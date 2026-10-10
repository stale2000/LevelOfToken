"""Describe a completed H3 pilot using paired, clip-clustered comparisons."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "teacher_gap_mse", "clean_mse", "temporal_mse", "temporal_teacher_mse",
    "spatial_teacher_mse", "spatial_detail_ratio", "spatial_detail_deviation", "forward_seconds",
)
POLICIES = ("detail50", "entropy50", "random50")


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _key(row):
    return str(row["clip"]), float(row["sigma"]), str(row["layout"])


def completed_evaluations(run: Path):
    """Summary files commit an evaluation; partial JSONL evaluations are ignored."""
    summaries = {}
    for path in sorted(run.glob("metrics-step-*.json")):
        summary = json.loads(path.read_text())
        if summary.get("eval_id") and "step" in summary:
            summaries[int(summary["step"])] = summary
    accepted = {summary["eval_id"]: step for step, summary in summaries.items()}
    evaluations = {step: {} for step in summaries}
    train_rows = []
    path = run / "metrics.jsonl"
    if path.exists():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(f"invalid metrics JSONL at line {index + 1}") from None
                continue  # A process may die partway through its last append.
            if row.get("event") == "train_step":
                train_rows.append(row)
            elif row.get("event") == "eval_sample" and row.get("eval_id") in accepted:
                step = accepted[row["eval_id"]]
                if int(row["step"]) != step:
                    raise ValueError("evaluation row step differs from its committed summary")
                if _finite(row.get("spatial_detail_ratio")):
                    row["spatial_detail_deviation"] = abs(float(row["spatial_detail_ratio"]) - 1)
                evaluations[step][_key(row)] = row  # Last identical resumed row wins.
    return summaries, evaluations, train_rows


def paired_stat(pairs, metric: str, bootstrap_samples: int = 2000):
    """Mean checkpoint-minus-reference change and CI resampled by whole clip."""
    clusters = defaultdict(list)
    references, currents = defaultdict(list), defaultdict(list)
    for before, after in pairs:
        if not (_finite(before.get(metric)) and _finite(after.get(metric))):
            continue
        clip = str(before["clip"])
        references[clip].append(float(before[metric]))
        currents[clip].append(float(after[metric]))
        clusters[clip].append(float(after[metric]) - float(before[metric]))
    if not clusters:
        return None
    clip_deltas = [statistics.fmean(values) for _, values in sorted(clusters.items())]
    delta = statistics.fmean(clip_deltas)
    ci = None
    if len(clip_deltas) > 1:
        rng = random.Random(0)
        draws = sorted(statistics.fmean(rng.choices(clip_deltas, k=len(clip_deltas)))
                       for _ in range(bootstrap_samples))
        ci = [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
    return {
        "paired_rows": sum(map(len, references.values())), "clips": len(clusters),
        "reference_mean": statistics.fmean(statistics.fmean(values) for values in references.values()),
        "current_mean": statistics.fmean(statistics.fmean(values) for values in currents.values()),
        "mean_paired_delta": delta, "clip_bootstrap_ci95": ci,
        "delta_direction": "checkpoint_minus_reference",
    }


def _strata(pairs):
    yield "all", pairs
    for name, predicate in (
        ("sigma_low", lambda row: float(row["sigma"]) <= 0.3 + 1e-6),
        ("sigma_mid", lambda row: 0.3 + 1e-6 < float(row["sigma"]) < 0.9 - 1e-6),
        ("sigma_high", lambda row: float(row["sigma"]) >= 0.9 - 1e-6),
    ):
        selected = [(a, b) for a, b in pairs if predicate(a)]
        if selected:
            yield name, selected
    for group in ("static", "moving", "unknown"):
        selected = [(a, b) for a, b in pairs
                    if a.get("camera_motion_group", "unknown") == group
                    and b.get("camera_motion_group", "unknown") == group]
        if selected:
            yield "camera_" + group, selected


def _comparison(pairs):
    return {stratum: {metric: stat for metric in METRICS
                      if (stat := paired_stat(selected, metric)) is not None}
            for stratum, selected in _strata(pairs)}


def _same_budget(a, b):
    field = "tokens_per_frame" if "tokens_per_frame" in a and "tokens_per_frame" in b else "token_fraction"
    return (_finite(a.get(field)) and _finite(b.get(field))
            and math.isclose(float(a[field]), float(b[field]), rel_tol=0.0, abs_tol=1e-9))


def _same_extent_histogram(a, b):
    first, second = a.get("extent_histogram"), b.get("extent_histogram")
    return isinstance(first, dict) and first == second


def build_report(run: Path):
    summaries, evaluations, train_rows = completed_evaluations(run)
    if 0 not in summaries or not evaluations.get(0):
        raise ValueError("a completed step-0 baseline with eval_sample rows is required")
    steps = sorted(step for step, rows in evaluations.items() if rows)
    final_step = steps[-1]
    baseline, final = evaluations[0], evaluations[final_step]
    config_path = run / "run.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    warnings = []
    drift = {}
    for step in steps[1:]:
        rows = evaluations[step]
        changes = {}
        for layout in sorted({key[2] for key in baseline} | {key[2] for key in rows}):
            common = sorted(key for key in baseline.keys() & rows.keys() if key[2] == layout)
            pairs = [(baseline[key], rows[key]) for key in common]
            expected = sum(key[2] == layout for key in baseline)
            observed = sum(key[2] == layout for key in rows)
            changes[layout] = {"baseline_rows": expected, "checkpoint_rows": observed,
                               "paired_rows": len(common), "metrics": _comparison(pairs)}
            if len(common) != expected or observed != expected:
                warnings.append(f"step {step} {layout}: sample identities differ; paired {len(common)} "
                                f"of baseline {expected}, checkpoint {observed}")
        drift[str(step)] = changes
    placement = {}
    for candidate in ("detail50", "entropy50"):
        pairs, mismatches = [], []
        for key, row in sorted(final.items()):
            if key[2] != candidate:
                continue
            control = final.get((key[0], key[1], "random_shuffle50"))
            if control is None:
                mismatches.append({"clip": key[0], "sigma": key[1], "reason": "missing extent-matched placement control"})
            elif not _same_budget(row, control):
                mismatches.append({"clip": key[0], "sigma": key[1], "reason": "token budget mismatch"})
            elif not _same_extent_histogram(row, control):
                mismatches.append({"clip": key[0], "sigma": key[1], "reason": "extent histogram mismatch"})
            else:
                pairs.append((control, row))
        # A partly mismatched control must not produce a policy ranking.
        placement[candidate + "_minus_random_shuffle50"] = {
            "status": "rejected" if mismatches or not pairs else "comparable",
            "mismatches": mismatches,
            "metrics": {} if mismatches else _comparison(pairs),
        }
        if mismatches:
            warnings.append(f"{candidate}: placement comparison rejected because controls are unmatched")
    elapsed = max((float(row["elapsed_seconds"]) for row in train_rows
                   if _finite(row.get("elapsed_seconds"))), default=None)
    preview_steps = sorted(
        int(path.parent.name.removeprefix("step-"))
        for path in (run / "previews").glob("step-*/manifest.json")
        if path.parent.name.removeprefix("step-").isdigit()
    )
    review_path = run / "visual-review.json"
    visual_review = json.loads(review_path.read_text()) if review_path.is_file() else {}
    observations = []
    latest_changes = drift.get(str(final_step), {})
    for control in ("base_dense", "dense_lora", "lot_dense"):
        comparison = latest_changes.get(control, {})
        for metric in ("clean_mse", "temporal_mse"):
            stat = comparison.get("metrics", {}).get("all", {}).get(metric)
            if stat is None:
                continue
            ci = stat["clip_bootstrap_ci95"]
            if control == "base_dense" and abs(stat["mean_paired_delta"]) > 1e-6:
                warnings.append(f"frozen base_dense {metric} changed by {stat['mean_paired_delta']:+.5g}; "
                                "check evaluation determinism/model identity")
            elif ci is not None and ci[0] > 0:
                observations.append(f"{control} {metric} increased across paired clips: "
                                    f"delta {stat['mean_paired_delta']:+.5g}, 95% interval {ci}")
    if not any(key[2] == "base_dense" for key in final):
        warnings.append("Frozen base_dense metric rows are missing; teacher-quality drift cannot be checked")
    report = {
        "run": str(run.resolve()), "run_config": config, "baseline_eval_id": summaries[0]["eval_id"],
        "latest_completed_step": final_step, "completed_steps": steps,
        "checkpoint_vs_baseline": drift, "placement_vs_random": placement,
        "control_layouts": [name for name in ("base_dense", "dense_lora", "lot_dense", "dense")
                            if any(key[2] == name for key in final)],
        "training_elapsed_seconds": elapsed, "warnings": warnings, "observations": observations,
        "visual_validation": (
            str(visual_review["summary"])
            if visual_review.get("status") == "reviewed"
            and set(preview_steps).issubset(visual_review.get("reviewed_steps", [])) else
            f"Sampler video previews exist for steps {preview_steps}; human visual review is pending. "
            "Proxy losses cannot establish quality"
            if preview_steps else
            "Real sampler video comparisons have not been generated; proxy losses cannot establish quality"
        ),
        "timing_scope": "forward_seconds measures DiT forward only; excludes layout selection, decoding, "
                        "teacher training forwards, backward, checkpointing and cloud setup",
        "ci_method": "paired clip means; deterministic percentile bootstrap resampling clips; "
                     "single-clip CI omitted; exploratory comparisons have no multiplicity correction",
        "next_run_decisions": [
            "Inspect frozen base and dense LoRA drift before attributing improvement to token placement.",
            ("Prefer a placement hypothesis only when its equal-budget difference from random is consistent "
             "across clips and low/mid sigma, and real sampler videos support it."),
            "If all layouts improve similarly, test objective/noise/weight changes before a new selector.",
            ("If teacher gap falls while spatial detail or temporal quality worsens, investigate smoothing "
             "and teacher weight before increasing training duration."),
            "If oracle metrics improve but inference videos fail, prioritize inference layout selection.",
        ],
    }
    if final_step == 0:
        warnings.append("Baseline only: no completed training evaluation is available")
    return report


def markdown_report(report):
    visual_validation = report["visual_validation"]
    lines = ["# H3 pilot evidence", "", f"Latest completed evaluation: step {report['latest_completed_step']}.",
             "", visual_validation + ("" if visual_validation.endswith((".", "!", "?")) else "."), "",
             "Each delta compares identical clip, sigma and layout against step 0; intervals resample whole clips.",
             "", "| Step | Layout | Metric | Clips | Paired delta | 95% interval |",
             "| --- | --- | --- | --- | --- | --- |"]
    for step, layouts in report["checkpoint_vs_baseline"].items():
        for layout, comparison in layouts.items():
            for metric in ("teacher_gap_mse", "clean_mse", "temporal_mse", "spatial_detail_deviation"):
                stat = comparison["metrics"].get("all", {}).get(metric)
                if stat is None:
                    continue
                ci = stat["clip_bootstrap_ci95"]
                interval = "insufficient clips" if ci is None else f"[{ci[0]:.5g}, {ci[1]:.5g}]"
                lines.append(f"| {step} | {layout} | {metric} | {stat['clips']} | "
                             f"{stat['mean_paired_delta']:+.5g} | {interval} |")
    lines += ["", ("Lower error is better. A detail ratio is evaluated by its distance from 1; "
                   "an increasing ratio alone is not evidence of better quality."),
              "", "## Equal-budget placement", ""]
    for name, comparison in report["placement_vs_random"].items():
        stat = comparison["metrics"].get("all", {}).get("teacher_gap_mse")
        evidence = "no paired teacher-gap evidence" if stat is None else (
            f"teacher-gap delta {stat['mean_paired_delta']:+.5g}, {stat['clips']} clips, "
            f"95% interval {stat['clip_bootstrap_ci95']}")
        lines.append(f"- {name}: {comparison['status']}; {evidence}.")
    lines += ["", "## Controls and measurement limits", "",
              f"Recorded dense controls: {', '.join(report['control_layouts']) or 'missing'}.", "",
              report["timing_scope"] + ".", "",
              (f"Training elapsed time recorded: {report['training_elapsed_seconds']} seconds. "
               "This is not a cloud billing total."), "",
              "Per-sigma and camera strata, reference/current means and sample counts are in the JSON report.",
              "", "## Next-run decisions", ""]
    lines += ["- " + text for text in report["next_run_decisions"]]
    if report["observations"]:
        lines += ["", "## Observed control changes", ""] + ["- " + text for text in report["observations"]]
    if report["warnings"]:
        lines += ["", "## Evidence gaps", ""] + ["- " + text for text in report["warnings"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, help="report output directory (default run directory)")
    args = parser.parse_args()
    report = build_report(args.run)
    output = args.out or args.run
    output.mkdir(parents=True, exist_ok=True)
    (output / "next-run-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "next-run-report.md").write_text(markdown_report(report))
    print(output / "next-run-report.md")


if __name__ == "__main__":
    main()
