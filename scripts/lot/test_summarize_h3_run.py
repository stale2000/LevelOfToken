"""Fixture checks for trustworthy pilot reporting, with no GPU or model dependency."""
import json
import tempfile
import unittest
from pathlib import Path

from summarize_h3_run import (
    build_report,
    completed_evaluations,
    markdown_report,
    paired_stat,
)


def row(step, clip, layout="entropy50", eval_id=None, gap=1.0, budget=0.5, sigma=0.3):
    return {"event": "eval_sample", "step": step, "eval_id": eval_id or f"eval-{step}",
            "clip": clip, "sigma": sigma, "layout": layout, "teacher_gap_mse": gap,
            "clean_mse": gap, "token_fraction": budget, "camera_motion_group": "unknown",
            "extent_histogram": {"1x2x2": 100}, "spatial_detail_ratio": 0.9}


class PilotReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        (self.run / "run.json").write_text("{}")

    def commit(self, step, eval_id=None):
        (self.run / f"metrics-step-{step:08d}.json").write_text(json.dumps(
            {"step": step, "eval_id": eval_id or f"eval-{step}"}))

    def write(self, rows):
        (self.run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_partial_evaluations_do_not_become_results(self):
        self.commit(0)
        self.write([row(0, "a"), row(250, "a", gap=0.1)])
        report = build_report(self.run)
        self.assertEqual(report["completed_steps"], [0])
        self.assertEqual(report["latest_completed_step"], 0)

    def test_report_distinguishes_generated_previews_from_visual_review(self):
        self.commit(0)
        self.write([row(0, "a")])
        preview = self.run / "previews/step-00000000/manifest.json"
        preview.parent.mkdir(parents=True)
        preview.write_text("{}")
        self.assertIn("human visual review is pending", build_report(self.run)["visual_validation"])

    def test_report_uses_review_notes_only_for_reviewed_preview_steps(self):
        self.commit(0)
        self.write([row(0, "a")])
        preview = self.run / "previews/step-00000000/manifest.json"
        preview.parent.mkdir(parents=True)
        preview.write_text("{}")
        review = {"status": "reviewed", "reviewed_steps": [0], "summary": "Review complete."}
        (self.run / "visual-review.json").write_text(json.dumps(review))
        self.assertEqual(build_report(self.run)["visual_validation"], "Review complete.")
        review["reviewed_steps"] = []
        (self.run / "visual-review.json").write_text(json.dumps(review))
        self.assertIn("human visual review is pending", build_report(self.run)["visual_validation"])

    def test_markdown_reports_detail_deviation_with_lower_is_better_direction(self):
        self.commit(0)
        self.commit(250)
        self.write([row(0, "a", gap=1), row(250, "a", gap=0.5)])
        table = markdown_report(build_report(self.run)).split("Lower error is better.")[0]
        self.assertIn("spatial_detail_deviation", table)
        self.assertNotIn("| 250 | entropy50 | spatial_detail_ratio |", table)

    def test_pairs_require_same_clip_sigma_and_layout(self):
        self.commit(0)
        self.commit(250)
        self.write([row(0, "a", gap=2), row(0, "b", gap=100), row(250, "a", gap=1),
                    row(250, "c", gap=0), row(250, "a", sigma=0.9, gap=200)])
        comparison = build_report(self.run)["checkpoint_vs_baseline"]["250"]["entropy50"]
        self.assertEqual(comparison["paired_rows"], 1)
        self.assertEqual(comparison["metrics"]["all"]["teacher_gap_mse"]["mean_paired_delta"], -1)

    def test_mismatched_budget_rejects_entire_policy_comparison(self):
        self.commit(0)
        self.commit(250)
        self.write([row(0, "a"), row(250, "a", gap=0.5),
                    row(250, "a", "random_shuffle50", budget=0.25)])
        comparison = build_report(self.run)["placement_vs_random"]["entropy50_minus_random_shuffle50"]
        self.assertEqual(comparison["status"], "rejected")
        self.assertEqual(comparison["metrics"], {})

    def test_mismatched_extent_histogram_rejects_equal_token_control(self):
        self.commit(0)
        self.commit(250)
        before = row(0, "a")
        entropy = row(250, "a")
        shuffled = row(250, "a", "random_shuffle50")
        shuffled["extent_histogram"] = {"1x1x1": 100}
        self.write([before, entropy, shuffled])
        comparison = build_report(self.run)["placement_vs_random"]["entropy50_minus_random_shuffle50"]
        self.assertEqual(comparison["status"], "rejected")
        self.assertEqual(comparison["mismatches"][0]["reason"], "extent histogram mismatch")

    def test_duplicate_resume_evaluations_use_committed_id_and_one_row(self):
        self.commit(0)
        self.commit(250, "resumed")
        self.write([row(0, "a", gap=2), row(250, "a", eval_id="old", gap=999),
                    row(250, "a", eval_id="resumed", gap=1),
                    row(250, "a", eval_id="resumed", gap=0.75)])
        _, evaluations, _ = completed_evaluations(self.run)
        self.assertEqual(len(evaluations[250]), 1)
        stat = build_report(self.run)["checkpoint_vs_baseline"]["250"]["entropy50"]["metrics"]["all"]
        self.assertEqual(stat["teacher_gap_mse"]["mean_paired_delta"], -1.25)

    def test_bootstrap_clusters_frames_by_clip(self):
        pairs = [(row(0, "a", gap=2, sigma=sigma), row(250, "a", gap=1, sigma=sigma))
                 for sigma in (0.3, 0.6, 0.9)]
        pairs.append((row(0, "b", gap=1), row(250, "b", gap=2)))
        stat = paired_stat(pairs, "teacher_gap_mse")
        self.assertEqual(stat["paired_rows"], 4)
        self.assertEqual(stat["clips"], 2)
        self.assertEqual(stat["mean_paired_delta"], 0)
        self.assertEqual(stat["clip_bootstrap_ci95"], [-1, 1])


if __name__ == "__main__":
    unittest.main()
