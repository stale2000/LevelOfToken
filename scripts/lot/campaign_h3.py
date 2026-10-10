"""Durable local supervisor for the bounded six-hour H3 LoT campaign.

All request intents and returned request IDs are written before waiting. A
request with an uncertain submit outcome is never submitted a second time;
the campaign app's request_key is the idempotency boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import tarfile
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import campaign_runtime
import fal_client
import fal_h3_client

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP = "bryanwade/lot-h3-campaign"
DEFAULT_PARENT_RUN = "h3-lot-openvid-pilot-v2"
MAX_START_TIMEOUT = 900
REQUEST_TIMEOUT = 2700
STARTUP_TIMEOUT = 900
TERMINATION_GRACE = 180
SAFETY_MARGIN = 300
MILESTONE = 250
MIN_MATCHED_UPDATES = 500
MAX_BRANCH_UPDATES = 4000
INITIAL_CAMPAIGN_ID = "lot-h3-six-hour-20261010-r4"
INITIAL_BRANCH = "control"
INITIAL_STEP = 500


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("campaign_started_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CampaignError(RuntimeError):
    """A campaign cannot safely proceed from its durable state."""


def read_primary_metrics(directory: Path, target: int) -> dict[str, Any] | None:
    """Read fixed 50% layout validation rows, aggregated by clip."""
    candidates = sorted(directory.rglob(f"metrics-step-{target:08d}.csv"))
    if not candidates:
        return None
    wanted_layouts = {"uniform50_1x2", "uniform50_2x1", "fixed50_1x2", "fixed50_2x1",
                      "uniform1x2", "uniform2x1"}
    per_clip: dict[str, dict[str, list[float]]] = {}
    dense_per_clip: dict[str, list[float]] = {}
    per_sigma: dict[str, dict[float, list[float]]] = {}
    for path in candidates:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("eval_split", "validation") != "validation":
                    continue
                try:
                    gap = float(row["teacher_gap_mse"])
                    sigma = float(row["sigma"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (gap >= 0 and gap < float("inf")):
                    continue
                clip = str(row.get("clip", row.get("source_id", "")))
                if not clip:
                    continue
                if row.get("layout") == "dense_lora":
                    dense_per_clip.setdefault(clip, []).append(gap)
                    continue
                if row.get("layout") not in wanted_layouts:
                    continue
                metrics = per_clip.setdefault(clip, {"teacher_gap_mse": [], "spatial_detail_deviation": [],
                                                     "temporal_teacher_mse": []})
                metrics["teacher_gap_mse"].append(gap)
                per_sigma.setdefault(clip, {}).setdefault(sigma, []).append(gap)
                for metric in ("spatial_detail_deviation", "temporal_teacher_mse"):
                    try:
                        value = (float(row["spatial_detail_deviation"])
                                 if metric == "spatial_detail_deviation" and row.get("spatial_detail_deviation")
                                 else abs(float(row["spatial_detail_ratio"]) - 1.0)
                                 if metric == "spatial_detail_deviation"
                                 else float(row[metric]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if value >= 0 and value < float("inf"):
                        metrics[metric].append(value)
    if not per_clip:
        return None
    return {
        "step": target,
        "clips": {clip: {key: statistics.fmean(values) for key, values in metrics.items() if values}
                  for clip, metrics in per_clip.items()},
        "dense_teacher_gap": {clip: statistics.fmean(values) for clip, values in dense_per_clip.items()},
        "sigma_teacher_gap": {
            clip: {str(sigma): statistics.fmean(values) for sigma, values in sigmas.items()}
            for clip, sigmas in per_sigma.items()
        },
        "layout_rows": sum(len(values["teacher_gap_mse"]) for values in per_clip.values()),
        "source_csv": str(candidates[-1]),
    }


def compare_primary(control: dict[str, Any], preserve: dict[str, Any]) -> dict[str, Any] | None:
    """Compute a clip-paired bootstrap for the predeclared fixed-50 endpoint."""
    shared = sorted(set(control["clips"]) & set(preserve["clips"]))
    if len(shared) < 2:
        return None
    deltas = [preserve["clips"][clip]["teacher_gap_mse"]
              - control["clips"][clip]["teacher_gap_mse"] for clip in shared]
    mean_delta = statistics.fmean(deltas)
    rng = random.Random(0)
    draws = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(1000))
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    control_mean = statistics.fmean(control["clips"][clip]["teacher_gap_mse"] for clip in shared)
    preserve_mean = statistics.fmean(preserve["clips"][clip]["teacher_gap_mse"] for clip in shared)
    def mean_metric(source, key):
        rows = [source["clips"][clip][key] for clip in shared if key in source["clips"][clip]]
        return statistics.fmean(rows) if rows else None
    detail_a, detail_b = mean_metric(control, "spatial_detail_deviation"), mean_metric(preserve, "spatial_detail_deviation")
    temporal_a, temporal_b = mean_metric(control, "temporal_teacher_mse"), mean_metric(preserve, "temporal_teacher_mse")
    dense_a_values = [control["dense_teacher_gap"][clip] for clip in shared if clip in control["dense_teacher_gap"]]
    dense_b_values = [preserve["dense_teacher_gap"][clip] for clip in shared if clip in preserve["dense_teacher_gap"]]
    dense_a = statistics.fmean(dense_a_values) if dense_a_values else None
    dense_b = statistics.fmean(dense_b_values) if dense_b_values else None
    per_sigma = {}
    shared_sigmas = sorted({sigma for clip in shared for sigma in control["sigma_teacher_gap"].get(clip, {})}
                           & {sigma for clip in shared for sigma in preserve["sigma_teacher_gap"].get(clip, {})},
                           key=float)
    for sigma in shared_sigmas:
        a = [control["sigma_teacher_gap"][clip][sigma] for clip in shared
             if sigma in control["sigma_teacher_gap"].get(clip, {}) and sigma in preserve["sigma_teacher_gap"].get(clip, {})]
        b = [preserve["sigma_teacher_gap"][clip][sigma] for clip in shared
             if sigma in control["sigma_teacher_gap"].get(clip, {}) and sigma in preserve["sigma_teacher_gap"].get(clip, {})]
        if a and b:
            per_sigma[sigma] = {"control_mean": statistics.fmean(a), "preserve_mean": statistics.fmean(b),
                                "relative_regression": statistics.fmean(b) / max(statistics.fmean(a), 1e-6) - 1}
    preserve_wins = (
        mean_delta <= -0.05 * max(control_mean, 1e-6) and hi < 0
        and (dense_a is None or dense_b is None or dense_b <= 1.05 * max(dense_a, 1e-6))
        and (detail_a is None or detail_b is None or detail_b <= 1.05 * max(detail_a, 1e-6))
        and (temporal_a is None or temporal_b is None or temporal_b <= 1.05 * max(temporal_a, 1e-6))
        and all(values["relative_regression"] <= 0.10 for values in per_sigma.values())
    )
    control_wins = (
        mean_delta >= 0.05 * max(control_mean, 1e-6) and lo > 0
        and (dense_a is None or dense_b is None or dense_a <= 1.05 * max(dense_b, 1e-6))
        and (detail_a is None or detail_b is None or detail_a <= 1.05 * max(detail_b, 1e-6))
        and (temporal_a is None or temporal_b is None or temporal_a <= 1.05 * max(temporal_b, 1e-6))
        and all(values["relative_regression"] >= -0.10 for values in per_sigma.values())
    )
    return {
        "target_step": control["step"], "paired_clips": len(shared),
        "control_mean": control_mean, "preserve_mean": preserve_mean,
        "preserve_minus_control": mean_delta, "clip_bootstrap_ci95": [lo, hi],
        "detail_deviation": {"control": detail_a, "preserve": detail_b},
        "temporal_teacher_error": {"control": temporal_a, "preserve": temporal_b},
        "dense_teacher_gap": {"control": dense_a, "preserve": dense_b},
        "per_sigma": per_sigma,
        "winner": "preserve" if preserve_wins else "control" if control_wins else None,
    }


class CampaignController:
    def __init__(
        self,
        campaign_id: str,
        *,
        parent_run_id: str = DEFAULT_PARENT_RUN,
        app: str = DEFAULT_APP,
        root: Path | None = None,
        fal_module: Any = fal_client,
        client_module: Any = fal_h3_client,
        now: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        log: Callable[[str], None] = print,
    ):
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", campaign_id) is None:
            raise ValueError("campaign id must contain lowercase letters, digits, and hyphens")
        self.campaign_id = campaign_id
        self.parent_run_id = parent_run_id
        self.app = app
        self.root = root or Path(__file__).resolve().parent / "runs" / "fal" / "campaigns" / campaign_id
        self.state_path = self.root / "campaign.json"
        self.stop_path = self.root / "STOP.json"
        self.lock_path = self.root / ".controller.lock"
        self.fal = fal_module
        self.client = client_module
        self.now = now
        self.monotonic = monotonic
        self.sleep = sleep
        self.command = command
        self.log = log
        self.state: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise CampaignError(f"campaign does not exist: {self.state_path}")
        self.state = read_json(self.state_path)
        if self.state.get("campaign_id") != self.campaign_id:
            raise CampaignError("campaign state identity mismatch")
        if self.state.get("app") != self.app or self.state.get("parent_run_id") != self.parent_run_id:
            raise CampaignError("app and parent identity are immutable after campaign creation")
        return self.state

    def save(self) -> None:
        atomic_json(self.state_path, self.state)

    def acquire_lock(self) -> None:
        """Prevent two local supervisors from racing a stage submission."""
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                pid = int(self.lock_path.read_text().strip())
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError, FileNotFoundError):
                self.lock_path.unlink(missing_ok=True)
                descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except PermissionError:
                raise CampaignError(f"campaign controller lock is held: {self.lock_path}") from None
            else:
                raise CampaignError(f"campaign controller is already running as PID {pid}")
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"{os.getpid()}\n")

    def release_lock(self) -> None:
        try:
            if int(self.lock_path.read_text().strip()) == os.getpid():
                self.lock_path.unlink(missing_ok=True)
        except (ValueError, FileNotFoundError):
            pass

    def initialize(self) -> dict[str, Any]:
        """Upload package and reuse the prepared data URL before freezing T0."""
        if self.state_path.exists():
            self.load()
            return self.state
        if self.stop_path.exists():
            raise CampaignError("campaign has a STOP marker and cannot be initialized")
        parent_dir = self.client.ROOT / "scripts/lot/runs/fal" / self.parent_run_id
        parent_state_path = parent_dir / "fal-state.json"
        if not parent_state_path.is_file():
            raise CampaignError(f"parent Fal state is missing: {parent_state_path}")
        parent_state = read_json(parent_state_path)
        checkpoint_dir = parent_dir / "artifacts/step-00000500"
        parent_result_path = checkpoint_dir / "result.json"
        if not parent_result_path.is_file():
            raise CampaignError("verified parent step-500 result is missing")
        parent_result = read_json(parent_result_path)
        if parent_result.get("checkpoint_step") != 500:
            raise CampaignError("parent result is not the expected step-500 checkpoint")
        for filename, key in (("lora.safetensors", "lora_file"), ("adapter.pt", "adapter_file")):
            artifact = checkpoint_dir / filename
            expected = parent_result.get("artifact_integrity", {}).get(key, {})
            if not artifact.is_file() or not self.client.artifact_integrity_matches(artifact, expected):
                raise CampaignError(f"parent step-500 {filename} is absent or failed SHA-256 verification")
        initial_dir = (self.client.ROOT / "scripts/lot/runs/fal/campaigns" / INITIAL_CAMPAIGN_ID
                       / "artifacts/control/train_control_0500/artifacts/step-00000500")
        initial_hashes = {}
        for filename in ("lora.safetensors", "adapter.pt"):
            path = initial_dir / filename
            if not path.is_file():
                raise CampaignError(f"selected prior control checkpoint is missing: {path}")
            initial_hashes[filename] = sha256_file(path)
        initial_result = read_json(initial_dir / "result.json")
        expected_initial = initial_result.get("artifact_integrity", {})
        for filename, key in (("lora.safetensors", "lora_file"), ("adapter.pt", "adapter_file")):
            if not self.client.artifact_integrity_matches(initial_dir / filename, expected_initial.get(key, {})):
                raise CampaignError(f"selected prior control {filename} failed its published SHA-256")
        data_url = parent_state.get("training_data_url")
        if not data_url or parent_state.get("completed_batches") != [0]:
            raise CampaignError("parent prepared data/cache identity is not ready")

        self.root.mkdir(parents=True, exist_ok=True)
        # Upload the exact current Python source before T0. No GPU request is
        # made until source/data/parent provenance are durable below.
        source_url, source_sha = self.client.upload_source(self.root)
        source_path = self.root / "lot-h3-source.zip"
        parent_hashes = {
            "lora_sha256": sha256_file(checkpoint_dir / "lora.safetensors"),
            "adapter_sha256": sha256_file(checkpoint_dir / "adapter.pt"),
            "parent_bundle_sha256": parent_result["artifact_integrity"]["bundle_file"]["sha256"],
        }
        now = self.now()
        self.state = {
            "schema": "lot-h3-campaign-v1",
            "campaign_id": self.campaign_id,
            "parent_run_id": self.parent_run_id,
            "app": self.app,
            "app_revision": self.state.get("app_revision"),
            "verified_app_configuration": self.state.get("verified_app_configuration"),
            "status": "ready_to_start",
            "campaign_started_at": None,
            "hard_deadline_at": None,
            "created_at": iso_utc(now),
            "source_bundle_url": source_url,
            "source_bundle_sha256": source_sha,
            "source_bundle_bytes": source_path.stat().st_size,
            "training_data_url": data_url,
            "parent_hashes": parent_hashes,
            "initial_checkpoint": {
                "campaign_id": INITIAL_CAMPAIGN_ID, "branch": INITIAL_BRANCH, "step": INITIAL_STEP,
                "lora_sha256": initial_hashes["lora.safetensors"],
                "adapter_sha256": initial_hashes["adapter.pt"],
                "source_result": str(initial_dir / "result.json"),
            },
            "phase": "qualify",
            "branches": {name: {"completed_updates": 0} for name in ("control", "detail", "dense")},
            "requests": {},
            "decisions": [],
            "shutdown_audits": [],
            "last_progress_at": iso_utc(now),
        }
        self.save()
        return self.state

    def freeze_start(self) -> None:
        self.load()
        if self.state.get("campaign_started_at"):
            return
        if self.state.get("status") != "ready_to_start":
            raise CampaignError("campaign cannot freeze T0 from its current status")
        started = self.now()
        self.state["campaign_started_at"] = iso_utc(started)
        self.state["hard_deadline_at"] = iso_utc(started + timedelta(seconds=campaign_runtime.CAMPAIGN_SECONDS))
        self.state["status"] = "running"
        self.state["started_monotonic_note"] = "UTC T0 is immutable; monotonic enforcement is per admitted request"
        self.save()

    def deadlines(self) -> campaign_runtime.CampaignDeadlines:
        if not self.state.get("campaign_started_at"):
            raise CampaignError("campaign T0 has not been frozen")
        return campaign_runtime.CampaignDeadlines.from_start(parse_utc(self.state["campaign_started_at"]))

    def ensure_active(self) -> None:
        self.load()
        if self.stop_path.exists() or self.state.get("status") in {"stopped", "finalized", "shutdown_unverified"}:
            raise CampaignError("campaign is stopped or finalized; no GPU request may be submitted")
        if not self.state.get("campaign_started_at"):
            raise CampaignError("campaign T0 is missing")

    def _request_id_for(self, key: str) -> str | None:
        record = self.state["requests"].get(key)
        return record.get("request_id") if record else None

    def request(
        self,
        phase: str,
        *,
        branch: str | None = None,
        recipe: str | None = None,
        target_step: int | None = None,
        evaluation_split: str = "validation",
        eval_items: int = 12,
        preview_items: int = 2,
        preview_steps: int = 8,
        preview_seed: int = 20261009,
        preview_split: str = "validation",
        preview_controls: str | None = None,
        preview_warmup_sigma: float = 0.7,
        preview_budget_fraction: float = 0.5,
        optimizer_seconds: float = 0,
        evaluation_seconds: float = 600,
        export_seconds: float = 900,
    ) -> dict[str, Any]:
        self.ensure_active()
        key = self.request_key(
            phase, branch, target_step, evaluation_split, preview_steps, preview_controls,
            preview_warmup_sigma, preview_budget_fraction, preview_seed,
        )
        prior = self.state["requests"].get(key)
        if prior:
            if prior.get("status") == "complete" and isinstance(prior.get("result"), dict):
                return prior["result"]
            if prior.get("status") == "incomplete_result_saved" and isinstance(prior.get("result"), dict):
                raise CampaignError(prior.get("error", f"request {key} ended incomplete; saved artifacts retained"))
            if not prior.get("request_id"):
                raise CampaignError(f"request {key} has an ambiguous submit outcome; refusing duplicate submission")
            return self._watch(key, self.fal.sync_client.get_handle(self.app, prior["request_id"]))

        if phase not in {"qualify", "train", "preview", "confirm"}:
            raise ValueError(f"unsupported phase: {phase}")
        recipe = recipe or {"control": "sparse50", "detail": "detail50", "dense": "dense_detail"}.get(branch or "", "sparse50")
        if phase == "train" and (branch not in {"control", "detail", "dense"} or target_step is None
                                  or target_step < 50 or target_step % 50):
            raise ValueError("train requests require branch and target_step")
        if phase == "preview" and (branch not in {"parent", "control", "detail", "dense", "preserve"} or target_step is None):
            raise ValueError("preview requests require target_step and an optional branch")
        if phase == "confirm" and (branch not in {"parent", "control", "detail", "dense", "preserve"} or target_step is None):
            raise ValueError("confirm requests require a parent/control/preserve branch and target_step")
        if target_step is not None and (target_step < 0 or target_step > MAX_BRANCH_UPDATES):
            raise ValueError("target_step is outside branch-local campaign limits")

        limits = campaign_runtime.FalLimits(
            server_start_timeout=MAX_START_TIMEOUT,
            startup_timeout=STARTUP_TIMEOUT,
            request_timeout=REQUEST_TIMEOUT,
            termination_grace=TERMINATION_GRACE,
            safety_margin=SAFETY_MARGIN,
        )
        admission = campaign_runtime.admit_request(
            now_utc=self.now(), deadlines=self.deadlines(),
            optimizer_seconds=optimizer_seconds,
            evaluation_seconds=evaluation_seconds,
            export_reserve_seconds=export_seconds,
            phase="optimizer" if phase == "train" else "evaluation",
            limits=limits,
        )
        if not admission.admitted:
            raise CampaignError(f"request {key} denied by deadline admission: {admission.reason}")
        full_lifecycle = MAX_START_TIMEOUT + STARTUP_TIMEOUT + REQUEST_TIMEOUT + TERMINATION_GRACE + SAFETY_MARGIN
        remaining = (self.deadlines().final_deadline.at - self.now()).total_seconds()
        if remaining < full_lifecycle:
            raise CampaignError(
                f"request {key} denied: only {remaining:.0f}s remain; campaign requires the full "
                f"{full_lifecycle}s queue/startup/request/termination/safety envelope"
            )

        payload: dict[str, Any] = {
            "phase": phase,
            "campaign_id": self.campaign_id,
            "parent_run_id": self.parent_run_id,
            "parent_lora_sha256": self.state["parent_hashes"]["lora_sha256"],
            "parent_adapter_sha256": self.state["parent_hashes"]["adapter_sha256"],
            "branch": branch,
            "recipe": recipe,
            "optimizer_budget_seconds": int(optimizer_seconds) if phase == "train" and optimizer_seconds else None,
            "initial_campaign_id": self.state["initial_checkpoint"]["campaign_id"],
            "initial_branch": self.state["initial_checkpoint"]["branch"],
            "initial_step": self.state["initial_checkpoint"]["step"],
            "initial_lora_sha256": self.state["initial_checkpoint"]["lora_sha256"],
            "initial_adapter_sha256": self.state["initial_checkpoint"]["adapter_sha256"],
            "target_step": 0 if target_step is None else target_step,
            "preview_step": target_step if phase == "preview" else None,
            "campaign_started_at": self.state["campaign_started_at"],
            "request_key": key,
            "code_bundle_url": self.state["source_bundle_url"],
            "code_bundle_sha256": self.state["source_bundle_sha256"],
            "training_data_url": self.state["training_data_url"],
            "evaluation_split": evaluation_split,
            "eval_items": eval_items,
            "preview_items": preview_items,
            "preview_steps": preview_steps,
            "preview_seed": preview_seed,
            "preview_split": preview_split,
            "preview_controls": preview_controls,
            "preview_warmup_sigma": preview_warmup_sigma,
            "preview_budget_fraction": preview_budget_fraction,
        }
        record = {
            "request_key": key,
            "phase": phase,
            "branch": branch,
            "recipe": recipe,
            "target_step": target_step,
            "preview_warmup_sigma": preview_warmup_sigma if phase == "preview" else None,
            "preview_budget_fraction": preview_budget_fraction if phase == "preview" else None,
            "preview_seed": preview_seed if phase == "preview" else None,
            "status": "submitting",
            "submitted_intent_at": iso_utc(self.now()),
            "start_timeout": MAX_START_TIMEOUT,
            "admission": {
                "reason": admission.reason,
                "processing_seconds": admission.processing_seconds,
                "latest_finish": iso_utc(admission.latest_finish),
                "optimizer_seconds": optimizer_seconds,
                "evaluation_seconds": evaluation_seconds,
                "export_seconds": export_seconds,
            },
        }
        self.state["requests"][key] = record
        self.save()
        self.ensure_active()
        record = self.state["requests"][key]
        try:
            handle = self.fal.sync_client.submit(self.app, payload, start_timeout=MAX_START_TIMEOUT)
        except Exception as exc:
            # The remote may have accepted the POST before a transport error.
            # Keep the intent ambiguous; never blindly submit it again.
            record = self.state["requests"][key]
            record.update({"status": "submit_outcome_ambiguous", "submit_error": f"{type(exc).__name__}: {exc}"})
            self.save()
            raise
        record.update({"request_id": handle.request_id, "status": "submitted", "submitted_at": iso_utc(self.now())})
        self.save()
        return self._watch(key, handle)

    @staticmethod
    def request_key(phase: str, branch: str | None, target: int | None,
                    evaluation_split: str, preview_steps: int,
                    preview_controls: str | None = None,
                    preview_warmup_sigma: float = 0.7,
                    preview_budget_fraction: float = 0.5,
                    preview_seed: int = 20261009) -> str:
        parts = [phase]
        if branch:
            parts.append(branch)
        if target is not None:
            parts.append(f"{target:04d}")
        if phase == "confirm":
            parts.extend((evaluation_split,))
        if phase == "preview":
            parts.append(f"{preview_steps}steps")
            signature = (f"{preview_controls or 'default'}|"
                         f"sigma={preview_warmup_sigma:.3f}|budget={preview_budget_fraction:.3f}|seed={preview_seed}")
            parts.append(hashlib.sha256(signature.encode()).hexdigest()[:8])
        return "/".join(parts)

    def _watch(self, key: str, handle: Any) -> dict[str, Any]:
        record = self.state["requests"][key]
        request_id = handle.request_id
        if not record.get("request_id"):
            record["request_id"] = request_id
            record["status"] = "submitted"
            self.save()
        try:
            seen: set[str] = set(record.get("seen_logs", []))
            for update in handle.iter_events(with_logs=True, interval=5.0):
                self.ensure_active()
                record = self.state["requests"][key]
                lines = getattr(update, "logs", None) or []
                for item in lines:
                    token = json.dumps(item, sort_keys=True, default=str)
                    if token in seen:
                        continue
                    seen.add(token)
                    message = item.get("message", str(item)) if isinstance(item, dict) else str(item)
                    self.log(message)
                    try:
                        self.client.consume_publication(message, self.root / "logs" / key.replace("/", "_") )
                    except Exception as exc:  # noqa: BLE001 - leave publication URL durable for retry
                        publication_errors = self.root / "publication-errors.jsonl"
                        publication_errors.parent.mkdir(parents=True, exist_ok=True)
                        with publication_errors.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"request_key": key, "message": message, "error": str(exc)}) + "\n")
                record.update({"status": type(update).__name__.lower(), "seen_logs": sorted(seen),
                               "updated_at": iso_utc(self.now())})
                self.state["last_progress_at"] = record["updated_at"]
                self.save()
                if getattr(update, "error", None):
                    raise RuntimeError(str(update.error))
            self.ensure_active()
            record = self.state["requests"][key]
            record["remote_terminal_at"] = iso_utc(self.now())
            self.save()
            result = handle.get()
            if not isinstance(result, dict):
                raise TypeError(f"Fal returned {type(result).__name__}, expected a result object")
            if result.get("request_key") not in (None, key):
                raise CampaignError("remote result request_key does not match the submitted stage")
            if result.get("campaign_id") not in (None, self.campaign_id):
                raise CampaignError("remote result campaign_id does not match")
            result_path = self.root / "requests" / f"{hashlib.sha256(key.encode()).hexdigest()}.result.json"
            atomic_json(result_path, result)
            # Keep a per-request artifact namespace: equal branch-local steps
            # from different arms can never overwrite each other.
            artifact_dir = self.root / "artifacts" / (record.get("branch") or "campaign") / key.replace("/", "_")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            self.client.save_result(artifact_dir, result)
            self._extract_preview_artifacts(artifact_dir, key, result)
            checkpoint_step = result.get("checkpoint_step")
            record.update({"status": "complete", "completed_at": iso_utc(self.now()),
                           "result_path": str(result_path), "result": result,
                           "reported_checkpoint_step": checkpoint_step,
                           "remote_result_status": result.get("status")})
            self.state["last_progress_at"] = record["completed_at"]
            if record.get("phase") == "train" and record.get("branch"):
                branch_state = self.state["branches"][record["branch"]]
                branch_state["last_checkpoint_step"] = checkpoint_step
            expected_status = {
                "qualify": {"qualified"}, "train": {"training_complete"},
                "preview": {"preview_complete"}, "confirm": {"confirmation_complete"},
            }[record["phase"]]
            if result.get("status") not in expected_status:
                record["status"] = "incomplete_result_saved"
                record["error"] = (
                    f"remote {record['phase']} returned status={result.get('status')!r}; "
                    f"expected one of {sorted(expected_status)}; any published checkpoint is saved"
                )
                self.save()
                raise CampaignError(record["error"])
            if record.get("phase") == "train" and record.get("branch"):
                target_step = int(record["target_step"] or 0)
                if checkpoint_step is None or int(checkpoint_step) < target_step:
                    record["status"] = "incomplete_result_saved"
                    record["error"] = (f"training returned checkpoint_step={checkpoint_step!r}, "
                                       f"below requested target {target_step}")
                    self.save()
                    raise CampaignError(record["error"])
                branch_state = self.state["branches"][record["branch"]]
                branch_state["completed_updates"] = max(
                    int(branch_state.get("completed_updates", 0)), target_step,
                )
            self.save()
            return result
        except KeyboardInterrupt:
            self.stop("keyboard_interrupt")
            raise
        except Exception as exc:
            record = self.state["requests"][key]
            if record.get("status") == "incomplete_result_saved":
                record["error"] = f"{type(exc).__name__}: {exc}"
                self.save()
                raise
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                           "failed_at": iso_utc(self.now())})
            self.save()
            raise

    def _extract_preview_artifacts(self, artifact_dir: Path, key: str, result: dict[str, Any]) -> None:
        bundle = result.get("bundle_file")
        if not isinstance(bundle, dict) or not bundle.get("file_name"):
            return
        path = next((candidate for candidate in artifact_dir.rglob(Path(bundle["file_name"]).name)
                     if candidate.is_file()), None)
        if path is None or not tarfile.is_tarfile(path):
            return
        destination_root = self.root / "results" / "previews" / key.replace("/", "_")
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.startswith("run/training/previews/"):
                    continue
                if not member.name.endswith((".mp4", ".png", "manifest.json")):
                    continue
                relative = Path(member.name.removeprefix("run/training/previews/"))
                if relative.is_absolute() or ".." in relative.parts:
                    raise CampaignError("preview archive contains an unsafe path")
                source = archive.extractfile(member)
                if source is None:
                    continue
                target = destination_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as stream:
                    stream.write(source.read())

    def stop(self, reason: str) -> None:
        self.load()
        if not self.stop_path.exists():
            atomic_json(self.stop_path, {"campaign_id": self.campaign_id, "reason": reason,
                                         "stopped_at": iso_utc(self.now()), "app": self.app})
        for record in self.state.get("requests", {}).values():
            if record.get("status") in {"complete", "completed", "cancelled"} or record.get("remote_terminal_at"):
                continue
            request_id = record.get("request_id")
            if request_id:
                try:
                    self.fal.sync_client.get_handle(self.app, request_id).cancel()
                    record.update({"status": "cancel_requested", "cancel_requested_at": iso_utc(self.now())})
                except Exception as exc:  # noqa: BLE001 - preserve stop and cancel remaining handles
                    record["cancel_error"] = f"{type(exc).__name__}: {exc}"
        self.state.update({"status": "stopped", "stop_reason": reason, "stopped_at": iso_utc(self.now())})
        self.save()

    def list_runners(self) -> list[dict[str, Any]]:
        alias = self.app.rsplit("/", 1)[-1]
        result = self.command(
            ["uvx", "--from", "fal", "fal", "apps", "runners", alias,
             "--state", "all", "--output", "json"],
            check=True, capture_output=True, text=True, timeout=45,
        )
        parsed = json.loads(result.stdout)
        runners = parsed.get("runners", []) if isinstance(parsed, dict) else parsed
        if not isinstance(runners, list):
            raise CampaignError("Fal runner listing returned an unrecognized shape")
        return [r for r in runners if isinstance(r, dict)]

    def shutdown_audit(self, *, snapshots: int = 2, spacing_seconds: float = 30) -> bool:
        if snapshots != 2 or spacing_seconds < 30:
            raise ValueError("shutdown audit requires two snapshots at least 30 seconds apart")
        self.load()
        self.stop("campaign_complete")
        self._refresh_request_termination()
        self._kill_idle_campaign_runners()
        snapshots_data: list[dict[str, Any]] = []
        for index in range(2):
            self._refresh_request_termination()
            runners = self.list_runners()
            active = [r for r in runners if int(r.get("in_flight_requests", 0) or 0) > 0]
            snapshot = {"captured_at": iso_utc(self.now()), "runner_count": len(runners),
                        "active_count": len(active), "runner_ids": [r.get("runner_id") for r in runners]}
            snapshots_data.append(snapshot)
            self.state["shutdown_audits"] = snapshots_data
            self.save()
            if index == 0:
                self.sleep(spacing_seconds)
        self._refresh_request_termination()
        unresolved = [key for key, record in self.state.get("requests", {}).items()
                      if not record.get("remote_terminal_at")
                      and record.get("status") not in {"complete", "completed", "cancelled"}]
        clean = not unresolved and all(item["runner_count"] == 0 and item["active_count"] == 0
                                       for item in snapshots_data)
        self.state["status"] = "finalized" if clean else "shutdown_unverified"
        self.state["shutdown_verified"] = clean
        self.state["unresolved_requests"] = unresolved
        self.state["finalized_at"] = iso_utc(self.now()) if clean else None
        self.save()
        self.write_report()
        return clean

    def finalize_with_retries(self, *, max_attempts: int = 8) -> bool:
        """Retry cancellation/scale-to-zero audits inside the campaign reserve."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        for attempt in range(max_attempts):
            if attempt:
                remaining = (self.deadlines().final_deadline.at - self.now()).total_seconds()
                if remaining <= 60:
                    break
                self.sleep(min(30, remaining - 60))
            try:
                if self.shutdown_audit():
                    return True
            except Exception as exc:  # noqa: BLE001 - retry provider/API audit errors within reserve
                self.log(f"campaign shutdown audit attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")
                try:
                    self.load()
                    self.state.update({"status": "shutdown_unverified", "shutdown_verified": False,
                                       "shutdown_audit_error": f"{type(exc).__name__}: {exc}"})
                    self.save()
                    self.write_report()
                except Exception as persist_exc:  # noqa: BLE001 - report durable-state failure
                    self.log(f"could not persist shutdown diagnostic: {type(persist_exc).__name__}: {persist_exc}")
        return False

    def _kill_idle_campaign_runners(self) -> None:
        unresolved = [record for record in self.state.get("requests", {}).values()
                      if not record.get("remote_terminal_at")
                      and record.get("status") not in {"complete", "completed", "cancelled"}]
        if unresolved:
            return
        alias = self.app.rsplit("/", 1)[-1]
        for runner in self.list_runners():
            if runner.get("alias") not in {alias, self.app} or int(runner.get("in_flight_requests", 0) or 0) > 0:
                continue
            runner_id = runner.get("runner_id")
            if not runner_id:
                continue
            try:
                self.command(["uvx", "--from", "fal", "fal", "runners", "kill", runner_id],
                             check=True, capture_output=True, text=True, timeout=45)
                self.state.setdefault("shutdown_killed_runners", []).append(runner_id)
            except Exception as exc:  # noqa: BLE001 - retain audit evidence for failed termination
                self.state.setdefault("shutdown_kill_errors", []).append(
                    {"runner_id": runner_id, "error": f"{type(exc).__name__}: {exc}"})
            self.save()

    def _refresh_request_termination(self) -> None:
        for record in self.state.get("requests", {}).values():
            if record.get("remote_terminal_at") or not record.get("request_id"):
                continue
            try:
                status = self.fal.sync_client.get_handle(self.app, record["request_id"]).status(with_logs=False)
            except Exception as exc:  # noqa: BLE001 - unverifiable status must keep audit open
                record["shutdown_status_error"] = f"{type(exc).__name__}: {exc}"
                continue
            remote_status = str(getattr(status, "status", type(status).__name__)).lower()
            if remote_status in {"completed", "complete", "cancelled", "canceled", "failed", "timed_out"}:
                record["remote_terminal_at"] = iso_utc(self.now())
                record["remote_terminal_status"] = remote_status
            else:
                record["shutdown_remote_status"] = remote_status
        self.save()

    def status(self) -> dict[str, Any]:
        state = self.load()
        if state.get("campaign_started_at"):
            state = dict(state)
            state["remaining_seconds"] = max(
                0.0, (parse_utc(state["hard_deadline_at"]) - self.now()).total_seconds()
            )
        return state

    def write_report(self) -> dict[str, Any]:
        """Write a compact, durable human and machine-readable campaign result."""
        self.load()
        requests = []
        for key, record in sorted(self.state.get("requests", {}).items()):
            result = record.get("result", {})
            requests.append({
                "request_key": key,
                "phase": record.get("phase"),
                "branch": record.get("branch"),
                "target_step": record.get("target_step"),
                "reported_checkpoint_step": record.get("reported_checkpoint_step"),
                "status": record.get("status"),
                "remote_result_status": record.get("remote_result_status", result.get("status")),
                "request_id": record.get("request_id"),
                "result_path": record.get("result_path"),
                "artifact_directory": str(self.root / "artifacts" /
                                           (record.get("branch") or "campaign") /
                                           key.replace("/", "_")),
                "error": record.get("error"),
            })
        preview_root = self.root / "results" / "previews"
        previews = sorted(str(path.relative_to(self.root)) for path in preview_root.rglob("*")
                          if path.is_file()) if preview_root.is_dir() else []
        report = {
            "campaign_id": self.campaign_id,
            "parent_run_id": self.parent_run_id,
            "app": self.app,
            "campaign_started_at": self.state.get("campaign_started_at"),
            "hard_deadline_at": self.state.get("hard_deadline_at"),
            "status": self.state.get("status"),
            "shutdown_verified": self.state.get("shutdown_verified", False),
            "shutdown_audits": self.state.get("shutdown_audits", []),
            "shutdown_killed_runners": self.state.get("shutdown_killed_runners", []),
            "shutdown_kill_errors": self.state.get("shutdown_kill_errors", []),
            "unresolved_requests": self.state.get("unresolved_requests", []),
            "parent_hashes": self.state.get("parent_hashes", {}),
            "branches": self.state.get("branches", {}),
            "matched_gates": [decision for decision in self.state.get("decisions", [])
                              if "comparison" in decision or str(decision.get("gate", "")).startswith("extension_")],
            "decisions": self.state.get("decisions", []),
            "confirmations": [request for request in requests if request["phase"] == "confirm"],
            "requests": requests,
            "preview_paths": previews,
            "artifact_retry_notes": self.state.get("artifact_retry_notes", []),
            "report_generated_at": iso_utc(self.now()),
        }
        atomic_json(self.root / "report.json", report)
        lines = [
            f"# LoT H3 campaign {self.campaign_id}",
            "",
            f"- Status: `{report['status']}`",
            f"- Parent: `{self.parent_run_id}`",
            f"- App: `{self.app}`",
            f"- T0: `{report['campaign_started_at']}`",
            f"- Hard deadline: `{report['hard_deadline_at']}`",
            f"- Shutdown verified: **{report['shutdown_verified']}**",
            "",
            "## Branch checkpoints",
            "",
        ]
        for branch, values in sorted(report["branches"].items()):
            lines.append(
                f"- **{branch}**: completed {values.get('completed_updates', 0)} new updates; "
                f"last checkpoint {values.get('last_checkpoint_step')}; best step "
                f"{values.get('best_step')}; best fixed-50 teacher gap {values.get('best_primary')}"
            )
        lines.extend(["", "## Matched comparisons and stopping decisions", ""])
        if report["decisions"]:
            for decision in report["decisions"]:
                lines.append(f"- `{decision.get('gate')}`: {json.dumps(decision, sort_keys=True)}")
        else:
            lines.append("- No comparison decision was recorded.")
        lines.extend(["", "## Confirmation artifacts", ""])
        if report["confirmations"]:
            for request in report["confirmations"]:
                lines.append(
                    f"- `{request['branch']}` step {request['target_step']}: "
                    f"{request['remote_result_status']} — `{request['artifact_directory']}`"
                )
        else:
            lines.append("- Confirmation evaluation was not completed.")
        lines.extend(["", "## Preview files", ""])
        lines.extend(f"- `{path}`" for path in previews) if previews else lines.append("- No preview files were downloaded.")
        lines.extend(["", "## Shutdown audit", ""])
        if report["shutdown_audits"]:
            lines.extend(f"- {json.dumps(snapshot, sort_keys=True)}" for snapshot in report["shutdown_audits"])
        else:
            lines.append("- No provider snapshots were collected.")
        if report["unresolved_requests"]:
            lines.extend(["", "Unresolved requests: " + ", ".join(f"`{key}`" for key in report["unresolved_requests"])])
        markdown = "\n".join(lines) + "\n"
        report_path = self.root / "report.md"
        temporary = report_path.with_suffix(".md.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(markdown)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, report_path)
        return report

    def run(self) -> dict[str, Any]:
        """Execute qualified A/B milestones, then confirmation and shutdown."""
        self.initialize()
        if not self.state.get("campaign_started_at"):
            self.freeze_start()
        try:
            return self._run_started_campaign()
        except (Exception, KeyboardInterrupt) as exc:
            # Once T0 exists, every exceptional exit gets a durable stop,
            # cancellation of known request IDs, and a two-snapshot audit.
            try:
                self.stop(f"campaign_error:{type(exc).__name__}")
                self.finalize_with_retries()
            except Exception as audit_exc:  # noqa: BLE001 - preserve failure and leave an explicit audit gap
                try:
                    self.load()
                    self.state.update({"status": "shutdown_unverified",
                                       "shutdown_audit_error": f"{type(audit_exc).__name__}: {audit_exc}",
                                       "shutdown_verified": False})
                    self.save()
                    self.write_report()
                except Exception as persist_exc:  # noqa: BLE001 - report both audit and durable-state failures
                    self.log(
                        f"campaign shutdown audit failed: {type(audit_exc).__name__}: {audit_exc}; "
                        f"could not persist diagnostic: {type(persist_exc).__name__}: {persist_exc}"
                    )
                else:
                    self.log(f"campaign shutdown unverified: {type(audit_exc).__name__}: {audit_exc}")
            raise

    def _run_started_campaign(self) -> dict[str, Any]:
        """Run full-pass replayed sparse recipes, then compute-match dense controls."""
        if "qualify" not in self.state["requests"]:
            self.request("qualify", evaluation_split="validation", eval_items=4,
                         optimizer_seconds=0, evaluation_seconds=300, export_seconds=600)

        # Keep each sparse arm resident through a complete 741-source pass.
        # The epoch sampler visits every source once before repeating. train_h3
        # saves every 50 updates and evaluates at 250/500 while this request is
        # resident, so the intermediate checkpoints remain recoverable.
        # The deployed Fal app still validates the original recipe enum. The
        # source bundle infers the predicted-selector candidate from the detail
        # branch path, so this remains schema-compatible without a redeploy.
        for branch, recipe in (("control", "sparse50"), ("detail", "detail50")):
            current = int(self.state["branches"][branch].get("completed_updates", 0))
            if current < 750:
                self.request("train", branch=branch, recipe=recipe, target_step=750,
                             optimizer_seconds=1_500, evaluation_seconds=300, export_seconds=600,
                             eval_items=4)

        for target in (250, 500, 750):
            control = read_primary_metrics(self.root / "artifacts/control", target)
            detail = read_primary_metrics(self.root / "artifacts/detail", target)
            comparison = compare_primary(control, detail) if control and detail else None
            self.state["decisions"].append({
                "at": iso_utc(self.now()), "gate": f"matched_{target}",
                "comparison": comparison,
                "decision": ("detail50" if comparison and comparison.get("winner") == "preserve"
                             else "sparse50" if comparison and comparison.get("winner") == "control"
                             else "sparse50 (inconclusive; simpler objective)"),
            })
            self.save()

        latest = self.state["decisions"][-1].get("comparison") if self.state.get("decisions") else None
        best_branch = "detail" if latest and latest.get("winner") == "preserve" else "control"
        best_recipe = "detail50" if best_branch == "detail" else "sparse50"
        sparse_target = 750

        # The dense arm starts from the same immutable LoRA. Its finite-difference
        # objective matches the selected sparse recipe; updates are selected from
        # measured active step seconds, then adjusted in 50-update checkpoint units.
        dense_recipe = "dense_detail" if best_branch == "detail" else "dense_control"
        dense_state = self.state["branches"]["dense"]
        if int(dense_state.get("completed_updates", 0)) < 250:
            self.request("train", branch="dense", recipe=dense_recipe, target_step=250,
                         optimizer_seconds=1_500, evaluation_seconds=300, export_seconds=600,
                         eval_items=4)
        sparse_seconds = self._active_training_seconds(best_branch, sparse_target)
        dense_step_seconds = self._step_seconds("dense", 250)
        if sparse_seconds is not None and dense_step_seconds:
            target_dense = max(250, int(round((sparse_seconds / dense_step_seconds) / 50) * 50))
            target_dense = min(target_dense, 4_000)
            while int(self.state["branches"]["dense"].get("completed_updates", 0)) < target_dense:
                current_dense = int(self.state["branches"]["dense"].get("completed_updates", 0))
                next_target = min(target_dense, current_dense + 250)
                next_target = int(round(next_target / 50) * 50)
                self.request("train", branch="dense", recipe=dense_recipe, target_step=next_target,
                             optimizer_seconds=1_500, evaluation_seconds=0, export_seconds=600,
                             eval_items=4)
            dense_seconds = self._active_training_seconds("dense", int(self.state["branches"]["dense"].get("completed_updates", 0)))
            if dense_seconds:
                rel = abs(dense_seconds - sparse_seconds) / max(sparse_seconds, 1e-9)
                self.state["decisions"].append({"at": iso_utc(self.now()), "gate": "equal_training_compute",
                                                "sparse_branch": best_branch, "sparse_updates": sparse_target,
                                                "sparse_active_training_seconds": sparse_seconds,
                                                "dense_updates": self.state["branches"]["dense"].get("completed_updates"),
                                                "dense_active_training_seconds": dense_seconds,
                                                "relative_compute_difference": rel,
                                                "matched_within_5_percent": rel <= .05})
                self.save()
        else:
            self.state["decisions"].append({"at": iso_utc(self.now()), "gate": "equal_training_compute",
                                            "matched_within_5_percent": False,
                                            "rationale": "step timing rows were unavailable"})
            self.save()

        # Same-noise selector, fixed-layout, and temporally coherent shuffle
        # controls on one sparse checkpoint; dense model gets its full-token arm.
        controls = ("dense_lora,uniform50_1x2,uniform50_2x1,detail_inference,"
                    "detail_inference_shuffle,entropy_inference,entropy_inference_shuffle")
        for seed in (20261009, 20261010):
            for branch, target, control_set, steps in (
                (best_branch, sparse_target, controls, 30),
                ("dense", int(self.state["branches"]["dense"].get("completed_updates", 0)), "dense_lora", 26),
            ):
                key = self.request_key("preview", branch, target, "validation", steps, control_set,
                                       .7, .5, seed)
                if key in self.state["requests"] and self.state["requests"][key].get("status") == "complete":
                    continue
                self.request("preview", branch=branch, target_step=target, preview_items=4,
                             preview_steps=steps, preview_seed=seed, preview_split="validation",
                             preview_controls=control_set, preview_warmup_sigma=.7,
                             preview_budget_fraction=.5, evaluation_split="validation", eval_items=4,
                             optimizer_seconds=0, evaluation_seconds=1_800, export_seconds=600)

        self.state["status"] = "evaluation_complete"
        self.save()
        clean = self.finalize_with_retries()
        if not clean:
            self.log("campaign shutdown is unverified; no new GPU work will be submitted")
        return self.state

    def _train_rows(self, branch: str) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        for path in (self.root / "artifacts" / branch).rglob("metrics.jsonl"):
            try:
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("event") == "train_step":
                        rows[int(row["step"])] = row
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return rows

    def _active_training_seconds(self, branch: str, target: int) -> float | None:
        rows = self._train_rows(branch)
        chosen = [row for step, row in rows.items() if 1 <= step <= target and row.get("step_seconds") is not None]
        if len(chosen) < target:
            return None
        return sum(float(row["step_seconds"]) for row in chosen)

    def _step_seconds(self, branch: str, target: int) -> float | None:
        rows = self._train_rows(branch)
        chosen = [float(row["step_seconds"]) for step, row in rows.items()
                  if 1 <= step <= target and row.get("step_seconds") is not None]
        return statistics.fmean(chosen) if len(chosen) >= target else None

    def _record_gate(self, name: str) -> None:
        self.state["decisions"].append({
            "at": iso_utc(self.now()), "gate": name,
            "decision": "required matched 500-update pair and interim visual review saved",
        })
        self.save()

    def _branch_metrics(self, branch: str, target: int) -> dict[str, Any] | None:
        return read_primary_metrics(self.root / "artifacts" / branch, target)

    def _paired_gate(self, target: int) -> dict[str, Any] | None:
        control = self._branch_metrics("control", target)
        preserve = self._branch_metrics("preserve", target)
        if control is None or preserve is None:
            return None
        comparison = compare_primary(control, preserve)
        if comparison is None:
            return None
        control_initial = self._branch_metrics("control", 0)
        preserve_initial = self._branch_metrics("preserve", 0)
        dense_guard = {"control": None, "preserve": None}
        if control_initial is not None and preserve_initial is not None:
            shared = sorted(set(control["dense_teacher_gap"]) & set(preserve["dense_teacher_gap"])
                            & set(control_initial["dense_teacher_gap"])
                            & set(preserve_initial["dense_teacher_gap"]))
            if shared:
                dense_guard = {
                    "control": all(control["dense_teacher_gap"][clip]
                                   <= 1.05 * max(preserve["dense_teacher_gap"][clip], 1e-6)
                                   and control["dense_teacher_gap"][clip]
                                   <= 1.05 * max(control_initial["dense_teacher_gap"][clip], 1e-6)
                                   for clip in shared),
                    "preserve": all(preserve["dense_teacher_gap"][clip]
                                    <= 1.05 * max(control["dense_teacher_gap"][clip], 1e-6)
                                    and preserve["dense_teacher_gap"][clip]
                                    <= 1.05 * max(preserve_initial["dense_teacher_gap"][clip], 1e-6)
                                    for clip in shared),
                    "paired_clips": len(shared),
                }
        if not isinstance(dense_guard["control"], bool) or not isinstance(dense_guard["preserve"], bool):
            comparison["dense_baseline_guard"] = {"available": False, "winner_suppressed": True}
            comparison["winner"] = None
        else:
            comparison["dense_baseline_guard"] = {"available": True, **dense_guard}
            if comparison["winner"] and not dense_guard[comparison["winner"]]:
                comparison["winner"] = None
        self.state["decisions"].append({
            "at": iso_utc(self.now()), "gate": f"matched_{target}", "comparison": comparison,
        })
        for branch, metrics in (("control", control), ("preserve", preserve)):
            primary = statistics.fmean(x["teacher_gap_mse"] for x in metrics["clips"].values())
            branch_state = self.state["branches"][branch]
            current_best = branch_state.get("best_primary")
            if current_best is None or primary < current_best:
                branch_state.update({"best_step": target, "best_primary": primary})
        self.state["last_paired_gate"] = target
        self.save()
        return comparison

    def _extension_supported(self, previous: int, current: int) -> tuple[bool, str]:
        scores = {}
        for branch in ("control", "preserve"):
            before = self._branch_metrics(branch, previous)
            after = self._branch_metrics(branch, current)
            if before is None or after is None:
                return False, f"fixed-50 validation CSV missing for {branch} at {previous}/{current}"
            before_mean = statistics.fmean(x["teacher_gap_mse"] for x in before["clips"].values())
            after_mean = statistics.fmean(x["teacher_gap_mse"] for x in after["clips"].values())
            scores[branch] = {"previous": before_mean, "current": after_mean,
                              "relative_change": after_mean / max(before_mean, 1e-6) - 1}
        changes = [value["relative_change"] for value in scores.values()]
        if max(changes) > 0.10:
            return False, f"quality guard: an arm regressed by more than 10%: {scores}"
        if min(changes) > -0.03:
            return False, f"neither arm improved fixed-50 teacher gap by at least 3%: {scores}"
        return True, f"at least one arm improved >=3% and neither regressed >10%: {scores}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "start", "status", "stop", "audit"))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--parent-run-id", default=DEFAULT_PARENT_RUN)
    parser.add_argument("--app", default=DEFAULT_APP)
    args = parser.parse_args()
    controller = CampaignController(args.campaign_id, parent_run_id=args.parent_run_id, app=args.app)
    if args.command == "status":
        print(json.dumps(controller.status(), indent=2, sort_keys=True))
    elif args.command == "stop":
        # Let the active owner's polling loop observe STOP; it retains the lock
        # and performs the ownership-safe cancellation itself.
        controller.stop("operator_stop")
    else:
        controller.acquire_lock()
        try:
            if args.command == "audit":
                print(json.dumps({"shutdown_verified": controller.shutdown_audit()}, indent=2))
            elif args.command == "start":
                controller.initialize()
                controller.freeze_start()
                result = controller.request("qualify", evaluation_split="validation", eval_items=12,
                                            optimizer_seconds=0, evaluation_seconds=600, export_seconds=900)
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                result = controller.run()
                print(json.dumps({"campaign_id": args.campaign_id, "status": result.get("status"),
                                  "shutdown_verified": result.get("shutdown_verified")}, indent=2))
        finally:
            controller.release_lock()


if __name__ == "__main__":
    main()
