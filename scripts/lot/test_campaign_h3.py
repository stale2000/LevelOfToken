"""Focused CPU-only lifecycle tests for campaign_h3; no Fal calls escape mocks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import campaign_h3


class FakeClock:
    def __init__(self, value: datetime):
        self.value = value

    def now(self) -> datetime:
        return self.value

    def monotonic(self) -> float:
        return self.value.timestamp()

    def sleep(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class FakeHandle:
    def __init__(self, request_id: str, app: str, payload: dict, fail_get: bool = False,
                 result_status: str | None = None):
        self.request_id = request_id
        self.payload = payload
        self.app = app
        self.fail_get = fail_get
        self.result_status = result_status
        self.cancelled = False

    def iter_events(self, *, with_logs: bool, interval: float):
        assert with_logs and interval == 5.0
        yield SimpleNamespace(logs=[], error=None)

    def get(self):
        if self.fail_get:
            raise TimeoutError("network dropped after submit")
        status = self.result_status or {"qualify": "qualified", "train": "training_complete",
                                        "preview": "preview_complete", "confirm": "confirmation_complete"}[self.payload["phase"]]
        return {"campaign_id": self.payload["campaign_id"],
                "request_key": self.payload["request_key"], "status": status,
                "checkpoint_step": self.payload.get("target_step")}

    def cancel(self):
        self.cancelled = True

    def status(self, *, with_logs: bool = False):
        assert not with_logs
        return SimpleNamespace(status="Queued")


class FakeFal:
    def __init__(self, *, fail_get: bool = False, result_status: str | None = None):
        self.calls = []
        self.handles = {}
        self.fail_get = fail_get
        self.result_status = result_status
        self.sync_client = self

    def submit(self, app, payload, *, start_timeout):
        self.calls.append((app, payload, start_timeout))
        handle = FakeHandle(f"request-{len(self.calls)}", app, payload, self.fail_get, self.result_status)
        self.handles[handle.request_id] = handle
        return handle

    def get_handle(self, app, request_id):
        assert self.handles[request_id].app == app
        return self.handles[request_id]


class FakeClient:
    def __init__(self, root: Path):
        self.ROOT = root
        self.uploaded = False
        self.saved = []

    def upload_source(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        archive = run_dir / "lot-h3-source.zip"
        archive.write_bytes(b"package-source")
        self.uploaded = True
        return "https://fal.invalid/source.zip", hashlib.sha256(archive.read_bytes()).hexdigest()

    @staticmethod
    def artifact_integrity_matches(path: Path, expected: dict):
        return path.is_file() and path.stat().st_size == expected["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]

    def consume_publication(self, _message: str, _run_dir: Path):
        return None

    def save_result(self, run_dir: Path, result: dict):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "saved.json").write_text(json.dumps(result))
        self.saved.append(run_dir)


def write_parent(parent: Path) -> None:
    run = parent / "scripts/lot/runs/fal/h3-lot-openvid-pilot-v2"
    checkpoint = run / "artifacts/step-00000500"
    checkpoint.mkdir(parents=True)
    (run / "fal-state.json").write_text(json.dumps({
        "training_data_url": "https://fal.invalid/data.zip", "completed_batches": [0],
    }))
    lora = b"verified lora"
    adapter = b"verified adapter"
    (checkpoint / "lora.safetensors").write_bytes(lora)
    (checkpoint / "adapter.pt").write_bytes(adapter)
    integrity = {
        "lora_file": {"bytes": len(lora), "sha256": hashlib.sha256(lora).hexdigest()},
        "adapter_file": {"bytes": len(adapter), "sha256": hashlib.sha256(adapter).hexdigest()},
        "bundle_file": {"sha256": "parent-bundle-sha"},
    }
    (checkpoint / "result.json").write_text(json.dumps({
        "checkpoint_step": 500, "artifact_integrity": integrity,
    }))


class CampaignControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lot-campaign-")
        self.root = Path(self.temp.name)
        write_parent(self.root)
        self.clock = FakeClock(datetime(2026, 10, 10, tzinfo=timezone.utc))
        self.fal = FakeFal()
        self.client = FakeClient(self.root)
        self.controller = campaign_h3.CampaignController(
            "campaign-test", root=self.root / "campaign", fal_module=self.fal,
            client_module=self.client, now=self.clock.now, monotonic=self.clock.monotonic,
            sleep=self.clock.sleep, log=lambda _message: None,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_source_upload_precedes_immutable_t0_and_bounded_submit(self):
        self.controller.initialize()
        self.assertTrue(self.client.uploaded)
        self.assertIsNone(self.controller.state["campaign_started_at"])
        self.controller.freeze_start()
        started = self.controller.state["campaign_started_at"]
        result = self.controller.request("qualify", evaluation_seconds=600, export_seconds=900)
        self.assertEqual(result["status"], "qualified")
        app, payload, start_timeout = self.fal.calls[0]
        self.assertEqual(app, "bryanwade/lot-h3-campaign")
        self.assertEqual(start_timeout, 900)
        self.assertEqual(payload["campaign_started_at"], started)
        self.assertEqual(payload["request_key"], "qualify")
        self.assertEqual(payload["target_step"], 0)
        self.assertEqual(payload["parent_lora_sha256"], self.controller.state["parent_hashes"]["lora_sha256"])
        self.assertEqual(payload["parent_adapter_sha256"], self.controller.state["parent_hashes"]["adapter_sha256"])
        self.assertTrue((self.root / "campaign/requests").is_dir())

    def test_final_preview_control_names_are_exact_renderer_contract(self):
        controls = {
            "parent": "base_dense,dense_lora,uniform50_1x2,detail50_warmup",
            "control": "dense_lora,uniform50_1x2,detail50_warmup",
            "preserve": "dense_lora,uniform50_1x2,detail50_warmup",
        }
        self.assertEqual(sum(len(value.split(",")) for value in controls.values()) * 4, 40)
        for value in controls.values():
            self.assertNotIn("fixed50", value)
            self.assertNotIn("detail50,", value)
        self.controller.initialize()
        self.controller.freeze_start()
        self.controller.request("preview", branch="parent", target_step=500,
                                preview_items=4, preview_steps=30, preview_split="confirmation",
                                preview_controls=controls["parent"], evaluation_split="confirmation",
                                preview_warmup_sigma=0.5, preview_budget_fraction=0.5,
                                evaluation_seconds=1800, export_seconds=900)
        payload = self.fal.calls[0][1]
        self.assertEqual(payload["preview_controls"], controls["parent"])
        self.assertEqual(payload["preview_step"], 500)
        self.assertEqual(payload["preview_items"], 4)
        self.assertEqual(payload["preview_steps"], 30)
        self.assertEqual(payload["preview_warmup_sigma"], 0.5)
        self.assertEqual(payload["preview_budget_fraction"], 0.5)

    def test_preview_request_key_tracks_layout_budget_and_warmup(self):
        base = campaign_h3.CampaignController.request_key(
            "preview", "control", 500, "validation", 30,
            "detail_inference,detail_inference_shuffle", 0.9, 0.5,
        )
        later_switch = campaign_h3.CampaignController.request_key(
            "preview", "control", 500, "validation", 30,
            "detail_inference,detail_inference_shuffle", 0.5, 0.5,
        )
        smaller_budget = campaign_h3.CampaignController.request_key(
            "preview", "control", 500, "validation", 30,
            "detail_inference,detail_inference_shuffle", 0.9, 0.25,
        )
        self.assertNotEqual(base, later_switch)
        self.assertNotEqual(base, smaller_budget)

    def test_preview_defaults_use_measured_budget_and_switch_point(self):
        self.controller.initialize()
        self.controller.freeze_start()
        self.controller.request("preview", branch="control", target_step=500)
        payload = self.fal.calls[0][1]
        self.assertEqual(payload["preview_warmup_sigma"], 0.7)
        self.assertEqual(payload["preview_budget_fraction"], 0.5)

    def test_resume_never_resets_t0_or_duplicates_completed_stage(self):
        self.controller.initialize()
        self.controller.freeze_start()
        original = self.controller.state["campaign_started_at"]
        self.controller.request("qualify", evaluation_seconds=600, export_seconds=900)
        self.clock.sleep(75)
        resumed = campaign_h3.CampaignController(
            "campaign-test", root=self.root / "campaign", fal_module=self.fal,
            client_module=self.client, now=self.clock.now, monotonic=self.clock.monotonic,
            sleep=self.clock.sleep, log=lambda _message: None,
        )
        resumed.initialize()
        resumed.freeze_start()
        again = resumed.request("qualify", evaluation_seconds=600, export_seconds=900)
        self.assertEqual(resumed.state["campaign_started_at"], original)
        self.assertEqual(len(self.fal.calls), 1)
        self.assertEqual(again["request_key"], "qualify")

    def test_ambiguous_submit_is_persisted_and_never_retried(self):
        class SubmitErrorFal(FakeFal):
            def submit(self, app, payload, *, start_timeout):
                super().submit(app, payload, start_timeout=start_timeout)
                raise TimeoutError("response lost")

        self.controller.fal = SubmitErrorFal()
        self.controller.initialize()
        self.controller.freeze_start()
        with self.assertRaises(TimeoutError):
            self.controller.request("qualify", evaluation_seconds=600, export_seconds=900)
        with self.assertRaises(campaign_h3.CampaignError):
            self.controller.request("qualify", evaluation_seconds=600, export_seconds=900)
        self.assertEqual(len(self.controller.fal.calls), 1)
        record = self.controller.state["requests"]["qualify"]
        self.assertEqual(record["status"], "submit_outcome_ambiguous")

    def test_interrupted_result_saves_checkpoint_but_does_not_advance_branch(self):
        self.controller.fal = FakeFal(result_status="interrupted_cleanly")
        self.controller.initialize()
        self.controller.freeze_start()
        with self.assertRaisesRegex(campaign_h3.CampaignError, "interrupted_cleanly"):
            self.controller.request("train", branch="control", target_step=250,
                                    optimizer_seconds=750, evaluation_seconds=600, export_seconds=900)
        state = self.controller.load()
        branch = state["branches"]["control"]
        self.assertEqual(branch["completed_updates"], 0)
        self.assertEqual(branch["last_checkpoint_step"], 250)
        request = state["requests"]["train/control/0250"]
        self.assertEqual(request["status"], "incomplete_result_saved")
        self.assertTrue(self.client.saved)

    def test_run_failure_sets_stop_and_audits_shutdown(self):
        self.controller.fal = FakeFal(fail_get=True)
        self.controller.command = lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps({"runners": []}))
        with self.assertRaises(TimeoutError):
            self.controller.run()
        self.assertTrue(self.controller.stop_path.is_file())
        state = self.controller.load()
        self.assertEqual(state["status"], "finalized")
        self.assertTrue(state["shutdown_verified"])
        self.assertTrue((self.root / "campaign/report.md").is_file())
        report = json.loads((self.root / "campaign/report.json").read_text())
        self.assertEqual(report["campaign_id"], "campaign-test")
        self.assertEqual(len(report["shutdown_audits"]), 2)

    def test_admission_denies_after_final_submission_cutoff(self):
        self.controller.initialize()
        self.controller.freeze_start()
        self.clock.sleep(4 * 3600 + 31 * 60)
        with self.assertRaisesRegex(campaign_h3.CampaignError, "deadline admission"):
            self.controller.request("qualify", evaluation_seconds=600, export_seconds=900)
        self.assertEqual(self.fal.calls, [])

    def test_shutdown_requires_two_empty_runner_snapshots(self):
        self.controller.initialize()
        self.controller.freeze_start()
        snapshots = iter(([], [], []))

        def command(*_args, **_kwargs):
            return SimpleNamespace(stdout=json.dumps({"runners": next(snapshots)}))

        self.controller.command = command
        self.assertTrue(self.controller.shutdown_audit())
        state = self.controller.load()
        self.assertEqual(state["status"], "finalized")
        audits = state["shutdown_audits"]
        self.assertEqual(len(audits), 2)
        self.assertGreaterEqual(
            (campaign_h3.parse_utc(audits[1]["captured_at"])
             - campaign_h3.parse_utc(audits[0]["captured_at"])).total_seconds(), 30,
        )

    def test_shutdown_with_idle_or_active_runner_is_not_verified(self):
        self.controller.initialize()
        self.controller.freeze_start()
        snapshots = iter(([], [{"runner_id": "gpu-1", "in_flight_requests": 1}], []))
        self.controller.command = lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps({"runners": next(snapshots)}))
        self.assertFalse(self.controller.shutdown_audit())
        self.assertEqual(self.controller.load()["status"], "shutdown_unverified")

    def test_shutdown_kills_only_exact_idle_campaign_alias_runner(self):
        self.controller.initialize()
        self.controller.freeze_start()
        inventory = iter((
            [{"alias": "lot-h3-campaign", "runner_id": "campaign-gpu", "in_flight_requests": 0}],
            [], [],
        ))
        calls = []

        def command(argv, **_kwargs):
            calls.append(argv)
            if "apps" in argv:
                return SimpleNamespace(stdout=json.dumps({"runners": next(inventory)}))
            return SimpleNamespace(stdout="{}")

        self.controller.command = command
        self.assertTrue(self.controller.shutdown_audit())
        self.assertIn("campaign-gpu", self.controller.load()["shutdown_killed_runners"])
        self.assertTrue(any(argv[-2:] == ["kill", "campaign-gpu"] for argv in calls))

    def test_shutdown_audit_detects_queued_request_even_without_runner(self):
        self.controller.initialize()
        self.controller.freeze_start()
        pending = FakeHandle("request-q", self.controller.app, {})
        self.fal.handles[pending.request_id] = pending
        self.controller.state["requests"]["train/control/0250"] = {
            "request_id": pending.request_id, "status": "submitted",
        }
        self.controller.save()
        self.controller.command = lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps({"runners": []}))
        self.assertFalse(self.controller.shutdown_audit())
        state = self.controller.load()
        self.assertEqual(state["unresolved_requests"], ["train/control/0250"])
        self.assertTrue(pending.cancelled)

    def test_fixed_50_clip_paired_metric_gate(self):
        def write_metrics(branch: str, value: float) -> Path:
            directory = self.root / "metrics" / branch
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "metrics-step-00000500.csv"
            fields = ["clip", "eval_split", "layout", "sigma", "teacher_gap_mse",
                      "spatial_detail_deviation", "spatial_detail_ratio", "temporal_teacher_mse"]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for clip in ("clip-a", "clip-b", "clip-c"):
                    for sigma in (0.3, 0.6, 0.9):
                        for layout in ("uniform50_1x2", "uniform50_2x1"):
                            writer.writerow({"clip": clip, "eval_split": "validation", "layout": layout,
                                             "sigma": sigma, "teacher_gap_mse": value,
                                             "spatial_detail_ratio": 1.1,
                                             "temporal_teacher_mse": 0.2})
                        writer.writerow({"clip": clip, "eval_split": "validation", "layout": "dense_lora",
                                         "sigma": sigma, "teacher_gap_mse": 0.2})
            return path

        control_path = write_metrics("control", 1.0)
        preserve_path = write_metrics("preserve", 0.9)
        control = campaign_h3.read_primary_metrics(control_path.parent, 500)
        preserve = campaign_h3.read_primary_metrics(preserve_path.parent, 500)
        comparison = campaign_h3.compare_primary(control, preserve)
        self.assertEqual(comparison["paired_clips"], 3)
        self.assertEqual(comparison["winner"], "preserve")
        self.assertLess(comparison["clip_bootstrap_ci95"][1], 0)
        self.assertAlmostEqual(comparison["dense_teacher_gap"]["control"], 0.2)
        self.assertAlmostEqual(comparison["dense_teacher_gap"]["preserve"], 0.2)

    def test_paired_gate_suppresses_winner_if_dense_path_drifts_from_initialization(self):
        self.controller.initialize()

        def write_metrics(branch: str, step: int, fixed_gap: float, dense_gap: float) -> None:
            directory = self.root / "campaign" / "artifacts" / branch / f"stage-{step}"
            directory.mkdir(parents=True)
            path = directory / f"metrics-step-{step:08d}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                fields = ["clip", "eval_split", "layout", "sigma", "teacher_gap_mse",
                          "spatial_detail_ratio", "temporal_teacher_mse"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for clip in ("clip-a", "clip-b", "clip-c"):
                    for sigma in (0.3, 0.6, 0.9):
                        for layout in ("uniform1x2", "uniform2x1"):
                            writer.writerow({"clip": clip, "eval_split": "validation", "layout": layout,
                                             "sigma": sigma, "teacher_gap_mse": fixed_gap,
                                             "spatial_detail_ratio": 1.0, "temporal_teacher_mse": 0.2})
                        writer.writerow({"clip": clip, "eval_split": "validation", "layout": "dense_lora",
                                         "sigma": sigma, "teacher_gap_mse": dense_gap})

        for branch in ("control", "preserve"):
            write_metrics(branch, 0, 1.0, 0.2)
        write_metrics("control", 500, 1.0, 0.3)
        write_metrics("preserve", 500, 0.9, 0.31)
        comparison = self.controller._paired_gate(500)
        self.assertEqual(comparison["winner"], None)
        self.assertFalse(comparison["dense_baseline_guard"]["preserve"])

    def test_campaign_preview_bundle_is_saved_for_review(self):
        artifact_dir = self.root / "campaign/artifacts/control/preview_control_0500"
        bundle_path = artifact_dir / "artifacts/step-00000500/preview.tar.gz"
        bundle_path.parent.mkdir(parents=True)
        preview_name = "run/training/previews/step-00000500/comparison.mp4"
        with tarfile.open(bundle_path, "w:gz") as archive:
            data = b"rendered-video"
            info = tarfile.TarInfo(preview_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        controller = campaign_h3.CampaignController(
            "campaign-test", root=self.root / "campaign", fal_module=self.fal,
            client_module=self.client, now=self.clock.now, monotonic=self.clock.monotonic,
            sleep=self.clock.sleep, log=lambda _message: None,
        )
        controller._extract_preview_artifacts(
            artifact_dir, "preview/control/0500/30steps/abcd1234",
            {"bundle_file": {"file_name": "preview.tar.gz"}},
        )
        saved = self.root / "campaign/results/previews/preview_control_0500_30steps_abcd1234/step-00000500/comparison.mp4"
        self.assertEqual(saved.read_bytes(), b"rendered-video")


if __name__ == "__main__":
    unittest.main()
