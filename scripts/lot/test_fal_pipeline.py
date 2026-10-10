"""CPU regression tests for Fal lifecycle; every cloud call is mocked."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import fal_h3_client as client
import torch
from pydantic import BaseModel
from run_artifacts import resolve_checkpoint, save_checkpoint_bundle


def load_app_without_fal():
    """The deployment SDK lives in uvx; lifecycle tests need no cloud SDK."""
    class App:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()

    class File(BaseModel):
        url: str
        file_name: str

    fal = types.ModuleType("fal")
    fal.App = App
    fal.cached = lambda function: function
    fal.endpoint = lambda _path: lambda function: function
    toolkit = types.ModuleType("fal.toolkit")
    toolkit.File = File
    toolkit.clone_repository = mock.Mock()
    toolkit.download_file = mock.Mock()
    provider = types.ModuleType("fal.toolkit.file.providers.fal")
    provider.FalFileRepositoryV3 = object
    modules = {"fal": fal, "fal.toolkit": toolkit,
               "fal.toolkit.file": types.ModuleType("fal.toolkit.file"),
               "fal.toolkit.file.providers": types.ModuleType("fal.toolkit.file.providers"),
               "fal.toolkit.file.providers.fal": provider}
    spec = importlib.util.spec_from_file_location("lot_test_fal_app", Path(__file__).with_name("fal_h3_app.py"))
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, File


app, FakeFile = load_app_without_fal()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lot-lifecycle-")
        self.root = Path(self.temporary.name)
        self.patch = mock.patch.object(app, "DATA_ROOT", self.root / "remote")
        self.patch.start()
        self.worker = app.H3LoTTraining()
        self.worker.runner_gpu = "mock RTX PRO 6000"
        self.worker.gpu_setup_error = None
        self.paths = self.worker._paths("pilot-test")
        self.paths["train"].mkdir(parents=True)

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def snapshot(self, step):
        return save_checkpoint_bundle(
            self.paths["train"], step, {"weight": torch.ones(1)},
            lambda path: path.write_bytes(b"lora"), {"step": step}, {},
        )

    def test_scan_recovers_commit_without_pointer_and_skips_corruption(self):
        first = self.snapshot(0)
        newest = self.snapshot(50)
        (self.paths["train"] / "latest.json").unlink()
        self.assertEqual(resolve_checkpoint(self.paths["train"]), newest)
        (self.paths["train"] / "latest.json").write_text("broken json")
        (newest / "adapter.pt").write_bytes(b"corrupt")
        self.assertEqual(resolve_checkpoint(self.paths["train"]), first)
        (self.paths["train"] / "checkpoints/.step-00000100.partial.tmp").mkdir()
        self.assertEqual(resolve_checkpoint(self.paths["train"]), first)

    def test_silent_command_deadline_allows_term_checkpoint(self):
        marker = self.root / "saved"
        script = ("import signal,time,pathlib,sys\n"
                  f"def stop(*_): pathlib.Path({str(marker)!r}).write_text('saved'); sys.exit(0)\n"
                  "signal.signal(signal.SIGTERM,stop)\ntime.sleep(20)\n")
        with self.assertRaises(TimeoutError):
            app.run_command([sys.executable, "-c", script], env=os.environ.copy(),
                            cwd=self.root, log_path=self.root / "command.log",
                            deadline=time.monotonic() + 0.4, grace_seconds=0.3)
        self.assertEqual(marker.read_text(), "saved")

    def test_deadline_force_kills_command_ignoring_term(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            app.run_command([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"],
                            env=os.environ.copy(), cwd=self.root, log_path=self.root / "kill.log",
                            deadline=started + 0.3, grace_seconds=0.2)
        self.assertLess(time.monotonic() - started, 2)

    def test_recovery_history_and_logs_export_without_checkpoint(self):
        self.worker._record_status(self.paths, {"status": "failed", "error": "cache failed"})
        self.worker._record_status(self.paths, {"status": "recovered"})
        logs = self.paths["run"] / "logs"
        logs.mkdir()
        (logs / "prepare.log").write_text("original failure details")
        archive = self.worker._archive(self.paths["run"], self.paths["train"], None, "diagnostics")
        with tarfile.open(archive) as contents:
            history = contents.extractfile("run/fal-history.jsonl").read().decode()
            self.assertIn("cache failed", history)
            self.assertEqual(contents.extractfile("run/logs/prepare.log").read(), b"original failure details")

    def test_preview_bundle_is_archived_and_extracted_for_local_review(self):
        preview = self.paths["train"] / "previews/step-00000500"
        preview.mkdir(parents=True)
        (preview / "comparison.png").write_bytes(b"contact-sheet")
        (preview / "p0_entropy.mp4").write_bytes(b"video")
        (preview / "manifest.json").write_text('{"status":"complete"}')
        bundle = self.worker._archive(self.paths["run"], self.paths["train"], None, "preview-test")
        with tarfile.open(bundle) as archive:
            self.assertIn("run/training/previews/step-00000500/comparison.png", archive.getnames())

        local = self.root / "local"
        data = bundle.read_bytes()
        import hashlib
        result = {"run_id": "pilot-test", "phase": "preview", "status": "preview_complete",
                  "bundle_file": {"url": "https://mock.invalid/bundle", "file_name": "preview.tar.gz"},
                  "artifact_integrity": {"bundle_file": {"bytes": len(data),
                                                              "sha256": hashlib.sha256(data).hexdigest()}}}
        def download(_url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        with mock.patch.object(client, "download_file", side_effect=download):
            client.save_result(local, result)
        self.assertEqual((local / "results/previews/step-00000500/comparison.png").read_bytes(), b"contact-sheet")
        self.assertEqual((local / "results/previews/step-00000500/p0_entropy.mp4").read_bytes(), b"video")

    def test_training_bundle_extracts_completed_evaluation_metrics(self):
        (self.paths["train"] / "metrics-step-00000500.csv").write_text("step,layout\n500,entropy50\n")
        (self.paths["train"] / "metrics-step-00000500.json").write_text('{"step":500}')
        (self.paths["train"] / "metrics-confirmation-step-00000500.csv").write_text(
            "step,eval_split,layout\n500,confirmation,uniform1x2\n")
        (self.paths["train"] / "metrics-confirmation-step-00000500.json").write_text(
            '{"step":500,"eval_split":"confirmation"}')
        bundle = self.worker._archive(self.paths["run"], self.paths["train"], None, "train-test")
        local = self.root / "local"
        local.mkdir(parents=True)
        client.atomic_json(local / "fal-state.json", {"runner_gpu": "mock RTX", "gpu_preflight_passed": True})
        data = bundle.read_bytes()
        import hashlib
        result = {"run_id": "pilot-test", "phase": "train", "status": "training_complete",
                  "checkpoint_step": 500,
                  "bundle_file": {"url": "https://mock.invalid/train", "file_name": "train.tar.gz"},
                  "artifact_integrity": {"bundle_file": {"bytes": len(data),
                                                              "sha256": hashlib.sha256(data).hexdigest()}}}
        def download(_url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        with mock.patch.object(client, "download_file", side_effect=download) as download_mock:
            client.save_result(local, result)
            client.save_result(local, result)
            download_mock.assert_called_once()
        self.assertEqual((local / "results/metrics-step-00000500.csv").read_text(),
                         "step,layout\n500,entropy50\n")
        self.assertEqual(json.loads((local / "results/metrics-step-00000500.json").read_text()),
                         {"step": 500})
        self.assertEqual((local / "results/metrics-confirmation-step-00000500.csv").read_text(),
                         "step,eval_split,layout\n500,confirmation,uniform1x2\n")
        self.assertEqual(json.loads((local / "results/metrics-confirmation-step-00000500.json").read_text()),
                         {"step": 500, "eval_split": "confirmation"})
        saved_state = json.loads((local / "fal-state.json").read_text())
        self.assertEqual(saved_state["runner_gpu"], "mock RTX")
        self.assertTrue(saved_state["gpu_preflight_passed"])

    def test_changed_source_invalidates_preflight(self):
        source = self.root / "source-sha-one"
        self.paths["run"].mkdir(exist_ok=True)
        prep = {"complete": True, "training_preflight": "passed",
                "preflight_identity": self.worker._preflight_identity(source)}
        app.atomic_status(self.paths["run"] / "prep.json", prep)
        self.worker._require_prepared(self.paths, source)
        with self.assertRaises(RuntimeError):
            self.worker._require_prepared(self.paths, self.root / "source-sha-two")

    def test_campaign_paths_share_only_parent_data_and_isolate_branch_writes(self):
        request = app.CampaignInput(
            phase="train", campaign_id="campaign-test", parent_run_id="pilot-test",
            branch="preserve", target_step=250, campaign_started_at="2026-10-10T12:00:00Z",
            request_key="train/preserve/0250", code_bundle_url="https://example.invalid/code.zip",
            code_bundle_sha256="a" * 64, parent_lora_sha256="b" * 64,
            parent_adapter_sha256="c" * 64,
        )
        campaign = app.H3LoTCampaign()
        paths = campaign._campaign_paths(request)
        parent = campaign._paths("pilot-test")
        self.assertEqual(paths["data"], parent["data"])
        self.assertEqual(paths["cache"], parent["cache"])
        self.assertEqual(paths["pairs"], parent["pairs"])
        self.assertEqual(paths["run"], self.root / "remote/campaigns/campaign-test/branches/preserve")
        self.assertNotEqual(paths["train"], parent["train"])

    def test_preview_controls_validate_budget_and_warmup(self):
        request = app.CampaignInput(
            phase="preview", campaign_id="campaign-test", parent_run_id="pilot-test",
            branch="control", target_step=500, campaign_started_at="2026-10-10T12:00:00Z",
            request_key="preview/control/0500/30steps/abcdef12",
            code_bundle_url="https://example.invalid/code.zip", code_bundle_sha256="a" * 64,
            parent_lora_sha256="b" * 64, parent_adapter_sha256="c" * 64,
            preview_warmup_sigma=0.5, preview_budget_fraction=0.5,
        )
        self.assertEqual(request.preview_warmup_sigma, 0.5)
        self.assertEqual(request.preview_budget_fraction, 0.5)

        defaults = app.CampaignInput(
            phase="preview", campaign_id="campaign-test", parent_run_id="pilot-test",
            branch="control", target_step=500, campaign_started_at="2026-10-10T12:00:00Z",
            request_key="preview/control/0500/30steps/default",
            code_bundle_url="https://example.invalid/code.zip", code_bundle_sha256="a" * 64,
            parent_lora_sha256="b" * 64, parent_adapter_sha256="c" * 64,
        )
        self.assertEqual(defaults.preview_warmup_sigma, 0.7)
        self.assertEqual(defaults.preview_budget_fraction, 0.5)

    def test_campaign_parent_requires_same_step500_hashes(self):
        request = app.CampaignInput(
            phase="qualify", campaign_id="campaign-test", parent_run_id="pilot-test",
            campaign_started_at="2026-10-10T12:00:00Z", request_key="qualify",
            code_bundle_url="https://example.invalid/code.zip", code_bundle_sha256="a" * 64,
            parent_lora_sha256="b" * 64, parent_adapter_sha256="c" * 64,
        )
        campaign = app.H3LoTCampaign()
        paths = campaign._campaign_paths(request)
        parent_run = self.root / "remote/runs/pilot-test"
        app.atomic_status(parent_run / "prep.json", {"complete": True, "training_preflight": "passed"})
        paths["data"].mkdir(parents=True)
        (paths["data"] / "manifest.csv").write_text("clip,source_id\n")
        (paths["data"] / "pairs-identity.json").write_text("{}")
        paths["cache"].mkdir(parents=True)
        for index in range(777):
            (paths["cache"] / f"clip-{index:03d}_0384x0640_minimaxh3.safetensors").touch()
            (paths["cache"] / f"clip-{index:03d}_minimaxh3_te.safetensors").touch()
        paths["pairs"].mkdir(parents=True)
        for index in range(8):
            (paths["pairs"] / f"pair-{index}.pt").touch()
        checkpoint = self.root / "remote/checkpoint"
        checkpoint.mkdir()
        (checkpoint / "lora.safetensors").write_bytes(b"lora")
        (checkpoint / "adapter.pt").write_bytes(b"adapter")
        with mock.patch.object(campaign, "_latest_step", return_value=(500, checkpoint)):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                campaign._campaign_parent(request, paths, self.root / "source")
            request.parent_lora_sha256 = hashlib.sha256(b"lora").hexdigest()
            request.parent_adapter_sha256 = hashlib.sha256(b"adapter").hexdigest()
            self.assertEqual(campaign._campaign_parent(request, paths, self.root / "source"), checkpoint)

    def test_cache_cleaner_cannot_purge_prior_batch(self):
        clips = self.paths["clips"]
        clips.mkdir(parents=True)
        (self.paths["data"] / "manifest.csv").write_text("mock metadata")
        for index in range(777):
            (clips / f"clip-{index:04d}.mp4").write_bytes(b"video")
            (clips / f"clip-{index:04d}.txt").write_text("caption")
        cache_roots = []

        def destructive_cache(command, **_kwargs):
            config = Path(command[command.index("--dataset_config") + 1]).read_text()
            import tomllib
            dataset = tomllib.loads(config)["datasets"][0]
            target = Path(dataset["cache_directory"])
            source = Path(dataset["image_directory"])
            cache_roots.append(target)
            stems = {path.stem for path in source.glob("*.mp4")}
            for path in target.glob("*.safetensors"):
                if not any(path.name.startswith(stem + "_") for stem in stems):
                    path.unlink()
            suffix = "_minimaxh3_te.safetensors" if "minimax_cache_text.py" in command[1] else "_7_minimaxh3.safetensors"
            for stem in stems:
                (target / (stem + suffix)).write_bytes(b"cache")

        with mock.patch.object(self.worker, "_run_command", side_effect=destructive_cache):
            self.worker._ensure_cache_batch(self.paths, self.root / "fizgig", self.root / "sha", None,
                                            "pilot-test", 0, 128)
            prior = self.paths["cache_batches"] / "batch-000/clip-0000_7_minimaxh3.safetensors"
            self.worker._ensure_cache_batch(self.paths, self.root / "fizgig", self.root / "sha", None,
                                            "pilot-test", 1, 128)
        self.assertTrue(prior.is_file())
        self.assertEqual(len(set(cache_roots)), 2)

    def test_stop_marker_cancels_inflight_and_watch_never_resumes(self):
        local = self.root / "local"
        request = local / "requests/train.json"
        client.atomic_json(request, {"app": "mock-app", "request_id": "req-one", "status": "running"})
        handle = mock.Mock()
        handle.request_id = "req-one"
        handle.iter_events.return_value = iter([types.SimpleNamespace(logs=[], error=None)])
        cli = mock.Mock()
        cli.stdout = json.dumps({"runners": [{"alias": "mock-app", "runner_id": "runner-one",
                                               "in_flight_requests": 1}]})
        empty = mock.Mock()
        empty.stdout = '{"runners": []}'
        with mock.patch.object(client.fal_client, "sync_client", types.SimpleNamespace(get_handle=lambda _app, _id: handle)), \
                mock.patch.object(client.fal_client, "submit") as submit, \
                mock.patch.object(client.subprocess, "run", side_effect=[cli, mock.Mock(), empty]) as fal_cli, \
                mock.patch.object(client, "terminate_active_runners", wraps=client.terminate_active_runners):
            client.stop_run(local, "mock-app")
            with self.assertRaises(client.RunStopped):
                client._watch(handle, request, "mock-app", "pilot-test")
            submit.assert_not_called()
            self.assertIn("runner-one", json.loads((local / "STOP.json").read_text())["killed_runner_ids"])
            self.assertIn("runner-one", str(fal_cli.call_args_list[1]))
        self.assertTrue((local / "STOP.json").is_file())
        self.assertTrue(handle.cancel.called)

    def test_snapshot_index_download_survives_without_gpu_recovery(self):
        checkpoint = self.snapshot(50)
        remote_objects = {}

        def mock_upload(path, _request):
            url = "https://mock.invalid/" + str(len(remote_objects))
            remote_objects[url] = Path(path).read_bytes()
            return FakeFile(url=url, file_name=Path(path).name)

        def mock_download(url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(remote_objects[url])

        with mock.patch.object(app, "upload", side_effect=mock_upload), contextlib.redirect_stdout(io.StringIO()) as output:
            self.worker._publish_checkpoint_line("LOT_H3 kind=checkpoint_committed step=50 path=unused complete=1",
                                                 "pilot-test", self.paths, self.root, None)
        marker = output.getvalue()
        self.assertIn("checkpoint_published", marker)
        local = self.root / "local"
        with mock.patch.object(client, "download_file", side_effect=mock_download), \
                mock.patch.object(client.fal_client, "submit") as submit:
            client.consume_publication(marker, local)
            submit.assert_not_called()
        artifacts = local / "artifacts/step-00000050"
        self.assertEqual((artifacts / "lora.safetensors").read_bytes(), (checkpoint / "lora.safetensors").read_bytes())
        self.assertTrue(any(artifacts.glob("*.tar.gz")))
        self.assertEqual(json.loads((local / "fal-state.json").read_text())["checkpoint_step"], 50)

    def test_publication_download_can_retry_offline_after_failure(self):
        local = self.root / "local"
        index = {"run_id": "pilot-test", "phase": "snapshot", "checkpoint_step": 50}
        marker = "LOT_FAL kind=checkpoint_published step=50 index_url=https://mock.invalid/index"
        with mock.patch.object(client, "download_file", side_effect=OSError("network unavailable")), self.assertRaises(OSError):
            client.consume_publication(marker, local)
        self.assertTrue(any((local / "published-links").glob("*.json")))
        def download(_url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(index))
        with mock.patch.object(client, "download_file", side_effect=download), \
                mock.patch.object(client.fal_client, "submit") as submit:
            client.download_saved_results(local)
            submit.assert_not_called()
        self.assertTrue((local / "artifacts/step-00000050/result.json").is_file())

    def test_app_reports_interrupted_progress_instead_of_complete(self):
        (self.paths["train"] / "baseline_metrics.json").write_text("{}")
        def train(*_args, **_kwargs):
            (self.paths["train"] / "progress.json").write_text(json.dumps({"status": "interrupted_cleanly", "step": 12}))
        def output(_run, _phase, _paths, _request, status, _source):
            return types.SimpleNamespace(status=status)
        with mock.patch.object(self.worker, "_ensure_lot_source", return_value=self.root), \
                mock.patch.object(self.worker, "_ensure_fizgig", return_value=self.root), \
                mock.patch.object(self.worker, "_ensure_lakonlab"), \
                mock.patch.object(self.worker, "_ensure_models"), \
                mock.patch.object(self.worker, "_require_prepared"), \
                mock.patch.object(self.worker, "_run_command", side_effect=train), \
                mock.patch.object(self.worker, "_build_output", side_effect=output):
            response = self.worker.run(app.Input(phase="train", run_id="pilot-test"),
                                       types.SimpleNamespace(headers={}), None)
        self.assertEqual(response.status, "interrupted_cleanly")

    def test_corrupt_published_artifact_is_rejected(self):
        payload = {"run_id": "pilot-test", "phase": "snapshot", "checkpoint_step": 50,
                   "lora_file": {"url": "mock", "file_name": "lora.safetensors"},
                   "artifact_integrity": {"lora_file": {"sha256": "bad", "bytes": 4}}}
        def download(_url, destination):
            destination.write_bytes(b"lora")
        with mock.patch.object(client, "download_file", side_effect=download), self.assertRaises(RuntimeError):
            client.save_result(self.root / "local", payload)
        self.assertFalse((self.root / "local/artifacts/step-00000050/lora.safetensors").exists())

    def test_new_source_rechecks_preflight_even_with_all_batches_cached(self):
        lot = self.root / "repo/scripts/lot"
        run_dir = lot / "runs/fal/pilot-test"
        client.atomic_json(run_dir / "fal-state.json", {
            "gpu_preflight_passed": True, "training_data_url": "mock",
            "completed_batches": list(range(7)), "batch_size": 128,
            "cache_layout_version": "per-batch-v1", "training_preflight": "passed",
            "preflight_source_sha256": "a" * 64,
        })
        result = {"phase": "prepare", "run_id": "pilot-test", "status": "preparation_complete",
                  "completed_batches": list(range(7)), "batch_index": 6, "batch_size": 128,
                  "cache_layout_version": "per-batch-v1", "training_preflight": "passed",
                  "preflight_identity": {"source_sha256": "b" * 64}, "runner_gpu": "mock RTX"}
        with mock.patch.object(client, "__file__", str(lot / "fal_h3_client.py")), \
                mock.patch.object(client, "upload_source", return_value=("mock-code", "b" * 64)), \
                mock.patch.object(client, "_watch", return_value=result), \
                mock.patch.object(client.fal_client, "submit") as submit, \
                mock.patch.object(sys, "argv", ["client", "prepare", "--run-id", "pilot-test",
                                                  "--batch-size", "128"]):
            client.main()
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(submit.call_args.args[1]["prepare_batch"], 6)

    def test_keyboard_interrupt_never_submits_recovery(self):
        lot = self.root / "repo/scripts/lot"
        run_dir = lot / "runs/fal/pilot-test"
        client.atomic_json(run_dir / "fal-state.json", {"gpu_preflight_passed": True, "training_data_url": "mock"})
        with mock.patch.object(client, "__file__", str(lot / "fal_h3_client.py")), \
                mock.patch.object(client, "upload_source", return_value=("mock-code", "a" * 64)), \
                mock.patch.object(client, "_watch", side_effect=KeyboardInterrupt), \
                mock.patch.object(client, "terminate_active_runners") as terminate, \
                mock.patch.object(client.fal_client, "submit") as submit, \
                mock.patch.object(sys, "argv", ["client", "baseline", "--run-id", "pilot-test"]):
            with self.assertRaises(KeyboardInterrupt):
                client.main()
            self.assertEqual(submit.call_count, 1)
            terminate.assert_not_called()
        self.assertTrue((run_dir / "STOP.json").is_file())


if __name__ == "__main__":
    unittest.main()
