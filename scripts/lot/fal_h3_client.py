"""Submit and monitor staged H3 LoT jobs on fal RTX PRO 6000."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import fal_client

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP = os.environ.get("LOT_H3_FAL_APP", "bryanwade/lot-h3-training")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RunStopped(RuntimeError):
    """An explicit stop must never cause a recovery or new GPU request."""


def ensure_running(run_dir: Path) -> None:
    if (run_dir / "STOP.json").is_file():
        raise RunStopped("run is stopped; use --resume-stopped to authorize a new request")


def terminate_active_runners(app: str, run_dir: Path, request_ids: set[str]) -> None:
    """Kill the sole app runner only when this run had one active request."""
    alias = app.rsplit("/", 1)[-1]
    try:
        if len(request_ids) != 1:
            print("FAL_H3 runner stop skipped: cannot map the stop to exactly one active request", flush=True)
            return
        listing = subprocess.run(
            ["uvx", "--from", "fal", "fal", "apps", "runners", alias,
             "--state", "all", "--output", "json"],
            check=True, capture_output=True, text=True, timeout=45,
        )
        runners = json.loads(listing.stdout).get("runners", [])
        active = [runner for runner in runners
                  if runner.get("alias") in {alias, app}
                  and int(runner.get("in_flight_requests", 0)) >= 1]
        # The app is configured with max_concurrency=1. If the runner list is
        # ambiguous, keep the durable stop marker and leave remote workers alone.
        if len(active) != 1 or not active[0].get("runner_id"):
            print(f"FAL_H3 runner stop skipped: found {len(active)} active app runners", flush=True)
            return
        runner_id = active[0]["runner_id"]
        subprocess.run(
            ["uvx", "--from", "fal", "fal", "runners", "kill", runner_id],
            check=True, capture_output=True, text=True, timeout=45,
        )
        print(f"FAL_H3 stopped active runner_id={runner_id} after cancellation did not end the request", flush=True)
        atomic_json(run_dir / "STOP.json", {
            **json.loads((run_dir / "STOP.json").read_text()), "killed_runner_ids": [runner_id],
        })
    except Exception as exc:  # noqa: BLE001 - preserve stop marker; surface any manual follow-up
        print(f"FAL_H3 runner stop could not be verified: {type(exc).__name__}: {exc}", flush=True)


def stop_run(run_dir: Path, app: str) -> None:
    prior_stop = json.loads((run_dir / "STOP.json").read_text()) if (run_dir / "STOP.json").is_file() else {}
    atomic_json(run_dir / "STOP.json", {**prior_stop, "stopped_unix": time.time(), "app": app})
    active_request_ids = set()
    for path in (run_dir / "requests").glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("status") in {"complete", "completed", "cancelled"} or not record.get("request_id"):
            continue
        active_request_ids.add(record["request_id"])
        try:
            fal_client.sync_client.get_handle(record.get("app", app), record["request_id"]).cancel()
            atomic_json(path, {**record, "status": "cancelled", "cancelled_unix": time.time()})
        except Exception as exc:  # noqa: BLE001 - keep cancelling other requests after SDK failure
            atomic_json(path, {**record, "cancel_error": f"{type(exc).__name__}: {exc}"})
            print(f"FAL_H3 cancellation needs checking: request_id={record['request_id']} error={exc}", flush=True)
    # A stop test, stale STOP marker, or run with no durable request record
    # must never kill an unrelated request sharing this private app alias.
    if active_request_ids:
        terminate_active_runners(app, run_dir, active_request_ids)


def download_saved_results(run_dir: Path) -> None:
    """Retry artifact fetches using saved CDN results without starting a GPU."""
    for path in sorted((run_dir / "published-links").glob("*.json")):
        reference = json.loads(path.read_text())
        consume_publication(reference["message"], run_dir)
    for path in sorted((run_dir / "artifacts").glob("*/result.json")):
        save_result(run_dir, json.loads(path.read_text()))


def consume_publication(message: str, run_dir: Path) -> None:
    match = re.search(r"LOT_FAL kind=checkpoint_published .*?index_url=(\S+)", message)
    if match is None:
        return
    url = match.group(1)
    name = hashlib.sha256(url.encode()).hexdigest() + ".json"
    atomic_json(run_dir / "published-links" / name, {"url": url, "message": message})
    index_path = run_dir / "published" / name
    download_file(url, index_path)
    save_result(run_dir, json.loads(index_path.read_text()))


def package_openvid(output: Path) -> Path:
    clips = ROOT / "data/lot_openvid_777/clips"
    manifest = ROOT / "data/lot_openvid_777/manifest.csv"
    readme = ROOT / "data/lot_openvid_777/README.md"
    videos = sorted(clips.glob("*.mp4"))
    if len(videos) != 777 or not manifest.is_file() or not readme.is_file():
        raise SystemExit("expected the curated 777 OpenVid clips, manifest, and CC BY attribution")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(manifest, "manifest.csv")
        archive.write(readme, "README.md")
        for video in videos:
            archive.write(video, f"clips/{video.name}")
            caption = video.with_suffix(".txt")
            if not caption.is_file():
                raise SystemExit(f"missing caption sidecar: {caption.name}")
            archive.write(caption, f"clips/{caption.name}")
    os.replace(temporary, output)
    return output


def package_source(output: Path) -> tuple[Path, str]:
    source_dir = ROOT / "scripts/lot"
    scripts = sorted(source_dir.glob("*.py"))
    if len(scripts) < 10:
        raise SystemExit(f"incomplete LoT source tree: {source_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in scripts:
            archive.write(path, f"scripts/lot/{path.name}")
    os.replace(temporary, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return output, digest


def upload_dataset(run_dir: Path) -> str:
    state_path = run_dir / "fal-state.json"
    state = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("training_data_url"):
            return state["training_data_url"]
    archive = package_openvid(run_dir / "openvid-777.zip")
    url = fal_client.upload_file(archive)
    state.update({"training_data_url": url, "archive_bytes": archive.stat().st_size})
    atomic_json(state_path, state)
    return url


def upload_source(run_dir: Path) -> tuple[str, str]:
    state_path = run_dir / "fal-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    archive, digest = package_source(run_dir / "lot-h3-source.zip")
    if state.get("code_bundle_url") and state.get("code_bundle_sha256") == digest:
        return state["code_bundle_url"], state["code_bundle_sha256"]
    url = fal_client.upload_file(archive)
    state.update({"code_bundle_url": url, "code_bundle_sha256": digest})
    atomic_json(state_path, state)
    return url, digest


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "lot-h3-run-tracker/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_result(run_dir: Path, result: dict[str, Any]) -> None:
    step = result.get("checkpoint_step")
    batch = result.get("batch_index")
    if step is not None:
        artifact_dir = run_dir / "artifacts" / f"step-{int(step):08d}"
    elif batch is not None:
        artifact_dir = run_dir / "artifacts" / f"prepare-batch-{int(batch):03d}"
    else:
        artifact_dir = run_dir / "artifacts" / str(result.get("phase", "result"))
    atomic_json(artifact_dir / "result.json", result)
    for key in ("lora_file", "adapter_file", "baseline_metrics_file",
                "metrics_csv_file", "metrics_log_file", "bundle_file",
                "preparation_file", "fal_progress_file"):
        item = result.get(key)
        if isinstance(item, dict) and item.get("url"):
            destination = artifact_dir / Path(item.get("file_name", key + ".bin")).name
            expected = result.get("artifact_integrity", {}).get(key)
            if not artifact_integrity_matches(destination, expected):
                download_file(item["url"], destination)
            if expected and not artifact_integrity_matches(destination, expected):
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"published artifact failed integrity check: {key}")
            if key == "bundle_file":
                with tarfile.open(destination, "r:gz") as archive:
                    if result.get("phase") == "preview":
                        prefix = "run/training/previews/"
                        for member in archive.getmembers():
                            if not member.isfile() or not member.name.startswith(prefix):
                                continue
                            if not member.name.endswith((".mp4", ".png", "manifest.json")):
                                continue
                            relative = Path(member.name.removeprefix(prefix))
                            if relative.is_absolute() or ".." in relative.parts:
                                raise RuntimeError("preview archive contains an unsafe path")
                            source = archive.extractfile(member)
                            if source is None:
                                continue
                            preview_path = run_dir / "results" / "previews" / relative
                            preview_path.parent.mkdir(parents=True, exist_ok=True)
                            preview_path.write_bytes(source.read())
                    for member in archive.getmembers():
                        if not member.isfile():
                            continue
                        name = member.name
                        leaf = Path(name).name
                        keep_training = name.startswith("run/training/") and (
                                     leaf in {"run.json", "progress.json", "baseline_metrics.json",
                                              "confirmation_progress.json",
                                             "metrics.jsonl", "entropy_calibration.pt"}
                                     or leaf.startswith(("metrics-step-", "metrics-confirmation-step-"))
                                     and leaf.endswith((".json", ".csv"))
                        )
                        keep_dataset = name in {"run/dataset/manifest.csv", "run/dataset/README.md"}
                        keep_status = name in {"run/fal-progress.json", "run/prep.json", "run/fal-history.jsonl"}
                        if not (keep_training or keep_dataset or keep_status):
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            continue
                        if keep_training:
                            destination = run_dir / "results" / leaf
                        elif keep_dataset:
                            destination = run_dir / "results" / "dataset" / leaf
                        else:
                            destination = run_dir / "results" / leaf
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(source.read())
    state_path = run_dir / "fal-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    state.update({
        "run_id": result.get("run_id"),
        "last_phase": result.get("phase"),
        "last_status": result.get("status"),
        "runner_gpu": result.get("runner_gpu", state.get("runner_gpu")),
        "gpu_setup_error": result.get("gpu_setup_error", state.get("gpu_setup_error")),
        "gpu_preflight_passed": result.get(
            "gpu_preflight_passed",
            result.get("gpu_setup_error", state.get("gpu_setup_error")) is None
            and result.get("runner_gpu", state.get("runner_gpu")) != "unverified",
        ),
        "updated_unix": time.time(),
    })
    if step is not None:
        state["checkpoint_step"] = int(step)
    if result.get("completed_batches") is not None:
        state["completed_batches"] = result["completed_batches"]
    if result.get("cache_layout_version") is not None:
        state["cache_layout_version"] = result["cache_layout_version"]
    if result.get("batch_count") is not None:
        state["batch_count"] = result["batch_count"]
    if result.get("batch_size") is not None:
        state["batch_size"] = result["batch_size"]
    if result.get("training_preflight") is not None:
        state["training_preflight"] = result["training_preflight"]
    if result.get("preflight_identity") is not None:
        state["preflight_source_sha256"] = result["preflight_identity"].get("source_sha256")
    atomic_json(state_path, state)


def artifact_integrity_matches(path: Path, expected: dict[str, Any] | None) -> bool:
    """Reuse a prior artifact download only when its saved size and digest match."""
    if not expected or not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return path.stat().st_size == expected["bytes"] and digest.hexdigest() == expected["sha256"]


def _watch(handle, state_path: Path, app: str, run_id: str) -> dict[str, Any]:
    seen: set[str] = set()
    request_id = handle.request_id
    atomic_json(state_path, {
        "app": app,
        "run_id": run_id,
        "request_id": request_id,
        "status": "submitted",
        "submitted_unix": time.time(),
    })
    print(f"FAL_H3 request_id={request_id} run_id={run_id}", flush=True)
    run_dir = state_path.parent.parent
    try:
        for update in handle.iter_events(with_logs=True, interval=5.0):
            ensure_running(run_dir)
            log_lines = getattr(update, "logs", None) or []
            for item in log_lines:
                token = json.dumps(item, sort_keys=True, default=str)
                if token not in seen:
                    seen.add(token)
                    message = item.get("message", str(item)) if isinstance(item, dict) else str(item)
                    print(message, flush=True)
                    try:
                        consume_publication(message, run_dir)
                    except Exception as exc:  # noqa: BLE001 - preserve publication URL for offline retry
                        # The index URL remains durably visible even if a download
                        # fails; the offline download phase can retry it.
                        with (run_dir / "publication-errors.jsonl").open("a") as stream:
                            stream.write(json.dumps({"message": message, "error": str(exc)}) + "\n")
                        print(f"FAL_H3 checkpoint download pending: {exc}", flush=True)
            status_name = type(update).__name__.lower()
            atomic_json(state_path, {
                "app": app,
                "run_id": run_id,
                "request_id": request_id,
                "status": status_name,
                "updated_unix": time.time(),
            })
            if getattr(update, "error", None):
                raise RuntimeError(str(update.error))
        ensure_running(run_dir)
        result = handle.get()
        if not isinstance(result, dict):
            raise TypeError(f"fal returned an unexpected result type: {type(result).__name__}")
        atomic_json(state_path, {"app": app, "run_id": run_id, "request_id": request_id,
                                "status": "complete", "completed_unix": time.time()})
        return result
    except (RunStopped, KeyboardInterrupt):
        stop_run(run_dir, app)
        raise
    except Exception:
        atomic_json(state_path, {
            "app": app,
            "run_id": run_id,
            "request_id": request_id,
            "status": "failed",
            "failed_unix": time.time(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "baseline", "train", "preview", "recover", "stop", "download"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--app", default=DEFAULT_APP, help="private Fal app endpoint")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-items", type=int, default=12)
    parser.add_argument("--preview-items", type=int, default=2)
    parser.add_argument("--preview-steps", type=int, default=12)
    parser.add_argument("--preview-split", choices=("validation", "confirmation"), default="validation")
    parser.add_argument("--no-preview", action="store_true", help="skip automatic baseline/final video previews")
    parser.add_argument("--batch-size", type=int, default=777,
                        help="cache all 777 clips together so the 32B text encoder loads once")
    parser.add_argument("--request-id", default=None, help="reattach to an already submitted Fal request")
    parser.add_argument("--resume-stopped", action="store_true", help="explicitly clear the durable stop marker")
    parser.add_argument("--recover-on-error", action="store_true", help="allow a new GPU request to export after failure")
    parser.add_argument("--check-config", action="store_true", help="validate configuration locally without cloud requests")
    args = parser.parse_args()

    if re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", args.run_id) is None:
        raise SystemExit("run id must contain lowercase letters, digits, and hyphens")
    run_dir = Path(__file__).resolve().parent / "runs" / "fal" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "fal-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if not 1 <= args.steps <= 500 or not 4 <= args.eval_items <= 24 or not 16 <= args.batch_size <= 777 \
            or not 1 <= args.preview_items <= 4 or not 2 <= args.preview_steps <= 30:
        raise SystemExit("steps/eval-items/batch-size are outside the Fal app limits")
    if args.check_config:
        clips = list((ROOT / "data/lot_openvid_777/clips").glob("*.mp4"))
        if len(clips) != 777 or any(not clip.with_suffix(".txt").is_file() for clip in clips):
            raise SystemExit("curated OpenVid clip/caption dataset is incomplete")
        print(json.dumps({"run_id": args.run_id, "phase": args.phase, "clips": len(clips),
                          "stopped": (run_dir / "STOP.json").is_file(),
                          "baseline_complete": (run_dir / "artifacts/step-00000000/baseline_metrics.json").is_file(),
                          "cloud_requests": 0}, indent=2))
        return
    if args.phase == "stop":
        stop_run(run_dir, args.app)
        print(f"FAL_H3 marked stopped run_id={args.run_id}; cancellation requested, no new GPU request submitted")
        return
    if args.phase == "download":
        download_saved_results(run_dir)
        return
    if args.resume_stopped:
        (run_dir / "STOP.json").unlink(missing_ok=True)
    ensure_running(run_dir)
    signal.signal(signal.SIGTERM, lambda _signal, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    data_url = state.get("training_data_url")
    source_url, source_sha256 = upload_source(run_dir)

    def submit_and_watch(phase: str, *, steps: int | None = None, batch: int | None = None,
                        request_id: str | None = None, preview_checkpoint_step: int | None = None) -> dict[str, Any]:
        arguments = {
            "phase": phase,
            "run_id": args.run_id,
            "training_data_url": data_url,
            "code_bundle_url": source_url,
            "code_bundle_sha256": source_sha256,
            "steps": steps or args.steps,
            "eval_items": args.eval_items,
            "batch_size": args.batch_size,
            "prepare_batch": batch,
            "preview_items": args.preview_items,
            "preview_steps": args.preview_steps,
            "preview_split": args.preview_split,
            "preview_checkpoint_step": preview_checkpoint_step,
        }
        ensure_running(run_dir)
        if request_id:
            handle = fal_client.sync_client.get_handle(args.app, request_id)
        else:
            handle = fal_client.submit(args.app, arguments)
        suffix = f"-batch-{batch:03d}" if batch is not None else f"-step-{steps:08d}" if steps else ""
        request_path = run_dir / "requests" / f"{phase}{suffix}.json"
        try:
            result = _watch(handle, request_path, args.app, args.run_id)
        except (RunStopped, KeyboardInterrupt):
            stop_run(run_dir, args.app)
            raise
        except Exception as exc:
            ensure_running(run_dir)
            if phase == "recover" or not args.recover_on_error:
                raise
            print(f"FAL_H3 request failed ({type(exc).__name__}); exporting last complete checkpoint.", flush=True)
            recovery = fal_client.submit(args.app, {
                "phase": "recover", "run_id": args.run_id,
                "code_bundle_url": source_url,
                "code_bundle_sha256": source_sha256,
                "steps": args.steps, "eval_items": args.eval_items,
                "batch_size": args.batch_size,
            })
            result = _watch(recovery, run_dir / "requests" / f"recover{suffix}.json",
                            args.app, args.run_id)
            result["failed_phase"] = phase
            result["recovered_after_request_error"] = f"{type(exc).__name__}: {exc}"
        save_result(run_dir, result)
        if result.get("status") in {"failed", "timed_out", "interrupted_cleanly", "incomplete"}:
            raise RuntimeError(f"Fal {phase} stopped with {result.get('status')}: {result.get('error', '')}; artifacts saved locally")
        return result

    if (args.phase in {"prepare", "baseline", "train", "preview"} and args.request_id is None
            and not state.get("gpu_preflight_passed")):
        probe = submit_and_watch("recover")
        if probe.get("gpu_setup_error") or probe.get("runner_gpu") == "unverified":
            raise RuntimeError(f"Fal RTX PRO 6000 preflight failed: {probe.get('gpu_setup_error')}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
    data_url = state.get("training_data_url")
    if args.phase in {"prepare", "baseline", "train"} and not data_url:
        data_url = upload_dataset(run_dir)

    if args.request_id and args.phase not in {"prepare", "train"}:
        result = submit_and_watch(args.phase, request_id=args.request_id)
    elif args.phase == "prepare":
        if state.get("preflight_source_sha256") != source_sha256:
            state["training_preflight"] = None
            current_state = json.loads(state_path.read_text())
            current_state["training_preflight"] = None
            atomic_json(state_path, current_state)
        if state.get("batch_size") not in (None, args.batch_size) and args.batch_size < int(state["batch_size"]):
            raise SystemExit("prepare cannot shrink a partially cached isolated batch")
        if state.get("cache_layout_version") != "per-batch-v1":
            print("FAL_H3 migrating to isolated per-batch cache directories; revalidating saved batches", flush=True)
            state["completed_batches"] = []
            state["training_preflight"] = None
            state["cache_layout_version"] = "per-batch-v1"
            atomic_json(state_path, state)
        total_batches = (777 + args.batch_size - 1) // args.batch_size
        completed = set(state.get("completed_batches", []))
        for batch in range(total_batches):
            if batch in completed and not (
                batch == total_batches - 1 and state.get("training_preflight") != "passed"
            ):
                continue
            batch_request = args.request_id if args.request_id and batch == min(
                set(range(total_batches)) - completed, default=batch
            ) else None
            for attempt in range(1, 4):
                result = submit_and_watch("prepare", batch=batch, request_id=batch_request)
                batch_request = None
                completed = set(result.get("completed_batches", []))
                if batch in completed:
                    break
                if not result.get("recovered_after_request_error"):
                    raise RuntimeError(
                        f"Fal did not record preparation batch {batch} as complete; "
                        "recovery artifacts are saved locally"
                    )
                if attempt < 3:
                    print(
                        f"FAL_H3 cache batch {batch} failed on attempt {attempt}; "
                        "checkpoint state recovered, retrying resumably",
                        flush=True,
                    )
            else:
                raise RuntimeError(
                    f"preparation batch {batch} failed three times; "
                    "recovery artifacts are saved locally"
                )
        final_state = json.loads((run_dir / "fal-state.json").read_text(encoding="utf-8"))
        if final_state.get("training_preflight") != "passed":
            raise RuntimeError(
                "all cache batches finished, but the H3 backward preflight did not; "
                "rerun prepare to retry the final extent-pair/preflight step"
            )
        result = json.loads((run_dir / "artifacts" / f"prepare-batch-{total_batches - 1:03d}" /
                             "result.json").read_text(encoding="utf-8"))
    elif args.phase == "train":
        target = args.steps
        current = int(state.get("checkpoint_step", 0))
        pending_request_id = args.request_id
        recoveries = 0
        while current < target:
            next_step = min(target, current + 250)
            result = submit_and_watch("train", steps=next_step, request_id=pending_request_id)
            pending_request_id = None
            next_current = result.get("checkpoint_step")
            if next_current is None or int(next_current) <= current:
                raise RuntimeError(
                    f"training stopped at step {next_current}; last saved step is {current}. "
                    "The complete metrics and LoRA are in the recovery artifacts."
                )
            current = int(next_current)
            if result.get("recovered_after_request_error"):
                recoveries += 1
                if recoveries > 2:
                    raise RuntimeError("training request failed repeatedly; latest complete checkpoint was exported")
        result = json.loads((run_dir / "artifacts" / f"step-{current:08d}" /
                             "result.json").read_text(encoding="utf-8"))
        if not args.no_preview:
            preview = submit_and_watch("preview")
            result["final_preview_status"] = preview.get("status")
    elif args.phase == "baseline":
        result = submit_and_watch("baseline", request_id=args.request_id)
        if result.get("status") == "baseline_complete" and not args.no_preview:
            preview = submit_and_watch("preview", preview_checkpoint_step=0)
            result["baseline_preview_status"] = preview.get("status")
    else:
        result = submit_and_watch(args.phase, request_id=args.request_id)
    if result.get("gpu_setup_error"):
        raise RuntimeError(f"Fal RTX PRO 6000 preflight failed: {result['gpu_setup_error']}")
    print(f"FAL_H3 phase={args.phase} status={result.get('status')} "
          f"step={result.get('checkpoint_step')} artifacts={run_dir}", flush=True)


if __name__ == "__main__":
    main()
