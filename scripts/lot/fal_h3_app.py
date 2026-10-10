"""Private fal Serverless worker for durable H3 LoT preparation and training."""

import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import ClassVar, Literal

import fal
from fal.toolkit import File, clone_repository, download_file
from fal.toolkit.file.providers.fal import FalFileRepositoryV3
from fastapi import Request, Response
from pydantic import BaseModel, Field

APP_NAME = "lot-h3-training"
MACHINE_TYPE = "GPU-RTXPRO6000"
NUM_GPUS = 1
REQUEST_TIMEOUT = 7_200
CAMPAIGN_APP_NAME = "lot-h3-campaign"
CAMPAIGN_REQUEST_TIMEOUT = 2_700
CAMPAIGN_STARTUP_TIMEOUT = 900
EXPORT_RESERVE_SECONDS = 900
COMMAND_GRACE_SECONDS = 180
FIZGIG_URL = "https://github.com/johndpope/Fizgig.git"
FIZGIG_COMMIT = "b7f485fbfa23af5be50ae85453a4badd9a27968b"
LAKONLAB_URL = "https://github.com/Lakonik/LakonLab.git"
LAKONLAB_COMMIT = "d62218dd448d84d6e0b8e194912e452a3e1d22f8"
HF_REPO = "Comfy-Org/MiniMax-H3"
HF_REVISION = "dc559027db79c174125df4d827db55cd11178860"
ASSETS = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors":
        "/media/2TB/Fizgig/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors":
        "/media/2TB/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors":
        "/media/2TB/minimax-h3-nvfp4/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
}
EXPECTED_BYTES = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20_970_379_616,
    "vae/minimax_h3_video_vae_fp16.safetensors": 5_207_808_496,
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15_687_142_551,
}
DATA_ROOT = Path("/data/lot-h3")


class Input(BaseModel):
    phase: Literal["prepare", "baseline", "train", "preview", "recover"]
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    training_data_url: str | None = None
    code_bundle_url: str | None = None
    code_bundle_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    prepare_batch: int | None = Field(default=None, ge=0, le=48)
    batch_size: int = Field(default=777, ge=16, le=777)
    steps: int = Field(default=500, ge=1, le=500)
    eval_items: int = Field(default=12, ge=4, le=24)
    preview_items: int = Field(default=2, ge=1, le=4)
    preview_steps: int = Field(default=12, ge=2, le=30)
    preview_checkpoint_step: int | None = Field(default=None, ge=0, le=500)
    preview_split: Literal["validation", "confirmation"] = "validation"


class Output(BaseModel):
    run_id: str
    phase: str
    status: str
    checkpoint_step: int | None = None
    runner_gpu: str
    gpu_setup_error: str | None = None
    lora_file: File | None = None
    adapter_file: File | None = None
    baseline_metrics_file: File | None = None
    metrics_csv_file: File | None = None
    metrics_log_file: File | None = None
    bundle_file: File | None = None
    preparation_file: File | None = None
    fal_progress_file: File | None = None
    completed_batches: list[int] = Field(default_factory=list)
    cache_layout_version: str | None = None
    batch_count: int | None = None
    batch_index: int | None = None
    batch_size: int | None = None
    training_preflight: str | None = None
    preflight_identity: dict | None = None
    artifact_integrity: dict = Field(default_factory=dict)
    error: str | None = None
    request_key: str | None = None
    campaign_id: str | None = None


class CampaignInput(BaseModel):
    phase: Literal["qualify", "train", "preview", "confirm"]
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,55}$")
    parent_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    branch: Literal["control", "detail", "dense", "preserve", "parent"] | None = None
    recipe: Literal["sparse50", "detail50", "predicted50", "dense_control", "dense_detail"] = "sparse50"
    initial_campaign_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    initial_branch: Literal["control", "detail", "dense", "preserve"] | None = None
    initial_step: int | None = Field(default=None, ge=0, le=4_000)
    initial_lora_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    initial_adapter_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    target_step: int = Field(default=0, ge=0, le=4_000)
    optimizer_budget_seconds: int | None = Field(default=None, ge=1, le=2_700)
    preview_step: int | None = Field(default=None, ge=0, le=4_000)
    campaign_started_at: str
    request_key: str = Field(pattern=r"^[a-z0-9][a-z0-9/-]{2,95}$")
    code_bundle_url: str
    code_bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_lora_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_adapter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_split: Literal["validation", "confirmation"] = "validation"
    eval_items: int = Field(default=12, ge=4, le=24)
    preview_items: int = Field(default=2, ge=1, le=4)
    preview_steps: int = Field(default=8, ge=2, le=30)
    preview_seed: int = Field(default=20261009, ge=0, le=2**31 - 1)
    preview_controls: str | None = None
    preview_split: Literal["validation", "confirmation"] = "validation"
    preview_warmup_sigma: float = Field(default=0.7, ge=0.0, le=1.0)
    preview_budget_fraction: float = Field(default=0.5, gt=0.0, le=1.0)


@fal.cached
def get_repository() -> FalFileRepositoryV3:
    return FalFileRepositoryV3()


def upload(path: Path, request: Request) -> File:
    return File.from_path(path=str(path), request=request, repository=get_repository())


def atomic_status(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_command(command: list[str], *, env: dict[str, str], cwd: Path, log_path: Path,
                deadline: float | None = None, grace_seconds: float = COMMAND_GRACE_SECONDS,
                on_line=None) -> None:
    """Bound silent subprocesses too; give the entire process group time to save."""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("request command budget is exhausted; export artifacts before retrying")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("LOT_FAL kind=command_start argv=" + json.dumps(command), flush=True)
    process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    lines = queue.Queue()

    def read_output():
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    expired = threading.Event()
    finished = threading.Event()

    def enforce_deadline():
        if deadline is None or finished.wait(max(0, deadline - time.monotonic())):
            return
        expired.set()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if not finished.wait(grace_seconds):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    watchdog = threading.Thread(target=enforce_deadline, daemon=True)
    watchdog.start()
    output_done = False
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        try:
            while not output_done or process.poll() is None:
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    output_done = True
                    continue
                log.write(line)
                print(line, end="", flush=True)
                if on_line is not None:
                    on_line(line)
            code = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise
        finally:
            finished.set()
            watchdog.join(timeout=1)
            reader.join(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
            log.flush()
            os.fsync(log.fileno())
    if expired.is_set():
        raise TimeoutError("command reached its wall-clock budget; last complete checkpoint retained")
    if code:
        raise RuntimeError(f"command exited {code}: {command[0]}")
    print(f"LOT_FAL kind=command_done code=0 log={log_path}", flush=True)


class H3LoTTraining(
    fal.App,
    name=APP_NAME,
    keep_alive=0,
    request_timeout=REQUEST_TIMEOUT,
    startup_timeout=3600,
):
    app_auth = "private"
    machine_type = MACHINE_TYPE
    num_gpus = NUM_GPUS
    min_concurrency = 0
    max_concurrency = 1
    requirements: ClassVar[list[str]] = [
        "torch==2.10.0",
        "torchvision==0.25.0",
        "accelerate==1.6.0",
        "diffusers==0.32.1",
        "safetensors==0.5.3",
        "einops==0.7.0",
        "transformers==4.57.6",
        "tokenizers==0.22.2",
        "sentencepiece==0.2.1",
        "bitsandbytes==0.48.2",
        "comfy-kitchen==0.2.31",
        "toml==0.10.2",
        "voluptuous==0.15.2",
        "pillow==12.3.0",
        "numpy>=1.26.4,<3",
        "opencv-python==4.10.0.84",
        "av==14.2.0",
        "imageio-ffmpeg>=0.5",
        "tqdm==4.67.1",
        "huggingface-hub==0.34.3",
        "hf_xet>=1.0",
        "pyyaml==6.0.3",
        "packaging==25.0",
        "psutil>=5.9",
    ]

    def setup(self) -> None:
        self.runner_gpu = "unverified"
        self.gpu_setup_error = None
        try:
            import torch

            available = torch.cuda.is_available()
            count = torch.cuda.device_count()
            if not available or count != 1:
                self.gpu_setup_error = f"expected one CUDA device, available={available}, count={count}"
            else:
                properties = torch.cuda.get_device_properties(0)
                self.runner_gpu = properties.name
                if "RTX PRO 6000" not in self.runner_gpu.upper():
                    self.gpu_setup_error = f"requested {MACHINE_TYPE}, received {self.runner_gpu}"
                print(
                    f"LOT_FAL kind=gpu_ready name={self.runner_gpu} "
                    f"vram_gb={properties.total_memory / 2**30:.1f} cuda={torch.version.cuda}",
                    flush=True,
                )
        # Preserve app health so the recovery endpoint can export checkpoints and
        # so the next request reports the original CUDA/driver failure explicitly.
        except Exception as exc:  # noqa: BLE001 - preserve health and diagnostics on runtime/SDK errors
            self.gpu_setup_error = f"CUDA preflight could not load torch/device: {type(exc).__name__}: {exc}"
        print(
            f"LOT_FAL kind=gpu_setup available={self.gpu_setup_error is None} "
            f"name={self.runner_gpu} error={self.gpu_setup_error or ''}",
            flush=True,
        )

    def _preflight_identity(self, lot_source: Path) -> dict:
        return {
            "source_sha256": lot_source.name,
            "fizgig_commit": FIZGIG_COMMIT,
            "lakonlab_commit": LAKONLAB_COMMIT,
            "hf_revision": HF_REVISION,
            "requirements": self.requirements,
            "machine_type": MACHINE_TYPE,
            "layout_policy": "entropy", "random_layout_p": 0.1, "gradient_checkpointing": True,
            "swap": 0, "shift": 3, "dense_shift": 12, "distill": 1, "data_weight": 1,
            "holdout": 36, "eval_items": 12, "split_seed": 0,
        }

    def _record_status(self, paths: dict[str, Path], value: dict) -> None:
        value = {**value, "recorded_unix": time.time()}
        history = paths["run"] / "fal-history.jsonl"
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_status(paths["status"], value)

    def _run_command(self, command: list[str], **kwargs) -> None:
        deadline = getattr(self, "_command_deadline", None)
        env = dict(kwargs["env"])
        if deadline is not None:
            env["LOT_H3_MAX_SECONDS"] = str(max(1, int(deadline - time.monotonic() - 300)))
        kwargs["env"] = env
        run_command(command, deadline=deadline,
                    on_line=getattr(self, "_checkpoint_callback", None), **kwargs)

    def _publish_checkpoint_line(self, line: str, run_id: str, paths: dict[str, Path],
                                 lot_source: Path, request: Request) -> None:
        if "LOT_H3 kind=checkpoint_committed " not in line:
            return
        try:
            step, checkpoint = self._latest_step(paths["train"], lot_source)
            if checkpoint is None:
                raise RuntimeError("committed checkpoint could not be resolved")
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            index = {"run_id": run_id, "phase": "snapshot", "status": "snapshot_published",
                     "checkpoint_step": step, "runner_gpu": self.runner_gpu,
                     "gpu_setup_error": self.gpu_setup_error, "artifact_integrity": {}}
            bundle = self._archive(paths["run"], paths["train"], checkpoint, f"{run_id}-step-{step:08d}")
            for key, path in (("lora_file", checkpoint / "lora.safetensors"),
                              ("adapter_file", checkpoint / "adapter.pt"),
                              ("bundle_file", bundle)):
                item = upload(path, request)
                index[key] = item.model_dump(mode="json")
                if path.name in manifest["files"]:
                    index["artifact_integrity"][key] = manifest["files"][path.name]
                else:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        while block := stream.read(8 * 1024 * 1024):
                            digest.update(block)
                    index["artifact_integrity"][key] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
            index_path = DATA_ROOT / "exports" / f"{run_id}-step-{step:08d}-index.json"
            atomic_status(index_path, index)
            index_file = upload(index_path, request)
            print(f"LOT_FAL kind=checkpoint_published step={step} index_url={index_file.url}", flush=True)
        except Exception as exc:  # noqa: BLE001 - preserve health and diagnostics on runtime/SDK errors
            # Upload failures do not destroy the local complete snapshot; record
            # them explicitly so a successful training step cannot hide the risk.
            self._record_status(paths, {"run_id": run_id, "status": "checkpoint_publication_failed",
                                       "error": f"{type(exc).__name__}: {exc}"})
            print(f"LOT_FAL kind=checkpoint_publication_failed error={type(exc).__name__}", flush=True)

    def _paths(self, run_id: str) -> dict[str, Path]:
        data_root = DATA_ROOT / "datasets" / run_id
        run_root = DATA_ROOT / "runs" / run_id
        return {
            "data": data_root,
            "clips": data_root / "clips",
            "cache": data_root / "cache",
            "cache_batches": data_root / "cache-batches",
            "pairs": data_root / "pairs",
            "pair_source": data_root / "pair-source",
            "run": run_root,
            "train": run_root / "training",
            "status": run_root / "fal-progress.json",
        }

    def _link(self, link_path: Path, target: Path) -> None:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink():
            if link_path.resolve() == target.resolve():
                return
            link_path.unlink()
        elif link_path.exists():
            raise RuntimeError(f"unexpected file blocks Fal model path: {link_path}")
        link_path.symlink_to(target)

    def _ensure_fizgig(self) -> Path:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        clone_root = DATA_ROOT / "Fizgig"
        fizgig_root = clone_root / f"Fizgig-{FIZGIG_COMMIT[:8]}"
        source = fizgig_root / "src"
        if not (source / "fizgig" / "minimax" / "model.py").is_file():
            if fizgig_root.exists():
                shutil.rmtree(fizgig_root)
            clone_repository(
                FIZGIG_URL,
                commit_hash=FIZGIG_COMMIT,
                target_dir=clone_root,
            )
        if not source.is_dir():
            raise RuntimeError(f"Fizgig source not found after pinned clone: {source}")
        revision = subprocess.check_output(
            ["git", "-C", str(fizgig_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != FIZGIG_COMMIT:
            raise RuntimeError(f"Fizgig revision mismatch: {revision} != {FIZGIG_COMMIT}")
        self._link(Path("/media/2TB/Fizgig"), fizgig_root)
        return source

    def _ensure_lakonlab(self) -> Path:
        clone_root = DATA_ROOT / "LakonLab"
        repository = clone_root / f"LakonLab-{LAKONLAB_COMMIT[:8]}"
        common = repository / "lakonlab/models/architectures/asymflow/common.py"
        if not common.is_file():
            if repository.exists():
                shutil.rmtree(repository)
            clone_repository(
                LAKONLAB_URL,
                commit_hash=LAKONLAB_COMMIT,
                target_dir=clone_root,
            )
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != LAKONLAB_COMMIT:
            raise RuntimeError(f"LakonLab revision mismatch: {revision} != {LAKONLAB_COMMIT}")
        if not common.is_file():
            raise RuntimeError(f"LakonLab asymmetric-flow source is missing: {common}")
        return repository

    def _ensure_lot_source(self, url: str | None, expected_sha256: str | None) -> Path:
        if not url or not expected_sha256:
            raise RuntimeError("request must include the uploaded LoT source bundle and SHA-256")
        root = DATA_ROOT / "lot-source" / expected_sha256
        if (root / "scripts/lot/train_h3.py").is_file():
            return root
        inbox = DATA_ROOT / "incoming"
        inbox.mkdir(parents=True, exist_ok=True)
        archive = Path(download_file(url, target_dir=str(inbox)))
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            archive.unlink(missing_ok=True)
            raise RuntimeError("uploaded LoT source bundle failed SHA-256 verification")
        staging = root.with_name(root.name + ".extracting")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as source:
                for info in source.infolist():
                    resolved = (staging / info.filename).resolve()
                    if staging.resolve() not in resolved.parents and resolved != staging.resolve():
                        raise RuntimeError(f"LoT source zip contains an unsafe path: {info.filename}")
                source.extractall(staging)
            if not (staging / "scripts/lot/train_h3.py").is_file():
                raise RuntimeError("LoT source bundle is missing scripts/lot/train_h3.py")
            root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, root)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            archive.unlink(missing_ok=True)
        print(f"LOT_FAL kind=source_ready sha256={expected_sha256} path={root}", flush=True)
        return root

    def _ensure_models(self) -> dict[str, Path]:
        os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        from huggingface_hub import hf_hub_download

        model_files = {}
        for filename, destination in ASSETS.items():
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            cached = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                revision=HF_REVISION,
                cache_dir=str(DATA_ROOT / "hf"),
            )
            cached_path = Path(cached)
            expected = EXPECTED_BYTES[filename]
            if cached_path.stat().st_size != expected:
                raise RuntimeError(
                    f"wrong file size for {filename}: {cached_path.stat().st_size} != {expected}"
                )
            self._link(path, cached_path)
            model_files[filename] = cached_path
            print(f"LOT_FAL kind=model_ready file={filename} bytes={expected}", flush=True)
        return model_files

    def _dataset_from_url(self, url: str, clips_dir: Path) -> None:
        if (clips_dir.is_dir() and len(list(clips_dir.glob("*.mp4"))) == 777
                and (clips_dir.parent / "manifest.csv").is_file()
                and (clips_dir.parent / "README.md").is_file()):
            return
        clips_dir.parent.mkdir(parents=True, exist_ok=True)
        if clips_dir.exists():
            shutil.rmtree(clips_dir)
        incoming = DATA_ROOT / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        archive = Path(download_file(url, target_dir=str(incoming)))
        stage = clips_dir.parent / f".{clips_dir.name}.extracting"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as source:
                for info in source.infolist():
                    resolved = (stage / info.filename).resolve()
                    if stage.resolve() not in resolved.parents and resolved != stage.resolve():
                        raise RuntimeError(f"dataset zip contains an unsafe path: {info.filename}")
                source.extractall(stage)
            extracted_clips = next(stage.rglob("clips"), None)
            if extracted_clips is None or len(list(extracted_clips.glob("*.mp4"))) != 777:
                raise RuntimeError("dataset archive must contain exactly 777 clip files in clips/")
            for name in ("manifest.csv", "README.md"):
                source_path = stage / name
                if not source_path.is_file():
                    raise RuntimeError(f"dataset archive is missing {name}")
                os.replace(source_path, clips_dir.parent / name)
            os.replace(extracted_clips, clips_dir)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            archive.unlink(missing_ok=True)

    def _write_dataset_config(
        self, paths: dict[str, Path], batch_dir: Path, cache_dir: Path
    ) -> Path:
        config = batch_dir / "dataset.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        batch_clips = batch_dir / "clips"
        config.write_text(
            "[general]\nresolution = [640, 384]\nbatch_size = 1\n"
            "enable_bucket = true\nbucket_no_upscale = true\n"
            'caption_extension = ".txt"\nnum_repeats = 1\n\n'
            "[[datasets]]\n"
            f'image_directory = "{batch_clips}"\n'
            f'cache_directory = "{cache_dir}"\n',
            encoding="utf-8",
        )
        return config

    def _ensure_cache_batch(self, paths: dict[str, Path], source: Path, lot_source: Path, url: str | None,
                            run_id: str, batch_index: int, batch_size: int) -> dict:
        if not paths["clips"].is_dir() or not (paths["data"] / "manifest.csv").is_file():
            if not url:
                raise RuntimeError("first preparation request needs training_data_url")
            self._dataset_from_url(url, paths["clips"])
        clips = sorted(paths["clips"].glob("*.mp4"))
        if len(clips) != 777:
            raise RuntimeError(f"expected 777 clips, found {len(clips)}")
        batches = (len(clips) + batch_size - 1) // batch_size
        if batch_index >= batches:
            raise ValueError(f"prepare batch {batch_index} outside [0, {batches - 1}]")
        prep_path = paths["run"] / "prep.json"
        prior = json.loads(prep_path.read_text()) if prep_path.is_file() else {}
        if prior.get("cache_layout_version") != "per-batch-v1":
            prior = {
                "run_id": run_id,
                "batch_size": prior.get("batch_size"),
                "completed_batches": [],
                "cache_layout_version": "per-batch-v1",
            }
        if prior.get("batch_size") not in (None, batch_size):
            if batch_size < int(prior["batch_size"]):
                raise RuntimeError("prepare cannot shrink a partially cached isolated batch")
            print(f"LOT_FAL kind=cache_batch_resize from={prior['batch_size']} to={batch_size} "
                  "existing latent/text files will be verified and reused", flush=True)
        identity = self._preflight_identity(lot_source)
        if prior.get("preflight_identity") != identity:
            prior["training_preflight"] = None
        if (prior.get("complete") and prior.get("training_preflight") == "passed"
                and batch_index in prior.get("completed_batches", [])):
            return prior
        start = batch_index * batch_size
        batch = clips[start:start + batch_size]
        batch_dir = paths["data"] / "cache-input" / f"batch-{batch_index:03d}"
        batch_cache = paths["cache_batches"] / f"batch-{batch_index:03d}"
        batch_cache.mkdir(parents=True, exist_ok=True)
        batch_clips = batch_dir / "clips"
        batch_clips.mkdir(parents=True, exist_ok=True)
        for clip in batch:
            for source_path in (clip, clip.with_suffix(".txt")):
                if not source_path.is_file():
                    raise RuntimeError(f"missing source clip/caption: {source_path.name}")
                target = batch_clips / source_path.name
                if not target.exists():
                    target.symlink_to(source_path)
        config = self._write_dataset_config(paths, batch_dir, batch_cache)
        missing_latents = [
            clip for clip in batch
            if not list(batch_cache.glob(f"{clip.stem}_*_minimaxh3.safetensors"))
        ]
        if missing_latents:
            command = [
                sys.executable,
                str(source / "fizgig/scripts/minimax_cache_latents.py"),
                "--dataset_config", str(config),
                "--vae", ASSETS["vae/minimax_h3_video_vae_fp16.safetensors"],
                "--skip_existing",
                "--num_workers", "8",
            ]
            self._run_command(command, env=self._fizgig_env(source), cwd=source,
                        log_path=paths["run"] / "logs" / f"prepare-latents-{batch_index:03d}.log")
        missing_text = [
            clip for clip in batch
            if not (batch_cache / f"{clip.stem}_minimaxh3_te.safetensors").is_file()
        ]
        if missing_text:
            command = [
                sys.executable,
                str(source / "fizgig/scripts/minimax_cache_text.py"),
                "--dataset_config", str(config),
                "--text_encoder", ASSETS["text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
                "--skip_existing",
                "--num_workers", "8",
            ]
            self._run_command(command, env=self._fizgig_env(source), cwd=source,
                        log_path=paths["run"] / "logs" / f"prepare-text-{batch_index:03d}.log")
        cache_dirs = [
            paths["cache_batches"] / f"batch-{index:03d}" for index in range(batches)
        ]
        latent_count = sum(
            any(list(cache.glob(f"{clip.stem}_*_minimaxh3.safetensors")) for cache in cache_dirs)
            for clip in clips
        )
        text_count = sum(
            any((cache / f"{clip.stem}_minimaxh3_te.safetensors").is_file() for cache in cache_dirs)
            for clip in clips
        )
        batch_complete = all(
            list(batch_cache.glob(f"{clip.stem}_*_minimaxh3.safetensors"))
            and (batch_cache / f"{clip.stem}_minimaxh3_te.safetensors").is_file()
            for clip in batch
        )
        if not batch_complete:
            raise RuntimeError(
                f"cache batch {batch_index} incomplete: {latent_count}/777 latents, "
                f"{text_count}/777 text files"
            )
        completed = set(prior.get("completed_batches", []))
        completed.add(batch_index)
        prep = {
            "run_id": run_id,
            "clips": len(clips),
            "latents": latent_count,
            "text": text_count,
            "cache": str(paths["cache"]),
            "cache_layout_version": "per-batch-v1",
            "batch_size": batch_size,
            "batch_count": batches,
            "completed_batches": sorted(completed),
            "complete": len(completed) == batches and latent_count == 777 and text_count == 777,
            "training_preflight": prior.get("training_preflight"),
            "preflight_identity": identity,
        }
        atomic_status(prep_path, prep)
        if not prep["complete"]:
            print(f"LOT_FAL kind=cache_batch batch={batch_index + 1}/{batches} "
                  f"latents={latent_count}/777 text={text_count}/777", flush=True)
            return prep

        self._assemble_cache(paths, cache_dirs)
        self._ensure_pairs(paths, source, lot_source)
        env = self._fizgig_env(source)
        command = [
            sys.executable,
            str(lot_source / "scripts/lot/train_h3.py"),
            "--pairs", str(paths["pairs"]),
            "--cache", str(paths["cache"]),
            "--data-manifest", str(paths["data"] / "manifest.csv"),
            "--holdout", "36",
            "--random-layout-p", "0.1",
            "--layout-policy", "entropy",
            "--gradient-checkpointing",
            "--swap", "0",
            "--shift", "3",
            "--dense-shift", "12",
            "--distill", "1",
            "--data-weight", "1",
            "--check",
        ]
        self._run_command(command, env=env, cwd=lot_source, log_path=paths["run"] / "logs" / "preflight.log")
        prep["training_preflight"] = "passed"
        atomic_status(prep_path, prep)
        return prep

    def _assemble_cache(self, paths: dict[str, Path], cache_dirs: list[Path]) -> None:
        target = paths["cache"]
        staging = target.with_name(target.name + ".assembling")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for cache_dir in cache_dirs:
            if not cache_dir.is_dir():
                raise RuntimeError(f"batch cache directory is missing: {cache_dir}")
            for source in cache_dir.glob("*.safetensors"):
                destination = staging / source.name
                if destination.exists():
                    if source.name.startswith("uncond_"):
                        continue
                    raise RuntimeError(f"duplicate cache artifact across batches: {source.name}")
                destination.symlink_to(source.resolve())
        latent_count = len(list(staging.glob("*_minimaxh3.safetensors")))
        text_count = len([
            path for path in staging.glob("*_minimaxh3_te.safetensors")
            if not path.name.startswith("uncond_")
        ])
        if latent_count != 777 or text_count != 777:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(
                f"assembled cache is incomplete: {latent_count}/777 latents, "
                f"{text_count}/777 text files"
            )
        if target.exists():
            legacy = paths["data"] / "cache-legacy-shared"
            if legacy.exists():
                shutil.rmtree(target)
            else:
                os.replace(target, legacy)
        os.replace(staging, target)

    def _fizgig_env(self, source: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
        env["HF_HOME"] = str(DATA_ROOT / "hf")
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["LAKONLAB_ROOT"] = str(
            DATA_ROOT / "LakonLab" / f"LakonLab-{LAKONLAB_COMMIT[:8]}"
        )
        return env

    def _ensure_pairs(self, paths: dict[str, Path], source: Path, lot_source: Path) -> None:
        import cv2

        source_dir = paths["pair_source"]
        source_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(lot_source / "scripts/lot"))
        from training_experiment import balanced_records, read_manifest, split_records

        metadata = read_manifest(paths["data"] / "manifest.csv")
        records = [{**metadata[clip.stem], "clip_path": str(clip)}
                   for clip in sorted(paths["clips"].glob("*.mp4"))]
        train_records, _ = split_records(records, holdout=36, seed=0)
        selected = balanced_records(train_records, 64)
        if len(selected) < 24:
            raise RuntimeError("need at least 24 training clips for independent extent-pair calibration")
        pair_identity = {"source_sha256": lot_source.name,
                         "training_clips": [record["clip_path"] for record in selected]}
        pair_manifest = paths["data"] / "pairs-identity.json"
        if (paths["pairs"].is_dir() and len(list(paths["pairs"].glob("*.pt"))) == 8
                and pair_manifest.is_file() and json.loads(pair_manifest.read_text()) == pair_identity):
            return
        if paths["pairs"].exists():
            shutil.rmtree(paths["pairs"])
        # The pair source must contain only the current train split; remove old
        # frames that could belong to this run's held-out video sources.
        shutil.rmtree(source_dir, ignore_errors=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        for record in selected:
            clip = Path(record["clip_path"])
            frame_dir = source_dir / clip.stem
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / "comfy_t00.png"
            if frame_path.is_file():
                continue
            capture = cv2.VideoCapture(str(clip))
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None:
                raise RuntimeError(f"could not read a first frame from {clip.name}")
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"could not write calibration frame for {clip.name}")

        staging = paths["data"] / "pairs.staging"
        shutil.rmtree(staging, ignore_errors=True)
        command = [
            sys.executable,
            str(lot_source / "scripts/lot/make_pairs_h3.py"),
            "--src", str(source_dir),
            "--limit", str(len(selected)),
            "--out", str(staging),
            "--vae", ASSETS["vae/minimax_h3_video_vae_fp16.safetensors"],
        ]
        self._run_command(command, env=self._fizgig_env(source), cwd=lot_source,
                    log_path=paths["run"] / "logs" / "prepare-pairs.log")
        if len(list(staging.glob("*.pt"))) != 8:
            raise RuntimeError("extent pair job did not produce all 8 non-unit H3 extents")
        os.replace(staging, paths["pairs"])
        atomic_status(pair_manifest, pair_identity)
        print(f"LOT_FAL kind=pairs_ready path={paths['pairs']} files=8", flush=True)

    def _require_prepared(self, paths: dict[str, Path], lot_source: Path) -> dict:
        prep_path = paths["run"] / "prep.json"
        if not prep_path.is_file():
            raise RuntimeError("preparation is incomplete; run the prepare phase first")
        prep = json.loads(prep_path.read_text(encoding="utf-8"))
        if (not prep.get("complete") or prep.get("training_preflight") != "passed"
                or prep.get("preflight_identity") != self._preflight_identity(lot_source)):
            raise RuntimeError("prepared H3 data/cache/extent preflight is not complete")
        return prep

    def _latest_step(self, run_dir: Path, lot_source: Path) -> tuple[int | None, Path | None]:
        try:
            sys.path.insert(0, str(lot_source / "scripts/lot"))
            from run_artifacts import resolve_checkpoint

            checkpoint = resolve_checkpoint(run_dir)
        except FileNotFoundError:
            return None, None
        except ValueError as exc:
            print(f"LOT_FAL kind=checkpoint_integrity_failed error={exc}", flush=True)
            return None, None
        manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        return int(manifest["step"]), checkpoint

    def _archive(self, run_root: Path, train_dir: Path, checkpoint: Path | None, name: str) -> Path:
        output = DATA_ROOT / "exports" / f"{name}.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tar.gz.tmp")
        with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
            if (train_dir / "run.json").is_file():
                archive.add(train_dir / "run.json", arcname="run/training/run.json")
            for filename in ("progress.json", "confirmation_progress.json", "baseline_metrics.json",
                             "metrics.jsonl", "entropy_calibration.pt", "latest.json"):
                path = train_dir / filename
                if path.is_file():
                    archive.add(path, arcname=f"run/training/{filename}")
            for path in sorted(train_dir.glob("metrics-step-*.json")):
                archive.add(path, arcname=f"run/training/{path.name}")
            for path in sorted(train_dir.glob("metrics-step-*.csv")):
                archive.add(path, arcname=f"run/training/{path.name}")
            for path in sorted(train_dir.glob("metrics-confirmation-step-*.json")):
                archive.add(path, arcname=f"run/training/{path.name}")
            for path in sorted(train_dir.glob("metrics-confirmation-step-*.csv")):
                archive.add(path, arcname=f"run/training/{path.name}")
            for path in sorted((train_dir / "previews").rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=f"run/training/{path.relative_to(train_dir)}")
            for path in sorted((run_root / "logs").glob("*.log")):
                archive.add(path, arcname=f"run/logs/{path.name}")
            if (run_root / "fal-progress.json").is_file():
                archive.add(run_root / "fal-progress.json", arcname="run/fal-progress.json")
            if (run_root / "prep.json").is_file():
                archive.add(run_root / "prep.json", arcname="run/prep.json")
            if (run_root / "fal-history.jsonl").is_file():
                archive.add(run_root / "fal-history.jsonl", arcname="run/fal-history.jsonl")
            dataset_root = DATA_ROOT / "datasets" / run_root.name
            for filename in ("manifest.csv", "README.md", "pairs-identity.json"):
                path = dataset_root / filename
                if path.is_file():
                    archive.add(path, arcname=f"run/dataset/{filename}")
            if checkpoint is not None:
                archive.add(checkpoint, arcname=f"run/training/checkpoints/{checkpoint.name}")
        os.replace(temporary, output)
        return output

    def _build_output(self, run_id: str, phase: str, paths: dict[str, Path],
                      request: Request, status: str, lot_source: Path) -> Output:
        step, checkpoint = self._latest_step(paths["train"], lot_source)
        archive_path = self._archive(paths["run"], paths["train"], checkpoint, f"{run_id}-{phase}")
        integrity = {}

        def upload_verified(key: str, path: Path) -> File:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            integrity[key] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
            return upload(path, request)

        archive_file = upload_verified("bundle_file", archive_path)
        lora_file = adapter_file = None
        if checkpoint is not None:
            lora_file = upload_verified("lora_file", checkpoint / "lora.safetensors")
            adapter_file = upload_verified("adapter_file", checkpoint / "adapter.pt")
        baseline_path = paths["train"] / "baseline_metrics.json"
        baseline_file = upload_verified("baseline_metrics_file", baseline_path) if baseline_path.is_file() else None
        metrics_path = paths["train"] / "metrics.jsonl"
        metrics_file = upload_verified("metrics_log_file", metrics_path) if metrics_path.is_file() else None
        csvs = sorted(paths["train"].glob("metrics-step-*.csv"))
        latest_csv = upload_verified("metrics_csv_file", csvs[-1]) if csvs else None
        prep_path = paths["run"] / "prep.json"
        prep_file = upload_verified("preparation_file", prep_path) if prep_path.is_file() else None
        fal_progress = upload_verified("fal_progress_file", paths["status"]) if paths["status"].is_file() else None
        preparation = json.loads(prep_path.read_text()) if prep_path.is_file() else {}
        return Output(
            run_id=run_id,
            phase=phase,
            status=status,
            checkpoint_step=step,
            runner_gpu=self.runner_gpu,
            gpu_setup_error=self.gpu_setup_error,
            lora_file=lora_file,
            adapter_file=adapter_file,
            baseline_metrics_file=baseline_file,
            metrics_csv_file=latest_csv,
            metrics_log_file=metrics_file,
            bundle_file=archive_file,
            preparation_file=prep_file,
            fal_progress_file=fal_progress,
            completed_batches=preparation.get("completed_batches", []),
            cache_layout_version=preparation.get("cache_layout_version"),
            batch_count=preparation.get("batch_count"),
            batch_index=preparation.get("last_batch"),
            batch_size=preparation.get("batch_size"),
            training_preflight=preparation.get("training_preflight"),
            preflight_identity=preparation.get("preflight_identity"),
            artifact_integrity=integrity,
        )

    @fal.endpoint("/")
    def run(self, request: Input, response: Response, http_request: Request) -> Output:
        paths = self._paths(request.run_id)
        lot_source = self._ensure_lot_source(request.code_bundle_url, request.code_bundle_sha256)
        started = time.perf_counter()
        self._command_deadline = time.monotonic() + REQUEST_TIMEOUT - EXPORT_RESERVE_SECONDS
        self._checkpoint_callback = lambda line: self._publish_checkpoint_line(
            line, request.run_id, paths, lot_source, http_request
        )
        paths["run"].mkdir(parents=True, exist_ok=True)
        paths["train"].mkdir(parents=True, exist_ok=True)
        self._record_status(paths, {
            "run_id": request.run_id, "phase": request.phase, "status": "starting",
            "runner_gpu": self.runner_gpu, "started_unix": time.time(),
        })
        try:
            if request.phase != "recover" and self.gpu_setup_error:
                raise RuntimeError(self.gpu_setup_error)
            if request.phase in {"prepare", "baseline", "train", "preview"}:
                source = self._ensure_fizgig()
                self._ensure_lakonlab()
                self._ensure_models()
            if request.phase == "prepare":
                if request.prepare_batch is None:
                    raise RuntimeError("prepare phase requires prepare_batch")
                prep = self._ensure_cache_batch(
                    paths, source, lot_source, request.training_data_url, request.run_id,
                    request.prepare_batch, request.batch_size,
                )
                prep["last_batch"] = request.prepare_batch
                atomic_status(paths["run"] / "prep.json", prep)
                status = "preparation_complete" if prep.get("complete") else "batch_complete"
            elif request.phase == "baseline":
                self._require_prepared(paths, lot_source)
                if not (paths["train"] / "baseline_metrics.json").is_file():
                    resume_args = (
                        ["--resume", str(paths["train"])]
                        if self._latest_step(paths["train"], lot_source)[1] is not None
                        else []
                    )
                    command = [
                        sys.executable, str(lot_source / "scripts/lot/train_h3.py"),
                        "--pairs", str(paths["pairs"]), "--cache", str(paths["cache"]),
                        "--data-manifest", str(paths["data"] / "manifest.csv"),
                        "--holdout", "36", "--random-layout-p", "0.1",
                        "--layout-policy", "entropy", "--gradient-checkpointing", "--swap", "0",
                        "--shift", "3", "--dense-shift", "12", "--distill", "1",
                        "--data-weight", "1", "--steps", str(request.steps),
                        "--eval-items", str(request.eval_items), "--eval-every", "250",
                        "--save-every", "50", "--baseline-only", *resume_args,
                        "--out", str(paths["train"]),
                    ]
                    self._run_command(command, env=self._fizgig_env(source),
                                cwd=lot_source,
                                log_path=paths["run"] / "logs" / "baseline-process.log")
                status = "baseline_complete"
            elif request.phase == "train":
                self._require_prepared(paths, lot_source)
                if not (paths["train"] / "baseline_metrics.json").is_file():
                    raise RuntimeError("baseline metrics must complete before the training phase")
                environment = self._fizgig_env(source)
                environment["LOT_H3_TRAIN"] = "1"
                command = [
                    sys.executable, str(lot_source / "scripts/lot/train_h3.py"),
                    "--pairs", str(paths["pairs"]), "--cache", str(paths["cache"]),
                        "--data-manifest", str(paths["data"] / "manifest.csv"),
                    "--holdout", "36", "--random-layout-p", "0.1",
                    "--layout-policy", "entropy", "--gradient-checkpointing", "--swap", "0",
                    "--shift", "3", "--dense-shift", "12",
                    "--distill", "1", "--data-weight", "1",
                    "--steps", str(request.steps), "--eval-items", str(request.eval_items),
                    "--eval-every", "250", "--save-every", "50", "--resume", str(paths["train"]),
                    "--out", str(paths["train"]),
                ]
                self._run_command(command, env=environment, cwd=lot_source,
                            log_path=paths["run"] / "logs" / "training-process.log")
                progress = json.loads((paths["train"] / "progress.json").read_text())
                status = "training_complete" if progress.get("status") == "complete" else progress.get("status", "incomplete")
            elif request.phase == "preview":
                self._require_prepared(paths, lot_source)
                step, checkpoint = self._latest_step(paths["train"], lot_source)
                if request.preview_checkpoint_step is not None:
                    from run_artifacts import resolve_checkpoint

                    checkpoint = resolve_checkpoint(
                        paths["train"] / "checkpoints" / f"step-{request.preview_checkpoint_step:08d}"
                    )
                    manifest = json.loads((checkpoint / "manifest.json").read_text())
                    step = int(manifest["step"])
                if checkpoint is None:
                    raise RuntimeError("preview requires an exported training checkpoint")
                preview_dir = paths["train"] / "previews" / f"step-{step:08d}"
                if (preview_dir / "manifest.json").is_file():
                    raise RuntimeError(f"preview already exists: {preview_dir}")
                command = [
                    sys.executable, str(lot_source / "scripts/lot/render_h3_pilot.py"),
                    "--run", str(paths["train"]), "--checkpoint", str(checkpoint),
                    "--items", str(request.preview_items), "--steps", str(request.preview_steps),
                    "--split", request.preview_split,
                    "--entropy-inference", "--fizgig-src", str(source / "src"),
                    "--vae", ASSETS["vae/minimax_h3_video_vae_fp16.safetensors"],
                    "--out", str(preview_dir),
                ]
                self._run_command(command, env=self._fizgig_env(source), cwd=lot_source,
                                  log_path=paths["run"] / "logs" / f"preview-{step:08d}.log")
                status = "preview_complete"
            else:
                status = "recovered"

            elapsed = time.perf_counter() - started
            self._record_status(paths, {
                "run_id": request.run_id, "phase": request.phase, "status": status,
                "runner_gpu": self.runner_gpu, "request_seconds": elapsed,
            })
            response.headers["x-lot-h3-run-seconds"] = str(round(elapsed, 3))
            print(f"LOT_FAL kind=phase_complete phase={request.phase} run_id={request.run_id} "
                  f"status={status} seconds={elapsed:.1f}", flush=True)
            return self._build_output(
                request.run_id, request.phase, paths, http_request, status, lot_source
            )
        except Exception as exc:  # noqa: BLE001 - preserve health and diagnostics on runtime/SDK errors
            step, checkpoint = self._latest_step(paths["train"], lot_source)
            self._record_status(paths, {
                "run_id": request.run_id, "phase": request.phase, "status": "failed",
                "runner_gpu": self.runner_gpu, "request_seconds": time.perf_counter() - started,
                "last_complete_step": step, "error": f"{type(exc).__name__}: {exc}",
                "recoverable": checkpoint is not None,
            })
            print(f"LOT_FAL kind=phase_failed phase={request.phase} run_id={request.run_id} "
                  f"last_complete_step={step} error={type(exc).__name__}", flush=True)
            output = self._build_output(
                request.run_id, request.phase, paths, http_request,
                "timed_out" if isinstance(exc, TimeoutError) else "failed", lot_source
            )
            output.error = f"{type(exc).__name__}: {exc}"
            return output


class H3LoTCampaign(H3LoTTraining, name=CAMPAIGN_APP_NAME, keep_alive=0,
                    request_timeout=CAMPAIGN_REQUEST_TIMEOUT,
                    startup_timeout=CAMPAIGN_STARTUP_TIMEOUT):
    """Isolated, deadline-bounded A/B service that reuses a verified pilot cache."""

    app_name = CAMPAIGN_APP_NAME
    regions: ClassVar[list[str]] = ["us-east"]

    def _campaign_paths(self, request: CampaignInput) -> dict[str, Path]:
        parent = self._paths(request.parent_run_id)
        root = DATA_ROOT / "campaigns" / request.campaign_id
        run_root = root / "branches" / (request.branch or "qualification")
        return {
            **parent,
            "run": run_root,
            "train": run_root / "training",
            "status": run_root / "fal-progress.json",
        }

    def _campaign_parent(self, request: CampaignInput, paths: dict[str, Path],
                         lot_source: Path) -> Path:
        prep_path = DATA_ROOT / "runs" / request.parent_run_id / "prep.json"
        if not prep_path.is_file():
            raise RuntimeError("parent cache preparation record is missing")
        prep = json.loads(prep_path.read_text(encoding="utf-8"))
        data, cache, pairs = paths["data"], paths["cache"], paths["pairs"]
        from training_experiment import clip_stem

        clips = list(cache.glob("*_minimaxh3.safetensors"))
        missing_text = [clip_stem(path) for path in clips
                        if not cache.joinpath(f"{clip_stem(path)}_minimaxh3_te.safetensors").is_file()]
        pair_count = len(list(pairs.glob("*.pt")))
        manifest_ready = (data / "manifest.csv").is_file() and (data / "pairs-identity.json").is_file()
        if (not prep.get("complete") or prep.get("training_preflight") != "passed"
                or len(clips) != 777 or missing_text or pair_count != 8 or not manifest_ready):
            raise RuntimeError(
                "parent cache preflight failed: "
                f"prepared={bool(prep.get('complete')) and prep.get('training_preflight') == 'passed'}, "
                f"latents={len(clips)}/777, missing_text={len(missing_text)}, "
                f"extent_pairs={pair_count}/8, manifest_identity={manifest_ready}"
            )
        parent_train = DATA_ROOT / "runs" / request.parent_run_id / "training"
        step, checkpoint = self._latest_step(parent_train, lot_source)
        if step != 500 or checkpoint is None:
            raise RuntimeError(f"campaign requires the verified parent step-500 checkpoint; found {step}")
        for filename, expected in (("lora.safetensors", request.parent_lora_sha256),
                                   ("adapter.pt", request.parent_adapter_sha256)):
            digest = hashlib.sha256()
            with (checkpoint / filename).open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError(f"remote parent {filename} SHA-256 differs from the local frozen baseline")
        initial_fields = (request.initial_campaign_id, request.initial_branch, request.initial_step,
                          request.initial_lora_sha256, request.initial_adapter_sha256)
        if any(value is not None for value in initial_fields):
            if any(value is None for value in initial_fields):
                raise RuntimeError("initial checkpoint campaign, branch, step, and both hashes are required together")
            init_dir = (DATA_ROOT / "campaigns" / request.initial_campaign_id / "branches"
                        / request.initial_branch / "training" / "checkpoints"
                        / f"step-{request.initial_step:08d}")
            for filename, expected in (("lora.safetensors", request.initial_lora_sha256),
                                       ("adapter.pt", request.initial_adapter_sha256)):
                digest = hashlib.sha256()
                with (init_dir / filename).open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(f"initial checkpoint {filename} SHA-256 differs from frozen run artifact")
            checkpoint = init_dir
            self._campaign_initial_train = init_dir.parent.parent
        else:
            self._campaign_initial_train = parent_train
        self._campaign_parent_checkpoint = checkpoint
        self._campaign_parent_train = parent_train
        return checkpoint

    def _campaign_deadline(self, request: CampaignInput) -> float:
        from datetime import datetime, timezone

        from campaign_runtime import CAMPAIGN_SECONDS, OPTIMIZER_SECONDS

        started = datetime.fromisoformat(request.campaign_started_at.replace("Z", "+00:00"))
        if started.tzinfo is None:
            raise RuntimeError("campaign_started_at must be timezone-aware")
        now_utc = datetime.now(timezone.utc)
        elapsed = (now_utc - started.astimezone(timezone.utc)).total_seconds()
        hard_limit = OPTIMIZER_SECONDS if request.phase == "train" else CAMPAIGN_SECONDS
        seconds_left = hard_limit - elapsed
        command_budget = min(CAMPAIGN_REQUEST_TIMEOUT - 600,
                             int(seconds_left))
        if command_budget <= 0:
            raise TimeoutError(f"campaign {request.phase} request is past its absolute deadline")
        return time.monotonic() + command_budget

    def _campaign_output(self, request: CampaignInput, paths: dict[str, Path],
                         http_request: Request, status: str, lot_source: Path) -> Output:
        branch_run_id = f"{request.campaign_id}-{request.branch or 'qualification'}"
        output = self._build_output(branch_run_id, f"campaign-{request.phase}", paths,
                                   http_request, status, lot_source)
        output.run_id = branch_run_id
        output.request_key = request.request_key
        output.campaign_id = request.campaign_id
        return output

    @fal.endpoint("/")
    def campaign(self, request: CampaignInput, response: Response,
                 http_request: Request) -> Output:
        self._checkpoint_callback = None
        paths = self._campaign_paths(request)
        paths["run"].mkdir(parents=True, exist_ok=True)
        paths["train"].mkdir(parents=True, exist_ok=True)
        request_record = paths["run"] / "requests" / f"{request.request_key.replace('/', '--')}.json"
        if request_record.is_file():
            recorded = json.loads(request_record.read_text(encoding="utf-8"))
            raise RuntimeError(f"request key already recorded ({recorded.get('status')}): {request.request_key}")
        atomic_status(request_record, {"request_key": request.request_key, "status": "running",
                                      "phase": request.phase, "started_unix": time.time()})
        lot_source = self._ensure_lot_source(
            getattr(request, "code_bundle_url", None),
            getattr(request, "code_bundle_sha256", None),
        )
        # Campaign-only helpers ship in the verified source bundle, separate
        # from the deployed Fal app module itself.
        sys.path.insert(0, str(lot_source / "scripts/lot"))
        started = time.perf_counter()
        self._command_deadline = self._campaign_deadline(request)
        parent_checkpoint = self._campaign_parent(request, paths, lot_source)
        source = self._ensure_fizgig()
        self._ensure_lakonlab()
        self._ensure_models()
        branch_run_id = f"{request.campaign_id}-{request.branch or 'qualification'}"
        self._record_status(paths, {
            "campaign_id": request.campaign_id, "parent_run_id": request.parent_run_id,
            "branch": request.branch, "phase": request.phase, "status": "starting",
            "runner_gpu": self.runner_gpu, "started_unix": time.time(),
        })
        try:
            if self.gpu_setup_error:
                raise RuntimeError(self.gpu_setup_error)
            environment = self._fizgig_env(source)
            train_script = lot_source / "scripts/lot/train_h3.py"
            base = [sys.executable, str(train_script), "--pairs", str(paths["pairs"]),
                    "--cache", str(paths["cache"]), "--data-manifest", str(paths["data"] / "manifest.csv"),
                    "--holdout", "36", "--random-layout-p", "0.2", "--layout-policy", "entropy",
                    "--gradient-checkpointing", "--swap", "0", "--shift", "3", "--dense-shift", "12",
                    "--lr", "5e-5", "--adapter-lr", "5e-5",
                    "--distill", "1", "--data-weight", "1", "--eval-items",
                    "12" if request.phase == "confirm" else str(request.eval_items),
                    "--budget-min", "0.5", "--budget-max", "0.5",
                    "--eval-budget", "0.5", "--eval-every", "250", "--save-every", "50",
                    "--sampling-policy", "epoch",
                    "--uniform-budget"]
            if request.phase == "qualify":
                command = [*base, "--dense-distill", "1", "--detail-distill", "0.1",
                           "--temporal-distill", "0.05", "--init", str(parent_checkpoint),
                           "--steps", "1", "--check", "--out", str(paths["train"])]
                self._run_command(command, env=environment, cwd=lot_source,
                                  log_path=paths["run"] / "logs" / "qualification.log")
                status = "qualified"
            elif request.phase == "train":
                if request.branch is None or request.target_step < 50 or request.target_step % 50:
                    raise RuntimeError("training requires a branch and a 50-update checkpoint target")
                latest, checkpoint = self._latest_step(paths["train"], lot_source)
                current = latest or 0
                if request.target_step <= current or request.target_step > current + 1_000:
                    raise RuntimeError(f"target must advance branch-local updates by 1-1000; current={current}")
                dense_weight = "1" if request.branch in {"preserve", "dense"} else "0"
                resume = ("--resume", str(paths["train"])) if checkpoint is not None else (
                    "--init", str(parent_checkpoint))
                recipe_flags = {
                    "sparse50": ["--dense-p", "0.2"],
                    "detail50": ["--dense-p", "0.2", "--detail-distill", "0.1", "--temporal-distill", "0.05"],
                    "predicted50": ["--dense-p", "0.2", "--selector-source", "predicted"],
                    "dense_control": ["--dense-only"],
                    "dense_detail": ["--dense-only", "--detail-distill", "0.1", "--temporal-distill", "0.05"],
                }[request.recipe]
                if request.branch == "dense" and request.recipe not in {"dense_control", "dense_detail"}:
                    raise RuntimeError("dense branch must use a dense-only recipe")
                if request.branch == "detail" and request.recipe not in {"detail50", "predicted50"}:
                    raise RuntimeError("detail branch must use the detail50 or predicted50 recipe")
                if request.branch == "control" and request.recipe != "sparse50":
                    raise RuntimeError("control branch must use the sparse50 recipe")
                train_environment = dict(environment)
                train_environment["LOT_H3_REPLAY_SEED"] = "20261010"
                if request.optimizer_budget_seconds is not None:
                    train_environment["LOT_H3_MAX_SECONDS"] = str(request.optimizer_budget_seconds)
                else:
                    train_environment["LOT_H3_MAX_SECONDS"] = "0"
                environment["LOT_H3_TRAIN"] = "1"
                train_environment["LOT_H3_TRAIN"] = "1"
                command = [*base, *recipe_flags, "--dense-distill", dense_weight, *resume,
                           "--steps", str(request.target_step), "--out", str(paths["train"])]

                def publish(line: str) -> None:
                    if "LOT_H3 kind=checkpoint_committed " not in line:
                        return
                    import re

                    match = re.search(r"step=(\d+)", line)
                    if match and (int(match.group(1)) % 250 == 0 or int(match.group(1)) == request.target_step):
                        self._publish_checkpoint_line(line, branch_run_id, paths, lot_source, http_request)

                self._checkpoint_callback = publish
                self._run_command(command, env=train_environment, cwd=lot_source,
                                  log_path=paths["run"] / "logs" / f"training-{request.target_step:04d}.log")
                progress_path = paths["train"] / "progress.json"
                progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
                status = "training_complete" if progress.get("status") == "complete" else progress.get("status", "incomplete")
            elif request.phase in {"confirm", "preview"}:
                if request.branch is None:
                    raise RuntimeError(f"{request.phase} requires a branch")
                if request.branch == "parent":
                    latest, checkpoint = 0, parent_checkpoint
                else:
                    latest, checkpoint = self._latest_step(paths["train"], lot_source)
                if checkpoint is None:
                    raise RuntimeError("branch has no recoverable checkpoint")
                if request.phase == "confirm":
                    if request.evaluation_split != "confirmation":
                        raise RuntimeError("confirm phase must use reserved confirmation split")
                    dense_weight = "1" if request.branch == "preserve" else "0"
                    selector_flags = ["--selector-source", "predicted"] if request.recipe == "predicted50" else []
                    command = [*base, "--dense-distill", dense_weight,
                               *selector_flags,
                               "--resume", str(checkpoint), "--evaluation-split", "confirmation",
                               "--baseline-only-confirmation",
                               "--out", str(paths["train"])]
                    status = "confirmation_complete"
                else:
                    from run_artifacts import resolve_checkpoint

                    target = request.preview_step if request.preview_step is not None else request.target_step
                    if request.branch == "parent":
                        target = 0
                        render_run = self._campaign_parent_train
                    else:
                        checkpoint = resolve_checkpoint(paths["train"] / "checkpoints" / f"step-{target:08d}")
                        render_run = paths["train"]
                    command = [sys.executable, str(lot_source / "scripts/lot/render_h3_pilot.py"),
                               "--run", str(render_run), "--checkpoint", str(checkpoint),
                               "--items", str(request.preview_items), "--steps", str(request.preview_steps),
                               "--seed", str(request.preview_seed),
                               "--split", request.preview_split, "--entropy-inference",
                               "--fizgig-src", str(source / "src"), "--vae",
                               ASSETS["vae/minimax_h3_video_vae_fp16.safetensors"],
                               "--out", str(paths["train"] / "previews" /
                                              f"step-{target:08d}-{request.preview_split}-"
                                              f"{request.preview_steps}-{hashlib.sha256(request.request_key.encode()).hexdigest()[:10]}")]
                    if request.preview_controls:
                        command = [arg for arg in command if arg != "--entropy-inference"]
                        command.extend(["--controls", request.preview_controls])
                    command.extend([
                        "--warmup-sigma", str(request.preview_warmup_sigma),
                        "--budget-fraction", str(request.preview_budget_fraction),
                    ])
                    status = "preview_complete"
                self._run_command(command, env=environment, cwd=lot_source,
                                  log_path=paths["run"] / "logs" / f"{request.phase}-{latest:04d}.log")
            else:
                raise RuntimeError(f"unsupported campaign phase: {request.phase}")

            elapsed = time.perf_counter() - started
            self._record_status(paths, {
                "campaign_id": request.campaign_id, "parent_run_id": request.parent_run_id,
                "branch": request.branch, "phase": request.phase, "status": status,
                "runner_gpu": self.runner_gpu, "request_seconds": elapsed,
                "checkpoint_step": self._latest_step(paths["train"], lot_source)[0],
            })
            atomic_status(request_record, {"request_key": request.request_key, "status": "complete",
                                          "phase": request.phase, "finished_unix": time.time()})
            response.headers["x-lot-h3-run-seconds"] = str(round(elapsed, 3))
            return self._campaign_output(request, paths, http_request, status, lot_source)
        except Exception as exc:  # noqa: BLE001 - always export the last complete branch checkpoint
            atomic_status(request_record, {"request_key": request.request_key, "status": "failed",
                                          "phase": request.phase,
                                          "error": f"{type(exc).__name__}: {exc}",
                                          "finished_unix": time.time()})
            self._record_status(paths, {
                "campaign_id": request.campaign_id, "branch": request.branch,
                "phase": request.phase, "status": "failed",
                "runner_gpu": self.runner_gpu, "error": f"{type(exc).__name__}: {exc}",
                "recoverable": self._latest_step(paths["train"], lot_source)[1] is not None,
            })
            output = self._campaign_output(request, paths, http_request, "failed", lot_source)
            output.error = f"{type(exc).__name__}: {exc}"
            return output

    # Fal discovers inherited endpoint methods by effective attribute. Hide the
    # parent app's '/' route so this alias has exactly one campaign handler.
    def run(self, *_args, **_kwargs):
        raise RuntimeError("use the campaign endpoint")


if __name__ == "__main__":
    app = fal.wrap_app(H3LoTTraining)
    app()
