"""Crash-safe manifests and resumable checkpoint bundles for LoT runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete JSON file with a same-filesystem atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, value: dict[str, Any], *, sync: bool = False) -> None:
    """Append one complete event record; optionally force it to durable storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        if sync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _verify_checkpoint(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"checkpoint is incomplete: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"checkpoint is incomplete: {directory}")
    for name, expected in manifest.get("files", {}).items():
        path = directory / name
        if not path.is_file() or _file_digest(path) != expected:
            raise ValueError(f"checkpoint file failed integrity check: {path}")
    required = {"adapter.pt", "lora.safetensors", "trainer_state.pt"}
    if not required <= set(manifest.get("files", {})):
        raise ValueError(f"checkpoint is missing required artifacts: {directory}")
    return manifest


def resolve_checkpoint(path: Path) -> Path:
    """Resolve a committed bundle, a run's atomic latest pointer, or a legacy run."""
    path = Path(path)
    if (path / "manifest.json").is_file():
        _verify_checkpoint(path)
        return path
    pointer = path / "latest.json"
    pointer_error = None
    pointed = None
    if pointer.is_file():
        try:
            reference = json.loads(pointer.read_text(encoding="utf-8"))
            target = (path / reference["checkpoint"]).resolve()
            if path.resolve() not in target.parents:
                raise ValueError("latest checkpoint pointer leaves the run directory")
            manifest = _verify_checkpoint(target)
            if manifest.get("step") != reference.get("step"):
                raise ValueError("latest checkpoint pointer step does not match its manifest")
            pointed = target
        except (ValueError, KeyError, TypeError, OSError) as exc:
            pointer_error = exc
    # A kill can occur after the complete directory rename but before publishing
    # latest.json. Ignore hidden partial directories and verify committed bundles.
    for candidate in sorted((path / "checkpoints").glob("step-*"), reverse=True):
        try:
            manifest = _verify_checkpoint(candidate)
            if candidate.name != f"step-{int(manifest['step']):08d}":
                continue
            return candidate
        except (ValueError, KeyError, TypeError, OSError):
            continue
    if pointed is not None:
        return pointed
    if pointer_error is not None:
        raise ValueError(f"no intact checkpoint after invalid latest pointer: {pointer_error}") from pointer_error
    if all((path / name).is_file() for name in ("adapter.pt", "lora.safetensors")):
        return path
    raise FileNotFoundError(f"no complete LoT checkpoint in {path}")


def save_checkpoint_bundle(
    run_dir: Path,
    step: int,
    adapter_state: dict[str, torch.Tensor],
    save_lora: Callable[[Path], None],
    trainer_state: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Save model, optimizer/RNG state and manifest as one committed directory.

    A process killed while writing can leave a hidden temporary directory, but
    it cannot move the run's `latest.json` pointer away from its last complete
    checkpoint.
    """
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    run_dir = Path(run_dir)
    parent = run_dir / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / f"step-{step:08d}"
    if target.exists():
        manifest = _verify_checkpoint(target)
        if manifest.get("step") != step:
            raise ValueError(f"checkpoint path contains the wrong step: {target}")
        atomic_write_json(run_dir / "latest.json", {"step": step, "checkpoint": str(target.relative_to(run_dir))})
        return target

    temporary = Path(tempfile.mkdtemp(prefix=f".step-{step:08d}.", suffix=".tmp", dir=parent))
    try:
        torch.save({key: value.detach().cpu() for key, value in adapter_state.items()}, temporary / "adapter.pt")
        save_lora(temporary / "lora.safetensors")
        torch.save(trainer_state, temporary / "trainer_state.pt")
        files = {}
        for name in ("adapter.pt", "lora.safetensors", "trainer_state.pt"):
            path = temporary / name
            if not path.is_file():
                raise RuntimeError(f"checkpoint writer did not create {path}")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            files[name] = _file_digest(path)
        manifest = {
            "complete": True,
            "step": step,
            "files": files,
            "metadata": {**metadata, "saved_unix_time": time.time()},
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        _fsync_dir(temporary)
        os.replace(temporary, target)
        _fsync_dir(parent)
        atomic_write_json(run_dir / "latest.json", {"step": step, "checkpoint": str(target.relative_to(run_dir))})
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_trainer_state(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[Path, dict[str, Any]]:
    directory = resolve_checkpoint(path)
    state_path = directory / "trainer_state.pt"
    if not state_path.is_file():
        raise ValueError(f"checkpoint has weights but cannot resume training: {directory}")
    try:
        state = torch.load(state_path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(state_path, map_location=map_location)
    return directory, state
