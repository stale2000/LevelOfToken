"""Small, testable contracts for the LoT video training pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def clip_stem(path):
    import re
    return re.sub(r"_\d{4}x\d{4}_minimaxh3$", "", Path(path).stem)


def json_safe(value):
    """Convert Path values recursively so run manifests work with repeated CLI args."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_manifest(path):
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        clip = Path(row["local_file"]).stem
        if clip in result:
            raise ValueError(f"duplicate manifest clip {clip}")
        motion = row.get("camera_motion", "").strip().lower()
        result[clip] = {
            "clip": clip, "source_id": row.get("source_id") or clip,
            "camera_motion_group": "static" if motion == "static" else "moving" if motion else "unknown",
            "caption": row.get("caption", ""),
        }
    return result


def split_records(records, holdout=24, seed=0):
    """Grouped, deterministic stratified split; holdout counts source groups."""
    groups = defaultdict(list)
    for record in sorted(records, key=lambda row: row["clip"]):
        groups[record.get("source_id") or record["clip"]].append(record)
    if holdout < 1 or len(groups) <= holdout:
        raise ValueError(f"need more than {holdout} independent sources, found {len(groups)}")
    strata = defaultdict(list)
    for source, rows in sorted(groups.items()):
        kinds = {row.get("camera_motion_group", "unknown") for row in rows}
        kind = "moving" if "moving" in kinds else "static" if kinds == {"static"} else "unknown"
        strata[kind].append(source)
    rng = random.Random(seed)
    for sources in strata.values():
        rng.shuffle(sources)
    selected = []
    while len(selected) < holdout:
        for kind in sorted(strata):
            if strata[kind] and len(selected) < holdout:
                selected.append(strata[kind].pop())
    held_sources = set(selected)
    train, held = [], []
    for source, rows in groups.items():
        (held if source in held_sources else train).extend(rows)
    rng.shuffle(train)
    rng.shuffle(held)
    return train, held


def balanced_records(records, limit):
    strata = defaultdict(list)
    for record in records:
        strata[record.get("camera_motion_group", "unknown")].append(record)
    selected = []
    while len(selected) < min(limit, len(records)):
        for kind in sorted(strata):
            if strata[kind] and len(selected) < limit:
                selected.append(strata[kind].pop(0))
    return selected


def evaluation_partition(held_records, validation_limit):
    """Fix validation clips and reserve independent sources for confirmation."""
    validation = balanced_records(held_records, validation_limit)
    sources = {row.get("source_id") or row["clip"] for row in validation}
    confirmation = [row for row in held_records if (row.get("source_id") or row["clip"]) not in sources]
    return validation, confirmation


def evaluation_contract(validation, confirmation, *, seed, holdout, eval_budget):
    """Identity settings describe both fixed sets, independent of the evaluated set."""
    return {
        "schema_version": "pilot-v2", "seed": seed, "holdout": holdout,
        "validation_items": validation, "confirmation_items": confirmation,
        "eval_budget": eval_budget, "eval_sigmas": [.3, .6, .9],
    }


def evaluation_report_paths(directory, step, split):
    if split not in {"validation", "confirmation"}:
        raise ValueError(f"unknown evaluation split {split!r}")
    prefix = "metrics" if split == "validation" else "metrics-confirmation"
    stem = Path(directory) / f"{prefix}-step-{step:08d}"
    return stem.with_suffix(".csv"), stem.with_suffix(".json")


def evaluate_confirmation_checkpoint(resume_state, set_lora_multiplier, evaluate):
    """Evaluate saved weights at their saved step without advancing trainer state."""
    if resume_state is None:
        raise ValueError("confirmation evaluation requires a resumable checkpoint")
    set_lora_multiplier(float(resume_state.get("lora_multiplier", 1.0)))
    return evaluate(int(resume_state["step"]))


def dataset_identity(records, settings):
    """Hash cached tensors and the split/evaluation contract, not just paths."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for record in sorted(records, key=lambda row: row["clip"]):
        digest.update(json.dumps(record, sort_keys=True).encode())
        for key in ("latent_path", "text_path"):
            with Path(record[key]).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


@contextmanager
def frozen_dense_teacher(dit, network):
    """Restore exact routing, mode, and LoRA multipliers even on failure."""
    previous_lot = getattr(dit, "_lot", None)
    multipliers = [module.multiplier for module in network.unet_loras]
    modes = [(module, module.training) for module in dit.modules()]
    try:
        dit._lot = None
        dit.eval()
        for module in network.unet_loras:
            module.multiplier = 0.0
        yield
    finally:
        dit._lot = previous_lot
        for module, value in zip(network.unet_loras, multipliers):
            module.multiplier = value
        for module, training in modes:
            module.training = training


def audio_noise_for(dit, x0, generator=None):
    import torch
    from fizgig.minimax.model import (
        AUDIO_CHANNELS,
        audio_latents_for_frames,
        pixel_frames_for_latent,
    )
    count = audio_latents_for_frames(pixel_frames_for_latent(x0.shape[2])) if dit.pack_audio_rows else 0
    return torch.randn(count * AUDIO_CHANNELS, dit.config.audio_latents_dim,
                       device=x0.device, dtype=x0.dtype, generator=generator)


def dense_teacher_prediction(dit, network, x0, noise, sigma, text, audio_noise, *, x_t=None):
    """Frozen dense clean estimate, optionally from the student's exact noisy input."""
    import torch
    scale = sigma.reshape(1, 1, 1, 1, 1)
    if x_t is None:
        x_t = (1.0 - scale) * x0 + scale * noise
    with frozen_dense_teacher(dit, network), torch.no_grad():
        out = dit(x_t, 1.0 - sigma, text, audio_noise=audio_noise)
    return (x_t.float() + scale * out.float()).detach()


def teacher_distance(student, teacher, sigma):
    return (student.float() - teacher.float()).square().mean() / float(sigma.clamp(min=0.05)) ** 2


def prediction_diagnostics(prediction, data, teacher):
    """Finite differences expose blur/motion loss hidden by pixel MSE."""
    import torch
    pred, target, ref = prediction.float(), data.float(), teacher.float()
    result = {"clean_mse": float((pred - target).square().mean())}
    for name, other in (("temporal_mse", target), ("temporal_teacher_mse", ref)):
        result[name] = (float((pred.diff(dim=2) - other.diff(dim=2)).square().mean())
                        if pred.shape[2] > 1 else None)
    pred_grad = torch.cat([pred.diff(dim=-1).flatten(), pred.diff(dim=-2).flatten()])
    ref_grad = torch.cat([ref.diff(dim=-1).flatten(), ref.diff(dim=-2).flatten()])
    result["spatial_teacher_mse"] = float((pred_grad - ref_grad).square().mean())
    ref_power = float(ref_grad.square().mean())
    result["spatial_detail_ratio"] = (float(pred_grad.square().mean()) / ref_power) ** 0.5 if ref_power > 1e-12 else None
    return result


def require_finite_loss(loss):
    import torch
    if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError("non-finite or non-scalar training loss; optimizer was not updated")
