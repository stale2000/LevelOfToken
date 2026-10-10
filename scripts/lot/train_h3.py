"""Phase 5: fine-tune H3 for Level-of-Token layouts. Stills first.

    LOT_H3_TRAIN=1 python3 scripts/lot/train_h3.py --pairs DIR [--steps 2000] [--swap 4]

What trains: the LoT adapter (per-extent ``in_proj`` / ``out_proj``, shape MLP)
and a rank-16 LoRA on ``attn.qkv_proj``, ``attn.out_proj``, ``mlp.fc1``,
``mlp.fc2`` in every block. AdaLN, the token refiner, the int8 base, and the
extent bank (``A``, ``s`` from ``--pairs``) stay frozen.

One step: sample a layout, move the clean latent to y-space (``to_y``), draw
sigma (Fizgig ``sample_sigmas``), noise, run the DiT with the LoT tail, and
take ``lot_h3_clean_loss`` (eq. 17 in H3's ``x0 - eps`` convention: velocity
MSE in y-space). About 20% of steps use the dense 1×1 layout, which is the
ordinary H3 LoRA loss and anchors the base.

``--distill W`` adds a frozen, dense-H3 teacher target on LoT steps. It uses
the same clean clip, noise, and sigma, then trains the LoT clean estimate to
match the dense model instead of averaging only against the stochastic data
target. Evaluation reports this teacher gap separately from data error.
``--dense-distill W`` also preserves the frozen dense-H3 clean estimate on
dense anchor steps, using the student's exact noisy input and conditioning.

Sigma: ``--dense-shift`` for the dense anchor steps (default 12, H3's own
density: median 0.92, ~3% of steps below 0.3) and ``--shift`` for LoT steps.
The first 2,000-step run used 12 for both and the coarse regions came out
soft: detail is decided at low sigma, which shift 12 almost never trains.
A float is the uniform-u shift map, ``sigmoid`` is logit-normal, ``lognorm:S``
is logit-normal under shift S (see Fizgig ``sample_sigmas``).

Data: one or more Fizgig H3 cache directories (``--cache``, repeatable) of
``*_minimaxh3.safetensors`` latents with ``*_minimaxh3_te.safetensors`` text,
so Qwen is never loaded here. Stills are ``(24, H, W)``; clips are
``(24, T, H, W)`` (cache 384×640, 22 frames for ``latent_t = 7``). Each latent
is randomly cropped to a token grid divisible by 4. ``--init DIR`` warm-starts
the adapter and LoRA from an earlier run's ``adapter.pt`` / ``lora.safetensors``
with a fresh optimizer.

Writes only under ``--out`` (default ``scripts/lot/runs/train_h3``, gitignored):
``log.jsonl``, and at ``--save-every`` and the end ``adapter.pt`` +
``lora.safetensors``. Refuses to start unless ``LOT_H3_TRAIN=1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import signal
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/media/2TB/Fizgig/src")

from flow import lot_h3_clean_loss, sample_noisy
from h3 import make_h3_adapter, patchify
from h3_splice import LotSplice
from layout import LotLayout, TokenRect, dense_layout, layout_from_rects
from run_artifacts import (
    append_jsonl,
    atomic_write_json,
    load_trainer_state,
    resolve_checkpoint,
    save_checkpoint_bundle,
)
from training_experiment import (
    audio_noise_for,
    clip_stem,
    dataset_identity,
    dense_teacher_prediction,
    evaluate_confirmation_checkpoint,
    evaluation_contract,
    evaluation_partition,
    evaluation_report_paths,
    json_safe,
    prediction_diagnostics,
    read_manifest,
    require_finite_loss,
    split_records,
    teacher_distance,
)

CACHE = Path("/media/2TB/lora-data/fizgig_minimax_h3/cache_iso3d")
LORA_PATTERNS = [
    r"blocks\.\d+\.attn\.qkv_proj",
    r"blocks\.\d+\.attn\.out_proj",
    r"blocks\.\d+\.mlp\.fc1",
    r"blocks\.\d+\.mlp\.fc2",
]
SUPER = 4  # layout super-cell, in tokens: every extent tiles a 4×4 cell
FINE = [(1, 1, 1)]
MID = [(1, 1, 2), (1, 2, 1), (1, 2, 2)]
COARSE = [(1, 2, 4), (1, 4, 2), (1, 4, 4), (1, 1, 4), (1, 4, 1)]


# ---------------------------------------------------------------------------------- data

def list_items(cache: Path, holdout: int) -> tuple[list, list]:
    """``(latent_path, te_path)`` pairs, split deterministically into train / held out."""
    if not cache.is_dir():
        raise SystemExit(f"cache directory {cache} does not exist")
    items = []
    for latent in sorted(cache.glob("*_minimaxh3.safetensors")):
        stem = re.sub(r"_\d{4}x\d{4}_minimaxh3\.safetensors$", "", latent.name)
        te = cache / f"{stem}_minimaxh3_te.safetensors"
        if te.is_file():
            items.append((latent, te))
    if len(items) <= holdout:
        raise SystemExit(f"{cache}: {len(items)} items with text, need more than {holdout}")
    rng = random.Random(0)
    rng.shuffle(items)
    return items[holdout:], items[:holdout]


def load_item(item, rng: random.Random, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean latent ``(1, 24, T, H, W)`` cropped to a 4-token multiple, and text ``(1, L, 5120)``.

    A still is ``T = 1``; a cached clip keeps its latent frames.
    """
    from safetensors import safe_open

    latent_path, te_path = item
    with safe_open(str(latent_path), "pt") as handle:
        key = next(k for k in handle.keys() if k.startswith("latent"))  # noqa: SIM118 - safetensors.safe_open exposes keys()
        latent = handle.get_tensor(key).float()
    with safe_open(str(te_path), "pt") as handle:
        hidden = handle.get_tensor("hidden_states")
        mask = handle.get_tensor("attention_mask")
    if latent.ndim == 3:
        latent = latent.unsqueeze(1)                      # (24, 1, H, W)
    step = 2 * SUPER                                       # latent pixels per super-cell
    height, width = latent.shape[-2] // step * step, latent.shape[-1] // step * step
    top = rng.randint(0, latent.shape[-2] - height)
    left = rng.randint(0, latent.shape[-1] - width)
    latent = latent[..., top:top + height, left:left + width].unsqueeze(0)
    return latent.to(device), hidden[mask].unsqueeze(0).to(device, torch.bfloat16)


# -------------------------------------------------------------------------------- layout

def layout_category(roll: float, *, dense_p: float, random_p: float) -> str:
    """Map one uniform draw onto the frozen dense/uniform/random/adaptive mix."""
    if not 0 <= dense_p < 1 or not 0 <= random_p < 1 or dense_p + .15 + random_p >= 1:
        raise ValueError("layout probabilities leave no adaptive training steps")
    if not 0 <= roll < 1:
        raise ValueError("layout roll must be in [0, 1)")
    if roll < dense_p:
        return "dense"
    if roll < dense_p + .15:
        return "uniform"
    if roll < dense_p + .15 + random_p:
        return "random_budget"
    return "adaptive"

def token_detail(latent: torch.Tensor) -> torch.Tensor:
    """``(T, H, W)`` detail per token: distance from its 4×4-cell mean, on the clean latent."""
    tokens = patchify(latent.float())[0]                   # (T, H, W, 96)
    grid = tokens.permute(0, 3, 1, 2)                       # (T, 96, H, W)
    coarse = F.avg_pool2d(grid, SUPER)
    coarse = F.interpolate(coarse, scale_factor=SUPER, mode="nearest")
    return (grid - coarse).norm(dim=1)


def sample_layout(
    latent: torch.Tensor,
    rng: random.Random,
    dense_p: float = 0.2,
    *,
    policy: str = "legacy",
    calibration=None,
    budget_range: tuple[float, float] = (0.25, 0.75),
    random_layout_p: float = 0.0,
    uniform_budget: bool = False,
    force_kind: str | None = None,
) -> tuple[LotLayout, str]:
    """A layout for one example and its kind.

    ``dense``: all 1×1. ``uniform``: one extent everywhere. ``mosaic``: each 4×4
    super-cell takes one extent, finer where the clean latent has more detail,
    with jittered thresholds, so mixed boundaries and rectangles both appear.
    """
    _b, _c, time, lat_h, lat_w = latent.shape
    height, width = lat_h // 2, lat_w // 2
    if not 0.0 <= random_layout_p < 0.80:
        raise ValueError("random layout probability must be in [0, .8)")
    chosen = (layout_category(rng.random(), dense_p=dense_p, random_p=random_layout_p)
              if force_kind is None else force_kind)
    if force_kind not in (None, "dense", "uniform", "random_budget", "adaptive"):
        raise ValueError(f"unknown forced layout kind {force_kind!r}")
    if chosen == "dense":
        return dense_layout(time, height, width), "dense"
    if chosen == "uniform":
        extent = rng.choice([(1, 1, 2), (1, 2, 1)] if uniform_budget else MID + COARSE)
        return _tile(time, height, width, lambda _t, _u, _v: extent), "uniform"
    if chosen == "random_budget":
        if not 0.0 < budget_range[0] <= budget_range[1] <= 1.0:
            raise ValueError("layout budget range must satisfy 0 < minimum <= maximum <= 1")
        from budget_layout import choose_budget_layout

        tokens = patchify(latent.float())[0]
        per_frame = tokens.shape[1] * tokens.shape[2]
        budget = round(per_frame * rng.uniform(*budget_range))
        selected = choose_budget_layout(
            tokens, budget, policy="random", calibration=calibration, seed=rng.randrange(2**31)
        )
        return selected.layout, "random_budget"
    if policy != "legacy":
        if policy not in {"detail", "entropy", "random"}:
            raise ValueError(f"unsupported layout policy {policy!r}")
        if not 0.0 < budget_range[0] <= budget_range[1] <= 1.0:
            raise ValueError("layout budget range must satisfy 0 < minimum <= maximum <= 1")
        from budget_layout import choose_budget_layout

        tokens = patchify(latent.float())[0]
        per_frame = tokens.shape[1] * tokens.shape[2]
        budget = round(per_frame * rng.uniform(*budget_range))
        selected = choose_budget_layout(
            tokens, budget, policy=policy, calibration=calibration, seed=rng.randrange(2**31)
        )
        return selected.layout, f"{policy}_budget"

    detail = token_detail(latent).cpu()
    cells = F.avg_pool2d(detail.unsqueeze(1), SUPER).squeeze(1)    # (T, H/4, W/4)
    flat = cells.flatten()
    low_q, high_q = sorted((rng.uniform(0.2, 0.6), rng.uniform(0.5, 0.9)))
    low, high = float(flat.quantile(low_q)), float(flat.quantile(high_q))

    def pick(t: int, u: int, v: int):
        value = float(cells[t, u // SUPER, v // SUPER])
        if value >= high:
            return FINE[0]
        if value >= low:
            return rng.choice(MID)
        return rng.choice(COARSE)

    return _tile(time, height, width, pick), "mosaic"


def fit_entropy_calibration(items, limit: int = 64):
    """Fit feature normalization on a deterministic prefix of the training split."""
    from budget_layout import EntropyCalibration
    from safetensors import safe_open

    def token_samples():
        for latent_path, _text_path in items[:limit]:
            with safe_open(str(latent_path), "pt") as handle:
                key = next(k for k in handle.keys() if k.startswith("latent"))  # noqa: SIM118 - safetensors.safe_open exposes keys()
                latent = handle.get_tensor(key).float()
            if latent.ndim == 3:
                latent = latent.unsqueeze(1)
            yield patchify(latent.unsqueeze(0))[0].cpu()

    return EntropyCalibration.fit(token_samples())


def _tile(time: int, height: int, width: int, choose) -> LotLayout:
    rects = []
    for t in range(time):
        for u0 in range(0, height, SUPER):
            for v0 in range(0, width, SUPER):
                _et, eh, ew = choose(t, u0, v0)
                for u in range(u0, u0 + SUPER, eh):
                    for v in range(v0, v0 + SUPER, ew):
                        rects.append(TokenRect(t, u, v, 1, eh, ew))
    return layout_from_rects(time, height, width, rects)


def extent_losses(out, y_t, y0, sigma, layout: LotLayout) -> dict[str, float]:
    """Held-out squared clean error per extent, on the token lattice."""
    sig = float(sigma)
    err = (y_t.float() + sig * out.float() - y0.float()).square() / max(sig, 0.05) ** 2
    tokens = patchify(err)[0].mean(dim=-1)                  # (T, H, W)
    sums: dict[str, list] = {}
    for rect in layout.rects:
        block = tokens[rect.t, rect.u:rect.u + rect.eh, rect.v:rect.v + rect.ew]
        sums.setdefault("x".join(map(str, rect.extent)), []).append(float(block.mean()))
    return {key: sum(values) / len(values) for key, values in sums.items()}


# --------------------------------------------------------------------------------- train

def parse_shift(value: str):
    """``12`` -> 12.0; ``sigmoid`` and ``lognorm:S`` pass through to ``sample_sigmas``."""
    if value == "sigmoid" or value.startswith("lognorm:"):
        return value
    return float(value)


def training_objective(dit, adapter, network, x0, text, layout, kind, sigma, noise,
                      audio_noise, *, data_weight: float, distill_weight: float,
                      dense_distill_weight: float = 0.0,
                      detail_distill_weight: float = 0.0,
                      temporal_distill_weight: float = 0.0,
                      aux_on_dense: bool = False,
                      aux_losses: dict | None = None):
    """Exact forward/loss path shared by the GPU dry check and optimizer."""
    splice = None if kind == "dense" else LotSplice(adapter, layout, y_space=True)
    y0 = x0 if splice is None else splice.to_y(x0)
    y_t, _ = sample_noisy(y0, sigma, noise)
    dit._lot = splice
    try:
        out = dit(y_t, 1.0 - sigma, text, audio_noise=audio_noise)
    finally:
        dit._lot = None
    data_loss = lot_h3_clean_loss(out, y_t, y0, sigma)
    gap = None
    teacher = None
    teacher_weight = dense_distill_weight if kind == "dense" else distill_weight
    active_detail_weight = detail_distill_weight if kind != "dense" or aux_on_dense else 0.0
    active_temporal_weight = temporal_distill_weight if kind != "dense" or aux_on_dense else 0.0
    if teacher_weight > 0 or active_detail_weight > 0 or active_temporal_weight > 0:
        teacher = dense_teacher_prediction(
            dit, network, x0, noise, sigma, text, audio_noise,
            **({"x_t": y_t} if kind == "dense" else {}),
        )
        clean = y_t.float() + sigma.reshape(1, 1, 1, 1, 1) * out.float()
        if splice is not None:
            clean = splice.to_x(clean)
        gap = teacher_distance(clean, teacher, sigma)
    detail_gap = temporal_gap = None
    if teacher is not None and active_detail_weight > 0:
        student_grad = torch.cat([clean.diff(dim=-1).flatten(), clean.diff(dim=-2).flatten()])
        teacher_grad = torch.cat([teacher.diff(dim=-1).flatten(), teacher.diff(dim=-2).flatten()])
        detail_gap = (student_grad - teacher_grad).square().mean() / float(sigma.clamp(min=0.05)) ** 2
    if teacher is not None and active_temporal_weight > 0 and clean.shape[2] > 1:
        temporal_gap = (clean.diff(dim=2) - teacher.diff(dim=2)).square().mean() / float(sigma.clamp(min=0.05)) ** 2
    loss = (data_weight * data_loss
            + (teacher_weight * gap if gap is not None else 0)
            + (active_detail_weight * detail_gap if detail_gap is not None else 0)
            + (active_temporal_weight * temporal_gap if temporal_gap is not None else 0))
    if aux_losses is not None:
        aux_losses.update(detail=detail_gap, temporal=temporal_gap)
    require_finite_loss(loss)
    return loss, out, y_t, y0, gap, data_loss


def check(adapter, dit, network, groups, items, rng, device, args, calibration) -> None:
    """Exercise the exact video/audio/distillation path for all pilot layouts."""
    from fizgig.minimax.trainer import sample_sigmas

    params = [p for group in groups for p in group["params"]]
    cases = ("dense", "uniform", "adaptive", "random_budget")
    for index, forced in enumerate(cases):
        x0, text = load_item(items[index % len(items)], rng, device)
        layout, kind = sample_layout(
            x0, rng, args.dense_p, policy=args.layout_policy, calibration=calibration,
            budget_range=(args.budget_min, args.budget_max),
            random_layout_p=args.random_layout_p, uniform_budget=args.uniform_budget, force_kind=forced,
        )
        sigma = sample_sigmas(1, device, shift=args.dense_shift if kind == "dense" else args.shift)
        noise = torch.randn_like(x0)
        audio_noise = audio_noise_for(dit, x0)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        aux = {}
        loss, _out, _state, _clean, gap, data_loss = training_objective(
            dit, adapter, network, x0, text, layout, kind, sigma, noise, audio_noise,
            data_weight=args.data_weight, distill_weight=args.distill,
            dense_distill_weight=args.dense_distill,
            detail_distill_weight=args.detail_distill,
            temporal_distill_weight=args.temporal_distill, aux_on_dense=args.dense_only, aux_losses=aux,
        )
        loss.backward()
        torch.cuda.synchronize()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in groups for p in group["params"]], 1.0, error_if_nonfinite=True
        )
        lora_grad = sum(float(p.grad.abs().sum()) for p in groups[0]["params"] if p.grad is not None)
        adapter_grad = sum(float(p.grad.abs().sum()) for p in groups[-1]["params"] if p.grad is not None)
        for parameter in params:
            parameter.grad = None
        print(f"LOT_H3 kind=train_check requested={forced} layout={kind} tokens={layout.count}/{layout.dense_count} "
              f"loss={float(loss.detach()):.4f} data={float(data_loss.detach()):.4f} "
              f"gap={float(gap.detach()) if gap is not None else 0:.4f} sigma={float(sigma):.3f} "
              f"detail={float(aux['detail'].detach()) if aux.get('detail') is not None else 0:.4f} "
              f"temporal={float(aux['temporal'].detach()) if aux.get('temporal') is not None else 0:.4f} "
              f"grad_norm={float(grad_norm):.3g} lora_grad={lora_grad:.3g} adapter_grad={adapter_grad:.3g} "
              f"seconds={time.perf_counter() - start:.2f} "
              f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} ok=1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, required=True, help="Procrustes pair dir for the frozen bank")
    parser.add_argument("--cache", type=Path, action="append", default=None,
                        help=f"Fizgig H3 cache dir; repeat to mix stills and clips (default {CACHE})")
    parser.add_argument("--data-manifest", type=Path, default=None, help="clip/source/camera metadata CSV")
    parser.add_argument("--shift", type=parse_shift, default=12.0,
                        help="sigma density for LoT steps: float shift, 'sigmoid', or 'lognorm:S'")
    parser.add_argument("--dense-shift", type=parse_shift, default=12.0,
                        help="sigma density for the dense anchor steps (12 = H3's own)")
    parser.add_argument("--distill", type=float, default=0.0,
                        help="weight of frozen dense-H3 teacher target on LoT steps")
    parser.add_argument("--dense-distill", type=float, default=0.0,
                        help="weight of frozen dense-H3 teacher preservation on dense anchor steps")
    parser.add_argument("--detail-distill", type=float, default=0.0,
                        help="weight of dense-teacher spatial finite-difference matching")
    parser.add_argument("--temporal-distill", type=float, default=0.0,
                        help="weight of dense-teacher adjacent-frame difference matching")
    parser.add_argument("--data-weight", type=float, default=1.0,
                        help="weight of the H3 eq. 17 data loss")
    parser.add_argument("--init", type=Path, default=None,
                        help="warm-start adapter.pt + lora.safetensors from an earlier run dir")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume weights, optimizer, and random state from a complete checkpoint")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--swap", type=int, default=4, help="blocks streamed from CPU; >0 turns on checkpointing")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="checkpoint block activations without streaming blocks from CPU")
    parser.add_argument("--dense-p", type=float, default=0.2)
    parser.add_argument("--dense-only", action="store_true",
                        help="train the LoRA on native dense H3 tokens; leaves the LoT adapter inactive")
    parser.add_argument("--uniform-budget", action="store_true",
                        help="use only 1x2 or 2x1 layouts for uniform anchors (exactly 50% tokens)")
    parser.add_argument("--holdout", type=int, default=24)
    parser.add_argument("--eval-items", type=int, default=12,
                        help="fixed held-out clips used for baseline and periodic comparisons")
    parser.add_argument("--eval-budget", type=float, default=0.5,
                        help="matched per-frame token fraction for detail/entropy/random controls")
    parser.add_argument("--evaluation-split", choices=("validation", "confirmation"), default="validation",
                        help="fixed held-out source set to evaluate; confirmation requires its evaluation-only mode")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=100,
                        help="write a verified, resumable checkpoint bundle every N steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-policy", choices=("replacement", "epoch"), default="replacement",
                        help="source replay: per-step replacement or deterministic shuffled passes")
    parser.add_argument("--selector-source", choices=("clean", "predicted"), default="clean",
                        help="adaptive layout input: clean training latent or frozen dense prediction")
    parser.add_argument("--selector-sigma", type=float, default=0.7063799788508988,
                        help="sigma used for predicted selector inputs; layout training sigma remains --shift")
    parser.add_argument("--layout-policy", choices=("legacy", "detail", "entropy", "random"), default="legacy",
                        help="adaptive layout policy for training steps outside dense/uniform anchors")
    parser.add_argument("--random-layout-p", type=float, default=0.0,
                        help="fraction of budgeted training examples using random quadtree placement")
    parser.add_argument("--budget-min", type=float, default=0.25)
    parser.add_argument("--budget-max", type=float, default=0.75)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline-only", action="store_true",
                        help="save the initialized LoRA, evaluate matched layout baselines, and exit")
    parser.add_argument("--baseline-only-confirmation", action="store_true",
                        help="evaluate only confirmation sources at the resumed checkpoint step; save no checkpoint")
    parser.add_argument("--check", action="store_true",
                        help="one forward + backward per layout kind, no optimizer step, writes nothing; "
                             "allowed without LOT_H3_TRAIN")
    args = parser.parse_args()

    if args.baseline_only_confirmation:
        if args.resume is None or args.baseline_only or args.check:
            raise SystemExit("--baseline-only-confirmation requires --resume and excludes --baseline-only/--check")
        args.evaluation_split = "confirmation"
    elif args.evaluation_split == "confirmation":
        raise SystemExit("confirmation sources require --baseline-only-confirmation")
    if args.init is not None and args.resume is not None:
        raise SystemExit("--init and --resume are mutually exclusive")
    if args.eval_items < 1 or not 0.0 < args.eval_budget <= 1.0:
        raise SystemExit("--eval-items must be positive and --eval-budget must be in (0, 1]")
    if not 0.0 < args.budget_min <= args.budget_max <= 1.0:
        raise SystemExit("layout budget range must satisfy 0 < --budget-min <= --budget-max <= 1")
    if not all(math.isfinite(v) and v >= 0 for v in (
            args.distill, args.dense_distill, args.detail_distill, args.temporal_distill, args.data_weight)):
        raise SystemExit("objective weights must be finite and non-negative")
    if args.distill == 0 and args.dense_distill == 0 and args.detail_distill == 0 \
            and args.temporal_distill == 0 and args.data_weight == 0:
        raise SystemExit("at least one objective weight must be positive")
    if not 0 <= args.dense_p < .85 or not 0 <= args.random_layout_p < .8 \
            or args.dense_p + .15 + args.random_layout_p >= 1 \
            or args.holdout < args.eval_items or args.eval_every < 0:
        raise SystemExit("layout probabilities, holdout, or evaluation interval are invalid")
    if not all(math.isfinite(v) and v > 0 for v in (args.lr, args.adapter_lr)) or args.rank < 1:
        raise SystemExit("learning rates and rank must be positive and finite")
    if not math.isfinite(args.selector_sigma) or not 0.0 < args.selector_sigma < 1.0:
        raise SystemExit("--selector-sigma must be finite and in (0, 1)")
    if args.save_every < 1:
        raise SystemExit("--save-every must be positive so a recoverable checkpoint is always produced")
    if not (args.baseline_only or args.baseline_only_confirmation or args.check) and args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.baseline_only and args.check:
        raise SystemExit("--baseline-only and --check are separate modes")
    if args.resume is not None:
        if args.out is None:
            resolved = resolve_checkpoint(args.resume)
            args.out = resolved.parent.parent
    elif args.out is None:
        args.out = Path(__file__).resolve().parent / "runs" / "train_h3"
    # Older deployed Fal app revisions only expose the original recipe schema.
    # Branch paths remain part of the durable campaign contract, so allow the
    # source bundle to select the new replay/selector behavior without requiring
    # an app redeploy. Explicit CLI flags still win for local runs.
    branch_parts = set(args.out.resolve().parts)
    if "control" in branch_parts or "detail" in branch_parts:
        args.sampling_policy = "epoch"
    if "detail" in branch_parts and args.selector_source == "clean":
        args.selector_source = "predicted"
        # The candidate isolates selector input; the old detail recipe's generic
        # gradient auxiliaries would otherwise confound the comparison.
        args.detail_distill = 0.0
        args.temporal_distill = 0.0
    if args.resume is None and args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(
            f"run directory already contains files: {args.out}; choose a new --out or use --resume"
        )

    if not (args.check or args.baseline_only or args.baseline_only_confirmation) and os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("phase 5 is gated: set LOT_H3_TRAIN=1 to train (or --check for a dry pass)")
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_h3.py")
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.trainer import sample_sigmas
    from fizgig.networks.lora import create_network
    from procrustes_h3 import CHECKPOINT, fit_bank, load_pairs, pretrained_maps

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    resume_directory = None
    resume_state = None
    if args.resume is not None:
        resume_directory, resume_state = load_trainer_state(args.resume, map_location="cpu")
        manifest = json.loads((resume_directory / "manifest.json").read_text(encoding="utf-8"))
        metadata = manifest.get("metadata", {})
        expected = {
            "rank": args.rank,
            "layout_policy": args.layout_policy,
            "seed": args.seed,
            "budget_min": args.budget_min,
            "budget_max": args.budget_max,
            "dense_p": args.dense_p,
            "random_layout_p": args.random_layout_p,
            "holdout": args.holdout,
            "shift": args.shift,
            "dense_shift": args.dense_shift,
            "distill": args.distill,
            "dense_distill": args.dense_distill,
            "detail_distill": args.detail_distill,
            "temporal_distill": args.temporal_distill,
            "dense_only": args.dense_only,
            "uniform_budget": args.uniform_budget,
            "data_weight": args.data_weight,
            "lr": args.lr,
            "adapter_lr": args.adapter_lr,
            "swap": args.swap,
            "gradient_checkpointing": args.gradient_checkpointing,
            "sampling_policy": args.sampling_policy,
            "selector_source": args.selector_source,
            "selector_sigma": args.selector_sigma,
        }
        optional_legacy = {"sampling_policy", "selector_source", "selector_sigma"}
        mismatch = []
        for key, value in expected.items():
            if key in optional_legacy and key not in metadata:
                continue
            if metadata.get(key, 0.0 if key == "dense_distill" else None) != value:
                mismatch.append(key)
        if mismatch:
            raise SystemExit(f"resume checkpoint training settings differ: {', '.join(mismatch)}")
        if (not args.baseline_only_confirmation
                and args.out.resolve() != resume_directory.parent.parent.resolve()):
            raise SystemExit("--resume output must point at the original run directory")
    caches = args.cache or [CACHE]
    metadata_by_clip = read_manifest(args.data_manifest)
    records = []
    for cache in caches:
        if not cache.is_dir():
            raise SystemExit(f"missing cache directory {cache}")
        for latent in sorted(cache.glob("*_minimaxh3.safetensors")):
            clip = clip_stem(latent)
            text_path = cache / f"{clip}_minimaxh3_te.safetensors"
            if not text_path.is_file():
                raise SystemExit(f"missing text cache for {clip}")
            if args.data_manifest and clip not in metadata_by_clip:
                raise SystemExit(f"cache clip missing from manifest: {clip}")
            records.append({**metadata_by_clip.get(clip, {
                "clip": clip, "source_id": clip, "camera_motion_group": "unknown", "caption": "",
            }), "latent_path": str(latent.resolve()), "text_path": str(text_path.resolve())})
    if len({row["clip"] for row in records}) != len(records):
        raise SystemExit("duplicate cached clips across cache directories")
    train_records, held_records = split_records(records, args.holdout, args.seed)
    validation_records, confirmation_records = evaluation_partition(held_records, args.eval_items)
    if len(validation_records) < args.eval_items:
        raise SystemExit("--eval-items exceeds the held-out clip count")
    eval_records = validation_records if args.evaluation_split == "validation" else confirmation_records
    if not eval_records:
        raise SystemExit(f"no sources reserved for {args.evaluation_split} evaluation")
    train_items = [(Path(row["latent_path"]), Path(row["text_path"])) for row in train_records]
    held_items = [(Path(row["latent_path"]), Path(row["text_path"])) for row in held_records]
    experiment_identity = dataset_identity(records, evaluation_contract(
        validation_records, confirmation_records, seed=args.seed, holdout=args.holdout,
        eval_budget=args.eval_budget,
    ))
    if resume_state is not None:
        previous_config = resume_state.get("run_config", {})
        if previous_config.get("dataset_identity") != experiment_identity:
            legacy_settings = {
                "schema_version": "pilot-v2", "seed": args.seed, "holdout": args.holdout,
                "evaluation_items": validation_records, "eval_budget": args.eval_budget, "eval_sigmas": [.3, .6, .9],
            }
            legacy_matches = (
                "evaluation_contract_version" not in previous_config
                and previous_config.get("validation_items") == validation_records
                and previous_config.get("confirmation_items") == confirmation_records
                and previous_config.get("dataset_identity") == dataset_identity(records, legacy_settings)
            )
            if not legacy_matches:
                raise SystemExit("resume data/split/evaluation contract differs; use a fresh run")
    if resume_state is not None and resume_state.get("entropy_calibration") is not None:
        from budget_layout import EntropyCalibration

        calibration_state = resume_state["entropy_calibration"]
        entropy_calibration = EntropyCalibration(
            calibration_state["mean"].float(), calibration_state["std"].float()
        )
    else:
        entropy_calibration = fit_entropy_calibration(train_items)

    adapter = make_h3_adapter()
    adapter.init_from_pretrained(*pretrained_maps(CHECKPOINT))
    fit_bank(adapter, load_pairs(args.pairs))
    adapter.to(device)
    adapter.bank.requires_grad_(False)

    dit = load_minimax_h3_dit(str(CHECKPOINT), device="cuda", compute_dtype=torch.bfloat16,
                              base_quant="int8", blocks_to_swap=args.swap)
    dit.requires_grad_(False)
    if args.swap:
        dit.enable_block_swap(args.swap, h2d_only=True)
    if args.swap or args.gradient_checkpointing:
        dit.enable_gradient_checkpointing(True)
    dit._tread = None
    network = create_network(None, "lora_unet", 1.0, args.rank, float(args.rank), None, [], dit,
                             include_patterns=LORA_PATTERNS)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    initial_source = None
    if resume_directory is not None:
        initial_source = resume_directory
    elif args.init is not None:
        initial_source = resolve_checkpoint(args.init)
    if initial_source is not None:
        state = torch.load(initial_source / "adapter.pt", map_location="cpu", weights_only=True)
        adapter.load_state_dict(state)
        info = network.load_weights(str(initial_source / "lora.safetensors"))
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(
                f"checkpoint LoRA mismatch: {len(info.missing_keys)} missing, "
                f"{len(info.unexpected_keys)} unexpected keys"
            )
        print(f"LOT_H3 kind={'resume' if resume_directory else 'train_init'} from={initial_source}", flush=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=torch.bfloat16)
    expected = len(LORA_PATTERNS) * len(dit.blocks)
    if len(network.unet_loras) != expected:
        raise RuntimeError(f"{len(network.unet_loras)} LoRA modules wrapped, expected {expected} "
                           f"(4 linears x {len(dit.blocks)} blocks)")

    lora_groups, _ = network.prepare_optimizer_params(args.lr)
    adapter_params = [p for name, p in adapter.named_parameters() if not name.startswith("bank.")]
    groups = lora_groups + [{"params": adapter_params, "lr": args.adapter_lr}]
    optimizer = None
    opt_name = "none"
    if not args.baseline_only_confirmation:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.0)
            opt_name = "adamw8bit"
        except ImportError:
            optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
            opt_name = "adamw"
    if resume_state is not None:
        if optimizer is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
        rng.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        start_step = int(resume_state["step"])
        if start_step >= args.steps and not args.baseline_only_confirmation:
            raise SystemExit(f"resume checkpoint is already at step {start_step}; increase --steps")
    else:
        start_step = 0
    trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"LOT_H3 kind=train_start items={len(train_items)} holdout={len(held_items)} "
          f"lora_modules={len(network.unet_loras)} trainable={trainable} optimizer={opt_name} "
          f"swap={args.swap} gradient_checkpointing={int(bool(args.swap or args.gradient_checkpointing))} "
          f"shift={args.shift} dense_shift={args.dense_shift} caches={len(caches)} "
          f"distill={args.distill} dense_distill={args.dense_distill} data_weight={args.data_weight} "
          f"allocated_mb={torch.cuda.memory_allocated() / 2**20:.0f}", flush=True)

    if args.check:
        check(adapter, dit, network, groups, train_items, rng, device, args, entropy_calibration)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    run_config = {
        **{key: json_safe(value) for key, value in vars(args).items()},
        "cache_directories": [str(cache) for cache in caches],
        "schema_version": "pilot-v2",
        "evaluation_contract_version": 2,
        "dataset_identity": experiment_identity,
        "evaluation_items": eval_records,
        "validation_items": validation_records,
        "confirmation_items": confirmation_records,
        "validation_source_ids": sorted({row["source_id"] for row in validation_records}),
        "confirmation_source_ids": sorted({row["source_id"] for row in confirmation_records}),
        "train_clip_ids": [row["clip"] for row in train_records],
        "heldout_clip_ids": [row["clip"] for row in held_records],
        "train_items": len(train_items),
        "heldout_items": len(held_items),
        "eval_seed": 1000,
        "eval_sigmas": [0.3, 0.6, 0.9],
        "eval_layouts": ["base_dense", "dense_lora", "lot_dense", "uniform1x2", "uniform2x1", "uniform2", "legacy_mosaic", "detail50", "entropy50", "random50", "random_shuffle50"],
        "entropy_calibration": "first 64 training clips only",
        "base_checkpoint": str(CHECKPOINT),
        "sampling_policy": args.sampling_policy,
        "selector_source": args.selector_source,
        "selector_sigma": args.selector_sigma,
    }
    if not args.baseline_only_confirmation and not (args.out / "run.json").exists():
        atomic_write_json(args.out / "run.json", run_config)
    calibration_path = args.out / "entropy_calibration.pt"
    if not args.baseline_only_confirmation and not calibration_path.exists():
        calibration_tmp = calibration_path.with_suffix(".pt.tmp")
        torch.save({"mean": entropy_calibration.mean.cpu(), "std": entropy_calibration.std.cpu()}, calibration_tmp)
        os.replace(calibration_tmp, calibration_path)
    log = open(args.out / "log.jsonl", "a", buffering=1)  # noqa: SIM115 - explicitly fsynced and closed on both exits
    lora_enabled = True

    def set_lora_multiplier(value: float) -> None:
        nonlocal lora_enabled
        lora_enabled = value != 0.0
        for module in network.unet_loras:
            module.multiplier = value

    session_started = time.perf_counter()
    previous_elapsed = float(resume_state.get("elapsed_seconds", 0.0)) if resume_state else 0.0
    token_counters = dict(resume_state.get("token_counters", {})) if resume_state else {}
    token_counters.setdefault("student_tokens", 0)
    token_counters.setdefault("dense_equivalent_tokens", 0)
    token_counters.setdefault("teacher_tokens", 0)
    max_seconds = float(os.environ.get("LOT_H3_MAX_SECONDS", "0"))
    signal_state = {"stop": False}

    def should_stop():
        return signal_state["stop"] or (max_seconds > 0 and time.perf_counter() - session_started >= max_seconds)

    def request_stop(_signum, _frame):
        signal_state["stop"] = True
        print("LOT_H3 kind=stop_requested checkpoint_after_current_step=1", flush=True)

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)

    def save_snapshot(step: int) -> Path:
        trainer_state = {
            "step": step,
            "token_counters": dict(token_counters),
            "optimizer": optimizer.state_dict(),
            "python_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "entropy_calibration": {
                "mean": entropy_calibration.mean.cpu(),
                "std": entropy_calibration.std.cpu(),
            },
            "lora_multiplier": 1.0 if lora_enabled else 0.0,
            "elapsed_seconds": previous_elapsed + time.perf_counter() - session_started,
            "run_config": run_config,
        }

        def write_lora(path: Path) -> None:
            network.save_weights(str(path), torch.bfloat16, {
                "lot_h3": "phase5",
                "rank": str(args.rank),
                "layout_policy": args.layout_policy,
                "step": str(step),
            })

        directory = save_checkpoint_bundle(
            args.out,
            step,
            adapter.state_dict(),
            write_lora,
            trainer_state,
            {
                "rank": args.rank,
                "layout_policy": args.layout_policy,
                "seed": args.seed,
                "budget_min": args.budget_min,
                "budget_max": args.budget_max,
                "dense_p": args.dense_p,
                "random_layout_p": args.random_layout_p,
                "holdout": args.holdout,
                "shift": args.shift,
                "dense_shift": args.dense_shift,
                "distill": args.distill,
                "dense_distill": args.dense_distill,
                "detail_distill": args.detail_distill,
                "temporal_distill": args.temporal_distill,
                "dense_only": args.dense_only,
                "uniform_budget": args.uniform_budget,
                "data_weight": args.data_weight,
                "lr": args.lr,
                "adapter_lr": args.adapter_lr,
                "swap": args.swap,
                "gradient_checkpointing": args.gradient_checkpointing,
                "sampling_policy": args.sampling_policy,
                "selector_source": args.selector_source,
                "selector_sigma": args.selector_sigma,
                "base_checkpoint": CHECKPOINT.name,
                "cache_directories": [str(cache) for cache in caches],
            },
        )
        print(f"LOT_H3 kind=checkpoint_committed step={step} path={directory} complete=1", flush=True)
        return directory

    def step_loss(item, item_rng):
        x0, text = load_item(item, item_rng, device)
        layout, kind = sample_layout(
            x0, item_rng, args.dense_p, policy=args.layout_policy,
            calibration=entropy_calibration, budget_range=(args.budget_min, args.budget_max),
            random_layout_p=args.random_layout_p, uniform_budget=args.uniform_budget,
            force_kind="dense" if args.dense_only else None,
        )
        sigma = sample_sigmas(1, device, shift=args.dense_shift if kind == "dense" else args.shift)
        noise = torch.randn_like(x0)
        audio_noise = audio_noise_for(dit, x0)
        selector_source = "clean"
        # The selector intervention is deliberately detached from optimization.
        # It uses a frozen LoRA-off dense prediction so the layout choice cannot
        # learn through the mask or silently change the denoising target. This is
        # a partial deployment match; inference may use a LoRA-on warm-up.
        if (args.selector_source == "predicted" and kind == "entropy_budget"
                and not args.dense_only):
            selector_sigma = torch.tensor([args.selector_sigma], device=device, dtype=x0.dtype)
            predicted_clean = dense_teacher_prediction(
                dit, network, x0, noise, selector_sigma, text, audio_noise
            )
            layout, predicted_kind = sample_layout(
                predicted_clean, item_rng, dense_p=0.0, policy=args.layout_policy,
                calibration=entropy_calibration,
                budget_range=(args.budget_min, args.budget_max),
                random_layout_p=0.0, uniform_budget=args.uniform_budget,
                force_kind="adaptive",
            )
            kind = predicted_kind
            selector_source = "predicted_dense_teacher"
        aux = {}
        loss, out, y_t, y0, gap, data_loss = training_objective(
            dit, adapter, network, x0, text, layout, kind, sigma, noise, audio_noise,
            data_weight=args.data_weight, distill_weight=args.distill,
            dense_distill_weight=args.dense_distill,
            detail_distill_weight=args.detail_distill,
            temporal_distill_weight=args.temporal_distill, aux_on_dense=args.dense_only, aux_losses=aux,
        )
        return (loss, out, y_t, y0, sigma, layout, kind, 0.0, data_loss, gap, aux, selector_source)

    def evaluate(step: int) -> dict:
        """Matched teacher/noise controls on the frozen, source-grouped held-out clips."""
        from budget_layout import choose_budget_layout, permute_root_layout

        layout_names = tuple(run_config["eval_layouts"])
        metric_keys = ("flow_mse", "clean_mse", "temporal_mse", "temporal_teacher_mse",
                       "spatial_teacher_mse", "spatial_detail_ratio", "dense_agreement_mse",
                       "forward_seconds", "teacher_gap_mse")
        metrics_by_layout = {name: {key: [] for key in metric_keys} for name in layout_names}
        extent_metrics = {name: {} for name in layout_names}
        rows = []
        sigmas = (.3, .6, .9)
        eval_id = f"{args.evaluation_split}-step-{step:08d}-{time.time_ns()}"
        progress_path = args.out / ("confirmation_progress.json" if args.evaluation_split == "confirmation" else "progress.json")
        network.eval()
        try:
            with torch.no_grad():
                for index, record in enumerate(eval_records):
                    if should_stop():
                        raise InterruptedError("evaluation interrupted; partial rows are not a completed evaluation")
                    item = Path(record["latent_path"]), Path(record["text_path"])
                    x0, text = load_item(item, random.Random(1000 + index), device)
                    frames, token_h, token_w = x0.shape[2], x0.shape[-2] // 2, x0.shape[-1] // 2
                    dense = dense_layout(frames, token_h, token_w)
                    layouts = {"base_dense": dense, "dense_lora": dense, "lot_dense": dense,
                               "uniform1x2": _tile(frames, token_h, token_w, lambda *_: (1, 1, 2)),
                               "uniform2x1": _tile(frames, token_h, token_w, lambda *_: (1, 2, 1)),
                               "uniform2": _tile(frames, token_h, token_w, lambda *_: (1, 2, 2))}
                    mosaic_rng = random.Random(2000 + index)
                    for _ in range(50):
                        layout, kind = sample_layout(x0, mosaic_rng, dense_p=0, policy="legacy")
                        if kind == "mosaic":
                            layouts["legacy_mosaic"] = layout
                            break
                    else:
                        raise RuntimeError("failed to draw legacy mosaic")
                    for policy in ("detail", "entropy", "random"):
                        layouts[policy + "50"] = choose_budget_layout(
                            patchify(x0.float())[0], round(token_h * token_w * args.eval_budget),
                            policy=policy, calibration=entropy_calibration, seed=3000 + index,
                        ).layout
                    # Pure placement control: move complete entropy-quadtree roots,
                    # preserving the exact token count and extent histogram.
                    layouts["random_shuffle50"] = permute_root_layout(
                        layouts["entropy50"], seed=4000 + index
                    )
                    for sigma_index, value in enumerate(sigmas):
                        sig = torch.tensor([value], device=device)
                        gen = torch.Generator(device=device).manual_seed(100_000 + index * len(sigmas) + sigma_index)
                        noise = torch.randn(x0.shape, device=device, generator=gen)
                        audio_noise = audio_noise_for(dit, x0, gen)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        target = dense_teacher_prediction(dit, network, x0, noise, sig, text, audio_noise)
                        torch.cuda.synchronize()
                        teacher_seconds = time.perf_counter() - start
                        for name in layout_names:
                            if should_stop():
                                raise InterruptedError("evaluation time budget exhausted")
                            layout = layouts[name]
                            splice = None if name in {"base_dense", "dense_lora"} else LotSplice(adapter, layout, y_space=True)
                            y0 = x0 if splice is None else splice.to_y(x0)
                            y_t, _ = sample_noisy(y0, sig, noise)
                            if name == "base_dense":
                                clean = target
                                out = (target - y_t) / value
                                forward_seconds = teacher_seconds
                            else:
                                dit._lot = splice
                                torch.cuda.synchronize()
                                start = time.perf_counter()
                                try:
                                    out = dit(y_t, 1.0 - sig, text, audio_noise=audio_noise)
                                finally:
                                    dit._lot = None
                                torch.cuda.synchronize()
                                forward_seconds = time.perf_counter() - start
                                clean_y = y_t.float() + value * out.float()
                                clean = clean_y if splice is None else splice.to_x(clean_y)
                            row = {
                                "eval_id": eval_id, "step": step, "clip": record["clip"],
                                "eval_split": args.evaluation_split,
                                "source_id": record["source_id"], "camera_motion_group": record["camera_motion_group"],
                                "sigma": value, "layout": name,
                                "tokens_per_frame": layout.count / frames,
                                "dense_tokens_per_frame": token_h * token_w,
                                "token_fraction": layout.count / layout.dense_count,
                                "extent_histogram": {
                                    "x".join(map(str, extent)): sum(
                                        rect.et * rect.eh * rect.ew for rect in layout.rects
                                        if rect.extent == extent
                                    ) for extent in sorted({rect.extent for rect in layout.rects})
                                },
                                "flow_mse": float((out.float() - (y0 - y_t) / value).square().mean()),
                                **prediction_diagnostics(clean, x0, target),
                                "dense_agreement_mse": float((clean.float() - target).square().mean()),
                                "teacher_gap_mse": float(teacher_distance(clean, target, sig)),
                                "forward_seconds": forward_seconds,
                                "trainable_lora_enabled": lora_enabled and name != "base_dense",
                                "layout_uses_clean_target": name in {
                                    "legacy_mosaic", "detail50", "entropy50", "random_shuffle50"
                                },
                            }
                            if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                                raise FloatingPointError(f"non-finite evaluation metric: {name}")
                            rows.append(row)
                            for key in metric_keys:
                                if row[key] is not None:
                                    metrics_by_layout[name][key].append(row[key])
                            for extent, error in extent_losses(out, y_t, y0, sig, layout).items():
                                extent_metrics[name].setdefault(extent, []).append(error)
                            append_jsonl(args.out / "metrics.jsonl", {"event": "eval_sample", **row}, sync=True)
                    print(f"LOT_H3 kind=eval_progress step={step} clips={index + 1}/{len(eval_records)} eval_id={eval_id}", flush=True)
                    atomic_write_json(progress_path, {
                        "status": "baseline_evaluating" if step == 0 else "evaluating", "step": step,
                        "eval_split": args.evaluation_split,
                        "eval_id": eval_id, "eval_clip": index + 1, "eval_clips": len(eval_records),
                        "last_checkpoint": "latest.json",
                    })
        finally:
            network.train()
        summary = {"eval_id": eval_id, "eval_split": args.evaluation_split,
                   "step": step, "clips": len(eval_records), "sigmas": list(sigmas),
                   "requested_token_fraction": args.eval_budget, "metrics": {}, "extent_flow_mse": {},
                   "timing_scope": "synchronized DiT forward only; excludes selector, backward, decoder",
                   "adaptive_layout_input": "detail/entropy/legacy oracle clean latent; random independent"}
        for name, values in metrics_by_layout.items():
            summary["metrics"][name] = {
                "samples": len(values["flow_mse"]),
                **{key: {"mean": statistics.fmean(vals), "median": statistics.median(vals),
                         "stddev": statistics.pstdev(vals)} for key, vals in values.items() if vals},
                "tokens_per_frame_mean": statistics.fmean(r["tokens_per_frame"] for r in rows if r["layout"] == name),
            }
            summary["extent_flow_mse"][name] = {key: statistics.fmean(vals) for key, vals in extent_metrics[name].items()}
        metrics_path, summary_path = evaluation_report_paths(args.out, step, args.evaluation_split)
        temporary = metrics_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, metrics_path)
        atomic_write_json(summary_path, summary)
        if step == 0 and args.evaluation_split == "validation":
            atomic_write_json(args.out / "baseline_metrics.json", summary)
        atomic_write_json(progress_path, {
            "status": ("confirmation_complete" if args.evaluation_split == "confirmation"
                       else "baseline_complete" if step == 0 else "running"), "step": step,
            "eval_split": args.evaluation_split,
            "last_eval": metrics_path.with_suffix(".json").name, "last_checkpoint": "latest.json",
        })
        log.write(json.dumps({"event": "eval_complete", "step": step, "eval_id": eval_id,
                              "eval_split": args.evaluation_split}, allow_nan=False) + "\n")
        log.flush(); os.fsync(log.fileno())
        print(f"LOT_H3 kind=train_eval step={step} eval_id={eval_id} metrics={metrics_path}", flush=True)
        return summary

    if args.baseline_only_confirmation:
        try:
            evaluate_confirmation_checkpoint(resume_state, set_lora_multiplier, evaluate)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            log.flush()
            os.fsync(log.fileno())
            log.close()
        _csv_path, summary_path = evaluation_report_paths(args.out, start_step, "confirmation")
        print(f"LOT_H3 kind=confirmation_done step={start_step} metrics={summary_path} ok=1", flush=True)
        return

    def save(tag: str) -> None:
        torch.save({k: v for k, v in adapter.state_dict().items()}, args.out / f"adapter{tag}.pt")
        network.save_weights(str(args.out / f"lora{tag}.safetensors"), torch.bfloat16,
                             {"lot_h3": "phase5", "rank": str(args.rank)})
        print(f"LOT_H3 kind=train_saved tag={tag or 'final'} out={args.out}", flush=True)

    baseline_complete = (args.out / "baseline_metrics.json").is_file()
    if args.resume is not None and start_step == 0 and not baseline_complete:
        set_lora_multiplier(float(resume_state.get("lora_multiplier", 0.0)))
        evaluate(0)
        baseline_complete = True
    elif args.resume is None:
        # The initial snapshot survives even if the baseline evaluation or GPU fails.
        set_lora_multiplier(1.0 if args.init is not None else 0.0)
        save_snapshot(0)
        evaluate(0)
        baseline_complete = True

    if args.baseline_only:
        if not baseline_complete:
            set_lora_multiplier(1.0 if args.init is not None else 0.0)
            evaluate(0)
        log.flush()
        os.fsync(log.fileno())
        log.close()
        print(f"LOT_H3 kind=baseline_done metrics={args.out / 'baseline_metrics.json'} "
              f"checkpoint={args.out / 'latest.json'} ok=1", flush=True)
        return

    if not baseline_complete:
        raise RuntimeError("baseline metrics are missing; refusing to start training")
    set_lora_multiplier(1.0)
    session_steps = 0
    final_step = start_step
    try:
        network.train()
        for step in range(start_step + 1, args.steps + 1):
            if should_stop():
                break
            step_started = time.perf_counter()
            torch.cuda.synchronize()
            # Derive sampling and noise from the global optimizer step. Every
            # branch starting from the same checkpoint sees the same replay,
            # even when its layout consumes a different number of random draws.
            step_seed = args.seed + step * 1_000_003
            item_rng = random.Random(step_seed)
            if args.sampling_policy == "replacement":
                replay_index = None
                item = item_rng.choice(train_items)
            else:
                epoch, offset = divmod(step - 1, len(train_items))
                order = list(range(len(train_items)))
                random.Random(args.seed + 17_171_717 + epoch).shuffle(order)
                replay_index = order[offset]
                item = train_items[replay_index]
            torch.manual_seed(step_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(step_seed)
            (loss, _out, _y_t, _y0, sigma, layout, kind, _forward_seconds,
             data_loss, distill_gap, aux_losses, selector_source_used) = step_loss(item, item_rng)
            optimizer.zero_grad(set_to_none=True)
            require_finite_loss(loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0, error_if_nonfinite=True)
            if not math.isfinite(float(loss.detach())):
                raise RuntimeError(f"step {step}: loss {float(loss)}")
            optimizer.step()
            torch.cuda.synchronize()
            optimizer.zero_grad(set_to_none=True)
            final_step = step
            session_steps += 1
            token_counters["student_tokens"] += layout.count
            token_counters["dense_equivalent_tokens"] += layout.dense_count
            token_counters["teacher_tokens"] += layout.dense_count if distill_gap is not None else 0
            total_elapsed = previous_elapsed + time.perf_counter() - session_started
            record = {
                "event": "train_step",
                "step": step,
                "loss": float(loss.detach()),
                "data_loss": float(data_loss.detach()),
                "dense_distill": args.dense_distill,
                "teacher_gap_mse": (
                    None if distill_gap is None else float(distill_gap.detach())
                ),
                "detail_teacher_grad_mse": (
                    None if aux_losses.get("detail") is None else float(aux_losses["detail"].detach())
                ),
                "temporal_teacher_grad_mse": (
                    None if aux_losses.get("temporal") is None else float(aux_losses["temporal"].detach())
                ),
                "detail_distill_weight": args.detail_distill,
                "temporal_distill_weight": args.temporal_distill,
                "sigma": float(sigma),
                "kind": kind,
                "selector_source": args.selector_source,
                "selector_source_used": selector_source_used,
                "sampling_policy": args.sampling_policy,
                "replay_index": replay_index,
                "clip": clip_stem(item[0]),
                "tokens": layout.count,
                "dense": layout.dense_count,
                "grad_norm": float(grad_norm),
                "step_seconds": time.perf_counter() - step_started,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                "cumulative_tokens": dict(token_counters),
                "elapsed_seconds": round(total_elapsed, 1),
            }
            log.write(json.dumps(record, allow_nan=False) + "\n")
            append_jsonl(args.out / "metrics.jsonl", record, sync=step % 25 == 0)
            if step % args.save_every == 0:
                save_snapshot(step)
            if step % 25 == 0:
                log.flush()
                os.fsync(log.fileno())
                atomic_write_json(args.out / "progress.json", {
                    "status": "running",
                    "step": step,
                    "target_steps": args.steps,
                    "elapsed_seconds": total_elapsed,
                    "loss": record["loss"],
                    "sigma": record["sigma"],
                    "layout": kind,
                    "tokens": layout.count,
                    "dense_tokens": layout.dense_count,
                    "last_checkpoint": str(args.out / "latest.json"),
                })
                gap_text = (
                    "" if record["teacher_gap_mse"] is None
                    else f" gap={record['teacher_gap_mse']:.4f}"
                )
                print(f"LOT_H3 kind=train step={step}/{args.steps} loss={record['loss']:.4f}"
                      f" data={record['data_loss']:.4f}{gap_text} sigma={record['sigma']:.3f} "
                      f"layout={kind} tokens={layout.count}/{layout.dense_count} "
                      f"s_per_step={(time.perf_counter() - session_started) / session_steps:.2f} "
                      f"elapsed_s={total_elapsed:.0f} peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}",
                      flush=True)
            if args.eval_every and step % args.eval_every == 0 and not should_stop():
                try:
                    evaluate(step)
                except InterruptedError:
                    signal_state["stop"] = True
            if should_stop():
                save_snapshot(step)
                atomic_write_json(args.out / "progress.json", {
                    "status": "interrupted_cleanly", "step": step, "target_steps": args.steps,
                    "elapsed_seconds": total_elapsed,
                    "last_checkpoint": str(args.out / "latest.json"),
                })
                break
        if final_step > start_step and final_step % args.save_every != 0:
            save_snapshot(final_step)
        status = "complete" if final_step >= args.steps and not should_stop() else "interrupted_cleanly"
        atomic_write_json(args.out / "progress.json", {
            "status": status, "step": final_step, "target_steps": args.steps,
            "elapsed_seconds": previous_elapsed + time.perf_counter() - session_started,
            "last_checkpoint": str(args.out / "latest.json"),
            "baseline_metrics": "baseline_metrics.json",
        })
        print(f"LOT_H3 kind=train_done steps={final_step}/{args.steps} "
              f"elapsed_s={previous_elapsed + time.perf_counter() - session_started:.0f} "
              f"checkpoint={args.out / 'latest.json'} status={status} ok=1", flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        log.flush()
        os.fsync(log.fileno())
        log.close()


if __name__ == "__main__":
    main()
