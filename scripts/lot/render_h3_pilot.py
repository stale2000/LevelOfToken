#!/usr/bin/env python3
"""Bounded same-noise video controls for a LoT pilot (CFG 1, pure Euler).

Read held-out prompts from the original training run's ``evaluation_items``.
Raw generated latents and an incremental manifest survive a later decode failure.
The optional entropy variant selects once from the warm-up prediction, never
from the reference clean latent. Requires the pinned LoT-aware Fizgig on GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from budget_layout import BudgetLayout
from h3_splice import (
    BudgetedLayoutController,
    LotSplice,
    h3_clean_estimate,
    transition_layout_state,
)
from layout import TokenRect, layout_from_rects
from run_artifacts import atomic_write_json, resolve_checkpoint

WIDTH, HEIGHT, LATENT_T, PIXEL_FRAMES, FPS = 640, 384, 7, 22, 24
VIDEO_SHIFT, AUDIO_SHIFT = 12.0, 3.0
BASE_CONTROLS = ("base_dense", "dense_lora", "uniform2")
SELECTOR_CONTROLS = (
    "detail_inference", "detail_inference_shuffle", "entropy_inference", "entropy_inference_shuffle",
    "oracle_detail", "oracle_detail_shuffle",
)
CONTROLS = BASE_CONTROLS + SELECTOR_CONTROLS
CAMPAIGN_CONTROLS = ("uniform50_1x2", "uniform50_2x1", "detail50_warmup")
ALL_CONTROLS = CONTROLS + CAMPAIGN_CONTROLS


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def initial_noise(seed: int, audio_shape: tuple[int, int], device="cpu"):
    """Independent generators make each explicit stream reproducible across variants."""
    video = torch.randn(1, 24, LATENT_T, HEIGHT // 16, WIDTH // 16,
                        generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    audio = torch.randn(*audio_shape, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float32)
    return video.to(device), audio.to(device)


def comfy_sigmas(steps: int) -> list[float]:
    """Fizgig sample_schedule(mode='comfy', shift=12), with steps forward calls."""
    if not 1 <= steps <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    table = [VIDEO_SHIFT * u / (1 + (VIDEO_SHIFT - 1) * u)
             for u in ((i + 1) / 1000 for i in range(1000))]
    return [table[-(1 + int(i * 1000 / steps))] for i in range(steps)] + [0.0]


def audio_sigma(video_sigma: float) -> float:
    """Match Fizgig remap_sigma(video, 12, 3)."""
    u = video_sigma / (VIDEO_SHIFT + video_sigma * (1 - VIDEO_SHIFT))
    return AUDIO_SHIFT * u / (1 + (AUDIO_SHIFT - 1) * u)


def make_layout(side: int):
    height, width = HEIGHT // 32, WIDTH // 32
    if side not in (1, 2):
        raise ValueError("pilot layouts support side 1 or 2")
    return layout_from_rects(LATENT_T, height, width, [
        TokenRect(t, u, v, 1, side, side)
        for t in range(LATENT_T) for u in range(0, height, side)
        for v in range(0, width, side)
    ])


def make_tiling(tile_h: int, tile_w: int):
    height, width = HEIGHT // 32, WIDTH // 32
    if height % tile_h or width % tile_w:
        raise ValueError("fixed tiling must divide the H3 latent grid")
    return layout_from_rects(LATENT_T, height, width, [
        TokenRect(t, u, v, 1, tile_h, tile_w)
        for t in range(LATENT_T)
        for u in range(0, height, tile_h)
        for v in range(0, width, tile_w)
    ])


class FixedLayoutController:
    """Switch once from dense inference to a declared fixed tiling."""

    def __init__(self, splice, layout, *, warmup_steps: int, warmup_sigma: float | None):
        if warmup_steps < 0 or (warmup_sigma is not None and not 0 <= warmup_sigma <= 1):
            raise ValueError("fixed-layout warmup is invalid")
        self.splice = splice
        self.layout = layout
        self.warmup_steps = warmup_steps
        self.warmup_sigma = warmup_sigma
        self.last_selection_step = None
        self.last_selection = None

    @torch.no_grad()
    def after_euler_step(self, state_before, output, sigma_before, state_after,
                         sigma_after, completed_steps):
        crossed_sigma = (self.warmup_sigma is not None
                         and float(torch.as_tensor(sigma_after)) <= self.warmup_sigma)
        warmup_complete = crossed_sigma if self.warmup_sigma is not None else completed_steps >= self.warmup_steps
        if self.last_selection_step is not None or not warmup_complete:
            return state_after, self.splice, None
        clean_x = self.splice.to_x(h3_clean_estimate(state_before, output, sigma_before))
        next_splice = LotSplice(self.splice.adapter, self.layout, y_space=True)
        changed = transition_layout_state(state_after, clean_x, sigma_after, self.splice, next_splice)
        self.splice = next_splice
        self.last_selection_step = completed_steps
        per_frame = self.layout.count // LATENT_T
        self.last_selection = BudgetLayout(self.layout, "fixed_tiling", per_frame, per_frame, 0.0)
        return changed, next_splice, self.last_selection


def configure_control(model, network, adapter, variant: str, *, calibration=None,
                      warmup_steps=3, warmup_sigma=None, budget_fraction=0.25,
                      seed=0, oracle_latent=None):
    if variant not in ALL_CONTROLS:
        raise ValueError(f"unknown preview control: {variant}")
    for module in network.unet_loras:
        module.multiplier = 0.0 if variant == "base_dense" else 1.0
    model._lot = LotSplice(adapter, make_layout(2), y_space=True) if variant == "uniform2" else None
    controller = None
    selectors = {
        "detail_inference": ("detail", False, False),
        "detail_inference_shuffle": ("detail", False, True),
        "entropy_inference": ("entropy", False, False),
        "entropy_inference_shuffle": ("entropy", False, True),
        "oracle_detail": ("detail", True, False),
        "oracle_detail_shuffle": ("detail", True, True),
    }
    if variant in selectors:
        policy, oracle, shuffle = selectors[variant]
        if policy == "entropy" and calibration is None:
            raise ValueError("entropy inference requires the training calibration")
        if oracle and oracle_latent is None:
            raise ValueError(f"{variant} requires the ground-truth clean latent")
        # This unit-layout splice is coordinate-identical to the dense warm-up.
        controller = BudgetedLayoutController(
            LotSplice(adapter, make_layout(1), y_space=True),
            math.floor((HEIGHT // 32) * (WIDTH // 32) * budget_fraction),
            policy=policy, calibration=calibration, warmup_steps=warmup_steps,
            warmup_sigma=warmup_sigma, refresh_interval=0, seed=seed,
            selection_latent=oracle_latent if oracle else None,
            shuffle_roots=shuffle,
        )
    elif variant in {"uniform50_1x2", "uniform50_2x1"}:
        tile_h, tile_w = (1, 2) if variant.endswith("1x2") else (2, 1)
        model._lot = None
        controller = FixedLayoutController(
            LotSplice(adapter, make_layout(1), y_space=True), make_tiling(tile_h, tile_w),
            warmup_steps=warmup_steps, warmup_sigma=warmup_sigma,
        )
    elif variant == "detail50_warmup":
        model._lot = None
        controller = BudgetedLayoutController(
            LotSplice(adapter, make_layout(1), y_space=True),
            math.floor((HEIGHT // 32) * (WIDTH // 32) * 0.5),
            policy="detail", warmup_steps=warmup_steps, warmup_sigma=warmup_sigma,
            refresh_interval=0, seed=seed,
        )
    return controller


@torch.inference_mode()
def sample_euler(model, text, video_noise, audio_noise, sigmas, *, controller=None,
                 dtype=torch.bfloat16):
    """H3 x0-minus-noise Euler, including the exact carried audio velocity.

    Inputs are cloned: every control begins with the exact same two streams.
    Layout conversion occurs after an Euler boundary and applies only to the
    estimated clean component. Once chosen, the layout is frozen.
    """
    video, audio = video_noise.float().clone(), audio_noise.float().clone()
    trace = []
    for index, (sigma, next_sigma) in enumerate(itertools.pairwise(sigmas)):
        if not (0 <= next_sigma < sigma <= 1):
            raise ValueError("sigmas must descend strictly from <=1 to >=0")
        before = video
        step_start = time.perf_counter()
        sa = audio_sigma(sigma)
        audio_in = audio * (sa / sigma)
        timestep = torch.tensor([1 - sigma], device=video.device, dtype=dtype)
        splice = getattr(model, "_lot", None)
        start = time.perf_counter()
        output, audio_raw = model(video.to(dtype), timestep, text,
                                  audio_rows=audio_in.to(dtype), return_audio=True)
        if video.is_cuda:
            torch.cuda.synchronize(video.device)
        seconds = time.perf_counter() - start
        output = output.float()
        if audio_raw is None:
            raise RuntimeError("video preview requires joint H3 audio predictions")
        if not torch.isfinite(output).all() or not torch.isfinite(audio_raw).all():
            raise RuntimeError(f"non-finite prediction at Euler step {index + 1}")
        scale = VIDEO_SHIFT / AUDIO_SHIFT
        audio_velocity = ((scale - 1) * audio_in
                          + (1 + (scale - 1) * sa) * audio_raw.float())
        video = video + (sigma - next_sigma) * output
        audio = audio + (sigma - next_sigma) * audio_velocity
        selection = None
        if controller is not None and next_sigma > 0:
            video, active, selection = controller.after_euler_step(
                before, output, sigma, video, next_sigma, index + 1)
            if controller.last_selection_step is not None:
                model._lot = active
        if video.is_cuda:
            torch.cuda.synchronize(video.device)
        trace.append({
            "step": index + 1, "sigma": sigma, "next_sigma": next_sigma,
            "audio_sigma": sa, "seconds": time.perf_counter() - step_start,
            "forward_seconds": seconds,
            "video_tokens": splice.layout.count if splice else LATENT_T * (HEIGHT // 32) * (WIDTH // 32),
            "layout_selected": selection is not None,
            "selected_tokens": selection.layout.count if selection else None,
            "selected_dense_tokens": selection.layout.dense_count if selection else None,
            "selection_policy": selection.policy if selection else None,
            "selection_mean_refinement_score": selection.mean_refinement_score if selection else None,
            "selected_extent_token_histogram": (
                _extent_token_histogram(selection.layout) if selection else None
            ),
            "selected_extent_rectangle_histogram": (
                _extent_rectangle_histogram(selection.layout) if selection else None
            ),
            "selected_rectangles": (
                _layout_rectangles(selection.layout) if selection else None
            ),
        })
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise RuntimeError("non-finite final latent")
    active = getattr(model, "_lot", None)
    if active is not None:
        video = active.to_x(video)
    return video.float().cpu(), (audio / (VIDEO_SHIFT / AUDIO_SHIFT)).cpu(), trace


def _extent_token_histogram(layout):
    counts = {}
    for rect in layout.rects:
        name = "x".join(map(str, rect.extent))
        counts[name] = counts.get(name, 0) + rect.et * rect.eh * rect.ew
    return dict(sorted(counts.items()))


def _extent_rectangle_histogram(layout):
    counts = {}
    for rect in layout.rects:
        name = "x".join(map(str, rect.extent))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _layout_rectangles(layout):
    return [{"t": rect.t, "u": rect.u, "v": rect.v, "extent": list(rect.extent)}
            for rect in layout.rects]


def load_reference(path: Path):
    """Center crop exactly 7 latent frames and the shared 384x640 canvas."""
    from safetensors import safe_open

    with safe_open(str(path), "pt") as handle:
        latent = handle.get_tensor(next(key for key in handle.keys() if key.startswith("latent"))).float()  # noqa: SIM118
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 4 or latent.shape[0] != 24:
        raise ValueError(f"expected cached (24,T,H,W), got {tuple(latent.shape)}")
    if any(a < b for a, b in zip(latent.shape[1:], (LATENT_T, HEIGHT // 16, WIDTH // 16))):
        raise ValueError(f"reference too small for pilot canvas: {tuple(latent.shape)}")
    top, left = (latent.shape[-2] - HEIGHT // 16) // 2, (latent.shape[-1] - WIDTH // 16) // 2
    crop = latent[:, :LATENT_T, top:top + HEIGHT // 16, left:left + WIDTH // 16].unsqueeze(0)
    return crop, {"latent_start_frame": 0, "top": top, "left": left, "height": HEIGHT // 16,
                  "width": WIDTH // 16, "latent_frames": LATENT_T}


def write_mp4(pixels, path: Path):
    """Use existing ffmpeg, with no Python video dependency."""
    frames = (pixels[0].permute(1, 2, 3, 0).float().cpu().clamp(0, 1) * 255).round().byte().numpy()
    if frames.shape != (PIXEL_FRAMES, HEIGHT, WIDTH, 3):
        raise ValueError(f"unexpected decoded video shape: {frames.shape}")
    process = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, error = process.communicate(frames.tobytes())
    if process.returncode:
        raise RuntimeError(f"ffmpeg encode failed: {error.decode(errors='replace')}")
    from PIL import Image

    return [Image.fromarray(frames[index]) for index in (0, PIXEL_FRAMES // 2, PIXEL_FRAMES - 1)]


def contact_sheet(rows, path: Path):
    from PIL import Image, ImageDraw

    cell_w, cell_h, label_h = WIDTH // 2, HEIGHT // 2, 36
    sheet = Image.new("RGB", (cell_w * 3, (cell_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, frames) in enumerate(rows):
        y = row * (cell_h + label_h)
        draw.text((6, y + 8), label, fill="black")
        for column, frame in enumerate(frames):
            sheet.paste(frame.resize((cell_w, cell_h)), (column * cell_w, y + label_h))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--items", type=int, default=2)
    parser.add_argument("--split", choices=("validation", "confirmation"), default="validation")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--warmup-sigma", type=float,
                        help="switch after the first Euler boundary with next sigma at or below this value")
    parser.add_argument("--budget-fraction", type=float, default=0.25,
                        help="selector budget; default 25% matches fixed 2x2")
    parser.add_argument("--entropy-inference", action="store_true",
                        help="run the full same-noise detail/entropy/oracle/shuffle selector bake-off")
    parser.add_argument("--controls", help="comma-separated control names (overrides default control set)")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--swap", type=int, default=0)
    parser.add_argument("--vae", type=Path, default=Path("/media/2TB/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors"))
    parser.add_argument("--fizgig-src", type=Path, default=Path("/media/2TB/Fizgig/src"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg must be available before allocating the GPU")
    if not 1 <= args.items <= 4 or not 2 <= args.steps <= 30:
        raise SystemExit("pilot must use 1-4 items and 2-30 Euler steps")
    if (not 1 <= args.warmup_steps < args.steps or not 0.25 <= args.budget_fraction <= 1
            or args.warmup_sigma is not None and not 0 <= args.warmup_sigma <= 1):
        raise SystemExit("warmup must precede final step; budget fraction must be in [0.25,1]")
    variants = ([part.strip() for part in args.controls.split(",") if part.strip()]
                if args.controls else list(CONTROLS if args.entropy_inference else BASE_CONTROLS))
    if not variants or any(variant not in ALL_CONTROLS for variant in variants):
        raise SystemExit(f"--controls must use: {', '.join(ALL_CONTROLS)}")
    run = json.loads((args.run / "run.json").read_text())
    available = run.get(f"{args.split}_items", run.get("evaluation_items", []))
    items = available[:args.items]
    if len(items) != args.items:
        raise SystemExit(f"run.json must contain enough explicit {args.split}_items")
    required = {"clip", "latent_path", "text_path", "caption", "source_id", "camera_motion_group"}
    if any(not required <= set(item) for item in items):
        raise SystemExit("evaluation_items missing clip/cache/caption/source/camera metadata")
    paths = [args.vae, Path(run["base_checkpoint"])]
    paths.extend(Path(item[key]) for item in items for key in ("latent_path", "text_path"))
    needs_calibration = args.entropy_inference or any(variant.startswith("entropy_") for variant in variants)
    if needs_calibration:
        paths.append(args.run / "entropy_calibration.pt")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"preview assets missing: {missing}")
    checkpoint = resolve_checkpoint(args.checkpoint or args.run)
    out_dir = args.out or args.run / "previews" / checkpoint.name
    if (out_dir / "manifest.json").exists():
        raise SystemExit(f"preview exists; choose a fresh --out: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = comfy_sigmas(args.steps)
    manifest = {"status": "sampling", "run": str(args.run.resolve()), "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": {name: file_hash(checkpoint / name) for name in ("adapter.pt", "lora.safetensors")},
                "cfg": 1.0, "sampler": "pure_euler", "schedule": "comfy_simple_shift12",
                "sigmas": sigmas, "audio_shift": AUDIO_SHIFT, "width": WIDTH, "height": HEIGHT,
                "pixel_frames": PIXEL_FRAMES, "latent_frames": LATENT_T, "fps": FPS,
                "warmup_steps": args.warmup_steps, "warmup_sigma": args.warmup_sigma,
                "budget_fraction": args.budget_fraction,
                "token_fraction_of_uniform2": args.budget_fraction / 0.25,
                "variants": variants,
                "oracle_layout_variants": [variant for variant in variants if variant.startswith("oracle_")],
                "selector_test_design": {
                    "detail": "within-block residual energy; warm-up-predicted clean latent",
                    "entropy": "normalized per-feature entropy; warm-up-predicted clean latent",
                    "oracle": "ground-truth reference latent used for layout selection only",
                    "matched_shuffle": "same selected rectangle extents, roots shuffled by supercell with one spatial permutation shared across latent frames",
                    "selector_budget_fraction": args.budget_fraction,
                },
                "torch_version": torch.__version__,
                "reference": "cached clean latent decode; generation is text-to-video, not reconstruction",
                "split": args.split, "records": []}
    atomic_write_json(out_dir / "manifest.json", manifest)
    sys.path.insert(0, str(args.fizgig_src))
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.model import AUDIO_CHANNELS, audio_latents_for_frames
    from fizgig.networks.lora import create_network
    from h3 import make_h3_adapter
    from train_h3 import LORA_PATTERNS

    model = load_minimax_h3_dit(run["base_checkpoint"], device="cuda", compute_dtype=torch.bfloat16,
                              base_quant="int8", blocks_to_swap=args.swap)
    if args.swap:
        model.enable_block_swap(args.swap, h2d_only=True)
    model.eval().requires_grad_(False)
    model._tread = None
    if not getattr(model, "pack_audio_rows", False):
        raise RuntimeError("H3 video preview requires packed joint audio")
    adapter = make_h3_adapter()
    adapter.load_state_dict(torch.load(checkpoint / "adapter.pt", map_location="cpu", weights_only=True))
    adapter.cuda().eval().requires_grad_(False)
    rank = int(run["rank"])
    network = create_network(None, "lora_unet", 1.0, rank, float(rank), None, [], model,
                             include_patterns=LORA_PATTERNS)
    network.apply_to(text_encoders=None, unet=model, apply_text_encoder=False, apply_unet=True)
    info = network.load_weights(str(checkpoint / "lora.safetensors"))
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"LoRA checkpoint does not match model: {info}")
    network.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    calibration = None
    if needs_calibration:
        from budget_layout import EntropyCalibration

        state = torch.load(args.run / "entropy_calibration.pt", map_location="cpu", weights_only=True)
        calibration = EntropyCalibration(state["mean"], state["std"])
    from safetensors import safe_open

    for index, item in enumerate(items):
        reference, crop = load_reference(Path(item["latent_path"]))
        torch.save(reference, out_dir / f"p{index}_reference.pt")
        with safe_open(str(item["text_path"]), "pt") as handle:
            text = handle.get_tensor("hidden_states")[handle.get_tensor("attention_mask")].unsqueeze(0).to("cuda", torch.bfloat16)
        seed = args.seed + 2 * index
        video_noise, audio_noise = initial_noise(seed, (audio_latents_for_frames(PIXEL_FRAMES) * AUDIO_CHANNELS,
                                                       model.config.audio_latents_dim), "cuda")
        torch.save({"video_noise": video_noise.cpu(), "audio_noise": audio_noise.cpu()}, out_dir / f"p{index}_noise.pt")
        for variant in variants:
            controller = configure_control(
                model, network, adapter, variant, calibration=calibration,
                warmup_steps=args.warmup_steps, warmup_sigma=args.warmup_sigma,
                budget_fraction=args.budget_fraction, seed=seed,
                oracle_latent=reference if variant.startswith("oracle_") else None,
            )
            torch.cuda.synchronize()
            start = time.perf_counter()
            latent, audio, trace = sample_euler(model, text, video_noise, audio_noise, sigmas, controller=controller)
            seconds = time.perf_counter() - start
            filename = f"p{index}_{variant}.pt"
            torch.save({"video": latent, "audio": audio}, out_dir / filename)
            record = {"item_index": index, **item, "variant": variant, "video_seed": seed,
                      "audio_seed": seed + 1, "reference_crop": crop, "raw_latent": filename,
                      "raw_latent_sha256": file_hash(out_dir / filename), "trace": trace,
                      "seconds": seconds, "forward_seconds": sum(step["forward_seconds"] for step in trace),
                      "oracle_layout": variant.startswith("oracle_")}
            manifest["records"].append(record)
            atomic_write_json(out_dir / "manifest.json", manifest)
            print(f"LOT_H3 kind=pilot_preview item={index} variant={variant} seconds={record['seconds']:.1f}", flush=True)
    model._lot = None
    del model, network, adapter, video_noise, audio_noise, text, controller
    gc.collect()
    manifest["status"] = "decoding"
    atomic_write_json(out_dir / "manifest.json", manifest)
    from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder

    decoder = MiniMaxH3VideoVAEDecoder()
    with safe_open(str(args.vae), "pt") as handle:
        info = decoder.load_state_dict({key: handle.get_tensor(key) for key in handle.keys()}, strict=False)  # noqa: SIM118
    if set(info.missing_keys) - {"latents_mean", "latents_std"}:
        raise RuntimeError(f"VAE decoder weights missing keys: {info.missing_keys}")
    manifest["vae_sha256"] = file_hash(args.vae)
    manifest["vae_missing_constant_buffers"] = list(info.missing_keys)
    decoder.to("cuda", torch.float16).eval().requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for index, item in enumerate(items):
            for variant in ["reference", *variants]:
                blob = torch.load(out_dir / f"p{index}_{variant}.pt", weights_only=True)
                latent = blob if variant == "reference" else blob["video"]
                torch.cuda.synchronize()
                start = time.perf_counter()
                pixels = decoder.decode(latent.to("cuda", torch.float16))
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                filename = f"p{index}_{variant}.mp4"
                frames = write_mp4(pixels, out_dir / filename)
                rows.append((f"p{index} {variant} | {item['camera_motion_group']}", frames))
                manifest.setdefault("videos", []).append({"item_index": index, "variant": variant,
                                                           "file": filename, "decode_seconds": seconds,
                                                           "sha256": file_hash(out_dir / filename)})
                atomic_write_json(out_dir / "manifest.json", manifest)
                del pixels
    contact_sheet(rows, out_dir / "comparison.png")
    manifest["status"] = "complete"
    manifest["contact_sheet"] = "comparison.png"
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(f"LOT_H3 kind=pilot_preview_complete out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
