#!/usr/bin/env python3
"""Same-seed stills: dense H3, LoT with the mean lift, LoT with a Procrustes fit.

    python3 scripts/lot/render_h3.py --pairs DIR [--prompts 3] [--steps 20] [--out DIR]

Text conditioning is a cached H3 text embedding (``*_minimaxh3_te.safetensors``
from a Fizgig cache), so the Qwen3-VL encoder is never loaded. Every column
starts from the same noise. LoT columns use the 768×1152 band layout
(``clip_layout(1, 36, 24)``: 1×1, 2×2, 4×2 bands). The fitted column samples in
y-space and maps the clean latent back with ``to_x`` before decoding. A frozen
base was never trained on fitted scales, so that column is a smoke, not a
quality claim.

Decodes with the fp16 H3 decoder after sampling and writes PNGs plus a grid
into ``--out`` (default ``scripts/lot/runs/render``, gitignored).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/media/2TB/Fizgig/src")

from gpu_guard import refuse_if_busy  # noqa: E402
from h3 import clip_layout, make_h3_adapter  # noqa: E402
from h3_splice import LotSplice  # noqa: E402
from layout import TokenRect, layout_from_rects  # noqa: E402
from procrustes_h3 import (  # noqa: E402
    CHECKPOINT,
    fit_bank,
    load_pairs,
    pretrained_maps,
)

TE_CACHE = Path("/media/2TB/lora-data/fizgig_minimax_h3/cache_iso3d")
CAPTIONS = Path("/media/2TB/lora-data/fizgig_minimax_h3/isometric_3d_stills")
VAE = Path("/media/2TB/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors")
WIDTH, HEIGHT = 768, 1152
# A still fits fully resident: 20.2 GB int8 weights, 20.8 GB peak on the 24 GB card.
# Streaming 32 blocks made every forward ~13.5 s instead of ~0.69 s.
SWAP = 0


def load_prompts(count: int, paths: list[Path] | None = None) -> list[tuple[str, torch.Tensor]]:
    """``count`` cached embeddings with their caption text: ``paths``, else the first by name."""
    from safetensors import safe_open

    prompts = []
    for path in (paths or sorted(TE_CACHE.glob("*_minimaxh3_te.safetensors")))[:count]:
        with safe_open(str(path), "pt") as handle:
            hidden = handle.get_tensor("hidden_states")
            mask = handle.get_tensor("attention_mask")
        stem = path.name.replace("_minimaxh3_te.safetensors", "")
        caption = CAPTIONS / f"{stem}.txt"
        text = caption.read_text().strip() if caption.is_file() else stem
        prompts.append((text, hidden[mask].unsqueeze(0)))
    return prompts


def make_layout(name: str):
    token_h, token_w = HEIGHT // 32, WIDTH // 32
    if name == "bands":
        return clip_layout(1, token_h, token_w)
    side = 2 if name == "uniform2" else 4
    rects = [TokenRect(0, u, v, 1, side, side)
             for u in range(0, token_h, side) for v in range(0, token_w, side)]
    return layout_from_rects(1, token_h, token_w, rects)


def to_image(pixels: torch.Tensor) -> Image.Image:
    array = (pixels[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round().clip(0, 255)
    return Image.fromarray(array.astype(np.uint8))


def grid(rows: list[list[Image.Image]], labels: list[str], scale: float = 0.33) -> Image.Image:
    cell_w, cell_h = int(WIDTH * scale), int(HEIGHT * scale)
    header = 28
    sheet = Image.new("RGB", (cell_w * len(labels), header + cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(labels):
        draw.text((col * cell_w + 6, 8), label, fill="black")
    for row, images in enumerate(rows):
        for col, image in enumerate(images):
            sheet.paste(image.resize((cell_w, cell_h), Image.LANCZOS), (col * cell_w, header + row * cell_h))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "render")
    parser.add_argument("--layout", choices=("bands", "uniform2", "uniform4"), default="bands",
                        help="bands: 1x1/2x2/4x2 thirds; uniformN: every token NxN")
    parser.add_argument("--variants", default="dense,lot_mean,lot_fit")
    parser.add_argument("--swap", type=int, default=SWAP, help="blocks streamed from CPU (0 = all resident)")
    parser.add_argument("--trained", type=Path, default=None,
                        help="train_h3 output dir (adapter.pt, lora.safetensors): enables the dense_lora and "
                             "lot_trained variants and renders the trainer's held-out prompts")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank train_h3 used")
    args = parser.parse_args()
    if args.trained is not None:
        from run_artifacts import resolve_checkpoint

        args.trained = resolve_checkpoint(args.trained)

    refuse_if_busy("render_h3.py")
    pairs = load_pairs(args.pairs)
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.sampling import _sample_image_impl

    # Build only what the requested variants use: each fp32 adapter is ~200 MB, and with
    # the int8 DiT fully resident there is no room for spares.
    wanted = {name for name in args.variants.split(",") if name}
    layout = make_layout(args.layout)
    splices = {}
    reports = []
    if "lot_mean" in wanted:
        mean = make_h3_adapter()
        mean.init_from_pretrained(*pretrained_maps(CHECKPOINT))
        splices["lot_mean"] = LotSplice(mean.cuda(), layout)
    if "lot_fit" in wanted:
        fitted = make_h3_adapter()
        fitted.init_from_pretrained(*pretrained_maps(CHECKPOINT))
        reports = fit_bank(fitted, pairs)
        splices["lot_fit"] = LotSplice(fitted.cuda(), layout, y_space=True)

    model = load_minimax_h3_dit(str(CHECKPOINT), device="cuda", compute_dtype=torch.bfloat16,
                                base_quant="int8", blocks_to_swap=args.swap)
    if args.swap:
        model.enable_block_swap(args.swap, h2d_only=True)
    model.eval()
    model._tread = None
    network = None
    if args.trained is not None:
        from fizgig.networks.lora import create_network
        from train_h3 import LORA_PATTERNS

        trained = make_h3_adapter()
        trained.load_state_dict(torch.load(args.trained / "adapter.pt", map_location="cpu", weights_only=True))
        if "lot_trained" in wanted:
            splices["lot_trained"] = LotSplice(trained.cuda(), layout, y_space=True)
        network = create_network(None, "lora_unet", 1.0, args.rank, float(args.rank), None, [], model,
                                 include_patterns=LORA_PATTERNS)
        network.apply_to(text_encoders=None, unet=model, apply_text_encoder=False, apply_unet=True)
        info = network.load_weights(str(args.trained / "lora.safetensors"))
        if info.missing_keys:
            raise RuntimeError(f"LoRA load missed {len(info.missing_keys)} keys")
        network.to(device="cuda", dtype=torch.bfloat16).eval()

    def set_lora(on: bool) -> None:
        for module in getattr(network, "unet_loras", []):
            module.multiplier = 1.0 if on else 0.0

    held = None
    if args.trained is not None:
        from train_h3 import CACHE, list_items

        held = [te for _latent, te in list_items(CACHE, 24)[1]]
    prompts = load_prompts(args.prompts, held)
    variants = [name for name in args.variants.split(",") if name]
    latents: dict[tuple[int, str], torch.Tensor] = {}
    timings: dict[str, list[float]] = {name: [] for name in variants}
    seq: dict[str, tuple] = {}
    def remember(_module, inputs, _output) -> None:
        # Must return None: a forward hook's return value replaces the block output.
        seq.setdefault("lot" if model._lot is not None else "dense", tuple(inputs[0].shape))

    hook = model.blocks[0].register_forward_hook(remember)
    try:
        for index, (_text, embeds) in enumerate(prompts):
            embeds = embeds.to("cuda", torch.bfloat16)
            for name in variants:
                model._lot = splices.get(name)
                set_lora(name in ("dense_lora", "lot_trained"))
                start = time.perf_counter()
                with torch.inference_mode():
                    latent = _sample_image_impl(model, embeds, width=WIDTH, height=HEIGHT, steps=args.steps,
                                                seed=args.seed + index, num_frames=1)
                torch.cuda.synchronize()
                timings[name].append(time.perf_counter() - start)
                if name in ("lot_fit", "lot_trained"):
                    latent = splices[name].to_x(latent.float())
                latents[(index, name)] = latent.float().cpu()
                print(f"LOT_H3 kind=render prompt={index} variant={name} seconds={timings[name][-1]:.1f} "
                      f"finite={int(bool(torch.isfinite(latent).all()))}", flush=True)
    finally:
        hook.remove()
        model._lot = None
    del model, splices, network
    gc.collect()

    from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
    from safetensors import safe_open

    decoder = MiniMaxH3VideoVAEDecoder()
    with safe_open(str(VAE), framework="pt", device="cpu") as handle:
        decoder.load_state_dict({key: handle.get_tensor(key) for key in handle.keys()}, strict=False)
    decoder = decoder.to("cuda", torch.float16).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(prompts)):
        row = []
        for name in variants:
            with torch.inference_mode():
                image = to_image(decoder.decode(latents[(index, name)].to("cuda")))
            image.save(args.out / f"p{index}_{name}.png")
            row.append(image)
        rows.append(row)
    medians = {name: float(np.median(values)) for name, values in timings.items()}
    labels = [f"{name}  {medians[name]:.0f}s/img" for name in variants]
    grid(rows, labels).save(args.out / "grid.png")
    meta = {
        "layout": args.layout, "steps": args.steps, "seed": args.seed, "canvas": f"{WIDTH}x{HEIGHT}", "layout_tokens": layout.count,
        "dense_tokens": layout.dense_count, "seq": {k: list(v) for k, v in seq.items()},
        "seconds_median": medians, "prompts": [text for text, _ in prompts],
        "fit": [{**r, "extent": list(r["extent"])} for r in reports],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"LOT_H3 kind=render_done out={args.out} seq={seq} "
          + " ".join(f"{k}_s={v:.1f}" for k, v in medians.items()) + " ok=1", flush=True)


if __name__ == "__main__":
    main()
