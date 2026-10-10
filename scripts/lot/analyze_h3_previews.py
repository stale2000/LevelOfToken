"""Offline paired ROI analysis for LoT preview videos.

The dense-LoRA render is a within-checkpoint fidelity reference, not ground
truth. Fine/coarse masks come from the recorded layout at the warm-up switch.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"could not decode video: {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def _layout_masks(record: dict, *, height: int, width: int, frames: int):
    selected = next((step for step in record["trace"] if step.get("layout_selected")), None)
    if selected is None:
        raise RuntimeError(f"no selected layout recorded for {record['variant']}")
    temporal = max((int(rect["t"]) for rect in selected["selected_rectangles"]), default=0) + 1
    fine = np.zeros((temporal, height, width), dtype=bool)
    coarse = np.zeros_like(fine)
    for rect in selected["selected_rectangles"]:
        _et, eh, ew = map(int, rect["extent"])
        top, left, t = int(rect["u"]), int(rect["v"]), int(rect["t"])
        region = (slice(top, top + eh), slice(left, left + ew))
        if eh == ew == 1:
            fine[t][region] = True
        else:
            coarse[t][region] = True
    frame_to_latent = np.rint(np.linspace(0, temporal - 1, frames)).astype(int)
    fine_video = fine[frame_to_latent]
    coarse_video = coarse[frame_to_latent]
    # Each dense H3 spatial token maps to an approximate 32x32 decoded-pixel cell.
    fine_video = np.repeat(np.repeat(fine_video, 32, axis=1), 32, axis=2)
    coarse_video = np.repeat(np.repeat(coarse_video, 32, axis=1), 32, axis=2)
    return selected, fine_video, coarse_video


def _masked_metrics(pred: np.ndarray, dense: np.ndarray, mask: np.ndarray) -> dict:
    if pred.shape != dense.shape or pred.shape[:3] != mask.shape or not mask.any():
        return {"dense_relative_rgb_mse": None, "dense_relative_temporal_mse": None,
                "dense_relative_spatial_gradient_mse": None, "spatial_detail_ratio": None}
    delta = pred - dense
    mse = float(np.square(delta[mask]).mean())
    motion = None
    if pred.shape[0] > 1:
        pred_motion = np.diff(pred, axis=0)
        dense_motion = np.diff(dense, axis=0)
        motion_mask = mask[:-1] & mask[1:]
        motion = (float(np.square((pred_motion - dense_motion)[motion_mask]).mean())
                  if motion_mask.any() else None)
    pred_dx, ref_dx = np.diff(pred, axis=2), np.diff(dense, axis=2)
    pred_dy, ref_dy = np.diff(pred, axis=1), np.diff(dense, axis=1)
    mx = mask[:, :, :-1] & mask[:, :, 1:]
    my = mask[:, :-1, :] & mask[:, 1:, :]
    gradient_terms = []
    pred_energy, ref_energy = [], []
    if mx.any():
        gradient_terms.append(np.square((pred_dx - ref_dx)[mx]).mean())
        pred_energy.append(np.square(pred_dx[mx]).mean())
        ref_energy.append(np.square(ref_dx[mx]).mean())
    if my.any():
        gradient_terms.append(np.square((pred_dy - ref_dy)[my]).mean())
        pred_energy.append(np.square(pred_dy[my]).mean())
        ref_energy.append(np.square(ref_dy[my]).mean())
    grad_mse = float(np.mean(gradient_terms)) if gradient_terms else None
    detail_ratio = (float(np.sqrt(np.mean(pred_energy) / max(np.mean(ref_energy), 1e-12)))
                    if pred_energy and ref_energy else None)
    return {"dense_relative_rgb_mse": mse, "dense_relative_temporal_mse": motion,
            "dense_relative_spatial_gradient_mse": grad_mse,
            "spatial_detail_ratio": detail_ratio}


def _bootstrap_ci(values: list[float], *, seed: int = 0, draws: int = 10_000):
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return [mean, mean]
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [samples[int(.025 * (draws - 1))], samples[int(.975 * (draws - 1))]]


def _crop_panel(sample: dict, output: Path, *, margin: int = 32) -> None:
    """Save the largest entropy-fine/shuffle-coarse patch as a matched crop triplet."""
    dense, entropy, shuffled = sample["frames"]
    fine = sample["entropy_fine"]
    shuffled_fine = sample["shuffle_fine"]
    center = dense.shape[0] // 2
    mask = fine[center] & ~shuffled_fine[center]
    cell_mask = mask.reshape(12, 32, 20, 32).any(axis=(1, 3)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cell_mask, connectivity=4)
    if count <= 1:
        return
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == component)
    y0, y1 = max(0, int(ys.min()) * 32 - margin), min(dense.shape[1], (int(ys.max()) + 1) * 32 + margin)
    x0, x1 = max(0, int(xs.min()) * 32 - margin), min(dense.shape[2], (int(xs.max()) + 1) * 32 + margin)
    images = []
    overlay = entropy[center].copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0
    overlay[mask] = .45 * overlay[mask] + .55 * red[mask]
    for frame in (dense[center], overlay, shuffled[center]):
        crop = (frame[y0:y1, x0:x1] * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(crop)
        image.thumbnail((512, 320))
        images.append(image)
    width, height = max(i.width for i in images), max(i.height for i in images)
    panel = Image.new("RGB", (width * 3, height + 40), "white")
    draw = ImageDraw.Draw(panel)
    for i, (image, label) in enumerate(zip(images, ("dense", "entropy fine ROI", "entropy shuffled"))):
        panel.paste(image, (i * width, 32))
        draw.text((i * width + 4, 8), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def analyze(preview_dirs: list[Path], output_dir: Path) -> dict:
    """Compare matched dense, entropy and coherent-shuffle samples by ROI."""
    sparse_records, dense_records = {}, {}
    for directory in preview_dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        branch = ("dense" if manifest["records"]
                  and all(r["variant"] == "dense_lora" for r in manifest["records"])
                  else "sparse")
        target = dense_records if branch == "dense" else sparse_records
        for record in manifest["records"]:
            key = (record["source_id"], int(record["video_seed"]))
            target.setdefault(key, {})[record["variant"]] = (directory, record)

    samples = []
    for key in sorted(set(sparse_records) & set(dense_records)):
        sparse = sparse_records[key]
        required = ("dense_lora", "entropy_inference", "entropy_inference_shuffle")
        if any(name not in sparse for name in required) or "dense_lora" not in dense_records[key]:
            continue
        dense_dir, dense_record = dense_records[key]["dense_lora"]
        dense = _read_video(dense_dir / f"p{dense_record['item_index']}_dense_lora.mp4")
        variants, masks = [], {}
        for variant, mode in (("entropy_inference", "entropy"),
                              ("entropy_inference_shuffle", "shuffle")):
            directory, record = sparse[variant]
            pred = _read_video(directory / f"p{record['item_index']}_{variant}.mp4")
            selected, fine, coarse = _layout_masks(record, height=12, width=20, frames=pred.shape[0])
            variants.append(pred)
            masks[mode] = (fine, coarse)
            if mode == "entropy":
                histogram = selected.get("selected_extent_token_histogram", {})
            if len(dense) != len(pred):
                raise RuntimeError(f"frame count mismatch for {key}: dense={len(dense)} adaptive={len(pred)}")
        fine_entropy = masks["entropy"][0]
        fine_shuffle = masks["shuffle"][0]
        regions = {
            "entropy_fine": fine_entropy,
            "entropy_coarse": masks["entropy"][1],
            "entropy_extra_fine": fine_entropy & ~fine_shuffle,
            "shuffle_extra_fine": fine_shuffle & ~fine_entropy,
        }
        row = {"source_id": key[0], "video_seed": key[1], "token_histogram": histogram,
               "metrics": {}}
        for region, mask in regions.items():
            row["metrics"][region] = {}
            for variant, pred in zip(("entropy", "shuffle"), variants):
                row["metrics"][region][variant] = {
                    **_masked_metrics(pred, dense, mask), "pixel_count": int(mask.sum())}
        samples.append(row)
        _crop_panel({"frames": (dense, variants[0], variants[1]),
                     "entropy_fine": fine_entropy, "shuffle_fine": fine_shuffle},
                    output_dir / "roi-crops" / f"{key[0]}-{key[1]}.png")

    by_source = defaultdict(list)
    for sample in samples:
        by_source[sample["source_id"]].append(sample)
    paired = {}
    for region in ("entropy_fine", "entropy_coarse", "entropy_extra_fine", "shuffle_extra_fine"):
        paired[region] = {}
        for metric in ("dense_relative_rgb_mse", "dense_relative_temporal_mse",
                       "dense_relative_spatial_gradient_mse", "spatial_detail_ratio"):
            by_source_delta = []
            for source_samples in by_source.values():
                ds = []
                for sample in source_samples:
                    vals = sample["metrics"][region]
                    a, b = vals["entropy"][metric], vals["shuffle"][metric]
                    if a is not None and b is not None:
                        ds.append(a - b)
                if ds:
                    by_source_delta.append(statistics.fmean(ds))
            paired[region][metric] = {
                "entropy_minus_shuffle": statistics.fmean(by_source_delta) if by_source_delta else None,
                "source_group_bootstrap_ci95": _bootstrap_ci(by_source_delta),
                "source_clusters": len(by_source_delta),
            }
    summary = {
        "metric_interpretation": "RGB MSE to the same sparse checkpoint's dense-LoRA render; a within-model fidelity proxy, not ground-truth quality.",
        "layout_interpretation": "Fine/coarse pixel masks are approximate 32x32 mappings of each saved warm-up layout. Entropy and shuffle share extent histograms and 840/1680 tokens.",
        "samples": samples,
        "paired_roi_effects": paired,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "roi_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    panels = sorted((output_dir / "roi-crops").glob("*.png"))
    if panels:
        opened = [Image.open(path).convert("RGB") for path in panels]
        width, height = max(i.width for i in opened), max(i.height for i in opened)
        sheet = Image.new("RGB", (width, height * len(opened)), "white")
        for index, image in enumerate(opened):
            sheet.paste(image, (0, index * height))
        sheet.save(output_dir / "roi-crops-contact-sheet.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-root", type=Path, required=True,
                        help="campaign results/previews directory")
    parser.add_argument("--out", type=Path, required=True, help="ROI analysis output directory")
    args = parser.parse_args()
    previews = [path.parent for path in args.preview_root.rglob("manifest.json")]
    summary = analyze(previews, args.out)
    print(json.dumps({"samples": len(summary["samples"]),
                      "paired_roi_effects": summary["paired_roi_effects"],
                      "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
