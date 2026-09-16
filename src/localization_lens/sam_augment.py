from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .io import read_jsonl, stratified_image_limit
from .views import render_views


def crop_boxes(width: int, height: int, layers: int) -> list[tuple[int, int, int, int]]:
    """Original image plus SAM-style overlapping crops, in pixel coordinates."""
    boxes = [(0, 0, width, height)]
    for layer in range(1, layers + 1):
        side = 2**layer
        overlap = int((512 / 1500) * min(width, height) * (2 / side))
        crop_width = math.ceil((width + overlap * (side - 1)) / side)
        crop_height = math.ceil((height + overlap * (side - 1)) / side)
        for x_index in range(side):
            for y_index in range(side):
                x = x_index * (crop_width - overlap)
                y = y_index * (crop_height - overlap)
                boxes.append((x, y, min(x + crop_width, width), min(y + crop_height, height)))
    return boxes


def touches_internal_crop_edge(mask, box, image_size, tolerance=20) -> bool:
    """Discard truncated proposals; actual image boundaries remain valid."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return False
    x0, y0, x1, y1 = box
    width, height = image_size
    return bool(
        (x0 > 0 and xs.min() <= tolerance)
        or (y0 > 0 and ys.min() <= tolerance)
        or (x1 < width and mask.shape[1] - 1 - xs.max() <= tolerance)
        or (y1 < height and mask.shape[0] - 1 - ys.max() <= tolerance)
    )


def generate_masks(generator, image: Image.Image, args) -> list[np.ndarray]:
    """Run crops independently to avoid Transformers stacking unequal raw crops."""
    proposals = []
    for box in crop_boxes(image.width, image.height, args.crops_n_layers):
        x0, y0, x1, y1 = box
        crop = image.crop(box)
        result = generator(
            crop,
            points_per_batch=args.points_per_batch,
            points_per_crop=args.points_per_crop,
            crops_n_layers=0,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )
        scores = result.get("scores", [0.0] * len(result["masks"]))
        for raw, score in zip(result["masks"], scores):
            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            mask = np.asarray(raw, dtype=bool).squeeze()
            if mask.shape != (crop.height, crop.width):
                raise ValueError(f"SAM mask shape {mask.shape} does not match crop size")
            if touches_internal_crop_edge(mask, box, image.size):
                continue
            full_mask = np.zeros((image.height, image.width), dtype=bool)
            full_mask[y0:y1, x0:x1] = mask
            if full_mask.any():
                proposals.append((float(score), full_mask))
    # Prefer higher-confidence proposals when the full-image and crop passes
    # describe essentially the same region. Nested, distinct regions survive.
    proposals.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _, mask in proposals:
        duplicate = False
        for previous in selected:
            intersection = np.count_nonzero(mask & previous)
            union = np.count_nonzero(mask | previous)
            if intersection / union > 0.7:
                duplicate = True
                break
        if not duplicate:
            selected.append(mask)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SAM localization-lens views")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam-vit-huge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--points-per-crop", type=int, default=32)
    parser.add_argument("--crops-n-layers", type=int, default=0)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.80)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import pipeline

    device = 0 if args.device.startswith("cuda") else -1
    generator = pipeline("mask-generation", model=args.model, device=device)
    rows = stratified_image_limit(read_jsonl(args.manifest), args.max_images)
    rng = np.random.default_rng(args.seed)
    root = args.manifest.parent
    completed = 0
    for row in rows:
        destination = args.output / row["image_id"]
        marker = destination / "sam_multicolor.png"
        if marker.exists() and not args.overwrite:
            completed += 1
            continue
        image = Image.open(root / row["image"]).convert("RGB")
        masks = generate_masks(generator, image, args)
        alpha = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))
        try:
            render_views(image, masks, destination, "sam", alpha)
            completed += 1
        except ValueError as exc:
            print(f"skip {row['image_id']}: {exc}")
    print(f"SAM views ready for {completed}/{len(rows)} unique images")


if __name__ == "__main__":
    main()
