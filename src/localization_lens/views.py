from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


PALETTE = np.asarray(
    [
        (230, 57, 70), (29, 185, 84), (0, 118, 182), (255, 183, 3),
        (131, 56, 236), (255, 127, 80), (42, 157, 143), (247, 37, 133),
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class MaskFilter:
    min_area_fraction: float = 0.005
    max_area_fraction: float = 0.95
    max_masks: int = 16
    max_extent_fraction: float = 0.60


def normalize_masks(
    masks: Sequence[np.ndarray],
    shape: tuple[int, int],
    config: MaskFilter = MaskFilter(),
) -> list[np.ndarray]:
    height, width = shape
    selected: list[tuple[int, np.ndarray]] = []
    for raw in masks:
        mask = np.asarray(raw).squeeze().astype(bool)
        if mask.shape != shape:
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (width, height), Image.Resampling.NEAREST
                )
            ) > 0
        area = int(mask.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(mask)
        box_height = int(ys.max() - ys.min() + 1)
        box_width = int(xs.max() - xs.min() + 1)
        # Reject broad annotations even when their actual foreground area is
        # small. Exactly 60% is allowed; either dimension above it is rejected.
        if (
            box_height > config.max_extent_fraction * height
            or box_width > config.max_extent_fraction * width
        ):
            continue
        fraction = area / float(height * width)
        if config.min_area_fraction <= fraction <= config.max_area_fraction:
            selected.append((area, mask))
    selected.sort(key=lambda pair: pair[0], reverse=True)
    return [mask for _, mask in selected[: config.max_masks]]


def filter_background_annotations(
    image: Image.Image, masks: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Conservative image-based filtering, not OCR or anatomical segmentation.

    Bright labels are rejected only near the perimeter with predominantly
    dark surrounding pixels. Interior structures/devices are not text-filtered.
    """
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    dark = gray < 20
    border = np.zeros_like(dark)
    margin_y = max(1, round(height * 0.15))
    margin_x = max(1, round(width * 0.15))
    border[:margin_y] = border[-margin_y:] = True
    border[:, :margin_x] = border[:, -margin_x:] = True
    kept = []
    for mask in masks:
        if dark[mask].mean() >= 0.80:
            continue
        if border[mask].mean() >= 0.80:
            ys, xs = np.nonzero(mask)
            pad_y = max(8, int((ys.max() - ys.min() + 1) * 0.5))
            pad_x = max(8, int((xs.max() - xs.min() + 1) * 0.5))
            y0, y1 = max(0, ys.min() - pad_y), min(height, ys.max() + pad_y + 1)
            x0, x1 = max(0, xs.min() - pad_x), min(width, xs.max() + pad_x + 1)
            surroundings = ~mask[y0:y1, x0:x1]
            if surroundings.any() and dark[y0:y1, x0:x1][surroundings].mean() >= 0.65:
                continue
        kept.append(mask)
    return kept


def render_views(
    image: Image.Image,
    masks: Sequence[np.ndarray],
    output_dir: Path,
    prefix: str,
    alpha: float,
) -> dict[str, str]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    masks = normalize_masks(masks, (height, width))
    masks = filter_background_annotations(image, masks)
    if not masks:
        raise ValueError("No usable masks after area filtering")
    union = np.logical_or.reduce(masks)
    # Resolve overlapping SAM proposals into a disjoint color map before
    # blending. Smaller regions paint over larger ones; adding overlapping
    # RGB values would clip to white and obscure the source anatomy.
    unicolor = union[..., None] * PALETTE[0]
    multicolor = np.zeros_like(rgb)
    for index, mask in enumerate(masks):
        multicolor[mask] = PALETTE[index % len(PALETTE)]
    blend_uni = np.clip(alpha * rgb + (1.0 - alpha) * unicolor, 0, 255)
    blend_multi = np.clip(alpha * rgb + (1.0 - alpha) * multicolor, 0, 255)
    blend_uni[~union] = rgb[~union]
    blend_multi[~union] = rgb[~union]
    masked = rgb * union[..., None]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"{prefix}_unicolor": output_dir / f"{prefix}_unicolor.png",
        f"{prefix}_multicolor": output_dir / f"{prefix}_multicolor.png",
        f"{prefix}_masked": output_dir / f"{prefix}_masked.png",
        f"{prefix}_union": output_dir / f"{prefix}_union.png",
        f"{prefix}_masks": output_dir / f"{prefix}_masks.npz",
    }
    Image.fromarray(blend_uni.astype(np.uint8)).save(outputs[f"{prefix}_unicolor"])
    Image.fromarray(blend_multi.astype(np.uint8)).save(outputs[f"{prefix}_multicolor"])
    Image.fromarray(masked.astype(np.uint8)).save(outputs[f"{prefix}_masked"])
    Image.fromarray(union.astype(np.uint8) * 255).save(outputs[f"{prefix}_union"])
    np.savez_compressed(outputs[f"{prefix}_masks"], masks=np.stack(masks).astype(np.uint8))
    return {key: str(value) for key, value in outputs.items()}
