from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def boxes_to_patch_mask(
    boxes: Iterable[tuple[float, float, float, float]],
    image_width: int,
    image_height: int,
    *,
    crop_size: int = 224,
    patch_grid: int = 16,
) -> np.ndarray:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if crop_size <= 0 or patch_grid <= 0 or crop_size % patch_grid:
        raise ValueError("crop_size must be positive and divisible by patch_grid")
    if image_height <= image_width:
        resized_height = crop_size
        resized_width = int(crop_size * image_width / image_height)
    else:
        resized_width = crop_size
        resized_height = int(crop_size * image_height / image_width)
    scale_x, scale_y = resized_width / image_width, resized_height / image_height
    crop_left = (resized_width - crop_size) // 2
    crop_top = (resized_height - crop_size) // 2
    cell = crop_size / patch_grid
    mask = np.zeros((patch_grid, patch_grid), dtype=np.bool_)
    for raw_box in boxes:
        if len(raw_box) != 4:
            raise ValueError(f"Box must have four coordinates, got {raw_box!r}")
        x1, y1, x2, y2 = map(float, raw_box)
        if not np.isfinite([x1, y1, x2, y2]).all():
            raise ValueError(f"Box contains non-finite coordinates: {raw_box!r}")
        left = min(max(x1 * scale_x - crop_left, 0.0), float(crop_size))
        right = min(max(x2 * scale_x - crop_left, 0.0), float(crop_size))
        top = min(max(y1 * scale_y - crop_top, 0.0), float(crop_size))
        bottom = min(max(y2 * scale_y - crop_top, 0.0), float(crop_size))
        if right <= left or bottom <= top:
            continue
        col0 = min(patch_grid - 1, int(math.floor(left / cell)))
        row0 = min(patch_grid - 1, int(math.floor(top / cell)))
        col1 = min(patch_grid - 1, int(math.ceil(right / cell) - 1))
        row1 = min(patch_grid - 1, int(math.ceil(bottom / cell) - 1))
        mask[row0 : row1 + 1, col0 : col1 + 1] = True
    return mask


def _proposal_value(proposal: Any, name: str) -> float:
    if isinstance(proposal, dict):
        if "xyxy" in proposal and name in ("x1", "y1", "x2", "y2"):
            return float(proposal["xyxy"][("x1", "y1", "x2", "y2").index(name)])
        return float(proposal[name])
    return float(getattr(proposal, name))


def proposal_union_mask(
    proposals: Iterable[Any],
    image_width: int,
    image_height: int,
    *,
    score_threshold: float = 0.4,
    max_proposals: int = 100,
    crop_size: int = 224,
    patch_grid: int = 16,
) -> np.ndarray:
    if max_proposals < 0:
        raise ValueError("max_proposals must be non-negative")
    ranked = sorted(proposals, key=lambda item: _proposal_value(item, "score"), reverse=True)
    selected = [item for item in ranked[:max_proposals] if _proposal_value(item, "score") >= score_threshold]
    boxes = [tuple(_proposal_value(item, key) for key in ("x1", "y1", "x2", "y2")) for item in selected]
    return boxes_to_patch_mask(boxes, image_width, image_height, crop_size=crop_size, patch_grid=patch_grid)
