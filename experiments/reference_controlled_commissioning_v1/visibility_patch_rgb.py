"""Runtime-observable visibility-patch features for RGB residual correction.

The detector's saved ``crop`` is a 96x96 rendering of the detected box with a
half-box margin on every side.  This module reconstructs the box location inside
that saved context crop and pools soft target-blue evidence into a 16x16 grid.
No reference pose, simulator segmentation, belief, innovation, or NIS enters the
feature.  The grid is therefore available both during commissioning and at
runtime from the current RGB frame and detected box.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


GRID_SIZE = 16
CONTEXT_SIZE = 96
BLUE_DOMINANCE_CENTER = 0.035
BLUE_DOMINANCE_SCALE = 0.035
CHROMA_CENTER = 0.075
CHROMA_SCALE = 0.040
BLUE_VALUE_CENTER = 0.12
BLUE_VALUE_SCALE = 0.055


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def soft_target_blue(rgb: np.ndarray) -> np.ndarray:
    """Return a soft blue-target score for an RGB image.

    The score deliberately uses colour relations rather than a hard hue range.
    Illumination can reduce saturation, so blue dominance, chroma, and blue
    intensity are combined smoothly.  Warehouse blue distractors are retained:
    the detected-box spatial pattern, not colour alone, must disambiguate them.
    """

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("RGB input must have shape [height,width,3]")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("floating RGB input must be finite and in [0,1]")
    red, green, blue = (image[..., index] for index in range(3))
    dominance = blue - np.maximum(red, green)
    chroma = np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue))
    score = (
        _sigmoid((dominance - BLUE_DOMINANCE_CENTER) / BLUE_DOMINANCE_SCALE)
        * _sigmoid((chroma - CHROMA_CENTER) / CHROMA_SCALE)
        * _sigmoid((blue - BLUE_VALUE_CENTER) / BLUE_VALUE_SCALE)
    )
    return np.asarray(score, dtype=np.float32)


def context_bounds(
    bbox_xyxy: Sequence[float], image_shape: Sequence[int]
) -> tuple[int, int, int, int]:
    """Reproduce the detector's integer half-box-margin crop bounds."""

    bbox = np.asarray(bbox_xyxy, dtype=float)
    shape = np.asarray(image_shape, dtype=int)
    if bbox.shape != (4,) or shape.shape != (2,) or not np.isfinite(bbox).all():
        raise ValueError("bbox and image shape must be finite [4] and [2] arrays")
    x0, y0, x1, y1 = bbox
    height, width = map(int, shape)
    if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("bbox and image dimensions must be positive")
    box_width, box_height = x1 - x0, y1 - y0
    left = int(max(0.0, x0 - 0.5 * box_width))
    top = int(max(0.0, y0 - 0.5 * box_height))
    right = int(min(float(width), x1 + 0.5 * box_width))
    bottom = int(min(float(height), y1 + 0.5 * box_height))
    if right <= left or bottom <= top:
        raise ValueError("context crop is empty")
    return left, top, right, bottom


def visibility_grid_from_saved_context(
    context_chw: np.ndarray,
    bbox_xyxy: Sequence[float],
    image_shape: Sequence[int],
    *,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Pool target-blue evidence inside the detected box into ``grid_size`` cells."""

    crop = np.asarray(context_chw)
    if crop.shape != (3, CONTEXT_SIZE, CONTEXT_SIZE) or crop.dtype != np.uint8:
        raise ValueError("saved context crop must be uint8 with shape [3,96,96]")
    if not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid size must be a positive integer")
    left, top, right, bottom = context_bounds(bbox_xyxy, image_shape)
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    scale_x = CONTEXT_SIZE / float(right - left)
    scale_y = CONTEXT_SIZE / float(bottom - top)
    bx0 = int(np.clip(math.floor((x0 - left) * scale_x), 0, CONTEXT_SIZE - 1))
    by0 = int(np.clip(math.floor((y0 - top) * scale_y), 0, CONTEXT_SIZE - 1))
    bx1 = int(np.clip(math.ceil((x1 - left) * scale_x), bx0 + 1, CONTEXT_SIZE))
    by1 = int(np.clip(math.ceil((y1 - top) * scale_y), by0 + 1, CONTEXT_SIZE))
    rgb = crop.transpose(1, 2, 0)
    score = soft_target_blue(rgb)[by0:by1, bx0:bx1]
    tensor = torch.from_numpy(score)[None, None]
    grid = F.adaptive_avg_pool2d(tensor, (grid_size, grid_size))[0].numpy()
    return np.asarray(grid, dtype=np.float32)


def visibility_summaries(grid: np.ndarray) -> dict[str, float]:
    """Return interpretable diagnostics without changing the model input."""

    value = np.asarray(grid, dtype=float)
    if value.shape != (1, GRID_SIZE, GRID_SIZE) or not np.isfinite(value).all():
        raise ValueError("visibility grid must be finite with shape [1,16,16]")
    mass = float(value.mean())
    lower = float(value[:, 3 * GRID_SIZE // 4 :, :].mean())
    rows = value[0].mean(axis=1)
    cols = value[0].mean(axis=0)
    normalizer = max(float(rows.sum()), 1e-9)
    row_centroid = float(np.dot(np.arange(GRID_SIZE), rows) / normalizer / (GRID_SIZE - 1))
    normalizer = max(float(cols.sum()), 1e-9)
    col_centroid = float(np.dot(np.arange(GRID_SIZE), cols) / normalizer / (GRID_SIZE - 1))
    return {
        "visible_blue_fraction": mass,
        "lower_quarter_blue_fraction": lower,
        "blue_row_centroid": row_centroid,
        "blue_column_centroid": col_centroid,
    }


class VisibilityPatchResidualNet(nn.Module):
    """Add a gated visibility-grid residual to a frozen tabular correction.

    The last residual layer is initialized to zero.  Before training, and after
    loading an explicitly zero residual, the output is exactly the supplied box
    correction.  This makes degradation an observed training outcome rather than
    an architectural consequence of replacing the baseline.
    """

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature count must be positive")
        self.feature_count = int(feature_count)
        self.visibility = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.features = nn.Sequential(
            nn.Linear(self.feature_count, 32), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(24 * 4 * 4 + 32, 96), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(96, 48), nn.ReLU(),
        )
        self.residual = nn.Linear(48, 2)
        self.gate = nn.Linear(48, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        visibility_grid: torch.Tensor,
        standardized_features: torch.Tensor,
        base_correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visibility_grid.ndim != 4 or visibility_grid.shape[1:] != (1, GRID_SIZE, GRID_SIZE):
            raise ValueError("visibility input must have shape [batch,1,16,16]")
        if standardized_features.ndim != 2 or standardized_features.shape[1] != self.feature_count:
            raise ValueError("feature input has the wrong shape")
        if base_correction.ndim != 2 or base_correction.shape[1] != 2:
            raise ValueError("base correction must have shape [batch,2]")
        encoded = torch.cat(
            (self.visibility(visibility_grid), self.features(standardized_features)), dim=1
        )
        hidden = self.trunk(encoded)
        residual = self.residual(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        return base_correction + gate * residual, residual, gate
