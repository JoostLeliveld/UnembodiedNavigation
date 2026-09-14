from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT / "experiments/reference_controlled_commissioning_v1/visibility_patch_rgb.py"
)
SPEC = importlib.util.spec_from_file_location("visibility_patch_rgb", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_soft_visibility_distinguishes_blue_from_neutral() -> None:
    blue = np.full((20, 20, 3), (15, 75, 195), dtype=np.uint8)
    neutral = np.full((20, 20, 3), (120, 120, 120), dtype=np.uint8)
    assert float(MODULE.soft_target_blue(blue).mean()) > 0.90
    assert float(MODULE.soft_target_blue(neutral).mean()) < 0.05


def test_grid_preserves_partial_visibility_pattern() -> None:
    # The detector crop is exactly twice the bbox size, so the bbox occupies the
    # central half of this 96x96 context image.
    context = np.full((96, 96, 3), 120, dtype=np.uint8)
    context[24:72, 24:48] = (15, 75, 195)
    grid = MODULE.visibility_grid_from_saved_context(
        context.transpose(2, 0, 1), (24.0, 24.0, 72.0, 72.0), (96, 96)
    )
    assert grid.shape == (1, 16, 16)
    assert float(grid[:, :, :8].mean()) > 0.85
    assert float(grid[:, :, 8:].mean()) < 0.10


def test_clamped_context_reconstructs_box_region() -> None:
    context = np.full((96, 96, 3), 120, dtype=np.uint8)
    # Source bbox (5,20)-(25,60) yields a context crop clamped on the left.
    left, top, right, bottom = MODULE.context_bounds((5, 20, 25, 60), (80, 100))
    assert (left, top, right, bottom) == (0, 0, 35, 80)
    scale_x, scale_y = 96 / 35, 96 / 80
    x0, x1 = int(np.floor(5 * scale_x)), int(np.ceil(25 * scale_x))
    y0, y1 = int(np.floor(20 * scale_y)), int(np.ceil(60 * scale_y))
    context[y0:y1, x0:x1] = (15, 75, 195)
    grid = MODULE.visibility_grid_from_saved_context(
        context.transpose(2, 0, 1), (5, 20, 25, 60), (80, 100)
    )
    assert float(grid.mean()) > 0.80


def test_zero_initialized_residual_preserves_box_baseline() -> None:
    model = MODULE.VisibilityPatchResidualNet(feature_count=5).eval()
    grid = torch.rand(3, 1, 16, 16)
    features = torch.rand(3, 5)
    base = torch.tensor([[0.1, -0.2], [0.0, 0.3], [-0.4, 0.2]])
    with torch.no_grad():
        corrected, residual, gate = model(grid, features, base)
    assert corrected.numpy() == pytest.approx(base.numpy())
    assert residual.numpy() == pytest.approx(np.zeros((3, 2)))
    assert torch.all((gate > 0.0) & (gate < 1.0))


def test_invalid_shapes_fail_closed() -> None:
    model = MODULE.VisibilityPatchResidualNet(feature_count=3)
    with pytest.raises(ValueError, match="visibility input"):
        model(torch.zeros(1, 1, 8, 8), torch.zeros(1, 3), torch.zeros(1, 2))
