"""Small deterministic checks for the physical RGB-D sanity-check geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("real_rgbd_evaluate", HERE / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def test_plane_round_trip_is_exact() -> None:
    K = np.array([[120.0, 0.0, 60.0], [0.0, 125.0, 45.0], [0.0, 0.0, 1.0]])
    normal = np.array([0.03, -0.995, -0.095], dtype=float)
    normal /= np.linalg.norm(normal)
    offset = 0.35
    depth = EVALUATE.plane_depth_map(K, (100, 120), normal, offset)
    mask = np.isfinite(depth)
    points = EVALUATE.backproject_z_depth(K, depth, mask)[::4]
    fit = EVALUATE.fit_floor_plane(points)
    expected = EVALUATE.plane_depth_map(K, depth.shape, normal, offset)
    actual = EVALUATE.plane_depth_map(K, depth.shape, fit["normal"], fit["offset"])
    common = np.isfinite(expected) & np.isfinite(actual)
    assert fit["passes"]
    assert np.max(np.abs(expected[common] - actual[common])) < 1e-10


def test_depth_metrics_use_only_the_requested_valid_pixels() -> None:
    truth = np.array([[1.0, 2.0], [3.0, 0.0]])
    prediction = np.array([[1.0, 3.0], [5.0, 7.0]])
    mask = np.array([[True, True], [False, True]])
    metrics = EVALUATE.depth_metrics(prediction, truth, mask)
    assert metrics["n"] == 2
    assert metrics["mae_m"] == 0.5
    assert metrics["bias_m"] == 0.5


def test_oracle_affine_recovers_known_scale_and_shift() -> None:
    prediction = np.arange(1.0, 10.0).reshape(3, 3)
    truth = 0.6 * prediction + 0.2
    mask = np.ones_like(prediction, dtype=bool)
    corrected, parameters = EVALUATE.oracle_affine(prediction, truth, mask)
    assert np.allclose(parameters, [0.6, 0.2])
    assert np.allclose(corrected, truth)

