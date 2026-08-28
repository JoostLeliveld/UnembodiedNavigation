#!/usr/bin/env python3
"""Feature and model contract for paired box-shape/contour-shape update MLPs."""

from __future__ import annotations

import numpy as np

from shape_conditioned_update_model import feature_names as box_feature_names
from shape_conditioned_update_model import feature_vector as box_feature_vector


CONTOUR_POINTS = 32


def resample_closed_contour(points, count: int = CONTOUR_POINTS) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        raise ValueError("a contour needs at least three finite 2-D points")
    points = points[:, :2]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        raise ValueError("a contour needs at least three finite 2-D points")
    # Standardize winding, then anchor at the leftmost point among the lowest 1 px band.
    signed_twice_area = np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                               - np.roll(points[:, 0], -1) * points[:, 1])
    if signed_twice_area < 0:
        points = points[::-1]
    bottom = points[:, 1].max()
    candidates = np.flatnonzero(points[:, 1] >= bottom - 1.0)
    anchor = int(candidates[np.argmin(points[candidates, 0])])
    points = np.roll(points, -anchor, axis=0)
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-9:
        raise ValueError("degenerate contour")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = np.linspace(0.0, perimeter, count, endpoint=False)
    output = np.empty((count, 2), dtype=np.float32)
    for index, target in enumerate(targets):
        segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
        fraction = (target - cumulative[segment]) / max(lengths[segment], 1e-12)
        output[index] = closed[segment] + fraction * (closed[segment + 1] - closed[segment])
    return output


def feature_names(use_contour: bool) -> tuple[str, ...]:
    base = tuple(box_feature_names(True))
    if not use_contour:
        return base
    predicted = tuple(f"pred_contour_{axis}_{index:02d}_norm" for index in range(CONTOUR_POINTS)
                      for axis in ("u", "v"))
    expected = tuple(f"expected_contour_{axis}_{index:02d}_norm" for index in range(CONTOUR_POINTS)
                     for axis in ("u", "v"))
    delta = tuple(f"contour_delta_{axis}_{index:02d}_norm" for index in range(CONTOUR_POINTS)
                  for axis in ("u", "v"))
    return base + predicted + expected + delta


def feature_vector(row, *, use_contour: bool) -> np.ndarray:
    base = box_feature_vector(row, use_shape=True)
    if not use_contour:
        return base
    width, height = float(row["image_width"]), float(row["image_height"])
    predicted = np.asarray([[float(row[f"contour_u_{i:02d}"]), float(row[f"contour_v_{i:02d}"])]
                            for i in range(CONTOUR_POINTS)], dtype=np.float32)
    expected = np.asarray([[float(row[f"expected_u_{i:02d}"]), float(row[f"expected_v_{i:02d}"])]
                           for i in range(CONTOUR_POINTS)], dtype=np.float32)
    scale = np.asarray([width, height], dtype=np.float32)
    values = np.concatenate([(predicted / scale).reshape(-1),
                             (expected / scale).reshape(-1),
                             ((predicted - expected) / scale).reshape(-1)])
    return np.concatenate([base, values]).astype(np.float32)


def make_model(input_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.SiLU(), nn.LayerNorm(256),
        nn.Linear(256, 256), nn.SiLU(),
        nn.Linear(256, 128), nn.SiLU(),
        nn.Linear(128, 2),
    )


def load_artifact(path, *, device="cpu"):
    import torch
    artifact = torch.load(path, map_location=device, weights_only=False)
    use_contour = bool(artifact["use_contour"])
    expected = list(feature_names(use_contour))
    if artifact["feature_names"] != expected:
        raise RuntimeError("contour-update feature contract does not match this code")
    model = make_model(len(expected)).to(device)
    model.load_state_dict(artifact["state_dict"]); model.eval()
    return model, artifact
