#!/usr/bin/env python3
"""Feature contracts for learned camera updates with and without robot shape."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D", "camera_E")
COMMON_FEATURES = (
    "prior_x_m", "prior_y_m", "sin_prior_yaw", "cos_prior_yaw",
    "det_u0_norm", "det_v0_norm", "det_u1_norm", "det_v1_norm",
    "det_confidence", "raw_ground_x_m", "raw_ground_y_m",
    *(f"is_{camera}" for camera in CAMERAS),
)
SHAPE_FEATURES = (
    "hull_u0_norm", "hull_v0_norm", "hull_u1_norm", "hull_v1_norm",
    "det_minus_hull_bottom_u_norm", "det_minus_hull_bottom_v_norm",
    "det_minus_hull_width_norm", "det_minus_hull_height_norm",
    "h_ground_x_m", "h_ground_y_m",
    "dhx_dx", "dhx_dy", "dhy_dx", "dhy_dy",
)


def feature_names(use_shape: bool) -> tuple[str, ...]:
    return COMMON_FEATURES + (SHAPE_FEATURES if use_shape else ())


def feature_vector(row: Mapping[str, object], *, use_shape: bool) -> np.ndarray:
    width, height = float(row["image_width"]), float(row["image_height"])
    yaw = float(row["prior_yaw"])
    camera = str(row["camera"])
    if camera not in CAMERAS:
        raise ValueError(f"unknown camera {camera!r}")
    common = [
        float(row["prior_x"]), float(row["prior_y"]), math.sin(yaw), math.cos(yaw),
        float(row["box_x1"]) / width, float(row["box_y1"]) / height,
        float(row["box_x2"]) / width, float(row["box_y2"]) / height,
        float(row["box_confidence"]),
        float(row["raw_ground_x"]), float(row["raw_ground_y"]),
        *(1.0 if camera == name else 0.0 for name in CAMERAS),
    ]
    if not use_shape:
        return np.asarray(common, dtype=np.float32)
    det_bottom_u = 0.5 * (float(row["box_x1"]) + float(row["box_x2"]))
    det_bottom_v = float(row["box_y2"])
    hull_bottom_u = 0.5 * (float(row["hull_u0"]) + float(row["hull_u1"]))
    hull_bottom_v = float(row["hull_v1"])
    shape = [
        float(row["hull_u0"]) / width, float(row["hull_v0"]) / height,
        float(row["hull_u1"]) / width, float(row["hull_v1"]) / height,
        (det_bottom_u - hull_bottom_u) / width,
        (det_bottom_v - hull_bottom_v) / height,
        ((float(row["box_x2"]) - float(row["box_x1"]))
         - (float(row["hull_u1"]) - float(row["hull_u0"]))) / width,
        ((float(row["box_y2"]) - float(row["box_y1"]))
         - (float(row["hull_v1"]) - float(row["hull_v0"]))) / height,
        float(row["h_ground_x"]), float(row["h_ground_y"]),
        float(row["dhx_dx"]), float(row["dhx_dy"]),
        float(row["dhy_dx"]), float(row["dhy_dy"]),
    ]
    return np.asarray(common + shape, dtype=np.float32)


def make_model(input_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.SiLU(), nn.LayerNorm(128),
        nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, 64), nn.SiLU(),
        nn.Linear(64, 2),
    )


def load_artifact(path, *, device: str = "cpu"):
    import torch

    artifact = torch.load(path, map_location=device, weights_only=False)
    expected = list(feature_names(bool(artifact["use_shape"])))
    if artifact["feature_names"] != expected:
        raise RuntimeError("shape-update feature contract does not match this code")
    model = make_model(len(expected)).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact
