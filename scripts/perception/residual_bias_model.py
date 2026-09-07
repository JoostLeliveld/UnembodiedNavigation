#!/usr/bin/env python3
"""Shared feature and model contract for heading-conditioned pixel corrections."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D", "camera_E")
FEATURE_NAMES = (
    "box_bottom_u_norm",
    "box_bottom_v_norm",
    "box_width_norm",
    "box_height_norm",
    "box_confidence",
    "log1p_baseline_range_m",
    "sin_heading_relative_to_camera_bearing",
    "cos_heading_relative_to_camera_bearing",
    *(f"is_{camera}" for camera in CAMERAS),
)


def feature_vector(
    row: Mapping[str, object], *, heading_rad: float | None = None,
    disable_heading: bool = False,
) -> np.ndarray:
    """Commissioning adapter; commanded yaw is allowed only in this offline API.

    Runtime callers must use ``online_feature_vector`` with their operational heading.
    """
    if not disable_heading and heading_rad is None:
        heading_rad = float(row['robot_yaw'])
    return online_feature_vector(row, heading_rad=heading_rad, disable_heading=disable_heading)


def online_feature_vector(
    row: Mapping[str, object], *, heading_rad: float | None = None,
    disable_heading: bool = False,
) -> np.ndarray:
    """Never reads commanded pose; enabled heading must be supplied operationally."""
    width = float(row["image_width"])
    height = float(row["image_height"])
    if not math.isfinite(width) or not math.isfinite(height) or min(width, height) <= 0:
        raise ValueError('image dimensions must be finite and positive')
    if disable_heading:
        sin_relative, cos_relative = 0.0, 0.0
    else:
        if heading_rad is None or not math.isfinite(heading_rad):
            raise ValueError('a finite operational heading is required')
        relative = float(heading_rad) - float(row['baseline_bearing_rad'])
        sin_relative, cos_relative = math.sin(relative), math.cos(relative)
    camera = str(row["camera"])
    if camera not in CAMERAS:
        raise ValueError(f"unknown camera {camera!r}")
    values = [
        float(row["box_bottom_u"]) / width,
        float(row["box_bottom_v"]) / height,
        (float(row["box_x2"]) - float(row["box_x1"])) / width,
        (float(row["box_y2"]) - float(row["box_y1"])) / height,
        float(row["box_confidence"]),
        math.log1p(float(row["baseline_range_m"])),
        sin_relative,
        cos_relative,
        *(1.0 if camera == name else 0.0 for name in CAMERAS),
    ]
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or values[2] <= 0 or values[3] <= 0 or not 0 <= values[4] <= 1:
        raise ValueError('nonfinite or invalid detector features')
    return values


def make_model(input_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.SiLU(),
        nn.LayerNorm(64),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 32),
        nn.SiLU(),
        nn.Linear(32, 2),
    )


def load_artifact(path, *, device: str = "cpu"):
    import torch

    artifact = torch.load(path, map_location=device, weights_only=False)
    if artifact.get('schema') != 'provisional_pixel_residual.v1':
        raise RuntimeError('unsupported residual-model artifact version')
    if artifact.get('frame') != 'original_image_pixels' or artifact.get('target') != 'projected_commanded_ground_reference_minus_semantic_mask_bottom_centre':
        raise RuntimeError('incompatible residual-model physical target')
    if artifact["feature_names"] != list(FEATURE_NAMES):
        raise RuntimeError("residual-model feature contract does not match this code")
    for key, length in [('x_mean',len(FEATURE_NAMES)),('x_std',len(FEATURE_NAMES)),('y_mean',2),('y_std',2)]:
        value = np.asarray(artifact.get(key), dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all() or (key.endswith('_std') and np.any(value <= 0)):
            raise RuntimeError(f'invalid residual-model normalization: {key}')
    model = make_model(len(FEATURE_NAMES)).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact
