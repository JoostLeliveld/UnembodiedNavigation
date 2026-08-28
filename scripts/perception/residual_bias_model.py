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
    """Build the online feature vector from one frozen detector observation."""
    width = float(row["image_width"])
    height = float(row["image_height"])
    yaw = float(row["robot_yaw"] if heading_rad is None else heading_rad)
    relative = yaw - float(row["baseline_bearing_rad"])
    if disable_heading:
        sin_relative, cos_relative = 0.0, 0.0
    else:
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
    return np.asarray(values, dtype=np.float32)


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
    if artifact["feature_names"] != list(FEATURE_NAMES):
        raise RuntimeError("residual-model feature contract does not match this code")
    model = make_model(len(FEATURE_NAMES)).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact
