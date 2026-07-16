"""Ground-plane projection of camera observations from world-SDF poses.

Library home for the helpers the commissioning recorder introduced, so live
nodes and offline tools share one projection.  ``unav_common`` is imported
lazily to keep this package importable in ROS-independent tooling contexts.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from reliability.contracts import CameraObservation


def camera_model_from_world(world_sdf: str | Path, *, include_name: str):
    """Build a ground-plane camera model from an SDF ``<include>`` pose."""

    from unav_common.camera_model import ObliqueCameraModel

    root = ET.parse(Path(world_sdf)).getroot()
    pose_text = None
    for include in root.findall(".//include"):
        name = (include.findtext("name") or "").strip()
        uri = (include.findtext("uri") or "").strip()
        model_name = uri.removeprefix("model://").split("/", 1)[0]
        if include_name in {name, model_name}:
            pose_text = include.findtext("pose")
            break
    if not pose_text:
        raise RuntimeError(f"Could not find a posed camera include {include_name!r} in {world_sdf}")
    values = [float(value) for value in pose_text.split()]
    if len(values) != 6:
        raise RuntimeError(f"Camera pose for {include_name!r} must contain six values")
    x, y, z, _roll, pitch, yaw = values
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    if forward[2] >= -1.0e-6:
        raise RuntimeError(f"Camera {include_name!r} does not point down towards the ground")
    scale = -z / forward[2]
    look_at = (x + scale * forward[0], y + scale * forward[1], 0.0)
    return ObliqueCameraModel(
        cam_pos=(x, y, z),
        look_at=look_at,
        img_width=1280,
        img_height=720,
        fov_h_rad=1.5708,
    )


def project_observation_to_world(
    observation: CameraObservation,
    camera,
    *,
    contact_z_m: float,
) -> tuple[float, float] | None:
    """Project a valid camera observation into the metric warehouse frame."""

    if not observation.detection_valid or observation.pixel_uv is None:
        return None
    u, v = observation.pixel_uv
    if contact_z_m > 0.0:
        return camera.pixel_to_world_at_z(u, v, contact_z_m)
    return camera.pixel_to_world(u, v)
