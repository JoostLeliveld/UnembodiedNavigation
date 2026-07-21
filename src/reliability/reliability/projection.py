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
    along_bearing_offset_m: float = 0.0,
    along_bearing_slope_per_m: float = 0.0,
) -> tuple[float, float] | None:
    """Project a valid camera observation into the metric warehouse frame.

    The correction ``along_bearing_offset_m + along_bearing_slope_per_m * d``
    (``d`` = horizontal camera-to-point ground distance) shifts the projected
    point along the camera bearing, positive = away from the camera.  It
    corrects the near-edge/contact-height pull toward the camera: the
    detector's box-bottom pixel images the robot's nearest silhouette edge, so
    the raw ground-plane intersection lands between the camera and the robot
    centre, by an amount that grows with viewing distance.  Values are
    per-camera commissioning calibration constants (see
    ``fit_projection_calibration.py``), never fitted during deployment.
    """

    if not observation.detection_valid or observation.pixel_uv is None:
        return None
    u, v = observation.pixel_uv
    if contact_z_m > 0.0:
        point = camera.pixel_to_world_at_z(u, v, contact_z_m)
    else:
        point = camera.pixel_to_world(u, v)
    if point is None or (not along_bearing_offset_m and not along_bearing_slope_per_m):
        return point
    bearing_x = point[0] - float(camera.cam_pos[0])
    bearing_y = point[1] - float(camera.cam_pos[1])
    norm = math.hypot(bearing_x, bearing_y)
    if norm <= 1.0e-9:
        return point
    offset = along_bearing_offset_m + along_bearing_slope_per_m * norm
    scale = offset / norm
    return (point[0] + bearing_x * scale, point[1] + bearing_y * scale)


def load_projection_calibration(path: str | Path) -> dict[str, dict[str, float]]:
    """Read per-camera along-bearing calibrations from a JSON file.

    Returns ``{camera_id: {"intercept_m": a, "slope_per_m": b}}`` where the
    applied correction is ``a + b * ground_distance``.  Accepts legacy entries
    that are a bare float or carry only ``along_bearing_offset_m`` (treated as
    intercept-only).
    """

    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = payload.get("cameras", payload)
    calibrations: dict[str, dict[str, float]] = {}
    for camera_id, entry in cameras.items():
        if isinstance(entry, dict):
            intercept = float(entry.get("intercept_m", entry.get("along_bearing_offset_m", 0.0)))
            slope = float(entry.get("slope_per_m", 0.0))
        else:
            intercept = float(entry)
            slope = 0.0
        calibrations[str(camera_id)] = {"intercept_m": intercept, "slope_per_m": slope}
    return calibrations
