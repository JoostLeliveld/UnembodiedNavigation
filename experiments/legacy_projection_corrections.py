"""The projection corrections that were DELETED from the runtime on 2026-08-07.

This module is a graveyard, not a library.  Nothing in `src/` may import it, and no new
study may use it to produce a result.

## Why it exists

`reliability.projection` used to carry a contact-plane constant, a per-camera along-bearing
correction and a gated cross-bearing correction.  e7 measured all of them against 1844 real
detections and every one scored **worse than applying no correction at all**:

    raw IPM, 0 fitted scalars    66.6 mm   <- what the runtime does now
    v4,      2 fitted scalars    70.1 mm
    v3,     10 fitted scalars    74.5 mm
    v2,      8 fitted scalars    68.2 mm

(`logs/studies/pixel_ground_path/e7_ipm_zero_parameter/RESULTS.md`.)  So they were removed
rather than carried.  The runtime is now one ray-plane intersection with no parameters.

But roughly eight completed studies have these corrections as their **subject** --
`external_camera_bias_model`, `projection_amplification`, `operational_residual_rcond`,
`calibration_drift_lifecycle`, `multicamera_fusion_extension`, the commissioning fit and
`pixel_ground_path` itself.  Deleting the code out from under them would make the evidence
that justified the deletion un-re-runnable, which is the exact failure e3's audit had to
repair once already (its scripts pointed at a dataset path a cold-archive move had removed).

So the deleted implementation lives here, in one place, byte-faithful to what `src/` held at
git revision prior to the 2026-08-07 cleanup.  Historical studies import it and keep
reproducing their published numbers; the runtime never sees it.

## The identity that made these corrections redundant

Intersecting the ray at height `z` instead of the floor shortens every estimate by
`z*d/(H-z)` -- exactly the form of the fitted `slope_per_m` term.  So a fitted along-bearing
correction and a raised contact plane are the same physical quantity seen twice, and fitting
one while an operator sets the other by hand lets the fit spend parameters undoing a free
constant.  v2 did precisely that: radial bias +8.2 mm with both, -94.1 mm with the plane
alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_projection_calibration(path: str | Path) -> dict[str, dict[str, float]]:
    """DELETED FROM RUNTIME. Read per-camera bearing-frame calibrations from JSON.

    Every key is optional and defaults to 0.0.  Legacy entries that are a bare float, or
    that carry only ``along_bearing_offset_m``, are accepted as intercept-only, and
    ``cross_bearing_offset_m`` is an alias for ``cross_intercept_m``.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = payload.get("cameras", payload)
    calibrations: dict[str, dict[str, float]] = {}
    for camera_id, entry in cameras.items():
        if isinstance(entry, dict):
            intercept = float(entry.get("intercept_m", entry.get("along_bearing_offset_m", 0.0)))
            slope = float(entry.get("slope_per_m", 0.0))
            cross_intercept = float(
                entry.get("cross_intercept_m", entry.get("cross_bearing_offset_m", 0.0))
            )
            cross_slope = float(
                entry.get("cross_slope_per_m", entry.get("cross_bearing_slope_per_m", 0.0))
            )
        else:
            intercept = float(entry)
            slope = 0.0
            cross_intercept = 0.0
            cross_slope = 0.0
        calibrations[str(camera_id)] = {
            "intercept_m": intercept,
            "slope_per_m": slope,
            "cross_intercept_m": cross_intercept,
            "cross_slope_per_m": cross_slope,
        }
    return calibrations


def load_projection_contact_z(path: str | Path, *, default: float = 0.05) -> float:
    """DELETED FROM RUNTIME. Read the contact plane an artifact was fitted against.

    ``default`` is the historical node default, for artifacts predating the field
    (``gt_validation_smoke_20260716``), whose intercepts were fitted against a 0.05 m plane.
    v2/v3 carry 0.05 explicitly; v4 carries 0.0.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return float(default)
    value = payload.get("contact_z_m")
    return float(default) if value is None else float(value)


def projection_kwargs_for_camera(
    calibrations: dict[str, dict[str, float]],
    camera_id: str,
    *,
    contact_z_m: float,
) -> dict[str, float]:
    """DELETED FROM RUNTIME. Map a loaded calibration onto the projection signature."""

    entry = calibrations.get(camera_id, {})
    return {
        "contact_z_m": float(contact_z_m),
        "along_bearing_offset_m": float(entry.get("intercept_m", 0.0)),
        "along_bearing_slope_per_m": float(entry.get("slope_per_m", 0.0)),
        "cross_bearing_offset_m": float(entry.get("cross_intercept_m", 0.0)),
        "cross_bearing_slope_per_m": float(entry.get("cross_slope_per_m", 0.0)),
    }


def project_pixel_to_world(
    u: float,
    v: float,
    camera,
    *,
    contact_z_m: float,
    along_bearing_offset_m: float = 0.0,
    along_bearing_slope_per_m: float = 0.0,
    cross_bearing_offset_m: float = 0.0,
    cross_bearing_slope_per_m: float = 0.0,
) -> tuple[float, float] | None:
    """DELETED FROM RUNTIME. The complete mean-calibration projection.

    The bearing basis is built from the RAW projected point, so the two corrections are
    orthogonal translations of that point and neither depends on the other's magnitude.
    """

    if contact_z_m > 0.0:
        point = camera.pixel_to_world_at_z(u, v, contact_z_m)
    else:
        point = camera.pixel_to_world(u, v)
    if point is None or not (
        along_bearing_offset_m
        or along_bearing_slope_per_m
        or cross_bearing_offset_m
        or cross_bearing_slope_per_m
    ):
        return point
    bearing_x = point[0] - float(camera.cam_pos[0])
    bearing_y = point[1] - float(camera.cam_pos[1])
    norm = math.hypot(bearing_x, bearing_y)
    if norm <= 1.0e-9:
        return point
    unit_along = (bearing_x / norm, bearing_y / norm)
    # Left of the bearing, matching the sign convention the calibration is fitted in.
    unit_cross = (-unit_along[1], unit_along[0])
    along = along_bearing_offset_m + along_bearing_slope_per_m * norm
    cross = cross_bearing_offset_m + cross_bearing_slope_per_m * norm
    return (
        point[0] + along * unit_along[0] + cross * unit_cross[0],
        point[1] + along * unit_along[1] + cross * unit_cross[1],
    )


#: Historical alias -- these studies imported the private runtime name.
_project_pixel_to_world = project_pixel_to_world


def project_observation_to_world(observation, camera, **kwargs):
    """DELETED FROM RUNTIME (correction-carrying form). Kept for historical studies."""

    if not observation.detection_valid or observation.pixel_uv is None:
        return None
    return project_pixel_to_world(
        observation.pixel_uv[0], observation.pixel_uv[1], camera, **kwargs
    )
