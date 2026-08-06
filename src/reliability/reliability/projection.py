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

Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]


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
    cross_bearing_offset_m: float = 0.0,
    cross_bearing_slope_per_m: float = 0.0,
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

    ``cross_bearing_*`` adds the perpendicular degree of freedom, positive to the
    LEFT of the bearing.  The along-bearing form above cannot represent a lateral
    offset at all, and the audit found one: camera C retained **+0.078 m** of
    uncorrected lateral bias, which is an extrinsic/rotation signature rather
    than the contact-point signature the along term targets.  See
    ``logs/studies/external_camera_bias_model/exp2_two_dof_bias/RESULTS.md`` --
    including the reason the cross term is **gated per camera** and left at zero
    where the lateral bias is not resolvable against that camera's own scatter
    (fitting it there made camera A 61 % worse on held-out data).

    Both defaults are 0.0, so an along-only calibration reproduces the previous
    behaviour exactly (asserted in ``tests/reliability/test_projection_cross_bearing.py``).
    """

    if not observation.detection_valid or observation.pixel_uv is None:
        return None
    return _project_pixel_to_world(
        observation.pixel_uv[0],
        observation.pixel_uv[1],
        camera,
        contact_z_m=contact_z_m,
        along_bearing_offset_m=along_bearing_offset_m,
        along_bearing_slope_per_m=along_bearing_slope_per_m,
        cross_bearing_offset_m=cross_bearing_offset_m,
        cross_bearing_slope_per_m=cross_bearing_slope_per_m,
    )


def project_observation_to_world_with_covariance(
    observation: CameraObservation,
    camera,
    *,
    contact_z_m: float,
    along_bearing_offset_m: float = 0.0,
    along_bearing_slope_per_m: float = 0.0,
    cross_bearing_offset_m: float = 0.0,
    cross_bearing_slope_per_m: float = 0.0,
    jacobian_step_px: float = 0.5,
    min_eigenvalue_m2: float = 1.0e-12,
) -> tuple[tuple[float, float], Matrix2x2] | None:
    """Project an observation and its full pixel covariance into map coordinates.

    This is the strict paper-1 precision-covariance reproduction path:
    ``CameraObservation.conditional_cov_uv`` is already the historical
    trust-to-R precision blend in px².  The complete corrected projection is
    differentiated numerically and the covariance is propagated as
    ``R_xy = J R_uv Jᵀ``.  The tiny eigenvalue floor is numerical only; it is
    many orders of magnitude below any physical measurement covariance.
    """

    if not observation.detection_valid or observation.pixel_uv is None:
        return None
    step = float(jacobian_step_px)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("jacobian_step_px must be finite and positive")
    floor = float(min_eigenvalue_m2)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("min_eigenvalue_m2 must be finite and positive")

    u, v = observation.pixel_uv
    kwargs = {
        "contact_z_m": contact_z_m,
        "along_bearing_offset_m": along_bearing_offset_m,
        "along_bearing_slope_per_m": along_bearing_slope_per_m,
        "cross_bearing_offset_m": cross_bearing_offset_m,
        "cross_bearing_slope_per_m": cross_bearing_slope_per_m,
    }
    centre = _project_pixel_to_world(u, v, camera, **kwargs)
    if centre is None:
        return None
    du = _projection_derivative(
        camera, u, v, axis=0, step=step, centre=centre, kwargs=kwargs
    )
    dv = _projection_derivative(
        camera, u, v, axis=1, step=step, centre=centre, kwargs=kwargs
    )
    if du is None or dv is None:
        return None

    # J rows are map x/y and columns are pixel u/v.
    j00, j10 = du
    j01, j11 = dv
    r_uv = observation.conditional_cov_uv
    a = float(r_uv[0][0])
    b = 0.5 * (float(r_uv[0][1]) + float(r_uv[1][0]))
    d = float(r_uv[1][1])
    xx = j00 * j00 * a + 2.0 * j00 * j01 * b + j01 * j01 * d
    xy = j00 * j10 * a + (j00 * j11 + j01 * j10) * b + j01 * j11 * d
    yy = j10 * j10 * a + 2.0 * j10 * j11 * b + j11 * j11 * d
    covariance = _floor_spd_2x2(((xx, xy), (xy, yy)), floor)
    return centre, covariance


def _project_pixel_to_world(
    u: float,
    v: float,
    camera,
    *,
    contact_z_m: float,
    along_bearing_offset_m: float,
    along_bearing_slope_per_m: float,
    cross_bearing_offset_m: float = 0.0,
    cross_bearing_slope_per_m: float = 0.0,
) -> tuple[float, float] | None:
    """Project one pixel through the complete mean-calibration function.

    The bearing basis is built from the RAW projected point, so the two
    corrections are orthogonal translations of that point and neither depends on
    the other's magnitude.
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


def _projection_derivative(
    camera,
    u: float,
    v: float,
    *,
    axis: int,
    step: float,
    centre: tuple[float, float],
    kwargs: dict[str, float],
) -> tuple[float, float] | None:
    """Central projection derivative with a one-sided image-edge fallback."""

    plus = _project_pixel_to_world(
        u + (step if axis == 0 else 0.0),
        v + (step if axis == 1 else 0.0),
        camera,
        **kwargs,
    )
    minus = _project_pixel_to_world(
        u - (step if axis == 0 else 0.0),
        v - (step if axis == 1 else 0.0),
        camera,
        **kwargs,
    )
    if plus is not None and minus is not None:
        return (
            (float(plus[0]) - float(minus[0])) / (2.0 * step),
            (float(plus[1]) - float(minus[1])) / (2.0 * step),
        )
    if plus is not None:
        return (
            (float(plus[0]) - float(centre[0])) / step,
            (float(plus[1]) - float(centre[1])) / step,
        )
    if minus is not None:
        return (
            (float(centre[0]) - float(minus[0])) / step,
            (float(centre[1]) - float(minus[1])) / step,
        )
    return None


def _floor_spd_2x2(matrix: Matrix2x2, floor: float) -> Matrix2x2:
    """Add only enough isotropic jitter to give the matrix a numerical SPD floor."""

    a = float(matrix[0][0])
    b = 0.5 * (float(matrix[0][1]) + float(matrix[1][0]))
    d = float(matrix[1][1])
    minimum = 0.5 * (a + d) - math.hypot(0.5 * (a - d), b)
    jitter = max(0.0, float(floor) - minimum)
    return ((a + jitter, b), (b, d + jitter))


def load_projection_calibration(path: str | Path) -> dict[str, dict[str, float]]:
    """Read per-camera bearing-frame calibrations from a JSON file.

    Returns ``{camera_id: {"intercept_m", "slope_per_m", "cross_intercept_m",
    "cross_slope_per_m"}}``.  The along-bearing correction is
    ``intercept_m + slope_per_m * d`` and the cross-bearing one is
    ``cross_intercept_m + cross_slope_per_m * d``, both in metres, ``d`` the
    ground distance.

    Every key is optional and defaults to 0.0, so **along-only calibration files
    (including the deployed v2) load to an identical correction to before** —
    the cross keys simply come back zero.  Legacy entries that are a bare float,
    or that carry only ``along_bearing_offset_m``, are still accepted as
    intercept-only.  ``cross_bearing_offset_m`` is accepted as an alias for
    ``cross_intercept_m``.
    """

    import json

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
    """Read the contact plane height a calibration artifact was fitted against.

    The plane the ray is intersected with and the along-bearing correction are the
    same physical quantity seen twice: intersecting at ``z`` instead of the floor
    shortens every estimate by ``z·d/(H−z)``, which is exactly the form of the
    ``slope_per_m`` term. Fitting one while the other is set independently lets a
    per-camera correction absorb a constant the operator chose, so the two must
    travel together — selecting a calibration selects its contact plane.

    ``default`` preserves the historical node default for artifacts predating this
    field, so v2/v3 load to a bit-identical projection.
    """

    import json

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
    """Complete projection keyword arguments for one camera.

    THE single place that maps a loaded calibration onto the projection
    signature.  Call this instead of picking ``intercept_m`` / ``slope_per_m`` out
    of the dict by hand: three call sites used to do that independently, so
    adding the cross-bearing degree of freedom would silently have left some of
    them at one DOF.  A camera absent from ``calibrations`` yields an all-zero
    (raw) correction, matching the previous per-site ``.get(..., 0.0)`` behaviour.
    """

    entry = calibrations.get(camera_id, {})
    return {
        "contact_z_m": float(contact_z_m),
        "along_bearing_offset_m": float(entry.get("intercept_m", 0.0)),
        "along_bearing_slope_per_m": float(entry.get("slope_per_m", 0.0)),
        "cross_bearing_offset_m": float(entry.get("cross_intercept_m", 0.0)),
        "cross_bearing_slope_per_m": float(entry.get("cross_slope_per_m", 0.0)),
    }
