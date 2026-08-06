"""Candidate box-to-ground projection used only by the pixel-ground study.

This is deliberately experiment-local.  Its constants are specific to the simulated
TurtleBot3 appearance, the studied detector and the 6.10 m / 0.92 rad camera mounts.  The
yaw-marginal covariance is temporally correlated and is not licensed as independent
per-frame measurement noise in a sequential filter.
"""

from __future__ import annotations

import math


Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]

# The statistic, projection plane and uncertainty terms are one candidate measurement
# definition.  Provenance is recorded in this experiment's README and e3-e6 outputs.
BOX_STATISTIC_ALPHA = 0.5
BOX_STATISTIC_PLANE_Z_M = 0.085
BOX_STATISTIC_SIGMA_UV_PX = (1.15, 0.77)
BOX_STATISTIC_SIGMA_YAW_M = (0.0303, 0.0222)
BOX_STATISTIC_REFERENCE_MOUNT = (6.10, 0.92)


def box_statistic_pixel(
    box_xyxy: tuple[float, float, float, float],
    *,
    alpha: float = BOX_STATISTIC_ALPHA,
) -> tuple[float, float]:
    """Return horizontal centre and ``alpha`` up from the box bottom."""

    try:
        u0, v0, u1, v1 = (float(value) for value in box_xyxy)
    except (TypeError, ValueError) as exc:
        raise ValueError("box_xyxy must contain exactly four numeric values") from exc
    if not all(math.isfinite(value) for value in (u0, v0, u1, v1)):
        raise ValueError("box_xyxy values must be finite")
    if u0 < 0.0 or v0 < 0.0:
        raise ValueError("box_xyxy must use non-negative image coordinates")
    if u1 <= u0 or v1 <= v0:
        raise ValueError("box_xyxy must have positive width and height")
    try:
        statistic_alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and in [0, 1]") from exc
    if not math.isfinite(statistic_alpha) or not 0.0 <= statistic_alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    return 0.5 * (u0 + u1), v1 + statistic_alpha * (v0 - v1)


def project_box_to_world(
    box_xyxy: tuple[float, float, float, float],
    camera,
    *,
    alpha: float = BOX_STATISTIC_ALPHA,
    plane_z_m: float = BOX_STATISTIC_PLANE_Z_M,
) -> tuple[float, float] | None:
    """Project the candidate statistic onto its coupled horizontal plane."""

    plane = _validated_projection_plane(camera, plane_z_m)
    u, v = box_statistic_pixel(box_xyxy, alpha=alpha)
    return _finite_projected_point(camera.pixel_to_world_at_z(u, v, plane))


def project_box_to_world_with_covariance(
    box_xyxy: tuple[float, float, float, float],
    camera,
    *,
    alpha: float = BOX_STATISTIC_ALPHA,
    plane_z_m: float = BOX_STATISTIC_PLANE_Z_M,
    sigma_uv_px: tuple[float, float] = BOX_STATISTIC_SIGMA_UV_PX,
    sigma_yaw_m: tuple[float, float] | None = BOX_STATISTIC_SIGMA_YAW_M,
    jacobian_step_px: float = 0.5,
) -> tuple[tuple[float, float], Matrix2x2] | None:
    """Return the point and per-detection map covariance for offline evaluation.

    ``R = J Sigma_uv J^T + Sigma_yaw``.  ``Sigma_yaw`` is rotated from the
    camera-to-estimate bearing frame into map axes.  It represents a yaw marginal across
    poses, not white noise across successive frames.
    """

    step = float(jacobian_step_px)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("jacobian_step_px must be finite and positive")
    try:
        su, sv = (float(value) for value in sigma_uv_px)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma_uv_px must contain exactly two numeric values") from exc
    if not all(math.isfinite(value) and value > 0.0 for value in (su, sv)):
        raise ValueError("sigma_uv_px values must be finite and positive")

    yaw_sigmas = None
    if sigma_yaw_m is not None:
        try:
            radial, lateral = (float(value) for value in sigma_yaw_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("sigma_yaw_m must contain exactly two numeric values") from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in (radial, lateral)):
            raise ValueError("sigma_yaw_m values must be finite and non-negative")
        yaw_sigmas = (radial, lateral)

    plane = _validated_projection_plane(camera, plane_z_m)
    point = project_box_to_world(box_xyxy, camera, alpha=alpha, plane_z_m=plane)
    if point is None:
        return None

    u, v = box_statistic_pixel(box_xyxy, alpha=alpha)
    jacobian = []
    for axis in (0, 1):
        du = step if axis == 0 else 0.0
        dv = step if axis == 1 else 0.0
        plus = _finite_projected_point(camera.pixel_to_world_at_z(u + du, v + dv, plane))
        minus = _finite_projected_point(camera.pixel_to_world_at_z(u - du, v - dv, plane))
        if plus is None or minus is None:
            return None
        jacobian.append(
            ((plus[0] - minus[0]) / (2.0 * step), (plus[1] - minus[1]) / (2.0 * step))
        )
    j00, j10 = jacobian[0]
    j01, j11 = jacobian[1]
    if not all(math.isfinite(value) for value in (j00, j01, j10, j11)):
        return None
    if abs(j00 * j11 - j01 * j10) <= 1.0e-12:
        return None

    xx = j00 * j00 * su * su + j01 * j01 * sv * sv
    xy = j00 * j10 * su * su + j01 * j11 * sv * sv
    yy = j10 * j10 * su * su + j11 * j11 * sv * sv

    if yaw_sigmas is not None:
        radial, lateral = yaw_sigmas
        try:
            camera_x = float(camera.cam_pos[0])
            camera_y = float(camera.cam_pos[1])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if not math.isfinite(camera_x) or not math.isfinite(camera_y):
            return None
        bearing_x = point[0] - camera_x
        bearing_y = point[1] - camera_y
        norm = math.hypot(bearing_x, bearing_y)
        if not math.isfinite(norm) or norm <= 1.0e-9:
            return None
        ux, uy = bearing_x / norm, bearing_y / norm
        rr, ll = radial * radial, lateral * lateral
        xx += ux * ux * rr + uy * uy * ll
        xy += ux * uy * (rr - ll)
        yy += uy * uy * rr + ux * ux * ll

    covariance = ((xx, xy), (xy, yy))
    if not all(math.isfinite(value) for row in covariance for value in row):
        return None
    if xx <= 0.0 or yy <= 0.0 or xx * yy - xy * xy <= 0.0:
        return None
    return point, covariance


def box_statistic_mount_deviation(
    camera,
    *,
    reference_mount: tuple[float, float] = BOX_STATISTIC_REFERENCE_MOUNT,
) -> tuple[float, float]:
    """Return signed camera height and downward-pitch deviations from the study mount."""

    try:
        cam_x, cam_y, cam_z = (float(value) for value in camera.cam_pos[:3])
        look_x, look_y, look_z = (float(value) for value in camera.look_at[:3])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("camera must expose finite cam_pos and look_at triples") from exc
    forward = (look_x - cam_x, look_y - cam_y, look_z - cam_z)
    norm = math.sqrt(sum(value * value for value in forward))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("camera look_at must differ from cam_pos")
    sin_pitch = min(1.0, max(-1.0, -forward[2] / norm))
    reference_height, reference_pitch = (float(value) for value in reference_mount)
    return cam_z - reference_height, math.asin(sin_pitch) - reference_pitch


def _validated_projection_plane(camera, plane_z_m: float) -> float:
    try:
        plane = float(plane_z_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("plane_z_m must be finite and non-negative") from exc
    if not math.isfinite(plane) or plane < 0.0:
        raise ValueError("plane_z_m must be finite and non-negative")
    try:
        camera_z = float(camera.cam_pos[2])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("camera.cam_pos must contain a finite camera height") from exc
    if not math.isfinite(camera_z) or camera_z <= 0.0:
        raise ValueError("camera height must be finite and above the floor")
    if plane >= camera_z:
        raise ValueError("plane_z_m must be below the camera height")
    return plane


def _finite_projected_point(point) -> tuple[float, float] | None:
    if point is None:
        return None
    try:
        x, y = (float(value) for value in point)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y
