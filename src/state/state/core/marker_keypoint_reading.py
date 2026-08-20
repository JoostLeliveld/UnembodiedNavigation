"""Turn a detected marker pair into a position, a heading, and an honest covariance.

The deployed reading takes the bottom edge of a detected box and back-projects it
onto the floor. That edge is not a fixed point on the robot — where it sits
depends on which way the robot faces — so the reading carries a repeated error
that no per-frame noise model can express.

Here the camera detects two marker disks whose positions in ``base_link`` are
known, so the observation model is the plain projection of known points:

    h(x, y, theta) = project(marker_front(x, y, theta)),
                     project(marker_rear(x, y, theta))

Inverting it is equally plain: each detected pixel is a ray, each ray meets the
plane the markers live on (``z = base_joint_z + marker_z``), and the two
recovered points give the robot's position *and* its heading — which the box
bottom never provided.

Covariance comes from the same geometry rather than from a fitted table: pixel
noise pushed through the Jacobian of that inversion. One consequence is worth
stating plainly, because it is why heading and position behave differently: the
heading's precision scales with how far apart the two markers appear, so it
degrades with range much faster than the position does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from unav_common.camera_model import ObliqueCameraModel

# Order the detector emits keypoints in. Must match
# perception.core.pose_keypoints and the training label layout.
KEYPOINT_FRONT = 0
KEYPOINT_REAR = 1


@dataclass(frozen=True)
class MarkerKeypointReading:
    """One sighting, expressed the way the filter wants it."""

    xy_m: tuple[float, float]
    heading_rad: float
    covariance_xy_m2: tuple[tuple[float, float], tuple[float, float]]
    heading_variance_rad2: float
    #: Full 3x3 over (x, y, heading), for a filter that updates heading too.
    covariance_xyh: tuple[tuple[float, float, float], ...]
    #: How far apart the markers appeared. Heading precision scales with this,
    #: so it is the honest health signal for the heading part of the reading.
    marker_separation_px: float


def marker_world_position(
    *, x: float, y: float, heading_rad: float, offset_x: float, plane_z: float,
) -> np.ndarray:
    """Where a marker at ``offset_x`` along the robot's axis sits in the world."""
    return np.asarray([
        float(x) + math.cos(float(heading_rad)) * float(offset_x),
        float(y) + math.sin(float(heading_rad)) * float(offset_x),
        float(plane_z),
    ], dtype=float)


def project_marker_pair(
    camera: ObliqueCameraModel,
    *,
    x: float,
    y: float,
    heading_rad: float,
    front_offset_x: float,
    rear_offset_x: float,
    plane_z: float,
) -> np.ndarray | None:
    """The observation model h: robot pose -> the four pixel numbers.

    Returns (u_front, v_front, u_rear, v_rear), or None if either marker is
    behind the camera.
    """
    out = []
    for offset in (front_offset_x, rear_offset_x):
        world = marker_world_position(
            x=x, y=y, heading_rad=heading_rad, offset_x=offset, plane_z=plane_z,
        )
        cam_pt = camera.R @ (world - camera.cam_pos)
        if cam_pt[2] <= 1e-6:
            return None
        pixel = camera.K @ cam_pt
        out.extend([pixel[0] / pixel[2], pixel[1] / pixel[2]])
    return np.asarray(out, dtype=float)


def _invert(
    camera: ObliqueCameraModel,
    pixels: np.ndarray,
    *,
    front_offset_x: float,
    rear_offset_x: float,
    plane_z: float,
) -> tuple[np.ndarray, float] | None:
    """Two detected pixels -> the robot's position and heading."""
    front = camera.pixel_to_world_at_z(pixels[0], pixels[1], plane_z)
    rear = camera.pixel_to_world_at_z(pixels[2], pixels[3], plane_z)
    if front is None or rear is None:
        return None
    front_xy = np.asarray(front, dtype=float)
    rear_xy = np.asarray(rear, dtype=float)
    heading = math.atan2(front_xy[1] - rear_xy[1], front_xy[0] - rear_xy[0])
    # The markers straddle base_link rather than centring on it, so the midpoint
    # has to be walked back along the heading by their mean offset.
    midpoint = 0.5 * (front_xy + rear_xy)
    mid_offset = 0.5 * (float(front_offset_x) + float(rear_offset_x))
    base = midpoint - mid_offset * np.asarray([math.cos(heading), math.sin(heading)])
    return base, heading


def read_marker_keypoints(
    camera: ObliqueCameraModel,
    front_uv: tuple[float, float] | np.ndarray,
    rear_uv: tuple[float, float] | np.ndarray,
    *,
    front_offset_x: float,
    rear_offset_x: float,
    plane_z: float,
    pixel_sigma: float,
    pixel_step: float = 0.5,
) -> MarkerKeypointReading | None:
    """Convert a detected marker pair into a pose reading with a covariance.

    ``pixel_sigma`` is the detector's per-coordinate spread in pixels, measured
    against projected ground truth. It is the ONLY fitted number here;
    everything else is geometry.
    """
    pixels = np.asarray([
        float(front_uv[0]), float(front_uv[1]), float(rear_uv[0]), float(rear_uv[1]),
    ], dtype=float)
    inverted = _invert(
        camera, pixels,
        front_offset_x=front_offset_x, rear_offset_x=rear_offset_x, plane_z=plane_z,
    )
    if inverted is None:
        return None
    base, heading = inverted

    # Jacobian of (x, y, heading) with respect to the four pixel numbers, by
    # central differences. The inversion is smooth, and a numerical Jacobian
    # cannot drift out of step with the inversion the way a hand-derived one can.
    jac = np.zeros((3, 4), dtype=float)
    step = float(pixel_step)
    for col in range(4):
        bumped_up = pixels.copy()
        bumped_down = pixels.copy()
        bumped_up[col] += step
        bumped_down[col] -= step
        up = _invert(camera, bumped_up, front_offset_x=front_offset_x,
                     rear_offset_x=rear_offset_x, plane_z=plane_z)
        down = _invert(camera, bumped_down, front_offset_x=front_offset_x,
                       rear_offset_x=rear_offset_x, plane_z=plane_z)
        if up is None or down is None:
            return None
        jac[0, col] = (up[0][0] - down[0][0]) / (2.0 * step)
        jac[1, col] = (up[0][1] - down[0][1]) / (2.0 * step)
        dheading = (up[1] - down[1] + math.pi) % (2.0 * math.pi) - math.pi
        jac[2, col] = dheading / (2.0 * step)

    cov_pixels = (float(pixel_sigma) ** 2) * np.eye(4)
    cov = jac @ cov_pixels @ jac.T
    cov = 0.5 * (cov + cov.T)  # kill asymmetry from floating point

    separation = float(np.linalg.norm(pixels[:2] - pixels[2:]))
    return MarkerKeypointReading(
        xy_m=(float(base[0]), float(base[1])),
        heading_rad=float(heading),
        covariance_xy_m2=(
            (float(cov[0, 0]), float(cov[0, 1])),
            (float(cov[1, 0]), float(cov[1, 1])),
        ),
        heading_variance_rad2=float(cov[2, 2]),
        covariance_xyh=tuple(
            (float(row[0]), float(row[1]), float(row[2])) for row in cov
        ),
        marker_separation_px=separation,
    )
