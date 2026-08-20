"""Sightline inference: would this camera see a robot standing there?

For a candidate robot position the question is whether anything lies between
the camera centre and the robot body. With a depth image in hand that is one
comparison per body point, at image resolution:

    project the body point -> read the depth the camera measured in that
    direction -> the sightline is clear exactly when the measured depth
    reaches at least as far as the body point.

This *is* the ray cast, evaluated exactly rather than by marching. Marching a
rasterised height map answers the same question and adds the grid's error on
top: a 2.5-D raster snaps an obstacle to whole cells, so a 0.8 m box becomes a
1.05 m box on a 0.25 m grid and its shadow grows accordingly. The height map is
still produced -- it is the map a planner reads and the artefact that makes a
stale-geometry failure visible -- but it is not what decides a sightline.

Two consequences worth stating, because both were tempting to get wrong:

**Uncertainty enters as a probability, not a threshold.** The comparison is
``P(measured depth >= body-point depth)`` under the depth sigma from the ground
fit, so a sightline the camera cannot resolve comes out near 0.5 instead of
being forced to a confident yes. It also means the two depths being compared
lie on the *same* pixel ray, so their errors are correlated and the difference
is far better determined than either absolute depth.

**Ground the camera cannot see is occluded, not unknown.** The region behind a
box is hidden *by the box*, which is a known reason, and a robot there would be
invisible for that reason. Unknown is reserved for the cases where the method
genuinely has no evidence: a pixel the depth model marked invalid, a body point
outside the image, and a frame whose ground fit was refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .contracts import CameraCalibration, FloorPlane, RaycastConfig, TargetVolume


@dataclass(frozen=True)
class SightlineField:
    """``p_visible + p_occluded + p_unknown == 1`` per cell.

    ``p_occluded`` means "not visible for a reason we can name": structure in
    the way, or the position falling outside the image. ``in_fov`` separates
    those two, so a consumer never has to guess which one it is looking at.
    """

    p_visible: np.ndarray
    p_occluded: np.ndarray
    p_unknown: np.ndarray
    in_fov: np.ndarray


def _project(calib: CameraCalibration, pts: np.ndarray):
    """World points ``(..., 3)`` -> ``(u, v, optical-axis depth, in_front)``."""
    cam_pt = (pts - calib.cam_pos) @ calib.R.T
    depth = cam_pt[..., 2]
    in_front = depth > 1e-9
    safe = np.where(in_front, depth, 1.0)
    u = calib.K[0, 0] * cam_pt[..., 0] / safe + calib.K[0, 2]
    v = calib.K[1, 1] * cam_pt[..., 1] / safe + calib.K[1, 2]
    return u, v, depth, in_front


def _plane_z(plane: FloorPlane, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    n = plane.normal
    return (plane.offset - n[0] * gx - n[1] * gy) / (n[2] if abs(n[2]) > 1e-9 else 1.0)


def _in_fov(
    calib: CameraCalibration, xs: np.ndarray, ys: np.ndarray, z_probe: float
) -> np.ndarray:
    """Cell centres that project inside the image, in front of the camera."""
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx, gy, np.full_like(gx, float(z_probe))], axis=-1)
    u, v, _, in_front = _project(calib, pts)
    return in_front & (u >= 0) & (u < calib.width) & (v >= 0) & (v < calib.height)


def line_of_sight_field(
    calib: CameraCalibration,
    depth_m: np.ndarray,
    sigma_m: np.ndarray,
    valid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    plane: FloorPlane | None = None,
    target: TargetVolume | None = None,
    config: RaycastConfig | None = None,
) -> SightlineField:
    """Per-cell visible / occluded / unknown for a robot body at that cell.

    ``depth_m`` and ``sigma_m`` are the *corrected metric* depth image and its
    1-sigma; ``valid`` marks the pixels the method is willing to read.
    """
    cfg = config or RaycastConfig()
    tgt = target or TargetVolume()
    plane = plane or FloorPlane()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    depth_m = np.asarray(depth_m, dtype=float)
    sigma_m = np.maximum(np.asarray(sigma_m, dtype=float), cfg.min_depth_sigma_m)
    valid = np.asarray(valid, dtype=bool)

    gx, gy = np.meshgrid(xs, ys)
    base_z = _plane_z(plane, gx, gy)
    offsets = tgt.sample_offsets()

    p_vis = np.zeros(gx.shape, dtype=float)
    p_unk = np.zeros(gx.shape, dtype=float)
    for ox, oy, oz in offsets:
        pts = np.stack([gx + ox, gy + oy, base_z + oz], axis=-1)
        u, v, depth_t, in_front = _project(calib, pts)
        in_img = in_front & (u >= 0) & (u < calib.width) & (v >= 0) & (v < calib.height)

        ui = np.clip(np.rint(u), 0, calib.width - 1).astype(int)
        vi = np.clip(np.rint(v), 0, calib.height - 1).astype(int)
        measured = depth_m[vi, ui]
        readable = valid[vi, ui] & np.isfinite(measured)

        p_clear = ndtr((measured - depth_t) / sigma_m[vi, ui])
        p_vis += np.where(in_img & readable, p_clear, 0.0)
        p_unk += np.where(in_img & ~readable, 1.0, 0.0)

    n_off = float(offsets.shape[0])
    p_visible = p_vis / n_off
    p_unknown = p_unk / n_off
    p_occluded = np.clip(1.0 - p_visible - p_unknown, 0.0, 1.0)
    z_probe = 0.5 * (tgt.z_min_m + tgt.z_max_m) + plane.offset
    return SightlineField(
        p_visible=p_visible,
        p_occluded=p_occluded,
        p_unknown=p_unknown,
        in_fov=_in_fov(calib, xs, ys, z_probe),
    )
