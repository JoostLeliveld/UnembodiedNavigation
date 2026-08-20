"""Corrected depth -> world points -> a 2.5-D obstacle-height representation.

One view gives one 2.5-D surface, so the representation has to carry three
things per cell, not one:

``h_max``     the tallest surface the camera actually saw in that cell;
``h_sigma``   how well that height is known, propagated from the depth sigma;
``observed``  whether the camera returned anything there at all.

``observed`` is what keeps the map honest. A cell with no returns is *not* a
cell that is known to be empty, and a map that cannot say which is which
invites the "false bright corridor" failure -- a rack that moved, or was never
scanned, read as clear floor. Anything consuming this map must treat
``observed == False`` as absence of evidence.

``observed`` is decided two ways, because back-projection alone conflates
"nothing there" with "we sub-sampled past it". Pixels are finite, so distant
cells can fall between back-projected samples and would look unobserved for a
bookkeeping reason rather than a physical one. The second, resolution-
independent test runs forwards: project the cell's own floor point into the
image and compare the measured depth against the analytic floor depth. If the
camera sees that far, nothing stands in that column -- for a downward-looking
camera anything in the column would have occluded the ground point first -- so
the cell is known and empty. If the measured depth is short, something is in
front and the ground there is hidden.

This map is an output in its own right, not the sightline oracle: visibility is
decided in :mod:`ground_anchoring.raycast` against the depth image directly, at
image resolution, without this grid's rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import CameraCalibration, FloorPlane


@dataclass(frozen=True)
class HeightMap:
    xs: np.ndarray
    ys: np.ndarray
    h_max: np.ndarray
    h_sigma: np.ndarray
    observed: np.ndarray
    n_points: int


def back_project(
    calib: CameraCalibration,
    depth_m: np.ndarray,
    valid: np.ndarray,
    *,
    step: int = 3,
    max_depth_m: float = 60.0,
    min_depth_m: float = 0.2,
    sigma_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sampled back-projection to world points.

    Returns ``(points, sigma_world_z)`` where ``points`` is ``(N, 3)``. The
    world-height sigma is ``|w_z| * sigma_depth``, with ``w_z`` the vertical
    component of the pixel's world ray -- a near-horizontal ray converts depth
    error into almost no height error, a steep one converts nearly all of it.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    step = max(1, int(step))
    h_img, w_img = depth_m.shape
    vv, uu = np.mgrid[0:h_img:step, 0:w_img:step]
    u = uu.ravel().astype(float)
    v = vv.ravel().astype(float)
    d = depth_m[vv.ravel(), uu.ravel()]
    ok = (
        valid[vv.ravel(), uu.ravel()]
        & np.isfinite(d)
        & (d > min_depth_m)
        & (d < max_depth_m)
    )
    u, v, d = u[ok], v[ok], d[ok]
    if u.size == 0:
        return np.zeros((0, 3)), np.zeros(0)
    dirs = calib.rays_world(u, v)  # (3, N) per unit optical-axis depth
    points = (calib.cam_pos[:, None] + dirs * d[None, :]).T
    if sigma_m is None:
        sig_world = np.zeros(u.size)
    else:
        s = np.asarray(sigma_m, dtype=float)[vv.ravel(), uu.ravel()][ok]
        sig_world = np.abs(dirs[2]) * s
    return points, sig_world


def ground_visibility_mask(
    calib: CameraCalibration,
    depth_m: np.ndarray,
    valid: np.ndarray,
    sigma_m: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    plane: FloorPlane,
    *,
    sigma_k: float = 3.0,
    abs_tol_m: float = 0.05,
) -> np.ndarray:
    """Cells whose floor point the camera demonstrably saw.

    A depth-buffer test, one lookup per cell: project the cell's floor point,
    and accept it when the measured depth reaches at least as far as the
    analytic floor depth, within ``sigma_k`` sigmas plus a small absolute
    allowance. Short of that, something is standing in the way.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    gx, gy = np.meshgrid(xs, ys)
    n = plane.normal
    gz = (plane.offset - n[0] * gx - n[1] * gy) / (n[2] if abs(n[2]) > 1e-9 else 1.0)
    pts = np.stack([gx, gy, gz], axis=-1)

    cam_pt = (pts - calib.cam_pos) @ calib.R.T
    expected = cam_pt[..., 2]
    in_front = expected > 1e-9
    safe = np.where(in_front, expected, 1.0)
    u = calib.K[0, 0] * cam_pt[..., 0] / safe + calib.K[0, 2]
    v = calib.K[1, 1] * cam_pt[..., 1] / safe + calib.K[1, 2]
    inside = in_front & (u >= 0) & (u < calib.width) & (v >= 0) & (v < calib.height)

    ui = np.clip(np.rint(u), 0, calib.width - 1).astype(int)
    vi = np.clip(np.rint(v), 0, calib.height - 1).astype(int)
    measured = np.asarray(depth_m, dtype=float)[vi, ui]
    tol = sigma_k * np.asarray(sigma_m, dtype=float)[vi, ui] + abs_tol_m
    reaches_floor = np.isfinite(measured) & (measured >= expected - tol)
    return inside & np.asarray(valid, dtype=bool)[vi, ui] & reaches_floor


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Nearest grid index for each value on a uniformly spaced axis."""
    axis = np.asarray(axis, dtype=float)
    n = axis.size
    if n == 1:
        return np.zeros(np.shape(values), dtype=int)
    d = float(axis[1] - axis[0])
    idx = np.rint((np.asarray(values, dtype=float) - axis[0]) / d)
    return np.clip(idx, 0, n - 1).astype(int)


def rasterize_heights(
    points: np.ndarray,
    sigma_world_z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    plane: FloorPlane | None = None,
    *,
    min_sigma_m: float = 0.0,
) -> HeightMap:
    """Tallest observed surface per cell, with its sigma and an observed mask.

    Heights are measured above the floor plane and clipped at zero: a slightly
    negative floor return is measurement noise, not a hole in the ground.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ny, nx = ys.size, xs.size
    h_max = np.zeros((ny, nx), dtype=float)
    h_sigma = np.zeros((ny, nx), dtype=float)
    observed = np.zeros((ny, nx), dtype=bool)
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return HeightMap(xs, ys, h_max, np.full((ny, nx), min_sigma_m), observed, 0)

    heights = (
        pts[:, 2] if plane is None else plane.height_above(pts)
    )
    heights = np.maximum(heights, 0.0)
    ix = _nearest_index(xs, pts[:, 0])
    iy = _nearest_index(ys, pts[:, 1])

    # Sort ascending by height so the last write into each cell is the tallest
    # point; that also carries *that point's* sigma, not an unrelated one.
    order = np.argsort(heights, kind="stable")
    h_max[iy[order], ix[order]] = heights[order]
    h_sigma[iy[order], ix[order]] = np.asarray(sigma_world_z, dtype=float)[order]
    observed[iy, ix] = True

    h_sigma = np.maximum(h_sigma, min_sigma_m)
    return HeightMap(xs, ys, h_max, h_sigma, observed, int(pts.shape[0]))
