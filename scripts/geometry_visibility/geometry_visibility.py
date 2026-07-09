#!/usr/bin/env python3
"""Pure, ROS-free functions for the geometry-derived observability prior.

The chain is:

    warehouse prisms + camera model
        -> 2.5D height map
        -> per-cell FOV / projection
        -> raycast occlusion clearance
        -> projection-Jacobian range/obliquity term
        -> visibility score in [0, 1]
        -> planner measurement covariance R_plan (pixel^2)

Everything here is deterministic and depends only on numpy plus the existing
``ObliqueCameraModel``. No ROS, Gazebo, YOLO, or matplotlib imports live in this
module so the toy tests stay fast and hermetic.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ``ObliqueCameraModel`` only imports math + numpy, so importing it here keeps
# this module ROS-free. The build script is responsible for putting unav_common
# on sys.path; tests inject a path fixture.
try:  # pragma: no cover - import shim
    from unav_common.camera_model import ObliqueCameraModel
except Exception:  # pragma: no cover - resolved by add_unav_common_to_path()
    ObliqueCameraModel = None  # type: ignore


# ---------------------------------------------------------------------------
# Geometry containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Prism:
    """Axis-aligned box occluder/driveable footprint (world frame, metres)."""

    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def covers(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)


def prisms_from_json(payload) -> list[Prism]:
    """Parse a ``{"prisms": [...]}`` payload into ``Prism`` objects."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    prisms = []
    for p in payload["prisms"]:
        prisms.append(
            Prism(
                name=str(p.get("name", "prism")),
                xmin=float(p["xmin"]),
                xmax=float(p["xmax"]),
                ymin=float(p["ymin"]),
                ymax=float(p["ymax"]),
                zmin=float(p.get("zmin", 0.0)),
                zmax=float(p["zmax"]),
            )
        )
    return prisms


def _decode_packed_json(raw) -> str:
    """Return the JSON string from a possibly array-wrapped npz field.

    The GP artifact stores ``geometry_json`` as ``np.array([json_string])`` so a
    naive ``str(...)`` yields ``"['{...}']"``. Unwrap the single element.
    """
    arr = np.asarray(raw)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def load_gp_artifact_geometry(npz_path: str | Path) -> dict:
    """Load prisms, grid, camera params, and empirical GP maps from the artifact."""
    data = np.load(str(npz_path), allow_pickle=True)

    def scalar(key):
        v = data[key]
        return v.item() if getattr(v, "shape", None) == (1,) else v

    geometry_json = _decode_packed_json(data["geometry_json"])
    prisms = prisms_from_json(geometry_json)
    out = {
        "xs": np.asarray(data["xs"], dtype=float),
        "ys": np.asarray(data["ys"], dtype=float),
        "prisms": prisms,
        "geometry_json": geometry_json,
        "geometry_sha256": hashlib.sha256(geometry_json.encode("utf-8")).hexdigest(),
        "camera_pos": np.asarray(data["camera_pos"], dtype=float),
        "look_at": np.asarray(data["look_at"], dtype=float),
        "img_width": int(scalar("img_width")),
        "img_height": int(scalar("img_height")),
        "fov_h_rad": float(scalar("fov_h_rad")),
        "target_height": float(scalar("target_height")),
    }
    # Empirical GP fields (optional; used for Stage 1/2 comparison only).
    for key in ("F_mean_map", "F_std_map", "P_mean_map", "P_conservative_plan_map", "X_train"):
        if key in data.files:
            out[key] = np.asarray(data[key], dtype=float)
    return out


# ---------------------------------------------------------------------------
# Height map (2.5D)
# ---------------------------------------------------------------------------
def heights_at(x: np.ndarray, y: np.ndarray, prisms: Iterable[Prism]) -> np.ndarray:
    """Max prism ``zmax`` over prisms covering each ``(x, y)`` sample; 0 elsewhere."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.zeros(np.broadcast(x, y).shape, dtype=float)
    for p in prisms:
        mask = p.covers(x, y)
        if np.any(mask):
            h = np.maximum(h, np.where(mask, p.zmax, 0.0))
    return h


def build_height_map(xs: np.ndarray, ys: np.ndarray, prisms: Iterable[Prism]) -> dict:
    """Rasterise prisms onto the grid. Returns ``h_max`` with shape ``(len(ys), len(xs))``."""
    gx, gy = np.meshgrid(xs, ys)  # (ny, nx)
    h_max = heights_at(gx, gy, prisms)
    known = np.ones_like(h_max, dtype=bool)  # v1 uses fully-known geometry
    occ_conf = (h_max > 0.0).astype(float)
    return {
        "h_max": h_max,
        "known": known,
        "occ_conf": occ_conf,
        "resolution": float(np.mean(np.diff(xs))),
        "origin_xy": np.array([xs[0], ys[0]], dtype=float),
    }


def height_map_from_points(points: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> dict:
    """Rasterise a sensed 3-D point cloud into a 2.5-D height map.

    This is the *warehouse-implementable* Level-3 entry point: feed it points from
    infrastructure stereo, a robot LiDAR/RGB-D mapping pass, or any depth sensor —
    it does not need CAD. ``h_max`` is the tallest return per cell; ``observed`` is
    True only where the sensor actually returned a point (cells with no returns are
    unknown and must be handled conservatively downstream).
    """
    pts = np.asarray(points, dtype=float)
    ny, nx = len(ys), len(xs)
    h_max = np.zeros((ny, nx), dtype=float)
    observed = np.zeros((ny, nx), dtype=bool)
    if pts.size == 0:
        return {"h_max": h_max, "observed": observed}
    ix = np.clip(np.searchsorted(xs, pts[:, 0]), 0, nx - 1)
    iy = np.clip(np.searchsorted(ys, pts[:, 1]), 0, ny - 1)
    np.maximum.at(h_max, (iy, ix), np.maximum(pts[:, 2], 0.0))
    observed[iy, ix] = True
    return {"h_max": h_max, "observed": observed}


def points_visible_from_camera(cam, points: np.ndarray, prisms: Iterable[Prism],
                               n_samples: int = 24, near_target_skip: float = 0.12) -> np.ndarray:
    """Boolean per point: is this 3-D surface point in the camera image AND unoccluded?

    Used to synthesise what an infrastructure stereo / depth sensor at the camera
    would actually return (occluded surfaces yield no points -> unknown regions).
    """
    pts = np.asarray(points, dtype=float)
    cp = np.asarray(cam.cam_pos, dtype=float)
    t = np.linspace(0.02, 1.0 - near_target_skip, int(n_samples)).reshape(1, -1, 1)
    s = cp.reshape(1, 1, 3) + t * (pts[:, None, :] - cp.reshape(1, 1, 3))
    obst = heights_at(s[..., 0], s[..., 1], prisms)
    blocked = np.any(s[..., 2] < obst - 1e-6, axis=1)
    u, v, in_front = project_points(cam, pts)
    in_img = (u >= 0) & (u < cam.img_width) & (v >= 0) & (v < cam.img_height) & in_front
    return in_img & ~blocked


def world_to_grid(xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> tuple[int, int]:
    """Nearest ``(iy, ix)`` cell index for a world point (row-major, matches maps)."""
    ix = int(np.argmin(np.abs(np.asarray(xs) - x)))
    iy = int(np.argmin(np.abs(np.asarray(ys) - y)))
    return iy, ix


def in_any_prism(xs: np.ndarray, ys: np.ndarray, prisms: Iterable[Prism]) -> np.ndarray:
    """Boolean grid mask: cell centre lies inside at least one prism footprint."""
    gx, gy = np.meshgrid(xs, ys)
    mask = np.zeros(gx.shape, dtype=bool)
    for p in prisms:
        mask |= p.covers(gx, gy)
    return mask


# ---------------------------------------------------------------------------
# Camera: FOV + projection (vectorised over the grid)
# ---------------------------------------------------------------------------
def make_camera(meta: dict):
    if ObliqueCameraModel is None:  # pragma: no cover
        raise RuntimeError("ObliqueCameraModel unavailable; call add_unav_common_to_path first")
    return ObliqueCameraModel(
        cam_pos=tuple(meta["camera_pos"]),
        look_at=tuple(meta["look_at"]),
        img_width=meta["img_width"],
        img_height=meta["img_height"],
        fov_h_rad=meta["fov_h_rad"],
    )


def project_points(cam, pts: np.ndarray):
    """Vectorised world->pixel. ``pts`` shape (..., 3). Returns u, v, in_front."""
    pts = np.asarray(pts, dtype=float)
    cam_pt = (pts - cam.cam_pos) @ cam.R.T  # (..., 3)
    z_c = cam_pt[..., 2]
    in_front = z_c > 1e-9
    z_safe = np.where(in_front, z_c, 1.0)
    f = cam.K[0, 0]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    u = f * cam_pt[..., 0] / z_safe + cx
    v = f * cam_pt[..., 1] / z_safe + cy
    return u, v, in_front


def fov_projection_grid(cam, xs: np.ndarray, ys: np.ndarray, z_marker: float) -> dict:
    """Per-cell pixel coords, in-front, and in-image FOV mask."""
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx, gy, np.full_like(gx, float(z_marker))], axis=-1)
    u, v, in_front = project_points(cam, pts)
    in_image = (u >= 0) & (u < cam.img_width) & (v >= 0) & (v < cam.img_height)
    fov_mask = in_front & in_image
    return {"u": u, "v": v, "in_front": in_front, "fov_mask": fov_mask}


# ---------------------------------------------------------------------------
# Raycast occlusion clearance
# ---------------------------------------------------------------------------
def raycast_min_clearance(
    cam,
    xs: np.ndarray,
    ys: np.ndarray,
    prisms: Iterable[Prism],
    z_marker: float,
    n_samples: int = 48,
) -> np.ndarray:
    """Minimum (ray height - obstacle height) between camera and each ground target.

    Negative -> the sightline passes through an obstacle (occluded). Positive ->
    clear line of sight. Sampled strictly between the endpoints to avoid the
    camera body and the target cell itself.
    """
    gx, gy = np.meshgrid(xs, ys)
    targets = np.stack(
        [gx, gy, np.full_like(gx, float(z_marker))], axis=-1
    ).reshape(-1, 3)  # (N, 3)
    cam_pos = np.asarray(cam.cam_pos, dtype=float)
    t = np.linspace(0.02, 0.98, int(n_samples)).reshape(1, -1, 1)  # (1, S, 1)
    # samples: (N, S, 3)
    samples = cam_pos.reshape(1, 1, 3) + t * (targets[:, None, :] - cam_pos.reshape(1, 1, 3))
    sx, sy, sz = samples[..., 0], samples[..., 1], samples[..., 2]
    obst_h = heights_at(sx, sy, prisms)  # (N, S)
    clearance = sz - obst_h  # (N, S)
    min_clear = clearance.min(axis=1)  # (N,)
    return min_clear.reshape(gx.shape)


# ---------------------------------------------------------------------------
# Range / obliquity via the projection Jacobian
# ---------------------------------------------------------------------------
def projection_jacobian_scale(cam, xs: np.ndarray, ys: np.ndarray, z_marker: float) -> dict:
    """Local pixels-per-metre resolution of the image<-ground map.

    d(u,v)/d(x,y) is estimated by central differences. Its smaller singular value
    is the worst-direction position resolution (px per m): high near the camera /
    head-on, low far away / grazing. This captures range AND obliquity, which is
    the dominant reliability driver when occluders are short and the camera is high.
    """
    gx, gy = np.meshgrid(xs, ys)
    z = np.full_like(gx, float(z_marker))
    h = 0.02  # metres, central-difference step

    def uv(dx, dy):
        pts = np.stack([gx + dx, gy + dy, z], axis=-1)
        u, v, _ = project_points(cam, pts)
        return u, v

    ux_p, vx_p = uv(h, 0.0)
    ux_m, vx_m = uv(-h, 0.0)
    uy_p, vy_p = uv(0.0, h)
    uy_m, vy_m = uv(0.0, -h)
    du_dx = (ux_p - ux_m) / (2 * h)
    dv_dx = (vx_p - vx_m) / (2 * h)
    du_dy = (uy_p - uy_m) / (2 * h)
    dv_dy = (vy_p - vy_m) / (2 * h)

    # Singular values of the 2x2 [[du_dx, du_dy],[dv_dx, dv_dy]] per cell.
    a, b, c, d = du_dx, du_dy, dv_dx, dv_dy
    tr = a * a + b * b + c * c + d * d
    det = a * d - b * c
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det * det, 0.0))
    s_max = np.sqrt(np.maximum((tr + disc) / 2.0, 0.0))
    s_min = np.sqrt(np.maximum((tr - disc) / 2.0, 0.0))
    return {"px_per_m_min": s_min, "px_per_m_max": s_max, "det_abs": np.abs(det)}


# ---------------------------------------------------------------------------
# Visibility components -> score
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def boundary_score(u: np.ndarray, v: np.ndarray, img_w: int, img_h: int, margin_frac: float = 0.06) -> np.ndarray:
    """1 in the image interior, smoothly -> 0 at the image edges."""
    mx = margin_frac * img_w
    my = margin_frac * img_h
    du = np.minimum(u, img_w - u)
    dv = np.minimum(v, img_h - v)
    sx = np.clip(du / mx, 0.0, 1.0)
    sy = np.clip(dv / my, 0.0, 1.0)
    return np.clip(np.minimum(sx, sy), 0.0, 1.0)


def compute_visibility(
    fov_mask: np.ndarray,
    min_clearance: np.ndarray,
    px_per_m_min: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    img_w: int,
    img_h: int,
    tau_clearance: float = 0.10,
    range_ref_pct: float = 90.0,
    use_range: bool = True,
    use_boundary: bool = True,
) -> dict:
    """Combine independent [0,1] components into a visibility score.

    f_fov   : hard in/out of image
    f_occ   : sigmoid of raycast clearance (occlusion)
    f_range : range/obliquity resolution, normalised to a robust in-FOV reference
    f_boundary : image-edge falloff
    """
    f_fov = fov_mask.astype(float)
    f_occ = _sigmoid(min_clearance / float(tau_clearance))

    if use_range:
        ref = np.percentile(px_per_m_min[fov_mask], range_ref_pct) if np.any(fov_mask) else 1.0
        ref = max(ref, 1e-9)
        f_range = np.clip(px_per_m_min / ref, 0.0, 1.0)
    else:
        f_range = np.ones_like(f_occ)

    f_boundary = boundary_score(u, v, img_w, img_h) if use_boundary else np.ones_like(f_occ)

    score = f_fov * f_occ * f_range * f_boundary
    score = np.clip(score, 0.0, 1.0)
    return {
        "f_fov": f_fov,
        "f_occ": f_occ,
        "f_range": f_range,
        "f_boundary": f_boundary,
        "visibility_score": score,
    }


def compute_label_noise(
    min_clearance: np.ndarray,
    f_boundary: np.ndarray,
    sigma_min2: float = 0.01,
    a_boundary: float = 0.25,
    tau_clearance: float = 0.10,
) -> np.ndarray:
    """Pseudo-label noise: high near the occlusion boundary and image edges."""
    occ_boundary = np.exp(-(min_clearance / float(tau_clearance)) ** 2)  # peaks at clearance 0
    edge_uncertainty = 1.0 - f_boundary
    noise = sigma_min2 + a_boundary * np.maximum(occ_boundary, edge_uncertainty)
    return noise


# ---------------------------------------------------------------------------
# Trust -> planner covariance R_plan (pixel^2)
# ---------------------------------------------------------------------------
def trust_to_r_plan(trust, r_visible_uv: float, r_miss_uv: float):
    """Precision-blend the visible/miss pixel covariances.

    1/var = trust / r_visible^2 + (1-trust) / r_miss^2.
    trust=1 -> std=r_visible_uv, trust=0 -> std=r_miss_uv, monotone in between.
    Returns (std_px, var_px2).
    """
    trust = np.clip(np.asarray(trust, dtype=float), 0.0, 1.0)
    inv_var = trust / (r_visible_uv ** 2) + (1.0 - trust) / (r_miss_uv ** 2)
    var = 1.0 / inv_var
    return np.sqrt(var), var


def r_plan_matrix(std_px: float) -> np.ndarray:
    """Explicit 2x2 diagonal measurement covariance in px^2."""
    v = float(std_px) ** 2
    return np.array([[v, 0.0], [0.0, v]])
