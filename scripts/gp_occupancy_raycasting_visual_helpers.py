"""Canonical helper for the supervisor-facing GP occupancy story notebook.

This helper intentionally consolidates the previous GP notebook/helper work into
one file. It covers:

- scene construction and oracle 3D raycasting
- historical old 2D GP baseline
- shared 2D opacity GP and 2.5D ray-opacity prior
- one online correction rule in opacity space
- one compact closed-loop planner comparison
- presentation-oriented plots
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm


def _ensure_repo_paths() -> None:
    repo_root = Path.cwd()
    if not (repo_root / "scripts").exists():
        repo_root = repo_root.parent
    for path in (
        repo_root,
        repo_root / "src" / "planning",
        repo_root / "src" / "unav_common",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)


_ensure_repo_paths()
jax.config.update("jax_enable_x64", True)

from scripts.botnav_efe_helpers import (
    ET1,
    ET2,
    PlanarCamera,
    UnicycleEFEAgent,
    ambiguity,
    planned_trajectory_unicycle,
    predict_unicycle,
    risk,
    unicycle_step,
)
from scripts.botnav_efe_jax import bind_unicycle_agent_jax, unicycle_jacobian_jax, unicycle_step_jax
from scripts.state_dependent_observation_helpers import (
    covariance_trace_series,
    make_occupancy_grid,
    observation_covariance,
    process_covariance_from_rho,
)
from unav_common.camera_model import ObliqueCameraModel


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.2
np.set_printoptions(precision=4, suppress=True)


# -----------------------------------------------------------------------------
# Core utilities and configs
# -----------------------------------------------------------------------------


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid_prime_from_mu(mu):
    s = sigmoid(mu)
    return s * (1.0 - s)


def wrap_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))


def _xy_visibility_sigma_points_np(mean_xy: np.ndarray, cov_xy: np.ndarray, kappa=1.0):
    mean_xy = np.asarray(mean_xy, dtype=float).reshape(2)
    cov_xy = np.asarray(cov_xy, dtype=float).reshape(2, 2)
    cov_xy = 0.5 * (cov_xy + cov_xy.T)
    kappa = max(float(kappa), 1e-6)
    scale = np.sqrt(2.0 + kappa)
    chol = np.linalg.cholesky(cov_xy + 1e-9 * np.eye(2))
    spread = scale * chol
    sigma_points = np.vstack(
        [
            mean_xy,
            mean_xy + spread[:, 0],
            mean_xy - spread[:, 0],
            mean_xy + spread[:, 1],
            mean_xy - spread[:, 1],
        ]
    )
    weights = np.array(
        [kappa / (2.0 + kappa)] + [1.0 / (2.0 * (2.0 + kappa))] * 4,
        dtype=float,
    )
    return sigma_points, weights


def _xy_visibility_sigma_points_jax(mean_xy: jnp.ndarray, cov_xy: jnp.ndarray, kappa=1.0):
    mean_xy = jnp.asarray(mean_xy, dtype=jnp.float64).reshape((2,))
    cov_xy = jnp.asarray(cov_xy, dtype=jnp.float64).reshape((2, 2))
    cov_xy = 0.5 * (cov_xy + cov_xy.T)
    kappa = max(float(kappa), 1e-6)
    scale = jnp.sqrt(jnp.asarray(2.0 + kappa, dtype=jnp.float64))
    chol = jnp.linalg.cholesky(cov_xy + 1e-9 * jnp.eye(2, dtype=jnp.float64))
    spread = scale * chol
    sigma_points = jnp.vstack(
        [
            mean_xy,
            mean_xy + spread[:, 0],
            mean_xy - spread[:, 0],
            mean_xy + spread[:, 1],
            mean_xy - spread[:, 1],
        ]
    )
    weights = jnp.asarray(
        [kappa / (2.0 + kappa)] + [1.0 / (2.0 * (2.0 + kappa))] * 4,
        dtype=jnp.float64,
    )
    return sigma_points, weights


@dataclass(frozen=True)
class StoryVariant:
    name: str
    rack_rect: Tuple[float, float, float, float]
    rack_zmax: float


@dataclass(frozen=True)
class SceneConfig:
    world_name: str = "warehouse_occ_light.world.sdf"
    task_name: str = "T2_diag_far"
    cam_pos: Tuple[float, float, float] = (-5.35, -5.20, 3.05)
    look_at: Tuple[float, float, float] = (-1.9100553137375051, -1.7600426780919225, 0.0)
    img_width: int = 1280
    img_height: int = 720
    fov_h_rad: float = 1.5708
    start_state: Tuple[float, float, float] = (-5.20, 1.65, -0.95)
    goal_state: Tuple[float, float, float] = (-1.0, -3.0, 0.0)
    rack_rect: Tuple[float, float, float, float] = (-3.58, -2.62, -2.52, -1.18)
    rack_zmin: float = 0.0
    rack_zmax: float = 2.70
    world_min: float = -6.0
    world_max: float = 6.0
    grid_resolution: float = 0.1
    plot_grid_n: int = 81
    target_point_body: Tuple[float, float, float] = (0.0, 0.0, 0.14)
    pre_occlusion_state: Tuple[float, float, float] = (-4.75, -2.10, -0.95)
    shadow_state: Tuple[float, float, float] = (-2.35, -0.55, -0.95)
    deep_shadow_state: Tuple[float, float, float] = (-1.45, 1.00, -0.95)


@dataclass(frozen=True)
class PlanningConfig:
    dt: float = 0.2
    horizon: int = 18
    n_steps: int = 90
    q_rho_xy: float = 1e-2
    q_exec_xy: float = 0.03
    q_exec_theta: float = 0.08
    r_visible_uv: float = 2.5
    r_miss_uv: float = 420.0
    goal_prior_u_std_start: float = 80.0
    goal_prior_v_std_start: float = 80.0
    goal_prior_u_std_final: float = 18.0
    goal_prior_v_std_final: float = 18.0
    mean0_xy_std: float = 0.18
    mean0_theta_std: float = 0.10
    eta: float = 0.0
    risk_scale: float = 1.25
    ambiguity_scale: float = 0.20
    discount_gamma: float = 0.98
    detection_power: float = 1.5
    visibility_power: float = 3.0
    visibility_sigma_kappa: float = 1.0
    goal_tightening_power: float = 0.45
    v_lims: Tuple[float, float] = (0.0, 0.60)
    w_lims: Tuple[float, float] = (-1.2, 1.2)
    optimizer_maxiter: int = 80
    optimizer_maxfun: int = 500
    optimizer_ftol: float = 1e-6


@dataclass(frozen=True)
class SharedGPConfig:
    ell_x: float = 0.32
    ell_y: float = 0.32
    signal_var: float = 1.0
    bias_var: float = 0.04
    noise_var: float = 0.02
    prior_occ: float = 0.005
    beta: float = 1.0
    ray_samples: int = 120
    height_tau: float = 0.08


@dataclass(frozen=True)
class OldGPConfig:
    length_scale: float = 0.18
    signal_var: float = 1.0
    noise_var: float = 0.01
    prior_mean: float = 0.02
    tau: float = 12.0
    ray_samples: int = 120


def _run_mode_settings(run_mode: str) -> Dict[str, int]:
    mode = str(run_mode).strip().lower()
    if mode == "smoke":
        return {
            "plot_grid_n": 41,
            "horizon": 10,
            "n_steps": 32,
            "optimizer_maxiter": 45,
            "optimizer_maxfun": 200,
        }
    if mode == "full":
        return {
            "plot_grid_n": 81,
            "horizon": 18,
            "n_steps": 90,
            "optimizer_maxiter": 110,
            "optimizer_maxfun": 900,
        }
    raise ValueError("run_mode must be 'smoke' or 'full'")


STORY_METHODS = ("old 2D GP", "offline prior", "corrected prior")


def _variant_specs() -> Dict[str, StoryVariant]:
    return {
        "Short box": StoryVariant(
            name="Short box",
            rack_rect=(-3.58, -2.62, -2.52, -1.18),
            rack_zmax=1.10,
        ),
        "Tall box": StoryVariant(
            name="Tall box",
            rack_rect=(-3.58, -2.62, -2.52, -1.18),
            rack_zmax=2.70,
        ),
    }


def _probe_state(scene: Mapping[str, object], probe_name: str) -> np.ndarray:
    mapping = {
        "start": "start_state",
        "pre-occ": "pre_occlusion_state",
        "shadow": "shadow_state",
        "deep-shadow": "deep_shadow_state",
        "goal": "goal_state",
    }
    if probe_name not in mapping:
        raise ValueError(f"Unknown probe '{probe_name}'")
    return np.asarray(scene[mapping[probe_name]], dtype=float)


def project_world_point_np(camera: ObliqueCameraModel, point_xyz: Sequence[float]) -> Tuple[float, float, bool]:
    point = np.asarray(point_xyz, dtype=float)
    cam_pt = camera.R @ (point - camera.cam_pos)
    if cam_pt[2] <= 1e-9:
        return math.nan, math.nan, False
    pixel = camera.K @ cam_pt
    u = float(pixel[0] / pixel[2])
    v = float(pixel[1] / pixel[2])
    visible = 0.0 <= u < camera.img_width and 0.0 <= v < camera.img_height
    return u, v, bool(visible)


def make_project_world_point_jax(camera: ObliqueCameraModel) -> Callable[[jnp.ndarray], jnp.ndarray]:
    R = jnp.asarray(camera.R, dtype=jnp.float64)
    K = jnp.asarray(camera.K, dtype=jnp.float64)
    cam_pos = jnp.asarray(camera.cam_pos, dtype=jnp.float64)

    def project_fn(point_xyz):
        point_xyz = jnp.asarray(point_xyz, dtype=jnp.float64)
        cam_pt = R @ (point_xyz - cam_pos)
        z = jnp.maximum(cam_pt[2], 1e-6)
        pixel = K @ (cam_pt / z)
        return pixel[:2]

    return project_fn


def state_to_world_target(state: Sequence[float], target_body: Sequence[float]) -> np.ndarray:
    x, y, theta = map(float, state[:3])
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    target_body = np.asarray(target_body, dtype=float)
    xy = np.array([x, y], dtype=float) + rot @ target_body[:2]
    return np.array([xy[0], xy[1], target_body[2]], dtype=float)


def make_target_world_jax(target_body: Sequence[float]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    target_body = jnp.asarray(target_body, dtype=jnp.float64)

    def target_fn(state):
        x, y, theta = state[0], state[1], state[2]
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]], dtype=jnp.float64)
        xy = jnp.array([x, y], dtype=jnp.float64) + rot @ target_body[:2]
        return jnp.array([xy[0], xy[1], target_body[2]], dtype=jnp.float64)

    return target_fn


def nominal_heading_to_goal(x: np.ndarray, y: np.ndarray, goal_xy: Sequence[float]) -> np.ndarray:
    goal_xy = np.asarray(goal_xy[:2], dtype=float)
    return np.arctan2(goal_xy[1] - y, goal_xy[0] - x)


def segment_intersects_aabb(
    p0: Sequence[float],
    p1: Sequence[float],
    box_min: Sequence[float],
    box_max: Sequence[float],
) -> bool:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    direction = p1 - p0
    t_min = 0.0
    t_max = 1.0
    for i in range(3):
        if abs(direction[i]) < 1e-12:
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return False
            continue
        inv_d = 1.0 / direction[i]
        t0 = (box_min[i] - p0[i]) * inv_d
        t1 = (box_max[i] - p0[i]) * inv_d
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < t_min:
            return False
    return True


def sample_occ(grid, x: float, y: float) -> float:
    xs_local = grid.xs
    ys_local = grid.ys
    occ = grid.occupancy

    if x <= xs_local[0] or x >= xs_local[-1] or y <= ys_local[0] or y >= ys_local[-1]:
        return 1.0

    ix = int(np.searchsorted(xs_local, x, side="right") - 1)
    iy = int(np.searchsorted(ys_local, y, side="right") - 1)
    ix = max(0, min(ix, xs_local.shape[0] - 2))
    iy = max(0, min(iy, ys_local.shape[0] - 2))

    x0, x1 = xs_local[ix], xs_local[ix + 1]
    y0, y1 = ys_local[iy], ys_local[iy + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    z00 = occ[iy, ix]
    z10 = occ[iy, ix + 1]
    z01 = occ[iy + 1, ix]
    z11 = occ[iy + 1, ix + 1]
    return float(
        (1.0 - ty) * ((1.0 - tx) * z00 + tx * z10)
        + ty * ((1.0 - tx) * z01 + tx * z11)
    )


def first_hit_point(grid, start: Sequence[float], end: Sequence[float], threshold=0.55, n=180):
    ts = np.linspace(0.0, 1.0, int(max(n, 2)))
    points = np.outer(1.0 - ts, np.asarray(start, dtype=float)) + np.outer(ts, np.asarray(end, dtype=float))
    occ_values = np.asarray([sample_occ(grid, p[0], p[1]) for p in points], dtype=float)
    hit_idx = np.flatnonzero(occ_values >= float(threshold))
    if hit_idx.size == 0:
        return None, points
    idx = int(hit_idx[0])
    return points[idx].copy(), points[: idx + 1].copy()


def obstacle_pixel_bbox(camera: PlanarCamera, obstacle_rect: Sequence[float]):
    ox0, ox1, oy0, oy1 = obstacle_rect
    corners = np.array([[ox0, oy0], [ox0, oy1], [ox1, oy0], [ox1, oy1]], dtype=float)
    uv = np.asarray([camera.world_to_pixel(x, y) for x, y in corners], dtype=float)
    return (
        float(np.min(uv[:, 0])),
        float(np.max(uv[:, 0])),
        float(np.min(uv[:, 1])),
        float(np.max(uv[:, 1])),
    )


def pixel_to_world_in_grid(camera: PlanarCamera, grid, u: float, v: float):
    x, y = camera.pixel_to_world(float(u), float(v))
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if x <= grid.xs[0] or x >= grid.xs[-1] or y <= grid.ys[0] or y >= grid.ys[-1]:
        return None
    return np.array([x, y], dtype=float)


def sample_free_points_along_ray(points, hit_exists):
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    if hit_exists:
        qs = np.array([0.15, 0.40, 0.70, 0.92], dtype=float)
    else:
        qs = np.array([0.20, 0.55, 0.90], dtype=float)
    idx = np.unique(np.clip((qs * (points.shape[0] - 1)).astype(int), 0, points.shape[0] - 1))
    return points[idx]


class OldSimpleRBFGP:
    def __init__(self, length_scale=0.20, signal_var=1.0, noise_var=0.01, prior_mean=0.02, jitter=1e-8):
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)
        self.noise_var = float(noise_var)
        self.prior_mean = float(prior_mean)
        self.jitter = float(jitter)
        self.X_train = None
        self.y_mean = None
        self.L = None
        self.alpha = None

    def _kernel(self, Xa, Xb):
        Xa = np.asarray(Xa, dtype=float)
        Xb = np.asarray(Xb, dtype=float)
        d2 = np.sum((Xa[:, None, :] - Xb[None, :, :]) ** 2, axis=2)
        ls2 = max(self.length_scale ** 2, 1e-12)
        return self.signal_var * np.exp(-0.5 * d2 / ls2)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.X_train = X
        self.y_mean = self.prior_mean
        y0 = y - self.y_mean
        K = self._kernel(X, X) + (self.noise_var + self.jitter) * np.eye(X.shape[0])
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y0))
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        Ks = self._kernel(X, self.X_train)
        mu = self.y_mean + Ks @ self.alpha
        if not return_std:
            return mu
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.signal_var - np.sum(v * v, axis=0), 1e-12)
        return mu, np.sqrt(var)


class LatentMaternARDGP:
    def __init__(
        self,
        *,
        ell_x=0.18,
        ell_y=0.18,
        signal_var=1.0,
        bias_var=0.10,
        noise_var=0.01,
        prior_mean_latent=0.0,
        jitter=1e-8,
    ):
        self.ell_x = float(ell_x)
        self.ell_y = float(ell_y)
        self.signal_var = float(signal_var)
        self.bias_var = float(bias_var)
        self.noise_var = float(noise_var)
        self.prior_mean_latent = float(prior_mean_latent)
        self.jitter = float(jitter)
        self.X_train = None
        self.L = None
        self.alpha = None

    def _kernel(self, Xa, Xb):
        Xa = np.asarray(Xa, dtype=float)
        Xb = np.asarray(Xb, dtype=float)
        dx = (Xa[:, None, 0] - Xb[None, :, 0]) / max(self.ell_x, 1e-12)
        dy = (Xa[:, None, 1] - Xb[None, :, 1]) / max(self.ell_y, 1e-12)
        r = np.sqrt(dx ** 2 + dy ** 2)
        matern = self.signal_var * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
        return matern + self.bias_var

    @property
    def prior_var(self) -> float:
        return self.signal_var + self.bias_var

    def fit(self, X, y_latent):
        X = np.asarray(X, dtype=float)
        y_latent = np.asarray(y_latent, dtype=float).reshape(-1)
        self.X_train = X
        y0 = y_latent - self.prior_mean_latent
        K = self._kernel(X, X) + (self.noise_var + self.jitter) * np.eye(X.shape[0])
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y0))
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        Ks = self._kernel(X, self.X_train)
        mu = self.prior_mean_latent + Ks @ self.alpha
        if not return_std:
            return mu
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.prior_var - np.sum(v * v, axis=0), 1e-12)
        return mu, np.sqrt(var)


# -----------------------------------------------------------------------------
# Scene, dataset, and GP prior construction
# -----------------------------------------------------------------------------


def build_old_gp_dataset(
    grid,
    camera: PlanarCamera,
    obstacle_rect: Sequence[float],
    start_state: Sequence[float],
    goal_state: Sequence[float],
    rng: np.random.Generator,
    *,
    n_rays=520,
    max_free=360,
    max_occ=180,
):
    free_points = []
    occ_points = []
    u_start, v_start = camera.world_to_pixel(float(start_state[0]), float(start_state[1]))
    u_goal, v_goal = camera.world_to_pixel(float(goal_state[0]), float(goal_state[1]))
    umin, umax, vmin, vmax = obstacle_pixel_bbox(camera, obstacle_rect)

    for ray_idx in range(int(n_rays)):
        mode = ray_idx % 5
        end = None
        for _ in range(40):
            if mode == 0:
                u = rng.uniform(60.0, camera.img_width - 60.0)
                v = rng.uniform(60.0, camera.img_height - 60.0)
            elif mode == 1:
                u = rng.uniform(umin - 25.0, umax + 25.0)
                v = rng.uniform(vmin - 20.0, vmax + 20.0)
            elif mode == 2:
                u = rng.uniform(umin - 40.0, umax + 90.0)
                v = rng.uniform(vmin - 40.0, vmax + 40.0)
            elif mode == 3:
                a = rng.uniform(0.0, 1.0)
                u = (1.0 - a) * u_start + a * u_goal + rng.normal(scale=60.0)
                v = (1.0 - a) * v_start + a * v_goal + rng.normal(scale=45.0)
            else:
                if rng.random() < 0.5:
                    u = rng.uniform(umin - 30.0, umax + 30.0)
                    v = rng.uniform(vmin - 55.0, vmin - 10.0)
                else:
                    u = rng.uniform(umin - 30.0, umax + 30.0)
                    v = rng.uniform(vmax + 10.0, vmax + 55.0)
            end = pixel_to_world_in_grid(camera, grid, u, v)
            if end is not None:
                break
        if end is None:
            continue

        hit_point, traversed_points = first_hit_point(
            grid,
            np.asarray(camera.cam_pos[:2], dtype=float),
            end,
            threshold=0.55,
            n=180,
        )
        if hit_point is not None:
            occ_points.append(hit_point.copy())
            free_segment = traversed_points[:-1]
        else:
            free_segment = traversed_points
        sampled_free = sample_free_points_along_ray(free_segment, hit_point is not None)
        for p in sampled_free:
            free_points.append(p.copy())

    free_points = np.asarray(free_points, dtype=float) if free_points else np.empty((0, 2), dtype=float)
    occ_points = np.asarray(occ_points, dtype=float) if occ_points else np.empty((0, 2), dtype=float)
    if free_points.shape[0]:
        free_points = np.unique(np.round(free_points, 3), axis=0)
    if occ_points.shape[0]:
        occ_points = np.unique(np.round(occ_points, 3), axis=0)
    if free_points.shape[0] > int(max_free):
        idx = rng.choice(free_points.shape[0], size=int(max_free), replace=False)
        free_points = free_points[idx]
    if occ_points.shape[0] > int(max_occ):
        idx = rng.choice(occ_points.shape[0], size=int(max_occ), replace=False)
        occ_points = occ_points[idx]

    X_train = np.vstack([free_points, occ_points])
    y_train = np.concatenate([
        np.zeros(free_points.shape[0], dtype=float),
        np.ones(occ_points.shape[0], dtype=float),
    ])
    return X_train, y_train, free_points, occ_points


def build_soft_opacity_dataset(
    grid,
    camera_planar: PlanarCamera,
    obstacle_rect: Sequence[float],
    start_state: Sequence[float],
    goal_state: Sequence[float],
    rng: np.random.Generator,
    *,
    n_rays_uniform=260,
    n_rays_focus=260,
    n_corridor=180,
    n_boundary=240,
    max_clear=560,
    max_transition=260,
    max_occ=220,
):
    x0, x1, y0, y1 = map(float, obstacle_rect)
    clear_points: List[np.ndarray] = []
    transition_points: List[np.ndarray] = []
    occ_points: List[np.ndarray] = []
    u_start, v_start = camera_planar.world_to_pixel(float(start_state[0]), float(start_state[1]))
    u_goal, v_goal = camera_planar.world_to_pixel(float(goal_state[0]), float(goal_state[1]))
    umin, umax, vmin, vmax = obstacle_pixel_bbox(camera_planar, obstacle_rect)

    def add_sample(arr: List[np.ndarray], p_xy: Sequence[float]):
        arr.append(np.asarray(p_xy, dtype=float).copy())

    def ray_endpoint_samples():
        endpoints = []
        for _ in range(int(n_rays_uniform)):
            u = rng.uniform(60.0, camera_planar.img_width - 60.0)
            v = rng.uniform(60.0, camera_planar.img_height - 60.0)
            p = pixel_to_world_in_grid(camera_planar, grid, u, v)
            if p is not None:
                endpoints.append(p)
        for _ in range(int(n_rays_focus)):
            if rng.random() < 0.5:
                u = rng.uniform(umin - 45.0, umax + 45.0)
                v = rng.uniform(vmin - 45.0, vmax + 45.0)
            else:
                a = rng.uniform(0.0, 1.0)
                u = (1.0 - a) * u_start + a * u_goal + rng.normal(scale=35.0)
                v = (1.0 - a) * v_start + a * v_goal + rng.normal(scale=28.0)
            p = pixel_to_world_in_grid(camera_planar, grid, u, v)
            if p is not None:
                endpoints.append(p)
        return endpoints

    for end in ray_endpoint_samples():
        hit_point, traversed_points = first_hit_point(
            grid,
            np.asarray(camera_planar.cam_pos[:2], dtype=float),
            end,
            threshold=0.55,
            n=220,
        )
        if traversed_points.shape[0] == 0:
            continue
        free_segment = traversed_points[:-1] if hit_point is not None else traversed_points
        if free_segment.shape[0]:
            idxs = np.unique(
                np.clip(
                    (np.array([0.15, 0.40, 0.70], dtype=float) * max(free_segment.shape[0] - 1, 1)).astype(int),
                    0,
                    free_segment.shape[0] - 1,
                )
            )
            for idx in idxs:
                add_sample(clear_points, free_segment[idx])
            if hit_point is not None and free_segment.shape[0] >= 2:
                add_sample(transition_points, free_segment[-1])
        if hit_point is not None:
            add_sample(transition_points, hit_point)

    corridor = np.linspace(0.0, 1.0, int(max(n_corridor, 2)))
    p0 = np.asarray(start_state[:2], dtype=float)
    p1 = np.asarray(goal_state[:2], dtype=float)
    tangent = p1 - p0
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    for a in corridor:
        base = (1.0 - a) * p0 + a * p1
        for d in (-0.35, 0.0, 0.35):
            p = base + d * normal + rng.normal(scale=0.03, size=2)
            if sample_occ(grid, p[0], p[1]) < 0.45:
                add_sample(clear_points, p)

    for _ in range(int(n_boundary)):
        side = rng.integers(0, 4)
        if side == 0:
            x = rng.uniform(x0, x1)
            y = y0 + rng.normal(scale=0.06)
        elif side == 1:
            x = rng.uniform(x0, x1)
            y = y1 + rng.normal(scale=0.06)
        elif side == 2:
            x = x0 + rng.normal(scale=0.06)
            y = rng.uniform(y0, y1)
        else:
            x = x1 + rng.normal(scale=0.06)
            y = rng.uniform(y0, y1)
        add_sample(transition_points, np.array([x, y], dtype=float))

    # Add a coarse free-space support lattice so the latent GP has labeled
    # evidence away from the rack instead of reverting to high uncertainty over
    # most of the map.
    support_spacing = 0.95
    obstacle_margin = 0.35
    for x in np.arange(grid.xs[4], grid.xs[-4], support_spacing):
        for y in np.arange(grid.ys[4], grid.ys[-4], support_spacing):
            if sample_occ(grid, x, y) >= 0.20:
                continue
            if (x0 - obstacle_margin) <= x <= (x1 + obstacle_margin) and (y0 - obstacle_margin) <= y <= (y1 + obstacle_margin):
                continue
            add_sample(clear_points, np.array([x, y], dtype=float))

    for _ in range(int(max_occ)):
        x = rng.uniform(x0 + 0.08, x1 - 0.08)
        y = rng.uniform(y0 + 0.08, y1 - 0.08)
        add_sample(occ_points, np.array([x, y], dtype=float))

    def finalize(points, max_count):
        arr = np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float)
        if arr.shape[0]:
            arr = np.unique(np.round(arr, 3), axis=0)
        if arr.shape[0] > int(max_count):
            idx = rng.choice(arr.shape[0], size=int(max_count), replace=False)
            arr = arr[idx]
        return arr

    clear_points_arr = finalize(clear_points, max_clear)
    transition_points_arr = finalize(transition_points, max_transition)
    occ_points_arr = finalize(occ_points, max_occ)

    X_train = np.vstack([clear_points_arr, transition_points_arr, occ_points_arr])
    y_soft = np.concatenate([
        np.full(clear_points_arr.shape[0], 0.005, dtype=float),
        np.full(transition_points_arr.shape[0], 0.15, dtype=float),
        np.full(occ_points_arr.shape[0], 0.995, dtype=float),
    ])
    if transition_points_arr.shape[0]:
        boundary_labels = np.full(transition_points_arr.shape[0], 0.15, dtype=float)
        inside_mask = (
            (transition_points_arr[:, 0] >= x0)
            & (transition_points_arr[:, 0] <= x1)
            & (transition_points_arr[:, 1] >= y0)
            & (transition_points_arr[:, 1] <= y1)
        )
        boundary_labels[inside_mask] = 0.60
        start_idx = clear_points_arr.shape[0]
        end_idx = start_idx + transition_points_arr.shape[0]
        y_soft[start_idx:end_idx] = boundary_labels

    labels = np.concatenate([
        np.full(clear_points_arr.shape[0], "clear", dtype=object),
        np.full(transition_points_arr.shape[0], "transition", dtype=object),
        np.full(occ_points_arr.shape[0], "occupied", dtype=object),
    ])
    return {
        "X_train": X_train,
        "y_soft": y_soft,
        "y_latent": logit(y_soft),
        "labels": labels,
        "clear_points": clear_points_arr,
        "transition_points": transition_points_arr,
        "occ_points": occ_points_arr,
    }


def build_scene(config: SceneConfig | None = None) -> Dict[str, object]:
    config = SceneConfig() if config is None else config
    camera_planar = PlanarCamera(
        cam_pos=config.cam_pos,
        look_at=config.look_at,
        img_width=config.img_width,
        img_height=config.img_height,
        fov_h_rad=config.fov_h_rad,
    )
    camera_3d = ObliqueCameraModel(
        cam_pos=config.cam_pos,
        look_at=config.look_at,
        img_width=config.img_width,
        img_height=config.img_height,
        fov_h_rad=config.fov_h_rad,
    )
    camera_xy = np.asarray(config.cam_pos[:2], dtype=float)
    obstacle_rect = tuple(map(float, config.rack_rect))
    obstacle_center_xy = np.array(
        [0.5 * (obstacle_rect[0] + obstacle_rect[1]), 0.5 * (obstacle_rect[2] + obstacle_rect[3])],
        dtype=float,
    )
    box_min = np.array([obstacle_rect[0], obstacle_rect[2], config.rack_zmin], dtype=float)
    box_max = np.array([obstacle_rect[1], obstacle_rect[3], config.rack_zmax], dtype=float)

    true_grid = make_occupancy_grid(
        xmin=config.world_min,
        xmax=config.world_max,
        ymin=config.world_min,
        ymax=config.world_max,
        resolution=config.grid_resolution,
        rectangles=[obstacle_rect],
        circles=None,
        border_occupancy=True,
    )
    start_state = np.array(config.start_state, dtype=float)
    goal_state = np.array(config.goal_state, dtype=float)
    pre_state = np.array(config.pre_occlusion_state, dtype=float)
    shadow_state = np.array(config.shadow_state, dtype=float)
    deep_shadow_state = np.array(config.deep_shadow_state, dtype=float)

    xs = np.linspace(true_grid.xs.min(), true_grid.xs.max(), int(config.plot_grid_n))
    ys = np.linspace(true_grid.ys.min(), true_grid.ys.max(), int(config.plot_grid_n))
    Xg, Yg = np.meshgrid(xs, ys)
    heading_ref = nominal_heading_to_goal(Xg, Yg, goal_state[:2])
    state_grid = np.stack([Xg, Yg, heading_ref], axis=-1)

    return {
        "config": config,
        "camera_planar": camera_planar,
        "camera_3d": camera_3d,
        "camera_xy": camera_xy,
        "obstacle_rect": obstacle_rect,
        "obstacle_center_xy": obstacle_center_xy,
        "box_min": box_min,
        "box_max": box_max,
        "target_point_body": np.asarray(config.target_point_body, dtype=float),
        "true_grid": true_grid,
        "start_state": start_state,
        "goal_state": goal_state,
        "pre_occlusion_state": pre_state,
        "shadow_state": shadow_state,
        "deep_shadow_state": deep_shadow_state,
        "xs": xs,
        "ys": ys,
        "Xg": Xg,
        "Yg": Yg,
        "heading_ref": heading_ref,
        "state_grid": state_grid,
    }


def build_variant_scene(variant_name: str, run_mode: str = "full"):
    if variant_name not in _variant_specs():
        raise ValueError(f"Unknown variant '{variant_name}'")
    spec = _variant_specs()[variant_name]
    settings = _run_mode_settings(run_mode)
    cfg = replace(
        SceneConfig(),
        world_name=f"gp_story::{variant_name.lower().replace(' ', '_')}",
        task_name=f"gp_story::{variant_name.lower().replace(' ', '_')}",
        rack_rect=spec.rack_rect,
        rack_zmax=spec.rack_zmax,
        plot_grid_n=settings["plot_grid_n"],
    )
    scene = build_scene(cfg)
    scene["variant_name"] = variant_name
    return scene


def _target_world_point(scene: Mapping[str, object], state: Sequence[float]) -> np.ndarray:
    return state_to_world_target(state, scene["target_point_body"])


def make_observation_fns(scene: Mapping[str, object]):
    camera_3d = scene["camera_3d"]
    target_body = scene["target_point_body"]
    project_jax = make_project_world_point_jax(camera_3d)
    target_jax = make_target_world_jax(target_body)

    def g_np(state):
        target_w = state_to_world_target(state, target_body)
        u, v, _ = project_world_point_np(camera_3d, target_w)
        return np.array([u, v], dtype=float)

    def g_jax(state):
        target_w = target_jax(state)
        return project_jax(target_w)

    return g_np, g_jax


def _oracle_p_vis(scene: Mapping[str, object], state: Sequence[float], min_visible=0.02) -> float:
    camera = scene["camera_3d"]
    target = _target_world_point(scene, state)
    blocked = segment_intersects_aabb(
        np.asarray(camera.cam_pos, dtype=float),
        target,
        np.asarray(scene["box_min"], dtype=float),
        np.asarray(scene["box_max"], dtype=float),
    )
    return float(min_visible if blocked else 1.0)


def evaluate_state_fn_on_grid(scene: Mapping[str, object], fn: Callable[[np.ndarray], float]) -> np.ndarray:
    state_grid = np.asarray(scene["state_grid"], dtype=float).reshape(-1, 3)
    vals = np.asarray([fn(s) for s in state_grid], dtype=float)
    return vals.reshape(scene["Xg"].shape)


def build_old_gp_prior(scene: Mapping[str, object], seed=12, cfg: OldGPConfig | None = None):
    cfg = OldGPConfig() if cfg is None else cfg
    rng = np.random.default_rng(int(seed))
    X_train, y_train, free_points, occ_points = build_old_gp_dataset(
        scene["true_grid"],
        scene["camera_planar"],
        scene["obstacle_rect"],
        scene["start_state"],
        scene["goal_state"],
        rng,
    )
    gp = OldSimpleRBFGP(
        length_scale=cfg.length_scale,
        signal_var=cfg.signal_var,
        noise_var=cfg.noise_var,
        prior_mean=cfg.prior_mean,
    ).fit(X_train, y_train)
    XY = np.column_stack([scene["Xg"].ravel(), scene["Yg"].ravel()])
    occ_mu, occ_std = gp.predict(XY, return_std=True)
    P_occ = np.clip(occ_mu.reshape(scene["Xg"].shape), 1e-4, 1.0 - 1e-4)
    P_occ_std = occ_std.reshape(scene["Xg"].shape)
    occ_interp = RegularGridInterpolator((scene["ys"], scene["xs"]), P_occ, bounds_error=False, fill_value=cfg.prior_mean)

    def p_vis_old(state):
        start = scene["camera_xy"]
        end = np.asarray(state[:2], dtype=float)
        ts = np.linspace(0.0, 1.0, int(max(cfg.ray_samples, 2)))
        points = np.outer(1.0 - ts, start) + np.outer(ts, end)
        occ_vals = np.clip(occ_interp(np.column_stack([points[:, 1], points[:, 0]])), 1e-4, 1.0)
        return float(np.clip(np.exp(-cfg.tau * float(np.mean(occ_vals))), 1e-4, 1.0 - 1e-4))

    P_vis_old = np.asarray([p_vis_old(s) for s in scene["state_grid"].reshape(-1, 3)], dtype=float).reshape(scene["Xg"].shape)
    return {
        "cfg": cfg,
        "gp": gp,
        "X_train": X_train,
        "y_train": y_train,
        "free_points": free_points,
        "occ_points": occ_points,
        "P_occ_gp": P_occ,
        "P_occ_std": P_occ_std,
        "P_vis_old_map": P_vis_old,
        "occ_interp": occ_interp,
        "p_vis_old": p_vis_old,
    }



def build_shared_gp_field(scene: Mapping[str, object], seed=12, cfg: SharedGPConfig | None = None):
    cfg = SharedGPConfig() if cfg is None else cfg
    rng = np.random.default_rng(int(seed))
    dataset = build_soft_opacity_dataset(
        scene["true_grid"],
        scene["camera_planar"],
        scene["obstacle_rect"],
        scene["start_state"],
        scene["goal_state"],
        rng,
    )
    prior_mean_latent = float(logit(cfg.prior_occ))
    gp = LatentMaternARDGP(
        ell_x=cfg.ell_x,
        ell_y=cfg.ell_y,
        signal_var=cfg.signal_var,
        bias_var=cfg.bias_var,
        noise_var=cfg.noise_var,
        prior_mean_latent=prior_mean_latent,
    ).fit(dataset["X_train"], dataset["y_latent"])

    XY = np.column_stack([scene["Xg"].ravel(), scene["Yg"].ravel()])
    mu_f, sigma_f = gp.predict(XY, return_std=True)
    mu_f_map = mu_f.reshape(scene["Xg"].shape)
    sigma_f_map = sigma_f.reshape(scene["Xg"].shape)
    rho_mean_map = np.clip(sigmoid(mu_f_map), 1e-5, 1.0 - 1e-5)
    rho_std_map = np.maximum(sigmoid_prime_from_mu(mu_f_map) * sigma_f_map, 1e-6)
    rho_conservative_map = np.clip(sigmoid(mu_f_map + cfg.beta * sigma_f_map), 1e-5, 1.0 - 1e-5)

    return {
        "cfg": cfg,
        "dataset": dataset,
        "gp": gp,
        "prior_mean_latent": prior_mean_latent,
        "mu_f_map": mu_f_map,
        "sigma_f_map": sigma_f_map,
        "rho_mean_map": rho_mean_map,
        "rho_std_map": rho_std_map,
        "rho_conservative_map": rho_conservative_map,
        "mu_f_interp": RegularGridInterpolator((scene["ys"], scene["xs"]), mu_f_map, bounds_error=False, fill_value=prior_mean_latent),
        "sigma_f_interp": RegularGridInterpolator((scene["ys"], scene["xs"]), sigma_f_map, bounds_error=False, fill_value=1e-3),
        "rho_mean_interp": RegularGridInterpolator((scene["ys"], scene["xs"]), rho_mean_map, bounds_error=False, fill_value=cfg.prior_occ),
        "rho_std_interp": RegularGridInterpolator((scene["ys"], scene["xs"]), rho_std_map, bounds_error=False, fill_value=1e-3),
        "rho_conservative_interp": RegularGridInterpolator((scene["ys"], scene["xs"]), rho_conservative_map, bounds_error=False, fill_value=cfg.prior_occ),
    }

# -----------------------------------------------------------------------------
# Offline prior models
# -----------------------------------------------------------------------------


def _height_gate(z_vals: np.ndarray, zmax: float, tau: float) -> np.ndarray:
    tau = max(float(tau), 1e-6)
    return 1.0 / (1.0 + np.exp((np.asarray(z_vals, dtype=float) - float(zmax)) / tau))



def build_height_aware_prior(
    scene: Mapping[str, object],
    shared_gp: Mapping[str, object],
    *,
    beta_override: float,
    height_tau_override: float | None = None,
):
    cfg = shared_gp["cfg"]
    mu_f_interp = shared_gp["mu_f_interp"]
    sigma_f_interp = shared_gp["sigma_f_interp"]
    cam_xyz = np.asarray(scene["camera_3d"].cam_pos, dtype=float)
    beta = float(beta_override)
    tau = float(cfg.height_tau if height_tau_override is None else height_tau_override)

    def ray_details_for_state(state):
        target = _target_world_point(scene, state)
        ts = np.linspace(0.0, 1.0, int(max(cfg.ray_samples, 2)))
        pts_xyz = np.outer(1.0 - ts, cam_xyz) + np.outer(ts, target)
        segs = np.diff(pts_xyz, axis=0)
        ds_segs = np.linalg.norm(segs, axis=1)
        ds = np.concatenate([ds_segs, ds_segs[-1:]], axis=0)
        arc = np.concatenate([[0.0], np.cumsum(ds_segs)], axis=0)
        horiz = np.linalg.norm(pts_xyz[:, :2] - cam_xyz[:2], axis=1)
        xy_query = np.column_stack([pts_xyz[:, 1], pts_xyz[:, 0]])
        mu_f = np.asarray(mu_f_interp(xy_query), dtype=float)
        sigma_f = np.clip(np.asarray(sigma_f_interp(xy_query), dtype=float), 1e-6, 10.0)
        rho_mean = np.clip(sigmoid(mu_f), 1e-5, 1.0)
        rho_std = np.clip(sigmoid_prime_from_mu(mu_f) * sigma_f, 1e-6, 10.0)
        rho_cons = np.clip(sigmoid(mu_f + beta * sigma_f), 1e-5, 1.0)
        gate_soft = _height_gate(pts_xyz[:, 2], scene["box_max"][2], tau)
        gate_hard = (pts_xyz[:, 2] <= scene["box_max"][2]).astype(float)

        term_mean_soft = ds * gate_soft * rho_mean
        term_unc_soft = ds * gate_soft * rho_cons
        cum_mean_soft = np.cumsum(term_mean_soft)
        cum_unc_soft = np.cumsum(term_unc_soft)
        opacity = float(cum_unc_soft[-1])
        visibility = float(np.clip(np.exp(-opacity), 1e-4, 1.0 - 1e-4))

        x0, x1, y0, y1 = map(float, scene["obstacle_rect"])
        inside_xy = (
            (pts_xyz[:, 0] >= x0)
            & (pts_xyz[:, 0] <= x1)
            & (pts_xyz[:, 1] >= y0)
            & (pts_xyz[:, 1] <= y1)
        )
        return {
            "target_xyz": target,
            "pts_xyz": pts_xyz,
            "arc": arc,
            "horiz": horiz,
            "ds": ds,
            "mu_f": mu_f,
            "sigma_f": sigma_f,
            "rho_mean": rho_mean,
            "rho_std": rho_std,
            "rho_cons": rho_cons,
            "gate_soft": gate_soft,
            "gate_hard": gate_hard,
            "term_mean_soft": term_mean_soft,
            "term_unc_soft": term_unc_soft,
            "cum_mean_soft": cum_mean_soft,
            "cum_unc_soft": cum_unc_soft,
            "vis_mean_soft": np.exp(-cum_mean_soft),
            "vis_unc_soft": np.exp(-cum_unc_soft),
            "opacity": opacity,
            "visibility": visibility,
            "inside_xy": inside_xy,
            "beta": beta,
            "tau": tau,
        }

    A0_map = np.zeros_like(scene["Xg"], dtype=float)
    O0_map = np.zeros_like(scene["Xg"], dtype=float)
    flat_states = np.asarray(scene["state_grid"], dtype=float).reshape(-1, 3)
    desc = f"Precomputing prior beta={beta_override:.1f}: {scene['variant_name']}"
    for idx, state in enumerate(tqdm(flat_states, desc=desc)):
        details = ray_details_for_state(state)
        iy, ix = np.unravel_index(idx, scene["Xg"].shape)
        A0_map[iy, ix] = details["opacity"]
        O0_map[iy, ix] = details["visibility"]

    return {
        "beta": beta,
        "height_tau": tau,
        "A0_map": A0_map,
        "O0_map": O0_map,
        "A0_interp": RegularGridInterpolator((scene["ys"], scene["xs"]), A0_map, bounds_error=False, fill_value=float(np.nanmax(A0_map))),
        "O0_interp": RegularGridInterpolator((scene["ys"], scene["xs"]), O0_map, bounds_error=False, fill_value=1e-4),
        "ray_details_for_state": ray_details_for_state,
    }

# -----------------------------------------------------------------------------
# Compact planning and online correction
# -----------------------------------------------------------------------------



def _make_planning_context(scene: Mapping[str, object], g_np, g_jax, cfg: PlanningConfig):
    dt = cfg.dt
    horizon = cfg.horizon
    len_trial = cfg.n_steps + 1
    Q = process_covariance_from_rho(dt=dt, rho_xy=cfg.q_rho_xy)
    Q_exec = np.diag([cfg.q_exec_xy ** 2, cfg.q_exec_xy ** 2, cfg.q_exec_theta ** 2])
    R_visible = observation_covariance("uv", uv_std=cfg.r_visible_uv)
    R_miss = observation_covariance("uv", uv_std=cfg.r_miss_uv)

    def smoothstep_np(x):
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def smoothstep_jax(x):
        x = jnp.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def goal_cov_for_progress(progress):
        progress_fast = np.clip(float(progress), 0.0, 1.0) ** cfg.goal_tightening_power
        a = smoothstep_np(progress_fast)
        sigma_u = (1.0 - a) * cfg.goal_prior_u_std_start + a * cfg.goal_prior_u_std_final
        sigma_v = (1.0 - a) * cfg.goal_prior_v_std_start + a * cfg.goal_prior_v_std_final
        return np.diag([sigma_u ** 2, sigma_v ** 2])

    def goal_cov_jax_for_progress(progress):
        progress_fast = jnp.clip(progress, 0.0, 1.0) ** cfg.goal_tightening_power
        a = smoothstep_jax(progress_fast)
        sigma_u = (1.0 - a) * cfg.goal_prior_u_std_start + a * cfg.goal_prior_u_std_final
        sigma_v = (1.0 - a) * cfg.goal_prior_v_std_start + a * cfg.goal_prior_v_std_final
        return jnp.diag(jnp.array([sigma_u ** 2, sigma_v ** 2], dtype=jnp.float64))

    def goal_cov_for_step(k):
        return goal_cov_for_progress(k / max(len_trial - 1, 1))

    goal_obs = g_np(scene["goal_state"])
    goal_obs_cov0 = goal_cov_for_step(0)
    mean0 = np.asarray(scene["start_state"], dtype=float).copy()
    cov0 = np.diag([cfg.mean0_xy_std ** 2, cfg.mean0_xy_std ** 2, cfg.mean0_theta_std ** 2])

    agent = UnicycleEFEAgent(
        goal=(goal_obs, goal_obs_cov0),
        g=g_np,
        Q=Q,
        R=R_visible,
        eta=cfg.eta,
        dt=dt,
        time_horizon=horizon,
    )
    params_j = bind_unicycle_agent_jax(agent)
    bounds = [cfg.v_lims, cfg.w_lims] * horizon
    R_visible_j = jnp.asarray(R_visible, dtype=jnp.float64)
    R_miss_j = jnp.asarray(R_miss, dtype=jnp.float64)
    goal_mu_j = jnp.asarray(goal_obs, dtype=jnp.float64)
    xs_j = jnp.asarray(scene["xs"], dtype=jnp.float64)
    ys_j = jnp.asarray(scene["ys"], dtype=jnp.float64)
    visibility_sigma_kappa = float(cfg.visibility_sigma_kappa)

    def observation_moments_np(mean, cov, R_eff, approx="ET2"):
        if approx == "ET1":
            return ET1(mean, cov, g_np, addmatrix=R_eff, forceHermitian=True)
        if approx == "ET2":
            return ET2(mean, cov, g_np, addmatrix=R_eff, forceHermitian=True)
        raise ValueError("approx must be 'ET1' or 'ET2'")

    def correct_with_R(y_k, m_pred, S_pred, R_eff, approx="ET2"):
        mu, Sigma, Gamma = observation_moments_np(m_pred, S_pred, R_eff, approx=approx)
        Sigma = 0.5 * (Sigma + Sigma.T) + 1e-9 * np.eye(Sigma.shape[0])
        K_gain = Gamma @ np.linalg.solve(Sigma, np.eye(Sigma.shape[0]))
        m_k = m_pred + K_gain @ (y_k - mu)
        m_k[2] = wrap_angle(m_k[2])
        S_k = S_pred - Gamma @ np.linalg.solve(Sigma, Gamma.T)
        S_k = 0.5 * (S_k + S_k.T) + 1e-9 * np.eye(S_k.shape[0])
        return m_k, S_k

    def evidence_with_R(y_k, m_k, S_k, R_eff, approx="ET2"):
        mu, Sigma, _ = observation_moments_np(m_k, S_k, R_eff, approx=approx)
        Sigma = 0.5 * (Sigma + Sigma.T) + 1e-9 * np.eye(Sigma.shape[0])
        return -multivariate_normal.logpdf(y_k, mu, Sigma)

    def observation_moments_jax(mean, cov, R_eff, approx="ET2"):
        Jm = jax.jacfwd(g_jax)(mean)
        if approx == "ET1":
            mu = g_jax(mean)
            Sigma = Jm @ cov @ Jm.T + R_eff
            Gamma = cov @ Jm.T
            return mu, Sigma, Gamma
        if approx == "ET2":
            H = jax.jacfwd(jax.jacrev(g_jax))(mean)
            aux1 = jnp.array([jnp.trace(H[i] @ cov) for i in range(H.shape[0])], dtype=jnp.float64)
            aux2 = jnp.array(
                [[jnp.trace(H[i] @ cov @ H[j] @ cov) for j in range(H.shape[0])] for i in range(H.shape[0])],
                dtype=jnp.float64,
            )
            mu = g_jax(mean) + 0.5 * aux1
            Sigma = Jm @ cov @ Jm.T + 0.5 * aux2 + R_eff
            Gamma = cov @ Jm.T
            return mu, Sigma, Gamma
        raise ValueError("approx must be 'ET1' or 'ET2'")

    def risk_goal_jax(mu, Sigma, goal_S_current):
        eps = 1e-9
        L0 = jnp.linalg.cholesky(Sigma + eps * jnp.eye(Sigma.shape[0], dtype=Sigma.dtype))
        L1 = jnp.linalg.cholesky(goal_S_current + eps * jnp.eye(goal_S_current.shape[0], dtype=goal_S_current.dtype))
        M = jnp.linalg.solve(L1, L0)
        y = jnp.linalg.solve(L1, goal_mu_j - mu)
        d = goal_mu_j.shape[0]
        return 0.5 * (
            jnp.sum(M ** 2) - d + jnp.sum(y ** 2) + 2.0 * jnp.sum(jnp.log(jnp.diag(L1) / jnp.diag(L0)))
        )

    def ambiguity_jax_local(Sigma, Gamma, S):
        Sigma_cond = Sigma - Gamma.T @ jnp.linalg.solve(S, Gamma)
        _, logdet = jnp.linalg.slogdet(Sigma_cond)
        d = Sigma_cond.shape[0]
        return 0.5 * (d * jnp.log(2.0 * jnp.pi * jnp.e) + logdet)

    def bilinear_field_jax(xy, xs_grid, ys_grid, field):
        x = jnp.clip(xy[0], xs_grid[0], xs_grid[-1] - 1e-9)
        y = jnp.clip(xy[1], ys_grid[0], ys_grid[-1] - 1e-9)
        ix = jnp.clip(jnp.searchsorted(xs_grid, x, side="right") - 1, 0, xs_grid.shape[0] - 2)
        iy = jnp.clip(jnp.searchsorted(ys_grid, y, side="right") - 1, 0, ys_grid.shape[0] - 2)
        x0, x1 = xs_grid[ix], xs_grid[ix + 1]
        y0, y1 = ys_grid[iy], ys_grid[iy + 1]
        tx = jnp.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
        ty = jnp.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))
        tx = jnp.clip(tx, 0.0, 1.0)
        ty = jnp.clip(ty, 0.0, 1.0)
        z00 = field[iy, ix]
        z10 = field[iy, ix + 1]
        z01 = field[iy + 1, ix]
        z11 = field[iy + 1, ix + 1]
        return (1.0 - ty) * ((1.0 - tx) * z00 + tx * z10) + ty * ((1.0 - tx) * z01 + tx * z11)

    def expected_visibility_np(interp, mean, cov, lo=1e-4, hi=1.0 - 1e-4):
        sigma_points_xy, weights = _xy_visibility_sigma_points_np(mean[:2], cov[:2, :2], kappa=visibility_sigma_kappa)
        query = np.column_stack([sigma_points_xy[:, 1], sigma_points_xy[:, 0]])
        vals = np.clip(np.asarray(interp(query), dtype=float), lo, hi)
        return float(np.clip(np.sum(weights * vals), lo, hi))

    def expected_visibility_jax(mean, cov, field):
        sigma_points_xy, weights = _xy_visibility_sigma_points_jax(mean[:2], cov[:2, :2], kappa=visibility_sigma_kappa)
        vals = jax.vmap(lambda xy: jnp.clip(bilinear_field_jax(xy, xs_j, ys_j, field), 1e-4, 1.0 - 1e-4))(sigma_points_xy)
        return jnp.clip(jnp.sum(weights * vals), 1e-4, 1.0 - 1e-4)

    trial_denom = float(max(len_trial - 1, 1))

    def make_map_valgrad_fn(approx="ET2"):
        def efe_sd(u, m0, S0, k0, p_vis_map):
            u = jnp.asarray(u, dtype=jnp.float64)
            if u.ndim == 1:
                u = u.reshape((horizon, 2))
            p_vis_map = jnp.asarray(p_vis_map, dtype=jnp.float64)
            m = m0
            S = S0
            cost = 0.0

            for t in range(horizon):
                m = unicycle_step_jax(m, u[t], params_j.dt)
                F = unicycle_jacobian_jax(m, u[t], params_j.dt)
                S = F @ S @ F.T + params_j.Q
                p_vis = expected_visibility_jax(m, S, p_vis_map)
                p_vis_eff = p_vis ** cfg.visibility_power
                R_plan = p_vis_eff * R_visible_j + (1.0 - p_vis_eff) * R_miss_j
                mu, Sigma, Gamma = observation_moments_jax(m, S, R_plan, approx=approx)
                progress_t = (k0 + t) / trial_denom
                goal_S_t = goal_cov_jax_for_progress(progress_t)
                weight_t = cfg.discount_gamma ** t
                cost += weight_t * (
                    cfg.risk_scale * risk_goal_jax(mu, Sigma, goal_S_t)
                    + params_j.eta * jnp.sum(u[t] ** 2)
                    + cfg.ambiguity_scale * ambiguity_jax_local(Sigma, Gamma, S)
                )
            return cost

        def valgrad(u, m, S, k0, p_vis_map):
            val = efe_sd(u, m, S, k0, p_vis_map)
            grad = jax.jacfwd(lambda uu: efe_sd(uu, m, S, k0, p_vis_map))(u)
            return val, grad

        return jax.jit(valgrad)

    return {
        "cfg": cfg,
        "g_np": g_np,
        "g_jax": g_jax,
        "Q": Q,
        "Q_exec": Q_exec,
        "R_visible": R_visible,
        "R_miss": R_miss,
        "goal_obs": goal_obs,
        "goal_cov_for_step": goal_cov_for_step,
        "mean0": mean0,
        "cov0": cov0,
        "agent": agent,
        "bounds": bounds,
        "correct_with_R": correct_with_R,
        "evidence_with_R": evidence_with_R,
        "observation_moments_np": observation_moments_np,
        "expected_visibility_np": expected_visibility_np,
        "map_valgrad_cache": {
            "ET1": make_map_valgrad_fn("ET1"),
            "ET2": make_map_valgrad_fn("ET2"),
        },
    }

def make_noise_pack(seed, plan_ctx: Mapping[str, object]):
    cfg = plan_ctx["cfg"]
    len_trial = cfg.n_steps + 1
    rng = np.random.default_rng(int(seed))
    return {
        "w_exec": rng.multivariate_normal(np.zeros(3), plan_ctx["Q_exec"], size=len_trial),
        "u_detect": rng.random(len_trial),
        "meas_eps": rng.standard_normal((len_trial, 2)),
    }


def map_lookup_np(interp: RegularGridInterpolator, state: Sequence[float], lo=1e-4, hi=1.0 - 1e-4):
    return float(np.clip(interp([[float(state[1]), float(state[0])]])[0], lo, hi))



def _make_opacity_correction_state(base_A_map: np.ndarray, base_O_map: np.ndarray, prior_strength=6.0):
    prior_strength = float(prior_strength)
    base_A_map = np.asarray(base_A_map, dtype=float)
    base_O_map = np.clip(np.asarray(base_O_map, dtype=float), 1e-4, 1.0 - 1e-4)
    alpha_prior = prior_strength * base_O_map
    beta_prior = prior_strength * (1.0 - base_O_map)
    return {
        "A_base_map": base_A_map,
        "O_base_map": base_O_map,
        "A_map": base_A_map.copy(),
        "O_map": base_O_map.copy(),
        "delta_A_map": np.zeros_like(base_A_map, dtype=float),
        "N_seen": np.zeros_like(base_A_map, dtype=float),
        "N_detect": np.zeros_like(base_A_map, dtype=float),
        "alpha_prior": alpha_prior,
        "beta_prior": beta_prior,
        "confidence_map": np.zeros_like(base_A_map, dtype=float),
        "p_calib_map": base_O_map.copy(),
        "prior_strength": prior_strength,
    }


def _update_opacity_correction(
    correction_state: Mapping[str, np.ndarray],
    *,
    sigma=1.0,
    delta_clip=(-3.0, 3.0),
):
    prior_mass = np.maximum(correction_state["alpha_prior"] + correction_state["beta_prior"], 1e-6)
    total_mass = np.maximum(correction_state["N_seen"] + prior_mass, 1e-6)
    p_emp = np.clip(
        (correction_state["N_detect"] + correction_state["alpha_prior"]) / total_mass,
        1e-4,
        1.0 - 1e-4,
    )
    A_emp = -np.log(p_emp)
    confidence = correction_state["N_seen"] / total_mass
    A_blend = (1.0 - confidence) * correction_state["A_base_map"] + confidence * A_emp
    delta_A = np.clip(A_blend - correction_state["A_base_map"], float(delta_clip[0]), float(delta_clip[1]))
    delta_A = gaussian_filter(delta_A, sigma=sigma, mode="nearest")
    A_map = np.clip(correction_state["A_base_map"] + delta_A, 1e-4, 12.0)
    O_map = np.clip(np.exp(-A_map), 1e-4, 1.0 - 1e-4)
    correction_state["p_calib_map"] = p_emp
    correction_state["confidence_map"] = confidence
    correction_state["delta_A_map"] = delta_A
    correction_state["A_map"] = A_map
    correction_state["O_map"] = O_map

def _run_story_experiment(
    variant_ctx: Mapping[str, object],
    label: str,
    planner_map: np.ndarray,
    *,
    base_A_map: np.ndarray | None = None,
    correction=False,
    approx="ET2",
    noise_pack=None,
    correction_update_every=5,
    correction_apply_online=True,
):
    scene = variant_ctx["scene"]
    plan_ctx = variant_ctx["planning"]
    oracle_p_vis = variant_ctx["oracle_p_vis"]

    cfg = plan_ctx["cfg"]
    len_trial = cfg.n_steps + 1
    g_np = plan_ctx["g_np"]
    R_visible = plan_ctx["R_visible"]
    R_miss = plan_ctx["R_miss"]
    mean0 = plan_ctx["mean0"]
    cov0 = plan_ctx["cov0"]
    agent = plan_ctx["agent"]
    bounds = plan_ctx["bounds"]
    correct_with_R = plan_ctx["correct_with_R"]
    evidence_with_R = plan_ctx["evidence_with_R"]
    observation_moments_np = plan_ctx["observation_moments_np"]
    expected_visibility_np = plan_ctx["expected_visibility_np"]
    goal_cov_for_step = plan_ctx["goal_cov_for_step"]
    goal_obs = plan_ctx["goal_obs"]
    valgrad = plan_ctx["map_valgrad_cache"][approx]

    if noise_pack is None:
        noise_pack = make_noise_pack(20260323, plan_ctx)

    current_planner_map = np.asarray(planner_map, dtype=float).copy()
    p_vis_interp = RegularGridInterpolator((scene["ys"], scene["xs"]), current_planner_map, bounds_error=False, fill_value=1e-4)

    correction_state = None
    if correction:
        if base_A_map is None:
            raise ValueError("base_A_map is required when correction=True")
        correction_state = _make_opacity_correction_state(base_A_map, planner_map)

    z_sim = np.zeros((3, len_trial))
    y_sim = np.full((agent.Dy, len_trial), np.nan)
    u_sim = np.zeros((2, len_trial))
    z_est_mean = np.zeros((3, len_trial))
    z_est_cov = np.zeros((3, 3, len_trial))
    risk_runtime = np.zeros(len_trial)
    ambiguity_runtime = np.zeros(len_trial)
    p_vis_exec_runtime = np.ones(len_trial)
    p_vis_plan_runtime = np.ones(len_trial)
    p_vis_plan_effective_runtime = np.ones(len_trial)
    detections = np.ones(len_trial, dtype=bool)
    optimizer_values = np.zeros(len_trial)
    F_runtime = np.full(len_trial, np.nan)
    planned_paths = []

    z_sim[:, 0] = scene["start_state"]
    y_sim[:, 0] = g_np(scene["start_state"])
    z_est_mean[:, 0] = mean0
    z_est_cov[:, :, 0] = cov0

    m_kmin1 = mean0.copy()
    S_kmin1 = cov0.copy()
    policy = np.zeros((2, cfg.horizon))

    for k in range(1, len_trial):
        z_sim[:, k] = unicycle_step(z_sim[:, k - 1], u_sim[:, k - 1], cfg.dt)
        z_sim[:, k] = z_sim[:, k] + noise_pack["w_exec"][k]
        z_sim[2, k] = wrap_angle(z_sim[2, k])

        p_vis_exec = float(np.clip(oracle_p_vis(z_sim[:, k]), 1e-4, 1.0))
        p_detect = p_vis_exec ** cfg.detection_power
        detected = bool(noise_pack["u_detect"][k] < p_detect)

        m_pred, S_pred = predict_unicycle(agent, m_kmin1, S_kmin1, u_sim[:, k - 1])
        if detected:
            y_mean = np.asarray(g_np(z_sim[:, k]), dtype=float)
            L_exec = np.linalg.cholesky(R_visible)
            y_sim[:, k] = y_mean + L_exec @ noise_pack["meas_eps"][k]
            m_k, S_k = correct_with_R(y_sim[:, k], m_pred, S_pred, R_visible, approx=approx)
            F_runtime[k] = evidence_with_R(y_sim[:, k], m_pred, S_pred, R_visible, approx=approx)
        else:
            m_k, S_k = m_pred, S_pred

        z_est_mean[:, k] = m_k
        z_est_cov[:, :, k] = S_k

        if correction and correction_state is not None:
            iy = int(np.searchsorted(scene["ys"], m_k[1], side="right") - 1)
            ix = int(np.searchsorted(scene["xs"], m_k[0], side="right") - 1)
            iy = max(0, min(iy, correction_state["N_seen"].shape[0] - 1))
            ix = max(0, min(ix, correction_state["N_seen"].shape[1] - 1))
            correction_state["N_seen"][iy, ix] += 1.0
            if detected:
                correction_state["N_detect"][iy, ix] += 1.0
            if (k % correction_update_every) == 0 or k == len_trial - 1:
                _update_opacity_correction(correction_state)
                if correction_apply_online:
                    current_planner_map = correction_state["O_map"]
                    p_vis_interp = RegularGridInterpolator(
                        (scene["ys"], scene["xs"]),
                        current_planner_map,
                        bounds_error=False,
                        fill_value=1e-4,
                    )

        p_vis_plan = expected_visibility_np(p_vis_interp, m_k, S_k)
        p_vis_plan_eff = float(np.clip(p_vis_plan ** cfg.visibility_power, 1e-4, 1.0 - 1e-4))
        R_plan_np = p_vis_plan_eff * R_visible + (1.0 - p_vis_plan_eff) * R_miss
        mu_y, Sigma_y, Gamma = observation_moments_np(m_k, S_k, R_plan_np, approx=approx)

        p_vis_exec_runtime[k] = p_vis_exec
        p_vis_plan_runtime[k] = p_vis_plan
        p_vis_plan_effective_runtime[k] = p_vis_plan_eff
        detections[k] = detected
        goal_cov_k = goal_cov_for_step(k)
        risk_runtime[k] = float(risk(mu_y, Sigma_y, (goal_obs, goal_cov_k)))
        ambiguity_runtime[k] = float(ambiguity(Sigma_y, Gamma, S_k))

        k0 = jnp.asarray(float(k), dtype=jnp.float64)
        p_vis_map_j = jnp.asarray(current_planner_map, dtype=jnp.float64)

        def f(u):
            val, _ = valgrad(jnp.asarray(u), jnp.asarray(m_k), jnp.asarray(S_k), k0, p_vis_map_j)
            return float(val)

        def grad_u(u):
            _, grad = valgrad(jnp.asarray(u), jnp.asarray(m_k), jnp.asarray(S_k), k0, p_vis_map_j)
            return np.asarray(grad, dtype=float)

        x0 = np.zeros(2 * cfg.horizon) if k == 1 else np.concatenate([policy[:, 1:].T.reshape(-1), policy[:, -1]])
        result = minimize(
            f,
            x0,
            jac=grad_u,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": cfg.optimizer_maxiter, "maxfun": cfg.optimizer_maxfun, "ftol": cfg.optimizer_ftol},
        )

        optimizer_values[k] = float(result.fun)
        policy = np.asarray(result.x, dtype=float).reshape((cfg.horizon, 2)).T
        u_sim[:, k] = policy[:, 0]
        planned_states, _ = planned_trajectory_unicycle(agent, policy, (m_k, S_k), approx=approx)
        planned_paths.append(planned_states[0].T.copy())

        m_kmin1 = m_k
        S_kmin1 = S_k

    return {
        "label": label,
        "approx": approx,
        "true_states": z_sim.T,
        "means": z_est_mean.T,
        "covs": np.moveaxis(z_est_cov, 2, 0),
        "measurements": y_sim.T,
        "chosen_controls": u_sim.T,
        "planned_paths": planned_paths,
        "p_vis_exec_runtime": p_vis_exec_runtime,
        "p_vis_plan_runtime": p_vis_plan_runtime,
        "p_vis_plan_effective_runtime": p_vis_plan_effective_runtime,
        "detections": detections,
        "risk_runtime": risk_runtime,
        "ambiguity_runtime": ambiguity_runtime,
        "log_evidence_runtime": F_runtime,
        "optimizer_values": optimizer_values,
        "correction_final_map": None if correction_state is None else correction_state["O_map"],
        "correction_delta_A_map": None if correction_state is None else correction_state["delta_A_map"],
        "correction_A_map": None if correction_state is None else correction_state["A_map"],
    }


# -----------------------------------------------------------------------------
# Public notebook API
# -----------------------------------------------------------------------------


def build_context(seed=12, run_mode="full"):
    settings = _run_mode_settings(run_mode)
    scenes = {name: build_variant_scene(name, run_mode=run_mode) for name in _variant_specs()}
    ref_scene = scenes["Tall box"]
    shared_gp = build_shared_gp_field(ref_scene, seed=seed)
    old_gp = build_old_gp_prior(ref_scene, seed=seed)

    variants = {}
    probe_frames = []
    for name, scene in scenes.items():
        g_np, g_jax = make_observation_fns(scene)
        oracle_map = evaluate_state_fn_on_grid(scene, lambda s: _oracle_p_vis(scene, s))
        prior_mean = build_height_aware_prior(scene, shared_gp, beta_override=0.0)
        prior_unc = build_height_aware_prior(scene, shared_gp, beta_override=shared_gp["cfg"].beta)
        planning_cfg = replace(
            PlanningConfig(),
            horizon=settings["horizon"],
            n_steps=settings["n_steps"],
            optimizer_maxiter=settings["optimizer_maxiter"],
            optimizer_maxfun=settings["optimizer_maxfun"],
        )
        planning = _make_planning_context(scene, g_np, g_jax, planning_cfg)

        rows = []
        for probe_name in ("start", "pre-occ", "shadow", "deep-shadow", "goal"):
            state = _probe_state(scene, probe_name)
            rows.append(
                {
                    "scene": name,
                    "probe": probe_name,
                    "oracle": _oracle_p_vis(scene, state),
                    "old_gp": old_gp["p_vis_old"](state),
                    "new_mean_only": map_lookup_np(prior_mean["O0_interp"], state),
                    "new_uncertainty_aware": map_lookup_np(prior_unc["O0_interp"], state),
                    "opacity_uncertainty_aware": float(prior_unc["A0_interp"]([[float(state[1]), float(state[0])]])[0]),
                }
            )

        probe_df = pd.DataFrame(rows)
        probe_frames.append(probe_df)
        variants[name] = {
            "scene": scene,
            "oracle_map": oracle_map,
            "oracle_p_vis": lambda state, _scene=scene: _oracle_p_vis(_scene, state),
            "prior_mean_only": prior_mean,
            "prior_uncertainty": prior_unc,
            "planning": planning,
        }

    return {
        "shared_gp": shared_gp,
        "old_gp": old_gp,
        "variants": variants,
        "probe_summary_df": pd.concat(probe_frames, ignore_index=True),
    }


def run_story_experiments(ctx: Mapping[str, object], base_seed=20260323):
    out = {}
    for idx, (name, variant) in enumerate(ctx["variants"].items()):
        noise_eval = make_noise_pack(base_seed + 10 * idx, variant["planning"])
        noise_calibration = make_noise_pack(base_seed + 10 * idx + 1, variant["planning"])
        prior = variant["prior_uncertainty"]
        calibration_res = _run_story_experiment(
            variant,
            f"{name} | correction calibration",
            prior["O0_map"],
            base_A_map=prior["A0_map"],
            correction=True,
            correction_apply_online=False,
            approx="ET2",
            noise_pack=noise_calibration,
        )
        corrected_eval = _run_story_experiment(
            variant,
            f"{name} | corrected prior",
            calibration_res["correction_final_map"],
            correction=False,
            approx="ET2",
            noise_pack=noise_eval,
        )
        corrected_eval["correction_final_map"] = calibration_res["correction_final_map"]
        corrected_eval["correction_delta_A_map"] = calibration_res["correction_delta_A_map"]
        corrected_eval["correction_A_map"] = calibration_res["correction_A_map"]
        out[name] = {
            "old 2D GP": _run_story_experiment(
                variant,
                f"{name} | old 2D GP",
                ctx["old_gp"]["P_vis_old_map"],
                correction=False,
                approx="ET2",
                noise_pack=noise_eval,
            ),
            "offline prior": _run_story_experiment(
                variant,
                f"{name} | offline prior",
                prior["O0_map"],
                correction=False,
                approx="ET2",
                noise_pack=noise_eval,
            ),
            "corrected prior": corrected_eval,
        }
    return out


def summarize_story_results(ctx: Mapping[str, object], story_results: Mapping[str, Mapping[str, Mapping[str, object]]]):
    rows = []
    for scene_name, methods in story_results.items():
        goal = ctx["variants"][scene_name]["scene"]["goal_state"][:2]
        for method in STORY_METHODS:
            res = methods[method]
            state_err = np.linalg.norm(res["true_states"][:, :2] - res["means"][:, :2], axis=1)
            rows.append(
                {
                    "scene": scene_name,
                    "method": method,
                    "final_true_goal_dist": float(np.linalg.norm(res["true_states"][-1, :2] - goal)),
                    "mean_state_error": float(np.mean(state_err)),
                    "detection_rate": float(np.mean(res["detections"][1:])),
                    "mean_p_vis_exec": float(np.mean(res["p_vis_exec_runtime"][1:])),
                    "mean_p_vis_plan": float(np.mean(res["p_vis_plan_effective_runtime"][1:])),
                    "cum_risk": float(np.sum(res["risk_runtime"])),
                    "cum_ambiguity": float(np.sum(res["ambiguity_runtime"])),
                    "final_trace": float(covariance_trace_series(res["covs"])[-1]),
                }
            )
    return pd.DataFrame(rows).sort_values(["scene", "method"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def plot_scene_overview(ctx: Mapping[str, object]):
    names = list(ctx["variants"].keys())
    fig, axes = plt.subplots(1, len(names), figsize=(14, 5), constrained_layout=True, sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]
    probe_specs = [
        ("pre_occlusion_state", "tab:blue"),
        ("shadow_state", "tab:red"),
        ("deep_shadow_state", "tab:purple"),
    ]
    for ax, name in zip(axes, names):
        variant = ctx["variants"][name]
        scene = variant["scene"]
        vis_extent = [scene["xs"].min(), scene["xs"].max(), scene["ys"].min(), scene["ys"].max()]
        ax.imshow(variant["oracle_map"], extent=vis_extent, origin="lower", cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0, alpha=0.9)
        ax.contour(scene["true_grid"].xs, scene["true_grid"].ys, scene["true_grid"].occupancy, levels=[0.5], colors="white", linewidths=1.4)
        ax.scatter(scene["camera_xy"][0], scene["camera_xy"][1], marker="^", s=85, color="white", edgecolor="black", label="camera")
        ax.scatter(scene["start_state"][0], scene["start_state"][1], s=70, color="white", edgecolor="black", label="start")
        ax.scatter(scene["goal_state"][0], scene["goal_state"][1], marker="*", s=170, color="tab:red", label="goal")
        for key, color in probe_specs:
            state = scene[key]
            label = key.replace("_state", "").replace("_", "-")
            ax.scatter(state[0], state[1], s=55, color=color, edgecolor="black", label=label)
        ax.set_title(f"{name}: scene and probe states")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axes[0].legend(loc="lower left", fontsize=8)
    return fig



def plot_shared_gp_field(ctx: Mapping[str, object]):
    ref_scene = ctx["variants"]["Tall box"]["scene"]
    gp = ctx["shared_gp"]
    vis_extent = [ref_scene["xs"].min(), ref_scene["xs"].max(), ref_scene["ys"].min(), ref_scene["ys"].max()]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True, sharex=True, sharey=True)
    ax = axes[0, 0]
    ax.imshow(ref_scene["true_grid"].occupancy, extent=ref_scene["true_grid"].extent, origin="lower", cmap="Greys", aspect="equal", vmin=0.0, vmax=1.0)
    dset = gp["dataset"]
    if dset["clear_points"].size:
        ax.scatter(dset["clear_points"][:, 0], dset["clear_points"][:, 1], s=5, color="tab:blue", alpha=0.18, label="clear")
    if dset["transition_points"].size:
        ax.scatter(dset["transition_points"][:, 0], dset["transition_points"][:, 1], s=8, color="tab:orange", alpha=0.35, label="transition")
    if dset["occ_points"].size:
        ax.scatter(dset["occ_points"][:, 0], dset["occ_points"][:, 1], s=8, color="tab:red", alpha=0.40, label="occupied")
    ax.set_title("Shared 2D training labels")
    ax.legend(loc="lower left", fontsize=7)

    panels = [
        (gp["mu_f_map"], r"Latent GP mean $\mu_f$", "coolwarm"),
        (gp["sigma_f_map"], r"Latent GP std $\sigma_f$", "cividis"),
        (gp["rho_mean_map"], r"Opacity density mean $\rho$", "viridis"),
        (gp["rho_conservative_map"], r"Conservative opacity $\rho_{cons}$", "magma"),
        (ctx["old_gp"]["P_vis_old_map"], "Old 2D GP visibility", "viridis"),
    ]
    for ax, (field, title, cmap) in zip(axes.ravel()[1:], panels):
        im = ax.imshow(field, extent=vis_extent, origin="lower", cmap=cmap, aspect="equal")
        ax.contour(ref_scene["true_grid"].xs, ref_scene["true_grid"].ys, ref_scene["true_grid"].occupancy, levels=[0.5], colors="white", linewidths=1.2)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def plot_raycasting_probe(ctx: Mapping[str, object], probe_name="shadow"):
    names = list(ctx["variants"].keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 5 * len(names)), constrained_layout=True)
    if len(names) == 1:
        axes = np.asarray(axes).reshape(1, 2)
    for row, name in enumerate(names):
        variant = ctx["variants"][name]
        scene = variant["scene"]
        state = _probe_state(scene, probe_name)
        details = variant["prior_uncertainty"]["ray_details_for_state"](state)
        target = details["target_xyz"]
        cam_xyz = np.asarray(scene["camera_3d"].cam_pos, dtype=float)

        ax = axes[row, 0]
        ax.imshow(variant["oracle_map"], extent=[scene["xs"].min(), scene["xs"].max(), scene["ys"].min(), scene["ys"].max()], origin="lower", cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0, alpha=0.9)
        ax.contour(scene["true_grid"].xs, scene["true_grid"].ys, scene["true_grid"].occupancy, levels=[0.5], colors="white", linewidths=1.4)
        ax.plot(details["pts_xyz"][:, 0], details["pts_xyz"][:, 1], color="tab:red", linewidth=2.0, label="ray")
        if np.any(details["inside_xy"]):
            ax.plot(details["pts_xyz"][details["inside_xy"], 0], details["pts_xyz"][details["inside_xy"], 1], color="cyan", linewidth=3.2, label="inside footprint")
        ax.scatter(cam_xyz[0], cam_xyz[1], marker="^", s=90, color="white", edgecolor="black", zorder=6)
        ax.scatter(target[0], target[1], s=70, color="tab:red", edgecolor="black", zorder=6)
        ax.set_title(f"{name}: top-down ray to {probe_name}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=8, loc="lower left")

        ax = axes[row, 1]
        ax.plot(details["horiz"], details["pts_xyz"][:, 2], color="tab:red", linewidth=2.2, label="camera-target ray")
        ax.fill_between(details["horiz"], 0.0, scene["box_max"][2], where=details["inside_xy"], color="tab:blue", alpha=0.35, label="box height under ray")
        ax.plot(details["horiz"], scene["box_max"][2] * details["gate_soft"], color="tab:green", linestyle="--", linewidth=2.0, label="height gate profile")
        ax.scatter([0.0], [cam_xyz[2]], marker="^", s=90, color="black")
        ax.scatter([details["horiz"][-1]], [target[2]], s=70, color="tab:red", edgecolor="black")
        ax.set_title(f"{name}: side view of the same ray")
        ax.set_xlabel("horizontal distance from camera [m]")
        ax.set_ylabel("z [m]")
        ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_ray_decomposition(ctx: Mapping[str, object], variant_name="Tall box", probe_name="shadow"):
    variant = ctx["variants"][variant_name]
    scene = variant["scene"]
    state = _probe_state(scene, probe_name)
    mean_details = variant["prior_mean_only"]["ray_details_for_state"](state)
    unc_details = variant["prior_uncertainty"]["ray_details_for_state"](state)
    cam_xyz = np.asarray(scene["camera_3d"].cam_pos, dtype=float)
    vis_extent = [scene["xs"].min(), scene["xs"].max(), scene["ys"].min(), scene["ys"].max()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(variant["oracle_map"], extent=vis_extent, origin="lower", cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0, alpha=0.9)
    ax.contour(scene["true_grid"].xs, scene["true_grid"].ys, scene["true_grid"].occupancy, levels=[0.5], colors="white", linewidths=1.4)
    ax.plot(unc_details["pts_xyz"][:, 0], unc_details["pts_xyz"][:, 1], color="tab:red", linewidth=2.0, label="ray")
    if np.any(unc_details["inside_xy"]):
        ax.plot(unc_details["pts_xyz"][unc_details["inside_xy"], 0], unc_details["pts_xyz"][unc_details["inside_xy"], 1], color="cyan", linewidth=3.0, label="inside footprint")
    ax.scatter(scene["camera_xy"][0], scene["camera_xy"][1], marker="^", s=90, color="white", edgecolor="black")
    ax.scatter(unc_details["target_xyz"][0], unc_details["target_xyz"][1], s=70, color="tab:red", edgecolor="black")
    ax.set_title(f"{variant_name}: top-down ray to {probe_name}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[0, 1]
    ax.plot(unc_details["horiz"], unc_details["pts_xyz"][:, 2], color="tab:red", linewidth=2.0, label="camera-target ray")
    ax.fill_between(unc_details["horiz"], 0.0, scene["box_max"][2], where=unc_details["inside_xy"], color="tab:blue", alpha=0.35, label="box height under ray")
    ax.plot(unc_details["horiz"], scene["box_max"][2] * unc_details["gate_hard"], color="black", linestyle="--", linewidth=1.8, label="hard gate")
    ax.plot(unc_details["horiz"], scene["box_max"][2] * unc_details["gate_soft"], color="tab:green", linestyle="--", linewidth=2.0, label="soft gate")
    ax.scatter([0.0], [cam_xyz[2]], marker="^", s=90, color="black")
    ax.scatter([unc_details["horiz"][-1]], [unc_details["target_xyz"][2]], s=70, color="tab:red", edgecolor="black")
    ax.set_title("Side view and height gate")
    ax.set_xlabel("horizontal distance from camera [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 2]
    ax.plot(unc_details["arc"], unc_details["rho_mean"], linewidth=2.0, color="tab:blue", label=r"$\rho_{mean}$")
    ax.plot(unc_details["arc"], unc_details["rho_cons"], linewidth=2.0, color="tab:orange", label=r"$\rho_{cons}$")
    ax.plot(unc_details["arc"], unc_details["gate_soft"], linewidth=2.0, color="tab:green", linestyle="--", label="soft gate")
    ax.plot(unc_details["arc"], unc_details["gate_hard"], linewidth=1.8, color="black", linestyle=":", label="hard gate")
    ax.set_title("Per-sample ray quantities")
    ax.set_xlabel("distance along ray [m]")
    ax.set_ylabel("value")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(mean_details["arc"], mean_details["term_mean_soft"], linewidth=2.0, color="tab:blue", label="local opacity: mean only")
    ax.plot(unc_details["arc"], unc_details["term_unc_soft"], linewidth=2.0, color="tab:red", label="local opacity: uncertainty-aware")
    ax.set_title("Local opacity contribution")
    ax.set_xlabel("distance along ray [m]")
    ax.set_ylabel(r"$\Delta A$")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(mean_details["arc"], mean_details["cum_mean_soft"], linewidth=2.0, color="tab:blue", label="cumulative opacity: mean only")
    ax.plot(unc_details["arc"], unc_details["cum_unc_soft"], linewidth=2.0, color="tab:red", label="cumulative opacity: uncertainty-aware")
    ax.set_title("Cumulative opacity")
    ax.set_xlabel("distance along ray [m]")
    ax.set_ylabel(r"$A(s)$")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(mean_details["arc"], mean_details["vis_mean_soft"], linewidth=2.0, color="tab:blue", label="visibility: mean only")
    ax.plot(unc_details["arc"], unc_details["vis_unc_soft"], linewidth=2.0, color="tab:red", label="visibility: uncertainty-aware")
    ax.axhline(_oracle_p_vis(scene, state), color="black", linewidth=1.5, linestyle="--", label="oracle final visibility")
    ax.set_title("Visibility implied by cumulative opacity")
    ax.set_xlabel("distance along ray [m]")
    ax.set_ylabel("visibility")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    return fig

def plot_method_maps(ctx: Mapping[str, object]):
    names = list(ctx["variants"].keys())
    fig, axes = plt.subplots(len(names), 4, figsize=(18, 5 * len(names)), constrained_layout=True, sharex=True, sharey=True)
    if len(names) == 1:
        axes = np.asarray(axes).reshape(1, 4)
    for row, name in enumerate(names):
        variant = ctx["variants"][name]
        scene = variant["scene"]
        vis_extent = [scene["xs"].min(), scene["xs"].max(), scene["ys"].min(), scene["ys"].max()]
        panels = [
            (variant["oracle_map"], f"{name}: oracle", "viridis"),
            (ctx["old_gp"]["P_vis_old_map"], f"{name}: old 2D GP", "viridis"),
            (variant["prior_mean_only"]["O0_map"], f"{name}: new mean-only", "viridis"),
            (variant["prior_uncertainty"]["O0_map"], f"{name}: new uncertainty-aware", "viridis"),
        ]
        for col, (field, title, cmap) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(field, extent=vis_extent, origin="lower", cmap=cmap, aspect="equal", vmin=0.0, vmax=1.0)
            ax.contour(scene["true_grid"].xs, scene["true_grid"].ys, scene["true_grid"].occupancy, levels=[0.5], colors="white", linewidths=1.4)
            ax.scatter(scene["camera_xy"][0], scene["camera_xy"][1], marker="^", s=80, color="white", edgecolor="black")
            ax.scatter(scene["start_state"][0], scene["start_state"][1], s=60, color="white", edgecolor="black")
            ax.scatter(scene["goal_state"][0], scene["goal_state"][1], marker="*", s=160, color="tab:red")
            ax.set_title(title)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def plot_height_difference_maps(ctx: Mapping[str, object]):
    short = ctx["variants"]["Short box"]
    tall = ctx["variants"]["Tall box"]
    scene = tall["scene"]
    vis_extent = [scene["xs"].min(), scene["xs"].max(), scene["ys"].min(), scene["ys"].max()]
    panels = [
        (tall["oracle_map"] - short["oracle_map"], "oracle: tall - short"),
        (ctx["old_gp"]["P_vis_old_map"] - ctx["old_gp"]["P_vis_old_map"], "old 2D GP: tall - short"),
        (tall["prior_mean_only"]["O0_map"] - short["prior_mean_only"]["O0_map"], "new mean-only: tall - short"),
        (tall["prior_uncertainty"]["O0_map"] - short["prior_uncertainty"]["O0_map"], "new uncertainty-aware: tall - short"),
    ]
    lim = max(float(np.max(np.abs(field))) for field, _ in panels)
    lim = max(lim, 1e-6)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True, sharex=True, sharey=True)
    for ax, (field, title) in zip(axes, panels):
        im = ax.imshow(field, extent=vis_extent, origin="lower", cmap="coolwarm", aspect="equal", vmin=-lim, vmax=lim)
        ax.contour(scene["true_grid"].xs, scene["true_grid"].ys, scene["true_grid"].occupancy, levels=[0.5], colors="black", linewidths=1.2)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def plot_probe_corridor(ctx: Mapping[str, object], n_points=80):
    tall_scene = ctx["variants"]["Tall box"]["scene"]
    p0 = np.asarray(tall_scene["pre_occlusion_state"], dtype=float)
    p1 = np.asarray(tall_scene["deep_shadow_state"], dtype=float)
    ts = np.linspace(0.0, 1.0, int(max(n_points, 2)))
    states = np.outer(1.0 - ts, p0) + np.outer(ts, p1)
    progress = np.linspace(0.0, 1.0, states.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    vis_extent = [tall_scene["xs"].min(), tall_scene["xs"].max(), tall_scene["ys"].min(), tall_scene["ys"].max()]
    ax = axes[0, 0]
    ax.imshow(ctx["variants"]["Tall box"]["oracle_map"], extent=vis_extent, origin="lower", cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0, alpha=0.9)
    ax.contour(tall_scene["true_grid"].xs, tall_scene["true_grid"].ys, tall_scene["true_grid"].occupancy, levels=[0.5], colors="white", linewidths=1.4)
    ax.plot(states[:, 0], states[:, 1], color="tab:cyan", linewidth=2.4, label="probe corridor")
    ax.scatter(states[0, 0], states[0, 1], s=60, color="white", edgecolor="black", label="visible end")
    ax.scatter(states[-1, 0], states[-1, 1], s=60, color="tab:red", edgecolor="black", label="shadow end")
    ax.set_title("Probe corridor from visible region into shadow")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="lower left", fontsize=8)

    old_vals = np.asarray([ctx["old_gp"]["p_vis_old"](s) for s in states], dtype=float)
    short = ctx["variants"]["Short box"]
    tall = ctx["variants"]["Tall box"]
    for ax, (name, variant) in zip([axes[0, 1], axes[1, 0]], [("Short box", short), ("Tall box", tall)]):
        oracle = np.asarray([variant["oracle_p_vis"](s) for s in states], dtype=float)
        mean_only = np.asarray([map_lookup_np(variant["prior_mean_only"]["O0_interp"], s) for s in states], dtype=float)
        unc = np.asarray([map_lookup_np(variant["prior_uncertainty"]["O0_interp"], s) for s in states], dtype=float)
        ax.plot(progress, oracle, color="black", linewidth=2.0, label="oracle")
        ax.plot(progress, old_vals, color="tab:orange", linewidth=2.0, label="old 2D GP")
        ax.plot(progress, mean_only, color="tab:blue", linewidth=2.0, linestyle="--", label="new mean-only")
        ax.plot(progress, unc, color="tab:red", linewidth=2.0, linestyle="--", label="new uncertainty-aware")
        ax.set_title(f"{name}: visibility along probe corridor")
        ax.set_xlabel("corridor progress")
        ax.set_ylabel("visibility")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    tall_oracle = np.asarray([tall["oracle_p_vis"](s) for s in states], dtype=float)
    short_oracle = np.asarray([short["oracle_p_vis"](s) for s in states], dtype=float)
    tall_mean = np.asarray([map_lookup_np(tall["prior_mean_only"]["O0_interp"], s) for s in states], dtype=float)
    short_mean = np.asarray([map_lookup_np(short["prior_mean_only"]["O0_interp"], s) for s in states], dtype=float)
    tall_unc = np.asarray([map_lookup_np(tall["prior_uncertainty"]["O0_interp"], s) for s in states], dtype=float)
    short_unc = np.asarray([map_lookup_np(short["prior_uncertainty"]["O0_interp"], s) for s in states], dtype=float)
    ax.plot(progress, tall_oracle - short_oracle, color="black", linewidth=2.0, label="oracle: tall - short")
    ax.plot(progress, tall_mean - short_mean, color="tab:blue", linewidth=2.0, label="new mean-only: tall - short")
    ax.plot(progress, tall_unc - short_unc, color="tab:red", linewidth=2.0, label="new uncertainty-aware: tall - short")
    ax.plot(progress, old_vals - old_vals, color="tab:orange", linewidth=2.0, linestyle="--", label="old 2D GP: tall - short")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title("Height effect along the same corridor")
    ax.set_xlabel("corridor progress")
    ax.set_ylabel(r"$\Delta p_{vis}$")
    ax.legend(fontsize=8)
    return fig


def plot_story_paths(ctx: Mapping[str, object], story_results: Mapping[str, Mapping[str, Mapping[str, object]]]):
    scenes = list(ctx["variants"].keys())
    methods = ["old 2D GP", "offline prior", "corrected prior"]
    fig, axes = plt.subplots(len(scenes), len(methods), figsize=(16, 5 * len(scenes)), constrained_layout=True, sharex=True, sharey=True)
    if len(scenes) == 1:
        axes = np.asarray(axes).reshape(1, len(methods))
    for row, scene_name in enumerate(scenes):
        variant = ctx["variants"][scene_name]
        scene = variant["scene"]
        vis_extent = [scene["xs"].min(), scene["xs"].max(), scene["ys"].min(), scene["ys"].max()]
        for col, method in enumerate(methods):
            res = story_results[scene_name][method]
            if method == "old 2D GP":
                field = ctx["old_gp"]["P_vis_old_map"]
            elif method == "offline prior":
                field = variant["prior_uncertainty"]["O0_map"]
            else:
                field = res["correction_final_map"]
            ax = axes[row, col]
            ax.imshow(field, extent=vis_extent, origin="lower", cmap="viridis", aspect="equal", alpha=0.92, vmin=0.0, vmax=1.0)
            ax.contour(scene["true_grid"].xs, scene["true_grid"].ys, scene["true_grid"].occupancy, levels=[0.5], colors="white", linewidths=1.4)
            for path in res["planned_paths"]:
                ax.plot(path[:, 0], path[:, 1], color="tab:orange", alpha=0.12, linewidth=0.95)
            ax.plot(res["true_states"][:, 0], res["true_states"][:, 1], color="black", linewidth=2.6, label="true state")
            ax.plot(res["means"][:, 0], res["means"][:, 1], color="tab:blue", linestyle="--", linewidth=2.0, label="belief mean")
            miss_idx = np.where(~res["detections"])[0]
            if miss_idx.size:
                ax.scatter(res["true_states"][miss_idx, 0], res["true_states"][miss_idx, 1], marker="x", s=40, color="tab:red", linewidths=1.4, label="missed detection")
            ax.scatter(scene["start_state"][0], scene["start_state"][1], s=65, color="white", edgecolor="black")
            ax.scatter(scene["goal_state"][0], scene["goal_state"][1], marker="*", s=170, color="tab:red")
            ax.scatter(scene["camera_xy"][0], scene["camera_xy"][1], marker="^", s=85, color="white", edgecolor="black")
            ax.set_title(f"{scene_name} | {method}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    return fig


def plot_story_timeseries(ctx: Mapping[str, object], story_results: Mapping[str, Mapping[str, Mapping[str, object]]]):
    scenes = list(ctx["variants"].keys())
    fig, axes = plt.subplots(len(scenes), 3, figsize=(15, 4.8 * len(scenes)), constrained_layout=True)
    if len(scenes) == 1:
        axes = np.asarray(axes).reshape(1, 3)
    for row, scene_name in enumerate(scenes):
        ax_vis, ax_err, ax_goal = axes[row]
        for method in STORY_METHODS:
            res = story_results[scene_name][method]
            ax_vis.plot(res["p_vis_exec_runtime"], linewidth=2.0, label=f"{method} exec")
            ax_vis.plot(res["p_vis_plan_effective_runtime"], linewidth=1.8, linestyle="--", label=f"{method} plan")
            ax_err.plot(np.linalg.norm(res["true_states"][:, :2] - res["means"][:, :2], axis=1), linewidth=2.0, label=method)
            goal = ctx["variants"][scene_name]["scene"]["goal_state"][:2]
            ax_goal.plot(np.linalg.norm(res["true_states"][:, :2] - goal, axis=1), linewidth=2.0, label=method)
        ax_vis.set_title(f"{scene_name}: visibility")
        ax_vis.set_xlabel("step")
        ax_vis.set_ylabel("visibility")
        ax_err.set_title(f"{scene_name}: state error")
        ax_err.set_xlabel("step")
        ax_err.set_ylabel("xy error [m]")
        ax_goal.set_title(f"{scene_name}: goal distance")
        ax_goal.set_xlabel("step")
        ax_goal.set_ylabel("distance [m]")
        ax_vis.legend(fontsize=7)
    return fig


def plot_story_risk_ambiguity(ctx: Mapping[str, object], story_results: Mapping[str, Mapping[str, Mapping[str, object]]]):
    scenes = list(ctx["variants"].keys())
    fig, axes = plt.subplots(len(scenes), 4, figsize=(20, 4.8 * len(scenes)), constrained_layout=True)
    if len(scenes) == 1:
        axes = np.asarray(axes).reshape(1, 4)
    for row, scene_name in enumerate(scenes):
        ax_pvis, ax_risk, ax_amb, ax_opt = axes[row]
        for method in STORY_METHODS:
            res = story_results[scene_name][method]
            ax_pvis.plot(res["p_vis_plan_effective_runtime"], linewidth=2.0, label=method)
            ax_risk.plot(res["risk_runtime"], linewidth=2.0, label=method)
            ax_amb.plot(res["ambiguity_runtime"], linewidth=2.0, label=method)
            valid = np.where(np.isfinite(res["optimizer_values"]), res["optimizer_values"], np.nan)
            ax_opt.plot(valid, linewidth=2.0, label=method)
        ax_pvis.set_title(f"{scene_name}: effective planner visibility")
        ax_pvis.set_xlabel("step")
        ax_pvis.set_ylabel("effective $p_{vis}$")
        ax_risk.set_title(f"{scene_name}: risk term")
        ax_risk.set_xlabel("step")
        ax_risk.set_ylabel("risk")
        ax_amb.set_title(f"{scene_name}: ambiguity term")
        ax_amb.set_xlabel("step")
        ax_amb.set_ylabel("ambiguity")
        ax_opt.set_title(f"{scene_name}: planner objective")
        ax_opt.set_xlabel("step")
        ax_opt.set_ylabel("objective value")
        ax_pvis.legend(fontsize=7)
    return fig


def _mask_to_spans(mask: np.ndarray) -> List[Tuple[float, float]]:
    mask = np.asarray(mask, dtype=bool)
    spans: List[Tuple[float, float]] = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx - 0.5
        elif not active and start is not None:
            spans.append((start, idx - 0.5))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 0.5))
    return spans


def plot_story_belief_diagnostics(
    ctx: Mapping[str, object],
    story_results: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    occlusion_threshold=0.35,
):
    scenes = list(ctx["variants"].keys())
    fig, axes = plt.subplots(len(scenes), len(STORY_METHODS), figsize=(16, 4.9 * len(scenes)), constrained_layout=True, sharex=True)
    if len(scenes) == 1:
        axes = np.asarray(axes).reshape(1, len(STORY_METHODS))
    for row, scene_name in enumerate(scenes):
        for col, method in enumerate(STORY_METHODS):
            res = story_results[scene_name][method]
            ax = axes[row, col]
            trace_vals = covariance_trace_series(res["covs"])
            occluded_mask = np.asarray(res["p_vis_exec_runtime"] < float(occlusion_threshold), dtype=bool)
            for left, right in _mask_to_spans(occluded_mask):
                ax.axvspan(left, right, color="tab:orange", alpha=0.14)
            ax.plot(trace_vals, color="tab:blue", linewidth=2.1, label=r"$\mathrm{tr}(S_k)$")
            miss_idx = np.where(~res["detections"])[0]
            if miss_idx.size:
                ax.scatter(
                    miss_idx,
                    trace_vals[miss_idx],
                    marker="x",
                    s=34,
                    color="tab:red",
                    linewidths=1.2,
                    label="missed detection",
                    zorder=4,
                )
            ax.set_title(f"{scene_name} | {method}")
            ax.set_xlabel("step")
            ax.set_ylabel(r"$\mathrm{tr}(S_k)$")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Belief impact under occlusion", fontsize=14)
    return fig


def plot_story_correction_maps(ctx: Mapping[str, object], story_results: Mapping[str, Mapping[str, Mapping[str, object]]]):
    scenes = list(ctx["variants"].keys())
    fig, axes = plt.subplots(len(scenes), 3, figsize=(15, 5 * len(scenes)), constrained_layout=True, sharex=True, sharey=True)
    if len(scenes) == 1:
        axes = np.asarray(axes).reshape(1, 3)
    for row, scene_name in enumerate(scenes):
        variant = ctx["variants"][scene_name]
        scene = variant["scene"]
        corrected = story_results[scene_name]["corrected prior"]
        offline = variant["prior_uncertainty"]["O0_map"]
        corrected_map = corrected["correction_final_map"]
        delta_A = corrected["correction_delta_A_map"]
        panels = [
            (offline, "offline prior", "viridis", 0.0, 1.0),
            (corrected_map, "corrected prior", "viridis", 0.0, 1.0),
            (delta_A, r"$\Delta_A$ correction", "coolwarm", None, None),
        ]
        vis_extent = [scene["xs"].min(), scene["xs"].max(), scene["ys"].min(), scene["ys"].max()]
        lim = max(float(np.max(np.abs(delta_A))), 1e-6)
        for col, (field, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            if title == r"$\Delta_A$ correction":
                im = ax.imshow(field, extent=vis_extent, origin="lower", cmap=cmap, aspect="equal", vmin=-lim, vmax=lim)
                contour_color = "black"
            else:
                im = ax.imshow(field, extent=vis_extent, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
                contour_color = "white"
            ax.contour(scene["true_grid"].xs, scene["true_grid"].ys, scene["true_grid"].occupancy, levels=[0.5], colors=contour_color, linewidths=1.2)
            ax.set_title(f"{scene_name}: {title}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig
