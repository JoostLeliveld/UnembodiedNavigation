"""Notebook-only helpers for state-dependent observation studies.

This module is intentionally separate from the runtime `src/` packages.
It exists to keep the canonical notebooks compact while the thesis mechanism is
still being worked out in notebooks first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scripts.botnav_efe_helpers import (
    ET1,
    ET2,
    PlanarCamera,
    UnicycleEFEAgent,
    ambiguity,
    risk,
    unicycle_jacobian,
    unicycle_step,
)


def make_default_camera():
    return PlanarCamera(
        cam_pos=(-3.0, -3.0, 6.0),
        look_at=(1.5, 1.5, 0.0),
        img_width=1280,
        img_height=720,
        fov_h_rad=1.5708,
    )


def default_process_covariance(dt=0.2, xy_std=0.05, theta_std=0.04):
    return np.diag([
        float(xy_std) ** 2 * dt,
        float(xy_std) ** 2 * dt,
        float(theta_std) ** 2 * dt,
    ]).astype(float)


def process_covariance_from_rho(dt=0.2, rho_xy=1e-2, rho_theta=None):
    """Notebook baseline aligned with `diffdrivenav-bev2image-efe .ipynb`."""
    rho_xy = float(rho_xy)
    rho_theta = rho_xy if rho_theta is None else float(rho_theta)
    sigma_process_xy = math.sqrt((float(dt) ** 3 / 3.0) * rho_xy)
    sigma_process_theta = math.sqrt(float(dt) * rho_theta)
    return np.diag([
        sigma_process_xy ** 2,
        sigma_process_xy ** 2,
        sigma_process_theta ** 2,
    ]).astype(float)


def observation_covariance(obs_mode="uv", uv_std=2.0, yaw_std=0.05):
    obs_mode = str(obs_mode).strip().lower()
    if obs_mode == "uv":
        return np.diag([float(uv_std) ** 2, float(uv_std) ** 2]).astype(float)
    if obs_mode == "uvt":
        return np.diag([
            float(uv_std) ** 2,
            float(uv_std) ** 2,
            float(yaw_std) ** 2,
        ]).astype(float)
    raise ValueError("obs_mode must be 'uv' or 'uvt'")


def make_observation_fn(camera: PlanarCamera, obs_mode="uv"):
    obs_mode = str(obs_mode).strip().lower()

    if obs_mode == "uv":
        def g(state):
            return camera.g(state)
        return g

    if obs_mode == "uvt":
        def g(state):
            uv = camera.g(state)
            return np.array([uv[0], uv[1], float(state[2])], dtype=float)
        return g

    raise ValueError("obs_mode must be 'uv' or 'uvt'")


def grid_states(xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, nx=81, ny=81, theta=0.0):
    xs = np.linspace(float(xmin), float(xmax), int(nx))
    ys = np.linspace(float(ymin), float(ymax), int(ny))
    X, Y = np.meshgrid(xs, ys)
    T = np.full_like(X, float(theta))
    return xs, ys, np.stack([X, Y, T], axis=-1)


def rollout_controls(initial_state, controls, dt):
    states = [np.asarray(initial_state, dtype=float)]
    x = np.asarray(initial_state, dtype=float)
    for u in np.asarray(controls, dtype=float):
        x = unicycle_step(x, u, dt)
        states.append(x.copy())
    return np.asarray(states, dtype=float)


def make_action_library(v_values=(0.0, 0.12, 0.22), w_values=(-0.8, -0.35, 0.0, 0.35, 0.8)):
    return np.asarray([(float(v), float(w)) for v in v_values for w in w_values], dtype=float)


def covariance_trace_series(covariances):
    covariances = np.asarray(covariances, dtype=float)
    return np.trace(covariances, axis1=-2, axis2=-1)


def covariance_logdet(matrix):
    matrix = np.asarray(matrix, dtype=float)
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        return np.nan
    return float(logdet)


def covariance_logdet_series(covariances):
    return np.asarray([covariance_logdet(S) for S in covariances], dtype=float)


def posterior_update(mean_pred, cov_pred, y_meas, g, R_eff, approx="ET2"):
    if approx == "ET1":
        mu_y, Sigma_y, Gamma = ET1(mean_pred, cov_pred, g, addmatrix=R_eff, forceHermitian=True)
    elif approx == "ET2":
        mu_y, Sigma_y, Gamma = ET2(mean_pred, cov_pred, g, addmatrix=R_eff, forceHermitian=True)
    else:
        raise ValueError("approx must be ET1 or ET2")

    innovation = np.asarray(y_meas, dtype=float) - np.asarray(mu_y, dtype=float)
    Sigma_inv = np.linalg.pinv(Sigma_y)
    K = Gamma @ Sigma_inv
    mean_post = np.asarray(mean_pred, dtype=float) + K @ innovation
    mean_post[2] = math.atan2(math.sin(mean_post[2]), math.cos(mean_post[2]))
    cov_post = np.asarray(cov_pred, dtype=float) - Gamma @ Sigma_inv @ Gamma.T
    cov_post = 0.5 * (cov_post + cov_post.T)
    return mean_post, cov_post, mu_y, Sigma_y, Gamma


def predict_unicycle_belief(mean, cov, control, dt, Q):
    mean_next = unicycle_step(mean, control, dt)
    F = unicycle_jacobian(mean_next, control, dt)
    cov_next = F @ cov @ F.T + np.asarray(Q, dtype=float)
    return mean_next, 0.5 * (cov_next + cov_next.T)


def radial_drop_q(state, center=(0.0, 0.0), radius=1.0, transition=0.5, q_min=0.15):
    xy = np.asarray(state[:2], dtype=float)
    c = np.asarray(center, dtype=float)
    d = float(np.linalg.norm(xy - c))
    if transition <= 1e-9:
        return float(q_min if d <= radius else 1.0)
    if d <= radius:
        return float(q_min)
    t = np.clip((d - radius) / transition, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return float(np.clip(q_min + (1.0 - q_min) * t, 0.0, 1.0))


def boundary_drop_q(state, camera: PlanarCamera, margin_px=140.0, q_min=0.15):
    u, v = camera.world_to_pixel(float(state[0]), float(state[1]))
    if not np.isfinite(u) or not np.isfinite(v):
        return float(q_min)
    if u < 0 or u >= camera.img_width or v < 0 or v >= camera.img_height:
        return float(q_min)
    distance_px = min(u, v, camera.img_width - 1.0 - u, camera.img_height - 1.0 - v)
    t = float(np.clip(distance_px / max(float(margin_px), 1e-6), 0.0, 1.0))
    return float(np.clip(q_min + (1.0 - q_min) * t, 0.0, 1.0))


def orientation_drop_q(state, preferred_heading=0.0, power=1.0, q_min=0.15):
    theta = float(state[2])
    delta = math.atan2(math.sin(theta - preferred_heading), math.cos(theta - preferred_heading))
    alignment = 0.5 * (1.0 + math.cos(delta))
    alignment = float(np.clip(alignment, 0.0, 1.0)) ** max(float(power), 1e-6)
    return float(np.clip(q_min + (1.0 - q_min) * alignment, 0.0, 1.0))


def narrow_passage_q(state, x_bounds=(-0.75, 0.75), y_bounds=(-2.5, 2.5), transition=0.75, q_min=0.15):
    x = float(state[0])
    y = float(state[1])
    if x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]:
        return float(q_min)
    dx = 0.0 if x_bounds[0] <= x <= x_bounds[1] else min(abs(x - x_bounds[0]), abs(x - x_bounds[1]))
    dy = 0.0 if y_bounds[0] <= y <= y_bounds[1] else min(abs(y - y_bounds[0]), abs(y - y_bounds[1]))
    d = math.hypot(dx, dy)
    t = np.clip(d / max(float(transition), 1e-6), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return float(np.clip(q_min + (1.0 - q_min) * t, 0.0, 1.0))


def effective_covariance(q_value, R_good, R_bad):
    q_value = float(np.clip(q_value, 0.0, 1.0))
    R_eff = q_value * np.asarray(R_good, dtype=float) + (1.0 - q_value) * np.asarray(R_bad, dtype=float)
    return 0.5 * (R_eff + R_eff.T)


def evaluate_q_field_on_grid(q_fn, states_grid):
    flat = states_grid.reshape(-1, states_grid.shape[-1])
    values = np.asarray([q_fn(state) for state in flat], dtype=float)
    return values.reshape(states_grid.shape[:2])


@dataclass
class CandidateSummary:
    control: np.ndarray
    total: float
    risk_term: float
    ambiguity_term: float
    control_term: float
    risk_raw: float
    ambiguity_raw: float
    control_raw: float
    final_state: np.ndarray
    min_q: float


@dataclass
class OccupancyGrid2D:
    xs: np.ndarray
    ys: np.ndarray
    occupancy: np.ndarray
    resolution: float

    @property
    def extent(self):
        return [float(self.xs.min()), float(self.xs.max()), float(self.ys.min()), float(self.ys.max())]


def make_unicycle_agent(goal_obs, goal_obs_cov, g, Q, R, eta=0.1, dt=0.2, horizon=5):
    return UnicycleEFEAgent(
        goal=(np.asarray(goal_obs, dtype=float), np.asarray(goal_obs_cov, dtype=float)),
        g=g,
        Q=np.asarray(Q, dtype=float),
        R=np.asarray(R, dtype=float),
        eta=float(eta),
        dt=float(dt),
        time_horizon=int(horizon),
    )


def score_action_library(
    mean0,
    cov0,
    actions,
    dt,
    Q,
    g,
    goal_obs,
    goal_obs_cov,
    R_good,
    *,
    R_bad=None,
    q_fn=None,
    horizon=4,
    risk_weight=1.0,
    ambiguity_weight=1.0,
    control_weight=0.1,
    approx="ET2",
    add_ambiguity=True,
):
    if R_bad is None:
        R_bad = np.asarray(R_good, dtype=float)
    results = []
    for action in np.asarray(actions, dtype=float):
        m = np.asarray(mean0, dtype=float).copy()
        S = np.asarray(cov0, dtype=float).copy()
        total_risk_raw = 0.0
        total_ambiguity_raw = 0.0
        total_control_raw = 0.0
        q_history = []
        for _ in range(int(horizon)):
            m, S = predict_unicycle_belief(m, S, action, dt, Q)
            q_value = 1.0 if q_fn is None else float(np.clip(q_fn(m), 0.0, 1.0))
            q_history.append(q_value)
            R_eff = effective_covariance(q_value, R_good, R_bad)
            if approx == "ET1":
                mu_y, Sigma_y, Gamma = ET1(m, S, g, addmatrix=R_eff, forceHermitian=True)
            elif approx == "ET2":
                mu_y, Sigma_y, Gamma = ET2(m, S, g, addmatrix=R_eff, forceHermitian=True)
            else:
                raise ValueError("approx must be ET1 or ET2")
            total_risk_raw += risk(mu_y, Sigma_y, (goal_obs, goal_obs_cov))
            if add_ambiguity:
                total_ambiguity_raw += ambiguity(Sigma_y, Gamma, S)
            total_control_raw += float(np.sum(action ** 2))

        risk_term = float(risk_weight) * float(total_risk_raw)
        ambiguity_term = float(ambiguity_weight) * float(total_ambiguity_raw)
        control_term = float(control_weight) * float(total_control_raw)

        results.append(
            CandidateSummary(
                control=np.asarray(action, dtype=float),
                total=float(risk_term + ambiguity_term + control_term),
                risk_term=risk_term,
                ambiguity_term=ambiguity_term,
                control_term=control_term,
                risk_raw=float(total_risk_raw),
                ambiguity_raw=float(total_ambiguity_raw),
                control_raw=float(total_control_raw),
                final_state=m.copy(),
                min_q=float(np.min(q_history) if q_history else 1.0),
            )
        )

    results.sort(key=lambda item: item.total)
    return results


def summarize_covariance_along_rollout(mean0, cov0, controls, dt, Q, g, R_sequence, approx="ET2"):
    means = [np.asarray(mean0, dtype=float).copy()]
    covs = [np.asarray(cov0, dtype=float).copy()]
    obs_means = []
    obs_covs = []

    m = np.asarray(mean0, dtype=float).copy()
    S = np.asarray(cov0, dtype=float).copy()
    for control, R_eff in zip(np.asarray(controls, dtype=float), list(R_sequence)):
        m, S = predict_unicycle_belief(m, S, control, dt, Q)
        if approx == "ET1":
            mu_y, Sigma_y, _ = ET1(m, S, g, addmatrix=R_eff, forceHermitian=True)
        elif approx == "ET2":
            mu_y, Sigma_y, _ = ET2(m, S, g, addmatrix=R_eff, forceHermitian=True)
        else:
            raise ValueError("approx must be ET1 or ET2")
        means.append(m.copy())
        covs.append(S.copy())
        obs_means.append(np.asarray(mu_y, dtype=float))
        obs_covs.append(np.asarray(Sigma_y, dtype=float))

    return (
        np.asarray(means, dtype=float),
        np.asarray(covs, dtype=float),
        np.asarray(obs_means, dtype=float),
        np.asarray(obs_covs, dtype=float),
    )


def simulate_receding_horizon(
    mean0,
    cov0,
    actions,
    dt,
    Q,
    g,
    goal_obs,
    goal_obs_cov,
    R_good,
    *,
    R_bad=None,
    q_fn=None,
    horizon=4,
    n_steps=15,
    risk_weight=1.0,
    ambiguity_weight=1.0,
    control_weight=0.1,
    approx="ET2",
    add_ambiguity=True,
):
    if R_bad is None:
        R_bad = np.asarray(R_good, dtype=float)

    m = np.asarray(mean0, dtype=float).copy()
    S = np.asarray(cov0, dtype=float).copy()
    means = [m.copy()]
    covs = [S.copy()]
    controls = []
    q_values = []
    ambiguity_terms = []
    risk_terms = []

    for _ in range(int(n_steps)):
        candidates = score_action_library(
            m,
            S,
            actions,
            dt,
            Q,
            g,
            goal_obs,
            goal_obs_cov,
            R_good,
            R_bad=R_bad,
            q_fn=q_fn,
            horizon=horizon,
            risk_weight=risk_weight,
            ambiguity_weight=ambiguity_weight,
            control_weight=control_weight,
            approx=approx,
            add_ambiguity=add_ambiguity,
        )
        best = candidates[0]
        u = best.control
        m, S = predict_unicycle_belief(m, S, u, dt, Q)
        q_value = 1.0 if q_fn is None else float(np.clip(q_fn(m), 0.0, 1.0))
        R_eff = effective_covariance(q_value, R_good, R_bad)
        if approx == "ET1":
            mu_y, Sigma_y, Gamma = ET1(m, S, g, addmatrix=R_eff, forceHermitian=True)
        elif approx == "ET2":
            mu_y, Sigma_y, Gamma = ET2(m, S, g, addmatrix=R_eff, forceHermitian=True)
        else:
            raise ValueError("approx must be ET1 or ET2")

        controls.append(u.copy())
        q_values.append(q_value)
        ambiguity_terms.append(float(ambiguity(Sigma_y, Gamma, S)) if add_ambiguity else 0.0)
        risk_terms.append(float(risk(mu_y, Sigma_y, (goal_obs, goal_obs_cov))))
        means.append(m.copy())
        covs.append(S.copy())

    return {
        "means": np.asarray(means, dtype=float),
        "covs": np.asarray(covs, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "q_values": np.asarray(q_values, dtype=float),
        "ambiguity_terms": np.asarray(ambiguity_terms, dtype=float),
        "risk_terms": np.asarray(risk_terms, dtype=float),
    }


def make_occupancy_grid(
    xmin=-5.0,
    xmax=5.0,
    ymin=-5.0,
    ymax=5.0,
    resolution=0.1,
    *,
    rectangles=None,
    circles=None,
    border_occupancy=True,
):
    xs = np.arange(float(xmin), float(xmax) + float(resolution), float(resolution))
    ys = np.arange(float(ymin), float(ymax) + float(resolution), float(resolution))
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros_like(X, dtype=float)

    if border_occupancy:
        occ[0, :] = 1.0
        occ[-1, :] = 1.0
        occ[:, 0] = 1.0
        occ[:, -1] = 1.0

    for rect in rectangles or []:
        x0, x1, y0, y1 = rect
        mask = (X >= float(x0)) & (X <= float(x1)) & (Y >= float(y0)) & (Y <= float(y1))
        occ[mask] = 1.0

    for circle in circles or []:
        cx, cy, radius = circle
        mask = (X - float(cx)) ** 2 + (Y - float(cy)) ** 2 <= float(radius) ** 2
        occ[mask] = 1.0

    return OccupancyGrid2D(xs=xs, ys=ys, occupancy=occ, resolution=float(resolution))


def _bilinear_sample(grid: OccupancyGrid2D, x: float, y: float) -> float:
    xs = grid.xs
    ys = grid.ys
    occ = grid.occupancy
    if x <= xs[0] or x >= xs[-1] or y <= ys[0] or y >= ys[-1]:
        return 1.0

    ix = int(np.searchsorted(xs, x, side="right") - 1)
    iy = int(np.searchsorted(ys, y, side="right") - 1)
    ix = max(0, min(ix, xs.shape[0] - 2))
    iy = max(0, min(iy, ys.shape[0] - 2))

    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    z00 = occ[iy, ix]
    z10 = occ[iy, ix + 1]
    z01 = occ[iy + 1, ix]
    z11 = occ[iy + 1, ix + 1]
    z0 = (1.0 - tx) * z00 + tx * z10
    z1 = (1.0 - tx) * z01 + tx * z11
    return float((1.0 - ty) * z0 + ty * z1)


def raycast_visibility_q(state, grid: OccupancyGrid2D, camera_xy, tau=6.0, n_samples=120):
    start = np.asarray(camera_xy, dtype=float)
    end = np.asarray(state[:2], dtype=float)
    ts = np.linspace(0.0, 1.0, int(max(n_samples, 2)))
    samples = np.outer(1.0 - ts, start) + np.outer(ts, end)
    occ_values = np.asarray([_bilinear_sample(grid, xy[0], xy[1]) for xy in samples], dtype=float)
    mean_occ = float(np.mean(occ_values))
    return float(np.clip(np.exp(-float(tau) * mean_occ), 0.0, 1.0))


def hard_line_of_sight_q(state, grid: OccupancyGrid2D, camera_xy, threshold=0.5, n_samples=120):
    start = np.asarray(camera_xy, dtype=float)
    end = np.asarray(state[:2], dtype=float)
    ts = np.linspace(0.0, 1.0, int(max(n_samples, 2)))
    samples = np.outer(1.0 - ts, start) + np.outer(ts, end)
    occ_values = np.asarray([_bilinear_sample(grid, xy[0], xy[1]) for xy in samples], dtype=float)
    return 0.0 if np.any(occ_values >= float(threshold)) else 1.0


def evaluate_visibility_on_grid(grid_states_xyz, q_fn):
    flat = grid_states_xyz.reshape(-1, grid_states_xyz.shape[-1])
    values = np.asarray([q_fn(state) for state in flat], dtype=float)
    return values.reshape(grid_states_xyz.shape[:2])
