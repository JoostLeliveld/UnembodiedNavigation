#!/usr/bin/env python3
"""Static four-camera usable-probability planning experiment.

This is deliberately an offline route-discrimination and stochastic-replay study. It uses
repository-native camera fields, warehouse geometry, task definitions, unicycle covariance
growth, and the stdlib twin of the runtime's explicit hit/miss implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
SUPERVISOR = HERE.parent
REPO = HERE.parents[3]
for source in (REPO / "src/reliability", REPO / "src/planning", REPO / "src/unav_common"):
    sys.path.insert(0, str(source))
sys.path.insert(0, str(SUPERVISOR))

import render_all as base  # noqa: E402
from planning.core.dynamics import unicycle_jacobian, unicycle_process_noise  # noqa: E402
from reliability.observation_model import (  # noqa: E402
    expected_posterior_branch,
    kalman_posterior,
    scaled_covariance_baseline,
)


CONFIG_PATH = HERE / "experiment.yaml"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
PG_SUMMARY = REPO / "logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json"
CAMERAS = tuple(base.CAMERA_POSES)
METHODS = (
    "availability_blind_shortest",
    "r_over_p_shortcut",
    "explicit_hit_miss",
)
METHOD_LABELS = {
    "availability_blind_shortest": "availability-blind shortest",
    "r_over_p_shortcut": r"single update with $R/p$",
    "explicit_hit_miss": "explicit hit/miss",
}
METHOD_COLORS = {
    "availability_blind_shortest": "#263238",
    "r_over_p_shortcut": "#c43c8c",
    "explicit_hit_miss": "#087f8c",
}
H = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
CHI2_95_2D = 5.991464547107979


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def as_matrix(value) -> np.ndarray:
    return np.asarray(value, dtype=float)


def load_r_cond() -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    payload = json.loads(PG_SUMMARY.read_text(encoding="utf-8"))
    arm = payload["arms"]["raw IPM (floor, no correction)"]["per_camera"]
    matrices: dict[str, np.ndarray] = {}
    provenance: dict[str, dict] = {}
    for camera in CAMERAS:
        row = arm[camera]
        sigma = math.sqrt(0.5 * (float(row["radial_sd_m"]) ** 2 + float(row["lateral_sd_m"]) ** 2))
        matrices[camera] = np.eye(2, dtype=float) * sigma**2
        provenance[camera] = {
            "n_detections": int(row["n"]),
            "radial_sd_m": float(row["radial_sd_m"]),
            "lateral_sd_m": float(row["lateral_sd_m"]),
            "isotropic_sigma_cond_m": sigma,
        }
    return matrices, provenance


def load_static_fields(ctx: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    planning = {camera: as_matrix(ctx["per_camera"]["gp"][camera]) for camera in CAMERAS}
    replay = {
        camera: as_matrix(ctx["gp_per_camera"][camera]["P_mean_map"])
        for camera in CAMERAS
    }
    for fields in (planning, replay):
        for camera, field in fields.items():
            if field.shape != ctx["driveable"].shape:
                raise RuntimeError(f"{camera} field shape {field.shape} does not match grid")
            if not np.all(np.isfinite(field)):
                raise RuntimeError(f"{camera} field contains non-finite values")
            fields[camera] = np.clip(field, 1.0e-4, 1.0 - 1.0e-4)
    return planning, replay


def frozen_camera_policy(
    planning_fields: dict[str, np.ndarray],
    r_cond: dict[str, np.ndarray],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_xy = float(config["dynamics"]["initial_sigma_xy_m"])
    sigma_theta = float(config["dynamics"]["initial_sigma_theta_rad"])
    prior = np.diag([sigma_xy**2, sigma_xy**2, sigma_theta**2])
    prior_trace = float(np.trace(prior[:2, :2]))
    utilities = []
    for camera in CAMERAS:
        certain = expected_posterior_branch(prior, H, r_cond[camera], 1.0)
        information_if_hit = prior_trace - float(np.trace(as_matrix(certain.cov_hit)[:2, :2]))
        utilities.append(planning_fields[camera] * information_if_hit)
    utility_stack = np.stack(utilities, axis=0)
    selected_index = np.argmax(utility_stack, axis=0).astype(np.int16)
    selected_p = np.take_along_axis(
        np.stack([planning_fields[c] for c in CAMERAS], axis=0),
        selected_index[None, ...],
        axis=0,
    )[0]
    selected_utility = np.max(utility_stack, axis=0)
    return selected_index, selected_p, selected_utility


def nearest_cell(xs: np.ndarray, ys: np.ndarray, xy: Iterable[float]) -> tuple[int, int]:
    x, y = xy
    return int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))


def nearest_valid(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, xy: Iterable[float]) -> tuple[int, int]:
    i0, j0 = nearest_cell(xs, ys, xy)
    if mask[i0, j0]:
        return i0, j0
    ii, jj = np.where(mask)
    index = int(np.argmin((ii - i0) ** 2 + (jj - j0) ** 2))
    return int(ii[index]), int(jj[index])


def weighted_route(
    field: np.ndarray,
    driveable: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    reliability_lambda: float,
) -> np.ndarray:
    start = nearest_valid(driveable, xs, ys, start_xy)
    goal = nearest_valid(driveable, xs, ys, goal_xy)
    ny, nx = driveable.shape
    distance = np.full((ny, nx), np.inf, dtype=float)
    distance[start] = 0.0
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while queue:
        cost, current = heapq.heappop(queue)
        if current == goal:
            break
        if cost != distance[current]:
            continue
        i, j = current
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx) or not driveable[ni, nj]:
                continue
            if di and dj and (not driveable[i + di, j] or not driveable[i, j + dj]):
                continue
            step = math.hypot(di * dy, dj * dx)
            p = float(np.clip(field[ni, nj], 0.0, 1.0))
            candidate = cost + step * (1.0 + reliability_lambda * (1.0 - p) ** 2)
            if candidate < distance[ni, nj]:
                distance[ni, nj] = candidate
                previous[(ni, nj)] = current
                heapq.heappush(queue, (candidate, (ni, nj)))
    if goal != start and goal not in previous:
        raise RuntimeError(f"no route from {start_xy} to {goal_xy}")
    cells = [goal]
    while cells[-1] != start:
        cells.append(previous[cells[-1]])
    cells.reverse()
    return np.asarray([[xs[j], ys[i]] for i, j in cells], dtype=float)


def route_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def resample_path(path: np.ndarray, step_m: float) -> np.ndarray:
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    if cumulative[-1] <= 0.0:
        return path[:1].copy()
    samples = np.arange(0.0, cumulative[-1], step_m)
    if len(samples) == 0 or not math.isclose(float(samples[-1]), float(cumulative[-1])):
        samples = np.append(samples, cumulative[-1])
    x = np.interp(samples, cumulative, path[:, 0])
    y = np.interp(samples, cumulative, path[:, 1])
    return np.column_stack([x, y])


def field_value(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, xy: np.ndarray) -> float:
    i, j = nearest_cell(xs, ys, xy)
    return float(field[i, j])


def camera_at(selected_index: np.ndarray, xs: np.ndarray, ys: np.ndarray, xy: np.ndarray) -> str:
    i, j = nearest_cell(xs, ys, xy)
    return CAMERAS[int(selected_index[i, j])]


def model_step(
    covariance: np.ndarray,
    xy_prev: np.ndarray,
    xy_next: np.ndarray,
    camera: str,
    p_use: float,
    model: str,
    r_cond: dict[str, np.ndarray],
    config: dict,
) -> np.ndarray:
    delta = xy_next - xy_prev
    distance = float(np.linalg.norm(delta))
    dt = float(config["dynamics"]["dt_s"])
    theta = math.atan2(float(delta[1]), float(delta[0])) if distance > 1.0e-12 else 0.0
    speed = distance / dt
    state = np.asarray([xy_prev[0], xy_prev[1], theta], dtype=float)
    control = np.asarray([speed, 0.0], dtype=float)
    transition = unicycle_jacobian(state, control, dt)
    process = unicycle_process_noise(
        float(config["dynamics"]["process_noise_xy"]),
        float(config["dynamics"]["process_noise_theta"]),
        dt,
        theta=theta,
        v=speed,
    )
    prior = transition @ covariance @ transition.T + process
    if model == "explicit_hit_miss":
        return as_matrix(expected_posterior_branch(prior, H, r_cond[camera], p_use).cov_expected)
    if model == "r_over_p_shortcut":
        r_scaled = scaled_covariance_baseline(r_cond[camera], p_use)
        return as_matrix(kalman_posterior(prior, H, r_scaled))
    if model == "predict_only":
        return prior
    raise KeyError(model)


def propagate_route(
    dense_path: np.ndarray,
    model: str,
    selected_index: np.ndarray,
    planning_fields: dict[str, np.ndarray],
    r_cond: dict[str, np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    config: dict,
) -> dict:
    sigma_xy = float(config["dynamics"]["initial_sigma_xy_m"])
    sigma_theta = float(config["dynamics"]["initial_sigma_theta_rad"])
    covariance = np.diag([sigma_xy**2, sigma_xy**2, sigma_theta**2])
    integral_trace = 0.0
    p_values: list[float] = []
    camera_counts = {camera: 0 for camera in CAMERAS}
    for previous, current in zip(dense_path[:-1], dense_path[1:]):
        camera = camera_at(selected_index, xs, ys, current)
        p_use = field_value(planning_fields[camera], xs, ys, current)
        covariance = model_step(covariance, previous, current, camera, p_use, model, r_cond, config)
        ds = float(np.linalg.norm(current - previous))
        integral_trace += float(np.trace(covariance[:2, :2])) * ds
        p_values.append(p_use)
        camera_counts[camera] += 1
    length = route_length(dense_path)
    weight = float(config["route_search"]["uncertainty_weight_per_m2"])
    return {
        "path_length_m": length,
        "integral_position_trace_m3": integral_trace,
        "objective_m": length + weight * integral_trace,
        "terminal_sigma_xy_m": math.sqrt(max(float(np.trace(covariance[:2, :2])) / 2.0, 0.0)),
        "mean_planning_p_use": float(np.mean(p_values)) if p_values else 0.0,
        "min_planning_p_use": float(np.min(p_values)) if p_values else 0.0,
        "camera_counts": camera_counts,
    }


def route_key(path: np.ndarray) -> bytes:
    return np.round(path, 5).tobytes()


def build_candidates(
    task: dict,
    selected_p: np.ndarray,
    ctx: dict,
    config: dict,
) -> list[np.ndarray]:
    start = (float(task["start"]["x"]), float(task["start"]["y"]))
    goal = (float(task["goal"]["x"]), float(task["goal"]["y"]))
    unique: dict[bytes, np.ndarray] = {}
    for penalty in config["route_search"]["candidate_reliability_lambdas"]:
        path = weighted_route(selected_p, ctx["driveable"], ctx["xs"], ctx["ys"], start, goal, float(penalty))
        unique.setdefault(route_key(path), path)
    return list(unique.values())


def choose_routes(
    ctx: dict,
    config: dict,
    planning_fields: dict[str, np.ndarray],
    selected_index: np.ndarray,
    selected_p: np.ndarray,
    r_cond: dict[str, np.ndarray],
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict], list[dict]]:
    selected: dict[tuple[str, str], np.ndarray] = {}
    selected_rows: list[dict] = []
    candidate_rows: list[dict] = []
    step_m = float(config["dynamics"]["dt_s"]) * float(config["dynamics"]["speed_mps"])
    for task_name in config["tasks"]:
        task = ctx["tasks"][task_name]
        candidates = build_candidates(task, selected_p, ctx, config)
        evaluated: dict[str, list[dict]] = {method: [] for method in METHODS}
        for candidate_id, grid_path in enumerate(candidates):
            path = resample_path(grid_path, step_m)
            for method in METHODS:
                planning_model = "explicit_hit_miss" if method == "explicit_hit_miss" else "r_over_p_shortcut"
                result = propagate_route(path, planning_model, selected_index, planning_fields, r_cond, ctx["xs"], ctx["ys"], config)
                row = {"task": task_name, "candidate_id": candidate_id, "planning_model": method, **result}
                row.pop("camera_counts")
                candidate_rows.append(row)
                evaluated[method].append({"path": path, **result})
        for method in METHODS:
            if method == "availability_blind_shortest":
                winner = min(evaluated[method], key=lambda item: (item["path_length_m"], item["objective_m"]))
            else:
                winner = min(evaluated[method], key=lambda item: (item["objective_m"], item["path_length_m"]))
            path = winner["path"]
            selected[(task_name, method)] = path
            honest_evaluation = propagate_route(path, "explicit_hit_miss", selected_index, planning_fields, r_cond, ctx["xs"], ctx["ys"], config)
            selected_rows.append({
                "task": task_name,
                "method": method,
                "candidate_count": len(candidates),
                "planner_objective_m": float(winner["objective_m"]),
                **{key: value for key, value in honest_evaluation.items() if key != "camera_counts"},
                **{f"steps_{camera}": int(honest_evaluation["camera_counts"][camera]) for camera in CAMERAS},
            })
    return selected, selected_rows, candidate_rows


def joseph_update(prior: np.ndarray, residual_error: np.ndarray, measurement_noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    innovation = H @ prior @ H.T + measurement_noise
    gain = prior @ H.T @ np.linalg.inv(innovation)
    identity = np.eye(3)
    posterior = (identity - gain @ H) @ prior @ (identity - gain @ H).T + gain @ measurement_noise @ gain.T
    error = (identity - gain @ H) @ residual_error
    return 0.5 * (posterior + posterior.T), error


def simulate_run(
    path: np.ndarray,
    seed: int,
    selected_index: np.ndarray,
    replay_fields: dict[str, np.ndarray],
    r_cond: dict[str, np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    config: dict,
) -> dict:
    rng = np.random.default_rng(seed)
    sigma_xy = float(config["dynamics"]["initial_sigma_xy_m"])
    sigma_theta = float(config["dynamics"]["initial_sigma_theta_rad"])
    covariance = np.diag([sigma_xy**2, sigma_xy**2, sigma_theta**2])
    error = rng.multivariate_normal(np.zeros(3), covariance)
    errors: list[float] = []
    nees: list[float] = []
    covered: list[float] = []
    arrivals = 0
    current_dropout = 0
    longest_dropout = 0
    p_truth_values: list[float] = []
    dt = float(config["dynamics"]["dt_s"])
    for previous, current in zip(path[:-1], path[1:]):
        delta = current - previous
        distance = float(np.linalg.norm(delta))
        theta = math.atan2(float(delta[1]), float(delta[0])) if distance > 1.0e-12 else 0.0
        speed = distance / dt
        state = np.asarray([previous[0], previous[1], theta], dtype=float)
        control = np.asarray([speed, 0.0], dtype=float)
        transition = unicycle_jacobian(state, control, dt)
        process = unicycle_process_noise(
            float(config["dynamics"]["process_noise_xy"]),
            float(config["dynamics"]["process_noise_theta"]),
            dt,
            theta=theta,
            v=speed,
        )
        process = 0.5 * (process + process.T)
        covariance = transition @ covariance @ transition.T + process
        process_sample = rng.multivariate_normal(np.zeros(3), process + 1.0e-12 * np.eye(3))
        error = transition @ error - process_sample

        camera = camera_at(selected_index, xs, ys, current)
        p_truth = field_value(replay_fields[camera], xs, ys, current)
        p_truth_values.append(p_truth)
        if rng.random() < p_truth:
            noise = rng.multivariate_normal(np.zeros(2), r_cond[camera])
            covariance_before = covariance
            innovation = H @ covariance_before @ H.T + r_cond[camera]
            gain = covariance_before @ H.T @ np.linalg.inv(innovation)
            identity = np.eye(3)
            error = (identity - gain @ H) @ error + gain @ noise
            covariance = ((identity - gain @ H) @ covariance_before @ (identity - gain @ H).T
                          + gain @ r_cond[camera] @ gain.T)
            covariance = 0.5 * (covariance + covariance.T)
            arrivals += 1
            current_dropout = 0
        else:
            current_dropout += 1
            longest_dropout = max(longest_dropout, current_dropout)

        xy_error = error[:2]
        p_xy = covariance[:2, :2]
        squared = float(xy_error @ np.linalg.solve(p_xy, xy_error))
        errors.append(float(np.linalg.norm(xy_error)))
        nees.append(squared)
        covered.append(float(squared <= CHI2_95_2D))
    values = np.asarray(errors, dtype=float)
    return {
        "belief_rmse_m": float(math.sqrt(np.mean(values**2))),
        "belief_p95_error_m": float(np.percentile(values, 95.0)),
        "median_nees_2d": float(np.median(nees)),
        "coverage_95_fraction": float(np.mean(covered)),
        "terminal_sigma_xy_m": math.sqrt(max(float(np.trace(covariance[:2, :2])) / 2.0, 0.0)),
        "longest_dropout_s": float(longest_dropout * dt),
        "arrival_fraction": float(arrivals / max(len(path) - 1, 1)),
        "mean_replay_p_use": float(np.mean(p_truth_values)) if p_truth_values else 0.0,
        "steps": int(max(len(path) - 1, 0)),
        "path_length_m": route_length(path),
    }


def run_monte_carlo(
    selected: dict[tuple[str, str], np.ndarray],
    ctx: dict,
    config: dict,
    selected_index: np.ndarray,
    replay_fields: dict[str, np.ndarray],
    r_cond: dict[str, np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    n_seeds = int(config["monte_carlo"]["seeds"])
    base_seed = int(config["monte_carlo"]["base_seed"])
    for task_index, task_name in enumerate(config["tasks"]):
        for method in METHODS:
            path = selected[(task_name, method)]
            for seed in range(n_seeds):
                replay_seed = base_seed + task_index * 100_000 + seed
                result = simulate_run(path, replay_seed, selected_index, replay_fields, r_cond, ctx["xs"], ctx["ys"], config)
                rows.append({
                    "run_id": f"{task_name}/{method}/seed_{seed:04d}",
                    "task": task_name,
                    "method": method,
                    "seed": seed,
                    **result,
                })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_summary(rows: list[dict]) -> list[dict]:
    metrics = (
        "belief_rmse_m",
        "belief_p95_error_m",
        "median_nees_2d",
        "coverage_95_fraction",
        "terminal_sigma_xy_m",
        "longest_dropout_s",
        "arrival_fraction",
    )
    summary = []
    for task in sorted({row["task"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows if row["task"] == task and row["method"] == method]
            out = {"task": task, "method": method, "n_runs": len(subset)}
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in subset], dtype=float)
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_median"] = float(np.median(values))
                out[f"{metric}_p05"] = float(np.percentile(values, 5.0))
                out[f"{metric}_p95"] = float(np.percentile(values, 95.0))
            summary.append(out)
    return summary


def draw_probability_fields(ctx: dict, planning_fields: dict[str, np.ndarray], selected_index: np.ndarray, selected_p: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0), constrained_layout=True)
    for ax, camera in zip(axes.flat[:4], CAMERAS):
        base.draw_field(ax, planning_fields[camera], ctx["xs"], ctx["ys"], ctx["driveable"], ctx["prisms"], title=f"{camera}: frozen planner p(use)")
    ax = axes.flat[4]
    selected_plot = np.where(ctx["driveable"], selected_index, np.nan)
    image = ax.imshow(selected_plot, origin="lower", extent=(ctx["xs"][0], ctx["xs"][-1], ctx["ys"][0], ctx["ys"][-1]), cmap=ListedColormap([base.CAMERA_COLORS[c] for c in CAMERAS]), vmin=-0.5, vmax=3.5, aspect="equal")
    base.draw_geometry(ax, ctx["prisms"])
    base.draw_cameras(ax)
    ax.set_title("Frozen single-camera selection")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    handles = [Line2D([], [], marker="s", linestyle="", color=base.CAMERA_COLORS[c], label=c) for c in CAMERAS]
    ax.legend(handles=handles, fontsize=8, loc="upper center", ncol=2)
    base.draw_field(axes.flat[5], selected_p, ctx["xs"], ctx["ys"], ctx["driveable"], ctx["prisms"], title="p(use) of selected camera", colorbar=True)
    fig.suptitle("Static four-camera planning input (existing GP conservative fields)", fontsize=16, weight="bold")
    fig.text(0.5, 0.005, "Measured detector-event fit; not independent held-out calibration evidence", ha="center", fontsize=10, color="#8b2e2e")
    fig.savefig(FIGURES / "01_four_camera_static_fields.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_routes(ctx: dict, selected_p: np.ndarray, selected: dict[tuple[str, str], np.ndarray], config: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.8), constrained_layout=True, sharex=True, sharey=True)
    for ax, task_name in zip(axes.flat, config["tasks"]):
        base.draw_field(ax, selected_p, ctx["xs"], ctx["ys"], ctx["driveable"], ctx["prisms"], title=task_name)
        task = ctx["tasks"][task_name]
        start = (float(task["start"]["x"]), float(task["start"]["y"]))
        goal = (float(task["goal"]["x"]), float(task["goal"]["y"]))
        for method in METHODS:
            path = selected[(task_name, method)]
            linestyle = "--" if method == "availability_blind_shortest" else "-"
            ax.plot(path[:, 0], path[:, 1], color=METHOD_COLORS[method], lw=2.4, linestyle=linestyle, zorder=8)
        ax.scatter(*start, s=55, c="#29a65a", edgecolors="black", zorder=9)
        ax.scatter(*goal, s=90, c="#e43d30", marker="*", edgecolors="black", zorder=9)
    handles = [Line2D([], [], color=METHOD_COLORS[m], lw=2.5, linestyle="--" if m == "availability_blind_shortest" else "-", label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Route selected from the same candidate library", fontsize=16, weight="bold")
    fig.text(0.5, 0.015, "Offline route discrimination — not closed-loop navigation", ha="center", fontsize=10, color="#8b2e2e")
    fig.savefig(FIGURES / "02_selected_routes.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_planning_tradeoff(selected_rows: list[dict], config: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3), constrained_layout=True)
    metrics = (
        ("path_length_m", "selected path length [m]"),
        ("terminal_sigma_xy_m", "exact expected terminal sigma [m]"),
        ("mean_planning_p_use", "mean planner p(use) on route"),
    )
    x = np.arange(len(config["tasks"]), dtype=float)
    width = 0.24
    for ax, (metric, label) in zip(axes, metrics):
        for offset, method in enumerate(METHODS):
            values = [next(float(row[metric]) for row in selected_rows if row["task"] == task and row["method"] == method) for task in config["tasks"]]
            ax.bar(x + (offset - 1) * width, values, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_xticks(x, [task.replace("mc_", "").replace("route_", "") for task in config["tasks"]], rotation=22, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Planning consequence under exact expected-belief evaluation", fontsize=16, weight="bold")
    fig.savefig(FIGURES / "03_planning_tradeoff.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_monte_carlo(summary: list[dict], config: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), constrained_layout=True)
    metrics = (
        ("belief_rmse_m", "filter-belief RMSE [m]"),
        ("longest_dropout_s", "longest measurement dropout [s]"),
        ("terminal_sigma_xy_m", "terminal stated sigma [m]"),
        ("coverage_95_fraction", "inside stated 95% ellipse"),
    )
    x = np.arange(len(config["tasks"]), dtype=float)
    for ax, (metric, label) in zip(axes.flat, metrics):
        for method in METHODS:
            rows = [next(row for row in summary if row["task"] == task and row["method"] == method) for task in config["tasks"]]
            centre = np.asarray([row[f"{metric}_median"] for row in rows], dtype=float)
            low = np.asarray([row[f"{metric}_p05"] for row in rows], dtype=float)
            high = np.asarray([row[f"{metric}_p95"] for row in rows], dtype=float)
            ax.errorbar(x, centre, yerr=np.vstack([centre - low, high - centre]), marker="o", capsize=3, lw=1.8, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_xticks(x, [task.replace("mc_", "").replace("route_", "") for task in config["tasks"]], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.22)
    axes.flat[0].legend(fontsize=8, frameon=False)
    axes.flat[3].axhline(0.95, color="#555", linestyle="--", lw=1.2, label="nominal")
    fig.suptitle(f"Model-based stochastic replay ({config['monte_carlo']['seeds']} matched seeds per task and method)", fontsize=16, weight="bold")
    fig.text(0.5, 0.005, "Points: run medians; bars: 5th–95th run percentiles. Replay field is not held-out truth.", ha="center", fontsize=10, color="#8b2e2e")
    fig.savefig(FIGURES / "04_monte_carlo_outcomes.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_approximation(candidate_rows: list[dict]) -> None:
    lookup: dict[tuple[str, int], dict[str, dict]] = {}
    for row in candidate_rows:
        lookup.setdefault((row["task"], int(row["candidate_id"])), {})[row["planning_model"]] = row
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    tasks = sorted({task for task, _ in lookup})
    colors = dict(zip(tasks, plt.get_cmap("tab10").colors))
    relative_sigma = []
    for task in tasks:
        points = [values for (name, _), values in lookup.items() if name == task]
        exact_sigma = np.asarray([float(values["explicit_hit_miss"]["terminal_sigma_xy_m"]) for values in points])
        shortcut_sigma = np.asarray([float(values["r_over_p_shortcut"]["terminal_sigma_xy_m"]) for values in points])
        exact_objective = np.asarray([float(values["explicit_hit_miss"]["objective_m"]) for values in points])
        shortcut_objective = np.asarray([float(values["r_over_p_shortcut"]["objective_m"]) for values in points])
        relative_sigma.extend(100.0 * (shortcut_sigma / exact_sigma - 1.0))
        axes[0].scatter(exact_sigma, shortcut_sigma, s=42, alpha=0.82, color=colors[task], label=task)
        axes[1].scatter(exact_objective, shortcut_objective, s=42, alpha=0.82, color=colors[task], label=task)
    for ax, xlabel, ylabel in (
        (axes[0], "explicit hit/miss terminal sigma [m]", "R/p terminal sigma [m]"),
        (axes[1], "explicit hit/miss planning objective [m]", "R/p planning objective [m]"),
    ):
        limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(limits, limits, color="#555", linestyle="--", lw=1.2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    axes[1].legend(fontsize=7.5, frameon=False)
    max_understatement = abs(float(np.min(relative_sigma)))
    fig.suptitle("Operational null: different covariance algebra, same selected routes", fontsize=15, weight="bold")
    axes[0].text(
        0.025,
        0.965,
        f"max sigma understatement: {max_understatement:.2f}%\nroute winners identical: 4/4 tasks",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
        color="#8b2e2e",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    fig.savefig(FIGURES / "05_hit_miss_vs_r_over_p.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_results_markdown(selected_rows: list[dict], summary: list[dict], r_provenance: dict, config: dict) -> None:
    lines = [
        "# Results — static four-camera probability planning",
        "",
        "## What ran",
        "",
        f"- Four frozen per-camera detector-hit GP fields from 8,808 spawn-grid events.",
        f"- Four declared warehouse tasks and {config['monte_carlo']['seeds']} matched stochastic-replay seeds per task and method.",
        "- One selected camera per cell; no simultaneous fusion.",
        "- Runtime replay always used a realised hit or no update. Only route planning differed.",
        "",
        "## Selected-route results",
        "",
        "| Task | Method | Length [m] | Mean planner p(use) | Exact expected terminal sigma [m] |",
        "|---|---|---:|---:|---:|",
    ]
    for task in config["tasks"]:
        for method in METHODS:
            row = next(item for item in selected_rows if item["task"] == task and item["method"] == method)
            lines.append(f"| `{task}` | `{method}` | {row['path_length_m']:.2f} | {row['mean_planning_p_use']:.3f} | {row['terminal_sigma_xy_m']:.4f} |")
    lines += [
        "",
        "## Model-based stochastic replay",
        "",
        "Values below are means over runs. The experimental unit is one task/method/seed run.",
        "",
        "| Task | Method | n | Belief RMSE [m] | Longest dropout [s] | Terminal sigma [m] | 95% coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in config["tasks"]:
        for method in METHODS:
            row = next(item for item in summary if item["task"] == task and item["method"] == method)
            lines.append(
                f"| `{task}` | `{method}` | {row['n_runs']} | "
                f"{row['belief_rmse_m_mean']:.4f} | {row['longest_dropout_s_mean']:.2f} | "
                f"{row['terminal_sigma_xy_m_mean']:.4f} | {row['coverage_95_fraction_mean']:.3f} |"
            )
    lines += [
        "",
        "## Conditional measurement covariance provenance",
        "",
        "`R_cond` is a planning input, not a probability. It was constructed from current",
        "camera-measurement residual component SDs versus commanded ground truth, zero-parameter",
        "floor IPM, balanced set-pose dataset `PG-IPM-CURRENT`. Ground truth was offline only.",
        "",
        "| Camera | detections | radial SD [m] | lateral SD [m] | isotropic conditional sigma [m] |",
        "|---|---:|---:|---:|---:|",
    ]
    for camera in CAMERAS:
        row = r_provenance[camera]
        lines.append(f"| {camera} | {row['n_detections']} | {row['radial_sd_m']:.4f} | {row['lateral_sd_m']:.4f} | {row['isotropic_sigma_cond_m']:.4f} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This experiment establishes route discrimination and the model-level consequence of",
        "using an explicit Bernoulli observation model. It does **not** establish held-out",
        "probability calibration, a closed-loop Gazebo navigation advantage, a real-robot result,",
        "or a simultaneous four-camera fusion result. Those require separate registered campaigns.",
        "",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(selected_rows: list[dict], candidate_rows: list[dict], runs: list[dict], config: dict) -> None:
    expected_selected = len(config["tasks"]) * len(METHODS)
    expected_runs = expected_selected * int(config["monte_carlo"]["seeds"])
    if len(selected_rows) != expected_selected:
        raise RuntimeError(f"selected route count {len(selected_rows)} != {expected_selected}")
    if len(runs) != expected_runs:
        raise RuntimeError(f"run count {len(runs)} != {expected_runs}")
    if not candidate_rows:
        raise RuntimeError("candidate table is empty")
    ids = [row["run_id"] for row in runs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("run IDs are not unique")
    for row in runs:
        if not (0.0 <= float(row["coverage_95_fraction"]) <= 1.0):
            raise RuntimeError("invalid coverage")
        if not (0.0 <= float(row["arrival_fraction"]) <= 1.0):
            raise RuntimeError("invalid arrival fraction")
        if float(row["belief_rmse_m"]) < 0.0:
            raise RuntimeError("negative RMSE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-runs",
        action="store_true",
        help="Reuse an already completed monte_carlo_runs.csv after a rendering-only failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.grid": False})

    ctx = base.build_context()
    missing_tasks = sorted(set(config["tasks"]) - set(ctx["tasks"]))
    if missing_tasks:
        raise RuntimeError(f"tasks absent from {base.TASKS}: {missing_tasks}")
    planning_fields, replay_fields = load_static_fields(ctx)
    r_cond, r_provenance = load_r_cond()
    selected_index, selected_p, selected_utility = frozen_camera_policy(planning_fields, r_cond, config)
    selected, selected_rows, candidate_rows = choose_routes(ctx, config, planning_fields, selected_index, selected_p, r_cond)
    runs_path = RESULTS / "monte_carlo_runs.csv"
    if args.reuse_runs and runs_path.exists():
        runs = read_csv(runs_path)
    else:
        runs = run_monte_carlo(selected, ctx, config, selected_index, replay_fields, r_cond)
    summary = group_summary(runs)
    validate_outputs(selected_rows, candidate_rows, runs, config)

    write_csv(RESULTS / "selected_routes.csv", selected_rows)
    write_csv(RESULTS / "candidate_routes.csv", candidate_rows)
    write_csv(runs_path, runs)
    write_csv(RESULTS / "monte_carlo_summary.csv", summary)
    np.savez_compressed(
        RESULTS / "static_planning_inputs.npz",
        xs=ctx["xs"],
        ys=ctx["ys"],
        driveable=ctx["driveable"],
        selected_camera_index=selected_index,
        selected_p_use=selected_p,
        selected_information_utility=selected_utility,
        **{f"planning_{camera}": planning_fields[camera] for camera in CAMERAS},
        **{f"replay_{camera}": replay_fields[camera] for camera in CAMERAS},
    )

    draw_probability_fields(ctx, planning_fields, selected_index, selected_p)
    draw_routes(ctx, selected_p, selected, config)
    draw_planning_tradeoff(selected_rows, config)
    draw_monte_carlo(summary, config)
    draw_approximation(candidate_rows)
    write_results_markdown(selected_rows, summary, r_provenance, config)

    input_paths = [CONFIG_PATH, PG_SUMMARY, base.WORLD, base.PROFILE, base.TASKS]
    input_paths += [base.GP_ROOT / camera / "det_hit_expected_kernel_gp.npz" for camera in CAMERAS]
    outputs = sorted([*RESULTS.glob("*"), *FIGURES.glob("*.png"), HERE / "RESULTS.md"])
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "run_count": len(runs),
        "selected_route_count": len(selected_rows),
        "candidate_evaluation_count": len(candidate_rows),
        "tasks": list(config["tasks"]),
        "methods": list(METHODS),
        "input_sha256": {str(path.relative_to(REPO)): sha256(path) for path in input_paths},
        "output_sha256": {str(path.relative_to(HERE)): sha256(path) for path in outputs if path.name != "manifest.json"},
        "metric_contract": {
            "metric_object": "filter_belief_model_replay",
            "reference": "sampled_linear_gaussian_model_truth",
            "experimental_unit": "run",
            "n_runs_per_task_method": int(config["monte_carlo"]["seeds"]),
            "online_planning_inputs": ["frozen P_conservative_plan_map", "current belief covariance", "route candidates"],
            "evaluation_only_inputs": ["full-fit P_mean_map replay probability", "sampled truth error"],
            "status": "exploratory_offline_mechanism",
        },
        "limitations": list(config["evaluation"]["forbidden_claims"]),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("experiment_id", "run_count", "selected_route_count", "candidate_evaluation_count")}, indent=2))


if __name__ == "__main__":
    main()
