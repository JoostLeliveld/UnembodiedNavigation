#!/usr/bin/env python3
"""Offline gate for waypoint-free hard routes in the four-camera warehouse.

This uses the real planner, GP artifact, world collision geometry, and task
start/goal poses, but never starts Gazebo. Task ``waypoints`` are intentionally
ignored. Route seeds are generated from the map alone and are identical across
measurement conditions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
for source_dir in ("planning", "unav_common", "experiments"):
    sys.path.insert(0, str(REPO / "src" / source_dir))

from experiments.core.world_profiles import (  # noqa: E402
    compute_look_at_from_pose,
    load_profile,
    serialize_collision_geometry_from_world,
    serialize_driveable_geometry_from_profile,
    serialize_occlusion_geometry_from_world,
)
from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402
from unav_common.lane_graph_routes import generate_route_seeds  # noqa: E402
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402

DEFAULT_CONFIG = REPO / "scripts" / "visibility_comparison" / "_rob_val.yaml"
TASKS_PATH = REPO / "src" / "experiments" / "config" / "tasks.yaml"
PROFILES_PATH = REPO / "src" / "experiments" / "config" / "world_profiles.yaml"
DEFAULT_OUT = REPO / "logs" / "studies" / "multicam_nav_demo" / "offline_hard_routes"
TASK_NAMES = ("rob_hardA", "rob_hardB")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _task(world: str, name: str) -> dict:
    tasks = (_load_yaml(TASKS_PATH).get("tasks") or {}).get(world) or []
    return next(task for task in tasks if task.get("name") == name)


def _camera_params(world: str) -> tuple[dict, dict, str]:
    profile, intrinsics, world_path, camera_pose = load_profile(
        str(PROFILES_PATH), world
    )
    cam_pos = [float(v) for v in camera_pose[:3]]
    look_at = compute_look_at_from_pose(
        cam_pos,
        float(camera_pose[3]),
        float(camera_pose[4]),
        float(camera_pose[5]),
    )
    return {
        "cam_pos": cam_pos,
        "look_at": look_at,
        "img_width": int(intrinsics["img_width"]),
        "img_height": int(intrinsics["img_height"]),
        "fov_h_rad": float(intrinsics["fov_h_rad"]),
    }, profile, world_path


def _planner(
    cfg: dict,
    *,
    horizon: int,
    seed: int,
    camera_params: dict,
    visibility_geometry_json: str,
    collision_geometry_json: str,
    driveable_geometry_json: str,
    route_seeds: list[dict],
    optimizer_maxiter: int,
) -> UnicyclePlannerBase:
    gp_path = Path(str(cfg["gp_artifact"])).expanduser()
    if not gp_path.is_absolute():
        gp_path = (REPO / gp_path).resolve()
    return UnicyclePlannerBase(
        horizon=horizon,
        dt=float(cfg["global_dt"]),
        v_min=-float(cfg["v_max"]),
        v_max=float(cfg["v_max"]),
        w_min=-float(cfg.get("w_max", 1.0)),
        w_max=float(cfg.get("w_max", 1.0)),
        control_weight=float(cfg.get("control_weight", 0.0)),
        process_noise_xy=float(cfg.get("process_noise_xy", 0.01)),
        process_noise_theta=float(cfg.get("process_noise_theta", 0.02)),
        obs_noise_uv=float(cfg.get("r_visible_uv", 2.5)),
        goal_sigma_uv=float(cfg.get("goal_prior_u_std_final", 20.0)),
        risk_weight_obs=float(cfg.get("risk_weight_obs", 1.0)),
        ambiguity_weight=float(cfg.get("ambiguity_weight", 6.0)),
        optimizer_maxiter=int(optimizer_maxiter),
        optimizer_maxfun=max(
            int(cfg.get("optimizer_maxfun", 900)),
            int(optimizer_maxiter) * 6,
        ),
        optimizer_ftol=float(cfg.get("optimizer_ftol", 1e-6)),
        optimizer_gtol=float(cfg.get("optimizer_gtol", 1e-4)),
        optimizer_warm_start=_as_bool(cfg.get("optimizer_warm_start", True)),
        optimizer_multistart=True,
        optimizer_multistart_include_direct=False,
        optimizer_initial_routes_json=json.dumps(route_seeds),
        approx_method=str(cfg.get("approx_method", "ET1")),
        use_obs_risk=True,
        use_ambiguity=_as_bool(cfg.get("global_use_ambiguity", True)),
        seed=seed,
        camera_params=camera_params,
        use_visibility_model=True,
        visibility_target_height_m=float(cfg.get("visibility_target_height_m", 0.35)),
        visibility_geometry_json=visibility_geometry_json,
        collision_geometry_json=collision_geometry_json,
        visibility_artifact_path=str(gp_path),
        r_visible_uv=float(cfg.get("r_visible_uv", 2.5)),
        r_miss_uv=float(cfg.get("r_miss_uv", 40.0)),
        visibility_sigma_kappa=float(cfg.get("visibility_sigma_kappa", 1.0)),
        goal_prior_u_std_start=float(cfg.get("goal_prior_u_std_start", 50.0)),
        goal_prior_v_std_start=float(cfg.get("goal_prior_v_std_start", 50.0)),
        goal_prior_u_std_final=float(cfg.get("goal_prior_u_std_final", 12.0)),
        goal_prior_v_std_final=float(cfg.get("goal_prior_v_std_final", 12.0)),
        goal_tightening_power=float(cfg.get("goal_tightening_power", 0.9)),
        goal_progress_n_steps=int(cfg.get("goal_progress_n_steps", 90)),
        observation_risk_scale=float(cfg.get("observation_risk_scale", 1.0)),
        ambiguity_term_scale=float(cfg.get("ambiguity_term_scale", 1.0)),
        discount_gamma=float(cfg.get("discount_gamma", 0.995)),
        use_nogo_cost=_as_bool(cfg.get("use_nogo_cost", True)),
        nogo_penalty_type=str(cfg.get("nogo_penalty_type", "warning_band")),
        nogo_weight=float(cfg.get("nogo_weight", 2000.0)),
        nogo_safe_distance=float(cfg.get("nogo_safe_distance", 0.25)),
        nogo_logbarrier_eps=float(cfg.get("nogo_logbarrier_eps", 0.01)),
        nogo_warning_band=float(cfg.get("nogo_warning_band", 0.05)),
        nogo_near_weight=float(cfg.get("nogo_near_weight", 50.0)),
        use_belief_nogo_cost=_as_bool(cfg.get("use_belief_nogo_cost", True)),
        nogo_belief_kappa=float(cfg.get("nogo_belief_kappa", 1.0)),
        nogo_mode=str(cfg.get("nogo_mode", "keep_out")),
        driveable_geometry_json=driveable_geometry_json,
        robot_collision_radius_m=float(cfg.get("robot_collision_radius_m", 0.125)),
        runtime_debug=False,
    )


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def _max_line_deviation(points: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    line = goal - start
    denom = float(np.linalg.norm(line))
    if denom <= 1e-12:
        return 0.0
    offsets = points[:, :2] - start[None, :]
    return float(np.max(np.abs(line[1] * offsets[:, 0] - line[0] * offsets[:, 1]) / denom))


def _total_heading_change(points: np.ndarray) -> float:
    return float(np.sum(np.abs(np.arctan2(
        np.sin(np.diff(points[:, 2])),
        np.cos(np.diff(points[:, 2])),
    ))))


def _draw_world(ax, collision_geometry_json: str) -> None:
    for prism in scene_from_json(collision_geometry_json).prisms:
        name = str(prism.name)
        if "wall_" in name:
            continue
        ax.add_patch(Rectangle(
            (prism.xmin, prism.ymin),
            prism.xmax - prism.xmin,
            prism.ymax - prism.ymin,
            facecolor="#5c6770",
            edgecolor="#2f3438",
            linewidth=0.5,
            alpha=0.85,
            zorder=2,
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-horizon", type=int, default=75)
    parser.add_argument("--fixed-horizon", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer-maxiter", type=int, default=120)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    world = str(cfg["world"])
    camera_params, profile, world_path = _camera_params(world)
    collision_json = serialize_collision_geometry_from_world(world_path)
    visibility_json = serialize_occlusion_geometry_from_world(world_path)
    driveable_json = serialize_driveable_geometry_from_profile(profile)
    clearance = float(cfg.get("robot_collision_radius_m", 0.125)) + float(
        cfg.get("nogo_safe_distance", 0.25)
    ) + float(cfg["v_max"]) * float(cfg["global_dt"])
    sigma_xy = float(cfg.get("init_belief_sigma_xy", 0.05))
    sigma_theta = float(cfg.get("init_belief_sigma_theta", 0.05))
    S0 = np.diag([sigma_xy**2, sigma_xy**2, sigma_theta**2])

    results = []
    trajectories = {}
    seed_paths = {}
    endpoints = {}
    for task_name in TASK_NAMES:
        task = _task(world, task_name)
        start = np.array([
            float(task["start"]["x"]),
            float(task["start"]["y"]),
            float(task["start"]["yaw"]),
        ])
        goal = np.array([float(task["goal"]["x"]), float(task["goal"]["y"])])
        # Paper-1 seeding only: lane-graph Manhattan routes over the drivable
        # region map. No collision-map fallback -- that was a fork from the
        # published method and is gone.
        route_seeds = generate_route_seeds(driveable_json, start[:2], goal)
        legacy_seeds = route_seeds
        if not route_seeds:
            raise RuntimeError(
                f"lane-graph seed generation produced no route for {task_name}; "
                "the drivable region map does not connect start to goal"
            )
        seed_points = np.asarray(
            [start[:2], *route_seeds[0]["waypoints"]], dtype=float
        )
        seed_length = _path_length(seed_points)
        seed_paths[task_name] = seed_points
        endpoints[task_name] = np.asarray([start[:2], goal], dtype=float)

        for label, horizon in (
            ("old_horizon", args.baseline_horizon),
            ("offline_fix", args.fixed_horizon),
        ):
            planner = _planner(
                cfg,
                horizon=horizon,
                seed=args.seed,
                camera_params=camera_params,
                visibility_geometry_json=visibility_json,
                collision_geometry_json=collision_json,
                driveable_geometry_json=driveable_json,
                route_seeds=route_seeds,
                optimizer_maxiter=(
                    int(cfg.get("optimizer_maxiter", 60))
                    if label == "old_horizon"
                    else args.optimizer_maxiter
                ),
            )
            seed_controls = planner._controls_for_waypoints(
                start, route_seeds[0]["waypoints"]
            ).reshape(horizon, 2)
            seed_eval = planner.evaluate_rollout_controls(
                start, S0, goal, seed_controls
            )
            started = time.perf_counter()
            plan = planner.plan(start, S0, goal)
            elapsed = time.perf_counter() - started
            states = np.asarray(plan.states, dtype=float)
            result = {
                "task": task_name,
                "setting": label,
                "horizon": int(horizon),
                "optimizer_maxiter": int(
                    cfg.get("optimizer_maxiter", 60)
                    if label == "old_horizon"
                    else args.optimizer_maxiter
                ),
                "lookahead_s": float(horizon * planner.dt),
                "kinematic_reach_m": float(horizon * planner.dt * planner.v_max),
                "task_waypoints_used": False,
                "legacy_driveable_only_seed_count": len(legacy_seeds),
                "collision_aware_seed_count": len(route_seeds),
                "seed_name": str(route_seeds[0]["name"]),
                "seed_path_length_m": seed_length,
                "seed_rollout_valid": bool(seed_eval["rollout_valid"]),
                "seed_terminal_goal_distance_m": float(
                    seed_eval["terminal_goal_distance_pred"]
                ),
                "seed_min_obstacle_clearance_m": float(
                    seed_eval["min_predicted_obstacle_distance_m"]
                ),
                "selected_source": str(plan.selected_source),
                "optimizer_success": bool(plan.optimizer_success),
                "rollout_valid": bool(plan.rollout_valid),
                "terminal_goal_distance_m": float(plan.terminal_goal_distance_pred),
                "min_obstacle_clearance_m": float(plan.min_predicted_obstacle_distance_m),
                "planned_path_length_m": _path_length(states),
                "max_straight_line_deviation_m": _max_line_deviation(
                    states, start[:2], goal
                ),
                "total_heading_change_rad": _total_heading_change(states),
                "solve_time_s": elapsed,
                "goal_gate_pass": bool(
                    plan.rollout_valid
                    and plan.terminal_goal_distance_pred
                    <= float(cfg.get("goal_success_radius", 0.25))
                ),
                "optimizer_message": str(plan.optimizer_message),
            }
            results.append(result)
            trajectories[(task_name, label)] = states
            print(json.dumps(result, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "world": world,
        "config": str(args.config),
        "route_policy": (
            "start+goal only; task waypoints ignored; map-derived collision-aware "
            "A* seed; no visibility/measurement data used in seed generation"
        ),
        "clearance_m": clearance,
        "results": results,
    }
    json_path = args.output_dir / "offline_multicam_hard_route_gate.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Export the solved trajectories so downstream figures can plot the routes
    # without re-solving. Keys are "<task>__<setting>"; the collision geometry
    # travels with them so a renderer needs nothing but this file.
    trajectory_path = args.output_dir / "offline_multicam_hard_route_trajectories.npz"
    np.savez_compressed(
        trajectory_path,
        collision_geometry_json=collision_json,
        **{f"{task}__{label}": states for (task, label), states in trajectories.items()},
        **{f"{task}__seed": pts for task, pts in seed_paths.items()},
        **{f"{task}__endpoints": pts for task, pts in endpoints.items()},
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), sharex=True, sharey=True)
    colors = {"old_horizon": "#d95f02", "offline_fix": "#0072b2"}
    for row, task_name in enumerate(TASK_NAMES):
        task = _task(world, task_name)
        start = np.array([task["start"]["x"], task["start"]["y"]], dtype=float)
        goal = np.array([task["goal"]["x"], task["goal"]["y"]], dtype=float)
        for col, label in enumerate(("old_horizon", "offline_fix")):
            ax = axes[row, col]
            _draw_world(ax, collision_json)
            states = trajectories[(task_name, label)]
            result = next(
                r for r in results
                if r["task"] == task_name and r["setting"] == label
            )
            ax.plot(
                states[:, 0],
                states[:, 1],
                color=colors[label],
                linewidth=2.2,
                zorder=5,
            )
            ax.scatter(*start, s=55, color="#009e73", edgecolor="white", zorder=6)
            ax.scatter(*goal, s=100, marker="*", color="#cc79a7", edgecolor="white", zorder=6)
            ax.plot(
                [start[0], goal[0]],
                [start[1], goal[1]],
                linestyle=":",
                color="#777777",
                linewidth=1.0,
                zorder=1,
            )
            ax.set_title(
                f"{task_name} · {label.replace('_', ' ')}\n"
                f"H={result['horizon']}, valid={result['rollout_valid']}, "
                f"goal gap={result['terminal_goal_distance_m']:.2f} m, "
                f"line deviation={result['max_straight_line_deviation_m']:.2f} m"
            )
            ax.set_aspect("equal")
            ax.set_xlim(-11.4, 11.7)
            ax.set_ylim(-8.8, 8.8)
            ax.grid(alpha=0.2)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
    fig.suptitle(
        "Offline four-camera hard-route gate — map-derived seeds, no mission waypoints",
        fontsize=14,
    )
    fig.tight_layout()
    figure_path = args.output_dir / "offline_multicam_hard_route_gate.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    print(f"wrote {json_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {trajectory_path}")


if __name__ == "__main__":
    main()
