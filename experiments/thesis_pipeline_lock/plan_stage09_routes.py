#!/usr/bin/env python3
"""Solve and freeze the canonical one-shot Stage-09 EFE route for each arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src/planning"), str(REPO / "src/reliability"),
                str(REPO / "src/unav_common"), str(REPO / "src/experiments")]
from experiments.core.world_profiles import (  # noqa: E402
    serialize_collision_geometry_from_world, serialize_driveable_geometry_from_profile,
)
from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402
from unav_common.lane_graph_routes import (  # noqa: E402
    generate_route_seeds, repair_route_seeds_for_footprint,
)

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CONDITIONS = (
    "global_intact", "global_removal", "per_camera_intact",
    "per_camera_removal", "spatial_intact", "spatial_removal",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planner(artifact: Path, collision: str, boundary: str, active: tuple[str, ...]) -> UnicyclePlannerBase:
    return UnicyclePlannerBase(
        horizon=75, dt=1.0, v_min=0.0, v_max=1.0, w_min=-1.0, w_max=1.0,
        control_weight=0.0, process_noise_xy=0.02, process_noise_theta=0.08,
        obs_noise_uv=2.5, goal_sigma_uv=2.0, risk_weight_obs=1.0,
        ambiguity_weight=1.0, discount_gamma=0.995,
        optimizer_maxiter=60, optimizer_gtol=1e-5,
        optimizer_warm_start=False, optimizer_maxfun=500, optimizer_ftol=1e-6,
        optimizer_terminal_goal_tolerance_m=0.35, optimizer_control_block_steps=1,
        optimizer_multistart=True, optimizer_multistart_include_direct=False, seed=900,
        camera_params={"cam_pos": [-6.0, -10.0, 5.0], "look_at": [0.0, 0.0, 0.0],
                       "img_width": 1280, "img_height": 720, "fov_h_rad": 1.2},
        use_visibility_model=True, camera_network_artifact_path=str(artifact),
        camera_network_camera_ids=",".join(CAMERAS),
        camera_network_active_camera_ids=",".join(active),
        camera_network_objective="metric_expected_belief", network_goal_std_m=0.10,
        network_goal_std_start_m=5.0, camera_network_updates_per_step=1,
        kouw_et1_ambiguity=True, terminal_risk_only=False, goal_tightening_power=0.9,
        observation_risk_scale=1.0, collision_geometry_json=collision,
        driveable_geometry_json=boundary, use_nogo_cost=True, nogo_mode="keep_in",
        nogo_weight=40.0, nogo_safe_distance=0.0, nogo_logbarrier_eps=0.05,
        nogo_warning_band=0.05, nogo_near_weight=50.0, use_belief_nogo_cost=True,
        nogo_belief_kappa=1.0, robot_collision_radius_m=math.hypot(0.4, 0.275),
        robot_length_m=0.80, robot_width_m=0.55,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help="comma-separated condition subset for development probes")
    args = parser.parse_args()
    campaign_path, output = args.campaign.resolve(), args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("global_planner_mode") != "efe":
        raise RuntimeError("canonical route planning requires global_planner_mode=efe")
    if args.task not in campaign["tasks"]:
        raise KeyError(args.task)
    task_file = REPO / campaign["tasks_yaml"]
    task_rows = yaml.safe_load(task_file.read_text(encoding="utf-8"))["tasks"]["warehouse_v2.world.sdf"]
    task = next(row for row in task_rows if row["name"] == args.task)
    start = np.asarray([task["start"][key] for key in ("x", "y", "yaw")], dtype=float)
    goal = np.asarray([task["goal"][key] for key in ("x", "y")], dtype=float)

    profile_path = REPO / campaign["world_profiles"]
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))["worlds"]["warehouse_v2.world.sdf"]
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    collision = serialize_collision_geometry_from_world(
        str(world), tuple(profile["collision_model_names"]),
        tuple(profile["collision_include_names"]), profile)
    boundary = serialize_driveable_geometry_from_profile(profile)
    declared_seeds = task.get("route_seeds")
    if declared_seeds:
        seeds = declared_seeds
    else:
        seeds = repair_route_seeds_for_footprint(
            generate_route_seeds(collision, start[:2], goal),
            collision, boundary, start,
            robot_length_m=0.80, robot_width_m=0.55,
            target_clearance_m=0.02,
        )
    if not seeds:
        raise RuntimeError("lane-graph seed generation returned no routes")
    prior = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(5.0) ** 2])

    requested_conditions = tuple(value.strip() for value in args.conditions.split(",")
                                 if value.strip())
    if not requested_conditions or not set(requested_conditions).issubset(CONDITIONS):
        raise ValueError(f"conditions must be a nonempty subset of {CONDITIONS}")
    results = {}
    for condition in requested_conditions:
        cfg = campaign["conditions"][condition]
        task_campaign_cfg = campaign["tasks"][args.task]
        task_override = (task_campaign_cfg.get("condition_overrides", {}) or {}).get(
            condition, {}) or {}
        artifact = Path(cfg["camera_network_artifact_path"])
        if digest(artifact) != cfg["camera_network_expected_sha256"]:
            raise RuntimeError(f"planning artifact hash drift for {condition}")
        active_text = task_override.get(
            "camera_network_active_camera_ids", cfg["camera_network_active_camera_ids"])
        active = tuple(value.strip() for value in active_text.split(","))
        model = planner(artifact, collision, boundary, active)
        model.optimizer_initial_routes = model._parse_initial_routes(json.dumps(seeds))
        result = model.plan(start, prior, goal)
        if not result.rollout_valid or result.terminal_goal_distance_pred > 0.35:
            raise RuntimeError(
                f"{condition}: invalid global plan ({result.invalid_reason}, "
                f"goal gap {result.terminal_goal_distance_pred:.3f} m)")
        states = np.asarray(result.states, dtype=float)
        distances = np.linalg.norm(states[:, :2] - goal[None, :], axis=1)
        arrival = int(np.flatnonzero(distances <= 0.35)[0]) if np.any(distances <= 0.35) else len(states) - 1
        route_path = staging / f"{condition}.npz"
        np.savez_compressed(route_path, states=states, controls=np.asarray(result.controls),
                            display_states=states[:arrival + 1], start=start, goal=goal)
        results[condition] = {
            "artifact": {"path": route_path.name, "sha256": digest(route_path)},
            "active_cameras": list(active), "selected_source": result.selected_source,
            "optimizer_success": bool(result.optimizer_success),
            "optimizer_status": int(result.optimizer_status),
            "optimizer_nit": int(result.optimizer_nit), "optimizer_nfev": int(result.optimizer_nfev),
            "optimizer_message": result.optimizer_message, "solve_time_s": result.solve_time_s,
            "total_cost": result.total_cost, "risk_cost": result.risk_cost,
            "ambiguity_cost": result.ambiguity_cost, "obstacle_cost": result.obstacle_cost,
            "terminal_goal_distance_pred_m": result.terminal_goal_distance_pred,
            "minimum_predicted_clearance_m": result.min_predicted_obstacle_distance_m,
            "rollout_valid": bool(result.rollout_valid), "arrival_state_index": arrival,
        }
        print(args.task, condition, result.selected_source,
              f"{result.solve_time_s:.1f}s", f"gap={result.terminal_goal_distance_pred:.3g}", flush=True)

    manifest = {
        "schema": ("thesis_stage09_offline_global_routes.v1"
                   if requested_conditions == CONDITIONS
                   else "thesis_stage09_route_probe.v1"),
        "status": ("frozen_before_gazebo_execution"
                   if requested_conditions == CONDITIONS else "development_probe"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task, "global_planner_mode": "efe",
        "purpose": "offline preview and deterministic verification of the same one-shot global EFE solve used at Gazebo startup",
        "navigation_execution_note": "campaign remains global_planner_mode=efe; these are not preselected-route inputs",
        "final_audit_used_for_route_selection": False,
        "final_audit_firewall": (
            "The audit was opened only after model freezing and is not read by this planner."
        ),
        "results": results,
        "source_hashes": {
            "campaign": digest(campaign_path), "tasks": digest(task_file),
            "world": digest(world), "profile": digest(profile_path),
            "implementation": digest(Path(__file__)),
        },
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": digest(manifest_path)}) + "\n")
    os.replace(staging, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
