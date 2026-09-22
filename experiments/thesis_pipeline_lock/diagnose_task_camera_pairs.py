#!/usr/bin/env python3
"""Score plausible route seeds for every task and single-camera removal."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src/planning"), str(REPO / "src/reliability"),
                str(REPO / "src/unav_common"), str(REPO / "src/experiments")]
from experiments.core.world_profiles import (  # noqa: E402
    serialize_collision_geometry_from_world, serialize_driveable_geometry_from_profile,
)
from planning.planners.base_planner import UnicyclePlannerBase, rollout_unicycle  # noqa: E402
from unav_common.lane_graph_routes import (  # noqa: E402
    generate_route_seeds, repair_route_seeds_for_footprint,
)

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
MODELS = ("spatial",)


def planner(artifact: Path, collision: str, boundary: str,
            active: tuple[str, ...]) -> UnicyclePlannerBase:
    return UnicyclePlannerBase(
        horizon=75, dt=1.0, v_min=0.0, v_max=1.0, w_min=-1.0, w_max=1.0,
        control_weight=0.0, process_noise_xy=0.02, process_noise_theta=0.08,
        obs_noise_uv=2.5, goal_sigma_uv=2.0, risk_weight_obs=1.0,
        ambiguity_weight=1.0, discount_gamma=0.995,
        optimizer_maxiter=60, optimizer_gtol=1e-5, optimizer_warm_start=False,
        optimizer_maxfun=500, optimizer_ftol=1e-6,
        optimizer_terminal_goal_tolerance_m=0.35, seed=900,
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
    parser.add_argument("--planning-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.planning_root.resolve()
    artifacts = {
        "global": root / "m0_planning_information.npz",
        "per_camera": root / "m1_planning_information.npz",
        "spatial": root / "m2_planning_information.npz",
    }
    profile_path = REPO / "src/experiments/config/world_profiles.yaml"
    profile = yaml.safe_load(profile_path.read_text())["worlds"]["warehouse_v2.world.sdf"]
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    collision = serialize_collision_geometry_from_world(
        str(world), tuple(profile["collision_model_names"]),
        tuple(profile["collision_include_names"]), profile)
    boundary = serialize_driveable_geometry_from_profile(profile)
    tasks = yaml.safe_load((
        REPO / "experiments/thesis_pipeline_lock/stage09_navigation_tasks.yaml"
    ).read_text())["tasks"][world.name]
    prior = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(5.0) ** 2])
    report = {"schema": "task_camera_pair_diagnostic.v1", "tasks": {}}
    ranked = []
    for task in tasks:
        start = np.asarray([task["start"][key] for key in ("x", "y", "yaw")])
        goal = np.asarray([task["goal"][key] for key in ("x", "y")])
        seeds = repair_route_seeds_for_footprint(
            generate_route_seeds(collision, start[:2], goal), collision, boundary, start,
            robot_length_m=0.80, robot_width_m=0.55, target_clearance_m=0.02)
        lengths = {}
        for seed in seeds:
            points = np.vstack((start[:2], np.asarray(seed["waypoints"], dtype=float)))
            lengths[seed["name"]] = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        states = {"intact": CAMERAS}
        states.update({f"remove_{camera[-1]}": tuple(c for c in CAMERAS if c != camera)
                       for camera in CAMERAS})
        task_rows = {}
        for model in MODELS:
            task_rows[model] = {}
            for state, active in states.items():
                engine = planner(artifacts[model], collision, boundary, active)
                candidates = []
                for seed in seeds:
                    controls = engine._controls_for_waypoints(
                        start, seed["waypoints"]).reshape(engine.horizon, 2)
                    total, metrics = engine._evaluate_metric_network_controls(
                        controls, start, prior, np.r_[goal, 0.0], return_metrics=True)
                    states_rollout = rollout_unicycle(start, controls, engine.dt)
                    goal_gap = float(np.linalg.norm(states_rollout[-1, :2] - goal))
                    candidates.append({
                        "route": seed["name"], "length_m": lengths[seed["name"]],
                        "cost": float(total),
                        "risk": float(metrics["risk_cost"]),
                        "ambiguity": float(metrics["ambiguity_cost"]),
                        "valid": not engine._trajectory_retraces_lane(states_rollout),
                        "goal_gap_m": goal_gap,
                    })
                feasible = [row for row in candidates
                            if row["valid"] and row["goal_gap_m"] <= 0.35]
                feasible.sort(key=lambda row: (row["cost"], row["route"]))
                task_rows[model][state] = {
                    "winner": feasible[0] if feasible else None,
                    "runner_up": feasible[1] if len(feasible) > 1 else None,
                    "candidates": candidates,
                }
        intact = task_rows["spatial"]["intact"]["winner"]
        for camera in CAMERAS:
            removal = task_rows["spatial"][f"remove_{camera[-1]}"]["winner"]
            if not all((intact, removal)):
                continue
            flip = intact["route"] != removal["route"]
            max_ratio = max(intact["length_m"], removal["length_m"]) / max(
                min(intact["length_m"], removal["length_m"]), 1e-9)
            row = {"task": task["name"], "removed_camera": camera,
                   "spatial_flip": flip,
                   "intact_route": intact["route"], "removal_route": removal["route"],
                   "intact_length_m": intact["length_m"],
                   "removal_length_m": removal["length_m"],
                   "length_ratio": max_ratio,
                   "removal_cost_delta": removal["cost"] - intact["cost"]}
            ranked.append(row)
        report["tasks"][task["name"]] = {
            "description": task["description"], "route_lengths_m": lengths,
            "scores": task_rows,
        }
    ranked.sort(key=lambda row: (not row["spatial_flip"],
                                 row["length_ratio"], row["task"], row["removed_camera"]))
    report["ranked_pairs"] = ranked
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for row in ranked:
        if row["spatial_flip"]:
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
