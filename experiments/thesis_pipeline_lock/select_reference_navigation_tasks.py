#!/usr/bin/env python3
"""Development-only route-selection pilot for the canonical six-arm campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
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
from unav_common.lane_graph_routes import generate_route_seeds, repair_route_seeds_for_footprint  # noqa: E402

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
REMOVED_CAMERA = "camera_E"
MODELS = ("global", "per_camera", "spatial")


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def planner(path, collision_json, boundary_json, active):
    return UnicyclePlannerBase(
        horizon=75, dt=1.0, v_min=0.0, v_max=1.0, w_min=-1.0, w_max=1.0,
        control_weight=0.0, process_noise_xy=0.02, process_noise_theta=0.08,
        obs_noise_uv=2.5, goal_sigma_uv=2.0, risk_weight_obs=1.0,
        ambiguity_weight=1.0, optimizer_maxiter=60, optimizer_gtol=1e-5,
        optimizer_warm_start=False, optimizer_maxfun=500, optimizer_ftol=1e-6,
        optimizer_terminal_goal_tolerance_m=0.35, seed=900,
        camera_params={"cam_pos": [-6.0, -10.0, 5.0], "look_at": [0.0, 0.0, 0.0],
                       "img_width": 1280, "img_height": 720, "fov_h_rad": 1.2},
        use_visibility_model=True, camera_network_artifact_path=str(path),
        camera_network_camera_ids=",".join(CAMERAS),
        camera_network_active_camera_ids=",".join(active),
        camera_network_objective="metric_expected_belief", network_goal_std_m=0.10,
        network_goal_std_start_m=5.0, camera_network_updates_per_step=1,
        kouw_et1_ambiguity=True, terminal_risk_only=False, goal_tightening_power=0.9,
        observation_risk_scale=1.0, collision_geometry_json=collision_json,
        driveable_geometry_json=boundary_json, use_nogo_cost=True, nogo_mode="keep_in",
        nogo_weight=40.0, nogo_safe_distance=0.0, nogo_logbarrier_eps=0.05,
        nogo_warning_band=0.05, nogo_near_weight=50.0, use_belief_nogo_cost=True,
        nogo_belief_kappa=1.0, robot_collision_radius_m=math.hypot(0.4, 0.275),
        robot_length_m=0.80, robot_width_m=0.55,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planning-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(); root, output = args.planning_root.resolve(), args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists(): raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)
    manifest = json.loads((root / "manifest.json").read_text())
    paths = {model: root / manifest["artifacts"][key]["path"]
             for model, key in zip(MODELS, ("M0", "M1", "M2"))}
    profile_path = REPO / "src/experiments/config/world_profiles.yaml"
    profile = yaml.safe_load(profile_path.read_text())["worlds"]["warehouse_v2.world.sdf"]
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    collision = serialize_collision_geometry_from_world(
        str(world), tuple(profile["collision_model_names"]),
        tuple(profile["collision_include_names"]), profile)
    boundary = serialize_driveable_geometry_from_profile(profile)
    task_path = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_tasks.yaml"
    tasks = yaml.safe_load(task_path.read_text())["tasks"][world.name]
    prior = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(5.0) ** 2])
    results = {}
    active_by_state = {"intact": CAMERAS,
                       "removal": tuple(c for c in CAMERAS if c != REMOVED_CAMERA)}
    for task in tasks:
        start = np.asarray([task["start"][x] for x in ("x", "y", "yaw")], dtype=float)
        goal = np.asarray([task["goal"][x] for x in ("x", "y")], dtype=float)
        seeds = repair_route_seeds_for_footprint(
            generate_route_seeds(collision, start[:2], goal), collision, boundary, start,
            robot_length_m=0.80, robot_width_m=0.55, target_clearance_m=0.02)
        task_result = {"description": task["description"], "candidate_count": len(seeds), "arms": {}}
        for model_name, path in paths.items():
            for state, active in active_by_state.items():
                arm = f"{model_name}_{state}"; engine = planner(path, collision, boundary, active)
                candidates = []
                for seed in seeds:
                    rollout = engine.evaluate_rollout_controls(
                        start, prior, goal, engine._controls_for_waypoints(start, seed["waypoints"]))
                    candidates.append({
                        "candidate": seed["name"], "total_cost": float(rollout["total_cost"]),
                        "rollout_valid": bool(rollout["rollout_valid"]),
                        "terminal_goal_distance_pred": float(rollout["terminal_goal_distance_pred"]),
                        "risk_cost": float(rollout.get("risk_cost", 0.0)),
                        "ambiguity_cost": float(rollout.get("ambiguity_cost", 0.0)),
                        "obstacle_cost": float(rollout.get("obstacle_cost", 0.0)),
                    })
                feasible = [x for x in candidates if x["rollout_valid"] and x["terminal_goal_distance_pred"] <= 0.35]
                task_result["arms"][arm] = {
                    "status": "selected" if feasible else "no_feasible_candidate",
                    "winner": None if not feasible else min(feasible, key=lambda x: (x["total_cost"], x["candidate"])),
                    "candidates": candidates,
                }
        winners = [x["winner"]["candidate"] for x in task_result["arms"].values() if x["winner"]]
        task_result["distinct_winners"] = sorted(set(winners))
        task_result["route_flip_score"] = len(set(winners)) - 1 if winners else -1
        task_result["all_arms_feasible"] = len(winners) == 6
        results[task["name"]] = task_result
        print(task["name"], task_result["route_flip_score"], task_result["distinct_winners"], flush=True)
    eligible = [name for name, value in results.items() if value["all_arms_feasible"]]
    selected = sorted(eligible, key=lambda name: (-results[name]["route_flip_score"], name))[:4]
    if len(selected) != 4: raise RuntimeError("fewer than four task candidates are feasible in all six arms")
    report = {
        "schema": "thesis_stage09_route_selection.v1", "status": "development_selection_complete",
        "final_audit_accessed": False, "removed_camera": REMOVED_CAMERA,
        "selection_rule": "all six arms feasible; descending distinct-winner route-flip score; task-name tie-break",
        "selected_tasks": selected, "matched_seeds": [91500, 91501, 91502, 91503, 91504],
        "results": results,
        "source_hashes": {"planning_manifest": digest(root / "manifest.json"),
                          "tasks": digest(task_path), "world": digest(world), "profile": digest(profile_path),
                          "implementation": digest(Path(__file__))},
    }
    path = staging / "report.json"; path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(staging, output); print(json.dumps({"selected_tasks": selected}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
