#!/usr/bin/env python3
"""Fast diagnostic version with verbose logging."""

import sys
from pathlib import Path

REPO_ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
sys.path.insert(0, str(REPO_ROOT / "src" / "planning"))
sys.path.insert(0, str(REPO_ROOT / "src" / "unav_common"))
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))

import json
import math
import time
import numpy as np
import yaml

from experiments.core.world_profiles import (
    resolve_world_path,
    serialize_collision_geometry_from_world,
    serialize_driveable_geometry_from_profile,
)
from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.dynamics import unicycle_step
from planning.core.efe_utils import wrap_angle

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "t", "yes", "y", "on")

def _normalize_geometry_json(raw) -> str:
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text

def _load_gp(npz_path: Path) -> dict:
    print(f"  Loading GP from {npz_path}...")
    with np.load(npz_path, allow_pickle=False) as d:
        out: dict = {
            "xs": np.asarray(d["xs"], dtype=float),
            "ys": np.asarray(d["ys"], dtype=float),
            "P_plan": np.asarray(d["P_conservative_plan_map"], dtype=float),
            "camera_pos": np.asarray(d["camera_pos"], dtype=float).reshape(-1),
            "camera_pose": np.asarray(d["camera_pose"], dtype=float).reshape(-1),
            "look_at": np.asarray(d["look_at"], dtype=float).reshape(-1),
            "img_width": int(np.asarray(d["img_width"]).reshape(-1)[0]),
            "img_height": int(np.asarray(d["img_height"]).reshape(-1)[0]),
            "fov_h_rad": float(np.asarray(d["fov_h_rad"]).reshape(-1)[0]),
            "target_height": float(np.asarray(d["target_height"]).reshape(-1)[0]),
            "geometry_json": _normalize_geometry_json(d["geometry_json"]),
        }
    print(f"  GP loaded OK.")
    return out

def _find_task(world: str, task_name: str) -> dict:
    tasks_path = REPO_ROOT / "src" / "experiments" / "config" / "tasks.yaml"
    payload = _load_yaml(tasks_path)
    tasks = list((payload.get("tasks") or {}).get(world) or [])
    for t in tasks:
        if t.get("name") == task_name:
            return t
    raise RuntimeError(f"task {task_name!r} not found for world {world!r}")

def main():
    cfg_path = Path("/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/aws_beliefloop_v1_config.yaml")
    out_root = Path("/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/beliefloop_v1/offline_sanity")
    out_root.mkdir(parents=True, exist_ok=True)

    print("\nLoading config...")
    cfg = _load_yaml(cfg_path)
    world = str(cfg["world"])
    print(f"World: {world}")

    print("Loading world profiles...")
    world_profiles_path = REPO_ROOT / "src" / "experiments" / "config" / "world_profiles.yaml"
    profiles = _load_yaml(world_profiles_path)
    profile = dict((profiles.get("worlds") or {}).get(world) or {})

    print("Loading task...")
    task_name = "F31_b1_apron_a3_mid"
    task = _find_task(world, task_name)
    print(f"Task: {task_name}")

    print("Loading GP...")
    gp_path = Path(cfg["gp_artifact"]).expanduser()
    if not gp_path.is_absolute():
        gp_path = (REPO_ROOT / gp_path).resolve()
    gp = _load_gp(gp_path)

    print("Loading world geometry...")
    world_path = resolve_world_path(world)
    collision_geometry_json = serialize_collision_geometry_from_world(world_path)
    driveable_geometry_json = serialize_driveable_geometry_from_profile(profile)

    print("Preparing setup...")
    camera_params = {
        "cam_pos": tuple(gp["camera_pos"][:3].tolist()),
        "look_at": tuple(gp["look_at"][:3].tolist()),
        "img_width": gp["img_width"],
        "img_height": gp["img_height"],
        "fov_h_rad": gp["fov_h_rad"],
    }

    sigma_xy = float(cfg.get("init_belief_sigma_xy", 0.15))
    sigma_th = float(cfg.get("init_belief_sigma_theta", 0.05))
    S0 = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_th ** 2])

    start = task["start"]
    goal = task["goal"]
    m0 = np.array([float(start["x"]), float(start["y"]), float(start["yaw"])], dtype=float)
    goal_xy = np.array([float(goal["x"]), float(goal["y"])], dtype=float)

    print(f"Start: {m0[:2]}, yaw={m0[2]:.4f}")
    print(f"Goal: {goal_xy}")

    print("\nBuilding planner (this may take a moment)...")
    print("  Creating UnicyclePlannerBase...")

    # Build planner manually to avoid the build_planner wrapper
    v_max = float(cfg.get("v_max", 0.22))
    v_min = -v_max
    w_max = float(cfg.get("w_max", 1.0))
    w_min = -w_max

    planner = UnicyclePlannerBase(
        horizon=int(cfg.get("local_horizon", 10)),
        dt=float(cfg.get("dt", 0.25)),
        v_min=v_min,
        v_max=v_max,
        w_min=w_min,
        w_max=w_max,
        control_weight=float(cfg.get("control_weight", 0.0)),
        process_noise_xy=float(cfg.get("process_noise_xy", 0.01)),
        process_noise_theta=float(cfg.get("process_noise_theta", 0.02)),
        obs_noise_uv=float(cfg.get("r_visible_uv", 2.5)),
        goal_sigma_uv=float(cfg.get("goal_prior_u_std_final", 20.0)),
        risk_weight_obs=float(cfg.get("risk_weight_obs", 1.0)),
        ambiguity_weight=float(cfg.get("ambiguity_weight", 6.0)),
        optimizer_maxiter=int(cfg.get("local_optimizer_maxiter", 45)),
        optimizer_gtol=float(cfg.get("optimizer_gtol", 1e-4)),
        optimizer_warm_start=bool(cfg.get("optimizer_warm_start", True)),
        approx_method=cfg.get("approx_method", "ET1"),
        use_obs_risk=True,
        use_ambiguity=False,
        seed=1,
        camera_params=camera_params,
        use_visibility_model=False,
        visibility_target_height_m=float(cfg.get("visibility_target_height_m", gp["target_height"])),
        visibility_geometry_json=collision_geometry_json,
        collision_geometry_json=collision_geometry_json,
        visibility_artifact_path="",
        r_visible_uv=float(cfg.get("r_visible_uv", 2.5)),
        r_miss_uv=float(cfg.get("r_miss_uv", 15.0)),
        visibility_sigma_kappa=float(cfg.get("visibility_sigma_kappa", 1.0)),
        goal_prior_u_std_start=float(cfg.get("local_goal_prior_u_std_start", 4.0)),
        goal_prior_v_std_start=float(cfg.get("local_goal_prior_v_std_start", 4.0)),
        goal_prior_u_std_final=float(cfg.get("local_goal_prior_u_std_final", 2.0)),
        goal_prior_v_std_final=float(cfg.get("local_goal_prior_v_std_final", 2.0)),
        goal_tightening_power=float(cfg.get("goal_tightening_power", 0.45)),
        goal_progress_n_steps=int(cfg.get("goal_progress_n_steps", 90)),
        observation_risk_scale=float(cfg.get("observation_risk_scale", 1.25)),
        ambiguity_term_scale=float(cfg.get("ambiguity_term_scale", 1.0)),
        discount_gamma=float(cfg.get("discount_gamma", 0.98)),
        optimizer_maxfun=int(cfg.get("local_optimizer_maxiter", 45) * 4),
        optimizer_ftol=float(cfg.get("optimizer_ftol", 1e-6)),
        optimizer_multistart=False,
        optimizer_multistart_include_direct=False,
        optimizer_multistart_lateral_offsets="",
        optimizer_initial_routes_json="",
        use_nogo_cost=_as_bool(cfg.get("use_nogo_cost", True)),
        nogo_penalty_type=str(cfg.get("local_nogo_penalty_type", "log_barrier")),
        nogo_weight=float(cfg.get("local_nogo_weight", 80.0)),
        nogo_safe_distance=float(cfg.get("local_nogo_safe_distance", 0.30)),
        nogo_gaussian_sigma=float(cfg.get("nogo_gaussian_sigma", 0.25)),
        nogo_softplus_scale=float(cfg.get("nogo_softplus_scale", 0.08)),
        nogo_logbarrier_scale=float(cfg.get("nogo_logbarrier_scale", 0.10)),
        nogo_logbarrier_eps=float(cfg.get("nogo_logbarrier_eps", 0.01)),
        use_belief_nogo_cost=_as_bool(cfg.get("local_use_belief_nogo_cost", True)),
        nogo_belief_kappa=float(cfg.get("nogo_belief_kappa", 2.0)),
        nogo_mode=str(cfg.get("nogo_mode", "keep_out")),
        driveable_geometry_json=str(driveable_geometry_json or ""),
        robot_collision_radius_m=float(cfg.get("robot_collision_radius_m", 0.125)),
        runtime_debug=False,
    )

    print(f"  Planner created OK.")
    print(f"  use_visibility_model: {planner.use_visibility_model}")
    print(f"  use_ambiguity: {planner.use_ambiguity}")
    print(f"  horizon: {planner.horizon}")
    print(f"  nogo_weight: {planner.nogo_weight}")

    # Quick freeze check on one route
    print("\n" + "="*70)
    print("QUICK FREEZE CHECK (20 steps max)")
    print("="*70)

    routes_json = cfg.get("optimizer_initial_routes_json", "")
    routes_dict = {}
    if routes_json:
        try:
            routes_list = json.loads(routes_json)
            for route in routes_list:
                name = route.get("name", "")
                waypoints = route.get("waypoints", [])
                routes_dict[name] = waypoints
        except Exception as e:
            print(f"  Failed to parse routes: {e}")

    if routes_dict:
        route_name = list(routes_dict.keys())[0]
        waypoints = routes_dict[route_name]
        print(f"\nTesting route: {route_name}")
        print(f"  Waypoints: {waypoints}")

        m = m0.copy()
        S = S0.copy()
        wp_idx = 0
        steps_taken = 0
        zero_v_steps = 0

        for step in range(20):
            if wp_idx >= len(waypoints):
                print(f"  Reached final waypoint at step {step}")
                break

            target = np.array(waypoints[wp_idx], dtype=float)
            dist_to_wp = np.linalg.norm(m[:2] - target)

            if dist_to_wp < 0.20:
                wp_idx += 1
                print(f"  Reached waypoint {wp_idx-1} at step {step}")
                if wp_idx >= len(waypoints):
                    break
                target = np.array(waypoints[wp_idx], dtype=float)

            print(f"  Step {step}: planning to waypoint {wp_idx}...", end=" ", flush=True)
            plan_start = time.perf_counter()
            try:
                plan = planner.plan(m, S, target)
                plan_time_ms = (time.perf_counter() - plan_start) * 1000.0
                print(f"OK ({plan_time_ms:.1f}ms)", end=" ")
            except Exception as e:
                print(f"FAILED: {e}")
                break

            if len(plan.controls) == 0:
                print("  (no controls)")
                break

            u0 = plan.controls[0]
            v_abs = abs(u0[0])
            if v_abs < 0.02:
                zero_v_steps += 1
                print(f"v={u0[0]:6.3f}")
            else:
                print(f"v={u0[0]:6.3f} (OK)")

            m, S = planner.predict(m, S, u0)
            steps_taken += 1

        print(f"\n  Result: {steps_taken} steps taken, {zero_v_steps} zero-v steps")

    print("\n" + "="*70)
    print("QUICK CHECK COMPLETE")
    print("="*70)
    print(f"\nDiagnostic output ready. Full run would write to: {out_root}")

if __name__ == "__main__":
    main()
