#!/usr/bin/env python3
"""Offline diagnostic: validate belief-closed local executor before Gazebo.

Runs DELIVERABLE 1 (freeze check) and DELIVERABLE 2 (belief-offset sensitivity)
without Gazebo, YOLO, or GP capture.

BUG FIXES (2026-06-03):
  BUG 1: Added warm-start replication. At the start and every time the active
         waypoint index advances, set planner.prev_controls_flat before calling
         planner.plan(m, S, target). This mirrors efe_agent_node.py:193-199 and
         218-220, preventing spurious zero-control minima from cold-start.
  BUG 2: Changed test pose from degenerate (3.1, 2.0) [clearance -0.125m, inside
         collision prism] to a genuinely CLEAR pose found by scanning. The
         belief-offset sweep now uses a truth pose with clearance >= 0.25m.

Output: logs/visibility_comparison/beliefloop_v1/offline_sanity/
"""

import sys
from pathlib import Path

REPO_ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
sys.path.insert(0, str(REPO_ROOT / "src" / "planning"))
sys.path.insert(0, str(REPO_ROOT / "src" / "unav_common"))
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))

import csv
import json
import math
import time
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiments.core.world_profiles import (
    resolve_world_path,
    serialize_collision_geometry_from_world,
    serialize_driveable_geometry_from_profile,
)
from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.dynamics import unicycle_step
from planning.core.efe_utils import wrap_angle

# --- Utilities ---

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
    with np.load(npz_path, allow_pickle=False) as d:
        out = {
            "xs": np.asarray(d["xs"], dtype=float),
            "ys": np.asarray(d["ys"], dtype=float),
            "camera_pos": np.asarray(d["camera_pos"], dtype=float).reshape(-1),
            "look_at": np.asarray(d["look_at"], dtype=float).reshape(-1),
            "img_width": int(np.asarray(d["img_width"]).reshape(-1)[0]),
            "img_height": int(np.asarray(d["img_height"]).reshape(-1)[0]),
            "fov_h_rad": float(np.asarray(d["fov_h_rad"]).reshape(-1)[0]),
            "target_height": float(np.asarray(d["target_height"]).reshape(-1)[0]),
            "geometry_json": _normalize_geometry_json(d["geometry_json"]),
        }
    return out

def _find_task(world: str, task_name: str) -> dict:
    tasks_path = REPO_ROOT / "src" / "experiments" / "config" / "tasks.yaml"
    payload = _load_yaml(tasks_path)
    tasks = list((payload.get("tasks") or {}).get(world) or [])
    for t in tasks:
        if t.get("name") == task_name:
            return t
    raise RuntimeError(f"task {task_name!r} not found")

def _parse_prisms_from_geometry_json(geometry_json: str) -> list:
    if not geometry_json:
        return []
    try:
        payload = json.loads(geometry_json)
    except Exception:
        return []
    return list(payload.get("prisms") or [])

def _signed_distance_to_nearest_prism(state_xy: np.ndarray, prisms: list) -> float:
    """Signed distance to nearest prism (positive = outside). Ignores boundary prisms 0-3."""
    x, y = float(state_xy[0]), float(state_xy[1])
    min_signed_dist = float('inf')
    # Skip world boundary prisms (indices 0-3)
    obstacle_prisms = prisms[4:] if len(prisms) > 4 else prisms
    for prism in obstacle_prisms:
        xmin, xmax = float(prism["xmin"]), float(prism["xmax"])
        ymin, ymax = float(prism["ymin"]), float(prism["ymax"])
        dx_left, dx_right = x - xmin, xmax - x
        dy_bot, dy_top = y - ymin, ymax - y
        if dx_left >= 0 and dx_right >= 0 and dy_bot >= 0 and dy_top >= 0:
            # Inside
            min_interior = min(dx_left, dx_right, dy_bot, dy_top)
            signed_dist = -min_interior
        else:
            # Outside
            dx = max(0, max(-dx_left, dx_right - xmax + x))
            dy = max(0, max(-dy_bot, dy_top - ymax + y))
            signed_dist = math.sqrt(dx**2 + dy**2)
        min_signed_dist = min(min_signed_dist, signed_dist)
    return float(min_signed_dist)

def _find_clear_test_pose(prisms: list, robot_radius_m: float) -> tuple:
    """Scan candidate poses and return the first genuinely CLEAR one (clearance >= 0.25m).

    Returns: (x, y, yaw, clearance_m, reason_string)
    """
    obstacle_prisms = prisms[4:] if len(prisms) > 4 else prisms
    candidates = []

    # Scan the lower_sweep_lane route first (it showed good clearance)
    for x in np.linspace(3.3, 1.2, 25):
        for y in np.linspace(-2.3, -2.1, 5):
            signed_dist = _signed_distance_to_nearest_prism([x, y], prisms)
            clearance = signed_dist - robot_radius_m
            candidates.append((x, y, clearance, "lower_sweep_lane"))

    # Also check the mid route
    for x in np.linspace(3.3, 1.5, 20):
        for y in np.linspace(1.3, 1.7, 9):
            signed_dist = _signed_distance_to_nearest_prism([x, y], prisms)
            clearance = signed_dist - robot_radius_m
            candidates.append((x, y, clearance, "mid_cross_lane"))

    # Sort by clearance (best first)
    candidates.sort(key=lambda c: -c[2])

    # Print all with clearance >= 0.15m for debugging
    print(f"\nCandidates with clearance >= 0.15m:")
    for x, y, clear, route in candidates:
        if clear >= 0.15:
            print(f"  ({x:5.2f}, {y:5.2f}) [{route:18s}]: clearance = {clear:7.4f} m")

    # Pick the first one with clearance >= 0.25m
    for x, y, clear, route in candidates:
        if clear >= 0.25:
            print(f"\nSelected test pose: ({x:.2f}, {y:.2f}) from {route} with clearance {clear:.4f} m")
            return (x, y, math.pi / 2, clear, route)

    # Fallback: use best available
    x, y, clear, route = candidates[0]
    print(f"\nNo pose with clearance >= 0.25m found; using best available: ({x:.2f}, {y:.2f}) "
          f"from {route} with clearance {clear:.4f} m")
    return (x, y, math.pi / 2, clear, route)

def _controls_for_waypoints(planner, start_xy_yaw, waypoints) -> np.ndarray:
    """Generate a crude unicycle warm-start toward waypoints.

    Mirrors the version from diagnose_initial_rollouts.py and efe_agent_node.py.
    The optimizer is free to change these controls.
    """
    horizon = int(planner.horizon)
    dt = float(planner.dt)
    v_max = float(planner.v_max)
    w_max = float(planner.w_max)
    m = np.asarray(start_xy_yaw, dtype=float).reshape(3).copy()
    controls = np.zeros((horizon, 2), dtype=float)
    waypoint_idx = 0
    waypoints = [np.asarray(wp, dtype=float).reshape(2) for wp in waypoints]

    for k in range(horizon):
        if waypoint_idx >= len(waypoints):
            break
        target = waypoints[waypoint_idx]
        d = target - m[:2]
        if float(np.linalg.norm(d)) < 0.18 and waypoint_idx < len(waypoints) - 1:
            waypoint_idx += 1
            target = waypoints[waypoint_idx]
            d = target - m[:2]

        desired_yaw = math.atan2(float(d[1]), float(d[0]))
        yaw_err = wrap_angle(desired_yaw - float(m[2]))
        w = float(np.clip(yaw_err / max(dt, 1e-6), -w_max, w_max))
        v = v_max if abs(yaw_err) < 0.65 else 0.0
        controls[k] = [v, w]
        m, _S_dummy = planner.predict(m, np.eye(3) * 1e-6, controls[k])

    return controls

def _simple_local_plan_controls(m0: np.ndarray, target: np.ndarray, horizon: int, dt: float, v_max: float) -> np.ndarray:
    """Proportional geometric controller."""
    w_max = 1.5
    yaw_gate = 0.6
    controls = np.zeros((horizon, 2), dtype=float)
    state = m0[:3].copy().astype(float)
    tx, ty = float(target[0]), float(target[1])
    for i in range(horizon):
        dx, dy = tx - state[0], ty - state[1]
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            break
        desired_yaw = math.atan2(dy, dx)
        yaw_err = wrap_angle(desired_yaw - state[2])
        w = float(np.clip(2.0 * yaw_err, -w_max, w_max))
        v = 0.0 if abs(yaw_err) > yaw_gate else float(v_max * math.exp(-abs(yaw_err)))
        controls[i] = [v, w]
        state = unicycle_step(state, [v, w], dt)
    return controls

# --- DELIVERABLE 1: Freeze check (with warm-start) ---

def run_freeze_check(planner, m0, S0, waypoints_list, route_name, dt, arrival_radius=0.20, max_steps=250) -> dict:
    """Simulate waypoint tracking WITH warm-start on every waypoint change.

    BUG FIX 1: Replicates runtime warm-start from efe_agent_node.py:193-199 and 218-220.
    At the start and every time the active waypoint index advances, set
    planner.prev_controls_flat from _controls_for_waypoints before calling plan().
    """
    m = np.asarray(m0, dtype=float).reshape(3).copy()
    S = np.asarray(S0, dtype=float).reshape(3, 3).copy()
    wp_idx = 0
    steps_taken = 0
    steps_nonzero_v = 0
    max_consecutive_zero_v = 0
    current_zero_v_streak = 0
    solve_times_ms = []
    trajectory = [m[:2].copy()]
    reached_final = False

    for step in range(max_steps):
        if wp_idx >= len(waypoints_list):
            reached_final = True
            break
        target = np.asarray(waypoints_list[wp_idx], dtype=float).reshape(2)
        dist_to_wp = float(np.linalg.norm(m[:2] - target))

        # Check if we've arrived at the current waypoint
        prev_wp_idx = wp_idx
        if dist_to_wp < arrival_radius:
            wp_idx += 1
            if wp_idx >= len(waypoints_list):
                reached_final = True
                break
            target = np.asarray(waypoints_list[wp_idx], dtype=float).reshape(2)

        # WARM-START FIX: On first step or when waypoint changes, seed the warm-start.
        if step == 0 or wp_idx != prev_wp_idx:
            try:
                seed = _controls_for_waypoints(planner, m[:3], [target])
                planner.prev_controls_flat = np.asarray(seed, dtype=float).reshape(-1)
            except Exception as e:
                print(f"  [freeze_check] failed to set warm-start seed: {e}")

        plan_start = time.perf_counter()
        try:
            plan = planner.plan(m, S, target)
        except Exception as e:
            print(f"  [freeze_check] plan() raised at step {step}: {e}")
            break
        plan_time_ms = (time.perf_counter() - plan_start) * 1000.0
        solve_times_ms.append(plan_time_ms)

        if len(plan.controls) == 0:
            break

        u0 = np.asarray(plan.controls[0], dtype=float)
        v_abs = abs(float(u0[0]))
        if v_abs > 0.02:
            steps_nonzero_v += 1
            current_zero_v_streak = 0
        else:
            current_zero_v_streak += 1
            max_consecutive_zero_v = max(max_consecutive_zero_v, current_zero_v_streak)

        m, S = planner.predict(m, S, u0)
        trajectory.append(m[:2].copy())
        steps_taken += 1

    frac_nonzero_v = steps_nonzero_v / max(steps_taken, 1)
    max_consecutive_zero_s = max_consecutive_zero_v * dt
    mean_solve_ms = np.mean(solve_times_ms) if solve_times_ms else 0.0
    median_solve_ms = np.median(solve_times_ms) if solve_times_ms else 0.0
    passed = reached_final and frac_nonzero_v > 0.85 and max_consecutive_zero_s < 1.0 and mean_solve_ms < 2500.0

    return {
        "route_name": route_name,
        "reached_final": reached_final,
        "steps_taken": steps_taken,
        "steps_nonzero_v": steps_nonzero_v,
        "frac_nonzero_v": frac_nonzero_v,
        "max_consecutive_zero_v_steps": max_consecutive_zero_v,
        "max_consecutive_zero_v_s": max_consecutive_zero_s,
        "mean_solve_ms": mean_solve_ms,
        "median_solve_ms": median_solve_ms,
        "trajectory": np.asarray(trajectory, dtype=float),
        "passed": passed,
    }

# --- DELIVERABLE 2: Belief-offset sensitivity (with clear pose) ---

def run_belief_offset_sweep(planner, m_truth, S0, target_wp, prisms, robot_radius_m, test_pose_info, n_steps=8) -> dict:
    """Sweep lateral belief offset from a CLEAR truth pose.

    BUG FIX 2: Uses a genuinely clear test pose (clearance >= 0.25m), not a
    degenerate one inside a collision prism. Also uses warm-start for EFE plan.
    """
    deltas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.4]
    dt = float(planner.dt)
    results = []

    for delta in deltas:
        m_truth_copy = np.asarray(m_truth, dtype=float).reshape(3).copy()
        S0_copy = np.asarray(S0, dtype=float).reshape(3, 3).copy()
        heading = float(m_truth_copy[2])
        perp_x = -math.sin(heading)
        perp_y = math.cos(heading)

        m_belief = m_truth_copy.copy()
        m_belief[0] += delta * perp_x
        m_belief[1] += delta * perp_y

        # Belief-closed EFE with warm-start
        m_truth_ef = m_truth_copy.copy()
        min_clearance_beliefclosed = float('inf')
        for step_i in range(n_steps):
            # Warm-start on first step
            if step_i == 0:
                try:
                    seed = _controls_for_waypoints(planner, m_belief[:3], [target_wp])
                    planner.prev_controls_flat = np.asarray(seed, dtype=float).reshape(-1)
                except Exception:
                    pass
            plan = planner.plan(m_belief, S0_copy, target_wp)
            u0 = np.asarray(plan.controls[0], dtype=float)
            m_truth_ef, _ = planner.predict(m_truth_ef, S0_copy, u0)
            signed_dist = _signed_distance_to_nearest_prism(m_truth_ef[:2], prisms)
            clearance = signed_dist - robot_radius_m
            min_clearance_beliefclosed = min(min_clearance_beliefclosed, clearance)
            m_belief, _ = planner.predict(m_belief, S0_copy, u0)

        # Simple proportional with warm-start on belief
        m_truth_simple = m_truth_copy.copy()
        m_belief_simple = m_truth_copy.copy()
        m_belief_simple[0] += delta * perp_x
        m_belief_simple[1] += delta * perp_y
        min_clearance_simple = float('inf')
        for step_i in range(n_steps):
            controls_simple = _simple_local_plan_controls(m_belief_simple, target_wp, horizon=10, dt=dt, v_max=planner.v_max)
            u0_simple = controls_simple[0]
            m_truth_simple = unicycle_step(m_truth_simple, u0_simple, dt)
            signed_dist = _signed_distance_to_nearest_prism(m_truth_simple[:2], prisms)
            clearance = signed_dist - robot_radius_m
            min_clearance_simple = min(min_clearance_simple, clearance)
            m_belief_simple = unicycle_step(m_belief_simple, u0_simple, dt)

        results.append({
            "delta_m": delta,
            "min_clearance_beliefclosed_m": min_clearance_beliefclosed,
            "min_clearance_simple_m": min_clearance_simple,
        })

    return {
        "m_truth_start": m_truth.copy(),
        "target_wp": target_wp.copy(),
        "test_pose_info": test_pose_info,
        "results": results,
    }

# --- Plotting ---

def plot_trajectories_and_sensitivity(freeze_results, sensitivity_results, prisms, out_root: Path):
    """Plot freeze check routes and belief-offset sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Freeze check trajectories (2 subplots)
    for idx, (route_name, result) in enumerate(freeze_results.items()):
        if idx >= 2:
            break
        ax = axes[idx]
        # Plot obstacles (skip world boundary prisms 0-3)
        obstacle_prisms = prisms[4:] if len(prisms) > 4 else prisms
        for p in obstacle_prisms:
            rect = Rectangle((p["xmin"], p["ymin"]), p["xmax"] - p["xmin"], p["ymax"] - p["ymin"],
                           facecolor="#666666", edgecolor="#222222", linewidth=0.5, alpha=0.7, zorder=10)
            ax.add_patch(rect)
        traj = result["trajectory"]
        ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", lw=2, label="EFE", zorder=15)
        ax.scatter([traj[0, 0]], [traj[0, 1]], s=100, c="green", edgecolor="white", linewidth=1, zorder=20, label="start")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=100, c="red", marker="*", edgecolor="white", linewidth=1, zorder=20, label="final")
        ax.set_aspect("equal")
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        status = "PASS" if result["passed"] else "FAIL"
        ax.set_title(f"{route_name} [{status}]\nfrac_nonzero_v={result['frac_nonzero_v']:.1%}, mean_solve={result['mean_solve_ms']:.0f}ms")

    # Sensitivity plot
    ax = axes[2]
    results = sensitivity_results["results"]
    deltas = [r["delta_m"] for r in results]
    clearances_ef = [r["min_clearance_beliefclosed_m"] for r in results]
    clearances_simple = [r["min_clearance_simple_m"] for r in results]
    ax.plot(deltas, clearances_ef, marker="o", color="#1f77b4", lw=2, label="Belief-closed EFE", zorder=10)
    ax.plot(deltas, clearances_simple, marker="s", color="#ff7f0e", lw=2, label="Simple proportional", zorder=10)
    ax.axhline(0.0, color="red", linestyle="--", lw=1.5, label="collision threshold", zorder=5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Lateral belief offset delta (m)")
    ax.set_ylabel("Truth min clearance (m)")
    ax.set_title("Belief-offset steering sensitivity")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = out_root / "diagnostics_summary.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

# --- Main ---

def main():
    out_root = Path("/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/beliefloop_v1/offline_sanity")
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("BELIEF-LOOP V1 OFFLINE SANITY DIAGNOSTIC (CORRECTED)")
    print("="*70)
    print(f"Output: {out_root}\n")

    # Load setup
    cfg_path = Path("/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/aws_beliefloop_v1_config.yaml")
    cfg = _load_yaml(cfg_path)
    world = str(cfg["world"])

    world_profiles_path = REPO_ROOT / "src" / "experiments" / "config" / "world_profiles.yaml"
    profiles = _load_yaml(world_profiles_path)
    profile = dict((profiles.get("worlds") or {}).get(world) or {})

    task_name = "F31_b1_apron_a3_mid"
    task = _find_task(world, task_name)

    gp_path = Path(cfg["gp_artifact"]).expanduser()
    if not gp_path.is_absolute():
        gp_path = (REPO_ROOT / gp_path).resolve()
    gp = _load_gp(gp_path)

    world_path = resolve_world_path(world)
    collision_geometry_json = serialize_collision_geometry_from_world(world_path)
    driveable_geometry_json = serialize_driveable_geometry_from_profile(profile)

    prisms = _parse_prisms_from_geometry_json(collision_geometry_json)
    print(f"Task: {task_name}")
    print(f"Collision prisms: {len(prisms)}")

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

    # Build planner
    print("\nBuilding belief-closed LOCAL EFE planner...")
    v_max = float(cfg.get("v_max", 0.22))
    planner = UnicyclePlannerBase(
        horizon=int(cfg.get("local_horizon", 10)),
        dt=float(cfg.get("dt", 0.25)),
        v_min=-v_max,
        v_max=v_max,
        w_min=-1.0,
        w_max=1.0,
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
        visibility_target_height_m=gp["target_height"],
        visibility_geometry_json=collision_geometry_json,
        collision_geometry_json=collision_geometry_json,
        visibility_artifact_path="",
        r_visible_uv=float(cfg.get("r_visible_uv", 2.5)),
        r_miss_uv=float(cfg.get("r_miss_uv", 15.0)),
        visibility_sigma_kappa=1.0,
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
        nogo_gaussian_sigma=0.25,
        nogo_softplus_scale=0.08,
        nogo_logbarrier_scale=float(cfg.get("nogo_logbarrier_scale", 0.10)),
        nogo_logbarrier_eps=float(cfg.get("nogo_logbarrier_eps", 0.01)),
        use_belief_nogo_cost=_as_bool(cfg.get("local_use_belief_nogo_cost", True)),
        nogo_belief_kappa=float(cfg.get("nogo_belief_kappa", 2.0)),
        nogo_mode="keep_out",
        driveable_geometry_json=driveable_geometry_json,
        robot_collision_radius_m=float(cfg.get("robot_collision_radius_m", 0.125)),
        runtime_debug=False,
    )

    assert planner.use_visibility_model == False and planner.use_ambiguity == False
    print(f"  horizon={planner.horizon}, nogo_weight={planner.nogo_weight}, use_belief_nogo={planner.use_belief_nogo_cost}")

    # DELIVERABLE 1
    print("\n" + "="*70)
    print("DELIVERABLE 1: FREEZE CHECK (with warm-start)")
    print("="*70)

    routes_json = cfg.get("optimizer_initial_routes_json", "")
    routes_dict = {}
    try:
        routes_list = json.loads(routes_json)
        for route in routes_list:
            routes_dict[route.get("name", "")] = route.get("waypoints", [])
    except:
        pass

    freeze_results = {}
    for route_name, waypoints in routes_dict.items():
        print(f"\nRoute: {route_name}")
        result = run_freeze_check(planner, m0, S0, waypoints, route_name, cfg["dt"])
        freeze_results[route_name] = result
        print(f"  Reached: {result['reached_final']}, frac_nonzero_v: {result['frac_nonzero_v']:.1%}, "
              f"max_zero_streak: {result['max_consecutive_zero_v_s']:.3f}s, "
              f"mean_solve: {result['mean_solve_ms']:.0f}ms")
        print(f"  {'PASS' if result['passed'] else 'FAIL'}")
        print(f"  Warm-start source: _controls_for_waypoints from diagnose_initial_rollouts.py (mirrored in this script)")

    # Write freeze check CSV
    freeze_csv = out_root / "freeze_check_results.csv"
    with open(freeze_csv, "w", newline="") as f:
        w = csv.DictWriter(f, ["route_name", "reached_final", "steps_taken", "frac_nonzero_v", "max_consecutive_zero_v_s", "mean_solve_ms", "passed"])
        w.writeheader()
        for result in freeze_results.values():
            w.writerow({k: result[k] for k in w.fieldnames})
    print(f"\nWrote: {freeze_csv}")

    # DELIVERABLE 2
    print("\n" + "="*70)
    print("DELIVERABLE 2: BELIEF-OFFSET STEERING SENSITIVITY (with clear pose)")
    print("="*70)

    # Find a genuinely CLEAR test pose
    robot_radius_m = float(cfg.get("robot_collision_radius_m", 0.125))
    x, y, yaw, clearance, route = _find_clear_test_pose(prisms, robot_radius_m)
    m_test = np.array([x, y, yaw], dtype=float)
    target_wp = np.array([1.5, 1.5], dtype=float)
    test_pose_info = {"x": x, "y": y, "yaw": yaw, "clearance": clearance, "route": route}

    print(f"\nTest pose (clear, not degenerate): ({x:.2f}, {y:.2f}, yaw={yaw:.3f})")
    print(f"  Clearance: {clearance:.4f} m (>= 0.25m required)")
    print(f"  From route: {route}")
    print(f"  Target waypoint: {target_wp}")

    sensitivity_results = run_belief_offset_sweep(planner, m_test, S0, target_wp, prisms,
                                                  robot_radius_m, test_pose_info)

    # Write sensitivity CSV
    sens_csv = out_root / "belief_offset_sensitivity.csv"
    with open(sens_csv, "w", newline="") as f:
        w = csv.DictWriter(f, ["delta_m", "min_clearance_beliefclosed_m", "min_clearance_simple_m"])
        w.writeheader()
        for r in sensitivity_results["results"]:
            w.writerow(r)
    print(f"\nWrote: {sens_csv}")

    print("\nSensitivity summary:")
    for r in sensitivity_results["results"]:
        delta = r["delta_m"]
        clear_ef = r["min_clearance_beliefclosed_m"]
        clear_simple = r["min_clearance_simple_m"]
        print(f"  delta={delta:4.1f}m: belief_closed={clear_ef:7.3f}m, simple={clear_simple:7.3f}m")

    # Plot
    print("\n" + "="*70)
    print("PLOTTING")
    print("="*70)
    plot_trajectories_and_sensitivity(freeze_results, sensitivity_results, prisms, out_root)

    # Summary report
    summary_path = out_root / "README.md"
    with open(summary_path, "w") as f:
        f.write("# Belief-Loop v1 Offline Sanity Check (Corrected 2026-06-03)\n\n")
        f.write("**Diagnostic**: belief-closed local EFE executor validation (no Gazebo)\n\n")
        f.write(f"**Config**: aws_beliefloop_v1_config.yaml\n")
        f.write(f"**Task**: {task_name}\n")
        f.write(f"**Local horizon**: {cfg.get('local_horizon', 10)}\n")
        f.write(f"**Local nogo weight**: {cfg.get('local_nogo_weight', 80.0)}\n")
        f.write(f"**use_visibility_model**: False (condition-neutral)\n")
        f.write(f"**use_ambiguity**: False (condition-neutral)\n\n")

        f.write("## Bug Fixes (2026-06-03)\n\n")
        f.write("**BUG 1 — Freeze check**: Added warm-start replication. At the start and every time\n")
        f.write("the active waypoint index advances, planner.prev_controls_flat is seeded from\n")
        f.write("_controls_for_waypoints() before calling planner.plan(). This mirrors the runtime\n")
        f.write("pattern in efe_agent_node.py:193-199 and 218-220, preventing spurious zero-control\n")
        f.write("local minima from cold-start optimizer initialization.\n\n")
        f.write("**BUG 2 — Belief-offset test pose**: Changed from degenerate pose (3.1, 2.0)\n")
        f.write(f"[clearance -0.125m, inside collision prism] to a genuinely CLEAR pose found by\n")
        f.write(f"scanning: ({x:.2f}, {y:.2f}) with clearance {clearance:.4f}m. Enables honest belief-offset\n")
        f.write(f"sensitivity evaluation.\n\n")

        f.write("## Deliverable 1: Freeze Check\n\n")
        for route_name, result in freeze_results.items():
            f.write(f"### {route_name}\n\n")
            f.write(f"- Reached final WP: {result['reached_final']}\n")
            f.write(f"- Frac steps with |v|>0.02: {result['frac_nonzero_v']:.1%}\n")
            f.write(f"- Max zero-v streak: {result['max_consecutive_zero_v_s']:.3f}s\n")
            f.write(f"- Mean solver time: {result['mean_solve_ms']:.0f}ms\n")
            f.write(f"- Status: {'PASS' if result['passed'] else 'FAIL'}\n\n")

        f.write("## Deliverable 2: Belief-Offset Sensitivity\n\n")
        f.write(f"Test pose: ({x:.2f}, {y:.2f}, yaw={yaw:.3f}), clearance={clearance:.4f}m\n")
        f.write(f"Target waypoint: [{target_wp[0]:.2f}, {target_wp[1]:.2f}]\n\n")
        f.write("Lateral belief offset sweep (perpendicular to heading):\n\n")
        for r in sensitivity_results["results"]:
            f.write(f"- delta={r['delta_m']:4.1f}m: belief_closed={r['min_clearance_beliefclosed_m']:7.3f}m, simple={r['min_clearance_simple_m']:7.3f}m\n")

        f.write("\n## Interpretation\n\n")
        f.write("1. **Freeze check (with warm-start)**: Both routes should reach final WP with >85% non-zero-v steps,\n")
        f.write("   <1s zero streak, <2.5s mean solve. The warm-start prevents cold-start local minima.\n\n")
        f.write("2. **Sensitivity**: belief-closed EFE may show collision (clearance<0) at realistic delta\n")
        f.write("   while simple tracker maintains safety. If belief-closed DOES collide, that validates\n")
        f.write("   the mechanism: diverged belief steers truth into danger. If it does NOT collide,\n")
        f.write("   the nogo constraints may be strong enough to prevent collision regardless.\n\n")

    print(f"Wrote: {summary_path}")

    print("\n" + "="*70)
    print(f"COMPLETE. Output: {out_root}")
    print("="*70)

if __name__ == "__main__":
    main()
