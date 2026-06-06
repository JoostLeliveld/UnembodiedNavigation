#!/usr/bin/env python3
"""Offline-only diagnostic: validate belief-closed local executor before Gazebo.

DELIVERABLE 1: Freeze check
  - Load the local executor with belief-loop parameters from aws_beliefloop_v1_config.yaml
  - Simulate tracking two canonical routes (occluded mid_cross_lane and visible lower_sweep_lane)
  - Verify non-frozen control tape, goal reach, and solver timing

DELIVERABLE 2: Belief-offset steering sensitivity
  - Test a representative pose inside the occluded aisle with lateral belief offsets
  - Compare belief-closed EFE vs simple proportional tracker
  - Record truth clearance-to-collision vs delta for both controllers
  - Validate that belief-closed EFE shows collision onset at realistic belief divergence

NO Gazebo, no YOLO, no GP capture.
NO source/config edits.
Output: logs/visibility_comparison/beliefloop_v1/offline_sanity/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import yaml

# --- Path setup
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "planning"))
sys.path.insert(0, str(REPO_ROOT / "src" / "unav_common"))
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))

from experiments.core.world_profiles import (
    resolve_world_path,
    serialize_collision_geometry_from_world,
    serialize_driveable_geometry_from_profile,
)
from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.dynamics import unicycle_step
from planning.core.efe_utils import wrap_angle


# --- Utilities from efe_offline_lab.py ---

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _normalize_geometry_json(raw) -> str:
    text = str(raw)
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text


def _load_gp(npz_path: Path) -> dict:
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
    return out


def _find_task(world: str, task_name: str) -> dict:
    tasks_path = REPO_ROOT / "src" / "experiments" / "config" / "tasks.yaml"
    payload = _load_yaml(tasks_path)
    tasks = list((payload.get("tasks") or {}).get(world) or [])
    for t in tasks:
        if t.get("name") == task_name:
            return t
    raise RuntimeError(f"task {task_name!r} not found for world {world!r}")


def _planner_kind_from_label(cfg: dict, condition: str) -> str:
    conds = cfg.get("conditions", {}) or {}
    return str(conds.get(condition, {}).get("planner", "constant_R_efe"))


def build_planner(
    cfg: dict,
    *,
    planner_kind: str,
    camera_params: dict,
    visibility_artifact_path: str,
    visibility_geometry_json: str,
    collision_geometry_json: str,
    driveable_geometry_json: str = '',
    seed: int = 1,
    visibility_target_height_m: float = 0.35,
) -> UnicyclePlannerBase:
    """Instantiate UnicyclePlannerBase matching the config exactly."""
    use_visibility_model = True
    use_ambiguity = True
    use_obs_risk = True
    if planner_kind == "constant_R_efe":
        use_visibility_model = False
        use_ambiguity = True
        use_obs_risk = True
    elif planner_kind == "risk_only_ablation":
        use_visibility_model = True
        use_ambiguity = False
        use_obs_risk = True

    v_max = float(cfg.get("v_max", 0.22))
    v_min = -v_max
    w_max = float(cfg.get("w_max", 1.0))
    w_min = -w_max

    planner = UnicyclePlannerBase(
        horizon=int(cfg.get("horizon", 40)),
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
        optimizer_maxiter=int(cfg.get("optimizer_maxiter", 150)),
        optimizer_gtol=float(cfg.get("optimizer_gtol", 1e-4)),
        optimizer_warm_start=bool(cfg.get("optimizer_warm_start", True)),
        approx_method=cfg.get("approx_method", "ET1"),
        use_obs_risk=use_obs_risk,
        use_ambiguity=use_ambiguity,
        seed=seed,
        camera_params=camera_params,
        use_visibility_model=use_visibility_model,
        visibility_target_height_m=float(cfg.get("visibility_target_height_m", visibility_target_height_m)),
        visibility_geometry_json=visibility_geometry_json,
        collision_geometry_json=collision_geometry_json,
        visibility_artifact_path=visibility_artifact_path if use_visibility_model else "",
        r_visible_uv=float(cfg.get("r_visible_uv", 2.5)),
        r_miss_uv=float(cfg.get("r_miss_uv", 15.0)),
        visibility_sigma_kappa=float(cfg.get("visibility_sigma_kappa", 1.0)),
        goal_prior_u_std_start=float(cfg.get("goal_prior_u_std_start", 80.0)),
        goal_prior_v_std_start=float(cfg.get("goal_prior_v_std_start", 80.0)),
        goal_prior_u_std_final=float(cfg.get("goal_prior_u_std_final", 20.0)),
        goal_prior_v_std_final=float(cfg.get("goal_prior_v_std_final", 20.0)),
        goal_tightening_power=float(cfg.get("goal_tightening_power", 0.45)),
        goal_progress_n_steps=int(cfg.get("goal_progress_n_steps", 90)),
        observation_risk_scale=float(cfg.get("observation_risk_scale", 1.25)),
        ambiguity_term_scale=float(cfg.get("ambiguity_term_scale", 1.0)),
        discount_gamma=float(cfg.get("discount_gamma", 0.98)),
        optimizer_maxfun=int(cfg.get("optimizer_maxfun", 900)),
        optimizer_ftol=float(cfg.get("optimizer_ftol", 1e-6)),
        optimizer_multistart=_as_bool(cfg.get("optimizer_multistart", False)),
        optimizer_multistart_include_direct=_as_bool(
            cfg.get("optimizer_multistart_include_direct", True)
        ),
        optimizer_multistart_lateral_offsets=cfg.get("optimizer_multistart_lateral_offsets", ""),
        optimizer_initial_routes_json=cfg.get("optimizer_initial_routes_json", ""),
        use_nogo_cost=_as_bool(cfg.get("use_nogo_cost", True)),
        nogo_penalty_type=str(cfg.get("nogo_penalty_type", "softplus")),
        nogo_weight=float(cfg.get("nogo_weight", 80.0)),
        nogo_safe_distance=float(cfg.get("nogo_safe_distance", 0.35)),
        nogo_gaussian_sigma=float(cfg.get("nogo_gaussian_sigma", 0.25)),
        nogo_softplus_scale=float(cfg.get("nogo_softplus_scale", 0.08)),
        nogo_logbarrier_scale=float(cfg.get("nogo_logbarrier_scale", 0.25)),
        nogo_logbarrier_eps=float(cfg.get("nogo_logbarrier_eps", 1e-3)),
        use_belief_nogo_cost=_as_bool(cfg.get("use_belief_nogo_cost", False)),
        nogo_belief_kappa=float(cfg.get("nogo_belief_kappa", 1.0)),
        nogo_mode=str(cfg.get("nogo_mode", "keep_out")),
        driveable_geometry_json=str(driveable_geometry_json or cfg.get("driveable_geometry_json", "")),
        robot_collision_radius_m=float(cfg.get("robot_collision_radius_m", 0.125)),
        runtime_debug=False,
    )
    return planner


def _parse_prisms_from_geometry_json(geometry_json: str) -> list:
    if not geometry_json:
        return []
    try:
        payload = json.loads(geometry_json)
    except Exception:
        return []
    return list(payload.get("prisms") or [])


def _signed_distance_to_nearest_prism(state_xy: np.ndarray, prisms: list) -> float:
    """Compute signed distance from (x, y) to nearest prism boundary.
    Positive = outside all prisms, Negative = inside/colliding with a prism."""
    x, y = float(state_xy[0]), float(state_xy[1])
    min_signed_dist = float('inf')
    for prism in prisms:
        xmin, xmax = float(prism["xmin"]), float(prism["xmax"])
        ymin, ymax = float(prism["ymin"]), float(prism["ymax"])
        # Point-to-rectangle signed distance
        dx_left = x - xmin
        dx_right = xmax - x
        dy_bot = y - ymin
        dy_top = ymax - y
        if dx_left >= 0 and dx_right >= 0 and dy_bot >= 0 and dy_top >= 0:
            # Inside rectangle
            min_interior = min(dx_left, dx_right, dy_bot, dy_top)
            signed_dist = -min_interior
        else:
            # Outside; compute distance to nearest corner or edge
            dx = max(0, max(-dx_left, dx_right - xmax + x))
            dy = max(0, max(-dy_bot, dy_top - ymax + y))
            signed_dist = math.sqrt(dx**2 + dy**2)
        min_signed_dist = min(min_signed_dist, signed_dist)
    return float(min_signed_dist)


def _simple_local_plan_controls(
    m0: np.ndarray,
    target: np.ndarray,
    horizon: int,
    dt: float,
    v_max: float,
) -> np.ndarray:
    """Proportional geometric controller (from efe_agent_node.py)."""
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
        if abs(yaw_err) > yaw_gate:
            v = 0.0
        else:
            v = float(v_max * math.exp(-abs(yaw_err)))
        controls[i] = [v, w]
        state = unicycle_step(state, [v, w], dt)

    return controls


# --- DELIVERABLE 1: Freeze check ---

def run_freeze_check(
    planner: UnicyclePlannerBase,
    m0: np.ndarray,
    S0: np.ndarray,
    waypoints_list: list,
    task_name: str,
    dt: float,
    waypoint_arrival_radius_m: float = 0.20,
    max_steps: int = 250,
) -> dict:
    """Simulate waypoint tracking with the belief-closed EFE planner.

    Returns dict with freeze check results.
    """
    m = np.asarray(m0, dtype=float).reshape(3).copy()
    S = np.asarray(S0, dtype=float).reshape(3, 3).copy()

    wp_idx = 0
    steps_taken = 0
    steps_nonzero_v = 0
    steps_zero_v = 0
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

        if dist_to_wp < waypoint_arrival_radius_m:
            wp_idx += 1
            if wp_idx >= len(waypoints_list):
                reached_final = True
                break
            target = np.asarray(waypoints_list[wp_idx], dtype=float).reshape(2)

        plan_start = time.perf_counter()
        try:
            plan = planner.plan(m, S, target)
        except Exception as e:
            print(f"  [freeze_check] plan() raised exception at step {step}: {e}")
            break
        plan_time_ms = (time.perf_counter() - plan_start) * 1000.0
        solve_times_ms.append(plan_time_ms)

        if len(plan.controls) == 0:
            print(f"  [freeze_check] empty controls at step {step}")
            break

        u0 = np.asarray(plan.controls[0], dtype=float)
        v_abs = abs(float(u0[0]))

        if v_abs > 0.02:
            steps_nonzero_v += 1
            current_zero_v_streak = 0
        else:
            steps_zero_v += 1
            current_zero_v_streak += 1
            max_consecutive_zero_v = max(max_consecutive_zero_v, current_zero_v_streak)

        m, S = planner.predict(m, S, u0)
        trajectory.append(m[:2].copy())
        steps_taken += 1

    frac_nonzero_v = steps_nonzero_v / max(steps_taken, 1)
    max_consecutive_zero_s = max_consecutive_zero_v * dt
    mean_solve_ms = np.mean(solve_times_ms) if solve_times_ms else 0.0
    median_solve_ms = np.median(solve_times_ms) if solve_times_ms else 0.0

    passed = (
        reached_final and
        frac_nonzero_v > 0.85 and
        max_consecutive_zero_s < 1.0 and
        mean_solve_ms < 2500.0
    )

    return {
        "task_name": task_name,
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


# --- DELIVERABLE 2: Belief-offset steering sensitivity ---

def run_belief_offset_sweep(
    planner_beliefclosed: UnicyclePlannerBase,
    m_truth: np.ndarray,
    S0: np.ndarray,
    target_wp: np.ndarray,
    prisms: list,
    robot_radius_m: float,
    n_steps_rollout: int = 8,
) -> dict:
    """Sweep belief offset perpendicular to the local heading.

    For each delta in {0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.4} m:
      - belief-closed EFE: plan from belief=truth+delta, execute on truth
      - simple proportional: plan from belief=truth+delta, execute on truth
      - record truth min clearance for each

    Returns dict with results and comparison.
    """
    deltas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.4]
    dt = float(planner_beliefclosed.dt)

    results = []

    for delta in deltas:
        m_truth_copy = np.asarray(m_truth, dtype=float).reshape(3).copy()
        S0_copy = np.asarray(S0, dtype=float).reshape(3, 3).copy()

        # Lateral offset perpendicular to heading
        heading = float(m_truth_copy[2])
        # Perpendicular direction (90 deg counterclockwise)
        perp_x = -math.sin(heading)
        perp_y = math.cos(heading)

        m_belief = m_truth_copy.copy()
        m_belief[0] += delta * perp_x
        m_belief[1] += delta * perp_y

        # --- Belief-closed EFE ---
        truth_traj_beliefclosed = [m_truth_copy[:2].copy()]
        m_truth_ef = m_truth_copy.copy()
        min_clearance_beliefclosed = float('inf')

        for _ in range(n_steps_rollout):
            plan = planner_beliefclosed.plan(m_belief, S0_copy, target_wp)
            u0 = np.asarray(plan.controls[0], dtype=float)
            m_truth_ef, _ = planner_beliefclosed.predict(m_truth_ef, S0_copy, u0)
            truth_traj_beliefclosed.append(m_truth_ef[:2].copy())

            signed_dist = _signed_distance_to_nearest_prism(m_truth_ef[:2], prisms)
            clearance = signed_dist - robot_radius_m
            min_clearance_beliefclosed = min(min_clearance_beliefclosed, clearance)

            # Update belief only (not truth)
            m_belief, _ = planner_beliefclosed.predict(m_belief, S0_copy, u0)

        # --- Simple proportional tracker ---
        truth_traj_simple = [m_truth_copy[:2].copy()]
        m_truth_simple = m_truth_copy.copy()
        min_clearance_simple = float('inf')

        for _ in range(n_steps_rollout):
            controls_simple = _simple_local_plan_controls(
                m_belief, target_wp,
                horizon=10, dt=dt, v_max=planner_beliefclosed.v_max
            )
            u0_simple = controls_simple[0]
            m_truth_simple = unicycle_step(m_truth_simple, u0_simple, dt)
            truth_traj_simple.append(m_truth_simple[:2].copy())

            signed_dist = _signed_distance_to_nearest_prism(m_truth_simple[:2], prisms)
            clearance = signed_dist - robot_radius_m
            min_clearance_simple = min(min_clearance_simple, clearance)

            m_belief = unicycle_step(m_belief, u0_simple, dt)

        results.append({
            "delta_m": delta,
            "min_clearance_beliefclosed_m": min_clearance_beliefclosed,
            "min_clearance_simple_m": min_clearance_simple,
            "truth_traj_beliefclosed": np.asarray(truth_traj_beliefclosed, dtype=float),
            "truth_traj_simple": np.asarray(truth_traj_simple, dtype=float),
        })

    return {
        "m_truth_start": m_truth.copy(),
        "target_wp": target_wp.copy(),
        "results": results,
    }


# --- Plotting ---

def plot_freeze_check_trajectories(
    setup_dict: dict,
    freeze_results: dict,
    prisms: list,
    out_path: Path,
):
    """Plot the two freeze-check route trajectories over the warehouse."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax_idx, (route_name, result) in enumerate(freeze_results.items()):
        ax = axes[ax_idx]

        # Draw prisms
        for p in prisms:
            rect = Rectangle(
                (p["xmin"], p["ymin"]),
                p["xmax"] - p["xmin"],
                p["ymax"] - p["ymin"],
                facecolor="#666666", edgecolor="#222222", linewidth=0.5, alpha=0.7, zorder=10
            )
            ax.add_patch(rect)

        # Draw trajectory
        traj = result["trajectory"]
        ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", lw=2, label="EFE trajectory", zorder=15)
        ax.scatter([traj[0, 0]], [traj[0, 1]], s=100, c="green", edgecolor="white", linewidth=1, zorder=20, label="start")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=100, c="red", marker="*", edgecolor="white", linewidth=1, zorder=20, label="final")

        ax.set_aspect("equal")
        ax.set_xlim(2.5, 3.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        status = "PASS" if result["passed"] else "FAIL"
        title = f"{route_name}\n({status})"
        if result["reached_final"]:
            title += f"\nReached final WP, {result['frac_nonzero_v']:.1%} steps with |v|>0.02"
            title += f"\nMax zero-v streak: {result['max_consecutive_zero_v_s']:.2f}s"
            title += f"\nMean solve: {result['mean_solve_ms']:.1f}ms"
        else:
            title += f"\nDid NOT reach final WP"

        ax.set_title(title, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_belief_offset_sensitivity(
    sensitivity_results: dict,
    prisms: list,
    out_path: Path,
):
    """Plot clearance vs delta for belief-closed EFE and simple tracker."""
    results = sensitivity_results["results"]
    deltas = [r["delta_m"] for r in results]
    clearances_ef = [r["min_clearance_beliefclosed_m"] for r in results]
    clearances_simple = [r["min_clearance_simple_m"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(deltas, clearances_ef, marker="o", color="#1f77b4", lw=2, label="Belief-closed EFE", zorder=10)
    ax.plot(deltas, clearances_simple, marker="s", color="#ff7f0e", lw=2, label="Simple proportional", zorder=10)
    ax.axhline(0.0, color="red", linestyle="--", lw=1.5, label="collision threshold", zorder=5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Lateral belief offset delta (m)")
    ax.set_ylabel("Truth min clearance (m)")
    ax.set_title("Belief-offset steering sensitivity: truth clearance vs lateral divergence")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Offline sanity check for belief-loop v1.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/aws_beliefloop_v1_config.yaml"),
        help="Path to the belief-loop config"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/beliefloop_v1/offline_sanity"),
        help="Output directory root"
    )
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"BELIEF-LOOP V1 OFFLINE SANITY CHECK")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Output: {out_root}")

    # Load config
    cfg = _load_yaml(args.config)
    world = str(cfg["world"])

    # Load world profiles
    world_profiles_path = REPO_ROOT / "src" / "experiments" / "config" / "world_profiles.yaml"
    profiles = _load_yaml(world_profiles_path)
    profile = dict((profiles.get("worlds") or {}).get(world) or {})

    # Load task
    task_name = "F31_b1_apron_a3_mid"
    task = _find_task(world, task_name)
    print(f"Task: {task_name}")

    # Load GP
    gp_path = Path(cfg["gp_artifact"]).expanduser()
    if not gp_path.is_absolute():
        gp_path = (REPO_ROOT / gp_path).resolve()
    gp = _load_gp(gp_path)
    print(f"GP: {gp_path}")

    # Load world geometry
    world_path = resolve_world_path(world)
    collision_geometry_json = serialize_collision_geometry_from_world(world_path)
    from experiments.core.world_profiles import serialize_driveable_geometry_from_profile
    driveable_geometry_json = serialize_driveable_geometry_from_profile(profile)

    # Parse prisms
    prisms = _parse_prisms_from_geometry_json(collision_geometry_json)
    print(f"Parsed {len(prisms)} collision prisms")

    # Camera params
    camera_params = {
        "cam_pos": tuple(gp["camera_pos"][:3].tolist()),
        "look_at": tuple(gp["look_at"][:3].tolist()),
        "img_width": gp["img_width"],
        "img_height": gp["img_height"],
        "fov_h_rad": gp["fov_h_rad"],
    }

    # Initial belief covariance
    sigma_xy = float(cfg.get("init_belief_sigma_xy", 0.15))
    sigma_th = float(cfg.get("init_belief_sigma_theta", 0.05))
    S0 = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_th ** 2])

    # Task start/goal
    start = task["start"]
    goal = task["goal"]
    m0 = np.array([float(start["x"]), float(start["y"]), float(start["yaw"])], dtype=float)
    goal_xy = np.array([float(goal["x"]), float(goal["y"])], dtype=float)

    print(f"\nStart: {m0[:2]}, yaw={m0[2]:.4f}")
    print(f"Goal: {goal_xy}")

    # Build belief-closed EFE planner with LOCAL parameters
    print("\n" + "-"*70)
    print("Building belief-closed LOCAL EFE planner...")
    print("-"*70)

    local_cfg = dict(cfg)
    # Override with LOCAL parameters
    local_cfg["horizon"] = cfg.get("local_horizon", 10)
    local_cfg["optimizer_maxiter"] = cfg.get("local_optimizer_maxiter", 45)
    local_cfg["nogo_weight"] = cfg.get("local_nogo_weight", 80.0)
    local_cfg["nogo_safe_distance"] = cfg.get("local_nogo_safe_distance", 0.30)
    local_cfg["goal_prior_u_std_start"] = cfg.get("local_goal_prior_u_std_start", 4.0)
    local_cfg["goal_prior_v_std_start"] = cfg.get("local_goal_prior_v_std_start", 4.0)
    local_cfg["goal_prior_u_std_final"] = cfg.get("local_goal_prior_u_std_final", 2.0)
    local_cfg["goal_prior_v_std_final"] = cfg.get("local_goal_prior_v_std_final", 2.0)
    local_cfg["use_visibility_model"] = False
    local_cfg["use_ambiguity"] = False
    local_cfg["use_belief_nogo_cost"] = cfg.get("local_use_belief_nogo_cost", True)
    local_cfg["nogo_penalty_type"] = cfg.get("local_nogo_penalty_type", "log_barrier")

    planner = build_planner(
        local_cfg,
        planner_kind="constant_R_efe",  # Condition-neutral
        camera_params=camera_params,
        visibility_artifact_path=str(gp_path),
        visibility_geometry_json=collision_geometry_json,
        collision_geometry_json=collision_geometry_json,
        driveable_geometry_json=driveable_geometry_json,
        seed=1,
        visibility_target_height_m=gp["target_height"],
    )


    # Override flags for LOCAL condition-neutral executor
    planner.use_visibility_model = False
    planner.use_ambiguity = False
    # Verify planner flags
    assert planner.use_visibility_model == False, "ERROR: use_visibility_model should be False"
    assert planner.use_ambiguity == False, "ERROR: use_ambiguity should be False"
    print(f"Planner config verified:")
    print(f"  use_visibility_model: {planner.use_visibility_model}")
    print(f"  use_ambiguity: {planner.use_ambiguity}")
    print(f"  horizon: {planner.horizon}")
    print(f"  v_max: {planner.v_max}")
    print(f"  dt: {planner.dt}")
    print(f"  nogo_weight: {planner.nogo_weight}")
    print(f"  use_belief_nogo_cost: {planner.use_belief_nogo_cost}")

    # --- DELIVERABLE 1: Freeze check ---
    print("\n" + "="*70)
    print("DELIVERABLE 1: FREEZE CHECK")
    print("="*70)

    # Parse routes from config
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
            print(f"  WARNING: failed to parse optimizer_initial_routes_json: {e}")

    freeze_results = {}
    for route_name, waypoints in routes_dict.items():
        print(f"\nTesting route: {route_name}")
        print(f"  Waypoints: {waypoints}")
        result = run_freeze_check(
            planner, m0, S0, waypoints, route_name,
            dt=float(cfg.get("dt", 0.25)),
            waypoint_arrival_radius_m=float(cfg.get("waypoint_arrival_radius_m", 0.20)),
            max_steps=250,
        )
        freeze_results[route_name] = result

        print(f"  Result:")
        print(f"    Reached final: {result['reached_final']}")
        print(f"    Steps taken: {result['steps_taken']}")
        print(f"    Frac steps with |v|>0.02: {result['frac_nonzero_v']:.3f}")
        print(f"    Max zero-v streak: {result['max_consecutive_zero_v_s']:.3f}s")
        print(f"    Mean solve time: {result['mean_solve_ms']:.1f}ms")
        print(f"    PASSED: {result['passed']}")

    # Write freeze check CSV
    freeze_csv_path = out_root / "freeze_check_results.csv"
    with open(freeze_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "route_name",
                "reached_final",
                "steps_taken",
                "frac_nonzero_v",
                "max_consecutive_zero_v_s",
                "mean_solve_ms",
                "median_solve_ms",
                "passed",
            ]
        )
        writer.writeheader()
        for route_name, result in freeze_results.items():
            writer.writerow({
                "route_name": route_name,
                "reached_final": result["reached_final"],
                "steps_taken": result["steps_taken"],
                "frac_nonzero_v": result["frac_nonzero_v"],
                "max_consecutive_zero_v_s": result["max_consecutive_zero_v_s"],
                "mean_solve_ms": result["mean_solve_ms"],
                "median_solve_ms": result["median_solve_ms"],
                "passed": result["passed"],
            })
    print(f"\nWrote freeze check CSV: {freeze_csv_path}")

    # --- DELIVERABLE 2: Belief-offset sensitivity ---
    print("\n" + "="*70)
    print("DELIVERABLE 2: BELIEF-OFFSET STEERING SENSITIVITY")
    print("="*70)

    # Pick a representative pose inside the occluded aisle
    # mid_cross_lane route: [[3.3,1.5],[1.5,1.5],[1.0,1.75]]
    # Representative pose: between (3.3,1.5) and (1.5,1.5), say (3.1, 2.0, yaw=pi/2)
    m_representative = np.array([3.1, 2.0, math.pi / 2], dtype=float)
    target_wp = np.array([1.5, 1.5], dtype=float)

    print(f"\nTesting pose: {m_representative}")
    print(f"Target waypoint: {target_wp}")

    sensitivity_results = run_belief_offset_sweep(
        planner, m_representative, S0, target_wp, prisms,
        robot_radius_m=float(cfg.get("robot_collision_radius_m", 0.125)),
        n_steps_rollout=8,
    )

    # Write sensitivity CSV
    sensitivity_csv_path = out_root / "belief_offset_sensitivity.csv"
    with open(sensitivity_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "delta_m",
                "min_clearance_beliefclosed_m",
                "min_clearance_simple_m",
            ]
        )
        writer.writeheader()
        for r in sensitivity_results["results"]:
            writer.writerow({
                "delta_m": r["delta_m"],
                "min_clearance_beliefclosed_m": r["min_clearance_beliefclosed_m"],
                "min_clearance_simple_m": r["min_clearance_simple_m"],
            })
    print(f"\nWrote sensitivity CSV: {sensitivity_csv_path}")

    # Print summary
    print(f"\nSensitivity summary:")
    for r in sensitivity_results["results"]:
        delta = r["delta_m"]
        clear_ef = r["min_clearance_beliefclosed_m"]
        clear_simple = r["min_clearance_simple_m"]
        print(f"  delta={delta:4.1f}m: belief_closed={clear_ef:7.3f}m, simple={clear_simple:7.3f}m")

    # --- Plotting ---
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    # Freeze check trajectory plot
    freeze_plot_path = out_root / "freeze_check_trajectories.png"
    plot_freeze_check_trajectories(
        {"start": m0, "goal": goal_xy},
        freeze_results, prisms, freeze_plot_path
    )

    # Sensitivity plot
    sensitivity_plot_path = out_root / "belief_offset_sensitivity.png"
    plot_belief_offset_sensitivity(sensitivity_results, prisms, sensitivity_plot_path)

    # --- Summary report ---
    print("\n" + "="*70)
    print("WRITING SUMMARY REPORT")
    print("="*70)

    summary_path = out_root / "README.md"
    with open(summary_path, "w") as f:
        f.write("# Belief-Loop v1 Offline Sanity Check\n\n")
        f.write("**Date**: 2026-06-03\n\n")
        f.write("**Purpose**: Validate the belief-closed local EFE executor before Gazebo\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- World: {world}\n")
        f.write(f"- Task: {task_name}\n")
        f.write(f"- GP: {gp_path}\n")
        f.write(f"- Local horizon: {cfg.get('local_horizon', 10)}\n")
        f.write(f"- Local optimizer maxiter: {cfg.get('local_optimizer_maxiter', 45)}\n")
        f.write(f"- Local nogo weight: {cfg.get('local_nogo_weight', 80.0)}\n")
        f.write(f"- Local nogo safe distance: {cfg.get('local_nogo_safe_distance', 0.30)}\n")
        f.write(f"- use_visibility_model: False (condition-neutral)\n")
        f.write(f"- use_ambiguity: False (condition-neutral)\n")
        f.write(f"- use_belief_nogo_cost: True\n\n")

        f.write("## Deliverable 1: Freeze Check\n\n")
        f.write("**Purpose**: Verify that the local EFE executor produces non-frozen, progressing control tape\n\n")
        f.write("**Routes tested**:\n")
        for route_name, result in freeze_results.items():
            status = "PASS" if result["passed"] else "FAIL"
            f.write(f"\n### {route_name} [{status}]\n\n")
            f.write(f"- Reached final waypoint: {result['reached_final']}\n")
            f.write(f"- Steps taken: {result['steps_taken']}\n")
            f.write(f"- Fraction of steps with |v|>0.02: {result['frac_nonzero_v']:.1%}\n")
            f.write(f"- Max consecutive zero-v steps: {result['max_consecutive_zero_v_steps']} ({result['max_consecutive_zero_v_s']:.3f}s)\n")
            f.write(f"- Mean solver time: {result['mean_solve_ms']:.1f}ms\n")
            f.write(f"- Median solver time: {result['median_solve_ms']:.1f}ms\n")

        f.write("\n## Deliverable 2: Belief-Offset Steering Sensitivity\n\n")
        f.write("**Purpose**: Verify that belief-closed EFE shows collision when truth diverges from belief\n\n")
        f.write("**Test setup**:\n")
        f.write(f"- Truth state: ({m_representative[0]}, {m_representative[1]}, yaw={m_representative[2]:.4f})\n")
        f.write(f"- Target waypoint: {target_wp}\n")
        f.write(f"- Lateral belief offsets (perpendicular to heading): 0.0 to 2.4 m\n")
        f.write(f"- Rollout steps: 8\n\n")
        f.write("**Key results**:\n\n")
        for r in sensitivity_results["results"]:
            delta = r["delta_m"]
            clear_ef = r["min_clearance_beliefclosed_m"]
            clear_simple = r["min_clearance_simple_m"]
            collision_ef = "COLLISION" if clear_ef < 0 else "SAFE"
            collision_simple = "COLLISION" if clear_simple < 0 else "SAFE"
            f.write(f"- delta={delta:4.1f}m: belief_closed={clear_ef:7.3f}m [{collision_ef}], ")
            f.write(f"simple={clear_simple:7.3f}m [{collision_simple}]\n")

        f.write("\n## Expected Behavior\n\n")
        f.write("1. **Freeze check**: Both routes should reach the final waypoint with >85% of steps having |v|>0.02,\n")
        f.write("   no sustained zero-v freeze (>1s), and mean solver time <2.5s.\n\n")
        f.write("2. **Belief-offset sensitivity**:\n")
        f.write("   - Simple tracker: truth clearance should be relatively stable (belief offset is along-track).\n")
        f.write("   - Belief-closed EFE: because the controller steers in the wrong belief frame,\n")
        f.write("     truth should collide (clearance < 0) once delta exceeds the aisle half-width minus margin\n")
        f.write("     (roughly 0.5-2.4m depending on geometry and controller tuning).\n\n")

        f.write("## Interpretation\n\n")
        f.write("- If both freeze checks PASS and sensitivity shows belief-closed EFE collision at realistic delta,\n")
        f.write("  then the mechanism is validated: the local executor is non-frozen and responds predictably\n")
        f.write("  to belief divergence.\n")
        f.write("- If belief-closed EFE does NOT show collision at realistic delta, the geometry may be too\n")
        f.write("  forgiving and Stage A may be insufficient to discriminate the failure mode in Gazebo.\n\n")

    print(f"Wrote summary: {summary_path}")

    print("\n" + "="*70)
    print(f"Offline sanity check complete. Output: {out_root}")
    print("="*70)


if __name__ == "__main__":
    main()
