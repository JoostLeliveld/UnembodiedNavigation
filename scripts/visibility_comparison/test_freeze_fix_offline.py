#!/usr/bin/env python3
"""Simple offline freeze-fix check (engineering gate, NOT scientific evidence).

Builds the CONDITION-NEUTRAL local executor (the H8 belief-space tracker: no GP,
no visibility; constant-R ambiguity is control-invariant so it does not affect the
argmin) with the local parameters from aws_beliefloop_v1_config.yaml, and runs the
MAINTAINED closed_loop_offline toward a waypoint that lies BEYOND the ~1.2 m local
reach (the situation that produced the zero-control freeze).

It sweeps local_goal_progress_weight in {0.0, 0.5}:
  - 0.0  -> expect the documented v=0 freeze (no forward incentive),
  - 0.5  -> expect a progressing, non-zero control tape (the fix).

Run:
  python3 scripts/visibility_comparison/test_freeze_fix_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "scripts" / "visibility_comparison"
sys.path.insert(0, str(LAB))

from efe_offline_lab import load_setup, build_planner, closed_loop_offline  # noqa: E402

CONFIG = LAB / "aws_beliefloop_v1_config.yaml"
TASK = "F31_b1_apron_a3_mid"
# A target ~2.5 m ahead of the start (3.3,-1.0): beyond the H8/dt0.25 (~1.2 m) reach,
# i.e. exactly the beyond-horizon waypoint that triggered the freeze.
TARGET = (3.3, 1.5)
N_STEPS = 50


def build_local(setup, gp_weight: float):
    """Rebuild the local executor planner with the LOCAL params + a chosen progress weight."""
    cfg = dict(setup.config)
    cfg["horizon"] = 8                       # local_horizon
    cfg["goal_progress_n_steps"] = 8         # == local horizon
    cfg["goal_progress_weight"] = gp_weight  # the freeze fix knob
    cfg["control_weight"] = 0.0
    cfg["optimizer_maxiter"] = 45
    cfg["optimizer_multistart"] = False
    cfg["optimizer_initial_routes_json"] = ""
    cfg["optimizer_multistart_lateral_offsets"] = ""
    # tight local goal priors
    cfg["goal_prior_u_std_start"] = 4.0
    cfg["goal_prior_v_std_start"] = 4.0
    cfg["goal_prior_u_std_final"] = 2.0
    cfg["goal_prior_v_std_final"] = 2.0
    # local belief-tube no-go (condition-neutral)
    cfg["use_nogo_cost"] = True
    cfg["nogo_penalty_type"] = "log_barrier"
    cfg["nogo_weight"] = 80.0
    cfg["nogo_safe_distance"] = 0.30
    cfg["use_belief_nogo_cost"] = True
    cfg["nogo_belief_kappa"] = 2.0
    # condition-neutral local: constant R, no GP. (constant-R ambiguity is a
    # control-invariant constant, so planner_kind=constant_R_efe is equivalent here.)
    return build_planner(
        cfg,
        planner_kind="constant_R_efe",
        camera_params=setup.camera_params,
        visibility_artifact_path="",                       # NO GP in the local executor
        visibility_geometry_json="",
        collision_geometry_json=setup.geometry_json,
        driveable_geometry_json=setup.driveable_geometry_json,
        seed=1,
        visibility_target_height_m=setup.gp["target_height"],
    )


def track_waypoints(planner, m0, S0, waypoints, n_steps=120, arrival_r=0.20, dt=0.25):
    """Mirror the runtime LOCAL executor: plan toward the current waypoint, advance
    when within arrival_r, execute controls[0], repeat. Re-seed the warm start on
    every waypoint advance (as efe_agent_node does)."""
    m = np.asarray(m0, dtype=float).reshape(3).copy()
    S = np.asarray(S0, dtype=float).reshape(3, 3).copy()
    wps = [np.asarray(w, dtype=float).reshape(2) for w in waypoints]
    wp_idx = 0
    planner.optimizer_warm_start = True
    planner.prev_controls_flat = None
    vs, traj = [], [m.copy()]
    for _ in range(n_steps):
        target = wps[wp_idx]
        advanced = False
        while wp_idx < len(wps) - 1 and float(np.linalg.norm(m[:2] - target)) < arrival_r:
            wp_idx += 1
            target = wps[wp_idx]
            advanced = True
        if advanced or planner.prev_controls_flat is None:
            try:
                planner.prev_controls_flat = np.asarray(
                    planner._controls_for_waypoints(m[:3], [target]), dtype=float).reshape(-1)
            except Exception:
                pass
        plan = planner.plan(m, S, target)
        u0 = np.asarray(plan.controls[0], dtype=float)
        vs.append(float(u0[0]))
        m, S = planner.predict(m, S, u0)
        traj.append(m.copy())
        if wp_idx == len(wps) - 1 and float(np.linalg.norm(m[:2] - wps[-1])) < arrival_r:
            break
    v = np.array(vs, dtype=float)
    traj = np.asarray(traj)
    return {
        "v": v,
        "traj": traj,
        "reached": bool(np.linalg.norm(traj[-1][:2] - wps[-1]) < arrival_r),
        "final_d": float(np.linalg.norm(traj[-1][:2] - wps[-1])),
        "net": float(np.linalg.norm(traj[-1][:2] - traj[0][:2])),
        "frac_moving": float(np.mean(np.abs(v) > 0.05)) if v.size else 0.0,
        "max_zero_streak": _max_zero_streak(v, dt),
        "wp_reached": wp_idx,
    }


def _max_zero_streak(v, dt):
    best = cur = 0
    for vi in v:
        cur = cur + 1 if abs(vi) <= 0.05 else 0
        best = max(best, cur)
    return best * dt


# Waypoint sequences from the config's optimizer_initial_routes_json (start 3.3,-1.0).
ROUTES = {
    "mid_cross_lane (occluded; N then W turn)": [(3.3, 1.5), (1.5, 1.5), (1.0, 1.75)],
    "lower_sweep_lane (visible; S then W then N)": [(3.3, -2.2), (1.2, -2.2), (1.0, 1.75)],
}


def main() -> int:
    setup = load_setup(CONFIG, condition="C1", task_override=TASK)
    m0 = np.array([setup.start_xy_yaw[0], setup.start_xy_yaw[1], setup.start_xy_yaw[2]], dtype=float)
    S0 = setup.S0.copy()
    reach = 8 * 0.25 * float(setup.config.get("v_max", 0.6))
    print(f"start={tuple(round(v,2) for v in m0)}  local reach (H8*dt0.25*vmax)={reach:.2f} m")
    print("Testing the documented freeze location: aisle-transition TURNS via waypoint tracking.\n")

    for route_name, wps in ROUTES.items():
        print(f"=== route: {route_name} ===")
        for gp_w in (0.0, 0.5):
            planner = build_local(setup, gp_w)
            assert planner.use_visibility_model is False, "local executor must not use GP"
            r = track_waypoints(planner, m0, S0, wps)
            verdict = ("MOVES (no freeze)" if r["frac_moving"] > 0.6 and r["max_zero_streak"] < 1.0
                       else "FROZEN / stalled")
            print(f"  goal_progress_weight={gp_w}: reached={r['reached']} "
                  f"wp={r['wp_reached']}/{len(wps)-1} final_d={r['final_d']:.2f}m "
                  f"net={r['net']:.2f}m frac|v|>0.05={r['frac_moving']:.2f} "
                  f"max_zero_streak={r['max_zero_streak']:.2f}s -> {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
