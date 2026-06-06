#!/usr/bin/env python3
"""Local-controller solve-time benchmark (engineering gate, NOT scientific evidence).

Sweeps local-controller variants of increasing informativeness and reports the
solve-time DISTRIBUTION (median / p90 / p95) against the 5 Hz control budget
(0.2 s; targets median<80, p90<150, p95<200 ms), plus goal-reach on both the
occluded (mid_cross_lane) and visible (lower_sweep_lane) routes. The point is to
choose the most informative local objective that still meets timing.

All variants are condition-neutral, H8/dt0.25, single-start, warm-started, no
multistart. Visibility/GP never enters the local layer.

Caveat: offline scipy-L-BFGS-B + compiled-CasADi timing is representative but the
final p90/p95 must be re-confirmed on the real ROS runtime.

Run:
  python3 scripts/visibility_comparison/benchmark_local_controllers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "scripts" / "visibility_comparison"
sys.path.insert(0, str(LAB))

from efe_offline_lab import load_setup, build_planner  # noqa: E402

CONFIG = LAB / "aws_beliefloop_v1_config.yaml"
TASK = "F31_b1_apron_a3_mid"
ROUTES = {
    "mid": [(3.3, 1.5), (1.5, 1.5), (1.0, 1.75)],     # occluded
    "low": [(3.3, -2.2), (1.2, -2.2), (1.0, 1.75)],   # visible
}


def build_variant(setup, *, horizon=8, maxiter=45, use_obs_risk=False, use_ambiguity=False,
                  belief_nogo=True, state_nogo=False, control_weight=0.0, goal_w=0.5):
    cfg = dict(setup.config)
    cfg.update(dict(
        horizon=horizon, goal_progress_n_steps=horizon, goal_progress_weight=goal_w,
        control_weight=control_weight, optimizer_maxiter=maxiter, optimizer_maxfun=maxiter * 4,
        optimizer_multistart=False, optimizer_initial_routes_json="",
        optimizer_multistart_lateral_offsets="",
        goal_prior_u_std_start=4.0, goal_prior_v_std_start=4.0,
        goal_prior_u_std_final=2.0, goal_prior_v_std_final=2.0,
        use_nogo_cost=bool(belief_nogo or state_nogo),
        nogo_penalty_type="log_barrier", nogo_weight=80.0, nogo_safe_distance=0.30,
        use_belief_nogo_cost=bool(belief_nogo), nogo_belief_kappa=2.0,
        use_obs_risk=use_obs_risk, use_ambiguity=use_ambiguity,
    ))
    return build_planner(
        cfg, planner_kind="constant_R_efe", camera_params=setup.camera_params,
        visibility_artifact_path="", visibility_geometry_json="",
        collision_geometry_json=setup.geometry_json,
        driveable_geometry_json=setup.driveable_geometry_json, seed=1,
        visibility_target_height_m=setup.gp["target_height"])


def track(planner, m0, S0, wps, n_steps=140, arrival_r=0.20):
    m = np.asarray(m0, float).reshape(3).copy()
    S = np.asarray(S0, float).reshape(3, 3).copy()
    wps = [np.asarray(w, float).reshape(2) for w in wps]
    wp_idx = 0
    planner.optimizer_warm_start = True
    planner.prev_controls_flat = None
    vs, solves = [], []
    for _ in range(n_steps):
        target = wps[wp_idx]
        adv = False
        while wp_idx < len(wps) - 1 and float(np.linalg.norm(m[:2] - target)) < arrival_r:
            wp_idx += 1; target = wps[wp_idx]; adv = True
        if adv or planner.prev_controls_flat is None:
            try:
                planner.prev_controls_flat = np.asarray(
                    planner._controls_for_waypoints(m[:3], [target]), float).reshape(-1)
            except Exception:
                pass
        plan = planner.plan(m, S, target)
        solves.append(float(getattr(plan, "solve_time_s", 0.0)) * 1000.0)
        u0 = np.asarray(plan.controls[0], float)
        vs.append(float(u0[0]))
        m, S = planner.predict(m, S, u0)
        if wp_idx == len(wps) - 1 and float(np.linalg.norm(m[:2] - wps[-1])) < arrival_r:
            break
    v = np.array(vs);
    return {
        "reached": bool(np.linalg.norm(m[:2] - wps[-1]) < arrival_r),
        "final_d": float(np.linalg.norm(m[:2] - wps[-1])),
        "frac_moving": float(np.mean(np.abs(v) > 0.05)) if v.size else 0.0,
        "solves_ms": solves,
    }


def run_variant(setup, m0, S0, label, **kw):
    solves_all = []
    reach = {}
    for rn, wps in ROUTES.items():
        pl = build_variant(setup, **kw)
        r = track(pl, m0, S0, wps)
        solves_all += r["solves_ms"]
        reach[rn] = (r["reached"], r["final_d"], r["frac_moving"])
    s = np.array(solves_all)
    med, p90, p95, mx = (np.median(s), np.percentile(s, 90), np.percentile(s, 95), s.max())
    ok = "OK " if (med < 80 and p90 < 150 and p95 < 200) else "SLOW"
    mid, low = reach["mid"], reach["low"]
    print(f"{label:38s} med={med:6.1f} p90={p90:6.1f} p95={p95:6.1f} max={mx:6.1f} ms [{ok}] "
          f"mid(reach={mid[0]},d={mid[1]:.2f},mv={mid[2]:.2f}) low(reach={low[0]},d={low[1]:.2f},mv={low[2]:.2f})")
    return med, p90, p95


def main() -> int:
    setup = load_setup(CONFIG, condition="C1", task_override=TASK)
    m0 = np.array([*setup.start_xy_yaw], float); S0 = setup.S0.copy()
    print("Local-controller solve-time benchmark (H8/dt0.25 unless swept; single-start, warm-start)")
    print("budget: median<80  p90<150  p95<200 ms  (5 Hz -> 200 ms cycle)\n")
    print("--- informativeness ladder ---")
    run_variant(setup, m0, S0, "V1 goal-only", belief_nogo=False, state_nogo=False)
    run_variant(setup, m0, S0, "V2 goal+belief-tube (chosen)", belief_nogo=True)
    run_variant(setup, m0, S0, "V3 goal+state-nogo", belief_nogo=False, state_nogo=True)
    run_variant(setup, m0, S0, "V4 V2+control-smooth(0.05)", belief_nogo=True, control_weight=0.05)
    run_variant(setup, m0, S0, "V5 V2+obs-risk", belief_nogo=True, use_obs_risk=True)
    run_variant(setup, m0, S0, "V6 V2+ambiguity", belief_nogo=True, use_ambiguity=True)
    run_variant(setup, m0, S0, "V7 V2+obs-risk+amb (full EFE)", belief_nogo=True,
                use_obs_risk=True, use_ambiguity=True)
    print("\n--- horizon sweep on V2 ---")
    for H in (6, 8, 12, 16):
        run_variant(setup, m0, S0, f"V2 H={H}", horizon=H, belief_nogo=True)
    print("\n--- maxiter sweep on V2 (H8) ---")
    for mi in (25, 45, 80):
        run_variant(setup, m0, S0, f"V2 maxiter={mi}", maxiter=mi, belief_nogo=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
