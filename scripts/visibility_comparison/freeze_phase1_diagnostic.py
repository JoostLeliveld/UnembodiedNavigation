#!/usr/bin/env python3
"""Phase 1 freeze diagnostic via the MAINTAINED efe_offline_lab.

Builds the LOCAL executor planner (mirrors efe_agent_node local-planner
construction: use_visibility_model=False, use_ambiguity=False, use_obs_risk=True,
goal_progress_n_steps=local_horizon, tight local goal prior, belief-nogo log
barrier) and runs closed_loop_offline from a clear pose toward nearby local
waypoint-style targets. Reports controls[0], truth progress, and an EFE term
breakdown at a (potentially frozen) step.

OFFLINE ONLY. No Gazebo, no ROS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "visibility_comparison"))

import efe_offline_lab as lab  # noqa: E402


# --- LOCAL executor parameter values (mirror runtime local solve) -----------
LOCAL_PARAMS = dict(
    horizon=10,
    dt=0.25,
    v_max=0.60,
    control_weight=0.0,
    goal_prior_u_std_start=4.0,
    goal_prior_v_std_start=4.0,
    goal_prior_u_std_final=2.0,
    goal_prior_v_std_final=2.0,
    goal_tightening_power=0.45,
    goal_progress_n_steps=10,  # = local_horizon, per agent node
    use_nogo_cost=True,
    nogo_penalty_type="log_barrier",
    nogo_weight=80.0,
    nogo_safe_distance=0.30,
    nogo_logbarrier_scale=0.10,
    nogo_logbarrier_eps=0.01,
    use_belief_nogo_cost=True,
    nogo_belief_kappa=2.0,
    use_visibility_model=False,
    use_ambiguity=False,
    use_obs_risk=True,
    optimizer_maxiter=45,
    optimizer_maxfun=180,
    optimizer_multistart=False,
    optimizer_multistart_include_direct=True,
    robot_collision_radius_m=0.125,
    process_noise_theta=0.046,
    process_noise_xy=0.01,
    risk_weight_obs=1.0,
    r_visible_uv=2.5,
    r_miss_uv=40.0,
    observation_risk_scale=1.25,
    ambiguity_weight=8.0,
    discount_gamma=0.98,
    init_belief_sigma_xy=0.05,
    init_belief_sigma_theta=0.05,
)


def build_local_planner(setup, *, overrides=None):
    cfg = dict(setup.config)
    cfg.update(LOCAL_PARAMS)
    if overrides:
        cfg.update(overrides)
    # LOCAL executor is condition-neutral: constant_R_efe == no GP, ambiguity off
    # via use_ambiguity flag. We force the flags directly through build_planner by
    # using planner_kind that yields use_visibility_model=False, but we also need
    # use_ambiguity=False, which constant_R_efe does NOT give. So build manually.
    planner = lab.build_planner(
        cfg,
        planner_kind="constant_R_efe",  # no GP, constant R (condition-neutral base)
        camera_params=setup.camera_params,
        visibility_artifact_path="",
        visibility_geometry_json=setup.geometry_json,
        collision_geometry_json=setup.geometry_json,
        driveable_geometry_json=setup.driveable_geometry_json,
        seed=1,
        visibility_target_height_m=setup.gp["target_height"],
    )
    # Force local-executor flags that build_planner's planner_kind doesn't set.
    planner.use_ambiguity = bool(cfg["use_ambiguity"])
    planner.use_obs_risk = bool(cfg["use_obs_risk"])
    planner.use_visibility_model = False
    return planner


def term_breakdown(planner, m, S, goal_xy, controls):
    res = lab.eval_controls(planner, m, S, goal_xy, controls)
    return {
        "total": res["total_cost"],
        "risk": res["risk_cost"],
        "ambiguity": res["ambiguity_cost"],
        "control": res["control_cost"],
        "obstacle": res["obstacle_cost"],
        "rollout_valid": res["rollout_valid"],
        "invalid_reason": res["invalid_reason"],
        "terminal_progress_m": res["terminal_goal_progress_m"],
    }


def probe_zero_vs_forward(planner, m, S, goal_xy):
    """Compare EFE of zero-control vs a straight forward-to-goal rollout."""
    H = planner.horizon
    dt = planner.dt
    vmax = planner.v_max
    # zero control
    zero = np.zeros((H, 2), dtype=float)
    # forward: turn toward goal instantly (use goal-heading seed), then drive
    yaw = float(m[2])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    desired = np.arctan2(gy - m[1], gx - m[0])
    fwd = lab.make_controls_to_heading(H, dt, vmax, yaw_start=yaw, yaw_target=desired, w_turn=1.0)
    return {
        "zero_control": term_breakdown(planner, m, S, goal_xy, zero),
        "forward_to_goal": term_breakdown(planner, m, S, goal_xy, fwd),
    }


def run_target(setup, label, start_xyz, goal_xy, n_steps=24, overrides=None):
    planner = build_local_planner(setup, overrides=overrides)
    m0 = np.asarray(start_xyz, dtype=float).reshape(3)
    sxy = LOCAL_PARAMS["init_belief_sigma_xy"]
    sth = LOCAL_PARAMS["init_belief_sigma_theta"]
    S0 = np.diag([sxy**2, sxy**2, sth**2])

    out = lab.closed_loop_offline(planner, m0, S0, goal_xy, n_steps, warm_start=True)
    infos = out["infos"]
    traj = out["traj_m"]

    u0s = np.asarray([i["u"] for i in infos], dtype=float)
    v0s = u0s[:, 0] if u0s.size else np.array([])
    solve_ms = np.asarray([i["solver_time_ms"] for i in infos], dtype=float)
    d0 = float(np.linalg.norm(m0[:2] - np.asarray(goal_xy)))
    dN = infos[-1]["d_goal"] if infos else d0
    total_disp = float(np.linalg.norm(traj[-1][:2] - traj[0][:2])) if len(traj) > 1 else 0.0

    frozen = bool(v0s.size and np.max(np.abs(v0s)) <= 0.05)
    # term breakdown at first step (cold-ish) — use zero vs forward probe
    probe = probe_zero_vs_forward(build_local_planner(setup, overrides=overrides),
                                  m0, S0, goal_xy)

    rec = {
        "label": label,
        "start": [float(x) for x in m0.tolist()],
        "goal": [float(goal_xy[0]), float(goal_xy[1])],
        "n_steps_run": len(infos),
        "d_goal_start": d0,
        "d_goal_end": float(dN),
        "progress_m": float(d0 - dN),
        "total_truth_displacement_m": total_disp,
        "v0_max_abs": float(np.max(np.abs(v0s))) if v0s.size else 0.0,
        "v0_mean": float(np.mean(v0s)) if v0s.size else 0.0,
        "v0_first": float(v0s[0]) if v0s.size else 0.0,
        "v0_all": [round(float(v), 4) for v in v0s.tolist()],
        "frozen": frozen,
        "solve_ms_mean": float(np.mean(solve_ms)) if solve_ms.size else 0.0,
        "solve_ms_max": float(np.max(solve_ms)) if solve_ms.size else 0.0,
        "plan_duration_s": float(planner.horizon * planner.dt),
        "probe_zero_vs_forward": probe,
    }
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        REPO_ROOT / "scripts/visibility_comparison/aws_beliefloop_v1_config.yaml"))
    ap.add_argument("--out", default=str(
        REPO_ROOT / "logs/visibility_comparison/beliefloop_v1/freeze_fix/phase1.json"))
    ap.add_argument("--tag", default="before")
    ap.add_argument("--override", default="", help="JSON dict of LOCAL_PARAMS overrides")
    args = ap.parse_args()

    overrides = json.loads(args.override) if args.override.strip() else None

    setup = lab.load_setup(Path(args.config), condition="C1")
    print(f"world={setup.world} task={setup.task_name} "
          f"start={setup.start_xy_yaw} goal={setup.goal_xy}")
    print(f"local overrides: {overrides}")

    # Targets: nearby local-waypoint-style targets on the occluded mid_cross_lane,
    # plus the two named intermediate-goal probes.
    start = (3.30, -1.00, 0.0)
    targets = [
        # ~1-1.5 m ahead toward the mid cross lane (north), occluded transition
        ("wp_north_1.0m", start, (3.30, 0.00)),
        ("wp_north_1.5m", start, (3.30, 0.50)),
        ("mid_cross_1.5_1.5", start, (1.5, 1.5)),
        ("goal_1.0_1.75", start, (1.0, 1.75)),
    ]

    results = []
    for label, s, g in targets:
        print(f"\n=== {label}: start={s} goal={g} ===")
        rec = run_target(setup, label, s, g, n_steps=24, overrides=overrides)
        results.append(rec)
        print(f"  v0_first={rec['v0_first']:.4f} v0_max={rec['v0_max_abs']:.4f} "
              f"frozen={rec['frozen']} progress={rec['progress_m']:.3f}m "
              f"disp={rec['total_truth_displacement_m']:.3f}m "
              f"solve_max={rec['solve_ms_max']:.0f}ms (plan_dur={rec['plan_duration_s']}s)")
        p = rec["probe_zero_vs_forward"]
        print(f"  PROBE zero:    total={p['zero_control']['total']:.3f} "
              f"risk={p['zero_control']['risk']:.3f} obs={p['zero_control']['obstacle']:.3f} "
              f"valid={p['zero_control']['rollout_valid']}")
        print(f"  PROBE forward: total={p['forward_to_goal']['total']:.3f} "
              f"risk={p['forward_to_goal']['risk']:.3f} obs={p['forward_to_goal']['obstacle']:.3f} "
              f"valid={p['forward_to_goal']['rollout_valid']} "
              f"prog={p['forward_to_goal']['terminal_progress_m']:.3f}m")
        better = "FORWARD" if p["forward_to_goal"]["total"] < p["zero_control"]["total"] else "ZERO"
        print(f"  -> objective prefers: {better}")

    out_path = Path(args.out).with_name(
        Path(args.out).stem + f"_{args.tag}" + Path(args.out).suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"overrides": overrides, "results": results}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()


def build_local_planner_ms(setup, *, overrides=None):
    """Local planner with multistart (direct-goal cold seed) enabled."""
    ovr = dict(overrides or {})
    ovr.update(dict(
        optimizer_multistart=True,
        optimizer_multistart_include_direct=True,
        optimizer_multistart_lateral_offsets="",
        optimizer_initial_routes_json="",
    ))
    return build_local_planner(setup, overrides=ovr)
