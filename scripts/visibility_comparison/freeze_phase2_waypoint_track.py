#!/usr/bin/env python3
"""Phase 2 faithful closed-loop: local-EFE executor tracking the advancing
planner-derived waypoint (0.40 m spacing), exactly like efe_agent_node._plan_once
LOCAL phase. This mirrors the REAL runtime, not the adversarial distant-target
probe. Verifies the freeze is gone with the goal-progress incentive while
remaining condition-neutral (C1 and C2 share the identical local executor).

OFFLINE ONLY.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "visibility_comparison"))

import efe_offline_lab as lab  # noqa: E402
from planning.planners.base_planner import extract_waypoints  # noqa: E402
import freeze_phase1_diagnostic as p1  # noqa: E402

WAYPOINT_SPACING_M = 0.40
WAYPOINT_ARRIVAL_RADIUS_M = 0.20


def build_global_planner(setup, planner_kind):
    cfg = dict(setup.config)
    cfg["horizon"] = 80
    return lab.build_planner(
        cfg, planner_kind=planner_kind, camera_params=setup.camera_params,
        visibility_artifact_path=str(setup.gp_path),
        visibility_geometry_json=setup.geometry_json,
        collision_geometry_json=setup.geometry_json,
        driveable_geometry_json=setup.driveable_geometry_json,
        seed=1, visibility_target_height_m=setup.gp["target_height"],
    )


def track_waypoints(setup, planner_kind, overrides, n_steps=120):
    gplan = build_global_planner(setup, planner_kind)
    m0 = np.array([3.30, -1.00, 0.0])
    S0 = np.diag([0.05**2, 0.05**2, 0.05**2])
    goal = np.array(setup.goal_xy, dtype=float)
    rg = gplan.plan(m0, S0, goal)
    wps = extract_waypoints(rg.states, spacing_m=WAYPOINT_SPACING_M, include_goal=True)
    if float(np.linalg.norm(np.asarray(wps[-1]) - goal)) > 1e-3:
        wps.append((float(goal[0]), float(goal[1])))

    if (overrides or {}).get("_multistart"):
        ovr2 = {k: v for k, v in (overrides or {}).items() if k != "_multistart"}
        local = p1.build_local_planner_ms(setup, overrides=ovr2)
    else:
        local = p1.build_local_planner(setup, overrides=overrides)
    # bootstrap warm start to first waypoint (matches agent node)
    wp0 = np.asarray(wps[0], dtype=float)
    local.prev_controls_flat = np.asarray(
        local._controls_for_waypoints(m0[:3], [wp0]), dtype=float).reshape(-1)
    local.optimizer_warm_start = True

    m = m0.copy()
    S = S0.copy()
    wp_idx = 0
    v0_list, dgoal_list, frozen_steps = [], [], 0
    solve_ms = []
    reached = False
    for step in range(n_steps):
        target = np.asarray(wps[wp_idx], dtype=float)
        prev = wp_idx
        while (wp_idx < len(wps) - 1
               and float(np.linalg.norm(m[:2] - target)) < WAYPOINT_ARRIVAL_RADIUS_M):
            wp_idx += 1
            target = np.asarray(wps[wp_idx], dtype=float)
        if wp_idx != prev:
            local.prev_controls_flat = np.asarray(
                local._controls_for_waypoints(m[:3], [target]), dtype=float).reshape(-1)
        plan = local.plan(m, S, target)
        u0 = np.asarray(plan.controls[0], dtype=float)
        v0_list.append(float(u0[0]))
        solve_ms.append(float(plan.solve_time_s) * 1000.0)
        if abs(float(u0[0])) <= 0.05:
            frozen_steps += 1
        m, S = local.predict(m, S, u0)
        dgoal = float(np.linalg.norm(m[:2] - goal))
        dgoal_list.append(dgoal)
        if dgoal < 0.20:
            reached = True
            break

    v0arr = np.asarray(v0_list)
    return {
        "planner_kind": planner_kind,
        "n_waypoints": len(wps),
        "max_wp_gap_m": float(max(
            np.linalg.norm(np.asarray(wps[i + 1]) - np.asarray(wps[i]))
            for i in range(len(wps) - 1))) if len(wps) > 1 else 0.0,
        "steps_run": len(v0_list),
        "reached_goal": reached,
        "d_goal_start": float(np.linalg.norm(m0[:2] - goal)),
        "d_goal_end": dgoal_list[-1] if dgoal_list else float("nan"),
        "v0_mean": float(np.mean(v0arr)) if v0arr.size else 0.0,
        "v0_min": float(np.min(v0arr)) if v0arr.size else 0.0,
        "frozen_steps": frozen_steps,
        "frozen_fraction": float(frozen_steps / max(len(v0_list), 1)),
        "solve_ms_mean": float(np.mean(solve_ms)) if solve_ms else 0.0,
        "solve_ms_max": float(np.max(solve_ms)) if solve_ms else 0.0,
        "plan_duration_s": float(local.horizon * local.dt),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        REPO_ROOT / "scripts/visibility_comparison/aws_beliefloop_v1_config.yaml"))
    ap.add_argument("--weight", type=float, default=-1.0,
                    help="goal_progress_weight; -1 => use config/0.0 (before)")
    ap.add_argument("--multistart", action="store_true",
                    help="enable local optimizer multistart (direct-goal cold seed)")
    ap.add_argument("--tag", default="after")
    args = ap.parse_args()

    overrides = {}
    if args.weight >= 0:
        overrides["goal_progress_weight"] = args.weight
    if args.multistart:
        overrides["_multistart"] = True
    overrides = overrides or None

    setup_c2 = lab.load_setup(Path(args.config), condition="C2")
    setup_c1 = lab.load_setup(Path(args.config), condition="C1")
    print(f"goal_progress_weight override = {overrides}")

    results = {}
    for cond, setup, kind in [
        ("C1", setup_c1, "constant_R_efe"),
        ("C2", setup_c2, "visibility_aware_efe"),
    ]:
        print(f"\n=== {cond} ({kind}) global route -> local waypoint tracking ===")
        rec = track_waypoints(setup, kind, overrides)
        results[cond] = rec
        print(f"  {rec['n_waypoints']} wps (max gap {rec['max_wp_gap_m']:.2f} m); "
              f"steps={rec['steps_run']} reached={rec['reached_goal']} "
              f"d_end={rec['d_goal_end']:.2f}m")
        print(f"  v0_mean={rec['v0_mean']:.3f} v0_min={rec['v0_min']:.3f} "
              f"frozen={rec['frozen_steps']}/{rec['steps_run']} "
              f"({rec['frozen_fraction']*100:.0f}%) "
              f"solve_max={rec['solve_ms_max']:.0f}ms (plan_dur={rec['plan_duration_s']}s)")

    out = (REPO_ROOT / "logs/visibility_comparison/beliefloop_v1/freeze_fix"
           / f"phase2_waypoint_track_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"overrides": overrides, "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
