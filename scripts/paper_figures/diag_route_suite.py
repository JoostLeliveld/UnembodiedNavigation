#!/usr/bin/env python3
"""Cross-task offline route-choice harness (6 visible-goal tasks).

For each task, two condition-neutral multistart SEEDS are offered to BOTH conditions:
  - "short"  : the shorter route that threads a camera-poor region (blind connector /
               rack-shadow / upper occlusion),
  - "long"   : the lane-feasible detour that stays in the GP-visible lower band + A2/A3.
The EFE solve picks one. We sweep the goal-prior final std (the allowed condition-neutral
schedule lever) and report, per task/condition, which route is chosen and the reach.

Goal: find a single hyperparameter setting where the visibility-aware condition (C2) takes
the observable route on the DISCRIMINATING tasks while the constant-R baseline (C1) takes the
blind shortcut, AND both agree on the visible route on the CONTROL tasks (no spurious detour).
No route-forcing (seeds offered to both; planner chooses), no ambiguity-weight inflation.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/visibility_comparison"))
sys.path.insert(0, str(ROOT/"src/planning"))
from efe_offline_lab import build_planner, load_setup

CONFIG = ROOT/"scripts/visibility_comparison/aws_f31b1_final_config.yaml"

# Per-task route-pair seeds (waypoints AFTER the task start). Lane centers:
# west x=-5.0, A1 x=-3.0, A2 x=-1.0, A3 x=1.0, A4 x=3.1; lower_main y=-2.4; connectors y=1.7; upper_cross y=4.5.
TASKS = {
  # ---- discriminating (center/east: real short-blind shortcut vs visible detour) ----
  "F31_b1_apron_a3_mid": dict(kind="discriminator",
    short=[[3.3,1.5],[1.5,1.5],[1.0,1.75]],
    long =[[3.125,-2.4],[1.075,-2.4],[1.075,0.2],[1.0,1.75]]),
  "b5_a4_apron_to_a2_mid": dict(kind="discriminator",
    short=[[3.1,1.7],[1.0,1.7],[-1.0,1.7],[-1.0,1.75]],
    long =[[3.125,-2.4],[-1.0,-2.4],[-1.0,0.2],[-1.0,1.75]]),
  "b2_a0_west_to_a1_upper": dict(kind="discriminator",
    short=[[-5.0,4.58],[-3.0,4.58],[-3.0,3.5]],
    long =[[-5.0,-2.4],[-3.0,-2.4],[-3.0,3.5]]),
  "b3_lowereast_to_a2_mid": dict(kind="discriminator",
    short=[[3.1,-1.0],[3.1,1.7],[-1.0,1.7],[-1.0,1.75]],
    long =[[-1.0,-2.4],[-1.0,0.2],[-1.0,1.75]]),
  # ---- west controls (R1 continuous: visible route dominates; expect C1==C2) ----
  "b6_a0_west_to_a1_low_control": dict(kind="control",
    short=[[-5.0,4.58],[-3.0,4.58],[-3.0,-0.75]],
    long =[[-5.0,-2.4],[-3.0,-2.4],[-3.0,-0.75]]),
  "b7_a0_west_to_a2_mid": dict(kind="control",
    short=[[-5.0,4.58],[-1.0,4.58],[-1.0,1.75]],
    long =[[-5.0,-2.4],[-1.0,-2.4],[-1.0,0.2],[-1.0,1.75]]),
}


def make_planner(cond, task, routes_json, gfs, maxiter, multistart):
    s = load_setup(CONFIG, condition=cond, seed=0, task_override=task)
    cfg = dict(s.config)
    cfg["horizon"] = int(cfg.get("global_horizon", 120))
    cfg["optimizer_maxiter"] = int(maxiter); cfg["optimizer_maxfun"] = int(maxiter)*6
    cfg["optimizer_warm_start"] = False
    cfg["optimizer_multistart"] = bool(multistart)
    cfg["optimizer_multistart_include_direct"] = False
    cfg["optimizer_multistart_lateral_offsets"] = ""
    cfg["optimizer_initial_routes_json"] = json.dumps(routes_json)
    if gfs is not None:
        cfg["goal_prior_u_std_final"] = float(gfs); cfg["goal_prior_v_std_final"] = float(gfs)
    pl = build_planner(cfg, planner_kind=s.planner_kind, camera_params=s.camera_params,
        visibility_artifact_path=str(s.gp_path), visibility_geometry_json=s.geometry_json,
        collision_geometry_json=s.geometry_json, driveable_geometry_json=s.driveable_geometry_json,
        seed=0, visibility_target_height_m=s.gp["target_height"])
    return s, pl


def _poly_dist(states, poly):
    """Mean over trajectory states of the min distance to a route polyline (start+waypoints)."""
    pts = np.asarray(poly, float)
    seg_d = []
    for p in np.asarray(states, float)[:, :2]:
        dmin = np.inf
        for a, b in zip(pts[:-1], pts[1:]):
            ab = b - a; L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            dmin = min(dmin, float(np.hypot(*(p - (a + t * ab)))))
        seg_d.append(dmin)
    return float(np.mean(seg_d))


def realized(st, start_xy=None, short_wps=None, long_wps=None):
    """Robust: assign the trajectory to whichever SEED route (short/long) it matches best.
    Falls back to the lower-band heuristic only if seeds are not provided."""
    if short_wps is not None and long_wps is not None and start_xy is not None:
        s0 = [list(np.asarray(start_xy, float)[:2])]
        ds = _poly_dist(st, s0 + [list(w) for w in short_wps])
        dl = _poly_dist(st, s0 + [list(w) for w in long_wps])
        return "observable/long" if dl < ds else "blind/short"
    return "observable/long" if np.asarray(st)[:, 1].min() < -1.5 else "blind/short"


def held_feasible(cond, task, name, wps):
    tc = TASKS[task]
    s, pl = make_planner(cond, task, [{"name": name, "waypoints": wps}], None, maxiter=1, multistart=False)
    controls = pl._controls_for_waypoints(np.asarray(s.start_xy_yaw, float), [np.asarray(w, float) for w in wps])
    r = pl.evaluate_rollout_controls(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float),
                                     np.asarray(s.goal_xy, float), controls)
    basin = realized(np.asarray(r["states"], float), s.start_xy_yaw, tc["short"], tc["long"])
    return float(r["obstacle_cost"]), basin, float(r["mean_p_vis_plan"])


def solve(cond, task, gfs, maxiter):
    cfg = TASKS[task]
    routes = [{"name":"short","waypoints":cfg["short"]}, {"name":"long","waypoints":cfg["long"]}]
    s, pl = make_planner(cond, task, routes, gfs, maxiter=maxiter, multistart=True)
    plan = pl.plan(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float), np.asarray(s.goal_xy, float))
    st = np.asarray(plan.states, float)
    c = {k: float(getattr(plan, k+"_cost", 0.0)) for k in ["risk","ambiguity","obstacle","total"]}
    d = float(np.hypot(st[-1,0]-s.goal_xy[0], st[-1,1]-s.goal_xy[1]))
    return realized(st, s.start_xy_yaw, cfg["short"], cfg["long"]), d, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfs", type=float, nargs="+", default=[24.0])
    ap.add_argument("--maxiter", type=int, default=350)
    ap.add_argument("--feas", action="store_true", help="also report held feasibility of each seed")
    args = ap.parse_args()

    if args.feas:
        print("== seed feasibility (held nogo; long seed must be ~0) ==")
        for task, cfg in TASKS.items():
            for nm in ("short","long"):
                nogo, basin, pvis = held_feasible("C2", task, nm, cfg[nm])
                print(f"  {task:32} {nm:5} nogo={nogo:9.1f} basin={basin:16} mean_pvis={pvis:.2f}")
        print()

    for gfs in args.gfs:
        print(f"================ goal_final_std = {gfs:.0f}  (maxiter={args.maxiter}) ================")
        print(f"{'task':32} {'kind':13} {'C1 route':17}{'C1 d':>6}   {'C2 route':17}{'C2 d':>6}  {'split?'}")
        for task, tc in TASKS.items():
            r1, d1, c1 = solve("C1", task, gfs, args.maxiter)
            r2, d2, c2 = solve("C2", task, gfs, args.maxiter)
            if tc["kind"] == "discriminator":
                ok = "SPLIT" if (r1=="blind/short" and r2=="observable/long") else ("same" if r1==r2 else "partial")
            else:
                ok = "ctrl-ok" if (r1==r2=="observable/long") else "ctrl-DIVERGE"
            print(f"{task:32} {tc['kind']:13} {r1:17}{d1:6.2f}   {r2:17}{d2:6.2f}  {ok}")


if __name__ == "__main__":
    main()
