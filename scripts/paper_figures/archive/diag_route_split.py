#!/usr/bin/env python3
"""Offline route-split diagnosis under the new Q_d + union-boundary keep-in.

For each condition (C1 constant-R, C2 visibility-aware) and each candidate route
(mid_cross_lane = short/blind, lower_sweep_lane = long/observable), force that ONE
route as the single optimizer seed (no multistart collapse), solve the global EFE,
and report the converged term breakdown + which basin it landed in. This isolates
whether the objective ranks the observable route below the blind one for C2 (and
not for C1) — i.e. whether the route split is latent in the cost.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/visibility_comparison"))
sys.path.insert(0, str(ROOT/"src/planning"))
from efe_offline_lab import build_planner, load_setup

CONFIG = ROOT/"scripts/visibility_comparison/aws_f31b1_final_config.yaml"
TASK = "F31_b1_apron_a3_mid"
ALL_ROUTES = json.loads(open(CONFIG).read().split("optimizer_initial_routes_json: '")[1].split("'")[0])
ROUTE_BY_NAME = {r["name"]: r for r in ALL_ROUTES}


def realized_route(st):
    if st[:, 1].min() < -1.5:
        return "lower-sweep"
    if np.any((st[:, 0] > 1.5) & (st[:, 0] < 2.6) & (st[:, 1] > 1.25) & (st[:, 1] < 2.2)):
        return "connector/mid"
    return "other"


def solve_single_route(cond, route_name, maxiter=800):
    s = load_setup(CONFIG, condition=cond, seed=0, task_override=TASK)
    cfg = dict(s.config)
    cfg["horizon"] = int(cfg.get("global_horizon", 120))
    cfg["optimizer_maxiter"] = int(maxiter)
    cfg["optimizer_maxfun"] = max(int(maxiter) * 6, 1)
    cfg["optimizer_warm_start"] = False
    cfg["optimizer_multistart"] = False               # single seed = this route only
    cfg["optimizer_multistart_include_direct"] = False
    cfg["optimizer_multistart_lateral_offsets"] = ""
    cfg["optimizer_initial_routes_json"] = json.dumps([ROUTE_BY_NAME[route_name]])
    pl = build_planner(
        cfg, planner_kind=s.planner_kind, camera_params=s.camera_params,
        visibility_artifact_path=str(s.gp_path), visibility_geometry_json=s.geometry_json,
        collision_geometry_json=s.geometry_json, driveable_geometry_json=s.driveable_geometry_json,
        seed=0, visibility_target_height_m=s.gp["target_height"])
    plan = pl.plan(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float), np.asarray(s.goal_xy, float))
    st = np.asarray(plan.states, float)
    costs = {k: float(getattr(plan, k + "_cost", 0.0)) for k in ["risk", "ambiguity", "obstacle", "total"]}
    d = float(np.hypot(st[-1, 0] - s.goal_xy[0], st[-1, 1] - s.goal_xy[1]))
    return costs, d, realized_route(st), st


def held_route(cond, route_name):
    """Evaluate the route as a HELD trajectory (route-seed controls, NO optimizer).
    Pure cost evaluation of 'what if the robot committed to this route'."""
    s = load_setup(CONFIG, condition=cond, seed=0, task_override=TASK)
    cfg = dict(s.config)
    cfg["horizon"] = int(cfg.get("global_horizon", 120))
    cfg["optimizer_multistart"] = False
    cfg["optimizer_initial_routes_json"] = json.dumps([ROUTE_BY_NAME[route_name]])
    pl = build_planner(
        cfg, planner_kind=s.planner_kind, camera_params=s.camera_params,
        visibility_artifact_path=str(s.gp_path), visibility_geometry_json=s.geometry_json,
        collision_geometry_json=s.geometry_json, driveable_geometry_json=s.driveable_geometry_json,
        seed=0, visibility_target_height_m=s.gp["target_height"])
    wps = [np.asarray(w, float) for w in ROUTE_BY_NAME[route_name]["waypoints"]]
    controls = pl._controls_for_waypoints(np.asarray(s.start_xy_yaw, float), wps)
    r = pl.evaluate_rollout_controls(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float),
                                     np.asarray(s.goal_xy, float), controls)
    st = np.asarray(r["states"], float)
    return r, realized_route(st)

print("\n================ HELD TRAJECTORY (route seed, NO optimizer) ================")
print(f"{'cond':4} {'route':16} {'-> realized':14} {'risk':>9} {'ambig':>10} {'nogo':>8} {'TOTAL':>10}  {'mean_pvis':>9} {'frac_low':>8}")
held = {}
for cond in ("C1", "C2"):
    for rname in ("mid_cross_lane", "lower_sweep_lane"):
        r, basin = held_route(cond, rname)
        held[(cond, rname)] = r
        print(f"{cond:4} {rname:16} {basin:14} {r['risk_cost']:9.1f} {r['ambiguity_cost']:10.1f} "
              f"{r['obstacle_cost']:8.2f} {r['total_cost']:10.1f}  {r['mean_p_vis_plan']:9.3f} {r['fraction_horizon_low_pvis']:8.2f}")
print("  -- ranking (HELD trajectory) --")
for cond in ("C1", "C2"):
    mid = held[(cond, "mid_cross_lane")]["total_cost"]
    low = held[(cond, "lower_sweep_lane")]["total_cost"]
    pref = "mid_cross (short/blind)" if mid < low else "lower_sweep (long/observable)"
    print(f"    {cond}: mid={mid:.1f}  lower_sweep={low:.1f}  -> prefers {pref}  (gap={abs(mid-low):.1f})")

for maxiter, label in [(800, "CONVERGED (maxiter=800)")]:
    print(f"\n================ {label} ================")
    print(f"{'cond':4} {'seed route':16} {'-> basin':14} {'d[m]':>5}  {'risk':>9} {'ambig':>10} {'nogo':>8} {'TOTAL':>10}")
    results = {}
    for cond in ("C1", "C2"):
        for rname in ("mid_cross_lane", "lower_sweep_lane"):
            c, d, basin, st = solve_single_route(cond, rname, maxiter=maxiter)
            results[(cond, rname)] = (c, d, basin)
            print(f"{cond:4} {rname:16} {basin:14} {d:5.2f}  {c['risk']:9.1f} {c['ambiguity']:10.1f} {c['obstacle']:8.2f} {c['total']:10.1f}")
    print(f"  -- ranking ({label}) --")
    for cond in ("C1", "C2"):
        mid = results[(cond, "mid_cross_lane")][0]["total"]
        low = results[(cond, "lower_sweep_lane")][0]["total"]
        pref = "mid_cross (short/blind)" if mid < low else "lower_sweep (long/observable)"
        print(f"    {cond}: mid={mid:.1f}  lower_sweep={low:.1f}  -> prefers {pref}  (gap={abs(mid-low):.1f})")
