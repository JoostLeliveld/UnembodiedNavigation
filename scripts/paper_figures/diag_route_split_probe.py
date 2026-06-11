#!/usr/bin/env python3
"""Route-split probe: feasible lane-following lower-sweep seed + goal-prior schedule sweep.

Two condition-neutral levers (allowed by the operating contract; NOT route-forcing waypoints
and NOT ambiguity-weight inflation):
  (1) a lane-feasible lower-sweep optimizer seed (so a clean lower-sweep basin can be held),
  (2) softening the goal-prior tightening (larger final std) so the goal-directed risk term
      stops acting as an implicit length penalty against the longer observable route.

Success = a reasonable setting where C2 (visibility-aware) converges to lower-sweep while
C1 (constant-R) stays on the blind mid-connector — the split emerging from the objective.
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
MID = next(r for r in ALL_ROUTES if r["name"] == "mid_cross_lane")

# Lane-feasible lower-sweep seed: down A4 aisle -> west along lower_main_aisle -> up A3 aisle to goal.
# (A4 center x~3.125, lower_main center y~-2.4, A3 center x~1.075; all inside the driveable union.)
FEAS_LOWER = {"name": "lower_sweep_feasible",
              "waypoints": [[3.125, -2.40], [1.075, -2.40], [1.075, 0.20], [1.0, 1.75]]}


def realized_route(st):
    if st[:, 1].min() < -1.5:
        return "lower-sweep"
    if np.any((st[:, 0] > 1.5) & (st[:, 0] < 2.6) & (st[:, 1] > 1.25) & (st[:, 1] < 2.2)):
        return "connector/mid"
    return "other"


def make_planner(cond, routes_json, goal_final_std, maxiter, multistart):
    s = load_setup(CONFIG, condition=cond, seed=0, task_override=TASK)
    cfg = dict(s.config)
    cfg["horizon"] = int(cfg.get("global_horizon", 120))
    cfg["optimizer_maxiter"] = int(maxiter)
    cfg["optimizer_maxfun"] = int(maxiter) * 6
    cfg["optimizer_warm_start"] = False
    cfg["optimizer_multistart"] = bool(multistart)
    cfg["optimizer_multistart_include_direct"] = False
    cfg["optimizer_multistart_lateral_offsets"] = ""
    cfg["optimizer_initial_routes_json"] = json.dumps(routes_json)
    if goal_final_std is not None:
        cfg["goal_prior_u_std_final"] = float(goal_final_std)
        cfg["goal_prior_v_std_final"] = float(goal_final_std)
    pl = build_planner(
        cfg, planner_kind=s.planner_kind, camera_params=s.camera_params,
        visibility_artifact_path=str(s.gp_path), visibility_geometry_json=s.geometry_json,
        collision_geometry_json=s.geometry_json, driveable_geometry_json=s.driveable_geometry_json,
        seed=0, visibility_target_height_m=s.gp["target_height"])
    return s, pl


# 1. Confirm the feasible lower-sweep seed is actually lane-feasible (low held nogo).
print("== feasibility of the new lower-sweep seed (held, no optimizer) ==")
for cond in ("C2",):
    s, pl = make_planner(cond, [FEAS_LOWER], None, maxiter=1, multistart=False)
    wps = [np.asarray(w, float) for w in FEAS_LOWER["waypoints"]]
    controls = pl._controls_for_waypoints(np.asarray(s.start_xy_yaw, float), wps)
    r = pl.evaluate_rollout_controls(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float),
                                     np.asarray(s.goal_xy, float), controls)
    st = np.asarray(r["states"], float)
    print(f"  {cond} held lower_sweep_feasible: nogo={r['obstacle_cost']:.1f} (crude was ~134k) "
          f"basin={realized_route(st)} mean_pvis={r['mean_p_vis_plan']:.3f} ambig={r['ambiguity_cost']:.1f}")

# 2. Goal-prior sweep with multistart over {mid, feasible-lower-sweep}; does C2 pick lower-sweep?
print("\n== goal-prior sweep, multistart over {mid, feasible lower-sweep}, maxiter=400 ==")
print(f"{'goal_final_std':>14} {'cond':4} {'-> route':14} {'d[m]':>5} {'risk':>9} {'ambig':>10} {'nogo':>8} {'TOTAL':>10}")
for gfs in (18.0, 21.0, 24.0, 27.0, 30.0, 36.0):
    for cond in ("C1", "C2"):
        s, pl = make_planner(cond, [MID, FEAS_LOWER], gfs, maxiter=400, multistart=True)
        plan = pl.plan(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float), np.asarray(s.goal_xy, float))
        st = np.asarray(plan.states, float)
        c = {k: float(getattr(plan, k + "_cost", 0.0)) for k in ["risk", "ambiguity", "obstacle", "total"]}
        d = float(np.hypot(st[-1, 0] - s.goal_xy[0], st[-1, 1] - s.goal_xy[1]))
        print(f"{gfs:14.0f} {cond:4} {realized_route(st):14} {d:5.2f} {c['risk']:9.1f} {c['ambiguity']:10.1f} {c['obstacle']:8.2f} {c['total']:10.1f}")
