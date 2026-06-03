#!/usr/bin/env python3
"""Rigor evidence: EFE cost of the north vs south route, per condition.

Proves the route choice is EMERGENT from the EFE objective (driven by the GP for
C2), not forced by the seeds. For C1 (constant R) and C2 (GP), evaluates the EFE
total + risk + ambiguity for the SAME two candidate routes (north occluded gap,
south visible sweep) and shows the crossover:
  C1 expected: J(north) < J(south)  -> picks north (shorter)
  C2 expected: J(south) < J(north)  -> picks south (GP raises ambiguity on north)

No Gazebo. Uses the offline lab planner + the config's route seeds.

Usage:
  python3 scripts/diagnostics/route_cost_crossover.py \
     --config scripts/visibility_comparison/aws_f69_simple_tracker_config.yaml
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/visibility_comparison"))
from efe_offline_lab import load_setup, eval_controls  # noqa: E402


def controls_for_route(planner, start_xytheta, route_xy):
    wps = [np.asarray(p, dtype=float) for p in route_xy]
    return np.asarray(planner._controls_for_waypoints(np.asarray(start_xytheta, dtype=float), wps),
                      dtype=float).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="F31_b1_apron_a3_mid")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()

    routes = json.loads(json.loads(json.dumps(  # tolerate str
        __import__("yaml").safe_load(cfg_path.read_text())["optimizer_initial_routes_json"]))
        if isinstance(__import__("yaml").safe_load(cfg_path.read_text())["optimizer_initial_routes_json"], str)
        else __import__("yaml").safe_load(cfg_path.read_text())["optimizer_initial_routes_json"])
    route_map = {r["name"]: r["waypoints"] for r in routes}
    print("Routes:", {k: f"{len(v)}wp" for k, v in route_map.items()})

    rows = []
    for cond in ("C1", "C2"):
        setup = load_setup(cfg_path, condition=cond, task_override=args.task)
        m0 = np.array([*setup.start_xy_yaw], dtype=float)
        for rname, rxy in route_map.items():
            u = controls_for_route(setup.planner, m0[:3], rxy)
            res = eval_controls(setup.planner, m0, setup.S0, setup.goal_xy, u)
            rows.append((cond, setup.planner_kind, rname,
                         float(res.get("total_cost", np.nan)),
                         float(res.get("risk_cost", np.nan)),
                         float(res.get("ambiguity_cost", np.nan)),
                         float(res.get("obstacle_cost", np.nan))))
    print(f"\n{'cond':4} {'planner':20} {'route':16} {'J_total':>10} {'J_risk':>10} {'J_amb':>10} {'J_obst':>10}")
    for r in rows:
        print(f"{r[0]:4} {r[1]:20} {r[2]:16} {r[3]:10.2f} {r[4]:10.2f} {r[5]:10.2f} {r[6]:10.2f}")

    # Crossover verdict
    def J(cond, rname):
        for r in rows:
            if r[0] == cond and rname in r[2]:
                return r[3]
        return np.nan
    print("\n=== CROSSOVER ===")
    for cond in ("C1", "C2"):
        north = J(cond, "mid_cross")
        south = J(cond, "lower_sweep")
        pick = "NORTH" if north < south else "SOUTH"
        print(f"  {cond}: J(north)={north:.2f}  J(south)={south:.2f}  -> picks {pick}")


if __name__ == "__main__":
    main()
