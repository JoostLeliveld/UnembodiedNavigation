#!/usr/bin/env python3
"""Offline acid test — given the SAME known warehouse lanes as candidates, does
the GP (C2) flip which lane is feasible vs constant-R (C1)?

For each condition and each lane seed, runs the REAL global optimizer (H=80,
multistart OFF, seeded from that lane) and reports the optimized
J_total / J_risk / J_ambiguity / J_obstacle and rollout_valid. This reproduces
the planner's per-candidate decision offline (no Gazebo, no run-timeout), and
shows WHICH cost term drives the feasibility flip.

Legitimacy: both conditions receive the identical lane set (domain prior — a
warehouse robot knows its aisles). The only difference is the GP. If C2 finds
the occluded lane rollout-invalid while C1 finds it valid+cheap, the route
choice is caused by the GP, not the seeds.

Usage:
  python3 scripts/diagnostics/route_feasibility_offline.py \
     --config scripts/visibility_comparison/aws_f70_turngate_config.yaml
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/visibility_comparison"))
from efe_offline_lab import load_setup, run_optimizer  # noqa: E402


def lane_controls(planner, start_xytheta, lane_xy):
    wps = [np.asarray(p, dtype=float) for p in lane_xy]
    return np.asarray(planner._controls_for_waypoints(np.asarray(start_xytheta, dtype=float), wps),
                      dtype=float).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="F31_b1_apron_a3_mid")
    ap.add_argument("--global-horizon", type=int, default=80)
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()

    raw = yaml.safe_load(cfg_path.read_text())["optimizer_initial_routes_json"]
    routes = json.loads(raw) if isinstance(raw, str) else raw
    lanes = {r["name"]: r["waypoints"] for r in routes}
    print(f"Lanes (domain prior, offered to BOTH conditions): "
          f"{ {k: f'{len(v)}wp' for k,v in lanes.items()} }")
    print(f"global_horizon={args.global_horizon}\n")

    # Build a temp config at the global horizon so plan() uses H=80.
    tmp = cfg_path.read_text()
    import re
    tmp = re.sub(r"^horizon:.*$", f"horizon: {args.global_horizon}", tmp, flags=re.M)
    tmp_path = Path("/tmp/_route_feasibility_cfg.yaml")
    tmp_path.write_text(tmp)

    rows = []
    for cond in ("C1", "C2"):
        setup = load_setup(tmp_path, condition=cond, task_override=args.task)
        # Single-start from each lane seed (no multistart): isolate per-lane cost.
        setup.planner.optimizer_multistart = False
        setup.planner.optimizer_initial_routes = []
        m0 = np.array([*setup.start_xy_yaw], dtype=float)
        for lname, lxy in lanes.items():
            u_seed = lane_controls(setup.planner, m0[:3], lxy)
            res = run_optimizer(setup.planner, m0, setup.S0, setup.goal_xy, u_init=u_seed)
            rows.append((cond, setup.planner_kind, lname,
                         float(getattr(res, "total_cost", np.nan)),
                         float(getattr(res, "risk_cost", np.nan)),
                         float(getattr(res, "ambiguity_cost", np.nan)),
                         float(getattr(res, "obstacle_cost", np.nan)),
                         bool(getattr(res, "rollout_valid", False))))

    print(f"{'cond':4} {'planner':20} {'lane':18} {'J_total':>12} {'J_risk':>9} {'J_amb':>9} {'J_obst':>12} valid")
    for r in rows:
        print(f"{r[0]:4} {r[1]:20} {r[2]:18} {r[3]:12.1f} {r[4]:9.1f} {r[5]:9.1f} {r[6]:12.1f} {r[7]}")

    def pick(cond):
        valids = [r for r in rows if r[0] == cond and r[7]]
        if not valids:
            return "NONE-VALID (safe-stop)"
        best = min(valids, key=lambda r: r[3])
        return f"{best[2]} (J={best[3]:.0f})"
    print("\n=== SELECTED LANE (lowest-J valid candidate) ===")
    for cond in ("C1", "C2"):
        print(f"  {cond}: {pick(cond)}")
    print("\nIf C1 selects the occluded (north/mid_cross) lane and C2 selects the")
    print("visible (south/lower_sweep) lane from the SAME candidate set, the GP")
    print("caused the route difference — not the seeds.")


if __name__ == "__main__":
    main()
