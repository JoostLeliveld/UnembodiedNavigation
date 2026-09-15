#!/usr/bin/env python3
"""Run the DEPLOYED global solve offline with the corrected objective.

`run_global_route_validation.py` validates the full-route *selector*. This script
validates the code path the robot actually runs: `UnicyclePlannerBase.plan`, i.e.
the CasADi-backed L-BFGS-B solve multistarted from the lane-graph route seeds,
with the corrected objective active. It reports, per task and condition:

  * optimizer convergence and iteration count
  * which route seed won
  * predicted terminal goal distance
  * the cost decomposition (goal/risk, localization, travel, no-go)
  * whether the returned rollout passed the planner's own feasibility gate

No Gazebo, no retraining, no artifact is overwritten: this only prints.

Run:
  python3 scripts/planning_validation/check_global_solve.py
  python3 scripts/planning_validation/check_global_solve.py --tasks route_apron_to_a3_mid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_planner_setup import (  # noqa: E402
    DEFAULT_WORLD,
    ObservabilityCondition,
    build_global_planner,
    lane_graph_candidates,
    load_campaign_config,
    load_tasks,
)

CONDITIONS = [
    ObservabilityCondition("unity", "constant_global"),
    ObservabilityCondition("commissioned", "commissioned"),
]
DEFAULT_TASKS = (
    "occlusion_transit_a4",
    "route_apron_to_a3_mid",
    "route_apron_to_a2_mid",
    "route_west_to_a1_upper",
    "control_west_to_a1_low",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    parser.add_argument("--localization-cost-mode", default="anchored_excess",
                        choices=("anchored_excess", "raw_ambiguity"))
    args = parser.parse_args()

    cfg = load_campaign_config()
    tasks = load_tasks()
    goal_radius = float(cfg.get("goal_success_radius", 0.25))

    header = (f"{'task':<24}{'condition':<34}{'ok':<5}{'valid':<7}{'term_gd':>8}"
              f"{'  seed':<26}{'risk':>10}{'loc':>9}{'travel':>8}{'nogo':>12}"
              f"{'nit':>5}{'solve_s':>9}")
    print(f"objective: localization_cost_mode={args.localization_cost_mode}, "
          f"risk_uses_reference_R=True, travel weights from campaign config")
    print(header)
    print("-" * len(header))

    failures = 0
    for task_name in args.tasks:
        task = tasks[task_name]
        start = np.array(
            [float(task["start"]["x"]), float(task["start"]["y"]),
             float(task["start"].get("yaw", 0.0))], dtype=float
        )
        goal = np.array([float(task["goal"]["x"]), float(task["goal"]["y"])], dtype=float)
        seeds = lane_graph_candidates(cfg, start[:2], goal)
        sigma0 = float(cfg.get("init_belief_sigma_xy", 0.05))
        sigma0_theta = float(cfg.get("init_belief_sigma_theta", 0.05))
        S0 = np.diag([sigma0 ** 2, sigma0 ** 2, sigma0_theta ** 2]).astype(float)

        for condition in CONDITIONS:
            planner = build_global_planner(
                cfg,
                condition,
                world_path=DEFAULT_WORLD,
                overrides={"localization_cost_mode": args.localization_cost_mode},
            )
            planner.optimizer_initial_routes = planner._parse_initial_routes(
                json.dumps([{"name": c.name, "waypoints": [list(w) for w in c.waypoints]}
                            for c in seeds])
            )
            t0 = time.perf_counter()
            try:
                plan = planner.plan(start, S0, goal)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"{task_name:<24}{condition.label:<34}ERROR {type(exc).__name__}: {exc}")
                continue
            solve_s = time.perf_counter() - t0
            reached = plan.terminal_goal_distance_pred <= goal_radius
            if not (plan.rollout_valid and reached):
                failures += 1
            print(
                f"{task_name:<24}{condition.label:<34}"
                f"{str(bool(plan.optimizer_success)):<5}{str(bool(plan.rollout_valid)):<7}"
                f"{plan.terminal_goal_distance_pred:8.3f}  "
                f"{str(plan.selected_source)[:24]:<24}"
                f"{plan.risk_cost:10.2f}{plan.ambiguity_cost:9.3f}"
                f"{plan.travel_cost:8.3f}{plan.obstacle_cost:12.1f}"
                f"{plan.optimizer_nit:5d}{solve_s:9.1f}"
            )

    print()
    print(f"{'ALL SOLVES PRODUCED A VALID ROLLOUT THAT REACHES THE GOAL' if failures == 0 else f'{failures} solve(s) did not produce a valid goal-reaching rollout'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
