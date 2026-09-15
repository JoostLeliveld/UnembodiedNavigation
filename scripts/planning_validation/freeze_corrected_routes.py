#!/usr/bin/env python3
"""Freeze the routes selected by the corrected objective, for a provisional campaign.

Reads a validation artifact directory produced by
`run_global_route_validation.py` and writes, next to it, a `frozen_routes.json`
holding -- per task and condition -- the route the corrected objective selected,
its waypoints, its safety verdict and its cost decomposition, plus the campaign
overrides needed to run that route.

This is a derived artifact: it is written into the validation directory it came
from, it never edits the validation outputs, and it refuses to overwrite an
existing `frozen_routes.json`.

Run:
  python3 scripts/planning_validation/freeze_corrected_routes.py \
      paper_artifacts/planning/global_objective_v2_<tag>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_planner_setup import (  # noqa: E402
    lane_graph_candidates,
    load_campaign_config,
    load_tasks,
)

# The footprint and gate set the deployed campaign is validated against.
DEPLOYMENT_FOOTPRINT = "deployed_burger"
DEPLOYMENT_GATE_SET = "full"
# The condition each campaign arm corresponds to.
ARM_CONDITIONS = {
    "C1_constant_R": "q=unity/R=constant_global",
    "C2_visibility_aware": "q=commissioned/R=commissioned",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", help="a global_objective_v2_* directory")
    args = parser.parse_args()

    art = Path(args.artifact_dir)
    if not art.is_absolute():
        art = REPO_ROOT / art
    results_path = art / "results.json"
    if not results_path.is_file():
        print(f"no results.json in {art}", file=sys.stderr)
        return 2
    out_path = art / "frozen_routes.json"
    if out_path.exists():
        print(f"refusing to overwrite {out_path}", file=sys.stderr)
        return 2

    results = json.loads(results_path.read_text(encoding="utf-8"))
    cfg = load_campaign_config()
    tasks = load_tasks()

    frozen: dict = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_artifact": str(art.relative_to(REPO_ROOT)),
        "footprint": DEPLOYMENT_FOOTPRINT,
        "gate_set": DEPLOYMENT_GATE_SET,
        "arms": {},
    }

    for arm, condition in ARM_CONDITIONS.items():
        arm_entry: dict = {"condition": condition, "tasks": {}}
        for task_name, task in tasks.items():
            start = (float(task["start"]["x"]), float(task["start"]["y"]))
            goal = (float(task["goal"]["x"]), float(task["goal"]["y"]))
            candidates = {c.name: c for c in lane_graph_candidates(cfg, start, goal)}
            selected = [
                r for r in results["candidates"]
                if r["task"] == task_name and r["condition"] == condition
                and r["objective"] == "corrected_v2"
                and r["footprint"] == DEPLOYMENT_FOOTPRINT
                and r["gate_set"] == DEPLOYMENT_GATE_SET and r["selected"]
            ]
            if not selected:
                arm_entry["tasks"][task_name] = {
                    "selected_route": None,
                    "note": "no safe candidate under the deployment gate set",
                }
                continue
            row = selected[0]
            candidate = candidates.get(row["route"])
            arm_entry["tasks"][task_name] = {
                "selected_route": row["route"],
                "waypoints": [list(w) for w in candidate.waypoints] if candidate else [],
                "start_xy": list(start),
                "goal_xy": list(goal),
                "safety_status": row["safety_status"],
                "min_body_clearance_m": row["min_body_clearance_m"],
                "route_length_m": row["route_length_m"],
                "travel_time_s": row["travel_time_s"],
                "localization_cost": row["localization_cost"],
                "total_cost": row["total_cost"],
                "optimizer_initial_routes_json": json.dumps(
                    [{"name": candidate.name, "waypoints": [list(w) for w in candidate.waypoints]}]
                ) if candidate else "",
            }
        frozen["arms"][arm] = arm_entry

    frozen["provisional_campaign"] = {
        "seeds": [0],
        "note": (
            "One-seed provisional campaign. Requires ROS 2 + Gazebo + the YOLO "
            "detector weights; it cannot be run from an environment without them."
        ),
        "command": (
            "python3 scripts/visibility_comparison/run_visibility_campaign.py "
            "--config scripts/visibility_comparison/warehouse_visibility_campaign.yaml "
            "--seeds 0 --out logs/campaigns/provisional_corrected_objective_<tag>"
        ),
        "planner_overrides": {
            "localization_cost_mode": "anchored_excess",
            "risk_uses_reference_R": True,
            "route_length_weight": 0.0,
            "travel_time_weight": 0.0,
            "r_reference_uv": -1.0,
        },
    }

    out_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for arm, entry in frozen["arms"].items():
        for task_name, item in entry["tasks"].items():
            print(f"  {arm:22s} {task_name:30s} -> {item.get('selected_route')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
