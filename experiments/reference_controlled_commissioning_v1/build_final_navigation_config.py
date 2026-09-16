#!/usr/bin/env python3
"""Build hash-bound final or pilot navigation YAML from frozen route selection."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


ARMS = ("C00", "C01", "C10", "C11")
TASKS = (
    "thesis09_west_to_east_north",
    "thesis09_east_to_west_south",
    "thesis09_aisle_to_crossaisle",
    "thesis09_south_to_north_central",
)
ROUTE_KEYS = (
    "preselected_route_json",
    "preselected_route_sha256",
    "preselected_route_source_path",
    "preselected_route_source_sha256",
)
FINAL_NOMINAL_CRUISE_MPS = 1.0
FINAL_FUSION_RATE_HZ = 5.0
FINAL_EXECUTION_CONTRACT = "final_1mps_5hz_m4_temporal_v1"
FINAL_EFFECTIVE_PLANNING_RATE_HZ = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-task", choices=TASKS)
    parser.add_argument("--pilot-arm", choices=ARMS)
    parser.add_argument("--pilot-seed", type=int, default=91599)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[91500, 91501, 91502, 91503, 91504],
        help="Matched execution seeds for a full campaign.",
    )
    args = parser.parse_args()
    if bool(args.pilot_task) != bool(args.pilot_arm):
        raise ValueError("pilot-task and pilot-arm must be supplied together")
    if args.output.exists():
        raise RuntimeError("navigation configs are immutable; choose a new output")
    config = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(args.selection_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("route selection is not complete")
    # The thesis-facing protocol fixes both rates.  Do not inherit a development
    # throttle from a route-selection or runtime-debug base configuration.
    config["v_max"] = FINAL_NOMINAL_CRUISE_MPS
    config["manager_decision_rate_hz"] = FINAL_FUSION_RATE_HZ
    expected_updates = int(round(
        FINAL_EFFECTIVE_PLANNING_RATE_HZ
        * float(config.get("global_dt", 1.0) or 1.0)
    ))
    config["camera_network_updates_per_step"] = expected_updates
    config["thesis_execution_contract"] = FINAL_EXECUTION_CONTRACT
    if float(config["v_max"]) != FINAL_NOMINAL_CRUISE_MPS:
        raise RuntimeError("final navigation must use 1.0 m/s nominal cruise")
    if float(config["manager_decision_rate_hz"]) != FINAL_FUSION_RATE_HZ:
        raise RuntimeError("final navigation must fuse at 5.0 Hz")
    selected = manifest["selected"]
    active_tasks = (args.pilot_task,) if args.pilot_task else TASKS
    active_arms = (args.pilot_arm,) if args.pilot_arm else ARMS
    seeds = [args.pilot_seed] if args.pilot_task else list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("execution seeds must be non-empty and unique")

    config["study_title"] = (
        "Final commissioned runtime route pilot"
        if args.pilot_task else "Final commissioned four-arm navigation campaign"
    )
    config["route_selection_manifest_path"] = str(args.selection_manifest)
    config["route_selection_manifest_sha256"] = hashlib.sha256(
        args.selection_manifest.read_bytes()
    ).hexdigest()
    config["study_comparison"] = (
        "Non-inferential hash-bound runtime and timing pilot."
        if args.pilot_task else
        "Matched four-by-two planner factorial: constant versus commissioned "
        "availability crossed with constant versus commissioned spatial covariance."
    )
    # Planner fields are consumed only during route selection.  Execution uses
    # the resulting route and source hash; forwarding a field would incorrectly
    # ask the runtime planner to solve again.
    config["conditions"] = {
        arm: {"label": config["conditions"][arm]["label"]}
        for arm in active_arms
    }
    config["tasks"] = {}
    for task in active_tasks:
        routes = {}
        for arm in active_arms:
            entry = selected[task][arm]
            if entry.get("status") != "selected":
                raise RuntimeError(f"route is not selected: {task}/{arm}")
            route = {key: entry[key] for key in ROUTE_KEYS}
            route.update({
                "preselected_route_clearance_m": 0.25,
                "preselected_route_endpoint_tolerance_m": 0.0,
                "preselected_route_sample_step_m": 0.04,
            })
            routes[arm] = route
        config["tasks"][task] = {
            "conditions": list(active_arms),
            "seeds": seeds,
            "preselected_routes": routes,
        }
    config["global_planner_mode"] = "preselected_route"
    config["camera_network_objective"] = "legacy_pixel_chart"
    config["global_optimizer_multistart"] = False
    config["optimizer_initial_routes_json"] = ""
    config["ros_domain_id_base"] = 120 if not args.pilot_task else 196
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False, width=100000), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
