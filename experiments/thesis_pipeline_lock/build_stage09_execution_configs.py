#!/usr/bin/env python3
"""Build immutable preselected-route Stage-09 execution configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_yaml(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable configuration: {path}")
    text = yaml.safe_dump(value, sort_keys=False, width=1000)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--campaign-output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.selection_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("route selection is not complete")
    protocol_path = (REPO / manifest["protocol_path"]).resolve()
    if _sha(protocol_path) != manifest["protocol_sha256"]:
        raise RuntimeError("route-selection protocol hash no longer matches")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_path = (REPO / protocol["route_selection_config"]).resolve()
    if _sha(base_path) != protocol["route_selection_config_sha256"]:
        raise RuntimeError("route-selection base configuration hash no longer matches")
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    config["study_title"] = "Thesis final frozen-route Q0-versus-Q1 navigation campaign"
    config["study_comparison"] = (
        "Routes are selected once with Q0 or Q1 under the frozen Stage-09 protocol; "
        "five matched stochastic seeds then execute the hash-bound routes with identical "
        "runtime perception, fusion, controller and noise."
    )
    for arm, condition in config["conditions"].items():
        condition.pop("camera_network_artifact_path", None)
        condition["label"] = (
            "route_selected_with_constant_Q0"
            if arm == "P0"
            else "route_selected_with_commissioned_Q1"
        )
    # YAML anchors in the design configuration intentionally share the paired
    # condition/seed block.  Detach every task before adding task-specific routes;
    # mutating an aliased mapping would otherwise assign the final task's route to
    # every task.
    detached_tasks = {}
    for task_name, source_task in config["tasks"].items():
        task = deepcopy(source_task)
        task["preselected_routes"] = {}
        for arm in task["conditions"]:
            selected = manifest["selected"][task_name][arm]
            task["preselected_routes"][arm] = {
                key: selected[key]
                for key in (
                    "preselected_route_json",
                    "preselected_route_sha256",
                    "preselected_route_source_path",
                    "preselected_route_source_sha256",
                )
            }
        detached_tasks[task_name] = task
    config["tasks"] = detached_tasks

    # At 960 px, the serialized causal EKF sustained fewer than five updates per
    # simulated second.  Two hertz is fixed for both arms before evidence collection;
    # the deeper event queue remains a loss-prevention backstop, not a rate control.
    config["manager_decision_rate_hz"] = 2.0
    config["global_planner_mode"] = "preselected_route"
    config["camera_network_objective"] = "legacy_pixel_chart"
    config["global_optimizer_multistart"] = False
    config["optimizer_multistart_include_direct"] = False
    config["preselected_route_clearance_m"] = float(
        protocol["route_gate"]["centerline_clearance_m"]
    )
    config["preselected_route_endpoint_tolerance_m"] = float(
        protocol["route_gate"]["endpoint_tolerance_m"]
    )
    config["preselected_route_sample_step_m"] = float(
        protocol["route_gate"]["sample_step_m"]
    )
    _write_yaml(args.campaign_output.resolve(), config)

    preflight = deepcopy(config)
    preflight["study_title"] = "Stage-09 960-pixel runtime and evidence-ledger preflight"
    preflight["study_comparison"] = (
        "Non-evidence run of one body-clearance-gated frozen route with the final detector, "
        "measurement model, availability model, correction rate and ledger runtime."
    )
    task_name = "thesis09_south_to_north_central"
    task = deepcopy(preflight["tasks"][task_name])
    task["conditions"] = ["P1"]
    task["seeds"] = [899]
    task["preselected_routes"] = {"P1": task["preselected_routes"]["P1"]}
    preflight["tasks"] = {task_name: task}
    preflight["conditions"] = {"P1": preflight["conditions"]["P1"]}
    preflight["ros_domain_id_base"] = 178
    _write_yaml(args.preflight_output.resolve(), preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
