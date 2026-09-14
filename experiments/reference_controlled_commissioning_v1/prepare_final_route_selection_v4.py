#!/usr/bin/env python3
"""Freeze the 1.0 m/s, M4-corrected, 1 Hz-fusion route-selection release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v5.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v6.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v6.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v7.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(relative_roots: list[str]) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for root in relative_roots
        for path in (REPO / root).rglob("*.py") if path.is_file()
    )
    for path in files:
        relative = str(path.relative_to(REPO)).encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"immutable release already exists: {path}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["study_title"] = "Final commissioned four-arm route selection V6"
    config["study_comparison"] = (
        "Four planner arms cross constant versus commissioned availability and "
        "covariance forecasts; runtime settings and route-feasibility rules are "
        "shared at 1.0 m/s with 5 Hz detection and covariance-calibrated 1 Hz fusion."
    )
    config["manager_decision_rate_hz"] = 1.0
    config["camera_network_updates_per_step"] = 1
    config["v_max"] = 1.0
    # Circumscribed body radius (0.485412 m) plus the predeclared 0.10 m
    # swept-body clearance. Use the same margin in global and local tracking.
    config["nogo_safe_distance"] = 0.58541219597369
    config["local_nogo_safe_distance"] = 0.58541219597369
    config["thesis_execution_contract"] = "final_1mps_1hz_m4_v1"
    config["run_timeout_after_first_cmd_s"] = 300
    config["ros_domain_id_base"] = 197
    for task in config["tasks"].values():
        task["seeds"] = [91440]
    write_new(
        OUTPUT_CONFIG,
        yaml.safe_dump(config, sort_keys=False, width=100000),
    )

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 7,
        "status": "frozen_before_route_selection",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The accepted single-cell global route exposed a global-to-local handoff "
            "bug before Gazebo: the old extractor selected 1 s global states and could "
            "therefore emit 1 m gaps despite a declared 0.20 m spacing. It now "
            "interpolates at at most 0.20 m, the local controller caps travel at the "
            "active waypoint, and local swept-clearance certification enforces the same "
            "0.10 m margin. The sparse-route diagnostic is excluded."
        ),
        "failed_execution_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_campaign_20260913_v4"
        ),
        "purpose": (
            "Select one hash-bound route for each task and each of the four "
            "commissioned planner arms with the launch-resolved CasADi/L-BFGS-B global "
            "planner at 1.0 m/s, 5 Hz detection and calibrated 1 Hz fusion."
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "route_selection_config_sha256": sha256(OUTPUT_CONFIG),
        "selection_seed": 91440,
        "selector_path": "experiments/icra_commissioning/network_route_probe.py",
        "selector_sha256": sha256(REPO / "experiments/icra_commissioning/network_route_probe.py"),
        "runtime_estimator_contract": {
            "mean_chain": "M3_box_mlp plus gated M4 16x16 visibility residual",
            "runtime_covariance": "R4_image_conditioned_scale fitted to M4-corrected residuals",
            "sensor_model_path": config["manager_visibility_sensor_model_path"],
            "sensor_model_sha256": config["manager_visibility_sensor_model_expected_sha256"],
            "detector_rate_hz": 5.0,
            "fusion_rate_hz": 1.0,
        },
        "route_freezer": {
            "path": "experiments/reference_controlled_commissioning_v1/freeze_single_casadi_route.py",
            "sha256": sha256(ROOT / "freeze_single_casadi_route.py"),
        },
    })
    protocol["launch_resolution"]["sha256"] = sha256(
        REPO / protocol["launch_resolution"]["path"])
    roots = protocol["planner_source_tree"]["roots"]
    protocol["planner_source_tree"]["sha256"] = tree_sha256(roots)
    protocol["route_probe"]["sha256"] = sha256(REPO / protocol["route_probe"]["path"])
    protocol["common_feasibility"]["optimizer_transition_check"] = (
        "CasADi state costs sample constant-twist transitions at no more than 0.20 m; "
        "release admission independently checks the swept rectangular body at 0.04 m."
    )
    protocol.pop("parallel_wrapper", None)
    protocol.pop("selector_dependencies", None)
    protocol.pop("reporting_acceleration", None)
    protocol["common_feasibility"]["definition"] = (
        "Static swept rectangular-body clearance is checked against both the "
        "driveable-lane union and the complete physical collision scene, including "
        "separately included props. Bounded-unicycle swept rollout and terminal-goal "
        "reachability at 1.0 m/s must also agree across all four arms."
    )
    protocol["execution_design"] = (
        "Hash-bind each selected route. Matched stochastic seeds execute routes at "
        "1.0 m/s with 5 Hz detection and 1 Hz camera-manager fusion, without rerunning "
        "the global optimizer."
    )
    write_new(
        OUTPUT_PROTOCOL,
        json.dumps(protocol, indent=2, sort_keys=False) + "\n",
    )
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
