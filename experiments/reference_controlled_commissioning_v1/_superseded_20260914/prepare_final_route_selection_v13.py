#!/usr/bin/env python3
"""Freeze the current-belief segment-feedback follower for every arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v13.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v15.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v14.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v16.json"


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
    config["study_title"] = "Final commissioned four-arm route selection V14"
    config["local_controller_type"] = "ff_fb"
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 16,
        "status": "frozen_before_route_selection_and_four_arm_waypoint_pilots",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "Execution uses the intermediate current-belief segment follower requested "
            "for causal separation. It combines route-segment tangent feedforward with "
            "heading and cross-track feedback, corner speed preview, and waypoint capture. "
            "It performs no q/R lookup, expected camera update, covariance rollout, or "
            "EFE optimization."
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "execution_design": (
            "Hash-bind each arm-specific global route, then execute matched seeds in "
            "Gazebo at 1.0 m/s using the identical current-belief segment-feedback "
            "waypoint follower, M3+M4 correction, temporally inflated R4 covariance, 5 Hz "
            "detector batches, and 5 Hz correction attempts."
        ),
        "local_controller_contract": {
            "type": "ff_fb",
            "state_input": "current planner belief mean [x,y,yaw]",
            "target": "current frozen-route segment and its next corner",
            "camera_quality_query": False,
            "belief_covariance_rollout": False,
            "efe_optimization": False,
            "shared_across_arms": True,
        },
    })
    protocol["route_selection_config_sha256"] = sha256(OUTPUT_CONFIG)
    protocol["selector_sha256"] = sha256(REPO / protocol["selector_path"])
    protocol["launch_resolution"]["sha256"] = sha256(
        REPO / protocol["launch_resolution"]["path"]
    )
    protocol["route_probe"]["sha256"] = sha256(REPO / protocol["route_probe"]["path"])
    protocol["route_freezer"]["sha256"] = sha256(REPO / protocol["route_freezer"]["path"])
    protocol["planner_source_tree"]["sha256"] = tree_sha256(
        protocol["planner_source_tree"]["roots"]
    )
    write_new(OUTPUT_PROTOCOL, json.dumps(protocol, indent=2) + "\n")
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
