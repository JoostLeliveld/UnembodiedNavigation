#!/usr/bin/env python3
"""Freeze the endpoint-robust final route-selection release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v6.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v7.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v7.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v8.json"
TASKS = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_tasks.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"immutable release already exists: {path}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["study_title"] = "Final commissioned four-arm route selection V7"
    write_new(
        OUTPUT_CONFIG,
        yaml.safe_dump(config, sort_keys=False, width=100000),
    )

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 8,
        "status": "frozen_before_route_selection",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The first hash-bound Gazebo pilot failed closed at the first in-place turn. "
            "The prescribed west endpoint (-7.90,-8.70), not the optimizer, placed the "
            "robot only 0.105 m from a physical pallet jack and left insufficient robust "
            "clearance for the commissioned camera-belief error. The reciprocal T1 start "
            "and T2 goal move to (-7.60,-8.50), where yaw-zero physical and driveable "
            "body clearances are 0.405 m and 0.500 m and the sampled all-yaw minimum is "
            "0.315 m. The failed pilot is retained as diagnostic provenance."
        ),
        "failed_execution_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v6"
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "route_selection_config_sha256": sha256(OUTPUT_CONFIG),
        "tasks_yaml_sha256": sha256(TASKS),
    })
    write_new(
        OUTPUT_PROTOCOL,
        json.dumps(protocol, indent=2, sort_keys=False) + "\n",
    )
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
