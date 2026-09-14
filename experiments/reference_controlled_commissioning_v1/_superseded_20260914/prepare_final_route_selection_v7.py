#!/usr/bin/env python3
"""Freeze the bounded look-ahead controller for final route selection/execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v8.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v9.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v9.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v10.json"


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
    config["study_title"] = "Final commissioned four-arm route selection V9"
    config["local_controller_type"] = "pure_pursuit"
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 10,
        "status": "frozen_after_controller_pilot_before_remaining_route_selection",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The turn-then-go pilot accumulated cross-track error between dense "
            "waypoints and proposed an unsafe in-place turn beside bin_office. The "
            "existing 0.60 m look-ahead pure-pursuit tracker passed the unchanged "
            "swept-body guard at the exact refused pose and completed the same frozen "
            "route in Gazebo. Its angular command is now clipped to the configured "
            "[-1,1] rad/s bounds. Pure pursuit is fixed identically for every arm."
        ),
        "controller_pilot_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "controller_pilot_t1_c00_pure_pursuit_v1"
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "route_selection_config_sha256": sha256(OUTPUT_CONFIG),
    })
    roots = protocol["planner_source_tree"]["roots"]
    protocol["planner_source_tree"]["sha256"] = tree_sha256(roots)
    protocol["route_freezer"]["sha256"] = sha256(
        REPO / protocol["route_freezer"]["path"])
    write_new(OUTPUT_PROTOCOL, json.dumps(protocol, indent=2) + "\n")
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
