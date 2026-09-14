#!/usr/bin/env python3
"""Freeze goal-entry truncation before accepting the remaining route matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v7.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v8.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v8.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v9.json"
FREEZER = ROOT / "freeze_single_casadi_route.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"immutable release already exists: {path}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["study_title"] = "Final commissioned four-arm route selection V8"
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 9,
        "status": "frozen_before_route_acceptance",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The safe-endpoint diagnostic entered the 0.35 m goal region but its "
            "fixed 200-step tail drifted back outside it under tiny nonzero controls at "
            "the iteration cap. Route finalization now truncates at the first goal-region "
            "entry, connects to the exact goal, and applies the unchanged swept-body "
            "clearance checks to that executable route. The goal criterion, optimizer "
            "budget, objective, and route ranking are unchanged."
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "route_selection_config_sha256": sha256(OUTPUT_CONFIG),
        "route_freezer": {
            "path": str(FREEZER.relative_to(REPO)),
            "sha256": sha256(FREEZER),
        },
    })
    write_new(OUTPUT_PROTOCOL, json.dumps(protocol, indent=2) + "\n")
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
