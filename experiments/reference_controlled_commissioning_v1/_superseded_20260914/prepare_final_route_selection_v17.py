#!/usr/bin/env python3
"""Freeze bounded diverse lane-graph candidates before final selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v19.json"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v20.json"


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


def main() -> int:
    if OUTPUT_PROTOCOL.exists():
        raise RuntimeError(f"immutable release already exists: {OUTPUT_PROTOCOL}")
    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 20,
        "status": "frozen_with_diverse_bounded_route_candidates",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The initial finite selector exposed only two T1/T2 polylines and "
            "collapsed nominal T4 alternatives onto one straight line. No final "
            "four-arm pilot used those manifests. V20 enumerates start-lane and "
            "goal-lane variants, removes collinear duplicates and immediate "
            "backtracks, preserves cross-aisle diversity, and caps the common set "
            "at eight candidates so each arm/task remains below 120 seconds."
        ),
        "candidate_generation": (
            "Enumerate geometry-valid start-lane, cross-aisle, and goal-lane "
            "combinations; remove duplicate collinear vertices and immediate "
            "backtracks; retain one distinct route per cross-aisle before filling "
            "by geometric length; cap at eight. The set is generated once by the "
            "same deterministic geometry-only rule for every arm."
        ),
    })
    protocol["route_candidate_parameterization"].update({
        "maximum_candidates_per_task": 8,
        "diversity_priority": "one distinct route per cross-aisle before shortest fillers",
        "duplicate_policy": "remove collinear duplicate polylines and immediate backtracks",
        "solve_time_requirement_s": 120,
    })
    protocol["selector_sha256"] = sha256(REPO / protocol["selector_path"])
    protocol["route_probe"]["sha256"] = sha256(REPO / protocol["route_probe"]["path"])
    protocol["planner_source_tree"]["sha256"] = tree_sha256(
        protocol["planner_source_tree"]["roots"]
    )
    OUTPUT_PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
