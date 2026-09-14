#!/usr/bin/env python3
"""Freeze distinct physical and driveable runtime clearance semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v10.json"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v11.json"


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
        "schema_version": 11,
        "status": "frozen_after_safety_diagnostic_before_remaining_route_selection",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The repeated pure-pursuit pilot remained physically safe but stopped at "
            "the east corner because the runtime guard applied the 0.10 m physical "
            "obstacle buffer a second time to the abstract driveable-region boundary. "
            "Runtime tracking now retains the 0.10 m buffer against physical collision "
            "geometry and requires non-penetration (positive clearance) against the "
            "driveable union. Global route admission still requires 0.10 m swept-body "
            "clearance against both geometries."
        ),
        "failed_execution_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v8"
        ),
        "runtime_clearance_policy": {
            "physical_collision_body_clearance_m": 0.10,
            "driveable_body_clearance_m": 0.0,
            "driveable_requirement": "nonpenetration",
        },
    })
    roots = protocol["planner_source_tree"]["roots"]
    protocol["planner_source_tree"]["sha256"] = tree_sha256(roots)
    OUTPUT_PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PROTOCOL.relative_to(REPO), hashlib.sha256(OUTPUT_PROTOCOL.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
