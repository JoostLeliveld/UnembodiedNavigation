#!/usr/bin/env python3
"""Freeze the zero-tolerance endpoint numerical-comparison repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v21.json"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v22.json"


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
        "schema_version": 22,
        "status": "frozen_after_preselected_endpoint_roundoff_repair",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The first V21 C00 Gazebo launch performed no motion because the "
            "runtime compared a zero-tolerance route endpoint without a numerical "
            "floor: 10.600000000000001 was rejected against 10.6. The on-disk "
            "hash and geometry validation had passed. V22 adds only a 1e-9 m "
            "floating-point comparison floor and binds the repaired source tree."
        ),
        "failed_execution_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v11_fffb"
        ),
    })
    protocol["planner_source_tree"]["sha256"] = tree_sha256(
        protocol["planner_source_tree"]["roots"]
    )
    OUTPUT_PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
