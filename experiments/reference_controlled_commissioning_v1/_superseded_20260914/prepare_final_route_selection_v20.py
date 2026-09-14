#!/usr/bin/env python3
"""Freeze neutral FF/FB speed behavior before final pilots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v22.json"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v23.json"


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
        "schema_version": 23,
        "status": "frozen_after_neutral_ff_fb_speed_repair",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The first moving V22 C00 commissioning run was stopped without an "
            "accepted outcome after its mean active command was measured at 0.287 "
            "m/s. The FF/FB arrival cap was treating every 0.2 m densified route "
            "sample as the mission endpoint. V23 preserves corner preview and "
            "feedback but applies arrival braking only on the final route segment, "
            "allowing the declared 1.0 m/s speed on aligned intermediate segments."
        ),
        "failed_execution_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v12_fffb"
        ),
    })
    protocol["local_controller_contract"].update({
        "aligned_intermediate_speed_mps": 1.0,
        "arrival_braking": "final route segment only",
        "corner_braking": "geometric look-ahead shared across arms",
    })
    protocol["planner_source_tree"]["sha256"] = tree_sha256(
        protocol["planner_source_tree"]["roots"]
    )
    OUTPUT_PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
