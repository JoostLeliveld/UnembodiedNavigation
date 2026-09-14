#!/usr/bin/env python3
"""Freeze the candidate-audit serialization repair before accepted routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v18.json"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v19.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError(f"immutable release already exists: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 19,
        "status": "frozen_after_candidate_audit_serialization_repair",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The first V18 C00 evaluation completed but its result record was not "
            "written because per-candidate NumPy arrays were not converted to JSON. "
            "No route manifest was accepted. V19 changes only audit serialization "
            "and binds the corrected selector source."
        ),
    })
    protocol["selector_sha256"] = sha256(REPO / protocol["selector_path"])
    protocol["route_probe"]["sha256"] = sha256(REPO / protocol["route_probe"]["path"])
    write_new(OUTPUT_PROTOCOL, protocol)
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
