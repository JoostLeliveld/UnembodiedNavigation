#!/usr/bin/env python3
"""Freeze removal of a duplicate legacy seed-evaluation pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v20.json"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v21.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT_PROTOCOL.exists():
        raise RuntimeError(f"immutable release already exists: {OUTPUT_PROTOCOL}")
    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 21,
        "status": "frozen_after_duplicate_probe_evaluation_removal",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The first V20 timing attempt redundantly evaluated every route once "
            "with the legacy CasADi initializer and again with the declared exact "
            "finite-selector controls. It was stopped before producing a result. "
            "V21 removes only that obsolete first pass; all eight declared "
            "expected-belief and geometry evaluations remain and are recorded."
        ),
    })
    protocol["selector_sha256"] = sha256(REPO / protocol["selector_path"])
    protocol["route_probe"]["sha256"] = sha256(REPO / protocol["route_probe"]["path"])
    OUTPUT_PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
