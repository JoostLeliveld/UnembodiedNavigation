#!/usr/bin/env python3
"""Freeze the controller and 5 Hz assimilation-throughput release contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v23.json"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v24.json"


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
        "schema_version": 24,
        "status": "frozen_after_clean_5hz_release_pilot",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "V23 commissioning exposed three execution-only defects without using "
            "scientific campaign outcomes: dense samples could remain behind the "
            "robot, direct cross-track angular feedback oscillated at 1 m/s, and "
            "the 10 Hz command timer repeatedly swept the complete remaining tape "
            "while holding the correction lock. The repairs advance crossed "
            "intermediate samples, use bounded path-heading guidance with lateral "
            "speed damping, and revalidate only the immediately executable control "
            "interval. Every later interval is still checked before publication. "
            "The V16 release pilot then reached the goal collision-free with exact "
            "journals, 5.0 Hz source/manager batches, and 4.98 Hz estimator attempts."
        ),
        "failed_execution_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v15_fffb"
        ),
        "commissioning_failure_evidence": [
            "logs/studies/reference_controlled_commissioning_v1/final_navigation_pilot_t1_c00_v13_fffb",
            "logs/studies/reference_controlled_commissioning_v1/final_navigation_pilot_t1_c00_v14_fffb",
            "logs/studies/reference_controlled_commissioning_v1/final_navigation_pilot_t1_c00_v15_fffb",
        ],
        "controller_pilot_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v16_fffb"
        ),
    })
    protocol["local_controller_contract"].update({
        "intermediate_waypoint_policy": "advance on arrival or crossed segment plane",
        "path_feedback": "bounded path-heading guidance with continuous cross-track speed damping",
        "command_time_safety_revalidation": (
            "one immediately executable interval per 10 Hz publication; complete "
            "tape checked at installation and every later interval before execution"
        ),
    })
    protocol["runtime_rate_audit"] = {
        "path": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v16_fffb/runtime_rate_audit_v1.json"
        ),
        "camera_and_detector_rate_hz": 5.0,
        "manager_rate_hz": 5.0,
        "estimator_assimilation_attempt_rate_hz": 4.976308015622831,
        "longest_correction_gap_s": 0.78,
    }
    protocol["planner_source_tree"]["sha256"] = tree_sha256(
        protocol["planner_source_tree"]["roots"]
    )
    OUTPUT_PROTOCOL.write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
