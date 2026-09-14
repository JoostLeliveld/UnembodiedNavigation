#!/usr/bin/env python3
"""Freeze finite lane-graph expected-belief route selection after V17 profiling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v15.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v17.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v16.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v18.json"


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
    config["study_title"] = (
        "Final commissioned four-arm lane-graph expected-belief route selection V16"
    )
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 18,
        "status": "frozen_before_lane_graph_route_selection_and_four_arm_pilots",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "Profiling showed that the 200-step symbolic camera-network function "
            "serialized to approximately 816 MB and dominated solve time before "
            "L-BFGS-B. No V17 route was accepted. V18 makes the permitted sparse "
            "decision explicit: enumerate the common map-derived lane-graph routes, "
            "evaluate every complete 200-step rollout with the exact NumPy "
            "expected-belief objective, and select the minimum-cost feasible route."
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "purpose": (
            "Select one hash-bound route per task and arm by exact enumeration of "
            "the common map-derived lane-graph candidate set. Each candidate uses "
            "bounded unicycle controls, a 200-second expected-belief rollout, and "
            "one temporally effective information update per second."
        ),
    })
    protocol["route_selection_config_sha256"] = sha256(OUTPUT_CONFIG)
    protocol["common_feasibility"]["optimizer_transition_check"] = (
        "Each enumerated route is converted to bounded stop-turn-go unicycle "
        "controls that land exactly on every map-derived waypoint. The full rollout "
        "uses the planner's swept rectangular-body validation; release admission "
        "independently checks the frozen polyline at 0.04 m."
    )
    protocol["route_candidate_parameterization"] = {
        "decision": "one map-derived lane-graph homotopy route",
        "generation": "condition-neutral and identical across all four arms",
        "rollout_steps": 200,
        "rollout_step_s": 1.0,
        "physical_horizon_s": 200.0,
        "candidate_controls": "bounded stop-turn-go with exact waypoint landing",
        "evaluation": "full NumPy metric expected-belief objective and geometry diagnostics",
        "selection": "minimum total cost among physically valid goal-reaching candidates",
        "camera_opportunities_per_rollout_step": 1,
        "runtime_correction_rate_hz": 5,
    }
    protocol.pop("sparse_global_parameterization", None)
    protocol["selector_sha256"] = sha256(REPO / protocol["selector_path"])
    protocol["route_probe"]["sha256"] = sha256(REPO / protocol["route_probe"]["path"])
    protocol["route_freezer"]["sha256"] = sha256(
        REPO / protocol["route_freezer"]["path"]
    )
    protocol["planner_source_tree"]["sha256"] = tree_sha256(
        protocol["planner_source_tree"]["roots"]
    )
    write_new(OUTPUT_PROTOCOL, json.dumps(protocol, indent=2) + "\n")
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
