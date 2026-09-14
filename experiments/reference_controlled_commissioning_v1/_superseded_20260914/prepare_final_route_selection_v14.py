#!/usr/bin/env python3
"""Freeze one effective planner update per second with five-Hz execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v14.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v16.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v15.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v17.json"
FIELD_ROOT = Path(
    "logs/studies/reference_controlled_commissioning_v1/"
    "final_planner_fields_20260913_v4_effective_1hz"
)


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
    fields = json.loads((REPO / FIELD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["study_title"] = "Final commissioned four-arm effective-rate route selection V15"
    for arm in ("C00", "C01", "C10", "C11"):
        config["conditions"][arm]["camera_network_artifact_path"] = fields["artifacts"][arm]["path"]
    config["camera_network_updates_per_step"] = 1
    config["thesis_execution_contract"] = "final_1mps_5hz_m4_temporal_v1"
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 17,
        "status": "frozen_before_effective_rate_route_selection_and_four_arm_pilots",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The explicit five-opportunity symbolic graph consumed 5.5 GB and treated "
            "temporally correlated frames as independent. It was stopped before any "
            "route result. V17 uses one conservative effective information update per "
            "one-second planner step, matching the fivefold covariance inflation used "
            "by five real runtime corrections. The 5 Hz runtime cadence is unchanged."
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "purpose": (
            "Select one hash-bound route per task and arm using launch-resolved "
            "CasADi/L-BFGS-B expected-belief planning with 100 sparse control blocks, "
            "a 200-step rollout, and one temporally effective update per second."
        ),
        "planning_execution_rate_map": {
            "runtime_detector_rate_hz": 5,
            "runtime_fusion_attempt_rate_hz": 5,
            "runtime_per_frame_covariance_multiplier": 5,
            "planner_effective_update_rate_hz": 1,
            "planner_updates_per_one_second_step": 1,
            "reason": "conservative temporal information thinning",
        },
    })
    protocol["route_selection_config_sha256"] = sha256(OUTPUT_CONFIG)
    protocol["sparse_global_parameterization"]["camera_opportunities_per_rollout_step"] = 1
    protocol["sparse_global_parameterization"]["camera_opportunities_per_second"] = 1
    protocol["sparse_global_parameterization"]["runtime_correction_rate_hz"] = 5
    protocol["runtime_estimator_contract"]["camera_network_updates_per_global_step"] = 1
    for arm in protocol["arm_order"]:
        protocol["planning_fields"][arm]["path"] = fields["artifacts"][arm]["path"]
        protocol["planning_fields"][arm]["sha256"] = fields["artifacts"][arm]["sha256"]
    protocol["selector_sha256"] = sha256(REPO / protocol["selector_path"])
    protocol["launch_resolution"]["sha256"] = sha256(
        REPO / protocol["launch_resolution"]["path"]
    )
    protocol["route_probe"]["sha256"] = sha256(REPO / protocol["route_probe"]["path"])
    protocol["route_freezer"]["sha256"] = sha256(REPO / protocol["route_freezer"]["path"])
    protocol["planner_source_tree"]["sha256"] = tree_sha256(
        protocol["planner_source_tree"]["roots"]
    )
    write_new(OUTPUT_PROTOCOL, json.dumps(protocol, indent=2) + "\n")
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
