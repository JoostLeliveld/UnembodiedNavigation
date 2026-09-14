#!/usr/bin/env python3
"""Freeze the no-reanchor, five-Hz planning and runtime contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v9.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v11.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v11.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v13.json"
FIELD_ROOT = Path(
    "logs/studies/reference_controlled_commissioning_v1/"
    "final_planner_fields_20260913_v2_5hz"
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
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["study_title"] = "Final commissioned four-arm route selection V11"
    config["study_comparison"] = (
        "Four planner arms cross constant versus commissioned availability and "
        "covariance forecasts; execution shares 1.0 m/s motion, 5 Hz detection, "
        "5 Hz fusion, M3+M4 correction, and R4 runtime covariance."
    )
    for arm in ("C00", "C01", "C10", "C11"):
        config["conditions"][arm]["camera_network_artifact_path"] = str(
            FIELD_ROOT / f"{arm.lower()}_planner_field.npz"
        )
    config["manager_decision_rate_hz"] = 5.0
    config["camera_network_updates_per_step"] = 5
    config["thesis_execution_contract"] = "final_1mps_5hz_m4_v1"
    config["state_reanchor_m"] = 0.0
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 13,
        "status": "frozen_before_five_hz_route_selection_and_arm_pilots",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "A source-time audit of the C00 V9 pilot proved 5.000 Hz camera capture "
            "and complete detector batches but only 1.000 Hz manager decisions and "
            "estimator correction attempts. This protocol makes the requested 5 Hz "
            "belief-correction cadence explicit in both planning and execution. "
            "Metric re-anchoring remains disabled: belief divergence and collision are "
            "retained outcomes rather than repaired from a large innovation."
        ),
        "purpose": (
            "Select one hash-bound route per task and arm with the launch-resolved "
            "CasADi/L-BFGS-B global planner at 1.0 m/s and five camera-network "
            "opportunities per 1 s global step, then execute at 5 Hz detection/fusion."
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "execution_design": (
            "Hash-bind each selected route. Matched seeds execute at 1.0 m/s with "
            "5 Hz detection and 5 Hz camera-manager correction attempts, without "
            "rerunning the global optimizer. All arms use the same pure-pursuit "
            "controller and the same M3+M4/R4 runtime estimator."
        ),
        "failed_execution_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v9"
        ),
        "runtime_rate_audit": {
            "path": (
                "logs/studies/reference_controlled_commissioning_v1/"
                "final_navigation_pilot_t1_c00_v9/runtime_rate_audit_v1.json"
            ),
            "camera_and_detector_rate_hz": 5.0,
            "manager_and_assimilation_attempt_rate_hz": 1.0,
        },
        "temporal_dependence_limit": {
            "five_hz_residual_lag1_median_absolute": 0.5505110860407658,
            "stride_five_residual_lag1_median_absolute": 0.10080566984856379,
            "interpretation": (
                "The five within-second opportunities are the requested nominal "
                "conditional-independence planning approximation; the observed "
                "temporal residual dependence must be disclosed when interpreting "
                "belief calibration."
            ),
        },
        "belief_recovery_policy": {
            "metric_reanchor_enabled": False,
            "state_reanchor_m": 0.0,
            "ordinary_nis_gate": 9.21,
        },
    })
    protocol["route_selection_config_sha256"] = sha256(OUTPUT_CONFIG)
    protocol["runtime_estimator_contract"].update({
        "detector_rate_hz": 5,
        "fusion_rate_hz": 5,
        "camera_network_updates_per_global_step": 5,
        "metric_reanchor_enabled": False,
    })
    manifest = json.loads((REPO / FIELD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for arm in protocol["arm_order"]:
        protocol["planning_fields"][arm]["path"] = manifest["artifacts"][arm]["path"]
        protocol["planning_fields"][arm]["sha256"] = manifest["artifacts"][arm]["sha256"]
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
