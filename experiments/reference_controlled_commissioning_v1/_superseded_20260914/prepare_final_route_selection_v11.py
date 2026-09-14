#!/usr/bin/env python3
"""Freeze the temporally conservative five-Hz sparse global-planning contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v11.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v13.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v12.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v14.json"
FIELD_ROOT = Path(
    "logs/studies/reference_controlled_commissioning_v1/"
    "final_planner_fields_20260913_v3_5hz_temporal"
)
SENSOR_MODEL = Path(
    "logs/studies/reference_controlled_commissioning_v1/"
    "visibility_patch_covariance_20260913_v7_5hz_temporal/"
    "commissioned_visibility_runtime_model.json"
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
    config["study_title"] = "Final commissioned four-arm sparse route selection V12"
    for arm in ("C00", "C01", "C10", "C11"):
        config["conditions"][arm]["camera_network_artifact_path"] = fields["artifacts"][arm]["path"]
    config["manager_visibility_sensor_model_path"] = str(SENSOR_MODEL)
    config["manager_visibility_sensor_model_expected_sha256"] = sha256(REPO / SENSOR_MODEL)
    # Sparse controls, unchanged 200 s horizon and 5 Hz information cadence.
    config["global_horizon"] = 100
    config["global_dt"] = 2.0
    config["camera_network_updates_per_step"] = 10
    config["manager_decision_rate_hz"] = 5.0
    config["state_reanchor_m"] = 0.0
    config["thesis_execution_contract"] = "final_1mps_5hz_m4_v1"
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 14,
        "status": "frozen_before_sparse_route_selection_and_four_arm_pilots",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The V13 preflight correctly refused the one-Hz-certified R4 bundle at "
            "5 Hz. V14 uses the same M3+M4 weights and R4 structure with a conservative "
            "fivefold per-frame temporal covariance inflation. The global control grid "
            "is sparsified from 200 one-second controls to 100 two-second controls; "
            "ten 5 Hz opportunities are evaluated per step, preserving the 200 s "
            "physical horizon and requested observation cadence."
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "purpose": (
            "Select one hash-bound route per task and arm using the launch-resolved "
            "CasADi/L-BFGS-B expected-belief planner with a sparse two-second control "
            "grid, 1.0 m/s bounds, and 5 Hz camera-network opportunities."
        ),
        "execution_design": (
            "Hash-bind each selected route, then execute matched seeds in Gazebo at "
            "1.0 m/s with the same pure-pursuit controller, M3+M4 correction, temporally "
            "inflated R4 covariance, 5 Hz detector batches, and 5 Hz correction attempts."
        ),
        "sparse_global_parameterization": {
            "horizon_steps": 100,
            "step_s": 2.0,
            "physical_horizon_s": 200.0,
            "camera_opportunities_per_step": 10,
            "camera_opportunities_per_second": 5,
            "maximum_obstacle_cost_sample_spacing_m": 0.20,
            "independent_frozen_route_sweep_spacing_m": 0.04,
        },
        "temporal_covariance_policy": {
            "runtime_model_path": str(SENSOR_MODEL),
            "runtime_model_sha256": sha256(REPO / SENSOR_MODEL),
            "per_frame_covariance_multiplier": 5.0,
            "information_interpretation": (
                "five successive frames contribute no more information than one "
                "uninflated commissioned frame"
            ),
        },
    })
    protocol["route_selection_config_sha256"] = sha256(OUTPUT_CONFIG)
    protocol["runtime_estimator_contract"].update({
        "sensor_model_path": str(SENSOR_MODEL),
        "sensor_model_sha256": sha256(REPO / SENSOR_MODEL),
        "detector_rate_hz": 5,
        "fusion_rate_hz": 5,
        "camera_network_updates_per_global_step": 10,
        "metric_reanchor_enabled": False,
        "temporal_covariance_inflation": 5.0,
    })
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
