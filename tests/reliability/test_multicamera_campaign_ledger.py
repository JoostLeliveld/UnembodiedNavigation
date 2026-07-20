from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
TOOL = (
    ROOT
    / "experiments"
    / "multicamera_commissioning_bigwarehouse"
    / "tools"
    / "campaign_ledger.py"
)
FINALIZER = TOOL.with_name("finalize_campaign_run.py")
STUDY = (
    ROOT
    / "experiments"
    / "multicamera_commissioning_bigwarehouse"
    / "config"
    / "study.yaml"
)
PROTOCOL = (
    ROOT
    / "experiments"
    / "multicamera_commissioning_bigwarehouse"
    / "config"
    / "paper_protocol.yaml"
)


def _module(name: str = "multicamera_campaign_ledger"):
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finalizer_module(name: str = "multicamera_campaign_finalizer"):
    spec = importlib.util.spec_from_file_location(name, FINALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _single_row_inputs(tmp_path: Path) -> dict[str, Path | list[Path]]:
    study = tmp_path / "study.yaml"
    protocol = tmp_path / "protocol.yaml"
    model = tmp_path / "model.pt"
    calibration = tmp_path / "calibration.json"
    runtime = tmp_path / "runtime.yaml"
    analysis_study = tmp_path / "analysis_study"
    analysis = analysis_study / "config" / "analysis.yaml"
    method_source = analysis_study / "fixture.py"
    _write_yaml(
        study,
        {
            "study_id": "fixture_study",
            "collection": {
                "routes": [
                    {
                        "name": "fixture_route",
                        "start": {"x": 0.0, "y": 0.0},
                        "goal": {"x": 1.0, "y": 0.0},
                    }
                ]
            },
        },
    )
    _write_yaml(
        protocol,
        {
            "protocol_id": "fixture_protocol",
            "randomization": {"seed_values": [0]},
            "route_disjoint_mapping": {
                "train_routes": ["fixture_route"],
                "heldout_routes": [],
                "repeats_per_route": 1,
                "lateral_offsets_m": [0.0],
                "speeds_mps": [0.1],
            },
            "overlap_qualification": {
                "repeats_per_route": 1,
                "lateral_offsets_m": [0.0],
                "speeds_mps": [0.1],
                "edges": [],
            },
        },
    )
    model.write_bytes(b"frozen detector fixture\n")
    calibration.write_text('{"calibration": "fixture"}\n', encoding="utf-8")
    _write_yaml(runtime, {"fresh_age_s": 0.15})
    method_source.parent.mkdir(parents=True)
    analysis.parent.mkdir(parents=True)
    method_source.write_text("# frozen fixture method\n", encoding="utf-8")
    _write_yaml(
        analysis,
        {
            "analysis_plan_id": "fixture_analysis",
            "method_freeze": {"source_files": ["fixture.py"]},
        },
    )
    return {
        "study_path": study,
        "protocol_path": protocol,
        "model_path": model,
        "calibration_path": calibration,
        "config_paths": [analysis, runtime],
    }


def _write_csv(path: Path, *, rows: int = 1, stamp_column: str = "stamp") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if stamp_column == "diag_stamp":
            writer.writerow(
                [
                    "diag_stamp",
                    "recorder_wall_elapsed_s",
                    "frame_age_at_publish_s",
                    "value",
                ]
            )
            for index in range(rows):
                stamp = 0.0 if rows <= 1 else index / (rows - 1)
                writer.writerow([stamp, stamp, 0.10, "ok"])
            return
        writer.writerow([stamp_column, "value"])
        for index in range(rows):
            writer.writerow([index, "ok"])


def _populate_valid_run(
    module,
    campaign_root: Path,
    ledger: dict,
    *,
    publish_completion: bool = True,
    attempt_id: str = "attempt_001",
) -> Path:
    row = ledger["rows"][0]
    attempt = next(item for item in row["attempts"] if item["attempt_id"] == attempt_id)
    run_dir = campaign_root / attempt["run_dir"]
    raw = run_dir / "raw"
    evaluation = run_dir / "evaluation_only"
    raw.mkdir(parents=True)
    evaluation.mkdir()
    exact = row["row_tuple"]
    frozen_hashes = module.input_sha256(ledger["provenance"])
    study_hash = frozen_hashes["study"]
    route_manifest_path = raw / "route_manifest.json"
    route_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "bigwarehouse_route_manifest_v2",
                "manifest_phase": "pre_run",
                "route": exact["route"],
                "lateral_offset_m": float(exact["lateral_offset_m"]),
                "speed_mps": float(exact["speed_mps"]),
                "study_config_sha256": study_hash,
                "contains_ground_truth": False,
                # This is written by the route driver immediately before the
                # route can receive its first control odometry sample.
                "created_wall_time_s": 2_000_000_000.0,
                "completion_artifact": str(raw / "route_completion.json"),
                "completion_status": "pending_route_success",
            }
        ),
        encoding="utf-8",
    )
    route_completion_path = raw / "route_completion.json"
    route_completion = {
        "schema_version": "bigwarehouse_route_completion_v1",
        "status": "completed",
        "route_complete": True,
        "end_reason": "route_complete",
        "completion_reason": "route_complete",
        "contains_ground_truth": False,
        "route_identity": {
            "route": exact["route"],
            "study_config_sha256": study_hash,
            "route_manifest": str(route_manifest_path),
            "route_manifest_sha256": module._sha256(route_manifest_path),
            "lateral_offset_m": float(exact["lateral_offset_m"]),
            "speed_mps": float(exact["speed_mps"]),
        },
        "waypoint_count": 2,
        "waypoints_reached": 2,
        "route_started_sim_time_s": 0.0,
        "route_completed_sim_time_s": 1.0,
        "terminal_stop_publish_count": 5,
        "safety_limits": {
            "terminal_stop_publish_count": 5,
            "odom_freshness_timeout_s": 0.5,
        },
    }
    route_completion_path.write_text(json.dumps(route_completion), encoding="utf-8")
    route_completion_hash = module._sha256(route_completion_path)
    model_path = next(
        item["path"] for item in ledger["provenance"] if item["label"] == "model"
    )
    detector_contract = {
        "schema_version": "four_camera_detector_runtime_contract.v1",
        "runtime_mode": "batched_four_camera",
        "executable": "batched_four_camera_yolo_node",
        "model_format": "native_ultralytics",
        "model_path": model_path,
        "model_sha256": frozen_hashes["model"],
        "imgsz": 640,
        "confidence_threshold": 0.05,
        "iou_threshold": 0.45,
        "use_masks": False,
        "model_instances": 1,
        "batch_size": 4,
        "camera_order": list(module.CAMERAS),
        "shared_device": "0",
        "cpu_threads": 2,
        "interop_threads": 1,
        "opencv_threads": 1,
        "max_batch_stamp_skew_s": 0.10,
        "max_pending_wall_s": 0.50,
        "frame_policy": "strict_new_unique_stamp_latest_only_no_reuse",
        "inference_timing_semantics": "full_batch_wall_ms_repeated_per_camera_result",
        "fault_policy": "fatal_process_exit_and_launch_shutdown_no_synthetic_miss",
        "synchronization_miss_policy": "no_output_detected_by_liveness_gate",
        "executable_source_sha256": "a" * 64,
        "source_hashes": {
            "perception.nodes.batched_four_camera_yolo_node": "a" * 64,
            "perception.core.four_camera_batch": "b" * 64,
            "perception.core.four_camera_runtime_contract": "c" * 64,
            "perception.core.yolo_selection": "d" * 64,
        },
    }
    detector_contract["contract_sha256"] = module._canonical_sha256(detector_contract)
    provenance_by_label = {item["label"]: item for item in ledger["provenance"]}
    frozen_configs = [
        {"path": item["path"], "sha256": item["sha256"]}
        for item in ledger["provenance"]
        if item["label"].startswith("config:")
    ]
    analysis_hash = module.method_freeze_snapshot(ledger)["analysis_plan_sha256"]
    analysis_record = next(
        item
        for item in frozen_configs
        if item["sha256"] == analysis_hash
    )
    recorder_provenance = {
        "study_config": {
            "path": provenance_by_label["study"]["path"],
            "sha256": frozen_hashes["study"],
        },
        "protocol": {
            "path": provenance_by_label["protocol"]["path"],
            "sha256": frozen_hashes["protocol"],
        },
        "analysis_plan": dict(analysis_record),
        "detector_model": {
            "path": provenance_by_label["model"]["path"],
            "sha256": frozen_hashes["model"],
        },
        "projection_calibration": {
            "path": provenance_by_label["calibration"]["path"],
            "sha256": frozen_hashes["calibration"],
        },
    }
    readiness_provenance = {
        "study": dict(recorder_provenance["study_config"]),
        "protocol": dict(recorder_provenance["protocol"]),
        "analysis_plan": dict(recorder_provenance["analysis_plan"]),
        "model": dict(recorder_provenance["detector_model"]),
        "calibration": dict(recorder_provenance["projection_calibration"]),
    }
    method_freeze = module.method_freeze_snapshot(ledger)
    method_freeze_sha256 = module.method_freeze_sha256(ledger)
    transport_environment = {
        "ROS_LOCALHOST_ONLY": "1",
        "IGN_IP": "127.0.0.1",
        "GZ_IP": "127.0.0.1",
        "IGN_PARTITION": attempt["ign_partition"],
        "ROS_DOMAIN_ID": str(attempt["ros_domain_id"]),
    }
    campaign_contract = {
        "ledger_path": str(campaign_root / module.DEFAULT_LEDGER_NAME),
        "plan_sha256": ledger["plan_sha256"],
        "row_id": row["row_id"],
        "attempt_id": attempt_id,
        "row_tuple": row["row_tuple"],
        "expected_evidence_role": row["expected_evidence_role"],
        "expected_analysis_split": row["expected_analysis_split"],
        "input_sha256": frozen_hashes,
        "method_freeze": method_freeze,
        "method_freeze_sha256": method_freeze_sha256,
        "transport_environment": transport_environment,
    }
    readiness_path = run_dir / module.RUNTIME_READINESS_ARTIFACT
    readiness_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "enforce",
                "pass": True,
                "evidence_eligible": True,
                "age_enforced": True,
                "would_pass_age_enforcement": True,
                "timed_out": False,
                "interrupted": False,
                "failures": [],
                "created_utc": "2026-01-01T00:00:00Z",
                "run_id": "fixture_run",
                "plan_row_id": row["row_id"],
                "attempt_id": attempt_id,
                "analysis_split": row["expected_analysis_split"],
                "seed": row["row_tuple"]["seed"],
                "evidence_role": row["expected_evidence_role"],
                "campaign_contract": campaign_contract,
                "transport_environment": transport_environment,
                "provenance": readiness_provenance,
                "frozen_configs": frozen_configs,
                "method_freeze": method_freeze,
                "method_freeze_sha256": method_freeze_sha256,
                "thresholds": {
                    "max_frame_age_s": 0.15,
                    "max_stream_wall_age_s": 3.0,
                    "min_fresh_fraction": 1.0,
                },
                "age_gate_by_camera": {camera: True for camera in module.CAMERAS},
                "detector_runtime_contract": {
                    "topic": "/perception/four_camera_detector_runtime_contract",
                    "qos": "reliable_transient_local_keep_last_1",
                    "message_count": 1,
                    "validated": True,
                    "contract_sha256": detector_contract["contract_sha256"],
                    "contract": detector_contract,
                    "error": None,
                },
                "cameras": {
                    camera: {
                        "summary_window_count": 2,
                        "sim_hz": 3.0,
                        "frame_age_s": {
                            "finite_count": 2,
                            "fresh_fraction": 1.0,
                            "threshold_s": 0.15,
                        },
                    }
                    for camera in module.CAMERAS
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (raw / "operational_recording_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "started_utc": "2026-01-01T00:01:00Z",
                "stop_reason": "route_completion_manifest",
                "contains_ground_truth": False,
                "run_id": "fixture_run",
                "plan_row_id": row["row_id"],
                "attempt_id": attempt_id,
                "seed": 0,
                "evidence_role": row["expected_evidence_role"],
                "analysis_split": row["expected_analysis_split"],
                "provenance": recorder_provenance,
                "frozen_configs": frozen_configs,
                "method_freeze": method_freeze,
                "method_freeze_sha256": method_freeze_sha256,
                "campaign_contract": campaign_contract,
                "transport_environment": campaign_contract["transport_environment"],
                "detector_runtime": {
                    "model_path": model_path,
                    "model_sha256": frozen_hashes["model"],
                    "topology": {
                        "runtime_mode": "batched_four_camera",
                        "executable": "batched_four_camera_yolo_node",
                        "launch_switch_yolo_batched_four_camera": True,
                        "model_format": "native_ultralytics",
                        "model_instances": 1,
                        "batch_size": 4,
                        "camera_order": list(module.CAMERAS),
                        "shared_device": "0",
                        "cpu_threads": 2,
                        "interop_threads": 1,
                        "opencv_threads": 1,
                        "max_batch_stamp_skew_s": 0.10,
                        "max_pending_wall_s": 0.50,
                        "frame_policy": "strict_new_unique_stamp_latest_only_no_reuse",
                        "inference_timing_semantics": "full_batch_wall_ms_repeated_per_camera_result",
                        "fault_policy": "fatal_process_exit_and_launch_shutdown_no_synthetic_miss",
                        "synchronization_miss_policy": "no_output_detected_by_liveness_gate",
                    },
                    "observed_contract": detector_contract,
                    "observed_contract_sha256": detector_contract["contract_sha256"],
                    "observed_contract_requirement": {
                        "state": "armed",
                        "expected_contract": detector_contract,
                        "expected_contract_sha256": detector_contract["contract_sha256"],
                    },
                },
                "projection_calibration": {"sha256": frozen_hashes["calibration"]},
                "route_completion_manifest": {"sha256": route_completion_hash},
                "fresh_age_s": 0.15,
                "stream_liveness_watchdog": {"max_stream_wall_age_s": 3.0},
                "minimum_stream_rows": {"per_camera": 1, "odometry": 1},
                "camera_rows": {camera: 3 for camera in module.CAMERAS},
                "odometry_rows": 1,
                "camera_stamp_ranges_s": {
                    camera: {"first": 0.0, "last": 1.0}
                    for camera in module.CAMERAS
                },
                "odometry_stamp_range_s": {"first": 0.0, "last": 1.0},
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "evaluation_truth_manifest.json").write_text(
        json.dumps(
            {
                "contains_ground_truth": True,
                "evaluation_only": True,
                "status": "completed",
                "stop_reason": "route_completion_manifest",
                "run_id": "fixture_run",
                "plan_row_id": row["row_id"],
                "attempt_id": attempt_id,
                "seed": 0,
                "evidence_role": row["expected_evidence_role"],
                "analysis_split": row["expected_analysis_split"],
                "provenance": recorder_provenance,
                "frozen_configs": frozen_configs,
                "campaign_contract": campaign_contract,
                "transport_environment": campaign_contract["transport_environment"],
                "route_completion_manifest": {"sha256": route_completion_hash},
                "sample_count": 1,
                "minimum_samples": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(raw / "experiment.csv")
    for camera in module.CAMERAS:
        _write_csv(raw / f"{camera}_perception.csv", rows=3, stamp_column="diag_stamp")
    _write_csv(evaluation / "ground_truth.csv")
    artifact_hashes = {
        relative: module._sha256(run_dir / relative)
        for relative in module.REQUIRED_ARTIFACTS
    }
    runtime_readiness, route_camera_health, readiness_errors = (
        module.runtime_readiness_completion_evidence(run_dir, ledger, row)
    )
    assert not readiness_errors
    assert runtime_readiness is not None
    assert route_camera_health is not None
    completion = {
        "schema_version": 1,
        "row_id": row["row_id"],
        "attempt_id": attempt_id,
        "row_tuple": row["row_tuple"],
        "input_sha256": module.input_sha256(ledger["provenance"]),
        "method_freeze": method_freeze,
        "method_freeze_sha256": method_freeze_sha256,
        "route_completion": route_completion,
        "runtime_readiness": runtime_readiness,
        "route_camera_health": route_camera_health,
        "run_identity": {
            "run_id": "fixture_run",
            "plan_row_id": row["row_id"],
            "attempt_id": attempt_id,
            "seed": 0,
            "truth_run_id": "fixture_run",
            "truth_plan_row_id": row["row_id"],
            "truth_attempt_id": attempt_id,
            "truth_seed": 0,
        },
        "artifacts_sha256": artifact_hashes,
    }
    if publish_completion:
        (run_dir / "completion_manifest.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return run_dir


def _refresh_completion_evidence(module, run_dir: Path, ledger: dict) -> list[str]:
    """Rehash a deliberate fixture mutation without hiding semantic failures."""

    row = ledger["rows"][0]
    completion_path = run_dir / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    readiness, route_health, errors = module.runtime_readiness_completion_evidence(
        run_dir, ledger, row
    )
    if readiness is not None:
        completion["runtime_readiness"] = readiness
    if route_health is not None:
        completion["route_camera_health"] = route_health
    completion["artifacts_sha256"] = {
        relative: module._sha256(run_dir / relative)
        for relative in module.REQUIRED_ARTIFACTS
        if (run_dir / relative).is_file()
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return errors


def test_frozen_protocol_has_360_unique_stable_collection_rows() -> None:
    module = _module("campaign_ledger_real_plan")
    study = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))

    first = module.build_collection_plan(study, protocol)
    second = module.build_collection_plan(study, protocol)
    first_ids = [module.row_id(row) for row in first]
    second_ids = [module.row_id(row) for row in second]

    assert len(first) == 360
    assert first_ids == second_ids
    assert len(set(first_ids)) == len(first_ids)
    assert {row["seed"] for row in first if row["phase"].startswith("D1_")} == set(
        range(5)
    )
    assert {row["seed"] for row in first if row["phase"].startswith("D2_")} == set(
        range(10)
    )
    assert {row["phase"].split("_")[0] for row in first} == {"D1", "D2"}


def test_recorder_preflight_binds_inputs_row_and_output_directory(tmp_path: Path) -> None:
    module = _module("campaign_recorder_preflight")
    inputs = _single_row_inputs(tmp_path)
    desired = module.build_ledger(**inputs)
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    row = ledger["rows"][0]
    attempt = row["attempts"][0]
    transport_environment = {
        "ROS_LOCALHOST_ONLY": "1",
        "IGN_IP": "127.0.0.1",
        "GZ_IP": "127.0.0.1",
        "IGN_PARTITION": attempt["ign_partition"],
        "ROS_DOMAIN_ID": str(attempt["ros_domain_id"]),
    }
    contract = module.recorder_preflight_contract(
        ledger_path=campaign_root / module.DEFAULT_LEDGER_NAME,
        plan_row_id=row["row_id"],
        attempt_id="attempt_001",
        seed=row["row_tuple"]["seed"],
        evidence_role=row["expected_evidence_role"],
        analysis_split=row["expected_analysis_split"],
        output_dir=campaign_root / row["run_dir"] / "raw",
        artifact_subdir="raw",
        expected_inputs={
            "study": inputs["study_path"],
            "protocol": inputs["protocol_path"],
            "model": inputs["model_path"],
            "calibration": inputs["calibration_path"],
        },
        frozen_config_paths=inputs["config_paths"],
        transport_environment=transport_environment,
    )

    assert contract["row_id"] == row["row_id"]
    assert contract["input_sha256"] == module.input_sha256(ledger["provenance"])
    with pytest.raises(module.LedgerError, match="Recorder output must be"):
        module.recorder_preflight_contract(
            ledger_path=campaign_root / module.DEFAULT_LEDGER_NAME,
            plan_row_id=row["row_id"],
            attempt_id="attempt_001",
            seed=row["row_tuple"]["seed"],
            evidence_role=row["expected_evidence_role"],
            analysis_split=row["expected_analysis_split"],
            output_dir=campaign_root / "wrong" / "raw",
            artifact_subdir="raw",
            expected_inputs={"study": inputs["study_path"]},
            frozen_config_paths=inputs["config_paths"],
            transport_environment=transport_environment,
        )


def test_quarantined_attempt_releases_next_predeclared_slot_without_overwrite(
    tmp_path: Path,
) -> None:
    module = _module("campaign_attempt_retry")
    inputs = _single_row_inputs(tmp_path)
    desired = module.build_ledger(**inputs, attempt_slots=2)
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    row = ledger["rows"][0]
    first_dir = campaign_root / row["attempts"][0]["run_dir"]
    (first_dir / "raw").mkdir(parents=True)
    partial = first_dir / "raw/camera_A_perception.csv.part"
    partial.write_text("partial evidence\n", encoding="utf-8")

    failure = module.quarantine_attempt(
        campaign_root,
        row_id_value=row["row_id"],
        attempt_id="attempt_001",
        reason="detector exited before route completion",
    )
    refreshed = module.refresh_campaign(campaign_root)

    assert failure["preserved_artifacts_sha256"][
        "raw/camera_A_perception.csv.part"
    ] == module._sha256(partial)
    assert refreshed["rows"][0]["status"] == "planned"
    assert refreshed["rows"][0]["next_attempt_id"] == "attempt_002"
    _populate_valid_run(
        module,
        campaign_root,
        refreshed,
        attempt_id="attempt_002",
    )
    completed = module.refresh_campaign(campaign_root)
    assert completed["rows"][0]["status"] == "completed"
    assert completed["rows"][0]["selected_attempt_id"] == "attempt_002"

    partial.write_text("mutated after quarantine\n", encoding="utf-8")
    tampered = module.refresh_campaign(campaign_root)
    assert tampered["rows"][0]["status"] == "failed"
    assert "quarantined attempt artifacts changed after sealing" in tampered["rows"][0][
        "validation_errors"
    ]


def test_duplicate_six_field_tuple_is_rejected() -> None:
    module = _module("campaign_ledger_duplicate_plan")
    study = {"collection": {"routes": [{"name": "same_route"}]}}
    protocol = {
        "randomization": {"seed_values": [0]},
        "route_disjoint_mapping": {
            "train_routes": ["same_route", "same_route"],
            "heldout_routes": [],
            "repeats_per_route": 1,
            "lateral_offsets_m": [0.0],
            "speeds_mps": [0.1],
        },
        "overlap_qualification": {"edges": []},
    }

    with pytest.raises(module.LedgerError, match="Duplicate six-field"):
        module.build_collection_plan(study, protocol)


def test_only_strictly_validated_completion_is_skipped(tmp_path: Path) -> None:
    module = _module("campaign_ledger_valid_completion")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)

    validated = module.refresh_campaign(campaign_root)

    assert validated["summary"] == {
        "planned": 0,
        "in_progress": 0,
        "completed": 1,
        "failed": 0,
        "total": 1,
    }
    assert module.runnable_rows(validated) == []

    # A completed row is re-hashed on every refresh. Post-completion mutation
    # revokes completion instead of being silently skipped.
    with (run_dir / "raw/camera_D_perception.csv").open("a", encoding="utf-8") as handle:
        handle.write("2,tampered\n")
    revalidated = module.refresh_campaign(campaign_root)

    assert revalidated["summary"]["completed"] == 0
    assert revalidated["summary"]["failed"] == 1
    assert len(module.runnable_rows(revalidated)) == 1
    assert "artifact hash mismatch for raw/camera_D_perception.csv" in revalidated["rows"][0][
        "validation_errors"
    ]


def test_zero_camera_rows_or_wrong_exact_tuple_fails_completion(tmp_path: Path) -> None:
    module = _module("campaign_ledger_invalid_completion")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    _write_csv(run_dir / "raw/camera_B_perception.csv", rows=0)
    completion_path = run_dir / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["row_tuple"] = dict(completion["row_tuple"], repeat=2)
    completion["artifacts_sha256"]["raw/camera_B_perception.csv"] = module._sha256(
        run_dir / "raw/camera_B_perception.csv"
    )
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    validated = module.refresh_campaign(campaign_root)
    errors = validated["rows"][0]["validation_errors"]

    assert validated["rows"][0]["status"] == "failed"
    assert "completion manifest does not contain the exact six-field tuple" in errors
    assert "camera_B operational CSV has zero data rows" in errors


def test_duplicate_completion_manifest_is_detected(tmp_path: Path) -> None:
    module = _module("campaign_ledger_duplicate_completion")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    duplicate = campaign_root / "unexpected" / "completion_manifest.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes((run_dir / "completion_manifest.json").read_bytes())

    validated = module.refresh_campaign(campaign_root)

    assert validated["rows"][0]["status"] == "failed"
    assert any(
        "duplicate completion manifests" in error
        for error in validated["rows"][0]["validation_errors"]
    )


def test_existing_run_dir_is_in_progress_and_plan_never_adopts_or_overwrites(tmp_path: Path) -> None:
    module = _module("campaign_ledger_no_overwrite")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    expected_dir = campaign_root / desired["rows"][0]["run_dir"]
    expected_dir.mkdir(parents=True)

    with pytest.raises(module.LedgerError, match="refusing to adopt/overwrite"):
        module.initialize_campaign(campaign_root, desired)

    # In a ledger-backed campaign, the same directory is a resumable active
    # attempt, never an implicitly successful row.
    expected_dir.rmdir()
    ledger = module.initialize_campaign(campaign_root, desired)
    (campaign_root / ledger["rows"][0]["run_dir"]).mkdir(parents=True)
    refreshed = module.refresh_campaign(campaign_root)

    assert refreshed["rows"][0]["status"] == "in_progress"
    assert refreshed["summary"]["completed"] == 0


def test_frozen_input_drift_blocks_refresh_without_rewriting_ledger(tmp_path: Path) -> None:
    module = _module("campaign_ledger_frozen_drift")
    inputs = _single_row_inputs(tmp_path)
    desired = module.build_ledger(**inputs)
    campaign_root = tmp_path / "campaign"
    module.initialize_campaign(campaign_root, desired)
    ledger_path = campaign_root / module.DEFAULT_LEDGER_NAME
    before = ledger_path.read_bytes()
    model_path = inputs["model_path"]
    assert isinstance(model_path, Path)
    model_path.write_bytes(b"changed detector bytes\n")

    with pytest.raises(module.LedgerError, match="Frozen input changed"):
        module.refresh_campaign(campaign_root)

    assert ledger_path.read_bytes() == before


def test_frozen_method_source_drift_blocks_refresh_without_rewriting_ledger(
    tmp_path: Path,
) -> None:
    module = _module("campaign_ledger_method_freeze_drift")
    inputs = _single_row_inputs(tmp_path)
    desired = module.build_ledger(**inputs)
    campaign_root = tmp_path / "campaign"
    module.initialize_campaign(campaign_root, desired)
    ledger_path = campaign_root / module.DEFAULT_LEDGER_NAME
    before = ledger_path.read_bytes()
    analysis_path = inputs["config_paths"][0]
    assert isinstance(analysis_path, Path)
    method_source = analysis_path.parent.parent / "fixture.py"
    method_source.write_text("# changed after plan\n", encoding="utf-8")

    with pytest.raises(module.LedgerError, match="Frozen method source changed"):
        module.refresh_campaign(campaign_root)

    assert ledger_path.read_bytes() == before


def test_readiness_campaign_contract_is_bound_to_the_selected_attempt(
    tmp_path: Path,
) -> None:
    module = _module("campaign_ledger_readiness_campaign_contract")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    readiness_path = run_dir / module.RUNTIME_READINESS_ARTIFACT
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["campaign_contract"]["attempt_id"] = "attempt_002"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    evidence_errors = _refresh_completion_evidence(module, run_dir, ledger)
    refreshed = module.refresh_campaign(campaign_root)

    assert (
        "runtime readiness campaign contract attempt_id does not match the campaign"
        in evidence_errors
    )
    assert (
        "runtime readiness campaign contract attempt_id does not match the campaign"
        in refreshed["rows"][0]["validation_errors"]
    )


def test_campaign_preflight_and_validation_allow_only_one_active_attempt(
    tmp_path: Path,
) -> None:
    module = _module("campaign_ledger_single_active_attempt")
    inputs = _single_row_inputs(tmp_path)
    study_path = inputs["study_path"]
    protocol_path = inputs["protocol_path"]
    assert isinstance(study_path, Path)
    assert isinstance(protocol_path, Path)
    study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    study["collection"]["routes"].append(
        {
            "name": "fixture_route_two",
            "start": {"x": 0.0, "y": 1.0},
            "goal": {"x": 1.0, "y": 1.0},
        }
    )
    _write_yaml(study_path, study)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["route_disjoint_mapping"]["train_routes"].append("fixture_route_two")
    _write_yaml(protocol_path, protocol)
    desired = module.build_ledger(**inputs)
    assert len(desired["rows"]) == 2
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)

    def preflight(row: dict) -> dict:
        attempt = row["attempts"][0]
        transport = {
            "ROS_LOCALHOST_ONLY": "1",
            "IGN_IP": "127.0.0.1",
            "GZ_IP": "127.0.0.1",
            "IGN_PARTITION": attempt["ign_partition"],
            "ROS_DOMAIN_ID": str(attempt["ros_domain_id"]),
        }
        return module.recorder_preflight_contract(
            ledger_path=campaign_root / module.DEFAULT_LEDGER_NAME,
            plan_row_id=row["row_id"],
            attempt_id=attempt["attempt_id"],
            seed=row["row_tuple"]["seed"],
            evidence_role=row["expected_evidence_role"],
            analysis_split=row["expected_analysis_split"],
            output_dir=campaign_root / attempt["run_dir"] / "raw",
            artifact_subdir="raw",
            expected_inputs={
                "study": inputs["study_path"],
                "protocol": inputs["protocol_path"],
                "model": inputs["model_path"],
                "calibration": inputs["calibration_path"],
            },
            frozen_config_paths=inputs["config_paths"],
            transport_environment=transport,
        )

    first, second = ledger["rows"]
    preflight(first)
    with pytest.raises(module.LedgerError, match="another campaign attempt is active"):
        preflight(second)

    # Validate the same invariant even if an external process created the
    # second declared directory without going through preflight.
    (campaign_root / second["attempts"][0]["run_dir"]).mkdir(parents=True)
    refreshed = module.refresh_campaign(campaign_root)
    for row in refreshed["rows"]:
        assert row["status"] == "failed"
        assert any(
            "more than one campaign attempt is active" in error
            for error in row["validation_errors"]
        )


def test_driver_completion_is_required_hashed_and_semantically_anchored(tmp_path: Path) -> None:
    module = _module("campaign_ledger_route_completion_anchor")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    route_completion_path = run_dir / "raw/route_completion.json"
    route_completion = json.loads(route_completion_path.read_text(encoding="utf-8"))
    route_completion["route_identity"]["route"] = "different_route"
    route_completion_path.write_text(json.dumps(route_completion), encoding="utf-8")

    # Update both published hashes and the embedded copy to prove that the
    # semantic route check is independent of byte-integrity checks.
    completion_path = run_dir / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["route_completion"] = route_completion
    completion["artifacts_sha256"]["raw/route_completion.json"] = module._sha256(
        route_completion_path
    )
    route_completion_hash = module._sha256(route_completion_path)
    for manifest_path in (
        run_dir / "raw/operational_recording_manifest.json",
        run_dir / "evaluation_only/evaluation_truth_manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["route_completion_manifest"]["sha256"] = route_completion_hash
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        relative = str(manifest_path.relative_to(run_dir))
        completion["artifacts_sha256"][relative] = module._sha256(manifest_path)
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    refreshed = module.refresh_campaign(campaign_root)

    assert refreshed["rows"][0]["status"] == "failed"
    assert "route completion route does not match campaign tuple" in refreshed["rows"][0][
        "validation_errors"
    ]


def test_completed_run_rejects_camera_that_died_before_route_end(tmp_path: Path) -> None:
    module = _module("campaign_ledger_stream_coverage")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    operational_path = run_dir / "raw/operational_recording_manifest.json"
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    operational["camera_stamp_ranges_s"]["camera_C"]["last"] = 0.10
    operational_path.write_text(json.dumps(operational), encoding="utf-8")
    completion_path = run_dir / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts_sha256"]["raw/operational_recording_manifest.json"] = (
        module._sha256(operational_path)
    )
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    refreshed = module.refresh_campaign(campaign_root)

    assert refreshed["rows"][0]["status"] == "failed"
    assert "camera_C did not cover the completed route interval" in refreshed["rows"][0][
        "validation_errors"
    ]


def test_completed_run_requires_a_passing_enforced_readiness_artifact(tmp_path: Path) -> None:
    module = _module("campaign_ledger_enforced_readiness")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    readiness_path = run_dir / module.RUNTIME_READINESS_ARTIFACT
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["mode"] = "pilot"
    readiness["evidence_eligible"] = False
    readiness["age_enforced"] = False
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    evidence_errors = _refresh_completion_evidence(module, run_dir, ledger)
    refreshed = module.refresh_campaign(campaign_root)

    assert "runtime readiness must use enforce mode" in evidence_errors
    assert refreshed["rows"][0]["status"] == "failed"
    assert "runtime readiness must use enforce mode" in refreshed["rows"][0][
        "validation_errors"
    ]


def test_completed_run_requires_readiness_before_the_pre_run_route_manifest(tmp_path: Path) -> None:
    module = _module("campaign_ledger_readiness_timestamp")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    readiness_path = run_dir / module.RUNTIME_READINESS_ARTIFACT
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["created_utc"] = "2034-01-01T00:00:00Z"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    evidence_errors = _refresh_completion_evidence(module, run_dir, ledger)
    refreshed = module.refresh_campaign(campaign_root)

    assert "runtime readiness timestamp is not before the pre-run route manifest" in evidence_errors
    assert "runtime readiness timestamp is not before the pre-run route manifest" in refreshed[
        "rows"
    ][0]["validation_errors"]


def test_completed_run_requires_route_wide_camera_freshness_and_bounded_gap(
    tmp_path: Path,
) -> None:
    module = _module("campaign_ledger_route_camera_health")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    camera_path = run_dir / "raw/camera_A_perception.csv"
    with camera_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    rows[1]["diag_stamp"] = "0.10"
    rows[1]["frame_age_at_publish_s"] = "0.20"
    rows[1]["recorder_wall_elapsed_s"] = "4.00"
    rows[2]["recorder_wall_elapsed_s"] = "8.00"
    with camera_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    evidence_errors = _refresh_completion_evidence(module, run_dir, ledger)
    refreshed = module.refresh_campaign(campaign_root)

    assert "camera_A route-wide frame freshness is below 1" in evidence_errors
    assert "camera_A route-wide recorder wall gap exceeds 3s" in evidence_errors
    assert "camera_A route-wide diagnostic-stamp gap exceeds 0.666667s" in evidence_errors
    assert refreshed["rows"][0]["status"] == "failed"
    assert "camera_A route-wide frame freshness is below 1" in refreshed["rows"][0][
        "validation_errors"
    ]
    assert "camera_A route-wide recorder wall gap exceeds 3s" in refreshed["rows"][0][
        "validation_errors"
    ]
    assert (
        "camera_A route-wide diagnostic-stamp gap exceeds 0.666667s"
        in refreshed["rows"][0]["validation_errors"]
    )


def test_runtime_readiness_binding_detects_detector_runtime_mutation(tmp_path: Path) -> None:
    module = _module("campaign_ledger_readiness_runtime_binding")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    operational_path = run_dir / "raw/operational_recording_manifest.json"
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    # Image size is not otherwise part of the legacy topology-shape check. The
    # immutable readiness binding must still catch a changed detector runtime.
    operational["detector_runtime"]["imgsz"] = 1234
    operational_path.write_text(json.dumps(operational), encoding="utf-8")
    completion_path = run_dir / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts_sha256"]["raw/operational_recording_manifest.json"] = (
        module._sha256(operational_path)
    )
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    refreshed = module.refresh_campaign(campaign_root)

    assert refreshed["rows"][0]["status"] == "failed"
    assert (
        "completion manifest runtime readiness binding does not match immutable artifacts"
        in refreshed["rows"][0]["validation_errors"]
    )


def test_runtime_contract_self_hash_and_observed_copy_are_revalidated(tmp_path: Path) -> None:
    module = _module("campaign_ledger_runtime_contract_tamper")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(module, campaign_root, ledger)
    operational_path = run_dir / "raw/operational_recording_manifest.json"
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    # Simulate a post-record edit that leaves the recorded contract digest in
    # place. Hashing the manifest again must not make this eligible.
    operational["detector_runtime"]["observed_contract"]["imgsz"] = 800
    operational_path.write_text(json.dumps(operational), encoding="utf-8")

    evidence_errors = _refresh_completion_evidence(module, run_dir, ledger)
    refreshed = module.refresh_campaign(campaign_root)

    assert "operational observed detector runtime contract differs from readiness" in evidence_errors
    assert "operational observed detector runtime contract self-hash is invalid" in evidence_errors
    assert refreshed["rows"][0]["status"] == "failed"
    assert "operational observed detector runtime contract differs from readiness" in refreshed[
        "rows"
    ][0]["validation_errors"]


def test_finalizer_publishes_only_a_complete_cross_checked_run(tmp_path: Path) -> None:
    module = _module("campaign_ledger_finalizer_valid")
    finalizer = _finalizer_module("campaign_run_finalizer_valid")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(
        module,
        campaign_root,
        ledger,
        publish_completion=False,
    )
    row_id = ledger["rows"][0]["row_id"]

    payload = finalizer.finalize_campaign_run(campaign_root, row_id)
    refreshed = module.refresh_campaign(campaign_root)

    assert payload["row_id"] == row_id
    assert payload["route_completion"] == json.loads(
        (run_dir / "raw/route_completion.json").read_text(encoding="utf-8")
    )
    assert "raw/route_completion.json" in payload["artifacts_sha256"]
    assert refreshed["rows"][0]["status"] == "completed"
    with pytest.raises(finalizer.FinalizationError, match="already exists"):
        finalizer.finalize_campaign_run(campaign_root, row_id)


def test_finalizer_rejects_recorder_identity_mismatch_without_success_artifact(
    tmp_path: Path,
) -> None:
    module = _module("campaign_ledger_finalizer_mismatch")
    finalizer = _finalizer_module("campaign_run_finalizer_mismatch")
    desired = module.build_ledger(**_single_row_inputs(tmp_path))
    campaign_root = tmp_path / "campaign"
    ledger = module.initialize_campaign(campaign_root, desired)
    run_dir = _populate_valid_run(
        module,
        campaign_root,
        ledger,
        publish_completion=False,
    )
    truth_manifest_path = run_dir / "evaluation_only/evaluation_truth_manifest.json"
    truth_manifest = json.loads(truth_manifest_path.read_text(encoding="utf-8"))
    truth_manifest["run_id"] = "wrong_run"
    truth_manifest_path.write_text(json.dumps(truth_manifest), encoding="utf-8")

    with pytest.raises(finalizer.FinalizationError, match="run_id values disagree"):
        finalizer.finalize_campaign_run(campaign_root, ledger["rows"][0]["row_id"])

    assert not (run_dir / "completion_manifest.json").exists()
