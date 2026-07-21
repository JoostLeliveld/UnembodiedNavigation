from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_SRC = ROOT / "src" / "perception"
if str(PERCEPTION_SRC) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_SRC))

from perception.core.four_camera_runtime_contract import (  # noqa: E402
    EXECUTABLE_SOURCE_KEY,
    FOUR_CAMERA_BATCH_SOURCE_KEY,
    REQUIRED_SOURCE_KEYS,
    RUNTIME_CONTRACT_SOURCE_KEY,
    RuntimeContractError,
    YOLO_SELECTION_SOURCE_KEY,
    build_batched_runtime_contract,
    compare_batched_runtime_contract,
    runtime_contract_json,
    runtime_contract_sha256,
)


RECORDER = (
    ROOT
    / "experiments"
    / "multicamera_commissioning_bigwarehouse"
    / "tools"
    / "record_operational_logs.py"
)


def _recorder_module():
    spec = importlib.util.spec_from_file_location("runtime_contract_recorder", RECORDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_hashes(token: str = "a") -> dict[str, str]:
    return {key: token * 64 for key in REQUIRED_SOURCE_KEYS}


def _contract(**overrides):
    values = {
        "model_path": "/tmp/frozen-model.pt",
        "model_sha256": "b" * 64,
        "imgsz": 640,
        "confidence_threshold": 0.05,
        "iou_threshold": 0.45,
        "use_masks": False,
        "shared_device": "0",
        "cpu_threads": 2,
        "interop_threads": 1,
        "opencv_threads": 1,
        "max_batch_stamp_skew_s": 0.10,
        "max_pending_wall_s": 0.50,
        "source_hashes": _source_hashes(),
    }
    values.update(overrides)
    return build_batched_runtime_contract(**values)


def _rehash(contract: dict) -> dict:
    changed = dict(contract)
    changed["contract_sha256"] = runtime_contract_sha256(changed)
    return changed


def test_contract_is_canonical_self_hashed_and_contains_full_topology() -> None:
    contract = _contract()

    encoded = runtime_contract_json(contract)
    assert json.loads(encoded) == contract
    assert contract["contract_sha256"] == runtime_contract_sha256(contract)
    assert contract["model_instances"] == 1
    assert contract["batch_size"] == 4
    assert contract["camera_order"] == ["camera_A", "camera_B", "camera_C", "camera_D"]
    assert contract["source_hashes"][EXECUTABLE_SOURCE_KEY] == "a" * 64
    assert contract["executable_source_sha256"] == contract["source_hashes"][
        EXECUTABLE_SOURCE_KEY
    ]
    assert set(contract["source_hashes"]) == {
        EXECUTABLE_SOURCE_KEY,
        FOUR_CAMERA_BATCH_SOURCE_KEY,
        RUNTIME_CONTRACT_SOURCE_KEY,
        YOLO_SELECTION_SOURCE_KEY,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_sha256", "c" * 64),
        ("imgsz", 800),
        ("shared_device", "cpu"),
        ("max_pending_wall_s", 0.75),
    ],
)
def test_observed_contract_rejects_any_model_or_runtime_difference(
    field: str, value: object
) -> None:
    expected = _contract()
    observed = _rehash({**expected, field: value})

    with pytest.raises(RuntimeContractError, match=field):
        compare_batched_runtime_contract(observed, expected)


def test_recorder_config_validation_requires_the_same_frozen_cli_contract(
    tmp_path: Path,
) -> None:
    recorder = _recorder_module()
    expected = _contract(model_path=tmp_path / "model.pt")
    config_path = tmp_path / "detector.yaml"
    runtime = {
        "image_sizes": [640, 800, 960],
        "model": expected["model_path"],
        "confidence_threshold": 0.05,
        "iou_threshold": 0.45,
        "detector_mode": "batched_four_camera",
        "runtime_executable": "batched_four_camera_yolo_node",
        "launch_switch": "yolo_batched_four_camera",
        "model_format": "native_ultralytics",
        "model_instances": 1,
        "batch_size": 4,
        "camera_order": ["camera_A", "camera_B", "camera_C", "camera_D"],
        "device": "0",
        "cpu_threads": 2,
        "cpu_interop_threads": 1,
        "opencv_threads": 1,
        "max_batch_stamp_skew_s": 0.10,
        "max_pending_wall_s": 0.50,
        "frame_policy": "strict_new_unique_stamp_latest_only_no_reuse",
        "inference_timing_semantics": "full_batch_wall_ms_repeated_per_camera_result",
        "fault_policy": "fatal_process_exit_and_launch_shutdown_no_synthetic_miss",
        "synchronization_miss_policy": "no_output_detected_by_liveness_gate",
        "masks": False,
    }
    config_path.write_text(
        yaml.safe_dump({"runtime_pilot": runtime}, sort_keys=True), encoding="utf-8"
    )

    assert recorder._validate_batched_cli_against_runtime_config(
        config_path=config_path, expected_contract=expected
    ) == runtime
    runtime["cpu_threads"] = 1
    config_path.write_text(
        yaml.safe_dump({"runtime_pilot": runtime}, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="cpu_threads"):
        recorder._validate_batched_cli_against_runtime_config(
            config_path=config_path, expected_contract=expected
        )


def test_recorder_rejects_unlocked_multi_size_config_for_evidence_roles(
    tmp_path: Path,
) -> None:
    recorder = _recorder_module()
    expected = _contract(model_path=tmp_path / "model.pt")
    config_path = tmp_path / "detector.yaml"
    runtime = {
        "evidence_selection": {"permitted": False, "selected_image_size": None},
        "image_sizes": [640, 800],
        "model": expected["model_path"],
        "confidence_threshold": 0.05,
        "iou_threshold": 0.45,
        "detector_mode": "batched_four_camera",
        "runtime_executable": "batched_four_camera_yolo_node",
        "launch_switch": "yolo_batched_four_camera",
        "model_format": "native_ultralytics",
        "model_instances": 1,
        "batch_size": 4,
        "camera_order": ["camera_A", "camera_B", "camera_C", "camera_D"],
        "device": "0",
        "cpu_threads": 2,
        "cpu_interop_threads": 1,
        "opencv_threads": 1,
        "max_batch_stamp_skew_s": 0.10,
        "max_pending_wall_s": 0.50,
        "frame_policy": "strict_new_unique_stamp_latest_only_no_reuse",
        "inference_timing_semantics": "full_batch_wall_ms_repeated_per_camera_result",
        "fault_policy": "fatal_process_exit_and_launch_shutdown_no_synthetic_miss",
        "synchronization_miss_policy": "no_output_detected_by_liveness_gate",
        "masks": False,
    }
    config_path.write_text(
        yaml.safe_dump({"runtime_pilot": runtime}, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="evidence_selection"):
        recorder._validate_batched_cli_against_runtime_config(
            config_path=config_path,
            expected_contract=expected,
            evidence_role="qualification",
        )

    runtime["evidence_selection"] = {"permitted": True, "selected_image_size": 640}
    runtime["image_sizes"] = [640]
    config_path.write_text(
        yaml.safe_dump(
            {"lock_status": "locked_before_evidence", "runtime_pilot": runtime},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert recorder._validate_batched_cli_against_runtime_config(
        config_path=config_path,
        expected_contract=expected,
        evidence_role="qualification",
    ) == runtime


def test_analysis_freeze_contains_every_source_that_the_live_contract_attests() -> None:
    recorder = _recorder_module()
    analysis = (
        ROOT
        / "experiments"
        / "multicamera_commissioning_bigwarehouse"
        / "config"
        / "paper_analysis_plan.yaml"
    )

    hashes = recorder._frozen_detector_source_hashes(
        analysis_path=analysis,
        method_freeze=recorder._method_freeze(analysis),
    )

    assert set(hashes) == set(REQUIRED_SOURCE_KEYS)
