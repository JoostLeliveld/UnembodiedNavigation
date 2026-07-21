from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools" / "record_operational_logs.py"
ATTACH = RECORDER.with_name("attach_evaluation_truth.py")


def _module():
    spec = importlib.util.spec_from_file_location("bigwarehouse_operational_recorder", RECORDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_at(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recorder_uses_the_actual_bigwarehouse_camera_models_and_operational_schema() -> None:
    module = _module()

    assert module.DEFAULT_CAMERA_MODELS == {
        "camera_A": "external_camera",
        "camera_B": "external_camera_b",
        "camera_C": "external_camera_c",
        "camera_D": "external_camera_d",
    }
    assert module.DEFAULT_CAMERA_TOPICS["camera_B"] == "/perception/camera_observation/camera_B"
    assert module.DETECTOR_CAMERA_ORDER == (
        "camera_A",
        "camera_B",
        "camera_C",
        "camera_D",
    )
    assert module.BATCHED_FRAME_POLICY == "strict_new_unique_stamp_latest_only_no_reuse"
    assert "pred_world_x" in module.PERCEPTION_FIELDS
    for field in (
        "recorder_wall_elapsed_s",
        "bbox_bottom_u",
        "mask_bottom_u",
        "selected_pixel_source",
        "yolo_inference_wall_ms",
        "detector_callback_wall_ms",
        "image_receive_stamp_s",
        "inference_finish_stamp_s",
        "frame_age_at_publish_s",
    ):
        assert field in module.PERCEPTION_FIELDS
    assert "ground_truth" not in " ".join(module.PERCEPTION_FIELDS)
    assert "true_x" not in " ".join(module.PERCEPTION_FIELDS)
    assert "--use-sim-time" in module.main.__code__.co_consts
    source = RECORDER.read_text(encoding="utf-8")
    assert "separate-process fallback remains launch-compatible" in source
    assert "reliable_transient_local_keep_last_1" in source
    assert "stream watchdog exceeded" in source


def test_recorder_calibrations_parse_all_fullwarehouse_camera_includes() -> None:
    module = _module()
    world = ROOT / "src" / "sim" / "gazebo_worlds" / "worlds" / "warehouse_full_4cam.world.sdf"

    expected = {
        "external_camera": (-6.0, -10.0, 6.1),
        "external_camera_b": (-6.0, 10.0, 6.1),
        "external_camera_c": (6.0, -10.0, 6.1),
        "external_camera_d": (6.0, 10.0, 6.1),
    }
    for include_name, pos in expected.items():
        camera = module.camera_model_from_world(world, include_name=include_name)
        assert tuple(camera.cam_pos) == pos
        assert camera.pixel_to_world(640.0, 360.0) is not None


def test_recorder_maps_spawn_local_odom_back_to_the_documented_warehouse_frame() -> None:
    module = _module()
    recorder = object.__new__(module.OperationalLogRecorder)
    recorder.odom_origin_x_m = -1.8
    recorder.odom_origin_y_m = -5.4
    recorder.odom_origin_yaw_rad = 1.5708
    recorder._origin_cos = __import__("math").cos(recorder.odom_origin_yaw_rad)
    recorder._origin_sin = __import__("math").sin(recorder.odom_origin_yaw_rad)

    world_x, world_y = recorder._odom_to_warehouse(11.6, 0.0)

    assert abs(world_x - (-1.8)) < 1e-4
    assert abs(world_y - 6.2) < 1e-4


def test_recorder_stream_watchdog_starts_only_after_contract_arming() -> None:
    module = _module()
    recorder = object.__new__(module.OperationalLogRecorder)
    recorder.detector_contract_armed = False
    recorder.detector_contract_received_wall_s = None
    recorder.max_stream_wall_age_s = 3.0
    recorder.last_camera_observation_wall_s = {
        camera: None for camera in module.DETECTOR_CAMERA_ORDER
    }
    recorder.last_odom_observation_wall_s = None

    assert recorder.stream_watchdog_failures(now_wall_s=100.0) == []

    recorder.detector_contract_armed = True
    recorder.detector_contract_received_wall_s = 10.0
    recorder.last_camera_observation_wall_s = {
        "camera_A": 12.0,
        "camera_B": 12.0,
        "camera_C": 12.0,
        "camera_D": 12.0,
    }
    recorder.last_odom_observation_wall_s = 12.0
    assert recorder.stream_watchdog_failures(now_wall_s=15.0) == []

    failures = recorder.stream_watchdog_failures(now_wall_s=15.001)
    assert len(failures) == 5
    assert any("camera_A stream watchdog exceeded" in failure for failure in failures)
    assert any("odometry stream watchdog exceeded" in failure for failure in failures)


def test_truth_attachment_requires_completed_matching_predeclared_sources(
    tmp_path: Path,
) -> None:
    module = _module_at(ATTACH, "bigwarehouse_attach_contract")
    run = tmp_path / "run"
    raw = run / "raw"
    evaluation = run / "evaluation_only"
    raw.mkdir(parents=True)
    evaluation.mkdir()
    camera_paths = []
    for camera in ("camera_A", "camera_B", "camera_C", "camera_D"):
        path = raw / f"{camera}_perception.csv"
        path.write_text("diag_stamp,detected\n1.0,0\n", encoding="utf-8")
        camera_paths.append(path)
    truth_csv = evaluation / "ground_truth.csv"
    truth_csv.write_text("stamp,gt_x,gt_y,gt_yaw\n1.0,0,0,0\n", encoding="utf-8")
    common = {
        "status": "completed",
        "run_id": "run-1",
        "plan_row_id": "row-1",
        "seed": 7,
        "evidence_role": "qualification",
        "route_completion_manifest": {"sha256": "a" * 64},
    }
    operational_path = raw / "operational_recording_manifest.json"
    operational_path.write_text(
        json.dumps(
            {
                **common,
                "contains_ground_truth": False,
                "detector_runtime": {"model_sha256": "b" * 64},
                "projection_calibration": {"sha256": "c" * 64},
            }
        ),
        encoding="utf-8",
    )
    truth_manifest_path = evaluation / "evaluation_truth_manifest.json"
    truth_manifest_path.write_text(
        json.dumps(
            {
                **common,
                "contains_ground_truth": True,
                "evaluation_only": True,
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        "raw/operational_recording_manifest.json": _sha256(operational_path),
        "evaluation_only/evaluation_truth_manifest.json": _sha256(truth_manifest_path),
        "evaluation_only/ground_truth.csv": _sha256(truth_csv),
        **{f"raw/{path.name}": _sha256(path) for path in camera_paths},
    }
    (run / "completion_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "row_tuple": {"phase": "D1_mapping_heldout", "route": "heldout"},
                "artifacts_sha256": artifacts,
            }
        ),
        encoding="utf-8",
    )

    contract = module._source_contract(
        raw_dir=raw, truth_csv=truth_csv, evidence_role="qualification"
    )
    assert contract["evidence_role"] == "qualification"
    assert contract["projection_calibration_sha256"] == "c" * 64

    (raw / "experiment.csv.part").write_text("partial", encoding="utf-8")
    with pytest.raises(SystemExit, match="partial or failed"):
        module._source_contract(
            raw_dir=raw, truth_csv=truth_csv, evidence_role="qualification"
        )
