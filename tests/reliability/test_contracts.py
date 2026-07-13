from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraObservation,
    CameraQuality,
    ContractValidationError,
    EvaluationOnlySample,
    LeakageError,
    OperationalReliabilitySample,
    PlanningCovariance,
    ReliabilityPrediction,
    UpdateCovariance,
)


def _operational_payload() -> dict:
    return {
        "sample_id": "experiment_00123:000184",
        "timestamp_s": 184.62,
        "run_id": "experiment_00123",
        "task_id": "route_apron_to_a3_mid",
        "seed": 3,
        "image_ref": "logs/experiments/experiment_00123/camera/frame_000184.png",
        "image_hash": "sha256:596c1d2e4f9f",
        "detector_result": {
            "detected": True,
            "raw_score": 0.83,
            "selected_score": 0.83,
            "bbox_bottom_u": 392.0,
            "bbox_bottom_v": 456.0,
            "bbox_area_px2": 2926.0,
            "mask_area_px": 0.0,
            "selected_pixel_source": "bbox_bottom",
            "inference_latency_ms": 18.6,
        },
        "selected_pixel": [392.0, 456.0],
        "projection_valid": True,
        "measurement_age_s": 0.04,
        "measurement_stale": False,
        "odometry": {
            "stamp_s": 184.55,
            "linear_speed_mps": 0.31,
            "yaw_rate_radps": 0.02,
        },
        "state_estimate": {
            "x_m": 1.2,
            "y_m": 1.66,
            "source_x": "yolo_mask_or_bbox_homography",
            "source_y": "yolo_mask_or_bbox_homography",
            "source_yaw": "none",
        },
        "belief": {
            "available": True,
            "stamp_s": 184.58,
            "x_m": 1.22,
            "y_m": 1.68,
            "yaw_rad": 0.04,
            "cov_xy_px2_proxy": 6.25,
        },
        "camera_relative_range_m": 6.82,
        "image_location": {"u_norm": 0.6125, "v_norm": 0.7125},
        "recent_detector_history": [True, True, False, True, True],
        "config_hash": "sha256:1b18a4e0",
        "artifact_hashes": {
            "yolo_model": "sha256:4b80b61a",
            "gp_artifact": "sha256:09dc5229",
        },
        "metadata": {
            "perception_backend": "yolo",
            "projection_backend": "homography",
        },
    }


def _evaluation_payload() -> dict:
    return {
        "sample_id": "experiment_00123:000184",
        "timestamp_s": 184.62,
        "run_id": "experiment_00123",
        "task_id": "route_apron_to_a3_mid",
        "seed": 3,
        "gazebo_ground_truth_pose": {"x_m": 1.18, "y_m": 1.64, "yaw_rad": 0.05},
        "ground_truth_projected_pixel": [389.8, 454.6],
        "ground_truth_localization_error_m": 0.045,
        "clearance_m": 0.44,
        "collision": False,
        "geometry_breach": False,
        "final_task_outcome": "goal_reached",
        "metrics": {
            "final_goal_distance_m": 0.08,
            "minimum_goal_distance_m": 0.08,
        },
    }


def test_operational_sample_json_round_trip() -> None:
    sample = OperationalReliabilitySample.from_dict(_operational_payload())
    restored = OperationalReliabilitySample.from_json(sample.to_json())

    assert restored == sample
    assert restored.selected_pixel == (392.0, 456.0)
    assert restored.recent_detector_history == (True, True, False, True, True)


def test_evaluation_only_sample_json_round_trip() -> None:
    sample = EvaluationOnlySample.from_dict(_evaluation_payload())
    restored = EvaluationOnlySample.from_json(sample.to_json())

    assert restored == sample
    assert restored.ground_truth_projected_pixel == (389.8, 454.6)


def test_optional_evaluation_numbers_serialize_as_json_null() -> None:
    payload = json.loads(EvaluationOnlySample(sample_id="missing-labels").to_json())
    restored = EvaluationOnlySample.from_dict(payload)

    assert payload["ground_truth_localization_error_m"] is None
    assert payload["clearance_m"] is None
    assert restored.sample_id == "missing-labels"


def test_docs_examples_match_contracts() -> None:
    operational_path = ROOT / "docs" / "reliability_contracts" / "example_operational_sample.json"
    evaluation_path = ROOT / "docs" / "reliability_contracts" / "example_evaluation_only_sample.json"

    operational = OperationalReliabilitySample.from_dict(json.loads(operational_path.read_text()))
    evaluation = EvaluationOnlySample.from_dict(json.loads(evaluation_path.read_text()))

    assert operational.sample_id == evaluation.sample_id
    assert operational.run_id == evaluation.run_id


def test_operational_sample_rejects_unknown_evaluation_fields() -> None:
    payload = _operational_payload()
    payload["gt_x"] = 1.18

    with pytest.raises(LeakageError, match="evaluation-only"):
        OperationalReliabilitySample.from_dict(payload)


def test_operational_sample_rejects_nested_evaluation_fields() -> None:
    payload = copy.deepcopy(_operational_payload())
    payload["detector_result"]["localization_error_m"] = 0.04

    with pytest.raises(LeakageError, match="localization_error_m"):
        OperationalReliabilitySample.from_dict(payload)


def test_operational_sample_rejects_non_eval_unknown_fields() -> None:
    payload = _operational_payload()
    payload["surprise_field"] = 1

    with pytest.raises(ContractValidationError, match="unknown fields"):
        OperationalReliabilitySample.from_dict(payload)


def test_prediction_json_round_trip_and_probability_gate() -> None:
    pred = ReliabilityPrediction(
        sample_id="sample",
        model_id="phase0_baseline",
        timestamp_s=1.0,
        p_available=0.7,
        raw_probability=0.65,
        calibrated_probability=0.7,
        conditional_quality={"score": 0.8},
        epistemic_uncertainty=0.1,
        staleness_s=0.2,
    )

    assert ReliabilityPrediction.from_json(pred.to_json()) == pred
    with pytest.raises(ContractValidationError, match="p_available"):
        ReliabilityPrediction(p_available=1.2)


def test_update_covariance_shape_units_and_spd_gate() -> None:
    covariance = UpdateCovariance(
        matrix_px2=((6.25, 0.1), (0.1, 9.0)),
        adapter_id="phase0",
        source="unit_test",
        epistemic_inflation=1.2,
        staleness_inflation=1.1,
    )

    assert UpdateCovariance.from_json(covariance.to_json()) == covariance
    with pytest.raises(ContractValidationError, match="units"):
        UpdateCovariance(units="m^2")
    with pytest.raises(ContractValidationError, match="2x2"):
        UpdateCovariance(matrix_px2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    with pytest.raises(ContractValidationError, match="symmetric"):
        UpdateCovariance(matrix_px2=((1.0, 0.2), (0.1, 1.0)))
    with pytest.raises(ContractValidationError, match="positive definite"):
        UpdateCovariance(matrix_px2=((1.0, 0.0), (0.0, 0.0)))


def test_planning_covariance_shape_probability_and_spd_gate() -> None:
    covariance = PlanningCovariance(
        matrix_px2=((25.0, 0.0), (0.0, 36.0)),
        p_available=0.4,
        conditional_covariance_px2=((6.25, 0.0), (0.0, 6.25)),
        epistemic_score=0.2,
        staleness_score=0.1,
        provenance="phase0",
        timestamp_s=2.0,
    )

    assert PlanningCovariance.from_json(covariance.to_json()) == covariance
    with pytest.raises(ContractValidationError, match="p_available"):
        PlanningCovariance(p_available=-0.1)
    with pytest.raises(ContractValidationError, match="conditional_covariance_px2"):
        PlanningCovariance(conditional_covariance_px2=((1.0, 0.0), (0.0, -1.0)))


def test_camera_observation_accepts_array_like_fields_and_round_trips() -> None:
    np = pytest.importorskip("numpy")

    observation = CameraObservation(
        camera_id="camera_A",
        timestamp_s=184.62,
        pixel_uv=np.asarray([392.0, 456.0]),
        detection_valid=True,
        detector_score=0.83,
        bbox_xyxy=np.asarray([360.0, 410.0, 424.0, 456.0]),
        measurement_age_s=0.04,
        calibration_id="warehouse_aws_external_camera_v1",
        image_frame_id="external_camera",
        conditional_cov_uv=np.asarray([[6.25, 0.0], [0.0, 6.25]]),
        availability_probability=0.74,
        association_probability=0.96,
    )
    restored = CameraObservation.from_json(observation.to_json())

    assert restored == observation
    assert restored.pixel_uv == (392.0, 456.0)
    assert restored.bbox_xyxy == (360.0, 410.0, 424.0, 456.0)
    assert restored.conditional_cov_uv == ((6.25, 0.0), (0.0, 6.25))

    quality = observation.quality(source_model="single_camera_gp")
    assert quality == CameraQuality(
        camera_id="camera_A",
        p_available=0.74,
        conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
        association_confidence=0.96,
        epistemic_score=0.0,
        stale=False,
        source_model="single_camera_gp",
    )


def test_camera_observation_requires_pixel_for_valid_detection() -> None:
    with pytest.raises(ContractValidationError, match="pixel_uv"):
        CameraObservation(detection_valid=True, detector_score=0.5)


def test_camera_contracts_reject_bad_probabilities_covariances_and_eval_fields() -> None:
    with pytest.raises(ContractValidationError, match="detector_score"):
        CameraObservation(detector_score=1.1)
    with pytest.raises(ContractValidationError, match="conditional_cov_uv"):
        CameraObservation(conditional_cov_uv=((1.0, 0.0), (0.0, -1.0)))
    with pytest.raises(ContractValidationError, match="association_confidence"):
        CameraQuality(association_confidence=-0.1)

    obs_payload = CameraObservation(
        camera_id="camera_A",
        pixel_uv=(1.0, 2.0),
        detection_valid=True,
    ).to_dict()
    obs_payload["oracle_visible"] = True
    with pytest.raises(LeakageError, match="oracle_visible"):
        CameraObservation.from_dict(obs_payload)

    quality_payload = CameraQuality(camera_id="camera_A").to_dict()
    quality_payload["ground_truth_pose"] = {"x_m": 1.0}
    with pytest.raises(LeakageError, match="ground_truth_pose"):
        CameraQuality.from_dict(quality_payload)
