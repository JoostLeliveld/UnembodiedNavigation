"""P2 exporter unit tests (Gate 2): firewall, misses, split integrity, manifest."""

from __future__ import annotations

import pytest

from reliability.contracts import LeakageError
from reliability.observation_exporter import (
    PERCEPTION_FORBIDDEN,
    PERCEPTION_WHITELIST,
    ExporterConfig,
    _gt_firewall_audit,
)
from reliability.observation_gates import UsableObservationGateConfig, evaluate_observation_opportunity
from reliability.contracts import _contains_evaluation_key


def _aws_gate() -> UsableObservationGateConfig:
    return UsableObservationGateConfig(
        image_width_px=1280, image_height_px=720,
        check_frame_freshness=False, check_confidence=True, confidence_threshold=0.25,
        check_projection=True, check_association=False, require_track=False,
        check_edge_clip=True, check_localizer_accept=False,
    )


def test_whitelist_has_no_gt_columns():
    assert [c for c in PERCEPTION_WHITELIST if _contains_evaluation_key(c)] == []


def test_forbidden_columns_are_actually_gt():
    # every declared forbidden column must be recognised as evaluation-only
    assert all(_contains_evaluation_key(c) for c in PERCEPTION_FORBIDDEN)


def test_firewall_audit_passes():
    audit = _gt_firewall_audit()
    assert audit["passed"] is True
    assert audit["whitelist_leak_hits"] == []
    assert audit["output_schema_leak_hits"] == []


def test_a_hit_becomes_usable_and_a_miss_becomes_no_detection():
    gate = _aws_gate()
    hit = {
        "timestamp": 1.0, "run_id": "r/C1/s/e", "route_id": "r", "camera_id": "cam",
        "state_x": 2.0, "state_y": 3.0, "state_yaw": 0.0, "state_source": "BELIEF",
        "frame_expected": True, "frame_received": True, "frame_age_ms": None,
        "detection_received": True, "detector_class": "robot", "detector_confidence": 0.81,
        "bbox_xmin": 949.0, "bbox_ymin": 278.0, "bbox_xmax": 971.0, "bbox_ymax": 298.0,
        "selected_pixel_u": 960.0, "selected_pixel_v": 298.0,
        "projection_attempted": True, "projection_valid": True,
        "association_attempted": False, "association_valid": False,
        "tracking_available": False, "tracking_valid": False, "accepted_by_localizer": True,
    }
    miss = dict(hit)
    miss.update(detection_received=False, detector_class=None, detector_confidence=0.0,
                projection_valid=False, accepted_by_localizer=False)
    assert evaluate_observation_opportunity(hit, gate).usable_label == 1
    row = evaluate_observation_opportunity(miss, gate)
    assert row.detection_label == 0 and row.usable_label == 0
    assert row.failure_reason == "NO_DETECTION"


def test_low_confidence_detection_is_a_quality_miss_not_a_detection_miss():
    gate = _aws_gate()
    raw = {
        "timestamp": 1.0, "run_id": "r/C1/s/e", "route_id": "r", "camera_id": "cam",
        "state_x": 2.0, "state_y": 3.0, "state_yaw": 0.0, "state_source": "BELIEF",
        "frame_expected": True, "frame_received": True, "frame_age_ms": None,
        "detection_received": True, "detector_class": "robot", "detector_confidence": 0.10,
        "bbox_xmin": 949.0, "bbox_ymin": 278.0, "bbox_xmax": 971.0, "bbox_ymax": 298.0,
        "selected_pixel_u": 960.0, "selected_pixel_v": 298.0,
        "projection_attempted": True, "projection_valid": True,
        "association_attempted": False, "association_valid": False,
        "tracking_available": False, "tracking_valid": False, "accepted_by_localizer": True,
    }
    row = evaluate_observation_opportunity(raw, gate)
    assert row.detection_label == 1 and row.quality_label == 0
    assert row.failure_reason == "LOW_CONFIDENCE"


def test_gt_field_in_raw_record_is_rejected():
    gate = _aws_gate()
    raw = {
        "timestamp": 1.0, "run_id": "r", "route_id": "r", "camera_id": "cam",
        "state_x": 2.0, "state_y": 3.0, "state_yaw": 0.0, "state_source": "BELIEF",
        "detection_received": True, "detector_confidence": 0.8,
        "projection_valid": True,
        "true_x": 2.01,  # GT leak
    }
    with pytest.raises(LeakageError):
        evaluate_observation_opportunity(raw, gate)


def test_exporter_config_hash_stable_and_sensitive():
    a = ExporterConfig(detection_floor=0.05)
    b = ExporterConfig(detection_floor=0.05)
    c = ExporterConfig(detection_floor=0.10)
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()
