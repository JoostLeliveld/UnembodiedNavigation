from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import ContractValidationError  # noqa: E402
from reliability.oracle import (  # noqa: E402
    EVALUATION_ONLY_LABEL,
    OracleAvailabilityRegion,
    OracleCameraFeasibilitySample,
    oracle_availability_for_xy,
    oracle_stream_from_evaluation_frames,
    summarize_oracle_redundancy,
)
from reliability.replay import EvaluationFrame  # noqa: E402


def test_oracle_availability_region_labels_evaluation_only_samples() -> None:
    region = OracleAvailabilityRegion(
        camera_id="camera_B",
        xmin=1.0,
        xmax=3.0,
        ymin=-1.0,
        ymax=1.0,
        p_available=0.9,
        reason="hypothetical_camera_B_clear_view",
    )

    inside = oracle_availability_for_xy(
        camera_id="camera_B",
        truth_xy_m=(2.0, 0.0),
        regions=[region],
        timestamp_s=4.0,
    )
    outside = oracle_availability_for_xy(
        camera_id="camera_B",
        truth_xy_m=(4.0, 0.0),
        regions=[region],
        timestamp_s=5.0,
    )

    assert inside.label == EVALUATION_ONLY_LABEL
    assert inside.available is True
    assert inside.p_available == pytest.approx(0.9)
    assert outside.available is False
    assert outside.reason == "outside_oracle_regions"


def test_oracle_stream_and_redundancy_summary_cover_dropouts() -> None:
    frames = (
        EvaluationFrame(timestamp_s=0.0, truth_xy_m=(0.0, 0.0)),
        EvaluationFrame(timestamp_s=1.0, truth_xy_m=(2.0, 0.0)),
        EvaluationFrame(timestamp_s=2.0, truth_xy_m=(2.5, 0.0)),
    )
    regions = (
        OracleAvailabilityRegion(camera_id="camera_B", xmin=1.0, xmax=3.0, ymin=-1.0, ymax=1.0),
    )
    stream = oracle_stream_from_evaluation_frames(
        frames,
        camera_id="camera_B",
        regions=regions,
    )

    summary = summarize_oracle_redundancy(
        primary_available=[True, False, False],
        candidate_samples=stream,
    )

    assert summary.label == EVALUATION_ONLY_LABEL
    assert summary.candidate_camera_id == "camera_B"
    assert summary.candidate_available_frames == 2
    assert summary.primary_dropout_frames == 2
    assert summary.covered_primary_dropout_frames == 2
    assert summary.primary_dropout_coverage_rate == pytest.approx(1.0)


def test_oracle_rejects_bad_shapes_and_labels() -> None:
    with pytest.raises(ContractValidationError, match="bounds"):
        OracleAvailabilityRegion(camera_id="camera_B", xmin=3.0, xmax=1.0, ymin=0.0, ymax=1.0)

    with pytest.raises(ContractValidationError, match="2-element"):
        oracle_availability_for_xy(camera_id="camera_B", truth_xy_m=(1.0,), regions=[])

    with pytest.raises(ContractValidationError, match=EVALUATION_ONLY_LABEL):
        OracleCameraFeasibilitySample(
            camera_id="camera_B",
            timestamp_s=0.0,
            truth_xy_m=(0.0, 0.0),
            available=True,
            p_available=1.0,
            label="operational",
        )
