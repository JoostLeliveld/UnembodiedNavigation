from __future__ import annotations

import pytest

from reliability.commissioning_contract import (
    AvailabilityModelSpec,
    CommissionedSensorBundle,
    CorrectionModelSpec,
    CovarianceModelSpec,
)
from reliability.contracts import ContractValidationError


def _bundle(**overrides) -> CommissionedSensorBundle:
    values = {
        "bundle_id": "candidate_raw_box_v1",
        "sensor_gate_id": "commissioning_sensor_gate_v1",
        "sensor_gate_belief_independent": True,
        "correction": CorrectionModelSpec("raw_box_correction_v1", "raw_box_features"),
        "covariance": CovarianceModelSpec(
            "raw_box_R_v1", "raw_box_correction_v1", "spatial_cholesky"
        ),
        "availability": AvailabilityModelSpec("q_sensor_v1"),
    }
    values.update(overrides)
    return CommissionedSensorBundle(**values)


def test_valid_bundle_keeps_correction_and_R_paired():
    assert _bundle().covariance.correction_model_id == "raw_box_correction_v1"


def test_reusing_R_from_another_correction_fails_closed():
    covariance = CovarianceModelSpec(
        "geometry_R_v1", "geometry_correction_v1", "spatial_cholesky"
    )
    with pytest.raises(ContractValidationError, match="paired model"):
        _bundle(covariance=covariance)


def test_q_sensor_cannot_include_nis_or_gate_routes():
    with pytest.raises(ContractValidationError, match="NIS"):
        AvailabilityModelSpec("bad_q", includes_nis=True)
    with pytest.raises(ContractValidationError, match="forecast"):
        AvailabilityModelSpec("bad_q", role="runtime_acceptance_gate")


def test_planner_R_query_cannot_require_future_image_evidence():
    with pytest.raises(ContractValidationError, match="future evidence"):
        CovarianceModelSpec(
            "rgb_R", "rgb_correction", "image_conditioned",
            planner_query_inputs=("camera_id", "position", "heading", "rgb"),
        )


def test_availability_never_changes_route_feasibility():
    with pytest.raises(ContractValidationError, match="route feasibility"):
        _bundle(route_feasibility_policy="q_sensor_above_threshold")


def test_naive_double_cascade_is_rejected():
    with pytest.raises(ContractValidationError, match="one robot filter"):
        _bundle(filter_policy="double_cascaded_gaussian_filter")
