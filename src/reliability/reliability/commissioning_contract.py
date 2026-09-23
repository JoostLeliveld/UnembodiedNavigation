"""Fail-closed contract for a commissioned camera sensor-model bundle.

This module records the interfaces that remain invariant while correction and covariance
families are compared.  It deliberately does not choose tomorrow's winning model.
"""

from __future__ import annotations

from dataclasses import dataclass

from reliability.contracts import ContractValidationError


Q_SENSOR_TARGET = "detector_hit_and_sensor_gate_pass"
Q_SENSOR_ROLE = "future_observation_forecast_only"
R_RESIDUAL_SOURCE = "whole_drive_out_of_fold_residuals_of_paired_correction"
ROUTE_FEASIBILITY_POLICY = "same_geometric_and_rollout_feasibility_for_all_conditions"
PRIMARY_FILTER_POLICY = "one_robot_filter_consuming_each_camera_frame_once"


@dataclass(frozen=True)
class CorrectionModelSpec:
    model_id: str
    family: str
    output: str = "corrected_ground_plane_position"

    def __post_init__(self) -> None:
        if not self.model_id or not self.family:
            raise ContractValidationError("correction model needs non-empty id and family")
        if self.output != "corrected_ground_plane_position":
            raise ContractValidationError("correction output must be a ground-plane position")


@dataclass(frozen=True)
class CovarianceModelSpec:
    model_id: str
    correction_model_id: str
    family: str
    residual_source: str = R_RESIDUAL_SOURCE
    planner_query_inputs: tuple[str, ...] = ("camera_id", "position", "heading")

    def __post_init__(self) -> None:
        if not self.model_id or not self.correction_model_id or not self.family:
            raise ContractValidationError("covariance model needs non-empty identifiers")
        if self.residual_source != R_RESIDUAL_SOURCE:
            raise ContractValidationError(
                "R must be learned from whole-drive out-of-fold residuals of its paired correction"
            )
        forbidden = {"rgb", "image", "bbox", "detector_confidence", "innovation", "nis"}
        leaked = forbidden.intersection(self.planner_query_inputs)
        if leaked:
            raise ContractValidationError(
                "planner covariance query cannot require unavailable future evidence: "
                + ", ".join(sorted(leaked))
            )


@dataclass(frozen=True)
class AvailabilityModelSpec:
    model_id: str
    target: str = Q_SENSOR_TARGET
    role: str = Q_SENSOR_ROLE
    includes_nis: bool = False
    query_inputs: tuple[str, ...] = ("camera_id", "position")

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ContractValidationError("availability model needs a non-empty id")
        if self.target != Q_SENSOR_TARGET:
            raise ContractValidationError("legacy availability must predict detector hit and sensor-gate pass")
        if self.role != Q_SENSOR_ROLE:
            raise ContractValidationError("legacy availability is a forecast, never a runtime acceptance gate")
        if self.includes_nis:
            raise ContractValidationError("NIS is downstream of legacy availability and cannot enter its target")
        if self.query_inputs != ("camera_id", "position"):
            raise ContractValidationError(
                "legacy availability inputs are exactly camera_id and 2-D position; heading is pooled"
            )


@dataclass(frozen=True)
class CommissionedSensorBundle:
    """Metadata joining one correction candidate to its own covariance model."""

    bundle_id: str
    sensor_gate_id: str
    sensor_gate_belief_independent: bool
    correction: CorrectionModelSpec
    covariance: CovarianceModelSpec
    availability: AvailabilityModelSpec
    route_feasibility_policy: str = ROUTE_FEASIBILITY_POLICY
    filter_policy: str = PRIMARY_FILTER_POLICY

    def __post_init__(self) -> None:
        if not self.bundle_id or not self.sensor_gate_id:
            raise ContractValidationError("bundle and sensor gate need non-empty identifiers")
        if not self.sensor_gate_belief_independent:
            raise ContractValidationError("sensor gate must be deterministic and belief-independent")
        if self.covariance.correction_model_id != self.correction.model_id:
            raise ContractValidationError(
                "correction and R are one paired model: covariance correction_model_id mismatch"
            )
        if self.route_feasibility_policy != ROUTE_FEASIBILITY_POLICY:
            raise ContractValidationError(
                "availability may change expected information, not route feasibility"
            )
        if self.filter_policy != PRIMARY_FILTER_POLICY:
            raise ContractValidationError(
                "each camera frame must enter one robot filter once; naive cascaded filtering is forbidden"
            )
