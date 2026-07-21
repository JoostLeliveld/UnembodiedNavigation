"""Evaluation-only oracle tools for second-camera feasibility checks.

This module deliberately consumes truth trajectories or hand-labeled regions.
It must not be used by planner-facing runtime code or reliability-model feature
loaders. Its purpose is to decide whether a hypothetical second camera is worth
implementing before building a real detector/calibration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from reliability.contracts import ContractValidationError
from reliability.replay import EvaluationFrame


EVALUATION_ONLY_LABEL = "evaluation_only_oracle"


@dataclass(frozen=True)
class OracleAvailabilityRegion:
    """Hand-labeled evaluation region for hypothetical camera availability."""

    camera_id: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    p_available: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        xmin = _finite(self.xmin, "xmin")
        xmax = _finite(self.xmax, "xmax")
        ymin = _finite(self.ymin, "ymin")
        ymax = _finite(self.ymax, "ymax")
        if xmax < xmin or ymax < ymin:
            raise ContractValidationError("OracleAvailabilityRegion bounds are inverted")
        object.__setattr__(self, "xmin", xmin)
        object.__setattr__(self, "xmax", xmax)
        object.__setattr__(self, "ymin", ymin)
        object.__setattr__(self, "ymax", ymax)
        object.__setattr__(self, "p_available", _probability(self.p_available, "p_available"))

    def contains(self, xy_m: Sequence[float]) -> bool:
        x, y = _pair(xy_m, "xy_m")
        return bool(self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax)


@dataclass(frozen=True)
class OracleCameraFeasibilitySample:
    """Per-timestamp evaluation-only availability label."""

    camera_id: str
    timestamp_s: float
    truth_xy_m: tuple[float, float]
    available: bool
    p_available: float
    reason: str = ""
    label: str = EVALUATION_ONLY_LABEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "truth_xy_m", _pair(self.truth_xy_m, "truth_xy_m"))
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "p_available", _probability(self.p_available, "p_available"))
        if self.label != EVALUATION_ONLY_LABEL:
            raise ContractValidationError(f"Oracle samples must be labeled {EVALUATION_ONLY_LABEL!r}")


@dataclass(frozen=True)
class OracleRedundancySummary:
    """Evaluation-only summary of whether a candidate camera covers dropouts."""

    candidate_camera_id: str
    total_frames: int
    candidate_available_frames: int
    primary_dropout_frames: int
    covered_primary_dropout_frames: int
    candidate_availability_rate: float
    primary_dropout_coverage_rate: float
    label: str = EVALUATION_ONLY_LABEL
    samples: tuple[OracleCameraFeasibilitySample, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.label != EVALUATION_ONLY_LABEL:
            raise ContractValidationError(f"Oracle summary must be labeled {EVALUATION_ONLY_LABEL!r}")


def oracle_availability_for_xy(
    *,
    camera_id: str,
    truth_xy_m: Sequence[float],
    regions: Sequence[OracleAvailabilityRegion],
    timestamp_s: float = 0.0,
    available_threshold: float = 0.5,
) -> OracleCameraFeasibilitySample:
    """Evaluate a hand-labeled oracle availability map at one truth position."""

    xy = _pair(truth_xy_m, "truth_xy_m")
    for region in regions:
        if region.camera_id != camera_id or not region.contains(xy):
            continue
        p_available = region.p_available
        return OracleCameraFeasibilitySample(
            camera_id=camera_id,
            timestamp_s=timestamp_s,
            truth_xy_m=xy,
            available=p_available >= float(available_threshold),
            p_available=p_available,
            reason=region.reason or "inside_oracle_region",
        )
    return OracleCameraFeasibilitySample(
        camera_id=camera_id,
        timestamp_s=timestamp_s,
        truth_xy_m=xy,
        available=False,
        p_available=0.0,
        reason="outside_oracle_regions",
    )


def oracle_stream_from_evaluation_frames(
    evaluation_frames: Sequence[EvaluationFrame],
    *,
    camera_id: str,
    regions: Sequence[OracleAvailabilityRegion],
    available_threshold: float = 0.5,
) -> tuple[OracleCameraFeasibilitySample, ...]:
    return tuple(
        oracle_availability_for_xy(
            camera_id=camera_id,
            truth_xy_m=frame.truth_xy_m,
            regions=regions,
            timestamp_s=frame.timestamp_s,
            available_threshold=available_threshold,
        )
        for frame in evaluation_frames
    )


def summarize_oracle_redundancy(
    primary_available: Sequence[bool],
    candidate_samples: Sequence[OracleCameraFeasibilitySample],
    *,
    candidate_camera_id: str = "",
) -> OracleRedundancySummary:
    """Measure candidate-camera coverage during primary-camera dropout frames."""

    if len(primary_available) != len(candidate_samples):
        raise ContractValidationError("primary_available and candidate_samples must have equal length")
    total = len(candidate_samples)
    candidate_available = sum(1 for sample in candidate_samples if sample.available)
    primary_dropout = 0
    covered_dropout = 0
    for primary_ok, sample in zip(primary_available, candidate_samples):
        if not bool(primary_ok):
            primary_dropout += 1
            if sample.available:
                covered_dropout += 1
    camera_id = candidate_camera_id or (candidate_samples[0].camera_id if candidate_samples else "")
    return OracleRedundancySummary(
        candidate_camera_id=camera_id,
        total_frames=total,
        candidate_available_frames=candidate_available,
        primary_dropout_frames=primary_dropout,
        covered_primary_dropout_frames=covered_dropout,
        candidate_availability_rate=(candidate_available / total) if total else math.nan,
        primary_dropout_coverage_rate=(covered_dropout / primary_dropout) if primary_dropout else math.nan,
        samples=tuple(candidate_samples),
    )


def _finite(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _pair(value: Sequence[float], field_name: str) -> tuple[float, float]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    return (_finite(value[0], f"{field_name}[0]"), _finite(value[1], f"{field_name}[1]"))


def _probability(value: float, field_name: str) -> float:
    out = _finite(value, field_name)
    if out < 0.0 or out > 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1]")
    return out
