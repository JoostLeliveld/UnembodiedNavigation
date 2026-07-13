"""Camera-overlap validation utilities for extension calibration gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Sequence

from reliability.contracts import ContractValidationError
from reliability.fusion import MapObservation
from reliability.replay import ReplayFrame


@dataclass(frozen=True)
class CameraOverlapPair:
    camera_a_id: str
    camera_b_id: str
    timestamp_a_s: float
    timestamp_b_s: float
    xy_a_m: tuple[float, float]
    xy_b_m: tuple[float, float]
    disagreement_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_a_id": self.camera_a_id,
            "camera_b_id": self.camera_b_id,
            "timestamp_a_s": self.timestamp_a_s,
            "timestamp_b_s": self.timestamp_b_s,
            "xy_a_m": list(self.xy_a_m),
            "xy_b_m": list(self.xy_b_m),
            "disagreement_m": self.disagreement_m,
        }


@dataclass(frozen=True)
class CameraOverlapValidationSummary:
    camera_a_id: str
    camera_b_id: str
    pair_count: int
    mean_disagreement_m: float
    max_disagreement_m: float
    rms_disagreement_m: float
    max_allowed_disagreement_m: float
    pass_gate: bool
    pairs: tuple[CameraOverlapPair, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_a_id": self.camera_a_id,
            "camera_b_id": self.camera_b_id,
            "pair_count": self.pair_count,
            "mean_disagreement_m": _none_nan(self.mean_disagreement_m),
            "max_disagreement_m": _none_nan(self.max_disagreement_m),
            "rms_disagreement_m": _none_nan(self.rms_disagreement_m),
            "max_allowed_disagreement_m": self.max_allowed_disagreement_m,
            "pass_gate": self.pass_gate,
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


def overlap_pairs_from_frames(
    frames: Sequence[ReplayFrame],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_time_delta_s: float = 0.05,
) -> tuple[CameraOverlapPair, ...]:
    """Build A/B overlap pairs from replay frames or near-synchronous frames."""

    observations: list[MapObservation] = []
    for frame in frames:
        observations.extend(frame.observations)
    return match_overlap_observations(
        observations,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        max_time_delta_s=max_time_delta_s,
    )


def match_overlap_observations(
    observations: Iterable[MapObservation],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_time_delta_s: float = 0.05,
) -> tuple[CameraOverlapPair, ...]:
    """Greedily match near-synchronous map observations from two cameras."""

    max_dt = _nonnegative(max_time_delta_s, "max_time_delta_s")
    obs_a = sorted((obs for obs in observations if obs.camera_id == camera_a_id), key=lambda obs: obs.timestamp_s)
    obs_b = sorted((obs for obs in observations if obs.camera_id == camera_b_id), key=lambda obs: obs.timestamp_s)
    used_b: set[int] = set()
    pairs: list[CameraOverlapPair] = []
    for a in obs_a:
        best_idx = None
        best_dt = math.inf
        for idx, b in enumerate(obs_b):
            if idx in used_b:
                continue
            dt = abs(float(a.timestamp_s) - float(b.timestamp_s))
            if dt <= max_dt and dt < best_dt:
                best_idx = idx
                best_dt = dt
        if best_idx is None:
            continue
        b = obs_b[best_idx]
        used_b.add(best_idx)
        disagreement = math.hypot(a.xy_m[0] - b.xy_m[0], a.xy_m[1] - b.xy_m[1])
        pairs.append(
            CameraOverlapPair(
                camera_a_id=camera_a_id,
                camera_b_id=camera_b_id,
                timestamp_a_s=float(a.timestamp_s),
                timestamp_b_s=float(b.timestamp_s),
                xy_a_m=a.xy_m,
                xy_b_m=b.xy_m,
                disagreement_m=float(disagreement),
            )
        )
    return tuple(pairs)


def validate_camera_overlap(
    frames: Sequence[ReplayFrame],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_time_delta_s: float = 0.05,
    max_allowed_disagreement_m: float = 0.30,
) -> CameraOverlapValidationSummary:
    pairs = overlap_pairs_from_frames(
        frames,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        max_time_delta_s=max_time_delta_s,
    )
    return summarize_overlap_pairs(
        pairs,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        max_allowed_disagreement_m=max_allowed_disagreement_m,
    )


def summarize_overlap_pairs(
    pairs: Sequence[CameraOverlapPair],
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_allowed_disagreement_m: float = 0.30,
) -> CameraOverlapValidationSummary:
    limit = _nonnegative(max_allowed_disagreement_m, "max_allowed_disagreement_m")
    values = [pair.disagreement_m for pair in pairs if math.isfinite(pair.disagreement_m)]
    mean = sum(values) / len(values) if values else math.nan
    max_value = max(values) if values else math.nan
    rms = math.sqrt(sum(value * value for value in values) / len(values)) if values else math.nan
    pass_gate = bool(values) and max_value <= limit
    return CameraOverlapValidationSummary(
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        pair_count=len(values),
        mean_disagreement_m=mean,
        max_disagreement_m=max_value,
        rms_disagreement_m=rms,
        max_allowed_disagreement_m=limit,
        pass_gate=pass_gate,
        pairs=tuple(pairs),
    )


def _nonnegative(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ContractValidationError(f"{field_name} must be finite and non-negative")
    return out


def _none_nan(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else float(value)
