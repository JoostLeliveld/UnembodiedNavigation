"""Simple camera selection and fusion policies for Phase 1/2 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence

from reliability.contracts import CameraObservation, CameraQuality, ContractValidationError


@dataclass(frozen=True)
class MapObservation:
    camera_id: str
    timestamp_s: float
    xy_m: tuple[float, float]
    covariance_m2: tuple[tuple[float, float], tuple[float, float]]
    quality: CameraQuality
    source: str = ""

    def __post_init__(self) -> None:
        xy = _pair(self.xy_m, "xy_m")
        cov = _matrix_2x2(self.covariance_m2, "covariance_m2")
        _validate_spd(cov, "covariance_m2")
        object.__setattr__(self, "xy_m", xy)
        object.__setattr__(self, "covariance_m2", cov)
        if self.quality.camera_id and self.quality.camera_id != self.camera_id:
            raise ContractValidationError("quality.camera_id must match MapObservation.camera_id")


@dataclass(frozen=True)
class SequentialFusionResult:
    mean_xy: tuple[float, float]
    covariance_m2: tuple[tuple[float, float], tuple[float, float]]
    accepted_camera_ids: tuple[str, ...]
    rejected_camera_ids: tuple[str, ...]
    nis_by_camera: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FixedZone:
    camera_id: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, xy: Sequence[float]) -> bool:
        x, y = _pair(xy, "xy")
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


def usable_observations(observations: Iterable[CameraObservation]) -> list[CameraObservation]:
    return [obs for obs in observations if obs.detection_valid and obs.pixel_uv is not None]


def select_primary_camera(
    observations: Iterable[CameraObservation],
    *,
    primary_camera_id: str = "camera_A",
) -> CameraObservation | None:
    valid = usable_observations(observations)
    for obs in valid:
        if obs.camera_id == primary_camera_id:
            return obs
    return valid[0] if valid else None


def select_fixed_zone(
    observations: Iterable[CameraObservation],
    *,
    state_xy: Sequence[float],
    zones: Sequence[FixedZone],
) -> CameraObservation | None:
    valid = usable_observations(observations)
    for zone in zones:
        if not zone.contains(state_xy):
            continue
        for obs in valid:
            if obs.camera_id == zone.camera_id:
                return obs
    return valid[0] if valid else None


def select_highest_detector_score(observations: Iterable[CameraObservation]) -> CameraObservation | None:
    valid = usable_observations(observations)
    if not valid:
        return None
    return max(valid, key=lambda obs: (obs.detector_score, -obs.measurement_age_s))


def select_freshest_valid(observations: Iterable[CameraObservation]) -> CameraObservation | None:
    valid = usable_observations(observations)
    if not valid:
        return None
    return min(valid, key=lambda obs: (obs.measurement_age_s, -obs.detector_score))


def select_best_static_reliability(
    observations: Iterable[CameraObservation],
    qualities: Mapping[str, CameraQuality],
) -> CameraObservation | None:
    valid = usable_observations(observations)
    if not valid:
        return None
    return max(
        valid,
        key=lambda obs: (
            qualities.get(obs.camera_id, obs.quality()).p_available,
            obs.detector_score,
        ),
    )


def conservative_camera_score(
    observation: CameraObservation,
    quality: CameraQuality | None = None,
    *,
    lambda_age: float = 1.0,
    gamma_epistemic: float = 1.0,
) -> float:
    quality = quality or observation.quality()
    age = max(float(observation.measurement_age_s), 0.0)
    return float(
        quality.p_available
        * quality.association_confidence
        * math.exp(-float(lambda_age) * age)
        * math.exp(-float(gamma_epistemic) * float(quality.epistemic_score))
    )


def select_conservative_best_camera(
    observations: Iterable[CameraObservation],
    qualities: Mapping[str, CameraQuality] | None = None,
    *,
    lambda_age: float = 1.0,
    gamma_epistemic: float = 1.0,
) -> CameraObservation | None:
    qualities = qualities or {}
    valid = usable_observations(observations)
    if not valid:
        return None
    return max(
        valid,
        key=lambda obs: conservative_camera_score(
            obs,
            qualities.get(obs.camera_id),
            lambda_age=lambda_age,
            gamma_epistemic=gamma_epistemic,
        ),
    )


def sequential_kalman_update_2d(
    prior_mean_xy: Sequence[float],
    prior_cov_m2: Sequence[Sequence[float]],
    observations: Iterable[MapObservation],
    *,
    nis_gate: float | None = 9.21,
    disagreement_gate_m: float | None = None,
) -> SequentialFusionResult:
    """Apply valid camera map observations sequentially with per-camera gates."""

    mean = _pair(prior_mean_xy, "prior_mean_xy")
    cov = _matrix_2x2(prior_cov_m2, "prior_cov_m2")
    _validate_spd(cov, "prior_cov_m2")
    accepted: list[str] = []
    rejected: list[str] = []
    nis_by_camera: dict[str, float] = {}

    for obs in observations:
        z = obs.xy_m
        if disagreement_gate_m is not None:
            residual_norm = math.hypot(z[0] - mean[0], z[1] - mean[1])
            if residual_norm > float(disagreement_gate_m):
                rejected.append(obs.camera_id)
                nis_by_camera[obs.camera_id] = math.nan
                continue
        innovation = (z[0] - mean[0], z[1] - mean[1])
        s_mat = _mat_add(cov, obs.covariance_m2)
        nis = _quad_form_inverse_2x2(innovation, s_mat)
        nis_by_camera[obs.camera_id] = nis
        if nis_gate is not None and math.isfinite(nis) and nis > float(nis_gate):
            rejected.append(obs.camera_id)
            continue
        k = _mat_mul(cov, _mat_inv_2x2(s_mat))
        delta = _mat_vec(k, innovation)
        mean = (mean[0] + delta[0], mean[1] + delta[1])
        cov = _mat_mul(_mat_sub(((1.0, 0.0), (0.0, 1.0)), k), cov)
        cov = _symmetrize(cov)
        accepted.append(obs.camera_id)

    return SequentialFusionResult(
        mean_xy=mean,
        covariance_m2=cov,
        accepted_camera_ids=tuple(accepted),
        rejected_camera_ids=tuple(rejected),
        nis_by_camera=nis_by_camera,
    )


def camera_disagreement_m(observations: Sequence[MapObservation]) -> float:
    if len(observations) < 2:
        return 0.0
    max_dist = 0.0
    for i, obs_a in enumerate(observations):
        for obs_b in observations[i + 1 :]:
            max_dist = max(
                max_dist,
                math.hypot(obs_a.xy_m[0] - obs_b.xy_m[0], obs_a.xy_m[1] - obs_b.xy_m[1]),
            )
    return float(max_dist)


def _pair(value: Sequence[float], field_name: str) -> tuple[float, float]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    out = (float(value[0]), float(value[1]))
    if not math.isfinite(out[0]) or not math.isfinite(out[1]):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _matrix_2x2(value: Sequence[Sequence[float]], field_name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2x2 matrix")
    rows = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise ContractValidationError(f"{field_name} must be a 2x2 matrix")
        rows.append((float(row[0]), float(row[1])))
    return (rows[0], rows[1])


def _validate_spd(matrix: tuple[tuple[float, float], tuple[float, float]], field_name: str) -> None:
    a, b = matrix[0]
    c, d = matrix[1]
    if not all(math.isfinite(v) for v in (a, b, c, d)):
        raise ContractValidationError(f"{field_name} must be finite")
    if abs(b - c) > 1.0e-9:
        raise ContractValidationError(f"{field_name} must be symmetric")
    if a <= 0.0 or d <= 0.0 or a * d - b * c <= 0.0:
        raise ContractValidationError(f"{field_name} must be symmetric positive definite")


def _mat_add(a, b):
    return ((a[0][0] + b[0][0], a[0][1] + b[0][1]), (a[1][0] + b[1][0], a[1][1] + b[1][1]))


def _mat_sub(a, b):
    return ((a[0][0] - b[0][0], a[0][1] - b[0][1]), (a[1][0] - b[1][0], a[1][1] - b[1][1]))


def _mat_mul(a, b):
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


def _mat_vec(a, v):
    return (a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1])


def _mat_inv_2x2(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) <= 1.0e-12:
        raise ContractValidationError("matrix is singular")
    inv_det = 1.0 / det
    return ((a[1][1] * inv_det, -a[0][1] * inv_det), (-a[1][0] * inv_det, a[0][0] * inv_det))


def _quad_form_inverse_2x2(v, a):
    inv = _mat_inv_2x2(a)
    iv = _mat_vec(inv, v)
    return float(v[0] * iv[0] + v[1] * iv[1])


def _symmetrize(a):
    off = 0.5 * (a[0][1] + a[1][0])
    return ((float(a[0][0]), float(off)), (float(off), float(a[1][1])))
