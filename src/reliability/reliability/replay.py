"""Offline replay core for camera-observation estimator comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Iterable, Mapping, Sequence

from reliability.contracts import CameraQuality, ContractValidationError
from reliability.fusion import MapObservation, SequentialFusionResult, sequential_kalman_update_2d
from reliability.providers import CameraReliabilityProvider


class ReplayMode(str, Enum):
    ODOM_ONLY = "R0_odom_only"
    SINGLE_FIXED_R = "R1_single_fixed_R"
    SINGLE_FIXED_R_NIS = "R2_single_fixed_R_NIS"
    DETECTOR_SCORE_R = "R3_detector_score_R"
    CURRENT_GP_R = "R4_current_GP_R"
    SEQUENTIAL_FUSION = "M5_sequential_fusion"
    CONSERVATIVE_SELECTION = "M6_conservative_selection"


@dataclass(frozen=True)
class ReplayFrame:
    timestamp_s: float
    odometry_xy_m: tuple[float, float]
    observations: tuple[MapObservation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "odometry_xy_m", _pair(self.odometry_xy_m, "odometry_xy_m"))
        object.__setattr__(self, "observations", tuple(self.observations))


@dataclass(frozen=True)
class EvaluationFrame:
    timestamp_s: float
    truth_xy_m: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "truth_xy_m", _pair(self.truth_xy_m, "truth_xy_m"))


@dataclass(frozen=True)
class ReplayConfig:
    mode: ReplayMode
    initial_cov_m2: tuple[tuple[float, float], tuple[float, float]] = ((0.05**2, 0.0), (0.0, 0.05**2))
    process_noise_m2: tuple[tuple[float, float], tuple[float, float]] = ((0.012**2, 0.0), (0.0, 0.012**2))
    fixed_measurement_cov_m2: tuple[tuple[float, float], tuple[float, float]] = ((0.08**2, 0.0), (0.0, 0.08**2))
    nis_gate: float | None = None
    detector_score_min_cov_m2: float = 0.04**2
    detector_score_max_cov_m2: float = 0.40**2
    max_error_divergence_m: float = 1.0
    quality_providers: Mapping[str, CameraReliabilityProvider] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayStep:
    timestamp_s: float
    mean_xy_m: tuple[float, float]
    covariance_m2: tuple[tuple[float, float], tuple[float, float]]
    accepted_camera_ids: tuple[str, ...]
    rejected_camera_ids: tuple[str, ...]
    nis_by_camera: dict[str, float]


@dataclass(frozen=True)
class ReplayMetrics:
    rmse_m: float
    max_error_m: float
    final_error_m: float
    mean_nis: float
    mean_nees: float
    update_acceptance_rate: float
    divergence_count: int


@dataclass(frozen=True)
class ReplayResult:
    config: ReplayConfig
    steps: tuple[ReplayStep, ...]
    metrics: ReplayMetrics | None = None


def required_replay_configs(
    *,
    quality_providers: Mapping[str, CameraReliabilityProvider] | None = None,
) -> tuple[ReplayConfig, ...]:
    providers = dict(quality_providers or {})
    return (
        ReplayConfig(mode=ReplayMode.ODOM_ONLY),
        ReplayConfig(mode=ReplayMode.SINGLE_FIXED_R),
        ReplayConfig(mode=ReplayMode.SINGLE_FIXED_R_NIS, nis_gate=9.21),
        ReplayConfig(mode=ReplayMode.DETECTOR_SCORE_R, nis_gate=9.21),
        ReplayConfig(mode=ReplayMode.CURRENT_GP_R, nis_gate=9.21, quality_providers=providers),
    )


def run_replay(
    frames: Sequence[ReplayFrame],
    config: ReplayConfig,
    *,
    evaluation_frames: Sequence[EvaluationFrame] | None = None,
) -> ReplayResult:
    if not frames:
        return ReplayResult(config=config, steps=tuple(), metrics=None)

    ordered = sorted(frames, key=lambda frame: frame.timestamp_s)
    mean = ordered[0].odometry_xy_m
    cov = _matrix(config.initial_cov_m2, "initial_cov_m2")
    previous_odom = ordered[0].odometry_xy_m
    steps: list[ReplayStep] = []
    accepted_total = 0
    rejected_total = 0

    for idx, frame in enumerate(ordered):
        if idx > 0:
            dx = frame.odometry_xy_m[0] - previous_odom[0]
            dy = frame.odometry_xy_m[1] - previous_odom[1]
            mean = (mean[0] + dx, mean[1] + dy)
            cov = _mat_add(cov, _matrix(config.process_noise_m2, "process_noise_m2"))
            previous_odom = frame.odometry_xy_m

        replay_obs = _observations_for_config(
            frame.observations,
            config,
            mean_xy=mean,
            timestamp_s=frame.timestamp_s,
        )
        if replay_obs:
            fusion = sequential_kalman_update_2d(
                mean,
                cov,
                replay_obs,
                nis_gate=config.nis_gate,
            )
            mean = fusion.mean_xy
            cov = fusion.covariance_m2
        else:
            fusion = SequentialFusionResult(
                mean_xy=mean,
                covariance_m2=cov,
                accepted_camera_ids=tuple(),
                rejected_camera_ids=tuple(),
                nis_by_camera={},
            )

        accepted_total += len(fusion.accepted_camera_ids)
        rejected_total += len(fusion.rejected_camera_ids)
        steps.append(
            ReplayStep(
                timestamp_s=frame.timestamp_s,
                mean_xy_m=mean,
                covariance_m2=cov,
                accepted_camera_ids=fusion.accepted_camera_ids,
                rejected_camera_ids=fusion.rejected_camera_ids,
                nis_by_camera=fusion.nis_by_camera,
            )
        )

    metrics = None
    if evaluation_frames:
        metrics = compute_replay_metrics(
            steps,
            evaluation_frames,
            accepted_total=accepted_total,
            rejected_total=rejected_total,
            max_error_divergence_m=config.max_error_divergence_m,
        )
    return ReplayResult(config=config, steps=tuple(steps), metrics=metrics)


def compute_replay_metrics(
    steps: Sequence[ReplayStep],
    evaluation_frames: Sequence[EvaluationFrame],
    *,
    accepted_total: int = 0,
    rejected_total: int = 0,
    max_error_divergence_m: float = 1.0,
) -> ReplayMetrics:
    eval_by_stamp = {round(frame.timestamp_s, 9): frame for frame in evaluation_frames}
    errors: list[float] = []
    nees: list[float] = []
    nis: list[float] = []
    divergence_count = 0
    for step in steps:
        eval_frame = eval_by_stamp.get(round(step.timestamp_s, 9))
        if eval_frame is not None:
            err = (
                step.mean_xy_m[0] - eval_frame.truth_xy_m[0],
                step.mean_xy_m[1] - eval_frame.truth_xy_m[1],
            )
            e_norm = math.hypot(err[0], err[1])
            errors.append(e_norm)
            if e_norm > max_error_divergence_m:
                divergence_count += 1
            try:
                nees.append(_quad_form_inverse_2x2(err, step.covariance_m2))
            except ContractValidationError:
                pass
        nis.extend(v for v in step.nis_by_camera.values() if math.isfinite(v))

    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else math.nan
    final_error = errors[-1] if errors else math.nan
    acceptance_den = accepted_total + rejected_total
    return ReplayMetrics(
        rmse_m=rmse,
        max_error_m=max(errors) if errors else math.nan,
        final_error_m=final_error,
        mean_nis=sum(nis) / len(nis) if nis else math.nan,
        mean_nees=sum(nees) / len(nees) if nees else math.nan,
        update_acceptance_rate=(accepted_total / acceptance_den) if acceptance_den else math.nan,
        divergence_count=divergence_count,
    )


def _observations_for_config(
    observations: Sequence[MapObservation],
    config: ReplayConfig,
    *,
    mean_xy: Sequence[float],
    timestamp_s: float,
) -> tuple[MapObservation, ...]:
    if config.mode == ReplayMode.ODOM_ONLY:
        return tuple()
    if config.mode in (ReplayMode.SINGLE_FIXED_R, ReplayMode.SINGLE_FIXED_R_NIS):
        obs = observations[:1]
        return tuple(_with_cov(item, config.fixed_measurement_cov_m2, "fixed_R") for item in obs)
    if config.mode == ReplayMode.DETECTOR_SCORE_R:
        return tuple(_with_detector_score_cov(item, config) for item in observations[:1])
    if config.mode == ReplayMode.CURRENT_GP_R:
        return tuple(_with_provider_quality(item, config, mean_xy, timestamp_s) for item in observations[:1])
    if config.mode == ReplayMode.CONSERVATIVE_SELECTION:
        if not observations:
            return tuple()
        selected = max(
            observations,
            key=lambda item: item.quality.p_available * item.quality.association_confidence,
        )
        return (selected,)
    return tuple(observations)


def _with_cov(obs: MapObservation, cov, source: str) -> MapObservation:
    return MapObservation(
        camera_id=obs.camera_id,
        timestamp_s=obs.timestamp_s,
        xy_m=obs.xy_m,
        covariance_m2=cov,
        quality=CameraQuality(
            camera_id=obs.camera_id,
            p_available=obs.quality.p_available,
            conditional_cov_uv=obs.quality.conditional_cov_uv,
            association_confidence=obs.quality.association_confidence,
            epistemic_score=obs.quality.epistemic_score,
            stale=obs.quality.stale,
            source_model=source,
        ),
        source=source,
    )


def _with_detector_score_cov(obs: MapObservation, config: ReplayConfig) -> MapObservation:
    score = max(min(float(obs.quality.p_available), 1.0), 0.0)
    return _with_probability_cov(obs, config, score, "detector_score_R", obs.quality)


def _with_provider_quality(
    obs: MapObservation,
    config: ReplayConfig,
    mean_xy: Sequence[float],
    timestamp_s: float,
) -> MapObservation:
    provider = dict(config.quality_providers or {}).get(obs.camera_id)
    if provider is None:
        return obs
    quality = provider.query(obs.camera_id, mean_xy, timestamp_s)
    return _with_probability_cov(obs, config, quality.p_available, quality.source_model, quality)


def _with_probability_cov(
    obs: MapObservation,
    config: ReplayConfig,
    score: float,
    source: str,
    quality: CameraQuality,
) -> MapObservation:
    score = max(min(float(score), 1.0), 0.0)
    var = (
        score * float(config.detector_score_min_cov_m2)
        + (1.0 - score) * float(config.detector_score_max_cov_m2)
    )
    return MapObservation(
        camera_id=obs.camera_id,
        timestamp_s=obs.timestamp_s,
        xy_m=obs.xy_m,
        covariance_m2=((var, 0.0), (0.0, var)),
        quality=quality,
        source=source,
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


def _matrix(value: Sequence[Sequence[float]], field_name: str):
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2x2 matrix")
    rows = []
    for i, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise ContractValidationError(f"{field_name} must be a 2x2 matrix")
        rows.append((_finite(row[0], f"{field_name}[{i}][0]"), _finite(row[1], f"{field_name}[{i}][1]")))
    return (rows[0], rows[1])


def _mat_add(a, b):
    return ((a[0][0] + b[0][0], a[0][1] + b[0][1]), (a[1][0] + b[1][0], a[1][1] + b[1][1]))


def _mat_inv_2x2(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) <= 1.0e-12:
        raise ContractValidationError("matrix is singular")
    inv_det = 1.0 / det
    return ((a[1][1] * inv_det, -a[0][1] * inv_det), (-a[1][0] * inv_det, a[0][0] * inv_det))


def _mat_vec(a, v):
    return (a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1])


def _quad_form_inverse_2x2(v, a):
    inv = _mat_inv_2x2(a)
    iv = _mat_vec(inv, v)
    return float(v[0] * iv[0] + v[1] * iv[1])
