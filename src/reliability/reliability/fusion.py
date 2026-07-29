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

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "timestamp_s": float(self.timestamp_s),
            "xy_m": [float(self.xy_m[0]), float(self.xy_m[1])],
            "covariance_m2": [
                [float(self.covariance_m2[0][0]), float(self.covariance_m2[0][1])],
                [float(self.covariance_m2[1][0]), float(self.covariance_m2[1][1])],
            ],
            "quality": self.quality.to_dict(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "MapObservation":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("MapObservation payload must be a mapping")
        missing = {"camera_id", "timestamp_s", "xy_m", "covariance_m2", "quality"} - set(payload)
        if missing:
            raise ContractValidationError(
                "MapObservation is missing fields: " + ", ".join(sorted(missing))
            )
        return cls(
            camera_id=str(payload["camera_id"]),
            timestamp_s=float(payload["timestamp_s"]),
            xy_m=tuple(payload["xy_m"]),
            covariance_m2=tuple(tuple(row) for row in payload["covariance_m2"]),
            quality=CameraQuality.from_dict(payload["quality"]),
            source=str(payload.get("source", "")),
        )


#: Wire format version for the per-camera map-observation batch. The planner
#: refuses a batch it does not recognise rather than silently misreading it.
MAP_OBSERVATION_BATCH_SCHEMA = "map_observation_batch/1"


def map_observations_to_json(observations: Sequence[MapObservation], *, frame_id: str = "map") -> str:
    """Serialize per-camera map observations for the planner's sequential filter.

    This is the seam between projection/covariance (camera_manager's job) and
    belief updating (the planner's job). Each observation keeps its OWN
    covariance: the planner folds them in one at a time, so a per-camera
    covariance model reaches the filter intact instead of being collapsed into
    a single fused pose first.
    """
    import json

    return json.dumps(
        {
            "schema": MAP_OBSERVATION_BATCH_SCHEMA,
            "frame_id": frame_id,
            "observations": [obs.to_dict() for obs in observations],
        },
        sort_keys=True,
        allow_nan=False,
    )


def map_observations_from_json(text: str) -> tuple[list[MapObservation], str]:
    """Inverse of :func:`map_observations_to_json`; returns (observations, frame_id)."""
    import json

    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ContractValidationError("map-observation batch must be a JSON object")
    schema = str(payload.get("schema", ""))
    if schema != MAP_OBSERVATION_BATCH_SCHEMA:
        raise ContractValidationError(
            f"unsupported map-observation batch schema {schema!r}; "
            f"expected {MAP_OBSERVATION_BATCH_SCHEMA!r}"
        )
    raw = payload.get("observations", [])
    if not isinstance(raw, Sequence):
        raise ContractValidationError("map-observation batch 'observations' must be a list")
    return [MapObservation.from_dict(item) for item in raw], str(payload.get("frame_id", "map"))


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


def independent_measurement_fusion_2d(
    observations: Iterable[MapObservation],
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
    """Fuse independent 2-D measurements without inventing a state prior.

    ``R_fused = (sum R_i^-1)^-1`` and
    ``z_fused = R_fused sum R_i^-1 z_i``.  This produces a measurement for the
    planner's one real belief filter.  It must not be replaced by a Kalman
    update against an arbitrary identity prior, which would filter the cameras
    once here and then filter the resulting pseudo-measurement a second time.
    Cross-camera correlation is deliberately outside this historical baseline.
    """

    items = tuple(observations)
    if not items:
        raise ContractValidationError("at least one map observation is required")
    information = ((0.0, 0.0), (0.0, 0.0))
    information_vector = (0.0, 0.0)
    for observation in items:
        precision = _mat_inv_2x2(observation.covariance_m2)
        information = _mat_add(information, precision)
        weighted = _mat_vec(precision, observation.xy_m)
        information_vector = (
            information_vector[0] + weighted[0],
            information_vector[1] + weighted[1],
        )
    covariance = _symmetrize(_mat_inv_2x2(_symmetrize(information)))
    mean = _mat_vec(covariance, information_vector)
    _validate_spd(covariance, "fused_measurement_covariance_m2")
    return mean, covariance


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


@dataclass(frozen=True)
class FuseOrSelectDecision:
    """Outcome of the fuse-vs-select consistency logic (plan 09, §12.5)."""

    mode: str
    selected: tuple[str, ...]
    excluded: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.mode not in ("fuse", "select", "none"):
            raise ContractValidationError("mode must be one of 'fuse', 'select', 'none'")
        object.__setattr__(self, "selected", tuple(str(c) for c in self.selected))
        object.__setattr__(self, "excluded", tuple(str(c) for c in self.excluded))


def joseph_update_2d(
    state_xy: Sequence[float],
    cov_xy: Sequence[Sequence[float]],
    observation: MapObservation,
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]], float]:
    """Single position update in Joseph form: (I−KH)P(I−KH)ᵀ + KRKᵀ, H = I.

    Numerically stable at small R. Returns (state, covariance, nis) where
    nis is the innovation Mahalanobis distance d² = νᵀ S⁻¹ ν, S = P + R.
    """

    mean = _pair(state_xy, "state_xy")
    cov = _matrix_2x2(cov_xy, "cov_xy")
    _validate_spd(cov, "cov_xy")
    r_mat = observation.covariance_m2

    z = observation.xy_m
    innovation = (z[0] - mean[0], z[1] - mean[1])
    s_mat = _mat_add(cov, r_mat)
    nis = _quad_form_inverse_2x2(innovation, s_mat)
    k = _mat_mul(cov, _mat_inv_2x2(s_mat))
    delta = _mat_vec(k, innovation)
    mean_post = (mean[0] + delta[0], mean[1] + delta[1])
    i_minus_k = _mat_sub(((1.0, 0.0), (0.0, 1.0)), k)
    cov_post = _mat_add(
        _mat_mul(_mat_mul(i_minus_k, cov), _transpose(i_minus_k)),
        _mat_mul(_mat_mul(k, r_mat), _transpose(k)),
    )
    return mean_post, _symmetrize(cov_post), float(nis)


def robust_reweight_covariance(
    covariance: Sequence[Sequence[float]],
    nis: float,
    *,
    dof: float = 4.0,
    w_min: float = 0.1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Student-t robust reweighting of an observation covariance.

    w = (dof + 2) / (dof + nis); returns R / clamp(w, w_min, 1.0).

    Deliberate deviation from plan 09's literal ``R / max(w, w_min)``: the
    weight is additionally capped at 1.0 so the mapping is monotone
    *inflating* only. Uncapped, nis < 2 would give w > 1 and shrink R
    ("sharpening"), violating plan 07's property that lower trust never
    reduces covariance — and a suspiciously-perfect observation should not
    be trusted beyond its nominal R.  Limits: nis → 0 ⇒ R unchanged;
    nis → ∞ ⇒ R / w_min floor. Ablatable: skipping this call = plain update.
    """

    r_mat = _matrix_2x2(covariance, "covariance")
    _validate_spd(r_mat, "covariance")
    nis_value = float(nis)
    if not math.isfinite(nis_value) or nis_value < 0.0:
        raise ContractValidationError("nis must be finite and non-negative")
    dof_value = float(dof)
    if not math.isfinite(dof_value) or dof_value <= 0.0:
        raise ContractValidationError("dof must be finite and positive")
    w_min_value = float(w_min)
    if not math.isfinite(w_min_value) or not 0.0 < w_min_value <= 1.0:
        raise ContractValidationError("w_min must be in (0, 1]")

    weight = (dof_value + 2.0) / (dof_value + nis_value)
    weight = min(max(weight, w_min_value), 1.0)
    scale = 1.0 / weight
    return (
        (r_mat[0][0] * scale, r_mat[0][1] * scale),
        (r_mat[1][0] * scale, r_mat[1][1] * scale),
    )


def expected_information_gain(
    cov_prior: Sequence[Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
) -> float:
    """Expected information gain Δ = log|P⁻| − log|P⁺| of a position update.

    P⁺ is the standard posterior for H = I: P⁺ = (I − K)P⁻ with
    K = P⁻ (P⁻ + R)⁻¹ (equivalently (P⁻¹ + R⁻¹)⁻¹). Always > 0 for SPD inputs.
    """

    p_mat = _matrix_2x2(cov_prior, "cov_prior")
    _validate_spd(p_mat, "cov_prior")
    r_mat = _matrix_2x2(observation_covariance, "observation_covariance")
    _validate_spd(r_mat, "observation_covariance")

    s_mat = _mat_add(p_mat, r_mat)
    k = _mat_mul(p_mat, _mat_inv_2x2(s_mat))
    p_post = _symmetrize(_mat_mul(_mat_sub(((1.0, 0.0), (0.0, 1.0)), k), p_mat))
    det_prior = _det_2x2(p_mat)
    det_post = _det_2x2(p_post)
    if det_post <= 0.0:
        raise ContractValidationError("posterior covariance is not positive definite")
    return float(math.log(det_prior) - math.log(det_post))


def select_information_best(
    observations: Sequence[MapObservation],
    cov_prior: Sequence[Sequence[float]],
) -> MapObservation | None:
    """B8: pick the observation with the largest expected information gain.

    Geometry-aware: a high-reliability observation with poor geometry
    (large variance along the prior's uncertain axis) can lose to a less
    reliable one that constrains the right direction.
    """

    p_mat = _matrix_2x2(cov_prior, "cov_prior")
    _validate_spd(p_mat, "cov_prior")
    best: MapObservation | None = None
    best_gain = -math.inf
    for obs in observations:
        if obs is None:
            continue
        gain = expected_information_gain(p_mat, obs.covariance_m2)
        if gain > best_gain:
            best = obs
            best_gain = gain
    return best


def fuse_or_select(
    observations: Sequence[MapObservation],
    taus: Mapping[str, float],
    healths: Mapping[str, float],
    *,
    tau_min: float,
    h_min: float,
    disagreement_gate_m: float,
) -> FuseOrSelectDecision:
    """Fuse-vs-select consistency logic (plan 09, §12.5).

    Candidate set C = {i : τ_i ≥ tau_min and h_i ≥ h_min} (a camera missing
    from ``taus``/``healths`` is treated as 0.0, i.e. excluded).

    - |C| = 0 → mode 'none'; |C| = 1 → mode 'select'.
    - If all pairwise position disagreements within C are ≤
      ``disagreement_gate_m`` → mode 'fuse' with all of C.
    - Otherwise iteratively drop the camera involved in the most
      above-gate pairs (ties broken by lowest τ, then camera_id for
      determinism) and re-check, until the remainder is mutually
      consistent (fuse) or a single camera remains (select).
    """

    tau_min_value = _require_probability(tau_min, "tau_min")
    h_min_value = _require_probability(h_min, "h_min")
    gate = float(disagreement_gate_m)
    if not math.isfinite(gate) or gate <= 0.0:
        raise ContractValidationError("disagreement_gate_m must be finite and positive")

    excluded: list[str] = []
    candidates: list[MapObservation] = []
    validated_taus: dict[str, float] = {}
    for obs in observations:
        tau = _require_probability(
            taus.get(obs.camera_id, 0.0),
            f"taus[{obs.camera_id!r}]",
        )
        health = _require_probability(
            healths.get(obs.camera_id, 0.0),
            f"healths[{obs.camera_id!r}]",
        )
        validated_taus[obs.camera_id] = tau
        if tau >= tau_min_value and health >= h_min_value:
            candidates.append(obs)
        else:
            excluded.append(obs.camera_id)

    if not candidates:
        return FuseOrSelectDecision(
            mode="none",
            selected=(),
            excluded=tuple(excluded),
            reason="no candidate passed tau_min/h_min thresholds",
        )
    if len(candidates) == 1:
        return FuseOrSelectDecision(
            mode="select",
            selected=(candidates[0].camera_id,),
            excluded=tuple(excluded),
            reason="single candidate above thresholds",
        )

    dropped_for_disagreement: list[str] = []
    while len(candidates) > 1:
        violations: dict[str, int] = {obs.camera_id: 0 for obs in candidates}
        any_violation = False
        for i, obs_a in enumerate(candidates):
            for obs_b in candidates[i + 1 :]:
                dist = math.hypot(
                    obs_a.xy_m[0] - obs_b.xy_m[0],
                    obs_a.xy_m[1] - obs_b.xy_m[1],
                )
                if dist > gate:
                    any_violation = True
                    violations[obs_a.camera_id] += 1
                    violations[obs_b.camera_id] += 1
        if not any_violation:
            break
        worst_id = min(
            violations,
            key=lambda cid: (-violations[cid], validated_taus[cid], cid),
        )
        dropped_for_disagreement.append(worst_id)
        excluded.append(worst_id)
        candidates = [obs for obs in candidates if obs.camera_id != worst_id]

    if len(candidates) == 1:
        return FuseOrSelectDecision(
            mode="select",
            selected=(candidates[0].camera_id,),
            excluded=tuple(excluded),
            reason=(
                "pairwise disagreement above gate; dropped "
                + ", ".join(dropped_for_disagreement)
            ),
        )
    reason = "all candidates mutually consistent"
    if dropped_for_disagreement:
        reason = (
            "consistent after dropping disagreeing camera(s): "
            + ", ".join(dropped_for_disagreement)
        )
    return FuseOrSelectDecision(
        mode="fuse",
        selected=tuple(obs.camera_id for obs in candidates),
        excluded=tuple(excluded),
        reason=reason,
    )


def _require_probability(value: object, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"{field_name} must be a finite probability in [0, 1]"
        ) from exc
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ContractValidationError(
            f"{field_name} must be a finite probability in [0, 1]"
        )
    return out


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


def _transpose(a):
    return ((a[0][0], a[1][0]), (a[0][1], a[1][1]))


def _det_2x2(a):
    return float(a[0][0] * a[1][1] - a[0][1] * a[1][0])


def _symmetrize(a):
    off = 0.5 * (a[0][1] + a[1][0])
    return ((float(a[0][0]), float(off)), (float(off), float(a[1][1])))
