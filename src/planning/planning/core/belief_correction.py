"""Measurement-space-agnostic belief correction chain.

Extracted verbatim from ``unicycle_planner_node._apply_pixel_correction`` so the
single-camera (paper-1) and multi-camera paths can share ONE gate chain instead
of maintaining two that drift apart. Paper-1 semantics are the reference: this
module must reproduce them bit-for-bit (see
``tests/planning/test_belief_correction.py``, which replays the locked
``honest_campaign_v1`` corrections through :func:`evaluate_gates`).

The only thing that genuinely differs between the two stacks is the
**measurement space**:

- paper-1: z = (u, v) pixels, h(x) nonlinear via the camera model, R = ``R_plan``
  from the visibility GP (:class:`PixelMeasurementSource`);
- multicam: z = (x, y) metres, h(x) = [I2 | 0], R = the fused covariance from
  ``camera_manager_node`` (:class:`FusedMapMeasurementSource`).

Everything downstream of ``linearize`` -- freshness, dt plausibility, the jump
limiter, the NIS gate, PSD projection, reason codes -- is measurement-space
agnostic and lives here, once.

Deliberately ROS-free and side-effect-free: it takes numpy arrays and returns a
:class:`CorrectionOutcome`. Committing the belief, publishing diagnostics,
throttling and belief bootstrap stay in the node, because they touch ROS state.

INVARIANT: ``R_eff`` is supplied by the measurement source and is used exactly as
given. This module must never rescale or reweight it. A measurement's stated
uncertainty is the measurement model's claim; silently adjusting it here would make
the covariance a property of the filter rather than of the sensor, and no downstream
calibration number would mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Protocol

import numpy as np

from planning.core.efe_utils import wrap_angle


class RejectReason(str, Enum):
    """Why a correction was not folded into the belief.

    Values match the strings the node has always passed to the reject-code
    lookup; ``ACCEPTED`` is the sentinel for "no gate fired".
    """

    ACCEPTED = 'accepted'
    STALE_AGE = 'stale_age'
    DT_IMPLAUSIBLE = 'dt_implausible'
    MISSING_SNAPSHOT = 'missing_snapshot'
    UPDATE_FAILED = 'update_failed'
    JUMP_TOO_LARGE = 'jump_too_large'
    NIS_TOO_LARGE = 'nis_too_large'
    #: The correction describes an instant at or before the committed belief.
    #: A causal filter cannot apply it without smoothing; record it explicitly.
    NOT_NEWER = 'not_newer_than_belief'
    #: The motion history cannot safely bridge the committed belief to this
    #: correction. This used to be a silent early return in the metric path.
    REPLAY_GAP = 'replay_gap_too_large'
    #: The belief diverged from a fresh, reliable correction. Not a plain
    #: rejection: the caller is expected to RE-ANCHOR (see `recover` below),
    #: because rejecting would lock the belief out of recovery.
    DIVERGED = 'diverged'


#: Wire codes published on ``/planner/pixel_correction_diagnostics`` and logged
#: as ``pixel_corr_reject_reason_code``. Frozen 1-6 -- existing campaign CSVs and
#: ``experiment_logger._pixel_correction_reject_reason_name`` decode those.
#: Append only; never renumber.
REJECT_CODES: dict[str, float] = {
    RejectReason.STALE_AGE.value: 1.0,
    RejectReason.DT_IMPLAUSIBLE.value: 2.0,
    RejectReason.MISSING_SNAPSHOT.value: 3.0,
    RejectReason.UPDATE_FAILED.value: 4.0,
    RejectReason.JUMP_TOO_LARGE.value: 5.0,
    RejectReason.NIS_TOO_LARGE.value: 6.0,
    RejectReason.DIVERGED.value: 7.0,
    RejectReason.NOT_NEWER.value: 8.0,
    RejectReason.REPLAY_GAP.value: 9.0,
}

#: How the caller should treat the belief after a non-accepted outcome.
#: ``REJECT`` = hold the prediction (the caller still advances the stamp and
#: inflates, so a rejection can never freeze the belief); ``REANCHOR`` = snap to
#: the measurement, the belief is the thing that is wrong.
RECOVER_REJECT = 'reject'
RECOVER_REANCHOR = 'reanchor'

#: Measurement-space markers, published so a CSV reader can tell which stack
#: produced a row (the pixel_corr_* columns are shared by both).
SPACE_PIXEL_UV = 0.0
SPACE_MAP_XY = 1.0

UNKNOWN_REJECT_CODE = 99.0
ACCEPTED_CODE = 0.0


def reject_code(reason: Any) -> float:
    """Wire code for ``reason``; unknown reasons map to 99, as before."""
    if reason is RejectReason.ACCEPTED or reason == RejectReason.ACCEPTED.value:
        return ACCEPTED_CODE
    key = reason.value if isinstance(reason, RejectReason) else str(reason or '').strip()
    return REJECT_CODES.get(key, UNKNOWN_REJECT_CODE)


@dataclass(frozen=True)
class CorrectionGates:
    """Thresholds for the shared gate chain, built once from node parameters.

    ``dt_nominal_s`` is the planner step (``self.dt``); it only feeds the
    implausible-dt ceiling.
    """

    pixel_timeout_s: float = 1.25
    dt_nominal_s: float = 0.1
    skip_stale: bool = True
    max_jump_m: float = 0.5
    nis_threshold: float = 9.21
    cov_eig_floor: float = 1e-9
    min_state_cov: float = 0.0

    #: Metric innovation beyond which the BELIEF (not the measurement) is judged
    #: to have diverged, so the caller re-anchors instead of rejecting. Only
    #: meaningful when the measurement is in metres; 0 disables it, which is the
    #: paper-1 pixel-path default (an innovation in pixels has no metric scale).
    reanchor_innov_m: float = 0.0

    #: Kinematic plausibility cap on the PREDICTION. The motion replay
    #: extrapolates the last command when odometry samples are missing, which
    #: can invent ~0.9 m of travel over a 1.5 s gap. 0 disables the cap, which
    #: is the paper-1 default -- the locked campaign ran without it.
    max_predict_speed_mps: float = 0.0
    predict_margin_m: float = 0.05

    @property
    def future_tolerance_s(self) -> float:
        return max(float(self.pixel_timeout_s), 0.25)

    @property
    def max_dt_s(self) -> float:
        return max(2.0 * float(self.pixel_timeout_s), 4.0 * float(self.dt_nominal_s), 0.5)

    def age_is_invalid(self, age: float) -> bool:
        return bool(
            self.skip_stale
            and (age > self.pixel_timeout_s or age < -self.future_tolerance_s)
        )

    def dt_is_implausible(self, dt_s: float) -> bool:
        return bool(dt_s > self.max_dt_s)

    def jump_is_too_large(self, xy_update_norm_m: float) -> bool:
        return bool(self.max_jump_m > 0.0 and xy_update_norm_m > self.max_jump_m)

    def nis_is_too_large(self, nis: float) -> bool:
        return bool(
            self.nis_threshold > 0.0
            and math.isfinite(nis)
            and nis > self.nis_threshold
        )

    def belief_has_diverged(self, innov_norm_m: float) -> bool:
        return bool(
            self.reanchor_innov_m > 0.0
            and math.isfinite(innov_norm_m)
            and innov_norm_m > self.reanchor_innov_m
        )

    def max_predict_step_m(self, dt_s: float) -> float:
        """Kinematic ceiling on one prediction step; inf when disabled."""
        if self.max_predict_speed_mps <= 0.0 or not math.isfinite(dt_s):
            return math.inf
        return self.max_predict_speed_mps * max(dt_s, 0.0) + self.predict_margin_m


def evaluate_gates(
    gates: CorrectionGates,
    *,
    age: float = math.nan,
    dt_s: float = math.nan,
    xy_update_norm_m: float = math.nan,
    nis: float = math.nan,
    innov_norm_m: float = math.nan,
    snapshot_missing: bool = False,
    update_failed: bool = False,
) -> RejectReason:
    """The single decision function: first gate that fires, in canonical order.

    Every threshold predicate is NaN-safe (``nan > x`` is ``False``), so calling
    this incrementally as quantities become known is equivalent to calling it
    once with the full record. :func:`apply_correction` relies on that; the
    golden-trace test exploits it to replay logged corrections that only carry
    the post-update quantities.
    """
    if gates.age_is_invalid(age):
        return RejectReason.STALE_AGE
    if gates.dt_is_implausible(dt_s):
        return RejectReason.DT_IMPLAUSIBLE
    if snapshot_missing:
        return RejectReason.MISSING_SNAPSHOT
    if update_failed:
        return RejectReason.UPDATE_FAILED
    # Ahead of the jump/NIS gates on purpose: a diverged belief trips both, and
    # rejecting on either would block the re-anchor that recovers it.
    if gates.belief_has_diverged(innov_norm_m):
        return RejectReason.DIVERGED
    if gates.jump_is_too_large(xy_update_norm_m):
        return RejectReason.JUMP_TOO_LARGE
    if gates.nis_is_too_large(nis):
        return RejectReason.NIS_TOO_LARGE
    return RejectReason.ACCEPTED


def clamp_prediction(m_before, m_pred, S_pred, *, max_step_m: float):
    """Cap an implausible predicted step and pay for it in covariance.

    ``_replay_cmd_log_interval`` falls back to integrating the LAST COMMAND when
    odometry samples are missing for the interval, which can invent ~0.9 m of
    travel over a 1.5 s gap and hand the update a prediction the robot could not
    physically have reached.

    Clamping rather than rejecting, because the *correction* is not at fault --
    the prediction is. The displacement removed is added back as isotropic xy
    variance, so the filter becomes less confident rather than quietly confident
    about a pose it invented.

    Returns ``(m_pred, S_pred, clipped_m)``; ``clipped_m`` is 0.0 when no clamp
    was applied.
    """
    m_before = np.asarray(m_before, dtype=float)
    m_pred = np.asarray(m_pred, dtype=float).copy()
    S_pred = np.asarray(S_pred, dtype=float)
    if not math.isfinite(max_step_m):
        return m_pred, S_pred, 0.0
    step = m_pred[:2] - m_before[:2]
    step_norm = float(np.linalg.norm(step))
    if not math.isfinite(step_norm) or step_norm <= max_step_m or step_norm <= 0.0:
        return m_pred, S_pred, 0.0
    clipped = step_norm - max_step_m
    m_pred[:2] = m_before[:2] + step * (max_step_m / step_norm)
    S_pred = S_pred.copy()
    S_pred[0, 0] += clipped ** 2
    S_pred[1, 1] += clipped ** 2
    return m_pred, S_pred, clipped


def project_to_psd(S, floor: float = 1e-9):
    S = np.asarray(S, dtype=float)
    w, v = np.linalg.eigh(S)
    w = np.maximum(w, floor)
    return (v * w) @ v.T


def regularize_covariance(S, min_state_cov: float):
    """Keep planner belief covariance positive enough for stable updates.

    Byte-for-byte the node's historical ``_regularize_state_covariance``: floor
    the leading 3 diagonal entries, then symmetrize.
    """
    S = np.asarray(S, dtype=float).copy()
    if min_state_cov > 0.0:
        for i in range(min(3, S.shape[0])):
            if S[i, i] < min_state_cov:
                S[i, i] = min_state_cov
    return (S + S.T) / 2.0


@dataclass
class CorrectionSnapshot:
    """Belief + measurement captured atomically before the update runs."""

    belief_m: np.ndarray
    belief_S: np.ndarray
    belief_stamp: Any
    cmd: np.ndarray
    meas: np.ndarray
    meas_stamp: Any = None
    yaw_meas: float | None = None
    yaw_sigma: float = math.nan
    yaw_source: float = 0.0


@dataclass
class Linearization:
    """Measurement model evaluated at the predicted belief.

    ``Gamma`` is the state/measurement cross-covariance (``S Hᵀ`` for a linear
    model, the moment-matched equivalent for ET1/ET2), ``Sigma_y`` the
    innovation covariance *including* R.
    """

    z: np.ndarray
    mu_y: np.ndarray
    Gamma: np.ndarray
    Sigma_y: np.ndarray
    R_eff: np.ndarray
    S_eff: np.ndarray
    p_vis: float = math.nan
    gain_scale: float = 1.0


@dataclass
class KalmanUpdate:
    next_m: np.ndarray
    next_S: np.ndarray
    innov: np.ndarray
    mu_y: np.ndarray
    S_y: np.ndarray
    K: np.ndarray


class MeasurementSource(Protocol):
    """The one seam between the single-camera and multi-camera stacks."""

    #: short identifier for diagnostics/logging, e.g. ``'pixel'`` / ``'fused_map'``
    label: str
    #: SPACE_PIXEL_UV or SPACE_MAP_XY -- published so a CSV reader can tell
    #: which stack produced a row.
    measurement_space: float

    def snapshot(self) -> CorrectionSnapshot | None:
        """Atomically capture belief + measurement, or None if incomplete."""

    def linearize(self, m_pred, S_pred, snapshot: CorrectionSnapshot) -> Linearization | None:
        """Evaluate the measurement model at the prediction, or None on failure."""


@dataclass
class CorrectionOutcome:
    """Everything the node needs to commit the belief and publish diagnostics."""

    reason: RejectReason
    age: float = math.nan
    dt_s: float = math.nan
    m_pred: np.ndarray | None = None
    #: Predicted covariance at the correction stamp, before the update. The
    #: caller needs it to hold-and-inflate on a rejection.
    S_pred: np.ndarray | None = None
    next_m: np.ndarray | None = None
    next_S: np.ndarray | None = None
    meas: np.ndarray | None = None
    mu_y: np.ndarray | None = None
    innov: np.ndarray | None = None
    R_eff: np.ndarray | None = None
    K: np.ndarray | None = None
    nis: float = math.nan
    xy_update_norm_m: float = math.nan
    p_vis: float = math.nan
    gain_scale: float = math.nan
    snapshot: CorrectionSnapshot | None = None
    replay_meta: dict = field(default_factory=dict)
    yaw_info: dict = field(default_factory=dict)
    #: Prediction step removed by the kinematic clamp, in metres (0 = none).
    predict_clipped_m: float = 0.0
    measurement_space: float = SPACE_PIXEL_UV

    @property
    def accepted(self) -> bool:
        return self.reason is RejectReason.ACCEPTED

    @property
    def reject_code(self) -> float:
        return reject_code(self.reason)

    @property
    def recover(self) -> str:
        """What the caller should do with the belief when not accepted.

        ``DIVERGED`` means the belief, not the measurement, is wrong -- snapping
        to the correction is recovery, whereas rejecting would lock the belief
        out of it and let it dead-reckon away.
        """
        return RECOVER_REANCHOR if self.reason is RejectReason.DIVERGED else RECOVER_REJECT

    @property
    def innov_norm_m(self) -> float:
        """Metric innovation norm; only meaningful for a map-xy measurement."""
        if self.innov is None:
            return math.nan
        return float(np.linalg.norm(np.asarray(self.innov, dtype=float).reshape(-1)[:2]))


def _report(callback, message: str) -> None:
    if callback is not None:
        callback(message)


def compute_update(m_pred, lin: Linearization, *, cov_eig_floor: float,
                   on_shape_error=None) -> KalmanUpdate | None:
    """Moment-matched Kalman update. Measurement-space agnostic.

    Reproduces ``_compute_pixel_uv_update`` exactly, and for a linear map-xy
    model (``Gamma = S[:, :2]``, ``Sigma_y = S[:2, :2] + R``) it reduces to the
    ``K = S[:, :2] S_y⁻¹`` / ``S - K S[:2, :]`` form the multicam path used.
    """
    m_pred = np.asarray(m_pred, dtype=float)
    mu_y = np.asarray(lin.mu_y, dtype=float).reshape(-1)
    meas = np.asarray(lin.z, dtype=float).reshape(-1)
    if meas.size != mu_y.size:
        _report(on_shape_error,
                "Pixel correction shape mismatch: "
                f"meas_dim={meas.size}, pred_dim={mu_y.size}. "
                "Skipping correction for this message.")
        return None

    Sigma_y = np.asarray(lin.Sigma_y, dtype=float)
    Gamma = np.asarray(lin.Gamma, dtype=float)
    if Sigma_y.shape != (meas.size, meas.size) or Gamma.shape[1] != meas.size:
        _report(on_shape_error,
                "Pixel correction covariance shape mismatch: "
                f"Sigma_y={Sigma_y.shape}, Gamma={Gamma.shape}, meas_dim={meas.size}. "
                "Skipping correction for this message.")
        return None

    innov = meas - mu_y
    if innov.size >= 3:
        innov[2] = wrap_angle(innov[2])
    Sigma_y = (Sigma_y + Sigma_y.T) / 2.0
    Sigma_inv = np.linalg.pinv(Sigma_y)
    K = Gamma @ Sigma_inv
    gain_scale = float(lin.gain_scale)
    next_m = m_pred + gain_scale * (K @ innov)
    next_m[2] = wrap_angle(next_m[2])
    # LIMITATION: this is the standard Kalman covariance form, which is correct only at
    # the optimal gain, i.e. gain_scale == 1. Every fused camera correction passes 1.0, so
    # the fusion experiment is unaffected. Only PixelMeasurementSource scales the gain (by
    # the visibility GP's predicted visibility), and on that path the covariance below is
    # inconsistent with the mean above. Switch that path to Joseph form before driving it
    # again. docs/open_questions.md, "Known limitations".
    next_S = np.asarray(lin.S_eff, dtype=float) - gain_scale * (Gamma @ Sigma_inv @ Gamma.T)
    next_S = 0.5 * (next_S + next_S.T)
    eig_min = np.min(np.linalg.eigvalsh(next_S))
    if eig_min < cov_eig_floor:
        next_S = project_to_psd(next_S, floor=cov_eig_floor)
    return KalmanUpdate(
        next_m=next_m,
        next_S=next_S,
        innov=innov,
        mu_y=mu_y,
        S_y=Sigma_y,
        K=K,
    )


def normalized_innovation_squared(innov, S_y) -> float:
    """NIS on the leading 2 components, matching the node's historical form."""
    if S_y is None:
        return float('nan')
    innov_2d = np.asarray(innov, dtype=float).reshape(-1)[:2]
    try:
        S_inv = np.linalg.inv(np.asarray(S_y, dtype=float)[:2, :2])
    except np.linalg.LinAlgError:
        return float('nan')
    return float(innov_2d @ S_inv @ innov_2d)


def yaw_report(next_m, next_S, m_pred) -> dict:
    """Report the theta component induced indirectly by the XY update.

    Pure reporting -- it does not modify the estimate (the historical
    ``_apply_yaw_anchor_after_pixel_update`` behaviour).
    """
    theta_update = float(wrap_angle(float(next_m[2]) - float(m_pred[2])))
    return {
        'next_m': next_m,
        'next_S': next_S,
        'theta_update_from_uv_rad': theta_update,
        'yaw_correction_applied': False,
        'innov_theta': math.nan,
        'k_theta_theta': math.nan,
        'theta_update_total_rad': theta_update,
    }


class PixelMeasurementSource:
    """Paper-1 source: z = (u, v) pixels, R = ``R_plan`` from the visibility GP.

    ``snapshot_fn`` is supplied by the node (it reads belief + latest pixel
    measurement under the data lock). ``planner`` provides the frozen
    observation model; ``planner_for_obs`` is the global planner when one
    exists, matching the node's historical choice.
    """

    label = 'pixel'
    measurement_space = SPACE_PIXEL_UV

    def __init__(self, *, planner, planner_for_obs, snapshot_fn, corr_method: str):
        self._planner = planner
        self._planner_for_obs = planner_for_obs
        self._snapshot_fn = snapshot_fn
        self._corr_method = corr_method

    def snapshot(self) -> CorrectionSnapshot | None:
        return self._snapshot_fn()

    def linearize(self, m_pred, S_pred, snapshot: CorrectionSnapshot) -> Linearization | None:
        p_vis, R_eff, S_eff, gain_scale = (
            self._planner_for_obs.observation_model_with_visibility(m_pred, S_pred)
        )
        mu_y, Sigma_y, Gamma = self._planner.approx_observation(
            m_pred, S_eff, method=self._corr_method, R_override=R_eff
        )
        return Linearization(
            z=np.asarray(snapshot.meas, dtype=float).reshape(-1),
            mu_y=np.asarray(mu_y, dtype=float).reshape(-1),
            Gamma=np.asarray(Gamma, dtype=float),
            Sigma_y=np.asarray(Sigma_y, dtype=float),
            R_eff=np.asarray(R_eff, dtype=float),
            S_eff=np.asarray(S_eff, dtype=float),
            p_vis=float(p_vis),
            gain_scale=float(gain_scale),
        )


class FusedMapMeasurementSource:
    """Multicam source: z = (x, y) metres, h(x) = [I2 | 0], R from the manager.

    Not yet wired into the node -- that is step 2 of the consolidation. Kept
    here so the seam is real rather than hypothetical, and unit-tested against
    the closed-form ``K = S[:, :2] S_y⁻¹`` update the multicam path used.
    """

    label = 'fused_map'
    measurement_space = SPACE_MAP_XY

    def __init__(self, *, snapshot_fn, measurement_cov_fn):
        self._snapshot_fn = snapshot_fn
        self._measurement_cov_fn = measurement_cov_fn

    def snapshot(self) -> CorrectionSnapshot | None:
        return self._snapshot_fn()

    def linearize(self, m_pred, S_pred, snapshot: CorrectionSnapshot) -> Linearization | None:
        z = np.asarray(snapshot.meas, dtype=float).reshape(-1)[:2]
        R = np.asarray(self._measurement_cov_fn(), dtype=float).reshape(2, 2)
        S_pred = np.asarray(S_pred, dtype=float)
        return Linearization(
            z=z,
            mu_y=np.asarray(m_pred, dtype=float)[:2].copy(),
            Gamma=S_pred[:, :2].copy(),
            Sigma_y=S_pred[:2, :2] + R,
            R_eff=R,
            S_eff=S_pred.copy(),
            p_vis=math.nan,
            gain_scale=1.0,
        )


def apply_correction(
    *,
    source: MeasurementSource,
    gates: CorrectionGates,
    replay,
    age: float,
    dt_s: float,
    on_shape_error=None,
) -> CorrectionOutcome:
    """Run one correction through the shared chain.

    ``replay(m, S, from_stamp, to_stamp, fallback_cmd, fallback_dt)`` performs
    latency compensation and returns ``(m_pred, S_pred, meta)``; the node
    supplies it because it reads the odom/cmd logs.

    Returns a :class:`CorrectionOutcome`; the caller decides what to commit. The
    age and dt gates are evaluated here, but the caller is responsible for the
    side effects the node has always attached to them (belief re-init on
    implausible dt, throttling, bootstrap).
    """
    space = float(getattr(source, 'measurement_space', SPACE_PIXEL_UV))

    reason = evaluate_gates(gates, age=age)
    if reason is not RejectReason.ACCEPTED:
        return CorrectionOutcome(reason=reason, age=age, measurement_space=space)

    reason = evaluate_gates(gates, age=age, dt_s=dt_s)
    if reason is not RejectReason.ACCEPTED:
        return CorrectionOutcome(reason=reason, age=age, dt_s=dt_s, measurement_space=space)

    snapshot = source.snapshot()
    if snapshot is None:
        return CorrectionOutcome(
            reason=RejectReason.MISSING_SNAPSHOT, age=age, dt_s=dt_s, measurement_space=space
        )

    m_pred, S_pred, replay_meta = replay(
        snapshot.belief_m,
        snapshot.belief_S,
        snapshot.belief_stamp,
        snapshot.meas_stamp,
        snapshot.cmd,
        dt_s,
    )
    m_pred, S_pred, predict_clipped_m = clamp_prediction(
        snapshot.belief_m, m_pred, S_pred, max_step_m=gates.max_predict_step_m(dt_s)
    )

    lin = source.linearize(m_pred, S_pred, snapshot)
    if lin is None:
        return CorrectionOutcome(
            reason=RejectReason.UPDATE_FAILED,
            age=age,
            dt_s=dt_s,
            m_pred=m_pred,
            S_pred=S_pred,
            meas=np.asarray(snapshot.meas, dtype=float),
            snapshot=snapshot,
            replay_meta=replay_meta,
            measurement_space=space,
            predict_clipped_m=predict_clipped_m,
        )

    update = compute_update(m_pred, lin, cov_eig_floor=gates.cov_eig_floor,
                            on_shape_error=on_shape_error)
    if update is None:
        return CorrectionOutcome(
            reason=RejectReason.UPDATE_FAILED,
            age=age,
            dt_s=dt_s,
            m_pred=m_pred,
            S_pred=S_pred,
            meas=np.asarray(lin.z, dtype=float),
            R_eff=np.asarray(lin.R_eff, dtype=float),
            p_vis=float(lin.p_vis),
            gain_scale=float(lin.gain_scale),
            snapshot=snapshot,
            replay_meta=replay_meta,
            measurement_space=space,
            predict_clipped_m=predict_clipped_m,
        )

    xy_update_norm_m = float(
        np.linalg.norm(np.asarray(update.next_m[:2] - np.asarray(m_pred, dtype=float)[:2], dtype=float))
    )
    nis = normalized_innovation_squared(update.innov, update.S_y)

    innov_norm_m = float(np.linalg.norm(np.asarray(update.innov, dtype=float).reshape(-1)[:2]))

    outcome = CorrectionOutcome(
        reason=evaluate_gates(
            gates,
            age=age,
            dt_s=dt_s,
            xy_update_norm_m=xy_update_norm_m,
            nis=nis,
            innov_norm_m=innov_norm_m,
        ),
        age=age,
        dt_s=dt_s,
        m_pred=np.asarray(m_pred, dtype=float),
        S_pred=np.asarray(S_pred, dtype=float),
        meas=np.asarray(lin.z, dtype=float).reshape(-1),
        mu_y=update.mu_y,
        innov=update.innov,
        R_eff=np.asarray(lin.R_eff, dtype=float),
        K=update.K,
        nis=nis,
        xy_update_norm_m=xy_update_norm_m,
        p_vis=float(lin.p_vis),
        gain_scale=float(lin.gain_scale),
        snapshot=snapshot,
        replay_meta=replay_meta,
        measurement_space=space,
        predict_clipped_m=predict_clipped_m,
    )
    if not outcome.accepted:
        return outcome

    info = yaw_report(update.next_m, update.next_S, m_pred)
    outcome.next_m = info['next_m']
    outcome.next_S = regularize_covariance(info['next_S'], gates.min_state_cov)
    outcome.yaw_info = info
    return outcome
