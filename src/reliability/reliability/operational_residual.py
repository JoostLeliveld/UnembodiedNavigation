"""Operational (ground-truth-free) camera residuals and state-corrected R_cond.

The repo's only measurement residual is ``eval_res_x/y = pred_world - ground_truth``,
which is EVALUATION-ONLY and firewalled (``^eval_`` in
``config/leakage_firewall.yaml``). It can characterise the camera network but can
never *train* a deployable covariance, which is why ``R_cond`` is the recorded
open blocker on the factorised observation model.

This module supplies the operational alternative
(``experiments/operational_residual_rcond/PLAN.md`` R2/R3): reference the residual
to a **smoothed operational belief** instead of truth,

    r_t = z_t - h(mu_t^s)                                    (operational residual)
    C_t = H_t P_t^s H_t^T + R_cond(s_t)                      (total predictive cov)

and estimate ``R_cond`` from the *state-corrected* second moment rather than from
``r r^T``. That distinction is the whole point: ``r r^T`` contains the robot's own
state uncertainty ``H P^s H^T``, and at these belief sizes it is not negligible, so
using it would silently inflate the camera's covariance by the filter's ignorance.

Two properties this module enforces that the incoming spec did not:

1. **Identifiability.** A residual measured against a trajectory that the same
   camera anchored is driven toward zero. Every estimate records whether its
   camera was held out of the reference (:attr:`OperationalResidual.held_out`),
   and :func:`circularity_factor` quantifies the gap. A not-held-out estimate may
   not be quoted as a measurement of camera noise (Gate R3).
2. **Bias is never absorbed into covariance.** The mean residual is reported
   separately (:attr:`ResidualSummary.mean_residual`) and the covariance is taken
   about that mean, matching :func:`reliability.conditional_covariance.estimate_conditional_covariance`.
   Inflating ``R`` to cover a deterministic projection bias is explicitly not done.

Frames: ``xy`` (metres, ``h(x) = [I2 | 0]`` -- the multicam path) or ``uv``
(pixels, nonlinear ``h`` -- the paper-1 path). The frame is carried on every record
and estimate; no number is returned without one. See PLAN.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from reliability.conditional_covariance import (
    CovarianceEstimate,
    default_shrinkage_lambda,
)
from reliability.contracts import ContractValidationError


#: Measurement frames, mirroring ``conditional_covariance._ALLOWED_FRAMES``.
ALLOWED_FRAMES = ("uv", "xy")

_PSD_FLOOR_EPS = 1.0e-9


@dataclass(frozen=True)
class OperationalResidual:
    """One accepted detection scored against the smoothed operational belief.

    ``state_projection`` is ``H_t P_t^s H_t^T`` -- the measurement-space image of
    the robot's own uncertainty. It is stored per record rather than recomputed so
    that the state-corrected estimator and the predictive likelihood cannot drift
    apart.
    """

    camera_id: str
    index: int
    frame: str
    measured: tuple[float, float]
    predicted: tuple[float, float]
    residual: tuple[float, float]
    state_projection: tuple[tuple[float, float], tuple[float, float]]
    anchored_by: tuple[str, ...]
    held_out: bool

    def total_covariance(
        self, r_cond: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """``C_t = H P^s H^T + R_cond`` -- the covariance the residual is scored under."""
        R = _as_matrix(r_cond, "r_cond")
        C = np.asarray(self.state_projection, dtype=float) + R
        return _as_tuple(_psd_floor(C))


@dataclass(frozen=True)
class ResidualSummary:
    """Bias and state-corrected covariance for one camera, with provenance.

    ``raw_second_moment`` is the uncorrected central covariance of the residuals
    (i.e. what a naive ``r r^T`` estimator would report, mean-removed) and
    ``mean_state_projection`` is what was subtracted from it. Both are kept so the
    size of the correction is always auditable rather than implicit.
    """

    camera_id: str
    frame: str
    sample_count: int
    held_out: bool
    anchored_by: tuple[str, ...]
    mean_residual: tuple[float, float]
    raw_second_moment: tuple[tuple[float, float], tuple[float, float]]
    mean_state_projection: tuple[tuple[float, float], tuple[float, float]]
    state_corrected: tuple[tuple[float, float], tuple[float, float]]
    psd_projection_applied: bool

    @property
    def bias_norm(self) -> float:
        return float(np.hypot(*self.mean_residual))


def build_operational_residuals(
    *,
    smoothed_mean: np.ndarray,
    smoothed_cov: np.ndarray,
    measurements: Iterable,
    camera_id: str,
    frame: str = "xy",
    anchored_by: Sequence[str] = (),
    observation_jacobian: Sequence[Sequence[float]] | None = None,
) -> list[OperationalResidual]:
    """Residual records for one camera against a smoothed trajectory.

    ``measurements`` are :class:`state.core.trajectory_smoother.Measurement`-shaped
    objects (``.index``, ``.z``, ``.source``); only those whose ``source`` matches
    ``camera_id`` are used, so passing the full set is safe and a mismatched
    ``camera_id`` yields an empty list rather than mislabelled residuals.

    ``held_out`` is derived from ``anchored_by`` -- the sources that actually
    anchored the trajectory -- not asserted by the caller, so a record cannot claim
    independence it does not have.

    ``observation_jacobian`` defaults to ``I2`` (the ``xy`` path, ``h(x) = [I2|0]``).
    For the ``uv`` path pass the camera projection Jacobian at ``mu_t^s``.
    """
    frame_key = _validated_frame(frame)
    if not str(camera_id):
        raise ContractValidationError("camera_id must be a non-empty id")

    mu = np.asarray(smoothed_mean, dtype=float)
    P = np.asarray(smoothed_cov, dtype=float)
    if mu.ndim != 2 or mu.shape[1] != 2:
        raise ContractValidationError(f"smoothed_mean must have shape (n, 2), got {mu.shape}")
    if P.shape != (mu.shape[0], 2, 2):
        raise ContractValidationError(
            f"smoothed_cov must have shape ({mu.shape[0]}, 2, 2), got {P.shape}"
        )

    H = np.eye(2) if observation_jacobian is None else _as_matrix(observation_jacobian, "observation_jacobian")
    sources = tuple(sorted({str(s) for s in anchored_by}))
    held_out = str(camera_id) not in sources

    out: list[OperationalResidual] = []
    for meas in measurements:
        if str(getattr(meas, "source", "")) != str(camera_id):
            continue
        index = int(meas.index)
        if index < 0 or index >= mu.shape[0]:
            raise ContractValidationError(
                f"measurement index {index} outside the {mu.shape[0]}-step smoothed trajectory"
            )
        z = np.asarray(meas.z, dtype=float)
        predicted = H @ mu[index]
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(predicted))):
            continue
        out.append(
            OperationalResidual(
                camera_id=str(camera_id),
                index=index,
                frame=frame_key,
                measured=(float(z[0]), float(z[1])),
                predicted=(float(predicted[0]), float(predicted[1])),
                residual=(float(z[0] - predicted[0]), float(z[1] - predicted[1])),
                state_projection=_as_tuple(H @ P[index] @ H.T),
                anchored_by=sources,
                held_out=held_out,
            )
        )
    return out


def summarize_residuals(residuals: Sequence[OperationalResidual]) -> ResidualSummary:
    """Bias plus state-corrected covariance for one camera's residuals.

    The estimator is the plan's ``S - mean(H P^s H^T)``: the central second moment
    of the residuals (population divisor ``n``, matching
    ``conditional_covariance``) minus the mean measurement-space state
    uncertainty, then projected onto the PSD cone.

    Subtraction can drive the result indefinite when the camera is sharper than
    the belief; that is real information (it says the residual is *smaller* than
    the state uncertainty alone would predict, i.e. the reference trajectory is
    correlated with the measurement -- the circularity of Gate R3). It is
    projected, and the projection is flagged rather than hidden.
    """
    records = list(residuals)
    if len(records) < 2:
        raise ContractValidationError("at least 2 residuals are required")
    frames = {r.frame for r in records}
    if len(frames) != 1:
        raise ContractValidationError(f"residuals mix frames: {sorted(frames)}")
    cameras = {r.camera_id for r in records}
    if len(cameras) != 1:
        raise ContractValidationError(f"residuals mix cameras: {sorted(cameras)}")
    held = {r.held_out for r in records}
    if len(held) != 1:
        raise ContractValidationError("residuals mix held-out and anchored references")

    r = np.array([record.residual for record in records], dtype=float)
    n = r.shape[0]
    mean_residual = r.mean(axis=0)
    centred = r - mean_residual
    sample = (centred.T @ centred) / n

    projections = np.array([record.state_projection for record in records], dtype=float)
    mean_projection = projections.mean(axis=0)

    corrected = sample - mean_projection
    floored = _psd_floor(corrected)
    projection_applied = not np.allclose(floored, corrected, atol=0.0, rtol=0.0)

    return ResidualSummary(
        camera_id=records[0].camera_id,
        frame=records[0].frame,
        sample_count=n,
        held_out=records[0].held_out,
        anchored_by=records[0].anchored_by,
        mean_residual=(float(mean_residual[0]), float(mean_residual[1])),
        raw_second_moment=_as_tuple(sample),
        mean_state_projection=_as_tuple(mean_projection),
        state_corrected=_as_tuple(floored),
        psd_projection_applied=bool(projection_applied),
    )


def shrink_summary(
    summary: ResidualSummary,
    shrinkage_target: Sequence[Sequence[float]],
    shrinkage_lambda: float | None = None,
) -> CovarianceEstimate:
    """Shrink a state-corrected covariance toward a pooled target.

    Uses the same pre-registered rule as
    :func:`reliability.conditional_covariance.estimate_conditional_covariance`
    (``lambda = min(1, 10/(10+n))`` by default) so the operational estimator and
    the GT-referenced one are directly comparable, and returns the same
    :class:`CovarianceEstimate` type so downstream consumers cannot tell them
    apart by shape -- only by the artifact that records which was used.
    """
    target = _as_matrix(shrinkage_target, "shrinkage_target")
    if float(np.min(np.linalg.eigvalsh(target))) <= 0.0:
        raise ContractValidationError("shrinkage_target must be positive definite")
    if shrinkage_lambda is None:
        lam = default_shrinkage_lambda(summary.sample_count)
    else:
        lam = float(shrinkage_lambda)
        if not np.isfinite(lam) or lam < 0.0 or lam > 1.0:
            raise ContractValidationError("shrinkage_lambda must be in [0, 1]")

    corrected = np.asarray(summary.state_corrected, dtype=float)
    shrunk = (1.0 - lam) * corrected + lam * target
    return CovarianceEstimate(
        camera_id=summary.camera_id,
        covariance=_as_tuple(_psd_floor(shrunk)),
        sample_count=summary.sample_count,
        frame=summary.frame,
        shrinkage_lambda=lam,
    )


def circularity_factor(held_out: ResidualSummary, anchored: ResidualSummary) -> float:
    """How much the reference trajectory absorbed of the camera's own error.

    ``sqrt(tr(R_held_out) / tr(R_anchored))``: 1.0 means the camera contributed
    nothing to its own reference; large values mean the anchored estimate is
    optimistic by that factor in std terms. Gate R3 requires this be reported
    beside any ``R_cond``.
    """
    if held_out.camera_id != anchored.camera_id:
        raise ContractValidationError("circularity_factor compares one camera against itself")
    if held_out.frame != anchored.frame:
        raise ContractValidationError("circularity_factor requires a common frame")
    if not held_out.held_out:
        raise ContractValidationError("the first argument must be the held-out estimate")
    if anchored.held_out:
        raise ContractValidationError("the second argument must be the anchored estimate")
    numerator = float(np.trace(np.asarray(held_out.state_corrected, dtype=float)))
    denominator = float(np.trace(np.asarray(anchored.state_corrected, dtype=float)))
    if denominator <= 0.0:
        return float("inf")
    return float(np.sqrt(max(numerator, 0.0) / denominator))


def total_covariances(
    residuals: Sequence[OperationalResidual],
    r_cond: Sequence[Sequence[float]] | Mapping[str, Sequence[Sequence[float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Per-record ``C_t = H P^s H^T + R_cond`` for scoring.

    Feed the result straight to ``conditional_covariance.matrix_nll`` /
    ``chi2_coverage`` / ``sharpness`` alongside ``[r.residual for r in residuals]``:
    the predictive likelihood must include the state term, or a model will be
    rewarded for absorbing the filter's uncertainty into the camera's covariance.

    ``r_cond`` may be one matrix or a per-camera mapping.
    """
    records = list(residuals)
    if not records:
        raise ContractValidationError("residuals must be non-empty")
    if isinstance(r_cond, Mapping):
        missing = sorted({r.camera_id for r in records} - set(r_cond))
        if missing:
            raise ContractValidationError(f"r_cond has no entry for cameras {missing}")
        return [record.total_covariance(r_cond[record.camera_id]) for record in records]
    return [record.total_covariance(r_cond) for record in records]


def pooled_target(summaries: Sequence[ResidualSummary]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Sample-count-weighted pooled state-corrected covariance across cameras.

    The shrinkage target for :func:`shrink_summary`. Weighting by count means a
    camera with 15 detections does not pull the pool as hard as one with 300.
    """
    items = list(summaries)
    if not items:
        raise ContractValidationError("summaries must be non-empty")
    frames = {s.frame for s in items}
    if len(frames) != 1:
        raise ContractValidationError(f"summaries mix frames: {sorted(frames)}")
    total_weight = float(sum(s.sample_count for s in items))
    if total_weight <= 0.0:
        raise ContractValidationError("summaries carry no samples")
    pooled = np.zeros((2, 2))
    for summary in items:
        pooled += (summary.sample_count / total_weight) * np.asarray(
            summary.state_corrected, dtype=float
        )
    return _as_tuple(_psd_floor(pooled))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _validated_frame(frame: str) -> str:
    key = str(frame)
    if key not in ALLOWED_FRAMES:
        raise ContractValidationError(f"frame must be one of {ALLOWED_FRAMES}, got {frame!r}")
    return key


def _as_matrix(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ContractValidationError(f"{name} must be a finite 2x2 matrix")
    return matrix


def _as_tuple(matrix: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    m = np.asarray(matrix, dtype=float)
    return (
        (float(m[0][0]), float(m[0][1])),
        (float(m[1][0]), float(m[1][1])),
    )


def _psd_floor(matrix: np.ndarray, eps: float = _PSD_FLOOR_EPS) -> np.ndarray:
    out = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(out)
    if float(np.min(eigenvalues)) < eps:
        eigenvalues = np.clip(eigenvalues, eps, None)
        out = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        out = 0.5 * (out + out.T)
    return out
