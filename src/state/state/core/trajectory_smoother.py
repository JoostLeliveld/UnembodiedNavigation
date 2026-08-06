"""Offline KF + RTS smoother over 2-D position — the operational training belief.

Promotes the one-off smoother in
the retired Option-A commissioning study (recoverable from the consolidation tag)
to an importable library, so that a *reusable* smoothed trajectory belief
``(mu_t^s, P_t^s)`` can back the operational residual used to estimate the
per-camera conditional covariance ``R_cond`` (registry experiment ``EXP-RCOND``).

What this is
------------
Forward Kalman filter driven by **wheel-odometry increments**, anchored by
absolute ``(x, y)`` fixes from the external camera network, then a
Rauch-Tung-Striebel backward pass that revises every historical belief. State is
planar position only::

    x_t = [p_x, p_y]^T,   x_t = x_{t-1} + u_t + w_t,   F_t = I

Heading is deliberately absent: it is odometry-driven under the locked campaign
configuration, exp5 (the behavioural reference) is 2-D, and the multicam
measurement path is ``z = (x, y)`` with ``h(x) = [I2 | 0]``. Widening the state
model is not a precondition for R_cond and is not done here.

Why the smoother at all
-----------------------
Not for point accuracy. exp5 measured that the online belief is already
camera-anchored, so an offline smoother does **not** beat its mean (p95 is in
fact somewhat worse -- the smoother absorbs anchor error). What smoothing fixes
is *covariance honesty*: NEES 16.8 -> 2.8, essentially calibrated. Since the
smoothed belief is consumed as a **training distribution** -- ``Sigma_{s,t}`` is
what weights the residual likelihood -- calibration is the property that matters.
That is the gate this module is held to (PLAN.md Gate R1), not point error.

Ground-truth firewall
---------------------
No function here accepts a truth argument. The public signatures are checked
against :data:`EVALUATION_ONLY_ARGUMENT_NAMES` at import-definition time by
``tests/state/test_trajectory_smoother.py``, which also cross-checks that list
against ``reliability.contracts.EVALUATION_ONLY_FIELD_NAMES``. Truth may only be
joined *after* smoothing, for scoring.

Leave-one-source-out
--------------------
:func:`without_source` drops one camera's measurements while keeping odometry and
the other cameras. Estimating ``R_cond,c`` against a trajectory that camera ``c``
itself anchored biases the residual toward zero; exp5 observed exactly that
absorption. See PLAN.md R3.

Pure numpy: no ROS, no file IO, no repo paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


#: chi-square 2 DOF at 0.99 -- the same threshold the runtime gate uses.
CHI2_2DOF_99 = 9.21

#: Argument names that would constitute a ground-truth input. Never parameters of
#: any public function in this module; asserted by the test suite.
EVALUATION_ONLY_ARGUMENT_NAMES = frozenset(
    {
        "gt",
        "gt_x",
        "gt_y",
        "gt_yaw",
        "gt_available",
        "ground_truth",
        "ground_truth_pose",
        "true_x",
        "true_y",
        "true_yaw",
        "true_available",
        "truth",
        "eval_res_x",
        "eval_res_y",
        "state_pos_error",
    }
)

_PSD_FLOOR = 1.0e-12


class SmootherError(ValueError):
    """Raised when inputs violate the smoother's contract."""


@dataclass(frozen=True)
class Measurement:
    """One absolute ``(x, y)`` fix from a named source, tied to a time index.

    ``index`` addresses the odometry track supplied to :func:`filter_forward`;
    several measurements may share an index (simultaneous cameras). ``source`` is
    the leave-one-out key -- a camera id -- and is required so that a residual can
    never be attributed to the wrong camera.
    """

    index: int
    z: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    source: str

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise SmootherError(f"measurement index must be >= 0, got {self.index}")
        z = np.asarray(self.z, dtype=float)
        if z.shape != (2,) or not np.all(np.isfinite(z)):
            raise SmootherError(f"measurement z must be a finite 2-vector, got {self.z!r}")
        cov = np.asarray(self.covariance, dtype=float)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            raise SmootherError(f"measurement covariance must be a finite 2x2, got {self.covariance!r}")
        if not np.allclose(cov, cov.T, atol=1e-12):
            raise SmootherError("measurement covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(cov))) <= 0.0:
            raise SmootherError("measurement covariance must be positive definite")
        if not str(self.source):
            raise SmootherError("measurement source must be a non-empty id")

    @property
    def R(self) -> np.ndarray:  # noqa: N802 - matches the filter equations
        return np.asarray(self.covariance, dtype=float)


#: exp5's per-step form -- ``std = q_base_m + q_per_metre * |u_t|``. Reproduces the
#: behavioural reference exactly, but is **rate-dependent**: the constant term
#: accumulates ``q_base_m^2`` of variance per *step*, so the same trajectory logged
#: at 50 Hz gets five times the drift variance of one logged at 10 Hz. Use only to
#: reproduce exp5 at exp5's own logging rate.
PROCESS_PER_STEP = "per_step"

#: Rate-invariant odometry random walk -- variance accumulates linearly in elapsed
#: time and in distance travelled::
#:
#:     Q_t = (sigma_per_sqrt_s^2 * dt + sigma_per_sqrt_m^2 * |u_t|) * I
#:
#: Two identical trajectories logged at different rates get the same total drift
#: variance, which is what makes ``P^s`` comparable across captures -- and
#: ``P^s`` is what the state-corrected R_cond estimator subtracts, so getting its
#: scale right is not cosmetic.
PROCESS_RATE_INVARIANT = "rate_invariant"

PROCESS_MODELS = (PROCESS_PER_STEP, PROCESS_RATE_INVARIANT)


@dataclass(frozen=True)
class SmootherConfig:
    """Frozen process/observation-gate parameters.

    Defaults are exp5's per-step form and an initial position std of 0.30 m, so
    this library reproduces the behavioural reference out of the box. Set
    ``process_model=PROCESS_RATE_INVARIANT`` (and supply ``dt``) for logs whose
    rate differs from exp5's -- see the constants above for why that matters.

    ``nis_soft_gate`` inflates rather than rejects::

        R_eff = R * max(1, NIS / gate)

    A hard reject would let the belief drift out of the measurement's reach and
    never recover -- the gate-runaway chain already diagnosed in this repo. Set to
    ``0.0`` to disable inflation entirely (the plain Kalman update).
    """

    q_base_m: float = 0.010
    q_per_metre: float = 0.15
    process_model: str = PROCESS_PER_STEP
    sigma_per_sqrt_s: float = 0.005
    sigma_per_sqrt_m: float = 0.040
    initial_position_std_m: float = 0.30
    nis_soft_gate: float = CHI2_2DOF_99
    psd_floor: float = _PSD_FLOOR

    def __post_init__(self) -> None:
        for name in (
            "q_base_m", "q_per_metre", "sigma_per_sqrt_s", "sigma_per_sqrt_m",
            "initial_position_std_m", "nis_soft_gate", "psd_floor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise SmootherError(f"{name} must be finite and >= 0, got {value}")
        if self.process_model not in PROCESS_MODELS:
            raise SmootherError(f"process_model must be one of {PROCESS_MODELS}, got {self.process_model!r}")

    def process_covariance(self, increment: np.ndarray, dt: float = float("nan")) -> np.ndarray:
        """``Q_t`` for one odometry increment -- isotropic in both forms."""
        distance = float(np.hypot(increment[0], increment[1]))
        if self.process_model == PROCESS_PER_STEP:
            std = self.q_base_m + self.q_per_metre * distance
            return (std**2) * np.eye(2)
        if not np.isfinite(dt) or dt < 0.0:
            raise SmootherError(
                f"process_model={PROCESS_RATE_INVARIANT!r} needs a finite dt >= 0; pass dt to filter_forward"
            )
        variance = (self.sigma_per_sqrt_s**2) * float(dt) + (self.sigma_per_sqrt_m**2) * distance
        return variance * np.eye(2)


@dataclass(frozen=True)
class ForwardPass:
    """Stored forward-pass quantities the RTS backward pass needs.

    ``predicted_mean[t]`` / ``predicted_cov[t]`` are ``mu_t^-`` / ``P_t^-`` -- the
    belief *before* any measurement at ``t`` is applied (for ``t = 0`` they are the
    initial belief). The backward recursion indexes them at ``t + 1``, matching
    the standard RTS statement.
    """

    filtered_mean: np.ndarray
    filtered_cov: np.ndarray
    predicted_mean: np.ndarray
    predicted_cov: np.ndarray
    update_counts: np.ndarray
    nis: np.ndarray
    config: SmootherConfig

    @property
    def n_steps(self) -> int:
        return int(self.filtered_mean.shape[0])


@dataclass(frozen=True)
class SmoothedTrajectory:
    """Filtered and smoothed belief over the whole track.

    ``smoothed_mean[t]`` / ``smoothed_cov[t]`` are ``mu_t^s`` / ``P_t^s`` -- the
    operational training belief. ``sources`` records which measurement sources
    were *included*, so an artifact can never misreport which camera anchored it.
    """

    filtered_mean: np.ndarray
    filtered_cov: np.ndarray
    predicted_mean: np.ndarray
    predicted_cov: np.ndarray
    smoothed_mean: np.ndarray
    smoothed_cov: np.ndarray
    update_counts: np.ndarray
    nis: np.ndarray
    sources: tuple[str, ...]
    config: SmootherConfig

    @property
    def n_steps(self) -> int:
        return int(self.smoothed_mean.shape[0])

    def position(self, index: int) -> np.ndarray:
        return np.asarray(self.smoothed_mean[index], dtype=float)

    def position_covariance(self, index: int) -> np.ndarray:
        return np.asarray(self.smoothed_cov[index], dtype=float)


def increments_from_track(track: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Odometry increments ``u_t = o_t - o_{t-1}`` with ``u_0 = 0``.

    The absolute odometry track is used only for differencing, so an arbitrary
    odometry frame offset (and any accumulated drift already in it) never enters
    as an absolute position.
    """
    odom = np.asarray(track, dtype=float)
    if odom.ndim != 2 or odom.shape[1] != 2:
        raise SmootherError(f"odometry track must have shape (n, 2), got {odom.shape}")
    if odom.shape[0] == 0:
        raise SmootherError("odometry track is empty")
    if not np.all(np.isfinite(odom)):
        raise SmootherError("odometry track must be finite -- drop or interpolate gaps first")
    increments = np.zeros_like(odom)
    increments[1:] = np.diff(odom, axis=0)
    return increments


def without_source(measurements: Iterable[Measurement], source: str) -> list[Measurement]:
    """Every measurement except those from ``source`` (leave-one-camera-out).

    Estimating ``R_cond,c`` from a residual whose reference trajectory camera
    ``c`` anchored drives that residual toward zero. Hold ``c`` out first.
    """
    held_out = str(source)
    if not held_out:
        raise SmootherError("source to hold out must be a non-empty id")
    return [m for m in measurements if m.source != held_out]


def _group_by_index(measurements: Iterable[Measurement], n_steps: int) -> dict[int, list[Measurement]]:
    grouped: dict[int, list[Measurement]] = {}
    for meas in measurements:
        index = int(meas.index)
        if index >= n_steps:
            raise SmootherError(
                f"measurement index {index} is past the end of the {n_steps}-step odometry track"
            )
        grouped.setdefault(index, []).append(meas)
    return grouped


def _symmetrize_psd(matrix: np.ndarray, floor: float) -> np.ndarray:
    out = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(out)
    if float(np.min(eigenvalues)) < floor:
        eigenvalues = np.clip(eigenvalues, floor, None)
        out = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        out = 0.5 * (out + out.T)
    return out


def filter_forward(
    increments: Sequence[Sequence[float]] | np.ndarray,
    measurements: Iterable[Measurement],
    *,
    initial_mean: Sequence[float],
    config: SmootherConfig | None = None,
    initial_cov: Sequence[Sequence[float]] | None = None,
    dt: Sequence[float] | np.ndarray | None = None,
) -> ForwardPass:
    """Forward Kalman pass over 2-D position.

    ``F_t = I`` (the state is advanced by the odometry increment itself) and
    ``H_t = I`` (the camera supplies an absolute position fix), so the equations
    reduce to::

        mu_t^- = mu_{t-1}^+ + u_t              P_t^- = P_{t-1}^+ + Q_t
        nu     = z - mu_t^-                    S = P_t^- + R_eff
        K      = P_t^- S^-1                    mu_t^+ = mu_t^- + K nu
        P_t^+  = (I - K) P_t^-

    Multiple measurements at one index are applied sequentially, which is the
    correct treatment for conditionally independent cameras (validated for this
    network: residual correlations collapse to 0.015-0.066 once conditioned on
    position).

    ``dt[t]`` is the elapsed time from ``t-1`` to ``t``; required by
    ``process_model=PROCESS_RATE_INVARIANT`` and ignored by the per-step form.
    """
    cfg = config if config is not None else SmootherConfig()
    u = np.asarray(increments, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise SmootherError(f"increments must have shape (n, 2), got {u.shape}")
    n = int(u.shape[0])
    if n == 0:
        raise SmootherError("increments are empty")
    if not np.all(np.isfinite(u)):
        raise SmootherError("increments must be finite")

    mean = np.asarray(initial_mean, dtype=float).reshape(2)
    if not np.all(np.isfinite(mean)):
        raise SmootherError("initial_mean must be finite")
    if initial_cov is None:
        cov = (cfg.initial_position_std_m**2) * np.eye(2)
    else:
        cov = np.asarray(initial_cov, dtype=float)
        if cov.shape != (2, 2):
            raise SmootherError(f"initial_cov must be 2x2, got {cov.shape}")
        cov = _symmetrize_psd(cov, cfg.psd_floor)

    if dt is None:
        steps = np.full(n, np.nan)
    else:
        steps = np.asarray(dt, dtype=float).reshape(-1)
        if steps.shape[0] != n:
            raise SmootherError(f"dt must have {n} entries to match the increments, got {steps.shape[0]}")
        if np.any(steps[1:] < 0.0) or not np.all(np.isfinite(steps[1:])):
            raise SmootherError("dt entries from index 1 onward must be finite and >= 0")

    grouped = _group_by_index(measurements, n)

    filtered_mean = np.zeros((n, 2))
    filtered_cov = np.zeros((n, 2, 2))
    predicted_mean = np.zeros((n, 2))
    predicted_cov = np.zeros((n, 2, 2))
    update_counts = np.zeros(n, dtype=int)
    nis_trace = np.full(n, np.nan)

    for k in range(n):
        if k > 0:
            mean = mean + u[k]
            cov = cov + cfg.process_covariance(u[k], steps[k])
            cov = _symmetrize_psd(cov, cfg.psd_floor)
        predicted_mean[k] = mean
        predicted_cov[k] = cov

        step_nis: list[float] = []
        for meas in grouped.get(k, ()):
            innovation = np.asarray(meas.z, dtype=float) - mean
            S = cov + meas.R
            nis = float(innovation @ np.linalg.solve(S, innovation))
            step_nis.append(nis)
            R_eff = meas.R
            if cfg.nis_soft_gate > 0.0 and np.isfinite(nis):
                R_eff = meas.R * max(1.0, nis / cfg.nis_soft_gate)
            S_eff = cov + R_eff
            # K = P S^-1 via a solve; S_eff is symmetric so no transpose dance.
            K = np.linalg.solve(S_eff, cov).T
            mean = mean + K @ innovation
            cov = _symmetrize_psd((np.eye(2) - K) @ cov, cfg.psd_floor)
            update_counts[k] += 1

        if step_nis:
            nis_trace[k] = float(np.max(step_nis))
        filtered_mean[k] = mean
        filtered_cov[k] = cov

    return ForwardPass(
        filtered_mean=filtered_mean,
        filtered_cov=filtered_cov,
        predicted_mean=predicted_mean,
        predicted_cov=predicted_cov,
        update_counts=update_counts,
        nis=nis_trace,
        config=cfg,
    )


def rts_backward(forward: ForwardPass) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel backward pass. Returns ``(mu^s, P^s)``.

    With ``F_t = I``::

        G_t   = P_t^+ (P_{t+1}^-)^-1
        mu_t^s = mu_t^+ + G_t (mu_{t+1}^s - mu_{t+1}^-)
        P_t^s  = P_t^+ + G_t (P_{t+1}^s - P_{t+1}^-) G_t^T

    Terminal step is the filtered belief. Gains use a linear solve, never an
    explicit inverse.
    """
    n = forward.n_steps
    smoothed_mean = forward.filtered_mean.copy()
    smoothed_cov = forward.filtered_cov.copy()
    floor = forward.config.psd_floor

    for k in range(n - 2, -1, -1):
        P_pred_next = forward.predicted_cov[k + 1]
        # G = P_f (P_p)^-1 with symmetric P_p: solve(P_p, P_f)^T == P_f P_p^-1.
        G = np.linalg.solve(P_pred_next, forward.filtered_cov[k]).T
        smoothed_mean[k] = forward.filtered_mean[k] + G @ (smoothed_mean[k + 1] - forward.predicted_mean[k + 1])
        correction = G @ (smoothed_cov[k + 1] - P_pred_next) @ G.T
        smoothed_cov[k] = _symmetrize_psd(forward.filtered_cov[k] + correction, floor)

    return smoothed_mean, smoothed_cov


def smooth_trajectory(
    increments: Sequence[Sequence[float]] | np.ndarray,
    measurements: Iterable[Measurement],
    *,
    initial_mean: Sequence[float],
    config: SmootherConfig | None = None,
    initial_cov: Sequence[Sequence[float]] | None = None,
    dt: Sequence[float] | np.ndarray | None = None,
) -> SmoothedTrajectory:
    """Forward filter then RTS backward pass in one call.

    The returned object carries both beliefs so a caller can report the
    filtered-vs-smoothed comparison Gate R1 requires without re-running anything.
    """
    included = list(measurements)
    forward = filter_forward(
        increments,
        included,
        initial_mean=initial_mean,
        config=config,
        initial_cov=initial_cov,
        dt=dt,
    )
    smoothed_mean, smoothed_cov = rts_backward(forward)
    sources = tuple(sorted({m.source for m in included}))
    return SmoothedTrajectory(
        filtered_mean=forward.filtered_mean,
        filtered_cov=forward.filtered_cov,
        predicted_mean=forward.predicted_mean,
        predicted_cov=forward.predicted_cov,
        smoothed_mean=smoothed_mean,
        smoothed_cov=smoothed_cov,
        update_counts=forward.update_counts,
        nis=forward.nis,
        sources=sources,
        config=forward.config,
    )


def nees(
    mean: np.ndarray,
    covariance: np.ndarray,
    reference: np.ndarray,
    *,
    floor: float = _PSD_FLOOR,
) -> np.ndarray:
    """Normalised estimation error squared, per step. **Evaluation only.**

    ``reference`` is ground truth, so this is scoring, never inference -- it is
    called *after* smoothing and its output never re-enters the filter. A
    calibrated 2-D belief gives NEES ~ chi2_2, i.e. mean 2 and median ~1.39.
    """
    mu = np.asarray(mean, dtype=float)
    P = np.asarray(covariance, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if mu.shape != ref.shape:
        raise SmootherError(f"mean {mu.shape} and reference {ref.shape} must match")
    out = np.full(mu.shape[0], np.nan)
    for k in range(mu.shape[0]):
        error = mu[k] - ref[k]
        if not np.all(np.isfinite(error)) or not np.all(np.isfinite(P[k])):
            continue
        S = _symmetrize_psd(P[k], floor)
        out[k] = float(error @ np.linalg.solve(S, error))
    return out
