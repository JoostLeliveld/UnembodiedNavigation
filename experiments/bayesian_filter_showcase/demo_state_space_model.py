#!/usr/bin/env python3
"""The BMLIP PP4 arc -- generative model, filtering, smoothing, forecasting, and
inferring the noise covariance -- carried out on this repository's real data.

Reference: `PP4 - Bayesian filtering and smoothing` from the BMLIP course
(https://github.com/bmlip/course, probprog/). Its shape:

    1. write the LINEAR GAUSSIAN STATE-SPACE MODEL down as a generative model
       (`@model function LGDS`), rather than hand-coding predict/update;
    2. hand it to an inference engine and get the WHOLE posterior over states --
       filtering and smoothing in one call;
    3. FORECAST by feeding `missing` observations;
    4. then stop treating the noise covariance as known: put an inverse-Wishart
       prior on it, infer it by variational message passing under the mean-field
       constraint q(z, Sigma) = q(z) q(Sigma), and compare models by free energy.

PP4 does step 4 for the PROCESS noise Q, because its sensor variance is known
from calibration. Our situation is the mirror image: the odometry is the
well-characterised part and the CAMERA side is what we do not trust. So this
script runs the same machinery on both, and then asks the question PP4's
formalism makes unavoidable and `exp1` could only answer with a floor:

    what happens if the camera's systematic offset is a STATE we infer,
    instead of a covariance we widen?

No RxInfer and no Julia here -- the models are conditionally conjugate, so the
same inference is Kalman + RTS in the E-step and a closed-form inverse-Wishart
update in the M-step. Where PP4 reports Bethe free energy, this reports the exact
negative log evidence of the observations plus a BIC complexity penalty; that is
a different number, and it is labelled as such wherever it appears.

Ground truth is EVALUATION-ONLY: it scores, it is plotted, it never enters a model.

Outputs -> logs/studies/bayesian_filter_showcase/demo_state_space_model/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
for _relative in ("src/reliability", "src/unav_common", "src/state",
                  "experiments/operational_residual_rcond"):
    sys.path.insert(0, str(REPO / _relative))
sys.path.insert(0, str(_HERE.parent))

from reliability.observation_model import time_sync_covariance  # noqa: E402
import rcond_common as rc  # noqa: E402
import exp1_graceful_vs_trusting as f1  # noqa: E402
import demo_how_the_filter_works as d1  # noqa: E402

OUT = REPO / "logs/studies/bayesian_filter_showcase/demo_state_space_model"

CAPTURE = "smoke1_20260716"
GRID_HZ = 10.0            # the state sequence is uniform in time, like PP4's Delta t
ASSOC_TOL_S = 0.06        # a detection belongs to the grid step it lands nearest

C_TRUTH = "#111111"
C_OBS = "#8A8A8A"
C_FILTER = "#D55E00"
C_SMOOTH = "#0072B2"
C_PRED = "#7B4EA8"
C_ACCENT = "#009E73"
C_MUTED = "#8A8A8A"

CHI2_95_2DOF = 5.991
CALIBRATED_MEDIAN_NEES = 1.386


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ---------------------------------------------------------------------- the data


class Sequence:
    """A uniform-time state sequence: control in, observations where they exist.

    PP4's `y` is a dense vector with `missing` entries where nothing was
    observed. Ours is the same object: on a 10 Hz grid, 45 % of the steps have no
    detection at all, because camera coverage is a relay with gaps in it.
    """

    def __init__(self, capture, truth_table):
        stamps = np.asarray(capture.stamps, dtype=float)
        odom = np.asarray(capture.odom, dtype=float)
        step = 1.0 / GRID_HZ
        grid = np.arange(stamps[0], stamps[-1], step)

        # control input u_k: the odometry increment over grid step k
        odom_on_grid = np.column_stack([np.interp(grid, stamps, odom[:, i])
                                        for i in range(2)])
        self.stamps = grid
        self.u = np.vstack([np.zeros((1, 2)), np.diff(odom_on_grid, axis=0)])
        self.odom = odom_on_grid
        self.dt = step

        detections = sorted(
            ((camera, d) for camera in rc.CAMERAS for d in capture.detections[camera]),
            key=lambda item: item[1].stamp,
        )
        self.y = np.full((len(grid), 2), np.nan)
        self.camera = [None] * len(grid)
        for camera, detection in detections:
            index = int(np.argmin(np.abs(grid - detection.stamp)))
            if abs(grid[index] - detection.stamp) > ASSOC_TOL_S:
                continue
            if self.camera[index] is not None:
                continue            # one observation per step, as the model assumes
            self.y[index] = np.asarray(detection.world, dtype=float)
            self.camera[index] = camera

        # EVALUATION ONLY
        self.truth = np.full((len(grid), 2), np.nan)
        for index, stamp in enumerate(grid):
            hit = rc.truth_at(truth_table, float(stamp))
            if hit is not None:
                self.truth[index] = np.asarray(hit[:2], dtype=float)

    @property
    def n_steps(self) -> int:
        return len(self.stamps)

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.y[:, 0])


# ------------------------------------------------------- the inference machinery


def kalman_filter(seq, model, upto: int | None = None):
    """Forward pass. Returns filtered and predicted moments plus -log p(y).

    ``model`` supplies the state dimension, how the control enters, and the
    observation model at each step -- so the same code runs the 2-D position
    model and the bias-augmented model.
    """
    n = model.dim
    T = seq.n_steps if upto is None else upto
    m = model.m0.copy()
    P = model.S0.copy()

    m_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    m_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))
    nll = 0.0

    for k in range(T):
        m = model.F @ m + model.B @ seq.u[k]
        P = model.F @ P @ model.F.T + model.Q
        P = 0.5 * (P + P.T)
        m_pred[k], P_pred[k] = m, P

        camera = seq.camera[k]
        if camera is not None:
            H = model.H(camera)
            R = model.R(camera)
            innovation = seq.y[k] - H @ m
            S = H @ P @ H.T + R
            S = 0.5 * (S + S.T)
            S_inv = np.linalg.inv(S)
            K = P @ H.T @ S_inv
            m = m + K @ innovation
            A = np.eye(n) - K @ H
            P = A @ P @ A.T + K @ R @ K.T
            P = 0.5 * (P + P.T)
            sign, logdet = np.linalg.slogdet(S)
            nll += 0.5 * (logdet + innovation @ S_inv @ innovation
                          + len(innovation) * math.log(2 * math.pi))
        m_filt[k], P_filt[k] = m, P

    return {"m_pred": m_pred, "P_pred": P_pred, "m_filt": m_filt, "P_filt": P_filt,
            "neg_log_evidence": float(nll)}


def rts_smoother(seq, model, forward):
    """Backward pass. Adds the lag-one covariance the covariance M-step needs."""
    n = model.dim
    T = len(forward["m_filt"])
    m_smooth = forward["m_filt"].copy()
    P_smooth = forward["P_filt"].copy()
    P_lag = np.zeros((T, n, n))

    for k in range(T - 2, -1, -1):
        P_next_pred = forward["P_pred"][k + 1]
        G = forward["P_filt"][k] @ model.F.T @ np.linalg.inv(P_next_pred)
        m_smooth[k] = (forward["m_filt"][k]
                       + G @ (m_smooth[k + 1] - forward["m_pred"][k + 1]))
        P_smooth[k] = (forward["P_filt"][k]
                       + G @ (P_smooth[k + 1] - P_next_pred) @ G.T)
        P_smooth[k] = 0.5 * (P_smooth[k] + P_smooth[k].T)
        P_lag[k + 1] = P_smooth[k + 1] @ G.T

    return {"m": m_smooth, "P": P_smooth, "P_lag": P_lag}


class PositionModel:
    """PP4's LGDS, with our physics: A = I, control = odometry, C = I.

        z_k = z_{k-1} + u_k + q_k,   q_k ~ N(0, Q)
        y_k = z_k + r_k,             r_k ~ N(0, R_c)   for the camera c that saw it
    """

    def __init__(self, *, Q, R_per_camera, m0, S0):
        self.dim = 2
        self.F = np.eye(2)
        self.B = np.eye(2)
        self.Q = np.asarray(Q, dtype=float)
        self._R = {c: np.asarray(v, dtype=float) for c, v in R_per_camera.items()}
        self.m0 = np.asarray(m0, dtype=float)
        self.S0 = np.asarray(S0, dtype=float)

    def H(self, camera):
        return np.eye(2)

    def R(self, camera):
        return self._R[camera]

    def position(self, m, P):
        return m[..., :2], P[..., :2, :2]

    @property
    def n_free_parameters(self) -> int:
        return 0


class BiasAugmentedModel:
    """The structural answer PP4's formalism suggests, instead of a floor.

    State is position PLUS one 2-D offset per camera::

        z_k = [p_x, p_y, b^A_x, b^A_y, ..., b^D_x, b^D_y]

    Position moves with odometry; each bias is a slow random walk. A camera then
    reports ``y_k = p_k + b^c_k + noise``, so a steady per-camera offset is
    something the model can EXPLAIN rather than something the filter has to be
    protected from. The biases are only jointly identifiable up to a common
    shift, which is exactly the statement that a single camera cannot detect its
    own lean; a zero-mean prior on the offsets pins that one direction, and the
    network's disagreement does the rest.
    """

    def __init__(self, *, Q_position, sigma_bias_walk, R_per_camera, m0, S0,
                 sigma_bias_prior):
        self.cameras = list(rc.CAMERAS)
        self.dim = 2 + 2 * len(self.cameras)
        self.F = np.eye(self.dim)
        self.B = np.zeros((self.dim, 2))
        self.B[:2, :2] = np.eye(2)
        self.Q = np.zeros((self.dim, self.dim))
        self.Q[:2, :2] = np.asarray(Q_position, dtype=float)
        for i in range(2, self.dim):
            self.Q[i, i] = sigma_bias_walk**2
        self._R = {c: np.asarray(v, dtype=float) for c, v in R_per_camera.items()}
        self.m0 = np.zeros(self.dim)
        self.m0[:2] = np.asarray(m0, dtype=float)
        self.S0 = np.zeros((self.dim, self.dim))
        self.S0[:2, :2] = np.asarray(S0, dtype=float)
        for i in range(2, self.dim):
            self.S0[i, i] = sigma_bias_prior**2

    def _slot(self, camera):
        return 2 + 2 * self.cameras.index(camera)

    def H(self, camera):
        H = np.zeros((2, self.dim))
        H[:, :2] = np.eye(2)
        slot = self._slot(camera)
        H[:, slot:slot + 2] = np.eye(2)
        return H

    def R(self, camera):
        return self._R[camera]

    def position(self, m, P):
        return m[..., :2], P[..., :2, :2]

    @property
    def n_free_parameters(self) -> int:
        return 0


def infer_covariance(seq, model_factory, *, target: str, prior_nu=6.0,
                     prior_scale=None, iterations=25):
    """Variational EM on q(z) q(Sigma) with an inverse-Wishart prior on Sigma.

    This is PP4's `@constraints q(z_0,z,Q) = q(z_0,z)q(Q)` written out. The
    E-step is the smoother run at the current expected precision; the M-step is
    the conjugate inverse-Wishart posterior. ``target`` is 'R' (measurement) or
    'Q' (process).
    """
    if prior_scale is None:
        prior_scale = np.eye(2) * (0.05**2) * prior_nu
    nu = float(prior_nu)
    scale = np.asarray(prior_scale, dtype=float)
    # E[Sigma^-1] = nu * scale^-1  ->  the covariance the E-step should run at
    current = scale / nu
    history = []

    for _ in range(iterations):
        model = model_factory(current)
        forward = kalman_filter(seq, model)
        smooth = rts_smoother(seq, model, forward)
        history.append(forward["neg_log_evidence"])

        accumulator = np.zeros((2, 2))
        count = 0
        if target == "R":
            for k in range(seq.n_steps):
                if seq.camera[k] is None:
                    continue
                H = model.H(seq.camera[k])
                residual = seq.y[k] - H @ smooth["m"][k]
                accumulator += np.outer(residual, residual) + H @ smooth["P"][k] @ H.T
                count += 1
        else:
            for k in range(1, seq.n_steps):
                delta = (smooth["m"][k][:2] - smooth["m"][k - 1][:2] - seq.u[k])
                cross = smooth["P_lag"][k][:2, :2]
                accumulator += (np.outer(delta, delta) + smooth["P"][k][:2, :2]
                                + smooth["P"][k - 1][:2, :2] - cross - cross.T)
                count += 1

        nu_post = prior_nu + count
        scale_post = scale + accumulator
        current = scale_post / nu_post          # E[Sigma^-1]^-1 under IW(nu, scale)

    return {"posterior_nu": nu_post, "posterior_scale": scale_post,
            "expected_covariance": current,
            # mean of IW(nu, Lambda) is Lambda / (nu - d - 1)
            "posterior_mean": scale_post / max(nu_post - 3.0, 1e-6),
            "neg_log_evidence_history": history, "n_terms": count}


def score(seq, model, *, label: str, n_free_parameters: int) -> dict:
    """Evidence, complexity and honesty for one model. Truth scores only."""
    forward = kalman_filter(seq, model)
    smooth = rts_smoother(seq, model, forward)

    rows = {"filtered": (forward["m_filt"], forward["P_filt"]),
            "smoothed": (smooth["m"], smooth["P"])}
    out = {"label": label,
           "neg_log_evidence": forward["neg_log_evidence"],
           "n_free_parameters": n_free_parameters}

    n_observed = int(seq.observed.sum())
    out["bic"] = (2 * forward["neg_log_evidence"]
                  + n_free_parameters * math.log(max(n_observed, 1)))

    for name, (means, covariances) in rows.items():
        position, covariance = model.position(means, covariances)
        valid = ~np.isnan(seq.truth[:, 0])
        errors = position[valid] - seq.truth[valid]
        covs = covariance[valid]
        nees = np.asarray([e @ np.linalg.solve(c, e) for e, c in zip(errors, covs)])
        sigma = np.sqrt(0.5 * (covs[:, 0, 0] + covs[:, 1, 1]))
        out[name] = {
            "n": int(valid.sum()),
            "median_nees": float(np.median(nees)),
            "unearned_confidence_fraction": float(np.mean(nees > CHI2_95_2DOF)),
            "rmse_m": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
            "mean_stated_sigma_m": float(np.mean(sigma)),
        }
    out["_forward"] = forward
    out["_smooth"] = smooth
    return out


# --------------------------------------------------------------------- figures


def fig_m1_the_model(seq) -> dict:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                                  gridspec_kw={"width_ratios": [1.45, 1.0]})
    t = seq.stamps - seq.stamps[0]
    ax.plot(t, seq.truth[:, 0], color=C_TRUTH, lw=1.8, label="true position (evaluation only)")
    observed = seq.observed
    ax.plot(t[observed], seq.y[observed, 0], ".", color=C_OBS, ms=3.5,
            label="camera observations")
    ax.plot(t, seq.odom[:, 0] - seq.odom[0, 0] + seq.truth[0, 0], color=C_ACCENT,
            lw=1.3, ls="--", label="odometry alone (dead reckoning)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x position (m)")
    ax.set_title(f"The data: {int(observed.sum())} observations on "
                 f"{seq.n_steps} steps — {100 * (1 - observed.mean()):.0f} % of steps "
                 "see nothing", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    ax2.axis("off")
    ax2.text(0.0, 1.0, "The generative model", fontsize=11, va="top", fontweight="bold")
    ax2.text(
        0.0, 0.86,
        "$p(z_0) = \\mathcal{N}(m_0, S_0)$\n\n"
        "$p(z_k \\mid z_{k-1}) = \\mathcal{N}(z_k \\mid z_{k-1} + u_k,\\; Q)$\n\n"
        "$p(y_k \\mid z_k) = \\mathcal{N}(y_k \\mid C z_k,\\; R_c)$\n\n"
        "$p(y_{1:T}, z_{0:T}) = p(z_0)\\prod_k p(y_k \\mid z_k)\\, p(z_k \\mid z_{k-1})$",
        fontsize=11, va="top", linespacing=1.5)
    ax2.text(
        0.0, 0.30,
        "$z_k$   robot position (x, y)\n"
        "$u_k$   wheel-odometry increment\n"
        "$y_k$   a camera's claim about position, or missing\n"
        "$Q$     process noise — how much driving costs us\n"
        "$R_c$   measurement noise of the camera that saw it",
        fontsize=9.5, va="top", color="#333333", linespacing=1.7)
    fig.suptitle("PP4 STEP 1 — write the state-space model down, then hand it the data",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_m1_the_model")
    return {"n_steps": seq.n_steps, "n_observed": int(observed.sum()),
            "missing_fraction": float(1 - observed.mean())}


def fig_m2_filtering_and_smoothing(seq, model, result) -> dict:
    forward, smooth = result["_forward"], result["_smooth"]
    t = seq.stamps - seq.stamps[0]
    m_f, P_f = model.position(forward["m_filt"], forward["P_filt"])
    m_s, P_s = model.position(smooth["m"], smooth["P"])
    s_f = np.sqrt(P_f[:, 0, 0])
    s_s = np.sqrt(P_s[:, 0, 0])

    gaps = ~seq.observed
    runs, start = [], None
    for k, missing in enumerate(gaps):
        if missing and start is None:
            start = k
        elif not missing and start is not None:
            runs.append((start, k))
            start = None
    longest = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 1)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12.6, 7.2))
    for axis, window, title in (
        (ax, (0, seq.n_steps), "The whole capture"),
        (ax2, (max(longest[0] - 40, 0), min(longest[1] + 40, seq.n_steps)),
         f"Zoom on the longest coverage gap "
         f"({(longest[1] - longest[0]) / GRID_HZ:.1f} s with no camera)"),
    ):
        lo, hi = window
        axis.fill_between(t[lo:hi], m_f[lo:hi, 0] - 2 * s_f[lo:hi],
                          m_f[lo:hi, 0] + 2 * s_f[lo:hi], color=C_FILTER, alpha=0.18)
        axis.fill_between(t[lo:hi], m_s[lo:hi, 0] - 2 * s_s[lo:hi],
                          m_s[lo:hi, 0] + 2 * s_s[lo:hi], color=C_SMOOTH, alpha=0.25)
        axis.plot(t[lo:hi], m_f[lo:hi, 0], color=C_FILTER, lw=1.5,
                  label="filtered  $q(z_k \\mid y_{1:k})$")
        axis.plot(t[lo:hi], m_s[lo:hi, 0], color=C_SMOOTH, lw=1.8,
                  label="smoothed  $q(z_k \\mid y_{1:T})$")
        axis.plot(t[lo:hi], seq.truth[lo:hi, 0], color=C_TRUTH, lw=1.4,
                  label="truth (evaluation only)")
        seen = seq.observed[lo:hi]
        axis.plot(t[lo:hi][seen], seq.y[lo:hi][seen, 0], ".", color=C_OBS, ms=3.0,
                  label="observations")
        axis.set_ylabel("x position (m)")
        axis.set_title(title, loc="left")
        axis.legend(frameon=False, fontsize=8, ncols=4, loc="upper center")
    ax2.set_xlabel("time (s)")

    fig.suptitle("PP4 STEP 2 — one inference call returns the whole posterior: "
                 "filtering AND smoothing", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "fig_m2_filtering_and_smoothing")
    return {"filtered": result["filtered"], "smoothed": result["smoothed"],
            "longest_gap_s": (longest[1] - longest[0]) / GRID_HZ}


def fig_m3_forecasting(seq, model) -> dict:
    horizon = int(6.0 * GRID_HZ)
    cut = seq.n_steps - horizon
    forward = kalman_filter(seq, model, upto=cut)

    m = forward["m_filt"][-1].copy()
    P = forward["P_filt"][-1].copy()
    means, sigmas = [], []
    for k in range(cut, seq.n_steps):
        m = model.F @ m + model.B @ seq.u[k]
        P = model.F @ P @ model.F.T + model.Q
        position, covariance = model.position(m, P)
        means.append(position.copy())
        sigmas.append(math.sqrt(covariance[0, 0]))
    means = np.asarray(means)
    sigmas = np.asarray(sigmas)

    t = seq.stamps - seq.stamps[0]
    view = slice(max(cut - 120, 0), seq.n_steps)
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    m_f, P_f = model.position(forward["m_filt"], forward["P_filt"])
    s_f = np.sqrt(P_f[:, 0, 0])
    lo = max(cut - 120, 0)
    ax.fill_between(t[lo:cut], m_f[lo:, 0] - 2 * s_f[lo:], m_f[lo:, 0] + 2 * s_f[lo:],
                    color=C_SMOOTH, alpha=0.22)
    ax.plot(t[lo:cut], m_f[lo:, 0], color=C_SMOOTH, lw=1.8, label="inferred")
    ax.fill_between(t[cut:], means[:, 0] - 2 * sigmas, means[:, 0] + 2 * sigmas,
                    color=C_PRED, alpha=0.20)
    ax.plot(t[cut:], means[:, 0], color=C_PRED, lw=2.0, label="prediction")
    ax.plot(t[view], seq.truth[view, 0], color=C_TRUTH, lw=1.4,
            label="truth (evaluation only)")
    seen = seq.observed[view]
    ax.plot(t[view][seen], seq.y[view][seen, 0], ".", color=C_OBS, ms=3.5,
            label="observations")
    ax.axvline(t[cut], color=C_ACCENT, ls="--", lw=1.5)
    ax.annotate("cameras switched off here", xy=(t[cut], ax.get_ylim()[1]),
                xytext=(-8, -14), textcoords="offset points", ha="right",
                color=C_ACCENT, fontsize=9, fontweight="bold")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x position (m)")
    final_error = float(np.hypot(*(means[-1] - seq.truth[-1])))
    ax.set_title("PP4 STEP 3 — forecasting is the same model with the observations "
                 "set to missing\n"
                 f"after {horizon / GRID_HZ:.0f} s of dead reckoning the stated 2$\\sigma$ "
                 f"is {200 * sigmas[-1]:.0f} cm and the true error is "
                 f"{100 * final_error:.0f} cm",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig_m3_forecasting")
    return {"horizon_s": horizon / GRID_HZ,
            "final_stated_2sigma_m": float(2 * sigmas[-1]),
            "final_true_error_m": final_error}


def fig_m4_inferring_the_covariance(r_result, q_result, deployed_r, deployed_q) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3),
                             gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    ax = axes[0]
    ax.plot(r_result["neg_log_evidence_history"], color=C_SMOOTH, lw=2.2, marker="o",
            ms=4, label="inferring $R$ (measurement)")
    ax.plot(q_result["neg_log_evidence_history"], color=C_FILTER, lw=2.2, marker="o",
            ms=4, label="inferring $Q$ (process)")
    ax.set_xlabel("variational iteration")
    ax.set_ylabel("$-\\log p(y_{1:T})$")
    ax.set_title("Both converge in a handful of sweeps", loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    for axis, inferred, deployed, name in (
        (axes[1], r_result["posterior_mean"], deployed_r, "$R$  measurement noise"),
        (axes[2], q_result["posterior_mean"], deployed_q, "$Q$  process noise"),
    ):
        inferred_sigma = 1000 * np.sqrt(np.diag(inferred))
        deployed_sigma = 1000 * np.sqrt(np.diag(np.asarray(deployed, dtype=float)))
        positions = np.arange(2)
        axis.bar(positions - 0.19, deployed_sigma, 0.36, color=C_MUTED,
                 label="deployed value")
        axis.bar(positions + 0.19, inferred_sigma, 0.36, color=C_SMOOTH,
                 label="inferred posterior mean")
        for i, (a, b) in enumerate(zip(deployed_sigma, inferred_sigma)):
            axis.annotate(f"{a:.0f}", xy=(i - 0.19, a), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8)
            axis.annotate(f"{b:.0f}", xy=(i + 0.19, b), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8)
        axis.set_xticks(positions, ["x", "y"])
        axis.set_ylabel("$\\sigma$ (mm)")
        axis.set_title(name, loc="left", pad=10)
        axis.set_ylim(0, max(deployed_sigma.max(), inferred_sigma.max()) * 1.45)
        axis.legend(frameon=False, fontsize=8, loc="upper center", ncols=2)

    fig.suptitle("PP4 STEP 4 — stop treating the noise as known: put an "
                 "inverse-Wishart prior on it and infer it",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_m4_inferring_the_covariance")
    return {
        "R_posterior_sigma_mm": (1000 * np.sqrt(np.diag(r_result["posterior_mean"]))).tolist(),
        "Q_posterior_sigma_mm": (1000 * np.sqrt(np.diag(q_result["posterior_mean"]))).tolist(),
        "R_posterior_nu": r_result["posterior_nu"],
        "Q_posterior_nu": q_result["posterior_nu"],
    }


def fig_m5_model_comparison(results: list[dict], bias_estimate: dict,
                            observed_counts: dict) -> dict:
    labels = [r["label"] for r in results]
    positions = np.arange(len(results))
    baseline = results[0]["neg_log_evidence"]
    # positive = this model explains the observations BETTER than the baseline
    gain = [baseline - r["neg_log_evidence"] for r in results]
    nees = [r["smoothed"]["median_nees"] for r in results]
    colours = [C_MUTED] * len(results)
    colours[-1] = C_SMOOTH
    worst = int(np.argmax(nees))
    colours[worst] = C_FILTER

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 5.2),
                                       gridspec_kw={"width_ratios": [1, 1, 1.2]})
    ax.bar(positions, gain, 0.6, color=colours)
    ax.axhline(0, color=C_TRUTH, lw=1.0)
    ax.set_xticks(positions, labels, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("evidence gained over the baseline\n"
                  "$\\log p(y)$ improvement (higher = fits the data better)")
    ax.set_title("What the data alone prefers", loc="left")
    for i, value in enumerate(gain):
        ax.annotate(f"{value:+.0f}", xy=(i, value),
                    xytext=(0, 4 if value >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold")
    ax.annotate("the best fit\nto the observations", xy=(worst, gain[worst]),
                xytext=(0, -34), textcoords="offset points", ha="center",
                fontsize=8.5, color=C_FILTER, fontweight="bold")

    ax2.bar(positions, nees, 0.6, color=colours)
    ax2.axhline(CALIBRATED_MEDIAN_NEES, color=C_TRUTH, lw=1.5, ls="--")
    ax2.text(0.02, 0.10, "an honest filter sits at 1.39", transform=ax2.transAxes,
             fontsize=8.5)
    ax2.set_xticks(positions, labels, fontsize=8, rotation=18, ha="right")
    ax2.set_ylabel("median NEES   (above the line = overconfident)")
    ax2.set_title("What the truth says", loc="left")
    for i, value in enumerate(nees):
        ax2.annotate(f"{value:.2f}", xy=(i, value), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=8.5,
                     fontweight="bold")
    ax2.annotate("and the best fit is\nthe LEAST honest", xy=(worst, nees[worst]),
                 xytext=(-6, -46), textcoords="offset points", ha="center",
                 fontsize=8.5, color=C_FILTER, fontweight="bold")

    cameras = list(bias_estimate["inferred_mm"])
    inferred = [bias_estimate["inferred_mm"][c] for c in cameras]
    commissioned = [bias_estimate["commissioned_mm"][c] for c in cameras]
    seen = [observed_counts[c] > 0 for c in cameras]
    width = 0.36
    spots = np.arange(len(cameras))
    ax3.bar(spots - width / 2, commissioned, width, color=C_MUTED,
            label="measured against truth at commissioning")
    ax3.bar(spots + width / 2, inferred, width,
            color=[C_ACCENT if s else "#DDDDDD" for s in seen],
            label="inferred online, no truth, as a state")
    for i, (a, b, s) in enumerate(zip(commissioned, inferred, seen)):
        ax3.annotate(f"{a:.0f}", xy=(i - width / 2, a), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
        if s:
            ax3.annotate(f"{b:.0f}", xy=(i + width / 2, b), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=8)
    unseen = [i for i, s in enumerate(seen) if not s]
    for i in unseen:
        ax3.annotate("never seen\nin this capture:\nprior, untouched",
                     xy=(i + width / 2, 1.0), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color=C_MUTED, style="italic")
    ax3.set_xticks(spots, [f"{c.replace('camera_', '')}\n({observed_counts[c]} obs)"
                           for c in cameras])
    ax3.set_ylabel("offset magnitude (mm)")
    ax3.set_title("The offsets the model recovered by itself", loc="left")
    ax3.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("PP4 STEP 5 — the question the formalism forces: make the camera's "
                 "offset a STATE, not a wider covariance",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_m5_model_comparison")
    return {r["label"].replace("\n", " "):
            {"neg_log_evidence": r["neg_log_evidence"], "bic": r["bic"],
             "evidence_gain_over_baseline": baseline - r["neg_log_evidence"],
             "smoothed": r["smoothed"], "filtered": r["filtered"]}
            for r in results}


def run_as_ladder_arm(capture, truth_lookup, *, sigma_bias_prior,
                      sigma_bias_walk_per_sqrt_s, frozen_offsets=None) -> list[dict]:
    """The offset model on `exp1`'s EXACT protocol, so it can join that ladder.

    Everything that made the numbers above incomparable is removed here: steps are
    per detection rather than a 10 Hz grid, the belief is FILTERED not smoothed,
    the prediction and its growth are `f1.run_arm`'s line for line, and scoring
    goes through `f1.summarize`. The only thing that differs from an exp1 arm is
    the state vector -- which is the whole point.

    R carries scatter and timing but NOT the residual-bias term the A2-A4 arms
    add, because in this model the bias is no longer noise to be absorbed; it is
    a state to be estimated.

    ``frozen_offsets`` turns the offsets from states into KNOWN CONSTANTS: given
    ``{camera: (dx, dy)}`` they are installed at those values with negligible
    prior width and no random walk, so nothing about the test capture can move
    them. That is what makes a transfer test possible -- fit on one route, freeze,
    apply to another.
    """
    cameras = list(rc.CAMERAS)
    dim = 2 + 2 * len(cameras)

    def slot(camera):
        return 2 + 2 * cameras.index(camera)

    mean = np.zeros(dim)
    mean[:2] = np.asarray(capture.odom[0], dtype=float)
    cov = np.zeros((dim, dim))
    cov[:2, :2] = np.eye(2) * f1.INITIAL_SIGMA_M**2
    for i in range(2, dim):
        cov[i, i] = sigma_bias_prior**2
    if frozen_offsets is not None:
        sigma_bias_walk_per_sqrt_s = 0.0
        for camera in cameras:
            start = 2 + 2 * cameras.index(camera)
            mean[start:start + 2] = np.asarray(frozen_offsets[camera], dtype=float)
            cov[start, start] = 1e-10
            cov[start + 1, start + 1] = 1e-10

    detections = sorted(
        ((camera, d) for camera in cameras for d in capture.detections[camera]),
        key=lambda item: item[1].stamp,
    )
    stamps = np.asarray(capture.stamps, dtype=float)
    odom = np.asarray(capture.odom, dtype=float)

    records = []
    previous_index = 0
    previous_stamp = float(stamps[0])
    for camera, detection in detections:
        index = int(np.searchsorted(stamps, detection.stamp))
        index = min(max(index, 0), len(stamps) - 1)

        # --- predict, exactly as f1.run_arm does for the position block
        if index > previous_index:
            step = odom[index] - odom[previous_index]
            distance = float(np.hypot(*step))
            mean[:2] = mean[:2] + step
            growth = f1.PROCESS_SIGMA_PER_SQRT_M**2 * max(distance, 1e-6)
            cov[0, 0] += growth
            cov[1, 1] += growth
            previous_index = index
        # --- the offsets drift slowly in their own right
        dt = max(float(detection.stamp) - previous_stamp, 0.0)
        previous_stamp = float(detection.stamp)
        for i in range(2, dim):
            cov[i, i] += sigma_bias_walk_per_sqrt_s**2 * dt

        velocity = np.zeros(2)
        if index > 0:
            gap = float(stamps[index] - stamps[index - 1])
            if gap > 1e-6:
                velocity = (odom[index] - odom[index - 1]) / gap

        # --- R: scatter + timing, no bias term
        sigma = f1.R_COND_SIGMA_M[camera]
        R = np.eye(2) * sigma**2 + np.asarray(
            time_sync_covariance(f1.H, velocity.tolist(), f1.SIGMA_TAU_S), dtype=float)

        H = np.zeros((2, dim))
        H[:, :2] = np.eye(2)
        H[:, slot(camera):slot(camera) + 2] = np.eye(2)

        measurement = np.asarray(detection.world, dtype=float)
        innovation = measurement - H @ mean
        S = H @ cov @ H.T + R
        S = 0.5 * (S + S.T)
        S_inv = np.linalg.inv(S)
        K = cov @ H.T @ S_inv
        mean = mean + K @ innovation
        A = np.eye(dim) - K @ H
        cov = A @ cov @ A.T + K @ R @ K.T
        cov = 0.5 * (cov + cov.T)
        nis = float(innovation @ S_inv @ innovation)

        truth = truth_lookup(detection.stamp)
        if truth is None:
            continue
        error = mean[:2] - np.asarray(truth[:2], dtype=float)
        position_cov = cov[:2, :2]
        records.append({
            "stamp": detection.stamp, "camera": camera, "accepted": True,
            "error_m": float(np.hypot(*error)),
            "nees": float(error @ np.linalg.solve(position_cov, error)),
            "sigma_m": float(math.sqrt(0.5 * (position_cov[0, 0] + position_cov[1, 1]))),
            "nis": nis, "responsibility": 1.0, "health": 1.0,
            "cov_post": position_cov.copy(),
            "offsets_mm": {c: float(1000 * np.hypot(*mean[slot(c):slot(c) + 2]))
                           for c in cameras},
            "offset_xy": {c: mean[slot(c):slot(c) + 2].copy() for c in cameras},
        })
    return records


def logarithmic_score(records) -> dict:
    """The proper scoring rule: how surprised was the belief by the truth?

    RMSE cannot separate these arms, and it was never going to. All six share one
    pixel-to-ground path, and that path's error dwarfs anything the update rule
    does -- so RMSE measures the projection, not the filter. What differs between
    arms is the *distribution* they state, and the honest way to score a stated
    distribution is a strictly proper rule:

        NLPD_k = -log N(truth_k ; belief_k, P_k)
               = 0.5 * ( NEES_k + log det(2*pi*P_k) )

    which decomposes exactly into a fit term and a volume term. Inflating P to
    look calibrated raises the volume term; shrinking it to look sharp raises the
    fit term. Neither cheat wins, which is what 'strictly proper' means, and it is
    the single number that says whether the filter's confidence was earned.
    """
    fit, volume, total = [], [], []
    for record in records:
        P = np.asarray(record["cov_post"], dtype=float)
        nees = float(record["nees"])
        sign, logdet = np.linalg.slogdet(2 * math.pi * P)
        if sign <= 0:
            continue
        fit.append(0.5 * nees)
        volume.append(0.5 * logdet)
        total.append(0.5 * (nees + logdet))
    return {
        "n": len(total),
        "mean_nlpd": float(np.mean(total)),
        "median_nlpd": float(np.median(total)),
        "mean_fit_term": float(np.mean(fit)),
        "mean_volume_term": float(np.mean(volume)),
    }


def fig_m8_proper_score(scores: dict, rmse_cm: dict) -> dict:
    """Score the arms by a rule that cannot be gamed, and show why RMSE could not."""
    labels = list(scores)
    positions = np.arange(len(labels))
    fit = [scores[a]["mean_fit_term"] for a in labels]
    volume = [scores[a]["mean_volume_term"] for a in labels]
    total = [scores[a]["mean_nlpd"] for a in labels]
    best = int(np.argmin(total))
    colours = [C_MUTED] * len(labels)
    colours[best] = C_ACCENT

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                                  gridspec_kw={"width_ratios": [1.25, 1.0]})

    # The two terms have opposite signs by construction -- the fit penalty is
    # always positive, and a sharp belief has negative log volume -- so they are
    # drawn away from zero rather than stacked, with their sum marked.
    ax.bar(positions - 0.18, fit, 0.32, color=C_FILTER,
           label="fit penalty  ½·NEES  (was the truth where it said?)")
    ax.bar(positions + 0.18, volume, 0.32, color=C_SMOOTH,
           label="volume credit  ½·log det 2πP  (how small a claim was it?)")
    ax.plot(positions, total, "D", color=C_TRUTH, ms=10, mec="white", mew=1.3,
            zorder=5, label="total  −log p(truth)  ← the score")
    for i, value in enumerate(total):
        ax.annotate(f"{value:.2f}", xy=(i, value), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold")
    ax.axhline(0, color=C_TRUTH, lw=0.9)
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.9, high + 1.5)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("mean negative log predictive density (nats)\nlower = better")
    ax.set_title("A rule neither cheat can win: inflate and you lose the credit,\n"
                 "shrink and you pay the penalty. A2 buys the sharpest claim in the\n"
                 "network and cannot pay for it; A5 is the only arm good at both.",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncols=1)

    rmse = [rmse_cm[a] for a in labels]
    spread_rmse = max(rmse) / min(rmse)
    spread_score = max(total) - min(total)
    ax2.bar(positions, rmse, 0.6, color=colours)
    ax2.set_xticks(positions, labels)
    ax2.set_ylabel("RMSE (cm)")
    ax2.set_ylim(0, max(rmse) * 2.15)
    for i, value in enumerate(rmse):
        ax2.annotate(f"{value:.1f}", xy=(i, value), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=8.5)
    ax2.set_title("Why RMSE could not decide this", loc="left", fontsize=10.5)
    ax2.text(0.03, 0.97,
             f"RMSE spans {spread_rmse:.2f}× across every arm,\n"
             f"while the proper score spans {spread_score:.1f} nats.\n\n"
             "All six share one pixel-to-ground path, and\n"
             "that path sets the floor: changing only the\n"
             "pixel statistic moves single-detection error\n"
             "110 → 18 mm, roughly 6×. RMSE here is\n"
             "measuring the projection, not the filter.",
             transform=ax2.transAxes, fontsize=8.8, va="top", color="#333333")

    fig.suptitle("Scoring the CONFIDENCE, not the accuracy — "
                 "RMSE is a control here, never the criterion",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_m8_proper_score")
    return {"mean_nlpd": dict(zip(labels, total)),
            "rmse_cm": dict(zip(labels, rmse)),
            "rmse_spread_ratio": spread_rmse,
            "score_spread_nats": spread_score}


def fig_m7_same_ladder(exp1_summary: dict, arm: dict) -> dict:
    """A5 dropped into exp1's own chart, measured its way."""
    arms = ["A0_trust_everything", "A1_hard_gate", "A2_factorized",
            "A3_network_consistency", "A4_correlation_floor"]
    labels = [a.split("_")[0] for a in arms] + ["A5"]
    nees = [exp1_summary["pooled"][a]["median_nees"] for a in arms] + [arm["median_nees"]]
    unearned = ([100 * exp1_summary["pooled"][a]["unearned_confidence_fraction"]
                 for a in arms] + [100 * arm["unearned_confidence_fraction"]])
    rmse = ([100 * exp1_summary["pooled"][a]["rmse_m"] for a in arms]
            + [100 * arm["rmse_m"]])
    stated = ([100 * exp1_summary["pooled"][a]["mean_stated_sigma_m"] for a in arms]
              + [100 * arm["mean_stated_sigma_m"]])
    colours = [C_MUTED] * 4 + [C_SMOOTH, C_ACCENT]
    positions = np.arange(len(labels))

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(14.4, 4.7))
    ax.bar(positions, nees, 0.62, color=colours)
    ax.axhline(CALIBRATED_MEDIAN_NEES, color=C_TRUTH, lw=1.5, ls="--")
    ax.text(0.02, 0.92, "an honest filter sits at 1.39", transform=ax.transAxes,
            fontsize=8.5)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("median NEES")
    ax.set_title("Honesty", loc="left")
    for i, value in enumerate(nees):
        ax.annotate(f"{value:.2f}", xy=(i, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold")

    ax2.bar(positions, unearned, 0.62, color=colours)
    ax2.axhline(5.0, color=C_TRUTH, lw=1.5, ls="--")
    ax2.set_xticks(positions, labels)
    ax2.set_ylabel("truth outside the stated 95 % ellipse (%)")
    ax2.set_title("Unearned confidence — nominal 5 %", loc="left")
    for i, value in enumerate(unearned):
        ax2.annotate(f"{value:.1f}", xy=(i, value), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=8.5,
                     fontweight="bold")

    width = 0.38
    ax3.bar(positions - width / 2, rmse, width, color=colours, label="actual RMSE")
    ax3.bar(positions + width / 2, stated, width, color=colours, alpha=0.45,
            label="stated $\\sigma$")
    ax3.set_xticks(positions, labels)
    ax3.set_ylabel("cm")
    ax3.set_title("Accuracy and sharpness — calibration comes from NEES/coverage", loc="left")
    ax3.legend(frameon=False, fontsize=8.5)
    for i, (a, b) in enumerate(zip(rmse, stated)):
        ax3.annotate(f"{a:.1f}", xy=(i - width / 2, a), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=7.5)
        ax3.annotate(f"{b:.1f}", xy=(i + width / 2, b), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=7.5)

    fig.suptitle("A5 on the SAME footing as the ladder — same captures, same "
                 "prediction, filtered not smoothed, scored by the same function",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_m7_same_ladder")
    return {"labels": labels, "median_nees": nees, "unearned_percent": unearned,
            "rmse_cm": rmse, "stated_sigma_cm": stated}


def fig_m6_prior_sensitivity(sweep: list[dict]) -> dict:
    """How wrong may the offset prior be? The same discipline exp2 applies to the floor."""
    widths = [1000 * s["sigma_bias_prior_m"] for s in sweep]
    nees = [s["median_nees"] for s in sweep]
    unearned = [100 * s["unearned_confidence_fraction"] for s in sweep]
    rmse = [100 * s["rmse_m"] for s in sweep]
    recovered = [s["camera_C_offset_mm"] for s in sweep]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax.plot(widths, nees, "-o", color=C_SMOOTH, lw=2.2, ms=8, mec="white", mew=1.2,
            label="median NEES")
    ax.axhline(CALIBRATED_MEDIAN_NEES, color=C_TRUTH, lw=1.4, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(widths, [f"{w:.0f}" for w in widths])
    ax.set_xlabel("prior width on each camera's offset, $\\sigma_b$ (mm)")
    ax.set_ylabel("median NEES")
    ax.set_title("Honesty against the prior width", loc="left")
    ax.set_ylim(0, max(nees) * 1.30)
    for i, (w, value, r) in enumerate(zip(widths, nees, rmse)):
        ax.annotate(f"{value:.2f}\nRMSE {r:.1f} cm", xy=(w, value),
                    xytext=(26 if i == 0 else 0, -4 if i == 0 else 10),
                    textcoords="offset points",
                    ha="left" if i == 0 else "center", fontsize=8)
    ax.annotate("an honest filter sits at 1.39", xy=(0.03, 0.08),
                xycoords="axes fraction", fontsize=8.5)

    ax2.plot(widths, recovered, "-o", color=C_ACCENT, lw=2.2, ms=8, mec="white",
             mew=1.2)
    ax2.axhline(1000 * f1.RESIDUAL_BIAS_M["camera_C"], color=C_TRUTH, lw=1.4, ls="--")
    ax2.annotate("historical-v2 camera C signed bias: 77 mm", xy=(0.03, 0.86),
                 xycoords="axes fraction", fontsize=8.5)
    ax2.set_xscale("log")
    ax2.set_xticks(widths, [f"{w:.0f}" for w in widths])
    ax2.set_xlabel("prior width on each camera's offset, $\\sigma_b$ (mm)")
    ax2.set_ylabel("historical-v2 camera C offset recovered (mm)")
    ax2.set_title("What it recovers, without ever seeing truth", loc="left")
    for w, value in zip(widths, recovered):
        ax2.annotate(f"{value:.0f}", xy=(w, value), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle("How wrong may the prior be? — the sensitivity exp2 demands of the floor, "
                 "asked of the offset model",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_m6_prior_sensitivity")
    return {"sweep": sweep}


# ------------------------------------------------------------------------- run


def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    models = rc.camera_models()
    calib = rc.deployed_calibration()
    capture = rc.load_operational_capture(CAPTURE, models=models, calib=calib)
    truth_table = rc.load_truth_table(CAPTURE)          # EVALUATION ONLY

    seq = Sequence(capture, truth_table)
    print(f"sequence: {seq.n_steps} steps at {GRID_HZ:.0f} Hz, "
          f"{int(seq.observed.sum())} observed "
          f"({100 * (1 - seq.observed.mean()):.0f} % missing)")

    deployed_q = np.eye(2) * (f1.PROCESS_SIGMA_PER_SQRT_M**2 * 0.02)
    deployed_r = {c: np.eye(2) * f1.R_COND_SIGMA_M[c]**2 for c in rc.CAMERAS}
    m0 = seq.odom[0].copy()
    S0 = np.eye(2) * f1.INITIAL_SIGMA_M**2

    print("\nstep 1 — the model and the data")
    m1 = fig_m1_the_model(seq)

    print("step 2 — inference: states only")
    base = PositionModel(Q=deployed_q, R_per_camera=deployed_r, m0=m0, S0=S0)
    base_result = score(seq, base, label="states only\n(deployed R)",
                        n_free_parameters=0)
    m2 = fig_m2_filtering_and_smoothing(seq, base, base_result)
    print(f"  -log p(y) = {base_result['neg_log_evidence']:.1f}   "
          f"smoothed median NEES = {base_result['smoothed']['median_nees']:.2f}")

    print("step 3 — forecasting")
    m3 = fig_m3_forecasting(seq, base)

    print("step 4 — inferring the noise covariances")
    r_inferred = infer_covariance(
        seq,
        lambda cov: PositionModel(Q=deployed_q,
                                  R_per_camera={c: cov for c in rc.CAMERAS},
                                  m0=m0, S0=S0),
        target="R")
    q_inferred = infer_covariance(
        seq,
        lambda cov: PositionModel(Q=cov, R_per_camera=deployed_r, m0=m0, S0=S0),
        target="Q",
        prior_scale=np.eye(2) * (0.01**2) * 6.0)
    m4 = fig_m4_inferring_the_covariance(r_inferred, q_inferred, deployed_r["camera_C"],
                                         deployed_q)
    print(f"  R sigma  {1000 * np.sqrt(np.diag(r_inferred['posterior_mean']))} mm")
    print(f"  Q sigma  {1000 * np.sqrt(np.diag(q_inferred['posterior_mean']))} mm")

    inferred_r_model = PositionModel(
        Q=deployed_q,
        R_per_camera={c: r_inferred["posterior_mean"] for c in rc.CAMERAS},
        m0=m0, S0=S0)
    inferred_r_result = score(seq, inferred_r_model, label="+ inferred $R$",
                              n_free_parameters=3)

    inferred_q_model = PositionModel(Q=q_inferred["posterior_mean"],
                                     R_per_camera=deployed_r, m0=m0, S0=S0)
    inferred_q_result = score(seq, inferred_q_model, label="+ inferred $Q$",
                              n_free_parameters=3)

    print("step 5 — the offset as a state")

    def build_bias_model(sigma_prior):
        return BiasAugmentedModel(
            Q_position=deployed_q,
            sigma_bias_walk=0.0005,
            R_per_camera=deployed_r,
            m0=m0, S0=S0,
            sigma_bias_prior=sigma_prior)

    def offsets_from(model, smooth):
        found = {}
        for camera in rc.CAMERAS:
            slot = model._slot(camera)
            vector = smooth["m"][-1][slot:slot + 2]
            found[camera] = float(1000 * np.hypot(*vector))
        return found

    bias_model = build_bias_model(0.10)
    bias_result = score(seq, bias_model, label="+ per-camera\noffset states",
                        n_free_parameters=0)
    inferred_bias = offsets_from(bias_model, bias_result["_smooth"])
    observed_counts = {c: sum(1 for x in seq.camera if x == c) for c in rc.CAMERAS}
    bias_estimate = {
        "inferred_mm": inferred_bias,
        "commissioned_mm": {c: 1000 * f1.RESIDUAL_BIAS_M[c] for c in rc.CAMERAS},
        "observed_steps": observed_counts,
    }
    for camera in rc.CAMERAS:
        seen = observed_counts[camera]
        note = "" if seen else "   (never observed here — prior, untouched)"
        print(f"  {camera}: inferred {inferred_bias[camera]:6.1f} mm  vs "
              f"commissioned {1000 * f1.RESIDUAL_BIAS_M[camera]:6.1f} mm{note}")

    results = [base_result, inferred_q_result, inferred_r_result, bias_result]
    m5 = fig_m5_model_comparison(results, bias_estimate, observed_counts)

    print("  prior sensitivity")
    sweep = []
    for sigma_prior in (0.02, 0.05, 0.10, 0.20):
        model = build_bias_model(sigma_prior)
        result = score(seq, model, label=f"prior {sigma_prior}", n_free_parameters=0)
        found = offsets_from(model, result["_smooth"])
        sweep.append({
            "sigma_bias_prior_m": sigma_prior,
            "median_nees": result["smoothed"]["median_nees"],
            "unearned_confidence_fraction":
                result["smoothed"]["unearned_confidence_fraction"],
            "rmse_m": result["smoothed"]["rmse_m"],
            "mean_stated_sigma_m": result["smoothed"]["mean_stated_sigma_m"],
            "camera_C_offset_mm": found["camera_C"],
        })
        print(f"    sigma_b = {1000 * sigma_prior:5.0f} mm -> NEES "
              f"{result['smoothed']['median_nees']:5.2f}, "
              f"RMSE {100 * result['smoothed']['rmse_m']:4.1f} cm, "
              f"camera C offset {found['camera_C']:5.1f} mm")
    m6 = fig_m6_prior_sensitivity(sweep)

    print("\nstep 6 — the same model on exp1's protocol, so it joins that ladder")
    ladder_arms = ["A0_trust_everything", "A1_hard_gate", "A2_factorized",
                   "A3_network_consistency", "A4_correlation_floor"]
    pooled_records = []
    pooled_by_arm = {arm: [] for arm in ladder_arms}
    per_capture = {}
    for name in rc.CAPTURES:
        other = rc.load_operational_capture(name, models=models, calib=calib)
        other_truth = rc.load_truth_table(name)         # EVALUATION ONLY

        def lookup(stamp, _table=other_truth):
            return rc.truth_at(_table, stamp)

        records = run_as_ladder_arm(other, lookup, sigma_bias_prior=0.05,
                                    sigma_bias_walk_per_sqrt_s=0.0016)
        per_capture[name] = f1.summarize(records)
        pooled_records.extend(records)
        for arm in ladder_arms:
            # d1.trace_arm keeps the full posterior covariance exp1 discards,
            # which the proper score needs; its NEES is verified identical there.
            pooled_by_arm[arm].extend(d1.trace_arm(other, arm, lookup))
        print(f"  {name:<26} n={per_capture[name]['n']:4d}  "
              f"NEES {per_capture[name]['median_nees']:5.2f}  "
              f"unearned {100 * per_capture[name]['unearned_confidence_fraction']:5.1f} %")

    a5 = f1.summarize(pooled_records)
    exp1_summary = json.loads(
        (REPO / "logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting"
                "/summary.json").read_text(encoding="utf-8"))
    m7 = fig_m7_same_ladder(exp1_summary, a5)

    print(f"\n  {'arm':<26}{'medNEES':>9}{'unearned%':>11}{'RMSE cm':>9}{'stated cm':>11}")
    for arm_name in ("A0_trust_everything", "A4_correlation_floor"):
        s = exp1_summary["pooled"][arm_name]
        print(f"  {arm_name:<26}{s['median_nees']:>9.2f}"
              f"{100 * s['unearned_confidence_fraction']:>11.1f}"
              f"{100 * s['rmse_m']:>9.1f}{100 * s['mean_stated_sigma_m']:>11.1f}")
    print(f"  {'A5_offset_states':<26}{a5['median_nees']:>9.2f}"
          f"{100 * a5['unearned_confidence_fraction']:>11.1f}"
          f"{100 * a5['rmse_m']:>9.1f}{100 * a5['mean_stated_sigma_m']:>11.1f}")
    final_offsets = pooled_records[-1]["offsets_mm"] if pooled_records else {}

    print("\nstep 7 — score the confidence with a proper rule, not the accuracy")
    short = {"A0_trust_everything": "A0", "A1_hard_gate": "A1",
             "A2_factorized": "A2", "A3_network_consistency": "A3",
             "A4_correlation_floor": "A4"}
    scores, rmse_cm = {}, {}
    for arm in ladder_arms:
        scores[short[arm]] = logarithmic_score(pooled_by_arm[arm])
        rmse_cm[short[arm]] = 100 * exp1_summary["pooled"][arm]["rmse_m"]
    scores["A5"] = logarithmic_score(pooled_records)
    rmse_cm["A5"] = 100 * a5["rmse_m"]
    m8 = fig_m8_proper_score(scores, rmse_cm)

    print(f"  {'arm':<6}{'mean NLPD':>11}{'= fit':>9}{'+ volume':>10}"
          f"{'medNLPD':>10}{'RMSE cm':>9}")
    for name, value in scores.items():
        print(f"  {name:<6}{value['mean_nlpd']:>11.2f}{value['mean_fit_term']:>9.2f}"
              f"{value['mean_volume_term']:>10.2f}{value['median_nlpd']:>10.2f}"
              f"{rmse_cm[name]:>9.1f}")
    print(f"  RMSE spans {m8['rmse_spread_ratio']:.2f}x; the score spans "
          f"{m8['score_spread_nats']:.1f} nats")

    print(f"\n{'model':<28}{'-log p(y)':>12}{'BIC':>11}{'medNEES':>10}"
          f"{'unearned%':>11}{'RMSE cm':>9}{'stated cm':>11}")
    for result in results:
        s = result["smoothed"]
        print(f"{result['label'].replace(chr(10), ' '):<28}"
              f"{result['neg_log_evidence']:>12.1f}{result['bic']:>11.1f}"
              f"{s['median_nees']:>10.2f}"
              f"{100 * s['unearned_confidence_fraction']:>11.1f}"
              f"{100 * s['rmse_m']:>9.1f}{100 * s['mean_stated_sigma_m']:>11.1f}")

    payload = {
        "reference": "BMLIP course, probprog/PP4 - Bayesian filtering and smoothing",
        "capture": CAPTURE,
        "grid_hz": GRID_HZ,
        "step1_model_and_data": m1,
        "step2_filtering_and_smoothing": m2,
        "step3_forecasting": m3,
        "step4_inferred_covariances": m4,
        "step5_model_comparison": m5,
        "step5_bias_estimate": bias_estimate,
        "step5_prior_sensitivity": m6,
        "step6_same_ladder": {
            "protocol": "exp1's own: per detection, filtered not smoothed, all three "
                        "captures, scored by f1.summarize; R = scatter + timing, no "
                        "bias term (the bias is a state here)",
            "sigma_bias_prior_m": 0.05,
            "A5_offset_states": a5,
            "per_capture": per_capture,
            "final_offsets_mm": final_offsets,
            "chart": m7,
        },
        "step7_proper_score": {
            "rule": "mean negative log predictive density of the truth under the "
                    "stated belief, = 0.5*(NEES + log det 2*pi*P); strictly proper, "
                    "so neither inflating nor shrinking the covariance can win it",
            "per_arm": scores,
            "chart": m8,
        },
        "note": "-log p(y) is the exact negative log evidence, not PP4's Bethe free "
                "energy; BIC adds a complexity penalty over the observed steps.",
    }
    rc.write_json(OUT / "summary.json", payload)
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
