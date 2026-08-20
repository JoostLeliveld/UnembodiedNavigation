#!/usr/bin/env python3
"""The state-space model, the estimators, and the measurements the notebooks score with.

Both notebooks import this. Neither defines an estimator of its own, so there is exactly
one `kalman_filter` in this study and both notebooks are demonstrably running it.

  the model         `Sequence` -- odometry increments on a uniform grid, observations
                    where a camera actually spoke, truth alongside for scoring only
  the estimators    `kalman_filter`, `rts_smoother`; `learn_R` (variational, learns the
                    observation covariance); `offset_state_filter` (carries a per-camera
                    offset in the state); `kalman_filter_jacobian_R` (rebuilds R at every
                    observation from the projection's derivative)
  the scorers       `honesty` (calibration and sharpness together), `error_summary`,
                    `score_offset_filter`; `forecast` / `forecast_summary` score the
                    forecast a filter makes for the NEXT camera reading, which needs no
                    ground truth and is therefore the only one a robot could run itself
  the measurements  `decompose_errors`, `prediction_residuals`, `message_outcomes`,
                    `what_the_innovation_sees`, `recovery_check`, `identifiability_sweep`

Anything that draws or prints lives in `notebook_views.py`; anything that loads the
capture lives in `notebook_data.py`.

EVALUATION ONLY: `Sequence.truth`, and every function that takes it, exists to score.
No estimator reads it.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy import stats

import notebook_data as nd

# ---------------------------------------------------------------- commissioned settings

PROCESS_SIGMA_PER_SQRT_M = 0.04     # commissioned wheel-odometry drift, m / sqrt(m)
INITIAL_SIGMA_M = 0.05              # how well the start pose is known
GRID_HZ = 10.0                      # the uniform time grid the state sequence lives on
ASSOC_TOL_S = 0.06                  # a detection belongs to the grid step it lands nearest
GATE_CHI2_2DOF = 5.991              # 95% of a chi-squared with 2 degrees of freedom

# A filter whose stated uncertainty is exactly right scores this, and not 2.0: the median
# of a chi-squared with two degrees of freedom is 2 ln 2. Every honesty number in both
# notebooks is read against it.
CALIBRATED_MEDIAN_NEES = 1.386

def offset_cameras():
    """The cameras the offset-state filter carries, in the world currently loaded."""
    return list(nd.CAMERAS)


def state_dim():
    """Position plus a two-dimensional offset per camera."""
    return 2 + 2 * len(offset_cameras())

# The robot's own surface, from its URDF meshes. The observation function needs it, and
# it is a tracked repo file, so loading it at import is the same class of dependency as
# reading the world file.
ROBOT_POINTS = nd.robot_point_cloud()

_YOLO_CACHE: dict = {}


def commissioned_noise(name=None):
    """The per-camera observation noise fitted on the three captures that predate this one.

    This capture is held out of that fit, so nothing either notebook scores is measured
    against a covariance fitted on itself. Two covariances come back, and the difference
    is not cosmetic:

      R_spread  the covariance about each camera's mean residual -- pure scatter
      R_total   the second moment about zero -- scatter AND offset together

    A model that says `y = x + zero-mean noise` has no term for a mean, so it can never
    subtract one. Commissioning honestly *for that model* therefore has to hand over
    R_total: the offset has to be paid for somewhere, and inflated noise is the only
    pocket the model has. R_spread describes a quantity that model cannot use.
    """
    path = nd.STUDY_ROOT / (name or nd.ACTIVE.commissioned_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"no commissioned noise for {nd.ACTIVE.key} at {path}; run "
            f"commission_observation_noise.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_camera = payload["per_camera"]
    cameras = [c for c in nd.CAMERAS if c in per_camera]
    return {
        "payload": payload,
        "R_spread": {c: np.asarray(per_camera[c]["R_spread"]) for c in cameras},
        "R_total": {c: np.asarray(per_camera[c]["R_total"]) for c in cameras},
        "offset_m": {c: np.asarray(per_camera[c]["mean_offset_m"]) for c in cameras},
        "sigma_total_m": {c: float(per_camera[c]["sigma_total_m"]) for c in cameras},
    }


# ------------------------------------------------------------------- the model

class Sequence:
    """A uniform-time state sequence: control in, observations where they exist.

    PP4's `y` is a dense array with gaps marked missing. This is the same object: on
    a 10 Hz grid most steps carry no detection, because camera coverage is a relay
    with holes in it.
    """

    def __init__(self, capture, truth_table, *, grid_hz=GRID_HZ, window=None):
        stamps = np.asarray(capture.stamps, dtype=float)
        odom = np.asarray(capture.odom, dtype=float)
        # Trim to the driven route. The recorders outlive the drive, and the robot sits
        # parked in one camera's view until shutdown; left in, that stationary tail
        # tripled camera B's detection count and would misrepresent every per-camera
        # statistic in this notebook.
        lo = stamps[0] if window is None else max(stamps[0], window[0])
        hi = stamps[-1] if window is None else min(stamps[-1], window[1])
        grid = np.arange(lo, hi, 1.0 / grid_hz)

        odom_on_grid = np.column_stack([np.interp(grid, stamps, odom[:, i]) for i in range(2)])
        self.stamps = grid
        self.dt = 1.0 / grid_hz
        self.odom = odom_on_grid
        self.u = np.vstack([np.zeros((1, 2)), np.diff(odom_on_grid, axis=0)])

        self.cameras = tuple(capture.cameras)
        detections = sorted(
            ((cam, d) for cam in self.cameras for d in capture.detections[cam]),
            key=lambda item: item[1].stamp,
        )
        self.y = np.full((len(grid), 2), np.nan)
        self.camera: list[str | None] = [None] * len(grid)
        self.pixel: list[tuple[float, float] | None] = [None] * len(grid)
        for cam, detection in detections:
            index = int(np.argmin(np.abs(grid - detection.stamp)))
            if abs(grid[index] - detection.stamp) > ASSOC_TOL_S:
                continue
            if self.camera[index] is not None:
                continue                       # one observation per step, as the model assumes
            self.y[index] = detection.world
            self.camera[index] = cam
            self.pixel[index] = (detection.u, detection.v)

        # EVALUATION ONLY -- for scoring, never for filtering
        self.truth = np.full((len(grid), 2), np.nan)
        for index, stamp in enumerate(grid):
            hit = nd.truth_at(truth_table, float(stamp))
            if hit is not None:
                self.truth[index] = hit[:2]

    @property
    def n_steps(self) -> int:
        return len(self.stamps)

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.y[:, 0])


class OffsetCorrected(Sequence):
    """The same sequence with each camera's average error removed. Diagnostic only."""

    def __init__(self, base, offsets):
        self.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else list(v)
                                  if isinstance(v, list) else v)
                              for k, v in base.__dict__.items()})
        for k in range(self.n_steps):
            cam = self.camera[k]
            if cam is not None and cam in offsets:
                self.y[k] = self.y[k] - offsets[cam]


# --------------------------------------------------------- filtering and smoothing

def kalman_filter(seq, R_per_camera, *, sigma_p=PROCESS_SIGMA_PER_SQRT_M,
                  initial_sigma=INITIAL_SIGMA_M, gate=GATE_CHI2_2DOF, m0=None):
    """Forward pass. Returns per-step means, covariances and diagnostics."""
    identity = np.eye(2)
    m = np.asarray(m0 if m0 is not None else seq.odom[0], dtype=float).copy()
    P = identity * initial_sigma**2

    out = {
        "m": np.zeros((seq.n_steps, 2)), "P": np.zeros((seq.n_steps, 2, 2)),
        "m_pred": np.zeros((seq.n_steps, 2)), "P_pred": np.zeros((seq.n_steps, 2, 2)),
        "innovation": np.full((seq.n_steps, 2), np.nan),
        "nis": np.full(seq.n_steps, np.nan),
        "used": np.zeros(seq.n_steps, dtype=bool),
        "rejected": np.zeros(seq.n_steps, dtype=bool),
        "log_evidence": 0.0,
    }

    for k in range(seq.n_steps):
        u = seq.u[k]
        Q = identity * (sigma_p**2 * float(np.linalg.norm(u)))
        m = m + u
        P = P + Q
        out["m_pred"][k], out["P_pred"][k] = m, P

        camera = seq.camera[k]
        if camera is not None:
            R = R_per_camera[camera]
            v = seq.y[k] - m                      # innovation
            S = P + R                             # innovation covariance
            S_inv = np.linalg.inv(S)
            nis = float(v @ S_inv @ v)
            out["innovation"][k], out["nis"][k] = v, nis
            if nis <= gate:
                K = P @ S_inv
                m = m + K @ v
                IKH = identity - K
                P = IKH @ P @ IKH.T + K @ R @ K.T      # Joseph form
                P = 0.5 * (P + P.T)
                out["used"][k] = True
                out["log_evidence"] += -0.5 * (
                    nis + math.log(max(np.linalg.det(2 * math.pi * S), 1e-300)))
            else:
                out["rejected"][k] = True

        out["m"][k], out["P"][k] = m, P
    return out


def rts_smoother(seq, forward):
    """Backward pass. Returns smoothed means and covariances."""
    n = seq.n_steps
    ms = forward["m"].copy()
    Ps = forward["P"].copy()
    for k in range(n - 2, -1, -1):
        P_next_pred = forward["P_pred"][k + 1]
        G = forward["P"][k] @ np.linalg.inv(P_next_pred)
        ms[k] = forward["m"][k] + G @ (ms[k + 1] - forward["m_pred"][k + 1])
        Ps[k] = forward["P"][k] + G @ (Ps[k + 1] - P_next_pred) @ G.T
        Ps[k] = 0.5 * (Ps[k] + Ps[k].T)
    return {"m": ms, "P": Ps}


def error_summary(means, seq, label):
    """Distance from truth, where truth exists."""
    ok = np.isfinite(seq.truth[:, 0])
    err = np.linalg.norm(means[ok] - seq.truth[ok], axis=1)
    print(f"  {label:22s} median {100 * np.median(err):5.1f} cm | "
          f"90th percentile {100 * np.quantile(err, 0.9):5.1f} cm | "
          f"worst {100 * err.max():5.1f} cm")
    return err


# ----------------------------------------------------------------------- the scorers

def honesty(result, seq, label, *, steps=None):
    """Calibration, sharpness and accuracy of a belief. EVALUATION ONLY.

    `steps` restricts the scoring to a (lo, hi) range, so a belief can be judged on the
    part of a drive its covariance was not fitted on.
    """
    ok = np.isfinite(seq.truth[:, 0])
    if steps is not None:
        inside = np.zeros(seq.n_steps, dtype=bool)
        inside[steps[0]:steps[1]] = True
        ok = ok & inside
    nees, nlpd = [], []
    for k in np.flatnonzero(ok):
        e = seq.truth[k] - result["m"][k]
        P = result["P"][k]
        P_inv = np.linalg.inv(P)
        value = float(e @ P_inv @ e)
        nees.append(value)
        nlpd.append(0.5 * (value + math.log(max(np.linalg.det(2 * math.pi * P), 1e-300))))
    nees = np.asarray(nees); nlpd = np.asarray(nlpd)
    err = np.linalg.norm(result["m"][ok] - seq.truth[ok], axis=1)
    return {"label": label, "median_nees": float(np.median(nees)),
            "mean_nlpd": float(np.mean(nlpd)), "rmse_cm": float(100 * np.sqrt((err**2).mean())),
            "nees": nees}


# ------------------------------------------------------------------- learning R

def iw_kl_from_prior(Psi_q, nu_q, Psi_p, nu_p, d=2):
    """KL( IW(Psi_q, nu_q) || IW(Psi_p, nu_p) ) -- how far the data moved the belief.

    Taking expectations of the inverse-Wishart log density under q, and using
    E_q[log|R|] = log|Psi_q| - d log 2 - sum_i psi((nu_q - i + 1)/2) and
    E_q[R^-1] = nu_q Psi_q^-1 (so that tr(Psi_q E_q[R^-1]) = nu_q d):

        KL = (nu_q/2) log|Psi_q| - (nu_p/2) log|Psi_p|
             - (d/2)(nu_q - nu_p) log 2
             - logGamma_d(nu_q/2) + logGamma_d(nu_p/2)
             - ((nu_q - nu_p)/2) E_q[log|R|]
             - (1/2) tr((Psi_q - Psi_p) E_q[R^-1])

    Verified below against the two properties any KL must have.
    """
    from scipy.special import multigammaln, digamma

    sign_q, logdet_q = np.linalg.slogdet(Psi_q)
    sign_p, logdet_p = np.linalg.slogdet(Psi_p)
    if sign_q <= 0 or sign_p <= 0:
        return float("nan")
    e_logdet = logdet_q - d * math.log(2.0) - sum(
        digamma((nu_q - i + 1) / 2.0) for i in range(1, d + 1))
    e_inv = nu_q * np.linalg.inv(Psi_q)
    return float(
        0.5 * nu_q * logdet_q - 0.5 * nu_p * logdet_p
        - 0.5 * d * (nu_q - nu_p) * math.log(2.0)
        - multigammaln(nu_q / 2.0, d) + multigammaln(nu_p / 2.0, d)
        - 0.5 * (nu_q - nu_p) * e_logdet
        - 0.5 * np.trace((Psi_q - Psi_p) @ e_inv)
    )


def learn_R(seq, *, iterations=12, prior_nu=6.0, prior_sigma_m=0.05,
            sigma_p=PROCESS_SIGMA_PER_SQRT_M, gate=GATE_CHI2_2DOF):
    """Mean-field variational inference for the per-camera R. Q is never touched.

    Returns the posterior parameters, not just a point estimate, so the notebook can
    plot how sure the data has made us.

    `gate` is exposed because it is a fair suspicion about this loop: the x step rejects
    observations whose innovation is too large for the current R, so as R shrinks the loop
    could be quietly throwing away the very residuals that would keep it honest. Passing
    `gate=float("inf")` learns from every observation and settles that question by
    measurement rather than by argument.
    """
    d = 2
    Psi = np.eye(d) * (prior_sigma_m**2) * prior_nu
    # start from the prior: R_bar = (E[R^-1])^-1 = Psi / nu
    cameras = list(getattr(seq, "cameras", nd.CAMERAS))
    R_bar = {c: Psi / prior_nu for c in cameras}
    posterior = {}
    history = []

    # q(R) starts at the prior, so the first ELBO is evaluated there.
    q_now = {c: {"Psi": Psi.copy(), "nu": float(prior_nu)} for c in cameras}

    for iteration in range(iterations):
        forward_i = kalman_filter(seq, R_bar, sigma_p=sigma_p, gate=gate)   # the x step
        smooth_i = rts_smoother(seq, forward_i)
        # The bound is evaluated HERE, straight after the x step, because that is the
        # point where q(x) is exactly p(x | y, R_bar) and the identity in `elbo` holds.
        bound = elbo(seq, R_bar, q_now, Psi, prior_nu, sigma_p=sigma_p)
        R_in = {c: R_bar[c].copy() for c in cameras}

        new_R_bar, counts = {}, {}
        for cam in cameras:                                             # the R step
            steps = [k for k in range(seq.n_steps)
                     if seq.camera[k] == cam and forward_i["used"][k]]
            counts[cam] = len(steps)
            scatter = np.zeros((d, d))
            for k in steps:
                v = (seq.y[k] - smooth_i["m"][k]).reshape(d, 1)
                scatter += v @ v.T + smooth_i["P"][k]
            Psi_post = Psi + scatter
            nu_post = prior_nu + len(steps)
            posterior[cam] = {"Psi": Psi_post, "nu": nu_post}
            # what the x step needs is the expected PRECISION, inverted
            new_R_bar[cam] = Psi_post / nu_post
        R_bar = new_R_bar
        q_now = {c: {"Psi": posterior[c]["Psi"].copy(), "nu": float(posterior[c]["nu"])}
                 for c in cameras if c in posterior}
        # The gate admits a different subset of observations for every R, so the gated
        # evidence sums over different data and cannot be compared across arms. Keep an
        # ungated figure, over all observations, for any comparison between models.
        # Snapshot the posterior every iteration, so the fitting can be replayed later --
        # `honesty` is not defined yet at this point in the notebook, so the calibration of
        # each iterate is scored where the animation is built rather than here.
        history.append({
            "iteration": iteration,
            "elbo": bound,
            # the R_bar that `bound` was evaluated at -- i.e. the state going INTO this
            # pass. `R_bar` below is the state coming out of it, one R step later.
            "R_in": {c: R_in[c].copy() for c in cameras},
            "log_evidence": forward_i["log_evidence"],
            "log_evidence_all": kalman_filter(seq, R_bar, sigma_p=sigma_p,
                                              gate=float("inf"))["log_evidence"],
            "R_bar": {c: R_bar[c].copy() for c in cameras},
            "posterior": {c: {"Psi": posterior[c]["Psi"].copy(),
                              "nu": float(posterior[c]["nu"])} for c in cameras},
            "sigma_m": {c: float(np.sqrt(np.trace(R_bar[c]) / d)) for c in cameras},
            "kl_from_prior": {c: iw_kl_from_prior(posterior[c]["Psi"], posterior[c]["nu"],
                                                  Psi, prior_nu, d) for c in cameras},
            "counts": counts,
        })

    # Each record above was scored with the R that went INTO that iteration, so the last
    # one does not describe the R actually returned. Score that too.
    final_forward = kalman_filter(seq, R_bar, sigma_p=sigma_p, gate=gate)
    history.append(dict(
        history[-1], iteration=iterations,
        elbo=elbo(seq, R_bar, q_now, Psi, prior_nu, sigma_p=sigma_p),
        R_in={c: R_bar[c].copy() for c in cameras},
        log_evidence=final_forward["log_evidence"],
        log_evidence_all=kalman_filter(seq, R_bar, sigma_p=sigma_p,
                                       gate=float("inf"))["log_evidence"],
        R_bar={c: R_bar[c].copy() for c in cameras},
        sigma_m={c: float(np.sqrt(np.trace(R_bar[c]) / d)) for c in cameras}))
    return R_bar, history, {"posterior": posterior, "Psi_prior": Psi, "nu_prior": prior_nu}


def sigma_density(Psi, nu, sigma_grid, axis=0, d=2):
    """Marginal density over one axis' standard deviation under IW(Psi, nu)."""
    # marginal variance ~ InvGamma(shape=(nu - d + 1)/2, scale=Psi[i,i]/2)
    shape = (nu - d + 1.0) / 2.0
    scale = Psi[axis, axis] / 2.0
    var = sigma_grid**2
    pdf_var = stats.invgamma.pdf(var, a=shape, scale=scale)
    return pdf_var * 2.0 * sigma_grid            # Jacobian d(var)/d(sigma)


# --------------------------------------------- R rebuilt from the projection itself

def projection_jacobian(model, u, v, step_px=0.5):
    """d(world)/d(pixel) by the same central difference the runtime uses."""
    columns = []
    for axis in (0, 1):
        delta = [step_px if axis == 0 else 0.0, step_px if axis == 1 else 0.0]
        plus = model.pixel_to_world(u + delta[0], v + delta[1])
        minus = model.pixel_to_world(u - delta[0], v - delta[1])
        if plus is None or minus is None:
            return None
        columns.append((np.asarray(plus) - np.asarray(minus)) / (2.0 * step_px))
    return np.column_stack(columns)


def kalman_filter_jacobian_R(seq, sigma_px, fallback, models, *,
                             sigma_p=PROCESS_SIGMA_PER_SQRT_M,
                             initial_sigma=INITIAL_SIGMA_M, gate=GATE_CHI2_2DOF):
    """The same filter, with R rebuilt at every observation from that pixel's Jacobian."""
    identity = np.eye(2)
    m = seq.odom[0].copy()
    P = identity * initial_sigma**2
    out = {"m": np.zeros((seq.n_steps, 2)), "P": np.zeros((seq.n_steps, 2, 2)),
           "m_pred": np.zeros((seq.n_steps, 2)), "P_pred": np.zeros((seq.n_steps, 2, 2)),
           "R": np.full((seq.n_steps, 2, 2), np.nan),
           "used": np.zeros(seq.n_steps, dtype=bool)}
    for k in range(seq.n_steps):
        u = seq.u[k]
        m = m + u
        P = P + identity * (sigma_p**2 * float(np.linalg.norm(u)))
        out["m_pred"][k], out["P_pred"][k] = m, P
        camera = seq.camera[k]
        if camera is not None:
            J = projection_jacobian(models[camera], *seq.pixel[k])
            R = fallback[camera] if J is None else (sigma_px**2) * (J @ J.T)
            out["R"][k] = R
            innovation = seq.y[k] - m
            S = P + R
            if float(innovation @ np.linalg.inv(S) @ innovation) <= gate:
                K = P @ np.linalg.inv(S)
                m = m + K @ innovation
                closed = identity - K
                P = closed @ P @ closed.T + K @ R @ K.T
                P = 0.5 * (P + P.T)
                out["used"][k] = True
        out["m"][k] = m
        out["P"][k] = P
    return out


# ------------------------------------------- the observation function, done properly

def silhouette_bottom(model, x, y, yaw, points=ROBOT_POINTS):
    """Where the bottom-centre of the robot's projected bounding box lands on the floor.

    This is the observation function the pipeline should have been using. Zero fitted
    parameters: the shape is the mesh, the camera is the world file.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = points @ rotation.T + np.array([x, y, 0.0])
    in_camera = (world - model.cam_pos) @ model.R.T
    ahead = in_camera[:, 2] > 1e-6
    if not ahead.any():
        return None
    projected = (model.K @ in_camera[ahead].T).T
    uv = projected[:, :2] / projected[:, 2:3]
    return model.pixel_to_world(0.5 * (uv[:, 0].min() + uv[:, 0].max()), uv[:, 1].max())


def silhouette_box(model, x, y, yaw, points=ROBOT_POINTS):
    """The robot's projected bounding box IN PIXELS, before anything is back-projected.

    Working in pixels is what makes the error decomposition clean. A world-space residual
    mixes three things -- how many pixels the detector was wrong by, how many centimetres
    a pixel is worth there, and the silhouette displacement -- and the projection factor
    cancels out of none of them. In pixel space the projection is simply absent, so the
    residual `observed_uv - box_bottom_uv` is the detector's error and nothing else.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = points @ rotation.T + np.array([x, y, 0.0])
    in_camera = (world - model.cam_pos) @ model.R.T
    ahead = in_camera[:, 2] > 1e-6
    if not ahead.any():
        return None
    projected = (model.K @ in_camera[ahead].T).T
    uv = projected[:, :2] / projected[:, 2:3]
    u0, u1 = float(uv[:, 0].min()), float(uv[:, 0].max())
    v0, v1 = float(uv[:, 1].min()), float(uv[:, 1].max())
    return {"bottom_centre_uv": (0.5 * (u0 + u1), v1),
            "box": (u0, v0, u1, v1),
            "width_px": u1 - u0, "height_px": v1 - v0,
            "diagonal_px": float(math.hypot(u1 - u0, v1 - v0))}


def predicted_offset(model, x, y, yaw):
    """What g_c says the observation will be displaced by, at this pose."""
    landing = silhouette_bottom(model, x, y, yaw)
    return None if landing is None else np.array([landing[0] - x, landing[1] - y])


def heading_from_odometry(seq, span=8):
    """Which way the robot is pointing, from the odometry track alone.

    No ground truth: this is the direction the dead-reckoned position is moving in. It is
    available to the filter at run time, which is the whole point.
    """
    out = np.full(seq.n_steps, np.nan)
    for k in range(seq.n_steps):
        a, b = max(0, k - span), min(seq.n_steps - 1, k + span)
        step = seq.odom[b] - seq.odom[a]
        if float(np.linalg.norm(step)) > 0.02:
            out[k] = math.atan2(step[1], step[0])
    return out


class GeometryCorrected(Sequence):
    """The same sequence with the observation function corrected by the object model.

    Uses the filter's own running estimate for position and odometry for heading, so it is
    deployable: nothing here is unavailable at run time.
    """

    def __init__(self, base, models, R_per_camera, heading, *,
                 sigma_p=PROCESS_SIGMA_PER_SQRT_M, initial_sigma=INITIAL_SIGMA_M):
        self.__dict__.update({
            k: (v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v)
            for k, v in base.__dict__.items()})
        identity = np.eye(2)
        m = self.odom[0].copy()
        P = identity * initial_sigma**2
        self.n_corrected = 0
        for k in range(self.n_steps):
            u = self.u[k]
            m = m + u
            P = P + identity * (sigma_p**2 * float(np.linalg.norm(u)))
            camera = self.camera[k]
            if camera is not None and np.isfinite(heading[k]):
                offset = predicted_offset(models[camera], m[0], m[1], float(heading[k]))
                if offset is not None:
                    self.y[k] = self.y[k] - offset
                    self.n_corrected += 1
            if camera is not None:
                R = R_per_camera[camera]
                innovation = self.y[k] - m
                S = P + R
                if float(innovation @ np.linalg.inv(S) @ innovation) <= GATE_CHI2_2DOF:
                    K = P @ np.linalg.inv(S)
                    m = m + K @ innovation
                    P = (identity - K) @ P @ (identity - K).T + K @ R @ K.T
                    P = 0.5 * (P + P.T)


def prediction_residuals(names, models, points=ROBOT_POINTS):
    """|observation - prediction| under several ways of supplying the heading."""
    out = {k: [] for k in ("no correction", "true heading", "odometry heading",
                           "heading assumed zero")}
    for name in names:
        cap = nd.load_capture(name, models=models)
        table = nd.load_truth(name)
        grid = Sequence(cap, table, window=nd.route_window(name))
        head = heading_from_odometry(grid)
        for cam in nd.CAMERAS:
            for det in cap.detections[cam]:
                hit = nd.truth_at(table, det.stamp, tol_s=0.05)
                if hit is None:
                    continue
                observed = np.asarray(det.world)
                truth_xy = np.asarray(hit[:2])
                out["no correction"].append(float(np.linalg.norm(observed - truth_xy)))
                index = int(np.argmin(np.abs(grid.stamps - det.stamp)))
                supplies = {"true heading": hit[2],
                            "odometry heading": (head[index] if index < grid.n_steps else np.nan),
                            "heading assumed zero": 0.0}
                for label, yaw in supplies.items():
                    if yaw is None or not np.isfinite(yaw):
                        continue
                    landing = silhouette_bottom(models[cam], truth_xy[0], truth_xy[1],
                                                float(yaw), points)
                    if landing is not None:
                        out[label].append(float(np.linalg.norm(observed - np.asarray(landing))))
    return {k: np.asarray(v) for k, v in out.items()}


# --------------------------------------------- the offset carried in the state

def offset_state_filter(seq, R_per_camera, *, sigma_b_prior=0.10, sigma_b_walk=0.0,
                        sigma_p=PROCESS_SIGMA_PER_SQRT_M, initial_sigma=INITIAL_SIGMA_M,
                        gate=GATE_CHI2_2DOF):
    """Position and one 2-D offset per camera, all estimated together."""
    cams = list(getattr(seq, "cameras", nd.CAMERAS))
    dim = 2 + 2 * len(cams)
    identity = np.eye(dim)
    m = np.zeros(dim)
    m[:2] = seq.odom[0]
    P = np.zeros((dim, dim))
    P[:2, :2] = np.eye(2) * initial_sigma**2
    for i in range(len(cams)):
        P[2 + 2 * i:4 + 2 * i, 2 + 2 * i:4 + 2 * i] = np.eye(2) * sigma_b_prior**2

    out = {"m": np.zeros((seq.n_steps, dim)),
           "sd": np.zeros((seq.n_steps, dim)),
           "P_position": np.zeros((seq.n_steps, 2, 2)),
           # the 2x2 block for each camera's offset, so its uncertainty can be drawn
           "P_offset": np.zeros((seq.n_steps, len(cams), 2, 2)),
           "used": np.zeros(seq.n_steps, dtype=bool)}

    for k in range(seq.n_steps):
        u = seq.u[k]
        m[:2] = m[:2] + u
        Q = np.zeros((dim, dim))
        Q[:2, :2] = np.eye(2) * (sigma_p**2 * float(np.linalg.norm(u)))
        for i in range(len(cams)):
            Q[2 + 2 * i:4 + 2 * i, 2 + 2 * i:4 + 2 * i] = np.eye(2) * sigma_b_walk**2
        P = P + Q

        camera = seq.camera[k]
        if camera is not None:
            i = cams.index(camera)
            H = np.zeros((2, dim))
            H[:, :2] = np.eye(2)
            H[:, 2 + 2 * i:4 + 2 * i] = np.eye(2)
            R = R_per_camera[camera]
            innovation = seq.y[k] - H @ m
            S = H @ P @ H.T + R
            if float(innovation @ np.linalg.inv(S) @ innovation) <= gate:
                K = P @ H.T @ np.linalg.inv(S)
                m = m + K @ innovation
                closed = identity - K @ H
                P = closed @ P @ closed.T + K @ R @ K.T
                P = 0.5 * (P + P.T)
                out["used"][k] = True
        out["m"][k] = m
        out["sd"][k] = np.sqrt(np.maximum(np.diag(P), 0.0))
        out["P_position"][k] = P[:2, :2]
        for i in range(len(cams)):
            out["P_offset"][k, i] = P[2 + 2 * i:4 + 2 * i, 2 + 2 * i:4 + 2 * i]
    return out


def score_offset_filter(result, seq, label):
    """Honesty of the POSITION marginal -- the offsets are means to an end."""
    ok = np.isfinite(seq.truth[:, 0])
    values = []
    for k in np.flatnonzero(ok):
        e = seq.truth[k] - result["m"][k, :2]
        values.append(float(e @ np.linalg.inv(result["P_position"][k]) @ e))
    err = np.linalg.norm(result["m"][ok, :2] - seq.truth[ok], axis=1)
    return {"label": label, "median_nees": float(np.median(values)),
            "rmse_cm": float(100 * np.sqrt((err**2).mean()))}


# ------------------------------------------- measurements on the observations

def decompose_errors(names, models):
    """Split every error into radial (along camera->robot) and across-track parts."""
    rows = {cam: {"radial": [], "tangential": [], "range": []} for cam in nd.CAMERAS}
    for name in names:
        cap = nd.load_capture(name, models=models)
        table = nd.load_truth(name)
        for cam in nd.CAMERAS:
            cx, cy = float(models[cam].cam_pos[0]), float(models[cam].cam_pos[1])
            for det in cap.detections[cam]:
                hit = nd.truth_at(table, det.stamp, tol_s=0.05)
                if hit is None:
                    continue
                err = np.array([det.world[0] - hit[0], det.world[1] - hit[1]])
                sight = np.array([hit[0] - cx, hit[1] - cy])
                rng = float(np.linalg.norm(sight))
                if rng < 1e-6:
                    continue
                along = sight / rng
                across = np.array([-along[1], along[0]])
                rows[cam]["radial"].append(float(err @ along))
                rows[cam]["tangential"].append(float(err @ across))
                rows[cam]["range"].append(rng)
    return {cam: {k: np.asarray(v) for k, v in d.items()}
            for cam, d in rows.items() if len(d["radial"]) > 5}


def message_outcomes(messages, truth_table, models, window=None):
    """Every message, with the robot's true range, image position, and whether it was
    inside that camera's image at all.

    The last one matters. On a single straight traverse the range to a camera and whether
    the robot is even in frame move together, so a raw "detection rate against range"
    mixes 'the detector failed' with 'there was nothing to detect' and comes out
    non-monotone and meaningless. Splitting them is the whole point of this cell.

    Inside the image is not the same as visible: a shelf can stand in the way, and nothing
    here accounts for occlusion. So 'in frame and missed' is an upper bound on the
    detector's own failures.
    """
    out = {}
    for cam in nd.CAMERAS:
        model = models[cam]
        cam_xy = np.asarray(model.cam_pos[:2], dtype=float)
        rows = {"range": [], "u": [], "v": [], "ok": [], "stamp": [], "in_frame": []}
        for stamp, ok in messages[cam]:
            if window is not None and not (window[0] <= stamp <= window[1]):
                continue
            hit = nd.truth_at(truth_table, stamp, tol_s=0.05)
            if hit is None:
                continue
            u, v, visible = model.world_to_pixel(hit[0], hit[1], 0.0)
            rows["range"].append(float(np.linalg.norm(np.asarray(hit[:2]) - cam_xy)))
            rows["u"].append(u); rows["v"].append(v)
            rows["ok"].append(bool(ok)); rows["stamp"].append(stamp)
            rows["in_frame"].append(bool(visible))
        out[cam] = {k: np.asarray(v) for k, v in rows.items()}
    return out


def detect_on_frame(image_bgr, model_path, *, imgsz=960, conf=0.05, iou=0.45):
    """Run the deployed detector on one frame; return boxes as (x1, y1, x2, y2, conf)."""
    from ultralytics import YOLO

    if model_path not in _YOLO_CACHE:
        _YOLO_CACHE[model_path] = YOLO(str(model_path))
    result = _YOLO_CACHE[model_path].predict(
        source=[image_bgr], imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confidence = boxes.conf.cpu().numpy()
    return [(*xyxy[i], float(confidence[i])) for i in np.argsort(-confidence)]


# ------------------------------------------- loops the notebooks used to carry inline

def pick_example_frame(seq, capture):
    """An observed step whose camera frame was also written to disk.

    Picked as near the middle of that camera's visibility as possible, so the picture
    shows the detector working rather than a hand-picked best case.
    """
    for cam in nd.CAMERAS:
        idx = [i for i, c in enumerate(seq.camera)
               if c == cam and np.isfinite(seq.truth[i, 0])]
        if not idx:
            continue
        middle = idx[len(idx) // 2]
        frame = capture.frame_at(cam, float(seq.stamps[middle]), tol_s=0.6)
        if frame is not None:
            return {"camera": cam, "step": middle, "stamp": float(seq.stamps[middle]),
                    "frame_stamp": frame[0], "frame_path": frame[1]}
    return None


def northing_of(outcomes, truth_table):
    """Where along the aisle each detector message happened.

    Range turns out not to be the controlling variable: conditioned on being in frame, the
    rate against range is non-monotone for every camera. Position along the aisle is the
    axis that explains the behaviour, because it is what moves the robot in and out of
    each camera's patch of floor and behind the shelf rows.
    """
    out = {}
    for cam in nd.CAMERAS:
        values = []
        for stamp in outcomes[cam]["stamp"]:
            hit = nd.truth_at(truth_table, float(stamp), tol_s=0.05)
            values.append(hit[1] if hit is not None else np.nan)
        out[cam] = np.asarray(values)
    return out


def geometric_offsets(seq, models, heading):
    """Each camera's mean predicted offset over this drive, from the robot's own mesh."""
    out = {}
    for cam in getattr(seq, "cameras", nd.CAMERAS):
        rows = []
        for k in range(seq.n_steps):
            if seq.camera[k] != cam or not np.isfinite(heading[k]):
                continue
            landing = silhouette_bottom(models[cam], seq.truth[k, 0], seq.truth[k, 1],
                                        float(heading[k]))
            if landing is not None:
                rows.append(np.asarray(landing) - seq.truth[k])
        out[cam] = np.mean(rows, axis=0) if rows else np.full(2, np.nan)
    return out


def pixel_scale_against_residual(seq, models, heading):
    """What one pixel is worth on the floor, against what is left of the error there.

    The scale is `sqrt(trace(J J^T) / 2)` at the pixel the detection actually landed on;
    the residual is measured after the mesh-predicted offset is removed, so what remains
    is the part a covariance could legitimately be asked to describe.
    """
    scale, residual, camera = [], [], []
    for k in range(seq.n_steps):
        cam = seq.camera[k]
        if cam is None or not np.isfinite(heading[k]) or not np.isfinite(seq.truth[k, 0]):
            continue
        J = projection_jacobian(models[cam], *seq.pixel[k])
        landing = silhouette_bottom(models[cam], seq.truth[k, 0], seq.truth[k, 1],
                                    float(heading[k]))
        if J is None or landing is None:
            continue
        scale.append(float(np.sqrt(np.trace(J @ J.T) / 2)))
        residual.append(float(np.linalg.norm(
            seq.y[k] - (np.asarray(landing) - seq.truth[k]) - seq.truth[k])))
        camera.append(cam)
    return {"cm_per_pixel": np.asarray(scale), "residual_m": np.asarray(residual),
            "camera": camera}


# The deployed detector, as shipped. Only the views that draw a frame need it.
def detector_path(capture=None):
    """The weights that produced this capture's detections. See `nd.detector_of`."""
    return nd.detector_of(None if capture is None else capture.name)


# --------------------------------------------- learning R, taken one camera at a time

def single_camera(seq, camera):
    """The same drive with only one camera's observations kept.

    Everything else is untouched: the same odometry, the same grid, the same truth. The
    other cameras' detections are simply never offered, so the filter dead-reckons
    through the stretches where this one camera cannot see the robot.
    """
    out = Sequence.__new__(Sequence)
    out.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray)
                             else list(v) if isinstance(v, list) else v)
                         for k, v in seq.__dict__.items()})
    for k in range(out.n_steps):
        if out.camera[k] is not None and out.camera[k] != camera:
            out.y[k] = np.nan
            out.camera[k] = None
            out.pixel[k] = None
    return out


def one_conjugate_update(seq, camera, *, prior_nu=6.0, prior_sigma_m=0.05,
                         R_start=None, sigma_p=PROCESS_SIGMA_PER_SQRT_M):
    """One turn of the crank for one camera, with every intermediate kept.

    This is the whole of `learn_R` for a single camera and a single iteration, opened up
    so the arithmetic can be printed rather than described:

      1  run the filter and smoother with the R we currently believe   (the x step)
      2  add up (residual residual' + smoother covariance) over that camera's steps
      3  Psi and nu go up by exactly that, and by the number of observations
      4  the new working covariance is Psi+ / nu+, the inverse of the expected precision

    Returns the prior, the scatter, the posterior and both, so the notebook can show
    what moved and by how much.
    """
    d = 2
    Psi = np.eye(d) * (prior_sigma_m ** 2) * prior_nu
    R_bar = {c: (Psi / prior_nu if R_start is None else R_start[c])
             for c in getattr(seq, "cameras", nd.CAMERAS)}

    forward = kalman_filter(seq, R_bar, sigma_p=sigma_p)
    smooth = rts_smoother(seq, forward)

    steps = [k for k in range(seq.n_steps)
             if seq.camera[k] == camera and forward["used"][k]]
    scatter = np.zeros((d, d))
    residual_part = np.zeros((d, d))
    smoother_part = np.zeros((d, d))
    for k in steps:
        v = (seq.y[k] - smooth["m"][k]).reshape(d, 1)
        residual_part += v @ v.T
        smoother_part += smooth["P"][k]
    scatter = residual_part + smoother_part

    Psi_post = Psi + scatter
    nu_post = prior_nu + len(steps)
    return {
        "camera": camera, "n": len(steps),
        "Psi_prior": Psi, "nu_prior": prior_nu, "R_prior": Psi / prior_nu,
        "residual_part": residual_part, "smoother_part": smoother_part,
        "scatter": scatter, "Psi_post": Psi_post, "nu_post": nu_post,
        "R_post": Psi_post / nu_post,
        "sigma_prior_m": float(np.sqrt(np.trace(Psi / prior_nu) / d)),
        "sigma_post_m": float(np.sqrt(np.trace(Psi_post / nu_post) / d)),
    }


def pixel_worth_over_floor(models, camera, *, extent=(-11.0, 11.0, -10.0, 10.0),
                           nx=110, ny=100):
    """What one pixel of detector error is worth, in metres of floor, at every floor point.

    Floor point -> pixel (through the camera model) -> J -> sqrt(trace(J J') / 2). NaN
    wherever that camera cannot see the point. This is the same J the deployed runtime
    differentiates, so the field is what `R` actually is, not an illustration of it.
    """
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    model = models[camera]
    worth = np.full((ny, nx), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            u, v, in_frame = model.world_to_pixel(float(x), float(y), 0.0)
            if not in_frame:
                continue
            J = projection_jacobian(model, u, v)
            if J is None:
                continue
            worth[j, i] = float(np.sqrt(np.trace(J @ J.T) / 2.0))
    return {"x": xs, "y": ys, "worth_m": worth, "camera": camera}


def R_at_floor_point(models, camera, x, y, sigma_px=1.0):
    """The observation covariance a detection at this floor point would carry."""
    u, v, in_frame = models[camera].world_to_pixel(float(x), float(y), 0.0)
    if not in_frame:
        return None
    J = projection_jacobian(models[camera], u, v)
    return None if J is None else (sigma_px ** 2) * (J @ J.T)


def conjugate_trace(seq, camera, counts, *, prior_nu=6.0, prior_sigma_m=0.05):
    """The posterior over R after the first n of this camera's observations, for each n.

    The trajectory is held at the one the prior's R gives, so this isolates the conjugate
    update itself: each observation adds its own outer product and one count to nu.
    """
    d = 2
    Psi = np.eye(d) * (prior_sigma_m ** 2) * prior_nu
    R_bar = {c: Psi / prior_nu for c in getattr(seq, "cameras", nd.CAMERAS)}
    forward = kalman_filter(seq, R_bar)
    smooth = rts_smoother(seq, forward)
    steps = [k for k in range(seq.n_steps) if seq.camera[k] == camera and forward["used"][k]]

    out, scatter = [], np.zeros((d, d))
    for n, k in enumerate(steps, start=1):
        v = (seq.y[k] - smooth["m"][k]).reshape(d, 1)
        scatter += v @ v.T + smooth["P"][k]
        if n in counts:
            out.append({"n": n, "Psi": Psi + scatter, "nu": prior_nu + n})
    return [{"n": 0, "Psi": Psi, "nu": prior_nu}] + out


def sigma_fixed_point(seq, camera, sigmas_m, *, prior_nu=6.0, prior_sigma_m=0.05):
    """Feed one sigma into the x step, read the sigma the R step hands back.

    Where the curve crosses the diagonal, the loop has nothing left to change: that is the
    fixed point the iteration converges to, and it is why x and R cannot be solved apart.
    """
    d = 2
    Psi = np.eye(d) * (prior_sigma_m ** 2) * prior_nu
    out = []
    for sigma in sigmas_m:
        R_bar = {c: np.eye(d) * sigma ** 2 for c in getattr(seq, "cameras", nd.CAMERAS)}
        forward = kalman_filter(seq, R_bar)
        smooth = rts_smoother(seq, forward)
        steps = [k for k in range(seq.n_steps)
                 if seq.camera[k] == camera and forward["used"][k]]
        scatter = np.zeros((d, d))
        for k in steps:
            v = (seq.y[k] - smooth["m"][k]).reshape(d, 1)
            scatter += v @ v.T + smooth["P"][k]
        R_post = (Psi + scatter) / (prior_nu + len(steps))
        out.append(float(np.sqrt(np.trace(R_post) / d)))
    return np.asarray(out)


def path_per_pass(seq, history):
    """The filtered trajectory each pass of the loop believed in."""
    return [kalman_filter(seq, h["R_bar"])["m"] for h in history if "R_bar" in h]


# ------------------------------------------------------------------ making a forecast

def observations_between(seq, lo, hi):
    """The same drive with only the observations in step range [lo, hi) offered.

    Used to hold data back: fit R on the first part of a drive, then ask what it says
    about the part it never saw. Odometry, grid and truth are untouched -- the filter
    still runs over every step, it simply gets no camera reading outside the window.
    """
    out = Sequence.__new__(Sequence)
    out.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray)
                             else list(v) if isinstance(v, list) else v)
                         for k, v in seq.__dict__.items()})
    for k in range(out.n_steps):
        if out.camera[k] is not None and not (lo <= k < hi):
            out.y[k] = np.nan
            out.camera[k] = None
            out.pixel[k] = None
    return out


def other_run(tag, camera, models=None):
    """A different recorded drive, as a one-camera sequence, for testing R somewhere else.

    Three captures predate the notebook run; they are the ones commissioning was fitted
    on. Scoring an R on these asks the only question that matters for deployment: does
    what this drive taught us still hold on a drive it has never seen?
    """
    capture = nd.load_capture(tag, models=models)
    return single_camera(
        Sequence(capture, nd.load_truth(tag), window=nd.route_window(tag)), camera)


def forecast(seq, R_per_camera, *, only=None, sigma_p=PROCESS_SIGMA_PER_SQRT_M,
             initial_sigma=INITIAL_SIGMA_M):
    """What the filter says the next camera reading will be, and what actually arrived.

    Before observation k is used, the filter already implies a distribution for it:

        y_k | everything so far  ~  N(m_k^-,  S_k),   S_k = P_k^- + R

    That is a genuine forecast -- it is formed from readings 1..k-1 only -- and checking
    it needs no ground truth whatever, so it is the one test a robot could run on itself
    while driving. Each row carries the two halves of its score:

        log p(y_k | past) = -1/2 v' S^-1 v   -   1/2 log det(2 pi S)
                            \\___ the miss ___/   \\___ the confidence ___/

    The first term punishes a reading that lands far from where it was predicted; the
    second rewards a narrow forecast. A model can raise the total either by predicting
    more accurately or by claiming to be surer -- which is exactly the ambiguity this
    notebook has to take apart.

    The filter runs UNGATED here so that every candidate R is scored on the same
    observations. A gate admits a different subset for every R, and totals over different
    data are not comparable.
    """
    result = kalman_filter(seq, R_per_camera, sigma_p=sigma_p,
                           initial_sigma=initial_sigma, gate=float("inf"))
    rows = []
    for k in np.flatnonzero(seq.observed):
        if only is not None and not (only[0] <= k < only[1]):
            continue
        S = result["P_pred"][k] + R_per_camera[seq.camera[k]]
        v = seq.y[k] - result["m_pred"][k]
        miss = float(v @ np.linalg.inv(S) @ v)
        confidence = -0.5 * math.log(max(np.linalg.det(2 * math.pi * S), 1e-300))
        rows.append({
            "k": int(k), "t": float(seq.stamps[k] - seq.stamps[0]),
            "camera": seq.camera[k],
            "predicted": result["m_pred"][k].copy(), "S": S,
            "arrived": seq.y[k].copy(), "innovation": v,
            "distance_m": float(np.linalg.norm(v)),
            "sigma_m": float(np.sqrt(np.trace(S) / 2.0)),
            "nis": miss, "miss_term": -0.5 * miss, "confidence_term": confidence,
            "log_p": -0.5 * miss + confidence,
            "truth": seq.truth[k].copy(),          # EVALUATION ONLY, carried for drawing
        })
    return {"rows": rows, "result": result}


def forecast_summary(forecast_out, label=""):
    """Score a set of forecasts. Nothing in here reads ground truth."""
    rows = forecast_out["rows"]
    nis = np.array([r["nis"] for r in rows])
    log_p = np.array([r["log_p"] for r in rows])
    return {
        "label": label, "n": len(rows),
        "median_nis": float(np.median(nis)),
        "inside_95": float(np.mean(nis <= GATE_CHI2_2DOF)),
        "log_p_total": float(log_p.sum()), "log_p_mean": float(log_p.mean()),
        "miss_mean": float(np.mean([r["miss_term"] for r in rows])),
        "confidence_mean": float(np.mean([r["confidence_term"] for r in rows])),
        "distance_cm": float(100 * np.mean([r["distance_m"] for r in rows])),
        "sigma_cm": float(100 * np.mean([r["sigma_m"] for r in rows])),
    }


def what_the_innovation_sees(seq, R_per_camera, *, sigma_p=PROCESS_SIGMA_PER_SQRT_M):
    """Three errors at the same instants: the camera's, the belief's, and the surprise.

    Only the third is available while driving. If a camera leans the same way every frame
    the first two carry that lean and the third does not, because the prediction it is
    measured against has already been pulled onto the leaning readings. That is the whole
    reason a forecast test can pass while the position estimate is wrong.
    """
    result = kalman_filter(seq, R_per_camera, sigma_p=sigma_p, gate=float("inf"))
    innovation, camera_error, belief_error = [], [], []
    for k in np.flatnonzero(seq.observed & np.isfinite(seq.truth[:, 0])):
        innovation.append(seq.y[k] - result["m_pred"][k])
        camera_error.append(seq.y[k] - seq.truth[k])         # EVALUATION ONLY
        belief_error.append(result["m"][k] - seq.truth[k])   # EVALUATION ONLY
    return {"innovation_m": np.asarray(innovation),
            "camera_error_m": np.asarray(camera_error),
            "belief_error_m": np.asarray(belief_error)}


# --------------------------------------------- is the estimator itself at fault?

def model_generated_observations(seq, sigma_m, bias_m=0.0, seed=0):
    """The real drive with the camera readings replaced by ones the model does describe.

    NOT DATA, AND NOT A RESULT. This is a self-test of the code. The timing, the gaps,
    the odometry and the truth are the recorded ones, but every reading is redrawn as
    `y = x_true + bias + N(0, sigma^2 I)`. Because the covariance that generated it is
    then known exactly, whatever `learn_R` hands back can be compared against the right
    answer. Anything concluded from it is a statement about the estimator, never about
    the warehouse.
    """
    rng = np.random.default_rng(seed)
    out = Sequence.__new__(Sequence)
    out.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray)
                             else list(v) if isinstance(v, list) else v)
                         for k, v in seq.__dict__.items()})
    for k in range(out.n_steps):
        if out.camera[k] is None:
            continue
        if not np.isfinite(seq.truth[k, 0]):          # nothing to draw around
            out.y[k] = np.nan
            out.camera[k] = None
            out.pixel[k] = None
            continue
        out.y[k] = seq.truth[k] + np.array([0.0, bias_m]) + rng.normal(0.0, sigma_m, 2)
    return out


def recovery_check(seq, camera, cases, *, seed=0, repeats=6, **learn_kwargs):
    """Hand `learn_R` data drawn from its own model and see whether it returns the truth.

    Repeated over several draws, because one draw of 160 readings carries enough sampling
    noise (a few percent) to read a trend into that is not there.
    """
    rows = []
    for i, (sigma_m, bias_m) in enumerate(cases):
        got = []
        for r in range(repeats):
            made = model_generated_observations(seq, sigma_m, bias_m,
                                                seed=seed + 1000 * r + i)
            R_hat, _, _ = learn_R(made, **learn_kwargs)
            got.append(float(np.sqrt(np.trace(R_hat[camera]) / 2.0)))
        got = np.asarray(got)
        rows.append({"true_sigma_m": float(sigma_m), "bias_m": float(bias_m),
                     "learned_sigma_m": float(got.mean()),
                     "spread_sigma_m": float(got.std()),
                     "ratio": float(got.mean() / sigma_m),
                     "ratio_spread": float(got.std() / sigma_m),
                     "repeats": int(repeats),
                     "n": int(model_generated_observations(seq, sigma_m, bias_m,
                                                           seed=seed).observed.sum())})
    return rows


def identifiability_sweep(seq, camera, sigma_ps, **learn_kwargs):
    """Learn R again under each assumed odometry noise level, and score the pair.

    R and Q both describe how far the readings and the dead-reckoned path are allowed to
    drift apart. One drive constrains their combination, not either alone, so what the
    loop reports as "this camera's noise" is an answer to a question that also contains
    an assumption about the wheels.
    """
    rows = []
    for sigma_p in sigma_ps:
        R_hat, _, _ = learn_R(seq, sigma_p=sigma_p, **learn_kwargs)
        scored = honesty(kalman_filter(seq, R_hat, sigma_p=sigma_p), seq, "")
        rows.append({"sigma_p": float(sigma_p),
                     "learned_sigma_m": float(np.sqrt(np.trace(R_hat[camera]) / 2.0)),
                     "median_nees": scored["median_nees"], "rmse_cm": scored["rmse_cm"]})
    return rows


def learning_gate_ablation(seq, camera, **learn_kwargs):
    """Learn R with the innovation gate on and off, to see whether censoring drives it."""
    rows = []
    for label, gate in (("gate on (as shipped)", GATE_CHI2_2DOF), ("gate off", float("inf"))):
        R_hat, history, _ = learn_R(seq, gate=gate, **learn_kwargs)
        rows.append({"label": label, "gate": gate,
                     "kept": int(history[-1]["counts"][camera]),
                     "offered": int(sum(1 for c in seq.camera if c == camera)),
                     "learned_sigma_m": float(np.sqrt(np.trace(R_hat[camera]) / 2.0)),
                     "sigma_per_pass_m": [h["sigma_m"][camera] for h in history]})
    return rows


# ------------------------------------------------- what the errors actually are

def oracle_noise(seq, camera=None):
    """The camera's error covariance on THIS drive, measured against ground truth.

    EVALUATION ONLY, and it is the reference the notebook judges a learned R against:
    the best a zero-mean model could possibly be told, measured on the very drive being
    filtered, using truth the robot does not have. If a fitted R disagrees with this, the
    fit is wrong about the camera and not merely optimistic.

    Two covariances come back and the difference is the whole point:

      R_spread  the covariance ABOUT the mean error -- pure frame-to-frame scatter
      R_total   the second moment about ZERO, E[(y-x)(y-x)'] -- scatter AND lean together

    A model that says `y = x + zero-mean noise` has no term for a mean and so can never
    subtract one. The honest thing to hand such a model is therefore R_total: the lean has
    to be paid for somewhere and inflated noise is the only pocket it has. R_spread
    describes a quantity that model cannot use.
    """
    cameras = [camera] if camera else list(getattr(seq, "cameras", nd.CAMERAS))
    out = {"R_total": {}, "R_spread": {}, "mean_m": {}, "n": {},
           "sigma_total_m": {}, "sigma_spread_m": {}, "offset_m": {}}
    for cam in cameras:
        rows = np.asarray([seq.y[k] - seq.truth[k] for k in range(seq.n_steps)
                           if seq.camera[k] == cam and np.isfinite(seq.truth[k, 0])])
        if not len(rows):
            continue
        mean = rows.mean(axis=0)
        centred = rows - mean
        out["n"][cam] = int(len(rows))
        out["mean_m"][cam] = mean
        out["R_spread"][cam] = centred.T @ centred / max(len(rows) - 1, 1)
        out["R_total"][cam] = rows.T @ rows / len(rows)
        out["sigma_spread_m"][cam] = float(np.sqrt(np.trace(out["R_spread"][cam]) / 2))
        out["sigma_total_m"][cam] = float(np.sqrt(np.trace(out["R_total"][cam]) / 2))
        out["offset_m"][cam] = float(np.hypot(*mean))
        out["errors_m"] = rows if camera else out.get("errors_m")
    return out


def times_too_confident(median_nees):
    """How much further the truth sits than the filter's own uncertainty allows.

    Median NEES is a squared quantity against a chi-squared reference, which is hard to
    read off an axis. This is the same fact as a plain multiplier: the typical distance
    from the estimate to the truth, divided by the typical distance an honest filter of
    that stated uncertainty would have. 1.0 is honest; 9 means the truth is nine times
    further away than the belief allows for.
    """
    out = np.sqrt(np.asarray(median_nees, dtype=float) / CALIBRATED_MEDIAN_NEES)
    return float(out) if out.ndim == 0 else out


# ------------------------------------------------ the quantity actually maximised

def expected_log_det_iw(Psi, nu, d=2):
    """E[log|R|] under IW(Psi, nu) -- needed by the ELBO, and not log|E[R]|."""
    from scipy.special import digamma
    sign, logdet = np.linalg.slogdet(Psi)
    if sign <= 0:
        return float("nan")
    return logdet - d * math.log(2.0) - sum(
        digamma((nu - i + 1) / 2.0) for i in range(1, d + 1))


def elbo(seq, R_bar, posterior, Psi_prior, nu_prior, *,
         sigma_p=PROCESS_SIGMA_PER_SQRT_M, d=2):
    """The ELBO at a (q(x), q(R)) pair -- the thing coordinate ascent climbs.

    Evaluated right after an x step, where q(x) is exactly p(x | y, R_bar). At that point
    the x-dependent part of the bound collapses onto the exact log marginal likelihood,

        E_q(x)[log p(y|x,R_bar)] - KL(q(x) || p(x)) = log p(y | R_bar),

    which `kalman_filter` already accumulates. Two corrections remain, because the bound
    is taken under the whole of q(R) and not at the point R_bar:

      * the likelihood's log-determinant term wants E[log|R|], not log|R_bar|, and by
        Jensen those differ;
      * the prior on R contributes -KL(q(R) || p(R)).

    So the value is exact rather than a proxy, which matters: the plug-in evidence the
    notebook plots elsewhere is NOT the objective and is not guaranteed to increase,
    whereas this is.

    Ungated throughout. The chi-squared gate is a robustness device bolted on outside the
    variational derivation; including it would score different data at different R.
    """
    forward = kalman_filter(seq, R_bar, sigma_p=sigma_p, gate=float("inf"))
    total = float(forward["log_evidence"])
    for cam in getattr(seq, "cameras", nd.CAMERAS):
        n = sum(1 for k in range(seq.n_steps) if seq.camera[k] == cam)
        if not n or cam not in posterior:
            continue
        Psi_q, nu_q = posterior[cam]["Psi"], posterior[cam]["nu"]
        _, logdet_bar = np.linalg.slogdet(R_bar[cam])
        total -= 0.5 * n * (expected_log_det_iw(Psi_q, nu_q, d) - logdet_bar)
        total -= iw_kl_from_prior(Psi_q, nu_q, Psi_prior, nu_prior, d)
    return total


# ================================================ R as a field, not a number
#
# The projection already says how R's SHAPE and ORIENTATION vary with position: one pixel
# of detector error maps to a floor displacement J du, so a pixel-isotropic error becomes
# R = sigma_px^2 J J' -- anisotropic, oriented, and different at every point. That leaves
# exactly one scalar to estimate, and it can be estimated the same way `learn_R` estimates
# a whole matrix: from the drive, with no ground truth.

def observation_shapes(seq, models, camera=None):
    """M_k = J J' at each reading: R's shape and orientation, before any scale."""
    out = {}
    for k in np.flatnonzero(seq.observed):
        cam = seq.camera[k]
        if camera is not None and cam != camera:
            continue
        J = projection_jacobian(models[cam], *seq.pixel[k])
        if J is not None:
            out[int(k)] = J @ J.T
    return out


def learn_sigma_px(seq, models, *, iterations=12, prior_a=3.0, prior_sigma_px=2.0,
                   sigma_p=PROCESS_SIGMA_PER_SQRT_M, gate=GATE_CHI2_2DOF, d=2):
    """Learn the ONE parameter left once the projection has supplied R's shape.

    Same coordinate ascent as `learn_R`, with `R_k = s M_k` for known `M_k = J J'` and a
    single unknown scale `s = sigma_px^2`. The conjugate prior for a scale on a known
    shape is inverse-gamma rather than inverse-Wishart, and the M step is again closed
    form: with `S_k = (y_k - m^s_k)(...)' + P^s_k`,

        a+ = a + d n / 2,     b+ = b + 1/2 sum_k tr(M_k^-1 S_k)

    and the x step wants the inverted expected precision, `(E[1/s])^-1 = b+ / a+`.

    Twelve hand-fitted numbers become one, and the one that remains is the only part of R
    the geometry cannot supply: how many pixels the detector is wrong by.
    """
    shapes = observation_shapes(seq, models)
    inverses = {k: np.linalg.inv(M) for k, M in shapes.items()}
    prior_b = prior_a * (prior_sigma_px ** 2)          # so that E[s] starts at sigma_px^2
    s_bar = prior_b / prior_a
    history = []

    for _ in range(iterations):
        R_bar = {c: np.eye(d) * s_bar for c in getattr(seq, "cameras", nd.CAMERAS)}
        forward = kalman_filter_jacobian_R(seq, math.sqrt(s_bar), R_bar, models,
                                           sigma_p=sigma_p, gate=gate)
        smooth = _smoother_for_jacobian_R(seq, forward)
        total, used = 0.0, 0
        for k, M_inv in inverses.items():
            if not forward["used"][k]:
                continue
            v = (seq.y[k] - smooth["m"][k]).reshape(d, 1)
            total += float(np.trace(M_inv @ (v @ v.T + smooth["P"][k])))
            used += 1
        a_post = prior_a + 0.5 * d * used
        b_post = prior_b + 0.5 * total
        s_bar = b_post / a_post
        history.append({"sigma_px": math.sqrt(s_bar), "a": a_post, "b": b_post, "n": used})
    return math.sqrt(s_bar), history


def _smoother_for_jacobian_R(seq, forward):
    """RTS pass over a filter whose R changed at every step."""
    n = seq.n_steps
    ms, Ps = forward["m"].copy(), forward["P"].copy()
    for k in range(n - 2, -1, -1):
        P_next_pred = forward["P_pred"][k + 1]
        G = forward["P"][k] @ np.linalg.inv(P_next_pred)
        ms[k] = forward["m"][k] + G @ (ms[k + 1] - forward["m_pred"][k + 1])
        Ps[k] = forward["P"][k] + G @ (Ps[k + 1] - P_next_pred) @ G.T
        Ps[k] = 0.5 * (Ps[k] + Ps[k].T)
    return {"m": ms, "P": Ps}


def R_field(models, camera, sigma_px, *, extent=(-5.5, 5.0, -5.0, 4.7), nx=90, ny=80):
    """R over the whole floor, as size, aspect and orientation.

    Computable at every point from the camera model alone, including points no drive has
    ever visited -- which is precisely what a planner scoring a candidate pose needs.
    """
    x0, x1, y0, y1 = extent
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    model = models[camera]
    size = np.full((ny, nx), np.nan)
    aspect = np.full((ny, nx), np.nan)
    angle = np.full((ny, nx), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            u, v, in_frame = model.world_to_pixel(float(x), float(y), 0.0)
            if not in_frame:
                continue
            J = projection_jacobian(model, u, v)
            if J is None:
                continue
            R = (sigma_px ** 2) * (J @ J.T)
            values, vectors = np.linalg.eigh(R)
            values = np.maximum(values, 1e-18)
            size[j, i] = math.sqrt(float(np.trace(R) / 2))
            aspect[j, i] = math.sqrt(values[-1] / values[0])
            angle[j, i] = math.degrees(math.atan2(vectors[1, -1], vectors[0, -1]))
    return {"x": xs, "y": ys, "size_m": size, "aspect": aspect, "angle_deg": angle,
            "camera": camera, "sigma_px": sigma_px}


# ============================================ R learned as a FIELD over the floor
#
# The blend in `reliability.covariance_mapping` turns one scalar trust into a covariance
# by interpolating between two asserted endpoints. That can move R's size and nothing
# else. What follows estimates R as a field instead: the projection supplies the shape and
# orientation at every point in closed form, and the SCALE is learned locally from the
# drives, together with a posterior that says how much the drives actually pinned it down
# at each point. Where nothing drove, the field returns the prior and says so -- which is
# the property a planner scoring an unvisited pose needs and a two-endpoint blend cannot
# express.

def field_observations(tags, camera, models, *, use_truth=True, sigma_p=PROCESS_SIGMA_PER_SQRT_M):
    """Per-reading (position, M = J J', residual outer product) over one or more drives.

    `use_truth=True` measures residuals against ground truth: an offline calibration, the
    same class of thing as commissioning. `use_truth=False` measures them against the
    smoothed trajectory instead -- no ground truth anywhere -- which is what a robot could
    do for itself, and which notebook 1 shows shrinks.
    """
    rows = []
    for tag in tags:
        seq = other_run(tag, camera, models=models)
        reference = seq.truth
        extra = None
        if not use_truth:
            forward = kalman_filter(seq, {camera: np.eye(2) * 0.05 ** 2}, sigma_p=sigma_p)
            smooth = rts_smoother(seq, forward)
            reference, extra = smooth["m"], smooth["P"]
        for k in np.flatnonzero(seq.observed):
            if seq.camera[k] != camera or not np.isfinite(reference[k][0]):
                continue
            J = projection_jacobian(models[camera], *seq.pixel[k])
            if J is None:
                continue
            rows.append({"tag": tag, "k": int(k),
                         "at": np.asarray(seq.truth[k] if use_truth else reference[k]),
                         "M": J @ J.T,
                         "S_root": np.asarray(seq.y[k] - reference[k]),
                         "P_extra": None if extra is None else extra[k]})
    return rows


def learn_R_field(rows, points, *, length_scale_m=1.5, prior_a=3.0, prior_sigma_px=2.0,
                  d=2):
    """Two fields, learned locally: where the camera LEANS, and how much it SCATTERS.

    A first attempt at this learned one field and got 9 to 13 "pixels" of noise, which is
    nonsense for a detector that places a box to about a pixel. The cause is the lean: a
    residual measured about zero is mostly bias, and dividing a bias by the local
    centimetres-per-pixel turns it into a large apparent pixel error that shrinks with
    range because the divisor grows. So the mean has to come out first, and because the
    lean itself depends on where the robot is, the mean is a field too.

    At each point, with Gaussian kernel weights w_k(x) = exp(-|x - x_k|^2 / 2 l^2):

        lean(x)  = sum_k w_k v_k / sum_k w_k                     (metres, a 2-vector)
        a(x)     = a0 + (d/2) sum_k w_k
        b(x)     = b0 + (1/2) sum_k w_k tr(M_k^-1 e_k e_k'),   e_k = v_k - lean(x)

    with the inverse-gamma posterior on the scale giving a credible interval for free:
    wide where no drive went, narrow where many did. `n_effective` is sum_k w_k, the
    amount of data actually standing behind each point.
    """
    from scipy.stats import invgamma

    at = np.asarray([r["at"] for r in rows])
    residual = np.asarray([r["S_root"] for r in rows])          # v_k, metres
    M_inv = [np.linalg.inv(r["M"]) for r in rows]
    extra = [r.get("P_extra") for r in rows]
    prior_b = prior_a * (prior_sigma_px ** 2)
    points = np.asarray(points, dtype=float).reshape(-1, 2)

    n = len(points)
    sigma, lo, hi, n_eff = (np.full(n, np.nan) for _ in range(4))
    lean = np.full((n, 2), np.nan)
    for i, p in enumerate(points):
        w = np.exp(-0.5 * np.sum((at - p) ** 2, axis=1) / length_scale_m ** 2)
        total = float(w.sum())
        centre = (w @ residual) / max(total, 1e-12)
        lean[i] = centre
        quad = 0.0
        for k, wk in enumerate(w):
            if wk < 1e-6:
                continue
            e = (residual[k] - centre).reshape(2, 1)
            S = e @ e.T + (extra[k] if extra[k] is not None else 0.0)
            quad += wk * float(np.trace(M_inv[k] @ S))
        a = prior_a + 0.5 * d * total
        b = prior_b + 0.5 * quad
        sigma[i] = math.sqrt(b / a)
        lo[i] = math.sqrt(invgamma.ppf(0.05, a=a, scale=b))
        hi[i] = math.sqrt(invgamma.ppf(0.95, a=a, scale=b))
        n_eff[i] = total
    return {"points": points, "sigma_px": sigma, "sigma_px_lo": lo, "sigma_px_hi": hi,
            "lean_m": lean, "n_effective": n_eff, "length_scale_m": length_scale_m,
            "prior_sigma_px": prior_sigma_px, "n_readings": len(rows)}


def R_field_learned(rows, models, camera, *, extent=(-5.5, 5.0, -5.0, 4.7), nx=64, ny=58,
                    **kwargs):
    """The learned scale field crossed with the projection's shape, over the whole floor."""
    x0, x1, y0, y1 = extent
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    grid = np.array([(x, y) for y in ys for x in xs])
    learned = learn_R_field(rows, grid, **kwargs)
    model = models[camera]

    size = np.full((ny, nx), np.nan)
    aspect = np.full((ny, nx), np.nan)
    angle = np.full((ny, nx), np.nan)
    for idx, (x, y) in enumerate(grid):
        u, v, in_frame = model.world_to_pixel(float(x), float(y), 0.0)
        if not in_frame:
            continue
        J = projection_jacobian(model, u, v)
        if J is None:
            continue
        R = (learned["sigma_px"][idx] ** 2) * (J @ J.T)
        values, vectors = np.linalg.eigh(R)
        j, i = divmod(idx, nx)
        size[j, i] = math.sqrt(float(np.trace(R) / 2))
        aspect[j, i] = math.sqrt(max(values[-1], 1e-18) / max(values[0], 1e-18))
        angle[j, i] = math.degrees(math.atan2(vectors[1, -1], vectors[0, -1]))
    shape = (ny, nx)
    return {"x": xs, "y": ys, "size_m": size, "aspect": aspect, "angle_deg": angle,
            "sigma_px": learned["sigma_px"].reshape(shape),
            "sigma_px_lo": learned["sigma_px_lo"].reshape(shape),
            "sigma_px_hi": learned["sigma_px_hi"].reshape(shape),
            "n_effective": learned["n_effective"].reshape(shape),
            "lean_m": learned["lean_m"].reshape(ny, nx, 2),
            "camera": camera, "learned": learned}


def R_at(rows, models, camera, x, y, **kwargs):
    """The learned R at one floor point: shape from the projection, scale from the field."""
    u, v, in_frame = models[camera].world_to_pixel(float(x), float(y), 0.0)
    if not in_frame:
        return None
    J = projection_jacobian(models[camera], u, v)
    if J is None:
        return None
    learned = learn_R_field(rows, [(x, y)], **kwargs)
    return {"R": (learned["sigma_px"][0] ** 2) * (J @ J.T),
            "sigma_px": float(learned["sigma_px"][0]),
            "sigma_px_lo": float(learned["sigma_px_lo"][0]),
            "sigma_px_hi": float(learned["sigma_px_hi"][0]),
            "lean_m": np.asarray(learned["lean_m"][0]),
            "n_effective": float(learned["n_effective"][0])}


# ------------------------------------------- learning R once the lean has been removed

def right_order_comparison(seq, models, camera, *, floor_m=0.025,
                           sigma_p=PROCESS_SIGMA_PER_SQRT_M, iterations=12):
    """Learn R before and after the observation function is corrected.

    The rest of this module measures a learned R against `oracle_noise`. This puts the two
    covariances that function returns to a different use, as a diagnostic of whether
    learning R is even a well-posed question:

        R_total   what a zero-mean model NEEDS  (scatter and lean)
        R_spread  what the learner can SEE      (scatter about the mean)

    Their ratio is how much lean the model has no term for. Nothing about the estimator
    can close that gap -- innovations are computed about a trajectory that has already
    absorbed the lean, so it is missing from the only quantity the fit is scored on. If
    the gap is the reason learning R shrinks, then removing the lean from the OBSERVATION
    FUNCTION should collapse the ratio to one and the same loop should start working, with
    nothing about the loop changed.

    `GeometryCorrected` supplies the correction: the robot's own meshes projected at the
    filter's running estimate, with heading from odometry. It needs an R to run, and using
    the learned one would confound the two steps, so the provisional pass uses the oracle
    and only the FINAL arms use each R on its own corrected sequence.

    The floor is one isotropic scalar added to R. The learner cannot produce it: it is the
    residual lean the mesh correction leaves behind, which is a mean and not a scatter.
    """
    heading = heading_from_odometry(seq)

    def scored(sequence, R):
        result = kalman_filter(sequence, R, sigma_p=sigma_p)
        judged = honesty(result, sequence, "arm")
        return {"nees": judged["median_nees"], "rmse_cm": judged["rmse_cm"]}

    def size_cm(R):
        return 100.0 * math.sqrt(np.trace(R) / 2.0)

    oracle_raw = oracle_noise(seq)
    learned_raw, _, _ = learn_R(seq, iterations=iterations, sigma_p=sigma_p)

    provisional = GeometryCorrected(seq, models, oracle_raw["R_total"], heading,
                                    sigma_p=sigma_p)
    oracle_cor = oracle_noise(provisional)
    learned_cor, _, _ = learn_R(provisional, iterations=iterations, sigma_p=sigma_p)
    floored = {c: R + np.eye(2) * floor_m**2 for c, R in learned_cor.items()}

    arms = {
        "raw + learned": scored(seq, learned_raw),
        "raw + oracle": scored(seq, oracle_raw["R_total"]),
        "corrected + learned": scored(
            GeometryCorrected(seq, models, learned_cor, heading, sigma_p=sigma_p),
            learned_cor),
        "corrected + oracle": scored(
            GeometryCorrected(seq, models, oracle_cor["R_total"], heading, sigma_p=sigma_p),
            oracle_cor["R_total"]),
        "corrected + learned + floor": scored(
            GeometryCorrected(seq, models, floored, heading, sigma_p=sigma_p), floored),
    }

    def side(oracle, learned):
        needed = size_cm(oracle["R_total"][camera])
        visible = size_cm(oracle["R_spread"][camera])
        return {"lean_cm": 100.0 * float(oracle["offset_m"][camera]),
                "needed_cm": needed, "visible_cm": visible,
                "learned_cm": size_cm(learned[camera]),
                "cannot_express": needed / visible if visible else float("nan")}

    return {"camera": camera, "floor_cm": 100.0 * floor_m,
            "raw": side(oracle_raw, learned_raw),
            "corrected": side(oracle_cor, learned_cor),
            "arms": arms,
            "n_corrected": int(provisional.n_corrected),
            "sigma_px_raw": float(learn_sigma_px(seq, models, sigma_p=sigma_p,
                                                 iterations=iterations)[0]),
            "sigma_px_corrected": float(learn_sigma_px(provisional, models, sigma_p=sigma_p,
                                                       iterations=iterations)[0]),
            "R_learned_raw": {c: R.copy() for c, R in learned_raw.items()},
            "R_learned_corrected": {c: R.copy() for c, R in learned_cor.items()},
            "oracle_raw": oracle_raw,
            "oracle_corrected": oracle_cor,
            "corrected_sequence": provisional}


# ============================== what a MIScalibrated camera does, and where it belongs

def miscalibrated(models, camera, *, dx=0.0, dy=0.0, dz=0.0, dpitch_deg=0.0,
                  dyaw_deg=0.0):
    """A copy of the camera set with a deliberate calibration error.

    CONTROLLED ABLATION, clearly labelled as such: the recorded pixels are untouched real
    data, and only our BELIEF about where the camera is moves. That is exactly what a
    miscalibration is -- the world is right and the model is wrong -- and it is the one
    thing these captures cannot contain on their own, because the projection is parsed
    from the same world file Gazebo renders from, so the calibration error in them is
    identically zero.
    """
    from unav_common.camera_model import ObliqueCameraModel

    root = ET.parse(nd.ACTIVE.world_sdf).getroot()
    include_name = nd.ACTIVE.model_includes[camera]
    pose_text = None
    for include in root.findall(".//include"):
        name = (include.findtext("name") or "").strip()
        uri = (include.findtext("uri") or "").strip()
        if include_name in {name, uri.removeprefix("model://").split("/", 1)[0]}:
            pose_text = include.findtext("pose")
            break
    if pose_text is None:
        raise RuntimeError(f"no posed include {include_name!r} in {nd.ACTIVE.world_sdf}")
    x, y, z, _roll, pitch, yaw = (float(v) for v in pose_text.split())
    x, y, z = x + dx, y + dy, z + dz
    pitch, yaw = pitch + math.radians(dpitch_deg), yaw + math.radians(dyaw_deg)
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
               -math.sin(pitch))
    scale = -z / forward[2]
    out = dict(models)
    out[camera] = ObliqueCameraModel(
        cam_pos=(x, y, z),
        look_at=(x + scale * forward[0], y + scale * forward[1], 0.0),
        img_width=models[camera].img_width, img_height=models[camera].img_height,
        fov_h_rad=models[camera].fov_h_rad)
    return out


def reprojected(seq, models):
    """The same recorded pixels, back-projected through a different camera model.

    Nothing about the detector changes -- the pixels are what the real camera produced.
    Only the map from pixel to floor moves, which is what a calibration error is.
    """
    out = Sequence.__new__(Sequence)
    out.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray)
                             else list(v) if isinstance(v, list) else v)
                         for k, v in seq.__dict__.items()})
    for k in range(out.n_steps):
        cam = out.camera[k]
        if cam is None or out.pixel[k] is None:
            continue
        point = models[cam].pixel_to_world(*out.pixel[k])
        if point is None or not all(np.isfinite(point)):
            out.y[k] = np.nan
            out.camera[k] = None
            out.pixel[k] = None
        else:
            out.y[k] = np.asarray(point, dtype=float)
    return out


def calibration_sensitivity(models, camera, cases, probes):
    """How far a given calibration error moves the floor point, at chosen ranges."""
    base = models[camera]
    rows = []
    for label, kwargs in cases:
        wrong = miscalibrated(models, camera, **kwargs)[camera]
        moved = []
        for x, y in probes:
            u, v, in_frame = base.world_to_pixel(float(x), float(y), 0.0)
            landing = wrong.pixel_to_world(u, v) if in_frame else None
            moved.append(float(np.linalg.norm(np.asarray(landing) - np.array([x, y])))
                         if landing is not None else float("nan"))
        rows.append({"label": label, "moved_m": moved})
    ranges = [float(np.hypot(x - base.cam_pos[0], y - base.cam_pos[1]))
              for x, y in probes]
    return {"rows": rows, "probes": probes, "ranges_m": ranges,
            "focal_px": (base.img_width / 2) / math.tan(base.fov_h_rad / 2)}


# ================================ the bias, predicted rather than absorbed into R

def bias_and_noise(seq, camera, models, truth_table):
    """Split the camera's error into the part geometry predicts and the part left over."""
    ks, obs, tru, yaw = [], [], [], []
    for k in np.flatnonzero(seq.observed):
        hit = nd.truth_at(truth_table, float(seq.stamps[k]))
        if hit is None or seq.camera[k] != camera or not np.isfinite(seq.truth[k, 0]):
            continue
        ks.append(int(k)); obs.append(seq.y[k]); tru.append(seq.truth[k]); yaw.append(hit[2])
    obs, tru, yaw = np.asarray(obs), np.asarray(tru), np.asarray(yaw)
    residual = obs - tru
    mean = residual.mean(axis=0)
    scatter = residual - mean
    predicted = np.asarray([
        nm_landing if (nm_landing := silhouette_bottom(models[camera], float(x), float(y),
                                                       float(t))) is not None
        else (np.nan, np.nan) for (x, y), t in zip(tru, yaw)], dtype=float)
    return {"steps": ks, "residual_m": residual, "mean_m": mean, "scatter_m": scatter,
            "after_prediction_m": obs - predicted,
            "total_cm": float(100 * np.median(np.linalg.norm(residual, axis=1))),
            "scatter_cm": float(100 * np.sqrt((scatter ** 2).sum(1).mean() / 2)),
            "after_cm": float(100 * np.median(np.linalg.norm(obs - predicted, axis=1)))}


def prediction_transfer(drives, camera, models):
    """The geometric prediction on every drive, with the heading it is given varied.

    Four columns and the last is the control: if the pose-dependence were not doing the
    work, assuming a heading of zero would score the same as using the real one.
    """
    rows, pooled = [], {k: [] for k in ("none", "truth", "odometry", "heading zero")}
    for tag in drives:
        capture = nd.load_capture(tag, models=models)
        truth_table = nd.load_truth(tag)
        seq = Sequence(capture, truth_table, window=nd.route_window(tag))
        odom_heading = heading_from_odometry(seq)
        got = {k: [] for k in pooled}
        for k in np.flatnonzero(seq.observed):
            hit = nd.truth_at(truth_table, float(seq.stamps[k]))
            if hit is None or seq.camera[k] != camera or not np.isfinite(seq.truth[k, 0]):
                continue
            x, y = seq.truth[k]
            got["none"].append(float(np.linalg.norm(seq.y[k] - np.array([x, y]))))
            supplies = {"truth": hit[2],
                        "odometry": odom_heading[k] if k < len(odom_heading) else np.nan,
                        "heading zero": 0.0}
            for key, heading in supplies.items():
                if heading is None or not np.isfinite(heading):
                    continue
                landing = silhouette_bottom(models[camera], float(x), float(y), float(heading))
                if landing is not None:
                    got[key].append(float(np.linalg.norm(seq.y[k] - np.asarray(landing))))
        for key in pooled:
            pooled[key].extend(got[key])
        rows.append({"drive": tag, "n": len(got["none"]),
                     **{k: 100 * float(np.median(v)) if v else float("nan")
                        for k, v in got.items()}})
    rows.append({"drive": "POOLED", "n": len(pooled["none"]),
                 **{k: 100 * float(np.median(v)) for k, v in pooled.items()}})
    return rows


def calibration_arms(seq, models, camera, *, degrees=(0.0, 0.1, 0.5), gate=float("inf"),
                     sigma_b_prior=0.20, sigma_b_walk=0.002):
    """Four ways of dealing with the error, under correct and incorrect calibration.

    Ungated throughout, deliberately: a correctly modelled observation function makes the
    innovations small, and a chi-squared gate whose threshold was set in the biased regime
    then rejects good readings and runs away. That is a real effect worth its own
    measurement, not something to let quietly advantage one arm over another.
    """
    table = []
    for degree in degrees:
        believed = models if degree == 0 else miscalibrated(models, camera,
                                                            dpitch_deg=degree)
        here = seq if degree == 0 else reprojected(seq, believed)
        oracle = oracle_noise(here, camera)
        R_scatter = {camera: oracle["R_spread"][camera]}
        R_total = {camera: oracle["R_total"][camera]}
        heading = heading_from_odometry(here)
        arms = [("R = the true scatter", here, R_scatter),
                ("R inflated to cover it all", here, R_total),
                ("observation function corrected",
                 GeometryCorrected(here, believed, R_total, heading), R_scatter)]
        for label, sequence, R in arms:
            scored = honesty(kalman_filter(sequence, R, gate=gate), sequence, label)
            table.append({"degrees": degree, "arm": label,
                          "median_nees": scored["median_nees"],
                          "rmse_cm": scored["rmse_cm"]})
        offset = offset_state_filter(here, R_scatter, sigma_b_prior=sigma_b_prior,
                                     sigma_b_walk=sigma_b_walk, gate=gate)
        scored = score_offset_filter(offset, here, "offset")
        table.append({"degrees": degree, "arm": "offset carried in the state",
                      "median_nees": scored["median_nees"], "rmse_cm": scored["rmse_cm"]})
    return table


def gate_after_correction(seq, models, camera, R_scatter, R_total, heading):
    """What the innovation gate does once the observation function is right."""
    corrected = GeometryCorrected(seq, models, R_total, heading)
    out = []
    for label, gate in (("gate on, as deployed", GATE_CHI2_2DOF),
                        ("gate off", float("inf"))):
        result = kalman_filter(corrected, R_scatter, gate=gate)
        scored = honesty(result, corrected, label)
        out.append({"label": label, "used": int(result["used"].sum()),
                    "offered": int(corrected.observed.sum()),
                    "median_nees": scored["median_nees"], "rmse_cm": scored["rmse_cm"]})
    return out
