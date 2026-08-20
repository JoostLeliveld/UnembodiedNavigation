#!/usr/bin/env python3
"""exp1: does the factorized measurement filter stay HONEST when a camera lies?

The goal here is NOT the lowest localization error. The goal is that the robot's
stated uncertainty tracks its real error, so it never claims confidence it has not
earned -- because unearned confidence is what drives a robot into a shelf.

The input residual is recorded, not injected: under the now-retired v2 calibration,
camera C carries a historical **+76.9 mm signed lateral bias** on these confounded driving
captures. This is mechanism evidence, not current camera accuracy. That bias is invisible to camera C
itself. It is only detectable because three other cameras disagree with it, which
is what makes this a multi-camera result rather than a filtering trick.

Three arms. Identical data, identical prediction step, identical everything except
the rule for turning a detection into a belief update:

    A0  trust everything     one fixed R for every camera, every detection
                             (the state of practice: static per-camera calibration
                             + fuse whatever arrives)
    A1  hard gate            same fixed R, plus a chi-square innovation gate --
                             the classical robust answer: reject the outlier
    A2  factorized (OURS)    per-camera conditional covariance R_cond, a
                             velocity-dependent timing term, and outliers treated
                             as a SEPARATE PROCESS via a contaminated-Gaussian
                             responsibility -- a soft, probabilistic down-weight
                             rather than a hard accept/reject

A2 is the measurement model from ``reliability.observation_model`` driving the
filter: every covariance it forms comes from ``innovation_covariance`` /
``state_projection_covariance`` / ``time_sync_covariance``, and the outlier
responsibility is the two-component mixture behind ``contaminated_gaussian_nll``.
No algebra is re-implemented here.

Reported together, always:

  * honesty   -- NEES, and the fraction of time the true error escapes the stated
                 95 % ellipse. A filter that just inflates its covariance passes
                 this trivially, which is why...
  * sharpness -- the mean stated sigma is reported next to it, and so is RMSE.
                 Being uncertain is not a virtue; being HONESTLY uncertain is.

Ground truth is EVALUATION-ONLY: it scores the arms and never enters a filter.

Outputs -> logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting/
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

from reliability.health_ewma import (  # noqa: E402
    InnovationHealthConfig,
    InnovationHealthMonitor,
)
from reliability.observation_model import (  # noqa: E402
    innovation_covariance,
    state_projection_covariance,
    time_sync_covariance,
)
import rcond_common as rc  # noqa: E402

OUT = REPO / "logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting"

H = ((1.0, 0.0), (0.0, 1.0))          # camera measures position directly
PROCESS_SIGMA_PER_SQRT_M = 0.04        # odometry drift, from the R_cond study sweep
INITIAL_SIGMA_M = 0.05
FIXED_R_SIGMA_M = 0.08                 # the single static R the baselines use
SIGMA_TAU_S = 0.05                     # camera/odometry timestamp uncertainty
NIS_GATE_CHI2 = 5.991                  # chi-square, 2 dof, 95 %
OUTLIER_PRIOR = 0.10                   # p(this detection is not a clean measurement)
OUTLIER_SIGMA_M = 0.50                 # broad component: a wrong box / stale frame

#: Per-camera conditional accuracy, GT-free, from the operational R_cond study
#: (logs/studies/operational_residual_rcond/exp2_operational_rcond, held-out
#: referenced, sigma per axis in metres). These are MEASURED, not tuned.
R_COND_SIGMA_M = {
    "camera_A": 0.0267,
    "camera_B": 0.0127,
    "camera_C": 0.0250,
    "camera_D": 0.0224,
}

#: Per-camera RESIDUAL BIAS BUDGET: how much systematic error is known to survive
#: commissioning, per camera, in metres. Measured (oracle bias norms from
#: logs/studies/operational_residual_rcond/exp2_operational_rcond) — not tuned.
#:
#: This is the bridge between the calibration work and the filter. R_cond measures
#: SCATTER only; a repeated historical-v2 lean has an error the scatter cannot explain.
#: Rather than trying to remove that error, the filter ACCOUNTS for it: the leftover
#: bias enters as the R_model term of the innovation decomposition, so a leaning
#: camera is honestly modelled as a less informative one. Caveat, documented in
#: reliability.observation_model.calibration_covariance: a bias is shared across
#: frames, not resampled, so this is the practical per-frame treatment of a
#: temporally correlated error, not an exact one.
RESIDUAL_BIAS_M = {
    "camera_A": 0.0071,
    "camera_B": 0.0123,
    "camera_C": 0.0768,
    "camera_D": 0.0328,
}

#: Each arm is a combination of independent mechanisms, so any one of them can be
#: switched off and re-measured. Name-based branching made that impossible and hid
#: which mechanism was actually doing the work.
#:   per_camera_r  measured R_cond + timing term + residual-bias budget
#:   gate          hard chi-square innovation rejection
#:   outlier       contaminated-Gaussian soft responsibility
#:   cross_check   leave-one-camera-out health, feeding covariance inflation
#:   floor         "per_camera" | "pooled" | None -- posterior covariance floor
ARM_SPEC = {
    "A0_trust_everything":    dict(per_camera_r=False, gate=False, outlier=False,
                                   cross_check=False, floor=None),
    "A1_hard_gate":           dict(per_camera_r=False, gate=True, outlier=False,
                                   cross_check=False, floor=None),
    "A2_factorized":          dict(per_camera_r=True, gate=False, outlier=True,
                                   cross_check=False, floor=None),
    "A3_network_consistency": dict(per_camera_r=True, gate=False, outlier=False,
                                   cross_check=True, floor=None),
    "A4_correlation_floor":   dict(per_camera_r=True, gate=False, outlier=False,
                                   cross_check=True, floor="per_camera"),
    # ---- ablations of A4: which mechanism is actually carrying the result? ----
    "X1_floor_only":          dict(per_camera_r=True, gate=False, outlier=False,
                                   cross_check=False, floor="per_camera"),
    "X2_pooled_floor":        dict(per_camera_r=True, gate=False, outlier=False,
                                   cross_check=True, floor="pooled"),
}
ARMS = tuple(ARM_SPEC)
#: Tick labels must say what the arm DOES, not just its code -- these figures are read
#: by people who have not read this file (see CLAUDE.md, "Figures must stand alone").
#: fig_s1 draws HORIZONTAL bars precisely so these can be full phrases: seven of them
#: as vertical-bar tick labels collide at any font size that stays legible.
ARM_LABEL = {
    "A0_trust_everything": "A0   trust every camera equally",
    "A1_hard_gate": "A1   + reject wild disagreements",
    "A2_factorized": "A2   + measure each camera's noise",
    "A3_network_consistency": "A3   + check each against the others",
    "A4_correlation_floor": "A4   + uncertainty floor   ← FULL METHOD",
    "X1_floor_only": "X1   floor only, no cross-checking",
    "X2_pooled_floor": "X2   one shared floor for all cameras",
}
#: Legend and results-table wording. Each entry says the MECHANISM in words, with the
#: familiar name of the technique in brackets for readers who know it.
ARM_LONG = {
    "A0_trust_everything": "A0  trust every camera equally (what is normally done)",
    "A1_hard_gate": "A1  + reject sightings that disagree wildly (classical robust filtering)",
    "A2_factorized": "A2  + measure how noisy each camera really is",
    "A3_network_consistency": "A3  + judge each camera against a position built WITHOUT it",
    "A4_correlation_floor": "A4  + a floor on each camera's uncertainty that repeated looks "
                            "cannot shrink  (full method)",
    "X1_floor_only": "X1  the floor alone, without judging cameras against each other",
    "X2_pooled_floor": "X2  one shared floor for all four cameras instead of one each",
}
ARM_COLOR = {"A0_trust_everything": "#D55E00", "A1_hard_gate": "#E69F00",
             "A2_factorized": "#CC79A7", "A3_network_consistency": "#56B4E9",
             "A4_correlation_floor": "#0072B2", "X1_floor_only": "#999999",
             "X2_pooled_floor": "#666666"}
#: One number for every camera, for the X2 ablation: the network-wide mean residual
#: bias. If this does as well as the per-camera floor, "per-camera" is decoration.
POOLED_BIAS_M = 0.0322
#: Health below this starts inflating that camera's covariance; the inflation is
#: continuous in health, so there is no accept/reject cliff.
HEALTH_INFLATE_BELOW = 0.9
HEALTH_MAX_INFLATION = 25.0


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


# --------------------------------------------------------------------- the filter


def _gaussian_pdf_2d(residual: np.ndarray, cov: np.ndarray) -> float:
    determinant = float(np.linalg.det(cov))
    if determinant <= 0.0:
        return 0.0
    quadratic = float(residual @ np.linalg.solve(cov, residual))
    return math.exp(-0.5 * quadratic) / (2.0 * math.pi * math.sqrt(determinant))


def measurement_update(arm: str, mean, cov, detection, velocity, camera_id, monitor=None):
    """One camera detection -> posterior belief. The ONLY thing that differs by arm.

    Returns ``(mean, cov, accepted, diagnostics)``. ``accepted`` is False only for
    the hard gate; the other arms never reject outright -- they price the detection
    and let the price do the work.
    """

    measurement = np.asarray(detection.world, dtype=float)
    innovation = measurement - mean
    state_term = state_projection_covariance(H, cov.tolist())

    spec = ARM_SPEC[arm]
    if spec["per_camera_r"]:
        sigma = R_COND_SIGMA_M[camera_id]
        r_cond = ((sigma**2, 0.0), (0.0, sigma**2))
        # Timing: an uncertain exposure instant images a state displaced by v*dt.
        # This is the reason a fast pass is noisier than a slow one at the same place.
        r_time = time_sync_covariance(H, velocity.tolist(), SIGMA_TAU_S)
        bias = RESIDUAL_BIAS_M[camera_id]
        r_model = ((bias**2, 0.0), (0.0, bias**2))
        total = np.asarray(
            innovation_covariance(state_term, r_cond, r_time, r_model), dtype=float
        )
        measurement_cov = (np.asarray(r_cond, dtype=float)
                           + np.asarray(r_time, dtype=float)
                           + np.asarray(r_model, dtype=float))
    else:
        r_fixed = ((FIXED_R_SIGMA_M**2, 0.0), (0.0, FIXED_R_SIGMA_M**2))
        total = np.asarray(innovation_covariance(state_term, r_fixed), dtype=float)
        measurement_cov = np.asarray(r_fixed, dtype=float)

    nis = float(innovation @ np.linalg.solve(total, innovation))

    if spec["gate"] and nis > NIS_GATE_CHI2:
        return mean, cov, False, {"nis": nis, "responsibility": 0.0, "health": 1.0}

    responsibility = 1.0
    health = 1.0
    if spec["cross_check"]:
        # THE multi-camera step, and the subtle part.
        #
        # Judging camera c by its innovation against the FULL belief does not
        # work, and gets it exactly backwards: a biased camera drags the belief
        # onto its own wrong answer, after which its innovations are small (it
        # looks healthy) and the honest cameras disagree with the dragged belief
        # (they look faulty). The operational R_cond study measured this
        # self-confirmation directly -- camera A's error was understated 4.2x by
        # a reference its own measurements anchored.
        #
        # So ``health`` here is supplied by the caller from a LEAVE-ONE-CAMERA-OUT
        # belief: camera c is checked against what the rest of the network thinks.
        # That comparison does not exist for a single camera, which is what makes
        # this a network capability rather than a filtering trick.
        health = monitor if monitor is not None else 1.0
        if health < HEALTH_INFLATE_BELOW:
            inflation = min(HEALTH_MAX_INFLATION, HEALTH_INFLATE_BELOW / max(health, 1e-3))
            measurement_cov = measurement_cov * inflation
            total = np.asarray(state_term, dtype=float) + measurement_cov

    if spec["outlier"]:
        # Outliers are a different process, not a fat tail of this one: a wrong
        # box, a stale frame, a partial occlusion. Their posterior responsibility
        # softly down-weights the update instead of an accept/reject cliff.
        inlier = _gaussian_pdf_2d(innovation, total)
        outlier_cov = total + np.eye(2) * OUTLIER_SIGMA_M**2
        outlier = _gaussian_pdf_2d(innovation, outlier_cov)
        denominator = (1.0 - OUTLIER_PRIOR) * inlier + OUTLIER_PRIOR * outlier
        responsibility = ((1.0 - OUTLIER_PRIOR) * inlier / denominator
                          if denominator > 0.0 else 0.0)
        if responsibility < 1.0e-6:
            return mean, cov, True, {"nis": nis, "responsibility": responsibility,
                                     "health": health}
        # A partially-trusted measurement is one with proportionally more noise.
        measurement_cov = measurement_cov / max(responsibility, 1.0e-6)
        total = np.asarray(state_term, dtype=float) + measurement_cov

    diagnostics = {"nis": nis, "responsibility": responsibility, "health": health}
    gain = cov @ np.linalg.inv(total)
    mean = mean + gain @ innovation
    identity = np.eye(2)
    a = identity - gain
    cov = a @ cov @ a.T + gain @ measurement_cov @ gain.T   # Joseph form
    cov = 0.5 * (cov + cov.T)
    if spec["floor"]:
        # A per-camera bias is TEMPORALLY CORRELATED: the same offset every frame,
        # not a fresh draw. Repeated looks from that camera are therefore not
        # independent evidence, but a Kalman filter treats them as if they were and
        # shrinks P as 1/n toward zero -- while the error floor set by the bias does
        # not shrink at all. No per-frame R, however well calibrated, can fix that;
        # this is the caveat spelled out in
        # reliability.observation_model.calibration_covariance.
        # The honest minimum: the belief may not claim to be sharper than the
        # systematic error of the camera informing it.
        floor = (RESIDUAL_BIAS_M[camera_id] if spec["floor"] == "per_camera"
                 else POOLED_BIAS_M) ** 2
        values, vectors = np.linalg.eigh(cov)
        values = np.maximum(values, floor)
        cov = vectors @ np.diag(values) @ vectors.T
        cov = 0.5 * (cov + cov.T)
    return mean, cov, True, diagnostics


def run_arm(capture, arm: str, truth_lookup) -> dict:
    """Odometry-driven filter over one capture. Prediction is identical per arm.

    For ``A3_network_consistency`` a bank of leave-one-camera-out beliefs runs
    alongside the main one. Camera c's health is judged by its innovation against
    the belief its own measurements never touched -- the only reference that
    cannot be captured by the camera being judged.
    """

    mean = np.asarray(capture.odom[0], dtype=float)
    cov = np.eye(2) * INITIAL_SIGMA_M**2
    cross_check = ARM_SPEC[arm]["cross_check"]
    # belief bank: excluded camera -> (mean, cov), each built without that camera
    bank = {c: [mean.copy(), cov.copy()] for c in rc.CAMERAS} if cross_check else {}
    monitors = {c: InnovationHealthMonitor(c, InnovationHealthConfig())
                for c in rc.CAMERAS}

    detections = sorted(
        ((camera, d) for camera in rc.CAMERAS for d in capture.detections[camera]),
        key=lambda item: item[1].stamp,
    )
    stamps = np.asarray(capture.stamps, dtype=float)
    odom = np.asarray(capture.odom, dtype=float)

    records = []
    previous_index = 0
    for camera, detection in detections:
        index = int(np.searchsorted(stamps, detection.stamp))
        index = min(max(index, 0), len(stamps) - 1)
        if index > previous_index:
            step = odom[index] - odom[previous_index]
            distance = float(np.hypot(*step))
            growth = np.eye(2) * (PROCESS_SIGMA_PER_SQRT_M**2 * max(distance, 1e-6))
            mean = mean + step
            cov = cov + growth
            for excluded in bank:
                bank[excluded][0] = bank[excluded][0] + step
                bank[excluded][1] = bank[excluded][1] + growth
            previous_index = index
        velocity = np.zeros(2)
        if index > 0:
            dt = float(stamps[index] - stamps[index - 1])
            if dt > 1e-6:
                velocity = (odom[index] - odom[index - 1]) / dt

        health = None
        if cross_check:
            # Judge this camera against the belief that excludes it, then feed the
            # verdict to the main filter.
            peer_mean, peer_cov = bank[camera]
            peer_innovation = np.asarray(detection.world, dtype=float) - peer_mean
            sigma = R_COND_SIGMA_M[camera]
            peer_total = (peer_cov + np.eye(2) * sigma**2)
            peer_nis = float(peer_innovation @ np.linalg.solve(peer_total, peer_innovation))
            health = monitors[camera].update(
                peer_nis, tuple(peer_innovation), dropped=False
            )
            # Every OTHER camera's reference absorbs this detection, so each bank
            # entry stays a belief built without its own camera.
            for excluded in bank:
                if excluded == camera:
                    continue
                bank_mean, bank_cov = bank[excluded]
                bank_total = bank_cov + np.eye(2) * sigma**2
                bank_gain = bank_cov @ np.linalg.inv(bank_total)
                innovation_here = np.asarray(detection.world, dtype=float) - bank_mean
                bank_mean = bank_mean + bank_gain @ innovation_here
                identity = np.eye(2)
                a = identity - bank_gain
                bank_cov = (a @ bank_cov @ a.T
                            + bank_gain @ (np.eye(2) * sigma**2) @ bank_gain.T)
                bank[excluded] = [bank_mean, 0.5 * (bank_cov + bank_cov.T)]

        mean, cov, accepted, diagnostics = measurement_update(
            arm, mean, cov, detection, velocity, camera, health
        )
        truth = truth_lookup(detection.stamp)
        if truth is None:
            continue
        error = mean - np.asarray(truth[:2], dtype=float)
        nees = float(error @ np.linalg.solve(cov, error))
        records.append({
            "stamp": detection.stamp, "camera": camera, "accepted": accepted,
            "error_m": float(np.hypot(*error)), "nees": nees,
            "sigma_m": float(math.sqrt(0.5 * (cov[0, 0] + cov[1, 1]))),
            "nis": diagnostics["nis"], "responsibility": diagnostics["responsibility"],
            "health": diagnostics.get("health", 1.0),
        })
    return {"records": records}


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    nees = np.asarray([r["nees"] for r in records])
    errors = np.asarray([r["error_m"] for r in records])
    sigmas = np.asarray([r["sigma_m"] for r in records])
    return {
        "n": len(records),
        # Calibrated median NEES for a 2-D belief is 1.39; >> means overconfident.
        "median_nees": float(np.median(nees)),
        "mean_nees": float(np.mean(nees)),
        "coverage_95": float(np.mean(nees <= 5.991)),
        "coverage_50": float(np.mean(nees <= 1.386)),
        "unearned_confidence_fraction": float(np.mean(nees > 5.991)),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "p95_error_m": float(np.percentile(errors, 95)),
        "mean_stated_sigma_m": float(np.mean(sigmas)),
        "accept_rate": float(np.mean([r["accepted"] for r in records])),
        "mean_responsibility": float(np.mean([r["responsibility"] for r in records])),
    }


def fig_s1(per_arm: dict) -> None:
    """Honesty and sharpness side by side — neither means anything alone."""

    # Horizontal bars, shared y axis: the arm descriptions are full phrases and only
    # fit as y-tick labels. Top-to-bottom reading order, so A0 sits at the top.
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=True)
    positions = np.arange(len(ARMS))
    colors = [ARM_COLOR[a] for a in ARMS]

    ax.barh(positions, [per_arm[a]["median_nees"] for a in ARMS], color=colors)
    ax.axvline(1.386, color="#009E73", lw=2.0, ls="--",
               label="perfectly honest = 1.39")
    ax.set_xscale("log")
    # Default log ticks give a single labelled decade here; readers need to be able to
    # see that A4 sits BELOW 1.39 and A2 sits ~4x above it.
    ax.set_xticks([0.5, 1.0, 1.39, 2.0, 5.0])
    ax.set_xticklabels(["0.5", "1", "1.39", "2", "5"])
    ax.minorticks_off()
    ax.set_yticks(positions, [ARM_LABEL[a] for a in ARMS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("(actual error)² ÷ (uncertainty the filter claims)\n"
                  "median over all updates, log scale")
    ax.set_title("Does the robot know how lost it is?\n"
                 "right of the line = claims more precision than it has",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax2.barh(positions, [100 * per_arm[a]["unearned_confidence_fraction"] for a in ARMS],
             color=colors)
    ax2.axvline(5.0, color="#009E73", lw=2.0, ls="--", label="should be 5 %")
    ax2.set_xlabel("% of updates where the robot's true position\n"
                   "lay outside its own 95 % confidence ellipse")
    ax2.set_title("How often the robot was somewhere\nit was sure it was not",
                  fontweight="bold", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")

    height = 0.38
    ax3.barh(positions - height / 2, [100 * per_arm[a]["rmse_m"] for a in ARMS], height,
             color=colors, label="2-D belief position RMSE")
    ax3.barh(positions + height / 2, [100 * per_arm[a]["mean_stated_sigma_m"] for a in ARMS],
             height, color=colors, alpha=0.45, hatch="//",
             label="mean RMS per-axis 1σ")
    ax3.set_xlabel("centimetres")
    # Headroom so the legend does not sit on top of the bars; every bar still starts at 0.
    ax3.set_xlim(0.0, 7.4)
    ax3.set_title("Accuracy and sharpness\ncalibration is judged by NEES/coverage, not bar equality",
                  fontweight="bold", fontsize=10)
    ax3.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Retired-v2 inputs contain repeated bias. What does each filter do about it?",
        fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.930,
             "1,424 update steps from three named 4-camera Gazebo captures under retired v2. "
             "The repeated residual is recorded but camera/route/yaw-confounded. Ground truth only scores the arms — no "
             "filter ever sees it.",
             ha="center", va="top", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_s1_honesty_and_sharpness.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_s2(per_arm_records: dict) -> None:
    """Where the trust goes: how each arm treated the biased camera vs the others."""

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax, ax2 = axes

    # LEFT: the leave-one-camera-out verdict, which is the quantity that actually
    # distinguishes the arms. (An earlier version plotted the outlier responsibility
    # here; only A2 uses that mechanism, so every other arm sat flat at 1.0 and the
    # panel said nothing.) 1.0 = "this camera looks perfectly fine to me".
    positions = np.arange(len(rc.CAMERAS))
    width = 0.115
    for index, arm in enumerate(ARMS):
        records = per_arm_records[arm]
        by_camera = []
        for camera in rc.CAMERAS:
            verdicts = [r["health"] for r in records if r["camera"] == camera]
            by_camera.append(float(np.mean(verdicts)) if verdicts else math.nan)
        # Centre the group of 7 on its tick; the old (index - 1) offset pushed every
        # group to the right of the camera it described.
        offset = (index - (len(ARMS) - 1) / 2.0) * width
        ax.bar(positions + offset, by_camera, width,
               color=ARM_COLOR[arm], label=ARM_LONG[arm])
    ax.axhline(1.0, color="#444444", lw=1.0, ls=":")
    ax.set_xticks(positions,
                  [f"camera {c.replace('camera_', '')}" for c in rc.CAMERAS],
                  fontsize=9)
    ax.set_ylim(0.0, 1.28)
    ax.set_ylabel("health this camera was given\n1.0 = “nothing wrong here”")
    ax.set_title("Which filters noticed the cameras that lean?\n"
                 "a flat row at 1.0 means the filter never suspected anything",
                 fontweight="bold", fontsize=10)
    # Name the actual lean of each camera on the plot, so the reader can compare the
    # verdict against the fault instead of taking the caption's word for it.
    for index, camera in enumerate(rc.CAMERAS):
        ax.text(index, 1.20, f"lean\n{1000 * RESIDUAL_BIAS_M[camera]:.0f} mm",
                ha="center", va="center", fontsize=8, color="#333333")

    # RIGHT: the full distribution, not just the median.
    for arm in ARMS:
        records = per_arm_records[arm]
        nees = np.asarray([r["nees"] for r in records])
        order = np.sort(nees)
        ax2.plot(order, np.linspace(0, 1, order.size), lw=2.0,
                 color=ARM_COLOR[arm], label=ARM_LONG[arm])
    ax2.axvline(5.991, color="#009E73", lw=1.6, ls="--",
                label="edge of the claimed 95 % ellipse")
    ax2.axhline(0.95, color="#009E73", lw=1.0, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("(actual error)² ÷ (uncertainty the filter claims), log scale")
    ax2.set_ylabel("fraction of updates at or below that value")
    ax2.set_title("Not just the typical case — the whole spread\n"
                  "honest = curve crosses 95 % left of the dashed line",
                  fontweight="bold", fontsize=10)

    fig.suptitle("A camera cannot detect its own lean. Only its neighbours can.",
                 fontsize=12.5, fontweight="bold")
    # ONE shared legend below both panels: seven long descriptions cannot sit inside
    # the axes without covering the data they explain.
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.155, 1, 0.91))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_s2_where_the_trust_goes.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    models = rc.camera_models()
    calib = rc.deployed_calibration()

    per_arm_records: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    per_capture: dict[str, dict] = {}
    for name in rc.CAPTURES:
        capture = rc.load_operational_capture(name, models=models, calib=calib)
        # EVALUATION ONLY — scores the arms, never enters a filter.
        truth_table = rc.load_truth_table(name)

        def truth(stamp: float, _table=truth_table):
            return rc.truth_at(_table, stamp)

        per_capture[name] = {}
        for arm in ARMS:
            result = run_arm(capture, arm, truth)
            per_arm_records[arm].extend(result["records"])
            per_capture[name][arm] = summarize(result["records"])

    per_arm = {arm: summarize(per_arm_records[arm]) for arm in ARMS}
    fig_s1(per_arm)
    fig_s2(per_arm_records)
    payload = {
        "config": {
            "context_id": "BELIEF-V2",
            "metric_object": "filter_belief_and_honesty",
            "experimental_unit": "filter_update_step",
            "run_ids": list(rc.CAPTURES),
            "projection_runtime": "projection_calibration_v2_retired",
            "mean_stated_sigma_definition": "mean(sqrt(trace(P)/2)); RMS per-axis 1-sigma",
            "fault": "retired v2 context — camera C carries a recorded +76.9 mm signed "
                     "lateral bias on route/yaw-confounded captures (not current accuracy)",
            "fixed_R_sigma_m": FIXED_R_SIGMA_M,
            "r_cond_sigma_m": R_COND_SIGMA_M,
            "outlier_prior": OUTLIER_PRIOR,
            "outlier_sigma_m": OUTLIER_SIGMA_M,
            "nis_gate_chi2": NIS_GATE_CHI2,
            "calibrated_median_nees": 1.386,
        },
        "pooled": per_arm,
        "per_capture": per_capture,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'arm':<22}{'medNEES':>9}{'unearned%':>11}{'RMSE cm':>9}"
          f"{'stated cm':>11}{'accept%':>9}")
    for arm in ARMS:
        s = per_arm[arm]
        print(f"{arm:<22}{s['median_nees']:>9.2f}"
              f"{100 * s['unearned_confidence_fraction']:>11.1f}"
              f"{100 * s['rmse_m']:>9.1f}{100 * s['mean_stated_sigma_m']:>11.1f}"
              f"{100 * s['accept_rate']:>9.1f}")
    print("\nwrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
