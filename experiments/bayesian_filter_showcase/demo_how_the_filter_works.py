#!/usr/bin/env python3
"""demo: how this filter works, in four acts.

An EXPLAINER, not a new experiment. It re-runs nothing in Gazebo, fits nothing,
and changes no number: every panel is drawn from the same three recorded captures
and the same update rule that produce `exp1_graceful_vs_trusting.py`'s result.

    Act 1  what a filter is        one real detection, drawn: prior x likelihood
                                   = posterior, then the predict/update sawtooth
    Act 2  how we use it here      state, prediction, and what R is actually made of
    Act 3  what that looks like    the failure and the fix on the recorded data
    Act 4  how it learns to filter which numbers are measured, where they come
                                   from, and how wrong they may be

The update rule is IMPORTED from exp1 (`measurement_update`), never re-implemented.
The prediction loop is duplicated here only because the demo needs the
intermediates exp1 discards -- and `_verify_against_exp1()` asserts the duplicate
reproduces exp1's NEES trace exactly, so the pictures cannot drift from the result.

Ground truth is EVALUATION-ONLY: it is drawn and it scores, it never enters a filter.

Outputs -> logs/studies/bayesian_filter_showcase/demo_how_the_filter_works/
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
from matplotlib.patches import Ellipse

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
for _relative in ("src/reliability", "src/unav_common", "src/state",
                  "experiments/operational_residual_rcond"):
    sys.path.insert(0, str(REPO / _relative))
sys.path.insert(0, str(_HERE.parent))

from reliability.health_ewma import (  # noqa: E402
    InnovationHealthConfig,
    InnovationHealthMonitor,
)
from reliability.observation_model import time_sync_covariance  # noqa: E402
import rcond_common as rc  # noqa: E402
import exp1_graceful_vs_trusting as f1  # noqa: E402

OUT = REPO / "logs/studies/bayesian_filter_showcase/demo_how_the_filter_works"

#: The capture every walkthrough panel is drawn from. smoke1 is the yaw = +90 deg
#: historical-v2 route, which is the one on which camera C's signed residual is largest.
DEMO_CAPTURE = "smoke1_20260716"

# Okabe-Ito, validated for CVD separation against a light surface.
C_PRIOR = "#E69F00"      # what we believed before
C_MEAS = "#009E73"       # what the camera said
C_POST = "#0072B2"       # what we believe now  (also: A4, the honest filter)
C_NAIVE = "#D55E00"      # A0, the trusting filter
C_TRUTH = "#111111"      # evaluation only
C_MUTED = "#8A8A8A"

CHI2_95_2DOF = 5.991     # the 95 % ellipse, 2 dof
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


def _ellipse(ax, mean, cov, chi2=CHI2_95_2DOF, **kwargs):
    """Draw the {x : (x-m)' P^-1 (x-m) <= chi2} contour -- the stated ellipse."""
    values, vectors = np.linalg.eigh(np.asarray(cov, dtype=float))
    values = np.maximum(values, 1e-12)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2.0 * np.sqrt(chi2 * values)
    patch = Ellipse(xy=(float(mean[0]), float(mean[1])), width=width, height=height,
                    angle=angle, **kwargs)
    ax.add_patch(patch)
    return patch


def _gauss_1d(grid, mu, sigma):
    return np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


# ------------------------------------------------------- the filter, instrumented


def trace_arm(capture, arm: str, truth_lookup) -> list[dict]:
    """`f1.run_arm`, but recording the intermediates the pictures need.

    The UPDATE is `f1.measurement_update` -- the same function exp1 scores. Only
    the prediction bookkeeping is written out here, so that the prior/posterior
    pair either side of each update can be drawn. `_verify_against_exp1` proves
    the two loops agree.
    """
    mean = np.asarray(capture.odom[0], dtype=float)
    cov = np.eye(2) * f1.INITIAL_SIGMA_M**2
    cross_check = f1.ARM_SPEC[arm]["cross_check"]
    bank = {c: [mean.copy(), cov.copy()] for c in rc.CAMERAS} if cross_check else {}
    monitors = {c: InnovationHealthMonitor(c, InnovationHealthConfig())
                for c in rc.CAMERAS}

    detections = sorted(
        ((camera, d) for camera in rc.CAMERAS for d in capture.detections[camera]),
        key=lambda item: item[1].stamp,
    )
    stamps = np.asarray(capture.stamps, dtype=float)
    odom = np.asarray(capture.odom, dtype=float)

    trace = []
    previous_index = 0
    for camera, detection in detections:
        index = int(np.searchsorted(stamps, detection.stamp))
        index = min(max(index, 0), len(stamps) - 1)
        mean_before, cov_before = mean.copy(), cov.copy()
        step = np.zeros(2)
        if index > previous_index:
            step = odom[index] - odom[previous_index]
            distance = float(np.hypot(*step))
            growth = np.eye(2) * (f1.PROCESS_SIGMA_PER_SQRT_M**2 * max(distance, 1e-6))
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
            peer_mean, peer_cov = bank[camera]
            peer_innovation = np.asarray(detection.world, dtype=float) - peer_mean
            sigma = f1.R_COND_SIGMA_M[camera]
            peer_total = (peer_cov + np.eye(2) * sigma**2)
            peer_nis = float(peer_innovation @ np.linalg.solve(peer_total, peer_innovation))
            health = monitors[camera].update(peer_nis, tuple(peer_innovation), dropped=False)
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

        mean_predicted, cov_predicted = mean.copy(), cov.copy()
        mean, cov, accepted, diagnostics = f1.measurement_update(
            arm, mean, cov, detection, velocity, camera, health
        )
        truth = truth_lookup(detection.stamp)
        if truth is None:
            continue
        error = mean - np.asarray(truth[:2], dtype=float)
        trace.append({
            "stamp": detection.stamp, "camera": camera, "accepted": accepted,
            "mean_before": mean_before, "cov_before": cov_before,
            "odom_step": step,
            "mean_predicted": mean_predicted, "cov_predicted": cov_predicted,
            "z": np.asarray(detection.world, dtype=float),
            "mean_post": mean.copy(), "cov_post": cov.copy(),
            "velocity": velocity, "range_m": detection.range_m,
            "truth": np.asarray(truth[:2], dtype=float),
            "error_m": float(np.hypot(*error)),
            "nees": float(error @ np.linalg.solve(cov, error)),
            "sigma_m": float(math.sqrt(0.5 * (cov[0, 0] + cov[1, 1]))),
            "nis": diagnostics["nis"],
            # carried so `f1.summarize` can score these records directly
            "responsibility": diagnostics["responsibility"],
            "health": diagnostics.get("health", 1.0),
        })
    return trace


def measurement_covariance(arm: str, camera: str, velocity) -> dict:
    """The R this arm hands the update, and what it is made of.

    Same construction as `f1.measurement_update`, term by term, so the picture in
    Act 2 is of the R the filter actually used.
    """
    if not f1.ARM_SPEC[arm]["per_camera_r"]:
        r_fixed = np.eye(2) * f1.FIXED_R_SIGMA_M**2
        return {"total": r_fixed, "terms": {"fixed R (one for every camera)": r_fixed}}
    sigma = f1.R_COND_SIGMA_M[camera]
    r_cond = np.eye(2) * sigma**2
    r_time = np.asarray(
        time_sync_covariance(f1.H, np.asarray(velocity, dtype=float).tolist(),
                             f1.SIGMA_TAU_S),
        dtype=float,
    )
    bias = f1.RESIDUAL_BIAS_M[camera]
    r_model = np.eye(2) * bias**2
    return {
        "total": r_cond + r_time + r_model,
        "terms": {
            "R_cond  measured scatter": r_cond,
            "R_time  timing x speed": r_time,
            "R_model residual bias": r_model,
        },
    }


def _verify_against_exp1(capture, truth_lookup) -> dict:
    """The demo's loop must reproduce exp1 exactly, or the pictures are fiction."""
    report = {}
    for arm in ("A0_trust_everything", "A4_correlation_floor"):
        mine = trace_arm(capture, arm, truth_lookup)
        theirs = f1.run_arm(capture, arm, truth_lookup)["records"]
        assert len(mine) == len(theirs), f"{arm}: {len(mine)} vs {len(theirs)} records"
        worst = max(abs(a["nees"] - b["nees"]) for a, b in zip(mine, theirs))
        assert worst < 1e-9, f"{arm}: NEES diverged by {worst}"
        report[arm] = {"n_updates": len(mine), "max_nees_difference": worst}
    return report


# ------------------------------------------------------------------------ act 1


def fig_a1_one_update(trace: list[dict]) -> dict:
    """A filter is one line of arithmetic, and this is it drawn once.

    Prior belief x what the camera said = new belief. Everything after this
    figure is a question about the width of the middle term.
    """
    # The update where the prior is at its WIDEST. Picked because a picture of a
    # filter is only legible when the two inputs are comparable in width: once the
    # belief is much sharper than the camera, the posterior sits on top of the prior
    # and the drawing says nothing. This is the honest such moment in the data.
    pick = int(np.argmax([math.sqrt(r["cov_predicted"][0, 0]) for r in trace]))
    r = trace[pick]

    prior_m, prior_P = r["mean_predicted"], r["cov_predicted"]
    z = r["z"]
    R = measurement_covariance("A0_trust_everything", r["camera"], r["velocity"])["total"]
    post_m, post_P = r["mean_post"], r["cov_post"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    # -- left: the same update as three 1-D curves, along x
    sx_prior = math.sqrt(prior_P[0, 0])
    sx_meas = math.sqrt(R[0, 0])
    sx_post = math.sqrt(post_P[0, 0])
    centre = 0.5 * (prior_m[0] + z[0])
    span = 4.0 * max(sx_prior, sx_meas)
    grid = np.linspace(centre - span, centre + span, 600)
    ax.plot(grid, _gauss_1d(grid, prior_m[0], sx_prior), color=C_PRIOR, lw=2.0)
    ax.plot(grid, _gauss_1d(grid, z[0], sx_meas), color=C_MEAS, lw=2.0)
    ax.plot(grid, _gauss_1d(grid, post_m[0], sx_post), color=C_POST, lw=2.4)
    ax.fill_between(grid, _gauss_1d(grid, post_m[0], sx_post), color=C_POST, alpha=0.12)
    ax.text(0.02, 0.94, f"prior belief   $\\sigma$ = {100 * sx_prior:.1f} cm",
            transform=ax.transAxes, color=C_PRIOR, fontsize=9.5, fontweight="bold")
    ax.text(0.02, 0.87, f"camera says   $\\sigma$ = {100 * sx_meas:.1f} cm",
            transform=ax.transAxes, color=C_MEAS, fontsize=9.5, fontweight="bold")
    ax.text(0.02, 0.80, f"new belief    $\\sigma$ = {100 * sx_post:.1f} cm",
            transform=ax.transAxes, color=C_POST, fontsize=9.5, fontweight="bold")
    ax.text(0.02, 0.70, "narrower than BOTH inputs:\nthat is the entire trick",
            transform=ax.transAxes, color="#333333", fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("probability density")
    ax.set_yticks([])
    ax.set_title("A filter multiplies two Gaussians\nthe answer is always narrower than either")

    # -- right: the identical update in 2-D, which is what the code does
    _ellipse(ax2, prior_m, prior_P, facecolor=C_PRIOR, alpha=0.18, edgecolor=C_PRIOR, lw=1.8)
    _ellipse(ax2, z, R, facecolor=C_MEAS, alpha=0.14, edgecolor=C_MEAS, lw=1.8, ls="--")
    _ellipse(ax2, post_m, post_P, facecolor=C_POST, alpha=0.30, edgecolor=C_POST, lw=2.2)
    ax2.plot(*prior_m, "o", color=C_PRIOR, ms=7, mec="white", mew=1.2)
    ax2.plot(*z, "o", color=C_MEAS, ms=7, mec="white", mew=1.2)
    ax2.plot(*post_m, "o", color=C_POST, ms=9, mec="white", mew=1.4)
    ax2.plot(*r["truth"], "x", color=C_TRUTH, ms=11, mew=2.2)
    ax2.annotate("prior", xy=prior_m, xytext=(10, 12), textcoords="offset points",
                 color=C_PRIOR, fontsize=9.5, fontweight="bold")
    ax2.annotate("measurement", xy=z, xytext=(10, -18), textcoords="offset points",
                 color=C_MEAS, fontsize=9.5, fontweight="bold")
    ax2.annotate("posterior", xy=post_m, xytext=(-78, 14), textcoords="offset points",
                 color=C_POST, fontsize=9.5, fontweight="bold")
    ax2.annotate("truth (evaluation only)", xy=r["truth"], xytext=(-30, 26),
                 textcoords="offset points", color=C_TRUTH, fontsize=9, ha="center")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("The same update in 2-D\nellipses are the stated 95 % regions")

    fig.suptitle("ACT 1 — what a filter is: one real detection from "
                 f"{r['camera'].replace('_', ' ')}, {DEMO_CAPTURE}",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_a1_one_update")
    return {
        "camera": r["camera"], "stamp": r["stamp"], "update_index": pick,
        "prior_sigma_x_m": sx_prior, "measurement_sigma_x_m": sx_meas,
        "posterior_sigma_x_m": sx_post,
        "innovation_m": float(np.hypot(*(z - prior_m))),
    }


def fig_a2_sawtooth(trace: list[dict]) -> dict:
    """Two steps, alternating forever: driving makes it vaguer, looking sharpens it."""
    def sigmas(record, key):
        cov = record[key]
        return 100 * math.sqrt(0.5 * (cov[0, 0] + cov[1, 1]))

    span = 60
    # Steady state, chosen as the window with the largest average sawtooth
    # amplitude AFTER the opening transient has died away.
    settle = 120
    amplitude = np.asarray([sigmas(r, "cov_predicted") - sigmas(r, "cov_post")
                            for r in trace])
    windowed = np.convolve(amplitude[settle:], np.ones(span) / span, mode="valid")
    start = settle + int(np.argmax(windowed))

    fig = plt.figure(figsize=(12.8, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.0], hspace=0.42, wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    steady = trace[start:start + span]
    gaps = np.diff([r["stamp"] for r in steady])
    worst = int(np.argmax(gaps)) if len(gaps) else 0
    gap_s = float(gaps[worst]) if len(gaps) else 0.0
    gap_grew = (sigmas(steady[worst + 1], "cov_predicted") / sigmas(steady[worst], "cov_post")
                if len(gaps) else 1.0)

    for axis, window, title in (
        (ax, trace[:span], "From a standing start: the cameras pull the belief in"),
        (ax2, steady,
         "When coverage lapses: the belief inflates until a camera sees it again"),
    ):
        times = np.asarray([r["stamp"] for r in window])
        times = times - times[0]
        before = [sigmas(r, "cov_before") for r in window]
        predicted = [sigmas(r, "cov_predicted") for r in window]
        post = [sigmas(r, "cov_post") for r in window]
        for t, a, b in zip(times, before, predicted):
            axis.plot([t, t], [a, b], color=C_PRIOR, lw=2.0, solid_capstyle="round",
                      zorder=3)
        for t, a, b in zip(times, predicted, post):
            axis.plot([t, t], [a, b], color=C_POST, lw=2.0, solid_capstyle="round",
                      zorder=3)
        axis.plot(times, predicted, "o", color=C_PRIOR, ms=4.0, mec="white", mew=0.7,
                  zorder=4)
        axis.plot(times, post, "o", color=C_POST, ms=4.0, mec="white", mew=0.7, zorder=4)
        axis.set_ylabel("stated $\\sigma$ (cm)")
        axis.set_xlabel("seconds")
        axis.set_title(title, loc="left", fontsize=10)
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + 0.30 * (high - low))
    ax.text(0.97, 0.93, "PREDICT (up)\nUPDATE (down)", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right", color="#333333")
    ax2.text(0.03, 0.93, f"while a camera is watching, each\ndrive step is worth "
                         f"tenths of a mm.\n{gap_s:.1f} s without a detection and the\n"
                         f"belief inflates {gap_grew:.1f}x — then the\nnetwork pulls it "
                         "back in.",
             transform=ax2.transAxes, fontsize=9, va="top", color="#333333")

    # -- which camera is informing the belief, across the whole capture
    order = {c: i for i, c in enumerate(rc.CAMERAS)}
    base = trace[0]["stamp"]
    for r in trace:
        ax3.plot(r["stamp"] - base, order[r["camera"]], "|", color=C_MEAS, ms=10,
                 mew=1.4)
    ax3.set_yticks(range(len(rc.CAMERAS)), [c.replace("camera_", "") for c in rc.CAMERAS])
    ax3.set_ylabel("informed by")
    ax3.set_xlabel(f"seconds into the capture — all {len(trace)} updates")
    ax3.set_ylim(-0.6, len(rc.CAMERAS) - 0.4)
    ax3.grid(axis="y", alpha=0.0)
    ax3.set_title("Which camera the belief is listening to — coverage is a "
                  "relay, not a chorus", loc="left", fontsize=10)

    fig.suptitle("ACT 1 — the whole loop: predict, update, repeat "
                 f"({DEMO_CAPTURE})", fontsize=12, fontweight="bold", y=0.98)
    _save(fig, "fig_a2_predict_update_loop")
    return {"steady_state_window_start": start,
            "opening_sigma_cm": sigmas(trace[0], "cov_predicted"),
            "settled_sigma_cm": sigmas(trace[start], "cov_post")}


# ------------------------------------------------------------------------ act 2


def fig_a3_what_r_is_made_of(trace: list[dict]) -> dict:
    """Act 2's whole point: R is the only thing we get to design, so what is in it?"""
    speeds = np.asarray([float(np.hypot(*r["velocity"])) for r in trace])
    speed = float(np.median(speeds[speeds > 1e-6])) if np.any(speeds > 1e-6) else 0.0
    velocity = np.asarray([speed, 0.0])

    cameras = list(rc.CAMERAS)
    term_names = ["R_cond  measured scatter", "R_time  timing x speed",
                  "R_model residual bias"]
    term_colors = {"R_cond  measured scatter": C_MEAS,
                   "R_time  timing x speed": C_MUTED,
                   "R_model residual bias": C_NAIVE}

    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    width = 0.24
    positions = np.arange(len(cameras))
    per_camera = {}
    for offset, name in enumerate(term_names):
        heights = []
        for camera in cameras:
            built = measurement_covariance("A4_correlation_floor", camera, velocity)
            sigma = math.sqrt(0.5 * (built["terms"][name][0, 0] + built["terms"][name][1, 1]))
            heights.append(1000 * sigma)
        ax.bar(positions + (offset - 1) * width, heights, width * 0.92,
               color=term_colors[name], label=name)
    for i, camera in enumerate(cameras):
        built = measurement_covariance("A4_correlation_floor", camera, velocity)
        total = 1000 * math.sqrt(0.5 * (built["total"][0, 0] + built["total"][1, 1]))
        per_camera[camera] = total
        ax.plot(i, total, "D", color=C_POST, ms=9, mec="white", mew=1.2, zorder=5)
        ax.annotate(f"{total:.0f} mm", xy=(i, total), xytext=(0, 11),
                    textcoords="offset points", ha="center", color=C_POST,
                    fontsize=9, fontweight="bold")

    fixed = 1000 * f1.FIXED_R_SIGMA_M
    ax.set_ylim(0, fixed * 1.42)
    ax.axhline(fixed, color=C_TRUTH, lw=1.4, ls="--")
    ax.text(0.5, fixed + 2.0,
            f"the state of practice: one fixed R for every camera, {fixed:.0f} mm",
            fontsize=9, color=C_TRUTH, va="bottom")
    ax.plot([], [], "D", color=C_POST, ms=8, label="total $\\sigma$ (terms in quadrature)")
    ax.set_xticks(positions, [c.replace("camera_", "camera ") for c in cameras])
    ax.set_ylabel("measurement $\\sigma$ (mm)")
    ax.set_title("ACT 2 — what R is made of, per camera\n"
                 "every term is MEASURED at commissioning; none is tuned to an outcome",
                 fontsize=12, fontweight="bold", loc="left", pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5,
              bbox_to_anchor=(0.0, 0.86))
    ax.annotate("under retired v2, camera C's R is dominated by BIAS,\n"
                "not by scatter — that is the whole story",
                xy=(2.24, 1000 * f1.RESIDUAL_BIAS_M["camera_C"]),
                xytext=(2.62, fixed * 1.20), fontsize=9, color=C_NAIVE,
                fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C_NAIVE, lw=1.3))
    fig.tight_layout()
    _save(fig, "fig_a3_what_r_is_made_of")
    return {"median_speed_mps": speed, "total_sigma_mm": per_camera,
            "fixed_r_sigma_mm": fixed}


# ------------------------------------------------------------------------ act 3


def fig_a4_shrinking_onto_a_lie(trace_a0: list[dict]) -> dict:
    """Why repeated looks from ONE camera are not repeated evidence."""
    rows = [r for r in trace_a0 if r["camera"] == "camera_C"][:60]
    n = np.arange(1, len(rows) + 1)
    stated = np.asarray([1000 * r["sigma_m"] for r in rows])
    actual = np.asarray([1000 * r["error_m"] for r in rows])

    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.plot(n, stated, color=C_NAIVE, lw=2.2, label="what the filter SAYS its error is")
    ax.plot(n, actual, color=C_TRUTH, lw=2.0, label="what the error ACTUALLY is")
    ax.fill_between(n, stated, actual, where=actual > stated, color=C_NAIVE, alpha=0.12,
                    interpolate=True)
    ax.axhline(1000 * f1.RESIDUAL_BIAS_M["camera_C"], color=C_MUTED, lw=1.3, ls=":")
    ax.annotate(f"historical-v2 signed lean: {1000 * f1.RESIDUAL_BIAS_M['camera_C']:.0f} mm\n"
                "this floor does not shrink with more looks",
                xy=(len(rows) * 0.52, 1000 * f1.RESIDUAL_BIAS_M["camera_C"]),
                xytext=(0, 12), textcoords="offset points", fontsize=9, color=C_MUTED)
    ax.set_xlabel("number of detections from camera C so far")
    ax.set_ylabel("mm")
    ax.set_title("ACT 3 — the failure: confidence grows, accuracy does not\n"
                 "A0 treats every look from the same camera as fresh independent evidence",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right")
    gap = float(np.median(actual / np.maximum(stated, 1e-9)))
    ax.text(0.30, 0.22, f"the gap is the overconfidence:\n"
                        f"the truth sits {gap:.1f}x further away\nthan the filter claims",
            transform=ax.transAxes, fontsize=9.5, color=C_NAIVE, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_a4_shrinking_onto_a_lie")
    return {"n_camera_c_updates": len(rows),
            "median_actual_over_stated": gap,
            "final_stated_mm": float(stated[-1]), "final_actual_mm": float(actual[-1])}


def fig_a5_why_gating_cannot_see_it(trace_a0: list[dict]) -> dict:
    """The most counter-intuitive panel: a lie that never looks like an outlier."""
    c_nis = np.asarray([r["nis"] for r in trace_a0 if r["camera"] == "camera_C"])
    other_nis = np.asarray([r["nis"] for r in trace_a0 if r["camera"] != "camera_C"])
    c_innov = np.asarray([1000 * float(np.hypot(*(r["z"] - r["mean_predicted"])))
                          for r in trace_a0 if r["camera"] == "camera_C"])
    other_innov = np.asarray([1000 * float(np.hypot(*(r["z"] - r["mean_predicted"])))
                              for r in trace_a0 if r["camera"] != "camera_C"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    bins = np.linspace(0, max(np.percentile(np.r_[c_innov, other_innov], 99), 1), 40)
    ax.hist(other_innov, bins=bins, color=C_MUTED, alpha=0.75, label="cameras A, B, D")
    ax.hist(c_innov, bins=bins, color=C_NAIVE, alpha=0.75, label="camera C")
    ax.axvline(float(np.median(other_innov)), color=C_MUTED, lw=2.0)
    ax.axvline(float(np.median(c_innov)), color=C_NAIVE, lw=2.0)
    ax.annotate(f"medians {np.median(other_innov):.0f} mm vs {np.median(c_innov):.0f} mm\n"
                "— the liar is not the loud one",
                xy=(0.40, 0.80), xycoords="axes fraction", fontsize=9, color="#333333")
    ax.set_xlabel("innovation |z - predicted| (mm)")
    ax.set_ylabel("detections")
    ax.set_title("The historical-v2 Camera C residual is not a wild outlier", loc="left")
    ax.legend(frameon=False, loc="upper right")

    bins2 = np.linspace(0, max(np.percentile(np.r_[c_nis, other_nis], 99), 8), 40)
    ax2.hist(other_nis, bins=bins2, color=C_MUTED, alpha=0.75, label="cameras A, B, D")
    ax2.hist(c_nis, bins=bins2, color=C_NAIVE, alpha=0.75, label="camera C")
    ax2.axvline(CHI2_95_2DOF, color=C_TRUTH, lw=1.6, ls="--")
    rejected = float(np.mean(np.r_[c_nis, other_nis] > CHI2_95_2DOF))
    ax2.annotate(f"95 % $\\chi^2$ gate\nrejects {100 * rejected:.1f} % of everything",
                 xy=(CHI2_95_2DOF, ax2.get_ylim()[1] * 0.72), xytext=(10, 0),
                 textcoords="offset points", fontsize=9, color=C_TRUTH)
    ax2.set_xlabel("normalized innovation squared (NIS)")
    ax2.set_ylabel("detections")
    ax2.set_title("...so it sits comfortably inside the outlier gate", loc="left")
    ax2.legend(frameon=False)

    fig.suptitle("ACT 3 — why classical robust filtering cannot help: "
                 "a steady lean is not an outlier",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_a5_why_gating_cannot_see_it")
    return {"camera_c_median_nis": float(np.median(c_nis)),
            "other_median_nis": float(np.median(other_nis)),
            "gate_rejection_fraction": rejected}


def fig_a6_the_promise_kept_or_broken(trace_a0, trace_a4) -> dict:
    """Count the escapes by eye.

    Each dot is one update's error, belief minus truth, in metres. The ellipse is
    the promise the filter made about where that dot would land. Drawn at the
    MEDIAN stated covariance, because the ellipse changes every step; the printed
    percentage uses each update's own ellipse, which is the honest number.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9), sharex=True, sharey=True)
    stats = {}
    for ax, trace, name, colour in (
        (axes[0], trace_a0, "A0  trust everything", C_NAIVE),
        (axes[1], trace_a4, "A4  correlation floor", C_POST),
    ):
        errors = np.asarray([r["mean_post"] - r["truth"] for r in trace])
        outside = np.asarray([r["nees"] > CHI2_95_2DOF for r in trace])
        median_cov = np.median(np.asarray([r["cov_post"] for r in trace]), axis=0)

        ax.axhline(0, color=C_MUTED, lw=0.9)
        ax.axvline(0, color=C_MUTED, lw=0.9)
        ax.plot(errors[~outside, 0], errors[~outside, 1], ".", color=C_MUTED, ms=4.0,
                label=f"inside the promise ({int((~outside).sum())})")
        ax.plot(errors[outside, 0], errors[outside, 1], ".", color=C_NAIVE, ms=4.5,
                label=f"OUTSIDE it ({int(outside.sum())})")
        _ellipse(ax, (0.0, 0.0), median_cov, facecolor=colour, alpha=0.20,
                 edgecolor=colour, lw=2.2, zorder=5)

        fraction = float(outside.mean())
        sigma = math.sqrt(0.5 * (median_cov[0, 0] + median_cov[1, 1]))
        stats[name] = {"n_updates": int(len(trace)), "outside_fraction": fraction,
                       "median_stated_sigma_m": sigma}
        ax.set_title(f"{name}\nit promises $\\sigma$ = {100 * sigma:.1f} cm\n"
                     f"the truth escapes {100 * fraction:.0f} % of the time",
                     loc="left", color=colour, fontweight="bold", fontsize=10.5)
        ax.set_xlabel("belief - truth, x (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    axes[0].set_ylabel("belief - truth, y (m)")
    fig.suptitle("ACT 3 — the fix: the ellipse is a promise about where the truth is, "
                 "and only one of these keeps it",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "fig_a6_the_promise_kept_or_broken")
    return stats


# ------------------------------------------------------------------------ act 4


def fig_a7_the_ladder(summary_exp1: dict) -> dict:
    """Four mechanisms tried in order. Each failed for a different, useful reason."""
    arms = ["A0_trust_everything", "A1_hard_gate", "A2_factorized",
            "A3_network_consistency", "A4_correlation_floor"]
    reasons = [
        "start here:\none fixed R",
        "reject outliers\n-> a lean is not\nan outlier",
        "sharper per-camera R\n-> trusts the lie\nHARDER",
        "cross-check against\nthe other cameras\n-> right idea,\ntoo small",
        "floor the posterior at\nthe camera's own bias\n-> honest",
    ]
    nees = [summary_exp1["pooled"][a]["median_nees"] for a in arms]
    unearned = [100 * summary_exp1["pooled"][a]["unearned_confidence_fraction"] for a in arms]
    colours = [C_NAIVE, C_NAIVE, C_NAIVE, C_NAIVE, C_POST]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.6, 8.2),
                                  gridspec_kw={"height_ratios": [1.0, 1.0]})
    positions = np.arange(len(arms))
    ax.bar(positions, nees, 0.62, color=colours)
    ax.axhline(CALIBRATED_MEDIAN_NEES, color=C_TRUTH, lw=1.5, ls="--")
    ax.text(len(arms) - 0.45, CALIBRATED_MEDIAN_NEES + 0.08,
            "a perfectly honest filter sits here: 1.39", ha="right", fontsize=9,
            color=C_TRUTH)
    for i, value in enumerate(nees):
        ax.annotate(f"{value:.2f}", xy=(i, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold")
    ax.set_ylabel("median NEES\n(above the line = overconfident)")
    ax.set_xticks(positions, [a.split("_")[0] for a in arms])
    ax.set_title("ACT 4 — how it learned to filter: every rung failed differently",
                 fontsize=12, fontweight="bold", loc="left")
    for i, text in enumerate(reasons):
        ax.annotate(text, xy=(i, max(nees) * 0.62), ha="center", va="top", fontsize=8,
                    color="#333333")

    ax2.bar(positions, unearned, 0.62, color=colours)
    ax2.axhline(5.0, color=C_TRUTH, lw=1.5, ls="--")
    ax2.text(len(arms) - 0.45, 6.2, "what '95 % confident' promises: 5 %", ha="right",
             fontsize=9, color=C_TRUTH)
    for i, value in enumerate(unearned):
        ax2.annotate(f"{value:.1f} %", xy=(i, value), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9,
                     fontweight="bold")
    ax2.set_ylabel("truth outside the stated\n95 % ellipse (%)")
    ax2.set_xticks(positions, [a.split("_")[0] for a in arms])
    ax2.set_xlabel("the same data, the same prediction — only the update rule changes")
    fig.tight_layout()
    _save(fig, "fig_a7_the_ladder")
    return {"median_nees": dict(zip(arms, nees)),
            "unearned_percent": dict(zip(arms, unearned))}


def fig_a8_what_it_learned(summary_exp2: dict) -> dict:
    """'Learning' here is commissioning: two measured numbers per camera, no tuning."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.9))

    cameras = list(rc.CAMERAS)
    positions = np.arange(len(cameras))
    scatter = [1000 * f1.R_COND_SIGMA_M[c] for c in cameras]
    bias = [1000 * f1.RESIDUAL_BIAS_M[c] for c in cameras]
    ax.bar(positions - 0.19, scatter, 0.36, color=C_MEAS,
           label="R_cond: random scatter (shrinks with more looks)")
    ax.bar(positions + 0.19, bias, 0.36, color=C_NAIVE,
           label="residual bias: the lean (never shrinks)")
    for i, value in enumerate(bias):
        ax.annotate(f"{value:.0f}", xy=(i + 0.19, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xticks(positions, [c.replace("camera_", "") for c in cameras])
    ax.set_ylabel("mm")
    ax.set_xlabel("camera")
    ax.set_title("The two numbers commissioning hands the filter", loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    scales = sorted(summary_exp2["floor_scale_sensitivity"], key=float)
    values = [100 * summary_exp2["floor_scale_sensitivity"][s]["unearned_confidence_fraction"]
              for s in scales]
    sigmas = [100 * summary_exp2["floor_scale_sensitivity"][s]["mean_stated_sigma_m"]
              for s in scales]
    ax2.plot([float(s) for s in scales], values, "-o", color=C_POST, lw=2.2, ms=8,
             mec="white", mew=1.2)
    ax2.axhline(5.0, color=C_TRUTH, lw=1.4, ls="--")
    ax2.set_ylim(-3.0, max(values) * 1.26)
    for s, value, sigma in zip(scales, values, sigmas):
        offset = {0.25: (14, -20), 2.0: (0, 20), 4.0: (0, 20)}.get(float(s), (0, 11))
        ax2.annotate(f"{value:.1f} %\n$\\sigma$={sigma:.0f} cm", xy=(float(s), value),
                     xytext=offset, textcoords="offset points",
                     ha="left" if float(s) <= 0.25 else "center", fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xticks([float(s) for s in scales], [f"x{s}" for s in scales])
    ax2.set_xlabel("floor scaled away from its measured value")
    ax2.set_ylabel("truth outside the stated 95 % ellipse (%)")
    ax2.set_title("A threshold, not a tuned optimum — so err HIGH", loc="left", pad=12)
    ax2.text(0.30, 0.62, "too small:\nconfidently wrong", transform=ax2.transAxes,
             fontsize=9, color=C_NAIVE, fontweight="bold")
    ax2.text(0.56, 0.26, "too large: only\nvague, never unsafe", transform=ax2.transAxes,
             fontsize=9, color=C_POST, fontweight="bold")

    fig.suptitle("ACT 4 — what the filter actually learns, and how wrong it may be",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_a8_what_it_learned")
    return {"floor_scale_unearned_percent": dict(zip(scales, values))}


# ---------------------------------------------------------------------------- run


def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    models = rc.camera_models()
    calib = rc.deployed_calibration()

    capture = rc.load_operational_capture(DEMO_CAPTURE, models=models, calib=calib)
    truth_table = rc.load_truth_table(DEMO_CAPTURE)   # EVALUATION ONLY

    def truth(stamp: float, _table=truth_table):
        return rc.truth_at(_table, stamp)

    print(f"capture {DEMO_CAPTURE}: {capture.n_steps} odometry steps, "
          f"{capture.duration_s:.1f} s, "
          f"{sum(len(v) for v in capture.detections.values())} detections")

    print("\nverifying the demo loop reproduces exp1 exactly ...")
    verification = _verify_against_exp1(capture, truth)
    for arm, report in verification.items():
        print(f"  {arm}: {report['n_updates']} updates, "
              f"max NEES difference {report['max_nees_difference']:.2e}")

    trace_a0 = trace_arm(capture, "A0_trust_everything", truth)
    trace_a4 = trace_arm(capture, "A4_correlation_floor", truth)

    exp1_summary = json.loads(
        (REPO / "logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting"
                "/summary.json").read_text(encoding="utf-8"))
    exp2_summary = json.loads(
        (REPO / "logs/studies/bayesian_filter_showcase/exp2_does_it_generalize"
                "/summary.json").read_text(encoding="utf-8"))

    print("\nact 1 — what a filter is")
    a1 = fig_a1_one_update(trace_a0)
    a2 = fig_a2_sawtooth(trace_a0)
    print("act 2 — how we use it here")
    a3 = fig_a3_what_r_is_made_of(trace_a4)
    print("act 3 — what that looks like on the recorded data")
    a4 = fig_a4_shrinking_onto_a_lie(trace_a0)
    a5 = fig_a5_why_gating_cannot_see_it(trace_a0)
    a6 = fig_a6_the_promise_kept_or_broken(trace_a0, trace_a4)
    print("act 4 — how it learns to filter")
    a7 = fig_a7_the_ladder(exp1_summary)
    a8 = fig_a8_what_it_learned(exp2_summary)

    payload = {
        "role": "explainer for EXP-BELIEF; re-runs nothing, fits nothing, "
                "changes no number",
        "capture": DEMO_CAPTURE,
        "verification_against_exp1": verification,
        "act1_one_update": a1,
        "act1_predict_update_loop": a2,
        "act2_what_r_is_made_of": a3,
        "act3_shrinking_onto_a_lie": a4,
        "act3_why_gating_cannot_see_it": a5,
        "act3_the_promise_kept_or_broken": a6,
        "act4_the_ladder": a7,
        "act4_what_it_learned": a8,
    }
    rc.write_json(OUT / "summary.json", payload)
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
