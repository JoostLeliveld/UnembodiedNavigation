#!/usr/bin/env python3
"""If the offset rotates with the robot, it belongs to the ROBOT, not the camera.

`exp1` fitted a 2-D offset per camera and found it route-conditioned: it transfers
to a repeat of its own route and actively harms across a heading change. The
offset vectors say why -- on camera C they read

    smoke1           (-2.7, -47.7) mm      yaw = +90 deg
    fusion_handover  (-16.7, -48.2) mm     the same route
    smoke2           (-75.6, -21.4) mm     yaw = 0 deg

which is the same vector pointing somewhere else once the robot turns. That is
the signature of a quantity fixed in the ROBOT's body frame, seen through an
unmodelled heading -- exactly what `pixel_ground_path/e6` concluded when a
zero-parameter CAD object model collapsed camera C's historical-v2 77 mm signed residual to 8 mm.

So this experiment re-parameterises the model accordingly. Instead of one offset
per camera in the world frame::

    y_k = p_k + b^c + noise            8 free states, none of them transferable

use ONE offset in the robot's body frame, shared by every camera and rotated into
the world by the robot's heading::

    y_k = p_k + R(theta_k) b + noise    2 states, shared, and heading-aware

`theta_k` comes from the ODOMETRY path tangent, never from truth. If this
transfers across a heading change where the per-camera version did not, then the
thing being estimated was never a camera property, and the per-camera offset state
was rediscovering the robot's own geometry one camera at a time.

Ground truth is EVALUATION-ONLY: it scores, it never enters a filter.

Outputs -> logs/studies/offset_state_model/exp2_offset_in_the_robot_frame/
"""

from __future__ import annotations

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
                  "experiments/operational_residual_rcond",
                  "experiments/bayesian_filter_showcase"):
    sys.path.insert(0, str(REPO / _relative))

from reliability.observation_model import time_sync_covariance  # noqa: E402
import rcond_common as rc  # noqa: E402
import exp1_graceful_vs_trusting as f1  # noqa: E402
import demo_how_the_filter_works as d1  # noqa: E402
import demo_state_space_model as m1  # noqa: E402

OUT = REPO / "logs/studies/offset_state_model/exp2_offset_in_the_robot_frame"

SIGMA_BODY_PRIOR_M = 0.05        # same prior width exp1 used, for a fair comparison
SIGMA_BODY_WALK = 0.0016
HEADING_WINDOW_S = 1.0           # odometry tangent is averaged over this
MIN_SPEED_MPS = 0.02             # below this the tangent is meaningless; hold the last

PAIRS = [
    ("smoke1_20260716", "fusion_handover_20260721", "same route, same heading"),
    ("smoke1_20260716", "smoke2_20260716", "DIFFERENT heading"),
    ("smoke2_20260716", "smoke1_20260716", "DIFFERENT heading"),
]

C_WORLD = "#CC79A7"
C_BODY = "#009E73"
C_RAW = "#D55E00"
C_FLOOR = "#0072B2"
C_TRUTH = "#111111"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def odometry_heading(capture) -> np.ndarray:
    """Heading per odometry sample from the path tangent. Odometry only, no truth."""
    stamps = np.asarray(capture.stamps, dtype=float)
    odom = np.asarray(capture.odom, dtype=float)
    dt = float(np.median(np.diff(stamps))) if len(stamps) > 1 else 0.02
    span = max(int(round(HEADING_WINDOW_S / max(dt, 1e-6))), 1)

    heading = np.zeros(len(stamps))
    last = 0.0
    for k in range(len(stamps)):
        lo = max(k - span, 0)
        hi = min(k + span, len(stamps) - 1)
        step = odom[hi] - odom[lo]
        elapsed = max(stamps[hi] - stamps[lo], 1e-6)
        if float(np.hypot(*step)) / elapsed >= MIN_SPEED_MPS:
            last = math.atan2(step[1], step[0])
        heading[k] = last
    return heading


def _rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def run_body_frame_arm(capture, truth_lookup, *, sigma_prior, sigma_walk,
                       frozen_offset=None) -> list[dict]:
    """One shared body-frame offset, rotated into the world by odometry heading.

    Deliberately the same protocol as `exp1`'s ladder arms -- per detection,
    filtered, identical prediction and growth -- so the only change from
    `m1.run_as_ladder_arm` is the state vector and the observation matrix.
    """
    dim = 4                                  # [p_x, p_y, b_x, b_y] with b in body frame
    mean = np.zeros(dim)
    mean[:2] = np.asarray(capture.odom[0], dtype=float)
    cov = np.zeros((dim, dim))
    cov[:2, :2] = np.eye(2) * f1.INITIAL_SIGMA_M**2
    cov[2, 2] = cov[3, 3] = sigma_prior**2
    if frozen_offset is not None:
        sigma_walk = 0.0
        mean[2:] = np.asarray(frozen_offset, dtype=float)
        cov[2, 2] = cov[3, 3] = 1e-10

    detections = sorted(
        ((camera, d) for camera in rc.CAMERAS for d in capture.detections[camera]),
        key=lambda item: item[1].stamp,
    )
    stamps = np.asarray(capture.stamps, dtype=float)
    odom = np.asarray(capture.odom, dtype=float)
    heading = odometry_heading(capture)

    records = []
    previous_index = 0
    previous_stamp = float(stamps[0])
    for camera, detection in detections:
        index = int(np.searchsorted(stamps, detection.stamp))
        index = min(max(index, 0), len(stamps) - 1)

        if index > previous_index:
            step = odom[index] - odom[previous_index]
            distance = float(np.hypot(*step))
            mean[:2] = mean[:2] + step
            growth = f1.PROCESS_SIGMA_PER_SQRT_M**2 * max(distance, 1e-6)
            cov[0, 0] += growth
            cov[1, 1] += growth
            previous_index = index
        dt = max(float(detection.stamp) - previous_stamp, 0.0)
        previous_stamp = float(detection.stamp)
        cov[2, 2] += sigma_walk**2 * dt
        cov[3, 3] += sigma_walk**2 * dt

        velocity = np.zeros(2)
        if index > 0:
            gap = float(stamps[index] - stamps[index - 1])
            if gap > 1e-6:
                velocity = (odom[index] - odom[index - 1]) / gap

        sigma = f1.R_COND_SIGMA_M[camera]
        R = np.eye(2) * sigma**2 + np.asarray(
            time_sync_covariance(f1.H, velocity.tolist(), f1.SIGMA_TAU_S), dtype=float)

        # THE change: the offset enters rotated by the robot's current heading
        rotation = _rotation(float(heading[index]))
        H = np.zeros((2, dim))
        H[:, :2] = np.eye(2)
        H[:, 2:] = rotation

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
            "nis": float(innovation @ S_inv @ innovation),
            "responsibility": 1.0, "health": 1.0,
            "cov_post": position_cov.copy(),
            "body_offset": mean[2:].copy(),
            "heading_rad": float(heading[index]),
        })
    return records


def score(records) -> dict:
    summary = f1.summarize(records)
    summary.update(m1.logarithmic_score(records))
    return summary


def fig_b1_body_offsets(fitted, headings) -> dict:
    """One number per capture now, instead of one per camera -- do they agree?"""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                  gridspec_kw={"width_ratios": [1.0, 1.1]})
    colors = {"smoke1_20260716": "#0072B2", "smoke2_20260716": "#D55E00",
              "fusion_handover_20260721": "#009E73"}
    vectors = []
    for name, vector in fitted.items():
        millimetres = 1000 * np.asarray(vector, dtype=float)
        vectors.append(millimetres)
        ax.annotate("", xy=(millimetres[0], millimetres[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2.6, color=colors[name]))
        ax.plot(*millimetres, "o", color=colors[name], ms=8, mec="white", mew=1.2,
                label=f"{name.replace('_20260716', '').replace('_20260721', '')}  "
                      f"({millimetres[0]:+.0f}, {millimetres[1]:+.0f})  "
                      f"|b| = {np.hypot(*millimetres):.0f} mm")
    ax.plot(0, 0, "+", color=C_TRUTH, ms=11, mew=1.8)
    ax.axhline(0, color="#8A8A8A", lw=0.7)
    ax.axvline(0, color="#8A8A8A", lw=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("body-frame offset x (mm)")
    ax.set_ylabel("body-frame offset y (mm)")
    pairwise = max((float(np.linalg.norm(a - b)) for i, a in enumerate(vectors)
                    for b in vectors[i + 1:]), default=0.0)
    magnitude = float(np.mean([np.linalg.norm(v) for v in vectors]))
    ax.set_title(f"ONE offset, in the robot's frame, shared by all four cameras\n"
                 f"fitted independently per capture: they disagree by "
                 f"{pairwise:.0f} mm on a {magnitude:.0f} mm vector",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")

    for name, series in headings.items():
        ax2.plot(np.degrees(series), color=colors[name], lw=2.0,
                 label=name.replace("_20260716", "").replace("_20260721", ""))
    ax2.set_xlabel("odometry sample")
    ax2.set_ylabel("heading from the odometry tangent (deg)")
    ax2.set_title("The heading each capture actually drove — odometry, never truth",
                  loc="left", fontsize=10)
    ax2.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Re-parameterised: the offset belongs to the robot, and the heading "
                 "carries it into each camera's view",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_b1_body_offsets")
    return {"max_pairwise_disagreement_mm": pairwise, "mean_magnitude_mm": magnitude,
            "disagreement_over_magnitude": pairwise / max(magnitude, 1e-9)}


def fig_b2_transfer(world, body) -> dict:
    labels = [f"{test.replace('_20260716', '').replace('_20260721', '')}\n"
              f"from {train.replace('_20260716', '').replace('_20260721', '')}\n"
              f"({note})" for train, test, note in PAIRS]
    positions = np.arange(len(PAIRS))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.1))
    width = 0.38

    for offset, (name, results, colour) in enumerate((
            ("per-camera offsets (exp1)", world, C_WORLD),
            ("one body-frame offset", body, C_BODY))):
        nlpd = [results[(train, test)]["mean_nlpd"] for train, test, _ in PAIRS]
        unearned = [100 * results[(train, test)]["unearned_confidence_fraction"]
                    for train, test, _ in PAIRS]
        ax.bar(positions + (offset - 0.5) * width, nlpd, width * 0.9, color=colour,
               label=name)
        ax2.bar(positions + (offset - 0.5) * width, unearned, width * 0.9,
                color=colour, label=name)
        for i, (a, b) in enumerate(zip(nlpd, unearned)):
            ax.annotate(f"{a:.1f}", xy=(positions[i] + (offset - 0.5) * width, a),
                        xytext=(0, 4 if a >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold")
            ax2.annotate(f"{b:.1f}", xy=(positions[i] + (offset - 0.5) * width, b),
                         xytext=(0, 4), textcoords="offset points", ha="center",
                         fontsize=8, fontweight="bold")

    ax.axhline(0, color=C_TRUTH, lw=0.9)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("mean $-\\log p(\\mathrm{truth})$   (lower = better)")
    ax.set_title("Frozen offsets applied to a capture they were not fitted on",
                 loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    ax2.axhline(5.0, color=C_TRUTH, lw=1.4, ls="--")
    ax2.annotate("nominal 5 %", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=8.5)
    ax2.set_xticks(positions, labels, fontsize=8)
    ax2.set_ylabel("truth outside the stated 95 % ellipse (%)")
    ax2.set_title("Unearned confidence on the held-out capture", loc="left")
    ax2.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Does the re-parameterisation transfer where the per-camera version "
                 "did not?", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_b2_transfer")
    return {}


def main() -> int:
    d1._style()
    OUT.mkdir(parents=True, exist_ok=True)
    models = rc.camera_models()
    calib = rc.deployed_calibration()

    loaded, fitted, headings = {}, {}, {}
    for name in rc.CAPTURES:
        capture = rc.load_operational_capture(name, models=models, calib=calib)
        table = rc.load_truth_table(name)                # EVALUATION ONLY

        def lookup(stamp, _table=table):
            return rc.truth_at(_table, stamp)

        loaded[name] = (capture, lookup)
        headings[name] = odometry_heading(capture)
        records = run_body_frame_arm(capture, lookup, sigma_prior=SIGMA_BODY_PRIOR_M,
                                     sigma_walk=SIGMA_BODY_WALK)
        fitted[name] = records[-1]["body_offset"].copy()

    print("one body-frame offset per capture, shared across all four cameras (mm):")
    for name, vector in fitted.items():
        millimetres = 1000 * np.asarray(vector)
        print(f"  {name:<26} ({millimetres[0]:+6.1f}, {millimetres[1]:+6.1f})   "
              f"|b| = {np.hypot(*millimetres):5.1f}   "
              f"heading {np.degrees(np.median(headings[name])):+7.1f} deg")
    agreement = fig_b1_body_offsets(fitted, headings)

    print("\ntransfer: freeze on train, apply to test")
    body, world = {}, {}
    for train, test, note in PAIRS:
        capture, lookup = loaded[test]
        body[(train, test)] = score(run_body_frame_arm(
            capture, lookup, sigma_prior=SIGMA_BODY_PRIOR_M,
            sigma_walk=SIGMA_BODY_WALK, frozen_offset=fitted[train]))
        # exp1's per-camera version, refitted here so the comparison is self-contained
        trained = m1.run_as_ladder_arm(
            loaded[train][0], loaded[train][1], sigma_bias_prior=0.05,
            sigma_bias_walk_per_sqrt_s=0.0016)
        per_camera = {c: np.asarray(v, dtype=float)
                      for c, v in trained[-1]["offset_xy"].items()}
        world[(train, test)] = score(m1.run_as_ladder_arm(
            capture, lookup, sigma_bias_prior=0.05,
            sigma_bias_walk_per_sqrt_s=0.0016, frozen_offsets=per_camera))
        print(f"\n  {train} -> {test}   ({note})")
        print(f"    {'parameterisation':<26}{'NLPD':>9}{'medNEES':>9}"
              f"{'unearned%':>11}{'RMSE cm':>9}{'stated cm':>11}")
        for label, summary in (("per-camera (exp1)", world[(train, test)]),
                               ("one body-frame offset", body[(train, test)])):
            print(f"    {label:<26}{summary['mean_nlpd']:>9.2f}"
                  f"{summary['median_nees']:>9.2f}"
                  f"{100 * summary['unearned_confidence_fraction']:>11.1f}"
                  f"{100 * summary['rmse_m']:>9.1f}"
                  f"{100 * summary['mean_stated_sigma_m']:>11.1f}")

    fig_b2_transfer(world, body)

    cross = [p for p in PAIRS if "DIFFERENT" in p[2]]
    body_wins = all(body[(t, s)]["mean_nlpd"] < world[(t, s)]["mean_nlpd"]
                    for t, s, _ in cross)
    body_honest = all(body[(t, s)]["unearned_confidence_fraction"] < 0.10
                      for t, s, _ in cross)
    verdict = (
        "the offset is a ROBOT property: one shared body-frame vector transfers "
        "across headings where eight per-camera vectors did not"
        if body_wins and body_honest else
        "the body-frame form transfers better but is not yet honest across headings"
        if body_wins else
        "the re-parameterisation does not rescue the transfer"
    )
    print(f"\nVERDICT: {verdict}")

    rc.write_json(OUT / "summary.json", {
        "question": "does the offset belong to the robot rather than the camera?",
        "body_offsets_mm": {n: (1000 * np.asarray(v)).tolist()
                            for n, v in fitted.items()},
        "median_heading_deg": {n: float(np.degrees(np.median(h)))
                               for n, h in headings.items()},
        "body_offset_agreement": agreement,
        "transfer": {f"{t}->{s}": {"per_camera": world[(t, s)],
                                   "body_frame": body[(t, s)]}
                     for t, s, _ in PAIRS},
        "verdict": verdict,
        "body_frame_transfers": bool(body_wins),
        "body_frame_honest_across_headings": bool(body_honest),
    })
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
