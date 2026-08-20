#!/usr/bin/env python3
"""Is a per-camera offset a property of the CAMERA, or of the route it was fitted on?

`bayesian_filter_showcase/demo_state_space_model.py` added a per-camera 2-D offset
to the state and it beat the correlation floor on a strictly proper score. Before
that becomes a claim it has to survive the objection E6 already raised against the
fitted calibration terms it replaces:

    nothing in the model forces the estimated offset to be a property of the
    camera. It absorbs whatever is systematic for that camera ON THAT ROUTE --
    including the robot's silhouette geometry seen from that camera's bearing at
    that route's heading, which `pixel_ground_path/e6` showed accounts for
    substantially all of camera C's historical-v2 77 mm signed residual.

So this experiment fits the offsets on one capture, FREEZES them, and applies them
to another. The design has a built-in positive control, because the three captures
are not three routes:

    smoke1              yaw = +90 deg on 2475/2475 truth rows
    fusion_handover     the SAME route as smoke1 (median nearest-neighbour 0.006 m)
    smoke2              yaw = 0 deg on 1140/1140 rows

  * smoke1 -> fusion_handover  same route, same heading. If an offset transfers
    anywhere it must transfer here. A failure here means the estimate is simply
    noisy and nothing further can be concluded.
  * smoke1 <-> smoke2          different heading. THIS is the test. Transfer here
    means the offset is a camera property; failure means it is route-conditioned.

Four arms are scored on every test capture so the transfer is judged against
alternatives rather than against zero:

    raw          sharp per-camera R, no offset handling at all (the A2 setting)
    frozen       offsets fixed at the values fitted on the TRAIN capture
    estimated    offsets estimated on the test capture itself -- the upper bound
    floor (A4)   the deployed correlation floor, for reference

Ground truth is EVALUATION-ONLY throughout: it scores, it never enters a filter.

Outputs -> logs/studies/offset_state_model/exp1_do_the_offsets_transfer/
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

import rcond_common as rc  # noqa: E402
import exp1_graceful_vs_trusting as f1  # noqa: E402
import demo_how_the_filter_works as d1  # noqa: E402
import demo_state_space_model as m1  # noqa: E402

OUT = REPO / "logs/studies/offset_state_model/exp1_do_the_offsets_transfer"

SIGMA_BIAS_PRIOR_M = 0.05          # the setting the prior sweep selected
SIGMA_BIAS_WALK = 0.0016

#: (train, test, what the pair tests)
PAIRS = [
    ("smoke1_20260716", "fusion_handover_20260721", "same route, same heading"),
    ("smoke1_20260716", "smoke2_20260716", "DIFFERENT heading"),
    ("smoke2_20260716", "smoke1_20260716", "DIFFERENT heading"),
]

C_RAW = "#D55E00"
C_FROZEN = "#CC79A7"
C_ESTIMATED = "#009E73"
C_FLOOR = "#0072B2"
C_TRUTH = "#111111"
C_MUTED = "#8A8A8A"

ARM_COLOR = {"raw": C_RAW, "frozen": C_FROZEN, "estimated": C_ESTIMATED,
             "floor (A4)": C_FLOOR}
CAPTURE_COLOR = {"smoke1_20260716": "#0072B2",
                 "smoke2_20260716": "#D55E00",
                 "fusion_handover_20260721": "#009E73"}


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def load(name, models, calib):
    capture = rc.load_operational_capture(name, models=models, calib=calib)
    table = rc.load_truth_table(name)              # EVALUATION ONLY

    def lookup(stamp, _table=table):
        return rc.truth_at(_table, stamp)

    return capture, lookup


def fit_offsets(capture, lookup) -> dict:
    """Estimate the offsets on this capture and return the final values."""
    records = m1.run_as_ladder_arm(
        capture, lookup, sigma_bias_prior=SIGMA_BIAS_PRIOR_M,
        sigma_bias_walk_per_sqrt_s=SIGMA_BIAS_WALK)
    if not records:
        return {c: np.zeros(2) for c in rc.CAMERAS}
    return {c: np.asarray(v, dtype=float)
            for c, v in records[-1]["offset_xy"].items()}


def observed_cameras(capture) -> dict:
    return {c: len(capture.detections[c]) for c in rc.CAMERAS}


def score_records(records) -> dict:
    summary = f1.summarize(records)
    summary.update(m1.logarithmic_score(records))
    return summary


def run_arms(capture, lookup, frozen) -> dict:
    """Four arms on one test capture, same protocol as exp1's ladder."""
    arms = {}
    # 'raw': the offsets exist as states but are pinned to zero and cannot move,
    # which is exactly 'sharp R, no offset handling'.
    arms["raw"] = score_records(m1.run_as_ladder_arm(
        capture, lookup, sigma_bias_prior=SIGMA_BIAS_PRIOR_M,
        sigma_bias_walk_per_sqrt_s=SIGMA_BIAS_WALK,
        frozen_offsets={c: np.zeros(2) for c in rc.CAMERAS}))
    arms["frozen"] = score_records(m1.run_as_ladder_arm(
        capture, lookup, sigma_bias_prior=SIGMA_BIAS_PRIOR_M,
        sigma_bias_walk_per_sqrt_s=SIGMA_BIAS_WALK, frozen_offsets=frozen))
    arms["estimated"] = score_records(m1.run_as_ladder_arm(
        capture, lookup, sigma_bias_prior=SIGMA_BIAS_PRIOR_M,
        sigma_bias_walk_per_sqrt_s=SIGMA_BIAS_WALK))
    arms["floor (A4)"] = score_records(
        d1.trace_arm(capture, "A4_correlation_floor", lookup))
    return arms


def fig_t1_the_offsets(per_capture_offsets, counts) -> dict:
    """The most direct evidence: do the fitted vectors agree across captures?"""
    cameras = [c for c in rc.CAMERAS
               if any(counts[name][c] > 0 for name in per_capture_offsets)]
    fig, axes = plt.subplots(1, len(cameras), figsize=(3.5 * len(cameras), 4.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    spread = {}
    for ax, camera in zip(axes, cameras):
        vectors = []
        for name, offsets in per_capture_offsets.items():
            if counts[name][camera] == 0:
                continue
            vector = 1000 * np.asarray(offsets[camera], dtype=float)
            vectors.append(vector)
            ax.annotate("", xy=(vector[0], vector[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", lw=2.4,
                                        color=CAPTURE_COLOR[name]))
            ax.plot(*vector, "o", color=CAPTURE_COLOR[name], ms=7, mec="white",
                    mew=1.2)
        ax.plot(0, 0, "+", color=C_TRUTH, ms=10, mew=1.6)
        ax.axhline(0, color=C_MUTED, lw=0.7)
        ax.axvline(0, color=C_MUTED, lw=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("offset x (mm)")
        if len(vectors) >= 2:
            pairwise = max(float(np.linalg.norm(a - b))
                           for i, a in enumerate(vectors)
                           for b in vectors[i + 1:])
            magnitude = float(np.mean([np.linalg.norm(v) for v in vectors]))
            spread[camera] = {"max_pairwise_disagreement_mm": pairwise,
                              "mean_magnitude_mm": magnitude,
                              "disagreement_over_magnitude": pairwise / max(magnitude, 1e-9)}
            ax.set_title(f"{camera.replace('camera_', 'camera ')}\n"
                         f"they disagree by {pairwise:.0f} mm on a "
                         f"{magnitude:.0f} mm offset",
                         loc="left", fontsize=9.5)
        else:
            ax.set_title(camera.replace("camera_", "camera "), loc="left",
                         fontsize=9.5)
    axes[0].set_ylabel("offset y (mm)")
    handles = [plt.Line2D([], [], color=CAPTURE_COLOR[n], lw=2.4,
                          label=n.replace("_20260716", "").replace("_20260721", ""))
               for n in per_capture_offsets]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("The offsets the model estimates, fitted independently on each capture — "
                 "an arrow per capture, per camera",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_t1_the_offsets")
    return spread


def fig_t2_transfer(results) -> dict:
    """Does a frozen offset help on a capture it was not fitted on?"""
    arms = ["raw", "floor (A4)", "frozen", "estimated"]
    labels = [f"{test.replace('_20260716', '').replace('_20260721', '')}\n"
              f"trained on {train.replace('_20260716', '').replace('_20260721', '')}\n"
              f"({note})" for train, test, note in PAIRS]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    width = 0.2
    positions = np.arange(len(PAIRS))

    for offset, arm in enumerate(arms):
        values = [results[(train, test)][arm]["mean_nlpd"] for train, test, _ in PAIRS]
        ax.bar(positions + (offset - 1.5) * width, values, width * 0.92,
               color=ARM_COLOR[arm], label=arm)
        unearned = [100 * results[(train, test)][arm]["unearned_confidence_fraction"]
                    for train, test, _ in PAIRS]
        ax2.bar(positions + (offset - 1.5) * width, unearned, width * 0.92,
                color=ARM_COLOR[arm], label=arm)
        for i, value in enumerate(unearned):
            ax2.annotate(f"{value:.1f}", xy=(positions[i] + (offset - 1.5) * width, value),
                         xytext=(0, 3), textcoords="offset points", ha="center",
                         fontsize=7)

    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("mean $-\\log p(\\mathrm{truth})$   (lower = better)")
    ax.set_title("The proper score on the held-out capture", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.axhline(0, color=C_TRUTH, lw=0.8)

    ax2.axhline(5.0, color=C_TRUTH, lw=1.4, ls="--")
    ax2.annotate("nominal 5 %", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=8.5)
    ax2.set_xticks(positions, labels, fontsize=8)
    ax2.set_ylabel("truth outside the stated 95 % ellipse (%)")
    ax2.set_title("Unearned confidence on the held-out capture", loc="left")
    ax2.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Fit the offsets on one capture, FREEZE them, apply to another",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_t2_transfer")
    return {f"{train}->{test}": {arm: results[(train, test)][arm] for arm in arms}
            for train, test, _ in PAIRS}


def main() -> int:
    d1._style()
    OUT.mkdir(parents=True, exist_ok=True)
    models = rc.camera_models()
    calib = rc.deployed_calibration()

    loaded, counts, offsets = {}, {}, {}
    for name in rc.CAPTURES:
        capture, lookup = load(name, models, calib)
        loaded[name] = (capture, lookup)
        counts[name] = observed_cameras(capture)
        offsets[name] = fit_offsets(capture, lookup)

    print("offsets fitted independently on each capture (mm):")
    header = "  " + "camera".ljust(10) + "".join(n.replace("_2026", " ").ljust(22)
                                                 for n in rc.CAPTURES)
    print(header)
    for camera in rc.CAMERAS:
        cells = []
        for name in rc.CAPTURES:
            if counts[name][camera] == 0:
                cells.append("not seen".ljust(22))
                continue
            vector = 1000 * np.asarray(offsets[name][camera], dtype=float)
            cells.append(f"({vector[0]:+6.1f},{vector[1]:+6.1f})".ljust(22))
        print("  " + camera.replace("camera_", "").ljust(10) + "".join(cells))

    spread = fig_t1_the_offsets(offsets, counts)

    print("\ntransfer, four arms on the held-out capture:")
    results = {}
    for train, test, note in PAIRS:
        capture, lookup = loaded[test]
        results[(train, test)] = run_arms(capture, lookup, offsets[train])
        print(f"\n  train {train} -> test {test}   ({note})")
        print(f"    {'arm':<12}{'NLPD':>9}{'medNEES':>9}{'unearned%':>11}"
              f"{'RMSE cm':>9}{'stated cm':>11}")
        for arm, summary in results[(train, test)].items():
            print(f"    {arm:<12}{summary['mean_nlpd']:>9.2f}"
                  f"{summary['median_nees']:>9.2f}"
                  f"{100 * summary['unearned_confidence_fraction']:>11.1f}"
                  f"{100 * summary['rmse_m']:>9.1f}"
                  f"{100 * summary['mean_stated_sigma_m']:>11.1f}")

    transfer = fig_t2_transfer(results)

    # --- the verdict, computed rather than asserted
    control = results[("smoke1_20260716", "fusion_handover_20260721")]
    cross = [results[("smoke1_20260716", "smoke2_20260716")],
             results[("smoke2_20260716", "smoke1_20260716")]]
    control_helps = control["frozen"]["mean_nlpd"] < control["raw"]["mean_nlpd"]
    cross_helps = all(r["frozen"]["mean_nlpd"] < r["raw"]["mean_nlpd"] for r in cross)
    verdict = (
        "offset is a CAMERA property: it transfers across headings"
        if control_helps and cross_helps else
        "offset is ROUTE-CONDITIONED: it transfers on the repeat route but not "
        "across headings" if control_helps and not cross_helps else
        "INCONCLUSIVE: the estimate does not even transfer to a repeat of its own route"
    )
    print(f"\nVERDICT: {verdict}")
    print(f"  same-route control: frozen {control['frozen']['mean_nlpd']:.2f} vs "
          f"raw {control['raw']['mean_nlpd']:.2f}  -> "
          f"{'helps' if control_helps else 'does not help'}")
    for (train, test, note), r in zip(PAIRS[1:], cross):
        print(f"  {train[:6]} -> {test[:6]}: frozen {r['frozen']['mean_nlpd']:.2f} vs "
              f"raw {r['raw']['mean_nlpd']:.2f}  -> "
              f"{'helps' if r['frozen']['mean_nlpd'] < r['raw']['mean_nlpd'] else 'does NOT help'}")

    rc.write_json(OUT / "summary.json", {
        "question": "is a per-camera offset a property of the camera or of the route?",
        "sigma_bias_prior_m": SIGMA_BIAS_PRIOR_M,
        "offsets_mm": {name: {c: (1000 * np.asarray(v)).tolist()
                              for c, v in offsets[name].items()}
                       for name in offsets},
        "detections_per_camera": counts,
        "offset_disagreement": spread,
        "transfer": transfer,
        "verdict": verdict,
        "control_transfers": bool(control_helps),
        "cross_heading_transfers": bool(cross_helps),
    })
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
