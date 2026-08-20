#!/usr/bin/env python3
"""Why letting the filter learn the camera's noise level makes it WORSE.

Three panels, one mechanism, in the order it happens. Everything is measured on
the real capture; ground truth is used only to score and to draw, never inside a
model.

  1  WHERE THE BIAS GOES.  Nothing pins the estimated path absolutely, so it
     slides sideways onto the biased camera's readings. Once it has, that camera's
     readings sit almost on top of the estimate -- the bias has moved OUT of the
     residuals and INTO the state. The estimator is left looking at scatter only.

  2  THE LOOP.  Each round: smaller estimated noise -> more trust in that camera
     -> the path slides further onto it -> its residuals shrink again. The
     estimated noise falls while the real error does not.

  3  THE TRAP.  The model that fits the observations best is the least honest one,
     so ordinary model selection picks it.

Reuses the machinery in ``bayesian_filter_showcase/demo_state_space_model.py``
rather than reimplementing the filter.

Outputs -> logs/studies/offset_state_closed_loop/why_learning_r_backfires/
"""

from __future__ import annotations

import json
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
for _rel in ("src/reliability", "src/unav_common", "src/state",
             "experiments/operational_residual_rcond",
             "experiments/bayesian_filter_showcase"):
    sys.path.insert(0, str(REPO / _rel))

import rcond_common as rc                       # noqa: E402
import exp1_graceful_vs_trusting as f1          # noqa: E402
import demo_state_space_model as d2             # noqa: E402

OUT = REPO / "logs/studies/offset_state_closed_loop/why_learning_r_backfires"
CAPTURE = "smoke1_20260716"
LIAR = "camera_C"
ITERATIONS = 12

TRUTH_C = "#009E73"      # ground truth
EST_C = "#0072B2"        # what the filter believes
CAM_C = "#D55E00"        # the leaning camera
INK = "#1A1A1A"
GREY = "#8A8A8A"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.grid": True, "grid.color": "#E2E2E2", "grid.linewidth": 0.6,
    })


def em_trace(seq, m0, S0, q):
    """The learn-R loop, recording what it believes and what it is actually doing.

    Same update as demo_state_space_model.infer_covariance, but scored against
    truth every round so the divergence is visible.
    """
    prior_nu, prior_scale = 6.0, np.eye(2) * (0.05**2) * 6.0
    current = prior_scale / prior_nu
    rows = []

    for _ in range(ITERATIONS):
        r_per_cam = {c: current for c in rc.CAMERAS}
        model = d2.PositionModel(Q=q, R_per_camera=r_per_cam, m0=m0, S0=S0)
        forward = d2.kalman_filter(seq, model)
        smooth = d2.rts_smoother(seq, model, forward)

        # What the estimator can see: residuals of the leaning camera.
        resid, nees, err = [], [], []
        accumulator = np.zeros((2, 2))
        count = 0
        for k in range(seq.n_steps):
            if seq.camera[k] is None:
                continue
            H = model.H(seq.camera[k])
            r = seq.y[k] - H @ smooth["m"][k]
            accumulator += np.outer(r, r) + H @ smooth["P"][k] @ H.T
            count += 1
            if seq.camera[k] == LIAR:
                resid.append(float(np.linalg.norm(r)))
        # What is actually true: error of the estimate against ground truth.
        for k in range(seq.n_steps):
            if np.isnan(seq.truth[k, 0]):
                continue
            d = smooth["m"][k][:2] - seq.truth[k]
            P = smooth["P"][k][:2, :2]
            err.append(float(d @ d))
            nees.append(float(d @ np.linalg.solve(P, d)))

        rows.append({
            "stated_sigma_mm": 1000.0 * float(np.sqrt(np.mean(np.diag(current)))),
            "liar_residual_mm": 1000.0 * float(np.mean(resid)) if resid else np.nan,
            "true_rmse_mm": 1000.0 * float(np.sqrt(np.mean(err))),
            "median_nees": float(np.median(nees)),
        })
        current = (prior_scale + accumulator) / (prior_nu + count)

    return rows, smooth


def panel_where(ax, seq, smooth_learned) -> None:
    """One coordinate against time: the estimate slides onto the biased camera.

    Drawn as a time series rather than a map: the stretches where camera C is the
    one reporting are nearly straight, so a map view collapses to a sliver under
    equal aspect and the annotations have nowhere to sit.
    """
    seen = [k for k in range(seq.n_steps)
            if seq.camera[k] == LIAR and not np.isnan(seq.truth[k, 0])]
    if len(seen) < 20:
        raise RuntimeError("not enough camera_C steps with truth to draw")

    # Show the axis the lean is most visible on.
    gaps = np.abs(seq.y[seen] - seq.truth[seen])
    axis = int(np.argmax(gaps.mean(axis=0)))
    name = "xy"[axis]

    lo, hi = seen[len(seen) // 3], seen[len(seen) // 3 + 18]
    span = np.arange(lo, hi + 1)
    t0 = float(seq.stamps[lo])
    secs = seq.stamps[span] - t0

    ax.plot(secs, seq.truth[span, axis], color=TRUTH_C, lw=3.2,
            label="where the robot really was", zorder=3)
    ax.plot(secs, [smooth_learned["m"][k][axis] for k in span], color=EST_C,
            lw=2.4, ls="--", label="where the filter thinks it was", zorder=4)
    in_span = [k for k in span if seq.camera[k] == LIAR]
    ax.plot(seq.stamps[in_span] - t0, seq.y[in_span, axis], marker="o", ms=6.5,
            ls="none", color=CAM_C, label="what the leaning camera reported",
            zorder=5)

    # The two gaps at one representative sighting.
    k = in_span[len(in_span) // 2]
    x = float(seq.stamps[k] - t0)
    z, e, tr = seq.y[k, axis], smooth_learned["m"][k][axis], seq.truth[k, axis]
    small, big = 1000.0 * abs(z - e), 1000.0 * abs(e - tr)

    ax.annotate("", xy=(x, z), xytext=(x, e),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
    ax.annotate("", xy=(x + 0.35, e), xytext=(x + 0.35, tr),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.8))
    ax.annotate(f"what the estimator can see:\n{small:.0f} mm — looks tiny",
                xy=(x, (z + e) / 2), xytext=(x + 0.45, z + 0.135),
                fontsize=9.5, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.annotate(f"what actually matters:\n{big:.0f} mm — invisible to it",
                xy=(x + 0.35, (e + tr) / 2), xytext=(x - 1.75, tr - 0.125),
                fontsize=9.5, fontweight="bold", color="#5A5A5A",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))

    ax.set_xlabel("seconds")
    ax.set_ylabel(f"position along {name} (m)")
    ax.set_title("1.  The estimate slides onto the leaning camera\n"
                 "so that camera's readings stop looking wrong",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.margins(y=0.24)


def panel_loop(ax, rows) -> None:
    """Each round it trusts the liar more, and is more wrong."""
    n = np.arange(1, len(rows) + 1)
    stated = [r["stated_sigma_mm"] for r in rows]
    true_err = [r["true_rmse_mm"] for r in rows]

    ax.plot(n, stated, marker="o", ms=5, color=EST_C, lw=2.2,
            label="noise it thinks the cameras have")
    ax.plot(n, true_err, marker="s", ms=5, color=TRUTH_C, lw=2.2,
            label="how wrong it actually is")
    ax.fill_between(n, stated, true_err, color="#D55E00", alpha=0.12)

    ax.annotate(f"starts by assuming {stated[0]:.0f} mm",
                xy=(1, stated[0]), xytext=(1.6, stated[0] + 14),
                fontsize=9, color=EST_C,
                arrowprops=dict(arrowstyle="->", color=EST_C, lw=1.2))
    ax.annotate(f"talks itself down to {stated[-1]:.0f} mm",
                xy=(len(rows), stated[-1]), xytext=(len(rows) - 5.4, stated[-1] - 20),
                fontsize=9, fontweight="bold", color=EST_C,
                arrowprops=dict(arrowstyle="->", color=EST_C, lw=1.2))
    ax.text(len(rows) / 2.0, (stated[len(rows) // 2] + true_err[len(rows) // 2]) / 2.0,
            "this gap is\nthe overconfidence", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#B4530A")

    ax.set_xlabel("round of learning")
    ax.set_ylabel("millimetres")
    ax.set_ylim(0, max(true_err + stated) * 1.35)
    ax.set_title("2.  Every round it trusts that camera more\n"
                 "the claim shrinks; the real error does not",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")


def panel_trap(ax) -> None:
    """The best-fitting model is the least honest one."""
    payload = json.loads((REPO / "logs/studies/bayesian_filter_showcase"
                          / "demo_state_space_model/summary.json").read_text())
    comparison = payload["step5_model_comparison"]

    wanted = [
        ("states only\n(fixed noise)", "states only (deployed R)"),
        ("+ learn how fast\nodometry drifts", "+ inferred $Q$"),
        ("+ learn the camera\nnoise level", "+ inferred $R$"),
        ("+ learn each camera's\nsideways offset", "+ per-camera offset states"),
    ]
    labels, fit, honesty = [], [], []
    for nice, key in wanted:
        entry = comparison.get(key)
        if entry is None:
            raise KeyError(f"{key!r} missing; summary.json keys: {list(comparison)}")
        labels.append(nice)
        fit.append(float(entry["evidence_gain_over_baseline"]))
        honesty.append(100.0 * float(entry["smoothed"]["unearned_confidence_fraction"]))

    positions = np.arange(len(labels))
    ax.bar(positions, fit, 0.58, color=[GREY, GREY, CAM_C, EST_C])
    ax.set_ylabel("how much better it fits what the cameras said\n"
                  "(taller = a better fit to the data)")
    ax.set_xticks(positions, labels, fontsize=8.5)
    ax.set_ylim(0, max(fit) * 1.45)
    for x, (v, h) in enumerate(zip(fit, honesty)):
        # Both numbers stacked above the bar; in-bar text gets clipped by the
        # narrow bars and by the zero-height baseline.
        ax.text(x, v + max(fit) * 0.035,
                f"+{v:.0f}\nwrong about itself\n{h:.0f} % of the time",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=INK, linespacing=1.45)
    ax.set_title("3.  The best fit is the worst filter\n"
                 "so picking by fit alone chooses the bad one",
                 fontweight="bold", fontsize=10.5)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()

    models = rc.camera_models()
    capture = rc.load_operational_capture(CAPTURE, models=models,
                                          calib=rc.deployed_calibration())
    seq = d2.Sequence(capture, rc.load_truth_table(CAPTURE))
    q = np.eye(2) * (f1.PROCESS_SIGMA_PER_SQRT_M**2 * 0.02)
    m0, S0 = seq.odom[0].copy(), np.eye(2) * f1.INITIAL_SIGMA_M**2

    rows, smooth_learned = em_trace(seq, m0, S0, q)
    for i, r in enumerate(rows, 1):
        print(f"  round {i:2d}: thinks {r['stated_sigma_mm']:5.1f} mm | "
              f"liar residual {r['liar_residual_mm']:5.1f} mm | "
              f"really off {r['true_rmse_mm']:5.1f} mm | NEES {r['median_nees']:6.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.2))
    panel_where(axes[0], seq, smooth_learned)
    panel_loop(axes[1], rows)
    panel_trap(axes[2])

    fig.suptitle("Letting the filter work out its own camera noise makes it "
                 "confidently wrong", fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.918,
             f"{CAPTURE}, 539 detections from four cameras under retired v2. Camera C carries a "
             "historical 77 mm signed sideways residual — MECHANISM evidence, not current "
             "camera accuracy (under the current path all four sit within 64.6–68.1 mm; see "
             "docs/localization_metrics.md). Ground truth scores and draws — it never "
             "enters a model.",
             ha="center", va="top", fontsize=8.0, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.878))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_r1_learning_r_backfires.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
