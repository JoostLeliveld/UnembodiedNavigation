#!/usr/bin/env python3
"""Give every camera its OWN learned noise level. Does the liar collapse hardest?

Prediction under test: whichever camera the estimated trajectory already sits
nearest gets the smallest residuals, so the smallest learned noise, so the most
weight, so it pulls the trajectory harder still. Rich-get-richer, with the worst
camera ending up claiming to be the most precise.

Same variational loop as ``fig_why_learning_r_backfires.py``, one change: the
inverse-Wishart posterior is accumulated per camera instead of pooled across all
four. Pooled is re-run here too so the two are directly comparable.

Camera A contributes zero detections to this capture, so its noise level can only
ever stay at the prior. Reported as "no data", never as an estimate.

Outputs -> logs/studies/offset_state_closed_loop/per_camera_r/
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

OUT = REPO / "logs/studies/offset_state_closed_loop/per_camera_r"
CAPTURE = "smoke1_20260716"
ITERATIONS = 14
PRIOR_NU = 6.0
PRIOR_SIGMA_M = 0.05

CAM_COLOR = {"camera_A": "#999999", "camera_B": "#009E73",
             "camera_C": "#D55E00", "camera_D": "#CC79A7"}
INK = "#1A1A1A"
POOLED_C = "#0072B2"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.grid": True, "grid.color": "#E2E2E2", "grid.linewidth": 0.6,
    })


def _score(seq, smooth) -> tuple[float, float, float]:
    nees, err = [], []
    for k in range(seq.n_steps):
        if np.isnan(seq.truth[k, 0]):
            continue
        d = smooth["m"][k][:2] - seq.truth[k]
        P = smooth["P"][k][:2, :2]
        err.append(float(d @ d))
        nees.append(float(d @ np.linalg.solve(P, d)))
    nees = np.asarray(nees)
    # chi2(2) 95th percentile = 5.991: truth outside the stated 95 % ellipse.
    return (float(np.median(nees)),
            float(np.mean(nees > 5.991)),
            1000.0 * float(np.sqrt(np.mean(err))))


def em(seq, m0, S0, q, *, per_camera: bool):
    """The learn-the-noise loop. ``per_camera`` switches pooled vs one each."""
    prior_scale = np.eye(2) * (PRIOR_SIGMA_M**2) * PRIOR_NU
    cameras = list(rc.CAMERAS)
    current = {c: prior_scale / PRIOR_NU for c in cameras}
    counts = {c: 0 for c in cameras}
    history = {c: [] for c in cameras}
    outcome = []

    for _ in range(ITERATIONS):
        model = d2.PositionModel(Q=q, R_per_camera=current, m0=m0, S0=S0)
        forward = d2.kalman_filter(seq, model)
        smooth = d2.rts_smoother(seq, model, forward)

        for c in cameras:
            history[c].append(1000.0 * float(np.sqrt(np.mean(np.diag(current[c])))))
        med, unearned, rmse = _score(seq, smooth)
        outcome.append({"median_nees": med, "unearned": unearned, "rmse_mm": rmse})

        acc = {c: np.zeros((2, 2)) for c in cameras}
        counts = {c: 0 for c in cameras}
        for k in range(seq.n_steps):
            cam = seq.camera[k]
            if cam is None:
                continue
            H = model.H(cam)
            r = seq.y[k] - H @ smooth["m"][k]
            acc[cam] += np.outer(r, r) + H @ smooth["P"][k] @ H.T
            counts[cam] += 1

        if per_camera:
            current = {c: (prior_scale + acc[c]) / (PRIOR_NU + counts[c])
                       for c in cameras}
        else:
            pooled_acc = sum(acc.values())
            pooled_n = sum(counts.values())
            shared = (prior_scale + pooled_acc) / (PRIOR_NU + pooled_n)
            current = {c: shared for c in cameras}

    return {"history": history, "outcome": outcome, "counts": counts,
            "final": {c: history[c][-1] for c in cameras}}


def panel_collapse(ax, per_cam, bias_mm) -> None:
    rounds = np.arange(1, ITERATIONS + 1)
    for cam, series in per_cam["history"].items():
        short = cam.replace("camera_", "")
        n = per_cam["counts"][cam]
        if n == 0:
            ax.plot(rounds, series, color=CAM_COLOR[cam], lw=2.0, ls=":",
                    label=f"camera {short} — never seen, no data")
            continue
        ax.plot(rounds, series, marker="o", ms=4.5, color=CAM_COLOR[cam], lw=2.4,
                label=f"camera {short} — {n} sightings, "
                      f"v2 signed bias {bias_mm[cam]:.0f} mm")
        ax.annotate(f"{series[-1]:.0f} mm", xy=(ITERATIONS, series[-1]),
                    xytext=(ITERATIONS + 0.35, series[-1]),
                    fontsize=9.5, fontweight="bold", color=CAM_COLOR[cam],
                    va="center")

    ax.set_xlabel("round of learning")
    ax.set_ylabel("noise each camera claims to have (mm)")
    ax.set_xlim(0.6, ITERATIONS + 2.6)
    ax.set_ylim(0, PRIOR_SIGMA_M * 1000 * 1.18)
    ax.set_title("Each camera now rates its own noise\n"
                 "and they do separate, in the order of this capture's residuals",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left")


def panel_claim_vs_truth(ax, per_cam, bias_mm) -> None:
    cams = [c for c in rc.CAMERAS if per_cam["counts"][c] > 0]
    positions = np.arange(len(cams))
    width = 0.38
    claimed = [per_cam["final"][c] for c in cams]
    actual = [bias_mm[c] for c in cams]

    ax.bar(positions - width / 2, actual, width, color="#555555",
           label="signed bias present in THIS capture (retired v2 path)")
    ax.bar(positions + width / 2, claimed, width,
           color=[CAM_COLOR[c] for c in cams],
           label="the noise it now claims (colour = camera, as on the left)")
    for x, (a, c) in enumerate(zip(actual, claimed)):
        ax.text(x - width / 2, a + 1.6, f"{a:.0f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#555555")
        ax.text(x + width / 2, c + 1.6, f"{c:.0f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)

    ax.set_xticks(positions, [f"camera {c.replace('camera_', '')}" for c in cams])
    ax.set_ylabel("millimetres")
    ax.set_ylim(0, max(actual + claimed) * 1.28)
    worst = max(cams, key=lambda c: bias_mm[c])
    ratio = bias_mm[worst] / max(per_cam["final"][worst], 1e-6)
    ax.set_title("But the SIZES come out far too small\n"
                 f"a {bias_mm[worst]:.0f} mm bias is owned up to as "
                 f"{per_cam['final'][worst]:.0f} — short by {ratio:.1f}×",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.0, loc="upper left")


def panel_outcome(ax, pooled, per_cam) -> None:
    labels = ["fixed noise\n(before any learning)",
              "one shared noise level\nlearned for all cameras",
              "a separate noise level\nlearned per camera"]
    unearned = [100 * pooled["outcome"][0]["unearned"],
                100 * pooled["outcome"][-1]["unearned"],
                100 * per_cam["outcome"][-1]["unearned"]]
    rmse = [pooled["outcome"][0]["rmse_mm"],
            pooled["outcome"][-1]["rmse_mm"],
            per_cam["outcome"][-1]["rmse_mm"]]

    positions = np.arange(len(labels))
    ax.bar(positions, unearned, 0.55, color=["#999999", POOLED_C, "#D55E00"])
    for x, (u, r) in enumerate(zip(unearned, rmse)):
        ax.text(x, u + 1.4, f"{u:.0f} %\n{r:.0f} mm off", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK, linespacing=1.4)
    ax.axhline(5.0, color="#009E73", lw=2.0, ls="--", label="should be 5 %")
    ax.set_xticks(positions, labels, fontsize=8.5)
    ax.set_ylabel("% of the time the robot's true position\n"
                  "was outside its own 95 % ellipse")
    ax.set_ylim(0, max(unearned) * 1.30)
    ax.set_title("And knowing the order does not help\n"
                 "both ways of learning end up worse than not learning at all",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()

    models = rc.camera_models()
    capture = rc.load_operational_capture(CAPTURE, models=models,
                                          calib=rc.deployed_calibration())
    seq = d2.Sequence(capture, rc.load_truth_table(CAPTURE))
    q = np.eye(2) * (f1.PROCESS_SIGMA_PER_SQRT_M**2 * 0.02)
    m0, S0 = seq.odom[0].copy(), np.eye(2) * f1.INITIAL_SIGMA_M**2
    bias_mm = {c: 1000.0 * v for c, v in f1.RESIDUAL_BIAS_M.items()}

    print("pooled:")
    pooled = em(seq, m0, S0, q, per_camera=False)
    print(f"  final shared noise {pooled['final']['camera_C']:.1f} mm | "
          f"NEES {pooled['outcome'][-1]['median_nees']:.2f} | "
          f"outside 95 % {100 * pooled['outcome'][-1]['unearned']:.1f} % | "
          f"{pooled['outcome'][-1]['rmse_mm']:.1f} mm off")

    print("per camera:")
    per_cam = em(seq, m0, S0, q, per_camera=True)
    for c in rc.CAMERAS:
        n = per_cam["counts"][c]
        tag = "NO DATA" if n == 0 else f"{n:4d} sightings"
        print(f"  {c}: {tag} | claims {per_cam['final'][c]:5.1f} mm | "
              f"really {bias_mm[c]:5.1f} mm off")
    print(f"  NEES {per_cam['outcome'][-1]['median_nees']:.2f} | "
          f"outside 95 % {100 * per_cam['outcome'][-1]['unearned']:.1f} % | "
          f"{per_cam['outcome'][-1]['rmse_mm']:.1f} mm off")

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.2))
    panel_collapse(axes[0], per_cam, bias_mm)
    panel_claim_vs_truth(axes[1], per_cam, bias_mm)
    panel_outcome(axes[2], pooled, per_cam)

    fig.suptitle("Letting each camera rate its own noise: it separates them, "
                 "and still does not help", fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.918,
             f"{CAPTURE}, 539 detections, RETIRED v2 projection — mechanism evidence, "
             "not current camera accuracy. Under the current path all four cameras sit "
             "within 64.6–68.1 mm, so this is not a camera ranking "
             "(docs/localization_metrics.md).",
             ha="center", va="top", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.878))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_r2_per_camera_r.{ext}", bbox_inches="tight")
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps({
        "capture": CAPTURE, "iterations": ITERATIONS,
        "prior_sigma_mm": PRIOR_SIGMA_M * 1000,
        "real_bias_mm": bias_mm,
        "sightings": per_cam["counts"],
        "pooled_final_mm": pooled["final"], "pooled_outcome": pooled["outcome"][-1],
        "per_camera_final_mm": per_cam["final"],
        "per_camera_outcome": per_cam["outcome"][-1],
        "before_learning": pooled["outcome"][0],
    }, indent=2), encoding="utf-8")
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
