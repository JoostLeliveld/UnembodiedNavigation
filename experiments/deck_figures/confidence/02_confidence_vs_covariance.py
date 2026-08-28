"""Confidence versus post-geometry pixel covariance: main result and role split.

Reads the frozen output of measurement_commissioning/confidence_covariance.py.
No model is fit and no commissioning statistic is recomputed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as D  # noqa: E402


RESULT = D.REPO / "logs/studies/measurement_commissioning/confidence_covariance.json"
OUT = D.REPO / "logs/studies/deck_figures/confidence"
R = json.loads(RESULT.read_text())
BINS = R["binning"]["bins"]
MODELS = R["models"]


def array(key):
    return np.array([b[key] for b in BINS], dtype=float)


def main_figure() -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16.0, 9.0),
                                 gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.subplots_adjust(left=0.065, right=0.965, top=0.70, bottom=0.18, wspace=0.29)

    # A — empirical residual spread against confidence.
    q = array("q_mean")
    pooled = array("sigma_pooled_px")
    ci = np.array([b["sigma_pooled_ci95_px"] for b in BINS])
    su, sv = array("sigma_u_px"), array("sigma_v_px")
    old = array("old_precision_blend_sigma_px")
    ax.fill_between(q, ci[:, 0], ci[:, 1], color=D.ROBOT, alpha=0.16, lw=0)
    ax.plot(q, pooled, "o-", color=D.ROBOT, lw=3.4, ms=9.5,
            label="observed pooled residual SD")
    ax.plot(q, su, "o--", color="#6aa1df", lw=1.6, ms=4.5, alpha=0.85,
            label="$u$ residual SD")
    ax.plot(q, sv, "s--", color="#164f91", lw=1.6, ms=4.5, alpha=0.85,
            label="$v$ residual SD")
    ax.plot(q, old, color=D.OLD, lw=3.0, ls=(0, (5, 3)),
            label="previous offline confidence→R blend")
    ax.annotate("minimum near 0.92\nthen the noise rises again",
                xy=(q[2], pooled[2]), xytext=(0.875, 1.47),
                color=D.ROBOT, fontsize=12.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=D.ROBOT, lw=2.0, shrinkB=6))
    ax.annotate("predicts 2.5–2.7 px\nwhere 0.6–1.1 px is observed",
                xy=(q[6], old[6]), xytext=(0.875, 2.15),
                color=D.OLD, fontsize=11.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=D.OLD, lw=2.0, shrinkB=6))
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1.0, 2.0, 3.0])
    ax.set_yticklabels(["0.5", "0.7", "1", "2", "3"])
    ax.set_ylim(0.48, 3.25)
    ax.set_xlim(0.84, 0.977)
    ax.set_xlabel("YOLO confidence · equal-count bins", fontsize=12.5)
    ax.set_ylabel("post-geometry pixel residual SD [px] · log scale", fontsize=12.5)
    ax.grid(True, color="#e8e7e2", lw=0.7)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("A.  Confidence predicts a U-shape, not inverse covariance",
                 loc="left", fontsize=15.5, color=D.INK, pad=12)

    # B — held-out covariance prediction.  Keep the useful range legible; the old
    # mapping is shown in an explicit off-scale card because its delta is +1.53 NLL.
    order = [
        ("constant", "constant $R$", D.MUTED),
        ("camera_only", "camera only", D.CAM_COLOUR["D"]),
        ("confidence_binned", "confidence · 8 bins", D.ROBOT),
        ("camera_plus_confidence_binned", "camera + confidence", D.GOOD),
    ]
    ys = np.arange(len(order))[::-1]
    for y, (key, label, col) in zip(ys, order):
        m = MODELS[key]
        delta = m["delta_nll_vs_constant"]
        lo, hi = m["delta_nll_vs_constant_ci95"]
        bx.errorbar(delta, y, xerr=[[delta - lo], [hi - delta]], fmt="o", ms=11,
                    color=col, ecolor=col, elinewidth=2.4, capsize=5, zorder=4)
        bx.text(0.046, y, f"{100*m['coverage_95']:.1f}%", va="center", ha="right",
                fontsize=11.5, color=D.INK2)
    bx.axvline(0, color=D.MUTED, lw=1.6, ls=(0, (4, 3)))
    bx.text(0.0, 3.47, "constant", ha="center", va="bottom", fontsize=10.5, color=D.MUTED)
    bx.set_yticks(ys)
    bx.set_yticklabels([x[1] for x in order], fontsize=11.5)
    bx.set_xlim(-0.095, 0.05)
    bx.set_ylim(-0.7, 3.7)
    bx.set_xlabel("held-out Δ Gaussian NLL vs constant · lower is better", fontsize=12.5)
    bx.grid(True, axis="x", color="#e8e7e2", lw=0.7)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.text(0.046, 3.55, "95% coverage", ha="right", va="bottom", fontsize=10.5,
            color=D.MUTED)
    oldm = MODELS["previous_offline_mapping"]
    bx.text(0.49, 0.055,
            f"previous mapping is off scale\nΔNLL  +{oldm['delta_nll_vs_constant']:.2f}  ·  "
            f"coverage {100*oldm['coverage_95']:.1f}%",
            transform=bx.transAxes, ha="left", va="bottom", fontsize=11.5,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=D.OLD, edgecolor="none", alpha=0.96))
    cmp = R["camera_incremental_comparison"]
    d = cmp["camera_plus_confidence_binned_minus_camera_only_nll"]
    lo, hi = cmp["ci95"]
    bx.text(0.00, -0.18,
            f"confidence beyond camera: ΔNLL {d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            transform=bx.transAxes, fontsize=10.8, color=D.GOOD, fontweight="bold")
    bx.set_title("B.  It adds a little held-out predictive value",
                 loc="left", fontsize=15.5, color=D.INK, pad=12)

    fig.text(0.025, 0.955, "CONFIDENCE GETS A FAIR TRIAL", fontsize=12.5,
             fontweight="bold", color=D.ROBOT, ha="left", va="top")
    fig.text(0.025, 0.900,
             "Useful signal, wrong rule: confidence cannot directly determine $R_{cond}$",
             fontsize=23.5, fontweight="bold", color=D.INK, ha="left", va="top")
    fig.text(0.025, 0.805,
             "3,151 held-out sightings · geometry and the frozen offset removed · six spatial folds · "
             "intervals bootstrap whole floor positions",
             fontsize=12.2, color=D.INK2, ha="left", va="top")
    fig.text(0.025, 0.055,
             "Diagnostic pending registry · one detector and one simulated stock state · "
             "previous curve is the historical offline 2.5↔40 px precision blend",
             fontsize=10.5, color=D.MUTED, ha="left", va="bottom")
    fig.savefig(OUT / "02_confidence_vs_covariance.png", dpi=175)
    plt.close(fig)


def role_split_figure() -> None:
    admission = R["admission_given_detection"]
    qa = np.array([b["q_mean"] for b in admission])
    pa = np.array([b["admission_rate"] for b in admission])
    ca = np.array([b["admission_rate_ci95"] for b in admission])
    q = array("q_mean")
    pooled = array("sigma_pooled_px")
    ci = np.array([b["sigma_pooled_ci95_px"] for b in BINS])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16.0, 7.5))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.76, bottom=0.16, wspace=0.25)
    ax.fill_between(qa, ca[:, 0], ca[:, 1], color=D.GOOD, alpha=0.16, lw=0)
    ax.plot(qa, pa, "o-", color=D.GOOD, lw=3.4, ms=9.5)
    ax.set_xlim(0.60, 0.98); ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("YOLO confidence · equal-count detected-box bins", fontsize=12.5)
    ax.set_ylabel("$P$(admitted | detector fired)", fontsize=12.5)
    ax.grid(True, color="#e8e7e2", lw=0.7)
    ax.annotate("strong gate signal", xy=(qa[6], pa[6]), xytext=(0.72, 0.83),
                fontsize=13, color=D.GOOD, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=D.GOOD, lw=2.2))
    ax.set_title("A.  Confidence strongly predicts whether a box survives",
                 loc="left", fontsize=15.5, pad=12)

    bx.fill_between(q, ci[:, 0], ci[:, 1], color=D.ROBOT, alpha=0.16, lw=0)
    bx.plot(q, pooled, "o-", color=D.ROBOT, lw=3.4, ms=9.5)
    bx.set_xlim(0.84, 0.977); bx.set_ylim(0.48, 1.25)
    bx.set_xlabel("YOLO confidence · admitted-sighting bins", fontsize=12.5)
    bx.set_ylabel("post-geometry residual SD [px]", fontsize=12.5)
    bx.grid(True, color="#e8e7e2", lw=0.7)
    bx.annotate("weak, non-monotonic\nnoise relationship", xy=(q[8], pooled[8]),
                xytext=(0.865, 1.08), fontsize=13, color=D.ROBOT, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=D.ROBOT, lw=2.2))
    bx.set_title("B.  Once admitted, noise follows a shallow U-shape",
                 loc="left", fontsize=15.5, pad=12)
    for axis in (ax, bx):
        for s in ("top", "right"):
            axis.spines[s].set_visible(False)

    fig.text(0.025, 0.95, "SEPARATE THE TWO ROLES", fontsize=12.5,
             fontweight="bold", color=D.GOOD, ha="left", va="top")
    fig.text(0.025, 0.895,
             "Confidence is more useful for admission than as a covariance dial",
             fontsize=24, fontweight="bold", color=D.INK, ha="left", va="top")
    fig.text(0.025, 0.825,
             "It only exists after detection, so this is not a route-planning availability field. "
             "It can still remain an online fusion feature.",
             fontsize=12.4, color=D.INK2, ha="left", va="top")
    fig.text(0.025, 0.055,
             "Left: all detected boxes. Right: admitted sightings held out from offset fitting. "
             "Shading = 95% floor-position cluster bootstrap interval.",
             fontsize=10.5, color=D.MUTED, ha="left", va="bottom")
    fig.savefig(OUT / "03_confidence_role_split.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main_figure()
    role_split_figure()
    print("wrote 02_confidence_vs_covariance.png and 03_confidence_role_split.png")
