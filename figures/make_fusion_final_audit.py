#!/usr/bin/env python3
"""Multi-camera fusion on the sealed final audit (matched position-heading camera batches).

(a) Fused RMSE by the number of simultaneously admitted cameras for every rule: the best
    single camera by the spatial model, equal weights, the three covariance models and the
    geometric baseline.
(b) Calibration and sharpness of the fused estimate: fraction of batches inside the
    predicted 95 % ellipse against its mean area (rules that predict a covariance only).

    python3 figures/make_fusion_final_audit.py
"""
from __future__ import annotations

import json

import paper as P

RULES = ("best_spatial_single", "equal", "rproj", "global", "per_camera", "spatial")
STYLE = {"best_spatial_single": ("s", (0, (1, 1.2))), "equal": ("D", (0, (1, 1.2))),
         "rproj": ("o", (0, (3, 1.5))), "global": ("o", "-"), "per_camera": ("o", "-"), "spatial": ("o", "-")}


def main():
    report = json.loads((P.THESIS / "final_audit_fusion/report.json").read_text())
    if report.get("status") != "complete" or report.get("selection_or_fitting_performed"):
        raise RuntimeError("final-audit fusion report is not a complete sealed evaluation")
    counts = sorted(int(c) for c in report["by_camera_count"]["spatial"])
    fig, (ax, bx) = P.plt.subplots(1, 2, figsize=(P.COLUMN, 2.15), gridspec_kw=dict(width_ratios=(1.2, 1), wspace=0.5))
    for rule in RULES:
        marker, ls = STYLE[rule]
        y = [100 * report["by_camera_count"][rule][str(c)]["rmse_m"] for c in counts]
        hollow = rule in ("rproj", "equal", "best_spatial_single")
        ax.plot(counts, y, ls=ls, marker=marker, ms=3.5, color=P.MODEL_COLOUR[rule],
                mfc="white" if hollow else P.MODEL_COLOUR[rule], mew=0.9, lw=1.1, label=P.MODEL_LABEL[rule])
    ax.set_xticks(counts)
    ax.set_xlabel("admitted cameras", labelpad=1); ax.set_ylabel("fused RMSE (cm)", labelpad=1)
    ax.set_ylim(0, None)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.33)
    for rule in ("rproj", "global", "per_camera", "spatial"):
        m = report["metrics"][rule]
        bx.plot(m["mean_fused_ellipse_area_95_cm2"], 100 * m["coverage_95"], "o", ms=5, color=P.MODEL_COLOUR[rule],
                mfc="white" if rule == "rproj" else P.MODEL_COLOUR[rule], mew=1.2)
        off = {"rproj": (-6, 0), "global": (6, -5), "per_camera": (6, 5), "spatial": (6, 0)}[rule]
        bx.annotate(P.MODEL_LABEL[rule], (m["mean_fused_ellipse_area_95_cm2"], 100 * m["coverage_95"]), xytext=off,
                    textcoords="offset points", fontsize=6.3, color=P.MODEL_COLOUR[rule],
                    ha="right" if rule == "rproj" else "left", va="center")
    bx.axhline(95, color=P.MUTED, lw=0.6, ls=(0, (2, 2)))
    bx.set_xscale("log"); bx.minorticks_off()
    areas = [report["metrics"][r]["mean_fused_ellipse_area_95_cm2"] for r in ("rproj", "global", "per_camera", "spatial")]
    bx.set_xlim(min(areas) / 1.6, max(areas) * 1.6)
    ticks = [t for t in (20, 50, 100, 200) if min(areas) / 1.6 <= t <= max(areas) * 1.6]
    bx.set_xticks(ticks); bx.set_xticklabels([str(t) for t in ticks])
    bx.set_xlabel("fused 95 % area (cm$^2$)", labelpad=1); bx.set_ylabel("inside 95 % ellipse (%)", labelpad=1)
    fig.text(0.0, 0.97, "(a)", fontweight="bold", va="top")
    fig.text(0.56, 0.97, "(b)", fontweight="bold", va="top")
    P.save(fig, "fusion_final_audit")
    print({r: round(100 * report["metrics"][r]["rmse_m"], 2) for r in RULES})


if __name__ == "__main__":
    main()
