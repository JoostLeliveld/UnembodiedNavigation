#!/usr/bin/env python3
"""Multi-camera fusion on the sealed final audit (matched position-heading camera batches).

Fused RMSE by the number of simultaneously admitted cameras for every rule: the single
camera selected by the spatial model, equal weights, the geometric baseline and the three
covariance models. Calibration and sharpness are in the fusion table, not here.

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
    fig, ax = P.plt.subplots(figsize=(P.COLUMN, 2.6))
    label = dict(P.MODEL_LABEL, best_spatial_single="$R_2$-selected single")
    for rule in RULES:
        marker, ls = STYLE[rule]
        y = [100 * report["by_camera_count"][rule][str(c)]["rmse_m"] for c in counts]
        hollow = rule in ("rproj", "equal", "best_spatial_single")
        ax.plot(counts, y, ls=ls, marker=marker, ms=4, color=P.MODEL_COLOUR[rule],
                mfc="white" if hollow else P.MODEL_COLOUR[rule], mew=0.9, lw=1.2, label=label[rule])
    ax.set_xticks(counts)
    ax.set_xlabel("admitted cameras", labelpad=1); ax.set_ylabel("fused RMSE (cm)", labelpad=1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=1.8,
              columnspacing=1.0, frameon=False)
    P.save(fig, "fusion_final_audit")
    print({r: round(100 * report["metrics"][r]["rmse_m"], 2) for r in RULES})


if __name__ == "__main__":
    main()
