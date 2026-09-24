#!/usr/bin/env python3
"""Covariance models on the sealed final audit: calibrated, and how sharp.

(a) Calibration: the fraction of corrected residuals inside the ellipse of nominal
    probability q, minus q, for q from 0.05 to 0.995. Zero is perfect; above zero the model
    is too wide, below it too narrow. Every position counts once (mean of per-position
    fractions), as in the audit report.
(b) Sharpness against likelihood: mean 95 % ellipse area and mean negative log likelihood.
    Lower left is better. The geometric baseline propagates a fitted pixel noise through the
    homography; it is an evaluation baseline and drives no planner.

    python3 figures/make_covariance.py
"""
from __future__ import annotations

import collections
import json

import numpy as np
from scipy.stats import chi2

import paper as P

MODELS = (("R0_global_full", "global"), ("R1_per_camera_full", "per_camera"),
          ("R2_spatial_full", "spatial"), ("Rproj", "rproj"))


def main():
    report = json.loads((P.THESIS / "final_audit/report.json").read_text())
    rows = [json.loads(line) for line in (P.THESIS / "final_audit/evaluated.jsonl").open()]
    admitted = [r for r in rows if r["outcome"] == "admitted"]
    q = np.concatenate([np.linspace(0.05, 0.95, 19), [0.975, 0.99, 0.995]])
    fig, (ax, bx) = P.plt.subplots(1, 2, figsize=(P.COLUMN, 1.8), gridspec_kw=dict(width_ratios=(1.35, 1), wspace=0.5))
    for key, name in MODELS:
        by_pos = collections.defaultdict(list)
        for r in admitted:
            by_pos[r["position_key"]].append(r["mahalanobis_d2"][key])
        d2 = [np.asarray(v) for v in by_pos.values()]
        cover = np.array([np.mean([np.mean(v <= chi2.ppf(p, 2)) for v in d2]) for p in q])
        c95 = np.mean([np.mean(v <= chi2.ppf(0.95, 2)) for v in d2])
        if abs(c95 - report["covariance"][key]["equal_position_coverage"]["95"]) > 1e-9:
            raise RuntimeError(f"{key}: recomputed C95 disagrees with the audit report")
        ls = (0, (3, 1.5)) if name == "rproj" else "-"
        ax.plot(100 * q, 100 * (cover - q), color=P.MODEL_COLOUR[name], ls=ls, lw=1.2, label=P.MODEL_LABEL[name])
        cov = report["covariance"][key]
        bx.plot(cov["equal_position_mean_ellipse_area_95_cm2"], cov["equal_position_mean_nll"], "o",
                ms=5.5 if name != "rproj" else 5, color=P.MODEL_COLOUR[name],
                mfc="white" if name == "rproj" else P.MODEL_COLOUR[name], mew=1.2)
        off = {"global": (-6, -6), "per_camera": (6, 6), "spatial": (6, 0), "rproj": (-6, 0)}[name]
        bx.annotate(P.MODEL_LABEL[name], (cov["equal_position_mean_ellipse_area_95_cm2"], cov["equal_position_mean_nll"]),
                    xytext=off, textcoords="offset points", fontsize=6.3, color=P.MODEL_COLOUR[name],
                    ha="right" if name in ("rproj", "global") else "left", va="center")
    ax.axhline(0, color=P.MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 95])
    ax.set_xlabel("nominal probability $q$ (%)", labelpad=1)
    ax.set_ylabel("coverage $-$ $q$ (pp)", labelpad=1)
    ax.legend(loc="upper right", fontsize=6, handlelength=1.5, borderaxespad=0.1)
    bx.set_xscale("log")
    bx.set_xlim(110, 1500); bx.set_xticks([200, 400, 800]); bx.set_xticklabels(["200", "400", "800"])
    bx.minorticks_off()
    bx.set_xlabel("95 % area (cm$^2$)", labelpad=1)
    bx.set_ylabel("NLL", labelpad=1)
    fig.text(0.0, 0.97, "(a)", fontweight="bold", va="top")
    fig.text(0.58, 0.97, "(b)", fontweight="bold", va="top")
    P.save(fig, "covariance")


if __name__ == "__main__":
    main()
