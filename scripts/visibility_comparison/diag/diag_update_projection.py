#!/usr/bin/env python3
"""Projection-compare + EKF-update sanity.

Projection: compare the THREE world positions a detected pixel implies vs
capture-time truth: raw, planner-offset path (reconstructed; the running belief
actually uses this), and the affine path (/state/bev). Quantifies what the
planner loses by ignoring the affine.

Update: innovation / NIS / accept-reject / applied-age over time, and flags
rejected-but-low-error updates (good measurements thrown away).

Usage: python diag_update_projection.py <run_dir> [--out <dir>]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diag_common as dc


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--offset", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    perc, exp, summary = dc.load_run(run_dir)
    perc = dc.add_planner_projection(perc, bev_y_offset=args.offset)
    out_dir = Path(args.out) if args.out else (run_dir / "diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    det = perc[perc["detected"] == 1].copy() if "detected" in perc else perc.copy()

    fig, axs = plt.subplots(2, 3, figsize=(18, 9))

    # ---- projection error vs robot y (position dependence) ----
    ax = axs[0, 0]
    yv = det["true_y"].to_numpy(float)
    for col, lab, c in [("localization_error_m", "raw", "#999"),
                        ("planner_loc_error_m", "planner (offset 0.05y, ACTUAL belief)", "#e8820c"),
                        ("localization_error_calibrated_m", "affine (/state/bev)", "#2a72c4")]:
        if col in det:
            ax.scatter(yv, det[col], s=10, alpha=0.5, c=c, label=f"{lab}: med={med(det[col]):.3f}")
    ax.set_xlabel("robot true y (m)"); ax.set_ylabel("loc error (m)")
    ax.set_title("Projection error vs y (live-truth)"); ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # ---- captime projection error (latency removed) ----
    ax = axs[0, 1]
    for col, lab, c in [("localization_error_captime_m", "captime (latency removed)", "#2a8"),
                        ("localization_error_calibrated_m", "affine @ logtime", "#2a72c4")]:
        if col in det:
            d = det[col].dropna()
            ax.hist(d, bins=30, alpha=0.5, color=c, label=f"{lab}: med={med(d):.3f}")
    ax.set_xlabel("loc error (m)"); ax.set_title("Projection error dist"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # ---- planner-offset vs affine gap (what planner loses) ----
    ax = axs[0, 2]
    if {"planner_loc_error_m", "localization_error_calibrated_m"}.issubset(det.columns):
        gap = det["planner_loc_error_m"] - det["localization_error_calibrated_m"]
        ax.hist(gap.dropna(), bins=30, color="#b05", alpha=0.7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"planner_err - affine_err  (med={med(gap):+.3f}m)\n>0 = planner offset path is WORSE")
        ax.set_xlabel("error gap (m)"); ax.grid(alpha=0.25)

    # ---- update: innovation over time ----
    corr = exp[exp.get("planner_pixel_correction_available", 0) == 1].copy() if exp is not None else None
    if corr is not None and len(corr):
        t = corr["stamp"].to_numpy(float)
        ax = axs[1, 0]
        for col, c in [("pixel_corr_innov_u", "#c33"), ("pixel_corr_innov_v", "#37c")]:
            if col in corr:
                ax.plot(t, corr[col], lw=0.8, color=c, label=col.replace("pixel_corr_", ""))
        ax.set_title("innovation (px)"); ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_xlabel("t (s)")

        ax = axs[1, 1]
        if "pixel_corr_nis" in corr:
            ax.plot(t, corr["pixel_corr_nis"], lw=0.8, color="#444")
            thr = float(corr["pixel_corr_nis_threshold"].dropna().iloc[0]) if "pixel_corr_nis_threshold" in corr and corr["pixel_corr_nis_threshold"].notna().any() else 9.21
            ax.axhline(thr, color="red", ls="--", label=f"thresh {thr:.2f}")
            rej = corr[corr["pixel_corr_accepted"] != 1]
            ax.scatter(rej["stamp"], rej["pixel_corr_nis"], s=18, c="red", zorder=5, label="rejected")
        ax.set_title("NIS + rejections"); ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_xlabel("t (s)")

        # ---- rejected-but-valid: applied age + accept rate text ----
        ax = axs[1, 2]
        if "pixel_corr_age_s" in corr:
            ax.plot(t, corr["pixel_corr_age_s"], lw=0.8, color="#777", label="applied age (s)")
        n = len(corr); nacc = int((corr["pixel_corr_accepted"] == 1).sum())
        reasons = {}
        if "pixel_corr_reject_reason" in corr:
            reasons = dict(corr[corr["pixel_corr_accepted"] != 1]["pixel_corr_reject_reason"].value_counts())
        ax.set_title(f"applied-age; accept {nacc}/{n} ({100*nacc/max(n,1):.0f}%)\nrejects={reasons}", fontsize=9)
        ax.set_xlabel("t (s)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle(f"{run_dir.name}: projection + update sanity", fontsize=13)
    fig.tight_layout()
    png = out_dir / "update_projection.png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    print(f"wrote {png}")
    # console summary
    print(f"  projection captime med = {med(det.get('localization_error_captime_m', [])):.3f}m")
    print(f"  planner-offset path med = {med(det.get('planner_loc_error_m', [])):.3f}m")
    print(f"  affine path med         = {med(det.get('localization_error_calibrated_m', [])):.3f}m")


if __name__ == "__main__":
    main()
