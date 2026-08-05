#!/usr/bin/env python3
"""Figures for the deployed-correction residual audit (fig_b1 .. fig_b6).

Reads the artifacts written by residual_audit.py and renders every figure as
both .png (dpi=150) and .pdf with tight bounding boxes.

Style contract (fixed across every figure):
  camera_A #0072B2, camera_B #E69F00, camera_C #009E73, camera_D #D55E00
  signed/directional quantities -> RdBu_r, symmetric about zero
  magnitude-only quantities     -> Blues
  no dual axes; legend whenever >=2 series; grid recessive, below data
  titles state the TAKEAWAY; headline numbers annotated on the figure
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
OUT = REPO / "logs/studies/external_camera_bias_model/exp1_residual_characterization"

CAMS = ("camera_A", "camera_B", "camera_C", "camera_D")
COLOR = {
    "camera_A": "#0072B2",
    "camera_B": "#E69F00",
    "camera_C": "#009E73",
    "camera_D": "#D55E00",
}
SHORT = {c: c[-1] for c in CAMS}
CAM_XY = {
    "camera_A": (-6.0, -10.0),
    "camera_B": (-6.0, 10.0),
    "camera_C": (6.0, -10.0),
    "camera_D": (6.0, 10.0),
}
SITE = (-11.20, 11.50, -8.60, 8.60)
RAW_GREY = "#9a9a9a"

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": "#cccccc",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png/.pdf")


def load():
    rows = []
    with (OUT / "residuals.csv").open(newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h):
            for k, v in r.items():
                if k not in ("capture", "camera"):
                    r[k] = float(v)
            rows.append(r)
    audit = json.loads((OUT / "audit.json").read_text())
    return rows, audit


def ellipse(ax, cx, cy, ex, ey, color, ls="-", lw=1.6, label=None):
    """1-sigma covariance ellipse of the (ex, ey) cloud, drawn at (cx, cy)."""
    if len(ex) < 3:
        return
    cov = np.cov(np.column_stack([ex, ey]), rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.maximum(vals[order], 0), vecs[:, order]
    ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse((cx, cy), 2 * math.sqrt(vals[0]), 2 * math.sqrt(vals[1]),
                         angle=ang, fill=False, edgecolor=color, lw=lw, ls=ls,
                         zorder=6, label=label))


# ---------------------------------------------------------------- fig_b1 (money)


def fig_b1(rows, audit):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    lim = 0.30
    for j, cam in enumerate(CAMS):
        grp = [r for r in rows if r["camera"] == cam]
        st = audit["stats"].get(f"{cam}|POOLED")
        for i, (fx, fy, fname) in enumerate(
            [("ex", "ey", "world frame"), ("along", "cross", "bearing frame")]
        ):
            ax = axes[i, j]
            ax.grid(True, zorder=0)
            ax.axhline(0, color="#555", lw=0.8, zorder=1)
            ax.axvline(0, color="#555", lw=0.8, zorder=1)
            rx = np.array([r[f"raw_{fx}"] for r in grp])
            ry = np.array([r[f"raw_{fy}"] for r in grp])
            cx = np.array([r[f"cor_{fx}"] for r in grp])
            cy = np.array([r[f"cor_{fy}"] for r in grp])
            ax.scatter(rx, ry, s=7, c=RAW_GREY, alpha=0.45, lw=0, zorder=3,
                       label="raw (pre-correction)")
            ax.scatter(cx, cy, s=7, c=COLOR[cam], alpha=0.65, lw=0, zorder=4,
                       label="after deployed correction")
            for xs, ys, col, ls in ((rx, ry, RAW_GREY, "--"), (cx, cy, COLOR[cam], "-")):
                bx, by = xs.mean(), ys.mean()
                ax.annotate("", xy=(bx, by), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                            shrinkA=0, shrinkB=0), zorder=5)
                ellipse(ax, bx, by, xs, ys, col, ls=ls)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(f"{cam}  (n={len(grp)})", color=COLOR[cam], fontweight="bold")
                ax.set_xlabel("x error [m]")
                ax.set_ylabel("y error [m]" if j == 0 else "")
                r_raw, r_cor = st["raw"]["ratio_world"], st["cor"]["ratio_world"]
                ax.text(0.03, 0.97,
                        f"|b| {st['raw']['bias_norm']:.3f}$\\to${st['cor']['bias_norm']:.3f} m\n"
                        f"bias/noise {r_raw:.1f}$\\to${r_cor:.1f}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        bbox=dict(fc="white", ec="#dddddd", alpha=0.9, pad=2.5))
            else:
                ax.set_xlabel("along-bearing error [m]")
                ax.set_ylabel("cross-bearing error [m]" if j == 0 else "")
                ax.text(0.03, 0.97,
                        f"along {st['raw']['along_bias']:+.3f}$\\to${st['cor']['along_bias']:+.3f}\n"
                        f"cross {st['raw']['cross_bias']:+.3f}$\\to${st['cor']['cross_bias']:+.3f}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        bbox=dict(fc="white", ec="#dddddd", alpha=0.9, pad=2.5))
            if (i, j) == (0, 0):
                ax.legend(loc="lower right", fontsize=7.5, markerscale=1.6)
    axes[0, 0].text(-0.30, 0.5, "WORLD FRAME", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontweight="bold", fontsize=10)
    axes[1, 0].text(-0.30, 0.5, "BEARING FRAME", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", fontweight="bold", fontsize=10)
    fig.suptitle(
        "The deployed correction removes the along-bearing bias entirely — but leaves a "
        "CROSS-bearing bias it cannot touch (camera C: +0.078 m)",
        fontweight="bold", fontsize=12.5, y=1.005)
    fig.tight_layout()
    save(fig, "fig_b1_residual_scatter_pre_post")


# --------------------------------------------------------------------- fig_b2


def fig_b2(audit):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    x = np.arange(len(CAMS))
    w = 0.36
    for ax_, key, lab in ((ax, "ratio_world", "bias-to-noise ratio  ||b|| / $\\sigma$"),
                          (ax2, "rmse", "residual RMSE [m]")):
        ax_.grid(True, axis="y", zorder=0)
        for k, (tag, hatch, alpha) in enumerate((("raw", "//", 0.45), ("cor", None, 1.0))):
            vals, los, his = [], [], []
            for cam in CAMS:
                st = audit["stats"][f"{cam}|POOLED"][tag]
                vals.append(st[key])
                if key == "ratio_world":
                    lo, hi = st["ratio_world_ci"]
                else:
                    lo = hi = st[key]
                los.append(max(st[key] - lo, 0))
                his.append(max(hi - st[key], 0))
            ax_.bar(x + (k - 0.5) * w, vals, w * 0.92,
                    color=[COLOR[c] for c in CAMS], alpha=alpha, hatch=hatch,
                    edgecolor="white", linewidth=0.8, zorder=3,
                    yerr=[los, his] if key == "ratio_world" else None,
                    error_kw=dict(ecolor="#333333", lw=1.1, capsize=3), )
            pad = 0.02 * max(vals)
            for xi, v, hi in zip(x + (k - 0.5) * w, vals, his):
                ax_.text(xi, v + hi + pad,
                         f"{v:.2f}" if key == "ratio_world" else f"{v:.3f}",
                         ha="center", va="bottom", fontsize=8)
        ax_.set_xticks(x)
        ax_.set_xticklabels([f"camera {SHORT[c]}" for c in CAMS])
        ax_.set_ylabel(lab)
        ax_.margins(y=0.16)
    ax.axhline(1.0, color="#444", ls=":", lw=1.4, zorder=4)
    ax.annotate("bias = noise", xy=(1.005, 1.0),
                xycoords=matplotlib.transforms.blended_transform_factory(
                    ax.transAxes, ax.transData),
                va="center", ha="left", fontsize=8, color="#444", annotation_clip=False)
    ax.set_title("Pre-correction every camera is bias-dominated (3.6-5.4x);\n"
                 "after correction only camera C still is (2.1x)", fontweight="bold")
    ax2.set_title("What remains is HETEROGENEOUS: camera C's residual is\n"
                  "4.3x camera B's — the planner models none of this",
                  fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#bbbbbb", hatch="//", alpha=0.6, ec="white"),
               plt.Rectangle((0, 0), 1, 1, fc="#bbbbbb", ec="white")]
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(handles, ["raw (pre-correction)", "after deployed correction"],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.0, "Error bars: 95% block-bootstrap CI on the ratio (left panel). "
             "Bar colour = camera identity.", ha="center", fontsize=8, color="#555")
    save(fig, "fig_b2_bias_to_noise_ratio")


# --------------------------------------------------------------------- fig_b3


def _segments(centers, values, counts):
    """Split into runs of ADJACENT populated bins so lines never interpolate
    across a range band where the camera has no observations."""
    runs, cur = [], []
    for c, v, n in zip(centers, values, counts):
        if n > 0 and np.isfinite(v):
            cur.append((c, v))
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def fig_b3(rows, audit):
    fig = plt.figure(figsize=(15.0, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.42,
                          right=0.90)
    scale = 12.0
    # ONE symmetric scale shared by all four panels so they are comparable
    vmax = max(abs(r["cor_cross"]) for r in rows)
    vmax = math.ceil(vmax * 100) / 100
    q = None
    for j, cam in enumerate(CAMS):
        ax = fig.add_subplot(gs[0, j])
        grp = [r for r in rows if r["camera"] == cam]
        ax.grid(True, zorder=0)
        ax.add_patch(plt.Rectangle((SITE[0], SITE[2]), SITE[1] - SITE[0], SITE[3] - SITE[2],
                                   fill=False, ec="#bbbbbb", lw=1.0, zorder=1))
        cx, cy = CAM_XY[cam]
        tx = np.array([r["true_x"] for r in grp])
        ty = np.array([r["true_y"] for r in grp])
        ex = np.array([r["cor_ex"] for r in grp])
        ey = np.array([r["cor_ey"] for r in grp])
        cross = np.array([r["cor_cross"] for r in grp])
        for i in range(0, len(tx), max(1, len(tx) // 12)):
            ax.plot([cx, tx[i]], [cy, ty[i]], color="#e2e2e2", lw=0.6, zorder=1)
        q = ax.quiver(tx, ty, ex * scale, ey * scale, cross, cmap="RdBu_r",
                      clim=(-vmax, vmax), angles="xy", scale_units="xy", scale=1,
                      width=0.008, zorder=4)
        ax.plot([cx], [cy], marker="s", ms=9, color=COLOR[cam], zorder=6)
        ax.annotate(f"camera {SHORT[cam]}", (cx, cy), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8.5, color=COLOR[cam],
                    fontweight="bold")
        ax.set_xlim(SITE[0] - 1.5, SITE[1] + 1.5)
        ax.set_ylim(SITE[2] - 2.6, SITE[3] + 2.6)
        ax.set_aspect("equal")
        ax.set_title(f"{cam}  (n={len(grp)}, arrows x{scale:.0f})",
                     color=COLOR[cam], fontweight="bold", fontsize=9.5)
        ax.set_xlabel("x [m]", fontsize=8.5)
        ax.set_ylabel("y [m]" if j == 0 else "", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    cax = fig.add_axes([0.915, 0.565, 0.011, 0.30])
    cb = fig.colorbar(q, cax=cax)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("cross-bearing residual [m]  (shared scale, symmetric about 0)", fontsize=8)

    ax2 = fig.add_subplot(gs[1, :2])
    ax3 = fig.add_subplot(gs[1, 2:])
    for ax_, lab in ((ax2, "along-bearing residual [m]"),
                     (ax3, "cross-bearing residual [m]")):
        ax_.grid(True, zorder=0)
        ax_.axhline(0, color="#555", lw=0.9, zorder=1)
        ax_.set_xlabel("ground range from camera [m]")
        ax_.set_ylabel(lab)
    for cam in CAMS:
        lo = audit["leftover_structure"][cam]
        cen = np.array(lo["range_bin_centers"])
        cnt = np.array(lo["range_bin_counts"])
        grp = [r for r in rows if r["camera"] == cam]
        rr = np.array([r["range_m"] for r in grp])
        cc = np.array([r["cor_cross"] for r in grp])
        edges = np.linspace(rr.min(), rr.max(), 6)
        cross_bin = []
        for a0, a1 in zip(edges[:-1], edges[1:]):
            mm = (rr >= a0) & (rr < a1)
            cross_bin.append(cc[mm].mean() if mm.sum() else np.nan)
        for ax_, vals in ((ax2, np.array(lo["range_bin_along"], float)),
                          (ax3, np.array(cross_bin, float))):
            first = True
            for run in _segments(cen, vals, cnt):
                xs = [p[0] for p in run]
                ys = [p[1] for p in run]
                ax_.plot(xs, ys, "-o", color=COLOR[cam], ms=5, lw=1.8, zorder=4,
                         label=f"camera {SHORT[cam]}" if first else None)
                first = False
    ax2.legend(fontsize=8, ncol=4)
    ax3.legend(fontsize=8, ncol=4)
    ax2.text(0.5, -0.30, "Line breaks = range bands with no observations "
             "(cameras A and B were only seen at two separated ranges).",
             transform=ax2.transAxes, ha="center", fontsize=7.5, color="#666")
    ax2.set_title("Along-bearing residual is within $\\pm$0.04 m at every range\n"
                  "(pre-correction it was $-$0.11 to $-$0.16 m): the range model works",
                  fontweight="bold", fontsize=9.5)
    ax3.set_title("The surviving error is cross-bearing and camera-specific —\n"
                  "camera C sits at +0.08 m across its whole range",
                  fontweight="bold", fontsize=9.5)
    fig.suptitle("Post-correction residuals have SPATIAL structure the deployed 1-D "
                 "along-bearing model cannot represent",
                 fontweight="bold", fontsize=12.5, y=0.985)
    save(fig, "fig_b3_spatial_structure")


# --------------------------------------------------------------------- fig_b4


def fig_b4(audit):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    lim = audit["jump_limit_m"]
    mats = {}
    for tag in ("raw", "cor"):
        m = np.full((4, 4), np.nan)
        for i, a in enumerate(CAMS):
            for j, b in enumerate(CAMS):
                v = audit["handover"][tag].get(f"{a}|{b}")
                if v is not None:
                    m[i, j] = v
        mats[tag] = m
    # Colour scale spans 0 -> the jump limiter itself, so "how close to the limit"
    # is read directly off the fill.
    vmax = lim
    for ax, tag, title in (
        (axes[0], "raw", "Pre-correction: worst handover step 0.313 m (63% of limit)"),
        (axes[1], "cor", "After deployed correction: worst step 0.089 m (18% of limit)"),
    ):
        m = mats[tag]
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(4):
            for j in range(4):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=9,
                            color="white" if m[i, j] > 0.6 * vmax else "#222222")
        ax.set_xticks(range(4), [f"cam {SHORT[c]}" for c in CAMS])
        ax.set_yticks(range(4), [f"cam {SHORT[c]}" for c in CAMS])
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("switching TO")
        if tag == "raw":
            ax.set_ylabel("switching FROM")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("bias-difference magnitude [m]", fontsize=8)
        cb.ax.axhline(lim, color="#D55E00", lw=2.0)
        cb.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        f"Handover steps stay FAR below the {lim:.1f} m jump limiter — both before and "
        f"after correction (worst case 0.31 m = 63% of the limit)",
        fontweight="bold", fontsize=11.5, y=1.02)
    fig.text(0.5, -0.03, f"Orange line on each colourbar = pixel_max_correction_jump_m = {lim:.1f} m "
             "(scripts/visibility_comparison/*.yaml). No pairwise step reaches it.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "fig_b4_handover_step_matrix")


# --------------------------------------------------------------------- fig_b5


def fig_b5(audit):
    caps = [c for c, v in audit["fusion"].items() if v.get("n_multi_camera_clusters", 0) > 0]
    fig, axes = plt.subplots(1, len(caps) + 1, figsize=(4.2 * (len(caps) + 1), 4.6))
    series = [("best_single_rmse", "best single camera", "#0072B2"),
              ("fused_rmse", "unweighted fusion", "#D55E00"),
              ("mean_single_rmse", "mean single camera", "#9a9a9a")]
    for k, cap in enumerate(caps):
        ax = axes[k]
        v = audit["fusion"][cap]
        errs = v["errors"]
        fused = np.array([e["fused_err"] for e in errs])
        best = np.array([min(e["per_cam"].values()) for e in errs])
        ax.grid(True, zorder=0)
        bins = np.linspace(0, max(fused.max(), best.max()) * 1.02, 26)
        ax.hist(best, bins=bins, color="#0072B2", alpha=0.55, label="best single camera", zorder=3)
        ax.hist(fused, bins=bins, histtype="step", lw=2.2, color="#D55E00",
                label="unweighted fusion", zorder=4)
        ax.axvline(best.mean(), color="#0072B2", ls="--", lw=1.5, zorder=5)
        ax.axvline(fused.mean(), color="#D55E00", ls="--", lw=1.5, zorder=5)
        ax.set_title(f"{cap}\n(n={v['n_multi_camera_clusters']} concurrent clusters)", fontsize=9)
        ax.set_xlabel("position error [m]")
        if k == 0:
            ax.set_ylabel("count")
            # Anchored below the stat box rather than auto-placed: "best" lands
            # top-right and collides with it.
            ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.74))
        ax.text(0.97, 0.97,
                f"fused RMSE {v['fused_rmse']:.3f} m\nbest single {v['best_single_rmse']:.3f} m\n"
                f"fusion wins {v['fused_beats_best_frac']*100:.0f}% of the time",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", ec="#dddddd", alpha=0.92, pad=3))
    ax = axes[-1]
    ax.grid(True, axis="y", zorder=0)
    x = np.arange(len(caps))
    w = 0.26
    for k, (key, lab, col) in enumerate(series):
        vals = [audit["fusion"][c][key] for c in caps]
        ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, label=lab,
               edgecolor="white", lw=0.8, zorder=3)
        for xi, vv in zip(x + (k - 1) * w, vals):
            ax.text(xi, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [c.replace("_", "\n") for c in caps], fontsize=8)
    ax.set_ylabel("RMSE [m]")
    ax.legend(fontsize=8)
    ax.set_title("Fusion never beats the best single camera", fontweight="bold", fontsize=9.5)
    fig.suptitle("Bias does NOT cancel under fusion: averaging cameras beats the AVERAGE "
                 "camera, but loses to the BEST one in all three captures",
                 fontweight="bold", fontsize=11.5, y=1.03)
    fig.tight_layout()
    save(fig, "fig_b5_fusion_simulation")


# --------------------------------------------------------------------- fig_b6


def fig_b6(rows, audit):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ex = audit["extrapolation"]
    x = np.arange(len(CAMS))
    ax.grid(True, axis="y", zorder=0)
    vals = [ex[c]["frac_outside_fitted_range"] * 100 for c in CAMS]
    ax.bar(x, vals, 0.6, color=[COLOR[c] for c in CAMS], edgecolor="white", lw=0.8, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x, [f"camera {SHORT[c]}" for c in CAMS])
    ax.set_ylabel("% of camera's on-site ground FOV\noutside its fitted range window")
    ax.set_ylim(0, 105)
    ax.set_title("Camera A's calibration is extrapolated over 92% of its\nfield of view "
                 "(fitted on just 11.2-12.2 m)", fontweight="bold", fontsize=10)
    ax2.grid(True, axis="x", zorder=0)
    for i, cam in enumerate(CAMS):
        e = ex[cam]
        ax2.barh(i, e["fov_range_max_m"] - e["fov_range_min_m"], left=e["fov_range_min_m"],
                 height=0.55, color="#dddddd", zorder=2,
                 label="reachable FOV range" if i == 0 else None)
        ax2.barh(i, e["fitted_max_m"] - e["fitted_min_m"], left=e["fitted_min_m"],
                 height=0.55, color=COLOR[cam], zorder=3,
                 label="fitted range window" if i == 0 else None)
        grp = [r["range_m"] for r in rows if r["camera"] == cam]
        ax2.scatter(grp, np.full(len(grp), i + 0.34), s=4, c=COLOR[cam], alpha=0.35,
                    lw=0, zorder=4, label="observed detections" if i == 0 else None)
        ax2.text(e["fitted_max_m"] + 0.25, i, f"{e['fitted_min_m']:.1f}-{e['fitted_max_m']:.1f} m",
                 va="center", fontsize=8, color=COLOR[cam], fontweight="bold")
    ax2.set_yticks(range(4), [f"camera {SHORT[c]}" for c in CAMS])
    ax2.set_xlabel("ground range from camera [m]")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("Every camera is used well outside the range band\nits constants were fitted on",
                  fontweight="bold", fontsize=10)
    fig.suptitle("Extrapolation risk: the deployed constants are applied far outside "
                 "their fitted support",
                 fontweight="bold", fontsize=11.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig_b6_extrapolation_risk")


def main() -> int:
    rows, audit = load()
    fig_b1(rows, audit)
    fig_b2(audit)
    fig_b3(rows, audit)
    fig_b4(audit)
    fig_b5(audit)
    fig_b6(rows, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
