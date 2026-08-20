#!/usr/bin/env python3
"""Box-bottom reading: what the bias depends on, and why R cannot carry it.

Serves the localization_reading_story brief, Part 1/Part 2. Recomposes an existing
measurement (logs/studies/keypoint_measurement/bbox_baseline) -- no new capture.

    python3 experiments/localization_reading_story/plot_bias_vs_noise.py
    -> logs/studies/localization_reading_story/bias_vs_noise/

Bins are EQUAL-COUNT (8 per axis). Coarse quantile bins that lump 1.2-5 m into one
near bin make the distance trend vanish entirely; that mistake was made twice.
"""
import csv

from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

SRC = Path(__file__).resolve().parents[2] / "logs/studies/keypoint_measurement/bbox_baseline/per_sample.csv"
OUT = Path(__file__).resolve().parents[2] / "logs/studies/localization_reading_story/bias_vs_noise"
BIAS, NOISE, REF = "#1b3b5f", "#d89000", "#888888"

rows = [r for r in csv.DictReader(open(SRC)) if int(float(r["detected"])) == 1]
g = lambda k, s=1.0: np.array([float(r[k]) for r in rows]) * s
x, y, ex, ey, yaw = g("x"), g("y"), g("err_x_m", 100), g("err_y_m", 100), g("yaw_rad")
CAM = np.array([0.0, -5.50])
d = np.hypot(x - CAM[0], y - CAM[1])
ur = np.stack([(x - CAM[0]) / d, (y - CAM[1]) / d], 1)
al = ex * ur[:, 0] + ey * ur[:, 1]
cr = -ex * ur[:, 1] + ey * ur[:, 0]

def binned(vals, keys, nbin=8):
    """Equal-COUNT bins, so neither axis is favoured by the binning."""
    q = np.quantile(keys, np.linspace(0, 1, nbin+1)); q[0] -= 1e-9; q[-1] += 1e-9
    out = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (keys >= lo) & (keys < hi)
        if m.sum() < 6: continue
        out.append((keys[m].mean(), vals[m].mean(), vals[m].std(ddof=1)/np.sqrt(m.sum())))
    return np.array(out)

hdeg = np.degrees(np.mod(yaw, 2*np.pi))
H, D = binned(al, hdeg), binned(al, d)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.8, 5.9), sharey=True)
for ax, B, xl, ttl, lab in [
    (a0, H, "which way the robot is facing (degrees)",
     "The error swings 4.5 cm with the robot's heading",
     "4.5 cm"),
    (a1, D, "distance from the camera (m)",
     "\u2026and 2.8 cm with distance \u2014 heading is twice as strong",
     "2.8 cm")]:
    ax.axhline(0, color=REF, lw=1.0)
    ax.axhline(al.mean(), color=REF, lw=1.2, ls="--")
    ax.fill_between(B[:, 0], B[:, 1]-1.96*B[:, 2], B[:, 1]+1.96*B[:, 2], color=BIAS, alpha=0.18, lw=0)
    ax.plot(B[:, 0], B[:, 1], "-o", color=BIAS, lw=2.4, ms=7, markeredgecolor="white", mew=1.3)
    lo, hi = B[:, 1].min(), B[:, 1].max()
    xa = B[:, 0].min() + 0.955*(B[:, 0].max()-B[:, 0].min())
    ax.annotate("", xy=(xa, lo), xytext=(xa, hi),
                arrowprops=dict(arrowstyle="<->", color="#b03030", lw=1.8))
    ax.text(xa - 0.02*(B[:, 0].max()-B[:, 0].min()), 0.5*(lo+hi), lab, rotation=90,
            va="center", ha="right", fontsize=10.5, color="#b03030", weight="bold")
    ax.set_xlabel(xl, fontsize=11)
    ax.set_title(ttl, fontsize=12.5, weight="bold")
    ax.grid(alpha=0.22); ax.set_axisbelow(True)
a0.set_ylabel("error along the camera's line of sight (cm)\nnegative = reported too close to the camera", fontsize=10.5)
a0.text(12, al.mean()+0.30, f"average lean {al.mean():.2f} cm", fontsize=9.5, color="#666")
a0.set_xticks(np.arange(0, 361, 90))
fig.text(0.5, 0.015,
 f"Deployed reading (bottom of the detected box projected onto the floor), {len(rows)} held-out detections, single-camera warehouse, scored against ground truth.\n"
 "Eight equal-count bins per panel so neither axis is favoured; points are bin means, bands 95% confidence on the mean. Both panels share one vertical scale.\n"
 "With the noise removed the spread of the true bias is 1.36 cm across heading and 0.71 cm across distance. Neither is carried by the measurement covariance, which is near-uniform.",
 ha="center", fontsize=8.5, color="#333")
fig.tight_layout(rect=(0, 0.10, 1, 1))
fig.savefig(OUT/"fig_bias_heading_not_range.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_bias_heading_not_range.png")

# ---- Figure B -------------------------------------------------------------
fig, (b0, b1) = plt.subplots(1, 2, figsize=(13.8, 6.0))
within = 2.95; bias_var = 1.36
n = np.arange(1, 31)
b0.plot(n, within/np.sqrt(n), "-o", color=NOISE, lw=2.4, ms=5, label="random error, after averaging $n$ readings")
b0.axhline(bias_var, color=BIAS, lw=2.4, label="how much the bias varies over the workspace")
ncross = (within/bias_var)**2
b0.axvline(ncross, color="#b03030", ls=":", lw=1.6)
b0.plot([ncross], [bias_var], "*", color="#b03030", ms=18, zorder=5)
b0.annotate(f"after ~{ncross:.0f} readings the part you\nCANNOT average away is the bigger one",
            xy=(ncross, bias_var), xytext=(7.5, 2.35), fontsize=10, color="#b03030",
            arrowprops=dict(arrowstyle="->", color="#b03030", lw=1.4))
b0.set_xlabel("number of camera readings averaged", fontsize=11)
b0.set_ylabel("remaining error (cm)  —  lower is better", fontsize=11)
b0.set_xlim(0.5, 30); b0.set_ylim(0, 3.2)
b0.grid(alpha=0.22); b0.set_axisbelow(True)
b0.legend(fontsize=9.6, loc="upper right", framealpha=0.95)
b0.set_title("Looking more often fixes the noise\nand never touches the bias", fontsize=12.5, weight="bold")

b1.axhline(0, color=REF, lw=1.0); b1.axvline(0, color=REF, lw=1.0)
b1.scatter(cr, al, s=13, color="#8fa8bd", alpha=0.55, edgecolors="none", zorder=2)
b1.add_patch(Ellipse((cr.mean(), al.mean()), 2*cr.std(ddof=1), 2*al.std(ddof=1),
                     fill=False, edgecolor=NOISE, lw=2.6, zorder=4))
b1.annotate("", xy=(0, al.mean()), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=BIAS, lw=3.4, mutation_scale=22), zorder=5)
b1.annotate(f"the lean: {al.mean():.1f} cm,\nalways along the\nline of sight",
            xy=(0.4, al.mean()*0.5), xytext=(6.6, 2.2), fontsize=10.5, color=BIAS, weight="bold",
            arrowprops=dict(arrowstyle="->", color=BIAS, lw=1.4))
b1.annotate(f"the noise, 1$\\sigma$: {cr.std(ddof=1):.1f} cm across\nthe view but only {al.std(ddof=1):.1f} cm along it",
            xy=(-cr.std(ddof=1)*0.72, al.mean()+al.std(ddof=1)*0.72), xytext=(-15.2, 11.4),
            fontsize=10.5, color="#a06a00", ha="left",
            arrowprops=dict(arrowstyle="->", color="#a06a00", lw=1.4))
b1.plot(0, 0, "P", color="#333", ms=11, zorder=6)
b1.text(0.6, 0.9, "no error", fontsize=9, color="#333")
b1.set_xlabel("error ACROSS the camera's view (cm)", fontsize=11)
b1.set_ylabel("error ALONG the camera's line of sight (cm)", fontsize=11)
b1.set_aspect("equal"); b1.set_xlim(-16, 16); b1.set_ylim(-16, 16)
b1.grid(alpha=0.22); b1.set_axisbelow(True)
b1.set_title("They point in different directions, so one\nuncertainty number cannot describe both",
             fontsize=12.5, weight="bold")
fig.text(0.5, 0.015,
 f"Same {len(rows)} detections. Left: the random part falls as 1/√n; the bias variation (1.36 cm across heading, noise removed) does not fall at all.\n"
 "Right: every detection's error in the camera's own frame. The lean runs along the line of sight (a wrong contact point on the robot); the noise is widest across it\n"
 "(the box's horizontal centre shifts as the robot turns). Deployed box-bottom reading, single-camera warehouse, offline against ground truth.",
 ha="center", fontsize=8.5, color="#333")
fig.tight_layout(rect=(0, 0.10, 1, 1))
fig.savefig(OUT/"fig_bias_vs_noise.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_bias_vs_noise.png")
