#!/usr/bin/env python3
"""F14 – Hierarchical planner Gazebo run diagnostic (enhanced multi-panel).

Uses the best-frame hier_v1 run (experiment_20260527_164918).
Panels:
  (a) Bird's-eye trajectory with 2σ covariance ellipses + waypoints
  (b) Belief-error (x, y, norm) with planner covariance envelope
  (c) Visibility / observation noise (p_vis, R_plan std)
  (d) EFE cost decomposition
  (e) Commands (v, ω) with pixel-update timing ticks
  (f) Pixel correction: innovation norms + update timing + YOLO detect/miss
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────
RUN_DIR = Path("/home/joostleliveld/Thesis/timing_presentation/runs/gazebo/hier_v1/experiment_20260527_164918")
PERC_CSV = RUN_DIR / "perception.csv"
EXP_CSV = RUN_DIR / "experiment.csv"
SUMMARY = RUN_DIR / "run_summary.json"
OUT_DIR = Path("/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures")

# ── load ─────────────────────────────────────────────────────────────────
df = pd.read_csv(EXP_CSV)
with open(SUMMARY) as f:
    summ = json.load(f)
perc = pd.read_csv(PERC_CSV) if PERC_CSV.exists() else None

# ── time origin at first cmd ─────────────────────────────────────────────
first_cmd = float(summ.get("first_cmd_stamp", df.loc[df["cmd_v"].abs() > 0.01, "stamp"].iloc[0]))
df["t"] = df["stamp"] - first_cmd
df = df[df["t"] >= -0.5].copy()

# ── derived columns ──────────────────────────────────────────────────────
df["belief_err_x"] = df["planner_belief_x"] - df["truth_x"]
df["belief_err_y"] = df["planner_belief_y"] - df["truth_y"]
df["belief_err_m"] = np.sqrt(df["belief_err_x"]**2 + df["belief_err_y"]**2)
df["state_err_m"]  = np.sqrt((df["state_x"] - df["truth_x"])**2 +
                              (df["state_y"] - df["truth_y"])**2)

# Planner 2-sigma envelope from covariance
df["planner_2sig_x"] = 2.0 * np.sqrt(df["planner_cov_x"].clip(lower=0))
df["planner_2sig_y"] = 2.0 * np.sqrt(df["planner_cov_y"].clip(lower=0))

# ── reconstruct global plan waypoints from offline planner ───────────────
# The global plan was published once (plan_points=61) at around t≈8.88s.
# We can reconstruct approximate waypoints from the belief trajectory
# at start, and the known waypoint_spacing_m=1.0 from the launch config.
# Use the plan's first-published belief as start.
global_plan_rows = df[df["plan_points"] == 61]
if len(global_plan_rows) > 0:
    gp_start_x = float(global_plan_rows.iloc[0]["planner_belief_x"])
    gp_start_y = float(global_plan_rows.iloc[0]["planner_belief_y"])
else:
    gp_start_x, gp_start_y = 3.4, -1.0

# Goal
gx = float(df["goal_x"].dropna().iloc[-1]) if "goal_x" in df.columns else 1.0
gy = float(df["goal_y"].dropna().iloc[-1]) if "goal_y" in df.columns else 2.0

# Since we know the plan goes: start → up A4 → west through mid cross-aisle → goal,
# approximate the global path from the truth trajectory shape
# (the global plan was published at plan_points=61, path=4.63m)
# For the figure we'll use truth as a proxy for the global route,
# with waypoints spaced every 1.0m of arc length.
def arc_length_waypoints(xs, ys, spacing=1.0):
    pts = np.column_stack([xs, ys])
    wps = [(pts[0, 0], pts[0, 1])]
    acc = 0.0
    for i in range(1, len(pts)):
        acc += np.linalg.norm(pts[i] - pts[i-1])
        if acc >= spacing:
            wps.append((pts[i, 0], pts[i, 1]))
            acc = 0.0
    # always include final
    if np.linalg.norm(np.array(wps[-1]) - pts[-1]) > 0.01:
        wps.append((pts[-1, 0], pts[-1, 1]))
    return wps

# Use the full truth trajectory to approximate the global plan route
truth_x = df["truth_x"].values
truth_y = df["truth_y"].values
approx_waypoints = arc_length_waypoints(truth_x, truth_y, spacing=1.0)

# ── pixel correction stamps ─────────────────────────────────────────────
pix_corr_stamps = df["planner_pixel_correction_stamp"].dropna()
pix_corr_unique = pix_corr_stamps.drop_duplicates()
pix_corr_t = pix_corr_unique - first_cmd

# Identify new correction events (stamp changes)
pcs = df[["t", "planner_pixel_correction_stamp"]].dropna()
pcs_new = pcs[pcs["planner_pixel_correction_stamp"].diff().abs() > 0.001]
pixel_update_times = pcs_new["t"].values

# ── driveable corridors ─────────────────────────────────────────────────
DRIVE_RECTS = [
    {"xmin": -4.85, "xmax": 4.85, "ymin": -3.45, "ymax": -2.55, "label": "apron"},
    {"xmin": 2.625, "xmax": 3.525, "ymin": -2.85, "ymax": 4.80, "label": "A4"},
    {"xmin": 0.625, "xmax": 1.525, "ymin": -2.85, "ymax": 4.80, "label": "A3"},
    {"xmin": -4.85, "xmax": 4.85, "ymin": 1.45, "ymax": 2.55, "label": "mid cross"},
    {"xmin": -4.85, "xmax": 4.85, "ymin": 4.28, "ymax": 4.80, "label": "upper cross"},
]

# ── covariance ellipse helper ────────────────────────────────────────────
def cov_ellipse(ax, cx, cy, cov_xx, cov_xy, cov_yy, n_sigma=2, **kwargs):
    """Draw a 2σ covariance ellipse."""
    S = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, 1e-8, None)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    w, h = 2 * n_sigma * np.sqrt(eigvals)
    ell = Ellipse((cx, cy), w, h, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell

# ── figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16), facecolor="white")
fig.suptitle(
    "F14 — Hierarchical EFE Planner Diagnostic  ·  hier_v1\n"
    f"Outcome: {summ['completion_reason']}  |  "
    f"min goal = {summ['minimum_goal_distance']:.2f} m  |  "
    f"path = {summ['path_length_m']:.1f} m  |  "
    f"elapsed = {summ['elapsed_after_first_cmd_s']:.1f} s  |  "
    f"mean belief err = {summ['mean_truth_belief_error_m']:.2f} m",
    fontsize=12, fontweight="bold", y=0.995,
)

gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.30, top=0.935, bottom=0.04,
                      left=0.06, right=0.97)

t = df["t"]
t_min, t_max = t.min() - 0.1, t.max() + 0.2

# ═══════════════════════════════════════════════════════════════════════════
# (a) Bird's-eye trajectory with covariance ellipses & waypoints
# ═══════════════════════════════════════════════════════════════════════════
ax_map = fig.add_subplot(gs[0, 0])

# Driveable regions
for r in DRIVE_RECTS:
    ax_map.add_patch(mpatches.Rectangle(
        (r["xmin"], r["ymin"]), r["xmax"] - r["xmin"], r["ymax"] - r["ymin"],
        facecolor="#e8f5e9", edgecolor="#66bb6a", linewidth=0.7, alpha=0.5, zorder=0,
    ))

# Covariance ellipses every ~0.5s along belief trajectory
ellipse_step = max(1, len(df) // 12)
for i in range(0, len(df), ellipse_step):
    row = df.iloc[i]
    if pd.isna(row["planner_cov_x"]) or pd.isna(row["planner_belief_x"]):
        continue
    cov_ellipse(
        ax_map,
        row["planner_belief_x"], row["planner_belief_y"],
        row["planner_cov_x"], row["planner_cov_xy"], row["planner_cov_y"],
        n_sigma=2,
        facecolor="#e5393520", edgecolor="#e53935", linewidth=0.6, zorder=3,
    )

# Truth trajectory
ax_map.plot(truth_x, truth_y, "k-", lw=2.2, label="truth", zorder=5)
# Belief trajectory
ax_map.plot(df["planner_belief_x"], df["planner_belief_y"], "-", color="#e53935",
            lw=1.3, alpha=0.8, label="planner belief", zorder=4)
# State estimate
ax_map.plot(df["state_x"], df["state_y"], "--", color="#1565c0", lw=0.9,
            alpha=0.5, label="state estimate", zorder=3)

# Waypoints (approx from truth arc-length)
wp_arr = np.array(approx_waypoints)
ax_map.plot(wp_arr[:, 0], wp_arr[:, 1], "D", color="#ff6f00", ms=7, mew=1.5,
            mfc="white", zorder=8, label=f"waypoints (≈{len(wp_arr)})")

# Start / goal
sx, sy = truth_x[0], truth_y[0]
ax_map.plot(sx, sy, "o", color="#2e7d32", ms=10, zorder=10,
            label=f"start ({sx:.1f}, {sy:.1f})")
ax_map.plot(gx, gy, "*", color="#c62828", ms=14, zorder=10,
            label=f"goal ({gx:.1f}, {gy:.1f})")

# Crash marker
crash_rows = df[df["collision_any"] == True]
if len(crash_rows) > 0:
    ci = crash_rows.index[0]
    ax_map.plot(df.loc[ci, "truth_x"], df.loc[ci, "truth_y"], "x",
                color="#b71c1c", ms=13, mew=3, zorder=11, label="crash")

ax_map.set_xlabel("x [m]")
ax_map.set_ylabel("y [m]")
ax_map.set_title("(a) Trajectory with 2σ belief covariance", fontsize=11, fontweight="bold")
ax_map.set_aspect("equal")
ax_map.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
ax_map.grid(True, alpha=0.25)

# ═══════════════════════════════════════════════════════════════════════════
# (b) Belief error with covariance envelope
# ═══════════════════════════════════════════════════════════════════════════
ax_err = fig.add_subplot(gs[0, 1])

ax_err.fill_between(t, -df["planner_2sig_y"], df["planner_2sig_y"],
                     alpha=0.12, color="#1e88e5", label="±2σ_y envelope", zorder=1)
ax_err.fill_between(t, -df["planner_2sig_x"], df["planner_2sig_x"],
                     alpha=0.10, color="#e53935", label="±2σ_x envelope", zorder=1)

ax_err.plot(t, df["belief_err_x"], color="#e53935", lw=1.0, alpha=0.9, label="belief err x")
ax_err.plot(t, df["belief_err_y"], color="#1e88e5", lw=1.0, alpha=0.9, label="belief err y")
ax_err.plot(t, df["belief_err_m"], color="black", lw=1.8, label="‖belief err‖")
ax_err.plot(t, df["state_err_m"],  color="#43a047", lw=1.0, ls="--", alpha=0.7,
            label="‖state err‖")
ax_err.axhline(0, color="gray", lw=0.4)

# Pixel update ticks on x-axis
for pt in pixel_update_times:
    ax_err.axvline(pt, color="#ff6f00", lw=0.4, alpha=0.35, zorder=0)

ax_err.set_xlabel("time after first cmd [s]")
ax_err.set_ylabel("error [m]")
ax_err.set_title("(b) Belief error with ±2σ covariance envelope", fontsize=11, fontweight="bold")
ax_err.legend(fontsize=6.5, ncol=3, loc="upper left", framealpha=0.85)
ax_err.grid(True, alpha=0.25)
ax_err.set_xlim(t_min, t_max)

# ═══════════════════════════════════════════════════════════════════════════
# (c) Visibility & observation noise
# ═══════════════════════════════════════════════════════════════════════════
ax_vis = fig.add_subplot(gs[1, 0])
ax_vis.plot(t, df["p_vis_plan"], color="#7b1fa2", lw=1.5, label="p_vis")
ax_vis.axhline(0.2, color="#7b1fa2", lw=0.6, ls=":", alpha=0.4)
ax_vis.set_ylabel("p_vis", color="#7b1fa2")
ax_vis.set_ylim(-0.05, 1.05)
ax_vis.tick_params(axis="y", labelcolor="#7b1fa2")

ax_r = ax_vis.twinx()
ax_r.plot(t, df["r_plan_u_std"], color="#ff6f00", lw=1.3, label="R_plan σ_u [px]")
ax_r.set_ylabel("R_plan std [px]", color="#ff6f00")
ax_r.tick_params(axis="y", labelcolor="#ff6f00")

# Pixel update ticks
for pt in pixel_update_times:
    ax_vis.axvline(pt, color="#00897b", lw=0.5, alpha=0.3, zorder=0)

ax_vis.set_xlabel("time after first cmd [s]")
ax_vis.set_title("(c) Visibility probability & observation noise", fontsize=11, fontweight="bold")
lines1, labs1 = ax_vis.get_legend_handles_labels()
lines2, labs2 = ax_r.get_legend_handles_labels()
ax_vis.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="center left", framealpha=0.85)
ax_vis.grid(True, alpha=0.25)
ax_vis.set_xlim(t_min, t_max)

# ═══════════════════════════════════════════════════════════════════════════
# (d) EFE cost decomposition
# ═══════════════════════════════════════════════════════════════════════════
ax_efe = fig.add_subplot(gs[1, 1])
efe_items = [
    ("efe_risk",      "risk",      "#e53935"),
    ("efe_ambiguity", "ambiguity", "#1e88e5"),
    ("efe_obstacle",  "obstacle",  "#f57c00"),
    ("efe_total",     "total",     "black"),
]
for col, label, color in efe_items:
    if col in df.columns:
        ax_efe.plot(t, df[col], label=label, color=color,
                    lw=2.0 if col == "efe_total" else 1.2)

ax_efe.set_xlabel("time after first cmd [s]")
ax_efe.set_ylabel("EFE cost")
ax_efe.set_title("(d) EFE cost decomposition", fontsize=11, fontweight="bold")
ax_efe.legend(fontsize=7, framealpha=0.85)
ax_efe.grid(True, alpha=0.25)
ax_efe.set_xlim(t_min, t_max)

# ═══════════════════════════════════════════════════════════════════════════
# (e) Commands with pixel-update timing
# ═══════════════════════════════════════════════════════════════════════════
ax_cmd = fig.add_subplot(gs[2, 0])
ax_cmd.plot(t, df["cmd_v"], color="#00897b", lw=1.5, label="v [m/s]")
ax_cmd.plot(t, df["cmd_w"], color="#5e35b1", lw=1.3, label="ω [rad/s]")
ax_cmd.axhline(0, color="gray", lw=0.4)

# Pixel update ticks on command plot
for i, pt in enumerate(pixel_update_times):
    ax_cmd.axvline(pt, color="#ff6f00", lw=0.7, alpha=0.5, zorder=0,
                   label="pixel update" if i == 0 else None)

ax_cmd.set_xlabel("time after first cmd [s]")
ax_cmd.set_ylabel("command")
ax_cmd.set_title("(e) Velocity commands + pixel update timing", fontsize=11, fontweight="bold")
ax_cmd.legend(fontsize=7, framealpha=0.85)
ax_cmd.grid(True, alpha=0.25)
ax_cmd.set_xlim(t_min, t_max)

# ═══════════════════════════════════════════════════════════════════════════
# (f) Pixel correction detail: innovation, update norm, YOLO status
# ═══════════════════════════════════════════════════════════════════════════
ax_pix = fig.add_subplot(gs[2, 1])

# Pixel update norm
ax_pix.plot(t, df["pixel_corr_xy_update_norm_m"], color="#00695c", lw=1.4,
            label="xy update norm [m]")

# Innovation magnitudes
innov_mag = np.sqrt(df["pixel_corr_innov_u"]**2 + df["pixel_corr_innov_v"]**2)
ax_pix2 = ax_pix.twinx()
ax_pix2.plot(t, innov_mag, color="#ad1457", lw=0.9, alpha=0.7,
             label="‖innovation‖ [px]")
ax_pix2.set_ylabel("innovation [px]", color="#ad1457")
ax_pix2.tick_params(axis="y", labelcolor="#ad1457")

# Pixel update timing ticks
for i, pt in enumerate(pixel_update_times):
    ax_pix.axvline(pt, color="#ff6f00", lw=0.8, alpha=0.4, zorder=0,
                   label="pixel update" if i == 0 else None)

# YOLO detection ticks from perception.csv
if perc is not None and "diag_stamp" in perc.columns:
    perc_t = perc["diag_stamp"] - first_cmd
    detected = perc["detected"].fillna(0).astype(float)
    mask_post = perc_t >= -0.5
    pt_post = perc_t[mask_post]
    det_post = detected[mask_post]
    y_tick = -0.003  # just below zero
    ax_pix.scatter(pt_post[det_post > 0.5],
                   np.full(int((det_post > 0.5).sum()), y_tick),
                   c="#43a047", s=20, marker="|", zorder=6, label="YOLO detect")
    ax_pix.scatter(pt_post[det_post < 0.5],
                   np.full(int((det_post < 0.5).sum()), y_tick),
                   c="#e53935", s=20, marker="|", zorder=6, label="YOLO miss")

ax_pix.set_xlabel("time after first cmd [s]")
ax_pix.set_ylabel("update norm [m]")
ax_pix.set_title("(f) Pixel correction: update norm, innovation & YOLO status",
                  fontsize=11, fontweight="bold")
lines_p, labs_p = ax_pix.get_legend_handles_labels()
lines_p2, labs_p2 = ax_pix2.get_legend_handles_labels()
ax_pix.legend(lines_p + lines_p2, labs_p + labs_p2,
              fontsize=6.5, loc="upper left", ncol=2, framealpha=0.85)
ax_pix.grid(True, alpha=0.25)
ax_pix.set_xlim(t_min, t_max)

# ── save ─────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in (".pdf", ".png"):
    out = OUT_DIR / f"F14_hier_diagnostic{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)
