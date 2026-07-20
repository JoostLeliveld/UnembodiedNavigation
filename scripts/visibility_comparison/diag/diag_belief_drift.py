#!/usr/bin/env python3
"""Belief-drift diagnostic: truth vs belief vs camera over the reliability map,
with error-vs-time decomposed (position + heading), occlusion (rho) shaded, and
camera-update accept/reject + turn markers.

This is the core figure showing WHERE and WHY the belief drifts, and the basis
for the per-task "error grows in occluded / recovers in visible" deliverable.

Usage:
    python diag_belief_drift.py <run_dir> [--out <dir>]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diag_common as dc

C_TRUTH = "#111111"
C_BELIEF = "#e8820c"
C_STATE = "#2a72c4"
C_CAM = "#cf2a4a"


def shade_blind(ax, t, blind):
    """Shade time intervals where the camera ACTUALLY did not detect the robot
    (genuine occlusion / blind dead-reckoning) -- NOT the GP reliability rho."""
    start = None
    for i in range(len(t)):
        if (not blind[i]) and start is None:
            # blind[i] True == detected; we shade where NOT detected
            pass
        if blind[i] and start is None:
            start = t[i]
        elif (not blind[i]) and start is not None:
            ax.axvspan(start, t[i], color="#555", alpha=0.18, lw=0)
            start = None
    if start is not None:
        ax.axvspan(start, t[-1], color="#555", alpha=0.18, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    perc, exp, summary = dc.load_run(run_dir)
    out_dir = Path(args.out) if args.out else (run_dir / "diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    e = exp.dropna(subset=["stamp", "odom_map_x", "odom_map_y"]).sort_values("stamp").reset_index(drop=True)
    t = e["stamp"].to_numpy(float)
    t0 = dc.first_cmd_time(exp) or t[0]
    rho_truth = dc.sample_rho(e["odom_map_x"].to_numpy(), e["odom_map_y"].to_numpy())
    detected = dc.detection_coverage(perc, t)      # True where camera actually sees robot
    blind = ~detected                              # genuine occlusion / blind dead-reckoning
    crash_t = float(summary.get("first_crash_stamp", "nan") or "nan")

    g = dc.gp()
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.33, wspace=0.18)

    # ---------- MAP ----------
    axm = fig.add_subplot(gs[:, 0])
    axm.imshow(g["rho"], origin="lower", extent=g["extent"], cmap="viridis",
               vmin=0, vmax=0.9, alpha=0.85, zorder=0)
    for (xm, xM, ym, yM) in dc.driveable_prisms(run_dir):
        axm.plot([xm, xM, xM, xm, xm], [ym, ym, yM, yM, ym], color="#39d", lw=0.8, alpha=0.5, zorder=1)
    axm.plot(e["odom_map_x"], e["odom_map_y"], "-", color=C_TRUTH, lw=2.2, label="truth", zorder=4)
    if "planner_belief_x" in e:
        b = e.dropna(subset=["planner_belief_x", "planner_belief_y"])
        axm.plot(b["planner_belief_x"], b["planner_belief_y"], "-", color=C_BELIEF, lw=1.8,
                 alpha=0.9, label="belief", zorder=5)
    # camera detections (state)
    det = perc[perc["detected"] == 1] if "detected" in perc else perc
    if "pred_world_x_calibrated" in det:
        axm.scatter(det["pred_world_x_calibrated"], det["pred_world_y_calibrated"], s=8,
                    c=C_CAM, alpha=0.5, label="camera (affine)", zorder=3)
    axm.scatter([e["odom_map_x"].iloc[0]], [e["odom_map_y"].iloc[0]], s=120, marker="o",
                c="lime", edgecolor="k", zorder=6, label="start")
    if "goal_x" in e and np.isfinite(e["goal_x"].iloc[-1]):
        axm.scatter([e["goal_x"].iloc[-1]], [e["goal_y"].iloc[-1]], s=200, marker="*",
                    c="red", edgecolor="k", zorder=6, label="goal")
    if np.isfinite(crash_t):
        ci = int(np.argmin(np.abs(t - crash_t)))
        axm.scatter([e["odom_map_x"].iloc[ci]], [e["odom_map_y"].iloc[ci]], s=240, marker="X",
                    c="red", edgecolor="k", zorder=7, label="collision")
    axm.set_title(f"{run_dir.name}\noutcome={summary.get('completion_reason','?')}", fontsize=11, loc="left")
    axm.set_xlabel("x (m)"); axm.set_ylabel("y (m)")
    axm.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axm.set_aspect("equal")

    # ---------- POSITION ERROR ----------
    ax1 = fig.add_subplot(gs[0, 1])
    shade_blind(ax1, t, blind)
    if "belief_error_odom_m" in e:
        ax1.plot(t, e["belief_error_odom_m"], color=C_BELIEF, lw=1.8, label="truth-belief")
    if "state_error_odom_m" in e:
        ax1.plot(t, e["state_error_odom_m"], color=C_STATE, lw=1.0, alpha=0.6, label="truth-state(cam)")
    ax1.axvline(t0, color="k", ls=":", lw=1, alpha=0.6)
    if np.isfinite(crash_t):
        ax1.axvline(crash_t, color="red", ls="--", lw=1.2)
    ax1.set_ylabel("pos error (m)"); ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Position error (gray band = camera BLIND: robot not detected)", fontsize=10, loc="left")
    ax1.grid(alpha=0.25)

    # ---------- HEADING ERROR ----------
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    shade_blind(ax2, t, blind)
    if "yaw_error_odom_map_vs_belief_rad" in e:
        ax2.plot(t, np.degrees(e["yaw_error_odom_map_vs_belief_rad"].abs()), color=C_BELIEF, lw=1.8, label="|truth-belief yaw|")
    if "yaw_error_odom_map_vs_state_rad" in e:
        ax2.plot(t, np.degrees(e["yaw_error_odom_map_vs_state_rad"].abs()), color=C_STATE, lw=1.0, alpha=0.6, label="|truth-state yaw|")
    ax2.axvline(t0, color="k", ls=":", lw=1, alpha=0.6)
    if np.isfinite(crash_t):
        ax2.axvline(crash_t, color="red", ls="--", lw=1.2)
    ax2.set_ylabel("heading error (deg)"); ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("Heading error (camera_xy_only -> uncorrected odom drift)", fontsize=10, loc="left")
    ax2.grid(alpha=0.25)

    # ---------- VISIBILITY + UPDATES + STALENESS ----------
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax3.plot(t, rho_truth, color="#2a8", lw=1.5, label="GP reliability rho@truth (PRIOR, not actual)")
    ax3.fill_between(t, 0, 1.0, where=blind, color="#555", alpha=0.18, step="mid", label="camera BLIND (actual)")
    ax3.axhline(0.5, color="#888", ls=":", lw=0.8)
    # camera update accept/reject ticks
    if "planner_pixel_correction_available" in e:
        corr = e[e["planner_pixel_correction_available"] == 1]
        acc = corr[corr["pixel_corr_accepted"] == 1]
        rej = corr[corr["pixel_corr_accepted"] != 1]
        ax3.scatter(acc["stamp"], np.full(len(acc), 1.04), marker="|", c="green", s=30, label="update accept")
        ax3.scatter(rej["stamp"], np.full(len(rej), 1.10), marker="|", c="red", s=40, label="update reject")
    ax3.set_ylim(-0.05, 1.18)
    ax3.set_ylabel("rho / updates"); ax3.set_xlabel("time (s)")
    ax3.legend(fontsize=7, loc="lower left", ncol=2)
    ax3.grid(alpha=0.25)

    # summary text
    def m(col, after=True):
        if col not in e:
            return float("nan")
        s = e[e["stamp"] >= t0][col] if after else e[col]
        return float(np.nanmedian(s.abs())) if len(s) else float("nan")
    txt = (f"after first cmd (t>={t0:.0f}s):  "
           f"pos_err med={m('belief_error_odom_m'):.3f}m  "
           f"yaw_err med={np.degrees(m('yaw_error_odom_map_vs_belief_rad')):.1f}deg  "
           f"yaw_err max={np.degrees(np.nanmax(e[e['stamp']>=t0]['yaw_error_odom_map_vs_belief_rad'].abs())) if 'yaw_error_odom_map_vs_belief_rad' in e else float('nan'):.1f}deg")
    fig.text(0.52, 0.005, txt, fontsize=9, color="#333")

    png = out_dir / "belief_drift.png"
    fig.savefig(png, dpi=115, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
