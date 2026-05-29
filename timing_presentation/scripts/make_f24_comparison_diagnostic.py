#!/usr/bin/env python3
"""F24 — Gazebo C1 vs C2 post-run diagnostic.

Produces a 2×3 panel figure:
  Row 0: BEV trajectory over visibility heatmap (C1 | C2)
  Row 1: goal distance, EFE obstacle cost, localization error vs time (C1 | C2)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml

DURABLE = Path("/home/joostleliveld/Thesis/timing_presentation")
REPO    = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
GP_PATH = REPO / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
OUT_DIR = DURABLE / "figures/F24"

C1_DIR = DURABLE / "runs/gazebo/hier_c1_v12/experiment_20260528_175437"
C2_DIR = DURABLE / "runs/gazebo/hier_c2_v12/experiment_20260528_174624"

GOAL   = np.array([1.00, 1.75])
START  = np.array([3.20, -1.00])
WORLD  = "warehouse_aws.world.sdf"
COLORS = {"C1": "#4477aa", "C2": "#e6194b"}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.facecolor": "white"})


def load_run(run_dir: Path):
    df = pd.read_csv(run_dir / "experiment.csv")
    ps = pd.read_csv(run_dir / "plan_samples.csv")
    with open(run_dir / "run_summary.json") as f:
        summ = json.load(f)
    wp = yaml.safe_load((run_dir / "world_profiles.yaml").read_text())
    regions = [r for r in wp["worlds"][WORLD].get("known_2d_regions", [])
               if r.get("type") == "traversable"]
    first_cmd = float(summ["first_cmd_stamp"])
    df["t"] = df["stamp"] - first_cmd
    df = df[df["t"] >= -0.5].copy()
    ps["t"] = ps["plan_stamp"] - first_cmd
    return df, ps, summ, regions


def draw_bev(ax, label, df, ps, summ, regions, gp):
    xs, ys = gp["xs"], gp["ys"]
    Pc = gp["P_conservative_plan_map"]
    ax.imshow(Pc, extent=[xs.min(), xs.max(), ys.min(), ys.max()],
              origin="lower", cmap="RdYlGn", vmin=0, vmax=1,
              aspect="equal", zorder=0, alpha=0.75)

    for r in regions:
        ax.add_patch(Rectangle((r["xmin"], r["ymin"]),
                               r["xmax"] - r["xmin"], r["ymax"] - r["ymin"],
                               facecolor="none", edgecolor="k",
                               lw=0.8, ls=":", alpha=0.5, zorder=1))

    # last planned trajectory before crash/stuck
    last_stamp = ps["plan_stamp"].max()
    last_plan = ps[ps["plan_stamp"] == last_stamp]
    ax.plot(last_plan["x"], last_plan["y"], "--", color="orange",
            lw=1.5, alpha=0.85, zorder=3, label="last plan")

    # executed trajectory colored by time
    traj = df[df["truth_x"].notna()][["t", "truth_x", "truth_y"]].copy()
    if len(traj) > 1:
        t_arr = traj["t"].values
        x_arr = traj["truth_x"].values
        y_arr = traj["truth_y"].values
        for i in range(len(t_arr) - 1):
            c = plt.cm.plasma(0.1 + 0.8 * i / max(len(t_arr) - 2, 1))
            ax.plot(x_arr[i:i+2], y_arr[i:i+2], "-", color=c, lw=2.5, zorder=5)

    crash_stamp = float(summ.get("first_crash_stamp", "nan"))
    if not np.isnan(crash_stamp):
        crash_t = crash_stamp - float(summ["first_cmd_stamp"])
        row = traj[traj["t"] <= crash_t].iloc[-1] if len(traj[traj["t"] <= crash_t]) else None
        if row is not None:
            ax.plot(row["truth_x"], row["truth_y"], "X", color="red",
                    ms=14, mec="k", mew=1.0, zorder=9, label=f"crash t={crash_t:.1f}s")

    ax.plot(*START, "o", color="k", ms=9, mec="w", mew=1.5, zorder=9, label="start")
    ax.plot(*GOAL, "*", color="k", ms=16, mec="w", mew=1.5, zorder=9, label="goal")

    reason = summ.get("completion_reason", "?")
    path_m = summ.get("path_length_m", 0)
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-3.0, 3.5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{label}  —  {reason}  |  path {path_m:.2f} m", pad=5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


def draw_timeseries(axes, label, df, summ, color):
    t = df["t"].values

    # goal distance
    ax0 = axes[0]
    ax0.plot(t, df["goal_dist"], color=color, lw=1.8, label=label)
    ax0.set_ylabel("goal dist (m)")
    ax0.set_xlabel("t (s)")
    ax0.set_title("Goal distance")
    ax0.axhline(0.2, ls="--", color="gray", lw=1)

    # EFE obstacle cost (log scale)
    ax1 = axes[1]
    obs = df["efe_obstacle"].clip(lower=1e-1)
    ax1.semilogy(t, obs, color=color, lw=1.8, label=label)
    ax1.set_ylabel("EFE obstacle (log)")
    ax1.set_xlabel("t (s)")
    ax1.set_title("EFE obstacle cost")

    # localization belief error
    ax2 = axes[2]
    ax2.plot(t, df["truth_belief_error_m"], color=color, lw=1.8, label=label)
    ax2.set_ylabel("belief error (m)")
    ax2.set_xlabel("t (s)")
    ax2.set_title("Belief localization error")

    crash_stamp = float(summ.get("first_crash_stamp", "nan"))
    if not np.isnan(crash_stamp):
        crash_t = crash_stamp - float(summ["first_cmd_stamp"])
        for ax in axes:
            ax.axvline(crash_t, color="red", lw=1.2, ls="--", alpha=0.7)

    stuck_t = float(summ.get("elapsed_after_first_cmd_s", "nan"))
    if not np.isnan(stuck_t) and summ.get("completion_reason") == "stuck":
        for ax in axes:
            ax.axvline(stuck_t, color="orange", lw=1.2, ls=":", alpha=0.8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gp = dict(np.load(GP_PATH, allow_pickle=True))

    df1, ps1, summ1, regions = load_run(C1_DIR)
    df2, ps2, summ2, _       = load_run(C2_DIR)

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 6)

    ax_bev_c1 = fig.add_subplot(gs[0, 0:3])
    ax_bev_c2 = fig.add_subplot(gs[0, 3:6])
    ts_axes_c1 = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ts_axes_c2 = [fig.add_subplot(gs[1, i+3]) for i in range(3)]

    draw_bev(ax_bev_c1, "C1 (constant-R EFE)", df1, ps1, summ1, regions, gp)
    draw_bev(ax_bev_c2, "C2 (GP-conditioned EFE)", df2, ps2, summ2, regions, gp)
    draw_timeseries(ts_axes_c1, "C1", df1, summ1, COLORS["C1"])
    draw_timeseries(ts_axes_c2, "C2", df2, summ2, COLORS["C2"])

    # shared legend for time series
    for axes in (ts_axes_c1, ts_axes_c2):
        for ax in axes:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"F24 — Gazebo C1 vs C2 diagnostic  |  task B1 ({START[0]:.1f},{START[1]:.1f}) → ({GOAL[0]:.1f},{GOAL[1]:.1f})\n"
        f"C1: {summ1['completion_reason']} path={summ1['path_length_m']:.2f}m  |  "
        f"C2: {summ2['completion_reason']} path={summ2['path_length_m']:.2f}m obstacle_penetration={summ2['max_obstacle_penetration_m']:.3f}m",
        fontsize=12, fontweight="bold"
    )

    out_png = OUT_DIR / "F24_gazebo_c1_c2_diagnostic.png"
    out_pdf = OUT_DIR / "F24_gazebo_c1_c2_diagnostic.pdf"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    # summary
    print("\n── Summary ──")
    for lbl, summ in (("C1", summ1), ("C2", summ2)):
        print(f"  {lbl}: reason={summ['completion_reason']}  path={summ['path_length_m']:.2f}m"
              f"  final_goal_dist={summ['final_goal_distance']:.2f}m"
              f"  mean_solve_ms={summ['mean_solve_time_ms']:.0f}"
              f"  efe_obstacle={summ['mean_efe_obstacle']:.0f}"
              f"  belief_err={summ['mean_truth_belief_error_m']:.3f}m"
              f"  crashed={summ['crashed']}")


if __name__ == "__main__":
    main()
