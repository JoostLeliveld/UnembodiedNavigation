#!/usr/bin/env python3
"""
Clean route-choice dashboard for any Fx B1 run.

Usage:
    python3 make_dashboard.py --fig 33 --log f33_b1_route_choice_v1

Layout (2×3):
  [BEV overlay — large]  |  [Goal distance]      |  [Solve time hist]
  [p_vis_plan both]      |  [Obstacle clearance]  |  [Stats table]

The BEV shows:
  - GP visibility field (correct orientation, no transpose)
  - Rack outlines with labels
  - C1 truth path (blue, time-coloured)
  - C2 truth path (red, time-coloured)
  - Start marker, goal star
  - Crash markers if applicable
  NO route seed lines — they add noise, not signal
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
GP_PATH = ROOT / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
WP_CFG  = ROOT / "src/experiments/config/world_profiles.yaml"
WORLD   = "warehouse_aws.world.sdf"

C  = {"C1": "#2563eb", "C2": "#dc2626"}
RACKS = [
    ("R4L", 1.725, 2.275, -0.8,  1.25),
    ("R4U", 1.725, 2.275,  2.2,  4.25),
    ("R5L", 3.875, 4.425, -0.8,  1.25),
    ("R5U", 3.875, 4.425,  2.2,  4.25),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_run(log_root: Path, task: str, cond: str):
    base = log_root / task / cond / "seed1"
    if not base.exists():
        return None, None, {}
    exps = sorted(base.glob("experiment_*"))
    for d in reversed(exps):
        csv = d / "experiment.csv"
        if csv.exists() and sum(1 for _ in open(csv)) > 30:
            exp  = pd.read_csv(csv)
            summ = json.loads((d / "run_summary.json").read_text()) if (d/"run_summary.json").exists() else {}
            return exp, d, summ
    return None, None, {}


def load_gp():
    if not GP_PATH.exists():
        return None
    d = np.load(GP_PATH, allow_pickle=True)
    if all(k in d for k in ("xs", "ys", "P_conservative_plan_map")):
        return d["xs"], d["ys"], d["P_conservative_plan_map"]
    return None


def load_regions():
    d = yaml.safe_load(WP_CFG.read_text())
    return d["worlds"][WORLD].get("known_2d_regions", [])


def moving(exp):
    if exp is None or "cmd_v" not in exp.columns:
        return exp
    return exp[exp["cmd_v"].abs() > 0.01]


def t_rel(exp, summ):
    t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
    return exp["stamp"] - t0


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_bev(ax, c1_exp, c1_sum, c2_exp, c2_sum, gp_data, regions):
    """Single BEV panel showing both routes on GP background."""

    # GP background — p stored y-first: contourf(xs, ys, p) is correct
    if gp_data is not None:
        xs, ys, p = gp_data
        cf = ax.contourf(xs, ys, p, levels=np.linspace(0, 1, 16),
                         cmap="RdYlGn", alpha=0.45, zorder=0)
        plt.colorbar(cf, ax=ax, fraction=0.03, pad=0.02, label="GP P_vis")

    # Driveable region boundaries (thin outline only, no fill — GP colour is enough)
    for r in regions:
        x, y = float(r["xmin"]), float(r["ymin"])
        w, h = float(r["xmax"]) - x, float(r["ymax"]) - y
        if "non_driveable" in str(r.get("type", "")):
            ax.add_patch(Rectangle((x, y), w, h, fc="#f28b82", ec="#d93025",
                                   alpha=0.30, lw=0.6, zorder=1))

    # Rack outlines
    for nm, xmin, xmax, ymin, ymax in RACKS:
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                fill=False, edgecolor="#222", lw=2.0, zorder=2))
        ax.text((xmin+xmax)/2, (ymin+ymax)/2, nm, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#222", zorder=3)

    # Trajectories — time-coloured with LineCollection
    for cond, exp, summ in [("C1", c1_exp, c1_sum), ("C2", c2_exp, c2_sum)]:
        if exp is None or len(exp) == 0:
            continue
        pts = exp[["truth_x", "truth_y", "stamp"]].dropna()
        if len(pts) < 2:
            continue
        xy = pts[["truth_x", "truth_y"]].values
        t  = pts["stamp"].values
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        cmap = "Blues_r" if cond == "C1" else "Reds_r"
        lc = LineCollection(segs, cmap=cmap,
                            norm=plt.Normalize(t.min(), t.max()),
                            lw=2.5, zorder=5, alpha=0.9)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        # crash marker
        if summ.get("crashed"):
            ax.scatter(*xy[-1], s=160, c=C[cond], marker="X", zorder=8,
                       edgecolors="white", linewidths=0.8, label=f"{cond} crash")

    # Start / goal
    for exp in [c1_exp, c2_exp]:
        if exp is not None and len(exp) > 0:
            ax.scatter(exp["truth_x"].iloc[0], exp["truth_y"].iloc[0],
                       s=120, c="#16a34a", marker="o", zorder=9, edgecolors="white", lw=1)
            gx = exp["goal_x"].dropna(); gy = exp["goal_y"].dropna()
            if len(gx):
                ax.scatter(gx.iloc[-1], gy.iloc[-1], s=200, c="#111827",
                           marker="*", zorder=9)
            break

    ax.set_xlim(-1.0, 5.0); ax.set_ylim(-3.5, 5.0)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]", fontsize=9); ax.set_ylabel("y [m]", fontsize=9)
    ax.grid(alpha=0.15)

    # Legend: colour patches
    from matplotlib.patches import Patch
    legend_els = [
        Patch(fc=C["C1"], label="C1 (constant-R)"),
        Patch(fc=C["C2"], label="C2 (GP-visibility)"),
        Patch(fc="#16a34a", label="start"),
        Patch(fc="#111827", label="goal ★"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="upper left",
              framealpha=0.85)


def draw_goal_dist(ax, c1_exp, c1_sum, c2_exp, c2_sum):
    ax.axhline(0.25, color="#9ca3af", lw=1.2, ls="--", label="success radius 0.25m")
    for cond, exp, summ in [("C1", c1_exp, c1_sum), ("C2", c2_exp, c2_sum)]:
        if exp is None or "goal_dist" not in exp.columns:
            continue
        t = t_rel(exp, summ)
        gd = exp["goal_dist"].ffill().bfill()
        ax.plot(t, gd, color=C[cond], lw=2.0, label=f"{cond}")
        if summ.get("first_crash_stamp"):
            crash_t = float(summ["first_crash_stamp"]) - float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
            ax.axvline(crash_t, color=C[cond], lw=1.0, ls=":", alpha=0.8)
    ax.set_ylabel("goal distance [m]", fontsize=9)
    ax.set_xlabel("t after first cmd [s]", fontsize=9)
    ax.set_title("Goal distance", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)


def draw_solve_hist(ax, c1_exp, c2_exp):
    for cond, exp in [("C1", c1_exp), ("C2", c2_exp)]:
        if exp is None or "solve_time_ms" not in exp.columns:
            continue
        st = exp["solve_time_ms"].dropna(); st = st[st > 10]
        # keep only unique values to avoid repeated log rows
        st_unique = st.drop_duplicates()
        ax.hist(st_unique, bins=20, color=C[cond], alpha=0.55, edgecolor="white",
                label=f"{cond}  μ={st_unique.mean():.0f} ms")
    ax.axvline(1000, color="#f97316", lw=1.5, ls="--", label="1 Hz budget")
    ax.set_xlabel("local solve [ms]", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title("Local solve times", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)


def draw_pvis(ax, c1_exp, c1_sum, c2_exp, c2_sum):
    for cond, exp, summ in [("C1", c1_exp, c1_sum), ("C2", c2_exp, c2_sum)]:
        if exp is None or "p_vis_plan" not in exp.columns:
            continue
        t = t_rel(exp, summ)
        ax.plot(t, exp["p_vis_plan"].clip(0, 1), color=C[cond], lw=1.8,
                label=cond)
    ax.axhline(0.2, color="#9ca3af", lw=1.0, ls="--", alpha=0.7, label="low-vis threshold")
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("p_vis_plan", fontsize=9)
    ax.set_xlabel("t after first cmd [s]", fontsize=9)
    ax.set_title("Planner-facing visibility (both conditions)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)


def draw_obstacle(ax, c1_exp, c1_sum, c2_exp, c2_sum):
    ax.axhline(0, color="#dc2626", lw=1.5, ls="--", alpha=0.8, label="forbidden (0 m)")
    for cond, exp, summ in [("C1", c1_exp, c1_sum), ("C2", c2_exp, c2_sum)]:
        if exp is None or "min_obstacle_distance_m" not in exp.columns:
            continue
        t = t_rel(exp, summ)
        ax.plot(t, exp["min_obstacle_distance_m"], color=C[cond], lw=1.8, label=cond)
        if summ.get("first_crash_stamp"):
            crash_t = float(summ["first_crash_stamp"]) - float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
            ax.axvline(crash_t, color=C[cond], lw=1.0, ls=":", alpha=0.8)
    ax.set_ylabel("obstacle clearance [m]", fontsize=9)
    ax.set_xlabel("t after first cmd [s]", fontsize=9)
    ax.set_title("Obstacle clearance", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)


def draw_stats(ax, fig_num, c1_sum, c1_exp, c2_sum, c2_exp):
    ax.axis("off")

    def fmt(summ, exp):
        if not summ:
            return "no data"
        p = summ.get("path_length_m") or 0
        g = summ.get("minimum_goal_distance") or 99
        e = summ.get("elapsed_after_first_cmd_s") or 0
        cr = summ.get("completion_reason", "?")
        obs = summ.get("min_obstacle_distance_m") or 0
        terr = summ.get("mean_truth_state_error_m") or 0
        st = pd.Series();
        if exp is not None and "solve_time_ms" in exp.columns:
            st = exp["solve_time_ms"].dropna(); st = st[st>10].drop_duplicates()
        rv = exp["rollout_valid"].mean() if exp is not None and "rollout_valid" in exp.columns else float("nan")
        return (f"outcome:  {cr}\n"
                f"path:     {p:.2f} m\n"
                f"min goal: {g:.3f} m\n"
                f"elapsed:  {e:.1f} s\n"
                f"min obs:  {obs:.3f} m\n"
                f"truth err:{terr:.3f} m\n"
                f"solve:    {st.mean():.0f} ms (μ)\n"
                f"rollout✓: {rv:.0%}")

    text = (f"F{fig_num} Results\n"
            f"{'─'*32}\n"
            f"C1 (constant-R)\n{fmt(c1_sum, c1_exp)}\n\n"
            f"C2 (GP-visibility)\n{fmt(c2_sum, c2_exp)}")

    ax.text(0.05, 0.97, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="#ccc", alpha=0.9))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig",  type=int, required=True, help="Figure number, e.g. 33")
    ap.add_argument("--log",  type=str, required=True, help="Log dir name under logs/visibility_comparison/")
    ap.add_argument("--task", type=str, default="F31_b1_apron_a3_mid")
    args = ap.parse_args()

    log_root = ROOT / "logs/visibility_comparison" / args.log
    out_dir  = ROOT / f"timing_presentation/figures/F{args.fig}"
    out_dir.mkdir(parents=True, exist_ok=True)

    c1_exp, _, c1_sum = load_run(log_root, args.task, "C1")
    c2_exp, _, c2_sum = load_run(log_root, args.task, "C2")
    gp_data  = load_gp()
    regions  = load_regions()

    # Build figure — 2 rows × 3 cols; BEV spans full left column
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            width_ratios=[1.6, 1, 1],
                            hspace=0.38, wspace=0.30,
                            left=0.05, right=0.97, top=0.91, bottom=0.07)

    ax_bev   = fig.add_subplot(gs[:, 0])   # full left column
    ax_goal  = fig.add_subplot(gs[0, 1])
    ax_solv  = fig.add_subplot(gs[0, 2])
    ax_pvis  = fig.add_subplot(gs[1, 1])
    ax_obs   = fig.add_subplot(gs[1, 2])

    draw_bev(ax_bev, c1_exp, c1_sum, c2_exp, c2_sum, gp_data, regions)

    # Overlay title on BEV
    c1p = c1_sum.get("path_length_m") or 0
    c2p = c2_sum.get("path_length_m") or 0
    c1g = c1_sum.get("minimum_goal_distance") or 99
    c2g = c2_sum.get("minimum_goal_distance") or 99
    ax_bev.set_title(
        f"F{args.fig} — B1 route-choice  |  start (3.3,−1.0,east) → goal (1.0,1.75)\n"
        f"C1: path={c1p:.2f}m, min_goal={c1g:.3f}m   "
        f"C2: path={c2p:.2f}m, min_goal={c2g:.3f}m",
        fontsize=9, loc="left"
    )

    draw_goal_dist(ax_goal, c1_exp, c1_sum, c2_exp, c2_sum)
    draw_solve_hist(ax_solv, c1_exp, c2_exp)
    draw_pvis(ax_pvis, c1_exp, c1_sum, c2_exp, c2_sum)
    draw_obstacle(ax_obs, c1_exp, c1_sum, c2_exp, c2_sum)

    # Stats in title area
    fig.suptitle(
        f"F{args.fig}  |  aws_gp_v5  |  r_vis=2.5  ctrl_wt=0.0  "
        f"local_H=10  v_max=0.60  local_maxiter=30",
        fontsize=9, y=0.975
    )

    png = out_dir / f"F{args.fig}_dashboard.png"
    pdf = out_dir / f"F{args.fig}_dashboard.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)

    # Stats table
    def s(summ, exp, cond):
        if not summ: return f"{cond}: no data"
        st = pd.Series()
        if exp is not None and "solve_time_ms" in exp.columns:
            st = exp["solve_time_ms"].dropna(); st = st[st>10].drop_duplicates()
        return (f"{cond}: {summ.get('completion_reason','?')} "
                f"path={summ.get('path_length_m',0) or 0:.2f}m "
                f"min_goal={summ.get('minimum_goal_distance',99) or 99:.3f}m "
                f"solve_μ={st.mean():.0f}ms")

    (out_dir / f"F{args.fig}_dashboard.md").write_text(
        f"# F{args.fig} Dashboard\n\n"
        f"Log: `{args.log}`\n\n"
        f"{s(c1_sum, c1_exp, 'C1')}\n\n"
        f"{s(c2_sum, c2_exp, 'C2')}\n\n"
        f"Figure: `{png}`\n"
    )


if __name__ == "__main__":
    main()
