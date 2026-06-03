#!/usr/bin/env python3
"""
Clean route-choice dashboard for any Fx B1 run.

Usage:
    python3 make_dashboard.py --fig 44 --log f44_b1_route_choice_v1

Layout (3×3):
  [BEV overlay — large]  |  [Goal distance]      |  [Solve time hist]
  [BEV (cont.)]          |  [p_vis_plan]          |  [Obstacle clearance]
  [BEV (cont.)]          |  [Yaw error]                                  |

The BEV shows:
  - GP visibility field
  - Rack outlines with labels
  - Global plan trajectory (dashed grey) with waypoint markers
  - C1 truth path (blue, time-coloured)
  - C2 truth path (red, time-coloured)
  - Start marker, goal star
  - Crash markers if applicable
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

def load_run(log_root: Path, task: str, cond: str, seed: int | None = None):
    """Load one run for a condition.

    If seed is given, load that specific seed. Otherwise scan seeds 0-9 and
    pick the one with the longest path_length_m (most interesting run).
    Falls back to most CSV rows if no run_summary exists.
    """
    cond_dir = log_root / task / cond
    if not cond_dir.exists():
        return None, None, {}, None

    # Determine seed search order
    if seed is not None:
        seed_dirs = [cond_dir / f"seed{seed}"]
    else:
        seed_dirs = sorted(cond_dir.glob("seed*"),
                           key=lambda p: int(p.name.replace("seed", "") or 0))

    best = None  # (path_length, csv_rows, exp, d, summ, plan_df)

    for seed_dir in seed_dirs:
        if not seed_dir.exists():
            continue
        exps = sorted(seed_dir.glob("experiment_*"))
        for d in reversed(exps):
            csv = d / "experiment.csv"
            if not csv.exists():
                continue
            row_count = sum(1 for _ in open(csv))
            if row_count <= 30:
                continue
            exp  = pd.read_csv(csv)
            summ = json.loads((d / "run_summary.json").read_text()) if (d/"run_summary.json").exists() else {}
            plan_csv = d / "plan_samples.csv"
            try:
                plan_df = pd.read_csv(plan_csv) if plan_csv.exists() else None
            except Exception:
                plan_df = None
            path_len = summ.get("path_length_m") or 0.0
            # Skip interrupted/invalid if a better run exists — but keep as fallback
            candidate = (path_len, row_count, exp, d, summ, plan_df)
            if best is None or path_len > (best[0] or 0):
                best = candidate
            break  # use most recent experiment_* per seed

    if best is None:
        return None, None, {}, None
    _, _, exp, d, summ, plan_df = best
    return exp, d, summ, plan_df


def extract_global_plan(plan_df):
    """Return (x, y) arrays for the global plan — first plan_stamp entry (H=80 → 81 points)."""
    if plan_df is None or plan_df.empty:
        return None, None
    first_stamp = plan_df["plan_stamp"].min()
    gp = plan_df[plan_df["plan_stamp"] == first_stamp].sort_values("point_idx")
    if len(gp) < 2:
        return None, None
    return gp["x"].values, gp["y"].values


def waypoints_from_global_plan(xs, ys, spacing_m=0.60):
    """Subsample the global plan trajectory at ~spacing_m intervals."""
    if xs is None or len(xs) < 2:
        return None, None
    wp_x, wp_y = [xs[0]], [ys[0]]
    dist_acc = 0.0
    for i in range(1, len(xs)):
        dist_acc += np.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
        if dist_acc >= spacing_m:
            wp_x.append(xs[i]); wp_y.append(ys[i])
            dist_acc = 0.0
    wp_x.append(xs[-1]); wp_y.append(ys[-1])
    return np.array(wp_x), np.array(wp_y)


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

def draw_bev(ax, c1_exp, c1_sum, c2_exp, c2_sum, gp_data, regions,
             c1_plan_df=None, c2_plan_df=None):
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

    # Global plan trajectory overlay (dashed grey = "what the planner intended")
    for plan_df, plan_color in [(c1_plan_df, "#1d4ed8"), (c2_plan_df, "#991b1b")]:
        gx, gy = extract_global_plan(plan_df)
        if gx is not None:
            ax.plot(gx, gy, color=plan_color, lw=1.2, ls="--", alpha=0.5, zorder=4)
            wx, wy = waypoints_from_global_plan(gx, gy, spacing_m=0.60)
            if wx is not None:
                ax.scatter(wx, wy, s=30, c=plan_color, marker="o",
                           edgecolors="white", linewidths=0.5, alpha=0.6, zorder=4)

    ax.set_xlim(-1.0, 5.0); ax.set_ylim(-3.5, 5.0)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]", fontsize=9); ax.set_ylabel("y [m]", fontsize=9)
    ax.grid(alpha=0.15)

    # Legend: colour patches
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_els = [
        Patch(fc=C["C1"], label="C1 (constant-R)"),
        Patch(fc=C["C2"], label="C2 (GP-visibility)"),
        Line2D([0], [0], color="#1d4ed8", lw=1.2, ls="--", alpha=0.7, label="C1 global plan"),
        Line2D([0], [0], color="#991b1b", lw=1.2, ls="--", alpha=0.7, label="C2 global plan"),
        Patch(fc="#16a34a", label="start"),
        Patch(fc="#111827", label="goal ★"),
    ]
    ax.legend(handles=legend_els, fontsize=7, loc="upper left",
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


def draw_pvis_truth(ax, c1_exp, c1_sum, c2_exp, c2_sum):
    """GP p_vis sampled at the robot's actual (truth) position over time.

    This is the direct mechanism panel: shows how visible the robot is to the
    camera at each moment. C1 (north rack route) drops to p_vis≈0.003; C2
    (visible apron) stays above 0.25.
    """
    gp_data = load_gp()
    if gp_data is None:
        ax.text(0.5, 0.5, "GP not found", transform=ax.transAxes, ha="center")
        return
    xs_gp, ys_gp, p_map = gp_data
    try:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (xs_gp, ys_gp), p_map.T, bounds_error=False, fill_value=None)
    except Exception:
        ax.text(0.5, 0.5, "scipy unavailable", transform=ax.transAxes, ha="center")
        return

    for cond, exp, summ in [("C1", c1_exp, c1_sum), ("C2", c2_exp, c2_sum)]:
        if exp is None or "truth_x" not in exp.columns:
            continue
        valid = exp[["truth_x", "truth_y"]].dropna()
        if len(valid) < 2:
            continue
        pvis = interp(valid.values)
        t_all = t_rel(exp, summ)
        t_valid = t_all[valid.index]
        ax.plot(t_valid, pvis, color=C[cond], lw=1.8, label=cond, alpha=0.85)

    ax.axhline(0.2, color="#9ca3af", lw=1.0, ls="--", alpha=0.7, label="low-vis 0.2")
    ax.axhline(0.5, color="#6b7280", lw=0.8, ls=":", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("GP p_vis at truth pos", fontsize=9)
    ax.set_xlabel("t after first cmd [s]", fontsize=9)
    ax.set_title("Visibility at actual robot position", fontsize=10)
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


def draw_yaw_error(ax, c1_exp, c1_sum, c2_exp, c2_sum):
    col = "yaw_error_truth_belief_rad"
    ax.axhline(0, color="#6b7280", lw=0.8, ls="-", alpha=0.5)
    ax.axhline(0.26, color="#9ca3af", lw=1.0, ls="--", alpha=0.7, label="±15°")
    ax.axhline(-0.26, color="#9ca3af", lw=1.0, ls="--", alpha=0.7)
    for cond, exp, summ in [("C1", c1_exp, c1_sum), ("C2", c2_exp, c2_sum)]:
        if exp is None or col not in exp.columns:
            continue
        t = t_rel(exp, summ)
        valid = exp[col].notna()
        ax.plot(t[valid], exp.loc[valid, col], color=C[cond], lw=1.8, label=cond, alpha=0.85)
    ax.set_ylabel("heading error (truth−belief) rad", fontsize=9)
    ax.set_xlabel("t after first cmd [s]", fontsize=9)
    ax.set_title("Yaw error (belief vs truth)", fontsize=10)
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
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed index to display (default: best-path-length seed)")
    args = ap.parse_args()

    log_root = ROOT / "logs/visibility_comparison" / args.log
    out_dir  = ROOT / f"timing_presentation/figures/F{args.fig}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_summary = "config not found"
    config_candidates = sorted((ROOT / "scripts/visibility_comparison").glob(f"aws_f{args.fig}_*.yaml"))
    if config_candidates:
        try:
            cfg = yaml.safe_load(config_candidates[0].read_text()) or {}
            config_summary = (
                f"{config_candidates[0].name}  |  "
                f"Hg={cfg.get('global_horizon', cfg.get('horizon', '?'))}  "
                f"Hl={cfg.get('local_horizon', '?')}  "
                f"v_max={cfg.get('v_max', '?')}  "
                f"local_iter={cfg.get('local_optimizer_maxiter', '?')}  "
                f"local_safe={cfg.get('local_nogo_safe_distance', '?')}"
            )
        except Exception as exc:
            config_summary = f"{config_candidates[0].name} unreadable: {exc}"

    c1_exp, _, c1_sum, c1_plan = load_run(log_root, args.task, "C1", seed=args.seed)
    c2_exp, _, c2_sum, c2_plan = load_run(log_root, args.task, "C2", seed=args.seed)
    gp_data  = load_gp()
    regions  = load_regions()

    # Build figure — 3 rows × 3 cols; BEV spans full left column
    fig = plt.figure(figsize=(18, 13), facecolor="white")
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            width_ratios=[1.6, 1, 1],
                            hspace=0.42, wspace=0.30,
                            left=0.05, right=0.97, top=0.92, bottom=0.06)

    ax_bev   = fig.add_subplot(gs[:, 0])   # full left column
    ax_goal  = fig.add_subplot(gs[0, 1])
    ax_solv  = fig.add_subplot(gs[0, 2])
    ax_terr  = fig.add_subplot(gs[1, 1])
    ax_obs   = fig.add_subplot(gs[1, 2])
    ax_yaw   = fig.add_subplot(gs[2, 1:])  # yaw error spans bottom two cols

    draw_bev(ax_bev, c1_exp, c1_sum, c2_exp, c2_sum, gp_data, regions,
             c1_plan_df=c1_plan, c2_plan_df=c2_plan)

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
    draw_pvis_truth(ax_terr, c1_exp, c1_sum, c2_exp, c2_sum)
    draw_obstacle(ax_obs, c1_exp, c1_sum, c2_exp, c2_sum)
    draw_yaw_error(ax_yaw, c1_exp, c1_sum, c2_exp, c2_sum)

    # Stats in title area
    fig.suptitle(
        f"F{args.fig}  |  aws_gp_v5  |  {config_summary}",
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

    def runtime_s(exp, cond):
        if exp is None:
            return f"{cond}: no runtime ledger"

        def finite(col):
            if col not in exp.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(exp[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        def p95(col):
            values = finite(col)
            return float(values.quantile(0.95)) if len(values) else float("nan")

        cmd_age = p95("cmd_age_s")
        raw_cmd_age = p95("cmd_raw_age_s")
        plan_age = p95("active_plan_age_s")
        nis = p95("pixel_corr_nis")
        odom_noisy = finite("odom_noisy_available")
        odom_noisy_frac = float((odom_noisy >= 0.5).mean()) if len(odom_noisy) else float("nan")
        accepted = finite("pixel_corr_accepted")
        accept_frac = float((accepted >= 0.5).mean()) if len(accepted) else float("nan")
        reject_reason = "not_logged"
        if "pixel_corr_reject_reason" in exp.columns:
            reasons = exp["pixel_corr_reject_reason"].dropna().astype(str)
            reasons = reasons[reasons != "accepted"]
            reject_reason = reasons.value_counts().idxmax() if len(reasons) else "none"
        exec_idx = finite("exec_control_index")
        exec_idx_max = float(exec_idx.max()) if len(exec_idx) else float("nan")
        return (
            f"{cond}: cmd_age_p95={cmd_age:.3f}s raw_cmd_age_p95={raw_cmd_age:.3f}s "
            f"plan_age_p95={plan_age:.3f}s pixel_accept={accept_frac:.0%} "
            f"nis_p95={nis:.1f} top_reject={reject_reason} "
            f"odom_noisy={odom_noisy_frac:.0%} exec_idx_max={exec_idx_max:.0f}"
        )

    (out_dir / f"F{args.fig}_dashboard.md").write_text(
        f"# F{args.fig} Dashboard\n\n"
        f"Log: `{args.log}`\n\n"
        f"Config: `{config_summary}`\n\n"
        f"{s(c1_sum, c1_exp, 'C1')}\n\n"
        f"{s(c2_sum, c2_exp, 'C2')}\n\n"
        "## Runtime Ledger\n\n"
        f"{runtime_s(c1_exp, 'C1')}\n\n"
        f"{runtime_s(c2_exp, 'C2')}\n\n"
        f"Figure: `{png}`\n"
    )


if __name__ == "__main__":
    main()


# ── standalone diagnostic: belief vs truth + detection events ────────────────
def make_belief_truth_diag(exp_csv: str, perc_csv: str, out_png: str, title: str = ""):
    """Plot belief xy vs truth xy, correction events, and detection events."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    exp  = pd.read_csv(exp_csv)
    perc = pd.read_csv(perc_csv)

    # align to first cmd
    fc = exp["stamp"].min()
    for col in ["cmd_v", "cmd_w"]:
        if col in exp.columns:
            first_nonzero = exp.index[exp[col].abs() > 0.01]
            if len(first_nonzero):
                fc = exp.loc[first_nonzero[0], "stamp"]
                break
    t = exp["stamp"] - fc
    perc_t = perc["log_stamp"] - fc if "log_stamp" in perc.columns else None

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="white")
    fig.suptitle(title, fontsize=10)

    # ── Panel 1: x over time ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, exp["truth_x"], color="#16a34a", lw=1.8, label="truth x")
    ax.plot(t, exp["planner_belief_x"], color="#dc2626", lw=1.2, alpha=0.8, label="belief x")
    # correction events
    if "planner_pixel_correction_stamp" in exp.columns:
        corr_stamps = exp["planner_pixel_correction_stamp"].dropna().unique()
        for cs in corr_stamps:
            ax.axvline(cs - fc, color="#f97316", lw=0.4, alpha=0.35)
    ax.set_ylabel("x [m]"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.2)
    ax.set_title("x position: truth (green) vs belief (red) | orange=correction events")

    # ── Panel 2: y over time ──────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, exp["truth_y"], color="#16a34a", lw=1.8, label="truth y")
    ax.plot(t, exp["planner_belief_y"], color="#dc2626", lw=1.2, alpha=0.8, label="belief y")
    if "planner_pixel_correction_stamp" in exp.columns:
        corr_stamps = exp["planner_pixel_correction_stamp"].dropna().unique()
        for cs in corr_stamps:
            ax.axvline(cs - fc, color="#f97316", lw=0.4, alpha=0.35)
    ax.set_ylabel("y [m]"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.2)
    ax.set_title("y position: truth (green) vs belief (red) | orange=correction events")

    # ── Panel 3: position error + detection bar ───────────────────────────────
    ax = axes[2]
    pos_err = np.hypot(exp["truth_x"] - exp["planner_belief_x"],
                       exp["truth_y"] - exp["planner_belief_y"])
    ax.plot(t, pos_err, color="#2563eb", lw=1.5, label="truth-belief dist")
    ax.axhline(0.25, color="#9ca3af", lw=1, ls="--", alpha=0.7, label="0.25 m")
    # detection events (from perception.csv)
    if perc_t is not None and "detected" in perc.columns:
        det = perc[perc["detected"].astype(bool)]
        nodet = perc[~perc["detected"].astype(bool)]
        det_t_vals = (det["log_stamp"] - fc).values if "log_stamp" in det.columns else []
        nodet_t_vals = (nodet["log_stamp"] - fc).values if "log_stamp" in nodet.columns else []
        ymax = float(pos_err.max() or 1.0)
        for tv in det_t_vals:
            ax.axvline(tv, color="#16a34a", lw=0.5, alpha=0.3)
        for tv in nodet_t_vals:
            ax.axvline(tv, color="#dc2626", lw=0.8, alpha=0.6)
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0],color="#2563eb",lw=1.5,label="belief-truth dist"),
            Line2D([0],[0],color="#16a34a",lw=0.8,label="detection (green)"),
            Line2D([0],[0],color="#dc2626",lw=0.8,label="no-detection (red)"),
        ], fontsize=8, loc="upper left")
    ax.set_ylabel("position error [m]")
    ax.set_xlabel("t after first cmd [s]")
    ax.set_title("Localisation error + YOLO detection events")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png
