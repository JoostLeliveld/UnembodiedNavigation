#!/usr/bin/env python3
"""F28 comprehensive diagnostic dashboard.

9-panel layout:
  Row 1 (BEV):    C1 trajectory map  |  C2 trajectory map  |  C1 vs C2 overlay
  Row 2 (errors): Position/yaw errors | Safety margins      | Local solve timing
  Row 3 (planner):EFE cost breakdown  | Belief covariance   | Command + noise

Each panel uses all available data — handles missing or incomplete C2 gracefully.
Reuse as template for future Fx dashboards by updating RUN_ROOT and OUT_DIR.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import pandas as pd
import yaml

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
FIG_NUM = 28
RUN_ROOT = ROOT / f"logs/visibility_comparison/f{FIG_NUM}_r01_gazebo_smoke_v1/F24_R01_a4_lower_to_a3_mid"
GP_PATH  = ROOT / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
WORLD_PROFILE = ROOT / "src/experiments/config/world_profiles.yaml"
OUT_DIR  = ROOT / f"timing_presentation/figures/F{FIG_NUM}"
WORLD    = "warehouse_aws.world.sdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"C1": "#2563eb", "C2": "#dc2626"}
ALPHA_PLAN = 0.18   # plan sample transparency

# ── data loaders ───────────────────────────────────────────────────────────

def latest_run(condition: str) -> Path | None:
    """Return the most recent experiment dir that has >20 rows."""
    base = RUN_ROOT / condition / "seed1"
    if not base.exists():
        return None
    dirs = sorted(base.glob("experiment_*"))
    for d in reversed(dirs):
        csv = d / "experiment.csv"
        if csv.exists() and sum(1 for _ in open(csv)) > 20:
            return d
    return None


def load_exp(run_dir: Path) -> pd.DataFrame | None:
    if run_dir is None:
        return None
    p = run_dir / "experiment.csv"
    return pd.read_csv(p) if p.exists() else None


def load_plans(run_dir: Path) -> pd.DataFrame | None:
    if run_dir is None:
        return None
    p = run_dir / "plan_samples.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(p)


def load_summary(run_dir: Path) -> dict:
    if run_dir is None:
        return {}
    p = run_dir / "run_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_regions() -> list[dict]:
    data = yaml.safe_load(WORLD_PROFILE.read_text())
    return data["worlds"][WORLD].get("known_2d_regions", [])


def load_gp() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not GP_PATH.exists():
        return None
    d = np.load(GP_PATH, allow_pickle=True)
    if "xs" in d and "ys" in d and "P_conservative_plan_map" in d:
        return d["xs"], d["ys"], d["P_conservative_plan_map"]
    return None

# ── drawing helpers ─────────────────────────────────────────────────────────

def draw_zones(ax, regions: list[dict], alpha_drive=0.14, alpha_forbid=0.22) -> None:
    for r in regions:
        x, y = float(r["xmin"]), float(r["ymin"])
        w = float(r["xmax"]) - x
        h = float(r["ymax"]) - y
        kind = str(r.get("type", ""))
        if kind == "traversable":
            ax.add_patch(Rectangle((x, y), w, h,
                facecolor="#79c779", edgecolor="#149447", alpha=alpha_drive, lw=0.8))
        elif "non_driveable" in kind:
            ax.add_patch(Rectangle((x, y), w, h,
                facecolor="#f28b82", edgecolor="#d93025", alpha=alpha_forbid, lw=0.8))


def draw_gp_background(ax, gp_data, alpha=0.35) -> None:
    if gp_data is None:
        return
    xs, ys, pmap = gp_data
    ax.contourf(xs, ys, pmap, levels=np.linspace(0, 1, 12),
                cmap="RdYlGn", alpha=alpha, zorder=0)


def draw_plan_samples(ax, plans: pd.DataFrame | None, exp: pd.DataFrame | None,
                      color: str) -> None:
    if plans is None or exp is None:
        return
    stamps = sorted(plans["plan_stamp"].unique())
    if len(stamps) == 0:
        return
    t0 = float(exp["stamp"].min())
    t1 = float(exp["stamp"].max())
    for stamp in stamps:
        frac = (float(stamp) - t0) / max(t1 - t0, 1e-6)
        alpha = ALPHA_PLAN + 0.25 * frac   # later plans slightly more visible
        g = plans[plans["plan_stamp"] == stamp].sort_values("point_idx")
        if len(g) > 1:
            ax.plot(g["x"], g["y"], color=color, lw=0.9, alpha=alpha, zorder=2)


def draw_belief_ellipses(ax, exp: pd.DataFrame, color: str,
                          every_n: int = 15) -> None:
    """Draw 2σ belief covariance ellipses at regular time intervals."""
    if exp is None:
        return
    for col in ("planner_cov_x", "planner_cov_y"):
        if col not in exp.columns:
            return
    rows = exp.iloc[::every_n]
    for _, row in rows.iterrows():
        sx = float(row.get("planner_cov_x", 0.01)) ** 0.5
        sy = float(row.get("planner_cov_y", 0.01)) ** 0.5
        if sx <= 0 or sy <= 0 or np.isnan(sx) or np.isnan(sy):
            continue
        e = Ellipse(xy=(row["planner_belief_x"], row["planner_belief_y"]),
                    width=4 * sx, height=4 * sy,   # 2σ radius → 4σ diameter
                    angle=0, edgecolor=color, facecolor="none",
                    alpha=0.4, lw=0.7, zorder=3)
        ax.add_patch(e)


def bev_panel(ax, condition: str, exp: pd.DataFrame | None,
              plans: pd.DataFrame | None, summary: dict,
              regions: list[dict], gp_data, title_prefix: str = "") -> None:
    draw_zones(ax, regions)
    draw_gp_background(ax, gp_data, alpha=0.28)
    color = COLORS[condition]

    if exp is not None and len(exp) > 0:
        draw_plan_samples(ax, plans, exp, color)
        draw_belief_ellipses(ax, exp, color)

        # belief path
        ax.plot(exp["planner_belief_x"], exp["planner_belief_y"].clip(-6, 6),
                color=color, lw=1.2, ls="--", alpha=0.7, label="belief", zorder=4)
        # truth path (colour-coded by time)
        pts = exp[["truth_x", "truth_y", "stamp"]].dropna()
        if len(pts) > 1:
            from matplotlib.collections import LineCollection
            xy = pts[["truth_x", "truth_y"]].values
            t  = pts["stamp"].values
            segs = np.stack([xy[:-1], xy[1:]], axis=1)
            lc = LineCollection(segs, cmap="plasma",
                                norm=plt.Normalize(t.min(), t.max()),
                                lw=2.2, zorder=5)
            lc.set_array(t[:-1])
            ax.add_collection(lc)
            plt.colorbar(lc, ax=ax, label="sim time [s]", fraction=0.04, pad=0.02)

        # start / goal / crash
        ax.scatter(exp["truth_x"].iloc[0], exp["truth_y"].iloc[0],
                   s=90, c="#16a34a", zorder=8, label="start")
        goal_rows = exp[["goal_x", "goal_y"]].dropna()
        if len(goal_rows):
            ax.scatter(goal_rows["goal_x"].iloc[-1], goal_rows["goal_y"].iloc[-1],
                       s=160, c="#111827", marker="*", zorder=8, label="goal")
        if summary.get("crashed"):
            ax.scatter(exp["truth_x"].iloc[-1], exp["truth_y"].iloc[-1],
                       s=140, c="#dc2626", marker="X", zorder=9, label="crash")

        outcome = summary.get("completion_reason", "running")
        path = summary.get("path_length_m", 0)
        d_goal = summary.get("minimum_goal_distance", 0)
        crash_type = summary.get("collision_reason", "")
        title = (f"{title_prefix}{condition} — {outcome}\n"
                 f"path={path:.2f}m  min_d_goal={d_goal:.2f}m  {crash_type}")
    else:
        draw_zones(ax, regions)
        title = f"{title_prefix}{condition} — no data yet"

    ax.set_xlim(-0.5, 4.1)
    ax.set_ylim(-3.2, 5.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2)


def timeseries_errors(ax, exp: pd.DataFrame | None, summary: dict,
                      condition: str) -> None:
    if exp is None or len(exp) == 0:
        ax.text(0.5, 0.5, f"{condition}: no data", transform=ax.transAxes, ha="center")
        return
    color = COLORS[condition]
    t0 = float(summary.get("first_cmd_stamp", exp["stamp"].iloc[0]))
    t = exp["stamp"] - t0

    ax.plot(t, exp["truth_state_error_m"],  color=color,    lw=1.5, label="truth-state err")
    ax.plot(t, exp["truth_belief_error_m"], color=color,    lw=1.0, ls="--", alpha=0.7,
            label="truth-belief err")
    if "yaw_error_truth_state_rad" in exp.columns:
        ax2 = ax.twinx()
        ax2.plot(t, exp["yaw_error_truth_state_rad"].abs(), color="#f97316",
                 lw=1.0, alpha=0.6, label="yaw err (rad)")
        ax2.set_ylabel("yaw err [rad]", fontsize=7, color="#f97316")
        ax2.tick_params(axis="y", labelcolor="#f97316", labelsize=6)
        ax2.legend(fontsize=6, loc="upper right")

    if summary.get("crashed"):
        crash_t = float(summary.get("first_crash_stamp", 0)) - t0
        ax.axvline(crash_t, color="#dc2626", lw=1.0, ls=":", label="crash")

    ax.set_title(f"{condition} — position/yaw error", fontsize=8)
    ax.set_xlabel("t after first cmd [s]", fontsize=7)
    ax.set_ylabel("pos error [m]", fontsize=7)
    ax.legend(fontsize=6); ax.grid(alpha=0.2)


def timeseries_safety(ax, exp_c1: pd.DataFrame | None, exp_c2: pd.DataFrame | None,
                      sum_c1: dict, sum_c2: dict) -> None:
    ax.axhline(0, color="#dc2626", lw=1.2, ls="--", alpha=0.7, label="forbidden boundary")
    for cond, exp, summ in [("C1", exp_c1, sum_c1), ("C2", exp_c2, sum_c2)]:
        if exp is None or len(exp) == 0:
            continue
        color = COLORS[cond]
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t = exp["stamp"] - t0
        ax.plot(t, exp["min_obstacle_distance_m"], color=color, lw=1.5,
                label=f"{cond} obstacle clearance")
        if "min_wall_distance_m" in exp.columns:
            ax.plot(t, exp["min_wall_distance_m"], color=color, lw=0.9, ls=":",
                    alpha=0.6, label=f"{cond} wall clearance")
        if summ.get("crashed"):
            crash_t = float(summ.get("first_crash_stamp", 0)) - t0
            ax.axvline(crash_t, color=color, lw=0.8, ls=":", alpha=0.7)
    ax.set_title("Safety: obstacle & wall clearance (both conditions)", fontsize=8)
    ax.set_xlabel("t after first cmd [s]", fontsize=7)
    ax.set_ylabel("clearance [m]", fontsize=7)
    ax.legend(fontsize=6); ax.grid(alpha=0.2)


def timeseries_solve(ax_hist, ax_ts, exp_c1, exp_c2) -> None:
    """Solve time histogram (left) and timeseries (right)."""
    for cond, exp in [("C1", exp_c1), ("C2", exp_c2)]:
        if exp is None or len(exp) == 0:
            continue
        color = COLORS[cond]
        st = exp["solve_time_ms"].dropna()
        st = st[st > 10]
        if len(st) == 0:
            continue
        ax_hist.hist(st, bins=20, color=color, alpha=0.55, edgecolor="white",
                     label=f"{cond} μ={st.mean():.0f}ms")
        # timeseries
        t0 = exp["stamp"].iloc[0]
        ax_ts.plot(exp["stamp"] - t0, exp["solve_time_ms"].fillna(method="ffill"),
                   color=color, lw=0.9, alpha=0.8, label=f"{cond}")

    ax_hist.axvline(500, color="#f97316", lw=1.2, ls="--", label="4Hz budget (500ms)")
    ax_hist.set_title("Local solve time distribution", fontsize=8)
    ax_hist.set_xlabel("solve [ms]", fontsize=7); ax_hist.set_ylabel("count", fontsize=7)
    ax_hist.legend(fontsize=6); ax_hist.grid(alpha=0.2)

    ax_ts.axhline(500, color="#f97316", lw=1.0, ls="--", alpha=0.7)
    ax_ts.set_title("Local solve time over run", fontsize=8)
    ax_ts.set_xlabel("sim time [s]", fontsize=7); ax_ts.set_ylabel("solve [ms]", fontsize=7)
    ax_ts.legend(fontsize=6); ax_ts.grid(alpha=0.2)


def timeseries_efe(ax, exp: pd.DataFrame | None, summary: dict,
                   condition: str) -> None:
    if exp is None or len(exp) == 0:
        ax.text(0.5, 0.5, f"{condition}: no data", transform=ax.transAxes, ha="center")
        return
    color = COLORS[condition]
    t0 = float(summary.get("first_cmd_stamp", exp["stamp"].iloc[0]))
    t = exp["stamp"] - t0

    def safe_col(name):
        return exp[name].fillna(0) if name in exp.columns else pd.Series(0, index=exp.index)

    risk  = safe_col("efe_risk")
    amb   = safe_col("efe_ambiguity")
    obs   = safe_col("efe_obstacle").clip(0, risk.max() * 3 + 1)  # clip runaway nogo
    ctrl  = safe_col("efe_control")

    ax.stackplot(t, [risk, amb, obs, ctrl],
                 labels=["risk", "ambiguity", "obstacle(clip)", "control"],
                 colors=["#3b82f6", "#8b5cf6", "#ef4444", "#6b7280"],
                 alpha=0.65)
    ax.set_title(f"{condition} — EFE cost decomposition", fontsize=8)
    ax.set_xlabel("t after first cmd [s]", fontsize=7); ax.set_ylabel("EFE cost", fontsize=7)
    ax.legend(fontsize=6, loc="upper right"); ax.grid(alpha=0.2)
    if summary.get("crashed"):
        crash_t = float(summary.get("first_crash_stamp", 0)) - t0
        ax.axvline(crash_t, color="#dc2626", lw=0.9, ls=":")


def timeseries_belief_cov(ax, exp_c1, exp_c2, sum_c1, sum_c2) -> None:
    ax2 = ax.twinx()
    for cond, exp, summ in [("C1", exp_c1, sum_c1), ("C2", exp_c2, sum_c2)]:
        if exp is None or len(exp) == 0:
            continue
        color = COLORS[cond]
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t = exp["stamp"] - t0
        if "planner_cov_x" in exp.columns and "planner_cov_y" in exp.columns:
            trace = exp["planner_cov_x"].fillna(0) + exp["planner_cov_y"].fillna(0)
            ax.plot(t, trace, color=color, lw=1.4, label=f"{cond} cov trace")
        if "p_vis_plan" in exp.columns:
            ax2.plot(t, exp["p_vis_plan"].clip(0, 1), color=color,
                     lw=1.0, ls="--", alpha=0.7, label=f"{cond} p_vis")
    ax.set_title("Belief covariance trace + p_vis_plan", fontsize=8)
    ax.set_ylabel("cov trace [m²]", fontsize=7)
    ax2.set_ylabel("p_vis_plan", fontsize=7); ax2.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc="upper left"); ax2.legend(fontsize=6, loc="upper right")
    ax.grid(alpha=0.2)


def timeseries_cmd(ax, exp_c1, exp_c2, sum_c1, sum_c2) -> None:
    for cond, exp, summ in [("C1", exp_c1, sum_c1), ("C2", exp_c2, sum_c2)]:
        if exp is None or len(exp) == 0:
            continue
        color = COLORS[cond]
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t = exp["stamp"] - t0
        ax.plot(t, exp["cmd_v"].fillna(0), color=color, lw=1.3,
                label=f"{cond} cmd_v")
        if "cmd_noise_v_error" in exp.columns:
            ax.fill_between(t,
                exp["cmd_v"].fillna(0) - exp["cmd_noise_v_error"].abs().fillna(0),
                exp["cmd_v"].fillna(0) + exp["cmd_noise_v_error"].abs().fillna(0),
                color=color, alpha=0.12, label=f"{cond} ±noise")
        if "rollout_valid" in exp.columns:
            ax2 = ax.twinx()
            ax2.fill_between(t, 0, exp["rollout_valid"].fillna(0),
                             color=color, alpha=0.15, label=f"{cond} rollout_valid")
            ax2.set_ylabel("rollout_valid", fontsize=7); ax2.set_ylim(-0.05, 1.2)
    ax.set_title("cmd_v + noise envelope + rollout validity", fontsize=8)
    ax.set_xlabel("t after first cmd [s]", fontsize=7); ax.set_ylabel("cmd_v [m/s]", fontsize=7)
    ax.legend(fontsize=6); ax.grid(alpha=0.2)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    regions = load_regions()
    gp_data = load_gp()

    c1_dir = latest_run("C1")
    c2_dir = latest_run("C2")

    c1_exp  = load_exp(c1_dir);   c2_exp  = load_exp(c2_dir)
    c1_plans= load_plans(c1_dir); c2_plans= load_plans(c2_dir)
    c1_sum  = load_summary(c1_dir); c2_sum = load_summary(c2_dir)

    # ── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 20), facecolor="#fafafa")
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.95, bottom=0.04)

    ax_bev_c1  = fig.add_subplot(gs[0, 0])
    ax_bev_c2  = fig.add_subplot(gs[0, 1])
    ax_bev_ov  = fig.add_subplot(gs[0, 2])   # overlay
    ax_err_c1  = fig.add_subplot(gs[1, 0])
    ax_safety  = fig.add_subplot(gs[1, 1])
    ax_solve_h = fig.add_subplot(gs[1, 2])   # histogram
    ax_efe_c1  = fig.add_subplot(gs[2, 0])
    ax_efe_c2  = fig.add_subplot(gs[2, 1])
    ax_solve_t = fig.add_subplot(gs[2, 2])   # timeseries
    ax_cov     = fig.add_subplot(gs[3, 0])
    ax_cmd     = fig.add_subplot(gs[3, 1])
    ax_err_c2  = fig.add_subplot(gs[3, 2])

    # Row 0 — BEV maps
    bev_panel(ax_bev_c1, "C1", c1_exp, c1_plans, c1_sum, regions, gp_data)
    bev_panel(ax_bev_c2, "C2", c2_exp, c2_plans, c2_sum, regions, gp_data)

    # Overlay: both truth paths on one map
    draw_zones(ax_bev_ov, regions)
    draw_gp_background(ax_bev_ov, gp_data, alpha=0.28)
    for cond, exp, summ in [("C1", c1_exp, c1_sum), ("C2", c2_exp, c2_sum)]:
        if exp is not None and len(exp) > 0:
            ax_bev_ov.plot(exp["truth_x"], exp["truth_y"],
                           color=COLORS[cond], lw=2.0, label=cond, zorder=5)
            ax_bev_ov.scatter(exp["truth_x"].iloc[0], exp["truth_y"].iloc[0],
                              s=70, c=COLORS[cond], marker="o", zorder=8)
            if summ.get("crashed"):
                ax_bev_ov.scatter(exp["truth_x"].iloc[-1], exp["truth_y"].iloc[-1],
                                  s=120, c=COLORS[cond], marker="X", zorder=9)
    ax_bev_ov.set_xlim(-0.5, 4.1); ax_bev_ov.set_ylim(-3.2, 5.5)
    ax_bev_ov.set_aspect("equal", adjustable="box")
    ax_bev_ov.set_title("C1 vs C2 truth paths (route comparison)", fontsize=8)
    ax_bev_ov.legend(fontsize=7); ax_bev_ov.grid(alpha=0.2)

    # Row 1
    timeseries_errors(ax_err_c1, c1_exp, c1_sum, "C1")
    timeseries_safety(ax_safety, c1_exp, c2_exp, c1_sum, c2_sum)
    timeseries_solve(ax_solve_h, ax_solve_t, c1_exp, c2_exp)

    # Row 2
    timeseries_efe(ax_efe_c1, c1_exp, c1_sum, "C1")
    timeseries_efe(ax_efe_c2, c2_exp, c2_sum, "C2")

    # Row 3
    timeseries_belief_cov(ax_cov, c1_exp, c2_exp, c1_sum, c2_sum)
    timeseries_cmd(ax_cmd, c1_exp, c2_exp, c1_sum, c2_sum)
    timeseries_errors(ax_err_c2, c2_exp, c2_sum, "C2")

    # ── summary stats box ─────────────────────────────────────────────────
    def stat_str(summ: dict, exp: pd.DataFrame | None) -> str:
        if not summ:
            return "running / no data"
        st = exp["solve_time_ms"].dropna() if exp is not None else pd.Series()
        st = st[st > 10]
        return (
            f"outcome={summ.get('completion_reason','?')}  "
            f"path={summ.get('path_length_m',0):.2f}m  "
            f"d_goal={summ.get('minimum_goal_distance',0):.2f}m\n"
            f"solve_ms={st.mean():.0f}μ/{st.quantile(0.95):.0f}p95  "
            f"truth_err={summ.get('mean_truth_state_error_m',0):.3f}m  "
            f"min_obs={summ.get('min_obstacle_distance_m',0):.3f}m  "
            f"min_wall={summ.get('min_wall_distance_m',0):.3f}m"
        )

    title = (
        f"F{FIG_NUM} Dashboard — F24-R01 A4-lower→A3-mid  |  "
        f"warehouse_aws  |  local_nogo_safe_dist=0.13  |  global_nogo=0.20\n"
        f"C1: {stat_str(c1_sum, c1_exp)}\n"
        f"C2: {stat_str(c2_sum, c2_exp)}"
    )
    fig.suptitle(title, fontsize=9, fontweight="bold", y=0.985,
                 ha="left", x=0.01, wrap=True)

    png = OUT_DIR / f"F{FIG_NUM}_dashboard.png"
    pdf = OUT_DIR / f"F{FIG_NUM}_dashboard.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
