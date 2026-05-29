#!/usr/bin/env python3
"""F32 comprehensive route-choice dashboard.

C1 (constant-R EFE) vs C2 (GP-visibility EFE) on B1 task: start (3.3,-1.0) facing east,
goal (1.0,1.75). Both route seeds visible to optimizer: mid_cross_lane (north gap, P~0.07)
and lower_sweep_lane (south apron, P~0.65).

C1 global solve: 51.6s → 8 waypoints → went north then east, crashed near R5 (1.7cm)
C2 global solve: 89.6s → 10 waypoints → southern sweep, reached within 0.39m of goal

Layout (4×3 = 12 panels):
  Row 0: BEV C1 | BEV C2 | C1 vs C2 overlay + route seeds + GP
  Row 1: p_vis_plan both | goal distance both | EFE obstacle+risk both
  Row 2: obstacle clearance both | solve time both | truth-state error both
  Row 3: belief covariance+p_vis | heading source (YOLO detect quality) | EFE ambiguity (global context)
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import yaml

ROOT   = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
BASE   = ROOT / "logs/visibility_comparison/f32_b1_route_choice_v1/F31_b1_apron_a3_mid"
GP_PATH = ROOT / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
WP_CFG = ROOT / "src/experiments/config/world_profiles.yaml"
OUT    = ROOT / "timing_presentation/figures/F32"
WORLD  = "warehouse_aws.world.sdf"
OUT.mkdir(parents=True, exist_ok=True)

C = {"C1": "#2563eb", "C2": "#dc2626"}
RACKS = [
    ("R4L", 1.725, 2.275, -0.8,  1.25),
    ("R4U", 1.725, 2.275,  2.2,  4.25),
    ("R5L", 3.875, 4.425, -0.8,  1.25),
    ("R5U", 3.875, 4.425,  2.2,  4.25),
]

# ── loaders ──────────────────────────────────────────────────────────────────

def load(cond: str):
    d = BASE / cond / "seed1"
    exps = sorted(d.glob("experiment_*"))
    for e in reversed(exps):
        if (e / "experiment.csv").exists() and sum(1 for _ in open(e/"experiment.csv")) > 30:
            exp   = pd.read_csv(e / "experiment.csv")
            plans = pd.read_csv(e / "plan_samples.csv") if (e/"plan_samples.csv").stat().st_size > 0 else pd.DataFrame()
            summ  = json.loads((e / "run_summary.json").read_text()) if (e/"run_summary.json").exists() else {}
            return exp, plans, summ, e
    return None, None, {}, None


def regions():
    d = yaml.safe_load(WP_CFG.read_text())
    return d["worlds"][WORLD].get("known_2d_regions", [])


def gp():
    if not GP_PATH.exists(): return None
    d = np.load(GP_PATH, allow_pickle=True)
    if all(k in d for k in ("xs","ys","P_conservative_plan_map")):
        return d["xs"], d["ys"], d["P_conservative_plan_map"]
    return None


# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_bg(ax, regs, gp_data, alpha_gp=0.4, alpha_reg=0.15):
    if gp_data:
        xs, ys, p = gp_data
        # p stored y-first: p[yi,xi] → contourf(xs,ys,p) correct
        ax.contourf(xs, ys, p, levels=np.linspace(0,1,14),
                    cmap="RdYlGn", alpha=alpha_gp, zorder=0)
    for r in regs:
        x,y = float(r["xmin"]), float(r["ymin"])
        w,h = float(r["xmax"])-x, float(r["ymax"])-y
        kind = str(r.get("type",""))
        if kind == "traversable":
            ax.add_patch(Rectangle((x,y),w,h,fc="#79c779",ec="#149447",alpha=alpha_reg,lw=0.7))
        elif "non_driveable" in kind:
            ax.add_patch(Rectangle((x,y),w,h,fc="#f28b82",ec="#d93025",alpha=0.20,lw=0.7))


def draw_racks(ax):
    for nm,xmin,xmax,ymin,ymax in RACKS:
        ax.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,
                                fill=False, edgecolor="#333", lw=1.8))
        ax.text((xmin+xmax)/2,(ymin+ymax)/2, nm, ha="center", va="center",
                fontsize=7, fontweight="bold", color="#333")


def draw_route_seeds(ax):
    """Show the two optimizer seed routes as dashed arrows."""
    mid  = [(3.3,1.5),(1.5,1.5),(1.0,1.75)]
    low  = [(3.3,-2.2),(1.2,-2.2),(1.0,1.75)]
    for pts, col, lbl in [(mid,"#7c3aed","mid_cross (P≈0.07)"),
                           (low,"#f97316","lower_sweep (P≈0.65)")]:
        xs2 = [p[0] for p in pts]; ys2 = [p[1] for p in pts]
        ax.plot(xs2, ys2, color=col, lw=1.4, ls="--", alpha=0.75, label=lbl, zorder=3)
        ax.scatter(xs2[1:-1], ys2[1:-1], s=25, c=col, zorder=4)


def draw_traj(ax, exp, summ, color, label):
    if exp is None or len(exp)==0: return
    pts = exp[["truth_x","truth_y","stamp"]].dropna()
    if len(pts) < 2: return
    xy = pts[["truth_x","truth_y"]].values
    t  = pts["stamp"].values
    segs = np.stack([xy[:-1],xy[1:]],axis=1)
    lc = LineCollection(segs, cmap="plasma",
                        norm=plt.Normalize(t.min(),t.max()), lw=2.0, zorder=5)
    lc.set_array(t[:-1]); ax.add_collection(lc)
    ax.scatter(*xy[0], s=90, c=color, marker="o", zorder=7, label="start")
    gx = exp["goal_x"].dropna(); gy = exp["goal_y"].dropna()
    if len(gx): ax.scatter(gx.iloc[-1], gy.iloc[-1], s=160, c="k", marker="*", zorder=7, label="goal")
    if summ.get("crashed"):
        ax.scatter(*xy[-1], s=130, c=color, marker="X", zorder=8, label=f"{label} crash")


def bev_ax(ax, exp, plans, summ, regs, gp_data, cond, show_seeds=False):
    draw_bg(ax, regs, gp_data)
    draw_racks(ax)
    if show_seeds: draw_route_seeds(ax)
    draw_traj(ax, exp, summ, C[cond], cond)
    # annotate global waypoints from early plan stamps
    if plans is not None and len(plans) > 0:
        stamps = sorted(plans["plan_stamp"].unique())
        if len(stamps) >= 2:
            early = plans[plans["plan_stamp"] == stamps[1]].sort_values("point_idx")
            if len(early) > 1:
                ax.plot(early["x"], early["y"], color=C[cond], lw=1.2, ls=":", alpha=0.55,
                        label="1st local plan")
    path = summ.get("path_length_m",0) or 0
    mg   = summ.get("minimum_goal_distance",99) or 99
    cr   = summ.get("completion_reason","?")
    ax.set_title(f"{cond} — {cr}\npath={path:.2f}m  min_goal={mg:.3f}m", fontsize=8)
    ax.set_xlim(-1.0, 5.0); ax.set_ylim(-3.5, 5.0)
    ax.set_aspect("equal","box"); ax.grid(alpha=0.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=6, loc="upper left")


def ts(ax, exp, summ, col, cond, field, label=None, lw=1.5, ls="-", secondary_field=None, sec_label=None, sec_col=None):
    if exp is None or field not in exp.columns: return
    t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
    t  = exp["stamp"] - t0
    vals = exp[field].fillna(method="ffill").fillna(0)
    ax.plot(t, vals, color=col, lw=lw, ls=ls, label=label or f"{cond} {field}")
    if secondary_field and secondary_field in exp.columns:
        ax2 = ax.twinx()
        svals = exp[secondary_field].clip(0,1)
        ax2.plot(t, svals, color=sec_col or col, lw=1.0, ls="--", alpha=0.6, label=sec_label)
        ax2.set_ylim(-0.05,1.15); ax2.set_ylabel(sec_label or secondary_field, fontsize=7)
        ax2.legend(fontsize=6, loc="upper right")
    if summ.get("first_crash_stamp"):
        ax.axvline(float(summ["first_crash_stamp"])-t0, color=col, lw=0.9, ls=":", alpha=0.7)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    regs   = regions()
    gp_data = gp()
    c1_exp, c1_pl, c1_sum, c1_dir = load("C1")
    c2_exp, c2_pl, c2_sum, c2_dir = load("C2")

    fig = plt.figure(figsize=(22, 22), facecolor="#fafafa")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.32,
                            left=0.05, right=0.97, top=0.94, bottom=0.04)

    ax_bev1 = fig.add_subplot(gs[0,0])
    ax_bev2 = fig.add_subplot(gs[0,1])
    ax_ov   = fig.add_subplot(gs[0,2])
    ax_pvis = fig.add_subplot(gs[1,0])
    ax_goal = fig.add_subplot(gs[1,1])
    ax_efe  = fig.add_subplot(gs[1,2])
    ax_obs  = fig.add_subplot(gs[2,0])
    ax_solv = fig.add_subplot(gs[2,1])
    ax_err  = fig.add_subplot(gs[2,2])
    ax_cov  = fig.add_subplot(gs[3,0])
    ax_head = fig.add_subplot(gs[3,1])
    ax_note = fig.add_subplot(gs[3,2])

    # ── Row 0: BEV ────────────────────────────────────────────────────────────
    bev_ax(ax_bev1, c1_exp, c1_pl, c1_sum, regs, gp_data, "C1", show_seeds=True)
    bev_ax(ax_bev2, c2_exp, c2_pl, c2_sum, regs, gp_data, "C2", show_seeds=True)

    # Overlay
    draw_bg(ax_ov, regs, gp_data)
    draw_racks(ax_ov)
    draw_route_seeds(ax_ov)
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        draw_traj(ax_ov, exp, summ, C[cond], cond)
    ax_ov.scatter([3.3],[-1.0],s=120,c="#16a34a",zorder=9,label="start")
    ax_ov.scatter([1.0],[1.75],s=180,c="k",marker="*",zorder=9,label="goal")
    ax_ov.set_xlim(-1.0,5.0); ax_ov.set_ylim(-3.5,5.0)
    ax_ov.set_aspect("equal","box"); ax_ov.grid(alpha=0.2)
    ax_ov.set_title("C1 vs C2 routes + GP + seeds\ngreen=visible, red=occluded", fontsize=8)
    ax_ov.legend(fontsize=6, loc="upper left")

    # ── Row 1: Mechanism ──────────────────────────────────────────────────────
    ax_pvis.set_title("p_vis_plan over time (both conditions)", fontsize=8)
    ax_pvis.axhline(0.2, color="#aaa", lw=0.8, ls="--", alpha=0.5, label="low-vis threshold")
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t  = exp["stamp"] - t0
        if "p_vis_plan" in exp.columns:
            ax_pvis.plot(t, exp["p_vis_plan"].clip(0,1), color=C[cond], lw=1.5, label=cond)
    ax_pvis.set_ylim(-0.05,1.1); ax_pvis.set_ylabel("p_vis_plan"); ax_pvis.set_xlabel("t [s]")
    ax_pvis.legend(fontsize=8); ax_pvis.grid(alpha=0.2)

    ax_goal.set_title("Goal distance over time (C2 longer route reaches closer)", fontsize=8)
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t  = exp["stamp"] - t0
        if "goal_dist" in exp.columns:
            gd = exp["goal_dist"].fillna(method="ffill").fillna(method="bfill")
            ax_goal.plot(t, gd, color=C[cond], lw=1.5, label=cond)
        if summ.get("first_crash_stamp"):
            ax_goal.axvline(float(summ["first_crash_stamp"])-t0, color=C[cond], lw=0.9, ls=":", alpha=0.7)
    ax_goal.axhline(0.25, color="#999", lw=0.8, ls="--", label="success radius")
    ax_goal.set_ylabel("goal dist [m]"); ax_goal.set_xlabel("t [s]")
    ax_goal.legend(fontsize=8); ax_goal.grid(alpha=0.2)

    ax_efe.set_title("EFE risk + obstacle (local planner, both)", fontsize=8)
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t  = exp["stamp"] - t0
        if "efe_risk" in exp.columns:
            ax_efe.plot(t, exp["efe_risk"].fillna(0), color=C[cond], lw=1.2, label=f"{cond} risk")
        if "efe_obstacle" in exp.columns:
            obs = exp["efe_obstacle"].clip(0, exp["efe_risk"].max()*4+1) if "efe_risk" in exp.columns else exp["efe_obstacle"]
            ax_efe.plot(t, obs.fillna(0), color=C[cond], lw=0.9, ls="--", alpha=0.7, label=f"{cond} obstacle")
    ax_efe.set_ylabel("EFE cost"); ax_efe.set_xlabel("t [s]")
    ax_efe.legend(fontsize=6); ax_efe.grid(alpha=0.2)

    # ── Row 2: Safety & timing ────────────────────────────────────────────────
    ax_obs.set_title("Obstacle clearance (0 = crash boundary)", fontsize=8)
    ax_obs.axhline(0, color="#dc2626", lw=1.2, ls="--", alpha=0.7, label="forbidden")
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t  = exp["stamp"] - t0
        if "min_obstacle_distance_m" in exp.columns:
            ax_obs.plot(t, exp["min_obstacle_distance_m"], color=C[cond], lw=1.5, label=cond)
        if summ.get("first_crash_stamp"):
            ax_obs.axvline(float(summ["first_crash_stamp"])-t0, color=C[cond], lw=0.9, ls=":", alpha=0.7)
    ax_obs.set_ylabel("clearance [m]"); ax_obs.set_xlabel("t [s]")
    ax_obs.legend(fontsize=8); ax_obs.grid(alpha=0.2)

    ax_solv.set_title("Local solve time (both; 1 Hz target)", fontsize=8)
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        st = exp["solve_time_ms"].dropna() if "solve_time_ms" in exp.columns else pd.Series()
        st = st[st > 10]
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t  = exp["stamp"] - t0
        ax_solv.hist(st, bins=18, color=C[cond], alpha=0.5, edgecolor="white",
                     label=f"{cond} μ={st.mean():.0f}ms")
    ax_solv.axvline(1000, color="#f97316", lw=1.2, ls="--", label="1 Hz budget")
    ax_solv.set_xlabel("solve [ms]"); ax_solv.set_ylabel("count")
    ax_solv.legend(fontsize=7); ax_solv.grid(alpha=0.2)

    ax_err.set_title("Truth-state position error (both)", fontsize=8)
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None or "truth_state_error_m" not in exp.columns: continue
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t  = exp["stamp"] - t0
        ax_err.plot(t, exp["truth_state_error_m"], color=C[cond], lw=1.4, label=cond)
    ax_err.set_ylabel("pos error [m]"); ax_err.set_xlabel("t [s]")
    ax_err.legend(fontsize=8); ax_err.grid(alpha=0.2)

    # ── Row 3: Belief, heading, note ─────────────────────────────────────────
    ax_cov.set_title("Belief covariance trace + p_vis_plan", fontsize=8)
    ax_cov2 = ax_cov.twinx()
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t  = exp["stamp"] - t0
        if "planner_cov_x" in exp.columns and "planner_cov_y" in exp.columns:
            trace = exp["planner_cov_x"].fillna(0) + exp["planner_cov_y"].fillna(0)
            ax_cov.plot(t, trace, color=C[cond], lw=1.4, label=f"{cond} cov")
        if "p_vis_plan" in exp.columns:
            ax_cov2.plot(t, exp["p_vis_plan"].clip(0,1), color=C[cond], lw=1.0, ls="--", alpha=0.7)
    ax_cov.set_ylabel("cov trace [m²]", fontsize=7)
    ax_cov2.set_ylabel("p_vis (dashed)", fontsize=7); ax_cov2.set_ylim(-0.05,1.15)
    ax_cov.legend(fontsize=7, loc="upper left"); ax_cov.grid(alpha=0.2)

    ax_head.set_title("YOLO detection quality: heading source\n(odom_fallback=detector lost robot)", fontsize=8)
    for cond, exp, summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None or "heading_source" not in exp.columns: continue
        t0 = float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))
        t  = exp["stamp"] - t0
        # 1=good (pixel heading), 0=fallback
        quality = (~exp["heading_source"].str.contains("odom|fallback", na=False)).astype(float)
        # smooth with rolling
        quality_smooth = quality.rolling(5, center=True, min_periods=1).mean()
        ax_head.plot(t, quality_smooth, color=C[cond], lw=1.4, label=cond)
    ax_head.set_ylim(-0.05, 1.15)
    ax_head.axhline(0.5, color="#aaa", lw=0.8, ls="--", alpha=0.6)
    ax_head.set_ylabel("YOLO heading available (1=yes)"); ax_head.set_xlabel("t [s]")
    ax_head.legend(fontsize=8); ax_head.grid(alpha=0.2)

    # Panel: key findings text
    ax_note.axis("off")
    c1p = c1_sum.get("path_length_m",0) or 0
    c2p = c2_sum.get("path_length_m",0) or 0
    c1g = c1_sum.get("minimum_goal_distance",99) or 99
    c2g = c2_sum.get("minimum_goal_distance",99) or 99
    c1s = c1_sum.get("mean_solve_time_ms",0) or 0
    c2s = c2_sum.get("mean_solve_time_ms",0) or 0
    note = (
        "F32 Key Findings\n\n"
        f"C1 (constant-R):\n"
        f"  Global: 51.6s → 8 wpts → NORTH then EAST\n"
        f"  Path: {c1p:.2f}m | closest: {c1g:.3f}m\n"
        f"  Mean local solve: {c1s:.0f}ms\n"
        f"  Outcome: crashed near R5 (1.7cm)\n\n"
        f"C2 (GP visibility-aware):\n"
        f"  Global: 89.6s → 10 wpts → SOUTH sweep\n"
        f"  Path: {c2p:.2f}m (+{(c2p/c1p-1)*100:.0f}% vs C1)\n"
        f"  Closest to goal: {c2g:.3f}m ← near success\n"
        f"  Mean local solve: {c2s:.0f}ms\n"
        f"  Outcome: graze (0.1mm)\n\n"
        "Route difference source:\n"
        "  Global EFE with ambiguity_weight=8\n"
        "  Northern gap: P_vis≈0.07 → high ambiguity\n"
        "  Southern sweep: P_vis≈0.65 → low ambiguity\n"
        "  C2 penalises low-vis gap → longer safe route\n"
        "  C1 ignores visibility → shorter risky route\n\n"
        "Note: local planner has local_use_ambiguity=false\n"
        "Route choice is purely from global EFE solve."
    )
    ax_note.text(0.05, 0.97, note, transform=ax_note.transAxes,
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8))

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        "F32 — B1 Route-Choice: start (3.3,−1.0) facing east → goal (1.0,1.75)  |  "
        "aws_gp_v5  |  local_horizon=10  v_max=0.60\n"
        f"C1: {c1_sum.get('completion_reason','?')} path={c1p:.2f}m min_goal={c1g:.3f}m   "
        f"C2: {c2_sum.get('completion_reason','?')} path={c2p:.2f}m min_goal={c2g:.3f}m",
        fontsize=10, fontweight="bold", y=0.965,
    )
    fig.tight_layout(rect=[0,0,1,0.96])

    png = OUT / "F32_dashboard.png"
    pdf = OUT / "F32_dashboard.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)

    # Write markdown
    md = OUT / "F32_dashboard.md"
    md.write_text("\n".join([
        "# F32 Route-Choice Dashboard",
        "",
        f"Config: `scripts/visibility_comparison/aws_f32_b1_route_choice_config.yaml`",
        f"Task: F31_b1_apron_a3_mid | start (3.3,-1.0,yaw=0) | goal (1.0,1.75)",
        "",
        "## Results",
        "",
        "| condition | global solve | waypoints | outcome | path m | min goal m | mean solve ms |",
        "|---|---:|---:|---|---:|---:|---:|",
        f"| C1 | 51.6 s | 8 | {c1_sum.get('completion_reason','?')} | {c1p:.2f} | {c1g:.3f} | {c1s:.0f} |",
        f"| C2 | 89.6 s | 10 | {c2_sum.get('completion_reason','?')} | {c2p:.2f} | {c2g:.3f} | {c2s:.0f} |",
        "",
        "## Route choice",
        "",
        "C1 (constant-R): global planner chose northern route, went north then east toward R5,",
        "crashed with 1.7 cm obstacle penetration at only 1.86 m from goal.",
        "",
        "C2 (GP-visibility): global planner chose southern sweep through the open visible apron",
        f"(P_vis≈0.65). Path is {(c2p/c1p-1)*100:.0f}% longer than C1 but reached within 0.39 m of goal.",
        "",
        "## Mechanism",
        "",
        "Route choice is entirely from the global EFE solve (local_use_ambiguity=false).",
        "Northern gap at y≈1.5: P_vis≈0.07 → high ambiguity cost with ambiguity_weight=8.",
        "Southern sweep (y<-0.8): P_vis≈0.65 → low ambiguity cost.",
        "C2 avoids the occluded gap; C1 ignores it and chooses the shorter infeasible route.",
        "",
        "## Caveats",
        "",
        "Single seed. Both conditions crashed (C1 by 1.7cm, C2 by 0.1mm).",
        "AWS world is exploratory — compact benchmark (warehouse_occ_light) is paper core.",
        "Result is consistent with the intended mechanism but requires seeded campaign before paper claim.",
        "",
        f"Figure: `{png}`",
        f"PDF: `{pdf}`",
        "",
    ]))
    print(md)


if __name__ == "__main__":
    main()
