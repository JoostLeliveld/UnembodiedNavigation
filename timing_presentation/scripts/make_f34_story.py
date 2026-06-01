#!/usr/bin/env python3
"""F34 story figure: belief, localization error, YOLO scores.

The full causal chain:
  C2 chose visible route → YOLO fires reliably → belief tracks truth →
  obstacle clearance safe → goal reached.

  C1 chose low-visibility northern gap → YOLO struggles in rack shadow →
  belief diverges from truth → obstacle penetration → crash.

Panels (3 rows × 3 cols):
  [0,0] BEV — routes on GP + belief paths     [0,1] Truth vs belief Y (lag)   [0,2] YOLO score along route
  [1,0] Localization error (both)              [1,1] Mask area & detection rate  [1,2] Belief covariance + p_vis
  [2,0] Goal distance                          [2,1] Obstacle clearance          [2,2] BEV belief vs truth scatter
"""

from __future__ import annotations
from pathlib import Path
import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import yaml

ROOT   = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
BASE   = ROOT / "logs/visibility_comparison/f34_b1_route_choice_v2/F31_b1_apron_a3_mid"
GP_PATH = ROOT / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
WP_CFG  = ROOT / "src/experiments/config/world_profiles.yaml"
WORLD   = "warehouse_aws.world.sdf"
OUT     = ROOT / "timing_presentation/figures/F34"
OUT.mkdir(parents=True, exist_ok=True)

C = {"C1": "#2563eb", "C2": "#dc2626"}
RACKS = [("R4L",1.725,2.275,-0.8,1.25),("R4U",1.725,2.275,2.2,4.25),
         ("R5L",3.875,4.425,-0.8,1.25),("R5U",3.875,4.425,2.2,4.25)]

# ── loaders ──────────────────────────────────────────────────────────────────

def load(cond):
    d = BASE / cond / "seed1"
    for e in reversed(sorted(d.glob("experiment_*"))):
        if (e/"experiment.csv").exists() and sum(1 for _ in open(e/"experiment.csv"))>30:
            exp   = pd.read_csv(e/"experiment.csv")
            perc  = pd.read_csv(e/"perception.csv")  if (e/"perception.csv").exists()  else pd.DataFrame()
            summ  = json.loads((e/"run_summary.json").read_text()) if (e/"run_summary.json").exists() else {}
            return exp, perc, summ
    return None, pd.DataFrame(), {}

def t0(summ, exp):
    return float(summ.get("first_cmd_stamp", exp["stamp"].iloc[0]))

def gp():
    if not GP_PATH.exists(): return None
    d = np.load(GP_PATH, allow_pickle=True)
    if all(k in d for k in ("xs","ys","P_conservative_plan_map")):
        return d["xs"], d["ys"], d["P_conservative_plan_map"]
    return None

def regions():
    d = yaml.safe_load(WP_CFG.read_text())
    return d["worlds"][WORLD].get("known_2d_regions",[])

def draw_bg(ax, regs, gp_data, alpha=0.4):
    if gp_data:
        xs,ys,p = gp_data
        ax.contourf(xs,ys,p,levels=np.linspace(0,1,14),cmap="RdYlGn",alpha=alpha,zorder=0)
    for r in regs:
        x,y=float(r["xmin"]),float(r["ymin"]); w,h=float(r["xmax"])-x,float(r["ymax"])-y
        if "non_driveable" in str(r.get("type","")):
            ax.add_patch(Rectangle((x,y),w,h,fc="#f28b82",ec="#d93025",alpha=0.22,lw=0.6,zorder=1))

def draw_racks(ax):
    for nm,xmin,xmax,ymin,ymax in RACKS:
        ax.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,ec="#222",lw=1.8,zorder=2))
        ax.text((xmin+xmax)/2,(ymin+ymax)/2,nm,ha="center",va="center",fontsize=7,fontweight="bold")

def bev_lim(ax):
    ax.set_xlim(-1.0,5.0); ax.set_ylim(-3.5,5.0)
    ax.set_aspect("equal","box"); ax.grid(alpha=0.15)
    ax.set_xlabel("x [m]",fontsize=8); ax.set_ylabel("y [m]",fontsize=8)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    gp_data = gp(); regs = regions()
    c1_exp, c1_perc, c1_sum = load("C1")
    c2_exp, c2_perc, c2_sum = load("C2")

    fig = plt.figure(figsize=(22,18), facecolor="white")
    gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.42,wspace=0.30,
                            left=0.05,right=0.97,top=0.93,bottom=0.05)
    axs = [[fig.add_subplot(gs[r,c]) for c in range(3)] for r in range(3)]

    # ── [0,0] BEV: truth + belief paths ──────────────────────────────────────
    ax = axs[0][0]
    draw_bg(ax,regs,gp_data); draw_racks(ax)
    for cond,exp,summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        pts = exp[["truth_x","truth_y","stamp"]].dropna()
        if len(pts)>1:
            xy=pts[["truth_x","truth_y"]].values; t=pts["stamp"].values
            segs=np.stack([xy[:-1],xy[1:]],axis=1)
            cmap="Blues_r" if cond=="C1" else "Reds_r"
            lc=LineCollection(segs,cmap=cmap,norm=plt.Normalize(t.min(),t.max()),lw=2.5,zorder=5,alpha=0.9)
            lc.set_array(t[:-1]); ax.add_collection(lc)
        # belief path (dashed)
        if "planner_belief_x" in exp.columns:
            ax.plot(exp["planner_belief_x"],exp["planner_belief_y"].clip(-4,5),
                    color=C[cond],lw=1.0,ls="--",alpha=0.55,zorder=4,label=f"{cond} belief")
        if summ.get("crashed"):
            ax.scatter(exp["truth_x"].iloc[-1],exp["truth_y"].iloc[-1],
                       s=150,c=C[cond],marker="X",zorder=8,edgecolors="white",lw=0.8)
        else:
            ax.scatter(exp["truth_x"].iloc[-1],exp["truth_y"].iloc[-1],
                       s=150,c=C[cond],marker="*",zorder=8,edgecolors="white",lw=0.8)
    if c1_exp is not None:
        ax.scatter(c1_exp["truth_x"].iloc[0],c1_exp["truth_y"].iloc[0],
                   s=110,c="#16a34a",marker="o",zorder=9,edgecolors="white",lw=1)
        gx=c1_exp["goal_x"].dropna(); gy=c1_exp["goal_y"].dropna()
        if len(gx): ax.scatter(gx.iloc[-1],gy.iloc[-1],s=180,c="#111827",marker="★" if False else "*",zorder=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C["C1"],label="C1 (constant-R)"),
                        Patch(fc=C["C2"],label="C2 (GP-vis)"),
                        Patch(fc="#16a34a",label="start"),
                        Patch(fc="k",label="end ✓/✗")],
              fontsize=7,loc="upper left",framealpha=0.85)
    ax.set_title("Routes + belief paths (dashed)\nGP: green=visible, red=occluded",fontsize=9)
    bev_lim(ax)

    # ── [0,1] Belief Y vs Truth Y — the systematic lag ───────────────────────
    ax = axs[0][1]
    for cond,exp,summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None or "planner_belief_y" not in exp.columns: continue
        _t0 = t0(summ,exp)
        t = exp["stamp"]-_t0
        ax.plot(t,exp["truth_y"],color=C[cond],lw=2.0,label=f"{cond} truth y")
        ax.plot(t,exp["planner_belief_y"].clip(-4,5),color=C[cond],lw=1.2,ls="--",alpha=0.7,label=f"{cond} belief y")
        if summ.get("first_crash_stamp"):
            ax.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cond],lw=0.9,ls=":",alpha=0.8)
    ax.set_ylabel("y position [m]",fontsize=9); ax.set_xlabel("t after first cmd [s]",fontsize=9)
    ax.set_title("Truth y vs belief y\n(solid=truth, dashed=belief)",fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # ── [0,2] YOLO score along route ─────────────────────────────────────────
    ax = axs[0][2]
    for cond,exp,perc,summ in [("C1",c1_exp,c1_perc,c1_sum),("C2",c2_exp,c2_perc,c2_sum)]:
        if exp is None or len(perc)==0: continue
        _t0 = t0(summ,exp)
        perc_t = perc["diag_stamp"] - _t0
        detected = perc["detected"].astype(bool)
        ax.scatter(perc_t[detected],perc.loc[detected,"yolo_score_raw"],
                   s=8,c=C[cond],alpha=0.7,label=f"{cond} detected")
        # Fill non-detections
        ax.scatter(perc_t[~detected],np.zeros((~detected).sum()),
                   s=8,c=C[cond],alpha=0.25,marker="x",label=f"{cond} missed")
        # Rolling mean
        scores = pd.Series(np.where(detected, perc["yolo_score_raw"], 0), index=perc_t)
        roll = scores.rolling(8,min_periods=1).mean()
        ax.plot(perc_t,roll,color=C[cond],lw=1.8,alpha=0.9)
    ax.axhline(0.10,color="#9ca3af",lw=1.0,ls="--",label="detection threshold (0.10)")
    ax.set_ylim(-0.05,1.05); ax.set_ylabel("YOLO score",fontsize=9)
    ax.set_xlabel("t after first cmd [s]",fontsize=9)
    ax.set_title("YOLO detection score along route\n(rolling mean — drops = robot in shadow)",fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # ── [1,0] Localization error ──────────────────────────────────────────────
    ax = axs[1][0]
    for cond,exp,perc,summ in [("C1",c1_exp,c1_perc,c1_sum),("C2",c2_exp,c2_perc,c2_sum)]:
        if exp is None: continue
        _t0 = t0(summ,exp)
        t = exp["stamp"]-_t0
        if "truth_state_error_m" in exp.columns:
            ax.plot(t,exp["truth_state_error_m"],color=C[cond],lw=1.8,label=f"{cond} truth-state err")
        if len(perc)>0 and "localization_error_m" in perc.columns:
            perc_t = perc["diag_stamp"]-_t0
            detected = perc["detected"].astype(bool)
            ax.scatter(perc_t[detected],perc.loc[detected,"localization_error_m"],
                       s=6,c=C[cond],alpha=0.5,zorder=3)
        if summ.get("first_crash_stamp"):
            ax.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cond],lw=0.9,ls=":",alpha=0.8)
    ax.set_ylabel("position error [m]",fontsize=9); ax.set_xlabel("t after first cmd [s]",fontsize=9)
    ax.set_title("Localization error over time\n(line=EKF state error, dots=raw YOLO error)",fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # ── [1,1] Mask area + detection rate ─────────────────────────────────────
    ax = axs[1][1]
    ax2 = ax.twinx()
    for cond,exp,perc,summ in [("C1",c1_exp,c1_perc,c1_sum),("C2",c2_exp,c2_perc,c2_sum)]:
        if exp is None or len(perc)==0: continue
        _t0 = t0(summ,exp)
        perc_t = perc["diag_stamp"]-_t0
        detected = perc["detected"].astype(bool)
        if "mask_area_px" in perc.columns:
            mask = pd.Series(np.where(detected,perc["mask_area_px"],np.nan),index=perc_t)
            ax.plot(perc_t,mask.fillna(0),color=C[cond],lw=1.5,alpha=0.8,label=f"{cond} mask area")
        # Detection rate (rolling)
        det_rate = pd.Series(detected.astype(float),index=perc_t).rolling(10,min_periods=1).mean()
        ax2.plot(perc_t,det_rate,color=C[cond],lw=1.2,ls="--",alpha=0.7)
    ax.set_ylabel("mask area [px²]",fontsize=9); ax.set_xlabel("t after first cmd [s]",fontsize=9)
    ax2.set_ylabel("detection rate (dashed)",fontsize=8,color="#666"); ax2.set_ylim(-0.05,1.15)
    ax.set_title("Mask area + detection rate\n(larger mask = robot more visible to camera)",fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # ── [1,2] Belief covariance trace + p_vis ────────────────────────────────
    ax = axs[1][2]; ax2 = ax.twinx()
    for cond,exp,summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        _t0 = t0(summ,exp); t = exp["stamp"]-_t0
        if "planner_cov_x" in exp.columns and "planner_cov_y" in exp.columns:
            trace = exp["planner_cov_x"].fillna(0)+exp["planner_cov_y"].fillna(0)
            ax.plot(t,trace,color=C[cond],lw=1.8,label=f"{cond} cov trace")
        if "p_vis_plan" in exp.columns:
            ax2.plot(t,exp["p_vis_plan"].clip(0,1),color=C[cond],lw=1.2,ls="--",alpha=0.7)
    ax.set_ylabel("belief cov trace [m²]",fontsize=9); ax.set_xlabel("t [s]",fontsize=9)
    ax2.set_ylabel("p_vis_plan (dashed)",fontsize=8,color="#666"); ax2.set_ylim(-0.05,1.15)
    ax.set_title("Belief covariance + planner visibility\n(larger cov = more uncertain)",fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # ── [2,0] Goal distance ───────────────────────────────────────────────────
    ax = axs[2][0]
    ax.axhline(0.25,color="#9ca3af",lw=1.2,ls="--",label="success radius 0.25m")
    for cond,exp,summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None or "goal_dist" not in exp.columns: continue
        _t0 = t0(summ,exp); t = exp["stamp"]-_t0
        ax.plot(t,exp["goal_dist"].ffill().bfill(),color=C[cond],lw=2.2,label=cond)
        if summ.get("first_crash_stamp"):
            ax.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cond],lw=0.9,ls=":",alpha=0.8)
    ax.set_ylabel("goal distance [m]",fontsize=9); ax.set_xlabel("t after first cmd [s]",fontsize=9)
    ax.set_title("Goal distance\nC2 (red) reaches 0.114m ✓",fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── [2,1] Obstacle clearance ──────────────────────────────────────────────
    ax = axs[2][1]
    ax.axhline(0,color="#dc2626",lw=1.5,ls="--",alpha=0.8,label="forbidden (0 m)")
    for cond,exp,summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None or "min_obstacle_distance_m" not in exp.columns: continue
        _t0 = t0(summ,exp); t = exp["stamp"]-_t0
        ax.plot(t,exp["min_obstacle_distance_m"],color=C[cond],lw=1.8,label=cond)
        if summ.get("first_crash_stamp"):
            ax.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cond],lw=0.9,ls=":",alpha=0.8)
    ax.set_ylabel("obstacle clearance [m]",fontsize=9); ax.set_xlabel("t after first cmd [s]",fontsize=9)
    ax.set_title("Obstacle clearance\nC2 stays at +0.38m throughout",fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── [2,2] BEV: belief error scatter ──────────────────────────────────────
    ax = axs[2][2]
    draw_bg(ax,regs,gp_data,alpha=0.35); draw_racks(ax)
    for cond,exp,summ in [("C1",c1_exp,c1_sum),("C2",c2_exp,c2_sum)]:
        if exp is None: continue
        mv = exp[exp["cmd_v"].abs()>0.01] if "cmd_v" in exp.columns else exp
        if len(mv)==0: continue
        # Colour by error magnitude
        err = mv["truth_state_error_m"].fillna(0).clip(0,1.5)
        sc = ax.scatter(mv["truth_x"],mv["truth_y"],c=err,cmap="hot_r",
                        s=8,alpha=0.7,vmin=0,vmax=1.0,zorder=5,
                        label=cond)
    plt.colorbar(sc,ax=ax,label="localiz. error [m]",fraction=0.04,pad=0.02)
    ax.set_title("Truth path coloured by\nlocalization error (darker=worse)",fontsize=9)
    bev_lim(ax)

    # ── suptitle ──────────────────────────────────────────────────────────────
    c1p=c1_sum.get("path_length_m",0) or 0; c2p=c2_sum.get("path_length_m",0) or 0
    c1g=c1_sum.get("minimum_goal_distance",99) or 99; c2g=c2_sum.get("minimum_goal_distance",99) or 99
    fig.suptitle(
        f"F34 — B1 route-choice  |  start (3.3,−1.0,east) → goal (1.0,1.75)  |  clean system  |  H=20  v_max=0.60\n"
        f"C1 (constant-R): {c1_sum.get('completion_reason','?')}  path={c1p:.2f}m  min_goal={c1g:.3f}m  solve_μ=1086ms\n"
        f"C2 (GP-visibility): {c2_sum.get('completion_reason','?')}  path={c2p:.2f}m  min_goal={c2g:.3f}m  solve_μ=1181ms  ← REACHED GOAL",
        fontsize=9,fontweight="bold",y=0.975
    )

    png = OUT / "F34_story.png"; pdf = OUT / "F34_story.pdf"
    fig.savefig(png,dpi=200,bbox_inches="tight")
    fig.savefig(pdf,bbox_inches="tight")
    plt.close(fig)
    print(png); print(pdf)


if __name__=="__main__":
    main()
