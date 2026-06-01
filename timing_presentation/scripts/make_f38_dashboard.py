#!/usr/bin/env python3
"""F38 dashboard — route choice + localization error featured prominently.

Same 6-panel layout as route_choice_uncertainty but with belief uncertainty bands
(2σ tube) added to the BEV panels, and an expanded localization error panel.

Run:
    python3 timing_presentation/scripts/make_f38_dashboard.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import Normalize

ROOT   = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
BASE   = ROOT / "logs/visibility_comparison/f38_b1_route_choice_v1/F31_b1_apron_a3_mid"
GP_NPZ = ROOT / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
FIGS   = Path("/home/joostleliveld/Thesis/thesis-report/figures")
FIGS.mkdir(parents=True, exist_ok=True)
OUT_F  = ROOT / "timing_presentation/figures/F38"
OUT_F.mkdir(parents=True, exist_ok=True)
WORLD  = "warehouse_aws.world.sdf"
RACKS  = [(1.725,2.275,-0.8,1.25),(1.725,2.275,2.2,4.25),
          (3.875,4.425,-0.8,1.25),(3.875,4.425,2.2,4.25)]
C      = {"C1":"#2563eb","C2":"#dc2626"}

def load(cond):
    d = BASE / cond / "seed1"
    for e in reversed(sorted(d.glob("experiment_*"))):
        if sum(1 for _ in open(e/"experiment.csv")) > 30:
            exp  = pd.read_csv(e/"experiment.csv")
            perc = pd.read_csv(e/"perception.csv") if (e/"perception.csv").exists() else pd.DataFrame()
            summ = json.loads((e/"run_summary.json").read_text())
            return exp, perc, summ
    return None, pd.DataFrame(), {}

gp  = np.load(GP_NPZ, allow_pickle=True)
xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]

def pvis(x, y):
    return float(P[np.argmin(abs(ys-y)), np.argmin(abs(xs-x))])

c1e, c1p, c1s = load("C1")
c2e, c2p, c2s = load("C2")
T0 = lambda s,e: float(s.get("first_cmd_stamp", e.stamp.iloc[0]))

def scene(ax):
    ax.contourf(xs, ys, P, levels=np.linspace(0,1,14), cmap="RdYlGn", alpha=0.45, zorder=0)
    for xmin,xmax,ymin,ymax in RACKS:
        ax.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,ec="#333",lw=1.8,zorder=2))
    ax.set_xlim(-0.5,4.8); ax.set_ylim(-3.2,5.0)
    ax.set_aspect("equal","box"); ax.grid(alpha=0.12)
    ax.set_xlabel("x [m]",fontsize=9); ax.set_ylabel("y [m]",fontsize=9)

def tube(ax, exp, color):
    bx = exp.planner_belief_x.ffill().values
    by = exp.planner_belief_y.ffill().clip(-4,5).values
    s  = 2*np.sqrt((exp.planner_cov_x.fillna(0)+exp.planner_cov_y.fillna(0)).values)
    step = max(1, len(bx)//80)
    ax.add_collection(PatchCollection(
        [Circle((bx[i],by[i]),s[i]) for i in range(0,len(bx),step) if s[i]>1e-4],
        fc=color, ec="none", alpha=0.18, zorder=3))
    ax.plot(bx, by, color=color, lw=0.9, ls="--", alpha=0.5, zorder=4)

def tpath(ax, exp):
    df = exp[["truth_x","truth_y","truth_state_error_m"]].dropna()
    if len(df) < 2: return
    xy = df[["truth_x","truth_y"]].values
    lc = LineCollection(np.stack([xy[:-1],xy[1:]],1), cmap="plasma",
                        norm=Normalize(0,1.0), lw=2.5, alpha=0.95, zorder=5)
    lc.set_array(df.truth_state_error_m.values[:-1])
    ax.add_collection(lc)

# ── 3 rows × 3 cols ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.30,
                        left=0.06, right=0.97, top=0.93, bottom=0.05)

ax_c1   = fig.add_subplot(gs[0,0])
ax_c2   = fig.add_subplot(gs[0,1])
ax_ov   = fig.add_subplot(gs[0,2])
ax_yerr = fig.add_subplot(gs[1,0])   # NEW: belief Y vs truth Y with 2σ band
ax_xerr = fig.add_subplot(gs[1,1])   # NEW: belief X vs truth X with 2σ band
ax_goal = fig.add_subplot(gs[1,2])
ax_yolo = fig.add_subplot(gs[2,0])
ax_loc  = fig.add_subplot(gs[2,1])   # localisation error vs 2σ
ax_obs  = fig.add_subplot(gs[2,2])

# Row 0: BEVs
for ax, cnd, exp, summ in [(ax_c1,"C1",c1e,c1s),(ax_c2,"C2",c2e,c2s)]:
    scene(ax); tube(ax,exp,C[cnd]); tpath(ax,exp)
    ax.scatter(exp.truth_x.iloc[0],exp.truth_y.iloc[0],s=100,c="#16a34a",marker="o",zorder=9,ec="white")
    gx=exp.goal_x.dropna(); gy=exp.goal_y.dropna()
    if len(gx): ax.scatter(gx.iloc[-1],gy.iloc[-1],s=180,c="#111",marker="*",zorder=9)
    cr = summ.get("completion_reason","?")
    mkr = "X" if summ.get("crashed") else ("o" if cr=="stuck" else "*")
    ax.scatter(exp.truth_x.iloc[-1],exp.truth_y.iloc[-1],s=150,c=C[cnd],marker=mkr,zorder=8,ec="white")
    te = exp.truth_state_error_m.mean()
    ax.set_title(f"{'C1 (constant-$R$)' if cnd=='C1' else 'C2 (GP-visibility)'} — {cr}\n"
                 f"mean loc.\\ err $= {te:.2f}$\\,m",fontsize=9)

sm=plt.cm.ScalarMappable(cmap="plasma",norm=Normalize(0,1)); sm.set_array([])
fig.colorbar(sm,ax=ax_c2,label="localisation error [m]",fraction=0.04,pad=0.03,shrink=0.75)

# Overlay
scene(ax_ov)
for cnd,exp,summ in [("C1",c1e,c1s),("C2",c2e,c2s)]:
    tube(ax_ov,exp,C[cnd])
    mv=exp[exp.cmd_v.abs()>0.01] if "cmd_v" in exp.columns else exp
    if len(mv)<2: continue
    xy=mv[["truth_x","truth_y"]].values
    ax_ov.add_collection(LineCollection(np.stack([xy[:-1],xy[1:]],1),
                          color=C[cnd],lw=2.4,alpha=0.88,zorder=5,label=cnd))
    cr=summ.get("completion_reason","?")
    ax_ov.scatter(*xy[-1],s=150,c=C[cnd],marker="X" if summ.get("crashed") else ("o" if cr=="stuck" else "*"),
                  zorder=8,ec="white",lw=0.8)
ax_ov.scatter(c1e.truth_x.iloc[0],c1e.truth_y.iloc[0],s=100,c="#16a34a",marker="o",zorder=9,ec="white")
gx=c1e.goal_x.dropna(); gy=c1e.goal_y.dropna()
if len(gx): ax_ov.scatter(gx.iloc[-1],gy.iloc[-1],s=180,c="#111",marker="*",zorder=9)
from matplotlib.patches import Patch
ax_ov.legend(handles=[Patch(fc=C["C1"],label="C1"),Patch(fc=C["C2"],label="C2")],fontsize=8,loc="upper left")
ax_ov.set_title("Route overlay (tube = $2\\sigma$ belief)",fontsize=9)
sm2=plt.cm.ScalarMappable(cmap="RdYlGn",norm=Normalize(0,1)); sm2.set_array([])
fig.colorbar(sm2,ax=ax_ov,label=r"$\rho_{\mathrm{plan}}$",fraction=0.04,pad=0.03,shrink=0.75)

# Row 1: Belief Y and X with 2σ bands
for ax, axis, cov_f in [(ax_yerr,"y","planner_cov_y"),(ax_xerr,"x","planner_cov_x")]:
    ax2 = ax.twinx()
    for cnd,exp,summ in [("C1",c1e,c1s),("C2",c2e,c2s)]:
        _t0=T0(summ,exp); t=exp.stamp-_t0
        sig = np.sqrt(exp[cov_f].clip(0).fillna(0))
        bv  = exp[f"planner_belief_{axis}"]
        tv  = exp[f"truth_{axis}"]
        lo, hi = min(bv.min(),tv.min())-0.4, max(bv.max(),tv.max())+0.4
        if "planner_pixel_correction_age_s" in exp.columns:
            age = exp.planner_pixel_correction_age_s.fillna(0)
            ax.fill_between(t,lo,hi,where=age>1.5,color="#fbbf24",alpha=0.10,zorder=0)
        ax.fill_between(t,bv-2*sig,bv+2*sig,color=C[cnd],alpha=0.18,zorder=1)
        ax.plot(t,bv,color=C[cnd],lw=1.3,ls="--",alpha=0.65,label=f"{cnd} belief")
        ax.plot(t,tv,color=C[cnd],lw=1.8,label=f"{cnd} truth")
        if summ.get("first_crash_stamp"):
            ax.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cnd],lw=0.9,ls=":")
        err = (bv-tv).abs().mean()
        # Error magnitude on right axis
        ax2.plot(t,(bv-tv).abs(),color=C[cnd],lw=0.7,alpha=0.4)
    ax2.set_ylabel(f"|belief−truth| {axis} [m]",fontsize=7,color="#555")
    ax2.tick_params(axis="y",labelsize=6,labelcolor="#555")
    ax.set_ylabel(f"{axis} [m]",fontsize=9); ax.set_xlabel("t [s]",fontsize=9)
    ax.set_title(f"Belief {axis} ±2σ (band) vs truth\nYellow = YOLO stale →dead-reckoning",fontsize=9)
    ax.legend(fontsize=6,loc="best"); ax.grid(alpha=0.2)

# Goal distance
ax_goal.axhline(0.25,color="#9ca3af",lw=1.2,ls="--",label="success 0.25m")
for cnd,exp,summ in [("C1",c1e,c1s),("C2",c2e,c2s)]:
    _t0=T0(summ,exp); t=exp.stamp-_t0
    ax_goal.plot(t,exp.goal_dist.ffill().bfill(),color=C[cnd],lw=2.0,label=cnd)
    if summ.get("first_crash_stamp"):
        ax_goal.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cnd],lw=0.9,ls=":")
ax_goal.set_ylabel("goal dist [m]",fontsize=9); ax_goal.set_xlabel("t [s]",fontsize=9)
ax_goal.set_title("Goal distance",fontsize=9); ax_goal.legend(fontsize=8); ax_goal.grid(alpha=0.2)

# Row 2: YOLO
for cnd,exp,perc,summ in [("C1",c1e,c1p,c1s),("C2",c2e,c2p,c2s)]:
    if len(perc)==0: continue
    _t0=T0(summ,exp); pt=perc.diag_stamp-_t0; det=perc.detected.astype(bool)
    sc=pd.Series(np.where(det,perc.yolo_score_raw,np.nan),index=pt)
    ax_yolo.plot(pt,sc.rolling(5,min_periods=1).mean(),color=C[cnd],lw=1.8,label=cnd)
    ax_yolo.scatter(pt[~det],np.zeros((~det).sum()),s=8,c=C[cnd],alpha=0.3,marker="x")
    if summ.get("first_crash_stamp"):
        ax_yolo.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cnd],lw=0.9,ls=":")
ax_yolo.axhline(0.10,color="#9ca3af",lw=1.0,ls="--")
ax_yolo.set_ylim(-0.05,1.05); ax_yolo.set_ylabel("YOLO score",fontsize=9)
ax_yolo.set_xlabel("t [s]",fontsize=9); ax_yolo.set_title("Detection score",fontsize=9)
ax_yolo.legend(fontsize=7); ax_yolo.grid(alpha=0.2)

# Localisation error with 2σ
for cnd,exp,summ in [("C1",c1e,c1s),("C2",c2e,c2s)]:
    _t0=T0(summ,exp); t=exp.stamp-_t0
    sig=np.sqrt((exp.planner_cov_x.fillna(0)+exp.planner_cov_y.fillna(0)))
    ax_loc.fill_between(t,0,2*sig,color=C[cnd],alpha=0.18,label=f"{cnd} 2σ")
    ax_loc.plot(t,exp.truth_state_error_m,color=C[cnd],lw=1.8,label=f"{cnd} error")
    if summ.get("first_crash_stamp"):
        ax_loc.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cnd],lw=0.9,ls=":")
ax_loc.set_ylabel("position error [m]",fontsize=9); ax_loc.set_xlabel("t [s]",fontsize=9)
ax_loc.set_title("Localisation error vs $2\\sigma$ belief\n(truth outside band = overconfident EKF)",fontsize=9)
ax_loc.legend(fontsize=7,ncol=2); ax_loc.grid(alpha=0.2)

# Obstacle clearance
ax_obs.axhline(0,color="#dc2626",lw=1.5,ls="--",alpha=0.8,label="forbidden")
for cnd,exp,summ in [("C1",c1e,c1s),("C2",c2e,c2s)]:
    _t0=T0(summ,exp); t=exp.stamp-_t0
    ax_obs.plot(t,exp.min_obstacle_distance_m,color=C[cnd],lw=1.8,label=cnd)
    if summ.get("first_crash_stamp"):
        ax_obs.axvline(float(summ["first_crash_stamp"])-_t0,color=C[cnd],lw=0.9,ls=":")
ax_obs.set_ylabel("clearance [m]",fontsize=9); ax_obs.set_xlabel("t [s]",fontsize=9)
ax_obs.set_title("Obstacle clearance",fontsize=9); ax_obs.legend(fontsize=7); ax_obs.grid(alpha=0.2)

c1p_=c1s.get("path_length_m",0) or 0; c2p_=c2s.get("path_length_m",0) or 0
c1g_=c1s.get("minimum_goal_distance",99) or 99; c2g_=c2s.get("minimum_goal_distance",99) or 99
fig.suptitle(
    f"F38 — waypoint_radius=0.10 + sequential YOLO velocity correction\n"
    f"C1: {c1s.get('completion_reason','?')} path={c1p_:.2f}m min_goal={c1g_:.3f}m   "
    f"C2: {c2s.get('completion_reason','?')} path={c2p_:.2f}m min_goal={c2g_:.3f}m",
    fontsize=9, fontweight="bold", y=0.975)
fig.tight_layout(rect=[0,0,1,0.965])

# Save to both locations
for out in [OUT_F/"F38_dashboard.png", OUT_F/"F38_dashboard.pdf",
            FIGS/"route_choice_uncertainty.pdf"]:
    fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved F38 dashboard")
