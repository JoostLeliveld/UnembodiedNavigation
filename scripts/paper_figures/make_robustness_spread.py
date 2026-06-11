#!/usr/bin/env python3
"""Robustness spread map: per task, overlay all per-seed truth trajectories for C1 and C2
on the GP reliability map, marking collided runs. Shows route consistency + spread + where
the constant-R baseline crashes vs the visibility-aware route staying observable."""
import sys, json, glob, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

def _dissolve_driveable(prisms):
    """Plotting-only: merge driveable prisms into clean polygons (dissolve internal cell seams)."""
    from shapely.ops import unary_union
    from shapely.geometry import box
    u = unary_union([box(p["xmin"], p["ymin"], p["xmax"], p["ymax"]) for p in prisms])
    return list(u.geoms) if u.geom_type == "MultiPolygon" else [u]

def _polygon_patch(poly, **kw):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        cs = list(ring.coords); verts += cs
        codes += [MplPath.MOVETO] + [MplPath.LINETO]*(len(cs)-2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kw)

ROOT = Path(__file__).resolve().parents[2]
CAMP = ROOT/"logs/visibility_comparison/paper_campaign_rawgp_v1"
GPZ  = ROOT/"logs/visibility_comparison/aws_gp_v7b/yolo_score_raw_gp.npz"
CONFIG = ROOT/"scripts/visibility_comparison/aws_f31b1_final_config.yaml"
OUT  = ROOT/"logs/paper_figures/robustness_spread.png"
TASKS = ["F31_b1_apron_a3_mid","b5_a4_apron_to_a2_mid","b2_a0_west_to_a1_upper","b6_a0_west_to_a1_low_control"]
SHORT = {"F31_b1_apron_a3_mid":"F31_b1","b5_a4_apron_to_a2_mid":"b5","b2_a0_west_to_a1_upper":"b2 (west)","b6_a0_west_to_a1_low_control":"b6 (control)"}

def load_runs(task, cond):
    """Return list of (states Nx2, collided bool) for each seed of (task,cond)."""
    out=[]
    base = CAMP/task/cond
    for seeddir in sorted(glob.glob(str(base/"seed*"))):
        exps = sorted(glob.glob(seeddir+"/experiment_*/experiment.csv"))
        if not exps: continue
        import csv
        rows=list(csv.DictReader(open(exps[-1])))
        tx=[]; ty=[]; collided=False
        for r in rows:
            try: x=float(r["truth_x"]); y=float(r["truth_y"])
            except (KeyError,ValueError): continue
            if not (math.isfinite(x) and math.isfinite(y)): continue
            tx.append(x); ty.append(y)
            if str(r.get("collision_any","0")) not in ("0","0.0","","nan"): collided=True
        # outcome from run_summary
        summ=glob.glob(seeddir+"/experiment_*/run_summary.json")
        if summ:
            try:
                s=json.load(open(sorted(summ)[-1]))
                if str(s.get("completion_reason",""))=="collision": collided=True
            except Exception: pass
        if tx: out.append((np.column_stack([tx,ty]), collided))
    return out

gp=np.load(GPZ,allow_pickle=True); xs,ys,P=gp["xs"],gp["ys"],gp["P_conservative_plan_map"]
extent=(float(xs[0]),float(xs[-1]),float(ys[0]),float(ys[-1]))
racks=[(p["xmin"],p["xmax"],p["ymin"],p["ymax"]) for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms",[])]
import yaml; drive=json.loads(yaml.safe_load(open(CONFIG))["driveable_geometry_json"])["prisms"]

fig,axes=plt.subplots(1,4,figsize=(22,5.6))
for col,task in enumerate(TASKS):
    ax=axes[col]
    ax.imshow(P,origin="lower",extent=extent,cmap="viridis",vmin=0,vmax=0.9,alpha=0.85,aspect="equal",zorder=0)
    # ONE clean dissolved driveable region (no blocky cell outlines)
    for poly in _dissolve_driveable(drive):
        ax.add_patch(_polygon_patch(poly,facecolor="none",edgecolor="#39ff14",lw=1.4,alpha=0.7,zorder=1))
    for (x0,x1,y0,y1) in racks: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor="#f2cf23",edgecolor="#0b6f8a",lw=0.5,zorder=4))
    counts={}
    for cond,colr in (("C1","#ff2d2d"),("C2","#1a1aff")):
        runs=load_runs(task,cond); nclean=sum(1 for _,c in runs if not c); ncoll=sum(1 for _,c in runs if c)
        counts[cond]=(nclean,ncoll,len(runs))
        for st,collided in runs:
            ax.plot(st[:,0],st[:,1],color=colr,lw=2.3,alpha=0.8,solid_capstyle="round",zorder=6)
            if collided:
                ax.scatter([st[-1,0]],[st[-1,1]],marker="x",s=110,color=colr,lw=3.0,zorder=9)
    # start/goal from first run
    r0=load_runs(task,"C1") or load_runs(task,"C2")
    if r0:
        st0=r0[0][0]; ax.scatter([st0[0,0]],[st0[0,1]],s=70,c="#16a34a",edgecolor="k",zorder=10)
    ax.set_title(f"{SHORT[task]}\nC1 {counts['C1'][0]}/{counts['C1'][2]} goal, {counts['C1'][1]} coll | "
                 f"C2 {counts['C2'][0]}/{counts['C2'][2]} goal, {counts['C2'][1]} coll",fontsize=10,fontweight="bold")
    ax.set_xlim(-5.6,5.4); ax.set_ylim(-3.7,5.0); ax.set_xticks([]); ax.set_yticks([])
handles=[Line2D([0],[0],color="#ff2d2d",lw=2,label="C1 constant-R (per seed)"),
         Line2D([0],[0],color="#1a1aff",lw=2,label="C2 visibility-aware (per seed)"),
         Line2D([0],[0],marker="x",color="#333",lw=0,markersize=9,label="collision"),
         Line2D([0],[0],marker="o",color="none",markerfacecolor="#16a34a",markeredgecolor="k",markersize=9,label="start")]
fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=10,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Robustness spread: per-seed truth trajectories (5 seeds/condition) under the locked config (belief-tube, ambiguity=1, aws_gp_v7b)",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0.03,1,0.96]); OUT.parent.mkdir(parents=True,exist_ok=True); fig.savefig(OUT,dpi=150,bbox_inches="tight")
print(f"wrote {OUT}")
