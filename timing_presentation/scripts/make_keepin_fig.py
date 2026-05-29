#!/usr/bin/env python3
"""Keep-in result: executed paths stay in driveable lanes; residual grazes are belief drift."""
from __future__ import annotations
import csv, glob, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, yaml

GAZ=Path("/home/joostleliveld/Thesis/timing_presentation/runs/gazebo")
OUT=Path("/home/joostleliveld/Thesis/timing_presentation/figures/F7_keepin_driveable.png")
PROF="/home/joostleliveld/Thesis/UnembodiedNavigation/src/experiments/config/world_profiles.yaml"
START=(3.0,-1.0); GOAL=(1.0,2.0)
plt.rcParams.update({"font.size":13,"axes.titlesize":15,"axes.titleweight":"bold","figure.facecolor":"white"})

def _f(v):
    try:return float(v)
    except:return np.nan
def latest(label):
    ds=sorted(glob.glob(f"{GAZ}/{label}/*/experiment.csv")); return Path(ds[-1]).parent if ds else None
def cols(p):
    r=list(csv.DictReader(open(p))); return {k:np.array([_f(x.get(k)) for x in r]) for k in r[0]}
def prisms(rd):
    cg=json.loads((rd/"run_manifest.json").read_text())["collision_geometry_json"]
    return (json.loads(cg) if isinstance(cg,str) else cg).get("prisms",[])
def trav():
    w=yaml.safe_load(open(PROF))["worlds"]["warehouse_aws.world.sdf"]
    return [r for r in w.get("known_2d_regions",[]) if r.get("type")=="traversable"]

fig,ax=plt.subplots(figsize=(11,9))
first=True
for r in trav():
    ax.add_patch(Rectangle((r["xmin"],r["ymin"]),r["xmax"]-r["xmin"],r["ymax"]-r["ymin"],
                 facecolor="#b8e0b8",edgecolor="#7cc47c",lw=0.5,alpha=0.55,zorder=0,
                 label="driveable region" if first else None)); first=False
rd0=latest("H200_ms1_ki")
for p in prisms(rd0):
    if "wall_" in p["name"]: continue
    r4="R4" in p["name"]; crate="low_crate" in p["name"]
    ax.add_patch(Rectangle((p["xmin"],p["ymin"]),p["xmax"]-p["xmin"],p["ymax"]-p["ymin"],
                 facecolor="#e8a0a0" if crate else "#c8c8c8",edgecolor="tab:red" if r4 else "0.5",
                 lw=1.6 if r4 else 0.8,alpha=0.7,zorder=1))
colors={"H80_ms1_ki":"tab:purple","H200_ms0_ki":"tab:blue","H200_ms1_ki":"tab:orange"}
for lab,c in colors.items():
    rd=latest(lab)
    if not rd: continue
    cc=cols(rd/"experiment.csv"); tx,cy=cc.get("truth_x"),cc.get("truth_y")
    m=np.isfinite(tx)&np.isfinite(cy)
    sm=json.load(open(rd/"run_summary.json"))
    be=sm.get("mean_truth_belief_error_m",float("nan"))
    ax.plot(tx[m],cy[m],"-",color=c,lw=2.5,zorder=4,label=f"{lab}: belief_err={be:.2f} m")
    ax.plot(tx[m][-1],cy[m][-1],"X",color=c,ms=15,mec="k",zorder=7)
ax.plot(*START,"o",color="k",ms=13,zorder=6,label="start"); ax.plot(*GOAL,"*",color="k",ms=22,zorder=6,label="goal")
ax.set_xlim(0.3,4.6); ax.set_ylim(-2.0,2.8); ax.grid(alpha=0.25)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Keep-in: plan stays in driveable lanes; X = ~1 cm graze where belief drifts ~0.1 m")
ax.legend(loc="lower left",fontsize=9,framealpha=0.92)
fig.tight_layout(); OUT.parent.mkdir(parents=True,exist_ok=True); fig.savefig(OUT,dpi=140)
print("wrote",OUT)
