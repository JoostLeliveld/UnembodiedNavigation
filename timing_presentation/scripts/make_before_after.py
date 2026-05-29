#!/usr/bin/env python3
"""Before/after: executed paths with no-go OFF vs ON (strong avoidance)."""
from __future__ import annotations
import csv, glob, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

GAZ = Path("/home/joostleliveld/Thesis/timing_presentation/runs/gazebo")
OUT = Path("/home/joostleliveld/Thesis/timing_presentation/figures/F6_avoidance_before_after.png")
START=(3.0,-1.0); GOAL=(1.0,2.0)
plt.rcParams.update({"font.size":13,"axes.titlesize":15,"axes.titleweight":"bold","figure.facecolor":"white"})

def latest(label):
    ds=sorted(glob.glob(f"{GAZ}/{label}/*/experiment.csv"))
    return Path(ds[-1]).parent if ds else None

def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan

def cols(p):
    r=list(csv.DictReader(open(p))); out={}
    for k in r[0]: out[k]=np.array([_f(x.get(k)) for x in r])
    return out

def prisms(rd):
    cg=json.loads((rd/"run_manifest.json").read_text())["collision_geometry_json"]
    return (json.loads(cg) if isinstance(cg,str) else cg).get("prisms",[])

def _trav():
    import yaml
    prof="/home/joostleliveld/Thesis/UnembodiedNavigation/src/experiments/config/world_profiles.yaml"
    w=yaml.safe_load(open(prof))["worlds"]["warehouse_aws.world.sdf"]
    return [r for r in w.get("known_2d_regions",[]) if r.get("type")=="traversable"]

def draw(ax,pr):
    first=True
    for r in _trav():
        ax.add_patch(Rectangle((r["xmin"],r["ymin"]),r["xmax"]-r["xmin"],r["ymax"]-r["ymin"],
                     facecolor="#b8e0b8",edgecolor="#7cc47c",lw=0.5,alpha=0.55,zorder=0,
                     label="driveable region" if first else None)); first=False
    for p in pr:
        if "wall_" in p["name"]: continue
        r4="R4" in p["name"]; crate="low_crate" in p["name"]
        ax.add_patch(Rectangle((p["xmin"],p["ymin"]),p["xmax"]-p["xmin"],p["ymax"]-p["ymin"],
                     facecolor="#e8a0a0" if crate else "#c8c8c8", edgecolor="tab:red" if r4 else "0.5",
                     lw=1.6 if r4 else 0.8, alpha=0.75, zorder=1))
    ax.plot(*START,"o",color="k",ms=13,zorder=6); ax.plot(*GOAL,"*",color="k",ms=22,zorder=6)
    ax.grid(alpha=0.25); ax.set_xlim(-2.0,3.6); ax.set_ylim(-2.0,2.8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

pairs=[("H80_ms1","H80 multistart"),("H200_ms1","H200 multistart")]
fig,axes=plt.subplots(1,len(pairs),figsize=(7.5*len(pairs),8))
pr=prisms(latest("H200_ms1_av"))
for ax,(base,title) in zip(axes,pairs):
    draw(ax,pr)
    for lab,color,tag in [(base,"tab:blue","no-go OFF"),(base+"_av","tab:green","no-go ON (strong)")]:
        rd=latest(lab)
        if rd is None: continue
        c=cols(rd/"experiment.csv"); tx,ty=c.get("truth_x"),c.get("truth_y")
        m=np.isfinite(tx)&np.isfinite(ty)
        sm=json.load(open(rd/"run_summary.json"))
        crashed=sm.get("crashed"); pl=sm.get("path_length_m",0)
        ax.plot(tx[m],ty[m],"-",color=color,lw=3,zorder=4,
                label=f"{tag}: {'CRASH' if crashed else 'no crash'}, {pl:.1f} m")
        mk="X" if crashed else "s"
        ax.plot(tx[m][-1],ty[m][-1],mk,color=color,ms=14,mec="k",zorder=7)
    ax.set_title(title); ax.legend(loc="upper left",fontsize=10,framealpha=0.9)
fig.suptitle("Obstacle avoidance OFF vs ON — executed paths (X=collision, ■=stopped safe)",fontsize=16,fontweight="bold")
fig.tight_layout(rect=(0,0,1,0.96)); OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=140); print("wrote",OUT)
