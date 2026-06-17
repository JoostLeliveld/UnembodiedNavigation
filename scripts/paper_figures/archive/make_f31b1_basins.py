#!/usr/bin/env python3
"""F31_b1 initial-basin decomposition (locked config).
4 single-seed solves (C1/C2 x short/long basin); for each, the converged planned path +
EFE cost decomposition. Shows WHY C2 rejects the blind (short) basin: its belief-tube/nogo
cost explodes there, while C1 (uniform cov) finds it cheap. Caches to JSON."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/paper_figures"))
sys.path.insert(0, str(ROOT/"scripts/visibility_comparison"))
sys.path.insert(0, str(ROOT/"src/planning"))
from diag_route_suite import TASKS, make_planner
TASK = "F31_b1_apron_a3_mid"
GPZ = ROOT/"paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
OUT = ROOT/"logs/paper_figures/f31b1_basins_decomposition.png"
CACHE = ROOT/"logs/paper_figures/f31b1_basins_cache.json"
CONFIG = ROOT/"scripts/visibility_comparison/aws_f31b1_final_config.yaml"
TERMS = ["risk","ambiguity","obstacle"]; TERMC={"risk":"#d94b41","ambiguity":"#7b3294","obstacle":"#2c7fb8"}

def solve(cond, seed):
    tc=TASKS[TASK]
    # maxiter=120 = the runtime optimizer budget (config optimizer_maxiter); keeps each seed in
    # its initial basin so the short/long basins stay distinct (full convergence collapses them
    # to each condition's single global optimum). Matches the Gazebo per-seed costs.
    s,pl=make_planner(cond,TASK,[{"name":seed,"waypoints":tc[seed]}],None,maxiter=120,multistart=False)
    plan=pl.plan(np.asarray(s.start_xy_yaw,float),np.asarray(s.S0,float),np.asarray(s.goal_xy,float))
    st=np.asarray(plan.states,float)
    c={k:float(getattr(plan,k+"_cost",0.0)) for k in ["risk","ambiguity","obstacle","total"]}
    return dict(states=st.tolist(),costs=c,start=list(map(float,s.start_xy_yaw)),goal=list(map(float,s.goal_xy)))

data={}
for cond in ("C1","C2"):
    for seed in ("short","long"):
        data[f"{cond}_{seed}"]=solve(cond,seed)
        print(f"{cond}_{seed}: total={data[f'{cond}_{seed}']['costs']['total']:.1f} costs={data[f'{cond}_{seed}']['costs']}")
CACHE.write_text(json.dumps(data))

gp=np.load(GPZ,allow_pickle=True); xs,ys,P=gp["xs"],gp["ys"],gp["P_conservative_plan_map"]
extent=(float(xs[0]),float(xs[-1]),float(ys[0]),float(ys[-1]))
racks=[(p["xmin"],p["xmax"],p["ymin"],p["ymax"]) for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms",[])]
import yaml; drive=json.loads(yaml.safe_load(open(CONFIG))["driveable_geometry_json"])["prisms"]
chosen={cond:min(("short","long"),key=lambda sd:data[f"{cond}_{sd}"]["costs"]["total"]) for cond in ("C1","C2")}

fig=plt.figure(figsize=(13,8))
gs=fig.add_gridspec(2,2,height_ratios=[1.4,1.0])
def mapfmt(ax,title):
    ax.imshow(P,origin="lower",extent=extent,cmap="viridis",vmin=0,vmax=0.9,alpha=0.9,aspect="equal",zorder=0)
    for pr in drive: ax.add_patch(Rectangle((pr["xmin"],pr["ymin"]),pr["xmax"]-pr["xmin"],pr["ymax"]-pr["ymin"],facecolor="none",edgecolor="#39ff14",lw=0.6,alpha=0.55,zorder=1))
    for (x0,x1,y0,y1) in racks: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor="#f2cf23",edgecolor="#0b6f8a",lw=0.6,zorder=4))
    ax.set_title(title,fontsize=11,fontweight="bold"); ax.set_xlim(-5.6,5.4); ax.set_ylim(-3.7,5.0); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
# one path panel per basin
for col,seed in enumerate(("short","long")):
    ax=fig.add_subplot(gs[0,col]); mapfmt(ax,f"'{seed}' basin seed — converged paths")
    for cond,colr in (("C1","#ff2d2d"),("C2","#111111")):
        st=np.asarray(data[f"{cond}_{seed}"]["states"],float)
        star=" ★chosen" if chosen[cond]==seed else ""
        ax.plot(st[:,0],st[:,1],color=colr,lw=2.4,zorder=6,label=f"{cond} (tot {data[f'{cond}_{seed}']['costs']['total']:.0f}){star}")
    st0=data["C1_short"]
    ax.scatter([st0['start'][0]],[st0['start'][1]],s=70,c="#16a34a",edgecolor="k",zorder=8)
    ax.scatter([st0['goal'][0]],[st0['goal'][1]],s=90,c="#e41a1c",marker="*",edgecolor="k",zorder=8)
    ax.legend(loc="upper left",fontsize=8,framealpha=0.85)
# decomposition bars
ax=fig.add_subplot(gs[1,:])
bars=["C1_short","C1_long","C2_short","C2_long"]; x=np.arange(len(bars)); bottoms=np.zeros(len(bars))
for term in TERMS:
    vals=np.array([data[b]["costs"][term] for b in bars]); ax.bar(x,vals,bottom=bottoms,color=TERMC[term],label=term,width=0.62); bottoms+=vals
for i,b in enumerate(bars):
    cond,seed=b.split("_")
    if chosen[cond]==seed: ax.text(x[i],bottoms[i]*1.02,"★ chosen",ha="center",va="bottom",fontsize=10,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["C1 · short(blind)","C1 · long(obs)","C2 · short(blind)","C2 · long(obs)"],fontsize=10)
ax.set_ylabel("EFE cost"); ax.set_title("Per-basin cost decomposition (★ = chosen by multistart)",fontsize=11); ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.3)
fig.suptitle("F31_b1 initial-basin decomposition (locked config: belief-tube no-go, ambiguity=1, aws_gp_v7b)",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT,dpi=150,bbox_inches="tight"); print(f"wrote {OUT}")
