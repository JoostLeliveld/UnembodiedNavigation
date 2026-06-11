#!/usr/bin/env python3
"""Pre-Gazebo offline validation + seed showcase for the robustness campaign.

For each campaign task, two CONDITION-NEUTRAL, geometry-named lane-graph seeds are offered to
BOTH C1 and C2 (no visibility labelling, no hand-tuned waypoints):
  - "via_connector": start aisle -> up to the rack-gap connector band (y~1.6) -> across to the
                     goal aisle -> goal     (the shorter, northern route through the R2..R5 gaps)
  - "via_lower":     start aisle -> down to the lower main aisle (y~-2.4) -> across -> up -> goal
                     (the longer southern corridor route)
(For the west control b6 the only northern lane is the upper cross-aisle, so "via_connector" is
the up-and-over route; R0/R1 is continuous so there is no direct west->A1 shortcut.)

Runs the EXACT Gazebo settings (locked config + optimizer_maxiter=120 = the runtime global
budget) and reports, per task/condition, which seed basin the EFE solve selects + reach, plus
held-seed feasibility (no-go). Produces a seed-showcase figure (raw seeds) + a converged figure.
"""
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
from diag_route_suite import make_planner, realized
from efe_offline_lab import load_setup

GPZ = ROOT/"logs/visibility_comparison/aws_gp_v7b/yolo_score_raw_gp.npz"
CONFIG = ROOT/"scripts/visibility_comparison/aws_f31b1_final_config.yaml"
OUT_SEEDS = ROOT/"logs/paper_figures/pregazebo_seeds.png"
OUT_CONV  = ROOT/"logs/paper_figures/pregazebo_converged.png"
MAXITER = 120   # matches the Gazebo runtime optimizer_maxiter

# Clean systematic lane-graph seeds (start is taken from tasks.yaml; these are the waypoints AFTER start).
SEEDS = {
  "F31_b1_apron_a3_mid":        dict(kind="discriminator",
      via_connector=[[3.1,1.6],[1.0,1.6],[1.0,1.75]],
      via_lower    =[[3.1,-2.4],[1.0,-2.4],[1.0,1.75]]),
  "b5_a4_apron_to_a2_mid":      dict(kind="discriminator",
      via_connector=[[3.1,1.6],[-1.0,1.6],[-1.0,1.75]],
      via_lower    =[[3.1,-2.4],[-1.0,-2.4],[-1.0,1.75]]),
  "b2_a0_west_to_a1_upper":     dict(kind="discriminator",
      via_connector=[[-5.0,4.58],[-3.0,4.58],[-3.0,3.5]],
      via_lower    =[[-5.0,-2.4],[-3.0,-2.4],[-3.0,3.5]]),
  "b6_a0_west_to_a1_low_control": dict(kind="control",
      via_connector=[[-5.0,4.58],[-3.0,4.58],[-3.0,-0.75]],
      via_lower    =[[-5.0,-2.4],[-3.0,-2.4],[-3.0,-0.75]]),
}
ORDER = list(SEEDS.keys())
SHORT = {"F31_b1_apron_a3_mid":"F31_b1","b5_a4_apron_to_a2_mid":"b5","b2_a0_west_to_a1_upper":"b2","b6_a0_west_to_a1_low_control":"b6(ctrl)"}


def held(cond, task, name, wps):
    s, pl = make_planner(cond, task, [{"name":name,"waypoints":wps}], None, maxiter=1, multistart=False)
    ctrl = pl._controls_for_waypoints(np.asarray(s.start_xy_yaw,float), [np.asarray(w,float) for w in wps])
    r = pl.evaluate_rollout_controls(np.asarray(s.start_xy_yaw,float), np.asarray(s.S0,float), np.asarray(s.goal_xy,float), ctrl)
    return float(r["obstacle_cost"]), float(r["mean_p_vis_plan"]), np.asarray(r["states"],float), s.start_xy_yaw, s.goal_xy


def solve(cond, task):
    sd = SEEDS[task]
    routes = [{"name":"via_connector","waypoints":sd["via_connector"]},{"name":"via_lower","waypoints":sd["via_lower"]}]
    s, pl = make_planner(cond, task, routes, None, maxiter=MAXITER, multistart=True)
    plan = pl.plan(np.asarray(s.start_xy_yaw,float), np.asarray(s.S0,float), np.asarray(s.goal_xy,float))
    st = np.asarray(plan.states,float)
    c = {k:float(getattr(plan,k+"_cost",0.0)) for k in ["risk","ambiguity","obstacle","total"]}
    d = float(np.hypot(st[-1,0]-s.goal_xy[0], st[-1,1]-s.goal_xy[1]))
    # classify by nearer seed: via_lower == "long", via_connector == "short"
    basin = realized(st, s.start_xy_yaw, sd["via_connector"], sd["via_lower"])
    sel = "via_lower" if basin=="observable/long" else "via_connector"
    return st, c, d, sel, s.start_xy_yaw, s.goal_xy


# ---- run ----
data = {}
print(f"{'task':10} {'seed feasibility (held nogo / mean_pvis)':52}")
for task in ORDER:
    sd = SEEDS[task]; data[task] = {}
    for nm in ("via_connector","via_lower"):
        nogo, pvis, st, start, goal = held("C2", task, nm, sd[nm])
        data[task][("seed",nm)] = dict(nogo=nogo, pvis=pvis, start=start, goal=goal, wps=sd[nm])
        print(f"  {SHORT[task]:8} {nm:14} nogo={nogo:9.1f}  mean_pvis={pvis:.2f}")

print(f"\n{'task':10} {'kind':13} {'C1 sel':14}{'C1 d':>6}   {'C2 sel':14}{'C2 d':>6}  verdict")
for task in ORDER:
    st1,c1,d1,sel1,start,goal = solve("C1", task)
    st2,c2,d2,sel2,_,_ = solve("C2", task)
    data[task]["C1"]=dict(states=st1.tolist(),sel=sel1,d=d1,costs=c1)
    data[task]["C2"]=dict(states=st2.tolist(),sel=sel2,d=d2,costs=c2)
    kind=SEEDS[task]["kind"]
    if kind=="discriminator":
        verdict = "SPLIT" if (sel1=="via_connector" and sel2=="via_lower") else ("same" if sel1==sel2 else "partial")
    else:
        verdict = "ctrl-ok" if (sel1==sel2=="via_lower") else "ctrl-CHECK"
    print(f"  {SHORT[task]:8} {kind:13} {sel1:14}{d1:6.2f}   {sel2:14}{d2:6.2f}  {verdict}")

# ---- maps ----
gp=np.load(GPZ,allow_pickle=True); xs,ys,P=gp["xs"],gp["ys"],gp["P_conservative_plan_map"]
extent=(float(xs[0]),float(xs[-1]),float(ys[0]),float(ys[-1]))
racks=[(p["xmin"],p["xmax"],p["ymin"],p["ymax"]) for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms",[])]
import yaml; drive=json.loads(yaml.safe_load(open(CONFIG))["driveable_geometry_json"])["prisms"]

def basemap(ax,title):
    ax.imshow(P,origin="lower",extent=extent,cmap="viridis",vmin=0,vmax=0.9,alpha=0.85,aspect="equal",zorder=0)
    for pr in drive: ax.add_patch(Rectangle((pr["xmin"],pr["ymin"]),pr["xmax"]-pr["xmin"],pr["ymax"]-pr["ymin"],facecolor="none",edgecolor="#39ff14",lw=0.5,alpha=0.5,zorder=1))
    for (x0,x1,y0,y1) in racks: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor="#f2cf23",edgecolor="#0b6f8a",lw=0.5,zorder=4))
    ax.set_title(title,fontsize=10,fontweight="bold"); ax.set_xlim(-5.6,5.4); ax.set_ylim(-3.7,5.0); ax.set_xticks([]); ax.set_yticks([])

# seed showcase
fig,axes=plt.subplots(1,4,figsize=(20,5.2))
for col,task in enumerate(ORDER):
    ax=axes[col]; basemap(ax,f"{SHORT[task]} initial seeds")
    start=data[task][("seed","via_connector")]["start"]; goal=data[task][("seed","via_connector")]["goal"]
    for nm,colr in (("via_connector","#ff7f0e"),("via_lower","#1f77b4")):
        wp=[start[:2]]+SEEDS[task][nm]
        wp=np.asarray(wp,float)
        ax.plot(wp[:,0],wp[:,1],"-o",color=colr,lw=2.0,ms=4,zorder=6,label=nm)
    ax.scatter([start[0]],[start[1]],s=70,c="#16a34a",edgecolor="k",zorder=8)
    ax.scatter([goal[0]],[goal[1]],s=90,c="#e41a1c",marker="*",edgecolor="k",zorder=8)
    ax.legend(loc="lower center",fontsize=8,ncol=1,framealpha=0.8)
fig.suptitle("Initial multistart route seeds (condition-neutral lane-graph routes; offered to BOTH C1 and C2)",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT_SEEDS,dpi=140,bbox_inches="tight"); print(f"\nwrote {OUT_SEEDS}")

# converged
fig,axes=plt.subplots(1,4,figsize=(20,5.2))
for col,task in enumerate(ORDER):
    ax=axes[col]; basemap(ax,f"{SHORT[task]} converged (maxiter={MAXITER})")
    start=data[task][("seed","via_connector")]["start"]; goal=data[task][("seed","via_connector")]["goal"]
    for cond,colr in (("C1","#ff2d2d"),("C2","#111111")):
        st=np.asarray(data[task][cond]["states"],float)
        ax.plot(st[:,0],st[:,1],color=colr,lw=2.3,zorder=6,label=f"{cond}: {data[task][cond]['sel']} (d={data[task][cond]['d']:.2f})")
    ax.scatter([start[0]],[start[1]],s=70,c="#16a34a",edgecolor="k",zorder=8)
    ax.scatter([goal[0]],[goal[1]],s=90,c="#e41a1c",marker="*",edgecolor="k",zorder=8)
    ax.legend(loc="lower center",fontsize=8,framealpha=0.8)
fig.suptitle("Offline EFE converged routes under EXACT Gazebo settings (locked config, ambiguity=1, belief-tube, aws_gp_v7b)",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT_CONV,dpi=140,bbox_inches="tight"); print(f"wrote {OUT_CONV}")
