#!/usr/bin/env python3
"""F31_b1 route-split figure under the LOCKED config (belief-tube no-go, ambiguity=1).
Solves C1 and C2 (multistart over the short/long seeds), caches states+costs to JSON,
and plots both chosen paths over the GP rho map. Single task = 2 solves only."""
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
from diag_route_suite import TASKS, make_planner, realized

TASK = "F31_b1_apron_a3_mid"
GPZ = ROOT/"paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
OUT = ROOT/"logs/paper_figures/f31b1_split.png"
CACHE = ROOT/"logs/paper_figures/f31b1_split_cache.json"
CONFIG = ROOT/"scripts/visibility_comparison/aws_f31b1_final_config.yaml"

def solve(cond):
    tc = TASKS[TASK]
    routes = [{"name":"short","waypoints":tc["short"]},{"name":"long","waypoints":tc["long"]}]
    s, pl = make_planner(cond, TASK, routes, None, maxiter=350, multistart=True)  # gfs=None -> config value
    plan = pl.plan(np.asarray(s.start_xy_yaw,float), np.asarray(s.S0,float), np.asarray(s.goal_xy,float))
    st = np.asarray(plan.states,float)
    c = {k: float(getattr(plan,k+"_cost",0.0)) for k in ["risk","ambiguity","obstacle","total"]}
    d = float(np.hypot(st[-1,0]-s.goal_xy[0], st[-1,1]-s.goal_xy[1]))
    route = realized(st, s.start_xy_yaw, tc["short"], tc["long"])
    return dict(states=st.tolist(), costs=c, d=d, route=route,
                start=list(map(float,s.start_xy_yaw)), goal=list(map(float,s.goal_xy)))

data = {}
for cond in ("C1","C2"):
    data[cond] = solve(cond)
    print(f"{cond}: route={data[cond]['route']} d={data[cond]['d']:.2f} costs={data[cond]['costs']}")
CACHE.parent.mkdir(parents=True, exist_ok=True)
CACHE.write_text(json.dumps(data))
print(f"cached {CACHE}")

gp = np.load(GPZ, allow_pickle=True)
xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
extent = (float(xs[0]),float(xs[-1]),float(ys[0]),float(ys[-1]))
geom = json.loads(str(gp["geometry_json"]).strip("[]'"))
racks = [(p["xmin"],p["xmax"],p["ymin"],p["ymax"]) for p in geom.get("prisms",[])]
import yaml
drive = json.loads(yaml.safe_load(open(CONFIG))["driveable_geometry_json"])["prisms"]

fig, ax = plt.subplots(figsize=(8.5,8.0))
im = ax.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=0.9, alpha=0.9, aspect="equal", zorder=0)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"learned reliability $\rho_{plan}$")
for pr in drive:
    ax.add_patch(Rectangle((pr["xmin"],pr["ymin"]),pr["xmax"]-pr["xmin"],pr["ymax"]-pr["ymin"],
                           facecolor="none",edgecolor="#39ff14",lw=0.7,alpha=0.6,zorder=1))
for (x0,x1,y0,y1) in racks:
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor="#f2cf23",edgecolor="#0b6f8a",lw=0.6,zorder=4))
for cond,colr,lab in (("C1","#ff2d2d","C1 constant-R"),("C2","#111111","C2 visibility-aware")):
    st = np.asarray(data[cond]["states"],float)
    ax.plot(st[:,0],st[:,1],color=colr,lw=2.6,zorder=6,
            label=f"{lab}: {data[cond]['route']} (d={data[cond]['d']:.2f}m)")
start=data["C1"]["start"]; goal=data["C1"]["goal"]
ax.scatter([start[0]],[start[1]],s=90,c="#16a34a",edgecolor="k",zorder=8,label="start")
ax.scatter([goal[0]],[goal[1]],s=120,c="#e41a1c",marker="*",edgecolor="k",zorder=8,label="goal")
ax.set_title("F31_b1 route split — locked config (belief-tube no-go, ambiguity=1, aws_gp_v7b)\n"
             "C1 takes the blind shortcut; C2 detours via the observable lower lane",
             fontsize=11, fontweight="bold")
ax.set_xlim(-5.6,5.4); ax.set_ylim(-3.7,5.0)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
