#!/usr/bin/env python3
"""F88 step-by-step build-up figure: world -> driveable(world) -> driveable(planner)
-> GP reliability -> planner seeds -> C1 outcome -> C2 outcome -> cost buildup.

Uses the COMMITTED aws_f31b1_final_config.yaml values (no goal-prior override) so it
shows the honest current F88 state. Saves logs/paper_figures/F88_stepwise.png.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/visibility_comparison"))
sys.path.insert(0, str(ROOT/"scripts/paper_figures"))
sys.path.insert(0, str(ROOT/"src/planning"))
from efe_offline_lab import build_planner, load_setup
from planning.planners.base_planner import extract_waypoints
from generate_driveable_overlay_sdf import build_overlay, _load_prisms, load_obstacle_footprints

CONFIG = ROOT/"scripts/visibility_comparison/aws_f31b1_final_config.yaml"
WORLD  = ROOT/"src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"
GPZ    = ROOT/"paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
TASK   = "F31_b1_apron_a3_mid"
OUT    = ROOT/"logs/paper_figures/F88_stepwise.png"

XLIM, YLIM = (-5.9, 5.4), (-5.7, 5.1)
CAM = (0.0, -5.5)

# ---- shared geometry ----
prisms = _load_prisms(CONFIG)
fps = load_obstacle_footprints(WORLD)
_model, summ = build_overlay(prisms, fps)
ux0, ux1, uy0, uy1 = summ["outer_bbox"]
nogo = summ["nogo_regions"]
gp = np.load(GPZ, allow_pickle=True)
xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
ROUTES = json.loads(open(CONFIG).read().split("optimizer_initial_routes_json: '")[1].split("'")[0])

def solve(cond):
    s = load_setup(CONFIG, condition=cond, seed=0, task_override=TASK)
    cfg = dict(s.config); cfg["horizon"] = int(cfg.get("global_horizon", 120))
    cfg["optimizer_maxiter"] = 800; cfg["optimizer_maxfun"] = 4800
    cfg["optimizer_multistart"] = True; cfg["optimizer_warm_start"] = False
    cfg["optimizer_multistart_include_direct"] = False; cfg["optimizer_multistart_lateral_offsets"] = ""
    pl = build_planner(cfg, planner_kind=s.planner_kind, camera_params=s.camera_params,
        visibility_artifact_path=str(s.gp_path), visibility_geometry_json=s.geometry_json,
        collision_geometry_json=s.geometry_json, driveable_geometry_json=s.driveable_geometry_json,
        seed=0, visibility_target_height_m=s.gp["target_height"])
    plan = pl.plan(np.asarray(s.start_xy_yaw, float), np.asarray(s.S0, float), np.asarray(s.goal_xy, float))
    st = np.asarray(plan.states, float)
    wp = np.asarray(extract_waypoints(st, spacing_m=float(cfg.get("waypoint_spacing_m", 0.12)), include_goal=True), float)
    costs = {k: float(getattr(plan, k+"_cost", 0.0)) for k in ["risk","ambiguity","obstacle","goal_progress","total"]}
    d = float(np.hypot(st[-1,0]-s.goal_xy[0], st[-1,1]-s.goal_xy[1]))
    route = "lower-sweep" if st[:,1].min() < -1.5 else ("connector/mid" if np.any((st[:,0]>1.5)&(st[:,0]<2.6)&(st[:,1]>1.25)&(st[:,1]<2.2)) else "other")
    return s, st, wp, costs, d, route

print("solving C1..."); s1, st1, wp1, c1, d1, r1 = solve("C1")
print("solving C2..."); s2, st2, wp2, c2, d2, r2 = solve("C2")
start, goal = s1.start_xy_yaw, s1.goal_xy

def base(ax, title):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.tick_params(labelsize=7)
def marks(ax, sg=True):
    ax.scatter([CAM[0]],[CAM[1]], marker="v", s=70, c="magenta", ec="k", zorder=20, label="camera")
    if sg:
        ax.scatter([start[0]],[start[1]], s=90, c="#2ca02c", ec="k", zorder=20, label="start")
        ax.scatter([goal[0]],[goal[1]], s=200, marker="*", c="#ffd51a", ec="k", zorder=20, label="goal")
def obstacles(ax):
    for nm, xmin, xmax, ymin, ymax in fps:
        c = "#9aa0a6" if "wall" in nm else "#6f6f6f"
        ax.add_patch(Rectangle((xmin,ymin), xmax-xmin, ymax-ymin,
                     facecolor=c, edgecolor="#333", lw=0.4, alpha=0.9, zorder=5))
def driveable_planner(ax):
    for p in prisms:
        ax.add_patch(Rectangle((p["xmin"],p["ymin"]), p["xmax"]-p["xmin"], p["ymax"]-p["ymin"],
                     facecolor="#cfe2ff", edgecolor="none", alpha=0.7, zorder=1))
    ax.add_patch(Rectangle((ux0,uy0), ux1-ux0, uy1-uy0, fill=False, ec="#0a3fcc", lw=1.6, ls=(0,(6,3)), zorder=4))
    for (a,b,c,d) in nogo:
        ax.add_patch(Rectangle((a,c), b-a, d-c, fill=False, ec="#0a8a2a", lw=1.1, ls=(0,(4,2)), zorder=4))
def gpfield(ax):
    return ax.pcolormesh(xs, ys, P, cmap="viridis", vmin=0, vmax=1, shading="auto", zorder=0)

fig, axes = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)

# Step 1 — world
ax = axes[0,0]; base(ax, "1. World geometry (SDF collision: racks/walls/crate stack)")
obstacles(ax); marks(ax); ax.legend(loc="lower left", fontsize=6)

# Step 2 — driveable per world (physical free floor = workspace minus obstacles)
ax = axes[0,1]; base(ax, "2. Driveable per WORLD (physical free floor between obstacles)")
ax.add_patch(Rectangle((XLIM[0],YLIM[0]), XLIM[1]-XLIM[0], YLIM[1]-YLIM[0], facecolor="#eaf4ea", zorder=0))
obstacles(ax); marks(ax, sg=False)
ax.text(0.02,0.02,"green = physically free; grey = obstacle\n(planner does NOT use all of this)", transform=ax.transAxes, fontsize=6.5, va="bottom")

# Step 3 — driveable per planner (keep-in prisms + no-go)
ax = axes[0,2]; base(ax, "3. Driveable per PLANNER (keep-in prism union + no-go)")
driveable_planner(ax); obstacles(ax); marks(ax, sg=False)
ax.text(0.02,0.02,"blue fill = aisles+connectors (keep_in)\ngreen dashed = no-go columns", transform=ax.transAxes, fontsize=6.5, va="bottom")

# Step 4 — GP reliability field (C2 input)
ax = axes[1,0]; im = gpfield(ax); base(ax, "4. GP reliability rho (v7) — the C2 visibility input")
driveable_planner(ax); marks(ax)
fig.colorbar(im, ax=ax, fraction=0.046, label="rho")

# Step 5 — planner route seeds
ax = axes[1,1]; gpfield(ax); base(ax, "5. Planner multistart route SEEDS (condition-neutral)")
driveable_planner(ax)
for rt, col in zip(ROUTES, ["#ff7f0e","#e377c2"]):
    w = np.array(rt["waypoints"], float)
    w = np.vstack([[start[0],start[1]], w])
    ax.plot(w[:,0], w[:,1], "o--", color=col, lw=1.8, ms=5, zorder=10, label=rt["name"])
marks(ax); ax.legend(loc="lower left", fontsize=6.5)

# Step 6 — C1 outcome
ax = axes[1,2]; gpfield(ax); base(ax, f"6. C1 outcome (constant-R)  d={d1:.2f}m  route={r1}")
driveable_planner(ax)
ax.plot(st1[:,0],st1[:,1], "-", color="#d62728", lw=2.2, zorder=9, label="C1 global plan")
ax.plot(wp1[:,0],wp1[:,1], "o", color="#d62728", ms=3.5, mec="k", mew=0.3, zorder=11)
marks(ax); ax.legend(loc="lower left", fontsize=6.5)

# Step 7 — C2 outcome
ax = axes[2,0]; gpfield(ax); base(ax, f"7. C2 outcome (visibility-aware)  d={d2:.2f}m  route={r2}")
driveable_planner(ax)
ax.plot(st2[:,0],st2[:,1], "-", color="#1f77b4", lw=2.2, zorder=9, label="C2 global plan")
ax.plot(wp2[:,0],wp2[:,1], "o", color="#1f77b4", ms=3.5, mec="k", mew=0.3, zorder=11)
marks(ax); ax.legend(loc="lower left", fontsize=6.5)

# Step 8 — cost buildup (decomposition)
ax = axes[2,1]; ax.set_title("8. Cost buildup (EFE term decomposition)", fontsize=10, fontweight="bold")
terms = ["risk","ambiguity","obstacle","goal_progress"]
x = np.arange(len(terms)); w = 0.38
ax.bar(x-w/2, [c1[t] for t in terms], w, color="#d62728", label=f"C1 (total {c1['total']:.0f})")
ax.bar(x+w/2, [c2[t] for t in terms], w, color="#1f77b4", label=f"C2 (total {c2['total']:.0f})")
ax.set_xticks(x); ax.set_xticklabels(terms, fontsize=8, rotation=15); ax.set_ylabel("cost"); ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)

# Step 9 — both routes overlaid + summary
ax = axes[2,2]; gpfield(ax); base(ax, "9. C1 vs C2 routes overlaid")
driveable_planner(ax)
ax.plot(st1[:,0],st1[:,1], "-", color="#d62728", lw=2.2, zorder=9, label=f"C1 ({r1})")
ax.plot(st2[:,0],st2[:,1], "-", color="#1f77b4", lw=2.2, zorder=9, label=f"C2 ({r2})")
marks(ax); ax.legend(loc="lower left", fontsize=6.5)

fig.suptitle("F88 step-by-step — warehouse_aws / F31_b1_apron_a3_mid / aws_gp_v7b / committed config (goal_prior 50->12)",
             fontsize=13, fontweight="bold")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
print(f"C1: d={d1:.2f} route={r1} costs={c1}")
print(f"C2: d={d2:.2f} route={r2} costs={c2}")
