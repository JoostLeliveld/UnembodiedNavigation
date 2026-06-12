#!/usr/bin/env python3
"""Offline plan-sanity across all campaign tasks, EXACTLY as Gazebo will solve them:
locked config, generated lane-graph seeds, 13-prism config geometry, full multistart,
maxiter 120, global_horizon 120. For C1 and C2 per task: chosen route, terminal goal
distance, %-outside the driveable lanes, GP-shadow exposure. Renders a 4-panel map."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in ["scripts/paper_figures", "scripts/visibility_comparison", "src/planning", "src/unav_common"]:
    sys.path.insert(0, str(ROOT / p))
import warnings; warnings.filterwarnings("ignore")
from diag_route_suite import make_planner  # noqa: E402
from unav_common.lane_graph_routes import generate_route_seeds  # noqa: E402
from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy  # noqa: E402

CONFIG = ROOT / "scripts/visibility_comparison/aws_f31b1_final_config.yaml"
GPZ = ROOT / "paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
OUT = ROOT / "logs/paper_figures/offline_plan_sanity.png"
CACHE = ROOT / "logs/paper_figures/offline_plan_sanity.json"
CFG = yaml.safe_load(open(CONFIG))
DJ = CFG["driveable_geometry_json"]
PRISMS = tuple(scene_from_json(DJ).prisms)
DRIVE = json.loads(DJ)["prisms"]
TASKS = list(CFG["tasks"].keys())
tasksy = yaml.safe_load(open(ROOT / "src/experiments/config/tasks.yaml"))["tasks"]["warehouse_aws.world.sdf"]
T = {t["name"]: t for t in tasksy}
gp = np.load(GPZ, allow_pickle=True)
xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]


def rho_path(st):
    out = []
    for x, y in st[:, :2]:
        i = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
        j = int(np.clip(np.searchsorted(ys, y), 0, len(ys) - 1))
        out.append(float(P[j, i]))
    return np.array(out)


def pct_out(st):
    sd = signed_distance_to_union_xy(PRISMS, st[:, :2], keep_in=True)
    return 100.0 * float((sd > 1e-4).mean())


def solve(cond, name, gen):
    sp, pl = make_planner(cond, name, gen, None, maxiter=60, multistart=True)
    pl.optimizer_initial_routes = pl._parse_initial_routes(json.dumps(gen))
    pl.optimizer_multistart = True
    pl.optimizer_multistart_include_direct = False
    pl.optimizer_multistart_lateral_offsets = [0.0]
    pl.optimizer_warm_start = True
    pl.prev_controls_flat = None
    pl._prev_goal_xy = None
    plan = pl.plan(np.asarray(sp.start_xy_yaw, float), np.asarray(sp.S0, float), np.asarray(sp.goal_xy, float))
    st = np.asarray(plan.states, float)
    r = rho_path(st)
    return dict(states=st.tolist(), route=str(plan.selected_source).replace("solver:route:", "").replace("solver:", ""),
                total=float(plan.total_cost), valid=bool(plan.rollout_valid), pct_out=pct_out(st),
                d=float(np.hypot(st[-1, 0] - sp.goal_xy[0], st[-1, 1] - sp.goal_xy[1])),
                mean_rho=float(r.mean()), shadow=float((r < 0.4).mean()),
                start=list(map(float, sp.start_xy_yaw)), goal=list(map(float, sp.goal_xy)))


def main():
    if "--from-cache" in sys.argv and CACHE.exists():
        data = json.loads(CACHE.read_text()); _render(data); return
    data = {}
    for name in TASKS:
        t = T[name]; s = (t["start"]["x"], t["start"]["y"]); g = (t["goal"]["x"], t["goal"]["y"])
        gen = generate_route_seeds(DJ, s, g)
        data[name] = {"seeds": gen, "C1": solve("C1", name, gen), "C2": solve("C2", name, gen)}
        for c in ("C1", "C2"):
            d = data[name][c]
            print(f"{name:28} {c}: {d['route']:18} d={d['d']:.2f} out={d['pct_out']:.1f}% valid={d['valid']} "
                  f"meanρ={d['mean_rho']:.2f} shadow={d['shadow']:.0%}", flush=True)
    CACHE.write_text(json.dumps(data)); _render(data)


def _dissolve(prisms):
    from shapely.ops import unary_union
    from shapely.geometry import box
    u = unary_union([box(p["xmin"], p["ymin"], p["xmax"], p["ymax"]) for p in prisms])
    return list(u.geoms) if u.geom_type == "MultiPolygon" else [u]


def _patch(poly, **kw):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        cs = list(ring.coords); verts += cs
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(cs) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kw)


def _render(data):
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"])
             for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]
    SEEDC = {"below_main_aisle": "#9467bd", "above_connector": "#ff7f0e", "above_cross_aisle": "#ff7f0e"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, name in zip(axes.ravel(), TASKS):
        d = data[name]
        ax.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=0.9, alpha=0.85, zorder=0)
        for poly in _dissolve(DRIVE):
            ax.add_patch(_patch(poly, facecolor="none", edgecolor="#39ff14", lw=1.4, alpha=0.75, zorder=1))
        for (x0, x1, y0, y1) in racks:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#f2cf23", edgecolor="#0b6f8a", lw=0.6, zorder=3))
        start = d["C1"]["start"]; goal = d["C1"]["goal"]
        for rt in d["seeds"]:
            w = np.vstack([[start[0], start[1]], np.array(rt["waypoints"], float)])
            ax.plot(w[:, 0], w[:, 1], "--", color=SEEDC.get(rt["name"], "#555"), lw=1.2, alpha=0.6, zorder=4)
        for cond, colr in (("C1", "#e8000b"), ("C2", "#111111")):
            st = np.asarray(d[cond]["states"], float)
            ax.plot(st[:, 0], st[:, 1], "-", color=colr, lw=2.6, zorder=7,
                    label=f"{cond}: {d[cond]['route']} d={d[cond]['d']:.2f} shadow={d[cond]['shadow']:.0%}")
        ax.scatter([start[0]], [start[1]], s=90, c="#16a34a", edgecolor="k", zorder=10)
        ax.scatter([goal[0]], [goal[1]], s=150, c="#e41a1c", marker="*", edgecolor="k", zorder=10)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlim(-5.7, 5.5); ax.set_ylim(-3.7, 5.0); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("Offline plan sanity — locked config, generated lane-graph seeds, 13-prism geometry "
                 "(exactly as Gazebo solves)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
