#!/usr/bin/env python3
"""Sanity plot for a single campaign run: generated seeds + solved global plan +
extracted waypoints + executed (truth) trajectory over the GP/map."""
import sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    sorted(glob.glob(str(ROOT/"logs/visibility_comparison/_smoke_lanegraph/*/*/*/experiment_*")))[-1])
OUT = ROOT / "logs/paper_figures/smoke_run_sanity.png"
GPZ = ROOT / "paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
CONFIG = ROOT / "scripts/visibility_comparison/aws_f31b1_final_config.yaml"
DRIVE = json.loads(yaml.safe_load(open(CONFIG))["driveable_geometry_json"])["prisms"]


def _dissolve(prisms):
    from shapely.ops import unary_union
    from shapely.geometry import box
    u = unary_union([box(p["xmin"], p["ymin"], p["xmax"], p["ymax"]) for p in prisms])
    return list(u.geoms) if u.geom_type == "MultiPolygon" else [u]


def _patch(poly, **kw):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        cs = list(ring.coords); verts += cs
        codes += [MplPath.MOVETO] + [MplPath.LINETO]*(len(cs)-2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kw)


meta = json.load(open(RUN/"global_plan_meta.json"))
gplan = pd.read_csv(RUN/"global_plan.csv")
wp = pd.read_csv(RUN/"global_waypoints.csv")
exp = pd.read_csv(RUN/"experiment.csv")
gp = np.load(GPZ, allow_pickle=True)
xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"])
         for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]

fig, ax = plt.subplots(figsize=(9.5, 9))
ax.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=0.9, alpha=0.85, zorder=0)
for poly in _dissolve(DRIVE):
    ax.add_patch(_patch(poly, facecolor="none", edgecolor="#39ff14", lw=1.6, alpha=0.8, zorder=1))
for (x0, x1, y0, y1) in racks:
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="#f2cf23", edgecolor="#0b6f8a", lw=0.6, zorder=3))

start = meta["start_xy_yaw"]; goal = meta["goal_xy"]
# generated seeds (dashed)
SEEDC = {"below_main_aisle": "#9467bd", "above_connector": "#ff7f0e", "above_cross_aisle": "#ff7f0e"}
chosen = meta["selected_source"].replace("solver:route:", "")
for rt in meta["route_seeds"]:
    w = np.vstack([[start[0], start[1]], np.array(rt["waypoints"], float)])
    star = "  ★CHOSEN" if rt["name"] == chosen else ""
    ax.plot(w[:, 0], w[:, 1], "o--", color=SEEDC.get(rt["name"], "#555"), lw=1.6, ms=4, alpha=0.85, zorder=5,
            label=f"seed: {rt['name']}{star}")
# solved global plan + waypoints
ax.plot(gplan["x"], gplan["y"], "-", color="#111111", lw=2.6, zorder=7, label="solved global plan")
ax.plot(wp["x"], wp["y"], ".", color="#1f77b4", ms=5, zorder=8, label=f"waypoints (n={len(wp)})")
# executed truth trajectory
if {"truth_x", "truth_y"}.issubset(exp.columns):
    t = exp.dropna(subset=["truth_x", "truth_y"])
    ax.plot(t["truth_x"], t["truth_y"], "-", color="#e8000b", lw=2.2, alpha=0.9, zorder=9, label="executed (truth)")
    ax.scatter([t["truth_x"].iloc[-1]], [t["truth_y"].iloc[-1]], s=150, marker="X", c="#e8000b",
               edgecolor="k", lw=1.5, zorder=11, label="end (collision)")
ax.scatter([start[0]], [start[1]], s=120, c="#16a34a", edgecolor="k", zorder=10, label="start")
ax.scatter([goal[0]], [goal[1]], s=180, c="#e41a1c", marker="*", edgecolor="k", zorder=10, label="goal")

ax.set_title(f"SMOKE: {RUN.parts[-4]} / {RUN.parts[-3]} / {RUN.parts[-2]}\n"
             f"route_seed_mode={meta['route_seed_mode']}  chose [{chosen}]  "
             f"J={meta['total_cost']:.0f} (risk {meta['risk_cost']:.0f}, amb {meta['ambiguity_cost']:.0f})  "
             f"valid={meta['rollout_valid']}", fontsize=10, fontweight="bold")
ax.set_xlim(-5.7, 5.5); ax.set_ylim(-3.7, 5.0); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=145, bbox_inches="tight")
print(f"wrote {OUT}  (run: {RUN})")
