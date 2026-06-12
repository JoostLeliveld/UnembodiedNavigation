#!/usr/bin/env python3
"""Plot ONLY the map-derived lane-graph route seeds (no solving)."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in ["src/unav_common"]:
    sys.path.insert(0, str(ROOT / p))
from unav_common.lane_graph_routes import generate_route_seeds  # noqa: E402

CONFIG = ROOT / "scripts/visibility_comparison/aws_f31b1_final_config.yaml"
GPZ = ROOT / "paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
OUT = ROOT / "logs/paper_figures/generated_seeds.png"
CFG = yaml.safe_load(open(CONFIG))
DJ = CFG["driveable_geometry_json"]
DRIVE = json.loads(DJ)["prisms"]
TASKS = list(CFG["tasks"].keys())


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


tasksy = yaml.safe_load(open(ROOT / "src/experiments/config/tasks.yaml"))["tasks"]["warehouse_aws.world.sdf"]
T = {t["name"]: t for t in tasksy}
gp = np.load(GPZ, allow_pickle=True)
xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"])
         for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]
COL = {"below_main_aisle": "#9467bd", "above_connector": "#ff7f0e", "above_cross_aisle": "#ff7f0e"}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for ax, task in zip(axes.ravel(), TASKS):
    t = T[task]; s = (t["start"]["x"], t["start"]["y"]); g = (t["goal"]["x"], t["goal"]["y"])
    seeds = generate_route_seeds(DJ, s, g)
    ax.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=0.9, alpha=0.85, zorder=0)
    for poly in _dissolve(DRIVE):
        ax.add_patch(_patch(poly, facecolor="none", edgecolor="#39ff14", lw=1.4, alpha=0.75, zorder=1))
    for (x0, x1, y0, y1) in racks:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#f2cf23", edgecolor="#0b6f8a", lw=0.6, zorder=3))
    for rt in seeds:
        w = np.vstack([[s[0], s[1]], np.array(rt["waypoints"], float)])
        ax.plot(w[:, 0], w[:, 1], "o--", color=COL.get(rt["name"], "#333"), lw=2.0, ms=6, alpha=0.95, zorder=6,
                label=f"seed: {rt['name']}")
    ax.scatter([s[0]], [s[1]], s=110, c="#16a34a", edgecolor="k", zorder=10, label="start")
    ax.scatter([g[0]], [g[1]], s=170, c="#e41a1c", marker="*", edgecolor="k", zorder=10, label="goal")
    ax.set_title(task, fontsize=10, fontweight="bold")
    ax.set_xlim(-5.7, 5.5); ax.set_ylim(-3.7, 5.0); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.suptitle("Map-derived lane-graph route SEEDS (condition-neutral, no GP, no solving)\n"
             "below = main aisle under racks; above = mid connector (east) or upper cross-aisle (west)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
