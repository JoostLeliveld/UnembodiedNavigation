#!/usr/bin/env python3
"""Pick scenario across horizons, over the visibility (P_conservative) heatmap.

Background = planner-facing visibility field from the GP (green=visible,
red=shadow). Overlaid: keep-in planned routes from the new pick start for
several horizons, plus obstacles and the oriented robot.
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow

REPO = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
SCRIPTS = REPO / "scripts/visibility_comparison"
sys.path.insert(0, str(SCRIPTS))
from efe_offline_lab import build_planner, load_setup  # noqa: E402
from experiments.core.world_profiles import (  # noqa: E402
    resolve_world_path, serialize_occlusion_geometry_from_world)
from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel  # noqa: E402

PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
WORLD = "warehouse_aws.world.sdf"
CONFIG = SCRIPTS / "aws_smoke_config.yaml"
GP = REPO / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
TASK = "B1_apron_a4_to_uppermid_a3"
OUT = Path("/home/joostleliveld/Thesis/timing_presentation/figures/F9_pick_horizons_visibility.png")
GAZ = Path("/home/joostleliveld/Thesis/timing_presentation/runs/gazebo")
HORIZONS = [40, 80, 120, 200]
V_MAX, LATERAL = 0.22, [-2.0, 2.0]
KW, KM, KS = 600.0, 0.10, 0.05
RADIUS = 0.125

plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
                     "figure.facecolor": "white"})


def driveable_json():
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    pr = [{"name": r["name"], "xmin": float(r["xmin"]), "xmax": float(r["xmax"]),
           "ymin": float(r["ymin"]), "ymax": float(r["ymax"]), "zmin": 0.0, "zmax": 0.1}
          for r in w.get("known_2d_regions", []) if r.get("type") == "traversable"]
    return json.dumps({"prisms": pr, "model_name": "driveable_region"})


def obstacles():
    p = sorted(glob.glob(f"{GAZ}/*/*/run_manifest.json"))[-1]
    cg = json.loads(Path(p).read_text())["collision_geometry_json"]
    cg = json.loads(cg) if isinstance(cg, str) else cg
    return cg.get("prisms", [])


def plan_route(setup, vg, dj, H):
    cfg = dict(setup.config); cfg["horizon"] = H; cfg["v_max"] = V_MAX
    cfg["use_nogo_cost"] = True; cfg["use_belief_nogo_cost"] = False
    p = build_planner(cfg, planner_kind=setup.planner_kind, camera_params=setup.camera_params,
                      visibility_artifact_path=str(setup.gp_path), visibility_geometry_json=vg,
                      collision_geometry_json=setup.geometry_json, seed=1,
                      visibility_target_height_m=setup.gp["target_height"])
    p.nogo_cost_model = NogoZoneCostModel(NogoCostConfig(
        penalty_type="softplus", weight=KW, safe_distance=KM, softplus_scale=KS,
        geometry_json=dj, mode="keep_in"))
    p.optimizer_multistart = True; p.optimizer_multistart_include_direct = True
    p.optimizer_multistart_lateral_offsets = list(LATERAL)
    p.prev_controls_flat = None
    r = p.plan(np.array(setup.start_xy_yaw), setup.S0, np.array(setup.goal_xy))
    return np.asarray(r.states), r.terminal_goal_distance_pred


def main():
    setup = load_setup(CONFIG, condition="C2", seed=1, task_override=TASK)
    vg = serialize_occlusion_geometry_from_world(resolve_world_path(setup.world))
    dj = driveable_json()
    d = np.load(GP, allow_pickle=True)
    xs, ys, Pcons = d["xs"], d["ys"], d["P_conservative_plan_map"]

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(Pcons, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower",
                   cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="equal", zorder=0, alpha=0.9)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("P_conservative  (camera visibility: green=visible, red=shadow)")

    # driveable lane outlines (keep-in region) over the heatmap
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    first = True
    for rr in w.get("known_2d_regions", []):
        if rr.get("type") != "traversable":
            continue
        ax.add_patch(Rectangle((rr["xmin"], rr["ymin"]), rr["xmax"]-rr["xmin"], rr["ymax"]-rr["ymin"],
                     facecolor="none", edgecolor="blue", lw=2.0, ls="--", alpha=0.9, zorder=3,
                     label="driveable lanes" if first else None)); first = False
    for q in obstacles():
        if "wall_" in q["name"]:
            continue
        r45 = ("R4" in q["name"]) or ("R5" in q["name"])
        ax.add_patch(Rectangle((q["xmin"], q["ymin"]), q["xmax"]-q["xmin"], q["ymax"]-q["ymin"],
                     facecolor="0.35", edgecolor="k", lw=1.4 if r45 else 0.7, alpha=0.85, zorder=2))

    cmap = plt.get_cmap("cool")
    for i, H in enumerate(HORIZONS):
        st, termd = plan_route(setup, vg, dj, H)
        ax.plot(st[:, 0], st[:, 1], "-", lw=2.8, color=cmap(i/(len(HORIZONS)-1)), zorder=5,
                label=f"H={H} (termd {termd:.2f} m)")
        print(f"H={H} termd={termd:.2f}")

    sx, sy, syaw = setup.start_xy_yaw
    gx, gy = setup.goal_xy
    ax.add_patch(Circle((sx, sy), RADIUS, facecolor="k", edgecolor="w", zorder=8))
    ax.add_patch(FancyArrow(sx, sy, 0.45*np.cos(syaw), 0.45*np.sin(syaw), width=0.05,
                            head_width=0.16, head_length=0.14, color="k", zorder=9,
                            length_includes_head=True))
    ax.plot(gx, gy, "*", color="k", ms=24, mec="w", zorder=8, label="goal")
    ax.annotate("just picked\nfrom R5", xy=(3.9, -1.0), xytext=(4.0, -1.9), fontsize=11,
                color="k", ha="center", arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlim(-0.3, 4.7); ax.set_ylim(-2.3, 3.0)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Keep-in route vs horizon, over camera-visibility field (pick scenario)")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140); print("wrote", OUT)


if __name__ == "__main__":
    main()
