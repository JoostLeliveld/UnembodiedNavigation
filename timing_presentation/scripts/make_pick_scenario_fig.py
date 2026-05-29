#!/usr/bin/env python3
"""Pick scenario figure: robot starts on the right of aisle A4 facing shelf R5
(just finished a pick), then plans the keep-in route to the goal in A3."""
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
TASK = "B1_apron_a4_to_uppermid_a3"
OUT = Path("/home/joostleliveld/Thesis/timing_presentation/figures/F8_pick_scenario.png")
GAZ = Path("/home/joostleliveld/Thesis/timing_presentation/runs/gazebo")
H, V_MAX, LATERAL = 200, 0.22, [-2.0, 2.0]
KW, KM, KS = 600.0, 0.10, 0.05  # keep-in weight, margin, scale
RADIUS = 0.125

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "figure.facecolor": "white"})


def driveable_json():
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    pr = [{"name": r["name"], "xmin": float(r["xmin"]), "xmax": float(r["xmax"]),
           "ymin": float(r["ymin"]), "ymax": float(r["ymax"]), "zmin": 0.0, "zmax": 0.1}
          for r in w.get("known_2d_regions", []) if r.get("type") == "traversable"]
    return json.dumps({"prisms": pr, "model_name": "driveable_region"}), w.get("known_2d_regions", [])


def obstacles():
    rd = sorted(glob.glob(f"{GAZ}/*/*/run_manifest.json"))
    cg = json.loads(json.loads(Path(rd[-1]).read_text())["collision_geometry_json"]) \
        if isinstance(json.loads(Path(rd[-1]).read_text())["collision_geometry_json"], str) \
        else json.loads(Path(rd[-1]).read_text())["collision_geometry_json"]
    return cg.get("prisms", [])


def main():
    setup = load_setup(CONFIG, condition="C2", seed=1, task_override=TASK)
    vg = serialize_occlusion_geometry_from_world(resolve_world_path(setup.world))
    dj, regions = driveable_json()
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
    m0 = np.array(setup.start_xy_yaw); S0 = setup.S0; goal = np.array(setup.goal_xy)
    p.prev_controls_flat = None
    r = p.plan(m0, S0, goal)
    st = np.asarray(r.states)
    print(f"start={tuple(m0)} goal={tuple(goal)} termd={r.terminal_goal_distance_pred:.2f} src={r.selected_source}")

    fig, ax = plt.subplots(figsize=(11, 9))
    first = True
    for rr in regions:
        if rr.get("type") != "traversable":
            continue
        ax.add_patch(Rectangle((rr["xmin"], rr["ymin"]), rr["xmax"]-rr["xmin"], rr["ymax"]-rr["ymin"],
                     facecolor="#b8e0b8", edgecolor="#7cc47c", lw=0.5, alpha=0.55, zorder=0,
                     label="driveable region" if first else None)); first = False
    for q in obstacles():
        if "wall_" in q["name"]:
            continue
        r4 = "R4" in q["name"]; r5 = "R5" in q["name"]; crate = "low_crate" in q["name"]
        ax.add_patch(Rectangle((q["xmin"], q["ymin"]), q["xmax"]-q["xmin"], q["ymax"]-q["ymin"],
                     facecolor="#e8a0a0" if crate else "#c8c8c8",
                     edgecolor="tab:orange" if r5 else ("tab:red" if r4 else "0.5"),
                     lw=1.8 if (r4 or r5) else 0.8, alpha=0.7, zorder=1))
    # planned route
    ax.plot(st[:, 0], st[:, 1], "-", color="tab:blue", lw=3, zorder=5, label="planned keep-in route")
    ax.plot(goal[0], goal[1], "*", color="k", ms=24, zorder=6, label="goal (A3)")
    # oriented robot at start: footprint + heading arrow (east, toward R5)
    sx, sy, syaw = m0
    ax.add_patch(Circle((sx, sy), RADIUS, facecolor="0.25", edgecolor="k", zorder=8))
    ax.add_patch(FancyArrow(sx, sy, 0.45*np.cos(syaw), 0.45*np.sin(syaw), width=0.05,
                            head_width=0.16, head_length=0.14, color="k", zorder=9, length_includes_head=True))
    ax.annotate("just picked\nfrom shelf R5", xy=(3.9, -1.0), xytext=(3.95, -1.85),
                fontsize=11, color="tab:orange", ha="center",
                arrowprops=dict(arrowstyle="->", color="tab:orange"))
    ax.annotate("start: right of A4,\nfacing R5", xy=(sx, sy), xytext=(2.2, -1.85),
                fontsize=11, ha="center", arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_xlim(0.3, 4.7); ax.set_ylim(-2.2, 2.8); ax.grid(alpha=0.25)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Pick scenario: start right-of-A4 facing R5, keep-in route out to A3 goal")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140); print("wrote", OUT)


if __name__ == "__main__":
    main()
