#!/usr/bin/env python3
"""C1 (no visibility) vs C2 (visibility-aware) route from the pick start, over
the visibility heatmap. Measures shadow exposure along each route to answer:
is visibility steering the route, and is a more-visible detour available?"""
from __future__ import annotations
import json, sys
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
WORLD = "warehouse_aws.world.sdf"; CONFIG = SCRIPTS / "aws_smoke_config.yaml"
GP = REPO / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
TASK = "B1_apron_a4_to_uppermid_a3"
OUT = Path("/home/joostleliveld/Thesis/timing_presentation/figures/F10_c1_vs_c2_visibility.png")
H, V_MAX, LATERAL = 200, 0.22, [-2.0, 2.0]
KW, KM, KS = 600.0, 0.10, 0.05


def driveable_json():
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    pr = [{"name": r["name"], "xmin": float(r["xmin"]), "xmax": float(r["xmax"]),
           "ymin": float(r["ymin"]), "ymax": float(r["ymax"]), "zmin": 0.0, "zmax": 0.1}
          for r in w.get("known_2d_regions", []) if r.get("type") == "traversable"]
    return json.dumps({"prisms": pr, "model_name": "driveable_region"})


def route(condition, vg, dj):
    setup = load_setup(CONFIG, condition=condition, seed=1, task_override=TASK)
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
    return p, np.asarray(r.states), r.terminal_goal_distance_pred, setup


def main():
    vg = serialize_occlusion_geometry_from_world(resolve_world_path(WORLD))
    dj = driveable_json()
    d = np.load(GP, allow_pickle=True)
    xs, ys, Pc = d["xs"], d["ys"], d["P_conservative_plan_map"]

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(Pc, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower",
                   cmap="RdYlGn", vmin=0, vmax=1, aspect="equal", zorder=0, alpha=0.9)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("P_conservative (green=visible, red=shadow)")
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    setup0 = None
    for cond, color in [("C1", "tab:blue"), ("C2", "magenta")]:
        p, st, termd, setup0 = route(cond, vg, dj)
        pv = np.array([p.visibility_probability([s[0], s[1], 0.0]) for s in st])
        ax.plot(st[:, 0], st[:, 1], "-", lw=3, color=color, zorder=5,
                label=f"{cond}: mean P_vis={pv.mean():.2f}, min={pv.min():.2f}, termd={termd:.2f}")
        print(f"{cond}: mean_pvis={pv.mean():.3f} min_pvis={pv.min():.3f} frac<0.3={float((pv<0.3).mean()):.2f} termd={termd:.2f}")
    sx, sy, syaw = setup0.start_xy_yaw; gx, gy = setup0.goal_xy
    ax.add_patch(Circle((sx, sy), 0.125, facecolor="k", edgecolor="w", zorder=8))
    ax.add_patch(FancyArrow(sx, sy, 0.45*np.cos(syaw), 0.45*np.sin(syaw), width=0.05,
                            head_width=0.16, color="k", zorder=9, length_includes_head=True))
    ax.plot(gx, gy, "*", color="k", ms=24, mec="w", zorder=8, label="goal (in shadow)")
    ax.set_xlim(-0.3, 4.7); ax.set_ylim(-2.3, 3.0)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("C1 (no visibility) vs C2 (visibility-aware) from pick start")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
    fig.tight_layout(); fig.savefig(OUT, dpi=140); print("wrote", OUT)


if __name__ == "__main__":
    main()
