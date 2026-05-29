#!/usr/bin/env python3
"""Offline two-stage (global->local) prototype, before any ROS code.

Stage A: ONE long-ish global plan (multistart, full visibility EFE, belief
log-barrier keep-in) to the final goal -> extract_waypoints.
Stage B: short-horizon LEAN local tracker (keep-in belief log-barrier + risk,
NO ambiguity), closed-loop: target current waypoint, execute 1 step, replan,
advance on arrival. Measures lane adherence, goal reach, and local solve time.
Plots global route + waypoints + executed local path over the visibility heatmap.
"""
from __future__ import annotations
import json, sys, time, glob
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
from planning.planners.base_planner import extract_waypoints  # noqa: E402

CONFIG = SCRIPTS / "aws_smoke_config.yaml"
PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
GP = REPO / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
WORLD = "warehouse_aws.world.sdf"; TASK = "B1_apron_a4_to_uppermid_a3"
OUT = Path("/home/joostleliveld/Thesis/timing_presentation/figures/F11_two_stage.png")
GAZ = Path("/home/joostleliveld/Thesis/timing_presentation/runs/gazebo")

DT, V_MAX = 0.25, 1.5
H_GLOBAL, H_LOCAL = 60, 12
LATERAL = [-2.0, 2.0]
# belief log-barrier keep-in (Phase 1 pick)
KW, KM, KSC, KEPS, KAPPA = 40.0, 0.15, 0.25, 0.01, 2.0
WP_SPACING, ARRIVE = 1.0, 0.35
MAX_STEPS = 120

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "figure.facecolor": "white"})


def driveable_json():
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    pr = [{"name": r["name"], "xmin": float(r["xmin"]), "xmax": float(r["xmax"]),
           "ymin": float(r["ymin"]), "ymax": float(r["ymax"]), "zmin": 0.0, "zmax": 0.1}
          for r in w.get("known_2d_regions", []) if r.get("type") == "traversable"]
    return json.dumps({"prisms": pr, "model_name": "driveable_region"})


DJ = driveable_json()
MEAS = NogoZoneCostModel(NogoCostConfig(weight=1.0, geometry_json=DJ, mode="keep_in"))
def d_out(xy): return max(MEAS.signed_distance_state_np([xy[0], xy[1], 0.0]), 0.0)


def keepin():
    return NogoZoneCostModel(NogoCostConfig(penalty_type="log_barrier", weight=KW,
        safe_distance=KM, logbarrier_scale=KSC, logbarrier_eps=KEPS,
        geometry_json=DJ, mode="keep_in"))


def build(setup, vg, H, multistart, ambiguity, local=False):
    cfg = dict(setup.config); cfg["horizon"] = H; cfg["dt"] = DT; cfg["v_max"] = V_MAX
    cfg["use_nogo_cost"] = True
    if local:
        # short-horizon tracker: strong, near-constant goal pull to the close waypoint
        cfg["goal_progress_n_steps"] = max(int(H), 1)
        cfg["goal_prior_u_std_start"] = cfg.get("goal_prior_u_std_final", 20.0)
        cfg["goal_prior_v_std_start"] = cfg.get("goal_prior_v_std_final", 20.0)
    p = build_planner(cfg, planner_kind=setup.planner_kind, camera_params=setup.camera_params,
                      visibility_artifact_path=str(setup.gp_path), visibility_geometry_json=vg,
                      collision_geometry_json=setup.geometry_json, seed=1,
                      visibility_target_height_m=setup.gp["target_height"])
    p.use_belief_nogo_cost = True; p.nogo_belief_kappa = KAPPA
    p.nogo_cost_model = keepin()
    p.use_ambiguity = bool(ambiguity)
    p.optimizer_multistart = bool(multistart)
    p.optimizer_multistart_include_direct = True
    p.optimizer_multistart_lateral_offsets = list(LATERAL) if multistart else []
    return p


def main():
    setup = load_setup(CONFIG, condition="C2", seed=1, task_override=TASK)
    vg = serialize_occlusion_geometry_from_world(resolve_world_path(setup.world))
    m0 = np.array(setup.start_xy_yaw); S0 = setup.S0; goal = np.array(setup.goal_xy)

    # ---- Stage A: global plan once ----
    g = build(setup, vg, H_GLOBAL, multistart=True, ambiguity=True)
    g.prev_controls_flat = None
    t0 = time.perf_counter(); rg = g.plan(m0, S0, goal); t_global = time.perf_counter() - t0
    gstates = np.asarray(rg.states)
    wps = extract_waypoints(gstates, spacing_m=WP_SPACING, include_goal=True)
    print(f"Stage A: global H={H_GLOBAL} solve={t_global:.1f}s termd={rg.terminal_goal_distance_pred:.2f} "
          f"src={rg.selected_source} -> {len(wps)} waypoints")

    # ---- Stage B: lean local tracker, closed loop ----
    L = build(setup, vg, H_LOCAL, multistart=False, ambiguity=False, local=True)
    m, S = m0.copy(), S0.copy()
    traj = [m[:2].copy()]; solve_ts = []; wp_idx = 0
    reached = False
    for step in range(MAX_STEPS):
        target = np.array(wps[wp_idx])
        S = S0.copy()  # bounded belief: EKF corrects via camera each cycle (no runaway covariance)
        L.prev_controls_flat = None
        t1 = time.perf_counter(); rl = L.plan(m, S, target); solve_ts.append(time.perf_counter() - t1)
        u0 = np.asarray(rl.controls)[0]
        m, _ = L.predict(m, S, u0)
        traj.append(m[:2].copy())
        if np.linalg.norm(m[:2] - target) < ARRIVE:
            if wp_idx == len(wps) - 1:
                reached = True; break
            wp_idx += 1
    traj = np.array(traj)
    douts = np.array([d_out(p) for p in traj])
    solve_ts = np.array(solve_ts)
    print(f"Stage B: H={H_LOCAL} steps={len(solve_ts)} reached_goal={reached} "
          f"final_d={np.linalg.norm(traj[-1]-goal):.2f} max_d_out={douts.max():.3f} "
          f"path={np.linalg.norm(np.diff(traj,axis=0),axis=1).sum():.2f}m "
          f"local_solve mean={solve_ts.mean()*1000:.0f}ms max={solve_ts.max()*1000:.0f}ms")

    # ---- plot over visibility heatmap ----
    d = np.load(GP, allow_pickle=True); xs, ys, Pc = d["xs"], d["ys"], d["P_conservative_plan_map"]
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(Pc, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower",
                   cmap="RdYlGn", vmin=0, vmax=1, aspect="equal", zorder=0, alpha=0.85)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("P_conservative (green=visible, red=shadow)")
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    for rr in w.get("known_2d_regions", []):
        if rr.get("type") == "traversable":
            ax.add_patch(Rectangle((rr["xmin"], rr["ymin"]), rr["xmax"]-rr["xmin"], rr["ymax"]-rr["ymin"],
                         facecolor="none", edgecolor="k", lw=1.0, ls=":", alpha=0.6, zorder=3))
    mani = sorted(glob.glob(f"{GAZ}/*/*/run_manifest.json"))
    if mani:
        cg = json.loads(Path(mani[-1]).read_text())["collision_geometry_json"]
        for q in (json.loads(cg) if isinstance(cg, str) else cg).get("prisms", []):
            if "wall_" in q["name"]: continue
            ax.add_patch(Rectangle((q["xmin"], q["ymin"]), q["xmax"]-q["xmin"], q["ymax"]-q["ymin"],
                         facecolor="0.35", edgecolor="k", lw=0.8, alpha=0.8, zorder=2))
    ax.plot(gstates[:, 0], gstates[:, 1], "--", color="tab:blue", lw=2, zorder=4, label="Stage A global plan")
    wpa = np.array(wps)
    ax.plot(wpa[:, 0], wpa[:, 1], "o", color="white", mec="tab:blue", ms=9, zorder=6, label="waypoints")
    ax.plot(traj[:, 0], traj[:, 1], "-", color="magenta", lw=3, zorder=7, label="Stage B local execution")
    sx, sy, syaw = setup.start_xy_yaw
    ax.add_patch(Circle((sx, sy), 0.125, facecolor="k", edgecolor="w", zorder=8))
    ax.add_patch(FancyArrow(sx, sy, 0.45*np.cos(syaw), 0.45*np.sin(syaw), width=0.05,
                            head_width=0.16, color="k", zorder=9, length_includes_head=True))
    ax.plot(goal[0], goal[1], "*", color="k", ms=24, mec="w", zorder=8, label="goal")
    ax.set_xlim(-0.3, 4.7); ax.set_ylim(-2.3, 3.0)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Two-stage: global({H_GLOBAL}) -> {len(wps)} waypoints -> lean local({H_LOCAL}) "
                 f"@ v_max={V_MAX} | local {solve_ts.mean()*1000:.0f}ms/step")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140); print("wrote", OUT)


if __name__ == "__main__":
    main()
