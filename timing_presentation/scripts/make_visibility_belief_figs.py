#!/usr/bin/env python3
"""F12: two-stage global plan with belief 2-sigma ellipses (visibility-EKF:
tight where visible, balloons in shadow) over the visibility heatmap.
F13: ambiguity_weight sweep showing how the visible detour emerges."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow, Ellipse

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
FIGDIR = Path("/home/joostleliveld/Thesis/timing_presentation/figures")
H, V_MAX, LATERAL = 60, 1.5, [-2.0, 2.0]
KW, KM, KSC, KEPS, KAPPA = 40.0, 0.15, 0.25, 0.01, 2.0
R_VIS, R_SHADOW = 0.05, 1.2  # position-measurement std (m): visible vs shadow

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "figure.facecolor": "white"})


def driveable_json():
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    pr = [{"name": r["name"], "xmin": float(r["xmin"]), "xmax": float(r["xmax"]),
           "ymin": float(r["ymin"]), "ymax": float(r["ymax"]), "zmin": 0.0, "zmax": 0.1}
          for r in w.get("known_2d_regions", []) if r.get("type") == "traversable"]
    return json.dumps({"prisms": pr, "model_name": "driveable_region"})


DJ = driveable_json()


def build(setup, vg, ambiguity_weight):
    cfg = dict(setup.config); cfg["horizon"] = H; cfg["dt"] = 0.25; cfg["v_max"] = V_MAX
    cfg["use_nogo_cost"] = True; cfg["ambiguity_weight"] = ambiguity_weight
    p = build_planner(cfg, planner_kind=setup.planner_kind, camera_params=setup.camera_params,
                      visibility_artifact_path=str(setup.gp_path), visibility_geometry_json=vg,
                      collision_geometry_json=setup.geometry_json, seed=1,
                      visibility_target_height_m=setup.gp["target_height"])
    p.use_belief_nogo_cost = True; p.nogo_belief_kappa = KAPPA
    p.nogo_cost_model = NogoZoneCostModel(NogoCostConfig(penalty_type="log_barrier",
        weight=KW, safe_distance=KM, logbarrier_scale=KSC, logbarrier_eps=KEPS,
        geometry_json=DJ, mode="keep_in"))
    p.optimizer_multistart = True; p.optimizer_multistart_include_direct = True
    p.optimizer_multistart_lateral_offsets = list(LATERAL)
    return p


def belief_rollout(p, m0, S0, controls):
    """EKF along the plan with visibility-dependent position measurement noise:
    small R where visible (S shrinks), large R in shadow (S grows)."""
    m, S = m0.copy(), S0.copy()
    out = [(m.copy(), S.copy())]
    for u in np.asarray(controls):
        m, S = p.predict(m, S, u)
        pv = float(p.visibility_probability([m[0], m[1], m[2]]))
        r = R_VIS + (R_SHADOW - R_VIS) * (1.0 - pv)
        R = np.diag([r * r, r * r])
        Sxy = 0.5 * (S[:2, :2] + S[:2, :2].T)
        Sxy = np.linalg.inv(np.linalg.inv(Sxy + 1e-6 * np.eye(2)) + np.linalg.inv(R))
        Sxy = 0.5 * (Sxy + Sxy.T)
        ev, evec = np.linalg.eigh(Sxy)
        ev = np.clip(ev, 1e-6, 4.0)  # PSD + cap runaway shadow growth
        Sxy = evec @ np.diag(ev) @ evec.T
        S[:2, :2] = Sxy
        out.append((m.copy(), S.copy()))
    return out


def heat_and_world(ax):
    d = np.load(GP, allow_pickle=True); xs, ys, Pc = d["xs"], d["ys"], d["P_conservative_plan_map"]
    im = ax.imshow(Pc, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower",
                   cmap="RdYlGn", vmin=0, vmax=1, aspect="equal", zorder=0, alpha=0.85)
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    for rr in w.get("known_2d_regions", []):
        if rr.get("type") == "traversable":
            ax.add_patch(Rectangle((rr["xmin"], rr["ymin"]), rr["xmax"]-rr["xmin"], rr["ymax"]-rr["ymin"],
                         facecolor="none", edgecolor="k", lw=1.0, ls=":", alpha=0.55, zorder=3))
    import glob
    mani = sorted(glob.glob("/home/joostleliveld/Thesis/timing_presentation/runs/gazebo/*/*/run_manifest.json"))
    if mani:
        cg = json.loads(Path(mani[-1]).read_text())["collision_geometry_json"]
        for q in (json.loads(cg) if isinstance(cg, str) else cg).get("prisms", []):
            if "wall_" in q["name"]: continue
            ax.add_patch(Rectangle((q["xmin"], q["ymin"]), q["xmax"]-q["xmin"], q["ymax"]-q["ymin"],
                         facecolor="0.4", edgecolor="k", lw=0.7, alpha=0.85, zorder=2))
    return im


def main():
    setup = load_setup(CONFIG, condition="C2", seed=1, task_override=TASK)
    vg = serialize_occlusion_geometry_from_world(resolve_world_path(setup.world))
    m0 = np.array(setup.start_xy_yaw); S0 = setup.S0; goal = np.array(setup.goal_xy)
    sx, sy, syaw = setup.start_xy_yaw

    # ---------- F12: two-stage global plan + belief 2-sigma ellipses ----------
    p = build(setup, vg, ambiguity_weight=6.0)
    p.prev_controls_flat = None
    r = p.plan(m0, S0, goal)
    st = np.asarray(r.states); wps = np.array(extract_waypoints(st, 1.0, True))
    bel = belief_rollout(p, m0, S0, r.controls)
    fig, ax = plt.subplots(figsize=(12, 9))
    cb = fig.colorbar(heat_and_world(ax), ax=ax, shrink=0.8)
    cb.set_label("P_conservative (green=visible, red=shadow)")
    ax.plot(st[:, 0], st[:, 1], "-", color="tab:blue", lw=2.5, zorder=5, label="global plan (visibility-aware)")
    ax.plot(wps[:, 0], wps[:, 1], "o", color="white", mec="tab:blue", ms=8, zorder=6, label="waypoints")
    first = True
    for (m, S) in bel[::3]:
        vals, vecs = np.linalg.eigh(S[:2, :2])
        vals = np.maximum(vals, 0.0)
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        e = Ellipse((m[0], m[1]), 2*2*np.sqrt(vals[-1]), 2*2*np.sqrt(vals[0]), angle=ang,
                    facecolor="none", edgecolor="navy", lw=1.3, alpha=0.8, zorder=7,
                    label="belief 2σ" if first else None); first = False
        ax.add_patch(e)
    ax.add_patch(Circle((sx, sy), 0.125, facecolor="k", edgecolor="w", zorder=8))
    ax.add_patch(FancyArrow(sx, sy, 0.45*np.cos(syaw), 0.45*np.sin(syaw), width=0.05,
                            head_width=0.16, color="k", zorder=9, length_includes_head=True))
    ax.plot(goal[0], goal[1], "*", color="k", ms=24, mec="w", zorder=8, label="goal (in shadow)")
    ax.set_xlim(-0.3, 4.7); ax.set_ylim(-2.3, 3.0); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("F12: two-stage global plan + belief 2σ (tight in green, balloons in shadow)")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
    fig.tight_layout(); fig.savefig(FIGDIR / "F12_two_stage_belief.png", dpi=140); plt.close(fig)
    print("wrote F12; belief 2sigma_x start->end:",
          f"{2*np.sqrt(bel[0][1][0,0]):.2f} -> {2*np.sqrt(bel[-1][1][0,0]):.2f} m")

    # ---------- F13: ambiguity_weight sweep (how to get the detour) ----------
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.colorbar(heat_and_world(ax), ax=ax, shrink=0.8).set_label("P_conservative (green=visible, red=shadow)")
    cmap = plt.get_cmap("cool")
    weights = [6.0, 30.0, 100.0]
    for i, aw in enumerate(weights):
        pw = build(setup, vg, ambiguity_weight=aw); pw.prev_controls_flat = None
        rw = pw.plan(m0, S0, goal); sw = np.asarray(rw.states)
        pv = np.array([pw.visibility_probability([s[0], s[1], 0.0]) for s in sw])
        ax.plot(sw[:, 0], sw[:, 1], "-", lw=3, color=cmap(i/(len(weights)-1)), zorder=5,
                label=f"amb_weight={aw:g} (mean P_vis={pv.mean():.2f}, termd {rw.terminal_goal_distance_pred:.2f})")
        print(f"amb_weight={aw}: mean_pvis={pv.mean():.3f} termd={rw.terminal_goal_distance_pred:.2f}")
    ax.add_patch(Circle((sx, sy), 0.125, facecolor="k", edgecolor="w", zorder=8))
    ax.add_patch(FancyArrow(sx, sy, 0.45*np.cos(syaw), 0.45*np.sin(syaw), width=0.05,
                            head_width=0.16, color="k", zorder=9, length_includes_head=True))
    ax.plot(goal[0], goal[1], "*", color="k", ms=24, mec="w", zorder=8, label="goal (in shadow)")
    ax.set_xlim(-0.3, 4.7); ax.set_ylim(-2.3, 3.0); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("F13: visibility weight sweep — higher amb_weight => stay-visible detour")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
    fig.tight_layout(); fig.savefig(FIGDIR / "F13_ambiguity_sweep.png", dpi=140); plt.close(fig)
    print("wrote F13")


if __name__ == "__main__":
    main()
