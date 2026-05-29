#!/usr/bin/env python3
"""Probe: belief log-barrier keep-in (driveable-region) penalty at v_max=1.5.

Penalises the part of the belief (2sigma, kappa=2) that lies OUTSIDE the
driveable lane union, via log_barrier, weighted by belief. For each config:
plan the B1 pick task and report how far outside the lanes the plan's MEAN and
its 2sigma-inflated band stray, whether it reaches the goal, and whether it
stalls. Goal: stay in lanes (incl. 2sigma), reach goal, no stall.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, yaml

REPO = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
SCRIPTS = REPO / "scripts/visibility_comparison"
sys.path.insert(0, str(SCRIPTS))
from efe_offline_lab import build_planner, load_setup  # noqa: E402
from experiments.core.world_profiles import (  # noqa: E402
    resolve_world_path, serialize_occlusion_geometry_from_world)
from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel  # noqa: E402

CONFIG = SCRIPTS / "aws_smoke_config.yaml"
PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
WORLD = "warehouse_aws.world.sdf"
TASK = "B1_apron_a4_to_uppermid_a3"
H, DT, V_MAX, LATERAL, KAPPA = 40, 0.25, 1.5, [-2.0, 2.0], 2.0


def driveable_json():
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    pr = [{"name": r["name"], "xmin": float(r["xmin"]), "xmax": float(r["xmax"]),
           "ymin": float(r["ymin"]), "ymax": float(r["ymax"]), "zmin": 0.0, "zmax": 0.1}
          for r in w.get("known_2d_regions", []) if r.get("type") == "traversable"]
    return json.dumps({"prisms": pr, "model_name": "driveable_region"})


DJ = driveable_json()
MEAS = NogoZoneCostModel(NogoCostConfig(weight=1.0, geometry_json=DJ, mode="keep_in"))


def d_out(xy):
    return max(MEAS.signed_distance_state_np([xy[0], xy[1], 0.0]), 0.0)


def make(setup, vg, weight, margin, lscale, leps):
    cfg = dict(setup.config); cfg["horizon"] = H; cfg["dt"] = DT; cfg["v_max"] = V_MAX
    cfg["use_nogo_cost"] = True
    p = build_planner(cfg, planner_kind=setup.planner_kind, camera_params=setup.camera_params,
                      visibility_artifact_path=str(setup.gp_path), visibility_geometry_json=vg,
                      collision_geometry_json=setup.geometry_json, seed=1,
                      visibility_target_height_m=setup.gp["target_height"])
    p.use_belief_nogo_cost = True
    p.nogo_belief_kappa = KAPPA
    p.nogo_cost_model = NogoZoneCostModel(NogoCostConfig(
        penalty_type="log_barrier", weight=weight, safe_distance=margin,
        logbarrier_scale=lscale, logbarrier_eps=leps, geometry_json=DJ, mode="keep_in"))
    p.optimizer_multistart = True; p.optimizer_multistart_include_direct = True
    p.optimizer_multistart_lateral_offsets = list(LATERAL)
    return p


# (label, weight, margin, logbarrier_scale, logbarrier_eps)
CONFIGS = [
    ("OFF (no keepin)",              None),
    ("logbar w=2  m=0.10 sc=0.25 eps=0.02", (2,  0.10, 0.25, 0.02)),
    ("logbar w=10 m=0.10 sc=0.25 eps=0.02", (10, 0.10, 0.25, 0.02)),
    ("logbar w=10 m=0.15 sc=0.25 eps=0.01", (10, 0.15, 0.25, 0.01)),
    ("logbar w=40 m=0.15 sc=0.25 eps=0.01", (40, 0.15, 0.25, 0.01)),
    ("logbar w=40 m=0.20 sc=0.40 eps=0.005",(40, 0.20, 0.40, 0.005)),
]


def main():
    setup = load_setup(CONFIG, condition="C2", seed=1, task_override=TASK)
    vg = serialize_occlusion_geometry_from_world(resolve_world_path(setup.world))
    m0 = np.array(setup.start_xy_yaw); S0 = setup.S0; goal = np.array(setup.goal_xy)
    print(f"task {TASK} start={tuple(m0)} goal={tuple(goal)} v_max={V_MAX} dt={DT} H={H} kappa={KAPPA}")
    print(f"{'config':38s} {'maxDout_mean':>12s} {'maxDout_2sig':>12s} {'termd':>7s} {'path_m':>7s} {'src':>20s}")
    for label, params in CONFIGS:
        if params is None:
            cfg = dict(setup.config); cfg["horizon"] = H; cfg["dt"] = DT; cfg["v_max"] = V_MAX
            cfg["use_nogo_cost"] = False
            p = build_planner(cfg, planner_kind=setup.planner_kind, camera_params=setup.camera_params,
                              visibility_artifact_path=str(setup.gp_path), visibility_geometry_json=vg,
                              collision_geometry_json=setup.geometry_json, seed=1,
                              visibility_target_height_m=setup.gp["target_height"])
            p.optimizer_multistart = True; p.optimizer_multistart_include_direct = True
            p.optimizer_multistart_lateral_offsets = list(LATERAL)
        else:
            p = make(setup, vg, *params)
        p.prev_controls_flat = None
        r = p.plan(m0, S0, goal)
        ctrls = np.asarray(r.controls)
        # roll out mean + covariance to get 2sigma band
        m, S = m0.copy(), S0.copy()
        dmean, d2s = [d_out(m)], [d_out(m)]
        for u in ctrls:
            m, S = p.predict(m, S, u)
            dm = d_out(m)
            # 2-sigma inflation: nearest-lane excursion grows by 2*sqrt(max eig of xy cov)
            sig = 2.0 * float(np.sqrt(max(np.linalg.eigvalsh(S[:2, :2])[-1], 0.0)))
            dmean.append(dm); d2s.append(max(dm - 0.0, 0.0) + (sig if dm > 1e-6 else 0.0))
        dmean = np.array(dmean)
        st = np.asarray(r.states); path = float(np.linalg.norm(np.diff(st[:, :2], axis=0), axis=1).sum())
        print(f"{label:38s} {dmean.max():12.3f} {np.array(d2s).max():12.3f} "
              f"{r.terminal_goal_distance_pred:7.2f} {path:7.2f} {r.selected_source:>20s}")


if __name__ == "__main__":
    main()
