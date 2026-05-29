#!/usr/bin/env python3
"""Offline solve-time scaling benchmark (durable out-of-repo copy).

Measures planner.plan() wall time vs horizon, multistart OFF/ON, using the real
AWS B1 setup. Writes CSV to /home/joostleliveld/Thesis/timing_presentation/runs/.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
DURABLE = Path("/home/joostleliveld/Thesis/timing_presentation")
SCRIPTS = REPO / "scripts/visibility_comparison"
sys.path.insert(0, str(SCRIPTS))

from efe_offline_lab import build_planner, load_setup  # noqa: E402
from experiments.core.world_profiles import (  # noqa: E402
    resolve_world_path, serialize_occlusion_geometry_from_world)

CONFIG = SCRIPTS / "aws_smoke_config.yaml"
TASK = "B1_apron_a4_to_uppermid_a3"
OUT = DURABLE / "runs/offline_solve_scaling.csv"
HORIZONS = [40, 80, 120, 200]
V_MAX = 0.22
LATERAL = [-2.0, 2.0]
N_REPEAT = 3


def make_planner(cfg, setup, horizon, multistart, vg):
    c = dict(cfg); c["horizon"] = int(horizon); c["v_max"] = V_MAX
    p = build_planner(c, planner_kind=setup.planner_kind, camera_params=setup.camera_params,
                      visibility_artifact_path=str(setup.gp_path), visibility_geometry_json=vg,
                      collision_geometry_json=setup.geometry_json, seed=1,
                      visibility_target_height_m=setup.gp["target_height"])
    p.optimizer_multistart = bool(multistart)
    p.optimizer_multistart_include_direct = True
    p.optimizer_multistart_lateral_offsets = list(LATERAL) if multistart else []
    return p


def main():
    setup = load_setup(CONFIG, condition="C2", seed=1, task_override=TASK)
    cfg = setup.config
    vg = serialize_occlusion_geometry_from_world(resolve_world_path(setup.world))
    m0 = np.array(setup.start_xy_yaw); S0 = setup.S0; goal = np.array(setup.goal_xy)
    print(f"setup start={tuple(m0)} goal={tuple(goal)} task={TASK} v_max={V_MAX}")
    rows = []
    for ms in (False, True):
        for H in HORIZONS:
            p = make_planner(cfg, setup, H, ms, vg)
            ncand = 1 + len(p._build_multistart_candidates(m0, goal))
            p.prev_controls_flat = None
            r0 = p.plan(m0, S0, goal); comp = r0.solve_time_s
            times = []
            for _ in range(N_REPEAT):
                p.prev_controls_flat = None
                t0 = time.perf_counter(); r = p.plan(m0, S0, goal)
                times.append(time.perf_counter() - t0)
            times = np.array(times)
            rows.append({"multistart": int(ms), "horizon": H, "n_candidates": ncand,
                         "first_compile_solve_s": round(comp, 4),
                         "solve_s_mean": round(float(times.mean()), 4),
                         "solve_s_min": round(float(times.min()), 4),
                         "solve_s_max": round(float(times.max()), 4),
                         "per_candidate_s": round(float(times.mean()) / max(ncand, 1), 4),
                         "selected_source": r.selected_source, "optimizer_nit": r.optimizer_nit,
                         "optimizer_nfev": r.optimizer_nfev,
                         "terminal_goal_distance_pred": round(r.terminal_goal_distance_pred, 3),
                         "rollout_valid": int(bool(r.rollout_valid)),
                         "total_cost": round(r.total_cost, 3)})
            print(f"ms={int(ms)} H={H:3d} nc={ncand} comp={comp:5.2f} mean={times.mean():5.2f} "
                  f"src={r.selected_source} termd={r.terminal_goal_distance_pred:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
