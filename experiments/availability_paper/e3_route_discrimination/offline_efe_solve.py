#!/usr/bin/env python3
"""Offline EFE route solve for C1-C4, using the RUNTIME objective.

WHY THIS EXISTS. E3 scored routes with a surrogate objective
(`length + 50 * integral trace(P)`), and under it availability changed the route.
The deployed planner does not use that objective. The 2026-08-18 closed-loop
campaign showed every arm selecting the SAME route, with a cost breakdown of
risk 511,683 / obstacle 1,017,055 / ambiguity 337 — availability enters at 0.02 %
of the objective and cannot move the choice. This harness reproduces that decision
offline, per field, with no Gazebo, so the arms can be compared in minutes.

DEFECTIVE -- DO NOT CITE ITS RESULTS (found 2026-08-20).  This harness does not pass
`nogo_mode` (so it silently defaults to 'keep_out' instead of the deployed 'keep_in'),
does not pass `driveable_geometry_json` (so the keep-in prisms are EMPTY), and does not
pass `use_belief_nogo_cost` (so it defaults to False).  Its output therefore carries
`obstacle = 0.0` in every row, which is exactly the channel through which availability
dominates the deployed objective: in the real closed-loop runs the availability-blind
arm pays 0 and the availability-aware arms pay 1,017,055 against a risk term of
511,683.  The conclusion recorded in the header below -- that availability enters at
0.02 % of the objective and cannot move the choice -- is an artefact of running with
the mechanism switched off.  Fix the four missing arguments before reusing this.

Arms:
  C1  availability-blind        (no visibility model; spatially uniform covariance)
  C2  operational GP            (GP on a geometric day-zero prior; needs a survey)
  C3  monocular depth           (camera RGB + calibration + drivable map)
  C4  monocular depth + GP      (GP residual on the depth prior; needs no survey)

Reads no ground truth and starts no simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

for _p in ("src/planning", "src/unav_common", "src/experiments", "src/reliability"):
    sys.path.insert(0, str(C.REPO / _p))
from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402
from experiments.core.world_profiles import load_profile, compute_look_at_from_pose  # noqa: E402

PROFILE_YAML = C.REPO / "src/experiments/config/world_profiles.yaml"
WORLD = "warehouse_full_4cam.world.sdf"
CAMPAIGN_PLAN = HERE.parent / "e4_closed_loop/campaign.yaml"
ROUTE_SEEDS = C.OUT_ROOT / "e3_route_discrimination/e3_selected_routes.json"

ARMS = {
    "C1_blind": None,
    "C2_operational_gp": C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz",
    "C3_mono_depth": C.OUT_ROOT / "mono_depth_planner_v1/fused_planner_four_camera.npz",
    "C4_depth_plus_gp": C.OUT_ROOT / "depth_gp_planner_v1/fused_planner_four_camera.npz",
}


def camera_params() -> dict:
    """Exactly what the launch file hands the planner."""
    profile, intrinsics, _model, pose = load_profile(str(PROFILE_YAML), WORLD)
    cam_pos = [float(pose[0]), float(pose[1]), float(pose[2])]
    look_at = compute_look_at_from_pose(cam_pos, float(pose[3]), float(pose[4]), float(pose[5]))
    return {
        "cam_pos": cam_pos, "look_at": look_at,
        "img_width": int(intrinsics["img_width"]), "img_height": int(intrinsics["img_height"]),
        "fov_h_rad": float(intrinsics["fov_h_rad"]),
    }


def build_planner(cfg: dict, artifact: Path | None, routes_json: str, cam: dict) -> UnicyclePlannerBase:
    """Global-planner configuration, mirroring unicycle_planner_node._construct_planner."""
    g = lambda k, d=None: cfg.get(k, d)
    return UnicyclePlannerBase(
        horizon=int(g("global_horizon")), dt=float(g("global_dt")),
        v_min=0.0, v_max=float(g("v_max")), w_min=-1.5, w_max=1.5,
        control_weight=float(g("control_weight", 0.0)),
        process_noise_xy=float(g("process_noise_xy")), process_noise_theta=float(g("process_noise_theta")),
        obs_noise_uv=2.0, goal_sigma_uv=2.0,
        risk_weight_obs=float(g("risk_weight_obs", 1.0)), ambiguity_weight=float(g("ambiguity_weight", 1.0)),
        optimizer_maxiter=int(g("optimizer_maxiter")), optimizer_maxfun=int(g("optimizer_maxfun")),
        optimizer_ftol=float(g("optimizer_ftol")), optimizer_gtol=float(g("optimizer_gtol")),
        optimizer_warm_start=False, optimizer_multistart=True,
        optimizer_multistart_include_direct=False,
        optimizer_initial_routes_json=routes_json,
        optimizer_terminal_goal_tolerance_m=float(g("optimizer_terminal_goal_tolerance_m", 0.5)),
        approx_method=None, use_obs_risk=True, use_ambiguity=bool(g("global_use_ambiguity", True)),
        seed=0, camera_params=cam,
        use_visibility_model=artifact is not None,
        visibility_target_height_m=0.0,
        visibility_geometry_json="", collision_geometry_json="",
        visibility_artifact_path=str(artifact) if artifact else "",
        r_visible_uv=float(g("r_visible_uv", 2.5)), r_miss_uv=float(g("r_miss_uv", 40.0)),
        goal_prior_u_std_start=float(g("goal_prior_u_std_start")),
        goal_prior_v_std_start=float(g("goal_prior_v_std_start")),
        goal_prior_u_std_final=float(g("goal_prior_u_std_final")),
        goal_prior_v_std_final=float(g("goal_prior_v_std_final")),
        goal_tightening_power=float(g("goal_tightening_power")),
        observation_risk_scale=float(g("observation_risk_scale", 1.0)),
        ambiguity_term_scale=float(g("ambiguity_term_scale", 1.0)),
        discount_gamma=float(g("discount_gamma")),
        use_nogo_cost=bool(g("use_nogo_cost", True)),
        nogo_penalty_type=str(g("nogo_penalty_type", "warning_band")),
        nogo_weight=float(g("nogo_weight", 2000.0)),
        nogo_safe_distance=float(g("nogo_safe_distance", 0.25)),
        nogo_logbarrier_eps=float(g("nogo_logbarrier_eps", 0.01)),
        nogo_warning_band=float(g("nogo_warning_band", 0.05)),
        nogo_near_weight=float(g("nogo_near_weight", 50.0)),
    )


def materialise_route_seeds(cfg: dict) -> None:
    """Add the concrete E3 route seeds required by the runtime planner."""

    routes = json.loads(ROUTE_SEEDS.read_text())["routes"]["four"]
    for task, task_cfg in cfg["tasks"].items():
        if "optimizer_initial_routes_json" in task_cfg:
            continue
        task_routes = routes[task]
        seeds = [
            {
                "name": name,
                "waypoints": [[float(x), float(y)] for x, y in task_routes[name]],
            }
            for name in ("availability_blind", "cad_reference")
        ]
        task_cfg["optimizer_initial_routes_json"] = json.dumps(seeds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", default=str(CAMPAIGN_PLAN))
    ap.add_argument("--out", default=str(C.OUT_ROOT / "e5_offline_efe_solve"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.campaign).read_text())
    materialise_route_seeds(cfg)
    cam = camera_params()
    apparatus = C.build_apparatus()
    out = Path(args.out)

    rows = []
    for task, tcfg in cfg["tasks"].items():
        spec = apparatus.tasks[task]
        start = spec["start"]; goal = spec["goal"]
        m0 = np.array([float(start["x"]), float(start["y"]), float(start.get("yaw", 0.0))])
        S0 = np.diag([float(cfg.get("init_belief_sigma_xy", 0.05))**2,
                      float(cfg.get("init_belief_sigma_xy", 0.05))**2,
                      float(cfg.get("init_belief_sigma_theta", 0.05))**2])
        goal_xy = (float(goal["x"]), float(goal["y"]))
        for arm, artifact in ARMS.items():
            if artifact is not None and not Path(artifact).is_file():
                print(f"  SKIP {arm}: {artifact} missing"); continue
            p = build_planner(cfg, artifact, tcfg["optimizer_initial_routes_json"], cam)
            res = p.plan(m0, S0, goal_xy, progress_index=0.0)
            sel = getattr(res, "selected_source", "?")
            rows.append(dict(task=task, arm=arm, selected=str(sel),
                             risk=float(res.risk_cost), ambiguity=float(res.ambiguity_cost),
                             obstacle=float(getattr(res, "obstacle_cost", 0.0)),
                             total=float(res.risk_cost + res.ambiguity_cost + getattr(res, "obstacle_cost", 0.0))))
            print(f"  {task:<24}{arm:<20}{str(sel)[:34]:<36}"
                  f"risk={rows[-1]['risk']:>12.1f} amb={rows[-1]['ambiguity']:>9.1f} obs={rows[-1]['obstacle']:>12.1f}")

    C.write_csv(out / "e5_offline_efe_solve.csv",
                ("task","arm","selected","risk","ambiguity","obstacle","total"), rows)
    C.write_json(out / "manifest.json", {"world": WORLD, "arms": {k: str(v) for k, v in ARMS.items()},
                                         "campaign_plan": str(Path(args.campaign)),
                                         "route_seeds": str(ROUTE_SEEDS),
                                         "objective": "runtime EFE (risk + ambiguity + nogo)",
                                         "note": "no Gazebo, no ground truth"})
    print(f"\nwrote {out}/e5_offline_efe_solve.csv")


if __name__ == "__main__":
    main()
