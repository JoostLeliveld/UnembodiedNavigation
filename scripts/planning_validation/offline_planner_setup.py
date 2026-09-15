#!/usr/bin/env python3
"""Build the deployed global planner offline (no ROS, no Gazebo).

This reproduces exactly what `efe_agent_node` constructs for its one-shot GLOBAL
stage -- `UnicyclePlannerBase(horizon=global_horizon, dt=global_dt, ...)` with the
campaign YAML's planner parameters -- so that route selection can be validated
before any simulator is launched (ground rule 1).

Geometry and camera provenance
------------------------------
* driveable region : `driveable_geometry_json` from the campaign YAML (the same
  string the launch file passes to the node).
* obstacle geometry: parsed from the world SDF with the same helper the runtime
  uses (`warehouse_walls` + `warehouse_rack_occluders` collision geometry).
* camera model     : taken from the fitted GP artifact, which records the camera
  pose/intrinsics it was fitted with. This is the same camera the runtime
  resolves from the world, and using the artifact's copy guarantees q and the
  observation model refer to one camera.

Nothing here retrains or refits any model (ground rule 2); artifacts are read
only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in ("src/planning", "src/unav_common"):
    _path = str(REPO_ROOT / _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402
from planning.core.global_route_selector import (  # noqa: E402
    ExecutionModelConfig,
    RouteCandidate,
)
from planning.core.route_safety import (  # noqa: E402
    RobotFootprint,
    RouteSafetyModel,
    SafetyGateConfig,
)
from unav_common.occlusion_geometry import (  # noqa: E402
    parse_collision_scene_from_world,
    scene_to_json,
)

DEFAULT_CAMPAIGN = REPO_ROOT / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
DEFAULT_TASKS = REPO_ROOT / "src/experiments/config/tasks.yaml"
DEFAULT_WORLD = REPO_ROOT / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"

# Condition axes of the validation grid.
Q_MODES = ("unity", "commissioned")
R_MODES = ("constant_global", "commissioned")


@dataclass
class ObservabilityCondition:
    """One (q, R) condition of the validation grid.

    q_mode:
      'unity'        -- observation availability is 1 everywhere (no GP field).
      'commissioned' -- the fitted GP availability field is used as-is.
    r_mode:
      'constant_global' -- the observation covariance does not depend on the map
                           at all (r_miss := r_visible), i.e. C1's constant R.
      'commissioned'    -- the commissioned (r_visible, r_miss) pair, so that
                           availability blends observation quality.
    `r_visible_scale` scales the measurement quality when an observation IS
    available, for the 'worsening R' sensitivity study.
    """

    q_mode: str = "commissioned"
    r_mode: str = "commissioned"
    r_visible_scale: float = 1.0

    @property
    def label(self) -> str:
        base = f"q={self.q_mode}/R={self.r_mode}"
        if abs(self.r_visible_scale - 1.0) > 1e-9:
            base += f"/Rvis_x{self.r_visible_scale:g}"
        return base

    def describe(self) -> dict:
        return {
            "q_mode": self.q_mode,
            "r_mode": self.r_mode,
            "r_visible_scale": float(self.r_visible_scale),
            "label": self.label,
        }


@dataclass
class OfflineSetup:
    """Everything a full-route validation run needs for one task/condition."""

    task_name: str
    condition: ObservabilityCondition
    planner: UnicyclePlannerBase
    safety_model: RouteSafetyModel
    execution: ExecutionModelConfig
    start_xy_yaw: np.ndarray
    goal_xy: np.ndarray
    S0: np.ndarray
    candidates: list[RouteCandidate]
    R_reference: np.ndarray
    config: dict


def load_campaign_config(path: str | os.PathLike = DEFAULT_CAMPAIGN) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_tasks(path: str | os.PathLike = DEFAULT_TASKS, world: str = "warehouse_aws.world.sdf") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    tasks = (data.get("tasks") or {}).get(world) or []
    return {str(t["name"]): t for t in tasks}


def camera_params_from_gp_artifact(gp_path: str | os.PathLike) -> dict:
    with np.load(str(gp_path), allow_pickle=False) as data:
        return {
            "cam_pos": np.asarray(data["camera_pos"], dtype=float).reshape(3).tolist(),
            "look_at": np.asarray(data["look_at"], dtype=float).reshape(3).tolist(),
            "img_width": int(np.asarray(data["img_width"]).reshape(-1)[0]),
            "img_height": int(np.asarray(data["img_height"]).reshape(-1)[0]),
            "fov_h_rad": float(np.asarray(data["fov_h_rad"], dtype=float).reshape(-1)[0]),
        }


def occlusion_geometry_from_gp_artifact(gp_path: str | os.PathLike) -> str:
    with np.load(str(gp_path), allow_pickle=False) as data:
        return str(np.asarray(data["geometry_json"]).reshape(-1)[0])


def collision_geometry_from_world(world_path: str | os.PathLike = DEFAULT_WORLD) -> str:
    scene = parse_collision_scene_from_world(
        str(world_path), model_names=("warehouse_walls", "warehouse_rack_occluders")
    )
    return scene_to_json(scene)


def build_global_planner(
    cfg: dict,
    condition: ObservabilityCondition,
    *,
    gp_artifact: str | os.PathLike | None = None,
    world_path: str | os.PathLike = DEFAULT_WORLD,
    overrides: dict[str, Any] | None = None,
) -> UnicyclePlannerBase:
    """Construct the GLOBAL-stage planner exactly as `efe_agent_node` would."""
    overrides = dict(overrides or {})
    gp_artifact = str(gp_artifact or (REPO_ROOT / cfg["gp_artifact"]))
    camera_params = camera_params_from_gp_artifact(gp_artifact)
    occlusion_json = occlusion_geometry_from_gp_artifact(gp_artifact)
    collision_json = collision_geometry_from_world(world_path)

    r_visible = float(cfg["r_visible_uv"]) * float(condition.r_visible_scale)
    r_miss = float(cfg["r_miss_uv"])
    if condition.r_mode == "constant_global":
        # Constant/global observation covariance: quality does not depend on the
        # map, so a miss costs exactly what a hit costs.
        r_miss = r_visible
    elif condition.r_mode != "commissioned":
        raise ValueError(f"unknown r_mode: {condition.r_mode}")

    if condition.q_mode == "unity":
        use_visibility_model = False
    elif condition.q_mode == "commissioned":
        use_visibility_model = True
    else:
        raise ValueError(f"unknown q_mode: {condition.q_mode}")

    kwargs = dict(
        horizon=int(cfg["global_horizon"]),
        dt=float(cfg["global_dt"]),
        v_min=0.0,
        v_max=float(cfg["v_max"]),
        w_min=-1.0,
        w_max=1.0,
        control_weight=float(cfg.get("control_weight", 0.0)),
        process_noise_xy=float(cfg["process_noise_xy"]),
        process_noise_theta=float(cfg["process_noise_theta"]),
        obs_noise_uv=2.0,
        goal_sigma_uv=2.0,
        risk_weight_obs=float(cfg["risk_weight_obs"]),
        ambiguity_weight=float(cfg["ambiguity_weight"]),
        optimizer_maxiter=int(cfg["optimizer_maxiter"]),
        optimizer_maxfun=int(cfg["optimizer_maxfun"]),
        optimizer_ftol=float(cfg["optimizer_ftol"]),
        optimizer_gtol=float(cfg["optimizer_gtol"]),
        optimizer_warm_start=True,
        # The runtime GLOBAL stage runs multistart over the lane-graph route
        # seeds (efe_agent_node passes global_optimizer_multistart); without it
        # the solve only ever sees the cold zero-control start.
        optimizer_multistart=bool(cfg.get("global_optimizer_multistart", True)),
        optimizer_multistart_include_direct=bool(
            cfg.get("optimizer_multistart_include_direct", False)
        ),
        seed=0,
        camera_params=camera_params,
        approx_method="ET1",
        use_obs_risk=True,
        use_ambiguity=bool(cfg.get("global_use_ambiguity", True)),
        use_visibility_model=use_visibility_model,
        visibility_target_height_m=0.0,
        visibility_geometry_json=occlusion_json,
        collision_geometry_json=collision_json,
        visibility_artifact_path=gp_artifact,
        r_visible_uv=r_visible,
        r_miss_uv=r_miss,
        visibility_sigma_kappa=1.0,
        goal_prior_u_std_start=float(cfg["goal_prior_u_std_start"]),
        goal_prior_v_std_start=float(cfg["goal_prior_v_std_start"]),
        goal_prior_u_std_final=float(cfg["goal_prior_u_std_final"]),
        goal_prior_v_std_final=float(cfg["goal_prior_v_std_final"]),
        goal_tightening_power=float(cfg["goal_tightening_power"]),
        goal_progress_n_steps=90,
        observation_risk_scale=float(cfg["observation_risk_scale"]),
        ambiguity_term_scale=float(cfg["ambiguity_term_scale"]),
        discount_gamma=float(cfg["discount_gamma"]),
        use_nogo_cost=bool(cfg.get("use_nogo_cost", True)),
        nogo_penalty_type=str(cfg.get("nogo_penalty_type", "warning_band")),
        nogo_weight=float(cfg.get("nogo_weight", 0.0)),
        nogo_safe_distance=float(cfg.get("nogo_safe_distance", 0.25)),
        nogo_logbarrier_eps=float(cfg.get("nogo_logbarrier_eps", 1e-2)),
        nogo_warning_band=float(cfg.get("nogo_warning_band", 0.05)),
        nogo_near_weight=float(cfg.get("nogo_near_weight", 50.0)),
        use_belief_nogo_cost=bool(cfg.get("use_belief_nogo_cost", True)),
        nogo_belief_kappa=float(cfg.get("nogo_belief_kappa", 1.0)),
        nogo_mode=str(cfg.get("nogo_mode", "keep_in")),
        driveable_geometry_json=str(cfg["driveable_geometry_json"]),
        robot_collision_radius_m=float(cfg.get("robot_collision_radius_m", 0.125)),
    )
    kwargs.update(overrides)
    return UnicyclePlannerBase(**kwargs)


def build_safety_model(
    cfg: dict,
    *,
    world_path: str | os.PathLike = DEFAULT_WORLD,
    footprint: RobotFootprint | None = None,
    gates: SafetyGateConfig | None = None,
) -> RouteSafetyModel:
    return RouteSafetyModel(
        driveable_geometry_json=str(cfg["driveable_geometry_json"]),
        obstacle_geometry_json=collision_geometry_from_world(world_path),
        footprint=footprint or RobotFootprint(),
        gates=gates or SafetyGateConfig(goal_radius_m=float(cfg.get("goal_success_radius", 0.25))),
    )


def lane_graph_candidates(cfg: dict, start_xy: Sequence[float], goal_xy: Sequence[float]) -> list[RouteCandidate]:
    """The deployed condition-neutral route seeds for this (start, goal)."""
    from unav_common.lane_graph_routes import generate_route_seeds

    seeds = generate_route_seeds(str(cfg["driveable_geometry_json"]), start_xy, goal_xy)
    return [RouteCandidate.from_dict(s) for s in seeds]


def configured_candidates(cfg: dict, task_name: str) -> list[RouteCandidate]:
    """The campaign YAML's explicit per-task route seeds, when present."""
    task_cfg = (cfg.get("tasks") or {}).get(task_name) or {}
    raw = task_cfg.get("optimizer_initial_routes_json") or cfg.get("optimizer_initial_routes_json") or ""
    if not str(raw).strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [RouteCandidate.from_dict(entry) for entry in payload if isinstance(entry, dict)]


def make_setup(
    task_name: str,
    condition: ObservabilityCondition,
    *,
    cfg: dict | None = None,
    tasks: dict | None = None,
    footprint: RobotFootprint | None = None,
    gates: SafetyGateConfig | None = None,
    candidate_source: str = "lane_graph",
    world_path: str | os.PathLike = DEFAULT_WORLD,
) -> OfflineSetup:
    cfg = cfg or load_campaign_config()
    tasks = tasks or load_tasks()
    if task_name not in tasks:
        raise KeyError(f"task '{task_name}' not found; known: {sorted(tasks)}")
    task = tasks[task_name]
    start = np.array(
        [float(task["start"]["x"]), float(task["start"]["y"]), float(task["start"].get("yaw", 0.0))],
        dtype=float,
    )
    goal = np.array([float(task["goal"]["x"]), float(task["goal"]["y"])], dtype=float)
    planner = build_global_planner(cfg, condition, world_path=world_path)
    safety_model = build_safety_model(cfg, world_path=world_path, footprint=footprint, gates=gates)
    execution = ExecutionModelConfig(
        dt=0.1,
        v_max=float(cfg["v_max"]),
        w_max=1.0,
    )
    sigma0 = float(cfg.get("init_belief_sigma_xy", 0.05))
    sigma0_theta = float(cfg.get("init_belief_sigma_theta", 0.05))
    S0 = np.diag([sigma0 ** 2, sigma0 ** 2, sigma0_theta ** 2]).astype(float)

    if candidate_source == "lane_graph":
        candidates = lane_graph_candidates(cfg, start[:2], goal)
    elif candidate_source == "configured":
        candidates = configured_candidates(cfg, task_name)
    else:
        raise ValueError(f"unknown candidate_source: {candidate_source}")

    # The reference measurement quality is the NOMINAL best-case R of the
    # campaign -- one fixed anchor shared by every condition, so the localization
    # cost of different conditions is measured on one common scale.
    r_ref = float(cfg["r_visible_uv"])
    R_reference = np.diag([r_ref ** 2, r_ref ** 2]).astype(float)

    return OfflineSetup(
        task_name=task_name,
        condition=condition,
        planner=planner,
        safety_model=safety_model,
        execution=execution,
        start_xy_yaw=start,
        goal_xy=goal,
        S0=S0,
        candidates=candidates,
        R_reference=R_reference,
        config=cfg,
    )


if __name__ == "__main__":  # pragma: no cover - smoke check
    cfg = load_campaign_config()
    tasks = load_tasks()
    print(f"tasks: {sorted(tasks)}")
    for name in sorted(tasks):
        task = tasks[name]
        start = (float(task["start"]["x"]), float(task["start"]["y"]))
        goal = (float(task["goal"]["x"]), float(task["goal"]["y"]))
        cands = lane_graph_candidates(cfg, start, goal)
        print(f"  {name:28s} {start} -> {goal}  seeds={[c.name for c in cands]}")
