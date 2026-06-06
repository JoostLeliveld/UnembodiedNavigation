#!/usr/bin/env python3
"""Offline EFE lab — evaluate the visibility-aware EFE planner without Gazebo.

This diagnostic script loads the planner with the exact parameters from a YAML
config, the GP visibility artifact, and the world geometry, then provides:

  * eval_controls   — full EFE breakdown for an arbitrary control sequence
  * run_optimizer   — run the real L-BFGS-B optimizer once (optional u_init)
  * closed_loop_offline — receding-horizon simulation against the truth dynamics
  * grid_eval       — evaluate a scalar field over the warehouse grid

The default entry point produces diagnostic plots
(GP+world overlay, nogo field, p_vis_eff field, R_plan field) into
logs/visibility_comparison/efe_offline_lab/<config_stem>/.

This is not paper evidence. AWS runs still require the full registered artifact
chain before interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import yaml

# --- Path setup so we can import the planner without ROS -------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "planning"))
sys.path.insert(0, str(REPO_ROOT / "src" / "unav_common"))
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))

from experiments.core.world_profiles import (  # noqa: E402
    resolve_world_path,
    serialize_collision_geometry_from_world,
    serialize_driveable_geometry_from_profile,
    serialize_occlusion_geometry_from_world,
)
from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402

# ---------------------------------------------------------------------------
# Config + setup loading
# ---------------------------------------------------------------------------

TASKS_PATH = REPO_ROOT / "src" / "experiments" / "config" / "tasks.yaml"
WORLD_PROFILES_PATH = REPO_ROOT / "src" / "experiments" / "config" / "world_profiles.yaml"


@dataclass
class Setup:
    config_path: Path
    config: dict
    task: dict
    profile: dict
    world: str
    task_name: str
    gp_path: Path
    gp: dict
    camera_params: dict
    geometry_json: str
    driveable_geometry_json: str
    planner: UnicyclePlannerBase
    planner_kind: str  # 'constant_R_efe' or 'visibility_aware_efe'
    start_xy_yaw: tuple
    goal_xy: tuple
    S0: np.ndarray


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _normalize_geometry_json(raw) -> str:
    """`np.load(...)['geometry_json']` returns an ndarray of strings; unwrap to one str."""
    text = str(raw)
    text = text.strip()
    # NumPy ndarray str-repr: starts with [ ' and ends with ' ]
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text


def _load_gp(npz_path: Path) -> dict:
    """Load GP artifact: visibility maps, camera info, world geometry JSON."""
    with np.load(npz_path, allow_pickle=False) as d:
        out: dict[str, Any] = {
            "xs": np.asarray(d["xs"], dtype=float),
            "ys": np.asarray(d["ys"], dtype=float),
            "P_plan": np.asarray(d["P_conservative_plan_map"], dtype=float),
            "P_mean": np.asarray(d["P_mean_map"], dtype=float) if "P_mean_map" in d.files else None,
            "F_std": np.asarray(d["F_std_map"], dtype=float) if "F_std_map" in d.files else None,
            "X_train": np.asarray(d["X_train"], dtype=float) if "X_train" in d.files else None,
            "p_train": np.asarray(d["p_train"], dtype=float) if "p_train" in d.files else None,
            "camera_pos": np.asarray(d["camera_pos"], dtype=float).reshape(-1),
            "camera_pose": np.asarray(d["camera_pose"], dtype=float).reshape(-1),
            "look_at": np.asarray(d["look_at"], dtype=float).reshape(-1),
            "img_width": int(np.asarray(d["img_width"]).reshape(-1)[0]),
            "img_height": int(np.asarray(d["img_height"]).reshape(-1)[0]),
            "fov_h_rad": float(np.asarray(d["fov_h_rad"]).reshape(-1)[0]),
            "target_height": float(np.asarray(d["target_height"]).reshape(-1)[0]),
            "geometry_json": _normalize_geometry_json(d["geometry_json"]),
        }
    return out


def _find_task(world: str, task_name: str) -> dict:
    payload = _load_yaml(TASKS_PATH)
    tasks = list((payload.get("tasks") or {}).get(world) or [])
    for t in tasks:
        if t.get("name") == task_name:
            return t
    raise RuntimeError(f"task {task_name!r} not found for world {world!r}")


def _planner_kind_from_label(cfg: dict, condition: str) -> str:
    conds = cfg.get("conditions", {}) or {}
    return str(conds.get(condition, {}).get("planner", "visibility_aware_efe"))


def build_planner(
    cfg: dict,
    *,
    planner_kind: str,
    camera_params: dict,
    visibility_artifact_path: str,
    visibility_geometry_json: str,
    collision_geometry_json: str,
    driveable_geometry_json: str = '',
    seed: int = 1,
    visibility_target_height_m: float = 0.35,
) -> UnicyclePlannerBase:
    """Instantiate UnicyclePlannerBase with the same flags the launcher uses."""
    use_visibility_model = True
    use_ambiguity = True
    use_obs_risk = True
    if planner_kind == "constant_R_efe":
        use_visibility_model = False
        # C1 still optimizes the EFE ambiguity term; it simply uses a constant
        # observation covariance instead of the GP-conditioned covariance.
        use_ambiguity = True
        use_obs_risk = True
    elif planner_kind == "risk_only_ablation":
        use_visibility_model = True
        use_ambiguity = False
        use_obs_risk = True
    # Diagnostic override: allow exercising a state-space (no observation-risk)
    # local tracker by setting use_obs_risk/use_ambiguity in cfg.
    if "use_obs_risk" in cfg:
        use_obs_risk = _as_bool(cfg["use_obs_risk"])
    if "use_ambiguity" in cfg:
        use_ambiguity = _as_bool(cfg["use_ambiguity"])

    v_max = float(cfg.get("v_max", 0.22))
    v_min = -v_max
    w_max = float(cfg.get("w_max", 1.0))
    w_min = -w_max

    planner = UnicyclePlannerBase(
        horizon=int(cfg.get("horizon", 40)),
        dt=float(cfg.get("dt", 0.25)),
        v_min=v_min,
        v_max=v_max,
        w_min=w_min,
        w_max=w_max,
        control_weight=float(cfg.get("control_weight", 0.0)),
        process_noise_xy=float(cfg.get("process_noise_xy", 0.01)),
        process_noise_theta=float(cfg.get("process_noise_theta", 0.02)),
        obs_noise_uv=float(cfg.get("r_visible_uv", 2.5)),
        goal_sigma_uv=float(cfg.get("goal_prior_u_std_final", 20.0)),
        risk_weight_obs=float(cfg.get("risk_weight_obs", 1.0)),
        ambiguity_weight=float(cfg.get("ambiguity_weight", 6.0)),
        optimizer_maxiter=int(cfg.get("optimizer_maxiter", 150)),
        optimizer_gtol=float(cfg.get("optimizer_gtol", 1e-4)),
        optimizer_warm_start=bool(cfg.get("optimizer_warm_start", True)),
        approx_method=cfg.get("approx_method", "ET1"),
        use_obs_risk=use_obs_risk,
        use_ambiguity=use_ambiguity,
        seed=seed,
        camera_params=camera_params,
        use_visibility_model=use_visibility_model,
        visibility_target_height_m=float(cfg.get("visibility_target_height_m", visibility_target_height_m)),
        visibility_geometry_json=visibility_geometry_json,
        collision_geometry_json=collision_geometry_json,
        visibility_artifact_path=visibility_artifact_path if use_visibility_model else "",
        r_visible_uv=float(cfg.get("r_visible_uv", 2.5)),
        r_miss_uv=float(cfg.get("r_miss_uv", 15.0)),
        visibility_sigma_kappa=float(cfg.get("visibility_sigma_kappa", 1.0)),
        goal_prior_u_std_start=float(cfg.get("goal_prior_u_std_start", 80.0)),
        goal_prior_v_std_start=float(cfg.get("goal_prior_v_std_start", 80.0)),
        goal_prior_u_std_final=float(cfg.get("goal_prior_u_std_final", 20.0)),
        goal_prior_v_std_final=float(cfg.get("goal_prior_v_std_final", 20.0)),
        goal_tightening_power=float(cfg.get("goal_tightening_power", 0.45)),
        goal_progress_n_steps=int(cfg.get("goal_progress_n_steps", 90)),
        goal_progress_weight=float(cfg.get("goal_progress_weight", 0.0)),
        ref_weight=float(cfg.get("ref_weight", 0.0)),
        terminal_ref_weight=float(cfg.get("terminal_ref_weight", 0.0)),
        du_weight=float(cfg.get("du_weight", 0.0)),
        observation_risk_scale=float(cfg.get("observation_risk_scale", 1.25)),
        ambiguity_term_scale=float(cfg.get("ambiguity_term_scale", 1.0)),
        discount_gamma=float(cfg.get("discount_gamma", 0.98)),
        optimizer_maxfun=int(cfg.get("optimizer_maxfun", 900)),
        optimizer_ftol=float(cfg.get("optimizer_ftol", 1e-6)),
        optimizer_multistart=_as_bool(cfg.get("optimizer_multistart", False)),
        optimizer_multistart_include_direct=_as_bool(
            cfg.get("optimizer_multistart_include_direct", True)
        ),
        optimizer_multistart_lateral_offsets=cfg.get("optimizer_multistart_lateral_offsets", ""),
        optimizer_initial_routes_json=cfg.get("optimizer_initial_routes_json", ""),
        use_nogo_cost=_as_bool(cfg.get("use_nogo_cost", True)),
        nogo_penalty_type=str(cfg.get("nogo_penalty_type", "softplus")),
        nogo_weight=float(cfg.get("nogo_weight", 80.0)),
        nogo_safe_distance=float(cfg.get("nogo_safe_distance", 0.35)),
        nogo_gaussian_sigma=float(cfg.get("nogo_gaussian_sigma", 0.25)),
        nogo_softplus_scale=float(cfg.get("nogo_softplus_scale", 0.08)),
        nogo_logbarrier_scale=float(cfg.get("nogo_logbarrier_scale", 0.25)),
        nogo_logbarrier_eps=float(cfg.get("nogo_logbarrier_eps", 1e-3)),
        use_belief_nogo_cost=_as_bool(cfg.get("use_belief_nogo_cost", False)),
        nogo_belief_kappa=float(cfg.get("nogo_belief_kappa", 1.0)),
        nogo_mode=str(cfg.get("nogo_mode", "keep_out")),
        driveable_geometry_json=str(driveable_geometry_json or cfg.get("driveable_geometry_json", "")),
        robot_collision_radius_m=float(cfg.get("robot_collision_radius_m", 0.125)),
        runtime_debug=False,
    )
    return planner


def load_setup(
    config_path: Path,
    *,
    condition: str = "C2",
    seed: int = 1,
    task_override: Optional[str] = None,
) -> Setup:
    """Load full setup (planner, GP, task, world) from a campaign YAML."""
    cfg = _load_yaml(config_path)
    world = str(cfg["world"])
    profiles = _load_yaml(WORLD_PROFILES_PATH)
    profile = dict((profiles.get("worlds") or {}).get(world) or {})

    # task name — pick the first task in the campaign config unless overridden
    task_name = task_override or next(iter(cfg.get("tasks", {}).keys()))
    task = _find_task(world, task_name)

    gp_path = Path(cfg["gp_artifact"]).expanduser()
    if not gp_path.is_absolute():
        gp_path = (REPO_ROOT / gp_path).resolve()
    gp = _load_gp(gp_path)
    world_path = resolve_world_path(world)
    current_visibility_geometry_json = serialize_occlusion_geometry_from_world(world_path)
    current_collision_geometry_json = serialize_collision_geometry_from_world(world_path)
    current_driveable_geometry_json = serialize_driveable_geometry_from_profile(profile)
    cfg = dict(cfg)
    cfg["driveable_geometry_json"] = current_driveable_geometry_json

    camera_params = {
        "cam_pos": tuple(gp["camera_pos"][:3].tolist()),
        "look_at": tuple(gp["look_at"][:3].tolist()),
        "img_width": gp["img_width"],
        "img_height": gp["img_height"],
        "fov_h_rad": gp["fov_h_rad"],
    }
    planner_kind = _planner_kind_from_label(cfg, condition)

    # Initial belief covariance — matches the planner_node default
    sigma_xy = float(cfg.get("init_belief_sigma_xy", 0.15))
    sigma_th = float(cfg.get("init_belief_sigma_theta", 0.05))
    S0 = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_th ** 2])

    planner = build_planner(
        cfg,
        planner_kind=planner_kind,
        camera_params=camera_params,
        visibility_artifact_path=str(gp_path),
        visibility_geometry_json=current_visibility_geometry_json,
        collision_geometry_json=current_collision_geometry_json,
        driveable_geometry_json=current_driveable_geometry_json,
        seed=seed,
        visibility_target_height_m=gp["target_height"],
    )

    start = task["start"]
    goal = task["goal"]
    return Setup(
        config_path=config_path,
        config=cfg,
        task=task,
        profile=profile,
        world=world,
        task_name=task_name,
        gp_path=gp_path,
        gp=gp,
        camera_params=camera_params,
        geometry_json=current_collision_geometry_json,
        driveable_geometry_json=current_driveable_geometry_json,
        planner=planner,
        planner_kind=planner_kind,
        start_xy_yaw=(float(start["x"]), float(start["y"]), float(start["yaw"])),
        goal_xy=(float(goal["x"]), float(goal["y"])),
        S0=S0,
    )


# ---------------------------------------------------------------------------
# Eval primitives
# ---------------------------------------------------------------------------

def eval_controls(
    planner: UnicyclePlannerBase,
    m0: np.ndarray,
    S0: np.ndarray,
    goal_xy: tuple,
    controls: np.ndarray,
) -> dict:
    """Wrap evaluate_rollout_controls — returns the full per-step trace + totals."""
    result = planner.evaluate_rollout_controls(
        np.asarray(m0, dtype=float).reshape(3),
        np.asarray(S0, dtype=float).reshape(3, 3),
        np.asarray(goal_xy, dtype=float).reshape(2),
        np.asarray(controls, dtype=float).reshape(-1, 2),
    )
    return result


def run_optimizer(
    planner: UnicyclePlannerBase,
    m0: np.ndarray,
    S0: np.ndarray,
    goal_xy: tuple,
    *,
    u_init: Optional[np.ndarray] = None,
) -> dict:
    """Run planner.plan() once; optionally with a custom u_init seed."""
    # Force-clear any persistent warm start, then optionally seed with u_init.
    planner.prev_controls_flat = None if u_init is None else np.asarray(u_init, dtype=float).reshape(-1)
    planner._prev_goal_xy = None
    plan = planner.plan(
        np.asarray(m0, dtype=float).reshape(3),
        np.asarray(S0, dtype=float).reshape(3, 3),
        np.asarray(goal_xy, dtype=float).reshape(2),
    )
    return plan


def closed_loop_offline(
    planner: UnicyclePlannerBase,
    m0: np.ndarray,
    S0: np.ndarray,
    goal_xy: tuple,
    n_steps: int,
    *,
    warm_start: bool = True,
    reset_each_step: bool = False,
    initial_u: Optional[np.ndarray] = None,
) -> dict:
    """Receding-horizon simulation against truth dynamics (no Gazebo).

    `initial_u`: optional starting warm-start for the very first plan call.
    Useful for testing whether the A3 basin is stable once the warm-start lives there.
    """
    planner.optimizer_warm_start = bool(warm_start)
    if reset_each_step or not warm_start:
        planner.prev_controls_flat = None
    elif initial_u is not None:
        planner.prev_controls_flat = np.asarray(initial_u, dtype=float).reshape(-1)
    else:
        planner.prev_controls_flat = None
    planner._prev_goal_xy = None

    m = np.asarray(m0, dtype=float).reshape(3).copy()
    S = np.asarray(S0, dtype=float).reshape(3, 3).copy()
    goal_xy_arr = np.asarray(goal_xy, dtype=float).reshape(2)

    traj_m, traj_S = [m.copy()], [S.copy()]
    plans, infos = [], []

    for step in range(n_steps):
        if reset_each_step:
            planner.prev_controls_flat = None
        plan = planner.plan(m, S, goal_xy_arr)
        plans.append(plan)
        # First control is what gets "executed" in the real loop.
        u0 = np.asarray(plan.controls[0], dtype=float)
        m, S = planner.predict(m, S, u0)
        traj_m.append(m.copy())
        traj_S.append(S.copy())
        # Distance to goal (truth, in m)
        d_goal = float(np.linalg.norm(m[:2] - goal_xy_arr))
        infos.append({
            "step": step,
            "u": u0.tolist(),
            "m_after": m.tolist(),
            "d_goal": d_goal,
            "solver_time_ms": float(plan.solve_time_s) * 1000.0,
            "efe_total": float(plan.total_cost),
            "efe_risk": float(plan.risk_cost),
            "efe_amb": float(plan.ambiguity_cost),
            "efe_obstacle": float(plan.obstacle_cost),
        })
        if d_goal < float(planner.dt * planner.v_max):
            break

    return {
        "traj_m": np.asarray(traj_m, dtype=float),
        "traj_S": np.asarray(traj_S, dtype=float),
        "plans": plans,
        "infos": infos,
    }


def grid_eval(
    fn: Callable[[np.ndarray], float],
    xs: np.ndarray,
    ys: np.ndarray,
    yaw: float = math.pi / 2,
) -> np.ndarray:
    """Evaluate fn(m=[x,y,yaw]) on a grid. Returns (ny, nx)."""
    out = np.full((ys.size, xs.size), np.nan, dtype=float)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            try:
                out[j, i] = float(fn(np.array([x, y, yaw], dtype=float)))
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Phase A — diagnostic plots
# ---------------------------------------------------------------------------

RACK_COLOR = "#666666"
RACK_EDGE = "#222222"


def _parse_prisms_from_geometry_json(geometry_json: str) -> list[dict]:
    if not geometry_json:
        return []
    try:
        payload = json.loads(geometry_json)
    except Exception:
        return []
    return list(payload.get("prisms") or [])


def _draw_world(ax, prisms, *, alpha=0.78, label_high_stack=True):
    """Draw rack rectangles and label the R4 high stack."""
    for p in prisms:
        name = str(p.get("name", "")).split("/")[-1].split(":")[0]
        is_stack = "high_stack" in name
        is_crate = name.startswith("low_crate")
        face = "#cc6633" if is_stack else (RACK_COLOR if not is_crate else "#aaaaaa")
        edge = RACK_EDGE
        rect = Rectangle(
            (p["xmin"], p["ymin"]),
            p["xmax"] - p["xmin"],
            p["ymax"] - p["ymin"],
            facecolor=face,
            edgecolor=edge,
            linewidth=0.5,
            alpha=alpha,
            zorder=10,
        )
        ax.add_patch(rect)
        if label_high_stack and "high_stack_lower" in name:
            cx = 0.5 * (p["xmin"] + p["xmax"])
            cy = 0.5 * (p["ymin"] + p["ymax"])
            ax.annotate(
                "R4 stack\n(occluder)",
                (cx, cy),
                xytext=(cx + 0.7, cy + 0.8),
                fontsize=7,
                color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                zorder=12,
            )


def _draw_task_markers(ax, setup: Setup, *, draw_lookahead: bool = False):
    sx, sy, _ = setup.start_xy_yaw
    gx, gy = setup.goal_xy
    ax.scatter([sx], [sy], s=70, c="#1976d2", edgecolor="white", linewidth=0.8, zorder=20, label="start")
    ax.scatter([gx], [gy], s=80, c="#1b9e5a", edgecolor="white", linewidth=0.8, marker="*", zorder=20, label="goal")
    cam = setup.gp["camera_pos"]
    ax.scatter([cam[0]], [cam[1]], marker="v", s=60, c="black", edgecolor="white", linewidth=0.6, zorder=20, label="camera")
    if draw_lookahead:
        h = setup.planner.horizon
        dt = setup.planner.dt
        vmax = setup.planner.v_max
        reach = h * dt * vmax
        circ = Circle((sx, sy), reach, fill=False, edgecolor="#1976d2", linewidth=1.0, linestyle="--", zorder=15)
        ax.add_patch(circ)
        ax.text(
            sx + 0.1, sy + reach + 0.1,
            f"lookahead = {reach:.2f} m  (H={h}, dt={dt}, v={vmax})",
            fontsize=7, color="#1976d2",
        )


def _format(ax, gp):
    xs, ys = gp["xs"], gp["ys"]
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(float(ys[0]), float(ys[-1]))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(color="#dddddd", linewidth=0.3, alpha=0.5)


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Phase B — candidate trajectory construction + EFE comparison
# ---------------------------------------------------------------------------

def _wrap_pi(angle: float) -> float:
    return ((angle + math.pi) % (2 * math.pi)) - math.pi


def make_controls_to_heading(
    horizon: int, dt: float, v_max: float, *,
    yaw_start: float, yaw_target: float, w_turn: float = 1.0,
) -> np.ndarray:
    """Turn-in-place from yaw_start to yaw_target, then drive at v_max forward."""
    yaw_err = _wrap_pi(yaw_target - yaw_start)
    n_full = int(abs(yaw_err) / (w_turn * dt))
    remaining = abs(yaw_err) - n_full * w_turn * dt
    has_partial = remaining > 1e-6
    n_turn = min(horizon, n_full + (1 if has_partial else 0))
    sign = math.copysign(1.0, yaw_err) if abs(yaw_err) > 1e-9 else 0.0

    controls = np.zeros((horizon, 2), dtype=float)
    for k in range(min(n_full, horizon)):
        controls[k, 1] = sign * w_turn
    if has_partial and n_full < horizon:
        controls[n_full, 1] = sign * (remaining / dt)
    if n_turn < horizon:
        controls[n_turn:, 0] = v_max
    return controls


def make_controls_west_then_north(
    horizon: int, dt: float, v_max: float, *,
    yaw_start: float, distance_west: float, w_turn: float = 1.0,
) -> np.ndarray:
    """Turn west, drive west `distance_west` m, then turn north, then drive forward."""
    # Phase 1: turn from yaw_start to yaw=π (west)
    c1 = make_controls_to_heading(
        horizon, dt, v_max, yaw_start=yaw_start, yaw_target=math.pi, w_turn=w_turn,
    )
    # Count turn steps used
    turn_steps_1 = int(np.sum(np.abs(c1[:, 1]) > 1e-9))
    # Then drive west for some steps
    n_drive_west = int(round(distance_west / max(v_max * dt, 1e-6)))
    n_drive_west = min(n_drive_west, horizon - turn_steps_1)
    # Phase 2: turn from yaw=π to yaw=π/2 (north)
    turn_steps_2 = int(math.ceil(math.pi / 2.0 / (w_turn * dt)))
    # Phase 3: drive north for remaining steps
    controls = np.zeros((horizon, 2), dtype=float)
    idx = 0
    # Use turn-from-yaw_start segments (preserved from c1)
    for k in range(turn_steps_1):
        controls[idx] = c1[k]
        idx += 1
        if idx >= horizon:
            return controls
    # Drive west
    for _ in range(n_drive_west):
        controls[idx] = [v_max, 0.0]
        idx += 1
        if idx >= horizon:
            return controls
    # Turn north
    for _ in range(turn_steps_2):
        if idx >= horizon:
            break
        controls[idx] = [0.0, -w_turn]
        idx += 1
    # Drive north
    while idx < horizon:
        controls[idx] = [v_max, 0.0]
        idx += 1
    return controls


def candidate_trajectories(setup: Setup, *, override_horizon: int = None,
                           override_v_max: float = None) -> dict:
    """Return a dict of {name: controls} for the canonical A4/A3 candidates."""
    horizon = override_horizon if override_horizon else setup.planner.horizon
    v_max = override_v_max if override_v_max else setup.planner.v_max
    dt = setup.planner.dt
    yaw_start = setup.start_xy_yaw[2]
    sx, sy, _ = setup.start_xy_yaw
    gx, gy = setup.goal_xy
    yaw_goal = math.atan2(gy - sy, gx - sx)

    out = {
        "A4_due_north": make_controls_to_heading(
            horizon, dt, v_max, yaw_start=yaw_start, yaw_target=math.pi / 2,
        ),
        "to_goal_direct": make_controls_to_heading(
            horizon, dt, v_max, yaw_start=yaw_start, yaw_target=yaw_goal,
        ),
        "A3_west_then_north": make_controls_west_then_north(
            horizon, dt, v_max, yaw_start=yaw_start, distance_west=2.20,
        ),
        "A3_west_partial": make_controls_west_then_north(
            horizon, dt, v_max, yaw_start=yaw_start, distance_west=1.10,
        ),
    }
    return out


def per_step_trace(planner, m0, S0, goal_xy, controls):
    """Forward propagate (m_t, S_t), recording p_vis_eff, r_plan, signed-distance."""
    m = np.asarray(m0, dtype=float).reshape(3).copy()
    S = np.asarray(S0, dtype=float).reshape(3, 3).copy()
    states, p_vis_eff, r_plan_u, sdist = [m[:2].copy()], [], [], []
    for u in controls:
        m, S = planner.predict(m, S, u)
        states.append(m[:2].copy())
        diag = planner.planning_visibility_diagnostics(m, S)
        p_vis_eff.append(float(diag["p_vis_eff"]))
        r_plan_u.append(float(diag["r_plan_u_std"]))
        if planner.nogo_cost_model is not None:
            sdist.append(float(planner.nogo_cost_model.signed_distance_state_np(m)))
        else:
            sdist.append(float("nan"))
    return {
        "states": np.asarray(states, dtype=float),
        "p_vis_eff": np.asarray(p_vis_eff, dtype=float),
        "r_plan_u": np.asarray(r_plan_u, dtype=float),
        "signed_distance": np.asarray(sdist, dtype=float),
    }


def plot_phase_b(setup: Setup, out_dir: Path, *, param_sweeps: list[tuple] = None) -> None:
    """Phase B — candidate trajectory comparison.

    Parameters
    ----------
    setup : Setup
        Baseline setup; we'll vary (W, horizon) by rebuilding planners off this.
    param_sweeps : list of (W, horizon) tuples
        EFE evaluation parameter sets. We construct candidate trajectories at the
        max horizon and reuse them, evaluating EFE per planner.
    """
    if param_sweeps is None:
        param_sweeps = [
            ("probe_v36 (W=10,h=60)", 10.0, 60),
            ("W=6,h=40 (smoke)", 6.0, 40),
            ("W=6,h=60", 6.0, 60),
            ("W=6,h=80", 6.0, 80),
            ("W=6,h=100", 6.0, 100),
            ("W=15,h=80", 15.0, 80),
        ]

    out_dir.mkdir(parents=True, exist_ok=True)

    # We need a planner *per* (W, horizon) to evaluate EFE. Build them now.
    cfg_base = dict(setup.config)
    planners = {}
    for label, W, H in param_sweeps:
        cfg_var = dict(cfg_base)
        cfg_var["ambiguity_weight"] = W
        cfg_var["horizon"] = H
        # Build with the same world + camera, visibility model on
        planner = build_planner(
            cfg_var,
            planner_kind="visibility_aware_efe",
            camera_params=setup.camera_params,
            visibility_artifact_path=str(setup.gp_path),
            visibility_geometry_json=setup.geometry_json,
            collision_geometry_json=setup.geometry_json,
            seed=1,
            visibility_target_height_m=setup.gp["target_height"],
        )
        planners[label] = (W, H, planner)

    # Build candidate trajectories at the maximum horizon — we'll use the truncated
    # leading slice for shorter horizons.
    max_h = max(H for _, H, _ in planners.values())
    base_for_traj = build_planner(
        {**cfg_base, "horizon": max_h},
        planner_kind="visibility_aware_efe",
        camera_params=setup.camera_params,
        visibility_artifact_path=str(setup.gp_path),
        visibility_geometry_json=setup.geometry_json,
        collision_geometry_json=setup.geometry_json,
        seed=1,
        visibility_target_height_m=setup.gp["target_height"],
    )
    candidates_max = candidate_trajectories(
        Setup(**{**setup.__dict__, "planner": base_for_traj})
    )

    m0 = np.array([setup.start_xy_yaw[0], setup.start_xy_yaw[1], setup.start_xy_yaw[2]])
    S0 = setup.S0
    goal_xy = setup.goal_xy

    # ---- Plot 1 — candidate trajectories on GP background ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    P = setup.gp["P_plan"]
    xs, ys = setup.gp["xs"], setup.gp["ys"]
    im = ax.imshow(P, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                   cmap="viridis", vmin=0, vmax=1.0, aspect="equal", zorder=1)
    _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json), alpha=0.55)
    _draw_task_markers(ax, setup, draw_lookahead=True)
    _format(ax, setup.gp)
    colors = {"A4_due_north": "#d62728", "to_goal_direct": "#ff7f0e",
              "A3_west_then_north": "#1f77b4", "A3_west_partial": "#9467bd"}
    for name, controls in candidates_max.items():
        trace = per_step_trace(base_for_traj, m0, S0, goal_xy, controls)
        st = trace["states"]
        ax.plot(st[:, 0], st[:, 1], color=colors.get(name, "white"), lw=1.5,
                label=name, zorder=15)
        ax.scatter([st[-1, 0]], [st[-1, 1]], s=50, c=colors.get(name, "white"),
                   edgecolor="black", linewidth=0.6, zorder=16, marker="x")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P_conservative_plan")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Phase B.1 — candidate trajectories (horizon={max_h}, v_max={base_for_traj.v_max})\n"
                 f"start={setup.start_xy_yaw[:2]}  goal={setup.goal_xy}")
    _save(fig, out_dir, "B1_candidate_trajectories")

    # ---- Plot 2 — per-step p_vis_eff and r_plan along each candidate ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax_p, ax_r = axes
    for name, controls in candidates_max.items():
        trace = per_step_trace(base_for_traj, m0, S0, goal_xy, controls)
        t_axis = np.arange(1, len(trace["p_vis_eff"]) + 1) * base_for_traj.dt
        ax_p.plot(t_axis, trace["p_vis_eff"], label=name, color=colors.get(name, "gray"))
        ax_r.plot(t_axis, trace["r_plan_u"], label=name, color=colors.get(name, "gray"))
    ax_p.set_ylabel("p_vis_eff")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.axhline(0.3, color="red", lw=0.5, linestyle="--", label="shadow threshold")
    ax_p.grid(alpha=0.4)
    ax_p.legend(loc="lower left", fontsize=7)
    ax_p.set_title("Phase B.2 — visibility and observation noise along each candidate")
    ax_r.set_ylabel("r_plan_u (px)")
    ax_r.set_xlabel("time (s)")
    ax_r.grid(alpha=0.4)
    _save(fig, out_dir, "B2_pvis_rplan_per_candidate")

    # ---- Plot 3 — EFE breakdown bar chart for each (W, horizon) setting ----
    # First, evaluate EVERYTHING and collect rows for CSV; then plot VALID-only
    # so the barrier cost from to_goal_direct doesn't crush the y-axis.
    rows = []
    eval_cache: dict[str, dict] = {}
    for label, (W, H, p) in planners.items():
        for name, ctrls_all in candidates_max.items():
            ctrls = np.asarray(ctrls_all, dtype=float)[:H]
            res = eval_controls(p, m0, S0, goal_xy, ctrls)
            rows.append({
                "setting": label, "W": W, "horizon": H, "trajectory": name,
                "total_cost": res["total_cost"], "risk_cost": res["risk_cost"],
                "ambiguity_cost": res["ambiguity_cost"], "obstacle_cost": res["obstacle_cost"],
                "mean_p_vis": res["mean_p_vis_plan_eff"],
                "terminal_progress_m": res["terminal_goal_progress_m"],
                "rollout_valid": res["rollout_valid"],
                "invalid_reason": res["invalid_reason"],
            })
            eval_cache[(label, name)] = res

    n_settings = len(planners)
    fig, axes = plt.subplots(2, n_settings, figsize=(4.0 * n_settings, 9), sharex="col")
    if n_settings == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for ax_idx, (label, (W, H, p)) in enumerate(planners.items()):
        ax_top = axes[0][ax_idx]
        ax_bot = axes[1][ax_idx]
        names = list(candidates_max.keys())
        valid_names = [n for n in names if eval_cache[(label, n)]["rollout_valid"]]
        invalid_names = [n for n in names if not eval_cache[(label, n)]["rollout_valid"]]
        # Top: stacked bar of risk + amb + obstacle for VALID candidates only
        n_v = len(valid_names)
        xs_bar = np.arange(n_v)
        per_risk = [eval_cache[(label, n)]["risk_cost"] for n in valid_names]
        per_amb = [eval_cache[(label, n)]["ambiguity_cost"] for n in valid_names]
        per_obs = [eval_cache[(label, n)]["obstacle_cost"] for n in valid_names]
        per_total = [eval_cache[(label, n)]["total_cost"] for n in valid_names]
        ax_top.bar(xs_bar, per_risk, color="#d62728", label="risk")
        ax_top.bar(xs_bar, per_amb, bottom=per_risk, color="#1f77b4", label="ambiguity")
        ax_top.bar(xs_bar, per_obs, bottom=np.array(per_risk) + np.array(per_amb),
                   color="#666666", label="obstacle")
        ax_top.scatter(xs_bar, per_total, color="black", marker="x", s=40, zorder=10)
        ax_top.set_xticks(xs_bar)
        ax_top.set_xticklabels(valid_names, rotation=30, ha="right", fontsize=7)
        # Highlight the minimum (optimizer's preferred valid candidate)
        if per_total:
            min_idx = int(np.argmin(per_total))
            ax_top.axvspan(min_idx - 0.4, min_idx + 0.4, color="#ffcc99", alpha=0.45, zorder=0)
        ax_top.set_title(f"{label}\n(invalid: {','.join(invalid_names) or 'none'})", fontsize=8)
        ax_top.grid(axis="y", alpha=0.4)
        for xi, t in zip(xs_bar, per_total):
            ax_top.text(xi, t * 1.02, f"{t:.0f}", ha="center", fontsize=6.5)
        if ax_idx == 0:
            ax_top.set_ylabel("EFE (valid candidates only)")
            ax_top.legend(fontsize=7, loc="upper right")
        # Bottom: Δrisk vs Δamb table (A3 minus A4 by convention)
        if "A4_due_north" in valid_names:
            ref = eval_cache[(label, "A4_due_north")]
            cmp_rows = []
            for n in valid_names:
                if n == "A4_due_north":
                    continue
                c = eval_cache[(label, n)]
                cmp_rows.append((n, c["risk_cost"] - ref["risk_cost"],
                                 c["ambiguity_cost"] - ref["ambiguity_cost"],
                                 c["total_cost"] - ref["total_cost"]))
            if cmp_rows:
                labels_cmp = [r[0] for r in cmp_rows]
                d_risk = [r[1] for r in cmp_rows]
                d_amb = [r[2] for r in cmp_rows]
                d_tot = [r[3] for r in cmp_rows]
                xs_cmp = np.arange(len(cmp_rows))
                width = 0.27
                ax_bot.bar(xs_cmp - width, d_risk, width, color="#d62728", label="Δrisk")
                ax_bot.bar(xs_cmp, d_amb, width, color="#1f77b4", label="Δambiguity")
                ax_bot.bar(xs_cmp + width, d_tot, width, color="black", label="Δtotal")
                ax_bot.axhline(0, color="gray", lw=0.5)
                ax_bot.set_xticks(xs_cmp)
                ax_bot.set_xticklabels(labels_cmp, rotation=20, ha="right", fontsize=7)
                ax_bot.set_title("delta vs A4_due_north (positive => A3 worse)", fontsize=8)
                ax_bot.grid(axis="y", alpha=0.4)
                if ax_idx == 0:
                    ax_bot.set_ylabel("EFE difference")
                    ax_bot.legend(fontsize=7)
    fig.suptitle(f"Phase B.3 — EFE breakdown (top) and A3-minus-A4 deltas (bottom)\n"
                 f"task: {setup.task_name}, start={setup.start_xy_yaw[:2]}, goal={setup.goal_xy}",
                 fontsize=10)
    _save(fig, out_dir, "B3_efe_breakdown_per_setting")

    # CSV dump for raw numbers
    import csv as _csv
    csv_path = out_dir / "B3_efe_breakdown.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  wrote {csv_path}")

    # ---- Plot 4 — run the real L-BFGS optimizer once and overlay its trajectory ----
    fig, axes = plt.subplots(1, n_settings, figsize=(5.5 * n_settings, 5.5),
                             squeeze=False)
    opt_rows = []
    for ax_idx, (label, (W, H, p)) in enumerate(planners.items()):
        ax = axes[0][ax_idx]
        # Background GP
        P = setup.gp["P_plan"]
        xs, ys = setup.gp["xs"], setup.gp["ys"]
        ax.imshow(P, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                  cmap="viridis", vmin=0, vmax=1.0, aspect="equal", zorder=1)
        _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json),
                    alpha=0.50, label_high_stack=False)
        _draw_task_markers(ax, setup, draw_lookahead=False)

        # Show the candidate trajectories thinly for context
        for name, ctrls_all in candidates_max.items():
            ctrls = np.asarray(ctrls_all, dtype=float)[:H]
            trace = per_step_trace(p, m0, S0, goal_xy, ctrls)
            st = trace["states"]
            ax.plot(st[:, 0], st[:, 1], color=colors.get(name, "white"), lw=0.7,
                    alpha=0.6, zorder=10)

        # Run the optimizer
        try:
            plan_result = run_optimizer(p, m0, S0, goal_xy)
            opt_controls = np.asarray(plan_result.controls, dtype=float)
            opt_trace = per_step_trace(p, m0, S0, goal_xy, opt_controls)
            opt_states = opt_trace["states"]
            opt_efe = float(plan_result.total_cost)
            opt_risk = float(plan_result.risk_cost)
            opt_amb = float(plan_result.ambiguity_cost)
            opt_obs = float(plan_result.obstacle_cost)
            opt_solver_time = float(plan_result.solve_time_s) * 1000.0
            opt_valid = bool(plan_result.rollout_valid)
            opt_nit = int(plan_result.optimizer_nit)
            opt_nfev = int(plan_result.optimizer_nfev)
            opt_success = bool(plan_result.optimizer_success)
            ax.plot(opt_states[:, 0], opt_states[:, 1], color="white", lw=2.5,
                    label="L-BFGS converged", zorder=20, path_effects=None)
            ax.plot(opt_states[:, 0], opt_states[:, 1], color="black", lw=1.0,
                    zorder=21)
            ax.scatter([opt_states[-1, 0]], [opt_states[-1, 1]],
                       s=80, c="white", edgecolor="black", linewidth=1.0,
                       zorder=22, marker="o")
            opt_rows.append({
                "setting": label, "W": W, "horizon": H,
                "total_cost": opt_efe, "risk_cost": opt_risk,
                "ambiguity_cost": opt_amb, "obstacle_cost": opt_obs,
                "solver_time_ms": opt_solver_time, "valid": opt_valid,
                "optimizer_success": opt_success, "optimizer_nit": opt_nit,
                "optimizer_nfev": opt_nfev,
                "x_min": float(opt_states[:, 0].min()),
                "x_max": float(opt_states[:, 0].max()),
                "y_min": float(opt_states[:, 1].min()),
                "y_max": float(opt_states[:, 1].max()),
                "terminal_x": float(opt_states[-1, 0]),
                "terminal_y": float(opt_states[-1, 1]),
            })
            ax.set_title(
                f"{label}\nEFE={opt_efe:.0f}, t={opt_solver_time:.0f}ms, nit={opt_nit}/nfev={opt_nfev}\n"
                f"x∈[{opt_states[:, 0].min():.2f},{opt_states[:, 0].max():.2f}], "
                f"y_term={opt_states[-1, 1]:.2f}, valid={opt_valid}",
                fontsize=7
            )
        except Exception as exc:
            ax.text(0.5, 0.5, f"optimizer error:\n{exc}", transform=ax.transAxes,
                    ha="center", va="center", color="red")
            opt_rows.append({"setting": label, "W": W, "horizon": H, "error": str(exc)})

        _format(ax, setup.gp)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Phase B.4 — Real L-BFGS-B optimizer from cold start (white/black line)\n"
                 "(thin colored lines = candidate trajectories from B1)", fontsize=10)
    _save(fig, out_dir, "B4_optimizer_converged")

    # CSV for optimizer outputs
    csv_path2 = out_dir / "B4_optimizer_results.csv"
    with open(csv_path2, "w", newline="") as f:
        keys = sorted(set().union(*[r.keys() for r in opt_rows]))
        writer = _csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in opt_rows:
            writer.writerow(r)
    print(f"  wrote {csv_path2}")


# ---------------------------------------------------------------------------
# Phase C — goal-prior sweep + closed-loop offline
# ---------------------------------------------------------------------------

def plot_phase_c(setup: Setup, out_dir: Path) -> None:
    """Sweep `goal_prior_u_std_final` and `ambiguity_weight` to see when A3 wins."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_base = dict(setup.config)
    # Use a long enough horizon to give A3 a fair chance
    cfg_base["horizon"] = 80
    # Increase v_max so the planner can reach the goal area in one horizon
    cfg_base["v_max"] = 0.22

    # Build "max horizon" planner once for trajectory construction (same controls everywhere)
    base_for_traj = build_planner(
        cfg_base,
        planner_kind="visibility_aware_efe",
        camera_params=setup.camera_params,
        visibility_artifact_path=str(setup.gp_path),
        visibility_geometry_json=setup.geometry_json,
        collision_geometry_json=setup.geometry_json,
        seed=1,
        visibility_target_height_m=setup.gp["target_height"],
    )
    candidates = candidate_trajectories(
        Setup(**{**setup.__dict__, "planner": base_for_traj})
    )

    m0 = np.array([setup.start_xy_yaw[0], setup.start_xy_yaw[1], setup.start_xy_yaw[2]])
    S0 = setup.S0
    goal_xy = setup.goal_xy

    # Sweep goal_prior_u_std_final × ambiguity_weight.
    goal_sigmas = [20.0, 35.0, 50.0, 80.0, 120.0]
    weights = [6.0, 10.0, 20.0]

    grid = np.full((len(weights), len(goal_sigmas)), np.nan, dtype=float)
    a3_minus_a4_total = np.full_like(grid, np.nan)
    a3_minus_a4_risk = np.full_like(grid, np.nan)
    a3_minus_a4_amb = np.full_like(grid, np.nan)
    rows = []

    for iw, W in enumerate(weights):
        for ig, sg in enumerate(goal_sigmas):
            cfg = dict(cfg_base)
            cfg["ambiguity_weight"] = W
            cfg["goal_prior_u_std_final"] = sg
            cfg["goal_prior_v_std_final"] = sg
            # Also push the starting sigma higher so we don't have a sharper-than-final start.
            cfg["goal_prior_u_std_start"] = max(sg * 2.0, 80.0)
            cfg["goal_prior_v_std_start"] = max(sg * 2.0, 80.0)
            try:
                p = build_planner(
                    cfg, planner_kind="visibility_aware_efe",
                    camera_params=setup.camera_params,
                    visibility_artifact_path=str(setup.gp_path),
                    visibility_geometry_json=setup.geometry_json,
                    collision_geometry_json=setup.geometry_json,
                    seed=1,
                    visibility_target_height_m=setup.gp["target_height"],
                )
            except Exception as exc:
                print(f"  build_planner failed at W={W}, sg={sg}: {exc}")
                continue
            res_a4 = eval_controls(p, m0, S0, goal_xy, candidates["A4_due_north"])
            res_a3 = eval_controls(p, m0, S0, goal_xy, candidates["A3_west_then_north"])
            d_tot = res_a3["total_cost"] - res_a4["total_cost"]
            d_risk = res_a3["risk_cost"] - res_a4["risk_cost"]
            d_amb = res_a3["ambiguity_cost"] - res_a4["ambiguity_cost"]
            grid[iw, ig] = d_tot
            a3_minus_a4_total[iw, ig] = d_tot
            a3_minus_a4_risk[iw, ig] = d_risk
            a3_minus_a4_amb[iw, ig] = d_amb
            rows.append({
                "W": W, "goal_sigma": sg,
                "A4_total": res_a4["total_cost"], "A3_total": res_a3["total_cost"],
                "A4_risk": res_a4["risk_cost"], "A3_risk": res_a3["risk_cost"],
                "A4_amb": res_a4["ambiguity_cost"], "A3_amb": res_a3["ambiguity_cost"],
                "delta_total_a3_minus_a4": d_tot,
                "delta_risk_a3_minus_a4": d_risk,
                "delta_amb_a3_minus_a4": d_amb,
                "A4_valid": res_a4["rollout_valid"],
                "A3_valid": res_a3["rollout_valid"],
            })

    # Save CSV
    import csv as _csv
    csv_path = out_dir / "C1_goal_prior_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  wrote {csv_path}")

    # ---- C1 plot — heatmap of Δtotal(A3 - A4) over (W, σ_goal) ----
    # Annotated negative = A3 wins; positive = A4 wins
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    titles = ["Δrisk = risk(A3) - risk(A4)", "Δamb = amb(A3) - amb(A4)", "Δtotal = total(A3) - total(A4)"]
    arrays = [a3_minus_a4_risk, a3_minus_a4_amb, a3_minus_a4_total]
    for ax, data, title in zip(axes, arrays, titles):
        # Symmetric colormap centred at 0
        vmax = float(np.nanmax(np.abs(data)))
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for iw in range(len(weights)):
            for ig in range(len(goal_sigmas)):
                v = data[iw, ig]
                if not np.isnan(v):
                    ax.text(ig, iw, f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="black" if abs(v) < vmax * 0.5 else "white")
        ax.set_xticks(range(len(goal_sigmas)))
        ax.set_xticklabels([f"{s:.0f}" for s in goal_sigmas])
        ax.set_yticks(range(len(weights)))
        ax.set_yticklabels([f"W={w:.0f}" for w in weights])
        ax.set_xlabel("goal_prior_u_std_final (px)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.suptitle(
        "Phase C.1 — Does widening the goal prior let A3 win?\n"
        f"(h={cfg_base['horizon']}, v_max={cfg_base['v_max']}, blue = A3 wins, red = A4 wins)",
        fontsize=10,
    )
    _save(fig, out_dir, "C1_goal_prior_sweep")

    # ---- C2 — closed-loop offline at the winning corner (if any) ----
    # Find the most A3-favourable (W, σ_goal) cell
    best = None
    for iw, W in enumerate(weights):
        for ig, sg in enumerate(goal_sigmas):
            v = a3_minus_a4_total[iw, ig]
            if not np.isnan(v):
                if best is None or v < best[0]:
                    best = (v, W, sg)
    print(f"  best A3-vs-A4 cell: Δtotal={best[0]:.1f} at W={best[1]}, σ_goal={best[2]}")

    # Run closed-loop at four reference configs:
    #   v36_probe: the current failing baseline
    #   h80_w6_default: smoke weight (W=6), default sigma, but longer horizon + faster v_max
    #   h80_best: best A3-vs-A4 cell from C1 sweep
    #   h80_w6_a3_seed: same as h80_w6_default BUT initialized with A3 candidate as warm-start
    cfg_list = [
        ("v36_probe", {"ambiguity_weight": 10.0, "goal_prior_u_std_final": 20.0,
                       "horizon": 60, "v_max": 0.15}, None),
        ("h80_w6_default", {"ambiguity_weight": 6.0, "goal_prior_u_std_final": 20.0,
                            "horizon": 80, "v_max": 0.22}, None),
        ("h80_best", {"ambiguity_weight": float(best[1]), "goal_prior_u_std_final": float(best[2]),
                      "horizon": 80, "v_max": 0.22}, None),
        ("h80_w6_a3_seed", {"ambiguity_weight": 6.0, "goal_prior_u_std_final": 20.0,
                            "horizon": 80, "v_max": 0.22}, "A3_west_then_north"),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7))
    xs, ys = setup.gp["xs"], setup.gp["ys"]
    P = setup.gp["P_plan"]
    ax.imshow(P, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
              cmap="viridis", vmin=0, vmax=1.0, aspect="equal", zorder=1)
    _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json), alpha=0.50)
    _draw_task_markers(ax, setup, draw_lookahead=False)

    style = {
        "v36_probe": dict(color="#d62728", lw=2.0, ls="-"),
        "h80_w6_default": dict(color="#ff7f0e", lw=2.0, ls="-"),
        "h80_best": dict(color="#2ca02c", lw=2.5, ls="-"),
        "h80_w6_a3_seed": dict(color="#1f77b4", lw=2.5, ls="-"),
    }
    closed_loop_rows = []
    n_steps_loop = 60  # 15 s of receding-horizon execution
    for tag, override, init_seed_name in cfg_list:
        cfg = dict(cfg_base)
        cfg.update(override)
        # Make sure goal_prior_start >= final
        cfg["goal_prior_u_std_start"] = max(2 * cfg["goal_prior_u_std_final"], 80.0)
        cfg["goal_prior_v_std_start"] = cfg["goal_prior_u_std_start"]
        try:
            p = build_planner(
                cfg, planner_kind="visibility_aware_efe",
                camera_params=setup.camera_params,
                visibility_artifact_path=str(setup.gp_path),
                visibility_geometry_json=setup.geometry_json,
                collision_geometry_json=setup.geometry_json,
                seed=1, visibility_target_height_m=setup.gp["target_height"],
            )
        except Exception as exc:
            print(f"  build_planner failed for tag={tag}: {exc}")
            continue
        # Build a seed initialisation if requested (using this planner's horizon/v_max).
        init_u = None
        if init_seed_name is not None:
            seed_cands = candidate_trajectories(Setup(**{**setup.__dict__, "planner": p}))
            init_u = seed_cands[init_seed_name].reshape(-1)
        cl = closed_loop_offline(p, m0, S0, goal_xy, n_steps=n_steps_loop,
                                 warm_start=True, initial_u=init_u)
        st = cl["traj_m"]
        st_ms = float(np.mean([i["solver_time_ms"] for i in cl["infos"]]))
        ax.plot(st[:, 0], st[:, 1], **style[tag], zorder=20, label=f"{tag} (mean_t={st_ms:.0f}ms)")
        ax.scatter([st[-1, 0]], [st[-1, 1]], s=70, c=style[tag]["color"],
                   edgecolor="white", linewidth=0.6, zorder=21, marker="o")
        closed_loop_rows.append({
            "tag": tag,
            "horizon": cfg["horizon"], "v_max": cfg["v_max"],
            "ambiguity_weight": cfg["ambiguity_weight"],
            "goal_prior_u_std_final": cfg["goal_prior_u_std_final"],
            "n_steps_ran": len(cl["infos"]),
            "terminal_x": float(st[-1, 0]), "terminal_y": float(st[-1, 1]),
            "x_min": float(st[:, 0].min()), "x_max": float(st[:, 0].max()),
            "y_min": float(st[:, 1].min()), "y_max": float(st[:, 1].max()),
            "mean_solver_time_ms": st_ms,
            "min_goal_dist": float(min(i["d_goal"] for i in cl["infos"])),
        })
    _format(ax, setup.gp)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Phase C.2 — Closed-loop offline rollouts (no Gazebo)\n"
                 f"start={setup.start_xy_yaw[:2]}, goal={setup.goal_xy}", fontsize=10)
    _save(fig, out_dir, "C2_closed_loop_rollouts")

    csv_path2 = out_dir / "C2_closed_loop.csv"
    with open(csv_path2, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(closed_loop_rows[0].keys()))
        writer.writeheader()
        for r in closed_loop_rows:
            writer.writerow(r)
    print(f"  wrote {csv_path2}")

    # ---- C3 — multi-start optimization to test the local-minimum hypothesis ----
    # At the same configs as C2, run the optimizer FROM both cold-start AND from the
    # A3 candidate as warm-start. If they converge to different EFE, the landscape is
    # multimodal and the cold-start is the bottleneck.
    fig, axes = plt.subplots(1, len(cfg_list), figsize=(5.5 * len(cfg_list), 6),
                             squeeze=False)
    multistart_rows = []
    for ax_idx, (tag, override, _seed_name) in enumerate(cfg_list):
        ax = axes[0][ax_idx]
        cfg = dict(cfg_base)
        cfg.update(override)
        cfg["goal_prior_u_std_start"] = max(2 * cfg["goal_prior_u_std_final"], 80.0)
        cfg["goal_prior_v_std_start"] = cfg["goal_prior_u_std_start"]
        p = build_planner(
            cfg, planner_kind="visibility_aware_efe",
            camera_params=setup.camera_params,
            visibility_artifact_path=str(setup.gp_path),
            visibility_geometry_json=setup.geometry_json,
            collision_geometry_json=setup.geometry_json,
            seed=1, visibility_target_height_m=setup.gp["target_height"],
        )
        # Build candidates at this planner's horizon and v_max
        cands = candidate_trajectories(Setup(**{**setup.__dict__, "planner": p}))

        # GP background
        P = setup.gp["P_plan"]
        xs_axes, ys_axes = setup.gp["xs"], setup.gp["ys"]
        ax.imshow(P, extent=[xs_axes[0], xs_axes[-1], ys_axes[0], ys_axes[-1]],
                  origin="lower", cmap="viridis", vmin=0, vmax=1.0, aspect="equal", zorder=1)
        _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json),
                    alpha=0.50, label_high_stack=False)
        _draw_task_markers(ax, setup, draw_lookahead=False)

        m0_arr = np.array([setup.start_xy_yaw[0], setup.start_xy_yaw[1], setup.start_xy_yaw[2]])

        # 3 starts: cold-start (default), A4 candidate, A3 candidate
        starts = {
            "cold_start": None,  # planner uses its neutral default cold start
            "init_A4": cands["A4_due_north"].reshape(-1),
            "init_A3": cands["A3_west_then_north"].reshape(-1),
        }
        col = {"cold_start": "white", "init_A4": "#ff7f0e", "init_A3": "#1f77b4"}
        results = {}
        for init_label, u_init in starts.items():
            try:
                pr = run_optimizer(p, m0_arr, S0, goal_xy, u_init=u_init)
                ctrls = np.asarray(pr.controls, dtype=float)
                trace = per_step_trace(p, m0_arr, S0, goal_xy, ctrls)
                states = trace["states"]
                ax.plot(states[:, 0], states[:, 1], color=col[init_label], lw=2.0,
                        label=f"{init_label}: EFE={pr.total_cost:.0f}", zorder=20)
                ax.scatter([states[-1, 0]], [states[-1, 1]], s=60, c=col[init_label],
                           edgecolor="black", linewidth=0.6, zorder=21)
                results[init_label] = {
                    "tag": tag, "init": init_label,
                    "total_cost": float(pr.total_cost),
                    "risk_cost": float(pr.risk_cost),
                    "ambiguity_cost": float(pr.ambiguity_cost),
                    "obstacle_cost": float(pr.obstacle_cost),
                    "valid": bool(pr.rollout_valid),
                    "terminal_x": float(states[-1, 0]),
                    "terminal_y": float(states[-1, 1]),
                    "x_min": float(states[:, 0].min()),
                }
                multistart_rows.append(results[init_label])
            except Exception as exc:
                print(f"  multistart fail {tag}/{init_label}: {exc}")
        _format(ax, setup.gp)
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title(f"{tag}  (W={cfg['ambiguity_weight']}, sigma={cfg['goal_prior_u_std_final']:.0f}, h={cfg['horizon']}, v={cfg['v_max']})", fontsize=8)
    fig.suptitle("Phase C.3 — Multi-start optimization\n"
                 "Same EFE? Single basin. Different EFE? Multimodal landscape (cold-start is the trap).",
                 fontsize=10)
    _save(fig, out_dir, "C3_multistart")
    csv_path3 = out_dir / "C3_multistart.csv"
    with open(csv_path3, "w", newline="") as f:
        keys = sorted(set().union(*[r.keys() for r in multistart_rows]))
        writer = _csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in multistart_rows:
            writer.writerow(r)
    print(f"  wrote {csv_path3}")


def plot_phase_a(setups: list[Setup], out_dir: Path) -> None:
    """All four Phase-A plots — overlaying smoke vs probe so we can compare."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. gp_and_world ---
    fig, axes = plt.subplots(1, len(setups), figsize=(9.5 * len(setups), 7.5))
    if len(setups) == 1:
        axes = [axes]
    for ax, setup in zip(axes, setups):
        P = setup.gp["P_plan"]
        xs, ys = setup.gp["xs"], setup.gp["ys"]
        im = ax.imshow(P, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                       cmap="viridis", vmin=0, vmax=1.0, aspect="equal", zorder=1)
        _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json), alpha=0.55)
        _draw_task_markers(ax, setup, draw_lookahead=True)
        _format(ax, setup.gp)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P_conservative_plan")
        ax.set_title(f"{setup.config_path.stem}\ntask={setup.task_name}, planner={setup.planner_kind}")
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Phase A.1 — GP visibility + world geometry + task look-ahead", fontsize=11)
    _save(fig, out_dir, "A1_gp_and_world")

    # --- 2. nogo_field ---
    fig, axes = plt.subplots(1, len(setups), figsize=(9.5 * len(setups), 7.5))
    if len(setups) == 1:
        axes = [axes]
    for ax, setup in zip(axes, setups):
        xs, ys = setup.gp["xs"], setup.gp["ys"]
        planner = setup.planner
        if planner.nogo_cost_model is None:
            ax.text(0.5, 0.5, "nogo model disabled in config", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        if planner.use_belief_nogo_cost:
            def _nogo_at_mean(m):
                return planner.nogo_cost_model.penalty_belief_np(
                    m,
                    setup.S0,
                    kappa=planner.nogo_belief_kappa,
                )
            nogo = grid_eval(_nogo_at_mean, xs, ys)
        else:
            nogo = grid_eval(planner.nogo_cost_model.penalty_state_np, xs, ys)
        # Plot log(1+nogo) for dynamic range
        im = ax.imshow(np.log1p(nogo), extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                       cmap="magma_r", aspect="equal", zorder=1)
        _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json), alpha=0.45)
        _draw_task_markers(ax, setup)
        _format(ax, setup.gp)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + nogo_penalty)")
        mode = "belief" if planner.use_belief_nogo_cost else "mean"
        ax.set_title(
            f"nogo_weight={planner.nogo_weight:.0f}, safe={planner.nogo_safe_distance:.2f}m, "
            f"mode={mode}"
        )
    fig.suptitle("Phase A.2 — nogo cost field", fontsize=11)
    _save(fig, out_dir, "A2_nogo_field")

    # --- 3. p_vis_eff field at S0 ---
    fig, axes = plt.subplots(1, len(setups), figsize=(9.5 * len(setups), 7.5))
    if len(setups) == 1:
        axes = [axes]
    for ax, setup in zip(axes, setups):
        planner = setup.planner
        xs, ys = setup.gp["xs"], setup.gp["ys"]
        S0 = setup.S0
        if planner.visibility_model is None:
            ax.text(0.5, 0.5, "visibility model disabled (constant_R_efe)", transform=ax.transAxes,
                    ha="center", va="center")
            _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json), alpha=0.55)
            _draw_task_markers(ax, setup)
            _format(ax, setup.gp)
            continue
        pveff = grid_eval(
            lambda m, S=S0, p=planner: float(p.visibility_probability_belief(m, S)),
            xs, ys,
        )
        im = ax.imshow(pveff, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                       cmap="viridis", vmin=0, vmax=1, aspect="equal", zorder=1)
        _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json), alpha=0.55)
        _draw_task_markers(ax, setup)
        _format(ax, setup.gp)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p_vis_eff(S0)")
        ax.set_title(f"sigma-point widened p_vis at sigma_xy={np.sqrt(S0[0,0]):.2f}m")
    fig.suptitle("Phase A.3 — p_vis_eff field at planner's initial belief", fontsize=11)
    _save(fig, out_dir, "A3_pvis_eff_field_at_S0")

    # --- 4. R_plan field ---
    fig, axes = plt.subplots(1, len(setups), figsize=(9.5 * len(setups), 7.5))
    if len(setups) == 1:
        axes = [axes]
    for ax, setup in zip(axes, setups):
        planner = setup.planner
        xs, ys = setup.gp["xs"], setup.gp["ys"]
        S0 = setup.S0
        if planner.visibility_model is None:
            r_const = float(planner.r_visible_uv)
            ax.text(0.5, 0.5, f"constant_R — r_plan = {r_const:.2f}px everywhere",
                    transform=ax.transAxes, ha="center", va="center")
            _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json), alpha=0.55)
            _draw_task_markers(ax, setup)
            _format(ax, setup.gp)
            continue

        def rplan_at(m):
            diag = planner.planning_visibility_diagnostics(m, S0)
            return float(diag["r_plan_u_std"])

        rfield = grid_eval(rplan_at, xs, ys)
        im = ax.imshow(rfield, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                       cmap="inferno", aspect="equal", zorder=1,
                       vmin=float(planner.r_visible_uv), vmax=float(planner.r_miss_uv))
        _draw_world(ax, _parse_prisms_from_geometry_json(setup.geometry_json), alpha=0.55)
        _draw_task_markers(ax, setup)
        _format(ax, setup.gp)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r_plan_u_std (px)")
        ax.set_title(
            f"R_plan = blend({planner.r_visible_uv}, {planner.r_miss_uv}) by p_vis_eff"
        )
    fig.suptitle("Phase A.4 — effective observation noise r_plan(p_vis_eff)", fontsize=11)
    _save(fig, out_dir, "A4_rplan_field")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-config",
                        default=str(REPO_ROOT / "scripts" / "visibility_comparison" / "aws_campaign_config.yaml"),
                        help="Exploratory AWS config to inspect. Diagnostic only.")
    parser.add_argument("--smoke-config",
                        default=str(REPO_ROOT / "scripts" / "visibility_comparison" / "aws_smoke_config.yaml"))
    parser.add_argument("--out-dir",
                        default=str(REPO_ROOT / "logs" / "visibility_comparison" / "efe_offline_lab"))
    parser.add_argument("--phase", choices=["a", "b", "c", "ab", "abc"], default="abc",
                        help="Which phase(s) to run.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AWS diagnostic setup from {args.aws_config}")
    aws_setup = load_setup(Path(args.aws_config), condition="C2", seed=1)
    print(f"  task={aws_setup.task_name}  start={aws_setup.start_xy_yaw}  goal={aws_setup.goal_xy}")
    print(f"  planner_kind={aws_setup.planner_kind}  horizon={aws_setup.planner.horizon}  v_max={aws_setup.planner.v_max}")
    print(f"  ambiguity_weight={aws_setup.planner.ambiguity_weight}  visibility={aws_setup.planner.use_visibility_model}")

    print(f"Loading smoke setup from {args.smoke_config}")
    smoke = load_setup(Path(args.smoke_config), condition="C2", seed=1)
    print(f"  task={smoke.task_name}  start={smoke.start_xy_yaw}  goal={smoke.goal_xy}")
    print(f"  planner_kind={smoke.planner_kind}  horizon={smoke.planner.horizon}  v_max={smoke.planner.v_max}")
    print(f"  ambiguity_weight={smoke.planner.ambiguity_weight}  visibility={smoke.planner.use_visibility_model}")

    if args.phase in ("a", "ab"):
        print(f"Phase A diagnostic plots → {out_dir}")
        plot_phase_a([aws_setup, smoke], out_dir)

    if args.phase in ("b", "ab"):
        print(f"Phase B candidate trajectory comparison (AWS diagnostic) → {out_dir}")
        plot_phase_b(aws_setup, out_dir)

    if args.phase in ("c", "ab", "abc"):
        print(f"Phase C: goal-prior sweep + closed-loop offline (AWS diagnostic) → {out_dir}")
        plot_phase_c(aws_setup, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
