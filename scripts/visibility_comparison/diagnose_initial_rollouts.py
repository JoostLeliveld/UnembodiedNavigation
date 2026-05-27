#!/usr/bin/env python3
"""Diagnose initial EFE rollouts without running Gazebo.

This is a planning diagnostic, not paper evidence and not a route-forcing
runtime mechanism. It runs the same local optimizer from one start/goal under
different horizons and optional initial guesses. If different initial guesses
converge to materially different costs or route families, the objective/solver
landscape is locally multimodal.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from efe_offline_lab import (  # noqa: E402
    _draw_task_markers,
    _draw_world,
    _format,
    _load_gp,
    _parse_prisms_from_geometry_json,
    build_planner,
    load_setup,
    per_step_trace,
    run_optimizer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _controls_for_waypoints(planner, start_xy_yaw, waypoints) -> np.ndarray:
    """Generate a crude unicycle warm-start toward waypoints.

    The optimizer is free to change these controls. The trajectory is only used
    to test whether a better basin exists than the neutral cold start.
    """
    horizon = int(planner.horizon)
    dt = float(planner.dt)
    v_max = float(planner.v_max)
    w_max = float(planner.w_max)
    m = np.asarray(start_xy_yaw, dtype=float).reshape(3).copy()
    controls = np.zeros((horizon, 2), dtype=float)
    waypoint_idx = 0
    waypoints = [np.asarray(wp, dtype=float).reshape(2) for wp in waypoints]

    for k in range(horizon):
        if waypoint_idx >= len(waypoints):
            break
        target = waypoints[waypoint_idx]
        d = target - m[:2]
        if float(np.linalg.norm(d)) < 0.18 and waypoint_idx < len(waypoints) - 1:
            waypoint_idx += 1
            target = waypoints[waypoint_idx]
            d = target - m[:2]

        desired_yaw = math.atan2(float(d[1]), float(d[0]))
        yaw_err = _wrap_pi(desired_yaw - float(m[2]))
        w = float(np.clip(yaw_err / max(dt, 1e-6), -w_max, w_max))
        v = v_max if abs(yaw_err) < 0.65 else 0.0
        controls[k] = [v, w]
        m, _S_dummy = planner.predict(m, np.eye(3) * 1e-6, controls[k])

    return controls


def _trace_from_controls(planner, start_xy_yaw, controls):
    m = np.asarray(start_xy_yaw, dtype=float).reshape(3).copy()
    S = np.diag([0.15 ** 2, 0.15 ** 2, 0.05 ** 2])
    states = [m[:2].copy()]
    pvis = []
    for u in np.asarray(controls, dtype=float).reshape(-1, 2):
        m, S = planner.predict(m, S, u)
        states.append(m[:2].copy())
        try:
            diag = planner.planning_visibility_diagnostics(m, S)
            pvis.append(float(diag.get("p_vis_eff", math.nan)))
        except Exception:
            pvis.append(math.nan)
    return np.asarray(states, dtype=float), np.asarray(pvis, dtype=float)


def _path_length(states: np.ndarray) -> float:
    if len(states) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(states, axis=0), axis=1).sum())


def _region_center(profile: dict, name: str) -> tuple[float, float] | None:
    for region in profile.get("known_2d_regions", []) or []:
        if str(region.get("name", "")) != name:
            continue
        try:
            return (
                0.5 * (float(region["xmin"]) + float(region["xmax"])),
                0.5 * (float(region["ymin"]) + float(region["ymax"])),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _aws_route_probe_waypoints(setup) -> dict[str, list[tuple[float, float]]]:
    """Region-derived AWS A3/A4 route initializations for local-optimum tests."""
    a3 = _region_center(setup.profile, "rack_aisle_A3")
    a4 = _region_center(setup.profile, "rack_aisle_A4")
    lower = _region_center(setup.profile, "lower_main_aisle")
    mid = _region_center(setup.profile, "mid_cross_aisle")
    if not (a3 and a4 and lower and mid):
        return {}

    gx, gy = setup.goal_xy
    lower_y = lower[1]
    # Use the goal y when it lies between the lower and mid corridors; otherwise
    # use the mid cross-aisle center. This keeps the diagnostic route inside the
    # same known driveable layer for both B1 variants.
    target_cross_y = float(gy)
    if target_cross_y < min(lower_y, mid[1]) or target_cross_y > max(lower_y, mid[1]):
        target_cross_y = float(mid[1])

    return {
        "aws_a3_detour_seed": [
            (float(a3[0]), float(lower_y)),
            (float(a3[0]), target_cross_y),
            (float(gx), float(gy)),
        ],
        "aws_a4_direct_seed": [
            (float(a4[0]), float(lower_y)),
            (float(a4[0]), target_cross_y),
            (float(gx), float(gy)),
        ],
    }


def _summarize_plan(setup, planner, horizon, condition, init_name, plan, error=""):
    start = np.asarray(setup.start_xy_yaw, dtype=float)
    goal = np.asarray(setup.goal_xy, dtype=float)
    if error:
        return {
            "condition": condition,
            "horizon": horizon,
            "init": init_name,
            "error": error,
        }, np.asarray([start[:2]], dtype=float), np.asarray([], dtype=float)

    states, pvis = _trace_from_controls(planner, start, plan.controls)
    goal_d = np.linalg.norm(states - goal.reshape(1, 2), axis=1)
    finite_pvis = pvis[np.isfinite(pvis)]
    row = {
        "condition": condition,
        "horizon": horizon,
        "init": init_name,
        "error": "",
        "total_cost": float(plan.total_cost),
        "risk_cost": float(plan.risk_cost),
        "ambiguity_cost": float(plan.ambiguity_cost),
        "obstacle_cost": float(plan.obstacle_cost),
        "terminal_goal_distance_pred": float(plan.terminal_goal_distance_pred),
        "terminal_goal_progress_m": float(plan.terminal_goal_progress_m),
        "min_goal_distance_pred": float(np.min(goal_d)),
        "terminal_x": float(states[-1, 0]),
        "terminal_y": float(states[-1, 1]),
        "x_min": float(np.min(states[:, 0])),
        "x_max": float(np.max(states[:, 0])),
        "y_min": float(np.min(states[:, 1])),
        "y_max": float(np.max(states[:, 1])),
        "path_length_m": _path_length(states),
        "mean_pvis_eff": float(np.mean(finite_pvis)) if finite_pvis.size else math.nan,
        "min_pvis_eff": float(np.min(finite_pvis)) if finite_pvis.size else math.nan,
        "outside_gp_steps": int(np.count_nonzero(~np.isfinite(pvis))),
        "rollout_valid": bool(plan.rollout_valid),
        "invalid_reason": str(plan.invalid_reason),
        "optimizer_success": bool(plan.optimizer_success),
        "optimizer_status": int(plan.optimizer_status),
        "optimizer_nit": int(plan.optimizer_nit),
        "optimizer_nfev": int(plan.optimizer_nfev),
        "solve_time_ms": float(plan.solve_time_s) * 1000.0,
    }
    return row, states, pvis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "scripts" / "visibility_comparison" / "putaway_route_choice_smoke_config.yaml"),
    )
    parser.add_argument("--gp", default="", help="Optional GP artifact override")
    parser.add_argument("--task", default="")
    parser.add_argument("--start", default="", help="Optional override x,y,yaw")
    parser.add_argument("--goal", default="", help="Optional override x,y")
    parser.add_argument("--conditions", default="C1,C2")
    parser.add_argument("--horizons", default="24,40,60,80,120")
    parser.add_argument(
        "--inits",
        default="",
        help="Optional comma-separated diagnostic initializations to run, e.g. cold,front_visible_seed",
    )
    parser.add_argument(
        "--seed-family",
        choices=("auto", "aws", "compact", "all"),
        default="auto",
        help=(
            "Which diagnostic initial guesses to include. auto uses AWS-specific "
            "seeds for warehouse_aws.world.sdf and compact-world seeds otherwise."
        ),
    )
    parser.add_argument("--v-max", type=float, default=None)
    parser.add_argument("--w-max", type=float, default=None)
    parser.add_argument("--control-weight", type=float, default=None)
    parser.add_argument("--ambiguity-weight", type=float, default=None)
    parser.add_argument("--observation-risk-scale", type=float, default=None)
    parser.add_argument("--goal-prior-final", type=float, default=None)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "logs" / "visibility_comparison" / "initial_rollout_diagnostics" / "putaway_v1"),
    )
    parser.add_argument("--optimizer-maxiter", type=int, default=80)
    parser.add_argument("--optimizer-maxfun", type=int, default=600)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    init_filter = {x.strip() for x in args.inits.split(",") if x.strip()}

    rows: list[dict] = []
    traces: dict[tuple[str, int, str], np.ndarray] = {}
    setup_for_plot = None

    for condition in conditions:
        setup = load_setup(
            config_path,
            condition=condition,
            seed=1,
            task_override=args.task or None,
        )
        if args.gp:
            gp_path = Path(args.gp).expanduser()
            if not gp_path.is_absolute():
                gp_path = (REPO_ROOT / gp_path).resolve()
            gp = _load_gp(gp_path)
            setup = replace(
                setup,
                gp_path=gp_path,
                gp=gp,
                camera_params={
                    "cam_pos": tuple(gp["camera_pos"][:3].tolist()),
                    "look_at": tuple(gp["look_at"][:3].tolist()),
                    "img_width": gp["img_width"],
                    "img_height": gp["img_height"],
                    "fov_h_rad": gp["fov_h_rad"],
                },
            )
        if args.start:
            parts = [float(x.strip()) for x in args.start.split(",")]
            if len(parts) != 3:
                raise SystemExit("--start must be x,y,yaw")
            setup = replace(setup, start_xy_yaw=(parts[0], parts[1], parts[2]))
        if args.goal:
            parts = [float(x.strip()) for x in args.goal.split(",")]
            if len(parts) != 2:
                raise SystemExit("--goal must be x,y")
            setup = replace(setup, goal_xy=(parts[0], parts[1]))
        setup_for_plot = setup
        sx, sy, _yaw = setup.start_xy_yaw
        gx, gy = setup.goal_xy

        compact_waypoint_sets = {
            "cold": None,
            "direct_goal_seed": [(gx, gy)],
            "rear_shadow_seed": [(sx, 1.25), (gx, 1.25), (gx, gy)],
            "front_visible_seed": [(gx, -2.0), (gx, gy)],
        }
        aws_waypoint_sets = {
            "cold": None,
            "direct_goal_seed": [(gx, gy)],
        }
        aws_waypoint_sets.update(_aws_route_probe_waypoints(setup))

        if args.seed_family == "all":
            waypoint_sets = dict(compact_waypoint_sets)
            waypoint_sets.update(aws_waypoint_sets)
        elif args.seed_family == "compact":
            waypoint_sets = dict(compact_waypoint_sets)
        elif args.seed_family == "aws":
            waypoint_sets = dict(aws_waypoint_sets)
        else:
            waypoint_sets = (
                dict(aws_waypoint_sets)
                if setup.world == "warehouse_aws.world.sdf"
                else dict(compact_waypoint_sets)
            )
        if init_filter:
            waypoint_sets = {
                name: waypoints
                for name, waypoints in waypoint_sets.items()
                if name in init_filter
            }

        for horizon in horizons:
            cfg = dict(setup.config)
            cfg["horizon"] = horizon
            if args.v_max is not None:
                cfg["v_max"] = float(args.v_max)
            if args.w_max is not None:
                cfg["w_max"] = float(args.w_max)
            if args.control_weight is not None:
                cfg["control_weight"] = float(args.control_weight)
            if args.ambiguity_weight is not None:
                cfg["ambiguity_weight"] = float(args.ambiguity_weight)
            if args.observation_risk_scale is not None:
                cfg["observation_risk_scale"] = float(args.observation_risk_scale)
            if args.goal_prior_final is not None:
                cfg["goal_prior_u_std_final"] = float(args.goal_prior_final)
                cfg["goal_prior_v_std_final"] = float(args.goal_prior_final)
            cfg["optimizer_maxiter"] = int(args.optimizer_maxiter)
            cfg["optimizer_maxfun"] = int(args.optimizer_maxfun)
            planner = build_planner(
                cfg,
                planner_kind=setup.planner_kind,
                camera_params=setup.camera_params,
                visibility_artifact_path=str(setup.gp_path),
                visibility_geometry_json=setup.geometry_json,
                collision_geometry_json=setup.geometry_json,
                seed=1,
                visibility_target_height_m=setup.gp["target_height"],
            )

            m0 = np.asarray(setup.start_xy_yaw, dtype=float)
            S0 = setup.S0
            goal_xy = np.asarray(setup.goal_xy, dtype=float)

            for init_name, waypoints in waypoint_sets.items():
                u_init = None
                if waypoints is not None:
                    u_init = _controls_for_waypoints(planner, setup.start_xy_yaw, waypoints).reshape(-1)
                try:
                    plan = run_optimizer(planner, m0, S0, goal_xy, u_init=u_init)
                    row, states, _pvis = _summarize_plan(
                        setup, planner, horizon, condition, init_name, plan
                    )
                    traces[(condition, horizon, init_name)] = states
                except Exception as exc:
                    row, states, _pvis = _summarize_plan(
                        setup, planner, horizon, condition, init_name, None, error=repr(exc)
                    )
                rows.append(row)
                print(
                    f"{condition:>2} H={horizon:>3} {init_name:<18} "
                    f"J={row.get('total_cost', math.nan):8.1f} "
                    f"d_goal={row.get('terminal_goal_distance_pred', math.nan):5.2f} "
                    f"pvis={row.get('mean_pvis_eff', math.nan):4.2f} "
                    f"t={row.get('solve_time_ms', math.nan):6.0f}ms "
                    f"{row.get('error', '')[:80]}"
                )

    csv_path = out_dir / "initial_rollout_sweep.csv"
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    if setup_for_plot is not None:
        xs, ys = setup_for_plot.gp["xs"], setup_for_plot.gp["ys"]
        P = setup_for_plot.gp["P_plan"]
        fig, axes = plt.subplots(
            len(conditions),
            len(horizons),
            figsize=(4.4 * len(horizons), 4.2 * len(conditions)),
            squeeze=False,
        )
        colors = {
            "cold": "black",
            "direct_goal_seed": "#ff7f0e",
            "rear_shadow_seed": "#d62728",
            "front_visible_seed": "#1f77b4",
            "aws_a3_detour_seed": "#2ca02c",
            "aws_a4_direct_seed": "#9467bd",
        }
        for iy, condition in enumerate(conditions):
            for ix, horizon in enumerate(horizons):
                ax = axes[iy][ix]
                ax.imshow(
                    P,
                    extent=[xs[0], xs[-1], ys[0], ys[-1]],
                    origin="lower",
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    aspect="equal",
                    zorder=1,
                )
                _draw_world(ax, _parse_prisms_from_geometry_json(setup_for_plot.geometry_json), alpha=0.50)
                _draw_task_markers(ax, setup_for_plot)
                for init_name, color in colors.items():
                    st = traces.get((condition, horizon, init_name))
                    if st is None or len(st) < 2:
                        continue
                    ax.plot(st[:, 0], st[:, 1], color=color, lw=1.8, label=init_name, zorder=20)
                    ax.scatter([st[-1, 0]], [st[-1, 1]], color=color, s=18, zorder=21)
                _format(ax, setup_for_plot.gp)
                ax.set_title(f"{condition}, H={horizon}")
                if ix == 0 and iy == 0:
                    ax.legend(fontsize=6, loc="upper left")
        fig.suptitle(
            "Initial optimizer rollouts from different diagnostic warm starts\n"
            "Different converged paths/costs imply local-optimum sensitivity.",
            fontsize=11,
        )
        plot_path = out_dir / "initial_rollout_sweep.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
