#!/usr/bin/env python3
"""F23: locked hierarchical offline smoke for the AWS route-choice task.

This is a pre-Gazebo smoke figure. It does not execute Gazebo or recapture
perception artifacts. It checks whether the locked runtime contract produces
reasonable route-level plans before spending time on closed-loop simulation:

* C1 and C2 get the same task, driveable floor, optimizer budget, and seeds.
* The global solve uses the long route-level horizon.
* Planner-derived waypoints are extracted from each global plan.
* A short local tracker is timed against those waypoints.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml


REPO = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
SCRIPT_DIR = REPO / "scripts" / "visibility_comparison"
CONFIG = SCRIPT_DIR / "aws_smoke_config.yaml"
WORLD = "warehouse_aws.world.sdf"
TASK = "B1_apron_a4_to_uppermid_a3"
PROFILE_PATH = REPO / "src" / "experiments" / "config" / "world_profiles.yaml"
OUT_DIR = REPO / "timing_presentation" / "figures" / "F23"
OUT_PNG = OUT_DIR / "F23_hier_locked_smoke.png"
OUT_PDF = OUT_DIR / "F23_hier_locked_smoke.pdf"
OUT_CSV = OUT_DIR / "F23_hier_locked_smoke.csv"
OUT_MD = OUT_DIR / "F23_hier_locked_smoke.md"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "src" / "experiments"))
sys.path.insert(0, str(REPO / "src" / "planning"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))

from efe_offline_lab import build_planner, load_setup  # noqa: E402
from planning.planners.base_planner import extract_waypoints  # noqa: E402


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to(src: np.ndarray, dst: np.ndarray, fallback: float = 0.0) -> float:
    d = np.asarray(dst, dtype=float) - np.asarray(src, dtype=float)
    if float(np.linalg.norm(d)) < 1e-6:
        return fallback
    return math.atan2(float(d[1]), float(d[0]))


def _profile() -> dict:
    return _load_yaml(PROFILE_PATH)["worlds"][WORLD]


def _condition_setup(condition: str):
    return load_setup(CONFIG, condition=condition, seed=1, task_override=TASK)


def _planner_from_setup(setup, cfg_updates: dict, *, planner_kind: str | None = None):
    cfg = dict(setup.config)
    cfg.update(cfg_updates)
    return build_planner(
        cfg,
        planner_kind=planner_kind or setup.planner_kind,
        camera_params=setup.camera_params,
        visibility_artifact_path=str(setup.gp_path),
        visibility_geometry_json=setup.geometry_json,
        collision_geometry_json=setup.geometry_json,
        driveable_geometry_json=setup.driveable_geometry_json,
        seed=1,
        visibility_target_height_m=setup.gp["target_height"],
    )


def _global_updates(cfg: dict) -> dict:
    return {
        "horizon": int(cfg["global_horizon"]),
        "dt": float(cfg["dt"]),
        "v_max": float(cfg["v_max"]),
        "ambiguity_weight": float(cfg["ambiguity_weight"]),
        "goal_prior_u_std_final": float(cfg["goal_prior_u_std_final"]),
        "goal_prior_v_std_final": float(cfg["goal_prior_v_std_final"]),
        "optimizer_maxiter": int(cfg["optimizer_maxiter"]),
        "optimizer_maxfun": int(cfg["optimizer_maxfun"]),
        "optimizer_multistart": bool(cfg["global_optimizer_multistart"]),
        "optimizer_multistart_include_direct": bool(cfg["optimizer_multistart_include_direct"]),
        "optimizer_multistart_lateral_offsets": str(cfg.get("optimizer_multistart_lateral_offsets", "")),
        "optimizer_initial_routes_json": str(cfg.get("optimizer_initial_routes_json", "")),
        "use_belief_nogo_cost": bool(cfg["use_belief_nogo_cost"]),
        "nogo_belief_kappa": float(cfg["nogo_belief_kappa"]),
        "nogo_penalty_type": str(cfg["nogo_penalty_type"]),
        "nogo_weight": float(cfg["nogo_weight"]),
        "nogo_safe_distance": float(cfg["nogo_safe_distance"]),
        "nogo_logbarrier_scale": float(cfg["nogo_logbarrier_scale"]),
        "nogo_logbarrier_eps": float(cfg["nogo_logbarrier_eps"]),
    }


def _local_updates(cfg: dict) -> dict:
    return {
        "horizon": int(cfg["local_horizon"]),
        "dt": float(cfg["dt"]),
        "v_max": float(cfg["v_max"]),
        "ambiguity_weight": float(cfg["ambiguity_weight"]),
        "goal_prior_u_std_start": float(cfg["goal_prior_u_std_final"]),
        "goal_prior_v_std_start": float(cfg["goal_prior_v_std_final"]),
        "goal_prior_u_std_final": float(cfg["goal_prior_u_std_final"]),
        "goal_prior_v_std_final": float(cfg["goal_prior_v_std_final"]),
        "optimizer_maxiter": int(cfg["local_optimizer_maxiter"]),
        "optimizer_maxfun": int(cfg["local_optimizer_maxiter"]) * 4,
        "optimizer_multistart": bool(cfg["local_optimizer_multistart"]),
        "optimizer_multistart_include_direct": bool(cfg["optimizer_multistart_include_direct"]),
        "optimizer_multistart_lateral_offsets": "",
        "optimizer_initial_routes_json": "",
        "use_belief_nogo_cost": bool(cfg["local_use_belief_nogo_cost"]),
        "nogo_belief_kappa": float(cfg["nogo_belief_kappa"]),
        "nogo_penalty_type": str(cfg["local_nogo_penalty_type"] or cfg["nogo_penalty_type"]),
        "nogo_weight": float(cfg["local_nogo_weight"] if float(cfg["local_nogo_weight"]) >= 0.0 else cfg["nogo_weight"]),
        "nogo_safe_distance": float(cfg["nogo_safe_distance"]),
        "nogo_logbarrier_scale": float(cfg["nogo_logbarrier_scale"]),
        "nogo_logbarrier_eps": float(cfg["nogo_logbarrier_eps"]),
    }


def _solve_global(setup):
    planner = _planner_from_setup(setup, _global_updates(setup.config))
    m0 = np.asarray(setup.start_xy_yaw, dtype=float)
    S0 = np.asarray(setup.S0, dtype=float)
    goal = np.asarray(setup.goal_xy, dtype=float)

    t0 = time.perf_counter()
    result = planner.plan(m0, S0, goal)
    wall_s = time.perf_counter() - t0
    waypoints = extract_waypoints(
        np.asarray(result.states, dtype=float),
        spacing_m=float(setup.config["waypoint_spacing_m"]),
        include_goal=True,
    )
    return planner, result, wall_s, waypoints


def _time_local_tracker(setup, waypoints: list[tuple[float, float]]) -> list[dict]:
    """Time short-horizon tracker solves between successive planner waypoints.

    This is not an execution simulation. It samples the local optimizer workload
    the hierarchical controller will face once the global plan has produced a
    route.
    """
    local = _planner_from_setup(
        setup,
        _local_updates(setup.config),
        planner_kind="constant_R_efe",
    )
    local.use_visibility_model = False
    local.use_ambiguity = bool(setup.config.get("local_use_ambiguity", False))

    points = [np.asarray(setup.start_xy_yaw[:2], dtype=float)]
    points += [np.asarray(wp, dtype=float) for wp in waypoints]
    rows: list[dict] = []
    S = np.asarray(setup.S0, dtype=float)
    prev_yaw = float(setup.start_xy_yaw[2])
    for i in range(min(len(points) - 1, 6)):
        src = points[i]
        dst = points[i + 1]
        if float(np.linalg.norm(dst - src)) < 0.18:
            continue
        yaw = _yaw_to(src, dst, fallback=prev_yaw)
        m = np.array([src[0], src[1], yaw], dtype=float)
        prev_yaw = yaw
        local.prev_controls_flat = None
        local._prev_goal_xy = None
        t0 = time.perf_counter()
        result = local.plan(m, S, dst)
        wall_s = time.perf_counter() - t0
        rows.append({
            "segment": len(rows) + 1,
            "from_x": float(src[0]),
            "from_y": float(src[1]),
            "to_x": float(dst[0]),
            "to_y": float(dst[1]),
            "local_solve_s": wall_s,
            "local_result_solve_s": float(result.solve_time_s),
            "local_total_cost": float(result.total_cost),
            "local_terminal_goal_distance_m": float(result.terminal_goal_distance_pred),
            "local_path_length_m": float(
                np.linalg.norm(np.diff(np.asarray(result.states)[:, :2], axis=0), axis=1).sum()
            ),
            "local_selected_source": str(result.selected_source),
        })
    return rows


def _draw_regions(ax, profile: dict) -> None:
    first_drive = True
    first_nondrive = True
    for r in profile.get("known_2d_regions", []) or []:
        kind = str(r.get("type", ""))
        if kind == "traversable":
            ax.add_patch(Rectangle(
                (r["xmin"], r["ymin"]),
                r["xmax"] - r["xmin"],
                r["ymax"] - r["ymin"],
                facecolor="#cdeccd",
                edgecolor="#43a65b",
                lw=0.8,
                alpha=0.45,
                zorder=0,
                label="known driveable floor" if first_drive else None,
            ))
            first_drive = False
        elif "non_driveable" in kind:
            ax.add_patch(Rectangle(
                (r["xmin"], r["ymin"]),
                r["xmax"] - r["xmin"],
                r["ymax"] - r["ymin"],
                facecolor="#f3b2b2",
                edgecolor="#e24a4a",
                lw=0.8,
                alpha=0.45,
                zorder=1,
                label="non-driveable" if first_nondrive else None,
            ))
            first_nondrive = False


def _draw_global_panel(ax, setup, result, waypoints, title: str, color: str) -> None:
    gp = setup.gp
    ax.imshow(
        gp["P_plan"],
        extent=[gp["xs"].min(), gp["xs"].max(), gp["ys"].min(), gp["ys"].max()],
        origin="lower",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        alpha=0.72,
        zorder=-2,
    )
    _draw_regions(ax, setup.profile)
    states = np.asarray(result.states, dtype=float)
    ax.plot(states[:, 0], states[:, 1], color=color, lw=2.6, label="global plan", zorder=6)
    if waypoints:
        wp = np.asarray(waypoints, dtype=float)
        ax.plot(wp[:, 0], wp[:, 1], "o", ms=6, mfc="white", mec=color, mew=1.6,
                label=f"{len(wp)} waypoints", zorder=7)
    sx, sy, yaw = setup.start_xy_yaw
    gx, gy = setup.goal_xy
    ax.plot(sx, sy, "o", color="#111827", ms=7, label="start", zorder=9)
    ax.arrow(sx, sy, 0.32 * math.cos(yaw), 0.32 * math.sin(yaw),
             width=0.015, head_width=0.11, head_length=0.12,
             color="#111827", length_includes_head=True, zorder=10)
    ax.plot(gx, gy, "*", color="#111827", ms=16, label="goal", zorder=9)
    ax.set_title(title)
    ax.set_xlim(0.15, 4.85)
    ax.set_ylim(-2.9, 2.35)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def _add_text_panel(ax, rows: list[dict], cfg: dict) -> None:
    ax.axis("off")
    lines = [
        "Locked hierarchical smoke",
        "",
        f"global horizon: {cfg['global_horizon']} steps",
        f"local horizon:  {cfg['local_horizon']} steps",
        f"dt: {cfg['dt']} s, v_max: {cfg['v_max']} m/s",
        f"goal prior final: {cfg['goal_prior_u_std_final']} px",
        f"R visible/miss: {cfg['r_visible_uv']} / {cfg['r_miss_uv']} px",
        f"command noise: {cfg['use_command_noise']}",
        f"encoder noise: {cfg['use_encoder_noise']}",
        "",
        "Global solve decomposition:",
    ]
    for r in rows:
        lines.extend([
            f"{r['condition']}:",
            f"  source: {r['global_selected_source']}",
            f"  solve: {r['global_solve_wall_s']:.2f}s",
            f"  J={r['global_total_cost']:.1f}",
            f"  risk={r['global_risk_cost']:.1f}",
            f"  amb={r['global_ambiguity_cost']:.1f}",
            f"  drive={r['global_drive_barrier_cost']:.1f}",
            f"  terminal d={r['global_terminal_goal_distance_m']:.2f}m",
            f"  waypoints={r['n_waypoints']}",
            f"  local tracker median={r['local_solve_median_s']:.2f}s",
            "",
        ])
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=8.5)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_yaml(CONFIG)

    all_rows: list[dict] = []
    results = {}
    for condition, label in (("C1", "C1 constant-R"), ("C2", "C2 visibility-aware")):
        setup = _condition_setup(condition)
        planner, global_result, wall_s, waypoints = _solve_global(setup)
        local_rows = _time_local_tracker(setup, waypoints)
        local_times = [r["local_solve_s"] for r in local_rows]
        summary = {
            "condition": label,
            "planner_kind": setup.planner_kind,
            "global_selected_source": str(global_result.selected_source),
            "global_solve_wall_s": wall_s,
            "global_result_solve_s": float(global_result.solve_time_s),
            "global_total_cost": float(global_result.total_cost),
            "global_risk_cost": float(global_result.risk_cost),
            "global_ambiguity_cost": float(global_result.ambiguity_cost),
            "global_drive_barrier_cost": float(global_result.obstacle_cost),
            "global_terminal_goal_distance_m": float(global_result.terminal_goal_distance_pred),
            "global_path_length_m": float(
                np.linalg.norm(np.diff(np.asarray(global_result.states)[:, :2], axis=0), axis=1).sum()
            ),
            "global_rollout_valid": int(bool(global_result.rollout_valid)),
            "global_optimizer_success": int(bool(global_result.optimizer_success)),
            "n_waypoints": len(waypoints),
            "waypoints_json": json.dumps([[round(float(x), 3), round(float(y), 3)] for x, y in waypoints]),
            "local_segments_timed": len(local_rows),
            "local_solve_min_s": float(np.min(local_times)) if local_times else float("nan"),
            "local_solve_median_s": float(np.median(local_times)) if local_times else float("nan"),
            "local_solve_max_s": float(np.max(local_times)) if local_times else float("nan"),
            "local_solve_times_json": json.dumps([round(float(t), 3) for t in local_times]),
        }
        all_rows.append(summary)
        for lr in local_rows:
            row = dict(summary)
            row.update(lr)
            all_rows.append(row)
        results[condition] = {
            "setup": setup,
            "global_result": global_result,
            "global_wall_s": wall_s,
            "waypoints": waypoints,
            "local_rows": local_rows,
            "summary": summary,
        }
        print(
            f"{label}: source={global_result.selected_source}, "
            f"global={wall_s:.2f}s, J={global_result.total_cost:.1f}, "
            f"wps={len(waypoints)}, local median={summary['local_solve_median_s']:.2f}s"
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in all_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    fig = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.82])
    ax_c1 = fig.add_subplot(gs[0, 0])
    ax_c2 = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[:, 2])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])

    _draw_global_panel(
        ax_c1,
        results["C1"]["setup"],
        results["C1"]["global_result"],
        results["C1"]["waypoints"],
        "C1 global original plan",
        "#2563eb",
    )
    _draw_global_panel(
        ax_c2,
        results["C2"]["setup"],
        results["C2"]["global_result"],
        results["C2"]["waypoints"],
        "C2 global original plan",
        "#dc2626",
    )

    labels = [r["condition"].replace(" ", "\n") for r in all_rows if "segment" not in r]
    summaries = [r for r in all_rows if "segment" not in r]
    x = np.arange(len(summaries))
    risk = np.array([r["global_risk_cost"] for r in summaries])
    amb = np.array([r["global_ambiguity_cost"] for r in summaries])
    drive = np.array([r["global_drive_barrier_cost"] for r in summaries])
    ax_bar.bar(x, risk, label="risk", color="#ef4444")
    ax_bar.bar(x, amb, bottom=risk, label="ambiguity", color="#60a5fa")
    ax_bar.bar(x, drive, bottom=risk + amb, label="drive barrier", color="#f59e0b")
    ax_bar.set_xticks(x, labels)
    ax_bar.set_ylabel("EFE cost")
    ax_bar.set_title("Global solve cost decomposition")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=0.25)

    global_times = np.array([r["global_solve_wall_s"] for r in summaries])
    local_median = np.array([r["local_solve_median_s"] for r in summaries])
    width = 0.34
    ax_time.bar(x - width / 2, global_times, width, label="global H80", color="#6b7280")
    ax_time.bar(x + width / 2, local_median, width, label="local H20 median", color="#10b981")
    for xi, val in zip(x - width / 2, global_times):
        ax_time.text(xi, val + 0.8, f"{val:.1f}s", ha="center", va="bottom", fontsize=8)
    for xi, val in zip(x + width / 2, local_median):
        ax_time.text(xi, val + 0.8, f"{val:.2f}s", ha="center", va="bottom", fontsize=8)
    ax_time.set_xticks(x, labels)
    ax_time.set_ylabel("wall time (s)")
    ax_time.set_title("Planning-time smoke")
    ax_time.legend(fontsize=8)
    ax_time.grid(axis="y", alpha=0.25)

    _add_text_panel(ax_text, summaries, cfg)

    fig.suptitle(
        "F23 - Locked hierarchical offline smoke: global route solve -> local tracker timing",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUT_PNG, dpi=160)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    md = [
        "# F23 Locked Hierarchical Offline Smoke",
        "",
        f"- figure: `{OUT_PNG}`",
        f"- pdf: `{OUT_PDF}`",
        f"- samples: `{OUT_CSV}`",
        "",
        "## Purpose",
        "",
        "Show the original long-horizon global plan for C1 and C2 under the locked",
        "AWS contract, including cost decomposition, planner-derived waypoints,",
        "global solve time, and short-horizon local tracker solve-time samples.",
        "",
        "This is an offline smoke test, not Gazebo evidence.",
        "",
        "## Locked Values",
        "",
        f"- task: `{TASK}`",
        f"- global horizon: `{cfg['global_horizon']}`",
        f"- local horizon: `{cfg['local_horizon']}`",
        f"- final goal-prior std: `{cfg['goal_prior_u_std_final']}` px",
        f"- visible/missed observation std: `{cfg['r_visible_uv']}` / `{cfg['r_miss_uv']}` px",
        f"- command noise in Gazebo contract: `{cfg['use_command_noise']}`",
        f"- encoder noise in Gazebo contract: `{cfg['use_encoder_noise']}`",
        f"- route seeds: `mid_cross_lane`, `lower_sweep_lane`",
        "",
        "## Summary",
        "",
    ]
    for r in summaries:
        md.append(
            f"- {r['condition']}: global solve `{r['global_solve_wall_s']:.2f}s`, "
            f"J `{r['global_total_cost']:.1f}` "
            f"(risk `{r['global_risk_cost']:.1f}`, amb `{r['global_ambiguity_cost']:.1f}`, "
            f"drive `{r['global_drive_barrier_cost']:.1f}`), "
            f"terminal d `{r['global_terminal_goal_distance_m']:.2f} m`, "
            f"{r['n_waypoints']} waypoints, local median `{r['local_solve_median_s']:.2f}s`."
        )
        md.append(f"  - waypoints: `{r['waypoints_json']}`")
        md.append(f"  - local H20 segment solve times: `{r['local_solve_times_json']}`")
    md.extend([
        "",
        "## Interpretation",
        "",
        "The local tracker timing is sampled on planner-derived waypoints. The",
        "waypoints are not mission waypoints: each condition first solves its own",
        "global EFE problem and the tracker follows that resulting plan.",
    ])
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
