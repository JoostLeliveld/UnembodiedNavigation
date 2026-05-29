#!/usr/bin/env python3
"""Plot the initial planned trajectory for each horizon.

This is a presentation/helper figure, deliberately outside the paper repo. It
shows the first offline C2 plan for the AWS B1 timing setup, so the plot is not
confounded by Gazebo execution latency or stale replanning.
"""
from __future__ import annotations

import csv
import json
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
DURABLE = Path("/home/joostleliveld/Thesis/timing_presentation")
SCRIPTS = REPO / "scripts/visibility_comparison"
CONFIG = SCRIPTS / "aws_smoke_config.yaml"
TASK = "B1_apron_a4_to_uppermid_a3"
WORLD = "warehouse_aws.world.sdf"
OUT = DURABLE / "figures/F7_initial_plan_horizon_pipeline.png"
CSV_OUT = DURABLE / "runs/initial_plan_horizon_pipeline.csv"

HORIZONS = [40, 80, 120, 200]
V_MAX = 0.22
LATERAL_OFFSETS = [-2.0, 2.0]

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / "src/experiments"))
sys.path.insert(0, str(REPO / "src/planning"))
sys.path.insert(0, str(REPO / "src/state"))
sys.path.insert(0, str(REPO / "src/perception"))
sys.path.insert(0, str(REPO / "src/unav_common"))

from efe_offline_lab import build_planner, load_setup  # noqa: E402
from experiments.core.world_profiles import (  # noqa: E402
    resolve_world_path,
    serialize_occlusion_geometry_from_world,
)


plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _load_profile() -> dict:
    profile_path = REPO / "src/experiments/config/world_profiles.yaml"
    return yaml.safe_load(profile_path.read_text())["worlds"][WORLD]


def _draw_regions(ax, profile: dict) -> None:
    first_drive = True
    first_staging = True
    for region in profile.get("known_2d_regions", []):
        kind = region.get("type")
        if kind == "traversable":
            ax.add_patch(
                Rectangle(
                    (region["xmin"], region["ymin"]),
                    region["xmax"] - region["xmin"],
                    region["ymax"] - region["ymin"],
                    facecolor="#b8e0b8",
                    edgecolor="#7cc47c",
                    lw=0.5,
                    alpha=0.52,
                    zorder=0,
                    label="known driveable region" if first_drive else None,
                )
            )
            first_drive = False
        elif kind == "non_driveable_staging":
            ax.add_patch(
                Rectangle(
                    (region["xmin"], region["ymin"]),
                    region["xmax"] - region["xmin"],
                    region["ymax"] - region["ymin"],
                    facecolor="#efb0b0",
                    edgecolor="tab:red",
                    lw=0.8,
                    alpha=0.55,
                    zorder=1,
                    label="non-driveable/staging" if first_staging else None,
                )
            )
            first_staging = False


def _parse_prisms(geometry_json: str) -> list[dict]:
    payload = json.loads(geometry_json)
    return payload.get("prisms", [])


def _draw_prisms(ax, prisms: list[dict]) -> None:
    first_rack = True
    first_crate = True
    for prism in prisms:
        name = str(prism.get("name", ""))
        if "wall_" in name:
            continue
        width = float(prism["xmax"] - prism["xmin"])
        height = float(prism["ymax"] - prism["ymin"])
        is_crate = "low_crate" in name or "stack" in name
        is_r4 = "R4" in name
        label = None
        if is_crate and first_crate:
            label = "crate / box pad"
            first_crate = False
        elif (not is_crate) and first_rack:
            label = "rack / forbidden zone"
            first_rack = False
        ax.add_patch(
            Rectangle(
                (float(prism["xmin"]), float(prism["ymin"])),
                width,
                height,
                facecolor="#e8a0a0" if is_crate else "#c8c8c8",
                edgecolor="tab:red" if is_r4 else "0.5",
                lw=1.6 if is_r4 else 0.8,
                alpha=0.75,
                zorder=2,
                label=label,
            )
        )


def _make_planner(cfg: dict, setup, horizon: int, visibility_geometry_json: str):
    cfg_h = dict(cfg)
    cfg_h["horizon"] = int(horizon)
    cfg_h["v_max"] = V_MAX
    planner = build_planner(
        cfg_h,
        planner_kind=setup.planner_kind,
        camera_params=setup.camera_params,
        visibility_artifact_path=str(setup.gp_path),
        visibility_geometry_json=visibility_geometry_json,
        collision_geometry_json=setup.geometry_json,
        seed=1,
        visibility_target_height_m=setup.gp["target_height"],
    )
    planner.optimizer_multistart = True
    planner.optimizer_multistart_include_direct = True
    planner.optimizer_multistart_lateral_offsets = list(LATERAL_OFFSETS)
    planner.prev_controls_flat = None
    return planner


def _axis_format(ax, start_pose: np.ndarray, goal_xy: np.ndarray, *, legend: bool) -> None:
    start_xy = np.asarray(start_pose[:2], dtype=float)
    start_yaw = float(start_pose[2])
    ax.plot(start_xy[0], start_xy[1], "o", color="black", ms=12, zorder=8, label="start")
    ax.arrow(
        start_xy[0],
        start_xy[1],
        0.34 * np.cos(start_yaw),
        0.34 * np.sin(start_yaw),
        width=0.018,
        head_width=0.11,
        head_length=0.13,
        color="black",
        length_includes_head=True,
        zorder=9,
    )
    ax.plot(goal_xy[0], goal_xy[1], "*", color="black", ms=23, zorder=8, label="goal")
    ax.set_xlim(-2.0, 3.9)
    ax.set_ylim(-2.0, 2.8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if legend:
        ax.legend(loc="upper left", framealpha=0.92)


def main() -> int:
    setup = load_setup(CONFIG, condition="C2", seed=1, task_override=TASK)
    cfg = setup.config
    profile = _load_profile()
    prisms = _parse_prisms(setup.geometry_json)
    visibility_geometry_json = serialize_occlusion_geometry_from_world(resolve_world_path(setup.world))
    start = np.asarray(setup.start_xy_yaw, dtype=float)
    start_xy = start[:2]
    goal = np.asarray(setup.goal_xy, dtype=float)

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5.5 * len(HORIZONS), 6.3), sharex=True, sharey=True)
    rows: list[dict] = []
    for ax, horizon in zip(axes, HORIZONS):
        planner = _make_planner(cfg, setup, horizon, visibility_geometry_json)
        t0 = time.perf_counter()
        result = planner.plan(start, setup.S0, goal)
        wall_s = time.perf_counter() - t0
        states = np.asarray(result.states, dtype=float)

        _draw_regions(ax, profile)
        _draw_prisms(ax, prisms)
        ax.plot(
            states[:, 0],
            states[:, 1],
            "-",
            color="#1f77b4",
            lw=3.0,
            zorder=5,
            label="initial planned rollout",
        )
        ax.plot(states[-1, 0], states[-1, 1], "s", color="#1f77b4", ms=10, mec="black", zorder=7)
        _axis_format(ax, start, goal, legend=(horizon == HORIZONS[0]))
        source = str(result.selected_source).replace("solver:", "")
        valid = "valid" if result.rollout_valid else "collision-predicted"
        ax.set_title(
            f"H{horizon} multistart\n"
            f"{source}, d_goal={result.terminal_goal_distance_pred:.2f} m\n"
            f"{wall_s:.1f} s, {valid}"
        )
        rows.append(
            {
                "horizon": horizon,
                "selected_source": result.selected_source,
                "solve_time_s": f"{float(result.solve_time_s):.4f}",
                "wall_time_s": f"{wall_s:.4f}",
                "terminal_goal_distance_pred_m": f"{float(result.terminal_goal_distance_pred):.4f}",
                "terminal_goal_progress_m": f"{float(result.terminal_goal_progress_m):.4f}",
                "min_predicted_obstacle_distance_m": f"{float(result.min_predicted_obstacle_distance_m):.4f}",
                "rollout_valid": int(bool(result.rollout_valid)),
                "total_cost": f"{float(result.total_cost):.4f}",
                "risk_cost": f"{float(result.risk_cost):.4f}",
                "ambiguity_cost": f"{float(result.ambiguity_cost):.4f}",
                "obstacle_cost": f"{float(result.obstacle_cost):.4f}",
            }
        )
        print(
            f"H{horizon}: source={result.selected_source}, "
            f"d_goal={result.terminal_goal_distance_pred:.3f}, "
            f"solve={result.solve_time_s:.2f}s, valid={result.rollout_valid}"
        )

    fig.suptitle(
        "Initial visibility-aware plan by horizon — post-pick start, before Gazebo execution",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT}")
    print(f"wrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
