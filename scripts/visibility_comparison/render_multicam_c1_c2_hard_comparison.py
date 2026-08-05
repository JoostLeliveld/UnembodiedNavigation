#!/usr/bin/env python3
"""Render the matched C1/C2 hard-route Gazebo executions.

The estimator is identical in all four panels: fused YOLO observations from
external cameras A--D.  C1 plans with constant observation covariance; C2 uses
the fused four-camera visibility field in R_plan.  Ground truth is only drawn
after each run for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multicam_c1_c2_comparison_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from render_multicam_hard_gazebo_showcase import (
    DEFAULT_CAMPAIGN,
    DEFAULT_GP,
    DEFAULT_WORLD,
    EXECUTED,
    MUTED,
    REPO,
    ROUTE_LABELS,
    _camera_xy,
    _candidate_metrics,
    _draw_world,
    _execution_segment,
    _load_json,
    _path_visibility,
)

DEFAULT_OUTPUT = (
    REPO
    / "logs/studies/multicam_nav_demo/figures"
    / "fig37_multicam_hard_c1_c2_gazebo_routes.png"
)
TASKS = ("rob_hardA_free", "rob_hardB_free")
CONDITIONS = ("C1", "C2")
CONDITION_COLOURS = {"C1": "#d95f02", "C2": "#0072b2"}


def render(
    campaign_root: Path,
    gp_artifact: Path,
    world_path: Path,
    output: Path,
) -> tuple[Path, Path]:
    campaign = _load_json(campaign_root / "campaign_log.json")
    with np.load(gp_artifact, allow_pickle=False) as artifact:
        xs = np.asarray(artifact["xs"], dtype=float)
        ys = np.asarray(artifact["ys"], dtype=float)
        p_vis = np.asarray(artifact["P_conservative_plan_map"], dtype=float)
        camera_ids = [str(item) for item in artifact["camera_ids"]]
    probability = RegularGridInterpolator(
        (ys, xs), p_vis, bounds_error=False, fill_value=np.nan
    )
    cameras = _camera_xy(world_path)

    fig, axes = plt.subplots(
        2, 2, figsize=(15.5, 12.8), sharex=True, sharey=True, constrained_layout=False
    )
    summaries: dict[str, dict] = {}
    heatmap = None

    for row, task in enumerate(TASKS):
        for column, condition in enumerate(CONDITIONS):
            axis = axes[row, column]
            key = f"{task}__{condition}__seed0"
            record = campaign[key]
            if not record.get("goal_reached"):
                raise RuntimeError(f"{key} is not a successful goal-reaching run")

            run_dir = Path(record["run_dir"])
            frame = pd.read_csv(run_dir / "experiment.csv")
            manifest = _load_json(run_dir / "run_manifest.json")
            run_summary = _load_json(run_dir / "run_summary.json")
            meta = _load_json(run_dir / "global_plan_meta.json")
            plan = pd.read_csv(run_dir / "global_plan.csv")[["x", "y"]].to_numpy(float)
            execution, belief_error = _execution_segment(frame)
            execution_metrics = _path_visibility(execution, probability)
            candidates = _candidate_metrics(meta, probability)
            selected_name = str(meta["selected_source"]).split(":")[-1]
            most_visible_name = max(
                candidates, key=lambda name: candidates[name]["mean_p_vis"]
            )
            colour = CONDITION_COLOURS[condition]

            heatmap = axis.imshow(
                p_vis,
                extent=(xs[0], xs[-1], ys[0], ys[-1]),
                origin="lower",
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
                alpha=0.72,
                interpolation="bilinear",
                zorder=0,
            )
            _draw_world(axis, manifest["collision_geometry_json"])

            start = np.asarray(meta["start_xy_yaw"][:2], dtype=float)
            for route in meta["route_seeds"]:
                name = str(route["name"])
                points = np.asarray([start, *route["waypoints"]], dtype=float)
                selected = name == selected_name
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    color=colour if selected else MUTED,
                    linewidth=2.1 if selected else 0.9,
                    linestyle=(0, (5, 3)),
                    alpha=0.92 if selected else 0.30,
                    zorder=5 if selected else 4,
                )

            axis.plot(
                plan[:, 0],
                plan[:, 1],
                color=colour,
                linewidth=2.8,
                alpha=0.98,
                zorder=7,
            )
            axis.plot(
                execution[:, 0],
                execution[:, 1],
                color=EXECUTED,
                linewidth=2.0,
                linestyle=(0, (1.0, 1.4)),
                zorder=9,
            )
            goal = np.asarray(meta["goal_xy"], dtype=float)
            axis.scatter(
                *start, s=62, color="#009e73", edgecolor="white", linewidth=1.0, zorder=10
            )
            axis.scatter(
                *goal,
                s=165,
                marker="*",
                color="#cc79a7",
                edgecolor="white",
                linewidth=0.9,
                zorder=10,
            )
            for camera_id, xy in cameras.items():
                axis.scatter(
                    *xy,
                    marker="^",
                    s=48,
                    color="white",
                    edgecolor="#222222",
                    zorder=11,
                )
                axis.annotate(
                    camera_id,
                    xy=xy,
                    xytext=(0, -13 if xy[1] > 0.0 else 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    weight="bold",
                    zorder=12,
                )

            planning_note = (
                "constant R (visibility not used)"
                if condition == "C1"
                else "fused visibility-conditioned R"
            )
            axis.set_title(
                f"{condition} · {task.replace('_free', '')} · "
                f"{ROUTE_LABELS.get(selected_name, selected_name)}\n"
                f"{planning_note}",
                fontsize=11.2,
                weight="bold",
                color=colour,
            )
            axis.text(
                0.015,
                0.015,
                f"executed mean Pvis  {execution_metrics['mean_p_vis']:.3f}\n"
                f"path / goal gap     {record['path_length_m']:.2f} / "
                f"{record['minimum_goal_distance']:.3f} m\n"
                f"obstacle clearance  {run_summary['min_obstacle_distance_m']:.3f} m\n"
                f"belief error         {np.mean(belief_error):.3f} m",
                transform=axis.transAxes,
                va="bottom",
                ha="left",
                fontsize=7.8,
                family="monospace",
                color="#1d2730",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.88,
                    "edgecolor": "#aab3bb",
                    "pad": 5,
                },
                zorder=20,
            )
            axis.set_xlim(-12.1, 12.1)
            axis.set_ylim(-10.2, 10.2)
            axis.set_aspect("equal")
            axis.set_xlabel("map x [m]")
            axis.set_ylabel("map y [m]")
            axis.grid(color="white", alpha=0.20, linewidth=0.5)

            summaries[key] = {
                "task": task,
                "condition": condition,
                "planner": record["planner"],
                "seed": record["seed"],
                "goal_reached": True,
                "selected_route": selected_name,
                "most_visible_candidate": most_visible_name,
                "selected_is_most_visible_candidate": selected_name
                == most_visible_name,
                "candidate_visibility": candidates,
                "executed_visibility": execution_metrics,
                "path_length_m": float(record["path_length_m"]),
                "minimum_goal_distance_m": float(record["minimum_goal_distance"]),
                "minimum_obstacle_distance_m": float(
                    run_summary["min_obstacle_distance_m"]
                ),
                "mean_belief_error_gt_m": float(np.mean(belief_error)),
                "global_terminal_goal_distance_pred_m": float(
                    meta["terminal_goal_distance_pred"]
                ),
                "selected_source": meta["selected_source"],
                "run_dir": str(run_dir),
            }

    handles = [
        Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (5, 3)),
               label="available lane-graph seed"),
        Line2D([], [], color=CONDITION_COLOURS["C1"], linewidth=2.8,
               label="C1 selected plan"),
        Line2D([], [], color=CONDITION_COLOURS["C2"], linewidth=2.8,
               label="C2 selected plan"),
        Line2D([], [], color=EXECUTED, linewidth=2.0, linestyle=(0, (1, 1.4)),
               label="executed GT (evaluation only)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=9.2,
    )
    colorbar = fig.colorbar(heatmap, ax=axes, fraction=0.021, pad=0.015)
    colorbar.set_label("four-camera conservative planning $P_{vis}$")
    fig.suptitle(
        "Matched Gazebo comparison: constant-R C1 versus visibility-aware C2",
        x=0.055,
        y=0.988,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.055,
        0.956,
        "Same waypoint-free tasks, fused external-camera A–D estimator, dynamics, "
        "goal prior, and EFE weights; only the planning observation model differs.",
        ha="left",
        fontsize=9.8,
        color="#4d5965",
    )
    fig.text(
        0.5,
        0.010,
        "SHOWCASE: one seed per task/condition, not a robustness campaign. "
        "Ground truth is never supplied to the planner.",
        ha="center",
        fontsize=8.7,
        color="#8a3232",
        weight="bold",
    )
    fig.subplots_adjust(
        top=0.915, bottom=0.085, left=0.055, right=0.925, hspace=0.20, wspace=0.08
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "visibility_artifact": str(gp_artifact),
                "visibility_field": "P_conservative_plan_map",
                "camera_ids": camera_ids,
                "state_correction_mode": "fused",
                "controlled_difference": {
                    "C1": "constant observation covariance in planning",
                    "C2": "fused four-camera visibility-conditioned covariance in planning",
                },
                "ground_truth_usage": "post-run evaluation only",
                "runs": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--gp-artifact", type=Path, default=DEFAULT_GP)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    figure, summary = render(
        args.campaign_root.resolve(),
        args.gp_artifact.resolve(),
        args.world.resolve(),
        args.output.resolve(),
    )
    print(f"Wrote {figure}")
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
