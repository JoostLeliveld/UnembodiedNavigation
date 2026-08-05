#!/usr/bin/env python3
"""Show the waypoint-free hard-route choices and their real Gazebo executions.

Each panel overlays:

* the fixed, four-camera conservative visibility field used by C2 planning;
* every map-derived lane-graph seed available to the one-shot global solve;
* the selected global plan; and
* the ground-truth trajectory, used only for post-run evaluation.

The renderer does not re-solve or feed ground truth into planning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multicam_hard_gazebo_showcase_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(Path(__file__))
sys.path.insert(0, str(REPO / "src" / "unav_common"))
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402

DEFAULT_CAMPAIGN = (
    REPO
    / "logs/visibility_comparison/multicam_hard_free_fused_terminal_gate_v2"
)
DEFAULT_GP = (
    REPO
    / "logs/visibility_comparison/spawn_grid_20260727"
    / "fused_planner_four_camera.npz"
)
DEFAULT_WORLD = (
    REPO
    / "src/sim/gazebo_worlds/worlds"
    / "warehouse_full_4cam.world.sdf"
)
DEFAULT_OUTPUT = (
    REPO
    / "logs/studies/multicam_nav_demo/figures"
    / "fig36_multicam_hard_gazebo_routes.png"
)

TASK_KEYS = (
    "rob_hardA_free__C2__seed0",
    "rob_hardB_free__C2__seed0",
)
ROUTE_LABELS = {
    "below_south_cross_aisle": "south crossing",
    "below_main_aisle": "main aisle",
    "above_connector": "connector",
    "above_cross_aisle": "north crossing",
}
SELECTED = "#cc4c02"
EXECUTED = "#111111"
MUTED = "#59636e"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _camera_xy(world_path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(world_path).getroot()
    cameras: dict[str, tuple[float, float]] = {}
    for include in root.findall(".//include"):
        name = (include.findtext("name") or "").strip()
        if not name.startswith("external_camera"):
            continue
        pose = (include.findtext("pose") or "").split()
        if len(pose) >= 2:
            camera_id = {
                "external_camera": "A",
                "external_camera_b": "B",
                "external_camera_c": "C",
                "external_camera_d": "D",
            }.get(name, name)
            cameras[camera_id] = (float(pose[0]), float(pose[1]))
    return cameras


def _draw_world(axis, collision_geometry_json: str) -> None:
    for prism in scene_from_json(collision_geometry_json).prisms:
        if "wall_" in str(prism.name):
            continue
        axis.add_patch(
            Rectangle(
                (prism.xmin, prism.ymin),
                prism.xmax - prism.xmin,
                prism.ymax - prism.ymin,
                facecolor="#737a80",
                edgecolor="#343a40",
                linewidth=0.45,
                alpha=0.85,
                zorder=3,
            )
        )


def _densify(points: np.ndarray, spacing_m: float = 0.04) -> np.ndarray:
    dense: list[np.ndarray] = []
    for first, second in zip(points[:-1], points[1:]):
        count = max(2, int(np.ceil(np.linalg.norm(second - first) / spacing_m)) + 1)
        dense.extend(np.linspace(first, second, count)[:-1])
    dense.append(points[-1])
    return np.asarray(dense, dtype=float)


def _path_visibility(
    points: np.ndarray,
    probability: RegularGridInterpolator,
) -> dict[str, float]:
    points = np.asarray(points, dtype=float)
    segment_length = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    midpoint = 0.5 * (points[:-1, :2] + points[1:, :2])
    values = np.asarray(probability(midpoint[:, [1, 0]]), dtype=float)
    valid = np.isfinite(values) & np.isfinite(segment_length)
    total = float(np.sum(segment_length[valid]))
    if total <= 1e-12:
        return {"length_m": 0.0, "mean_p_vis": float("nan"), "low_fraction": float("nan")}
    return {
        "length_m": total,
        "mean_p_vis": float(np.sum(segment_length[valid] * values[valid]) / total),
        "low_fraction": float(np.sum(segment_length[valid & (values < 0.2)]) / total),
    }


def _execution_segment(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    linear = pd.to_numeric(frame["cmd_v"], errors="coerce").fillna(0.0).to_numpy(float)
    angular = pd.to_numeric(frame["cmd_w"], errors="coerce").fillna(0.0).to_numpy(float)
    moving = np.flatnonzero(np.hypot(linear, angular) > 0.03)
    if moving.size == 0:
        raise RuntimeError("run has no non-zero command")
    goal_distance = pd.to_numeric(frame["goal_dist"], errors="coerce").to_numpy(float)
    finish = int(np.nanargmin(goal_distance))
    start = int(moving[0])
    if finish <= start:
        raise RuntimeError("closest-goal sample precedes the first command")
    segment = frame.iloc[start : finish + 1]
    xy = segment[["gt_x", "gt_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    error = pd.to_numeric(segment["belief_error_gt_m"], errors="coerce").to_numpy(float)
    finite_xy = np.isfinite(xy).all(axis=1)
    return xy[finite_xy], error[np.isfinite(error)]


def _candidate_metrics(
    meta: dict,
    probability: RegularGridInterpolator,
) -> dict[str, dict[str, float]]:
    start = np.asarray(meta["start_xy_yaw"][:2], dtype=float)
    metrics = {}
    for route in meta["route_seeds"]:
        points = np.asarray([start, *route["waypoints"]], dtype=float)
        metrics[str(route["name"])] = _path_visibility(_densify(points), probability)
    return metrics


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

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.8), sharex=True, sharey=True)
    summaries: dict[str, dict] = {}
    heatmap = None

    for axis, key in zip(axes, TASK_KEYS):
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

        heatmap = axis.imshow(
            p_vis,
            extent=(xs[0], xs[-1], ys[0], ys[-1]),
            origin="lower",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            alpha=0.78,
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
                color=SELECTED if selected else MUTED,
                linewidth=2.3 if selected else 1.15,
                linestyle=(0, (5, 3)),
                alpha=0.95 if selected else 0.46,
                zorder=5 if selected else 4,
            )
            anchor = points[max(1, len(points) // 2)]
            label_on_right = bool(anchor[0] > 4.0)
            axis.annotate(
                f"{ROUTE_LABELS.get(name, name)}  {candidates[name]['mean_p_vis']:.3f}",
                xy=anchor,
                xytext=(-4 if label_on_right else 4, 4),
                textcoords="offset points",
                ha="right" if label_on_right else "left",
                fontsize=7.5,
                color=SELECTED if selected else "#454d55",
                weight="bold" if selected else "normal",
                zorder=8,
            )

        axis.plot(
            plan[:, 0],
            plan[:, 1],
            color=SELECTED,
            linewidth=2.6,
            alpha=0.95,
            zorder=7,
            label="selected global plan",
        )
        axis.plot(
            execution[:, 0],
            execution[:, 1],
            color=EXECUTED,
            linewidth=2.1,
            linestyle=(0, (1.0, 1.4)),
            zorder=9,
            label="executed GT (evaluation only)",
        )
        goal = np.asarray(meta["goal_xy"], dtype=float)
        axis.scatter(*start, s=70, color="#009e73", edgecolor="white", linewidth=1.1, zorder=10)
        axis.scatter(*goal, s=180, marker="*", color="#cc79a7", edgecolor="white", linewidth=1.0, zorder=10)
        for camera_id, xy in cameras.items():
            axis.scatter(*xy, marker="^", s=55, color="#f7f7f7", edgecolor="#222222", zorder=11)
            camera_label_dy = -14 if xy[1] > 0.0 else 7
            axis.annotate(camera_id, xy=xy, xytext=(0, camera_label_dy), textcoords="offset points",
                          ha="center", fontsize=7.5, weight="bold", zorder=12)

        route_lines = "\n".join(
            f"{'●' if name == selected_name else '○'} "
            f"{ROUTE_LABELS.get(name, name):<14}  "
            f"mean Pvis {values['mean_p_vis']:.3f}  "
            f"low {values['low_fraction']:.1%}"
            for name, values in candidates.items()
        )
        axis.text(
            0.015,
            0.015,
            route_lines,
            transform=axis.transAxes,
            va="bottom",
            ha="left",
            fontsize=7.3,
            family="monospace",
            color="#1d2730",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#aab3bb", "pad": 5},
            zorder=20,
        )
        task_label = str(record["task"]).replace("_free", "")
        axis.set_title(
            f"{task_label}: {ROUTE_LABELS.get(selected_name, selected_name)} selected\n"
            f"executed mean $P_{{vis}}$ {execution_metrics['mean_p_vis']:.3f} · "
            f"goal gap {record['minimum_goal_distance']:.3f} m · "
            f"clearance {run_summary['min_obstacle_distance_m']:.3f} m",
            fontsize=11.5,
            weight="bold",
        )
        axis.set_xlim(-12.1, 12.1)
        axis.set_ylim(-10.2, 10.2)
        axis.set_aspect("equal")
        axis.set_xlabel("map x [m]")
        axis.set_ylabel("map y [m]")
        axis.grid(color="white", alpha=0.22, linewidth=0.55)

        summaries[str(record["task"])] = {
            "condition": record["condition"],
            "seed": record["seed"],
            "goal_reached": bool(record["goal_reached"]),
            "collision_detected": bool(record.get("crash_detected", False)),
            "selected_route": selected_name,
            "most_visible_candidate": most_visible_name,
            "selected_is_most_visible_candidate": selected_name == most_visible_name,
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
        Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (5, 3)),
               label="available lane-graph seed"),
        Line2D([], [], color=SELECTED, linewidth=2.6, label="selected global plan"),
        Line2D([], [], color=EXECUTED, linewidth=2.1, linestyle=(0, (1, 1.4)),
               label="executed GT (post-run evaluation)"),
        Line2D([], [], marker="^", color="none", markerfacecolor="white",
               markeredgecolor="#222222", label="external cameras A–D"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    colorbar = fig.colorbar(heatmap, ax=axes, fraction=0.025, pad=0.015)
    colorbar.set_label("four-camera conservative planning $P_{vis}$")
    fig.suptitle(
        "Real Gazebo: hard routes choose the visible cross-aisles",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.055,
        0.945,
        "Waypoint-free global solve · fused four-camera YOLO measurement · fixed visibility/R-plan weights · "
        "candidate labels report distance-weighted fused visibility",
        ha="left",
        fontsize=9.7,
        color="#4d5965",
    )
    fig.text(
        0.5,
        0.010,
        "SHOWCASE: one seed per hard task (C2), not a robustness campaign. "
        "Ground truth is logged for evaluation only and is never supplied to the planner.",
        ha="center",
        fontsize=8.8,
        color="#8a3232",
        weight="bold",
    )
    fig.subplots_adjust(top=0.88, bottom=0.135, left=0.055, right=0.92, wspace=0.10)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    summary_path = output.with_suffix(".json")
    payload = {
        "campaign_root": str(campaign_root),
        "visibility_artifact": str(gp_artifact),
        "visibility_field": "P_conservative_plan_map",
        "camera_ids": camera_ids,
        "state_correction_mode": "fused",
        "ground_truth_usage": "post-run evaluation only",
        "tasks": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
