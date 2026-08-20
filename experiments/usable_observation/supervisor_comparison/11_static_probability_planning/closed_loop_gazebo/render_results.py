#!/usr/bin/env python3
"""Render the prespecified matched C1/C2/C3 closed-loop Gazebo smoke test.

Evaluation position is always Gazebo GT (gt_x/gt_y). Belief position is always the
canonical planner belief checked by scripts/geometry_visibility/campaign_metrics.py.
The script only consumes the exact run directories frozen in campaign_log.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/static_puse_closed_loop_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "scripts" / "geometry_visibility"))
sys.path.insert(0, str(REPO / "scripts" / "visibility_comparison"))

import campaign_metrics as canonical  # noqa: E402
from render_multicam_hard_gazebo_showcase import _camera_xy, _draw_world  # noqa: E402


DEFAULT_CAMPAIGN = REPO / "logs/visibility_comparison/static_puse_closed_loop_gazebo_v1"
DEFAULT_GP = REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz"
DEFAULT_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
CONDITIONS = ("C1", "C2", "C3")
LABELS = {
    "C1": "constant R",
    "C2": r"static $p_{use}$: $R/p$",
    "C3": r"static $p_{use}$: hit/miss",
}
COLOURS = {"C1": "#d55e00", "C2": "#0072b2", "C3": "#009e73"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_record(campaign: dict, condition: str) -> dict:
    matches = [v for v in campaign.values() if v.get("condition") == condition]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {condition} record, found {len(matches)}")
    record = matches[0]
    if not record.get("run_dir"):
        raise RuntimeError(f"{condition} has no completed run_dir: {record}")
    return record


def _execution(run_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    frame = pd.read_csv(run_dir / "experiment.csv")
    summary = _json(run_dir / "run_summary.json")
    manifest = _json(run_dir / "run_manifest.json")
    first = float(summary["first_cmd_stamp"])
    stamp = pd.to_numeric(frame["stamp"], errors="coerce")
    execution = frame.loc[stamp >= first].copy()
    if execution.empty:
        raise RuntimeError(f"No post-command rows in {run_dir}")
    # Invoke the repository contract's self-checking loader before any metric use.
    canonical.load_run(str(run_dir / "experiment.csv"))
    return execution, summary, manifest


def _finite(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def _metrics(frame: pd.DataFrame, summary: dict, probability) -> dict:
    bx, by = _finite(frame, "planner_belief_x"), _finite(frame, "planner_belief_y")
    gx, gy = _finite(frame, "gt_x"), _finite(frame, "gt_y")
    error = _finite(frame, "belief_error_gt_m")
    t = _finite(frame, "stamp") - float(summary["first_cmd_stamp"])
    mask = np.isfinite(bx) & np.isfinite(by) & np.isfinite(gx) & np.isfinite(gy) & np.isfinite(error)
    nees = []
    coverage = []
    cxx = _finite(frame, "planner_cov_x")
    cxy = _finite(frame, "planner_cov_xy")
    cyy = _finite(frame, "planner_cov_y")
    for i in np.flatnonzero(mask):
        cov = np.array([[cxx[i], cxy[i]], [cxy[i], cyy[i]]], dtype=float)
        if not np.all(np.isfinite(cov)) or np.linalg.det(cov) <= 1e-12:
            continue
        delta = np.array([bx[i] - gx[i], by[i] - gy[i]])
        value = float(delta @ np.linalg.solve(cov, delta))
        nees.append(value)
        coverage.append(value <= 5.991)  # 95% chi-square threshold, 2 DoF.
    p = probability(np.column_stack((gy[mask], gx[mask]))) if mask.any() else np.array([])
    return {
        "samples_after_first_cmd": int(mask.sum()),
        "belief_rmse_gt_m": float(np.sqrt(np.mean(np.square(error[mask])))),
        "belief_error_p95_gt_m": float(np.percentile(error[mask], 95)),
        "belief_error_max_gt_m": float(np.max(error[mask])),
        "median_nees_xy": float(np.median(nees)) if nees else math.nan,
        "coverage_95_xy": float(np.mean(coverage)) if coverage else math.nan,
        "mean_static_puse_along_gt": float(np.nanmean(p)) if len(p) else math.nan,
        "minimum_static_puse_along_gt": float(np.nanmin(p)) if len(p) else math.nan,
        "path_length_m": float(summary["path_length_m"]),
        "minimum_goal_distance_m": float(summary["minimum_goal_distance"]),
        "elapsed_after_first_cmd_s": float(summary["elapsed_after_first_cmd_s"]),
        "min_obstacle_distance_m": float(summary["min_obstacle_distance_m"]),
        "collision_any": bool(summary["collision_any"]),
        "goal_reached": bool(summary["goal_region_success"]),
        "completion_reason": str(summary["completion_reason"]),
        "time_s": t[mask],
        "error_m": error[mask],
        "gt_x": gx[mask],
        "gt_y": gy[mask],
        "belief_x": bx[mask],
        "belief_y": by[mask],
    }


def render(campaign_root: Path, gp_path: Path, world_path: Path, output_dir: Path) -> list[Path]:
    campaign = _json(campaign_root / "campaign_log.json")
    with np.load(gp_path, allow_pickle=False) as artifact:
        xs = np.asarray(artifact["xs"], dtype=float)
        ys = np.asarray(artifact["ys"], dtype=float)
        p_use = np.asarray(artifact["P_conservative_plan_map"], dtype=float)
        camera_ids = [str(item) for item in artifact["camera_ids"]]
    probability = RegularGridInterpolator((ys, xs), p_use, bounds_error=False, fill_value=np.nan)
    cameras = _camera_xy(world_path)

    records: dict[str, dict] = {}
    metrics: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    manifests: dict[str, dict] = {}
    for condition in CONDITIONS:
        records[condition] = _condition_record(campaign, condition)
        run_dir = Path(records[condition]["run_dir"])
        frame, summary, manifest = _execution(run_dir)
        frames[condition], summaries[condition], manifests[condition] = frame, summary, manifest
        metrics[condition] = _metrics(frame, summary, probability)
        metrics[condition]["run_dir"] = str(run_dir)
        meta_path = run_dir / "global_plan_meta.json"
        metrics[condition]["selected_source"] = _json(meta_path).get("selected_source", "unknown")

    output_dir.mkdir(parents=True, exist_ok=True)
    route_path = output_dir / "01_closed_loop_routes.png"
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.0), sharex=True, sharey=True)
    heatmap = None
    for axis, condition in zip(axes, CONDITIONS):
        frame = frames[condition]
        run_dir = Path(records[condition]["run_dir"])
        heatmap = axis.imshow(
            p_use, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower",
            cmap="YlGnBu", vmin=0.0, vmax=1.0, alpha=0.72, interpolation="bilinear",
        )
        _draw_world(axis, manifests[condition]["collision_geometry_json"])
        plan = pd.read_csv(run_dir / "global_plan.csv")
        px, py = _finite(plan, "x"), _finite(plan, "y")
        axis.plot(px, py, color=COLOURS[condition], linewidth=2.7, label="global plan")
        axis.plot(metrics[condition]["gt_x"], metrics[condition]["gt_y"], color="#111111",
                  linewidth=2.0, linestyle=(0, (1.0, 1.2)), label="executed Gazebo GT")
        axis.plot(metrics[condition]["belief_x"], metrics[condition]["belief_y"], color="#cc79a7",
                  linewidth=1.0, alpha=0.85, label="planner belief")
        goal = np.array([_finite(frame, "goal_x")[-1], _finite(frame, "goal_y")[-1]])
        start = np.array([metrics[condition]["gt_x"][0], metrics[condition]["gt_y"][0]])
        axis.scatter(*start, s=58, color="#f0e442", edgecolor="#222222", zorder=8)
        axis.scatter(*goal, s=150, marker="*", color="#cc79a7", edgecolor="white", zorder=8)
        for camera_id, xy in cameras.items():
            axis.scatter(*xy, marker="^", s=45, color="white", edgecolor="#222222", zorder=9)
            axis.annotate(camera_id, xy=xy, xytext=(0, 6), textcoords="offset points",
                          ha="center", fontsize=7, weight="bold")
        m = metrics[condition]
        axis.set_title(f"{condition}: {LABELS[condition]}\n{m['completion_reason'].upper()} · "
                       f"goal gap {m['minimum_goal_distance_m']:.2f} m",
                       color=COLOURS[condition], weight="bold")
        axis.text(0.02, 0.02,
                  f"path       {m['path_length_m']:.2f} m\n"
                  f"belief RMSE {m['belief_rmse_gt_m']:.3f} m\n"
                  f"mean p_use  {m['mean_static_puse_along_gt']:.3f}\n"
                  f"clearance   {m['min_obstacle_distance_m']:.3f} m",
                  transform=axis.transAxes, fontsize=8, family="monospace", va="bottom",
                  bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#aab3bb"})
        axis.set_xlim(-12.1, 12.1)
        axis.set_ylim(-10.2, 10.2)
        axis.set_aspect("equal")
        axis.set_xlabel("map x [m]")
        axis.grid(color="white", alpha=0.20, linewidth=0.5)
    axes[0].set_ylabel("map y [m]")
    fig.colorbar(heatmap, ax=axes, fraction=0.019, pad=0.014,
                 label=r"frozen four-camera planning $p_{use}(x)$")
    fig.legend(handles=[
        Line2D([], [], color=COLOURS["C2"], linewidth=2.7, label="planned path (condition colour)"),
        Line2D([], [], color="#111111", linewidth=2, linestyle=(0, (1, 1.2)), label="executed Gazebo GT"),
        Line2D([], [], color="#cc79a7", linewidth=1, label="planner belief"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.052), ncol=3, frameon=False)
    fig.suptitle("Matched four-camera closed-loop Gazebo feasibility run", x=0.04, ha="left",
                 fontsize=16, weight="bold")
    fig.text(0.04, 0.925, "Same world, detector, filter, dynamics, noise seed and EFE weights; only the planning observation model changes.",
             fontsize=9.5, color="#4d5965")
    fig.text(0.5, 0.014, "One prespecified seed per condition: feasibility evidence, not a powered comparison. Gazebo GT is evaluation-only.",
             ha="center", fontsize=8.5, color="#8a3232", weight="bold")
    fig.subplots_adjust(left=0.045, right=0.93, top=0.86, bottom=0.18, wspace=0.07)
    fig.savefig(route_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    diagnostic_path = output_dir / "02_closed_loop_metrics.png"
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6))
    for condition in CONDITIONS:
        m = metrics[condition]
        axes[0].plot(m["time_s"], m["error_m"], color=COLOURS[condition], label=condition, linewidth=1.8)
    axes[0].set_title("Belief error against Gazebo GT")
    axes[0].set_xlabel("time after first command [s]")
    axes[0].set_ylabel("position error [m]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    x = np.arange(3)
    axes[1].bar(x, [metrics[c]["belief_rmse_gt_m"] for c in CONDITIONS],
                color=[COLOURS[c] for c in CONDITIONS])
    axes[1].set_xticks(x, CONDITIONS)
    axes[1].set_title("Belief RMSE after motion starts")
    axes[1].set_ylabel("RMSE [m]")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, [metrics[c]["minimum_goal_distance_m"] for c in CONDITIONS],
                color=[COLOURS[c] for c in CONDITIONS])
    axes[2].set_xticks(x, CONDITIONS)
    axes[2].set_title("Closest approach to goal")
    axes[2].set_ylabel("minimum goal distance [m]")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Closed-loop outcomes and estimator behavior", x=0.045, ha="left",
                 fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(diagnostic_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    serializable: dict[str, dict] = {}
    csv_rows = []
    for condition in CONDITIONS:
        serializable[condition] = {
            key: value for key, value in metrics[condition].items()
            if not isinstance(value, np.ndarray)
        }
        row = {"condition": condition, "label": LABELS[condition], **serializable[condition]}
        csv_rows.append(row)
    json_path = output_dir / "closed_loop_metrics.json"
    json_path.write_text(json.dumps({
        "evidence_level": "one-seed closed-loop Gazebo feasibility experiment",
        "experimental_unit": "one complete navigation run",
        "metric_object": "planner belief during navigation",
        "reference": "Gazebo ground truth gt_x/gt_y, evaluation only",
        "position_frame": "map_bev",
        "covariance_for_nees": "planner_cov_x/planner_cov_xy/planner_cov_y",
        "visibility_field": "P_conservative_plan_map",
        "visibility_artifact": str(gp_path),
        "camera_ids": camera_ids,
        "campaign_log": str(campaign_root / "campaign_log.json"),
        "runs": serializable,
    }, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    csv_path = output_dir / "closed_loop_metrics.csv"
    fields = list(csv_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    return [route_path, diagnostic_path, json_path, csv_path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--gp-artifact", type=Path, default=DEFAULT_GP)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    for path in render(args.campaign_root.resolve(), args.gp_artifact.resolve(),
                       args.world.resolve(), args.output_dir.resolve()):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
