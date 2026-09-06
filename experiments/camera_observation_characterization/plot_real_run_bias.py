#!/usr/bin/env python3
"""Compare camera-reading bias over actual elapsed time across exact drive directories.

Every camera reading is loaded through fusion_on_fixed_routes/aligned.py, deduplicated by
(camera, capture stamp), and scored against ground truth at that same capture stamp. The
figure never pools runs: each drive keeps its own column, and the shared row scales exist
only so the three box interpretations can be read against each other. Camera-reading error
is kept separate from fused and belief error throughout.

The runs are separate closed-loop drives on the same frozen route. They did not see an
identical observation stream, so this is one-seed diagnostic evidence, not a replicated
comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "logs/studies/camera_observation_characterization_20260831"
for rel in ("experiments/deck_figures", "experiments/fusion_on_fixed_routes"):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
import aligned as A  # noqa: E402

FOLDER = "08_on_a_real_drive"
FILENAME = "19_raw_fixed_hull_over_time.png"
MODEL_LABEL = {
    "raw_box": "Raw YOLO box bottom-centre → floor",
    "fixed_offset": "Fixed-offset YOLO box interpretation",
    "hull": "Analytic-hull YOLO box interpretation",
}
MODEL_SHORT = {
    "raw_box": "Raw box → floor",
    "fixed_offset": "Fixed 30.9 cm shift",
    "hull": "Analytic hull",
    "learned_linear": "Learned linear correction",
    "learned_neural": "Learned neural correction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_geometry() -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray([float(item.x), float(item.y)])
        for item in D.layout().cameras
    }


def components(reading: dict, camera_xy: dict[str, np.ndarray]) -> tuple[float, float]:
    truth = np.asarray(reading["truth"], dtype=float)
    residual = np.asarray(reading["error"], dtype=float)
    ray = truth - camera_xy[reading["camera"]]
    norm = float(np.linalg.norm(ray))
    if norm <= 1e-12:
        return math.nan, math.nan
    along = ray / norm
    left = np.asarray([-along[1], along[0]])
    return float(np.dot(residual, along)), float(np.dot(residual, left))


def binned_median(x: np.ndarray, y: np.ndarray, *, width_s: float = 4.0):
    if not x.size:
        return np.array([]), np.array([])
    edges = np.arange(0.0, float(np.max(x)) + width_s, width_s)
    centres, medians = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = y[(x >= low) & (x < high)]
        if selected.size:
            centres.append(0.5 * (low + high))
            medians.append(float(np.median(selected)))
        elif centres and not math.isnan(centres[-1]):
            centres.append(math.nan)
            medians.append(math.nan)
    return np.asarray(centres), np.asarray(medians)


def blind_spans(times: np.ndarray, *, minimum_s: float = 1.5) -> list[tuple[float, float]]:
    unique = np.asarray(sorted(set(float(value) for value in times)))
    return [
        (float(start), float(end))
        for start, end in zip(unique[:-1], unique[1:])
        if end - start >= minimum_s
    ]


def load_drive(run: Path) -> dict:
    """Everything one exact run contributes to the comparison sheet."""
    required = tuple(run / name for name in (
        "run_manifest.json", "run_summary.json", "experiment.csv",
        "fusion_observations.csv", "correction_assimilations.csv",
    ))
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Missing required run evidence: {path}")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    schema = A.schema_version(run)
    if schema < 4:
        raise RuntimeError(f"{run.name}: actual-run plot requires schema 4 or newer")
    if not summary.get("completed") or not summary.get("valid_run", False):
        raise RuntimeError(f"{run.name}: run must be completed and valid")

    # Contract loader: dedupe physical readings and score each at obs_stamp.
    loaded = A.readings(run, admitted_only=False, dedupe=True, require_capture_time=True)
    first_cmd = float(summary["first_cmd_stamp"])
    stop_stamp = float(summary["stop_stamp"])
    readings = [item for item in loaded if first_cmd <= item["obs_stamp"] <= stop_stamp]
    if not readings:
        raise RuntimeError(f"{run.name}: no capture-aligned readings during the driven interval")

    camera_xy = camera_geometry()
    for item in readings:
        along, across = components(item, camera_xy)
        item["along_m"] = along
        item["across_m"] = across
        item["magnitude_m"] = item["error_cm"] / 100.0

    times = np.asarray([item["obs_stamp"] - first_cmd for item in readings])
    table = A.rows(run)
    truth = A.truth_series(run, table)
    in_drive = (truth.t >= first_cmd) & (truth.t <= stop_stamp)
    errors_cm = np.asarray([item["error_cm"] for item in readings])
    return {
        "run": run,
        "manifest": manifest,
        "summary": summary,
        "schema": schema,
        "readings": readings,
        "times": times,
        "errors_cm": errors_cm,
        "duration_s": float(summary["elapsed_after_first_cmd_s"]),
        "collision_s": (
            float(summary["first_crash_stamp"]) - first_cmd
            if summary.get("collision_any") and summary.get("first_crash_stamp") is not None
            else None
        ),
        "spans": blind_spans(times),
        "route": np.asarray(json.loads(manifest["preselected_route_json"]), dtype=float),
        "truth_xy": (truth.x[in_drive], truth.y[in_drive]),
        "observation_model": str(manifest.get("manager_observation_model", "unknown")),
        "run_id": str(manifest.get("run_id", run.name)),
        "arm": str(manifest.get("method", "unknown")).replace("closed_loop_", ""),
        "completion": str(summary.get("completion_reason", "unknown")),
        "per_camera": {camera: sum(item["camera"] == camera for item in readings)
                       for camera in "ABCDE"},
    }


def row_limits(drives: list[dict], field: str, *, symmetric: bool) -> tuple[float, float]:
    """One scale per row so the three interpretations are directly comparable."""
    pooled = np.concatenate([
        np.asarray([float(reading[field]) for reading in drive["readings"]])
        for drive in drives
    ])
    if symmetric:
        limit = max(0.10, float(np.quantile(np.abs(pooled), 0.995)))
        return -limit, limit
    return 0.0, max(0.10, float(np.quantile(pooled, 0.995)) * 1.08)


def draw_time_panel(ax, drive: dict, field: str, bounds: tuple[float, float],
                    *, symmetric: bool) -> dict:
    readings = drive["readings"]
    times = drive["times"]
    all_values = np.asarray([float(reading[field]) for reading in readings])
    lower, upper = bounds

    for start, end in drive["spans"]:
        ax.axvspan(start, end, color="#efc37f", alpha=0.24, zorder=0)
    clipped = 0
    for camera in "ABCDE":
        indices = [index for index, reading in enumerate(readings)
                   if reading["camera"] == camera]
        for index in indices:
            value = all_values[index]
            shown = float(np.clip(value, lower, upper))
            clipped += int(shown != value)
            marker = "^" if value > upper else ("v" if value < lower else "o")
            ax.scatter(times[index], shown, s=25, marker=marker, color=D.CAM_COLOUR[camera],
                       alpha=0.72, edgecolors="white", linewidths=0.35, zorder=3)
    bx, by = binned_median(times, all_values)
    ax.plot(bx, np.clip(by, lower, upper), color=D.INK, lw=2.5, zorder=5)
    if symmetric:
        ax.axhline(0.0, color=D.MUTED, lw=1.2, linestyle="--", zorder=2)
    if drive["collision_s"] is not None:
        ax.axvline(drive["collision_s"], color="#c63131", lw=2.0, linestyle="--", zorder=6)
    ax.set_xlim(0.0, drive["duration_s"])
    ax.set_ylim(lower, upper)
    ax.grid(color="#e4e2dc", lw=0.8)
    ax.set_axisbelow(True)
    return {
        "plot_bounds_m": [lower, upper],
        "clipped_readings": clipped,
        "whole_run_median_m": float(np.median(all_values)),
        "whole_run_mean_m": float(np.mean(all_values)),
    }


def draw_sheet(
    drives: list[dict],
    target: Path,
    *,
    filename: str = FILENAME,
    suptitle: str | None = None,
    footnote: str | None = None,
) -> dict:
    columns = len(drives)
    magnitude_bounds = row_limits(drives, "magnitude_m", symmetric=False)
    along_bounds = row_limits(drives, "along_m", symmetric=True)
    across_bounds = row_limits(drives, "across_m", symmetric=True)

    fig = plt.figure(figsize=(6.6 * columns, 20.0))
    grid = fig.add_gridspec(
        4, columns, height_ratios=(1.20, 1.0, 1.0, 1.0),
        left=0.075, right=0.985, bottom=0.090, top=0.880, hspace=0.30, wspace=0.16,
    )

    panels: dict[str, dict] = {}
    for column, drive in enumerate(drives):
        map_ax = fig.add_subplot(grid[0, column])
        D.draw_warehouse(map_ax, D.layout(), show_cameras=True, camera_labels=True,
                         rack_alpha=0.72)
        map_ax.plot(drive["route"][:, 0], drive["route"][:, 1], color=D.MUTED, lw=2.0,
                    linestyle="--", zorder=4)
        x, y = drive["truth_xy"]
        map_ax.plot(x, y, color=D.ROBOT, lw=3.3, zorder=6)
        map_ax.plot(x[0], y[0], marker="o", ms=10, mfc="white", mec=D.ROBOT, mew=2.5,
                    zorder=7)
        if drive["summary"].get("collision_any"):
            map_ax.plot(x[-1], y[-1], marker="X", ms=14, color="#c63131", mec="white",
                        mew=1.2, zorder=8)
        per_camera = drive["per_camera"]
        errors_cm = drive["errors_cm"]
        context = drive.get(
            "context_line", f"arm {drive['arm']} · run {drive['run_id']}"
        )
        map_ax.set_title(
            f"{MODEL_SHORT.get(drive['observation_model'], drive['observation_model'])}\n"
            f"{context}\n"
            f"{len(drive['readings'])} readings "
            f"(A {per_camera['A']}, B {per_camera['B']}, C {per_camera['C']}, "
            f"D {per_camera['D']}, E {per_camera['E']})\n"
            f"median {np.median(errors_cm):.1f} cm · "
            f"90th percentile {np.quantile(errors_cm, 0.90):.1f} cm\n"
            f"{drive['completion']} after {drive['duration_s']:.1f} s, "
            f"{float(drive['summary']['path_length_m']):.1f} m driven",
            fontsize=12.2, fontweight="bold",
            color="#8a5a1a" if drive.get("oracle") else D.INK,
        )

        specifications = (
            (1, "magnitude_m", magnitude_bounds, False,
             "Camera-reading position error (m)"),
            (2, "along_m", along_bounds, True,
             "Signed error along camera ray (m)\npositive = lands away from camera"),
            (3, "across_m", across_bounds, True,
             "Signed error across camera ray (m)\npositive = lands left of ray"),
        )
        for row, field, bounds, symmetric, ylabel in specifications:
            ax = fig.add_subplot(grid[row, column])
            if drive.get("oracle"):
                ax.set_facecolor("#fbf3e6")
            report = draw_time_panel(ax, drive, field, bounds, symmetric=symmetric)
            panel_id = drive.get("panel_id", drive["run_id"])
            panels.setdefault(panel_id, {})[field] = report
            if column == 0:
                ax.set_ylabel(ylabel, fontsize=11.5)
            else:
                ax.tick_params(labelleft=False)
            if row == 3:
                ax.set_xlabel("Elapsed time after first motion command (s)", fontsize=11.5)

    for row, label in ((1, "How far each camera reading lands from the robot"),
                       (2, "Radial bias over actual elapsed time"),
                       (3, "Lateral bias over actual elapsed time")):
        position = fig.add_subplot(grid[row, :], frameon=False)
        position.set_xticks([])
        position.set_yticks([])
        position.set_title(label, fontsize=15, fontweight="bold", pad=16)
        position.patch.set_alpha(0.0)

    handles = [
        Line2D([0], [0], marker="o", lw=0, ms=8, color=D.CAM_COLOUR[camera],
               label=f"Camera {camera}") for camera in "ABCDE"
    ] + [
        Line2D([0], [0], color=D.INK, lw=2.5, label="4 s median"),
        Line2D([0], [0], color="#efc37f", lw=8, alpha=0.55,
               label="≥1.5 s without a logged reading"),
    ]
    if any(drive["collision_s"] is not None for drive in drives):
        handles.append(Line2D([0], [0], color="#c63131", lw=2, linestyle="--",
                              label="physical collision / run stop"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, fontsize=11,
               bbox_to_anchor=(0.5, 0.036))
    fig.suptitle(
        suptitle or (
            "ACTUAL RECORDED GAZEBO DRIVES — camera-reading bias over elapsed time\n"
            "Three separate closed-loop drives on the same frozen route, one box "
            "interpretation each\n"
            "Each row shares one scale so the columns compare; each reading is counted once "
            "and scored against truth at its own capture timestamp"
        ),
        fontsize=19,
        fontweight="bold",
        y=0.980,
    )
    fig.text(
        0.5, 0.006,
        footnote or (
            "Camera-reading layer only — not fused, belief or planner error. The runtime "
            "admission gate was enabled, so detector misses and boxes rejected before "
            "fusion_observations.csv cannot be recovered here. One seed per arm: diagnostic "
            "evidence, not a replicated comparison.\n"
            "Time axes are per drive: the drives are not the same length, and the hull run is "
            "the only one that reached the goal."
        ),
        ha="center", fontsize=11.5, color=D.MUTED,
    )

    target.mkdir(parents=True, exist_ok=True)
    image_path = target / filename
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {image_path}")
    return panels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Exact run directory; repeat for each column. "
                             "Globs and latest are not accepted.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    runs = [path.expanduser().resolve() for path in args.run]
    if len(set(runs)) != len(runs):
        raise RuntimeError("The same run directory was passed more than once")
    drives = [load_drive(run) for run in runs]
    models = [drive["observation_model"] for drive in drives]
    if len(set(models)) != len(models):
        raise RuntimeError(f"Each column needs a distinct observation model, got {models}")
    tasks = {drive["manifest"].get("task") for drive in drives}
    if len(tasks) != 1:
        raise RuntimeError(f"All runs must share one frozen route, got {tasks}")

    out = (args.out or DEFAULT_OUT).expanduser().resolve()
    target = out / FOLDER
    image_path = target / FILENAME
    if image_path.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {image_path}; pass --overwrite")

    panels = draw_sheet(drives, target)

    output_manifest = {
        "status": "complete",
        "schema": "actual_drive_camera_bias_profile.v2",
        "evidence_status": "diagnostic_only; not in a frozen replicated paper selection",
        "comparison_note": (
            "separate closed-loop drives on one frozen route; they did not see an identical "
            "observation stream, so column differences mix interpretation and drive"
        ),
        "task": sorted(tasks)[0],
        "shared_row_scales": True,
        "drives": [
            {
                "run_dir": str(drive["run"]),
                "run_manifest_sha256": sha256(drive["run"] / "run_manifest.json"),
                "run_summary_sha256": sha256(drive["run"] / "run_summary.json"),
                "fusion_observations_sha256": sha256(drive["run"] / "fusion_observations.csv"),
                "experiment_sha256": sha256(drive["run"] / "experiment.csv"),
                "correction_assimilations_sha256": sha256(
                    drive["run"] / "correction_assimilations.csv"),
                "run_id": drive["run_id"],
                "arm": drive["arm"],
                "seed": drive["manifest"].get("seed"),
                "logging_schema_version": drive["schema"],
                "observation_model": drive["observation_model"],
                "fusion_rule": drive["manifest"].get("manager_fusion_rule"),
                "manager_admission_gate": drive["manifest"].get("manager_admission_gate"),
                "completion_reason": drive["completion"],
                "duration_after_first_command_s": drive["duration_s"],
                "path_length_m": float(drive["summary"]["path_length_m"]),
                "camera_reading_layer": {
                    "n": len(drive["readings"]),
                    "dedupe_key": "(camera, obs_stamp)",
                    "reference": "ground truth interpolated at each obs_stamp via aligned.py",
                    "median_error_cm": float(np.median(drive["errors_cm"])),
                    "p90_error_cm": float(np.quantile(drive["errors_cm"], 0.90)),
                    "rmse_cm": float(np.sqrt(np.mean(drive["errors_cm"] ** 2))),
                    "per_camera_counts": drive["per_camera"],
                },
                "blind_spans_at_least_1_5s": drive["spans"],
                "panels": panels[drive["run_id"]],
            }
            for drive in drives
        ],
        "logging_boundary": (
            "fusion_observations.csv contains readings reaching the manager; detector misses "
            "and boxes rejected before this log are not available in this figure"
        ),
        "figure": FILENAME,
    }
    json_path = (target / FILENAME).with_suffix(".json")
    json_path.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "figure": str(target / FILENAME),
        "manifest": str(json_path),
        "drives": {
            drive["run_id"]: {
                "observation_model": drive["observation_model"],
                "n": len(drive["readings"]),
                "median_cm": float(np.median(drive["errors_cm"])),
                "p90_cm": float(np.quantile(drive["errors_cm"], 0.90)),
            }
            for drive in drives
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
