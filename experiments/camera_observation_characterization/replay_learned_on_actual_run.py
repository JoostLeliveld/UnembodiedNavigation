#!/usr/bin/env python3
"""Replay characterization-trained box corrections on one exact recorded drive.

The drive is real recorded Gazebo data.  The learned corrections are offline replays on
the raw camera readings from that drive: they did not steer the robot, alter fusion, or
change the recorded trajectory.  Every physical reading is counted once and scored at its
capture timestamp through fusion_on_fixed_routes/aligned.py.

Training uses only the checkerboard TRAIN tiles from the frozen characterization capture.
Ground truth from the drive is used only after prediction to score the replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
for rel in (
    "experiments/deck_figures",
    "experiments/fusion_on_fixed_routes",
    "experiments/camera_observation_characterization",
    "experiments/measurement_commissioning",
    "src/experiments",
    "src/unav_common",
):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import aligned as A  # noqa: E402
import style as D  # noqa: E402
import plot_real_run_bias as R  # noqa: E402
from experiments.core.world_profiles import compute_look_at_from_pose  # noqa: E402
from observation import h as hull_h, jacobian as hull_jacobian  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402
from fit_bias_updates import (  # noqa: E402
    CAMERAS,
    FEATURE_NAMES,
    apply_correction,
    camera_geometry,
    features,
    ray_frame,
    target,
    tile_split,
)

DEFAULT_CAPTURE = (
    REPO / "logs/perception_datasets/warehouse_v2_bbox_characterization_20260831"
)
DEFAULT_RUN = (
    REPO / "logs/studies/fusion_on_fixed_routes/diagnostic_schema5_20260831/"
    "fusion_network_traverse/O1/seed0/experiment_20260831_110742"
)
DEFAULT_OUT = REPO / "logs/studies/camera_observation_characterization_20260831"
FOLDER = "09_learned_fixes_replayed"
METHODS = ("raw", "linear", "neural", "hull")
METHOD_LABEL = {
    "raw": "Raw box → floor",
    "linear": "Learned linear (current)",
    "neural": "Learned neural (current)",
    "hull": "Analytic hull (oracle bound)",
}
METHOD_COLOUR = {
    "raw": D.BAD,
    "linear": D.ROBOT,
    "neural": D.OLD,
    "hull": D.GOOD,
}
# The hull rung linearises the analytic silhouette around the TRUE pose and yaw, so it is an
# evaluation-only upper bound on what this box could give if the robot's pose were already
# known. PLAN.md permits it as an oracle bound; it is never an operational correction, and
# every label that carries it says so.
ORACLE_METHODS = ("hull",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values_m: np.ndarray) -> dict[str, float | int]:
    """Median/p90/RMSE over the readings a rung actually produced, NaN-aware."""
    values_m = np.asarray(values_m, dtype=float)
    good = values_m[np.isfinite(values_m)]
    if not good.size:
        return {"n": 0, "offered": int(values_m.size), "median_cm": math.nan,
                "p90_cm": math.nan, "rmse_cm": math.nan}
    return {
        "n": int(good.size),
        "offered": int(values_m.size),
        "median_cm": float(np.median(good) * 100.0),
        "p90_cm": float(np.quantile(good, 0.90) * 100.0),
        "rmse_cm": float(np.sqrt(np.mean(good ** 2)) * 100.0),
    }


def signed_summary(values_m: np.ndarray) -> dict[str, float]:
    values_m = np.asarray(values_m, dtype=float)
    values_m = values_m[np.isfinite(values_m)]
    if not values_m.size:
        return {"mean_cm": math.nan, "std_cm": math.nan, "q05_cm": math.nan, "q95_cm": math.nan}
    q05, q95 = np.quantile(values_m, [0.05, 0.95])
    return {
        "mean_cm": float(np.mean(values_m) * 100.0),
        "std_cm": float(np.std(values_m) * 100.0),
        "q05_cm": float(q05 * 100.0),
        "q95_cm": float(q95 * 100.0),
    }


def pooled_features(row: dict[str, str], geometry: dict[str, dict]) -> np.ndarray:
    camera_id = row["camera_id"]
    onehot = np.zeros(len(CAMERAS), dtype=float)
    onehot[CAMERAS.index(camera_id)] = 1.0
    return np.concatenate([features(row, geometry[camera_id]), onehot])


def look_at_from_pose(pose: list[float]) -> np.ndarray:
    x, y, z, _roll, pitch, yaw = pose
    forward = np.asarray([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    ])
    scale = -z / forward[2]
    return np.asarray([x, y, z]) + scale * forward


class Projector:
    """Small forward projector matching ObliqueCameraModel.world_to_pixel."""

    def __init__(self, pose: list[float], width: int, height: int, fov_h_rad: float):
        self.position = np.asarray(pose[:3], dtype=float)
        forward = look_at_from_pose(pose) - self.position
        z_axis = forward / np.linalg.norm(forward)
        x_axis = np.cross(z_axis, np.asarray([0.0, 0.0, 1.0]))
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        self.rotation = np.asarray([x_axis, y_axis, z_axis])
        self.focal = (float(width) / 2.0) / math.tan(float(fov_h_rad) / 2.0)
        self.width = int(width)
        self.height = int(height)

    def world_to_pixel(self, point_xy: np.ndarray) -> tuple[float, float]:
        point = np.asarray([point_xy[0], point_xy[1], 0.0], dtype=float)
        camera = self.rotation @ (point - self.position)
        if camera[2] <= 1e-12:
            raise RuntimeError("Raw floor point projects behind its stated camera")
        return (
            float(self.focal * camera[0] / camera[2] + self.width / 2.0),
            float(self.focal * camera[1] / camera[2] + self.height / 2.0),
        )


def camera_models(capture_manifest: dict) -> dict[str, ObliqueCameraModel]:
    """Full oblique camera models, needed by the analytic-hull forward projection."""
    profiles_path = Path(capture_manifest["world_profiles_path"])
    if sha256(profiles_path) != capture_manifest["world_profiles_sha256"]:
        raise RuntimeError("World-profile file changed after the frozen capture")
    profiles = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
    intrinsics = profiles["camera_intrinsics"]
    models = {}
    for item in capture_manifest["cameras"]:
        pose = [float(value) for value in item["pose_xyz_rpy"]]
        models[item["camera_id"]] = ObliqueCameraModel(
            cam_pos=pose[:3],
            look_at=compute_look_at_from_pose(pose[:3], *pose[3:]),
            img_width=int(item["image_width"]),
            img_height=int(item["image_height"]),
            fov_h_rad=float(intrinsics["fov_h_rad"]),
        )
    return models


def hull_estimates(readings: list[dict], rows: list[dict[str, str]],
                   models: dict[str, ObliqueCameraModel]) -> np.ndarray:
    """Analytic-hull reading for each drive observation, around the true pose.

    Mirrors derive_interpretations.py exactly: x_hull = x_ref + J^-1 (z - h(x_ref)), with the
    reference taken as ground truth at that capture stamp. That reference is the reason this
    rung is an oracle, and the reason it can never be a deployment input.
    """
    out = np.full((len(readings), 2), np.nan, dtype=float)
    for index, (reading, row) in enumerate(zip(readings, rows)):
        cam = models[row["camera_id"]]
        truth = np.asarray(reading["truth"], dtype=float)
        yaw = float(reading["truth_yaw"])
        if not math.isfinite(yaw):
            continue
        measured = np.asarray([float(row["u_bbox_bottom"]), float(row["v_bbox_bottom"])])
        try:
            predicted = hull_h(cam, truth[0], truth[1], yaw)
            if predicted is None:
                continue
            jac = hull_jacobian(cam, truth[0], truth[1], yaw)
            out[index] = truth + np.linalg.solve(jac, measured - np.asarray(predicted))
        except (ValueError, TypeError, np.linalg.LinAlgError):
            continue
    return out


def projectors(capture_manifest: dict) -> dict[str, Projector]:
    profiles_path = Path(capture_manifest["world_profiles_path"])
    if sha256(profiles_path) != capture_manifest["world_profiles_sha256"]:
        raise RuntimeError("World-profile file changed after the frozen capture")
    profiles = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
    fov = float(profiles["camera_intrinsics"]["fov_h_rad"])
    result = {}
    for item in capture_manifest["cameras"]:
        result[item["camera_id"]] = Projector(
            [float(value) for value in item["pose_xyz_rpy"]],
            int(item["image_width"]),
            int(item["image_height"]),
            fov,
        )
    return result


def fit_models(train: list[dict[str, str]], geometry: dict[str, dict], seed: int) -> dict:
    linear = {}
    for camera_id in CAMERAS:
        subset = [row for row in train if row["camera_id"] == camera_id]
        x = np.stack([features(row, geometry[camera_id]) for row in subset])
        y = np.stack([target(row, geometry[camera_id]) for row in subset])
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x, y)
        linear[camera_id] = model

    x_train = np.stack([pooled_features(row, geometry) for row in train])
    y_train = np.stack([target(row, geometry[row["camera_id"]]) for row in train])
    neural = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
            alpha=1e-4, learning_rate_init=1e-3, max_iter=4000,
            early_stopping=True, n_iter_no_change=40, validation_fraction=0.15,
            random_state=seed,
        ),
    )
    neural.fit(x_train, y_train)

    return {"linear": linear, "neural": neural}


def corrections_for_rows(
    rows: list[dict[str, str]], models: dict, geometry: dict[str, dict]
) -> dict[str, np.ndarray]:
    output = {"raw": np.zeros((len(rows), 2), dtype=float)}
    linear = np.zeros((len(rows), 2), dtype=float)
    for camera_id in CAMERAS:
        indices = [i for i, row in enumerate(rows) if row["camera_id"] == camera_id]
        if indices:
            design = np.stack([features(rows[i], geometry[camera_id]) for i in indices])
            linear[indices] = models["linear"][camera_id].predict(design)
    output["linear"] = linear
    pooled = np.stack([pooled_features(row, geometry) for row in rows])
    output["neural"] = models["neural"].predict(pooled)
    return output


def replay_rows(
    readings: list[dict], project: dict[str, Projector]
) -> list[dict[str, str]]:
    rows = []
    for reading in readings:
        camera_id = f"camera_{reading['camera']}"
        raw = np.asarray([reading["obs_x"], reading["obs_y"]], dtype=float)
        u, v = project[camera_id].world_to_pixel(raw)
        width = float(reading["bbox_w_px"])
        height = float(reading["bbox_h_px"])
        if not all(math.isfinite(value) and value > 0.0 for value in (width, height)):
            raise RuntimeError("Actual run lacks a finite native-resolution box size")
        rows.append({
            "camera_id": camera_id,
            "raw_x": str(raw[0]),
            "raw_y": str(raw[1]),
            "x0": str(u - width / 2.0),
            "x1": str(u + width / 2.0),
            "y0": str(v - height),
            "y1": str(v),
            "u_bbox_bottom": str(u),
            "v_bbox_bottom": str(v),
            "confidence": str(reading["conf"]),
        })
    return rows


def evaluate(
    rows: list[dict[str, str]], truths: np.ndarray, corrections: dict[str, np.ndarray],
    geometry: dict[str, dict], direct: dict[str, np.ndarray] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Score every rung on the same rows.

    `direct` carries rungs that produce a world estimate outright rather than a correction to
    the raw point — the analytic hull is one. Those may fail to solve, so every downstream
    statistic here is NaN-aware and each rung reports how many readings it actually covered.
    """
    direct = direct or {}
    result = {}
    for method in METHODS:
        if method in direct:
            estimates = np.asarray(direct[method], dtype=float)
        else:
            estimates = np.stack([
                apply_correction(row, geometry[row["camera_id"]], correction)
                for row, correction in zip(rows, corrections[method])
            ])
        residual = estimates - truths
        magnitude = np.linalg.norm(residual, axis=1)
        along, across = [], []
        for row, truth_xy, error in zip(rows, truths, residual):
            unit, left = ray_frame(truth_xy, geometry[row["camera_id"]]["xy"])
            along.append(float(error @ unit))
            across.append(float(error @ left))
        result[method] = {
            "estimate": estimates,
            "error": residual,
            "magnitude": magnitude,
            "along": np.asarray(along),
            "across": np.asarray(across),
            "valid": np.isfinite(magnitude),
        }
    return result


def learned_time_columns(
    run: Path,
    run_manifest: dict,
    run_summary: dict,
    readings: list[dict],
    elapsed: np.ndarray,
    evaluated: dict,
) -> list[dict]:
    """Adapt the two offline replays to the full actual-drive plotting contract."""
    first_cmd = float(run_summary["first_cmd_stamp"])
    stop_stamp = float(run_summary["stop_stamp"])
    table = A.rows(run)
    truth = A.truth_series(run, table)
    in_drive = (truth.t >= first_cmd) & (truth.t <= stop_stamp)
    route = np.asarray(json.loads(run_manifest["preselected_route_json"]), dtype=float)
    run_id = str(run_manifest.get("run_id", run.name))
    collision_s = (
        float(run_summary["first_crash_stamp"]) - first_cmd
        if run_summary.get("collision_any")
        and run_summary.get("first_crash_stamp") is not None
        else None
    )
    columns = []
    for method, observation_model in (
        ("linear", "learned_linear"), ("neural", "learned_neural")
    ):
        replayed = []
        for index, source in enumerate(readings):
            item = dict(source)
            item["error"] = evaluated[method]["error"][index]
            item["error_cm"] = float(evaluated[method]["magnitude"][index] * 100.0)
            item["magnitude_m"] = float(evaluated[method]["magnitude"][index])
            item["along_m"] = float(evaluated[method]["along"][index])
            item["across_m"] = float(evaluated[method]["across"][index])
            replayed.append(item)
        errors_cm = np.asarray([item["error_cm"] for item in replayed], dtype=float)
        columns.append({
            "run": run,
            "manifest": run_manifest,
            "summary": run_summary,
            "readings": replayed,
            "times": elapsed,
            "errors_cm": errors_cm,
            "duration_s": float(run_summary["elapsed_after_first_cmd_s"]),
            "collision_s": collision_s,
            "spans": R.blind_spans(elapsed),
            "route": route,
            "truth_xy": (truth.x[in_drive], truth.y[in_drive]),
            "observation_model": observation_model,
            "run_id": run_id,
            "panel_id": observation_model,
            "arm": "offline_replay",
            "context_line": f"offline replay on raw-box arm O1\nsource run {run_id}",
            "completion": str(run_summary.get("completion_reason", "unknown")),
            "per_camera": {
                camera: sum(item["camera"] == camera for item in replayed)
                for camera in "ABCDE"
            },
        })
    return columns


def binned_mean(x: np.ndarray, y: np.ndarray, width_s: float = 4.0):
    edges = np.arange(0.0, float(np.max(x)) + width_s, width_s)
    centres, values = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = y[(x >= low) & (x < high)]
        if selected.size:
            centres.append(0.5 * (low + high))
            values.append(float(np.mean(selected)))
    return np.asarray(centres), np.asarray(values)


def plot_time_group(
    output: Path,
    readings: list[dict],
    elapsed: np.ndarray,
    evaluated: dict,
    methods: tuple[str, ...],
    *,
    run_id: str,
    collision_s: float | None,
    duration_s: float,
    title: str,
    filename: str,
    per_panel_scale: bool = False,
) -> str:
    def scale_for(values: np.ndarray) -> float:
        return max(0.08, float(np.quantile(np.abs(values), 0.997)) * 1.08)

    all_signed = np.concatenate([evaluated[method]["along"] for method in methods])
    shared_limit = scale_for(all_signed)
    # The raw bias is roughly four times the learned residual. On one shared axis the
    # learned panels collapse to a flat line, so each panel keeps its own scale and the
    # common-scale summary column on the right carries the magnitude comparison.
    limits = {
        method: (scale_for(evaluated[method]["along"]) if per_panel_scale else shared_limit)
        for method in methods
    }
    height = 7.6 if len(methods) == 1 else 3.9 * len(methods)
    fig = plt.figure(figsize=(18.0, height))
    grid = fig.add_gridspec(
        len(methods), 2, width_ratios=(3.35, 1.30), left=0.070, right=0.975,
        bottom=0.170 if len(methods) == 1 else 0.125,
        top=0.790 if len(methods) == 1 else 0.825,
        hspace=0.24, wspace=0.20,
    )
    axes = [fig.add_subplot(grid[row, 0]) for row in range(len(methods))]
    summary_ax = fig.add_subplot(grid[:, 1])
    for ax, method in zip(axes, methods):
        signed = evaluated[method]["along"]
        y_limit = limits[method]
        for camera in "ABCDE":
            indices = np.asarray([
                i for i, reading in enumerate(readings) if reading["camera"] == camera
            ], dtype=int)
            values = signed[indices]
            clipped = np.clip(values, -y_limit, y_limit)
            markers = np.where(values > y_limit, "^", np.where(values < -y_limit, "v", "o"))
            for marker in ("o", "^", "v"):
                selected = markers == marker
                if np.any(selected):
                    ax.scatter(
                        elapsed[indices][selected], clipped[selected], s=22,
                        marker=marker, color=D.CAM_COLOUR[camera], alpha=0.68,
                        edgecolors="white", linewidths=0.30, zorder=3,
                    )
        bx, by = binned_mean(elapsed, signed)
        mean_signed = float(np.mean(signed))
        q05, q95 = np.quantile(signed, [0.05, 0.95])
        ax.plot(bx, np.clip(by, -y_limit, y_limit), color=D.INK, lw=2.5, zorder=5)
        ax.axhline(0.0, color=D.MUTED, lw=1.15, linestyle=":", zorder=2)
        ax.axhline(mean_signed, color=METHOD_COLOUR[method], lw=1.8,
                   linestyle="--", zorder=4)
        if collision_s is not None:
            ax.axvline(collision_s, color="#c63131", lw=1.8, linestyle="--")
        ax.set_xlim(0.0, duration_s)
        ax.set_ylim(-y_limit, y_limit)
        ax.grid(color="#e4e2dc", lw=0.8)
        ax.set_axisbelow(True)
        ax.text(
            0.012, 0.88,
            f"{METHOD_LABEL[method]}   mean {mean_signed * 100:+.1f} cm · "
            f"central 90% [{q05 * 100:+.1f}, {q95 * 100:+.1f}] cm",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold",
            color=METHOD_COLOUR[method], va="top",
        )
        ax.set_ylabel("Signed along-ray error (m)\n− toward camera   + away")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Actual elapsed time after first motion command (s)")

    ypos = np.arange(len(methods))[::-1]
    means_cm, q05_cm, q95_cm = [], [], []
    for method in methods:
        values_cm = evaluated[method]["along"] * 100.0
        means_cm.append(float(np.mean(values_cm)))
        low, high = np.quantile(values_cm, [0.05, 0.95])
        q05_cm.append(float(low))
        q95_cm.append(float(high))
    for y, method in zip(ypos, methods):
        index = methods.index(method)
        summary_ax.errorbar(
            means_cm[index], y,
            xerr=[[means_cm[index] - q05_cm[index]],
                  [q95_cm[index] - means_cm[index]]],
            fmt="o", ms=9, capsize=6, elinewidth=2.4,
            color=METHOD_COLOUR[method], ecolor=METHOD_COLOUR[method], zorder=4,
        )
    signed_limit_cm = max(abs(min(q05_cm)), abs(max(q95_cm))) * 1.18
    summary_ax.axvline(0.0, color=D.INK, lw=1.2, linestyle=":")
    summary_ax.set_xlim(-signed_limit_cm, signed_limit_cm)
    summary_ax.set_yticks(ypos, [METHOD_LABEL[method] for method in methods])
    summary_ax.set_xlabel("Signed along-camera error (cm)\n− toward camera   + away")
    summary_ax.set_title("Mean and central 90% interval\non the same readings",
                         fontsize=15, fontweight="bold", pad=13)
    summary_ax.grid(axis="x", color="#e4e2dc", lw=0.8)
    note = "Dot = signed mean\nWhisker = 5th–95th percentile"
    summary_ax.text(
        0.04, 0.10 if len(methods) > 1 else 0.20, note,
        transform=summary_ax.transAxes, fontsize=11.2, va="bottom",
        bbox=dict(boxstyle="round,pad=0.6", fc="#f2f0ea", ec="#d0cec7"),
    )

    handles = [
        Line2D([0], [0], marker="o", lw=0, ms=8, color=D.CAM_COLOUR[camera],
               label=f"Camera {camera}") for camera in "ABCDE"
    ] + [
        Line2D([0], [0], color=D.INK, lw=2.5, label="4 s mean"),
        Line2D([0], [0], color=D.MUTED, lw=1.2, linestyle=":", label="zero error"),
        Line2D([0], [0], color=D.MUTED, lw=1.8, linestyle="--",
               label="whole-run mean"),
    ]
    if collision_s is not None:
        handles.append(Line2D([0], [0], color="#c63131", lw=2, linestyle="--",
                              label="collision / run stop"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True,
               fontsize=10.8, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle(
        f"{title}\n"
        f"Raw observations and timestamps from {run_id}; recorded trajectory is unchanged",
        fontsize=20, fontweight="bold", y=0.975,
    )
    subtitle = (
        "One deduplicated raw reading per (camera, capture time)\n"
        "Drive truth used only for scoring · camera-reading layer, not fused or belief error"
        if methods == ("raw",)
        else
        "Learned models trained only on frozen characterization TRAIN tiles · "
        "each time panel keeps its own y scale\n"
        "The common-scale summary on the right is where the magnitudes compare · "
        "drive truth used only for scoring"
    )
    fig.text(
        0.385, 0.850 if len(methods) == 1 else 0.880,
        subtitle,
        ha="center", fontsize=11.5, color=D.MUTED,
    )
    fig.savefig(output / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_component_distributions(
    output: Path,
    readings: list[dict],
    evaluated: dict,
    run_id: str,
    *,
    component: str,
    filename: str,
) -> str:
    if component not in {"along", "across"}:
        raise ValueError("component must be along or across")
    combined = np.concatenate([evaluated[method][component] for method in METHODS])
    limit = max(0.15, float(np.quantile(np.abs(combined), 0.997)) * 1.05)
    bins = np.linspace(-limit, limit, 35)
    fig, axes = plt.subplots(3, 2, figsize=(18.0, 12.0), sharex=True)
    flat_axes = axes.ravel()
    for row_index, camera in enumerate("ABCDE"):
        ax = flat_axes[row_index]
        indices = np.asarray([
            i for i, reading in enumerate(readings) if reading["camera"] == camera
        ], dtype=int)
        for method in METHODS:
            values = evaluated[method][component][indices]
            ax.hist(values, bins=bins, histtype="step", lw=2.2,
                    color=METHOD_COLOUR[method], label=METHOD_LABEL[method])
            ax.axvline(float(np.mean(values)), color=METHOD_COLOUR[method],
                       lw=1.5, linestyle="--", alpha=0.90)
        ax.axvline(0.0, color=D.INK, lw=1.1, linestyle="--")
        ax.grid(color="#e4e2dc", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"Camera {camera} · {len(indices)} readings", fontsize=15,
                     fontweight="bold")
        ax.set_ylabel("Count")

    info_ax = flat_axes[-1]
    info_ax.axis("off")
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLOUR[method], lw=2.0,
               label=METHOD_LABEL[method]) for method in METHODS
    ] + [
        Line2D([0], [0], color=D.MUTED, lw=1.4, linestyle="--",
               label="coloured dashed line = mean"),
        Line2D([0], [0], color=D.INK, lw=1.1, linestyle="--", label="zero error"),
    ]
    info_ax.legend(handles=legend_handles, loc="upper left", frameon=True,
                   fontsize=12.0)
    sign_text = (
        "Negative = reading lands toward camera\nPositive = reading lands away from camera"
        if component == "along"
        else "Negative = reading lands right of camera ray\nPositive = reading lands left of camera ray"
    )
    overall = "\n".join(
        f"{METHOD_LABEL[method]}: {np.mean(evaluated[method][component]) * 100:+.1f} cm"
        for method in METHODS
    )
    info_ax.text(
        0.02, 0.48,
        f"Sign convention\n{sign_text}\n\nOverall signed means\n{overall}",
        fontsize=13.0, va="top",
        bbox=dict(boxstyle="round,pad=0.7", fc="#f2f0ea", ec="#d0cec7"),
    )
    component_title = "ALONG-CAMERA (RADIAL) ERROR" if component == "along" else (
        "ACROSS-CAMERA (LATERAL) ERROR"
    )
    fig.suptitle(
        f"{component_title} ON THE REAL-DRIVE OBSERVATIONS\n"
        f"Actual timestamps from {run_id}; 385 deduplicated camera readings",
        fontsize=20, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.900,
        "Signed distributions: coloured dashed lines are per-camera means; black dashed "
        "line is zero. Trajectory and fusion were not replayed.",
        ha="center", fontsize=12.5, color=D.MUTED,
    )
    fig.supxlabel("Signed camera-reading error (m)", fontsize=13.0, y=0.018)
    fig.tight_layout(rect=(0.035, 0.045, 0.985, 0.885), h_pad=1.15, w_pad=0.85)
    fig.savefig(output / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return filename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    run = args.run.expanduser().resolve()
    output = args.out.expanduser().resolve() / FOLDER
    output.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output / "20_signed_bias_raw_vs_learned.png",
        output / "21_along_ray_distributions.png",
        output / "22_across_ray_distributions.png",
        output / "23_learned_linear_neural_over_time.png",
        output / "learned_replay_manifest.json",
    ]
    if not args.overwrite and any(path.exists() for path in expected_outputs):
        raise RuntimeError("Replay output exists; pass --overwrite")

    capture_paths = {
        name: capture / name for name in (
            "capture_manifest.json", "bbox_detector_manifest.json",
            "observation_interpretations.csv", "observation_interpretations_manifest.json",
            "bias_update_interpretations.csv", "bias_update_interpretations_manifest.json",
        )
    }
    run_paths = {
        name: run / name for name in (
            "run_manifest.json", "run_summary.json", "experiment.csv",
            "fusion_observations.csv", "correction_assimilations.csv",
        )
    }
    for path in (*capture_paths.values(), *run_paths.values()):
        if not path.is_file():
            raise RuntimeError(f"Missing required evidence: {path}")

    capture_manifest = json.loads(capture_paths["capture_manifest.json"].read_text())
    detector_manifest = json.loads(capture_paths["bbox_detector_manifest.json"].read_text())
    run_manifest = json.loads(run_paths["run_manifest.json"].read_text())
    run_summary = json.loads(run_paths["run_summary.json"].read_text())
    bias_manifest = json.loads(
        capture_paths["bias_update_interpretations_manifest.json"].read_text()
    )
    if bias_manifest.get("status") != "complete":
        raise RuntimeError("Frozen bias-update interpretations are not complete")
    if A.schema_version(run) < 5:
        raise RuntimeError("Replay needs schema 5 box width/height and capture timestamps")
    if run_manifest.get("manager_observation_model") != "raw_box":
        raise RuntimeError("Replay source must be the exact raw-box O1 arm")
    if run_manifest.get("yolo_model_sha256") != detector_manifest.get("weights_sha256"):
        raise RuntimeError("Characterization and actual run used different YOLO weights")
    native_shapes = {
        (int(item["image_width"]), int(item["image_height"]))
        for item in capture_manifest["cameras"]
    }
    if native_shapes != {(1280, 720)}:
        raise RuntimeError(f"Unexpected characterization image dimensions: {native_shapes}")

    geometry = camera_geometry(capture_manifest)
    project = projectors(capture_manifest)
    optics = camera_models(capture_manifest)
    source_rows = list(csv.DictReader(
        capture_paths["observation_interpretations.csv"].open(encoding="utf-8")
    ))
    split = tile_split(source_rows, 2.0)
    usable = [row for row in source_rows if row["raw_valid"] == "1"]
    train = [row for row in usable if split[row["position_id"]] == "train"]
    test = [row for row in usable if split[row["position_id"]] == "test"]
    models = fit_models(train, geometry, args.seed)

    # Recreate the two previously plotted model outputs before applying them to the run.
    frozen_rows = list(csv.DictReader(
        capture_paths["bias_update_interpretations.csv"].open(encoding="utf-8")
    ))
    frozen_by_key = {
        (row["pose_id"], row["repetition_id"], row["camera_id"]): row
        for row in frozen_rows
    }
    test_corrections = corrections_for_rows(test, models, geometry)
    reproduction = {}
    for method, frozen_name in (("linear", "learned"), ("neural", "nn")):
        differences = []
        for row, correction in zip(test, test_corrections[method]):
            estimate = apply_correction(row, geometry[row["camera_id"]], correction)
            frozen = frozen_by_key[(row["pose_id"], row["repetition_id"], row["camera_id"])]
            expected = np.asarray([
                float(frozen[f"{frozen_name}_x"]), float(frozen[f"{frozen_name}_y"])
            ])
            differences.append(float(np.max(np.abs(estimate - expected))))
        reproduction[method] = float(max(differences))
        if reproduction[method] > 1e-9:
            raise RuntimeError(
                f"{method} refit does not reproduce frozen model outputs: "
                f"max |difference|={reproduction[method]}"
            )

    test_truth = np.stack([
        [float(row["robot_x"]), float(row["robot_y"])] for row in test
    ])
    grid_hull = np.asarray([
        [float(row["hull_x"]), float(row["hull_y"])] if row["hull_valid"] == "1"
        else [math.nan, math.nan]
        for row in test
    ], dtype=float)
    grid_evaluated = evaluate(test, test_truth, test_corrections, geometry,
                              direct={"hull": grid_hull})
    grid_scores = {
        method: summary(grid_evaluated[method]["magnitude"]) for method in METHODS
    }

    loaded = A.readings(run, admitted_only=False, dedupe=True, require_capture_time=True)
    first_cmd = float(run_summary["first_cmd_stamp"])
    stop_stamp = float(run_summary["stop_stamp"])
    readings = [
        item for item in loaded if first_cmd <= item["obs_stamp"] <= stop_stamp
    ]
    if not readings:
        raise RuntimeError("No deduplicated readings during actual motion")
    rows = replay_rows(readings, project)
    truths = np.stack([reading["truth"] for reading in readings])
    replay_corrections = corrections_for_rows(rows, models, geometry)
    drive_hull = hull_estimates(readings, rows, optics)
    evaluated = evaluate(rows, truths, replay_corrections, geometry,
                         direct={"hull": drive_hull})
    actual_scores = {
        method: summary(evaluated[method]["magnitude"]) for method in METHODS
    }
    elapsed = np.asarray([reading["obs_stamp"] - first_cmd for reading in readings])
    collision_s = (
        float(run_summary["first_crash_stamp"]) - first_cmd
        if run_summary.get("collision_any") and run_summary.get("first_crash_stamp") is not None
        else None
    )

    run_id = str(run_manifest.get("run_id", run.name))
    duration_s = float(run_summary["elapsed_after_first_cmd_s"])
    figure18 = plot_time_group(
        output, readings, elapsed, evaluated, METHODS, run_id=run_id,
        collision_s=collision_s, duration_s=duration_s,
        title="THE LEARNED UPDATES MOVE THE RADIAL BIAS FROM -23.6 cm TO NEAR ZERO",
        filename="20_signed_bias_raw_vs_learned.png",
        per_panel_scale=True,
    )
    figure19 = plot_component_distributions(
        output, readings, evaluated, run_id, component="along",
        filename="21_along_ray_distributions.png",
    )
    figure20 = plot_component_distributions(
        output, readings, evaluated, run_id, component="across",
        filename="22_across_ray_distributions.png",
    )
    time_columns = learned_time_columns(
        run, run_manifest, run_summary, readings, elapsed, evaluated
    )
    replay_time_panels = R.draw_sheet(
        time_columns,
        output,
        filename="23_learned_linear_neural_over_time.png",
        suptitle=(
            "OFFLINE REPLAY ON ONE ACTUAL RECORDED GAZEBO DRIVE — learned corrections "
            "over elapsed time\n"
            f"The same {len(readings)} deduplicated raw-box readings and the same recorded "
            "trajectory in both columns\n"
            "Each row shares one scale; every reading is scored against truth at its own "
            "capture timestamp"
        ),
        footnote=(
            "Camera-reading layer only — not fused, belief or planner error. Both learned "
            "models are offline replays on the same O1 raw-box drive; they did not steer the "
            "robot, alter fusion, or change the trajectory.\n"
            "The historical runtime gate was already enabled in the source drive, so detector "
            "misses and earlier rejects cannot be recovered. One diagnostic drive, not a "
            "replicated closed-loop comparison."
        ),
    )
    figure21 = "23_learned_linear_neural_over_time.png"

    feature_train = np.stack([pooled_features(row, geometry) for row in train])
    feature_run = np.stack([pooled_features(row, geometry) for row in rows])
    feature_support = {}
    for index, name in enumerate(FEATURE_NAMES):
        feature_support[name] = {
            "train_q01_q50_q99": [
                float(value) for value in np.quantile(feature_train[:, index], [.01, .5, .99])
            ],
            "actual_run_q01_q50_q99": [
                float(value) for value in np.quantile(feature_run[:, index], [.01, .5, .99])
            ],
            "actual_run_fraction_outside_train_min_max": float(np.mean(
                (feature_run[:, index] < np.min(feature_train[:, index]))
                | (feature_run[:, index] > np.max(feature_train[:, index]))
            )),
        }

    manifest = {
        "status": "complete",
        "schema": "actual_drive_offline_learned_replay.v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "exploratory diagnostic; not a frozen replicated paper result",
        "interpretation_boundary": (
            "Actual raw camera observations and timestamps; learned corrections applied "
            "offline. Recorded trajectory, fusion decisions, belief, and planner are unchanged."
        ),
        "training": {
            "capture": str(capture),
            "split": "2 m checkerboard TRAIN tiles only",
            "n_train": len(train),
            "n_heldout_test": len(test),
            "truth_role": "regression target on characterization TRAIN tiles only",
            "feature_names": list(FEATURE_NAMES),
            "models": {
                "linear": "exact current per-camera StandardScaler + Ridge(alpha=1)",
                "neural": (
                    "exact current pooled StandardScaler + MLP(64,64), camera one-hot, "
                    "seed 0"
                ),
            },
            "frozen_output_reproduction_max_abs_m": reproduction,
            "heldout_grid_scores": grid_scores,
        },
        "actual_drive": {
            "run_dir": str(run),
            "run_id": run_id,
            "task": run_manifest.get("task"),
            "seed": run_manifest.get("seed"),
            "source_arm": run_manifest.get("method"),
            "source_observation_model": run_manifest.get("manager_observation_model"),
            "completion_reason": run_summary.get("completion_reason"),
            "duration_after_first_command_s": float(
                run_summary["elapsed_after_first_cmd_s"]
            ),
            "path_length_m": float(run_summary["path_length_m"]),
            "n_deduplicated_readings_during_motion": len(readings),
            "dedupe_key": "(camera, obs_stamp)",
            "truth_role": "offline score at obs_stamp only; never a replay feature",
            "scores": actual_scores,
            "signed_components": {
                method: {
                    "along_camera_ray": signed_summary(evaluated[method]["along"]),
                    "across_camera_ray": signed_summary(evaluated[method]["across"]),
                }
                for method in METHODS
            },
            "runtime_gate_boundary": (
                "fusion_observations.csv starts after the runtime admission gate; detector "
                "misses and earlier gate rejects are unavailable"
            ),
        },
        "compatibility": {
            "yolo_weights_sha256": run_manifest.get("yolo_model_sha256"),
            "native_image_dimensions": [1280, 720],
            "bbox_bottom_reconstruction": (
                "schema-5 run logs raw box-to-floor xy plus native bbox width/height; "
                "bbox bottom pixel is reconstructed by the frozen camera homography"
            ),
            "feature_support": feature_support,
        },
        "hashes": {
            "capture_manifest": sha256(capture_paths["capture_manifest.json"]),
            "bbox_detector_manifest": sha256(capture_paths["bbox_detector_manifest.json"]),
            "observation_interpretations": sha256(
                capture_paths["observation_interpretations.csv"]
            ),
            "bias_update_interpretations": sha256(
                capture_paths["bias_update_interpretations.csv"]
            ),
            "run_manifest": sha256(run_paths["run_manifest.json"]),
            "run_summary": sha256(run_paths["run_summary.json"]),
            "fusion_observations": sha256(run_paths["fusion_observations.csv"]),
            "experiment": sha256(run_paths["experiment.csv"]),
        },
        "figures": [figure18, figure19, figure20, figure21],
        "learned_linear_neural_time_panels": replay_time_panels,
        "next_required_evidence": (
            "Choose and freeze a model without using this diagnostic run for tuning, deploy "
            "it, then collect independent repeated closed-loop drives."
        ),
    }
    (output / "learned_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "actual_drive_scores": actual_scores,
        "heldout_grid_scores": grid_scores,
        "reproduction_max_abs_m": reproduction,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
