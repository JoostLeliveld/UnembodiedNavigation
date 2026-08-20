#!/usr/bin/env python3
"""Render the complete supervisor method-comparison figure package.

Every panel is deterministic and uses repository-native sources:

* the real ``warehouse_full_4cam`` SDF and Gazebo screenshot;
* the calibrated four-camera day-zero prior;
* the recorded spawn-grid detector opportunities and fitted GP artifacts;
* the declared D2 range-limited commissioning-depth degradation model; and
* the actual driveable regions and task start/goal definitions.

The route overlays are *offline explanatory routes*: one common reliability-weighted
grid search is applied to every field. They are not closed-loop navigation results and
the figures state that explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for source_dir in ("unav_common", "reliability"):
    sys.path.insert(0, str(REPO / "src" / source_dir))
sys.path.insert(0, str(REPO / "scripts" / "geometry_visibility"))

from reliability.prior import CameraCalibration  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402
from unav_common.occlusion_geometry import (  # noqa: E402
    AxisAlignedPrism,
    parse_occlusion_scene_from_world,
)
import geometry_visibility as gv  # noqa: E402


WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
TASKS = REPO / "src/experiments/config/tasks.yaml"
GAZEBO_ROOT = (
    REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse"
    / "four_camera_showcase/live_gazebo_views"
)
GAZEBO_OVERVIEW = GAZEBO_ROOT / "overview/frame_000040.000.png"
GAZEBO_CAMERA = {
    "camera_A": GAZEBO_ROOT / "camera_A/frame_000040.000.png",
    "camera_B": GAZEBO_ROOT / "camera_B/frame_000040.000.png",
    "camera_C": GAZEBO_ROOT / "camera_C/frame_000040.200.png",
    "camera_D": GAZEBO_ROOT / "camera_D/frame_000040.000.png",
}
DAYZERO = (
    REPO
    / "paper_artifacts/gp/warehouse_full_4cam_dayzero_v1"
    / "camera_a_planner_with_four_camera_maps.npz"
)
GP_ROOT = REPO / "logs/visibility_comparison/spawn_grid_20260727/gp"
GP_FUSED = REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz"
EVENT_ROOT = (
    REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/spawn_grid_20260727/events"
)

CAMERA_POSES = {
    "camera_A": (-6.0, -10.0, 6.10, 0.0, 0.92, 1.5708),
    "camera_B": (-6.0, 10.0, 6.10, 0.0, 0.92, -1.5708),
    "camera_C": (6.0, -10.0, 6.10, 0.0, 0.92, 1.5708),
    "camera_D": (6.0, 10.0, 6.10, 0.0, 0.92, -1.5708),
}
CAMERA_COLORS = {
    "camera_A": "#1674d1",
    "camera_B": "#159447",
    "camera_C": "#8d3cc7",
    "camera_D": "#ef8b2c",
}
METHODS = [
    ("constant_distance", "Constant / distance", "01_constant_distance", "#5875a4"),
    ("fov_range", "FOV / range", "02_fov_range", "#00a6a6"),
    ("depth_raycast", "Depth / raycast", "03_depth_raycast", "#d89000"),
    ("gp", "Operational GP", "04_gp", "#7b53b5"),
    ("hybrid", "Depth + GP residual", "05_hybrid", "#2a9d58"),
    ("cad_reference", "Complete CAD reference", "06_cad_reference", "#d94b4b"),
]
TARGET_Z = 0.35
DEPTH_RANGE_M = 10.0
FIELD_CMAP = "viridis"
EXTENT = (-11.7, 11.7, -9.0, 9.0)
ROUTE_LAMBDA = 5.0
FIG_DPI = 170


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def noisy_or(per_camera: dict[str, np.ndarray]) -> np.ndarray:
    stack = np.stack([np.clip(per_camera[c], 0.0, 1.0) for c in CAMERA_POSES])
    return np.clip(1.0 - np.prod(1.0 - stack, axis=0), 0.0, 1.0)


def profile_regions() -> list[AxisAlignedPrism]:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    profile = payload["worlds"][WORLD.name]
    regions = []
    for item in profile["known_2d_regions"]:
        if item.get("type") != "traversable":
            continue
        regions.append(
            AxisAlignedPrism(
                name=str(item["name"]),
                xmin=float(item["xmin"]),
                xmax=float(item["xmax"]),
                ymin=float(item["ymin"]),
                ymax=float(item["ymax"]),
                zmin=0.0,
                zmax=0.05,
            )
        )
    return regions


def mask_from_prisms(xs: np.ndarray, ys: np.ndarray, prisms: Iterable[AxisAlignedPrism]) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys)
    mask = np.zeros_like(gx, dtype=bool)
    for p in prisms:
        mask |= (gx >= p.xmin) & (gx <= p.xmax) & (gy >= p.ymin) & (gy <= p.ymax)
    return mask


def height_map(xs: np.ndarray, ys: np.ndarray, prisms: Iterable[AxisAlignedPrism]) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys)
    heights = np.zeros_like(gx, dtype=float)
    for p in prisms:
        inside = (gx >= p.xmin) & (gx <= p.xmax) & (gy >= p.ymin) & (gy <= p.ymax)
        heights[inside] = np.maximum(heights[inside], float(p.zmax))
    return heights


def cameras() -> tuple[dict[str, CameraCalibration], dict[str, ObliqueCameraModel]]:
    calibrations = {}
    models = {}
    for camera_id, pose in CAMERA_POSES.items():
        calibration = CameraCalibration.from_gazebo_pose(camera_id, pose)
        calibrations[camera_id] = calibration
        models[camera_id] = ObliqueCameraModel(
            cam_pos=calibration.cam_pos_m,
            look_at=calibration.look_at_m,
            img_width=calibration.img_width_px,
            img_height=calibration.img_height_px,
            fov_h_rad=calibration.fov_h_rad,
        )
    return calibrations, models


def load_events() -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for camera_id in CAMERA_POSES:
        path = EVENT_ROOT / f"{camera_id}_events.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        result[camera_id] = {
            "xy": np.asarray([[float(r["m_x"]), float(r["m_y"])] for r in rows]),
            "hit": np.asarray([float(r["det_hit"]) for r in rows]),
        }
    return result


def fit_distance_baseline(events: dict[str, dict[str, np.ndarray]]) -> tuple[float, float, float]:
    distances = []
    labels = []
    for camera_id, event in events.items():
        cam = np.asarray(CAMERA_POSES[camera_id][:3], dtype=float)
        xyz = np.column_stack([event["xy"], np.full(len(event["xy"]), TARGET_Z)])
        distances.append(np.linalg.norm(xyz - cam[None, :], axis=1))
        labels.append(event["hit"])
    d = np.concatenate(distances)
    y = np.concatenate(labels)
    prevalence = float(np.mean(y))

    def objective(params: np.ndarray) -> float:
        p = expit(params[0] + params[1] * d)
        return float(-np.sum(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)))

    fit = minimize(objective, np.asarray([logit(np.clip(prevalence, 1e-3, 1 - 1e-3)), -0.1]))
    if not fit.success:
        raise RuntimeError(f"distance baseline fit failed: {fit.message}")
    return prevalence, float(fit.x[0]), float(fit.x[1])


def distance_fields(
    xs: np.ndarray,
    ys: np.ndarray,
    prevalence: float,
    intercept: float,
    slope: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(xs, ys)
    per_camera = {}
    for camera_id, pose in CAMERA_POSES.items():
        d = np.sqrt((gx - pose[0]) ** 2 + (gy - pose[1]) ** 2 + (TARGET_Z - pose[2]) ** 2)
        per_camera[camera_id] = expit(intercept + slope * d)
    constant = np.full_like(gx, 1.0 - (1.0 - prevalence) ** len(CAMERA_POSES))
    return per_camera, noisy_or(per_camera), constant


def fov_fields(dayzero: np.lib.npyio.NpzFile) -> tuple[dict[str, np.ndarray], np.ndarray]:
    per_camera = {
        camera_id: np.asarray(dayzero[f"P_{camera_id}_map"], dtype=float)
        for camera_id in CAMERA_POSES
    }
    return per_camera, np.asarray(dayzero["P_union_4cam_map"], dtype=float)


def exact_occlusion_fields(
    xs: np.ndarray,
    ys: np.ndarray,
    models: dict[str, ObliqueCameraModel],
    fov_per_camera: dict[str, np.ndarray],
    prisms: Iterable[AxisAlignedPrism],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    prisms = tuple(prisms)
    ray_prisms = tuple(
        gv.Prism(p.name, p.xmin, p.xmax, p.ymin, p.ymax, p.zmin, p.zmax)
        for p in prisms
    )
    per_camera = {}
    for camera_id, model in models.items():
        clearance = gv.raycast_min_clearance(model, xs, ys, ray_prisms, TARGET_Z, n_samples=48)
        occ = expit(clearance / 0.10)
        per_camera[camera_id] = np.clip(fov_per_camera[camera_id] * occ, 0.0, 1.0)
    return per_camera, noisy_or(per_camera)


def prism_sensor_range(prism: AxisAlignedPrism, cam_xyz: np.ndarray) -> float:
    dx = max(prism.xmin - cam_xyz[0], 0.0, cam_xyz[0] - prism.xmax)
    dy = max(prism.ymin - cam_xyz[1], 0.0, cam_xyz[1] - prism.ymax)
    dz = max(prism.zmin - cam_xyz[2], 0.0, cam_xyz[2] - prism.zmax)
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def depth_d2_fields(
    xs: np.ndarray,
    ys: np.ndarray,
    models: dict[str, ObliqueCameraModel],
    fov_per_camera: dict[str, np.ndarray],
    prisms: Iterable[AxisAlignedPrism],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, tuple[AxisAlignedPrism, ...]], dict[str, np.ndarray]]:
    """Declared D2 mechanism: 10 m sensor range plus conservative unknown fallback.

    This is an analytic sensor-realism model over the real SDF, not a captured depth frame.
    It is used only for method explanation and is labelled that way in every depth panel.
    """
    prisms = tuple(prisms)
    gx, gy = np.meshgrid(xs, ys)
    per_camera = {}
    sensed_by_camera = {}
    unknown_by_camera = {}
    for camera_id, model in models.items():
        cam = np.asarray(model.cam_pos, dtype=float)
        sensed = tuple(p for p in prisms if prism_sensor_range(p, cam) <= DEPTH_RANGE_M)
        sensed_by_camera[camera_id] = sensed
        ray_sensed = tuple(
            gv.Prism(p.name, p.xmin, p.xmax, p.ymin, p.ymax, p.zmin, p.zmax)
            for p in sensed
        )
        clearance = gv.raycast_min_clearance(model, xs, ys, ray_sensed, TARGET_Z, n_samples=48)
        occ = expit(clearance / 0.10)
        target_range = np.sqrt((gx - cam[0]) ** 2 + (gy - cam[1]) ** 2 + (TARGET_Z - cam[2]) ** 2)
        unknown = target_range > DEPTH_RANGE_M
        unknown_by_camera[camera_id] = unknown
        score = fov_per_camera[camera_id] * occ
        # Explicit conservative fallback for depth-unknown target cells.
        score = np.where(unknown, 0.20 * fov_per_camera[camera_id], score)
        per_camera[camera_id] = np.clip(score, 0.0, 1.0)
    return per_camera, noisy_or(per_camera), sensed_by_camera, unknown_by_camera


def load_gp_fields() -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    per_camera = {}
    for camera_id in CAMERA_POSES:
        path = GP_ROOT / camera_id / "det_hit_expected_kernel_gp.npz"
        with np.load(path, allow_pickle=False) as data:
            per_camera[camera_id] = {k: np.asarray(data[k]) for k in data.files}
    with np.load(GP_FUSED, allow_pickle=False) as data:
        fused = np.asarray(data["P_conservative_plan_map"], dtype=float)
    return per_camera, fused


def hybrid_fields(
    depth_per_camera: dict[str, np.ndarray],
    gp_per_camera: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    per_camera = {}
    residuals = {}
    eps = 1e-4
    for camera_id in CAMERA_POSES:
        gp = gp_per_camera[camera_id]
        prior = np.clip(gp["prior_P_mean_map"], eps, 1.0 - eps)
        posterior = np.clip(gp["P_mean_map"], eps, 1.0 - eps)
        residual = np.clip(logit(posterior) - logit(prior), -4.0, 4.0)
        residuals[camera_id] = residual
        depth = np.clip(depth_per_camera[camera_id], eps, 1.0 - eps)
        mean = expit(logit(depth) + residual)
        uncertainty_penalty = np.maximum(gp["P_mean_map"] - gp["P_conservative_plan_map"], 0.0)
        per_camera[camera_id] = np.clip(mean - uncertainty_penalty, 0.0, 1.0)
    return per_camera, residuals, noisy_or(per_camera)


def tasks() -> dict[str, dict]:
    payload = yaml.safe_load(TASKS.read_text(encoding="utf-8"))
    rows = payload["tasks"][WORLD.name]
    return {row["name"]: row for row in rows}


def nearest_valid(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, xy: tuple[float, float]) -> tuple[int, int]:
    j0 = int(np.argmin(np.abs(xs - xy[0])))
    i0 = int(np.argmin(np.abs(ys - xy[1])))
    if mask[i0, j0]:
        return i0, j0
    ii, jj = np.where(mask)
    k = int(np.argmin((ii - i0) ** 2 + (jj - j0) ** 2))
    return int(ii[k]), int(jj[k])


def dijkstra_route(
    field: np.ndarray,
    driveable: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> np.ndarray:
    start = nearest_valid(driveable, xs, ys, start_xy)
    goal = nearest_valid(driveable, xs, ys, goal_xy)
    ny, nx = field.shape
    dist = np.full((ny, nx), np.inf)
    dist[start] = 0.0
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start)]
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    while queue:
        cost, current = heapq.heappop(queue)
        if current == goal:
            break
        if cost != dist[current]:
            continue
        i, j = current
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx) or not driveable[ni, nj]:
                continue
            step = math.hypot(di * dy, dj * dx)
            reliability = float(np.clip(field[ni, nj], 0.0, 1.0))
            new_cost = cost + step * (1.0 + ROUTE_LAMBDA * (1.0 - reliability) ** 2)
            if new_cost < dist[ni, nj]:
                dist[ni, nj] = new_cost
                previous[(ni, nj)] = current
                heapq.heappush(queue, (new_cost, (ni, nj)))
    if goal not in previous and goal != start:
        raise RuntimeError(f"no route from {start_xy} to {goal_xy}")
    cells = [goal]
    while cells[-1] != start:
        cells.append(previous[cells[-1]])
    cells.reverse()
    path = np.asarray([[xs[j], ys[i]] for i, j in cells], dtype=float)
    return simplify_path(path)


def simplify_path(path: np.ndarray) -> np.ndarray:
    if len(path) <= 2:
        return path
    keep = [0]
    prev = np.sign(path[1] - path[0])
    for idx in range(1, len(path) - 1):
        direction = np.sign(path[idx + 1] - path[idx])
        if not np.array_equal(direction, prev):
            keep.append(idx)
        prev = direction
    keep.append(len(path) - 1)
    return path[np.asarray(keep)]


def sample_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, path: np.ndarray) -> np.ndarray:
    jj = np.clip(np.searchsorted(xs, path[:, 0]), 0, len(xs) - 1)
    ii = np.clip(np.searchsorted(ys, path[:, 1]), 0, len(ys) - 1)
    return field[ii, jj]


def draw_geometry(ax, prisms: Iterable[AxisAlignedPrism], *, alpha: float = 0.92) -> None:
    for p in prisms:
        ax.add_patch(
            Rectangle(
                (p.xmin, p.ymin),
                p.xmax - p.xmin,
                p.ymax - p.ymin,
                facecolor="#7d858e" if p.zmax < 2.5 else "#cf6b28",
                edgecolor="#26313b",
                linewidth=0.55,
                alpha=alpha,
                zorder=4,
            )
        )


def draw_cameras(ax) -> None:
    for camera_id, pose in CAMERA_POSES.items():
        ax.scatter(pose[0], pose[1], s=58, c=CAMERA_COLORS[camera_id], edgecolors="black", zorder=7)
        ax.text(pose[0], pose[1] + (-0.42 if pose[1] > 0 else 0.42), camera_id[-1], ha="center", va="center", fontsize=8, weight="bold")


def draw_field(
    ax,
    field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    driveable: np.ndarray,
    prisms: Iterable[AxisAlignedPrism],
    *,
    title: str,
    colorbar: bool = False,
    unknown: np.ndarray | None = None,
):
    shown = np.where(driveable, field, np.nan)
    im = ax.imshow(shown, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=FIELD_CMAP, vmin=0, vmax=1, aspect="equal", zorder=1)
    if unknown is not None:
        masked = np.where(driveable & unknown, 1.0, np.nan)
        ax.contourf(xs, ys, masked, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=3)
    draw_geometry(ax, prisms)
    draw_cameras(ax)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=11, weight="bold")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025, label=r"planner reliability $p_{use}$")
    return im


def add_status(fig, text: str = "EXPLORATORY METHOD EXPLANATION — deterministic repository renderer") -> None:
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8.5, color="#4b5563")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gazebo_panel(ax, title: str, image_path: Path = GAZEBO_OVERVIEW) -> None:
    ax.imshow(plt.imread(image_path))
    ax.set_title(title, weight="bold")
    ax.axis("off")


def route_spec(task_map: dict[str, dict], key: str) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    if key == "R1":
        names = ["mc_blind_L"]
    elif key == "R2":
        names = ["mc_south_we", "mc_north_we"]
    elif key == "R3":
        names = ["mc_central_ns"]
    elif key == "R6":
        names = ["rob_easy"]
    else:
        raise KeyError(key)
    out = []
    for name in names:
        task = task_map[name]
        out.append((name, (float(task["start"]["x"]), float(task["start"]["y"])), (float(task["goal"]["x"]), float(task["goal"]["y"]))))
    return out


def draw_routes(ax, field, driveable, xs, ys, specs, color, label=True):
    paths = []
    for index, (name, start, goal) in enumerate(specs):
        path = dijkstra_route(field, driveable, xs, ys, start, goal)
        paths.append(path)
        ax.plot(path[:, 0], path[:, 1], color=color, lw=2.6, zorder=8, label="selected explanatory route" if label and index == 0 else None)
        ax.scatter(*start, s=55, c="#21a65a", edgecolors="black", zorder=9)
        ax.scatter(*goal, s=80, c="#e3342f", marker="*", edgecolors="black", zorder=9)
        values = sample_field(field, xs, ys, path)
        ax.text(0.02, 0.02 + 0.075 * index, f"{name}: mean p={np.mean(values):.2f}", transform=ax.transAxes, fontsize=7.5, bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"), zorder=10)
    return paths


def render_method_figures(ctx: dict) -> None:
    xs, ys = ctx["xs"], ctx["ys"]
    driveable, prisms = ctx["driveable"], ctx["prisms"]
    fields, per_camera = ctx["fields"], ctx["per_camera"]

    # 01 Constant/distance -------------------------------------------------
    out = HERE / "01_constant_distance/figures"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    gazebo_panel(axes[0], "Actual Gazebo world\n(no scene input used by baseline)")
    draw_field(axes[1], ctx["constant"], xs, ys, driveable, prisms, title=f"Constant: pooled event prevalence\nper-camera p={ctx['prevalence']:.2f}")
    draw_field(axes[2], fields["constant_distance"], xs, ys, driveable, prisms, title=f"Distance-only fitted from 8,808 events\nlogit(p)={ctx['distance_fit'][0]:.2f}{ctx['distance_fit'][1]:+.2f}d", colorbar=True)
    fig.suptitle("Begin state — constant and distance baselines", fontsize=16, weight="bold")
    add_status(fig); save(fig, out / "01_begin_state.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    draw_field(axes[0], ctx["constant"], xs, ys, driveable, prisms, title="Uniform fused constant field", colorbar=True)
    draw_field(axes[1], fields["constant_distance"], xs, ys, driveable, prisms, title="Fused distance field used below\n(no FOV, no racks, no experience)", colorbar=True)
    fig.suptitle("Planning fields — simple baselines", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "02_planning_field.png")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4))
    for ax, field, title in [(axes[0,0], ctx["constant"], "constant before route"), (axes[0,1], ctx["constant"], "constant after observations"), (axes[1,0], fields["constant_distance"], "distance before route"), (axes[1,1], fields["constant_distance"], "distance after observations")]:
        draw_field(ax, field, xs, ys, driveable, prisms, title=title)
    axes[0,1].text(.5,.5,"NO RELIABILITY-FIELD UPDATE",transform=axes[0,1].transAxes,ha="center",weight="bold",bbox=dict(facecolor="white",alpha=.9))
    axes[1,1].text(.5,.5,"belief updates; map stays fixed",transform=axes[1,1].transAxes,ha="center",weight="bold",bbox=dict(facecolor="white",alpha=.9))
    fig.suptitle("Update handling — static until explicit recommissioning", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "03_update_sequence.png")

    # 02 FOV/range ---------------------------------------------------------
    out = HERE / "02_fov_range/figures"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    gazebo_panel(
        axes[0],
        "Actual Gazebo camera-A view\nfrom the four-camera installation",
        GAZEBO_CAMERA["camera_A"],
    )
    draw_field(axes[1], per_camera["fov_range"]["camera_A"], xs, ys, driveable, prisms, title="Camera A analytic FOV/range prior")
    draw_field(axes[2], fields["fov_range"], xs, ys, driveable, prisms, title="Four-camera noisy-OR field\nno occluder input", colorbar=True)
    fig.suptitle("Begin state — calibration and projection geometry only", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "01_begin_state.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6), sharex=True, sharey=True)
    for ax, camera_id in zip(axes.flat, CAMERA_POSES):
        draw_field(ax, per_camera["fov_range"][camera_id], xs, ys, driveable, prisms, title=f"{camera_id}: calibrated footprint")
    fig.suptitle("Map used in planning — per-camera FOV/range priors", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "02_planning_field.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], fields["fov_range"], xs, ys, driveable, prisms, title="calibration valid")
    axes[1].axis("off"); axes[1].text(.5,.62,"runtime hits / misses\nupdate the robot belief",ha="center",va="center",fontsize=14,weight="bold"); axes[1].annotate("",xy=(.82,.5),xytext=(.18,.5),arrowprops=dict(arrowstyle="->",lw=3),xycoords="axes fraction") ; axes[1].text(.5,.35,"they do not refit this map",ha="center",fontsize=11,color="#9b2c2c")
    draw_field(axes[2], fields["fov_range"], xs, ys, driveable, prisms, title="same field\nrecompute only after recalibration")
    fig.suptitle("Update handling — static camera-geometry source", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "03_update_sequence.png")

    # 03 Depth/raycast -----------------------------------------------------
    out = HERE / "03_depth_raycast/figures"
    sensed_a = ctx["sensed_prisms"]["camera_A"]
    h_sensed = height_map(xs, ys, sensed_a)
    unknown_a = ctx["unknown"]["camera_A"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    gazebo_panel(
        axes[0],
        "Actual camera-A Gazebo view\nD2 model applied to the world SDF",
        GAZEBO_CAMERA["camera_A"],
    )
    im = axes[1].imshow(np.where(h_sensed > 0, h_sensed, np.nan), origin="lower", extent=(xs[0],xs[-1],ys[0],ys[-1]), cmap="cividis", vmin=0, vmax=5.2, aspect="equal")
    draw_geometry(axes[1], sensed_a, alpha=.35); draw_cameras(axes[1]); axes[1].set_title("Camera A modeled 10 m commissioning scan\nsensed obstacle heights"); axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]"); plt.colorbar(im,ax=axes[1],fraction=.035,pad=.02,label="height [m]")
    draw_field(axes[2], fields["depth_raycast"], xs, ys, driveable, prisms, title="D2 raycast field\nhatching = depth-unknown for camera A", colorbar=True, unknown=unknown_a)
    fig.suptitle("Begin state — modeled sensor-realistic commissioning depth", fontsize=16, weight="bold"); add_status(fig,"EXPLORATORY D2 MODEL — actual SDF + 10 m range + conservative unknown fallback; not a captured RGB-D frame"); save(fig, out / "01_begin_state.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], per_camera["depth_raycast"]["camera_A"], xs, ys, driveable, prisms, title="Camera A depth/raycast field", unknown=unknown_a)
    draw_field(axes[1], fields["depth_raycast"], xs, ys, driveable, prisms, title="Four-camera D2 noisy-OR", colorbar=True)
    draw_field(axes[2], fields["cad_reference"] - fields["depth_raycast"], xs, ys, driveable, prisms, title="CAD − D2 reliability\nwhere missing depth costs trust")
    fig.suptitle("Map used in planning — explicit line-of-sight with unknown-depth fallback", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "02_planning_field.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], ctx["depth_initial"], xs, ys, driveable, prisms, title="t₀ scan: low E2-south variant")
    draw_field(axes[1], ctx["depth_initial"], xs, ys, driveable, prisms, title="layout changes: stored map is stale")
    pchg = ctx["changed_prism"]
    axes[1].add_patch(Rectangle((pchg.xmin,pchg.ymin),pchg.xmax-pchg.xmin,pchg.ymax-pchg.ymin,fill=False,edgecolor="red",lw=3,zorder=12))
    axes[1].text(.62,.13,"actual rack raised\nmap still predicts old shadow",transform=axes[1].transAxes,color="red",weight="bold",bbox=dict(facecolor="white",alpha=.85))
    draw_field(axes[2], fields["depth_raycast"], xs, ys, driveable, prisms, title="rescan: height map and shadow replaced", colorbar=True)
    fig.suptitle("Update handling — scan age, stale layout, explicit rescan", fontsize=16, weight="bold"); add_status(fig,"CONTROLLED LOW→HIGH RACK VARIANT — deterministic D2 sensitivity, not a measured navigation result"); save(fig, out / "03_update_sequence.png")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    draw_field(axes[0], fields["fov_range"], xs, ys, driveable, prisms, title="D0 input absent: FOV/range")
    draw_field(axes[1], fields["depth_raycast"], xs, ys, driveable, prisms, title="D2 operational model\n10 m + unknown fallback")
    draw_field(axes[2], fields["cad_reference"], xs, ys, driveable, prisms, title="D0 complete CAD\nevaluation reference", colorbar=True)
    fig.suptitle("Depth provenance ladder — do not collapse these sources", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "05_depth_provenance_ladder.png")

    # 04 GP ----------------------------------------------------------------
    out = HERE / "04_gp/figures"
    gp_a = ctx["gp_per_camera"]["camera_A"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], gp_a["prior_P_mean_map"], xs, ys, driveable, prisms, title="Begin: camera A day-zero prior")
    draw_field(axes[1], gp_a["prior_P_mean_map"], xs, ys, driveable, prisms, title="Operational observations")
    xy = gp_a["X_train"]; hit = gp_a["p_train"]
    axes[1].scatter(xy[:,0],xy[:,1],c=hit,cmap=FIELD_CMAP,vmin=0,vmax=1,s=4,alpha=.65,zorder=8)
    draw_field(axes[2], gp_a["F_std_map"], xs, ys, driveable, prisms, title="Posterior latent uncertainty\nseparate from reliability", colorbar=True)
    fig.suptitle("Begin state — prior, spatial support and epistemic uncertainty", fontsize=16, weight="bold"); add_status(fig,"EXPLORATORY VISUALIZATION OF THE EXISTING 2,202-EVENT CAMERA-A GP ARTIFACT"); save(fig, out / "01_begin_state.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], gp_a["P_mean_map"], xs, ys, driveable, prisms, title="Camera A posterior mean")
    draw_field(axes[1], gp_a["P_conservative_plan_map"], xs, ys, driveable, prisms, title="Camera A conservative planner map")
    draw_field(axes[2], fields["gp"], xs, ys, driveable, prisms, title="Four-camera GP planner field", colorbar=True)
    fig.suptitle("Map used in planning — mean is not the conservative planner field", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "02_planning_field.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], gp_a["prior_P_mean_map"], xs, ys, driveable, prisms, title="1. day-zero prior")
    draw_field(axes[1], gp_a["prior_P_mean_map"], xs, ys, driveable, prisms, title="2. aggregate hit/miss opportunities")
    axes[1].scatter(xy[:,0],xy[:,1],c=hit,cmap=FIELD_CMAP,vmin=0,vmax=1,s=4,alpha=.72,zorder=8)
    draw_field(axes[2], gp_a["P_conservative_plan_map"], xs, ys, driveable, prisms, title="3. updated conservative map", colorbar=True)
    fig.suptitle("Update handling — operational evidence changes the reliability field", fontsize=16, weight="bold"); add_status(fig,"ACTUAL SPAWN-GRID EVENTS + EXISTING GP FIT; no held-out performance claim in this mechanism panel"); save(fig, out / "03_update_sequence.png")

    # 05 Hybrid -------------------------------------------------------------
    out = HERE / "05_hybrid/figures"
    residual_a = ctx["residuals"]["camera_A"]
    zero = np.zeros_like(residual_a)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], per_camera["depth_raycast"]["camera_A"], xs, ys, driveable, prisms, title="Cold-start depth prior")
    rim = axes[1].imshow(np.where(driveable,zero,np.nan),origin="lower",extent=(xs[0],xs[-1],ys[0],ys[-1]),cmap="coolwarm",vmin=-4,vmax=4,aspect="equal"); draw_geometry(axes[1],prisms); draw_cameras(axes[1]); axes[1].set_title("Residual GP begins at zero\nwith no operational support"); axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]"); plt.colorbar(rim,ax=axes[1],fraction=.035,pad=.02,label="logit residual")
    draw_field(axes[2], per_camera["depth_raycast"]["camera_A"], xs, ys, driveable, prisms, title="Combined cold-start field")
    fig.suptitle("Begin state — geometry supplies structure, residual begins neutral", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "01_begin_state.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], per_camera["depth_raycast"]["camera_A"], xs, ys, driveable, prisms, title="1. D2 depth prior")
    rim = axes[1].imshow(np.where(driveable,residual_a,np.nan),origin="lower",extent=(xs[0],xs[-1],ys[0],ys[-1]),cmap="coolwarm",vmin=-4,vmax=4,aspect="equal"); draw_geometry(axes[1],prisms); draw_cameras(axes[1]); axes[1].set_title("2. learned logit residual"); axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]"); plt.colorbar(rim,ax=axes[1],fraction=.035,pad=.02,label="GP residual")
    draw_field(axes[2], fields["hybrid"], xs, ys, driveable, prisms, title="3. fused conservative hybrid field", colorbar=True)
    fig.suptitle("Map used in planning — components remain inspectable", fontsize=16, weight="bold"); add_status(fig,"CANDIDATE EQUATION: logit(p_hybrid)=logit(p_D2)+GP residual; uncertainty penalty retained"); save(fig, out / "02_planning_field.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    draw_field(axes[0], fields["depth_raycast"], xs, ys, driveable, prisms, title="1. commissioned D2 prior")
    draw_field(axes[1], fields["depth_raycast"], xs, ys, driveable, prisms, title="2. hit/miss events update residual only")
    all_xy = ctx["events"]["camera_A"]["xy"]; all_hit=ctx["events"]["camera_A"]["hit"]
    axes[1].scatter(all_xy[:,0],all_xy[:,1],c=all_hit,cmap=FIELD_CMAP,vmin=0,vmax=1,s=2.5,alpha=.45,zorder=8)
    draw_field(axes[2], fields["hybrid"], xs, ys, driveable, prisms, title="3. combined field after updates", colorbar=True)
    fig.suptitle("Update handling — detector evidence updates residual; rescan updates geometry", fontsize=16, weight="bold"); add_status(fig); save(fig, out / "03_update_sequence.png")

    x_idx = int(np.argmin(np.abs(xs + 6.7)))
    fig, ax = plt.subplots(figsize=(11.2,5.2))
    for key, label, color in [("fov_range","FOV/range","#00a6a6"),("depth_raycast","D2 depth","#d89000"),("gp","GP","#7b53b5"),("hybrid","hybrid","#2a9d58"),("cad_reference","CAD ref.","#d94b4b")]:
        ax.plot(ys, fields[key][:,x_idx],lw=2.2,label=label,color=color)
    ax.axvspan(3.2,6.9,color="#7d858e",alpha=.18,label="W2-north rack band")
    ax.set(xlabel="y along x=-6.7 m cross-section [m]",ylabel=r"$p_{use}$",ylim=(-.03,1.03),title="Rack-shadow boundary: sharp geometry plus learned residual")
    ax.grid(alpha=.25); ax.legend(ncol=3)
    fig.tight_layout(); add_status(fig); save(fig, out / "05_boundary_cross_section.png")

    # 06 CAD ---------------------------------------------------------------
    out = HERE / "06_cad_reference/figures"
    h_cad = height_map(xs,ys,prisms)
    fig, axes = plt.subplots(1,3,figsize=(16,5.2))
    gazebo_panel(axes[0],"Actual Gazebo world")
    im=axes[1].imshow(np.where(h_cad>0,h_cad,np.nan),origin="lower",extent=(xs[0],xs[-1],ys[0],ys[-1]),cmap="cividis",vmin=0,vmax=5.2,aspect="equal"); draw_geometry(axes[1],prisms,alpha=.35); draw_cameras(axes[1]); axes[1].set_title("Exact SDF collision heights"); axes[1].set_xlabel("x [m]");axes[1].set_ylabel("y [m]");plt.colorbar(im,ax=axes[1],fraction=.035,pad=.02,label="height [m]")
    draw_field(axes[2],fields["cad_reference"],xs,ys,driveable,prisms,title="Complete-map raycast field",colorbar=True)
    fig.suptitle("Begin state — complete CAD is available only because Gazebo exposes truth",fontsize=16,weight="bold"); add_status(fig,"EVALUATION-ONLY REFERENCE — exact SDF collision geometry is forbidden to operational sensing arms"); save(fig,out/"01_begin_state.png")

    fig,axes=plt.subplots(2,2,figsize=(12,9.6),sharex=True,sharey=True)
    for ax,camera_id in zip(axes.flat,CAMERA_POSES): draw_field(ax,per_camera["cad_reference"][camera_id],xs,ys,driveable,prisms,title=f"{camera_id}: exact-prism raycast")
    fig.suptitle("Map used for reference planning — complete static line of sight",fontsize=16,weight="bold");add_status(fig,"EVALUATION-ONLY REFERENCE");save(fig,out/"02_planning_field.png")

    fig,axes=plt.subplots(1,3,figsize=(16,5.2))
    draw_field(axes[0],fields["cad_reference"],xs,ys,driveable,prisms,title="exact world at t₀")
    axes[1].axis("off");axes[1].text(.5,.64,"NO OPERATIONAL UPDATE",ha="center",fontsize=16,weight="bold");axes[1].text(.5,.45,"loading a changed SDF supplies\nnew simulator truth — it is not a rescan",ha="center",fontsize=12,color="#9b2c2c")
    draw_field(axes[2],fields["cad_reference"],xs,ys,driveable,prisms,title="same exact-reference field")
    fig.suptitle("Update handling — oracle/reference has no deployment clock",fontsize=16,weight="bold");add_status(fig,"EVALUATION-ONLY REFERENCE");save(fig,out/"03_update_sequence.png")

    fig,axes=plt.subplots(1,3,figsize=(16,5.2))
    im=axes[0].imshow(np.where(h_cad>0,h_cad,np.nan),origin="lower",extent=(xs[0],xs[-1],ys[0],ys[-1]),cmap="cividis",vmin=0,vmax=5.2,aspect="equal");draw_geometry(axes[0],prisms,alpha=.3);axes[0].set_title("complete CAD height map");axes[0].set_xlabel("x [m]");axes[0].set_ylabel("y [m]")
    im2=axes[1].imshow(np.where(h_sensed>0,h_sensed,np.nan),origin="lower",extent=(xs[0],xs[-1],ys[0],ys[-1]),cmap="cividis",vmin=0,vmax=5.2,aspect="equal");draw_geometry(axes[1],sensed_a,alpha=.3);axes[1].set_title("camera-A D2 sensed subset");axes[1].set_xlabel("x [m]");axes[1].set_ylabel("y [m]")
    diff=np.clip(h_cad-h_sensed,0,None);imd=axes[2].imshow(np.where(diff>0,diff,np.nan),origin="lower",extent=(xs[0],xs[-1],ys[0],ys[-1]),cmap="magma",vmin=0,vmax=5.2,aspect="equal");draw_geometry(axes[2],prisms,alpha=.15);axes[2].set_title("geometry missing from D2 scan");axes[2].set_xlabel("x [m]");axes[2].set_ylabel("y [m]");plt.colorbar(imd,ax=axes[2],fraction=.035,pad=.02,label="missing height [m]")
    fig.suptitle("Complete model geometry and sensed structure are different inputs",fontsize=16,weight="bold");add_status(fig);save(fig,out/"05_cad_vs_sensed_surface.png")

    # Shared method-specific route grids ----------------------------------
    route_keys=("R1","R2","R3","R6")
    for key,label,folder,color in METHODS:
        fig,axes=plt.subplots(2,2,figsize=(12.5,9.6),sharex=True,sharey=True)
        for ax,route_key in zip(axes.flat,route_keys):
            draw_field(ax,fields[key],xs,ys,driveable,prisms,title=f"{route_key}: {route_title(route_key)}")
            draw_routes(ax,fields[key],driveable,xs,ys,route_spec(ctx["tasks"],route_key),color)
        fig.suptitle(f"{label} — common offline explanatory routes",fontsize=16,weight="bold")
        fig.legend(handles=[Line2D([],[],color=color,lw=2.6,label="reliability-weighted grid route"),Line2D([],[],marker="o",color="w",markerfacecolor="#21a65a",markeredgecolor="black",label="start"),Line2D([],[],marker="*",color="w",markerfacecolor="#e3342f",markeredgecolor="black",markersize=12,label="goal")],loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.02))
        add_status(fig,"OFFLINE EXPLANATORY ROUTE SEARCH — same driveable grid and cost for every method; not a closed-loop result")
        save(fig,HERE/folder/"figures/04_route_grid.png")


def route_title(key: str) -> str:
    return {"R1":"short blind vs detour","R2":"equal-length occlusion pair","R3":"camera handover","R6":"uniformly good control"}[key]


def render_cross_method_routes(ctx: dict) -> None:
    xs,ys=ctx["xs"],ctx["ys"];driveable=ctx["driveable"];prisms=ctx["prisms"];fields=ctx["fields"]
    out=HERE/"07_route_comparison/figures";out.mkdir(parents=True,exist_ok=True)
    paths={}
    for route_key,filename in [("R1","R1_short_vs_visible.png"),("R2","R2_equal_length_occlusion.png"),("R3","R3_handover.png"),("R6","R6_uniform_control.png")]:
        fig,axes=plt.subplots(2,3,figsize=(17,10.2),sharex=True,sharey=True)
        for ax,(key,label,folder,color) in zip(axes.flat,METHODS):
            draw_field(ax,fields[key],xs,ys,driveable,prisms,title=label)
            method_paths=draw_routes(ax,fields[key],driveable,xs,ys,route_spec(ctx["tasks"],route_key),color)
            paths[(route_key,key)]=method_paths
        fig.suptitle(f"{route_key} — {route_title(route_key)}: same task and route cost across methods",fontsize=17,weight="bold")
        add_status(fig,"OFFLINE EXPLANATORY ROUTE SEARCH — visual mechanism comparison, not navigation-performance evidence")
        save(fig,out/filename)

    # Contact sheet uses the generated route matrices without recomputation.
    files=[out/"R1_short_vs_visible.png",out/"R2_equal_length_occlusion.png",out/"R3_handover.png",out/"R6_uniform_control.png"]
    fig,axes=plt.subplots(2,2,figsize=(18,11.5))
    for ax,path in zip(axes.flat,files):
        ax.imshow(plt.imread(path));ax.axis("off");ax.set_title(path.stem.replace("_"," "),fontsize=12,weight="bold")
    fig.suptitle("Supervisor contact sheet — source → field → route consequence",fontsize=18,weight="bold")
    fig.tight_layout();save(fig,out/"all_methods_contact_sheet.png")


def build_context() -> dict:
    with np.load(DAYZERO,allow_pickle=False) as dayzero_data:
        dayzero={k:np.asarray(dayzero_data[k]) for k in dayzero_data.files}
    xs=np.asarray(dayzero["xs"],dtype=float);ys=np.asarray(dayzero["ys"],dtype=float)
    driveable_prisms=profile_regions();driveable=mask_from_prisms(xs,ys,driveable_prisms)
    prisms=tuple(parse_occlusion_scene_from_world(str(WORLD),model_name="warehouse_rack_occluders",geometry_tags=("collision",)).prisms)
    calibrations,models=cameras()
    events=load_events();prevalence,intercept,slope=fit_distance_baseline(events)
    distance_per,distance_fused,constant=distance_fields(xs,ys,prevalence,intercept,slope)
    fov_per,fov_fused=fov_fields(dayzero)
    cad_per,cad_fused=exact_occlusion_fields(xs,ys,models,fov_per,prisms)
    depth_per,depth_fused,sensed,unknown=depth_d2_fields(xs,ys,models,fov_per,prisms)
    gp_per,gp_fused=load_gp_fields()
    hybrid_per,residuals,hybrid_fused=hybrid_fields(depth_per,gp_per)

    # Controlled, preregistered low→high rack-height variant for the rescan panel.
    changed=next(p for p in prisms if "rack_E2_south" in p.name)
    low_variant=tuple(replace(p,zmax=1.20) if p.name==changed.name else p for p in prisms)
    _,depth_initial,_,_=depth_d2_fields(xs,ys,models,fov_per,low_variant)

    fields={
        "constant_distance":distance_fused,
        "fov_range":fov_fused,
        "depth_raycast":depth_fused,
        "gp":gp_fused,
        "hybrid":hybrid_fused,
        "cad_reference":cad_fused,
    }
    per_camera={
        "constant_distance":distance_per,
        "fov_range":fov_per,
        "depth_raycast":depth_per,
        "gp":{c:gp_per[c]["P_conservative_plan_map"] for c in CAMERA_POSES},
        "hybrid":hybrid_per,
        "cad_reference":cad_per,
    }
    return {
        "xs":xs,"ys":ys,"driveable":driveable,"driveable_prisms":driveable_prisms,"prisms":prisms,
        "fields":fields,"per_camera":per_camera,"constant":constant,"prevalence":prevalence,
        "distance_fit":(intercept,slope),"events":events,"gp_per_camera":gp_per,"residuals":residuals,
        "sensed_prisms":sensed,"unknown":unknown,"depth_initial":depth_initial,"changed_prism":changed,
        "tasks":tasks(),"calibrations":calibrations,
    }


def write_artifacts(ctx: dict) -> None:
    data_dir=HERE/"generated_data";data_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(data_dir/"method_fields.npz",xs=ctx["xs"],ys=ctx["ys"],driveable_mask=ctx["driveable"],**{f"{k}_field":v for k,v in ctx["fields"].items()})
    inputs=[WORLD,PROFILE,TASKS,GAZEBO_OVERVIEW,*GAZEBO_CAMERA.values(),DAYZERO,GP_FUSED]
    inputs += [GP_ROOT/c/"det_hit_expected_kernel_gp.npz" for c in CAMERA_POSES]
    inputs += [EVENT_ROOT/f"{c}_events.csv" for c in CAMERA_POSES]
    outputs=sorted(HERE.glob("*/figures/*.png"))
    manifest={
        "status":"exploratory_method_explanation",
        "renderer":str(Path(__file__).relative_to(REPO)),
        "depth_semantics":"D2 analytic sensor-realism model over actual SDF: 10 m structure range and 0.20×FOV conservative target-cell fallback; not a captured RGB-D frame",
        "hybrid_semantics":"candidate logit(depth_D2)+operational_GP_logit_residual with existing GP uncertainty penalty",
        "route_semantics":f"common 8-neighbor driveable-grid Dijkstra; edge cost=distance×[1+{ROUTE_LAMBDA}(1-p_use)^2]; explanatory only",
        "constant_per_camera_prevalence":ctx["prevalence"],
        "distance_logit_intercept":ctx["distance_fit"][0],
        "distance_logit_slope_per_m":ctx["distance_fit"][1],
        "inputs":{str(p.relative_to(REPO)):sha256(p) for p in inputs},
        "outputs":{str(p.relative_to(HERE)):sha256(p) for p in outputs},
    }
    (data_dir/"render_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


def main() -> None:
    plt.rcParams.update({"font.size":9.5,"axes.grid":False,"figure.constrained_layout.use":True})
    ctx=build_context()
    render_method_figures(ctx)
    render_cross_method_routes(ctx)
    write_artifacts(ctx)
    count=len(list(HERE.glob("*/figures/*.png")))
    print(f"rendered {count} PNG figures")
    print(f"manifest: {(HERE/'generated_data/render_manifest.json').relative_to(REPO)}")


if __name__=="__main__":
    main()
