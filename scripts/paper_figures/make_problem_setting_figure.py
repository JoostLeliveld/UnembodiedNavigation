#!/usr/bin/env python3
"""Create the paper problem-setting figure.

The intended paper figure is deliberately simple and caveat-first:

1. panel (a): a real external-camera/Gazebo image, supplied with
   --panel-a-image or --gazebo-screenshot;
2. panel (b): a clean top-down problem statement for a planner with constant
   observation covariance R0.

The top-down panel should motivate state-dependent observation covariance,
not show the learned GP solution. A pale region marks where external-camera
updates are expected to be weak; it is not a reward, cost, or GP map.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_REPORT = REPO_ROOT.parent / "thesis-report"
TASKS_PATH = REPO_ROOT / "src" / "experiments" / "config" / "tasks.yaml"
WORLD_PATH = REPO_ROOT / "src" / "sim" / "gazebo_worlds" / "worlds" / "warehouse_occ_light.world.sdf"
DEFAULT_VISIBILITY_ARTIFACT = REPO_ROOT / "logs" / "visibility_comparison" / "current_gp" / "yolo_score_raw_gp.npz"
OUT_DIR = THESIS_REPORT / "figures"


COLORS = {
    "camera": "#222222",
    "start": "#0a8f2a",
    "goal": "#e41a1c",
    "truth": "#2f61d5",
    "belief": "#7b3294",
    "short": "#e69f00",
    "fail": "#c83f2d",
    "low_rel": "#e6ad3f",
    "weak_ray": "#777777",
    "shelf": "#a6a6a6",
    "shelf_edge": "#4d4d4d",
    "floor": "#fbfbfb",
    "grid": "#d8d8d8",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gazebo-screenshot",
        type=Path,
        default=None,
        help="Deprecated alias for --panel-a-image.",
    )
    parser.add_argument(
        "--panel-a-image",
        type=Path,
        default=None,
        help="Path to a real external-camera or Gazebo screenshot for panel (a).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        required=True,
        help="Constant-R0 experiment run directory for snapshot panels.",
    )
    parser.add_argument(
        "--snapshot-times",
        type=float,
        nargs="+",
        default=(0.0, 16.0),
        metavar="T",
        help=(
            "Snapshot times in seconds after first command for top-down panels. "
            "Use two times for the compact paper figure, or three for early/stale/recovered sequences."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where the PDF/PNG outputs are written.",
    )
    parser.add_argument(
        "--visibility-artifact-path",
        type=Path,
        default=DEFAULT_VISIBILITY_ARTIFACT,
        help="Optional detector-derived visibility artifact for --visibility-field gp.",
    )
    parser.add_argument(
        "--visibility-field",
        choices=("geometry", "gp"),
        default="geometry",
        help="Auxiliary setup plot only: strict geometric line-of-sight or learned detector-derived GP field.",
    )
    parser.add_argument(
        "--write-auxiliary",
        action="store_true",
        help="Also write auxiliary setup/topdown panels. The main paper figure is problem_setup.*.",
    )
    parser.add_argument(
        "--allow-nonconstant-r0",
        action="store_true",
        help="Allow plotting a run that is not planner=constant_R_efe/use_visibility_model=false.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="shadow_tradeoff_a",
        help="Task entry from tasks.yaml whose start/goal define the figure markers.",
    )
    parser.add_argument(
        "--use-raw-state",
        action="store_true",
        help=(
            "Use state_x/state_y (raw YOLO-based position) instead of planner_belief_x/y "
            "for the belief trace. Shows the camera estimate that freezes in the shadow zone."
        ),
    )
    return parser.parse_args()


def _load_task(task_name: str = "shadow_tradeoff_a") -> tuple[dict, dict]:
    payload = yaml.safe_load(TASKS_PATH.read_text(encoding="utf-8"))
    tasks = payload["tasks"]["warehouse_occ_light.world.sdf"]
    for task in tasks:
        if task["name"] == task_name:
            return task["start"], task["goal"]
    raise RuntimeError(f"{task_name!r} not found in tasks.yaml")


def _parse_vec(text: str) -> list[float]:
    return [float(v) for v in str(text).split()]


def _first_pose(parent: ET.Element) -> list[float]:
    pose = parent.find("pose")
    if pose is None or pose.text is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    values = _parse_vec(pose.text)
    return values + [0.0] * (6 - len(values))


def _box_size(link: ET.Element) -> tuple[float, float, float]:
    size = link.find(".//geometry/box/size")
    if size is None or size.text is None:
        raise RuntimeError("Expected box size in SDF link")
    sx, sy, sz = _parse_vec(size.text)
    return float(sx), float(sy), float(sz)


def _load_world_geometry() -> dict:
    tree = ET.parse(WORLD_PATH)
    root = tree.getroot()

    shelf = {}
    for model in root.findall(".//model"):
        if model.attrib.get("name") == "warehouse_rack_occluders":
            for link in model.findall("link"):
                name = link.attrib.get("name", "")
                pose = _first_pose(link)
                sx, sy, sz = _box_size(link)
                shelf[name] = {"x": pose[0], "y": pose[1], "z": pose[2], "sx": sx, "sy": sy, "sz": sz}

    camera_pose = None
    for include in root.findall(".//include"):
        uri = include.findtext("uri", "")
        if "external_camera" in uri:
            camera_pose = _first_pose(include)
            break
    if camera_pose is None:
        raise RuntimeError("External camera include not found in world")

    return {"shelf": shelf, "camera_pose": camera_pose}


def _ray_intersects_prism(
    camera: np.ndarray,
    target: np.ndarray,
    prism: dict[str, float],
) -> bool:
    bounds = (
        (float(prism["x"] - prism["sx"] / 2.0), float(prism["x"] + prism["sx"] / 2.0)),
        (float(prism["y"] - prism["sy"] / 2.0), float(prism["y"] + prism["sy"] / 2.0)),
        (float(prism["z"] - prism["sz"] / 2.0), float(prism["z"] + prism["sz"] / 2.0)),
    )
    direction = target - camera
    t_min = 0.0
    t_max = 1.0
    for axis, (lo, hi) in enumerate(bounds):
        d = float(direction[axis])
        origin = float(camera[axis])
        if abs(d) < 1e-12:
            if origin < lo or origin > hi:
                return False
            continue
        t1 = (lo - origin) / d
        t2 = (hi - origin) / d
        t_axis_min = min(t1, t2)
        t_axis_max = max(t1, t2)
        t_min = max(t_min, t_axis_min)
        t_max = min(t_max, t_axis_max)
        if t_max < t_min:
            return False
    return 0.0 < t_max and t_min < 1.0


def _compute_geometry_visibility_field(
    geom: dict,
    *,
    n: int = 220,
    target_height_m: float = 0.10,
) -> dict[str, np.ndarray]:
    xs = np.linspace(-3.0, 3.0, int(n))
    ys = np.linspace(-3.0, 3.0, int(n))
    cam_pose = geom["camera_pose"]
    camera = np.array([float(cam_pose[0]), float(cam_pose[1]), float(cam_pose[2])], dtype=float)
    shelf_prisms = [
        dict(item)
        for name, item in geom["shelf"].items()
        if name in ("shelf_body", "shelf_top")
    ]
    visible = np.ones((ys.size, xs.size), dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            target = np.array([float(x), float(y), float(target_height_m)], dtype=float)
            blocked = any(_ray_intersects_prism(camera, target, prism) for prism in shelf_prisms)
            visible[iy, ix] = 0.0 if blocked else 1.0
    return {"xs": xs, "ys": ys, "p_map": visible, "kind": "geometry"}


def _load_gp_visibility_field(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.is_file():
        raise RuntimeError(f'GP visibility field not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        xs = np.asarray(data["xs"], dtype=float)
        ys = np.asarray(data["ys"], dtype=float)
        if "P_conservative_plan_map" not in data.files:
            raise RuntimeError(f'Paper GP artifact is missing P_conservative_plan_map: {path}')
        p_map = np.asarray(data["P_conservative_plan_map"], dtype=float)
    return {"xs": xs, "ys": ys, "p_map": p_map, "kind": "gp"}


def _load_visibility_field(path: Path | None, geom: dict, *, mode: str) -> dict[str, np.ndarray]:
    if str(mode).strip().lower() == "gp":
        return _load_gp_visibility_field(path)
    return _compute_geometry_visibility_field(geom)


def _ray_endpoint_on_workspace(cam_xy: np.ndarray, direction: np.ndarray, *, bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0)) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    candidates = []
    if abs(direction[0]) > 1e-12:
        for x_bound in (xmin, xmax):
            t = (x_bound - cam_xy[0]) / direction[0]
            y = cam_xy[1] + t * direction[1]
            if t > 0.0 and ymin - 1e-9 <= y <= ymax + 1e-9:
                candidates.append((t, np.array([x_bound, y], dtype=float)))
    if abs(direction[1]) > 1e-12:
        for y_bound in (ymin, ymax):
            t = (y_bound - cam_xy[1]) / direction[1]
            x = cam_xy[0] + t * direction[0]
            if t > 0.0 and xmin - 1e-9 <= x <= xmax + 1e-9:
                candidates.append((t, np.array([x, y_bound], dtype=float)))
    if not candidates:
        return cam_xy
    return min(candidates, key=lambda item: item[0])[1]


def _draw_geometry_shadow_boundaries(ax, geom: dict) -> None:
    shelf = geom["shelf"].get("shelf_top") or geom["shelf"]["shelf_body"]
    cam_x, cam_y, *_ = geom["camera_pose"]
    cam_xy = np.array([float(cam_x), float(cam_y)], dtype=float)
    corners = np.array(
        [
            [shelf["x"] - shelf["sx"] / 2.0, shelf["y"] - shelf["sy"] / 2.0],
            [shelf["x"] + shelf["sx"] / 2.0, shelf["y"] - shelf["sy"] / 2.0],
            [shelf["x"] + shelf["sx"] / 2.0, shelf["y"] + shelf["sy"] / 2.0],
            [shelf["x"] - shelf["sx"] / 2.0, shelf["y"] + shelf["sy"] / 2.0],
        ],
        dtype=float,
    )
    vectors = corners - cam_xy[None, :]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    # In this task all shelf-corner rays lie in one angular branch; unwrap keeps
    # the tangent selection robust if the camera is moved later.
    unwrapped = np.unwrap(angles)
    for idx in (int(np.argmin(unwrapped)), int(np.argmax(unwrapped))):
        direction = vectors[idx] / max(np.linalg.norm(vectors[idx]), 1e-9)
        end = _ray_endpoint_on_workspace(cam_xy, direction)
        ax.plot(
            [cam_xy[0], end[0]],
            [cam_xy[1], end[1]],
            color="#b9821f",
            linewidth=0.85,
            linestyle=(0, (4, 2)),
            alpha=0.78,
            zorder=3,
        )


def _resolve_run_dir(run_dir: Path | None) -> Path | None:
    if run_dir is not None:
        return run_dir if (run_dir / "experiment.csv").exists() else None
    return None


def _load_run_manifest(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {}
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_run_trace(run_dir: Path | None, use_raw_state: bool = False) -> dict[str, np.ndarray]:
    if run_dir is None:
        raise RuntimeError("--run-dir is required for paper figure generation")

    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required to load experiment traces") from None

    exp_path = run_dir / "experiment.csv"
    if not exp_path.exists():
        raise RuntimeError(f"Missing experiment.csv in run directory: {run_dir}")

    df = pd.read_csv(exp_path)
    traces: dict[str, np.ndarray] = {}

    if use_raw_state:
        # Use state_x/y (raw YOLO-homography position) as the belief trace.
        # This column only updates when YOLO fires; it freezes in shadow, showing
        # camera-based localization failure in the invisible region.
        required_columns = ("stamp", "truth_x", "truth_y", "state_x", "state_y")
        missing_columns = [name for name in required_columns if name not in df.columns]
        if missing_columns:
            raise RuntimeError(f"experiment.csv is missing required columns: {', '.join(missing_columns)}")
        for name in required_columns:
            traces[name] = df[name].to_numpy(dtype=float)
        # Map state columns → planner_belief keys so the rest of the code is unchanged
        traces["planner_belief_x"] = traces.pop("state_x")
        traces["planner_belief_y"] = traces.pop("state_y")
        # Covariance: use EKF covariance if available, else small fixed value
        for cov_col, default in (("planner_cov_x", 1e-6), ("planner_cov_xy", 0.0), ("planner_cov_y", 1e-6)):
            if cov_col in df.columns:
                traces[cov_col] = df[cov_col].to_numpy(dtype=float)
            else:
                traces[cov_col] = np.full(len(df), default)
    else:
        required_columns = (
            "stamp",
            "truth_x",
            "truth_y",
            "planner_belief_x",
            "planner_belief_y",
            "planner_cov_x",
            "planner_cov_xy",
            "planner_cov_y",
        )
        missing_columns = [name for name in required_columns if name not in df.columns]
        if missing_columns:
            raise RuntimeError(f"experiment.csv is missing required columns: {', '.join(missing_columns)}")
        for name in required_columns:
            if name in df.columns:
                traces[name] = df[name].to_numpy(dtype=float)
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            summary = {}
        if isinstance(summary, dict) and "first_cmd_stamp" in summary:
            try:
                traces["first_cmd_stamp"] = np.asarray([float(summary["first_cmd_stamp"])], dtype=float)
            except (TypeError, ValueError):
                pass
    plan_path = run_dir / "plan_samples.csv"
    if plan_path.exists():
        plan_df = pd.read_csv(plan_path)
        if {"plan_stamp", "point_idx", "x", "y"}.issubset(plan_df.columns):
            first_stamp = plan_df["plan_stamp"].dropna().iloc[0]
            group = plan_df[plan_df["plan_stamp"] == first_stamp].sort_values("point_idx")
            traces["plan_x"] = group["x"].to_numpy(dtype=float)
            traces["plan_y"] = group["y"].to_numpy(dtype=float)
            traces["all_plan_stamp"] = plan_df["plan_stamp"].to_numpy(dtype=float)
            traces["all_plan_point_idx"] = plan_df["point_idx"].to_numpy(dtype=float)
            traces["all_plan_x"] = plan_df["x"].to_numpy(dtype=float)
            traces["all_plan_y"] = plan_df["y"].to_numpy(dtype=float)
        else:
            raise RuntimeError(f"plan_samples.csv is missing required columns in {run_dir}")
    else:
        raise RuntimeError(f"Missing plan_samples.csv in run directory: {run_dir}")
    return traces


def _bezier(points: list[tuple[float, float]], **kwargs) -> PathPatch:
    verts = [points[0], points[1], points[2], points[3]]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return PathPatch(MplPath(verts, codes), fill=False, capstyle="round", joinstyle="round", **kwargs)


def _draw_screenshot_panel(ax, screenshot: Path | None) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(a) simulation setup", fontsize=10, loc="center", pad=6)
    if screenshot is not None and screenshot.exists():
        image = mpimg.imread(screenshot)
        ax.imshow(image, aspect="auto")
        ax.text(
            0.03, 0.06, "External-camera / Gazebo view",
            transform=ax.transAxes, fontsize=8, color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3},
        )
        return

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eeeeee", edgecolor="#999999", linewidth=0.6))
    ax.text(0.5, 0.58, "External-camera /\nGazebo view",
            ha="center", va="center", fontsize=11, color="#444444")
    ax.text(0.5, 0.30, "(insert image with\n--panel-a-image)",
            ha="center", va="center", fontsize=8, color="#777777", style="italic")


def _draw_reliability_overlay(ax, geom: dict, visibility: dict[str, np.ndarray]) -> None:
    if visibility:
        xs = visibility["xs"]
        ys = visibility["ys"]
        p_map = np.asarray(visibility["p_map"], dtype=float)
        if str(visibility.get("kind", "")).strip().lower() == "geometry":
            low_alpha = np.asarray(1.0 - p_map, dtype=float) * 0.28
        else:
            low_alpha = np.clip((0.45 - p_map) / 0.45, 0.0, 1.0) * 0.34
        ax.imshow(
            np.ones_like(p_map),
            origin="lower",
            extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
            cmap=ListedColormap([COLORS["low_rel"]]),
            alpha=low_alpha,
            aspect="equal",
            zorder=1,
        )
        finite = p_map[np.isfinite(p_map)]
        if finite.size and float(np.nanmin(finite)) < 0.30 < float(np.nanmax(finite)):
            ax.contour(xs, ys, p_map, levels=[0.30], colors="#ad7a10", linewidths=0.55, alpha=0.65, zorder=2)
        if str(visibility.get("kind", "")).strip().lower() == "geometry":
            _draw_geometry_shadow_boundaries(ax, geom)
        return

    # Fallback only: approximate the shelf shadow when no detector-derived map is available.
    shelf_body = geom["shelf"]["shelf_body"]
    cam_x, cam_y, *_ = geom["camera_pose"]
    cam = np.array([cam_x, cam_y])
    shelf_center = np.array([shelf_body["x"], shelf_body["y"]])
    direction = shelf_center - cam
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    behind_center = shelf_center + 1.45 * direction
    low_poly = np.array(
        [
            shelf_center - 1.25 * normal - 0.03 * direction,
            shelf_center + 1.25 * normal - 0.03 * direction,
            behind_center + 1.85 * normal + 1.05 * direction,
            behind_center - 1.85 * normal + 1.05 * direction,
        ]
    )
    ax.add_patch(Polygon(low_poly, closed=True, facecolor=COLORS["low_rel"], alpha=0.16, edgecolor="none", zorder=1))


def _draw_workspace(ax, geom: dict, visibility: dict[str, np.ndarray] | None = None) -> None:
    shelf_body = geom["shelf"]["shelf_body"]
    shelf_top = geom["shelf"].get("shelf_top", shelf_body)
    cam_x, cam_y, *_ = geom["camera_pose"]

    ax.add_patch(Rectangle((-3, -3), 6, 6, facecolor=COLORS["floor"], edgecolor="#777777", linewidth=0.8, zorder=0))
    ax.add_patch(
        Rectangle(
            (shelf_body["x"] - shelf_body["sx"] / 2.0, shelf_body["y"] - shelf_body["sy"] / 2.0),
            shelf_body["sx"],
            shelf_body["sy"],
            facecolor=COLORS["shelf"],
            edgecolor=COLORS["shelf_edge"],
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (shelf_top["x"] - shelf_top["sx"] / 2.0, shelf_top["y"] - shelf_top["sy"] / 2.0),
            shelf_top["sx"],
            shelf_top["sy"],
            facecolor="none",
            edgecolor="#666666",
            linewidth=0.8,
            zorder=6,
        )
    )

    _draw_reliability_overlay(ax, geom, visibility or {})

    ax.scatter([cam_x], [cam_y], marker="<", s=40, color=COLORS["camera"], zorder=9)


def _camera_shadow_polygon(geom: dict) -> np.ndarray:
    shelf = geom["shelf"].get("shelf_top") or geom["shelf"]["shelf_body"]
    cam_x, cam_y, *_ = geom["camera_pose"]
    cam_xy = np.array([float(cam_x), float(cam_y)], dtype=float)
    corners = np.array(
        [
            [shelf["x"] - shelf["sx"] / 2.0, shelf["y"] - shelf["sy"] / 2.0],
            [shelf["x"] + shelf["sx"] / 2.0, shelf["y"] - shelf["sy"] / 2.0],
            [shelf["x"] + shelf["sx"] / 2.0, shelf["y"] + shelf["sy"] / 2.0],
            [shelf["x"] - shelf["sx"] / 2.0, shelf["y"] + shelf["sy"] / 2.0],
        ],
        dtype=float,
    )
    vectors = corners - cam_xy[None, :]
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    low_idx = int(np.argmin(angles))
    high_idx = int(np.argmax(angles))
    low_corner = corners[low_idx]
    high_corner = corners[high_idx]
    low_end = _ray_endpoint_on_workspace(cam_xy, vectors[low_idx] / max(np.linalg.norm(vectors[low_idx]), 1e-9))
    high_end = _ray_endpoint_on_workspace(cam_xy, vectors[high_idx] / max(np.linalg.norm(vectors[high_idx]), 1e-9))
    return np.array([low_corner, high_corner, high_end, low_end], dtype=float)


def _draw_paper_workspace(ax, geom: dict, *, show_camera_direction: bool = False) -> None:
    shelf_body = geom["shelf"]["shelf_body"]
    shelf_top = geom["shelf"].get("shelf_top", shelf_body)
    cam_x, cam_y, *_ = geom["camera_pose"]

    ax.add_patch(Rectangle((-3, -3), 6, 6, facecolor=COLORS["floor"], edgecolor="#777777", linewidth=0.8, zorder=0))
    shadow = _camera_shadow_polygon(geom)
    ax.add_patch(
        Polygon(
            shadow,
            closed=True,
            facecolor=COLORS["low_rel"],
            edgecolor="none",
            alpha=0.16,
            zorder=1,
        )
    )
    if show_camera_direction:
        shelf_center = np.array([float(shelf_body["x"]), float(shelf_body["y"])], dtype=float)
        ax.plot(
            [cam_x, shelf_center[0]],
            [cam_y, shelf_center[1]],
            color="#777777",
            linewidth=0.6,
            linestyle=(0, (2, 3)),
            alpha=0.45,
            zorder=2,
        )
    ax.add_patch(
        Rectangle(
            (shelf_body["x"] - shelf_body["sx"] / 2.0, shelf_body["y"] - shelf_body["sy"] / 2.0),
            shelf_body["sx"],
            shelf_body["sy"],
            facecolor="#bdbdbd",
            edgecolor="#1f1f1f",
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (shelf_top["x"] - shelf_top["sx"] / 2.0, shelf_top["y"] - shelf_top["sy"] / 2.0),
            shelf_top["sx"],
            shelf_top["sy"],
            facecolor="none",
            edgecolor="#4a4a4a",
            linewidth=0.75,
            zorder=6,
        )
    )
    ax.scatter([cam_x], [cam_y], marker="<", s=38, color=COLORS["camera"], zorder=9)


def _covariance_ellipse(
    ax,
    xy: tuple[float, float],
    cov: np.ndarray,
    *,
    color: str,
    alpha: float = 0.17,
    zorder: int = 4,
    sigma: float = 2.0,
) -> None:
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-5, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * float(sigma) * np.sqrt(vals)
    ax.add_patch(Ellipse(xy, width, height, angle=angle, facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.9, zorder=zorder))


def _extract_problem_paths_from_trace(traces: dict[str, np.ndarray], start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> dict[str, np.ndarray]:
    del start_xy, goal_xy
    if not {"truth_x", "truth_y", "planner_belief_x", "planner_belief_y"}.issubset(traces):
        raise RuntimeError("Problem top-down panel requires truth and planner-belief columns")
    truth_x, truth_y = _finite_xy(traces["truth_x"], traces["truth_y"])
    belief_x, belief_y = _finite_xy(traces["planner_belief_x"], traces["planner_belief_y"])
    if truth_x.size < 4 or belief_x.size < 4:
        raise RuntimeError("Problem top-down panel requires at least four valid truth/belief samples")
    n = min(truth_x.size, belief_x.size)
    truth = np.column_stack([truth_x[:n], truth_y[:n]])
    belief = np.column_stack([belief_x[:n], belief_y[:n]])
    if "plan_x" in traces and "plan_y" in traces:
        plan_x, plan_y = _finite_xy(traces["plan_x"], traces["plan_y"])
        plan = np.column_stack([plan_x, plan_y]) if plan_x.size >= 2 else truth
    else:
        plan = truth

    if "stamp" in traces and len(traces["stamp"]) >= n:
        stamps = np.asarray(traces["stamp"][:n], dtype=float)
        finite = np.isfinite(stamps)
        if np.count_nonzero(finite) >= 2:
            t0 = float(np.nanmin(stamps[finite]))
            t1 = float(np.nanmax(stamps[finite]))
            marker_times = np.linspace(t0, t1, min(6, n))
            marker_indices = np.array([int(np.nanargmin(np.abs(stamps - t))) for t in marker_times], dtype=int)
        else:
            marker_indices = np.linspace(0, n - 1, min(6, n)).astype(int)
    else:
        marker_indices = np.linspace(0, n - 1, min(6, n)).astype(int)

    cov_x = traces.get("planner_cov_x")
    cov_y = traces.get("planner_cov_y")
    cov_xy = traces.get("planner_cov_xy")
    if cov_x is None or cov_y is None:
        raise RuntimeError("Problem top-down panel requires planner covariance columns")
    covs = []
    for idx in marker_indices:
        if idx >= len(cov_x) or idx >= len(cov_y):
            raise RuntimeError("Planner covariance columns are shorter than the selected trace markers")
        xy_val = float(cov_xy[idx]) if cov_xy is not None and idx < len(cov_xy) and math.isfinite(float(cov_xy[idx])) else 0.0
        cov = np.array([[float(cov_x[idx]), xy_val], [xy_val, float(cov_y[idx])]], dtype=float)
        if not np.all(np.isfinite(cov)):
            raise RuntimeError("Problem top-down panel encountered a non-finite planner covariance")
        covs.append(cov)
    return {"truth": truth, "belief": belief, "plan": plan, "covs": np.asarray(covs), "marker_indices": marker_indices}


def _draw_problem_statement_topdown(ax, geom: dict, start: dict, goal: dict, traces: dict[str, np.ndarray]) -> None:
    start_xy = (float(start["x"]), float(start["y"]))
    goal_xy = (float(goal["x"]), float(goal["y"]))
    cam_x, cam_y, *_ = geom["camera_pose"]
    paths = _extract_problem_paths_from_trace(traces, start_xy, goal_xy)
    truth = paths["truth"]
    belief = paths["belief"]
    plan = paths["plan"]
    covs = paths["covs"]
    marker_indices = np.asarray(
        paths.get(
            "marker_indices",
            np.linspace(0, min(len(truth), len(belief)) - 1, min(6, len(truth), len(belief))).astype(int),
        ),
        dtype=int,
    )

    _style_topdown_axis(ax)
    _draw_paper_workspace(ax, geom)
    ax.plot(plan[:, 0], plan[:, 1], color=COLORS["fail"], linewidth=1.25, alpha=0.88, label=r"constant-$R_0$ plan", zorder=7)
    ax.plot(truth[:, 0], truth[:, 1], color="#222222", linewidth=1.55, label="truth path", zorder=8)
    ax.plot(belief[:, 0], belief[:, 1], color=COLORS["belief"], linewidth=1.25, linestyle=(0, (4, 2)), label="belief mean", zorder=8)
    ax.text(-0.62, 1.16, r"constant-$R_0$ plan", fontsize=5.7, color=COLORS["fail"], ha="center", va="bottom")
    ax.text(1.52, -0.45, "truth", fontsize=5.7, color="#222222", ha="left", va="center")
    ax.text(1.05, 0.16, "belief", fontsize=5.7, color=COLORS["belief"], ha="left", va="bottom")

    for k, idx in enumerate(marker_indices):
        idx = int(np.clip(idx, 0, min(len(truth), len(belief)) - 1))
        truth_xy = truth[idx]
        belief_xy = belief[idx]
        ax.scatter([truth_xy[0]], [truth_xy[1]], color="#222222", s=12, zorder=10)
        ax.scatter([belief_xy[0]], [belief_xy[1]], color=COLORS["belief"], s=14, zorder=10)
        ax.plot([truth_xy[0], belief_xy[0]], [truth_xy[1], belief_xy[1]], color="#777777", linewidth=0.55, linestyle=":", zorder=6)
        cov = covs[min(k, len(covs) - 1)]
        _covariance_ellipse(ax, (float(belief_xy[0]), float(belief_xy[1])), cov, color=COLORS["belief"], alpha=0.14, zorder=4)
        ax.text(float(truth_xy[0]) + 0.05, float(truth_xy[1]) + 0.07, rf"$t_{k}$", fontsize=5.8, color="#333333", zorder=12)

    ax.scatter([start_xy[0]], [start_xy[1]], s=32, color=COLORS["start"], edgecolor="black", linewidth=0.5, zorder=11)
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=36, color=COLORS["goal"], edgecolor="black", linewidth=0.5, zorder=11)
    ax.text(start_xy[0] - 0.10, start_xy[1] + 0.20, "start", fontsize=6.2, ha="right", va="bottom")
    ax.text(goal_xy[0] + 0.10, goal_xy[1] + 0.20, "goal", fontsize=6.2, ha="left", va="bottom")
    ax.text(cam_x + 0.14, cam_y - 0.04, "fixed camera", fontsize=5.8, ha="left", va="top")
    ax.text(-0.05, 0.10, "shelf", fontsize=5.8, ha="center", va="bottom")
    ax.annotate(
        "reduced camera\nupdate reliability",
        xy=(0.95, 0.28),
        xytext=(1.72, 1.05),
        fontsize=5.8,
        color="#8a5b0b",
        ha="center",
        arrowprops={"arrowstyle": "->", "linewidth": 0.55, "color": "#8a5b0b"},
    )


def _first_command_stamp(traces: dict[str, np.ndarray]) -> float | None:
    if "first_cmd_stamp" in traces and traces["first_cmd_stamp"].size:
        value = float(traces["first_cmd_stamp"][0])
        if math.isfinite(value):
            return value
    if "all_plan_stamp" in traces and traces["all_plan_stamp"].size:
        finite = traces["all_plan_stamp"][np.isfinite(traces["all_plan_stamp"])]
        if finite.size:
            return float(np.nanmin(finite))
    if "stamp" in traces and traces["stamp"].size:
        finite = traces["stamp"][np.isfinite(traces["stamp"])]
        if finite.size:
            return float(np.nanmin(finite))
    return None


def _nearest_experiment_index(traces: dict[str, np.ndarray], target_stamp: float) -> int | None:
    if "stamp" not in traces:
        return None
    stamps = np.asarray(traces["stamp"], dtype=float)
    if not stamps.size:
        return None
    required = ("truth_x", "truth_y", "planner_belief_x", "planner_belief_y")
    valid = np.isfinite(stamps)
    for key in required:
        if key not in traces:
            return None
        valid &= np.isfinite(np.asarray(traces[key], dtype=float))
    if not np.any(valid):
        return None
    valid_indices = np.flatnonzero(valid)
    nearest = valid_indices[int(np.nanargmin(np.abs(stamps[valid] - float(target_stamp))))]
    return int(nearest)


def _plan_at_stamp(traces: dict[str, np.ndarray], target_stamp: float) -> tuple[np.ndarray, float] | None:
    required = ("all_plan_stamp", "all_plan_point_idx", "all_plan_x", "all_plan_y")
    if not all(key in traces for key in required):
        if "plan_x" in traces and "plan_y" in traces:
            x, y = _finite_xy(np.asarray(traces["plan_x"], dtype=float), np.asarray(traces["plan_y"], dtype=float))
            return (np.column_stack([x, y]), float("nan")) if x.size >= 2 else None
        return None
    stamps = np.asarray(traces["all_plan_stamp"], dtype=float)
    finite_stamps = np.unique(stamps[np.isfinite(stamps)])
    if not finite_stamps.size:
        return None
    plan_stamp = float(finite_stamps[int(np.nanargmin(np.abs(finite_stamps - float(target_stamp))))])
    mask = np.isclose(stamps, plan_stamp, atol=1e-9)
    order = np.argsort(np.asarray(traces["all_plan_point_idx"], dtype=float)[mask])
    x = np.asarray(traces["all_plan_x"], dtype=float)[mask][order]
    y = np.asarray(traces["all_plan_y"], dtype=float)[mask][order]
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return None
    return np.column_stack([x[finite], y[finite]]), plan_stamp


def _snapshot_from_trace(
    traces: dict[str, np.ndarray],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    rel_time_s: float,
    *,
    late: bool,
) -> dict:
    del start_xy, goal_xy
    first_cmd = _first_command_stamp(traces)
    if first_cmd is None:
        raise RuntimeError("Cannot determine first command stamp from trace")
    target_stamp = float(first_cmd) + float(rel_time_s)
    idx = _nearest_experiment_index(traces, target_stamp)
    if idx is None:
        raise RuntimeError(f"No valid experiment row near requested snapshot time {rel_time_s:.3f}s")

    stamps = np.asarray(traces["stamp"], dtype=float)
    current_stamp = float(stamps[idx])
    history_mask = (
        np.isfinite(stamps)
        & (stamps >= float(first_cmd) - 1e-6)
        & (stamps <= current_stamp + 1e-6)
        & np.isfinite(np.asarray(traces["truth_x"], dtype=float))
        & np.isfinite(np.asarray(traces["truth_y"], dtype=float))
        & np.isfinite(np.asarray(traces["planner_belief_x"], dtype=float))
        & np.isfinite(np.asarray(traces["planner_belief_y"], dtype=float))
    )
    truth_history = np.column_stack([
        np.asarray(traces["truth_x"], dtype=float)[history_mask],
        np.asarray(traces["truth_y"], dtype=float)[history_mask],
    ])
    belief_history = np.column_stack([
        np.asarray(traces["planner_belief_x"], dtype=float)[history_mask],
        np.asarray(traces["planner_belief_y"], dtype=float)[history_mask],
    ])
    if truth_history.shape[0] < 1 or belief_history.shape[0] < 1:
        raise RuntimeError(f"No valid truth/belief history at snapshot time {rel_time_s:.3f}s")
    plan_result = _plan_at_stamp(traces, current_stamp)
    if plan_result is None:
        raise RuntimeError(f"No planned horizon found near stamp {current_stamp:.3f}")
    plan, plan_stamp = plan_result

    cov_xy = 0.0
    if "planner_cov_xy" in traces and idx < len(traces["planner_cov_xy"]):
        value = float(np.asarray(traces["planner_cov_xy"], dtype=float)[idx])
        cov_xy = value if math.isfinite(value) else 0.0
    cov = np.array(
        [
            [float(np.asarray(traces["planner_cov_x"], dtype=float)[idx]), cov_xy],
            [cov_xy, float(np.asarray(traces["planner_cov_y"], dtype=float)[idx])],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(cov)):
        raise RuntimeError(f"Invalid planner covariance at experiment row {idx}")

    return {
        "truth_history": truth_history,
        "belief_history": belief_history,
        "truth_xy": truth_history[-1],
        "belief_xy": belief_history[-1],
        "plan": plan,
        "cov": cov,
        "time_label": rf"$t={max(0.0, current_stamp - first_cmd):.1f}\,\mathrm{{s}}$",
        "has_run": True,
        "requested_rel_time_s": float(rel_time_s),
        "first_cmd_stamp": float(first_cmd),
        "target_stamp": float(target_stamp),
        "current_stamp": float(current_stamp),
        "experiment_row": int(idx),
        "plan_stamp": float(plan_stamp),
        "plan_points": int(plan.shape[0]),
    }


def _draw_snapshot_topdown(
    ax,
    geom: dict,
    start: dict,
    goal: dict,
    traces: dict[str, np.ndarray],
    *,
    rel_time_s: float,
    panel: str,
    title_suffix: str,
    annotation: str,
) -> None:
    start_xy = (float(start["x"]), float(start["y"]))
    goal_xy = (float(goal["x"]), float(goal["y"]))
    cam_x, cam_y, *_ = geom["camera_pose"]
    snap = _snapshot_from_trace(
        traces,
        start_xy,
        goal_xy,
        rel_time_s,
        late=annotation != "early",
    )

    _style_topdown_axis(ax)
    _draw_paper_workspace(ax, geom)
    ax.set_title(f"{panel} {title_suffix}\n{snap['time_label']}", fontsize=10)

    plan = np.asarray(snap["plan"], dtype=float)
    if plan.ndim == 2 and plan.shape[0] >= 2:
        ax.plot(plan[:, 0], plan[:, 1], color=COLORS["fail"], linewidth=1.4, alpha=0.85, zorder=7)
    truth = np.asarray(snap["truth_history"], dtype=float)
    belief = np.asarray(snap["belief_history"], dtype=float)
    ax.plot(truth[:, 0], truth[:, 1], color="#222222", linewidth=1.7, zorder=8)
    ax.plot(belief[:, 0], belief[:, 1], color=COLORS["belief"], linewidth=1.4, linestyle=(0, (4, 2)), zorder=8)
    ax.scatter([snap["truth_xy"][0]], [snap["truth_xy"][1]], color="#222222", s=34, zorder=10)
    ax.scatter([snap["belief_xy"][0]], [snap["belief_xy"][1]], color=COLORS["belief"], s=38, zorder=10)
    ax.plot(
        [snap["truth_xy"][0], snap["belief_xy"][0]],
        [snap["truth_xy"][1], snap["belief_xy"][1]],
        color="#777777",
        linewidth=0.7,
        linestyle=":",
        zorder=6,
    )
    # Realized posterior covariance is small; visualize at 3sigma so the
    # cross-condition difference reads at this figure size. Legend marks it
    # explicitly as "covariance illustration".
    _covariance_ellipse(
        ax,
        (float(snap["belief_xy"][0]), float(snap["belief_xy"][1])),
        np.asarray(snap["cov"], dtype=float),
        color=COLORS["belief"],
        alpha=0.35,
        zorder=4,
        sigma=3.0,
    )
    ax.scatter([start_xy[0]], [start_xy[1]], s=46, color=COLORS["start"], edgecolor="black", linewidth=0.6, zorder=11)
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=52, color=COLORS["goal"], edgecolor="black", linewidth=0.6, zorder=11)
    ax.text(start_xy[0] - 0.15, start_xy[1] + 0.22, "start", fontsize=8.5, ha="right", va="bottom")
    ax.text(goal_xy[0] + 0.15, goal_xy[1] + 0.22, "goal", fontsize=8.5, ha="left", va="bottom")
    ax.text(cam_x + 0.18, cam_y - 0.05, "fixed camera", fontsize=8, ha="left", va="top")
    ax.text(-0.05, 0.12, "shelf", fontsize=8, ha="center", va="bottom")
    if annotation == "stale":
        ax.text(1.5, 1.85, "reduced\nupdates", fontsize=8.5, ha="center", va="center", color="#8a5b0b")
        bx = float(snap["belief_xy"][0]); by = float(snap["belief_xy"][1])
        # Anchor "predicted uncertainty" label to upper-left of belief, away from shelf and goal
        lx = bx - 0.85
        ly = by + 0.95
        ax.annotate(
            "predicted\nuncertainty",
            xy=(bx, by),
            xytext=(lx, ly),
            fontsize=8.5,
            color=COLORS["belief"],
            ha="center",
            va="bottom",
            arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": COLORS["belief"]},
        )
    elif annotation == "recovered":
        ax.text(1.5, 1.85, "updates\nreturn", fontsize=8.5, ha="center", va="center", color="#8a5b0b")


def _snapshot_panel_spec(index: int, count: int) -> tuple[str, str, str]:
    panel = f"({chr(ord('b') + index)})"
    if index == 0:
        return panel, r"initial constant-$R_0$ rollout", "early"
    if count >= 3 and index == count - 1:
        return panel, "after fresh corrections return", "recovered"
    if count >= 3:
        return panel, "stale camera-update interval", "stale"
    return panel, "near reduced camera-update reliability", "stale"


def _problem_legend_handles() -> list:
    return [
        Line2D([0], [0], color="#222222", linewidth=1.7, label="truth path"),
        Line2D([0], [0], color=COLORS["belief"], linewidth=1.4, linestyle=(0, (4, 2)), label="belief mean"),
        Line2D([0], [0], color=COLORS["fail"], linewidth=1.4, alpha=0.85, label="current horizon"),
        Ellipse((0, 0), 0.18, 0.10, facecolor=COLORS["belief"], edgecolor=COLORS["belief"], alpha=0.35, label=r"3$\sigma$ posterior covariance"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["start"], markeredgecolor="black", markersize=7, label="start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["goal"], markeredgecolor="black", markersize=7, label="goal"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["low_rel"], edgecolor="none", alpha=0.25, label="reduced camera-update reliability"),
    ]


def _style_topdown_axis(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.15, 3.15)
    ax.set_ylim(-3.15, 3.15)
    ax.set_xlabel("position x [m]", fontsize=9)
    ax.set_ylabel("position y [m]", fontsize=9)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.grid(True, color=COLORS["grid"], linewidth=0.45, zorder=-1)
    ax.tick_params(labelsize=8, length=2)


def _draw_markers(ax, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> None:
    ax.scatter([start_xy[0]], [start_xy[1]], s=34, color=COLORS["start"], edgecolor="black", linewidth=0.5, zorder=10)
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=36, color=COLORS["goal"], edgecolor="black", linewidth=0.5, zorder=10)


def _draw_weak_camera_rays(ax, cam_xy: tuple[float, float], target_xy: tuple[float, float]) -> None:
    cx, cy = cam_xy
    tx, ty = target_xy
    for offset, alpha in ((-0.08, 0.45), (0.0, 0.55), (0.08, 0.45)):
        ax.plot(
            [cx, tx + offset],
            [cy, ty - offset],
            color=COLORS["weak_ray"],
            linewidth=0.8,
            linestyle=(0, (2, 3)),
            alpha=alpha,
            zorder=3,
        )


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _draw_trace_or_route(ax, traces: dict[str, np.ndarray], start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> None:
    if "truth_x" in traces and "truth_y" in traces:
        tx, ty = _finite_xy(traces["truth_x"], traces["truth_y"])
        if len(tx) > 1:
            ax.plot(tx, ty, color=COLORS["truth"], linewidth=1.7, marker="o", markersize=2.0, markevery=max(1, len(tx) // 12), zorder=8)
    else:
        ax.add_patch(_bezier([start_xy, (-1.55, -0.7), (-0.15, -2.15), goal_xy], color=COLORS["truth"], linewidth=1.8, zorder=8))

    if "planner_belief_x" in traces and "planner_belief_y" in traces:
        bx, by = _finite_xy(traces["planner_belief_x"], traces["planner_belief_y"])
        if len(bx) > 1:
            ax.plot(bx, by, color=COLORS["belief"], linewidth=1.4, marker="o", markersize=2.0, markevery=max(1, len(bx) // 12), zorder=7)
            step = max(1, len(bx) // 9)
            cov_x = traces.get("planner_cov_x")
            cov_y = traces.get("planner_cov_y")
            for idx in range(0, len(bx), step):
                if cov_x is not None and cov_y is not None and idx < len(cov_x) and idx < len(cov_y):
                    sx = 2.0 * math.sqrt(max(float(cov_x[idx]), 0.0))
                    sy = 2.0 * math.sqrt(max(float(cov_y[idx]), 0.0))
                else:
                    sx = sy = 0.35
                ax.add_patch(
                    Ellipse((bx[idx], by[idx]), max(sx, 0.12), max(sy, 0.12), facecolor=COLORS["belief"], edgecolor=COLORS["belief"], alpha=0.12, zorder=3)
                )


def _draw_problem_routes(ax, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> None:
    """Draw only the paper problem statement: the fixed-noise failure mode."""
    # The red/orange route is intentionally not a successful trace. It is the
    # failure mode we want the problem-setting figure to communicate.
    ax.add_patch(
        _bezier(
            [start_xy, (-1.15, 1.08), (0.95, 0.52), (1.24, 0.30)],
            color=COLORS["fail"],
            linewidth=1.8,
            zorder=8,
        )
    )
    ax.scatter([1.24], [0.30], marker="x", s=55, color=COLORS["fail"], linewidth=1.5, zorder=11)
    ax.text(1.34, 0.26, "belief degrades\nnear shelf", fontsize=5.8, ha="left", va="top", color=COLORS["fail"])

    # Covariance/reliability rings in the low-reliability region, matching the
    # style of the reference figure without calling it a reward.
    for p, r in [((-1.15, 0.85), 0.62), ((-0.25, 0.82), 0.88), ((0.55, 0.52), 0.76)]:
        ax.add_patch(Circle(p, r, facecolor=COLORS["low_rel"], edgecolor=COLORS["short"], linewidth=0.5, alpha=0.17, zorder=2))


def _draw_topdown_panel(
    ax,
    geom: dict,
    start: dict,
    goal: dict,
    traces: dict[str, np.ndarray],
    *,
    visibility: dict[str, np.ndarray] | None = None,
    mode: str = "problem",
) -> None:
    start_xy = (float(start["x"]), float(start["y"]))
    goal_xy = (float(goal["x"]), float(goal["y"]))
    cam_x, cam_y, *_ = geom["camera_pose"]

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.15, 3.15)
    ax.set_ylim(-3.15, 3.15)
    ax.set_xlabel("position x [m]", fontsize=8)
    ax.set_ylabel("position y [m]", fontsize=8)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.grid(True, color=COLORS["grid"], linewidth=0.45, zorder=-1)
    ax.tick_params(labelsize=7, length=2)

    _draw_workspace(ax, geom, visibility)

    if mode == "run":
        ax.add_patch(
            _bezier(
                [start_xy, (-1.25, 0.95), (0.35, 0.92), goal_xy],
                color=COLORS["short"],
                linewidth=1.5,
                linestyle="-",
                zorder=4,
            )
        )
        for p, r in [((-1.05, 0.90), 0.62), ((0.05, 0.88), 0.92), ((0.95, 0.75), 0.72)]:
            ax.add_patch(Circle(p, r, facecolor=COLORS["low_rel"], edgecolor=COLORS["short"], linewidth=0.5, alpha=0.16, zorder=2))
        _draw_trace_or_route(ax, traces, start_xy, goal_xy)
    else:
        _draw_problem_routes(ax, start_xy, goal_xy)

    ax.scatter([start_xy[0]], [start_xy[1]], s=32, color=COLORS["start"], edgecolor="black", linewidth=0.5, zorder=10)
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=34, color=COLORS["goal"], edgecolor="black", linewidth=0.5, zorder=10)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="<", color="none", markerfacecolor=COLORS["camera"], markeredgecolor=COLORS["camera"], markersize=7, label="fixed camera"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["start"], markeredgecolor="black", markersize=6, label="start state"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["goal"], markeredgecolor="black", markersize=6, label="goal state"),
        Line2D([0], [0], color=COLORS["fail"], linewidth=1.8, label="fixed-noise short route"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["shelf"], edgecolor=COLORS["shelf_edge"], label="occluding shelf"),
    ]
    if mode == "run":
        legend_handles.insert(3, Line2D([0], [0], color=COLORS["truth"], linewidth=1.7, marker="o", markersize=3, label="system states"))
        legend_handles.insert(4, Line2D([0], [0], color=COLORS["belief"], linewidth=1.4, marker="o", markersize=3, label="state estimate"))
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=5.2,
        frameon=True,
        framealpha=0.95,
        fancybox=False,
        borderpad=0.25,
        handlelength=1.5,
        labelspacing=0.25,
    )
    ax.text(
        0.75,
        0.76,
        "geometric camera\nline-of-sight shadow",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color="#8a5b0b",
    )
    ax.text(cam_x + 0.15, cam_y - 0.05, "camera", fontsize=6.5, ha="left", va="top")


def _save_single_route_panel(
    out_dir: Path,
    geom: dict,
    start: dict,
    goal: dict,
    traces: dict[str, np.ndarray],
    mode: str,
    visibility: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.25), constrained_layout=True)
    _draw_problem_statement_topdown(ax, geom, start, goal, traces)
    fig.savefig(out_dir / "problem_setting_routes.pdf")
    fig.savefig(out_dir / "problem_setting_routes.png", dpi=300)
    fig.savefig(out_dir / "problem_statement_topdown.pdf")
    fig.savefig(out_dir / "problem_statement_topdown.png", dpi=300)
    plt.close(fig)


def _save_snapshot_panels(
    out_dir: Path,
    geom: dict,
    start: dict,
    goal: dict,
    traces: dict[str, np.ndarray],
    snapshot_times: tuple[float, ...],
) -> None:
    count = len(snapshot_times)
    fig_width = 7.05 if count == 2 else max(7.05, 3.2 * count)
    fig, axes = plt.subplots(1, count, figsize=(fig_width, 3.55), constrained_layout=False)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.26, wspace=0.25)
    for index, rel_time_s in enumerate(snapshot_times):
        panel, title, annotation = _snapshot_panel_spec(index, count)
        _draw_snapshot_topdown(
            axes[index],
            geom,
            start,
            goal,
            traces,
            rel_time_s=float(rel_time_s),
            panel=panel,
            title_suffix=title,
            annotation=annotation,
        )
    fig.legend(
        handles=_problem_legend_handles(),
        loc="lower center",
        ncol=4,
        fontsize=6.0,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.savefig(out_dir / "problem_statement_snapshots.pdf")
    fig.savefig(out_dir / "problem_statement_snapshots.png", dpi=300)
    plt.close(fig)


def _snapshot_provenance(
    traces: dict[str, np.ndarray],
    start: dict,
    goal: dict,
    snapshot_times: tuple[float, ...],
) -> list[dict]:
    start_xy = (float(start["x"]), float(start["y"]))
    goal_xy = (float(goal["x"]), float(goal["y"]))
    rows = []
    count = len(snapshot_times)
    for index, rel_time_s in enumerate(snapshot_times):
        _, _, annotation = _snapshot_panel_spec(index, count)
        snap = _snapshot_from_trace(
            traces,
            start_xy,
            goal_xy,
            rel_time_s,
            late=annotation != "early",
        )
        rows.append(
            {
                "requested_rel_time_s": snap.get("requested_rel_time_s"),
                "first_cmd_stamp": snap.get("first_cmd_stamp"),
                "target_stamp": snap.get("target_stamp"),
                "current_stamp": snap.get("current_stamp"),
                "experiment_row": snap.get("experiment_row"),
                "plan_stamp": snap.get("plan_stamp"),
                "plan_points": snap.get("plan_points"),
                "has_run": bool(snap.get("has_run", False)),
                "truth_xy": [float(v) for v in np.asarray(snap["truth_xy"], dtype=float).reshape(2)],
                "belief_xy": [float(v) for v in np.asarray(snap["belief_xy"], dtype=float).reshape(2)],
                "covariance_xy": np.asarray(snap["cov"], dtype=float).tolist(),
            }
        )
    return rows


def _draw_setup_topdown_panel(ax, geom: dict, start: dict, goal: dict, visibility: dict[str, np.ndarray]) -> None:
    start_xy = (float(start["x"]), float(start["y"]))
    goal_xy = (float(goal["x"]), float(goal["y"]))
    cam_x, cam_y, *_ = geom["camera_pose"]

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.15, 3.15)
    ax.set_ylim(-3.15, 3.15)
    ax.set_xlabel("position x [m]", fontsize=8)
    ax.set_ylabel("position y [m]", fontsize=8)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.grid(True, color=COLORS["grid"], linewidth=0.45, zorder=-1)
    ax.tick_params(labelsize=7, length=2)

    _draw_workspace(ax, geom, visibility)

    ax.scatter([start_xy[0]], [start_xy[1]], s=38, color=COLORS["start"], edgecolor="black", linewidth=0.6, zorder=10)
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=44, color=COLORS["goal"], edgecolor="black", linewidth=0.6, zorder=10)
    ax.text(start_xy[0] - 0.15, start_xy[1] + 0.20, "start", fontsize=7, ha="right", va="bottom")
    ax.text(goal_xy[0] + 0.15, goal_xy[1] + 0.20, "goal", fontsize=7, ha="left", va="bottom")
    ax.text(cam_x + 0.16, cam_y - 0.05, "fixed external\ncamera", fontsize=6.5, ha="left", va="top")
    ax.text(0.20, 0.10, "occluding shelf", fontsize=6.8, ha="center", va="bottom")
    ax.text(
        0.73,
        0.77,
        "geometric camera\nline-of-sight shadow",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color="#8a5b0b",
    )

    legend_handles = [
        Line2D([0], [0], marker="<", color="none", markerfacecolor=COLORS["camera"], markeredgecolor=COLORS["camera"], markersize=7, label="fixed camera"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["start"], markeredgecolor="black", markersize=6, label="start state"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["goal"], markeredgecolor="black", markersize=6, label="goal state"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["shelf"], edgecolor=COLORS["shelf_edge"], label="occluding shelf"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["low_rel"], edgecolor="none", alpha=0.25, label="camera shadow"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=5.4,
        frameon=True,
        framealpha=0.95,
        fancybox=False,
        borderpad=0.25,
        handlelength=1.4,
        labelspacing=0.25,
    )


def _save_setup_topdown_panel(out_dir: Path, geom: dict, start: dict, goal: dict, visibility: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.25), constrained_layout=True)
    _draw_setup_topdown_panel(ax, geom, start, goal, visibility)
    fig.savefig(out_dir / "setup_topdown.pdf")
    fig.savefig(out_dir / "setup_topdown.png", dpi=300)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    start, goal = _load_task(args.task_name)
    geom = _load_world_geometry()
    visibility = _load_visibility_field(args.visibility_artifact_path, geom, mode=args.visibility_field)
    run_dir = _resolve_run_dir(args.run_dir)
    if run_dir is None:
        raise RuntimeError(f"Invalid --run-dir: {args.run_dir}")
    manifest = _load_run_manifest(run_dir)
    planner = str(manifest.get("planner", "") or "")
    use_visibility_model = bool(manifest.get("use_visibility_model", False))
    if (planner != "constant_R_efe" or use_visibility_model) and not args.allow_nonconstant_r0:
        raise RuntimeError(
            "Refusing to label this as a constant-R0 problem figure because the run is not "
            f"constant_R_efe/use_visibility_model=false: planner={planner!r}, "
            f"use_visibility_model={manifest.get('use_visibility_model', None)!r}. "
            "Pass --allow-nonconstant-r0 only for debugging."
        )
    traces = _load_run_trace(run_dir, use_raw_state=args.use_raw_state)
    panel_a_image = args.panel_a_image if args.panel_a_image is not None else args.gazebo_screenshot

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    snapshot_times = tuple(float(value) for value in args.snapshot_times)
    if len(snapshot_times) < 2:
        raise RuntimeError("--snapshot-times needs at least two values")

    n_snap = len(snapshot_times)
    fig_width = 4.4 + 3.6 * n_snap
    fig = plt.figure(figsize=(fig_width, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 1 + n_snap, width_ratios=[1.0] + [1.05] * n_snap)
    ax_a = fig.add_subplot(gs[0, 0])

    _draw_screenshot_panel(ax_a, panel_a_image)
    for index, rel_time_s in enumerate(snapshot_times):
        ax = fig.add_subplot(gs[0, index + 1])
        panel, title, annotation = _snapshot_panel_spec(index, len(snapshot_times))
        _draw_snapshot_topdown(
            ax,
            geom,
            start,
            goal,
            traces,
            rel_time_s=rel_time_s,
            panel=panel,
            title_suffix=title,
            annotation=annotation,
        )
    fig.legend(
        handles=_problem_legend_handles(),
        loc="lower center",
        ncol=7,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )

    fig.savefig(args.out_dir / "problem_setup.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "problem_setup.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _save_snapshot_panels(
        args.out_dir,
        geom,
        start,
        goal,
        traces,
        snapshot_times,
    )
    provenance = {
        "figure": "problem_setup",
        "run_dir": str(run_dir),
        "run_manifest": {
            "planner": manifest.get("planner"),
            "use_visibility_model": manifest.get("use_visibility_model"),
            "task": manifest.get("task"),
            "world": manifest.get("world"),
            "process_noise_xy": manifest.get("process_noise_xy"),
            "process_noise_theta": manifest.get("process_noise_theta"),
            "r_visible_uv": manifest.get("r_visible_uv"),
            "r_miss_uv": manifest.get("r_miss_uv"),
        },
        "snapshot_times_requested_s_after_first_command": [float(v) for v in snapshot_times],
        "snapshots": _snapshot_provenance(
            traces,
            start,
            goal,
            snapshot_times,
        ),
        "panel_a_image": str(panel_a_image) if panel_a_image is not None else None,
        "notes": [
            "The current horizon is loaded from plan_samples.csv at the nearest available plan_stamp.",
            "Truth and belief paths are loaded from experiment.csv up to the selected snapshot row.",
            "The script refuses to generate paper snapshots without real run traces.",
        ],
    }
    (args.out_dir / "problem_setup_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if args.write_auxiliary:
        _save_single_route_panel(args.out_dir, geom, start, goal, traces, "problem", visibility)
        _save_setup_topdown_panel(args.out_dir, geom, start, goal, visibility)
    print(f"Wrote {args.out_dir / 'problem_setup.pdf'}")
    print(f"Wrote {args.out_dir / 'problem_setup.png'}")
    print(f"Wrote {args.out_dir / 'problem_statement_snapshots.pdf'}")
    print(f"Wrote {args.out_dir / 'problem_statement_snapshots.png'}")
    print(f"Wrote {args.out_dir / 'problem_setup_provenance.json'}")
    if args.write_auxiliary:
        print(f"Wrote {args.out_dir / 'problem_setting_routes.pdf'}")
        print(f"Wrote {args.out_dir / 'problem_setting_routes.png'}")
        print(f"Wrote {args.out_dir / 'problem_statement_topdown.pdf'}")
        print(f"Wrote {args.out_dir / 'problem_statement_topdown.png'}")
        print(f"Wrote {args.out_dir / 'setup_topdown.pdf'}")
        print(f"Wrote {args.out_dir / 'setup_topdown.png'}")
    if panel_a_image is None:
        print("Panel (a) is a placeholder. Rerun with --panel-a-image path/to/image.png after taking a screenshot.")
    if run_dir is not None:
        print(f"Panel (b) used trace data from {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
