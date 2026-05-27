#!/usr/bin/env python3
"""Generate paper-support figures for the supervisor feedback pass.

The figures are deliberately diagnostic rather than new experiment evidence.
They help the paper separate:

* known 2D traversability / forbidden zones,
* learned camera-observation reliability,
* planner-facing ambiguity and shared no-go feasibility costs.

Default input is the compact paper-facing Task A artifact. Experiment B can be
inspected with --world warehouse_aws.world.sdf and the matching AWS GP artifact
once that artifact is paper-ready.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORLD_DIR = REPO_ROOT / "src" / "sim" / "gazebo_worlds" / "worlds"
TASKS_PATH = REPO_ROOT / "src" / "experiments" / "config" / "tasks.yaml"
WORLD_PROFILES_PATH = REPO_ROOT / "src" / "experiments" / "config" / "world_profiles.yaml"
DEFAULT_GP = REPO_ROOT / "logs" / "visibility_comparison" / "current_gp" / "yolo_score_raw_gp.npz"
DEFAULT_OUT_DIR = REPO_ROOT / "logs" / "visibility_comparison" / "supervisor_feedback_figures"

sys.path.insert(0, str(REPO_ROOT / "src" / "unav_common"))
from unav_common.occlusion_geometry import (  # noqa: E402
    OcclusionScene,
    parse_collision_scene_from_world,
    signed_distance_to_union_xy,
)


GREEN = "#158a46"
RED = "#d84a4a"
RACK = "#666666"
WALL = "#b7b7b7"
CAMERA = "#111111"
START = "#1976d2"
GOAL = "#1b9e5a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="warehouse_occ_light.world.sdf")
    parser.add_argument("--task", default=None, help="Task name from tasks.yaml. Defaults to the world profile recommendation.")
    parser.add_argument("--gp-artifact", type=Path, default=DEFAULT_GP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--r-visible-uv", type=float, default=2.5)
    parser.add_argument("--r-miss-uv", type=float, default=120.0)
    parser.add_argument("--constant-r-uv", type=float, default=2.5)
    parser.add_argument("--ambiguity-weight", type=float, default=3.0)
    parser.add_argument("--ambiguity-term-scale", type=float, default=1.0)
    parser.add_argument("--risk-weight", type=float, default=1.25)
    parser.add_argument("--nogo-weight", type=float, default=40.0)
    parser.add_argument("--nogo-safe-distance", type=float, default=0.35)
    parser.add_argument("--nogo-softplus-scale", type=float, default=0.08)
    parser.add_argument("--min-prob", type=float, default=1.0e-4)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_profile(world: str) -> dict[str, Any]:
    payload = load_yaml(WORLD_PROFILES_PATH)
    return dict((payload.get("worlds") or {}).get(world) or {})


def load_task(world: str, task_name: str | None) -> tuple[str, dict[str, Any]]:
    profile = load_profile(world)
    selected = task_name or profile.get("recommended_task")
    payload = load_yaml(TASKS_PATH)
    tasks = list((payload.get("tasks") or {}).get(world) or [])
    if selected is None and tasks:
        selected = str(tasks[0].get("name"))
    for task in tasks:
        if task.get("name") == selected:
            return str(selected), task
    raise RuntimeError(f"Task {selected!r} not found for world {world!r} in {TASKS_PATH}")


def load_gp(path: Path) -> dict[str, np.ndarray | float]:
    path = resolve_path(path)
    if not path.is_file():
        raise RuntimeError(f"GP artifact not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = ["xs", "ys", "P_conservative_plan_map", "X_train", "p_train"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise RuntimeError(f"GP artifact missing required keys: {missing}")
        xs = np.asarray(data["xs"], dtype=float)
        ys = np.asarray(data["ys"], dtype=float)
        p_plan = np.asarray(data["P_conservative_plan_map"], dtype=float)
        if p_plan.shape != (ys.size, xs.size):
            if p_plan.T.shape == (ys.size, xs.size):
                p_plan = p_plan.T
            else:
                raise RuntimeError(
                    f"P_conservative_plan_map shape {p_plan.shape} does not match ys/xs {(ys.size, xs.size)}"
                )
        out: dict[str, np.ndarray | float] = {
            "xs": xs,
            "ys": ys,
            "P_plan": p_plan,
            "X_train": np.asarray(data["X_train"], dtype=float),
            "p_train": np.asarray(data["p_train"], dtype=float),
        }
        if "P_mean_map" in data.files:
            out["P_mean"] = _as_map(data["P_mean_map"], xs, ys)
        else:
            out["P_mean"] = p_plan
        if "F_std_map" in data.files:
            out["F_std"] = _as_map(data["F_std_map"], xs, ys)
        else:
            out["F_std"] = np.full_like(p_plan, np.nan, dtype=float)
        if "camera_pos" in data.files:
            out["camera_pos"] = np.asarray(data["camera_pos"], dtype=float).reshape(-1)
        else:
            out["camera_pos"] = np.array([np.nan, np.nan, np.nan], dtype=float)
        if "beta" in data.files:
            out["beta"] = float(np.asarray(data["beta"], dtype=float).reshape(-1)[0])
        else:
            out["beta"] = math.nan
    return out


def _as_map(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    field = np.asarray(arr, dtype=float)
    if field.shape == (ys.size, xs.size):
        return field
    if field.T.shape == (ys.size, xs.size):
        return field.T
    raise RuntimeError(f"Map shape {field.shape} does not match ys/xs {(ys.size, xs.size)}")


def world_path(world: str) -> Path:
    path = WORLD_DIR / world
    if not path.is_file():
        raise RuntimeError(f"World file not found: {path}")
    return path


def grid_from_gp(gp: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(gp["xs"], dtype=float)
    ys = np.asarray(gp["ys"], dtype=float)
    x_grid, y_grid = np.meshgrid(xs, ys)
    pts = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    return xs, ys, x_grid, pts


def effective_plan_variance(
    p_plan: np.ndarray,
    *,
    r_visible_uv: float,
    r_miss_uv: float,
    min_prob: float,
) -> np.ndarray:
    p_eff = np.clip(np.asarray(p_plan, dtype=float), min_prob, 1.0 - min_prob)
    visible_var = float(r_visible_uv) ** 2
    miss_var = float(r_miss_uv) ** 2
    precision = p_eff / visible_var + (1.0 - p_eff) / miss_var
    return 1.0 / np.maximum(precision, 1.0e-12)


def ambiguity_from_variance(plan_var: np.ndarray, *, weight: float, scale: float) -> np.ndarray:
    # R is diagonal with equal x/y variances in the current planner-facing model,
    # so 0.5 log|R| = log(var). Keep the formula explicit for paper traceability.
    return float(weight) * float(scale) * 0.5 * np.log(np.clip(plan_var * plan_var, 1.0e-12, None))


def no_go_field(
    scene: OcclusionScene,
    pts: np.ndarray,
    shape: tuple[int, int],
    *,
    weight: float,
    safe_distance: float,
    softplus_scale: float,
) -> np.ndarray:
    if not scene.prisms or weight <= 0.0:
        return np.zeros(shape, dtype=float)
    signed_d = signed_distance_to_union_xy(scene.prisms, pts)
    clearance = signed_d - float(safe_distance)
    z = np.clip(-clearance / max(float(softplus_scale), 1.0e-9), -60.0, 60.0)
    return (float(weight) * np.log1p(np.exp(z))).reshape(shape)


def robust01(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def draw_collision_prisms(ax: plt.Axes, scene: OcclusionScene, *, alpha: float = 0.72, zorder: int = 8) -> None:
    for prism in scene.prisms:
        name = prism.name.lower()
        face = WALL if "wall" in name else RACK
        edge = "#555555" if "wall" in name else "#222222"
        ax.add_patch(
            Rectangle(
                (prism.xmin, prism.ymin),
                prism.xmax - prism.xmin,
                prism.ymax - prism.ymin,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.55,
                alpha=alpha,
                zorder=zorder,
            )
        )


def draw_known_regions(ax: plt.Axes, profile: dict[str, Any]) -> bool:
    regions = list(profile.get("known_2d_regions") or [])
    if not regions:
        return False
    for region in regions:
        xmin = float(region["xmin"])
        xmax = float(region["xmax"])
        ymin = float(region["ymin"])
        ymax = float(region["ymax"])
        rtype = str(region.get("type", "")).lower()
        if "traversable" in rtype:
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    facecolor="none",
                    edgecolor=GREEN,
                    linewidth=1.2,
                    zorder=9,
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    facecolor="#ffd7d7",
                    edgecolor=RED,
                    hatch="///",
                    linewidth=0.8,
                    alpha=0.85,
                    zorder=7,
                )
            )
    return True


def draw_task_and_camera(
    ax: plt.Axes,
    task: dict[str, Any],
    gp: dict[str, np.ndarray | float],
    *,
    labels: bool = True,
) -> None:
    start = task["start"]
    goal = task["goal"]
    sx, sy = float(start["x"]), float(start["y"])
    gx, gy = float(goal["x"]), float(goal["y"])
    ax.scatter([sx], [sy], s=42, facecolor=START, edgecolor="black", linewidth=0.6, zorder=20)
    ax.scatter([gx], [gy], s=48, facecolor=GOAL, edgecolor="black", linewidth=0.6, zorder=20)
    if labels:
        ax.text(sx + 0.05, sy + 0.05, "start", fontsize=7, color=START, zorder=21)
        ax.text(gx + 0.05, gy + 0.05, "goal", fontsize=7, color=GOAL, zorder=21)
    cam = np.asarray(gp.get("camera_pos", np.array([np.nan, np.nan, np.nan])), dtype=float)
    if cam.size >= 2 and np.all(np.isfinite(cam[:2])):
        ax.scatter([cam[0]], [cam[1]], marker="v", s=70, facecolor=CAMERA, edgecolor="white", linewidth=0.7, zorder=20)
        if labels:
            ax.text(cam[0] + 0.05, cam[1] + 0.05, "camera", fontsize=7, color=CAMERA, zorder=21)


def format_axes(ax: plt.Axes, gp: dict[str, np.ndarray | float]) -> None:
    xs = np.asarray(gp["xs"], dtype=float)
    ys = np.asarray(gp["ys"], dtype=float)
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(float(ys[0]), float(ys[-1]))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(color="#dddddd", linewidth=0.35, alpha=0.5)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        written.append(path)
    plt.close(fig)
    return written


def plot_traversability_vs_observability(
    *,
    gp: dict[str, np.ndarray | float],
    scene: OcclusionScene,
    profile: dict[str, Any],
    task: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    extent = _extent(gp)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.set_title("(a) known 2D forbidden-zone / traversability layer")
    has_regions = draw_known_regions(ax, profile)
    draw_collision_prisms(ax, scene, alpha=0.78)
    draw_task_and_camera(ax, task, gp)
    format_axes(ax, gp)
    if not has_regions:
        ax.text(
            0.02,
            0.03,
            "green regions not annotated for this compact world;\n"
            "gray footprints are the known no-go geometry",
            transform=ax.transAxes,
            fontsize=7,
            color="#444444",
            va="bottom",
        )

    ax = axes[1]
    ax.set_title("(b) learned planner-facing observation reliability")
    im = ax.imshow(
        np.asarray(gp["P_plan"], dtype=float),
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.65, float(np.nanpercentile(np.asarray(gp["P_plan"], dtype=float), 99))),
        aspect="equal",
    )
    draw_collision_prisms(ax, scene, alpha=0.58)
    draw_known_regions(ax, profile)
    draw_task_and_camera(ax, task, gp)
    format_axes(ax, gp)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="rho_plan")

    handles = [
        Patch(facecolor=WALL, edgecolor="#555555", label="known non-driveable geometry"),
        Line2D([0], [0], color=GREEN, linewidth=1.4, label="known driveable boundary"),
        Patch(facecolor="#ffd7d7", edgecolor=RED, hatch="///", label="non-driveable staging pad"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=CAMERA, markeredgecolor="white", markersize=7, label="external camera"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=7, frameon=True)
    return save_figure(fig, out_dir, "traversability_vs_observability")


def plot_gp_uncertainty_coverage(
    *,
    gp: dict[str, np.ndarray | float],
    scene: OcclusionScene,
    task: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    extent = _extent(gp)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.25), constrained_layout=True)

    ax = axes[0]
    ax.set_title("(a) GP training samples")
    sc = ax.scatter(
        np.asarray(gp["X_train"])[:, 0],
        np.asarray(gp["X_train"])[:, 1],
        c=np.asarray(gp["p_train"]),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=18,
        edgecolor="none",
        zorder=12,
    )
    draw_collision_prisms(ax, scene, alpha=0.42)
    draw_task_and_camera(ax, task, gp, labels=False)
    format_axes(ax, gp)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="sample score")

    ax = axes[1]
    ax.set_title("(b) GP posterior uncertainty")
    im = ax.imshow(np.asarray(gp["F_std"]), extent=extent, origin="lower", cmap="magma", aspect="equal")
    draw_collision_prisms(ax, scene, alpha=0.42, zorder=9)
    draw_task_and_camera(ax, task, gp, labels=False)
    format_axes(ax, gp)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="posterior std")

    ax = axes[2]
    beta = gp.get("beta", math.nan)
    beta_text = f"beta={beta:.2f}" if isinstance(beta, float) and math.isfinite(beta) else "beta unknown"
    ax.set_title(f"(c) conservative reliability ({beta_text})")
    im = ax.imshow(np.asarray(gp["P_plan"]), extent=extent, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    draw_collision_prisms(ax, scene, alpha=0.42, zorder=9)
    draw_task_and_camera(ax, task, gp, labels=False)
    format_axes(ax, gp)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="rho_plan")

    fig.suptitle("Sample coverage matters: unsampled regions are GP extrapolation, not measured visibility", fontsize=10)
    return save_figure(fig, out_dir, "gp_uncertainty_coverage")


def plot_ambiguity_constant_vs_learned(
    *,
    gp: dict[str, np.ndarray | float],
    scene: OcclusionScene,
    task: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[Path], np.ndarray]:
    extent = _extent(gp)
    p_plan = np.asarray(gp["P_plan"], dtype=float)
    learned_var = effective_plan_variance(
        p_plan,
        r_visible_uv=args.r_visible_uv,
        r_miss_uv=args.r_miss_uv,
        min_prob=args.min_prob,
    )
    learned_amb = ambiguity_from_variance(
        learned_var,
        weight=args.ambiguity_weight,
        scale=args.ambiguity_term_scale,
    )
    constant_var = np.full_like(learned_var, float(args.constant_r_uv) ** 2)
    constant_amb = ambiguity_from_variance(
        constant_var,
        weight=args.ambiguity_weight,
        scale=args.ambiguity_term_scale,
    )

    vmin = min(float(np.nanmin(constant_amb)), float(np.nanmin(learned_amb)))
    vmax = max(float(np.nanmax(constant_amb)), float(np.nanmax(learned_amb)))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    for ax, title, field in (
        (axes[0], "(a) constant covariance ambiguity (spatially uniform)", constant_amb),
        (axes[1], "(b) learned covariance ambiguity", learned_amb),
    ):
        ax.set_title(title)
        im = ax.imshow(field, extent=extent, origin="lower", cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
        draw_collision_prisms(ax, scene, alpha=0.46)
        draw_task_and_camera(ax, task, gp)
        format_axes(ax, gp)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02, label="lambda_amb * 0.5 log|R|")
    fig.suptitle("C1 uses a constant observation covariance; C2 turns reliability into spatial ambiguity", fontsize=10)
    written = save_figure(fig, out_dir, "ambiguity_constant_vs_learned")
    return written, learned_amb


def plot_planner_cost_fields(
    *,
    gp: dict[str, np.ndarray | float],
    scene: OcclusionScene,
    task: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
    learned_amb: np.ndarray,
) -> list[Path]:
    xs, ys, x_grid, pts = grid_from_gp(gp)
    y_grid = np.meshgrid(xs, ys)[1]
    shape = (ys.size, xs.size)
    goal = task["goal"]
    gx, gy = float(goal["x"]), float(goal["y"])
    risk = float(args.risk_weight) * ((x_grid - gx) ** 2 + (y_grid - gy) ** 2)
    nogo = no_go_field(
        scene,
        pts,
        shape,
        weight=args.nogo_weight,
        safe_distance=args.nogo_safe_distance,
        softplus_scale=args.nogo_softplus_scale,
    )
    risk_n = robust01(risk)
    amb_n = robust01(learned_amb)
    nogo_n = robust01(nogo)
    total_n = risk_n + amb_n + nogo_n

    extent = _extent(gp)
    panels = [
        ("(a) J_risk diagnostic\n(goal-distance proxy)", risk_n, "Blues"),
        ("(b) J_amb from learned R(x,y)", amb_n, "magma"),
        ("(c) shared no-go / forbidden-zone cost", nogo_n, "Greys"),
        ("(d) normalized diagnostic total", total_n, "inferno"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.6), constrained_layout=True)
    for ax, (title, field, cmap) in zip(axes.ravel(), panels):
        ax.set_title(title)
        im = ax.imshow(field, extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=max(1.0, float(np.nanmax(field))), aspect="equal")
        draw_collision_prisms(ax, scene, alpha=0.44)
        draw_task_and_camera(ax, task, gp, labels=False)
        format_axes(ax, gp)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        "Planner-cost explanation field: risk, ambiguity, shared forbidden-zone cost, and total\n"
        "Fields are robust-normalized for visualization; use logged runs for final numeric claims.",
        fontsize=10,
    )
    return save_figure(fig, out_dir, "planner_cost_fields")


def _extent(gp: dict[str, np.ndarray | float]) -> tuple[float, float, float, float]:
    xs = np.asarray(gp["xs"], dtype=float)
    ys = np.asarray(gp["ys"], dtype=float)
    return float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])


def main() -> int:
    args = parse_args()
    gp = load_gp(args.gp_artifact)
    profile = load_profile(args.world)
    task_name, task = load_task(args.world, args.task)
    scene = parse_collision_scene_from_world(
        str(world_path(args.world)),
        model_names=("warehouse_walls", "warehouse_rack_occluders"),
    )

    out_dir = resolve_path(args.out_dir)
    written: list[Path] = []
    written.extend(
        plot_traversability_vs_observability(
            gp=gp,
            scene=scene,
            profile=profile,
            task=task,
            out_dir=out_dir,
        )
    )
    written.extend(
        plot_gp_uncertainty_coverage(
            gp=gp,
            scene=scene,
            task=task,
            out_dir=out_dir,
        )
    )
    ambiguity_written, learned_amb = plot_ambiguity_constant_vs_learned(
        gp=gp,
        scene=scene,
        task=task,
        out_dir=out_dir,
        args=args,
    )
    written.extend(ambiguity_written)
    written.extend(
        plot_planner_cost_fields(
            gp=gp,
            scene=scene,
            task=task,
            out_dir=out_dir,
            args=args,
            learned_amb=learned_amb,
        )
    )

    print(f"Generated supervisor-feedback figures for world={args.world}, task={task_name}:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

