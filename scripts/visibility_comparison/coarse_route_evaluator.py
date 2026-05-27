#!/usr/bin/env python3
"""Offline coarse route evaluator for visibility-aware route sanity checks.

This is deliberately not an online planner and not a waypoint generator. It
answers a narrower question: given the known 2D driveable floor and a fitted GP
reliability artifact, would a warehouse-scale route score prefer the direct
short path or a longer more observable path?

The graph is generated automatically from `world_profiles.yaml` traversable
rectangles. Non-driveable/staging rectangles are excluded. The visibility term
is a planner-facing ambiguity proxy derived from the same precision-blended
observation covariance used by the local EFE planner.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_PATH = REPO_ROOT / "src" / "experiments" / "config" / "tasks.yaml"
PROFILES_PATH = REPO_ROOT / "src" / "experiments" / "config" / "world_profiles.yaml"
DEFAULT_OUT_ROOT = REPO_ROOT / "logs" / "visibility_comparison" / "coarse_route_eval"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _find_task(world: str, task_name: str) -> dict:
    data = _load_yaml(TASKS_PATH)
    for task in (data.get("tasks") or {}).get(world, []) or []:
        if str(task.get("name", "")) == task_name:
            return dict(task)
    raise SystemExit(f"Task '{task_name}' not found for world '{world}' in {TASKS_PATH}")


def _load_profile(world: str) -> dict:
    data = _load_yaml(PROFILES_PATH)
    profile = dict((data.get("worlds") or {}).get(world) or {})
    if not profile:
        raise SystemExit(f"World profile '{world}' not found in {PROFILES_PATH}")
    return profile


def _rects(profile: dict, want_traversable: bool) -> list[dict]:
    out = []
    for region in profile.get("known_2d_regions", []) or []:
        rtype = str(region.get("type", "")).lower()
        is_trav = "traversable" in rtype
        if is_trav == want_traversable:
            out.append(dict(region))
    return out


def _rect_contains(rect: dict, x: float, y: float, margin: float = 0.0) -> bool:
    return (
        float(rect["xmin"]) + margin <= x <= float(rect["xmax"]) - margin
        and float(rect["ymin"]) + margin <= y <= float(rect["ymax"]) - margin
    )


def _default_gp_path(world: str) -> Path:
    if world == "warehouse_aws.world.sdf":
        return REPO_ROOT / "logs" / "visibility_comparison" / "aws_gp_v5" / "yolo_score_raw_gp.npz"
    return REPO_ROOT / "logs" / "visibility_comparison" / "current_gp" / "yolo_score_raw_gp.npz"


def _load_gp(gp_path: Path):
    with np.load(gp_path, allow_pickle=False) as data:
        xs = np.asarray(data["xs"], dtype=float)
        ys = np.asarray(data["ys"], dtype=float)
        if "P_conservative_plan_map" not in data.files:
            raise SystemExit(f"GP artifact missing P_conservative_plan_map: {gp_path}")
        p_plan = np.asarray(data["P_conservative_plan_map"], dtype=float)
    interp = RegularGridInterpolator(
        (ys, xs),
        p_plan,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return xs, ys, p_plan, interp


def _make_grid(profile: dict, resolution: float, boundary_margin: float):
    defaults = profile.get("visibility_defaults", {}) or {}
    xmin = float(defaults.get("visibility_map_min_x", -5.5))
    xmax = float(defaults.get("visibility_map_max_x", 5.5))
    ymin = float(defaults.get("visibility_map_min_y", -5.0))
    ymax = float(defaults.get("visibility_map_max_y", 5.0))
    xs = np.arange(xmin, xmax + 0.5 * resolution, resolution)
    ys = np.arange(ymin, ymax + 0.5 * resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)

    passable = np.zeros_like(xx, dtype=bool)
    for rect in _rects(profile, want_traversable=True):
        passable |= (
            (xx >= float(rect["xmin"]) + boundary_margin)
            & (xx <= float(rect["xmax"]) - boundary_margin)
            & (yy >= float(rect["ymin"]) + boundary_margin)
            & (yy <= float(rect["ymax"]) - boundary_margin)
        )
    for rect in _rects(profile, want_traversable=False):
        passable &= ~(
            (xx >= float(rect["xmin"]) - boundary_margin)
            & (xx <= float(rect["xmax"]) + boundary_margin)
            & (yy >= float(rect["ymin"]) - boundary_margin)
            & (yy <= float(rect["ymax"]) + boundary_margin)
        )
    return xs, ys, xx, yy, passable


def _snap_to_passable(x: float, y: float, xs: np.ndarray, ys: np.ndarray, passable: np.ndarray):
    ix = int(np.argmin(np.abs(xs - x)))
    iy = int(np.argmin(np.abs(ys - y)))
    if passable[iy, ix]:
        return iy, ix, 0.0

    yy, xx = np.where(passable)
    if yy.size == 0:
        raise SystemExit("No passable grid cells after applying known 2D regions.")
    d2 = (xs[xx] - x) ** 2 + (ys[yy] - y) ** 2
    k = int(np.argmin(d2))
    return int(yy[k]), int(xx[k]), float(math.sqrt(float(d2[k])))


def _ambiguity_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    gp_interp,
    r_visible_uv: float,
    r_miss_uv: float,
):
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    rho = np.asarray(gp_interp(pts), dtype=float).reshape(xx.shape)
    rho = np.where(np.isfinite(rho), rho, 0.5)
    rho = np.clip(rho, 1e-4, 1.0 - 1e-4)

    var_vis = float(r_visible_uv) ** 2
    var_miss = float(r_miss_uv) ** 2
    precision = rho / var_vis + (1.0 - rho) / var_miss
    plan_var = 1.0 / np.maximum(precision, 1e-12)
    denom = max(math.log(var_miss / var_vis), 1e-9)
    amb = np.clip(np.log(plan_var / var_vis) / denom, 0.0, 1.0)
    return rho, amb


def _neighbors(iy: int, ix: int, passable: np.ndarray, resolution: float):
    h, w = passable.shape
    for dy, dx, dist in (
        (-1, 0, resolution), (1, 0, resolution), (0, -1, resolution), (0, 1, resolution),
        (-1, -1, resolution * math.sqrt(2.0)),
        (-1, 1, resolution * math.sqrt(2.0)),
        (1, -1, resolution * math.sqrt(2.0)),
        (1, 1, resolution * math.sqrt(2.0)),
    ):
        ny, nx = iy + dy, ix + dx
        if ny < 0 or ny >= h or nx < 0 or nx >= w or not passable[ny, nx]:
            continue
        if dy != 0 and dx != 0:
            if not (passable[iy, nx] and passable[ny, ix]):
                continue
        yield ny, nx, dist


def _shortest_path(passable: np.ndarray, amb: np.ndarray, start, goal, resolution: float, visibility_weight: float):
    h, w = passable.shape
    dist_grid = np.full((h, w), np.inf, dtype=float)
    prev_y = np.full((h, w), -1, dtype=np.int32)
    prev_x = np.full((h, w), -1, dtype=np.int32)
    sy, sx = start
    gy, gx = goal
    dist_grid[sy, sx] = 0.0
    heap = [(0.0, sy, sx)]

    while heap:
        cost, iy, ix = heapq.heappop(heap)
        if cost != dist_grid[iy, ix]:
            continue
        if (iy, ix) == (gy, gx):
            break
        for ny, nx, step_m in _neighbors(iy, ix, passable, resolution):
            edge_amb = 0.5 * (amb[iy, ix] + amb[ny, nx])
            edge_cost = step_m * (1.0 + visibility_weight * edge_amb)
            new_cost = cost + edge_cost
            if new_cost < dist_grid[ny, nx]:
                dist_grid[ny, nx] = new_cost
                prev_y[ny, nx] = iy
                prev_x[ny, nx] = ix
                heapq.heappush(heap, (new_cost, ny, nx))

    if not math.isfinite(float(dist_grid[gy, gx])):
        raise SystemExit("No route found through the known driveable 2D layer.")

    cells = []
    iy, ix = gy, gx
    while iy >= 0 and ix >= 0:
        cells.append((iy, ix))
        if (iy, ix) == (sy, sx):
            break
        py, px = int(prev_y[iy, ix]), int(prev_x[iy, ix])
        iy, ix = py, px
    cells.reverse()
    return float(dist_grid[gy, gx]), cells


def _route_between(passable, amb_grid, start_idx, goal_idx, resolution, visibility_weight):
    total_cost, cells = _shortest_path(
        passable, amb_grid, start_idx, goal_idx, resolution, visibility_weight
    )
    return total_cost, cells


def _route_via(passable, amb_grid, start_idx, via_idx, goal_idx, resolution, visibility_weight):
    cost_a, cells_a = _route_between(
        passable, amb_grid, start_idx, via_idx, resolution, visibility_weight
    )
    cost_b, cells_b = _route_between(
        passable, amb_grid, via_idx, goal_idx, resolution, visibility_weight
    )
    return cost_a + cost_b, cells_a + cells_b[1:]


def _path_xy(cells: list[tuple[int, int]], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.asarray([(xs[ix], ys[iy]) for iy, ix in cells], dtype=float)


def _segment_lengths(path: np.ndarray) -> np.ndarray:
    if len(path) < 2:
        return np.asarray([], dtype=float)
    return np.linalg.norm(np.diff(path, axis=0), axis=1)


def _meters_in_rect(path: np.ndarray, rect: dict) -> float:
    if len(path) < 2:
        return 0.0
    mids = 0.5 * (path[:-1] + path[1:])
    lens = _segment_lengths(path)
    total = 0.0
    for mid, length in zip(mids, lens):
        if _rect_contains(rect, float(mid[0]), float(mid[1])):
            total += float(length)
    return total


def _summarize_path(
    path: np.ndarray,
    total_cost: float,
    visibility_weight: float,
    rho_grid: np.ndarray,
    amb_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    profile: dict,
    low_rho_threshold: float,
):
    cells_x = np.searchsorted(xs, path[:, 0])
    cells_y = np.searchsorted(ys, path[:, 1])
    cells_x = np.clip(cells_x, 0, len(xs) - 1)
    cells_y = np.clip(cells_y, 0, len(ys) - 1)
    rho_vals = rho_grid[cells_y, cells_x]
    amb_vals = amb_grid[cells_y, cells_x]
    lengths = _segment_lengths(path)
    if len(lengths):
        seg_amb = 0.5 * (amb_vals[:-1] + amb_vals[1:])
        ambiguity_integral = float(np.sum(lengths * seg_amb))
        length_m = float(np.sum(lengths))
    else:
        ambiguity_integral = 0.0
        length_m = 0.0

    region_m = {}
    for rect in _rects(profile, want_traversable=True):
        name = str(rect.get("name", ""))
        if name:
            region_m[name] = _meters_in_rect(path, rect)
    a3_m = region_m.get("rack_aisle_A3", 0.0)
    a4_m = region_m.get("rack_aisle_A4", 0.0)
    if a3_m > a4_m + 0.25:
        route_hint = "A3_detour"
    elif a4_m > a3_m + 0.25:
        route_hint = "A4_direct"
    else:
        route_hint = "mixed_or_cross_aisle"

    return {
        "visibility_weight": float(visibility_weight),
        "total_cost": float(total_cost),
        "length_m": length_m,
        "ambiguity_integral_m": ambiguity_integral,
        "mean_rho": float(np.mean(rho_vals)),
        "min_rho": float(np.min(rho_vals)),
        "frac_low_rho": float(np.mean(rho_vals < low_rho_threshold)),
        "meters_rack_aisle_A3": float(a3_m),
        "meters_rack_aisle_A4": float(a4_m),
        "route_hint": route_hint,
    }


def _parse_weights(raw: str) -> list[float]:
    weights = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            weights.append(float(part))
    return weights or [0.0]


def _find_region(profile: dict, name: str) -> dict:
    for region in profile.get("known_2d_regions", []) or []:
        if str(region.get("name", "")) == name:
            return dict(region)
    raise SystemExit(f"Region '{name}' not found in world profile.")


def _candidate_point_for_region(region: dict, start_xy, goal_xy) -> tuple[float, float]:
    x = 0.5 * (float(region["xmin"]) + float(region["xmax"]))
    y_mid = 0.5 * (float(start_xy[1]) + float(goal_xy[1]))
    y = min(max(y_mid, float(region["ymin"])), float(region["ymax"]))
    return x, y


def _parse_region_names(raw: str) -> list[str]:
    names = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if part:
            names.append(part)
    return names


def _draw_regions(ax, profile: dict):
    for rect in _rects(profile, want_traversable=True):
        x0, x1 = float(rect["xmin"]), float(rect["xmax"])
        y0, y1 = float(rect["ymin"]), float(rect["ymax"])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="none", edgecolor="#0d8a3d",
                               linewidth=1.3, zorder=5))
    for rect in _rects(profile, want_traversable=False):
        x0, x1 = float(rect["xmin"]), float(rect["xmax"])
        y0, y1 = float(rect["ymin"]), float(rect["ymax"])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="#ffd7d7", edgecolor="#c93030",
                               linewidth=0.9, hatch="///", alpha=0.55, zorder=6))


def _plot_routes(
    out_path: Path,
    profile: dict,
    gp_xs: np.ndarray,
    gp_ys: np.ndarray,
    gp_map: np.ndarray,
    start_xy,
    goal_xy,
    summaries: list[dict],
    paths: list[np.ndarray],
):
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 9.5))
    ax.imshow(gp_map, extent=[gp_xs[0], gp_xs[-1], gp_ys[0], gp_ys[-1]],
              origin="lower", cmap="Greys", vmin=0.0, vmax=1.0,
              alpha=0.45, aspect="equal", zorder=1)
    _draw_regions(ax, profile)

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(len(paths), 2)))
    for summary, path, color in zip(summaries, paths, colors):
        label = (
            f"{summary.get('candidate', 'route')} "
            f"w={summary['visibility_weight']:g} "
            f"{summary['route_hint']} "
            f"L={summary['length_m']:.2f}m"
        )
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.5,
                alpha=0.9, label=label, zorder=20)
    ax.scatter([start_xy[0]], [start_xy[1]], s=95, c="#1976d2",
               edgecolor="white", linewidth=0.9, zorder=25, label="start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=95, c="#1b9e55",
               edgecolor="white", linewidth=0.9, zorder=25, label="goal")

    handles = [
        Patch(facecolor="0.65", alpha=0.45, label="GP reliability underlay"),
        Line2D([0], [0], color="#0d8a3d", lw=1.3, label="known driveable boundary"),
        Patch(facecolor="#ffd7d7", edgecolor="#c93030", hatch="///",
              label="known forbidden/staging zone"),
    ]
    route_handles, route_labels = ax.get_legend_handles_labels()
    ax.legend(handles + route_handles, [h.get_label() for h in handles] + route_labels,
              loc="upper left", fontsize=7.5, framealpha=0.94)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "Offline coarse route evaluator\n"
        "Routes are generated from known driveable 2D regions; GP only changes route cost.",
        loc="left",
        fontsize=11,
    )
    ax.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Iterable[dict]):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="warehouse_aws.world.sdf")
    parser.add_argument("--task", default="B1_clean_route_choice")
    parser.add_argument("--start", default="", help="Optional x,y override for diagnostic route scoring")
    parser.add_argument("--goal", default="", help="Optional x,y override for diagnostic route scoring")
    parser.add_argument("--gp", default=None, help="GP .npz artifact. Defaults to current world line.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--boundary-margin-m", type=float, default=0.0)
    parser.add_argument("--weights", default="0,1,2,4,8,16",
                        help="Comma-separated visibility weights. 0 is length-only.")
    parser.add_argument("--via-candidates", default="",
                        help=(
                            "Optional comma-separated known_2d_regions to score as offline "
                            "route candidates, e.g. rack_aisle_A3,rack_aisle_A4. "
                            "These are not published as mission waypoints."
                        ))
    parser.add_argument("--r-visible-uv", type=float, default=2.5)
    parser.add_argument("--r-miss-uv", type=float, default=15.0)
    parser.add_argument("--low-rho-threshold", type=float, default=0.35)
    args = parser.parse_args()

    profile = _load_profile(args.world)
    task = _find_task(args.world, args.task)
    start_xy = (float(task["start"]["x"]), float(task["start"]["y"]))
    goal_xy = (float(task["goal"]["x"]), float(task["goal"]["y"]))
    if args.start:
        parts = [float(v.strip()) for v in str(args.start).split(",")]
        if len(parts) != 2:
            raise SystemExit("--start must be x,y")
        start_xy = (parts[0], parts[1])
    if args.goal:
        parts = [float(v.strip()) for v in str(args.goal).split(",")]
        if len(parts) != 2:
            raise SystemExit("--goal must be x,y")
        goal_xy = (parts[0], parts[1])
    gp_path = Path(args.gp).expanduser() if args.gp else _default_gp_path(args.world)
    if not gp_path.is_file():
        raise SystemExit(f"GP artifact not found: {gp_path}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (
        DEFAULT_OUT_ROOT / f"{Path(args.world).stem}_{args.task}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, xx, yy, passable = _make_grid(profile, args.resolution_m, args.boundary_margin_m)
    start_idx = _snap_to_passable(start_xy[0], start_xy[1], xs, ys, passable)
    goal_idx = _snap_to_passable(goal_xy[0], goal_xy[1], xs, ys, passable)
    gp_xs, gp_ys, gp_map, gp_interp = _load_gp(gp_path)
    rho_grid, amb_grid = _ambiguity_grid(
        xx, yy, gp_interp, args.r_visible_uv, args.r_miss_uv
    )
    amb_grid = np.where(passable, amb_grid, 1.0)

    summaries = []
    paths = []
    candidate_specs = [("unconstrained", None, start_idx[:2])]
    for region_name in _parse_region_names(args.via_candidates):
        region = _find_region(profile, region_name)
        via_xy = _candidate_point_for_region(region, start_xy, goal_xy)
        via_idx = _snap_to_passable(via_xy[0], via_xy[1], xs, ys, passable)
        candidate_specs.append((f"via_{region_name}", region_name, via_idx[:2]))

    for weight in _parse_weights(args.weights):
        for candidate, via_region, via_idx in candidate_specs:
            if via_region is None:
                total_cost, cells = _route_between(
                    passable, amb_grid, start_idx[:2], goal_idx[:2],
                    args.resolution_m, weight
                )
            else:
                total_cost, cells = _route_via(
                    passable, amb_grid, start_idx[:2], via_idx, goal_idx[:2],
                    args.resolution_m, weight
                )
            path = _path_xy(cells, xs, ys)
            summary = _summarize_path(
                path, total_cost, weight, rho_grid, amb_grid, xs, ys,
                profile, args.low_rho_threshold
            )
            summary = {
                "candidate": candidate,
                "via_region": via_region or "",
                **summary,
            }
            summaries.append(summary)
            paths.append(path)

    summary_csv = out_dir / "coarse_route_summary.csv"
    _write_csv(summary_csv, summaries)
    plot_path = out_dir / "coarse_route_overlay.png"
    _plot_routes(plot_path, profile, gp_xs, gp_ys, gp_map, start_xy, goal_xy, summaries, paths)

    meta = {
        "world": args.world,
        "task": args.task,
        "gp_artifact": str(gp_path.resolve()),
        "resolution_m": args.resolution_m,
        "boundary_margin_m": args.boundary_margin_m,
        "r_visible_uv": args.r_visible_uv,
        "r_miss_uv": args.r_miss_uv,
        "via_candidates": _parse_region_names(args.via_candidates),
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "start_snap_distance_m": start_idx[2],
        "goal_snap_distance_m": goal_idx[2],
        "outputs": {
            "summary_csv": str(summary_csv),
            "overlay": str(plot_path),
        },
    }
    (out_dir / "coarse_route_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"World/task: {args.world} / {args.task}")
    print(f"GP: {gp_path}")
    if start_idx[2] > 1e-9 or goal_idx[2] > 1e-9:
        print(
            f"Snapped start by {start_idx[2]:.3f} m, "
            f"goal by {goal_idx[2]:.3f} m to nearest driveable grid cell."
        )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_path}")
    for row in summaries:
        print(
            f"  {row['candidate']:<28} "
            f"w={row['visibility_weight']:>5g}  "
            f"{row['route_hint']:<18}  "
            f"L={row['length_m']:.2f}m  "
            f"amb_int={row['ambiguity_integral_m']:.2f}  "
            f"mean_rho={row['mean_rho']:.3f}  "
            f"low={100.0 * row['frac_low_rho']:.0f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
