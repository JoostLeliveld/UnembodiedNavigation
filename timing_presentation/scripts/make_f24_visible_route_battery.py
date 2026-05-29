#!/usr/bin/env python3
"""F24: offline visible-to-visible route battery for the hierarchical method.

This is a pre-Gazebo sanity check. It samples ten visible start/goal pairs in
the AWS world and solves the same reduced-time route-level problem for C1 and
C2. Route seeds are generated only from the known 2D lane graph and passed
identically to both conditions.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml


REPO = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
SCRIPT_DIR = REPO / "scripts" / "visibility_comparison"
CONFIG = SCRIPT_DIR / "aws_smoke_config.yaml"
TASK = "B1_apron_a4_to_uppermid_a3"
WORLD = "warehouse_aws.world.sdf"
OUT_DIR = REPO / "timing_presentation" / "figures" / "F24"
OUT_PNG = OUT_DIR / "F24_visible_route_battery.png"
OUT_CSV = OUT_DIR / "F24_visible_route_battery.csv"
OUT_MD = OUT_DIR / "F24_visible_route_battery.md"

sys.path.insert(0, str(REPO / "timing_presentation" / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "src" / "experiments"))
sys.path.insert(0, str(REPO / "src" / "planning"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))

import make_f23_hier_offline as f23  # noqa: E402
from planning.planners.base_planner import extract_waypoints  # noqa: E402


VISIBLE_THRESHOLD = 0.60

# Named lane points in the accepted AWS geometry. Endpoints are checked against
# the GP before running, so this list is not assumed valid a priori.
POINTS = {
    "A4_lower": (3.125, -2.44),
    "A3_lower": (1.075, -2.44),
    "A2_lower": (-0.975, -2.44),
    "A1_lower": (-3.025, -2.44),
    "A3_mid": (1.075, 1.64),
    "A2_mid": (-0.975, 1.64),
    "apron_left": (-4.20, -3.15),
    "apron_mid": (0.0, -3.15),
    "apron_right": (3.20, -3.15),
}

ROUTES = [
    # Harder visible-endpoint tasks: longer diagonal/cross-aisle moves that
    # can expose low-reliability interior regions even though both endpoints
    # remain visible. These replace the earlier mostly lower-lane-only smoke.
    ("R01", "A4_lower", "A3_mid"),
    ("R02", "apron_right", "A2_mid"),
    ("R03", "apron_left", "A4_lower"),
    ("R04", "A1_lower", "A3_mid"),
    ("R05", "A4_lower", "A2_mid"),
    ("R06", "apron_left", "A3_mid"),
    ("R07", "A1_lower", "A4_lower"),
    ("R08", "A2_mid", "apron_right"),
    ("R09", "A3_mid", "apron_left"),
    ("R10", "A4_lower", "apron_left"),
]

# Known lane graph centers. These are topology seeds, not visibility labels.
X_LANES = [-3.025, -0.975, 1.075, 3.125]
Y_LANES = [-3.15, -2.44, 1.64, 4.54]


def _interp_field(xs: np.ndarray, ys: np.ndarray, field: np.ndarray, x: float, y: float) -> float:
    ix = int(np.searchsorted(xs, x) - 1)
    iy = int(np.searchsorted(ys, y) - 1)
    ix = max(0, min(ix, len(xs) - 2))
    iy = max(0, min(iy, len(ys) - 2))
    tx = float((x - xs[ix]) / max(xs[ix + 1] - xs[ix], 1e-9))
    ty = float((y - ys[iy]) / max(ys[iy + 1] - ys[iy], 1e-9))
    return float(
        (1 - tx) * (1 - ty) * field[iy, ix]
        + tx * (1 - ty) * field[iy, ix + 1]
        + (1 - tx) * ty * field[iy + 1, ix]
        + tx * ty * field[iy + 1, ix + 1]
    )


def _pvis(setup, xy: tuple[float, float]) -> float:
    return _interp_field(setup.gp["xs"], setup.gp["ys"], setup.gp["P_plan"], xy[0], xy[1])


def _node(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 3), round(float(y), 3))


def _nearest_lane_node(xy: tuple[float, float]) -> tuple[float, float]:
    x, y = xy
    candidates = [_node(xl, yl) for xl in X_LANES for yl in Y_LANES]
    return min(candidates, key=lambda n: (n[0] - x) ** 2 + (n[1] - y) ** 2)


def _lane_graph() -> dict[tuple[float, float], list[tuple[tuple[float, float], float]]]:
    nodes = [_node(x, y) for x in X_LANES for y in Y_LANES]
    graph = {n: [] for n in nodes}
    for y in Y_LANES:
        row = [_node(x, y) for x in X_LANES]
        for a, b in zip(row[:-1], row[1:]):
            d = abs(b[0] - a[0])
            graph[a].append((b, d))
            graph[b].append((a, d))
    for x in X_LANES:
        col = [_node(x, y) for y in Y_LANES]
        for a, b in zip(col[:-1], col[1:]):
            d = abs(b[1] - a[1])
            graph[a].append((b, d))
            graph[b].append((a, d))
    return graph


def _shortest_path(start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
    graph = _lane_graph()
    q: list[tuple[float, tuple[float, float], list[tuple[float, float]]]] = [(0.0, start, [start])]
    seen: dict[tuple[float, float], float] = {}
    while q:
        dist, node, path = heapq.heappop(q)
        if node in seen and seen[node] <= dist:
            continue
        seen[node] = dist
        if node == goal:
            return path
        for nb, w in graph[node]:
            heapq.heappush(q, (dist + w, nb, path + [nb]))
    return [start, goal]


def _route_json(start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> str:
    s_node = _nearest_lane_node(start_xy)
    g_node = _nearest_lane_node(goal_xy)
    shortest = _shortest_path(s_node, g_node)

    route_specs = [{"name": "lane_shortest", "waypoints": shortest[1:] + [goal_xy]}]
    for y in (-2.44, 1.64, 4.54):
        via1 = _nearest_lane_node((start_xy[0], y))
        via2 = _nearest_lane_node((goal_xy[0], y))
        path = _shortest_path(s_node, via1)[:-1] + _shortest_path(via1, via2)[:-1] + _shortest_path(via2, g_node)
        # de-duplicate while preserving order
        clean: list[tuple[float, float]] = []
        for p in path[1:] + [goal_xy]:
            p = _node(*p)
            if not clean or clean[-1] != p:
                clean.append(p)
        route_specs.append({"name": f"lane_via_y_{y:+.2f}", "waypoints": clean})
    return json.dumps(route_specs)


def _yaw_to(src: tuple[float, float], dst: tuple[float, float]) -> float:
    dx = float(dst[0] - src[0])
    dy = float(dst[1] - src[1])
    if dx * dx + dy * dy < 1e-9:
        return 0.0
    return math.atan2(dy, dx)


def _path_length(states: np.ndarray) -> float:
    xy = np.asarray(states, dtype=float)[:, :2]
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _route_class(states: np.ndarray) -> str:
    xy = np.asarray(states, dtype=float)[:, :2]
    if bool(np.any(xy[:, 1] < -2.0)):
        return "lower_lane"
    if bool(np.any(xy[:, 1] > 4.1)):
        return "upper_lane"
    if bool(np.any((xy[:, 0] < 1.7) & (xy[:, 1] < 0.5))):
        return "early_left"
    return "mid_or_direct"


def _solve(condition: str, route_id: str, start_name: str, goal_name: str, base_setup) -> dict:
    start_xy = POINTS[start_name]
    goal_xy = POINTS[goal_name]
    yaw = _yaw_to(start_xy, goal_xy)
    setup = f23._condition_setup(condition)
    cfg = dict(setup.config)
    cfg.update({
        # Reduced-time route-level solve: preserves 20 s lookahead but reduces
        # variables relative to H80, dt=0.25.
        "global_horizon": 50,
        "horizon": 50,
        "dt": 0.40,
        "optimizer_maxiter": 70,
        "optimizer_maxfun": 420,
        "optimizer_initial_routes_json": _route_json(start_xy, goal_xy),
        "optimizer_multistart_lateral_offsets": "-1.0,1.0",
        "optimizer_multistart_include_direct": False,
    })
    setup = replace(
        setup,
        config=cfg,
        start_xy_yaw=(float(start_xy[0]), float(start_xy[1]), float(yaw)),
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
    )
    planner = f23._planner_from_setup(setup, f23._global_updates(setup.config))
    t0 = time.perf_counter()
    result = planner.plan(np.asarray(setup.start_xy_yaw), setup.S0, np.asarray(setup.goal_xy))
    wall = time.perf_counter() - t0
    wps = extract_waypoints(np.asarray(result.states), spacing_m=float(setup.config["waypoint_spacing_m"]), include_goal=True)
    return {
        "route_id": route_id,
        "condition": condition,
        "planner_kind": setup.planner_kind,
        "start": start_name,
        "goal": goal_name,
        "start_x": start_xy[0],
        "start_y": start_xy[1],
        "goal_x": goal_xy[0],
        "goal_y": goal_xy[1],
        "start_pvis": _pvis(base_setup, start_xy),
        "goal_pvis": _pvis(base_setup, goal_xy),
        "source": str(result.selected_source),
        "route_class": _route_class(np.asarray(result.states)),
        "solve_wall_s": wall,
        "total_cost": float(result.total_cost),
        "risk_cost": float(result.risk_cost),
        "ambiguity_cost": float(result.ambiguity_cost),
        "drive_barrier_cost": float(result.obstacle_cost),
        "terminal_goal_distance_m": float(result.terminal_goal_distance_pred),
        "path_length_m": _path_length(np.asarray(result.states)),
        "rollout_valid": int(bool(result.rollout_valid)),
        "optimizer_success": int(bool(result.optimizer_success)),
        "n_waypoints": len(wps),
        "waypoints_json": json.dumps([[round(float(x), 3), round(float(y), 3)] for x, y in wps]),
        "states": np.asarray(result.states, dtype=float),
    }


def _draw_regions(ax, profile: dict) -> None:
    for r in profile.get("known_2d_regions", []) or []:
        kind = str(r.get("type", ""))
        if kind == "traversable":
            ax.add_patch(Rectangle(
                (r["xmin"], r["ymin"]),
                r["xmax"] - r["xmin"],
                r["ymax"] - r["ymin"],
                facecolor="#cdeccd",
                edgecolor="#43a65b",
                lw=0.5,
                alpha=0.35,
                zorder=0,
            ))
        elif "non_driveable" in kind:
            ax.add_patch(Rectangle(
                (r["xmin"], r["ymin"]),
                r["xmax"] - r["xmin"],
                r["ymax"] - r["ymin"],
                facecolor="#f3b2b2",
                edgecolor="#e24a4a",
                lw=0.5,
                alpha=0.35,
                zorder=1,
            ))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = f23._condition_setup("C2")

    rows: list[dict] = []
    plot_rows: list[dict] = []
    for route_id, start_name, goal_name in ROUTES:
        sp = _pvis(base, POINTS[start_name])
        gp = _pvis(base, POINTS[goal_name])
        if sp < VISIBLE_THRESHOLD or gp < VISIBLE_THRESHOLD:
            raise RuntimeError(
                f"{route_id} endpoint below visible threshold: {start_name}={sp:.3f}, {goal_name}={gp:.3f}"
            )
        for condition in ("C1", "C2"):
            row = _solve(condition, route_id, start_name, goal_name, base)
            plot_rows.append(row)
            flat = {k: v for k, v in row.items() if k != "states"}
            rows.append(flat)
            print(
                f"{route_id} {condition}: {row['route_class']:12s} "
                f"d={row['terminal_goal_distance_m']:.2f} valid={row['rollout_valid']} "
                f"wall={row['solve_wall_s']:.1f}s J={row['total_cost']:.0f} "
                f"src={row['source']}",
                flush=True,
            )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5), constrained_layout=True)
    for ax, (route_id, start_name, goal_name) in zip(axes.ravel(), ROUTES):
        gp = base.gp
        ax.imshow(
            gp["P_plan"],
            extent=[gp["xs"].min(), gp["xs"].max(), gp["ys"].min(), gp["ys"].max()],
            origin="lower",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            alpha=0.62,
            zorder=-3,
        )
        _draw_regions(ax, base.profile)
        for row in [r for r in plot_rows if r["route_id"] == route_id]:
            color = "#2563eb" if row["condition"] == "C1" else "#dc2626"
            label = f"{row['condition']} {row['route_class']} {row['terminal_goal_distance_m']:.2f}m"
            states = row["states"]
            ax.plot(states[:, 0], states[:, 1], color=color, lw=2.0, label=label, zorder=5)
        sx, sy = POINTS[start_name]
        gx, gy = POINTS[goal_name]
        ax.plot(sx, sy, "o", color="#111827", ms=5, zorder=8)
        ax.plot(gx, gy, "*", color="#111827", ms=11, zorder=8)
        ax.set_title(
            f"{route_id}: {start_name}->{goal_name}\n"
            f"P { _pvis(base, POINTS[start_name]):.2f}->{ _pvis(base, POINTS[goal_name]):.2f}",
            fontsize=9,
        )
        ax.set_xlim(-5.35, 5.35)
        ax.set_ylim(-3.65, 4.95)
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(
        "F24 - Visible-to-visible route battery: reduced-time global solve (H50, dt=0.40)",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)

    summary = []
    for route_id, _, _ in ROUTES:
        c1 = next(r for r in rows if r["route_id"] == route_id and r["condition"] == "C1")
        c2 = next(r for r in rows if r["route_id"] == route_id and r["condition"] == "C2")
        summary.append((route_id, c1, c2))

    reached = lambda r: bool(r["rollout_valid"]) and float(r["terminal_goal_distance_m"]) <= 0.30
    md = [
        "# F24 Visible-To-Visible Route Battery",
        "",
        f"- figure: `{OUT_PNG}`",
        f"- csv: `{OUT_CSV}`",
        "",
        "## Contract",
        "",
        "- Offline only; this is not Gazebo evidence.",
        "- Ten harder routes have GP conservative endpoint reliability above the visible threshold.",
        "- Compared with the first F24 pass, short lower-lane-only tasks were replaced by longer diagonal/cross-aisle tasks.",
        f"- visible endpoint threshold: `{VISIBLE_THRESHOLD}`",
        "- C1 and C2 receive the same known driveable layer and same neutral lane-graph route seeds.",
        "- Reduced-time global solve: `H=50`, `dt=0.40`, `maxiter=70`, preserving about 20 s lookahead.",
        "",
        "## Summary",
        "",
        "| route | C1 class | C1 d | C1 t | C2 class | C2 d | C2 t |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route_id, c1, c2 in summary:
        md.append(
            f"| {route_id} | {c1['route_class']} | {float(c1['terminal_goal_distance_m']):.2f} | "
            f"{float(c1['solve_wall_s']):.1f}s | {c2['route_class']} | "
            f"{float(c2['terminal_goal_distance_m']):.2f} | {float(c2['solve_wall_s']):.1f}s |"
        )
    md.extend([
        "",
        "## Aggregate",
        "",
        f"- C1 reached/valid: `{sum(reached(r) for r in rows if r['condition'] == 'C1')}/10`",
        f"- C2 reached/valid: `{sum(reached(r) for r in rows if r['condition'] == 'C2')}/10`",
        f"- C1 median solve time: `{np.median([float(r['solve_wall_s']) for r in rows if r['condition'] == 'C1']):.2f}s`",
        f"- C2 median solve time: `{np.median([float(r['solve_wall_s']) for r in rows if r['condition'] == 'C2']):.2f}s`",
        "",
        "Interpretation should focus on whether the reduced global solve remains stable across visible endpoints.",
        "Route differences are diagnostics only until confirmed by Gazebo runs with command and encoder noise.",
    ])
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
