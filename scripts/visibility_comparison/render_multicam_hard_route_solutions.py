#!/usr/bin/env python3
"""fig34 — the two hard routes get two genuinely different solutions.

Reads the offline hard-route gate artifacts (JSON metrics + solved trajectory
npz) written by ``diag/offline_multicam_hard_route_gate.py`` and renders the
route-level story that the 2x2 gate figure does not show: rob_hardA and
rob_hardB are solved by *different* detours, both collision-valid, neither
using mission waypoints.

OFFLINE PLANNER SOLVES ONLY. No Gazebo, no perception, no measurement data.
Colour encodes the ROUTE (the entity); line style encodes the setting, so the
superseded short-horizon solve stays neutral grey and never competes with the
route identity.

Usage:
  python3 scripts/visibility_comparison/render_multicam_hard_route_solutions.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multicam_hard_route_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(Path(__file__))
sys.path.insert(0, str(REPO / "src" / "unav_common"))
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402

DEFAULT_IN = REPO / "logs/studies/multicam_nav_demo/offline_hard_routes"
DEFAULT_OUT = REPO / "logs/studies/multicam_nav_demo/figures/fig34_hard_route_solutions.png"

# Okabe-Ito; validated CVD-safe (worst adjacent dE 22.0 protan, 31.7 normal).
ROUTE_COLOR = {"rob_hardA": "#0072b2", "rob_hardB": "#d95f02"}
OLD_GREY = "#9a9a9a"
INK = "#222222"
MUTED = "#6b6b6b"
GOAL_RADIUS_M = 0.25


def _draw_world(ax, collision_geometry_json: str) -> None:
    for prism in scene_from_json(collision_geometry_json).prisms:
        if "wall_" in str(prism.name):
            continue
        ax.add_patch(Rectangle(
            (prism.xmin, prism.ymin),
            prism.xmax - prism.xmin,
            prism.ymax - prism.ymin,
            facecolor="#d6d6d6", edgecolor="#b0b0b0", linewidth=0.6, zorder=1,
        ))


def _result(results: list[dict], task: str, setting: str) -> dict:
    return next(r for r in results if r["task"] == task and r["setting"] == setting)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = json.loads(
        (args.input_dir / "offline_multicam_hard_route_gate.json").read_text("utf-8"))
    results = payload["results"]
    data = np.load(args.input_dir / "offline_multicam_hard_route_trajectories.npz",
                   allow_pickle=False)
    collision_json = str(data["collision_geometry_json"])
    tasks = ["rob_hardA", "rob_hardB"]

    fig = plt.figure(figsize=(15.0, 8.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.62, 1.0], height_ratios=[1.0, 1.02],
                            wspace=0.20, hspace=0.34)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_gap = fig.add_subplot(grid[0, 1])
    ax_tab = fig.add_subplot(grid[1, 1])

    # ---- Panel A: both solved routes on one map --------------------------
    _draw_world(ax_map, collision_json)
    for task in tasks:
        colour = ROUTE_COLOR[task]
        ends = data[f"{task}__endpoints"]
        start, goal = ends[0], ends[1]
        old = data[f"{task}__old_horizon"]
        fixed = data[f"{task}__offline_fix"]
        seed = data[f"{task}__seed"]

        ax_map.plot([start[0], goal[0]], [start[1], goal[1]],
                    linestyle=(0, (1, 3)), color=MUTED, linewidth=1.0, zorder=2)
        # Paper-1 lane-graph seed drawn as a pale halo under the solve, so the
        # optimiser's departure from the seed is visible.
        ax_map.plot(seed[:, 0], seed[:, 1], color=colour, linewidth=8.0,
                    alpha=0.22, solid_capstyle="round", zorder=3)
        ax_map.plot(old[:, 0], old[:, 1], color=OLD_GREY, linewidth=1.8,
                    linestyle="-", alpha=0.9, zorder=4)
        ax_map.plot(fixed[:, 0], fixed[:, 1], color=colour, linewidth=3.0,
                    solid_capstyle="round", zorder=6)
        ax_map.scatter(*start, s=90, color=colour, edgecolor="white",
                       linewidth=1.6, zorder=8)
        ax_map.scatter(*goal, s=230, marker="*", color=colour, edgecolor="white",
                       linewidth=1.2, zorder=8)
        ax_map.annotate(f"{task}  goal", xy=goal, xytext=(9, 9),
                        textcoords="offset points", color=colour,
                        fontsize=10.5, fontweight="bold", zorder=9)
        ax_map.annotate("start", xy=start, xytext=(9, -15),
                        textcoords="offset points", color=colour, fontsize=9.5, zorder=9)

    handles = [
        plt.Line2D([], [], color=ROUTE_COLOR["rob_hardA"], lw=3.0, label="rob_hardA solution"),
        plt.Line2D([], [], color=ROUTE_COLOR["rob_hardB"], lw=3.0, label="rob_hardB solution"),
        plt.Line2D([], [], color=OLD_GREY, lw=1.8, label="superseded H=75 solve"),
        plt.Line2D([], [], color="#8c8c8c", lw=8.0, alpha=0.32,
                   label="paper-1 lane-graph seed (drivable map)"),
        plt.Line2D([], [], color=MUTED, lw=1.0, linestyle=(0, (1, 3)),
                   label="straight line start→goal"),
    ]
    ax_map.legend(handles=handles, loc="lower right", fontsize=9.5, framealpha=0.94,
                  borderpad=0.7, labelspacing=0.55)
    ax_map.set_title("Two hard routes, two different solutions\n"
                     "waypoint-free · paper-1 lane-graph seeds from the drivable map",
                     fontsize=12.5, color=INK)
    ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
    ax_map.set_aspect("equal")
    ax_map.set_xlim(-11.4, 11.7); ax_map.set_ylim(-8.8, 9.4)
    ax_map.grid(alpha=0.18)

    # ---- Panel B: goal gap, superseded -> fixed --------------------------
    ax_gap.axvspan(0, GOAL_RADIUS_M, color="#2e7d32", alpha=0.12, zorder=0)
    ax_gap.axvline(GOAL_RADIUS_M, color="#2e7d32", linewidth=1.2,
                   linestyle="--", zorder=1)
    for row, task in enumerate(tasks):
        colour = ROUTE_COLOR[task]
        old_gap = _result(results, task, "old_horizon")["terminal_goal_distance_m"]
        new_gap = _result(results, task, "offline_fix")["terminal_goal_distance_m"]
        ax_gap.plot([new_gap, old_gap], [row, row], color=colour, linewidth=2.4,
                    alpha=0.45, zorder=2, solid_capstyle="round")
        ax_gap.scatter(old_gap, row, s=110, facecolor="white", edgecolor=OLD_GREY,
                       linewidth=2.2, zorder=3)
        ax_gap.scatter(new_gap, row, s=140, color=colour, edgecolor="white",
                       linewidth=1.4, zorder=4)
        ax_gap.annotate(f"{old_gap:.2f} m", xy=(old_gap, row), xytext=(0, 13),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=MUTED)
        ax_gap.annotate(f"{new_gap:.2f} m", xy=(new_gap, row), xytext=(0, -21),
                        textcoords="offset points", ha="center", fontsize=10,
                        color=colour, fontweight="bold")
    ax_gap.annotate(f"goal radius {GOAL_RADIUS_M} m", xy=(GOAL_RADIUS_M, 1.42),
                    xytext=(6, 0), textcoords="offset points", fontsize=9,
                    color="#2e7d32", va="center")
    ax_gap.set_yticks(range(len(tasks)))
    ax_gap.set_yticklabels(tasks, fontsize=10.5)
    ax_gap.set_ylim(-0.6, 1.75)
    ax_gap.set_xlim(-0.35, 6.4)
    ax_gap.set_xlabel("terminal goal gap (m)")
    ax_gap.set_title("Goal gap: superseded ○ → fixed ●",
                     fontsize=11.5, color=INK)
    ax_gap.grid(axis="x", alpha=0.18)
    for side in ("top", "right", "left"):
        ax_gap.spines[side].set_visible(False)

    # ---- Panel C: per-route blocks (a column grid clips at this width)
    ax_tab.axis("off")
    ax_tab.set_xlim(0, 1); ax_tab.set_ylim(0, 1)
    y = 0.98
    for task in tasks:
        fixed = _result(results, task, "offline_fix")
        ax_tab.text(0.0, y, task, fontsize=11, color=ROUTE_COLOR[task],
                    fontweight="bold", va="top")
        ax_tab.text(0.0, y - 0.10,
                    f"goal gap {fixed['terminal_goal_distance_m']:.2f} m   ·   "
                    f"clearance {fixed['min_obstacle_clearance_m']:.3f} m   ·   "
                    f"deviation {fixed['max_straight_line_deviation_m']:.2f} m",
                    fontsize=10, color=INK, va="top")
        ax_tab.text(0.0, y - 0.20,
                    f"path {fixed['planned_path_length_m']:.1f} m   ·   "
                    f"clearance seed {fixed['seed_min_obstacle_clearance_m']:.3f} → "
                    f"solve {fixed['min_obstacle_clearance_m']:.3f} m   ·   "
                    f"chose {fixed['selected_source'].split(':')[-1]}",
                    fontsize=10, color=MUTED, va="top")
        y -= 0.34

    ax_tab.plot([0, 1], [0.28, 0.28], color="#dddddd", linewidth=1.0)
    ax_tab.text(0.0, 0.23,
                "Seeds are paper-1's lane-graph method on the drivable region map —\n"
                "two candidates per route (below / above the rack block), and the\n"
                "solver genuinely chooses between them. Reach is the other constraint:\n"
                "18.0 m at H=75 could not express the 18.4–21.5 m lane route; 28.8 m can.\n\n"
                "Open issue — clearance. Both solves clear obstacles by only 5 cm, and the\n"
                "seed rollout is already tight (0.12 / 0.08 m). Corner rounding in 1.2 m\n"
                "cross aisles, not a seeding fault; well inside the 0.25 m safety margin.",
                fontsize=9.4, color="#7a4a00", va="top", linespacing=1.6)

    fig.text(0.5, 0.012,
             "OFFLINE PLANNER SOLVES — no Gazebo, no perception, no measurements.  "
             "Horizon 120 / optimizer budget 120 (was 75 / 60).",
             fontsize=10, color="#a33", ha="center", fontweight="bold")

    fig.suptitle("fig34 — Waypoint-free hard-route solves in the four-camera warehouse",
                 fontsize=14.5, color=INK, y=0.985)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
