#!/usr/bin/env python3
"""fig35 — paper-1 lane-graph seeds and the solves they produce.

One panel per hard route. Each shows BOTH candidate seeds the published
`generate_route_seeds` returns from the drivable region map (below / above the
rack block), which one the EFE solver selected, and the trajectory it settled
on. This is the route-choice mechanism the collision-map A* fork destroyed:
a single shortest path offers nothing to choose between.

OFFLINE PLANNER SOLVES ONLY. No Gazebo, no perception, no measurement data.
Colour encodes the SEED IDENTITY (below vs above); the solve is drawn on top in
the colour of the seed it chose, so "which route won" is readable at a glance.

Usage:
  python3 scripts/visibility_comparison/render_multicam_seed_and_solve.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multicam_seed_solve_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(Path(__file__))
sys.path.insert(0, str(REPO / "src" / "unav_common"))
from unav_common.occlusion_geometry import (  # noqa: E402
    scene_from_json,
    signed_distance_to_union_xy,
)

DEFAULT_IN = REPO / "logs/studies/multicam_nav_demo/offline_hard_routes"
DEFAULT_OUT = REPO / "logs/studies/multicam_nav_demo/figures/fig35_seed_and_solve.png"

# Okabe-Ito; validated CVD-safe (worst adjacent dE 22.0 protan, 31.7 normal).
SEED_COLOR = {"below_main_aisle": "#0072b2", "above_connector": "#d95f02"}
SEED_LABEL = {"below_main_aisle": "seed: below_main_aisle", "above_connector": "seed: above_connector"}
INK, MUTED = "#222222", "#6b6b6b"
ROBOT_R = 0.125
TASKS = ["rob_hardA", "rob_hardB"]


def _draw_world(ax, collision_geometry_json: str) -> None:
    for prism in scene_from_json(collision_geometry_json).prisms:
        if "wall_" in str(prism.name):
            continue
        ax.add_patch(Rectangle(
            (prism.xmin, prism.ymin), prism.xmax - prism.xmin, prism.ymax - prism.ymin,
            facecolor="#d6d6d6", edgecolor="#b0b0b0", linewidth=0.6, zorder=1,
        ))


def _true_clearance(prisms, pts) -> float:
    sd = np.asarray(signed_distance_to_union_xy(prisms, pts, keep_in=False), dtype=float)
    return float(sd.min() - ROBOT_R)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = json.loads((args.input_dir / "offline_multicam_hard_route_gate.json").read_text("utf-8"))
    results = payload["results"]
    data = np.load(args.input_dir / "offline_multicam_hard_route_trajectories.npz", allow_pickle=False)
    prisms = tuple(scene_from_json(str(data["collision_geometry_json"])).prisms)

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.6))
    for ax, task in zip(axes, TASKS):
        fixed = next(r for r in results if r["task"] == task and r["setting"] == "offline_fix")
        chosen = fixed["selected_source"].split(":")[-1]
        ends = data[f"{task}__endpoints"]
        start, goal = ends[0], ends[1]

        _draw_world(ax, str(data["collision_geometry_json"]))

        # every candidate seed the published generator returned
        cands = sorted(k.split("__seedcand__")[1] for k in data.files
                       if k.startswith(f"{task}__seedcand__"))
        for name in cands:
            pts = np.asarray(data[f"{task}__seedcand__{name}"], dtype=float)
            picked = (name == chosen)
            ax.plot(pts[:, 0], pts[:, 1],
                    color=SEED_COLOR.get(name, MUTED),
                    linewidth=3.4 if picked else 1.9,
                    alpha=0.95 if picked else 0.42,
                    linestyle="-" if picked else (0, (6, 4)),
                    marker="o", markersize=5.5 if picked else 4.0,
                    markerfacecolor="white", zorder=5 if picked else 4)

        solve = np.asarray(data[f"{task}__offline_fix"], dtype=float)[:, :2]
        ax.plot(solve[:, 0], solve[:, 1], color=SEED_COLOR.get(chosen, INK),
                linewidth=2.0, linestyle=(0, (1, 1.6)), zorder=7)

        ax.scatter(*start, s=95, color="#009e73", edgecolor="white", linewidth=1.6, zorder=9)
        ax.scatter(*goal, s=250, marker="*", color="#cc79a7", edgecolor="white", linewidth=1.2, zorder=9)
        ax.annotate("start", xy=start, xytext=(9, -16), textcoords="offset points",
                    fontsize=9.5, color=MUTED, zorder=10)
        ax.annotate("goal", xy=goal, xytext=(10, 8), textcoords="offset points",
                    fontsize=10.5, color="#cc79a7", fontweight="bold", zorder=10)

        clr = _true_clearance(prisms, solve)
        ax.set_title(
            f"{task} — solver chose {chosen}\n"
            f"goal gap {fixed['terminal_goal_distance_m']:.3f} m  ·  "
            f"clearance {clr:+.3f} m  ·  {len(cands)} candidate seeds",
            fontsize=11.5, color=INK)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal"); ax.grid(alpha=0.18)
        ax.set_xlim(-11.4, 11.7); ax.set_ylim(-8.8, 9.2)

    handles = [
        plt.Line2D([], [], color=SEED_COLOR["below_main_aisle"], lw=3.4, marker="o",
                   markerfacecolor="white", label="below_main_aisle"),
        plt.Line2D([], [], color=SEED_COLOR["above_connector"], lw=3.4, marker="o",
                   markerfacecolor="white", label="above_connector"),
        plt.Line2D([], [], color=MUTED, lw=1.9, linestyle=(0, (6, 4)), marker="o",
                   markerfacecolor="white", label="candidate not selected (faded)"),
        plt.Line2D([], [], color=INK, lw=2.0, linestyle=(0, (1, 1.6)),
                   label="EFE solve (in the chosen seed's colour)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle(
        "fig35 — Paper-1 lane-graph seeds and the solves they produce\n"
        "two candidates per route from the drivable region map; the EFE objective picks one",
        fontsize=13.5, color=INK, y=0.98)
    fig.text(0.5, 0.012,
             "OFFLINE PLANNER SOLVES — no Gazebo, no perception, no measurements.   "
             "keep_in · H=120 · goal prior 24→6 px (metric-matched to paper 1)",
             fontsize=9.5, color="#a33", ha="center", fontweight="bold")
    fig.tight_layout(rect=(0, 0.125, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
