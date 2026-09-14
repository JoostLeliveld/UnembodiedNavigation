#!/usr/bin/env python3
"""Draw the proposed commissioning tours: one continuous fit drive and one held-out drive.

Waypoints come from proposed_commissioning_routes.json.  The warehouse layout, the racks
and the camera positions are the real ones, so a route drawn clear of an obstacle here is
clear in the world.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/commissioning_drives_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "experiments/warehouse_v2_sketches")]

import warehouse_v2 as warehouse  # noqa: E402

SURFACE = "#f4f4f2"
FIT_CMAP = "viridis"
HELD_CMAP = "plasma"
AISLES = {"A1": -10.80, "A2": -6.90, "A3": -3.05, "A4": 0.82, "A5": 10.57}


def draw_warehouse(ax) -> None:
    layout = warehouse.build()
    ax.add_patch(Rectangle((-12, -10), 24, 20, facecolor=SURFACE,
                           edgecolor="#62696d", linewidth=1.0, zorder=0))
    for zone in layout.zones:
        ax.add_patch(Rectangle(
            (zone.xmin, zone.ymin), zone.xmax - zone.xmin, zone.ymax - zone.ymin,
            facecolor="#dfe1df", edgecolor="#b9bdba", linewidth=0.45, zorder=1))
    for camera in layout.cameras:
        angle = np.deg2rad(camera.yaw_deg)
        ax.scatter(camera.x, camera.y, marker="s", s=26, color="#26343c",
                   edgecolor="white", linewidth=0.55, zorder=9)
        ax.annotate("", xy=(camera.x + 1.25 * np.cos(angle),
                            camera.y + 1.25 * np.sin(angle)),
                    xytext=(camera.x, camera.y),
                    arrowprops=dict(arrowstyle="-|>", color="#26343c", lw=0.9,
                                    shrinkA=3, shrinkB=0), zorder=8)
        ax.text(camera.x, camera.y + 0.62, camera.name.replace("camera_", ""),
                fontsize=8, fontweight="bold", color="#26343c",
                ha="center", va="center", zorder=10)
    ax.set(xlim=(-12.4, 12.4), ylim=(-10.4, 10.4), aspect="equal",
           xlabel="world $x$ [m]", ylabel="world $y$ [m]")
    ax.tick_params(labelsize=8, length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def densify(waypoints: np.ndarray, step: float = 0.05) -> np.ndarray:
    points = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(2, int(distance / step))
        for fraction in np.linspace(0.0, 1.0, count, endpoint=False):
            points.append(start + (end - start) * fraction)
    points.append(waypoints[-1])
    return np.asarray(points)


def draw_tour(ax, waypoints: np.ndarray, cmap: str) -> None:
    """Colour the tour by distance travelled so the order of the drive is readable."""
    path = densify(waypoints)
    segments = np.stack([path[:-1], path[1:]], axis=1)
    travelled = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    ax.add_collection(LineCollection(segments, cmap=cmap, linewidth=2.6,
                                     array=travelled[:-1], zorder=5,
                                     capstyle="round", joinstyle="round"))

    for target in np.linspace(travelled[-1] * 0.03, travelled[-1] * 0.97, 24):
        index = min(max(int(np.searchsorted(travelled, target)), 1), len(path) - 2)
        direction = path[index + 1] - path[index - 1]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        direction = 0.34 * direction / norm
        ax.annotate("", xy=path[index] + direction, xytext=path[index],
                    arrowprops=dict(arrowstyle="-|>", color="#1b2a33", lw=0.85,
                                    mutation_scale=8, alpha=0.8), zorder=7)

    ax.scatter(waypoints[1:-1, 0], waypoints[1:-1, 1], s=15, facecolor="white",
               edgecolor="#1b2a33", linewidth=0.7, zorder=10)
    ax.scatter(*waypoints[0], s=95, marker="o", facecolor="#13a10e",
               edgecolor="white", linewidth=1.3, zorder=11)
    ax.text(waypoints[0][0] + 0.1, waypoints[0][1] - 1.1, "start / end",
            fontsize=8, fontweight="bold", color="#0d6b0a", ha="center",
            va="center", zorder=12,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#13a10e", linewidth=0.6, alpha=0.93))


def label_aisles(ax) -> None:
    for name, x in AISLES.items():
        ax.text(x, 9.6, name, fontsize=7.6, fontweight="bold", color="#3d4448",
                ha="center", va="center", zorder=12,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#8d9490", linewidth=0.5, alpha=0.92))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path,
                        default=HERE / "proposed_commissioning_routes.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    routes = json.loads(arguments.routes.read_text(encoding="utf-8"))

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 7.6))
    for ax, key, cmap, title, extra in (
        (axes[0], "fit_tour", FIT_CMAP, "Fit drive",
         "every aisle, one continuous loop"),
        (axes[1], "held_tour", HELD_CMAP, "Held-out drive",
         "the same loop, shifted 25 cm sideways"),
    ):
        entry = routes[key]
        waypoints = np.asarray(entry["waypoints"], dtype=float)
        draw_warehouse(ax)
        draw_tour(ax, waypoints, cmap)
        label_aisles(ax)
        ax.set_title(
            f"{title}: {extra}\n{entry['length_m']:.0f} m, {entry['turns']} turns, "
            f"{entry['diagonal_metres']:.0f} m of it diagonal, "
            f"{entry['heading_bins_15deg']} headings\n"
            f"never closer than {entry['min_obstacle_clearance_m']:.2f} m to a rack",
            fontsize=10.2, fontweight="bold", color="#26343c")
        ax.legend(handles=[
            Line2D([0], [0], color="#3b528b", linewidth=2.6,
                   label="drive; colour runs dark to light with distance"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#13a10e",
                   markeredgecolor="white", markersize=8, label="start and end"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="#1b2a33", markersize=5, label="turn"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#26343c",
                   markeredgecolor="white", markersize=6,
                   label="fixed camera, arrow shows aim"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8,
            frameon=False, handlelength=1.8, columnspacing=1.4)

    figure.suptitle(
        "Proposed commissioning drives. Each partition is ONE continuous drive, so the "
        "reference pose is tracked without a break and every camera opportunity\nalong the "
        "way is logged in order. A1 to A5 mark the five north-south aisles; both drives "
        "enter all five. The sawtooth sections are driven diagonally,\nso the robot is seen "
        "at nine headings rather than the four an axis-aligned drive produces. The held-out "
        "drive repeats the loop 25 cm to the side.",
        fontsize=9.5, color="#3d4448", y=0.012)
    figure.tight_layout(rect=(0, 0.155, 1, 1))
    target = arguments.output / "proposed_commissioning_routes.pdf"
    figure.savefig(target, bbox_inches="tight")
    figure.savefig(target.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
