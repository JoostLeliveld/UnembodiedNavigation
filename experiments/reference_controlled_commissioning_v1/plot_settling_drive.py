#!/usr/bin/env python3
"""Draw the settling drive: the route, and the heading coverage it produces per position.

Left panel is the drive itself, one lap per colour.  Right panel answers the question the
availability model needs: at how many distinct headings is each square metre of the
workspace seen?
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "experiments/warehouse_v2_sketches")]

import warehouse_v2 as warehouse  # noqa: E402

SURFACE = "#f4f4f2"
LAP_COLORS = ("#2b5d8a", "#c2571a", "#2d7d55")
LAP_NAMES = ("lap 1, forward", "lap 2, reversed", "lap 3, shifted sawtooth")
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


def sample(waypoints, step: float = 0.15):
    poses = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end - start))
        if distance < 1e-9:
            continue
        heading = np.degrees(np.arctan2(*(end - start)[::-1])) % 360.0
        for fraction in np.linspace(0.0, 1.0, max(2, int(distance / step))):
            point = start + (end - start) * fraction
            poses.append((point[0], point[1], heading))
    return np.asarray(poses)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path, default=HERE / "settling_drive.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(arguments.drive.read_text(encoding="utf-8"))["drive"]
    waypoints = np.asarray(payload["waypoints"], dtype=float)
    bounds = payload["lap_boundaries_waypoint_index"]

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 7.4))

    draw_warehouse(axes[0])
    # Lap 2 retraces lap 1 in reverse, so drawn on the same line it would be hidden.
    # Offset each lap slightly perpendicular to its own legs; the offset is a drawing
    # device and is stated in the caption.
    for lap, shift in zip(range(3), (0.0, 0.22, -0.22)):
        section = waypoints[bounds[lap]:bounds[lap + 1] + 1].copy()
        if shift:
            drawn = section.copy()
            for index in range(len(section)):
                before = section[max(index - 1, 0)]
                after = section[min(index + 1, len(section) - 1)]
                delta = after - before
                norm = float(np.linalg.norm(delta))
                if norm > 1e-9:
                    drawn[index] = section[index] + shift * np.array(
                        [-delta[1], delta[0]]) / norm
            section = drawn
        axes[0].plot(section[:, 0], section[:, 1], color=LAP_COLORS[lap],
                     linewidth=1.7, alpha=0.9, solid_capstyle="round",
                     zorder=5 - lap * 0.1)
    axes[0].scatter(*waypoints[0], s=95, marker="o", facecolor="#13a10e",
                    edgecolor="white", linewidth=1.3, zorder=11)
    axes[0].text(waypoints[0][0] + 0.1, waypoints[0][1] - 1.1, "start / end",
                 fontsize=8, fontweight="bold", color="#0d6b0a", ha="center",
                 va="center", zorder=12,
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                           edgecolor="#13a10e", linewidth=0.6, alpha=0.93))
    for name, x in AISLES.items():
        axes[0].text(x, 9.6, name, fontsize=7.6, fontweight="bold", color="#3d4448",
                     ha="center", va="center", zorder=12,
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                               edgecolor="#8d9490", linewidth=0.5, alpha=0.92))
    axes[0].set_title(
        f"One continuous drive, three laps of the same loop\n"
        f"{payload['length_m']:.0f} m, {payload['duration_minutes_at_1mps']:.0f} min at "
        f"{payload['deployment_speed_mps']:.1f} m/s, never closer than "
        f"{payload['min_obstacle_clearance_m']:.2f} m to a rack",
        fontsize=10.5, fontweight="bold", color="#26343c")
    axes[0].legend(handles=[Line2D([0], [0], color=LAP_COLORS[i], linewidth=2.0,
                                   label=LAP_NAMES[i]) for i in range(3)]
                   + [Line2D([0], [0], marker="s", color="none",
                             markerfacecolor="#26343c", markeredgecolor="white",
                             markersize=6, label="fixed camera, arrow shows aim")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
                   fontsize=8, frameon=False, handlelength=1.8)

    poses = sample(waypoints)
    cells: dict[tuple[int, int], set] = {}
    for x, y, heading in poses:
        cells.setdefault((int(np.floor(x)), int(np.floor(y))), set()).add(int(heading // 45))

    draw_warehouse(axes[1])
    counts = np.asarray([len(value) for value in cells.values()])
    colours = plt.get_cmap("YlGnBu")(np.clip(counts, 1, 5) / 5.0)
    for (cell, colour) in zip(cells, colours):
        axes[1].add_patch(Rectangle(cell, 1.0, 1.0, facecolor=colour,
                                    edgecolor="white", linewidth=0.3, zorder=3))
    share = float(np.mean(counts >= 2))
    axes[1].set_title(
        "Heading coverage: distinct headings seen in each square metre\n"
        f"{len(cells)} cells visited, {100 * share:.0f} percent seen from two or more "
        "directions\n(one heading only means availability there cannot vary with heading)",
        fontsize=10.5, fontweight="bold", color="#26343c")
    axes[1].legend(handles=[
        Rectangle((0, 0), 1, 1, facecolor=plt.get_cmap("YlGnBu")(value / 5.0),
                  edgecolor="white",
                  label=f"{value} heading" + ("" if value == 1 else "s")
                        + (" or more" if value == 5 else ""))
        for value in (1, 2, 3, 4, 5)
    ], loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, fontsize=8,
        frameon=False, handlelength=1.2, columnspacing=1.0)

    figure.suptitle(
        "The settling drive. Three laps of one loop, driven without stopping, so the "
        "reference pose is tracked unbroken and every camera opportunity is logged in "
        "order.\nLap 2 reverses the loop and lap 3 shifts the sawtooth, so most of the "
        "workspace is seen from several directions: that heading variation is what makes "
        "availability\na probability rather than a yes or no, and what separates a "
        "heading-dependent error from a position-dependent one.\nThe three laps are drawn "
        "a few centimetres apart so each stays visible; they are driven on the same line.",
        fontsize=9.5, color="#3d4448", y=0.015)
    figure.tight_layout(rect=(0, 0.135, 1, 1))
    target = arguments.output / "settling_drive.pdf"
    figure.savefig(target, bbox_inches="tight")
    figure.savefig(target.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
