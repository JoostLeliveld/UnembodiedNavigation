#!/usr/bin/env python3
"""Step 0: the setup, and why getting a position out of it is not uniform.

No estimator, no arms, no ground truth. Pure geometry, computed from the real
camera poses in ``warehouse_full_4cam.world.sdf`` through the deployed camera
model, so every number here is what the runtime actually works with.

  left   side elevation of one camera. Equally spaced pixels going down the image
         land on the floor at very UNEQUAL spacing: steeply near the camera, almost
         grazing far away. So one pixel is worth a few mm of floor close by and
         many cm far off. This is the whole reason camera quality is a field and
         not a constant.

  right  the same quantity across the real warehouse, best of the four cameras.
         Where the floor is pale the best available camera is coarse.

Line of sight only -- racks are NOT treated as blocking here, so the true picture
is worse than this. Stated in the caption rather than implied.

Outputs -> logs/studies/offset_state_closed_loop/the_setup/
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
for _rel in ("src/reliability", "src/unav_common"):
    sys.path.insert(0, str(REPO / _rel))

from reliability.projection import camera_model_from_world  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
OUT = REPO / "logs/studies/offset_state_closed_loop/the_setup"

CAMERAS = {
    "A": "external_camera",
    "B": "external_camera_b",
    "C": "external_camera_c",
    "D": "external_camera_d",
}
CAM_XY = {"A": (-6.0, -10.0), "B": (-6.0, 10.0), "C": (6.0, -10.0), "D": (6.0, 10.0)}

INK = "#1A1A1A"
ACCENT = "#0072B2"
WARN = "#D55E00"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555",
    })


def floor_mm_per_pixel(camera, x: float, y: float) -> float:
    """How many mm of floor one pixel covers at this floor point (worst direction)."""
    u, v, visible = camera.world_to_pixel(x, y, 0.0)
    if not visible:
        return np.nan
    centre = camera.pixel_to_world(u, v)
    if centre is None:
        return np.nan
    cols = []
    for du, dv in ((0.5, 0.0), (0.0, 0.5)):
        plus = camera.pixel_to_world(u + du, v + dv)
        minus = camera.pixel_to_world(u - du, v - dv)
        if plus is None or minus is None:
            return np.nan
        cols.append(((plus[0] - minus[0]), (plus[1] - minus[1])))
    jac = np.array(cols, dtype=float).T          # world (x,y) per pixel
    # Worst-case stretch = largest singular value, in mm.
    return float(np.linalg.svd(jac, compute_uv=False)[0]) * 1000.0


def panel_side(ax, camera) -> None:
    """One camera in elevation: equal pixel steps, unequal floor steps."""
    cam_x, cam_y, cam_z = camera.cam_pos
    # March straight down the image centre column and see where each pixel lands.
    us = camera.img_width / 2.0
    rows = np.linspace(camera.img_height * 0.06, camera.img_height * 0.97, 13)
    hits = []
    for v in rows:
        world = camera.pixel_to_world(us, v)
        if world is None:
            continue
        hits.append((float(np.hypot(world[0] - cam_x, world[1] - cam_y)), float(v)))
    ranges = np.array([h[0] for h in hits])

    ax.plot([0, ranges.max() * 1.06], [0, 0], color="#8A8A8A", lw=2.2, zorder=1)
    for r in ranges:
        ax.plot([0, r], [cam_z, 0.0], color=ACCENT, lw=0.9, alpha=0.55, zorder=2)
        ax.plot([r], [0.0], marker="|", ms=11, color=ACCENT, mew=1.6, zorder=3)

    ax.plot([0], [cam_z], marker="s", ms=13, color=INK, zorder=5)
    ax.annotate(f"camera on the wall\n{cam_z:.2f} m up",
                xy=(0, cam_z), xytext=(1.4, cam_z - 0.55),
                fontsize=9.5, color=INK, va="top")

    # Quantify the two ends with the real model.
    near_r, far_r = ranges.min(), ranges.max()
    near_mm = floor_mm_per_pixel(camera, cam_x, cam_y + near_r)
    far_mm = floor_mm_per_pixel(camera, cam_x, cam_y + far_r)

    ax.annotate(f"{near_mm:.0f} mm of floor\nper pixel",
                xy=(near_r, 0), xytext=(near_r + 0.3, 2.15),
                fontsize=10, fontweight="bold", color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))
    ax.annotate(f"{far_mm:.0f} mm of floor\nper pixel",
                xy=(far_r, 0), xytext=(far_r - 5.2, 2.15),
                fontsize=10, fontweight="bold", color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3))

    ax.set_xlim(-0.6, ranges.max() * 1.06)
    ax.set_ylim(-0.9, cam_z + 0.9)
    ax.set_xlabel("distance across the floor, away from the wall (m)")
    ax.set_ylabel("height (m)")
    ax.set_yticks([0, 2, 4, 6])
    ax.set_title("Evenly spaced pixels do not land evenly on the floor\n"
                 f"so one pixel is worth {near_mm:.0f} mm here and "
                 f"{far_mm:.0f} mm at the far end",
                 fontweight="bold", fontsize=10.5)
    ax.text(ranges.max() * 0.52, -0.62, "floor", fontsize=9, color="#8A8A8A")


def panel_map(ax, cameras) -> None:
    """Best-of-four floor resolution across the real warehouse."""
    xs = np.arange(-12.0, 12.01, 0.2)
    ys = np.arange(-10.0, 10.01, 0.2)
    best = np.full((ys.size, xs.size), np.nan)
    for cam in cameras.values():
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                mm = floor_mm_per_pixel(cam, x, y)
                if np.isnan(mm):
                    continue
                if np.isnan(best[j, i]) or mm < best[j, i]:
                    best[j, i] = mm

    ax.set_facecolor("#DCDCDC")
    # Scale to the data, not to a guessed range -- the whole point is the structure.
    lo = float(np.nanpercentile(best, 0.5))
    hi = float(np.nanpercentile(best, 99.5))
    mesh = ax.pcolormesh(xs, ys, best, cmap="viridis_r",
                         norm=LogNorm(vmin=lo, vmax=hi), shading="nearest")
    bar = plt.colorbar(mesh, ax=ax, pad=0.02, extend="both")
    bar.set_label("mm of floor per pixel, best camera available\n(smaller is better)",
                  fontsize=9.5)

    levels = [12.0, 16.0, 20.0, 24.0]
    cs = ax.contour(xs, ys, best, levels=levels, colors="white",
                    linewidths=0.9, alpha=0.75)
    ax.clabel(cs, fmt="%.0f mm", fontsize=8, inline=True)

    worst = float(np.nanmax(best))
    ax.annotate(f"worst-served strip: {worst:.0f} mm per pixel\n"
                "every camera is at its longest range here",
                xy=(0.0, 0.0), xytext=(0.0, 1.5),
                fontsize=9.5, fontweight="bold", color="white",
                va="center", ha="center")
    ax.axhline(0.0, color="white", lw=1.0, ls=(0, (5, 4)), alpha=0.8)
    print(f"  best-of-four floor resolution: {np.nanmin(best):.1f} to {worst:.1f} mm/px")

    for name, (cx, cy) in CAM_XY.items():
        ax.plot([cx], [cy], marker="s", ms=11, color=INK, zorder=6)
        ax.annotate(name, xy=(cx, cy), xytext=(cx, cy + (-1.15 if cy > 0 else 1.15)),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=INK, zorder=7)

    ax.set_xlim(-12, 12)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("The same thing across the real warehouse\n"
                 "grey = no camera has a view at all",
                 fontweight="bold", fontsize=10.5)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    cameras = {k: camera_model_from_world(WORLD, include_name=v)
               for k, v in CAMERAS.items()}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.4),
                                  gridspec_kw={"width_ratios": [1.0, 1.15]})
    panel_side(ax, cameras["A"])
    panel_map(ax2, cameras)

    fig.suptitle("The setup: a camera does not measure position equally well everywhere",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.918,
             "Four fixed cameras 6.1 m up on the north and south walls, tilted down. "
             "Pure geometry from the world file — no detector, no filter, no ground truth. "
             "Racks are not treated as blocking, so real coverage is worse than shown.",
             ha="center", va="top", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.875))

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_s0_the_setup.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}")

    for name, cam in cameras.items():
        cx, cy, _ = cam.cam_pos
        inward = -1.0 if cy > 0 else 1.0        # north cameras look back down the map
        probes = [floor_mm_per_pixel(cam, cx, cy + inward * d) for d in (2.0, 8.0, 14.0)]
        print(f"  camera {name}: 2 m {probes[0]:6.1f} mm/px | "
              f"8 m {probes[1]:6.1f} | 14 m {probes[2]:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
