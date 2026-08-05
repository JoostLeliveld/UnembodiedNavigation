#!/usr/bin/env python3
"""exp1: sketches of realistic ASYMMETRIC warehouse camera layouts.

Exploratory, not a proposal to change the evaluation world. The current 4-camera
world places cameras with four-fold symmetry at (+-6, +-10), 6.1 m, 52.7 deg
pitch, which is a deliberate control: any spatial structure in achievable
precision is then attributable to per-camera error rather than to geometry. These
sketches ask what a *realistic* network would look like instead, and what it would
do to coverage.

Realism the sketches try to capture, none of which the current world has:

  * Cameras mount where building structure exists -- column grid, wall plates,
    dock lintels -- not at geometrically convenient points.
  * Racking is tall and opaque. Coverage is aisle-shaped, not circular, and a
    camera on one side of a block sees nothing on the other side.
  * The network was installed for security and inventory, and robot localization
    inherits it. Dock doors and high-value zones are watched; the interior of a
    storage aisle is watched by whatever happens to spill over.
  * Budget is uneven on purpose. Traffic concentrates on a main aisle; dead
    corners get nothing.

Three layouts, increasing in how much the network was designed for the robot:

  L1  inherited security network        8 cameras, none aimed down an aisle
  L2  security + robot retrofit        12 cameras, 4 added on the main routes
  L3  designed for robot localization  16 cameras on the column grid, deliberate
                                       overlap at junctions where handover happens

Footprints are computed with the SAME projection the rest of the repo uses
(``unav_common.camera_model.ObliqueCameraModel``), and occlusion is a true
line-of-sight test against rack height -- so these are quantitative sketches that
can be scored, not drawings.

Outputs -> logs/studies/warehouse_layout_sketches/exp1_layout_candidates/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "unav_common"))

from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

OUT = REPO / "logs/studies/warehouse_layout_sketches/exp1_layout_candidates"

# ---------------------------------------------------------------- the building
# 44 x 26 m: a small-but-real distribution centre bay, roughly 2x the current world.
SITE = (-22.0, 22.0, -13.0, 13.0)
GRID_M = 0.25
RACK_HEIGHT_M = 4.5
ROBOT_Z_M = 0.05

#: Storage blocks (x0, x1, y0, y1). Aisles are the gaps between them. The layout is
#: deliberately irregular: two deep blocks north, three shallower south, a cross
#: aisle offset from centre, and an open staging area by the docks.
RACKS = [
    (-19.0, -11.0, 2.5, 11.5),
    (-9.0, -1.0, 2.5, 11.5),
    (1.0, 9.0, 2.5, 11.5),
    (11.0, 19.0, 2.5, 8.0),      # shorter block: cross-traffic to the east wall
    (-19.0, -13.0, -11.0, -2.0),
    (-11.0, -5.0, -11.0, -2.0),
    (-3.0, 3.0, -11.0, -6.0),    # half-depth: staging spills into this bay
    (5.0, 11.0, -11.0, -2.0),
    (13.0, 19.0, -11.0, -2.0),
]
#: Dock doors along the south wall (x ranges); the busiest floor in the building.
DOCK_X = [(-9.0, -5.0), (-3.0, 1.0), (3.0, 7.0)]

CAMERA_Z_M = 6.5
CAMERA_PITCH_DEG = 52.7          # matches the deployed cameras
IMG_W, IMG_H, FOV_H_RAD = 1280, 720, 1.5708

#: (x, y, yaw_deg). Pitch and height are common; yaw is where it looks.
LAYOUTS: dict[str, dict] = {
    "L1_inherited_security": {
        "blurb": "8 cameras installed for security/inventory.\nDocks and perimeter "
                 "watched; aisle interiors are an afterthought.",
        "cameras": [
            (-7.0, -12.5, 90), (-1.0, -12.5, 90), (5.0, -12.5, 90),   # dock lintels
            (-21.0, -12.0, 45), (21.0, -12.0, 135),                    # SW / SE corners
            (-21.0, 12.0, -45), (21.0, 12.0, -135),                    # NW / NE corners
            (0.0, 12.5, -90),                                          # north wall centre
        ],
    },
    "L2_security_plus_retrofit": {
        "blurb": "12 cameras: the security network plus 4 retrofitted\nonto the main "
                 "aisle and the cross-aisle junctions.",
        "cameras": [
            (-7.0, -12.5, 90), (-1.0, -12.5, 90), (5.0, -12.5, 90),
            (-21.0, -12.0, 45), (21.0, -12.0, 135),
            (-21.0, 12.0, -45), (21.0, 12.0, -135), (0.0, 12.5, -90),
            # Retrofit aims cameras UP the storage aisles, not along the cross
            # aisle -- an aisle is only observable from its mouth.
            (0.0, 1.0, 90),                                  # centre north aisle
            (-4.0, -1.0, -90), (4.0, -1.0, -90),             # two busiest south aisles
            (-16.0, 0.0, 0),                                 # west cross-aisle run
        ],
    },
    "L3_designed_for_robot": {
        "blurb": "16 cameras aimed UP the storage aisles from their\nmouths, with "
                 "far-end partners for handover.",
        "cameras": [
            (-7.0, -12.5, 90), (-1.0, -12.5, 90), (5.0, -12.5, 90),
            (-21.0, -12.0, 45), (21.0, -12.0, 135),
            (-21.0, 12.0, -45), (21.0, 12.0, -135), (0.0, 12.5, -90),
            # North aisle mouths, looking up the aisle (racks gap at x = -10, 0, 10)
            (-10.0, 1.0, 90), (0.0, 1.0, 90), (10.0, 1.0, 90),
            # ...and from the far end, so a robot mid-aisle has a handover partner
            (-10.0, 12.5, -90), (10.0, 12.5, -90),
            # South aisle mouths -- note these do NOT line up with the north ones
            # (rack blocks are offset), which is where the asymmetry really bites
            (-12.0, -1.0, -90), (-4.0, -1.0, -90), (4.0, -1.0, -90),
        ],
    },
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


def build_grid():
    x0, x1, y0, y1 = SITE
    xs = np.arange(x0, x1 + GRID_M, GRID_M)
    ys = np.arange(y0, y1 + GRID_M, GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    drivable = np.ones(gx.shape, dtype=bool)
    for rx0, rx1, ry0, ry1 in RACKS:
        drivable &= ~((gx >= rx0) & (gx <= rx1) & (gy >= ry0) & (gy <= ry1))
    return xs, ys, gx, gy, drivable


def camera_from_pose(x: float, y: float, yaw_deg: float) -> ObliqueCameraModel:
    """Same construction as reliability.projection.camera_model_from_world."""

    pitch = math.radians(CAMERA_PITCH_DEG)
    yaw = math.radians(yaw_deg)
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
               -math.sin(pitch))
    scale = -CAMERA_Z_M / forward[2]
    look_at = (x + scale * forward[0], y + scale * forward[1], 0.0)
    return ObliqueCameraModel(cam_pos=(x, y, CAMERA_Z_M), look_at=look_at,
                              img_width=IMG_W, img_height=IMG_H, fov_h_rad=FOV_H_RAD)


def visible_mask(model: ObliqueCameraModel, gx, gy) -> np.ndarray:
    """In-frame and in-front-of-camera, vectorised through the model's own R and K."""

    points = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, ROBOT_Z_M)], axis=1)
    cam = (points - np.asarray(model.cam_pos, float)) @ np.asarray(model.R, float).T
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = cam @ np.asarray(model.K, float).T
        u = pixels[:, 0] / pixels[:, 2]
        v = pixels[:, 1] / pixels[:, 2]
    ok = (depth > 0) & np.isfinite(u) & np.isfinite(v)
    ok &= (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return ok.reshape(gx.shape)


def line_of_sight(model: ObliqueCameraModel, gx, gy) -> np.ndarray:
    """False where a rack blocks the ray from the camera to the floor point.

    The ray is sampled between the camera and the target; at each sample, if the
    horizontal position lies inside a rack footprint and the ray's height is below
    that rack's top, the view is blocked. Racks are opaque and full height, which
    is what tall pallet racking is to a 6.5 m camera.
    """

    cam = np.asarray(model.cam_pos, float)
    targets = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, ROBOT_Z_M)], axis=1)
    clear = np.ones(targets.shape[0], dtype=bool)
    # 0 is the camera, 1 is the target; sample the interior only.
    for t in np.linspace(0.04, 0.96, 24):
        sample = cam[None, :] * (1.0 - t) + targets * t
        below_rack_top = sample[:, 2] < RACK_HEIGHT_M
        for rx0, rx1, ry0, ry1 in RACKS:
            inside = ((sample[:, 0] >= rx0) & (sample[:, 0] <= rx1)
                      & (sample[:, 1] >= ry0) & (sample[:, 1] <= ry1))
            clear &= ~(inside & below_rack_top)
    return clear.reshape(gx.shape)


def evaluate(cameras, gx, gy, drivable):
    counts = np.zeros(gx.shape, dtype=int)
    footprints = []
    for x, y, yaw in cameras:
        model = camera_from_pose(x, y, yaw)
        seen = visible_mask(model, gx, gy) & line_of_sight(model, gx, gy) & drivable
        counts += seen.astype(int)
        footprints.append(seen)
    return counts, footprints


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    xs, ys, gx, gy, drivable = build_grid()
    n_drivable = int(drivable.sum())

    results = {}
    fig, axes = plt.subplots(1, len(LAYOUTS), figsize=(6.4 * len(LAYOUTS), 5.4))
    for ax, (name, spec) in zip(np.atleast_1d(axes), LAYOUTS.items()):
        counts, _ = evaluate(spec["cameras"], gx, gy, drivable)
        shown = np.where(drivable, np.minimum(counts, 4), np.nan)
        mesh = ax.pcolormesh(xs, ys, shown, cmap="YlGnBu", vmin=0, vmax=4, shading="auto")
        for rx0, rx1, ry0, ry1 in RACKS:
            ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                   facecolor="#4a4a4a", edgecolor="#222222", lw=0.6))
        for dx0, dx1 in DOCK_X:
            ax.plot([dx0, dx1], [SITE[2], SITE[2]], lw=5, color="#D55E00",
                    solid_capstyle="butt")
        for x, y, yaw in spec["cameras"]:
            ax.plot([x], [y], marker="o", ms=6, color="#C1121F",
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            ax.plot([x, x + 2.4 * math.cos(math.radians(yaw))],
                    [y, y + 2.4 * math.sin(math.radians(yaw))],
                    lw=1.4, color="#C1121F", zorder=5)
        share = {k: float(np.mean(counts[drivable] == k)) for k in range(4)}
        share["4+"] = float(np.mean(counts[drivable] >= 4))
        results[name] = {
            "cameras": len(spec["cameras"]),
            "drivable_cells": n_drivable,
            "share_by_camera_count": share,
            "unseen_fraction": share[0],
            "single_camera_fraction": share[1],
            "redundant_fraction": float(np.mean(counts[drivable] >= 2)),
            "mean_cameras_per_cell": float(np.mean(counts[drivable])),
        }
        ax.set_title(f"{name.replace('_', ' ')}  ({len(spec['cameras'])} cameras)\n"
                     f"{spec['blurb']}", fontweight="bold", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_xlabel(f"unseen {100 * share[0]:.0f} %   ·   single-camera "
                      f"{100 * share[1]:.0f} %   ·   redundant "
                      f"{100 * results[name]['redundant_fraction']:.0f} %",
                      fontsize=9, fontweight="bold")
    np.atleast_1d(axes)[0].set_ylabel("y [m]")
    cbar = fig.colorbar(mesh, ax=np.atleast_1d(axes).tolist(), shrink=0.8,
                        ticks=range(5))
    cbar.set_label("cameras seeing this point")
    cbar.ax.set_yticklabels(["0", "1", "2", "3", "4+"])
    fig.suptitle("Realistic asymmetric layouts — grey = racking, orange = dock doors, "
                 "red = cameras\n(footprints use the deployed camera model; occlusion is "
                 "true line-of-sight against 4.5 m racking)",
                 fontsize=12.5, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_w1_layout_candidates.{ext}", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "site_m": SITE, "grid_m": GRID_M, "rack_height_m": RACK_HEIGHT_M,
        "camera_z_m": CAMERA_Z_M, "camera_pitch_deg": CAMERA_PITCH_DEG,
        "racks": RACKS, "layouts": results,
        "current_world_reference": {
            "note": "warehouse_full_4cam, from the frozen coverage artifact",
            "cameras": 4, "single_camera_fraction": 0.57, "redundant_fraction": 0.42,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"drivable cells: {n_drivable}  ({GRID_M} m grid over "
          f"{SITE[1] - SITE[0]:.0f} x {SITE[3] - SITE[2]:.0f} m)\n")
    print(f"{'layout':<28}{'cams':>5}{'unseen':>9}{'1 cam':>8}{'2+':>8}{'mean':>7}")
    for name, entry in results.items():
        print(f"{name:<28}{entry['cameras']:>5}"
              f"{100 * entry['unseen_fraction']:>8.1f}%"
              f"{100 * entry['single_camera_fraction']:>7.1f}%"
              f"{100 * entry['redundant_fraction']:>7.1f}%"
              f"{entry['mean_cameras_per_cell']:>7.2f}")
    print("\ncurrent symmetric world, for reference: 4 cams, "
          "single-camera 57%, redundant 42%")
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
