#!/usr/bin/env python3
"""exp2: the PROPOSED asymmetric warehouse — BEV plan for approval.

Nothing is built or run by this script. It draws the layout, places every object
at its measured AWS RoboMaker footprint, and computes what the proposed camera
network would actually see, so the design can be approved or changed before any
capture/retrain time is spent.

Design intent, and what each choice is for:

  * TWO STORAGE BLOCKS WITH DIFFERENT AISLE TOPOLOGY. West rows run north-south
    (deep 12 m aisles); east rows run east-west (shallower 8.5 m aisles). A single
    camera geometry cannot serve both, which is the point -- it forces per-camera
    heterogeneity to be structural rather than incidental.
  * DOUBLE-HEIGHT STACKS. The AWS shelf mesh is 2.61 m and the cameras sit at
    6.1 m, so a single rack barely occludes anything at range. Stacking to 5.2 m
    on selected rows is what creates genuinely blind floor -- the quantity the
    current world almost entirely lacks (~1 % unseen).
  * OFFSET CENTRAL AISLE. The highway runs between the blocks, not down the
    building centreline, so no reflection symmetry survives.
  * ASYMMETRIC CAMERA NETWORK. Two dock cameras inherited from a security
    install, one corner camera, and three placed for the robot -- one up the west
    aisles, one along the east block, one over the north end. No two cameras see
    equivalent geometry.

Every prop footprint below is the measured native size already used by
``scripts/geometry_visibility/make_warehouse_full.py``; nothing is invented.

Outputs -> logs/studies/warehouse_layout_sketches/exp2_proposed_world/
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
from matplotlib.patches import Rectangle, Circle

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE.parent))

from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

OUT = REPO / "logs/studies/warehouse_layout_sketches/exp2_proposed_world"

# Building kept at the CURRENT footprint on purpose: capture time scales with
# floor area, and the layout sketches showed the interesting structure comes from
# occlusion and aiming, not from a bigger box.
X0, X1, Y0, Y1 = -12.0, 12.0, -10.0, 10.0
SITE = (-11.2, 11.5, -8.6, 8.6)
GRID_M = 0.20
ROBOT_Z_M = 0.05

RACK_W = 0.55            # as built today
H_STD, H_TALL = 2.09, 2.61
H_STACK = 5.20           # double stack — this is what actually blocks

#: (name, x_centre, y_centre, size_x, size_y, height, note)
RACKS = [
    # ---- WEST BLOCK: deep north-south aisles -----------------------------
    ("W_row1", -10.40, 2.00, RACK_W, 12.0, H_STACK, "double-stacked, wall side"),
    ("W_row2", -8.30, 2.00, RACK_W, 12.0, H_STACK, "double-stacked"),
    ("W_row3", -6.20, 2.00, RACK_W, 12.0, H_TALL, "single height"),
    # ---- EAST BLOCK: east-west aisles, rotated 90 deg ---------------------
    ("E_row1", 6.25, -1.00, 8.5, RACK_W, H_TALL, "east block, aisle runs E-W"),
    ("E_row2", 6.25, 1.10, 8.5, RACK_W, H_STACK, "double-stacked"),
    ("E_row3", 6.25, 3.20, 8.5, RACK_W, H_TALL, ""),
    ("E_row4", 6.25, 5.30, 8.5, RACK_W, H_STACK, "double-stacked"),
    # ---- north-west wall-backed row (reachable from one side only) --------
    ("NW_wall_row", -11.45, 4.50, RACK_W, 7.0, H_STD, "backed to west wall"),
]
#: Building column: a small, tall, awkward occluder in open floor.
PILLAR = ("support_pillar", -2.00, -2.00, 0.50, 0.50, 5.20)

#: (name, model, x, y, yaw_deg) with footprints from PROP_SIZES in the generator.
PROP_SIZES = {
    "Bucket_01": (0.42, 0.42), "TrashCanC_01": (0.55, 0.55),
    "ClutteringA_01": (0.90, 0.75), "ClutteringC_01": (1.00, 0.80),
    "ClutteringD_01": (0.90, 0.80), "PalletJackB_01": (1.35, 0.65),
    "DeskC_01": (1.55, 0.85),
}
PROPS = [
    ("palletjack_dock", "PalletJackB_01", -4.60, -7.60, 15),
    ("clutterD_dock", "ClutteringD_01", -1.20, -8.10, 0),
    ("clutterA_dock_w", "ClutteringA_01", -8.40, -8.20, 0),
    ("bucket_dock_e", "Bucket_01", 3.40, -8.30, 0),
    ("trashcan_nw", "TrashCanC_01", -10.90, 8.60, 35),
    ("desk_qc_ne", "DeskC_01", 9.40, 7.90, 180),
    ("clutterC_qc", "ClutteringC_01", 6.80, 8.10, 180),
    ("dropped_stack_w2", "ClutteringA_01", -7.25, 5.40, 0),   # narrows west aisle 2
]

#: Dock doors on the south wall (x ranges).
DOCK_X = [(-7.0, -4.0), (-2.5, 0.5), (2.0, 5.0)]
CHARGER_XY = (10.60, -7.80)

CAMERA_Z_M = 6.10        # unchanged from the deployed rig
CAMERA_PITCH_DEG = 52.7
IMG_W, IMG_H, FOV_H_RAD = 1280, 720, 1.5708

#: (name, x, y, yaw_deg, provenance)
CAMERAS = [
    ("cam_dock_w", -5.50, -9.60, 90, "inherited: dock apron west"),
    ("cam_dock_e", 3.00, -9.60, 90, "inherited: dock apron east"),
    ("cam_nw", -11.40, 9.40, -50, "inherited: NW security corner"),
    ("cam_west_aisles", -8.30, -5.40, 90, "for the robot: up the west aisles"),
    ("cam_east_block", 11.40, 2.10, 180, "for the robot: along the east block"),
    ("cam_north", 1.00, 9.60, -90, "for the robot: north end of the highway"),
]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


def obstacles():
    out = [(n, x, y, sx, sy, h) for n, x, y, sx, sy, h, _ in RACKS]
    out.append(PILLAR)
    for name, model, x, y, _yaw in PROPS:
        sx, sy = PROP_SIZES[model]
        out.append((name, x, y, sx, sy, 0.9))     # props are low: rarely occlude
    return out


def build_grid():
    xs = np.arange(SITE[0], SITE[1] + GRID_M, GRID_M)
    ys = np.arange(SITE[2], SITE[3] + GRID_M, GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    drivable = np.ones(gx.shape, dtype=bool)
    for _n, cx, cy, sx, sy, _h in obstacles():
        drivable &= ~((np.abs(gx - cx) <= sx / 2 + 0.18)
                      & (np.abs(gy - cy) <= sy / 2 + 0.18))   # robot half-width clearance
    return xs, ys, gx, gy, drivable


def camera_from_pose(x, y, yaw_deg):
    pitch = math.radians(CAMERA_PITCH_DEG)
    yaw = math.radians(yaw_deg)
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
               -math.sin(pitch))
    scale = -CAMERA_Z_M / forward[2]
    return ObliqueCameraModel(
        cam_pos=(x, y, CAMERA_Z_M),
        look_at=(x + scale * forward[0], y + scale * forward[1], 0.0),
        img_width=IMG_W, img_height=IMG_H, fov_h_rad=FOV_H_RAD)


def in_frame(model, gx, gy):
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, ROBOT_Z_M)], axis=1)
    cam = (pts - np.asarray(model.cam_pos, float)) @ np.asarray(model.R, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pix = cam @ np.asarray(model.K, float).T
        u, v = pix[:, 0] / pix[:, 2], pix[:, 1] / pix[:, 2]
    ok = (cam[:, 2] > 0) & np.isfinite(u) & np.isfinite(v)
    ok &= (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return ok.reshape(gx.shape)


def line_of_sight(model, gx, gy):
    """Ray from camera to floor point; blocked where it passes through an obstacle
    below that obstacle's height. With 6.1 m cameras and 2.6 m racks the ray often
    clears — which is exactly why the double stacks at 5.2 m matter."""
    cam = np.asarray(model.cam_pos, float)
    tgt = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, ROBOT_Z_M)], axis=1)
    clear = np.ones(tgt.shape[0], dtype=bool)
    for t in np.linspace(0.03, 0.97, 32):
        s = cam[None, :] * (1.0 - t) + tgt * t
        for _n, cx, cy, sx, sy, h in obstacles():
            inside = (np.abs(s[:, 0] - cx) <= sx / 2) & (np.abs(s[:, 1] - cy) <= sy / 2)
            clear &= ~(inside & (s[:, 2] < h))
    return clear.reshape(gx.shape)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    xs, ys, gx, gy, drivable = build_grid()

    counts = np.zeros(gx.shape, dtype=int)
    per_camera = {}
    for name, x, y, yaw, _why in CAMERAS:
        model = camera_from_pose(x, y, yaw)
        seen = in_frame(model, gx, gy) & line_of_sight(model, gx, gy) & drivable
        per_camera[name] = float(seen.sum() / max(drivable.sum(), 1))
        counts += seen.astype(int)

    share = {str(k): float(np.mean(counts[drivable] == k)) for k in range(4)}
    share["4+"] = float(np.mean(counts[drivable] >= 4))
    stats = {
        "drivable_cells": int(drivable.sum()),
        "drivable_area_m2": float(drivable.sum() * GRID_M**2),
        "share_by_camera_count": share,
        "unseen": share["0"], "single_camera": share["1"],
        "redundant": float(np.mean(counts[drivable] >= 2)),
        "mean_cameras_per_cell": float(np.mean(counts[drivable])),
        "per_camera_floor_share": per_camera,
        "cameras": len(CAMERAS),
        "reference_current_world": {"cameras": 4, "unseen": 0.01,
                                    "single_camera": 0.57, "redundant": 0.42},
    }
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.6))

    def draw_scene(a, shade=None):
        a.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False,
                              edgecolor="#333333", lw=2.0))
        a.add_patch(Rectangle((SITE[0], SITE[2]), SITE[1] - SITE[0], SITE[3] - SITE[2],
                              fill=False, edgecolor="#2166AC", lw=1.2, ls="--"))
        if shade is not None:
            a.pcolormesh(xs, ys, shade, cmap="YlGnBu", vmin=0, vmax=4, shading="auto",
                         alpha=0.85, zorder=0)
        for name, cx, cy, sx, sy, h, _note in RACKS:
            tall = h >= H_STACK - 1e-6
            a.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy,
                                  facecolor="#3d3d3d" if tall else "#8a8a8a",
                                  edgecolor="#111111", lw=0.7, zorder=3))
        px, py, psx, psy, ph = PILLAR[1], PILLAR[2], PILLAR[3], PILLAR[4], PILLAR[5]
        a.add_patch(Rectangle((px - psx / 2, py - psy / 2), psx, psy,
                              facecolor="#111111", edgecolor="k", zorder=3))
        for _n, model, x, y, _yaw in PROPS:
            sx, sy = PROP_SIZES[model]
            a.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy,
                                  facecolor="#E69F00", edgecolor="#7a5200", lw=0.6,
                                  zorder=4))
        for dx0, dx1 in DOCK_X:
            a.plot([dx0, dx1], [Y0, Y0], lw=6, color="#D55E00", solid_capstyle="butt",
                   zorder=5)
        a.add_patch(Circle(CHARGER_XY, 0.55, facecolor="#009E73", edgecolor="k",
                           lw=0.6, zorder=4))
        for name, x, y, yaw, _why in CAMERAS:
            a.plot([x], [y], marker="o", ms=8, color="#C1121F",
                   markeredgecolor="white", markeredgewidth=1.0, zorder=6)
            a.plot([x, x + 2.6 * math.cos(math.radians(yaw))],
                   [y, y + 2.6 * math.sin(math.radians(yaw))],
                   lw=2.0, color="#C1121F", zorder=6)
        a.set_xlim(X0 - 0.6, X1 + 0.6)
        a.set_ylim(Y0 - 0.6, Y1 + 0.6)
        a.set_aspect("equal")
        a.set_xlabel("x [m]")

    draw_scene(ax)
    ax.set_ylabel("y [m]")
    ax.set_title("Proposed layout\ndark grey = double-stacked (5.2 m, blocks) · "
                 "light grey = single (2.6 m)\norange = AWS props · red = cameras · "
                 "green = charger", fontweight="bold", fontsize=9.5)
    for name, x, y, yaw, why in CAMERAS:
        ax.annotate(name.replace("cam_", ""), xy=(x, y), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=7,
                    fontweight="bold", color="#C1121F", zorder=7)

    shaded = np.where(drivable, np.minimum(counts, 4), np.nan)
    draw_scene(ax2, shade=shaded)
    ax2.set_title(f"What this network sees   ({len(CAMERAS)} cameras)\n"
                  f"unseen {100 * stats['unseen']:.0f} %  ·  single-camera "
                  f"{100 * stats['single_camera']:.0f} %  ·  redundant "
                  f"{100 * stats['redundant']:.0f} %",
                  fontweight="bold", fontsize=9.5)

    fig.suptitle("PROPOSED asymmetric warehouse — BEV plan for approval "
                 "(nothing built yet)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_p1_proposed_world.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"drivable {stats['drivable_area_m2']:.0f} m^2 "
          f"({stats['drivable_cells']} cells @ {GRID_M} m)")
    print(f"unseen {100 * stats['unseen']:.1f}%   single {100 * stats['single_camera']:.1f}%"
          f"   redundant {100 * stats['redundant']:.1f}%"
          f"   mean {stats['mean_cameras_per_cell']:.2f}")
    print("\nper-camera share of drivable floor:")
    for name, x, y, yaw, why in CAMERAS:
        print(f"   {name:<18}{100 * per_camera[name]:>6.1f}%   {why}")
    print("\ncurrent world for reference: 4 cams, unseen ~1%, single 57%, redundant 42%")
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
