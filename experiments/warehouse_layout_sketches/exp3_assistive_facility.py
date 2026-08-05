#!/usr/bin/env python3
"""exp3: the facility, redesigned for the ASSISTIVE-INFRASTRUCTURE framing.

Supersedes exp2, which was a box arrangement at half scale and did not correspond
to any real class of facility. This one is built from published dimensions and
from the reference deployment described in "Infrastructure-based Autonomous Mobile
Robots for Internal Logistics" (arXiv 2512.15215): ceiling cameras, on-premise
compute, minimal robots, MIXED TRAFFIC with pedestrians and manually driven
vehicles, brownfield.

Dimensions are from warehouse design practice, not invented:

    rack depth          1.10 m      (industry standard 1.07-1.22 m)
    picking aisle       1.60 m      (narrow-aisle class; AMR + person)
    rack pitch          2.70 m      (= 1.10 + 1.60)
    cross aisle         2.80 m      (direct routes between picking aisles)
    perimeter haul lane 3.00 m      (wide-aisle class, mixed traffic)

Layout follows the conventional warehouse form the MAPF warehouse grids and the
AMR literature both describe: long rows of shelving forming narrow aisles,
perpendicular cross-aisles, and WORKSTATIONS AROUND THE PERIMETER.

What makes it the right testbed for this work, and none of it is decoration:

  * The camera network is INHERITED. Four cameras sit where a security/operations
    install would put them -- over the dock doors, over the pick stations, and in
    the two corners holding assets (QC bench, charging bank). They were not placed
    for the robot, and it shows: they cover the perimeter zones and barely reach
    into the storage field.
  * Only three cameras are retrofitted for the robot, on the cross-aisle and the
    north haul lane.
  * Storage genuinely occludes. Cameras are at 6.1 m; a single 2.6 m rack is seen
    over from a distance, so selected rows are DOUBLE-STACKED to 5.2 m. Those are
    what create blind aisles.
  * Zones are asymmetric by function: dock south-west, pick stations east,
    charging north-west, QC north-east. Nothing reflects.
  * The haul route is long relative to the building -- dock to stations runs the
    perimeter, echoing the reference deployment's ~150 m routes.

Nothing is built or run by this script. It draws the plan and computes what the
proposed network would see.

Outputs -> logs/studies/warehouse_layout_sketches/exp3_assistive_facility/
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

from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

OUT = REPO / "logs/studies/warehouse_layout_sketches/exp3_assistive_facility"

# ------------------------------------------------------------------ dimensions
X0, X1, Y0, Y1 = -12.0, 12.0, -10.0, 10.0          # building, 24 x 20 m
SITE = (-11.4, 11.4, -9.4, 9.4)                     # drivable envelope
GRID_M = 0.20
ROBOT_Z_M = 0.05
ROBOT_CLEARANCE_M = 0.18

RACK_DEPTH = 1.10
PICK_AISLE = 1.60
RACK_PITCH = RACK_DEPTH + PICK_AISLE                # 2.70
CROSS_AISLE = 2.80
HAUL_LANE = 3.00

H_SINGLE, H_DOUBLE = 2.61, 5.20                     # AWS shelf mesh, and stacked

FIELD_X0 = SITE[0] + HAUL_LANE                      # storage field bounds
FIELD_X1 = SITE[1] - HAUL_LANE
FIELD_Y0 = SITE[2] + HAUL_LANE
FIELD_Y1 = SITE[3] - HAUL_LANE

#: Two bands of racking split by a central cross-aisle.
BAND_N = (CROSS_AISLE / 2.0, FIELD_Y1)
BAND_S = (FIELD_Y0, -CROSS_AISLE / 2.0)

#: Rows are double-stacked on a 1-in-2 pattern: enough to create blind aisles
#: without walling the whole field off.
DOUBLE_STACK_ROWS = {0, 2, 4}


def racks():
    out = []
    x = FIELD_X0 + RACK_DEPTH / 2.0
    row = 0
    while x + RACK_DEPTH / 2.0 <= FIELD_X1 - PICK_AISLE:
        height = H_DOUBLE if row in DOUBLE_STACK_ROWS else H_SINGLE
        for tag, (y_lo, y_hi) in (("N", BAND_N), ("S", BAND_S)):
            out.append((f"rack{row}{tag}", x, (y_lo + y_hi) / 2.0,
                        RACK_DEPTH, y_hi - y_lo, height))
        x += RACK_PITCH
        row += 1
    return out


RACKS = racks()
PILLARS = [("pillar_w", -6.0, 0.0, 0.40, 0.40, 5.60),
           ("pillar_e", 4.2, 0.0, 0.40, 0.40, 5.60)]

PROP_SIZES = {
    "Bucket_01": (0.42, 0.42), "TrashCanC_01": (0.55, 0.55),
    "ClutteringA_01": (0.90, 0.75), "ClutteringC_01": (1.00, 0.80),
    "ClutteringD_01": (0.90, 0.80), "PalletJackB_01": (1.35, 0.65),
    "DeskC_01": (1.55, 0.85),
}
PROPS = [
    ("palletjack_dock", "PalletJackB_01", -6.20, -8.30, 10),
    ("clutterD_inbound", "ClutteringD_01", -2.60, -8.60, 0),
    ("clutterA_inbound", "ClutteringA_01", -9.60, -8.40, 0),
    ("bucket_lane_s", "Bucket_01", 1.60, -8.70, 0),
    ("desk_qc", "DeskC_01", 9.90, 8.40, 180),
    ("clutterC_qc", "ClutteringC_01", 7.40, 8.60, 180),
    ("trash_charge", "TrashCanC_01", -8.20, 8.70, 30),
    ("blocked_aisle_stack", "ClutteringA_01", -4.85, 4.20, 0),   # narrows one aisle
]

DOCK_DOORS = [(-9.0, -6.0), (-5.0, -2.0)]           # south wall, west-biased
PICK_STATIONS = [(10.9, -4.0), (10.9, 0.0), (10.9, 4.0)]   # east wall
CHARGERS = [(-10.4, 7.4), (-10.4, 6.0)]             # north-west bank

CAMERA_Z_M = 6.10
CAMERA_PITCH_DEG = 52.7
IMG_W, IMG_H, FOV_H_RAD = 1280, 720, 1.5708

#: (name, x, y, yaw_deg, provenance). FOUR inherited, THREE retrofitted.
CAMERAS = [
    ("dock", -7.0, -9.3, 90, "inherited — dock doors (goods in/out)"),
    ("stations", 11.1, 0.0, 180, "inherited — pick/pack stations"),
    ("qc_ne", 11.1, 9.2, -135, "inherited — QC bench, NE corner"),
    ("charge_nw", -11.1, 9.2, -45, "inherited — charging bank (fire watch)"),
    ("cross_w", -11.1, 0.0, 0, "retrofit — cross-aisle, from the west"),
    ("cross_e", 11.1, -1.6, 175, "retrofit — cross-aisle, from the east"),
    ("lane_n", -1.0, 9.3, -90, "retrofit — north haul lane"),
]

#: Long haul route: dock (SW) -> west lane -> north lane -> east stations.
HAUL_ROUTE = [(-7.0, -8.2), (-9.9, -8.2), (-9.9, 8.0), (0.0, 8.0),
              (9.9, 8.0), (9.9, 0.0)]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


def obstacles():
    out = [(n, x, y, sx, sy, h) for n, x, y, sx, sy, h in RACKS]
    out += PILLARS
    for name, model, x, y, _yaw in PROPS:
        sx, sy = PROP_SIZES[model]
        out.append((name, x, y, sx, sy, 0.95))
    return out


def build_grid():
    xs = np.arange(SITE[0], SITE[1] + GRID_M, GRID_M)
    ys = np.arange(SITE[2], SITE[3] + GRID_M, GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    drivable = np.ones(gx.shape, dtype=bool)
    for _n, cx, cy, sx, sy, _h in obstacles():
        drivable &= ~((np.abs(gx - cx) <= sx / 2 + ROBOT_CLEARANCE_M)
                      & (np.abs(gy - cy) <= sy / 2 + ROBOT_CLEARANCE_M))
    return xs, ys, gx, gy, drivable


def camera_from_pose(x, y, yaw_deg):
    pitch = math.radians(CAMERA_PITCH_DEG)
    yaw = math.radians(yaw_deg)
    fwd = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
           -math.sin(pitch))
    scale = -CAMERA_Z_M / fwd[2]
    return ObliqueCameraModel(
        cam_pos=(x, y, CAMERA_Z_M),
        look_at=(x + scale * fwd[0], y + scale * fwd[1], 0.0),
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
    cam = np.asarray(model.cam_pos, float)
    tgt = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, ROBOT_Z_M)], axis=1)
    clear = np.ones(tgt.shape[0], dtype=bool)
    for t in np.linspace(0.03, 0.97, 30):
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
    inherited_counts = np.zeros(gx.shape, dtype=int)
    per_camera = {}
    for name, x, y, yaw, why in CAMERAS:
        seen = (in_frame(camera_from_pose(x, y, yaw), gx, gy)
                & line_of_sight(camera_from_pose(x, y, yaw), gx, gy) & drivable)
        per_camera[name] = float(seen.sum() / max(drivable.sum(), 1))
        counts += seen.astype(int)
        if why.startswith("inherited"):
            inherited_counts += seen.astype(int)

    # Is the storage field the blind part, as an inherited network implies?
    in_field = ((gx >= FIELD_X0) & (gx <= FIELD_X1)
                & (gy >= FIELD_Y0) & (gy <= FIELD_Y1) & drivable)
    perimeter = drivable & ~in_field

    stats = {
        "building_m": [X1 - X0, Y1 - Y0],
        "dimensions": {"rack_depth_m": RACK_DEPTH, "pick_aisle_m": PICK_AISLE,
                       "rack_pitch_m": RACK_PITCH, "cross_aisle_m": CROSS_AISLE,
                       "haul_lane_m": HAUL_LANE},
        "rack_segments": len(RACKS),
        "drivable_area_m2": float(drivable.sum() * GRID_M**2),
        "cameras": {"total": len(CAMERAS),
                    "inherited": sum(1 for c in CAMERAS if c[4].startswith("inherited")),
                    "retrofit": sum(1 for c in CAMERAS if c[4].startswith("retrofit"))},
        "all_cameras": {
            "unseen": float(np.mean(counts[drivable] == 0)),
            "single": float(np.mean(counts[drivable] == 1)),
            "redundant": float(np.mean(counts[drivable] >= 2)),
            "mean": float(np.mean(counts[drivable])),
        },
        "inherited_only": {
            "unseen": float(np.mean(inherited_counts[drivable] == 0)),
            "redundant": float(np.mean(inherited_counts[drivable] >= 2)),
        },
        "storage_field_vs_perimeter": {
            "field_unseen": float(np.mean(counts[in_field] == 0)),
            "perimeter_unseen": float(np.mean(counts[perimeter] == 0)),
            "field_mean_cameras": float(np.mean(counts[in_field])),
            "perimeter_mean_cameras": float(np.mean(counts[perimeter])),
        },
        "per_camera_floor_share": per_camera,
        "reference_current_world": {"cameras": 4, "unseen": 0.01, "single": 0.57},
    }
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.4, 6.8))

    def draw(a, shade=None):
        if shade is not None:
            a.pcolormesh(xs, ys, shade, cmap="YlGnBu", vmin=0, vmax=4,
                         shading="auto", zorder=0)
        a.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False,
                              edgecolor="#222222", lw=2.2))
        for _n, cx, cy, sx, sy, h in RACKS:
            tall = h >= H_DOUBLE - 1e-6
            a.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy,
                                  facecolor="#3a3a3a" if tall else "#8f8f8f",
                                  edgecolor="#111111", lw=0.6, zorder=3))
        for _n, cx, cy, sx, sy, _h in PILLARS:
            a.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy,
                                  facecolor="#000000", zorder=3))
        for _n, model, x, y, _yaw in PROPS:
            sx, sy = PROP_SIZES[model]
            a.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy,
                                  facecolor="#E69F00", edgecolor="#7a5200",
                                  lw=0.5, zorder=4))
        for dx0, dx1 in DOCK_DOORS:
            a.plot([dx0, dx1], [Y0, Y0], lw=7, color="#D55E00",
                   solid_capstyle="butt", zorder=5)
        for sx_, sy_ in PICK_STATIONS:
            a.add_patch(Rectangle((sx_ - 0.35, sy_ - 0.9), 0.7, 1.8,
                                  facecolor="#0072B2", edgecolor="k", lw=0.5, zorder=5))
        for cx_, cy_ in CHARGERS:
            a.add_patch(Circle((cx_, cy_), 0.42, facecolor="#009E73",
                               edgecolor="k", lw=0.5, zorder=5))
        route = np.asarray(HAUL_ROUTE)
        a.plot(route[:, 0], route[:, 1], lw=2.2, ls=(0, (6, 3)), color="#7B3294",
               zorder=6, alpha=0.9)
        for name, x, y, yaw, why in CAMERAS:
            colour = "#C1121F" if why.startswith("inherited") else "#1B7837"
            a.plot([x], [y], marker="o", ms=8, color=colour,
                   markeredgecolor="white", markeredgewidth=1.0, zorder=7)
            a.plot([x, x + 2.5 * math.cos(math.radians(yaw))],
                   [y, y + 2.5 * math.sin(math.radians(yaw))],
                   lw=2.0, color=colour, zorder=7)
        a.set_xlim(X0 - 0.7, X1 + 0.7)
        a.set_ylim(Y0 - 0.7, Y1 + 0.7)
        a.set_aspect("equal")
        a.set_xlabel("x [m]")

    draw(ax)
    ax.set_ylabel("y [m]")
    ax.set_title("Layout at real dimensions\n"
                 f"rack {RACK_DEPTH:.2f} m · pick aisle {PICK_AISLE:.2f} m · "
                 f"cross {CROSS_AISLE:.1f} m · haul lane {HAUL_LANE:.1f} m\n"
                 "dark = double-stacked 5.2 m · orange = AWS props · "
                 "blue = pick stations · green = chargers · purple = haul route",
                 fontweight="bold", fontsize=9)
    for name, x, y, yaw, why in CAMERAS:
        ax.annotate(name, xy=(x, y), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=7, fontweight="bold",
                    color="#C1121F" if why.startswith("inherited") else "#1B7837",
                    zorder=8)

    draw(ax2, shade=np.where(drivable, np.minimum(counts, 4), np.nan))
    a = stats["all_cameras"]
    f = stats["storage_field_vs_perimeter"]
    ax2.set_title(f"What the network sees   (4 inherited + 3 retrofit)\n"
                  f"unseen {100 * a['unseen']:.0f} %  ·  single {100 * a['single']:.0f} %"
                  f"  ·  redundant {100 * a['redundant']:.0f} %\n"
                  f"storage field {100 * f['field_unseen']:.0f} % blind  vs  "
                  f"perimeter {100 * f['perimeter_unseen']:.0f} % blind",
                  fontweight="bold", fontsize=9)

    fig.suptitle("Brownfield internal-logistics floor — inherited camera network "
                 "(red) plus three retrofits (green)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_f1_assistive_facility.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"building {X1 - X0:.0f} x {Y1 - Y0:.0f} m · drivable "
          f"{stats['drivable_area_m2']:.0f} m^2 · {len(RACKS)} rack segments")
    print(f"all 7 cameras : unseen {100 * a['unseen']:.1f}%  single "
          f"{100 * a['single']:.1f}%  redundant {100 * a['redundant']:.1f}%  "
          f"mean {a['mean']:.2f}")
    print(f"inherited only: unseen {100 * stats['inherited_only']['unseen']:.1f}%  "
          f"redundant {100 * stats['inherited_only']['redundant']:.1f}%")
    print(f"storage field {100 * f['field_unseen']:.1f}% blind "
          f"({f['field_mean_cameras']:.2f} cams) vs perimeter "
          f"{100 * f['perimeter_unseen']:.1f}% blind "
          f"({f['perimeter_mean_cameras']:.2f} cams)")
    print("\nper-camera share of drivable floor:")
    for name, x, y, yaw, why in CAMERAS:
        print(f"   {name:<12}{100 * per_camera[name]:>6.1f}%   {why}")
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
