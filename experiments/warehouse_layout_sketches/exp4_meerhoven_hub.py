#!/usr/bin/env python3
"""exp4: MEERHOVEN REGIONAL HUB — a facility derived from a storyline.

Supersedes exp2/exp3, which arranged boxes and then wrote a caption. Here the
storyline comes first and every geometric feature has a business reason. Nothing is
built or run; this draws the plan and scores what the network would see.

================================ THE STORYLINE ================================

MEERHOVEN REGIONAL HUB — a mid-size e-commerce returns-and-outbound hub occupying a
1983 general-warehouse shell that was repurposed in 2024.

Two flows share one floor, and they do not get on:

  OUTBOUND (planned).  Pallets arrive at two inbound doors, are broken down into
  totes, picked from selective racking, consolidated on a pack line, wrapped,
  weighed, and shipped from a single outbound door.

  RETURNS (unplanned).  The returns stream tripled in two years. Nobody designed for
  it, so it was shoehorned into the old bulk-storage bay at the north-east, because
  that was the only free floor. It has no racking — goods are BLOCK-STACKED on the
  slab and the stacks move weekly.

Eighteen low-cost AMRs ferry totes between pick faces, the pack line and the returns
benches, alongside nine pickers and manual pallet-jack traffic (an AMR:picker ratio of
about 2:1, the throughput optimum reported for AMR-assisted picking).

WHY THIS SITE USES INFRASTRUCTURE CAMERAS, and not onboard SLAM:

  1. Eighteen robots. A LiDAR per robot costs more than the whole camera network.
  2. Six cameras were ALREADY THERE — an insurer's condition covering the dock doors,
     the high-value cage and the returns bench, where shrinkage happens. The
     localization system inherits a network installed to watch people, not robots.
  3. The decisive reason: THE FLOOR LAYOUT CHANGES WEEKLY. Block stacks in the
     returns bay migrate; staging spills into aisles. An onboard SLAM map goes stale
     between shifts, whereas an overhead network sees the current state and can
     re-publish the drivable map centrally.
  4. The main cross-aisle is shared with pedestrians and manual pallet jacks, so a
     central view of where every robot is has a safety argument the robots cannot
     make individually.

WHAT THE 1983 SHELL IMPOSES:  a structural column grid on 8 m centres that predates
the racking, so columns land inside aisles; a mezzanine added over the pack line in
2024 for the QC office, whose deck blocks the cameras above it; and a charging bank
placed not where robots wanted it but where three-phase power already existed, on the
west wall beside the old switch room.

============================== WHAT IT SHOWCASES ==============================

Each measured result has a named place in this building that produces it:

  per-camera lateral bias        `returns_e` is a 2011 analogue-era mount on a long
                                 bracket over the returns bench, in the aisle manual
                                 pallet jacks use. It is the leaning camera, and it is
                                 the one that gets knocked.
  overconfident belief           the haul run under the mezzanine deck (x 8..14,
                                 y -9..-5), reachable by one camera only.
  cross-check needs overlap      the cross-aisle midpoint, where `xaisle_w`,
                                 `xaisle_e` and `pick_mouths` all reach.
  blind runs                     picking aisles between double-stacked runs, and the
                                 floor behind the block stacks in the returns bay.
  precision varies by task       pack-line infeed docking (tight) sits 12 m from the
                                 open haul lane (loose). Same robot, same minute.
  stale calibration              the knocked bracket on `returns_e` is the drift
                                 source, and it is the camera whose correction is
                                 largest — exactly where staleness hurts.

Outputs -> logs/studies/warehouse_layout_sketches/exp4_meerhoven_hub/
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

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
sys.path.insert(0, str(_HERE.parent))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "unav_common"))

import facility_draw as fd  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

OUT = REPO / "logs/studies/warehouse_layout_sketches/exp4_meerhoven_hub"

# ============================================================== the building
# 30 x 20 m. Larger than the current 24 x 20 world by 1.25x, so a detector capture
# grows from ~25 to ~31 min — the cost of having two flows and a repurposed bay
# instead of one storage block.
X0, X1, Y0, Y1 = -15.0, 15.0, -10.0, 10.0
GRID_M = 0.20
ROBOT_Z_M = 0.05
ROBOT_CLEARANCE_M = 0.18

# Dimensions from warehouse design practice (see exp3 header for sources).
RACK_DEPTH, PICK_AISLE = 1.10, 1.60
RACK_PITCH = RACK_DEPTH + PICK_AISLE          # 2.70
CROSS_AISLE = 2.80
HAUL_LANE = 3.00
H_SINGLE, H_DOUBLE = 2.61, 5.20

CAMERA_PITCH_DEG = 52.7
IMG_W, IMG_H, FOV_H_RAD = 1280, 720, 1.5708

#: Per-camera downward pitch where it differs from the deployed 52.7 deg. A lower,
#: steeper mount sees a SMALLER patch at FINER resolution — the right trade for a
#: work area, and the wrong one for a lane.
CAMERA_PITCH = {
    "office_ne": 70.0,     # stair head only; it has no business watching the bay
    "pack_mezz": 62.0,     # under the deck, so it must look steeply down anyway
    "cage_hv": 62.0,       # column mount aimed at the cage, not a floor sweep
}

# ------------------------------------------------------------------ structure
#: 1983 column grid on 8 m centres — predates the racking, so columns sit in aisles.
COLUMN_XS = [-11.0, -3.0, 5.0, 13.0]
COLUMN_YS = [-5.6, 2.2]
COLUMN_SIZE = 0.42

MEZZANINE = (7.4, -9.5, 14.5, -6.4)          # QC office deck over the pack line, 2024

# ------------------------------------------------------------------ elements
# kind, name, x, y, sx, sy, height, occludes, zone (+ per-kind extras)
ELEMENTS: list[dict] = []


def _add(kind, name, x, y, sx, sy, h, occl, zone, **kw):
    ELEMENTS.append(dict(kind=kind, name=name, x=x, y=y, sx=sx, sy=sy,
                         height=h, occludes=occl, zone=zone, **kw))


# --- band structure, north to south, summing to the 20 m clear span ----------
#   y  7.4 ..  9.6   north lane + charging (2.2 m)
#   y  3.4 ..  7.4   rack band N (4.0 m)
#   y  0.6 ..  3.4   cross aisle (2.8 m)
#   y -3.4 ..  0.6   rack band S (4.0 m)
#   y -6.4 .. -3.4   MAIN HAUL LANE (3.0 m), shared with manual traffic
#   y -9.6 .. -6.6   dock-side zones (3.0 m)
RACK_BAND_S = (-3.4, 0.6)
RACK_BAND_N = (3.4, 7.4)
NORTH_LANE_Y = 8.4
HAUL_LANE_Y = -4.9
_rack_x = []
_x = -12.0 + RACK_DEPTH / 2.0
_row = 0
while _x + RACK_DEPTH / 2.0 <= 2.7:
    tall = _row in (0, 2, 4)                  # 1-in-2 double-stacked
    for tag, (y_lo, y_hi) in (("S", RACK_BAND_S), ("N", RACK_BAND_N)):
        _add("rack", f"R{_row}{tag}", _x, (y_lo + y_hi) / 2.0,
             RACK_DEPTH, y_hi - y_lo, H_DOUBLE if tall else H_SINGLE, True,
             "storage", tall=tall)
    _rack_x.append(_x)
    _x += RACK_PITCH
    _row += 1

#: Aisle centrelines between adjacent rack runs — the only drivable lines through
#: the storage field. The mission MUST use these; earlier versions put a waypoint
#: inside a rack because the route was drawn by eye.
AISLE_X = [0.5 * (a + b) for a, b in zip(_rack_x[:-1], _rack_x[1:])]
WEST_LANE_X = -13.6

# --- carton flow rack: shallow, fast-mover pick face on the cross-aisle -------
# A carton flow-rack module bolted to the end of run 2, inside that run's own x
# footprint so it blocks no aisle mouth. (It previously spanned two of them.)
_add("rack", "flow_rack", _rack_x[2], 2.05, RACK_DEPTH, 0.70, 2.20, False, "storage",
     tall=False)

# --- the old bulk bay, now returns: BLOCK-STACKED goods, opaque, they move ----
_add("block", "stack_A", 5.8, 7.6, 2.6, 2.4, 2.40, True, "returns")
_add("block", "stack_B", 9.2, 7.9, 2.2, 1.8, 2.20, True, "returns")
_add("block", "stack_C", 6.2, 3.8, 2.8, 2.0, 2.60, True, "returns")
_add("block", "stack_D", 9.6, 4.4, 1.8, 2.4, 2.00, True, "returns")
_add("block", "stack_E", 6.0, 0.4, 1.6, 1.4, 1.80, True, "returns")

# --- returns benches along the east wall -------------------------------------
for i, y in enumerate((7.2, 4.4, 1.6)):
    _add("station", f"returns_bench_{i + 1}", 14.0, y, 0.9, 1.6, 0.95, False, "returns")

# --- pack line: conveyor + wrapper + scale, under the mezzanine ---------------
CONVEYOR = [(5.6, -8.6), (13.2, -8.6)]
_add("machine", "stretch_wrapper", 8.4, -7.1, 1.9, 1.9, 2.60, True, "pack",
     glyph="wrapper")
_add("machine", "checkweigher", 11.2, -7.2, 1.4, 1.0, 1.20, False, "pack",
     glyph="scale")
_add("machine", "case_erector", 6.9, -7.1, 1.5, 1.4, 2.10, True, "pack", glyph="box")
_add("station", "pack_bench_1", 13.6, -7.4, 1.1, 1.6, 0.95, False, "pack")

# --- inbound: receiving desk, staged pallets, jack ---------------------------
_add("machine", "receiving_desk", -14.0, -7.6, 0.9, 1.6, 1.10, False, "inbound",
     glyph="box")
_add("prop", "pallet_jack", -12.6, -8.9, 0.65, 1.35, 1.20, False, "inbound")
_add("block", "inbound_staged_1", -9.8, -8.4, 2.4, 1.8, 1.60, True, "inbound")
_add("block", "inbound_staged_2", -6.4, -8.6, 1.8, 1.4, 1.40, True, "inbound")
_add("prop", "clutter_inbound", -11.4, -6.9, 0.90, 0.75, 0.80, False, "inbound")

# --- outbound staging -------------------------------------------------------
_add("block", "outbound_staged", 3.0, -7.6, 2.4, 1.5, 1.60, True, "outbound")

# --- charging bank on the NORTH wall, beside the old switch room -------------
# It sits where three-phase power already ran, not where robots would want it.
# (An earlier version put it on the west wall, where it walled off the west lane.)
for i, x in enumerate((-10.6, -9.0, -7.4)):
    _add("machine", f"charger_{i + 1}", x, 9.1, 1.0, 0.8, 1.30, False, "charge",
         glyph="charger")
_add("machine", "switch_room", -13.2, 9.0, 2.0, 1.4, 3.00, True, "charge", glyph="box")

# --- high-value cage: mesh panels, see-through but a hard obstacle ----------
# High-value returns cage. It sat across the north lane in an earlier version, which
# the route validator caught; here it occupies open floor between the haul lane and
# the returns bay, which is also where such a cage would actually go.
_add("cage", "hv_cage", 7.0, -1.8, 3.4, 1.8, 2.40, False, "cage")

# --- misc ---------------------------------------------------------------------
_add("prop", "trash_ne", 14.2, 0.0, 0.55, 0.55, 0.90, False, "returns")
_add("prop", "bucket_pack", 14.2, -9.2, 0.42, 0.42, 0.50, False, "pack")

DOCK_DOORS = [(-12.4, -9.4, "IN-1"), (-8.0, -5.0, "IN-2"), (1.4, 4.4, "OUT-1")]

#: Pedestrians walk the DOCK SIDE of the haul lane, fenced off from AMR traffic, so
#: they reach the docks and the pack line without entering the lane. The fence runs
#: along the lane's south edge, NOT between the lane and the storage aisles — an
#: earlier version did the latter and fenced every picking aisle off from the
#: highway behind a single gap, which the route validator caught immediately.
#: Gaps are where AMRs cross: the outbound/pack crossing and the east-lane turn.
WALKWAYS = [[(-14.4, -6.15), (14.4, -6.15)]]
GUARDRAILS = [[(-14.4, -6.2), (0.8, -6.2)],
              [(4.4, -6.2), (11.0, -6.2)],
              [(13.4, -6.2), (14.4, -6.2)]]

ZONES = [
    dict(name="inbound / receiving", x0=-14.6, y0=-9.6, x1=-4.6, y1=-6.5, kind="inbound"),
    dict(name="outbound staging", x0=1.0, y0=-9.6, x1=4.0, y1=-6.5, kind="outbound"),
    dict(name="pack line", x0=4.6, y0=-9.6, x1=14.6, y1=-6.5, kind="pack"),
    dict(name="main haul lane (mixed traffic)", x0=-14.6, y0=-6.4, x1=14.6, y1=-3.5,
         kind="walkway"),
    dict(name="selective racking — pick faces", x0=-13.0, y0=-3.5, x1=3.4, y1=7.6,
         kind="storage"),
    dict(name="returns — old bulk bay", x0=4.2, y0=-0.6, x1=14.6, y1=9.6,
         kind="returns"),
    dict(name="charging", x0=-11.4, y0=8.2, x1=-6.6, y1=9.6, kind="charge"),
    dict(name="HV returns cage", x0=5.1, y0=-2.9, x1=8.9, y1=-0.7, kind="qc"),
]

# --------------------------------------------------------------------- cameras
# name, x, y, z, yaw, provenance, note
CAMERAS = [
    ("dock_in", -8.6, -9.6, 6.10, 88, "inherited",
     "2019 insurer condition: both inbound doors in one frame"),
    ("dock_out", 2.7, -9.6, 6.10, 92, "inherited",
     "2019 insurer condition: the outbound door"),
    ("cage_hv", 5.0, -5.6, 6.10, 62, "inherited",
     "high-value returns cage, viewed from the nearest BUILDING COLUMN (grid C1). "
     "An earlier version had it floating 2.4 m from that column in open floor, which "
     "is not a thing you can bolt a camera to"),
    ("returns_e", 14.7, 5.4, 5.60, -125, "inherited",
     "2011 analogue-era mount, long bracket over the returns benches; shrinkage "
     "watch. THE LEANING CAMERA, and the one manual pallet jacks knock. A rendered "
     "12-view audit corrected the original +168 deg yaw, which looked through the "
     "north rack band instead of centring the returns work area"),
    ("pack_mezz", 10.95, -6.5, 2.90, -100, "inherited",
     "bolted to the EDGE BEAM of the 2024 mezzanine deck, underneath — the only way "
     "to see the pack line at all, since the deck blocks everything above it. At "
     "2.9 m it is the odd camera out: short reach, finest ground resolution here"),
    ("office_ne", 14.2, 9.4, 3.30, -128, "inherited",
     "QC office stair head. Mounted low and steep because that is all it was for — "
     "it watches the stairs, not the floor. Re-aiming it off the returns bay is what "
     "leaves the returns benches observed by ONE camera, the leaning one"),
    # --- 2024 retrofit: six cameras, on real mounts, chosen by measurement ---
    # Route-weighted greedy over a pool of surfaces an installer can bolt to, with a
    # third view scored at zero so it cannot pile up overlap. Two earlier pools were
    # wrong and each was caught by a check rather than by eye: one allowed mid-floor
    # positions (half the network hung in mid-air), the other sampled the walls on a
    # 3 m grid that never aligned with the 2.70 m rack pitch, so a wall camera ended
    # up staring down the length of a rack run.
    ("north_wlane", -12.0, 9.7, 6.10, -90, "retrofit",
     "2024 retrofit #1: NORTH WALL at the west end, covering the west lane and the "
     "north lane — the robot's whole return leg"),
    ("cross_col", 5.0, 2.2, 6.10, -90, "retrofit",
     "2024 retrofit #2: building COLUMN grid C2, looking south across the cross "
     "aisle, the returns approach and the east pick faces"),
    ("aisle2_n", -7.4, 9.7, 6.10, -90, "retrofit",
     "2024 retrofit #3: NORTH WALL, ON THE CENTRELINE OF PICKING AISLE 2, looking "
     "straight down it. Alignment is the whole point: a wall mount half a metre off "
     "sees the side of a rack instead of the aisle floor"),
    ("south_wlane", -12.0, -9.7, 6.10, 90, "retrofit",
     "2024 retrofit #4: SOUTH WALL at the west end, covering inbound and the west "
     "end of the haul lane"),
    ("haul_col", -3.0, -5.6, 6.10, 0, "retrofit",
     "2024 retrofit #5: building COLUMN grid B1, looking east along the haul lane. "
     "The column stands in the lane anyway, so it is the natural mount"),
    ("west_w", -14.7, 2.0, 6.10, 0, "retrofit",
     "2024 retrofit #6: WEST WALL, looking east across the storage field. Last "
     "camera worth buying — the seventh candidate's marginal gain falls to 13 %"),
]

# ------------------------------------------------------------- robot mission
#: The tote run. Every waypoint lies on an aisle centreline, a lane, or a dock face,
#: and `main()` asserts each one is drivable — an earlier version drew this by eye and
#: put a waypoint inside a rack.
def _mission():
    a2 = AISLE_X[1]                    # second picking aisle
    east_lane = 12.2
    return [
        dict(wp="charger 2", x=-9.0, y=NORTH_LANE_Y, act="undock from charge",
             prec="±0.05 m"),
        dict(wp="aisle 2 north entry", x=a2, y=NORTH_LANE_Y - 0.6,
             act="enter picking aisle", prec="±0.10 m"),
        dict(wp="pick face 2N", x=a2, y=5.4,
             act="tote pick, deep in a double-stacked aisle", prec="±0.08 m"),
        dict(wp="cross aisle", x=a2, y=2.0, act="exit to the cross aisle",
             prec="±0.15 m"),
        dict(wp="aisle 2 south", x=a2, y=-1.4, act="transit the south band",
             prec="±0.10 m"),
        dict(wp="haul lane west", x=a2, y=HAUL_LANE_Y, act="join the haul lane",
             prec="±0.20 m"),
        dict(wp="guardrail gap", x=1.4, y=HAUL_LANE_Y,
             act="cross the guardrail gap southbound", prec="±0.20 m"),
        dict(wp="dock face", x=1.4, y=-9.0, act="turn onto the pack-line lane",
             prec="±0.15 m"),
        dict(wp="pack infeed", x=5.6, y=-9.0, act="dock to the conveyor infeed",
             prec="±0.02 m — the tightest task in the building"),
        dict(wp="pack line east", x=east_lane, y=-9.0,
             act="run the length of the pack line", prec="±0.15 m"),
        dict(wp="east lane", x=east_lane, y=HAUL_LANE_Y,
             act="turn north for returns", prec="±0.20 m"),
        dict(wp="bench approach", x=east_lane, y=4.4,
             act="north up the east lane, past the column", prec="±0.20 m"),
        dict(wp="returns bench 2", x=east_lane + 0.9, y=4.4,
             act="drop a returns tote", prec="±0.05 m"),
        dict(wp="bench depart", x=east_lane, y=4.4,
             act="back out to the east lane", prec="±0.20 m"),
        dict(wp="east lane return", x=east_lane, y=HAUL_LANE_Y,
             act="back onto the haul lane", prec="±0.20 m"),
        dict(wp="haul lane west", x=WEST_LANE_X, y=HAUL_LANE_Y,
             act="run the lane west", prec="±0.20 m"),
        dict(wp="west lane north", x=WEST_LANE_X, y=7.9,
             act="north along the west lane", prec="±0.20 m"),
        dict(wp="charger 2", x=-9.0, y=7.9, act="return to charge", prec="±0.05 m"),
    ]


MISSION = _mission()

# =========================================================== geometry helpers


def obstacles():
    """Everything the ROBOT cannot drive through.

    Guardrails belong here. Rendering the generated world showed them as solid
    barriers in Gazebo while this grid ignored them — so the plan said the robot
    could drive where the world said it could not. The single crossing gap between
    the two rail runs is the only way through, and the route validator now has to
    honour it.
    """
    out = [(e["name"], e["x"], e["y"], e["sx"], e["sy"], e["height"])
           for e in ELEMENTS]
    for x in COLUMN_XS:
        for y in COLUMN_YS:
            out.append((f"col_{x}_{y}", x, y, COLUMN_SIZE, COLUMN_SIZE, 5.60))
    for gi, pts in enumerate(GUARDRAILS):
        (gx0, gy0), (gx1, gy1) = pts[0], pts[-1]
        out.append((f"guardrail_{gi}", 0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1),
                    abs(gx1 - gx0) or 0.10, abs(gy1 - gy0) or 0.10, 0.95))
    return out


def occluders():
    """Only what actually blocks a 6.1 m camera, plus the mezzanine deck.

    Guardrails are deliberately absent: posts and two thin rails stop a robot but
    hide almost nothing from a camera looking down from 6.1 m.
    """
    out = [(e["name"], e["x"], e["y"], e["sx"], e["sy"], e["height"])
           for e in ELEMENTS if e["occludes"]]
    for x in COLUMN_XS:
        for y in COLUMN_YS:
            out.append((f"col_{x}_{y}", x, y, COLUMN_SIZE, COLUMN_SIZE, 5.60))
    return out


#: Height of the mezzanine deck. It is a horizontal SLAB, not a solid volume: it
#: blocks rays that cross its footprint from ABOVE, and leaves the floor beneath it
#: visible only to something mounted underneath. Modelling it as a box from the slab
#: down (the obvious mistake) made the pack-line camera see nothing at all.
MEZZANINE_DECK_Z = 3.20


def mount_of(x, y, z, tol=0.75):
    """What is this camera bolted to? Returns None if the answer is 'nothing'."""
    best, best_d = None, tol
    for _tag, kind, mx, my, mz, _yaw, _pitch in mount_points():
        d = math.hypot(x - mx, y - my)
        if d < best_d and abs(z - mz) < 1.6:
            best, best_d = kind, d
    if best is None:
        for cx in COLUMN_XS:
            for cy in COLUMN_YS:
                if math.hypot(x - cx, y - cy) < tol:
                    return "column"
        if min(abs(x - X0), abs(X1 - x), abs(y - Y0), abs(Y1 - y)) < 0.7:
            return "wall"
    return best


def build_grid():
    xs = np.arange(X0 + 0.4, X1 - 0.4 + GRID_M, GRID_M)
    ys = np.arange(Y0 + 0.4, Y1 - 0.4 + GRID_M, GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    drivable = np.ones(gx.shape, dtype=bool)
    for _n, cx, cy, sx, sy, _h in obstacles():
        drivable &= ~((np.abs(gx - cx) <= sx / 2 + ROBOT_CLEARANCE_M)
                      & (np.abs(gy - cy) <= sy / 2 + ROBOT_CLEARANCE_M))
    return xs, ys, gx, gy, drivable


def camera_model(x, y, z, yaw_deg, pitch_deg=None):
    pitch = math.radians(CAMERA_PITCH_DEG if pitch_deg is None else pitch_deg)
    yaw = math.radians(yaw_deg)
    fwd = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
           -math.sin(pitch))
    scale = -z / fwd[2]
    return ObliqueCameraModel(cam_pos=(x, y, z),
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
    boxes = occluders()
    mx0, my0, mx1, my1 = MEZZANINE
    for t in np.linspace(0.03, 0.97, 34):
        s = cam[None, :] * (1.0 - t) + tgt * t
        for _n, cx, cy, sx, sy, h in boxes:
            inside = (np.abs(s[:, 0] - cx) <= sx / 2) & (np.abs(s[:, 1] - cy) <= sy / 2)
            clear &= ~(inside & (s[:, 2] < h))
        # The mezzanine slab: blocks anything looking down through it.
        under_deck = ((s[:, 0] >= mx0) & (s[:, 0] <= mx1)
                      & (s[:, 1] >= my0) & (s[:, 1] <= my1))
        clear &= ~(under_deck & (s[:, 2] > MEZZANINE_DECK_Z))
    return clear.reshape(gx.shape)


# ====================================================== retrofit placement
# The inherited six are fixed by the storyline. The 2024 retrofit budget should be
# spent where it buys the most, which is NOT where hand-placement put it: an earlier
# version aimed two retrofits along the haul lane, which was already covered by the
# dock cameras, leaving the picking aisles 87 % blind.
#
# Selection is greedy over a pool of physically mountable ceiling positions, scoring
# each cell with DIMINISHING RETURNS so a third or fourth look at the same floor is
# worth almost nothing. That is what stops the optimiser from stacking overlap.

#: value of having exactly k cameras on a cell. w(2) > w(1) because cross-checking
#: needs a second view; beyond two the gain is nearly flat.
#: A second view is worth having — cross-checking a leaning camera needs one. A THIRD
#: is worth nothing: it cannot resolve a disagreement the second did not already
#: expose, and it costs a camera. Flat after 2 is what stops the optimiser blanketing
#: the haul lane, which 6.1 m / 52.7 deg wall mounts do naturally (45 m reach).
COVERAGE_VALUE = {0: 0.0, 1: 1.00, 2: 1.30, 3: 1.30, 4: 1.30}

#: Cells are weighted by whether the ROBOT GOES THERE. Maximising covered floor area
#: is the wrong objective and produced the wrong answer: the haul lane is 90 m^2 and
#: a picking aisle is 6 m^2, so an area-greedy optimiser spends the whole budget on
#: lanes and leaves the aisles blind — which is exactly the imbalance in the
#: hand-placed network (87 % blind aisles, 71 % redundant haul lane).
OPERATING_CORRIDOR_M = 1.6
OFF_ROUTE_WEIGHT = 0.12


def importance_weights(gx, gy, drivable):
    """1.0 within the operating corridor of the mission, small elsewhere."""
    pts = [(m["x"], m["y"]) for m in MISSION]
    dist = np.full(gx.shape, np.inf)
    for (ax0, ay0), (ax1, ay1) in zip(pts[:-1], pts[1:]):
        dx, dy = ax1 - ax0, ay1 - ay0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-9:
            continue
        s = np.clip(((gx - ax0) * dx + (gy - ay0) * dy) / seg2, 0.0, 1.0)
        dist = np.minimum(dist, np.hypot(gx - (ax0 + s * dx), gy - (ay0 + s * dy)))
    w = np.where(dist <= OPERATING_CORRIDOR_M, 1.0, OFF_ROUTE_WEIGHT)
    return np.where(drivable, w, 0.0)


def _cell_value(counts):
    v = np.full(counts.shape, COVERAGE_VALUE[4], dtype=float)
    for k in range(4):
        v = np.where(counts == k, COVERAGE_VALUE[k], v)
    return v


# ------------------------------------------------------------------- mounting
# A camera has to be BOLTED TO SOMETHING. The first version of this pool let
# candidates hang in mid-floor: six of twelve cameras ended up 2-6 m from the
# nearest wall or column, which is why the placement did not make sense however
# good its coverage looked. Everything below is a surface an installer can reach,
# with power and data already near it.

MOUNT_WALL_INSET = 0.35
RACK_MOUNT_Z = 4.60          # on a rack end frame, below the top of a 5.2 m run
COLUMN_MOUNT_Z = 6.10        # near the top of a building column


def mount_points():
    """Every physically mountable camera position, tagged with what it bolts to."""
    out = []                                    # (tag, kind, x, y, z, yaw, pitch)
    wi = MOUNT_WALL_INSET

    # --- walls: ON THE AISLE CENTRELINES first -------------------------------
    # A wall mount whose x does not line up with an aisle stares down the length of
    # a rack run and sees nothing useful. The rack pitch is 2.70 m, so a 3 m sampling
    # grid never aligns — that mistake put the first north-wall camera directly above
    # a rack edge, which the rendered Gazebo view showed straight away.
    for i, ax in enumerate(AISLE_X):
        out.append((f"wS_aisle{i}", "wall", float(ax), Y0 + wi, 6.10, 90, 52.7))
        out.append((f"wN_aisle{i}", "wall", float(ax), Y1 - wi, 6.10, -90, 52.7))
    out.append(("wS_west", "wall", WEST_LANE_X, Y0 + wi, 6.10, 90, 52.7))
    out.append(("wN_west", "wall", WEST_LANE_X, Y1 - wi, 6.10, -90, 52.7))

    # --- walls: a coarser grid for the open zones ----------------------------
    for x in np.arange(X0 + 3.0, X1 - 2.9, 3.0):
        out.append((f"wS{x:+.0f}", "wall", float(x), Y0 + wi, 6.10, 90, 52.7))
        out.append((f"wN{x:+.0f}", "wall", float(x), Y1 - wi, 6.10, -90, 52.7))
    for y in np.arange(Y0 + 3.0, Y1 - 2.9, 3.0):
        out.append((f"wW{y:+.0f}", "wall", X0 + wi, float(y), 6.10, 0, 52.7))
        out.append((f"wE{y:+.0f}", "wall", X1 - wi, float(y), 6.10, 180, 52.7))

    # --- building columns: four aims each ------------------------------------
    for cx in COLUMN_XS:
        for cy in COLUMN_YS:
            for yaw in (0, 90, 180, -90):
                out.append((f"col{cx:+.0f}{cy:+.0f}y{yaw}", "column",
                            cx, cy, COLUMN_MOUNT_Z, yaw, 52.7))

    # --- rack END FRAMES at the aisle mouths ---------------------------------
    # The standard way to watch an aisle: bolt the camera to the end frame of a
    # rack run and look along the aisle. Mounted on the aisle-side face, so the
    # position is the rack edge, not the aisle centreline.
    for i, ax in enumerate(AISLE_X):
        west_face = _rack_x[i] + RACK_DEPTH / 2 + 0.06
        for band, (y_lo, y_hi) in (("S", RACK_BAND_S), ("N", RACK_BAND_N)):
            out.append((f"rk{i}{band}s", "rack end frame", west_face, y_lo - 0.12,
                        RACK_MOUNT_Z, 90, 60.0))
            out.append((f"rk{i}{band}n", "rack end frame", west_face, y_hi + 0.12,
                        RACK_MOUNT_Z, -90, 60.0))

    # --- mezzanine edge beam and underside ----------------------------------
    mx0, my0, mx1, my1 = MEZZANINE
    for x in (mx0 + 1.6, 0.5 * (mx0 + mx1), mx1 - 1.6):
        out.append((f"mezz{x:+.0f}", "mezzanine edge", float(x), my1 - 0.1, 2.90,
                    -100, 62.0))
    return out


def candidate_mounts():
    """The retrofit candidate pool: real mounting surfaces only.

    Two mount classes, because one geometry does not serve both jobs:
      WALL/COLUMN  6.1 m at 52.7 deg — the deployed rig. A 90 deg FOV at that pitch
                   gives a long narrow footprint reaching ~45 m: right for a lane.
      RACK END     4.6 m at 60 deg — shorter and steeper, bolted to a rack end frame
                   at an aisle mouth, looking along the aisle.
    """
    return [(tag, x, y, z, yaw, pitch)
            for tag, _kind, x, y, z, yaw, pitch in mount_points()]


def choose_retrofits(gx, gy, drivable, inherited_masks, budget):
    base = np.zeros(gx.shape, dtype=int)
    for m in inherited_masks:
        base += m.astype(int)
    weights = importance_weights(gx, gy, drivable)
    pool = candidate_mounts()
    masks = {}
    for name, x, y, z, yaw, pitch in pool:
        model = camera_model(x, y, z, yaw, pitch_deg=pitch)
        masks[name] = (in_frame(model, gx, gy) & line_of_sight(model, gx, gy)
                       & drivable)
    chosen, counts = [], base.copy()
    for _ in range(budget):
        current = float((_cell_value(counts) * weights).sum())
        best, best_gain = None, 0.0
        for cand in pool:
            if cand[0] in [c[0] for c in chosen]:
                continue
            gain = float((_cell_value(counts + masks[cand[0]].astype(int))
                          * weights).sum()) - current
            if gain > best_gain:
                best, best_gain = cand, gain
        if best is None or best_gain <= 1e-6:
            break
        chosen.append((*best, best_gain))
        counts += masks[best[0]].astype(int)
    return chosen, counts


# ================================================================== the plan


def draw_plan(ax, *, with_fov=False):
    for z in ZONES:
        fd.zone(ax, z["x0"], z["y0"], z["x1"], z["y1"], name=z["name"], kind=z["kind"])
    for pts in WALKWAYS:
        fd.walkway(ax, pts, width=1.1)

    fd.wall_poche(ax, X0, Y0, X1, Y1, thickness=0.45)
    for x0, x1, tag in DOCK_DOORS:
        fd.dock_door(ax, x0, x1, Y0, thickness=0.45, label=tag)

    fd.mezzanine(ax, *MEZZANINE, label="QC OFFICE MEZZANINE (2024)")

    for e in ELEMENTS:
        if e["kind"] == "rack":
            fd.rack_run(ax, e["x"], e["y"], e["sx"], e["sy"], tall=e.get("tall", False))
        elif e["kind"] == "block":
            fd.block_stack(ax, e["x"], e["y"], e["sx"], e["sy"])
        elif e["kind"] == "machine":
            fd.machine(ax, e["x"], e["y"], e["sx"], e["sy"],
                       kind=e.get("glyph", "box"))
        elif e["kind"] == "station":
            fd.station(ax, e["x"], e["y"], e["sx"], e["sy"])
        elif e["kind"] == "cage":
            fd.cage(ax, e["x"], e["y"], e["sx"], e["sy"], label="HV CAGE")
        elif e["kind"] == "prop":
            fd.machine(ax, e["x"], e["y"], e["sx"], e["sy"], kind="box")

    fd.conveyor(ax, CONVEYOR, width=0.8, label="take-away conveyor")
    for pts in GUARDRAILS:
        fd.guardrail(ax, pts)
    fd.column_grid(ax, COLUMN_XS, COLUMN_YS, size=COLUMN_SIZE, tags=True)

    label_offsets = {
        "dock_in": (-34, -2), "dock_out": (34, -2), "cage_hv": (-36, 2),
        "returns_e": (-40, 6), "pack_mezz": (30, -12), "office_ne": (-40, -6),
        "north_wlane": (26, -6), "aisle2_n": (0, -15), "south_wlane": (26, 6),
        "west_w": (28, -6), "cross_col": (30, 4), "haul_col": (0, 15),
    }
    for name, x, y, z, yaw, prov, _note in CAMERAS:
        pitch = CAMERA_PITCH.get(name, CAMERA_PITCH_DEG)
        reach = min(14.0, z / math.tan(math.radians(max(pitch - 45.0, 4.0))))
        fd.camera(ax, x, y, yaw, reach=reach, fov_deg=90.0,
                  inherited=(prov == "inherited"), label=name, show_fov=with_fov,
                  label_offset=label_offsets.get(name, (0, -13)),
                  clip_to=(X0, Y0, X1, Y1))

    fd.style_axes(ax, X0, Y0, X1, Y1, margin=2.0)


def annotate_plan(ax, *, legend_ax=None):
    fd.route(ax, [(m["x"], m["y"]) for m in MISSION], label=None)
    # Dimensions read against clear floor, not over the racking.
    fd.dimension(ax, (_rack_x[0], Y1 + 0.55), (_rack_x[1], Y1 + 0.55),
                 label=f"rack pitch {RACK_PITCH:.2f} m", offset=0.0)
    fd.dimension(ax, (_rack_x[3] + RACK_DEPTH / 2, RACK_BAND_N[0] - 0.5),
                 (_rack_x[4] - RACK_DEPTH / 2, RACK_BAND_N[0] - 0.5),
                 label=f"aisle {PICK_AISLE:.2f} m", offset=0.0)
    fd.dimension(ax, (X0, Y1 + 1.9), (X1, Y1 + 1.9),
                 label=f"{X1 - X0:.0f} m", offset=0.0)
    fd.dimension(ax, (X0 - 1.5, Y0), (X0 - 1.5, Y1), label=f"{Y1 - Y0:.0f} m",
                 offset=0.0)
    fd.scale_bar(ax, X0 + 0.4, Y0 - 1.9, length=5.0)
    fd.north_arrow(ax, X1 + 1.15, Y1 - 0.6, size=0.85)
    (legend_ax or ax).axis("off") if legend_ax else None
    fd.legend_panel(legend_ax or ax, [
        ("wall", "1983 shell / structure"), ("rack", "selective racking 2.6 m"),
        ("rack_tall", "double-stacked 5.2 m (blocks view)"),
        ("block", "block-stacked goods (opaque, moves weekly)"),
        ("machine", "fixed plant"), ("station", "pick / pack / returns bench"),
        ("dock", "dock door"), ("walkway", "pedestrian walkway"),
        ("guardrail", "guardrail"), ("cam_inherited", "camera — inherited (security)"),
        ("cam_retrofit", "camera — 2024 robot retrofit"),
        ("route", "AMR tote run (numbered)"),
        ("other", "high-value cage (mesh, see-through)"),
    ], loc="upper left" if True else "lower left")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 170, "savefig.dpi": 170, "font.size": 8})

    xs, ys, gx, gy, drivable = build_grid()

    # Validate the mission against the drivable grid rather than by eye. An earlier
    # version had a waypoint inside a rack run and a leg cutting across the racking.
    bad = []
    for m in MISSION:
        i = int(np.argmin(np.abs(ys - m["y"])))
        j = int(np.argmin(np.abs(xs - m["x"])))
        if not drivable[i, j]:
            bad.append(f"{m['wp']} ({m['x']:.1f}, {m['y']:.1f})")
    leg_blocked = []
    pts = [(m["x"], m["y"]) for m in MISSION]
    for (ax0, ay0), (ax1, ay1) in zip(pts[:-1], pts[1:]):
        n = max(2, int(math.hypot(ax1 - ax0, ay1 - ay0) / 0.15))
        for s in np.linspace(0.0, 1.0, n):
            px, py = ax0 + s * (ax1 - ax0), ay0 + s * (ay1 - ay0)
            i = int(np.argmin(np.abs(ys - py)))
            j = int(np.argmin(np.abs(xs - px)))
            if not drivable[i, j]:
                leg_blocked.append(f"({ax0:.1f},{ay0:.1f})->({ax1:.1f},{ay1:.1f})"
                                   f" at ({px:.1f},{py:.1f})")
                break
    if bad:
        print("MISSION WAYPOINTS NOT DRIVABLE:", "; ".join(bad))
    if leg_blocked:
        print("MISSION LEGS CROSSING OBSTACLES:", "; ".join(sorted(set(leg_blocked))))
    if not bad and not leg_blocked:
        print("mission route validated: all waypoints and legs are drivable")

    # What would a measured placement choose for the retrofit budget?
    inherited_masks = []
    for name, x, y, z, yaw, prov, _note in CAMERAS:
        if prov != "inherited":
            continue
        model = camera_model(x, y, z, yaw, pitch_deg=CAMERA_PITCH.get(name))
        inherited_masks.append(in_frame(model, gx, gy)
                               & line_of_sight(model, gx, gy) & drivable)
    n_retrofit = sum(1 for c in CAMERAS if c[5] == "retrofit")
    picked, _pc = choose_retrofits(gx, gy, drivable, inherited_masks, 10)
    # Every camera must be bolted to something. This check exists because six of
    # twelve were once hanging in mid-floor.
    unmounted = [(nm, cx, cy) for nm, cx, cy, cz, _yw, _pv, _nt in CAMERAS
                 if mount_of(cx, cy, cz) is None]
    if unmounted:
        print("CAMERAS MOUNTED ON NOTHING: "
              + "; ".join(f"{nm} ({cx:+.1f},{cy:+.1f})" for nm, cx, cy in unmounted))
    else:
        print("all cameras are bolted to a wall, column, rack end frame or the "
              "mezzanine")

    print(f"\nroute-weighted greedy retrofit placement ({n_retrofit} + 3 shown):")
    first_gain = picked[0][-1] if picked else 1.0
    for i, (nm, px, py, pz, pyaw, ppitch, gain) in enumerate(picked, 1):
        print(f"   {i}. {nm:<16} ({px:+6.1f}, {py:+5.1f}) z {pz:.1f} yaw {pyaw:+4.0f} "
              f"pitch {ppitch:.0f}   marginal gain {100 * gain / first_gain:5.1f}% "
              f"of the first")

    counts = np.zeros(gx.shape, dtype=int)
    inherited = np.zeros(gx.shape, dtype=int)
    per_camera = {}
    for name, x, y, z, yaw, prov, _note in CAMERAS:
        model = camera_model(x, y, z, yaw, pitch_deg=CAMERA_PITCH.get(name))
        seen = in_frame(model, gx, gy) & line_of_sight(model, gx, gy) & drivable
        per_camera[name] = float(seen.sum() / max(drivable.sum(), 1))
        counts += seen.astype(int)
        if prov == "inherited":
            inherited += seen.astype(int)

    def region(x0, y0, x1, y1):
        m = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1) & drivable
        if not m.any():
            return None
        return {"unseen": float(np.mean(counts[m] == 0)),
                "single": float(np.mean(counts[m] == 1)),
                "redundant": float(np.mean(counts[m] >= 2)),
                "mean": float(np.mean(counts[m]))}

    # The metric that matters for an assistive service: coverage along the corridor
    # the robot actually drives. Whole-floor coverage counts aisles the robot never
    # enters, and blind floor it never visits costs nothing.
    corridor = importance_weights(gx, gy, drivable) >= 0.99
    corridor_stats = {
        "cells": int(corridor.sum()),
        "unseen": float(np.mean(counts[corridor] == 0)),
        "single": float(np.mean(counts[corridor] == 1)),
        "two": float(np.mean(counts[corridor] == 2)),
        "three_plus": float(np.mean(counts[corridor] >= 3)),
        "mean": float(np.mean(counts[corridor])),
    }

    showcase = {
        "pick aisles (blind runs)": region(-13.0, 3.4, 2.6, 7.4),
        "returns bay (block stacks)": region(4.2, 0.6, 14.6, 9.6),
        "under mezzanine (single camera)": region(7.6, -9.4, 14.4, -4.8),
        "cross-aisle junction (overlap)": region(-3.0, -3.8, 3.0, -1.8),
        "pack infeed (high precision)": region(4.4, -9.6, 6.8, -8.2),
        "returns bench 2 (precision, 1 cam?)": region(11.4, 3.4, 13.6, 5.4),
        "haul lane (low precision)": region(-14.0, -3.8, 14.0, -1.8),
    }

    stats = {
        "site": "Meerhoven Regional Hub",
        "building_m": [X1 - X0, Y1 - Y0],
        "capture_cost_vs_current": round(((X1 - X0) * (Y1 - Y0)) / (24 * 20), 2),
        "drivable_area_m2": float(drivable.sum() * GRID_M**2),
        "element_count": len(ELEMENTS),
        "element_types": sorted({e["kind"] for e in ELEMENTS}),
        "cameras": {"inherited": sum(1 for c in CAMERAS if c[5] == "inherited"),
                    "retrofit": sum(1 for c in CAMERAS if c[5] == "retrofit")},
        "whole_floor": {
            "unseen": float(np.mean(counts[drivable] == 0)),
            "single": float(np.mean(counts[drivable] == 1)),
            "redundant": float(np.mean(counts[drivable] >= 2)),
            "mean": float(np.mean(counts[drivable])),
        },
        "inherited_only_unseen": float(np.mean(inherited[drivable] == 0)),
        "per_camera_floor_share": per_camera,
        "showcase_regions": showcase,
        "operating_corridor": corridor_stats,
    }
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- figure A: the facility plan -------------------------------------
    figA = plt.figure(figsize=(18.6, 10.6))
    gsA = figA.add_gridspec(1, 2, width_ratios=[1.0, 0.235], wspace=0.02)
    axA = figA.add_subplot(gsA[0, 0])
    axL = figA.add_subplot(gsA[0, 1])
    axL.axis("off")
    draw_plan(axA, with_fov=False)
    annotate_plan(axA, legend_ax=axL)
    axA.set_title("MEERHOVEN REGIONAL HUB — facility plan\n"
                  "1983 shell repurposed 2024 · outbound pick-pack + unplanned "
                  "returns bay · 18 AMRs, 9 pickers",
                  fontsize=12.5, fontweight="bold", pad=16)
    fd.title_block(figA, project="Infrastructure-assisted AMR localization",
                   drawing="A-01 facility plan", revision="A",
                   notes=f"rack {RACK_DEPTH:.2f} m · pick aisle {PICK_AISLE:.2f} m · "
                         f"cross aisle {CROSS_AISLE:.1f} m · haul lane {HAUL_LANE:.1f} m "
                         f"· columns 8 m grid · {len(ELEMENTS)} elements")
    figA.tight_layout()
    for ext in ("png", "pdf"):
        figA.savefig(OUT / f"fig_h1_facility_plan.{ext}", bbox_inches="tight")
    plt.close(figA)

    # ---- figure B: the camera network -------------------------------------
    figB, axB = plt.subplots(1, 2, figsize=(19.0, 8.2))
    draw_plan(axB[0], with_fov=True)
    n_inh = sum(1 for c in CAMERAS if c[5] == "inherited")
    n_ret = len(CAMERAS) - n_inh
    axB[0].set_title(f"Camera network — {n_inh} inherited (red) + {n_ret} retrofit "
                     f"(green)\nwedges are floor footprints, clipped to the shell; "
                     "reach follows each mount's height and pitch",
                     fontsize=10.5, fontweight="bold")

    shaded = np.where(drivable, np.minimum(counts, 4), np.nan)
    mesh = axB[1].pcolormesh(xs, ys, shaded, cmap="YlGnBu", vmin=0, vmax=4,
                             shading="auto", zorder=0)
    for e in ELEMENTS:
        if e["kind"] == "rack":
            fd.rack_run(axB[1], e["x"], e["y"], e["sx"], e["sy"],
                        tall=e.get("tall", False), bays=1)
        elif e["kind"] == "block":
            fd.block_stack(axB[1], e["x"], e["y"], e["sx"], e["sy"])
    fd.wall_poche(axB[1], X0, Y0, X1, Y1, thickness=0.45)
    fd.column_grid(axB[1], COLUMN_XS, COLUMN_YS, size=COLUMN_SIZE, tags=False)
    for name, x, y, z, yaw, prov, _note in CAMERAS:
        fd.camera(axB[1], x, y, yaw, inherited=(prov == "inherited"), show_fov=False)
    fd.route(axB[1], [(m["x"], m["y"]) for m in MISSION])
    fd.style_axes(axB[1], X0, Y0, X1, Y1, margin=1.2)
    w = stats["whole_floor"]
    axB[1].set_title(f"How many cameras see each point\n"
                     f"unseen {100 * w['unseen']:.0f} % · single "
                     f"{100 * w['single']:.0f} % · redundant "
                     f"{100 * w['redundant']:.0f} %   "
                     f"(inherited alone: {100 * stats['inherited_only_unseen']:.0f} % "
                     f"unseen)", fontsize=10.5, fontweight="bold")
    cb = figB.colorbar(mesh, ax=axB[1], shrink=0.75, ticks=range(5))
    cb.set_label("cameras with line of sight")
    cb.ax.set_yticklabels(["0", "1", "2", "3", "4+"])
    figB.suptitle("Meerhoven Regional Hub — what an inherited network actually covers",
                  fontsize=12.5, fontweight="bold")
    figB.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        figB.savefig(OUT / f"fig_h2_camera_network.{ext}", bbox_inches="tight")
    plt.close(figB)

    print(f"MEERHOVEN REGIONAL HUB — {X1 - X0:.0f} x {Y1 - Y0:.0f} m, "
          f"{stats['drivable_area_m2']:.0f} m2 drivable, {len(ELEMENTS)} elements "
          f"({', '.join(stats['element_types'])})")
    print(f"capture cost vs current world: {stats['capture_cost_vs_current']}x")
    print(f"cameras: {stats['cameras']['inherited']} inherited + "
          f"{stats['cameras']['retrofit']} retrofit")
    print(f"whole floor : unseen {100 * w['unseen']:.1f}%  single "
          f"{100 * w['single']:.1f}%  redundant {100 * w['redundant']:.1f}%  "
          f"mean {w['mean']:.2f}")
    print(f"inherited only: unseen {100 * stats['inherited_only_unseen']:.1f}%")
    c = corridor_stats
    print(f"operating corridor ({c['cells']} cells, {OPERATING_CORRIDOR_M} m either "
          f"side of the route):")
    print(f"   unseen {100 * c['unseen']:.1f}%  single {100 * c['single']:.1f}%  "
          f"two {100 * c['two']:.1f}%  three+ {100 * c['three_plus']:.1f}%  "
          f"mean {c['mean']:.2f}")
    print("\nshowcase regions:")
    for name, r in showcase.items():
        if r is None:
            print(f"   {name:<36} (no drivable cells)")
            continue
        print(f"   {name:<36} unseen {100 * r['unseen']:5.1f}%  "
              f"single {100 * r['single']:5.1f}%  redundant "
              f"{100 * r['redundant']:5.1f}%  mean {r['mean']:.2f}")
    print("\nper-camera share of drivable floor:")
    for name, x, y, z, yaw, prov, _note in CAMERAS:
        print(f"   {name:<13}{100 * per_camera[name]:>6.1f}%  ({prov})")
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
