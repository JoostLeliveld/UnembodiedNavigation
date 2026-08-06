#!/usr/bin/env python3
"""Generate warehouse_full_4cam.world.sdf - the flagship: a fully-dressed AWS-mesh
warehouse at ~24 x 20 m with FOUR wall-mounted external cameras.

Style = warehouse_aws (the 'old' world): box-collision racks with contact sensors
(planner/logger/CAD-parser compatible) + blue top rails + tan shelf boxes + REAL
AWS RoboMaker shelf-mesh visual overlays (ShelfD/E, measured native dims
3.917 x 0.880 x 2.613 m), docks, staging clutter (real mesh models), hanging lamps,
and floor safety markings: blue site bounds plus conservative green no-go
envelopes around operational obstacles.

Cameras are placed like fixed warehouse operations/security cameras along the
north/south walls, centered over four dock-apron columns:
  external_camera    (-6.0, -10.0), south wall, looking north
  external_camera_b  (-6.0,  10.0), north wall, looking south
  external_camera_c  ( 6.0, -10.0), south wall, looking north
  external_camera_d  ( 6.0,  10.0), north wall, looking south
This avoids a lab-like centerline camera pair while bringing the two columns
slightly closer to the central aisles. That deliberately widens the horizontal
overlap available for validation and source handover. The rack blocks are spaced
around a 4.5 m central aisle rather than separated by a broad empty hall, so the
layout reads as one warehouse with real handover corridors. A west-wall backed
shelf is flush with the wall and reachable only from its aisle side. Adjacent
fields overlap around the center cross aisles, while opposite wall pairs create
handover regions through rack occlusion. Tall (2.61 m) rack segments are patterned
asymmetrically so the cameras disagree in useful ways:
W-block NORTH segments tall, E-block SOUTH segments tall.

Collision boxes match the mesh visual heights, so physics = CAD prisms = visuals.
Occluder heights remain EVALUATION-ONLY inputs downstream.
"""
from __future__ import annotations
import math
import os
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DOC = REPO / "docs/warehouse_full_4cam_layout.md"
SVG = REPO / "docs/assets/warehouse_full_4cam_map.svg"
PNG = REPO / "docs/assets/warehouse_full_4cam_map.png"
PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
WORLD_FILE = "warehouse_full_4cam.world.sdf"

RACK_YELLOW = "0.93 0.80 0.08 1"
RAIL_BLUE = "0.00 0.28 0.50 1"
BOX_TAN = "0.75 0.56 0.24 1"
WALL_GRAY = "0.62 0.64 0.67 1"
FLOOR = "0.56 0.55 0.51 1"
BLUE_MARK = "0.05 0.30 0.85 1"
GREEN_MARK = "0.05 0.55 0.20 1"

# native mesh dims (measured from the DAEs, unit=0.01 applied)
MESH_L, MESH_W, MESH_H = 3.917, 0.880, 2.613
RACK_W = 0.55
H_STD, H_TALL = 2.09, 2.61

X0, X1, Y0, Y1 = -12.0, 12.0, -10.0, 10.0
CENTRAL_CLEARANCE_M = 4.50
ROW_SPACING = 2.10
INNER_ROW_ABS_X = CENTRAL_CLEARANCE_M / 2.0 + RACK_W / 2.0
ROWS_W = [-(INNER_ROW_ABS_X + 3 * ROW_SPACING), -(INNER_ROW_ABS_X + 2 * ROW_SPACING), -(INNER_ROW_ABS_X + ROW_SPACING), -INNER_ROW_ABS_X]
ROWS_E = [INNER_ROW_ABS_X, INNER_ROW_ABS_X + ROW_SPACING, INNER_ROW_ABS_X + 2 * ROW_SPACING, INNER_ROW_ABS_X + 3 * ROW_SPACING]
WALL_BACKED_ROW_X = X0 + 0.09 + RACK_W / 2.0
SITE_X0, SITE_X1, SITE_Y0, SITE_Y1 = -11.20, 11.50, -8.60, 8.60
# The 1.60 m raw cross-aisles retain 0.96 m after applying the 0.32 m
# operational envelope on each adjoining rack end. The former 1.20 m gaps left
# only 0.56 m between the painted no-go lines and almost no useful planner
# corridor after its 0.25 m keep-in margin.
SEGS = [("south", -6.5, -2.8), ("mid", -1.2, 1.6), ("north", 3.2, 6.9)]

NON_DRIVEABLE_SPOTS = [
    ("central_support_pillar", 0.0, -0.90, 0.50, 0.50, "fixed building column inside the 4.5 m central aisle"),
    ("west_wall_backed_shelf", WALL_BACKED_ROW_X, 0.20, RACK_W, 13.40, "ShelfD/E row backed directly against the west wall; reachable only from the east aisle side"),
]

# Floor paint represents a conservative operational exclusion envelope, not a
# tracing of mesh/collision geometry.  The margin is deliberately wider than
# the TurtleBot footprint and visually separates the safety contract from the
# exact shelf/box dimensions.
NOGO_MARGIN_M = 0.32

CAMERA_ITEMS = [
    (
        "Camera A",
        "external_camera",
        "external_camera",
        (-6.00, -10.00, 6.10, 0.0, 0.92, 1.5708),
        "/external_camera/image_raw",
        "#2f80ed",
        "south wall, west dock column",
    ),
    (
        "Camera B",
        "external_camera_b",
        "external_camera_b",
        (-6.00, 10.00, 6.10, 0.0, 0.92, -1.5708),
        "/external_camera_b/image_raw",
        "#27ae60",
        "north wall, west dock column",
    ),
    (
        "Camera C",
        "external_camera_c",
        "external_camera_c",
        (6.00, -10.00, 6.10, 0.0, 0.92, 1.5708),
        "/external_camera_c/image_raw",
        "#9b51e0",
        "south wall, east dock column",
    ),
    (
        "Camera D",
        "external_camera_d",
        "external_camera_d",
        (6.00, 10.00, 6.10, 0.0, 0.92, -1.5708),
        "/external_camera_d/image_raw",
        "#f2994a",
        "north wall, east dock column",
    ),
]

CLUTTER_ITEMS = [
    ("bucket_dock", "Bucket_01", "11.2 -8.9 0 0 0 0", "bucket"),
    ("trashcan_nw", "TrashCanC_01", "-11.3 9.0 0 0 0 0.6", "trash"),
    ("clutterA_sw", "ClutteringA_01", "-10.8 -9.2 0 0 0 0", "clutter A"),
    ("clutterC_ne", "ClutteringC_01", "10.6 9.0 0 0 0 3.1416", "clutter C"),
    ("clutterD_dock", "ClutteringD_01", "-2.5 -9.3 0 0 0 0", "clutter D"),
    ("palletjack_apron", "PalletJackB_01", "-4.6 -9.1 0 0 0 0", "pallet jack"),
    ("desk_qc_ne", "DeskC_01", "8.75 8.85 0 0 0 3.1416", "QC desk"),
]

PROP_SIZES = {
    "Bucket_01": (0.42, 0.42),
    "TrashCanC_01": (0.55, 0.55),
    "ClutteringA_01": (0.90, 0.75),
    "ClutteringC_01": (1.00, 0.80),
    "ClutteringD_01": (0.90, 0.80),
    "PalletJackB_01": (1.35, 0.65),
    "DeskC_01": (1.55, 0.85),
}

LAMP_POSES = [(-6.0, -4.0), (6.0, -4.0), (-6.0, 4.0), (6.0, 4.0)]

links, mesh_links, occluders = [], [], []


def box_link(name, cx, cy, sx, sy, sz, color, *, collide=True, contact=True, z=None, visual=True):
    zc = sz / 2 if z is None else z
    parts = [f'      <link name="{name}"><pose>{cx:.3f} {cy:.3f} {zc:.3f} 0 0 0</pose>']
    if collide:
        parts.append(f'<collision name="collision"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>')
        if contact:
            parts.append('<sensor name="contact" type="contact"><contact><collision>collision</collision></contact><update_rate>60</update_rate></sensor>')
    if visual:
        parts.append(f'<visual name="visual"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>'
                     f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>')
    parts.append('</link>')
    return "".join(parts)


def pose_xy_from_text(pose: str) -> tuple[float, float]:
    vals = [float(v) for v in pose.split()]
    return vals[0], vals[1]


def profile_regions(region_type: str) -> list[dict]:
    """Return the exact map rectangles consumed by the runtime planner."""
    data = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    regions = data["worlds"][WORLD_FILE].get("known_2d_regions", [])
    return [region for region in regions if str(region.get("type", "")) == region_type]


def outline_links(name, cx, cy, sx, sy, color=GREEN_MARK, *, pad=NOGO_MARGIN_M, thickness=0.055):
    ox, oy = sx + 2.0 * pad, sy + 2.0 * pad
    return [
        box_link(f"{name}_outline_s", cx, cy - oy / 2.0, ox, thickness, 0.014, color, collide=False, z=0.035),
        box_link(f"{name}_outline_n", cx, cy + oy / 2.0, ox, thickness, 0.014, color, collide=False, z=0.035),
        box_link(f"{name}_outline_w", cx - ox / 2.0, cy, thickness, oy, 0.014, color, collide=False, z=0.035),
        box_link(f"{name}_outline_e", cx + ox / 2.0, cy, thickness, oy, 0.014, color, collide=False, z=0.035),
    ]


def rack_segment(row, seg, xc, ya, yb, h, mesh):
    """AWS-house-style rack: collision box (invisible; the mesh is the look) +
    blue rails + a tan shelf box on top + scaled real shelf-mesh visual."""
    L = yb - ya; yc = (ya + yb) / 2
    name = f"{row}_{seg}"
    links.append(box_link(f"rack_{name}", xc, yc, RACK_W, L, h, RACK_YELLOW, visual=False))
    occluders.append((name, xc, yc, RACK_W, L, h))
    for side, dx in (("west", -RACK_W / 2 + 0.02), ("east", RACK_W / 2 - 0.02)):
        links.append(box_link(f"rail_{name}_{side}", xc + dx, yc, 0.04, L, 0.10, RAIL_BLUE,
                              collide=False, z=h + 0.07))
    links.append(box_link(f"shelfbox_{name}", xc, yc - L / 4, 0.39, 0.27, 0.16, BOX_TAN,
                          collide=False, z=h + 0.12 + 0.08))
    sx, sy, sz = L / MESH_L, RACK_W / MESH_W, h / MESH_H
    mesh_links.append(
        f'      <link name="mesh_{name}"><pose>{xc:.3f} {yc:.3f} 0 0 0 0</pose>'
        f'<visual name="visual"><pose>0 0 0 0 0 1.5708</pose><geometry><mesh>'
        f'<uri>model://aws_robomaker_warehouse_{mesh}/meshes/aws_robomaker_warehouse_{mesh}_visual.DAE</uri>'
        f'<scale>{sx:.4f} {sy:.4f} {sz:.4f}</scale></mesh></geometry></visual></link>')


# ---------------- rack blocks: tall pattern is camera-asymmetric --------------
for bi, (rows, blk) in enumerate([(ROWS_W, "W"), (ROWS_E, "E")]):
    for ri, xc in enumerate(rows, 1):
        for seg, ya, yb in SEGS:
            inner = ri in (2, 3)
            tall = inner and ((blk == "W" and seg == "north") or (blk == "E" and seg == "south"))
            h = H_TALL if tall else H_STD
            mesh = "ShelfD_01" if (ri + len(seg)) % 2 == 0 else "ShelfE_01"
            rack_segment(f"{blk}{ri}", seg, xc, ya, yb, h, mesh)

# One-sided shelf row: backed directly against the west wall, only reachable
# from the aisle side. It deliberately uses the same ShelfD/E footprint as the
# rest of the warehouse so the camera view stays in-distribution.
for seg, ya, yb in SEGS:
    rack_segment("W0", seg, WALL_BACKED_ROW_X, ya, yb, H_STD, "ShelfD_01")

# ---------------- central feature ---------------------------------------------
links.append(box_link("support_pillar", 0.0, -0.9, 0.50, 0.50, 5.20, WALL_GRAY))
occluders.append(("support_pillar", 0.0, -0.9, 0.50, 0.50, 5.20))

rack_model = "\n".join(links)
mesh_model = "\n".join(mesh_links)

walls = "\n".join([
    box_link("wall_east", X1, 0.0, 0.18, Y1 - Y0, 4.50, WALL_GRAY, z=2.25),
    box_link("wall_west", X0, 0.0, 0.18, Y1 - Y0, 4.50, WALL_GRAY, z=2.25),
    box_link("wall_north", 0.0, Y1, X1 - X0, 0.18, 4.50, WALL_GRAY, z=2.25),
    box_link("wall_south", 0.0, Y0, X1 - X0, 0.18, 4.50, WALL_GRAY, z=2.25),
] + [box_link(f"south_dock_{c}", x, Y0 + 0.08, 1.10, 0.07, 0.15, "0.50 0.50 0.50 1", collide=False, z=0.075)
     for c, x in zip("abcd", [-7.5, -2.5, 2.5, 7.5])]
  + [box_link(f"north_dock_{c}", x, Y1 - 0.08, 1.10, 0.07, 0.15, "0.50 0.50 0.50 1", collide=False, z=0.075)
     for c, x in zip("ab", [-5.0, 5.0])])

# Blue site boundary + conservative green no-go envelopes (visual only). The
# paint includes clearance beyond collision geometry and is emitted only for
# obstacles inside the operational boundary.
marks = []
bx0, bx1, by0, by1 = SITE_X0, SITE_X1, SITE_Y0, SITE_Y1
bcx, bcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
marks += [box_link("bnd_s", bcx, by0, bx1 - bx0, 0.05, 0.014, BLUE_MARK, collide=False, z=0.035),
          box_link("bnd_n", bcx, by1, bx1 - bx0, 0.05, 0.014, BLUE_MARK, collide=False, z=0.035),
          box_link("bnd_w", bx0, bcy, 0.05, by1 - by0, 0.014, BLUE_MARK, collide=False, z=0.035),
          box_link("bnd_e", bx1, bcy, 0.05, by1 - by0, 0.014, BLUE_MARK, collide=False, z=0.035)]
for name, cx, cy, sx, sy, _h in occluders:
    marks.extend(outline_links(f"nogo_{name}", cx, cy, sx, sy))
# CLUTTER_ITEMS are already outside the blue operational boundary.  Their old
# approximate footprint outlines sometimes enclosed empty floor because the
# visual mesh origin is not its footprint centre; omit those stale markings.
marker_model = "\n".join(marks)

# clutter: full mesh model includes (own scale/collisions), all OUTSIDE drivable regions
clutter = "\n".join([
    f'    <include><name>{n}</name><uri>model://aws_robomaker_warehouse_{m}</uri><pose>{p}</pose></include>'
    for n, m, p, _label in CLUTTER_ITEMS]) + "\n" + "\n".join([
    f'    <include><name>lamp_{i}</name><uri>model://aws_robomaker_warehouse_Lamp_01</uri><pose>{x:.1f} {y:.1f} 5.95 0 0 0</pose></include>'
    for i, (x, y) in enumerate(LAMP_POSES)])


def pose_text(pose):
    x, y, z, roll, pitch, yaw = pose
    return f"{x:.2f} {y:.2f} {z:.2f} {roll:.0f} {pitch:.2f} {yaw:.4f}"


camera_includes = "\n".join([
    f'    <include><name>{include}</name><uri>model://{model}</uri><pose>{pose_text(pose)}</pose></include>'
    for _label, include, model, pose, _topic, _color, _mount in CAMERA_ITEMS
])
# This fifth camera is deliberately presentation-only: it is high enough to
# show the whole facility in one frame, but it is never part of the four-camera
# detector, GP, fusion, or planner interfaces.
overview_camera_include = (
    '    <include><name>presentation_overview_camera</name>'
    '<uri>model://presentation_overview_camera</uri>'
    '<pose>0.00 0.00 26.00 0 1.5708 0</pose></include>'
)


def _pose_xy(pose: str) -> tuple[float, float]:
    return pose_xy_from_text(pose)


def _clipped_fov_points(x: float, y: float, yaw: float) -> list[tuple[float, float]]:
    # Visual planning aid: 90 degree FOV projected as a clipped floor wedge.
    # Runtime geometry still comes from the calibrated camera model.
    span = 0.785398
    reach = 22.0

    def clipped_point(angle: float) -> tuple[float, float]:
        dx, dy = reach * math.cos(angle), reach * math.sin(angle)
        scale = 1.0
        if dx > 0:
            scale = min(scale, (X1 - x) / dx)
        elif dx < 0:
            scale = min(scale, (X0 - x) / dx)
        if dy > 0:
            scale = min(scale, (Y1 - y) / dy)
        elif dy < 0:
            scale = min(scale, (Y0 - y) / dy)
        scale = max(0.0, min(1.0, scale))
        return x + dx * scale, y + dy * scale

    return [(x, y), clipped_point(yaw - span), clipped_point(yaw + span)]


def _svg_map() -> str:
    width, height, margin = 1180, 900, 74
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    traversable = profile_regions("traversable")

    def sx(x: float) -> float:
        return margin + (x - X0) / (X1 - X0) * plot_w

    def sy(y: float) -> float:
        return margin + (Y1 - y) / (Y1 - Y0) * plot_h

    def esc(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def rect(cx, cy, rx, ry, fill, stroke="#2f2f2f", opacity=1.0, label=None, label_size=13):
        x = sx(cx - rx / 2)
        y = sy(cy + ry / 2)
        w = sx(cx + rx / 2) - x
        h = sy(cy - ry / 2) - y
        out = [
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2" opacity="{opacity:.3f}"/>'
        ]
        if label:
            out.append(
                f'<text x="{sx(cx):.1f}" y="{sy(cy) + label_size / 3:.1f}" '
                f'class="label" font-size="{label_size}" text-anchor="middle">{esc(label)}</text>'
            )
        return "\n".join(out)

    def line(xa, ya, xb, yb, color, width_px=2.0, dash=""):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<line x1="{sx(xa):.1f}" y1="{sy(ya):.1f}" x2="{sx(xb):.1f}" y2="{sy(yb):.1f}" '
            f'stroke="{color}" stroke-width="{width_px:.1f}"{dash_attr}/>'
        )

    def poly(points, fill, stroke, opacity=1.0):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" opacity="{opacity:.3f}"/>'

    def outline_rect(cx, cy, rx, ry, pad=NOGO_MARGIN_M):
        return rect(cx, cy, rx + 2.0 * pad, ry + 2.0 * pad, "none", "#138a43", 1.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="warehouse full four camera top down map">',
        "<defs>",
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#1f2933}",
        ".title{font-size:28px;font-weight:700}",
        ".subtitle{font-size:15px;fill:#405261}",
        ".label{font-weight:700;paint-order:stroke;stroke:#f8fafc;stroke-width:3px}",
        ".tiny{font-size:12px;fill:#34495e}",
        ".axis{font-size:12px;fill:#637381}",
        "</style>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#f5f7f9"/>',
        '<text x="74" y="42" class="title">warehouse_full_4cam: 24.5 x 20.5 m AWS-style warehouse</text>',
        '<text x="74" y="66" class="subtitle">Cyan planner aisles never cross green no-go envelopes; loose aisle boxes have been removed.</text>',
        rect(0, 0, X1 - X0, Y1 - Y0, "#d6d0c8", "#42505c", 1.0),
    ]

    for x in range(-12, 13, 2):
        parts.append(line(x, Y0, x, Y1, "#ffffff", 0.8, "3 7"))
        parts.append(f'<text x="{sx(x):.1f}" y="{sy(Y0) + 26:.1f}" class="axis" text-anchor="middle">x={x}</text>')
    for y in range(-10, 11, 2):
        parts.append(line(X0, y, X1, y, "#ffffff", 0.8, "3 7"))
        parts.append(f'<text x="{sx(X0) - 14:.1f}" y="{sy(y) + 4:.1f}" class="axis" text-anchor="end">y={y}</text>')

    for region in traversable:
        xmin, xmax = float(region["xmin"]), float(region["xmax"])
        ymin, ymax = float(region["ymin"]), float(region["ymax"])
        parts.append(rect(
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            xmax - xmin,
            ymax - ymin,
            "#67d5e8",
            "#1282a2",
            0.20,
        ))

    for _label, _include, _model, pose, _topic, color, _mount in CAMERA_ITEMS:
        parts.append(poly(_clipped_fov_points(pose[0], pose[1], pose[5]), color, color, 0.14))
    boundary_cx = 0.5 * (SITE_X0 + SITE_X1)
    boundary_cy = 0.5 * (SITE_Y0 + SITE_Y1)
    parts.append(rect(boundary_cx, SITE_Y0, SITE_X1 - SITE_X0, 0.07, "#1f6feb", "#1f6feb", 1.0))
    parts.append(rect(boundary_cx, SITE_Y1, SITE_X1 - SITE_X0, 0.07, "#1f6feb", "#1f6feb", 1.0))
    parts.append(rect(SITE_X0, boundary_cy, 0.07, SITE_Y1 - SITE_Y0, "#1f6feb", "#1f6feb", 1.0))
    parts.append(rect(SITE_X1, boundary_cy, 0.07, SITE_Y1 - SITE_Y0, "#1f6feb", "#1f6feb", 1.0))

    for name, cx, cy, rx, ry, h in occluders:
        if name.startswith(("W", "E")):
            fill = "#d96b27" if h >= 2.5 else "#f2c94c"
            label = name if (h >= 2.5 or name.startswith("W0")) else None
            parts.append(rect(cx, cy, rx, ry, fill, "#5b4222", 0.96, label, 11))
        elif name == "support_pillar":
            parts.append(rect(cx, cy, rx, ry, "#6b7280", "#374151", 1.0, "pillar", 11))

    for _name, model, pose, label in CLUTTER_ITEMS:
        cx, cy = _pose_xy(pose)
        rx, ry = PROP_SIZES.get(model, (0.7, 0.7))
        parts.append(rect(cx, cy, rx, ry, "#7f8c8d", "#34495e", 0.93, label, 10))

    for _name, cx, cy, rx, ry, _h in occluders:
        parts.append(outline_rect(cx, cy, rx, ry))
    for i, (x, y) in enumerate(LAMP_POSES):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="8" fill="#fff2a8" stroke="#8f7a20" stroke-width="1.4"/>')
        parts.append(f'<text x="{sx(x):.1f}" y="{sy(y) + 22:.1f}" class="tiny" text-anchor="middle">lamp {i}</text>')

    for label, include, _model, pose, topic, color, mount in CAMERA_ITEMS:
        x, y = pose[0], pose[1]
        yaw = pose[5]
        ux, uy = math.cos(yaw), math.sin(yaw)
        left_x, left_y = math.cos(yaw + 2.55), math.sin(yaw + 2.55)
        right_x, right_y = math.cos(yaw - 2.55), math.sin(yaw - 2.55)
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="12" fill="{color}" stroke="#111827" stroke-width="2"/>')
        parts.append(
            poly(
                [
                    (x + 0.62 * ux, y + 0.62 * uy),
                    (x + 0.42 * left_x, y + 0.42 * left_y),
                    (x + 0.42 * right_x, y + 0.42 * right_y),
                ],
                color,
                "#111827",
                1.0,
            )
        )
        anchor = "end" if include == "external_camera" else ("start" if x < 0 else "end")
        offset_x = -0.45 if anchor == "end" else 0.45
        text_x = sx(x + offset_x)
        text_y = sy(y + (0.58 if y < 0 else -0.58))
        parts.append(f'<text x="{text_x:.1f}" y="{text_y:.1f}" class="label" font-size="14" text-anchor="{anchor}">{label}</text>')
        parts.append(f'<text x="{text_x:.1f}" y="{text_y + 15:.1f}" class="tiny" text-anchor="{anchor}">{esc(include)}</text>')

    legend_x, legend_y = 850, 104
    legend = [
        ("#67d5e8", "planner driveable aisle union"),
        ("#f2c94c", "standard ShelfD/E rack"),
        ("#d96b27", "tall 2.61 m rack / handover occluder"),
        ("#138a43", "conservative no-go envelope"),
        ("#1f6feb", "blue site/dock boundary"),
        ("#7f8c8d", "AWS prop / staging object"),
    ]
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="250" height="172" fill="#ffffff" stroke="#b6c2cc" rx="6"/>')
    parts.append(f'<text x="{legend_x + 16}" y="{legend_y + 25}" font-size="15" font-weight="700">Legend</text>')
    for i, (color, label) in enumerate(legend):
        y = legend_y + 50 + 20 * i
        parts.append(f'<rect x="{legend_x + 16}" y="{y - 11}" width="18" height="12" fill="{color}" stroke="#34495e"/>')
        parts.append(f'<text x="{legend_x + 42}" y="{y}" class="tiny">{label}</text>')

    parts.append(f'<text x="{sx(0):.1f}" y="{sy(0.45) + 5:.1f}" class="label" font-size="16" text-anchor="middle">4.5 m central aisle / local pillar bypass</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _layout_markdown() -> str:
    traversable = profile_regions("traversable")
    tall_segments = [
        f"- `{name}` at `({cx:+.2f}, {cy:+.2f})`, height `{h:.2f} m`"
        for name, cx, cy, _rx, _ry, h in occluders
        if name.startswith(("W", "E")) and h >= 2.5
    ]
    props = [
        f"- `{name}`: AWS `{model}` at `{pose}` ({label})"
        for name, model, pose, label in CLUTTER_ITEMS
    ]
    cameras = [
        f"- {label}: include `{include}`, mount `{mount}`, pose `{pose_text(pose)}`, RGB topic `{topic}`"
        for label, include, _model, pose, topic, _color, mount in CAMERA_ITEMS
    ]
    drive_regions = [
        f"- `{region['name']}`: x `[{float(region['xmin']):+.2f}, {float(region['xmax']):+.2f}]`, "
        f"y `[{float(region['ymin']):+.2f}, {float(region['ymax']):+.2f}]` - {region.get('note', '')}"
        for region in traversable
    ]
    blocked_spots = [
        f"- `{name}`: center `({cx:+.2f}, {cy:+.2f})`, footprint `{sx:.2f} x {sy:.2f} m` - {note}"
        for name, cx, cy, sx, sy, note in NON_DRIVEABLE_SPOTS
    ]
    return "\n".join([
        "# warehouse_full_4cam Layout",
        "",
        "Canonical world: `src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf`",
        "",
        "![Top-down layout](assets/warehouse_full_4cam_map.svg)",
        "",
        "PNG preview: `docs/assets/warehouse_full_4cam_map.png`",
        "",
        "## Map Decision",
        "",
        "- Footprint: ground plane `24.5 x 20.5 m`, walls at `x = +/-12 m`, `y = +/-10 m`.",
        "- Structure: two four-row rack blocks copied from the old warehouse style, spaced so the inner rack faces bound a `4.50 m` central aisle instead of a broad empty hall. Rack segments were shortened so both interior cross-aisles retain `0.96 m` between conservative no-go envelopes.",
        "- One-sided shelf: an extra ShelfD/E row is backed directly against the west wall, so it can only be reached from the aisle side.",
        f"- Floor semantics: beige is the physical floor, cyan is the exact planner driveable union from `world_profiles.yaml`, blue is the site boundary, and green is a conservative no-go envelope expanded by {NOGO_MARGIN_M:.2f} m beyond collision geometry. Cyan and green regions are contract-checked not to overlap.",
        "- Aisle cleanup: the dropped west-aisle stack, four loose south-apron crates, and east-service pallet-box island were removed. The fixed building pillar remains and is represented as a local two-sided bypass instead of removing the whole center strip.",
        "- Camera placement: four high wall-mounted cameras sit at `(-6.0, -10)`, `(-6.0, 10)`, `(6.0, -10)`, and `(6.0, 10)` and look perpendicular to their wall. Bringing the columns closer to the central aisle widens adjacent-camera overlap for handover validation.",
        "- Presentation overview: a separate top-down camera at `(0.0, 0.0, 26.0)` produces a full-facility Gazebo view for media only. It is not a fifth localization camera and is excluded from the GP/fusion/planner interfaces.",
        "- Occlusion/handover: W-block inner north racks and E-block inner south racks are tall, creating different blind regions for the north/south camera pairs.",
        "- Styling: box collision/contact sensors remain the operational geometry; ShelfD/E AWS meshes are visual overlays so rendered cameras see the same shelf footprint.",
        "- Added AWS props: bucket, trash can, clutter piles, pallet jack, lamps, and a `DeskC_01` quality-control desk near the north-east wall.",
        "",
        "## Cameras",
        "",
        *cameras,
        "",
        "Approximate FOV wedges in the SVG are visual planning aids only. Runtime projection still comes from the camera model and calibration.",
        "",
        "## Planner Driveable Aisles (cyan)",
        "",
        *drive_regions,
        "",
        "## Conservative No-Go Envelopes",
        "",
        *blocked_spots,
        "",
        "## Tall Handover Segments",
        "",
        *tall_segments,
        "",
        "## AWS Props",
        "",
        *props,
        "",
        "## Launch",
        "",
        "```bash",
        "ros2 launch sim bringup_sim.launch.py world:=warehouse_full_4cam.world.sdf bridge_camera_b:=true bridge_camera_c:=true bridge_camera_d:=true use_lidar:=false bridge_scan:=false",
        "```",
        "",
        "Generated by `scripts/geometry_visibility/make_warehouse_full.py`; edit the generator, not the SDF.",
        "",
    ])


def _write_png_map() -> bool:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/warehouse_full_4cam_matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon, Rectangle
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(12.4, 9.2), dpi=150)
    traversable = profile_regions("traversable")
    ax.set_facecolor("#f5f7f9")
    ax.set_xlim(X0 - 0.7, X1 + 0.7)
    ax.set_ylim(Y0 - 0.9, Y1 + 0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("warehouse_full_4cam: four wall-mounted inward-looking cameras", fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="white", linewidth=1.0, linestyle=(0, (2, 5)))

    def add_rect(cx, cy, rx, ry, color, edge="#2f2f2f", label=None, alpha=1.0):
        ax.add_patch(Rectangle((cx - rx / 2, cy - ry / 2), rx, ry, facecolor=color, edgecolor=edge, linewidth=0.8, alpha=alpha))
        if label:
            ax.text(cx, cy, label, ha="center", va="center", fontsize=6.5, weight="bold", color="#1f2933")

    def add_outline(cx, cy, rx, ry, pad=NOGO_MARGIN_M):
        ax.add_patch(Rectangle(
            (cx - rx / 2 - pad, cy - ry / 2 - pad),
            rx + 2.0 * pad,
            ry + 2.0 * pad,
            facecolor="none",
            edgecolor="#138a43",
            linewidth=1.35,
            alpha=0.98,
        ))

    ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, facecolor="#d6d0c8", edgecolor="#42505c", linewidth=1.2))
    for region in traversable:
        xmin, xmax = float(region["xmin"]), float(region["xmax"])
        ymin, ymax = float(region["ymin"]), float(region["ymax"])
        ax.add_patch(Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="#67d5e8",
            edgecolor="#1282a2",
            linewidth=0.7,
            alpha=0.20,
        ))
    for _label, _include, _model, pose, _topic, color, _mount in CAMERA_ITEMS:
        ax.add_patch(Polygon(_clipped_fov_points(pose[0], pose[1], pose[5]), closed=True, facecolor=color, edgecolor=color, alpha=0.14))

    boundary_cx = 0.5 * (SITE_X0 + SITE_X1)
    boundary_cy = 0.5 * (SITE_Y0 + SITE_Y1)
    add_rect(boundary_cx, SITE_Y0, SITE_X1 - SITE_X0, 0.07, "#1f6feb", "#1f6feb")
    add_rect(boundary_cx, SITE_Y1, SITE_X1 - SITE_X0, 0.07, "#1f6feb", "#1f6feb")
    add_rect(SITE_X0, boundary_cy, 0.07, SITE_Y1 - SITE_Y0, "#1f6feb", "#1f6feb")
    add_rect(SITE_X1, boundary_cy, 0.07, SITE_Y1 - SITE_Y0, "#1f6feb", "#1f6feb")

    for name, cx, cy, rx, ry, h in occluders:
        if name.startswith(("W", "E")):
            label = name if (h >= 2.5 or name.startswith("W0")) else None
            add_rect(cx, cy, rx, ry, "#d96b27" if h >= 2.5 else "#f2c94c", "#5b4222", label, 0.96)
        elif name == "support_pillar":
            add_rect(cx, cy, rx, ry, "#6b7280", "#374151", "pillar")

    for _name, model, pose, label in CLUTTER_ITEMS:
        cx, cy = _pose_xy(pose)
        rx, ry = PROP_SIZES.get(model, (0.7, 0.7))
        add_rect(cx, cy, rx, ry, "#7f8c8d", "#34495e", label, 0.93)

    for _name, cx, cy, rx, ry, _h in occluders:
        add_outline(cx, cy, rx, ry)
    for i, (x, y) in enumerate(LAMP_POSES):
        ax.add_patch(Circle((x, y), 0.18, facecolor="#fff2a8", edgecolor="#8f7a20", linewidth=0.8))
        ax.text(x, y - 0.42, f"lamp {i}", ha="center", va="top", fontsize=6.5)

    for label, include, _model, pose, topic, color, mount in CAMERA_ITEMS:
        x, y, yaw = pose[0], pose[1], pose[5]
        ax.add_patch(Circle((x, y), 0.24, facecolor=color, edgecolor="#111827", linewidth=1.0, zorder=5))
        ax.arrow(x, y, 0.72 * math.cos(yaw), 0.72 * math.sin(yaw), width=0.05, head_width=0.28, head_length=0.28, color="#111827", zorder=6)
        ha = "right" if include == "external_camera" else ("left" if x < 0 else "right")
        tx = x + (-0.45 if ha == "right" else 0.45)
        ty = y + (0.65 if y < 0 else -0.65)
        ax.text(tx, ty, f"{label}\n{include}", ha=ha, va="center", fontsize=7.0, weight="bold", color="#1f2933")

    ax.text(0, 0.45, "4.5 m central aisle / local pillar bypass", ha="center", va="center", fontsize=9, weight="bold", color="#1f2933")
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#67d5e8", edgecolor="#1282a2", alpha=0.40, label="planner driveable aisle union"),
        Rectangle((0, 0), 1, 1, facecolor="#f2c94c", edgecolor="#5b4222", label="standard ShelfD/E rack"),
        Rectangle((0, 0), 1, 1, facecolor="#d96b27", edgecolor="#5b4222", label="tall handover occluder"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#138a43", label="conservative no-go envelope"),
        Rectangle((0, 0), 1, 1, facecolor="#1f6feb", edgecolor="#1f6feb", label="site/dock boundary"),
        Rectangle((0, 0), 1, 1, facecolor="#7f8c8d", edgecolor="#34495e", label="AWS prop/staging"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.96)
    fig.tight_layout()
    fig.savefig(PNG)
    plt.close(fig)
    return True


def _write_layout_artifacts() -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    SVG.parent.mkdir(parents=True, exist_ok=True)
    SVG.write_text(_svg_map(), encoding="utf-8")
    DOC.write_text(_layout_markdown(), encoding="utf-8")
    _write_png_map()

sdf = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <!-- warehouse_full_4cam: flagship AWS-mesh-styled large warehouse (~24 x 20 m)
       with FOUR wall-mounted external cameras looking perpendicular to their walls.
       Generated by scripts/geometry_visibility/make_warehouse_full.py - edit the
       generator, not this file. Collision boxes == mesh visual heights, so
       physics == CAD prisms == what the cameras see. Occluder heights are
       EVALUATION-ONLY for the visibility priors. -->
  <world name="warehouse_full_4cam">
    <physics name="default_physics" type="ode"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor><real_time_update_rate>1000</real_time_update_rate></physics>
    <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>
    <plugin name="gz::sim::systems::UserCommands" filename="gz-sim-user-commands-system"/>
    <plugin name="gz::sim::systems::SceneBroadcaster" filename="gz-sim-scene-broadcaster-system"/>
    <plugin name="gz::sim::systems::Contact" filename="gz-sim-contact-system"/>
    <plugin name="gz::sim::systems::Sensors" filename="gz-sim-sensors-system"><render_engine>ogre2</render_engine></plugin>
    <gravity>0 0 -9.81</gravity><magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field><atmosphere type="adiabatic"/>
    <scene><ambient>0.58 0.58 0.58 1</ambient><background>0.70 0.72 0.74 1</background><shadows>true</shadows></scene>

    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>24.5 20.5</size></plane></geometry></collision>
      <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>24.5 20.5</size></plane></geometry><material><ambient>{FLOOR}</ambient><diffuse>{FLOOR}</diffuse><specular>0.05 0.05 0.05 1</specular></material></visual>
    </link></model>

    <model name="warehouse_walls"><static>true</static>
{walls}
    </model>

    <model name="warehouse_rack_occluders"><static>true</static>
{rack_model}
    </model>

    <model name="aws_shelf_mesh_visuals"><static>true</static>
{mesh_model}
    </model>

    <model name="known_driveable_boundary"><static>true</static>
{marker_model}
    </model>

{clutter}

{camera_includes}
{overview_camera_include}

    <light name="sun" type="directional"><pose>0 0 12 0 0 0</pose><cast_shadows>true</cast_shadows><intensity>0.74</intensity><direction>-0.25 0.15 -1</direction></light>
    <light name="south_overhead_light" type="point"><pose>0 -6.0 5.1 0 0 0</pose><cast_shadows>true</cast_shadows><intensity>0.60</intensity><attenuation><range>18</range><constant>0.45</constant><linear>0.06</linear><quadratic>0.010</quadratic></attenuation></light>
    <light name="center_overhead_light" type="point"><pose>0 0.5 5.2 0 0 0</pose><cast_shadows>true</cast_shadows><intensity>0.65</intensity><attenuation><range>18</range><constant>0.45</constant><linear>0.06</linear><quadratic>0.010</quadratic></attenuation></light>
    <light name="north_overhead_light" type="point"><pose>0 6.5 5.1 0 0 0</pose><cast_shadows>true</cast_shadows><intensity>0.62</intensity><attenuation><range>18</range><constant>0.45</constant><linear>0.06</linear><quadratic>0.010</quadratic></attenuation></light>
  </world>
</sdf>
"""

OUT.write_text(sdf)
_write_layout_artifacts()
print(f"wrote {OUT}")
print(f"wrote {SVG}")
if PNG.exists():
    print(f"wrote {PNG}")
print(f"wrote {DOC}")
print(f"  rack segments: {sum(1 for n,*_ in occluders if n[0] in 'WE')}, mesh overlays: {len(mesh_links)}")
print("  tall (2.61 m) segments:")
for n, cx, cy, sx, sy, h in occluders:
    if h >= 2.5 and n != "support_pillar":
        print(f"    {n:12s} at ({cx:+.2f},{cy:+.2f})")
