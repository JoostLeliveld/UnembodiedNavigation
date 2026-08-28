#!/usr/bin/env python3
"""Emit warehouse_v2.world.sdf from warehouse_v2.py -- edit the layout, not the SDF.

Follows the same construction as make_warehouse_full.py, which is the pattern the
rest of the toolchain expects:

  * box collisions WITH contact sensors are the operational geometry (planner,
    logger and CAD parser all read those),
  * the AWS meshes are visual-only overlays sitting exactly on those boxes,
  * floor paint is visual-only: a blue site boundary and a green keep-out
    envelope around every declared zone.

Meshes are never stretched vertically to reach a stock height. A 5.23 m rack is
two native 2.613 m ShelfD modules; a
three-high block stack is three native pallets at 0.00 / 1.06 / 2.12 m. That
keeps the rendered shelf in the distribution the detector was trained on.
"""
from __future__ import annotations

import argparse
import math
import pathlib
from pathlib import Path
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mesh_library import MESHES                      # noqa: E402
from warehouse_v2 import (APRON_TOP, ARTERY, A_TOP, A_X, A_Y1, B_TOP, C_ROWS,
                          DOCK_DOORS, DRUM_D, DRUM_H, MODULE, SITE, TOTE_H,
                          TOTE_L, TOTE_W, build)  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
WORLD_FILES = {
    "A": "warehouse_v2.world.sdf",
    "B": "warehouse_v2_shipout.world.sdf",
}
WORLD_NAMES = {
    "A": "warehouse_v2",
    "B": "warehouse_v2_shipout",
}
WORLD_DIR = REPO / "src/sim/gazebo_worlds/worlds"

WALL_H = 9.0
WALL_T = 0.20
HALL_X, HALL_Y = (-12.0, 12.0), (-10.0, 10.0)
NOGO = 0.32

WALL_C = "0.70 0.70 0.68 1"
FLOOR_C = "0.55 0.545 0.52 1"
DOOR_C = "0.32 0.36 0.40 1"
BLUE = "0.05 0.30 0.85 1"
GREEN = "0.05 0.55 0.20 1"
OFFICE_C = "0.80 0.79 0.75 1"
# One painted colour per section. The AWS pack is a single ochre palette -- ShelfD
# and ShelfE share all four texture files -- so colour variety cannot be picked,
# it has to be painted on, the way this repo's older world paints its blue rails.
# Steel tints are DIVIDED by the metal albedo's mean (0.489, 0.536, 0.578), so
# the painted colour that comes out of the multiply is the colour named here
# rather than a darkened version of it. Measured, not eyeballed.
FRAME_A = "0.20 0.56 0.95 1"     # -> 0.10 0.30 0.55, blue racking uprights
BEAM_A = "0.86 0.42 0.06 1"      # section A: orange load beams on the blue uprights
FRAME_B = "1.00 0.63 0.17 1"     # -> 0.78 0.34 0.10, orange uprights
BEAM_B = "0.12 0.34 0.60 1"      # section B: blue load beams -- the reverse of A
# Floor markings and fittings. Real DC floors are not one grey: traffic lanes are
# white, pedestrian routes are green, anything you must not stand in is yellow,
# and fire equipment is red. None of this is geometry -- it is paint on the slab
# and colour on the steel, which is the only lever available here because every
# dominant colour in the AWS mesh pack is the same ochre/yellow/dark-frame trio.
# Yellow is the standard aisle/gangway colour and is kept for exactly three
# things -- the artery boundary, the dock keep-clear hatching and the rack-end
# guards. It was previously also on the kick plates, the section-C bay outlines
# and the dock sills, which put so much of it in frame that the marks that
# should carry meaning stopped carrying any.
LANE_YELLOW = "0.88 0.72 0.05 1"   # aisle and storage-block demarcation
PERIMETER = "0.86 0.86 0.83 1"     # white: operating-field perimeter
# Pallets are where the colour actually is in a real DC. Pooled pallets are
# painted by their pool operator -- CHEP blue and LPR red are the two you cannot
# miss -- and everything else is natural timber. Loads are then stretch-wrapped,
# and the film is clear, black or tinted depending on the packer.
# NO painted dressing on the goods. Three attempts went on the block stacks and
# all three were cut, which is the answer rather than a failure:
#
#   1. stretch-film wraps -- film is transparent and skin-tight, so an opaque box
#      standing proud of the cartons renders as a coloured shelf pushed between
#      them. No thickness or height fixes an object that is not opaque.
#   2. pallet decks cycling by stack height -- a rainbow layer cake, because one
#      bay is one SKU on one supplier's pallets, not three liveries in a column.
#   3. pallet decks, one livery per bay, weighted to timber -- correct in every
#      respect above and STILL a legible coloured band at every seam, because a
#      0.14 m slab across a 2.15 x 1.98 m footprint is a large flat face at the
#      exact height the cameras look at.
#
# The goods are cardboard on timber and that is what they look like. Colour in
# this building comes from the racking steel, the floor markings and the
# building fabric -- things that carry colour because someone chose to paint
# them, not because a render needed livening up.
WALK_GREEN = "0.09 0.42 0.20 1"  # pedestrian walkway
HATCH_YELLOW = "0.92 0.78 0.05 1"  # keep-clear hatching at the dock doors
GUARD_YELLOW = "1.00 1.00 0.07 1"  # -> 0.93 0.72 0.04, rack-end guards
DADO = "0.36 0.40 0.45 1"        # painted lower wall band, 1.4 m


#: TEST: does an SDF material tint the DAE or flatten it? Cm gets a muted
#: blue-grey, Cn a muted olive, Cs stays natural as the control.
# Box variety: three MESHES, not three paint jobs. Four things were rendered and
# looked at before settling on that:
#
#   1. <material><diffuse> alone REPLACES the DAE material -- the loads came out
#      as flat untextured monoliths, no grain, no seams, no printing.
#   2. <pbr><metal><albedo_map> at the model's own texture, modulated by
#      <diffuse>, does keep the grain and does shift the colour. Promising.
#   3. ...but every Cluttering mesh has THREE material slots, and ClutteringA's
#      DAE genuinely uses all three of its textures. Forcing one albedo_map
#      collapses all three slots onto one image, and the faces whose UVs expect a
#      different image render black. That is not fixable from the SDF: it would
#      need per-slot materials the format does not expose here.
#   4. So the loads keep their own materials, untouched, and the variety comes
#      from using ClutteringA, C and D on the three rows -- genuinely different
#      product families at genuinely different sizes -- rather than from
#      repainting one mesh nine times.

RACK_MESH_H = MESHES["ShelfD_01"]["size"][2]         # 2.613
PALLET = "ClutteringA_01"
PALLET_H = MESHES[PALLET]["size"][2]                 # 1.058
PALLET_SX, PALLET_SY = MESHES[PALLET]["size"][:2]    # 2.146 x 1.983


# ---------------------------------------------------------------------------
# Lighting. These are the numbers the 2026-08-21 fix turns, and they are set by
# measurement: render the world, project the sixteen lamp positions into the five
# camera frames, sample the floor under each fixture and midway between them, and
# require (a) a working aisle floor that is clearly lit rather than grey, (b) a
# visible excess under each fitting, and (c) no clipped white. The measurement
# lives in the study log; do not tune these by eye.
SCENE_AMBIENT = 0.20      # was 0.34: a uniform wash that swamped every fixture
SUN_DIFFUSE = 0.42        # was 0.74: interior form light, not an outdoor sun
LAMP_DIFFUSE = 0.60       # was 0.22, which arrived as 0.10 after attenuation
LAMP_ATT_CONST = 1.00     # was 1.10
LAMP_ATT_LINEAR = 0.035   # was 0.08
LAMP_ATT_QUAD = 0.006     # was 0.010
LAMP_RANGE = 12.0         # a fitting lights its own bay; at 20 m distant
                          # fittings piled up and clipped the open apron
LAMP_LIGHT_Z = 6.95       # was 7.10, which sat inside the housing (7.00-8.52 m)
LAMP_GLOW_Z = 6.97        # the emissive face, just under the fitting

# How each painted surface responds to light. The AWS pack ships diffuse
# textures and nothing else -- no normal, roughness or metalness maps anywhere --
# which is why the meshes read as flat plastic. The MESHES cannot be fixed from
# here without destroying their material slots, but every painted box in this
# world has no texture to lose, so it can carry a real PBR response for free.
# Concrete is rough and dielectric, painted rack steel is semi-gloss and
# metallic, floor paint is matte. That difference is most of what "flat" means.
FINISH = {"concrete": (0.92, 0.0),      # (roughness, metalness)
          "steel": (0.35, 0.65),
          "paint": (0.55, 0.0),
          "wall": (0.80, 0.0),
          "card": (0.85, 0.0)}


PBR_TEX = "model://warehouse_pbr/materials/textures"
# Divided through the metal albedo mean (0.489, 0.536, 0.578) so the drum renders
# the colour named: pooled-chemical blue, a plain galvanised one, and safety red.
DRUM_TINT = ["0.31 0.60 1.00 1",     # -> 0.15 0.32 0.58 blue
             "0.90 0.90 0.88 1",     # -> galvanised grey
             "1.00 0.30 0.16 1"]     # -> safety red
# The tote mesh's albedo is near-black, so a stack of them is a solid dark mass.
# Returnable transit packaging really does come in grey and blue as well as
# black, and the tint lifts the albedo rather than replacing it, so the moulding
# detail and the handles survive.
# SDF colour components are clamped to [0,1], so a >1 diffuse cannot brighten a
# near-black albedo -- Fortress rejects the world outright. The lift therefore
# happens in the TEXTURE: two derivative albedos are generated from the crate's
# own map (mean 20 -> 76 grey, -> 48/62/86 blue) multiplicatively, so the
# moulding detail and handle shadows survive instead of being flat-filled.
TOTE_ALBEDO = [f"{PBR_TEX}/tote_grey_albedo.png",
               None,                                   # as moulded, near black
               f"{PBR_TEX}/tote_blue_albedo.png"]


def pbr_slab(name, cx, cy, z, sx, sy, sz, stem, yaw=0.0, tint="1 1 1 1",
             metal_map=False):
    """A visual-only box carrying a tileable PBR material set.

    Primitives get 0..1 UVs per face in ogre2, so one texture is stretched over
    the whole face however big it is. The floor is therefore laid as SLABS rather
    than one sheet: a 2048 px map over a 6 m bay is ~340 px/m, and poured
    concrete really is cast in bays with saw-cut joints between them, so the
    tiling boundary is a feature of the real thing rather than an artefact.
    """
    metal_extra = (f'<metalness_map>{PBR_TEX}/{stem}_metalness.png</metalness_map>'
                   if metal_map else '<metalness>0.0</metalness>')
    return (f'      <link name="{name}"><pose>{cx:.3f} {cy:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>'
            f'<visual name="visual"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size>'
            f'</box></geometry><material>'
            f'<ambient>{tint}</ambient><diffuse>{tint}</diffuse>'
            f'<specular>0.08 0.08 0.08 1</specular>'
            f'<pbr><metal>'
            f'<albedo_map>{PBR_TEX}/{stem}_albedo.png</albedo_map>'
            f'<normal_map>{PBR_TEX}/{stem}_normal.png</normal_map>'
            f'<roughness_map>{PBR_TEX}/{stem}_roughness.png</roughness_map>'
            f'{metal_extra}'
            f'</metal></pbr></material></visual></link>')


def tote_link(name, cx, cy, z, yaw=0.0, albedo=None):
    """A returnable plastic tote, drawn as the mesh with its own PBR maps.

    Deliberately NOT an <include> of model://Large_Crate. That model ships a
    collision box with a NEGATIVE z extent (Box019: 0.662 0.028538 -0.0406942)
    and DART asserts on it -- the whole world fails to start. The totes are
    visual dressing standing inside an already-declared occluder box, so they
    need no collision of their own and the broken model is simply not loaded.
    """
    T = "model://Large_Crate"
    return (f'      <link name="{name}"><pose>{cx:.3f} {cy:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>'
            f'<visual name="visual"><geometry><mesh>'
            f'<uri>{T}/meshes/LargeCrate.dae</uri></mesh></geometry>'
            f'<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>'
            f'<specular>0.2 0.2 0.2 1</specular><pbr><metal>'
            f'<albedo_map>{albedo or (T + "/materials/textures/Crate_Albedo.jpg")}</albedo_map>'
            f'<normal_map>{T}/materials/textures/Crate_Normal.jpg</normal_map>'
            f'<roughness_map>{T}/materials/textures/Crate_Roughness.jpg</roughness_map>'
            f'<metalness>0.0</metalness></metal></pbr></material></visual></link>')


def pbr_cyl(name, cx, cy, z, r, h, stem, tint="1 1 1 1", metal_map=False):
    """A visual-only cylinder on a PBR material set: a steel drum."""
    metal_extra = (f'<metalness_map>{PBR_TEX}/{stem}_metalness.png</metalness_map>'
                   if metal_map else '<metalness>0.0</metalness>')
    return (f'      <link name="{name}"><pose>{cx:.3f} {cy:.3f} {z:.3f} 0 0 0</pose>'
            f'<visual name="visual"><geometry><cylinder><radius>{r:.3f}</radius>'
            f'<length>{h:.3f}</length></cylinder></geometry><material>'
            f'<ambient>{tint}</ambient><diffuse>{tint}</diffuse>'
            f'<specular>0.25 0.25 0.25 1</specular>'
            f'<pbr><metal>'
            f'<albedo_map>{PBR_TEX}/{stem}_albedo.png</albedo_map>'
            f'<normal_map>{PBR_TEX}/{stem}_normal.png</normal_map>'
            f'<roughness_map>{PBR_TEX}/{stem}_roughness.png</roughness_map>'
            f'{metal_extra}</metal></pbr></material></visual></link>')


def box_link(name, cx, cy, z, sx, sy, sz, colour, *, collide=True, contact=True,
             visual=True, yaw=0.0, finish="wall"):
    p = [f'      <link name="{name}"><pose>{cx:.3f} {cy:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>']
    if collide:
        p.append(f'<collision name="collision"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}'
                 f'</size></box></geometry></collision>')
        if contact:
            p.append('<sensor name="contact" type="contact"><contact><collision>collision'
                     '</collision></contact><update_rate>60</update_rate></sensor>')
    if visual:
        rough, metal = FINISH[finish]
        p.append(f'<visual name="visual"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size>'
                 f'</box></geometry><material><ambient>{colour}</ambient>'
                 f'<diffuse>{colour}</diffuse>'
                 f'<specular>{0.5*(1-rough):.3f} {0.5*(1-rough):.3f} {0.5*(1-rough):.3f} 1</specular>'
                 f'<pbr><metal><metalness>{metal:.2f}</metalness>'
                 f'<roughness>{rough:.2f}</roughness></metal></pbr>'
                 f'</material></visual>')
    p.append("</link>")
    return "".join(p)


def mesh_link(name, model, cx, cy, z, yaw, sx, sy, sz):
    return (f'      <link name="{name}"><pose>{cx:.3f} {cy:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>'
            f'<visual name="visual"><geometry><mesh><uri>model://aws_robomaker_warehouse_{model}'
            f'/meshes/aws_robomaker_warehouse_{model}_visual.DAE</uri>'
            f'<scale>{sx:.4f} {sy:.4f} {sz:.4f}</scale></mesh></geometry></visual></link>')


def rack_tiers(h: float):
    """Split a stock height into whole native rack modules plus a short top one.

    Returns (z_base, z_scale) pairs. Never stretches: the tallest scale returned
    is 1.0, so beams keep their spacing and only a partial top tier is squashed.
    """
    tiers, z = [], 0.0
    while h - z > RACK_MESH_H * 1.02:
        tiers.append((z, 1.0))
        z += RACK_MESH_H
    if h - z > 0.05:
        tiers.append((z, (h - z) / RACK_MESH_H))
    return tiers


def outline(name, xmin, xmax, ymin, ymax, colour=GREEN, pad=NOGO, t=0.06):
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    ox, oy = (xmax - xmin) + 2 * pad, (ymax - ymin) + 2 * pad
    z, h = 0.02, 0.012
    return [
        box_link(f"{name}_s", cx, cy - oy / 2, z, ox, t, h, colour, collide=False),
        box_link(f"{name}_n", cx, cy + oy / 2, z, ox, t, h, colour, collide=False),
        box_link(f"{name}_w", cx - ox / 2, cy, z, t, oy, h, colour, collide=False),
        box_link(f"{name}_e", cx + ox / 2, cy, z, t, oy, h, colour, collide=False),
    ]


def camera_include(c):
    return (f'    <include><name>{c.include}</name><uri>model://{c.model}</uri>'
            f'<pose>{c.x:.3f} {c.y:.3f} {c.z:.3f} 0 {math.radians(c.pitch_deg):.4f} '
            f'{math.radians(c.yaw_deg):.4f}</pose></include>')


class Cam:
    def __init__(self, c, include, model):
        self.x, self.y, self.z = c.x, c.y, c.z
        self.pitch_deg, self.yaw_deg = c.pitch_deg, c.yaw_deg
        self.include, self.model = include, model


CAMERA_MODELS = {"A": "external_camera", "B": "external_camera_b", "C": "external_camera_c",
                 "D": "external_camera_d", "E": "external_camera_e"}


#: Diagnostic switch, off by default. With it on, the sixteen fixture LIGHTS are
#: omitted while every other element -- including the fittings themselves and their
#: emissive faces -- stays identical, so differencing two renders isolates exactly
#: what the fixtures contribute to the floor. Never used for a captured dataset.
LAMP_LIGHTS_ON = True


def build_world(state: str = "A") -> str:
    if state not in WORLD_FILES:
        raise ValueError(f"state must be one of {sorted(WORLD_FILES)}, got {state!r}")
    L = build()
    fill = L.fill_a if state == "A" else L.fill_b
    world_name = WORLD_NAMES[state]

    # ---- walls, doors, floor -------------------------------------------------
    walls = [
        box_link("wall_east", HALL_X[1], 0, WALL_H / 2, WALL_T, 20.0, WALL_H, WALL_C),
        box_link("wall_west", HALL_X[0], 0, WALL_H / 2, WALL_T, 20.0, WALL_H, WALL_C),
        box_link("wall_north", 0, HALL_Y[1], WALL_H / 2, 24.0, WALL_T, WALL_H, WALL_C),
    ]
    # south wall in segments so the three dock doors are real openings
    doors = sorted(DOCK_DOORS)
    edges = [HALL_X[0]] + [v for d in doors for v in (d - 1.60, d + 1.60)] + [HALL_X[1]]
    for i in range(0, len(edges) - 1, 2):
        a, b = edges[i], edges[i + 1]
        if b - a > 0.05:
            walls.append(box_link(f"wall_south_{i//2}", 0.5 * (a + b), HALL_Y[0], WALL_H / 2,
                                  b - a, WALL_T, WALL_H, WALL_C))
    # The lintel has to run from the door head all the way to the wall top, or
    # the wall has a 4.9 m slot above every door and the cameras see daylight.
    for i, d in enumerate(doors):
        head = 4.50                                     # dock door clear height
        walls.append(box_link(f"door_head_{i}", d, HALL_Y[0], 0.5 * (head + WALL_H),
                              3.20, WALL_T, WALL_H - head, WALL_C))
        walls.append(box_link(f"door_frame_{i}", d, HALL_Y[0] - 0.13, 0.5 * head,
                              3.36, 0.08, head, DOOR_C, collide=False))
        walls.append(box_link(f"door_sill_{i}", d, HALL_Y[0] + 0.30, 0.008,
                              3.20, 0.60, 0.014, DOOR_C, collide=False,
                              finish="steel"))
        # Rubber dock bumpers and the shutter guide rails: the two details that
        # say "loading dock" rather than "hole in a wall". Both sit on the wall
        # face outside the operating field, so neither costs a driveable cell.
        for sgn in (-1, 1):
            walls.append(box_link(f"door_bumper_{i}{'lr'[sgn > 0]}",
                                  d + sgn * 1.74, HALL_Y[0] + 0.16, 0.55,
                                  0.28, 0.22, 0.50, "0.12 0.12 0.13 1",
                                  collide=False, finish="paint"))
            walls.append(box_link(f"door_guide_{i}{'lr'[sgn > 0]}",
                                  d + sgn * 1.66, HALL_Y[0] + 0.14, head / 2,
                                  0.14, 0.14, head, "0.44 0.46 0.48 1",
                                  collide=False, finish="steel"))
        walls.append(box_link(f"door_lintel_{i}", d, HALL_Y[0] + 0.14,
                              head + 0.22, 3.60, 0.20, 0.44,
                              "0.44 0.46 0.48 1", collide=False, finish="steel"))

    # ---- storage contents ----------------------------------------------------
    boxes, meshes, office_vis, goods = [], [], [], []
    for o in fill:
        if o.zone == "DOCK_OFFICE":
            # Collision stays one clean box -- the planner and the sight-line
            # model want a solid 2.80 m block. The VISUAL is built as a small
            # building on top of it, because from camera A at 5 m the old single
            # pale box filled a third of the frame and read as a blank slab.
            boxes.append(box_link("dock_office", o.cx, o.cy, o.h / 2, o.sx, o.sy,
                                  o.h, OFFICE_C, visual=False))
            office_vis.append(pbr_slab("office_body", o.cx, o.cy, 1.30,
                                       o.sx, o.sy, 2.60, "concrete",
                                       tint="1.00 0.99 0.95 1"))
            # The roof is what a 5 m camera actually sees of this building, so
            # it is a pale rough membrane, not a dark metallic slab -- the first
            # attempt swapped one blank slab for a darker one. The parapet is a
            # RIM around the edge; as a full slab it just re-covered the roof.
            office_vis.append(pbr_slab("office_roof", o.cx, o.cy, 2.66,
                                       o.sx + 0.06, o.sy + 0.06, 0.12,
                                       "concrete", tint="0.83 0.83 0.80 1"))
            for tag, dx, dy, px, py in (("s", 0, -(o.sy + 0.06) / 2, o.sx + 0.06, 0.10),
                                        ("n", 0, +(o.sy + 0.06) / 2, o.sx + 0.06, 0.10),
                                        ("w", -(o.sx + 0.06) / 2, 0, 0.10, o.sy + 0.06),
                                        ("e", +(o.sx + 0.06) / 2, 0, 0.10, o.sy + 0.06)):
                office_vis.append(box_link(f"office_parapet_{tag}", o.cx + dx,
                                           o.cy + dy, 2.80, px, py, 0.20,
                                           "0.50 0.51 0.49 1", collide=False,
                                           finish="paint"))
            office_vis.append(box_link("office_window", o.cx + o.sx / 2 - 0.01,
                                       o.cy, 1.70, 0.05, o.sy * 0.62, 0.80,
                                       "0.20 0.28 0.34 1", collide=False,
                                       finish="steel"))
            office_vis.append(box_link("office_door", o.cx + o.sx / 2 - 0.01,
                                       o.cy - o.sy / 2 + 0.55, 1.02, 0.06, 0.90,
                                       2.04, DOOR_C, collide=False,
                                       finish="paint"))
            office_vis.append(box_link("office_step", o.cx + o.sx / 2 + 0.22,
                                       o.cy - o.sy / 2 + 0.55, 0.06, 0.44, 1.10,
                                       0.12, "0.52 0.52 0.50 1", collide=False,
                                       finish="concrete"))
            continue
        boxes.append(box_link(f"obs_{o.name}", o.cx, o.cy, o.h / 2, o.sx, o.sy, o.h,
                              WALL_C, visual=False))
        if o.zone.startswith("C"):                       # block-stacked pallets
            bay = int(o.name.rsplit("_p", 1)[-1]) if "_p" in o.name else 0
            unit_h = {"drums": DRUM_H, "totes": TOTE_H}.get(
                o.mesh, MESHES[o.mesh]["size"][2] if o.mesh in MESHES else 1.0)
            n = max(1, int(round(o.h / unit_h)))
            if o.mesh == "drums":
                # 2 x 2 steel drums per pallet layer, in pool colours by bay.
                # Tints are divided through the metal albedo mean the same way
                # the racking steel is, so the drum comes out the colour named.
                col = DRUM_TINT[bay % len(DRUM_TINT)]
                for k in range(n):
                    for dx in (-DRUM_D / 2 - 0.01, DRUM_D / 2 + 0.01):
                        for dy in (-DRUM_D / 2 - 0.01, DRUM_D / 2 + 0.01):
                            goods.append(pbr_cyl(
                                f"drum_{o.name}_{k}_{'wm'[dx > 0]}{'sn'[dy > 0]}",
                                o.cx + dx, o.cy + dy,
                                k * DRUM_H + DRUM_H / 2, DRUM_D / 2, DRUM_H,
                                "metal", tint=col, metal_map=True))
                    goods.append(pbr_slab(f"drumpallet_{o.name}_{k}", o.cx, o.cy,
                                           k * DRUM_H + 0.02, 1.20, 1.20, 0.04,
                                           "concrete", tint="0.80 0.66 0.44 1"))
                continue
            if o.mesh == "totes":
                # 2 x 3 returnable plastic totes per layer, included as props so
                # Gazebo resolves the mesh's own node transforms.
                for k in range(n):
                    for ix in range(2):
                        for iy in range(3):
                            goods.append(tote_link(
                                f"tote_{o.name}_{k}_{ix}{iy}",
                                o.cx + (ix - 0.5) * (TOTE_L + 0.02),
                                o.cy + (iy - 1) * (TOTE_W + 0.02),
                                k * TOTE_H,
                                albedo=TOTE_ALBEDO[bay % len(TOTE_ALBEDO)]))
                continue
            # Each carton row carries its own product family, so the mesh comes
            # from the obstacle rather than from one global pallet constant.
            msx, msy, msz = MESHES[o.mesh]["size"]
            # Whether the layout turned this load to fit the row depth is decided
            # by comparing its footprint to the mesh's, not by which side is
            # longer: ClutteringC_01 is 1.77 x 2.05 and comes out 2.05 x 1.77
            # once turned, so a sx<sy test reads it as un-turned and stretches it.
            rot = abs(o.sx - msy) + abs(o.sy - msx) < abs(o.sx - msx) + abs(o.sy - msy)
            yaw = math.pi / 2 if rot else 0.0
            fx = (o.sy if rot else o.sx) / msx
            fy = (o.sx if rot else o.sy) / msy
            for k in range(n):
                meshes.append(mesh_link(f"m_{o.name}_{k}", o.mesh, o.cx, o.cy,
                                        k * msz, yaw, fx, fy, 1.0))

        elif o.zone.startswith(("A", "B")):              # racking
            ns_run = o.sy > o.sx
            yaw = math.pi / 2 if ns_run else 0.0
            run_len = o.sy if ns_run else o.sx
            depth = o.sx if ns_run else o.sy
            fx = run_len / MESHES["ShelfD_01"]["size"][0]
            fy = depth / MESHES["ShelfD_01"]["size"][1]
            for k, (z0, zs) in enumerate(rack_tiers(o.h)):
                meshes.append(mesh_link(f"m_{o.name}_{k}", o.mesh, o.cx, o.cy, z0, yaw,
                                        fx, fy, zs))
        else:                                            # apron staging pallets
            meshes.append(mesh_link(f"m_{o.name}", "ClutteringC_01", o.cx, o.cy, 0.0, 0.0,
                                    o.sx / MESHES["ClutteringC_01"]["size"][0],
                                    o.sy / MESHES["ClutteringC_01"]["size"][1],
                                    o.h / MESHES["ClutteringC_01"]["size"][2]))

    # ---- permanent racking steel, and floor bay markings ---------------------
    # The frames stand at the section's DESIGN height in every stock state: real
    # racking steel does not come and go, only the goods on it do. They are
    # visual-only, so the occluder is the goods (a solid pallet load), not the
    # open steel -- which is the honest way round for a sight-line model.
    frames = []
    for z in L.zones:
        if z.name.startswith("A"):
            colour, accent, design = FRAME_A, BEAM_A, A_TOP["full"]
        elif z.name.startswith("B"):
            colour, accent, design = FRAME_B, BEAM_B, B_TOP["full"]
        else:
            continue
        long_x = z.sx >= z.sy
        run = z.sx if long_x else z.sy
        n_bays = max(1, int(round(run / MODULE)))
        posts = [(-run / 2) + run * k / n_bays for k in range(n_bays + 1)]
        for k, t in enumerate(posts):
            for side in (-1, 1):
                cx = (z.cx + t) if long_x else (z.cx + side * (z.sx / 2 - 0.06))
                cy = (z.cy + side * (z.sy / 2 - 0.06)) if long_x else (z.cy + t)
                frames.append(pbr_slab(f"post_{z.name}_{k}{'ns'[side > 0]}", cx, cy,
                                       design / 2, 0.085, 0.085, design, "metal",
                                       tint=colour, metal_map=True))
        # NO painted beams. Selective racking is two-tone -- coloured uprights,
        # contrasting load beams -- and this rack already is: the ShelfD/E mesh
        # supplies the beams, in its own yellow-orange steel. Painting a second
        # set of beams on top put THREE steel colours in one rack, which is not a
        # thing that exists. The uprights below are the painted half of the pair,
        # and the mesh is the other half.
        #
        # It also keeps the section difference honest: A and B differ by upright
        # colour, which is how two racking installations from different suppliers
        # actually differ, rather than by a decorative stripe.
        _ = accent
        for side in (-1, 1):                              # kick plate at floor level
            cx = z.cx if long_x else z.cx + side * (z.sx / 2 - 0.06)
            cy = z.cy + side * (z.sy / 2 - 0.06) if long_x else z.cy
            frames.append(pbr_slab(f"kick_{z.name}_{'ns'[side > 0]}", cx, cy, 0.10,
                                   run if long_x else 0.10, 0.10 if long_x else run,
                                   0.20, "metal", tint=colour, metal_map=True))
        # End-of-run guards. They sit 0.16 m outside the rack, i.e. INSIDE the
        # 0.32 m keep-out envelope the map already declares, so they add colour
        # and a real-looking detail without taking a cell off the driveable map.
        for sx_s in (-1, 1):
            for sy_s in (-1, 1):
                frames.append(pbr_slab(
                    f"guard_{z.name}_{'wm'[sx_s > 0]}{'sn'[sy_s > 0]}",
                    z.cx + sx_s * (z.sx / 2 + 0.16), z.cy + sy_s * (z.sy / 2 + 0.16),
                    0.30, 0.13, 0.13, 0.60, "metal", tint=GUARD_YELLOW,
                    metal_map=True))

    # ---- wall lining: precast panels -----------------------------------------
    # The shell's structural boxes stay exactly as they are -- they are what the
    # sight-line model and the closure checks read. These are a visual lining on
    # the INNER face only, panelised for the same UV reason as the floor bays,
    # and reusing the floor's material set so they cost no extra texture memory.
    # Precast concrete panels are what a DC of this size is actually built from.
    PANEL = 6.0
    WALL_TINT = "0.97 0.97 0.94 1"   # -> ~0.70 precast grey
    wall_panels = []
    for i in range(int(round((HALL_X[1] - HALL_X[0]) / PANEL))):
        w = (HALL_X[1] - HALL_X[0]) / int(round((HALL_X[1] - HALL_X[0]) / PANEL))
        cx = HALL_X[0] + w * (i + 0.5)
        wall_panels.append(pbr_slab(f"panel_n_{i}", cx, HALL_Y[1] - 0.11,
                                    WALL_H / 2, w - 0.02, 0.02, WALL_H - 0.02,
                                    "concrete", tint=WALL_TINT))
    for j in range(int(round((HALL_Y[1] - HALL_Y[0]) / PANEL))):
        h = (HALL_Y[1] - HALL_Y[0]) / int(round((HALL_Y[1] - HALL_Y[0]) / PANEL))
        cy = HALL_Y[0] + h * (j + 0.5)
        for side, x in (("w", HALL_X[0] + 0.11), ("e", HALL_X[1] - 0.11)):
            wall_panels.append(pbr_slab(f"panel_{side}_{j}", x, cy, WALL_H / 2,
                                        0.02, h - 0.02, WALL_H - 0.02,
                                        "concrete", tint=WALL_TINT))

    # ---- floor: poured in bays -----------------------------------------------
    FLOOR_BAY = 6.0
    floor_slabs = []
    nx = int(round((HALL_X[1] - HALL_X[0]) / FLOOR_BAY))
    ny = int(round((HALL_Y[1] - HALL_Y[0]) / FLOOR_BAY))
    bw = (HALL_X[1] - HALL_X[0]) / nx
    bh = (HALL_Y[1] - HALL_Y[0]) / ny
    for i in range(nx):
        for j in range(ny):
            floor_slabs.append(pbr_slab(
                f"floor_bay_{i}_{j}",
                HALL_X[0] + bw * (i + 0.5), HALL_Y[0] + bh * (j + 0.5), 0.005,
                bw - 0.015, bh - 0.015, 0.01, "concrete"))

    # ---- floor markings ------------------------------------------------------
    # These used to be two nested rectangles round every zone -- a blue "operating
    # field" boundary and a green line at the planner's 0.32 m keep-out envelope
    # -- and neither is a thing that exists on a warehouse floor. They were the
    # declared MAP, painted onto the floor of the world the cameras look at. That
    # is indefensible twice over: no DC paints a planner envelope on its slab,
    # and baking the map into the pixels means a perception system could read the
    # map off the floor, in a study whose whole point is what the map does not
    # declare.
    #
    # What is left is what a real floor carries: a painted bay outline round each
    # FLOOR-STORAGE row, because block storage really is marked out that way, and
    # nothing round the racking, because a rack's own foot is its boundary and
    # nobody paints a second line beside it.
    marks = []
    # The line every storage block gets is drawn at the 0.32 m keep-out envelope,
    # so the clear floor inside it IS the driveable region and you can see where
    # the robot may go by looking at the floor. That is a real marking, not a
    # debug overlay: demarcating storage blocks so vehicles know where the
    # gangway ends is exactly what these lines are for in a working DC, and the
    # envelope is where the edge genuinely is. It replaces the previous pair of
    # nested rectangles -- a blue "operating field" bound and a separate green
    # envelope line -- which showed the same thing twice in two colours that mean
    # nothing on a warehouse floor.
    for z in L.zones:
        if z.kind == "structure":
            continue                      # the dock office is a building, not a bay
        marks += outline(f"bay_{z.name}", z.xmin, z.xmax, z.ymin, z.ymax,
                         LANE_YELLOW, pad=NOGO, t=0.10)
    # Operating-field perimeter in white: a different marking with a different
    # meaning, so it gets a different colour rather than more yellow.
    cx, cy = 0.5 * (SITE[0] + SITE[1]), 0.5 * (SITE[2] + SITE[3])
    marks += [box_link("site_s", cx, SITE[2], 0.02, SITE[1] - SITE[0], 0.10, 0.012, PERIMETER, collide=False),
              box_link("site_n", cx, SITE[3], 0.02, SITE[1] - SITE[0], 0.10, 0.012, PERIMETER, collide=False),
              box_link("site_w", SITE[0], cy, 0.02, 0.10, SITE[3] - SITE[2], 0.012, PERIMETER, collide=False),
              box_link("site_e", SITE[1], cy, 0.02, 0.10, SITE[3] - SITE[2], 0.012, PERIMETER, collide=False)]

    # ---- painted floor markings, wall finish and signage --------------------
    # All visual-only and all either on the slab or on a wall, so none of it
    # touches the driveable map or the sight-line model -- coverage.py builds its
    # height map from the layout, not from this file. It exists because the AWS
    # mesh pack is one ochre palette: colour in this building has to be painted.
    decor = []
    PZ, PT = 0.022, 0.014                       # paint sits just above the slab

    def paint(name, xmin, xmax, ymin, ymax, colour, yaw=0.0):
        decor.append(box_link(name, 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), PZ,
                              xmax - xmin, ymax - ymin, PT, colour,
                              collide=False, yaw=yaw, finish="paint"))

    # Main artery, marked the way a real one is: solid YELLOW boundary lines
    # defining the aisle (yellow is the industry standard for aisles and
    # gangways; OSHA 1910.22(b) puts the line width at 2-6 in, 4 in typical,
    # which is the 0.10 m used here) and a white dashed line dividing the two
    # directions of travel.
    # A marking that runs past the thing it marks is worse than no marking. These
    # two lines start where the artery starts (the apron edge) and stop where the
    # racking it separates stops, instead of running wall to wall. There is no
    # centre divider: a 2.35 m aisle is one vehicle wide, so a line down the
    # middle of it would be claiming two lanes that do not exist.
    for nm, x in (("w", ARTERY[0] + 0.10), ("e", ARTERY[1] - 0.10)):
        paint(f"lane_artery_{nm}", x - 0.05, x + 0.05, APRON_TOP, A_Y1, LANE_YELLOW)

    # pedestrian walkways: green, against the wall, out of the traffic lanes
    paint("walk_west", -11.28, -10.48, SITE[2], SITE[3], WALK_GREEN)
    paint("walk_north", -11.28, ARTERY[0], 8.48, 9.28, WALK_GREEN)

    # keep-clear hatching in front of every dock door
    for i, d in enumerate(sorted(DOCK_DOORS)):
        for j in range(6):
            sx0 = d - 1.50 + j * 0.52
            decor.append(box_link(f"hatch_{i}_{j}", sx0 + 0.26, -8.84, 0.023,
                                  0.14, 1.55, PT, HATCH_YELLOW,
                                  collide=False, yaw=0.70))

    # painted lower wall band, the industrial two-tone every hall has
    DADO_H, DADO_Z = 1.40, 0.70
    decor.append(box_link("dado_west", HALL_X[0] + 0.14, 0, DADO_Z, 0.03, 20.0,
                          DADO_H, DADO, collide=False))
    decor.append(box_link("dado_east", HALL_X[1] - 0.14, 0, DADO_Z, 0.03, 20.0,
                          DADO_H, DADO, collide=False))
    decor.append(box_link("dado_north", 0, HALL_Y[1] - 0.14, DADO_Z, 24.0, 0.03,
                          DADO_H, DADO, collide=False))
    for i in range(0, len(edges) - 1, 2):        # south wall is in door segments
        a, b = edges[i], edges[i + 1]
        if b - a > 0.05:
            decor.append(box_link(f"dado_south_{i//2}", 0.5 * (a + b),
                                  HALL_Y[0] + 0.14, DADO_Z, b - a, 0.03, DADO_H,
                                  DADO, collide=False))

    # NO wall signage. An untextured coloured rectangle stuck on a wall does not
    # read as a sign, it reads as a rendering artefact -- and under ANSI Z535.1
    # red specifically means fire equipment, so a bare red panel is worse than
    # nothing. Colour in this building comes from things that are actually
    # coloured in a real DC: the racking steel, the paint on the slab, and the
    # pallets under the goods.
    # ---- props ---------------------------------------------------------------
    # Everything a working DC has standing about that this world did not. The
    # placement rule is strict and is the reason the list is short: a prop may
    # only stand inside a DECLARED non-driveable zone (plus its 0.32 m envelope)
    # or outside the operating field. Anything else would put a solid object in
    # driveable space that the map does not declare, and the planner would drive
    # into it. Where a prop does not fit that rule it is left out rather than
    # squeezed in.
    #
    # They are included <static>, unlike dyn_forklift's own dynamic variant:
    # these are parked, not traffic.
    def prop(name, model, x, y, z=0.0, yaw=0.0):
        return (f'    <include><name>{name}</name><uri>model://{model}</uri>'
                f'<static>true</static>'
                f'<pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose></include>')

    zmap = {z.name: z for z in L.zones}
    se, sm, ofc = zmap["STAGE_E"], zmap["STAGE_MID"], zmap["DOCK_OFFICE"]
    props_xml = [
        # A counterbalance truck parked on the apron, forks to the dock. Its
        # 2.73 m footprint fits STAGE_E's envelope only lengthwise along x.
        prop("forklift_parked", "dyn_forklift", se.cx - 0.20, se.cy, 0.0, math.pi),
        # Hand pallet truck and a bin beside the dock office, in its envelope.
        prop("pallet_jack", "aws_robomaker_warehouse_PalletJackB_01",
             ofc.xmax + 0.16, ofc.cy - 0.55, 0.0, math.pi / 2),
        prop("bin_office", "aws_robomaker_warehouse_TrashCanC_01",
             ofc.xmax + 0.14, ofc.ymax + 0.10, 0.0, 0.0),
        # Loose pallets waiting to be put away, on the staging bays.
        prop("pallet_loose_1", "dyn_pallet_box", sm.xmin + 0.62, sm.cy + 0.10, 0.0, 0.12),
        prop("pallet_loose_2", "dyn_pallet_box", se.xmax - 0.55, se.cy + 0.12, 0.0, -0.20),
    ]

    # Lamp_01 is 1.52 m tall and its geometry sits at z in
    # [12.50, 14.02] inside its own DAE, so the include pose has to be NEGATIVE:
    # -5.50 hangs the fixture at 7.00-8.52 m, clear of the 5.23 m racks and under
    # the 9.0 m roof. They never block a wall camera: a ray from 8.0 m at 44 deg
    # is already below 7.0 m one metre out, and no lamp is within 2.5 m of a wall.
    # Sixteen fixtures on the aisle centrelines, which is how a DC is actually
    # lit: continuous rows ALONG each aisle rather than a scatter over the
    # racking, so the light lands on the floor the robot drives on instead of on
    # the top beam.
    #
    # THE FIXTURES WERE NOT ACTUALLY LIGHTING ANYTHING (measured 2026-08-21).
    # Projecting all sixteen lamp positions into the five camera frames and
    # sampling the floor under each one gave a median luminance of 97 against 107
    # midway between them: the floor under a lamp was, if anything, DARKER. The
    # cause is arithmetic rather than mysterious. Ignition attenuates a point
    # light by 1/(constant + linear*d + quadratic*d^2), so at the 7.1 m drop to
    # the floor the old constants (1.10, 0.08, 0.010) left a factor of 1/2.17,
    # and 0.22 diffuse through that is 0.10 of scale -- against a 0.34 scene
    # ambient and a 0.74 directional sun. Every fixture was contributing about a
    # tenth of what the ambient wash already supplied, so the room was lit by
    # nothing in particular and no fixture read as switched on.
    #
    # The fix has three parts, and all three are needed:
    #  1. the fixture output goes up and its attenuation is flattened, so the
    #     light actually arrives at the floor 7.1 m below it;
    #  2. the ambient wash and the sun come down, so the building is lit BY its
    #     fixtures instead of by a uniform grey that hides them. A DC has bright
    #     pools under the fittings and dimmer floor between them, and that
    #     spatial structure is real: it is one of the reasons a place is easier or
    #     harder to detect a robot in, which is exactly what this study measures.
    #     Ambient stays high enough that the darkest aisle interior is still a
    #     usable image rather than a black hole;
    #  3. each fixture gets a visible emissive face. Lamp_01 ships as a passive
    #     housing -- no light, no emissive material -- so even a working fixture
    #     rendered as a grey lump hanging from the roof, which is what "some of
    #     the lights are not turned on" looks like.
    # The light itself moves from 7.10 m, which is INSIDE the housing, to 6.95 m,
    # just below its underside, so the fitting is never between its own lamp and
    # the floor.
    #
    # Nothing here can occlude a camera: the fixtures hang at 7.00-8.52 m and
    # every mount is at 5.00 m looking DOWN, so no sight-line ever reaches lamp
    # height. Lighting is a known null for this detector, so this is expected to
    # change the pictures and not the detection numbers -- but it is now an
    # honest picture, and the illumination field it creates is measurable.
    lamp_xy = [(-7.05, -2.40), (-7.05, 1.50), (-7.05, 5.40),
               (-3.35, -2.40), (-3.35, 1.50), (-3.35, 5.40),
               (0.62, -6.50), (0.62, -1.50), (0.62, 3.50), (0.62, 8.00),
               (3.60, -1.50), (8.40, -1.50), (3.60, 2.15), (8.40, 2.15),
               (-9.00, -6.90), (6.00, -6.90)]
    lamps = "\n".join(
        f'    <include><name>lamp_{i}</name><uri>model://aws_robomaker_warehouse_Lamp_01</uri>'
        f'<pose>{x:.2f} {y:.2f} -5.50 0 0 0</pose></include>\n'
        f'    <model name="lamp_glow_{i}"><static>true</static><pose>{x:.2f} {y:.2f} '
        f'{LAMP_GLOW_Z:.2f} 0 0 0</pose><link name="link"><visual name="visual">'
        f'<geometry><box><size>1.20 0.22 0.03</size></box></geometry>'
        f'<material><ambient>1 0.98 0.92 1</ambient><diffuse>1 0.98 0.92 1</diffuse>'
        f'<specular>0.2 0.2 0.2 1</specular><emissive>1 0.97 0.90 1</emissive></material>'
        f'<cast_shadows>false</cast_shadows></visual></link></model>\n'
        + ('' if not LAMP_LIGHTS_ON else
        f'    <light type="point" name="lamp_light_{i}"><pose>{x:.2f} {y:.2f} '
        f'{LAMP_LIGHT_Z:.2f} 0 0 0</pose>'
        f'<cast_shadows>false</cast_shadows>'
        f'<diffuse>{LAMP_DIFFUSE:.3f} {LAMP_DIFFUSE * 0.977:.3f} {LAMP_DIFFUSE * 0.909:.3f} 1</diffuse>'
        f'<specular>0.05 0.05 0.05 1</specular><attenuation><range>{LAMP_RANGE:.1f}</range>'
        f'<constant>{LAMP_ATT_CONST:.2f}</constant><linear>{LAMP_ATT_LINEAR:.3f}</linear>'
        f'<quadratic>{LAMP_ATT_QUAD:.4f}</quadratic>'
        f'</attenuation></light>') for i, (x, y) in enumerate(lamp_xy))

    cams = [Cam(c, CAMERA_MODELS[c.name], CAMERA_MODELS[c.name]) for c in L.cameras]

    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <!-- warehouse_v2: three storage sections (north-south racking, racking turned
       90 degrees, and floor block stacks) around a dock apron and a central
       artery.  Stock state {state}.
       Generated by experiments/warehouse_v2_sketches/make_world.py - edit the
       generator, not this file. -->
  <world name="{world_name}">
    <physics name="default_physics" type="ode"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor><real_time_update_rate>1000</real_time_update_rate></physics>
    <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>
    <plugin name="gz::sim::systems::UserCommands" filename="gz-sim-user-commands-system"/>
    <plugin name="gz::sim::systems::SceneBroadcaster" filename="gz-sim-scene-broadcaster-system"/>
    <plugin name="gz::sim::systems::Contact" filename="gz-sim-contact-system"/>
    <plugin name="gz::sim::systems::Sensors" filename="gz-sim-sensors-system"><render_engine>ogre2</render_engine></plugin>
    <gravity>0 0 -9.81</gravity><magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field><atmosphere type="adiabatic"/>
    <scene><ambient>{SCENE_AMBIENT:.2f} {SCENE_AMBIENT:.2f} {SCENE_AMBIENT * 1.06:.2f} 1</ambient><background>0.74 0.78 0.82 1</background><shadows>true</shadows></scene>

    <!-- One directional light so the racking has form instead of reading as flat
         clay, plus hung fixtures over the aisles.  Lamp_01's geometry sits at
         z in [12.50, 14.02] inside its own DAE, so the include pose has to be
         NEGATIVE to hang it in this building: -5.30 puts the fixture at 7.20 m. -->
    <light type="directional" name="sun">
      <pose>0 0 12 0 0 0</pose><cast_shadows>true</cast_shadows>
      <diffuse>{SUN_DIFFUSE:.2f} {SUN_DIFFUSE * 0.986:.2f} {SUN_DIFFUSE * 0.946:.2f} 1</diffuse><specular>0.12 0.12 0.12 1</specular>
      <direction>-0.35 0.55 -0.80</direction>
      <attenuation><range>60</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
    </light>
{lamps}
{chr(10).join(props_xml)}

    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>24.5 20.5</size></plane></geometry></collision>
    </link></model>

    <model name="warehouse_v2_floor"><static>true</static>
{chr(10).join(floor_slabs)}
    </model>

    <model name="warehouse_v2_wall_lining"><static>true</static>
{chr(10).join(wall_panels)}
    </model>

    <model name="warehouse_shell"><static>true</static>
{chr(10).join(walls)}
{chr(10).join(office_vis)}
    </model>

    <model name="warehouse_v2_occluders"><static>true</static>
{chr(10).join(boxes)}
    </model>

    <model name="warehouse_v2_mesh_visuals"><static>true</static>
{chr(10).join(meshes)}
    </model>

    <!-- Goods built from primitives rather than meshes: steel drums and the
         pallets under them. Kept out of the mesh-visuals model so that model
         stays mesh-only and its 1:1 scale check keeps meaning something. -->
    <model name="warehouse_v2_goods"><static>true</static>
{chr(10).join(goods)}
    </model>

    <model name="warehouse_v2_rack_frames"><static>true</static>
{chr(10).join(frames)}
    </model>

    <model name="warehouse_v2_floor_markings"><static>true</static>
{chr(10).join(marks)}
    </model>

    <!-- Paint, wall finish and signage. Visual-only: no collision anywhere in
         here, and nothing stands in driveable space, so the map and the
         sight-line model are untouched by it. -->
    <model name="warehouse_v2_decor"><static>true</static>
{chr(10).join(decor)}
    </model>

{chr(10).join(camera_include(c) for c in cams)}
    <!-- Overview camera: 19.0 m so the 24 x 20 m hall just fills a 1280x720
         frame at 90 deg, and yawed so north is up and east is right. -->
    <include><name>presentation_overview_camera</name><uri>model://presentation_overview_camera</uri><pose>0.00 0.00 19.00 0 1.5708 1.5708</pose></include>
    <!-- Checking aid: 0.90 rad at 42 m spans 40.6 x 30.5 m, so a 5.23 m rack at the
         far wall spills sideways by 1.1 m instead of 2.6 m at 19 m. -->
    <include><name>plan_view_camera</name><uri>model://plan_view_camera</uri><pose>0.00 0.00 42.00 0 1.5708 1.5708</pose></include>
  </world>
</sdf>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="A", choices=["A", "B"])
    ap.add_argument("--no-lamp-lights", action="store_true",
                    help=("diagnostic only: omit the sixteen fixture lights so their "
                          "contribution can be measured by differencing two renders. "
                          "Never use for a captured dataset."))
    ap.add_argument(
        "--out",
        default=None,
        help=("output SDF; defaults to the canonical filename for --state so "
              "--state B cannot overwrite the peak world"),
    )
    a = ap.parse_args()
    if a.no_lamp_lights:
        LAMP_LIGHTS_ON = False
        if not a.out:
            raise SystemExit("--no-lamp-lights is a diagnostic and requires an explicit --out")
    p = pathlib.Path(a.out) if a.out else WORLD_DIR / WORLD_FILES[a.state]
    p.write_text(build_world(a.state))
    shown = p.resolve()
    try:
        shown = shown.relative_to(Path(REPO).resolve())
    except ValueError:
        pass
    print(f"wrote {shown}  ({p.stat().st_size/1024:.0f} kB)"
          + ("   [DIAGNOSTIC: fixture lights omitted]" if a.no_lamp_lights else ""))


def world_profile() -> str:
    """The world_profiles.yaml entry: spawn, lanes and the non-drivable zones.

    Lanes are the ones coverage.py derives from the zone map, so the map the
    planner is handed and the map the sketch was scored against are one object.
    """
    import coverage as cov
    L = build()
    cov.analyse(L)                       # populates L.lanes with the derived rectangles
    lanes = "\n".join(
        f"      - {{name: {ln.name}, type: traversable, xmin: {ln.xmin:.2f}, xmax: {ln.xmax:.2f}, "
        f"ymin: {ln.ymin:.2f}, ymax: {ln.ymax:.2f}, note: derived lane rectangle}}"
        for ln in L.lanes)
    zones = "\n".join(
        f"      - {{name: {z.name}, type: non_driveable_obstacle, "
        f"xmin: {z.xmin - NOGO:.2f}, xmax: {z.xmax + NOGO:.2f}, "
        f"ymin: {z.ymin - NOGO:.2f}, ymax: {z.ymax + NOGO:.2f}, "
        f"note: {z.kind} zone plus its 0.32 m keep-out envelope}}"
        for z in L.zones)
    return f"""  warehouse_v2.world.sdf: &warehouse_v2_peak_profile
    world_name: warehouse_v2
    stock_state: peak
    evidence_role: primary_commissioning_and_evaluation
    paired_world: warehouse_v2_shipout.world.sdf
    geometry_freeze_id: WHV2-GEOMETRY-V1
    camera_layout_decision_id: WHV2-CAMLAYOUT-V1
    evidence_status: geometry_only_no_detector_data
    investigation_focus: >-
      Three-section warehouse: north-south selective racking in the west (top beam
      5.23 m), racking turned 90 degrees in the north-east (top beam 2.61 m), and
      floor block-stacking in the south-east where the same pallet is stacked one,
      two and three high. The map declares the storage ZONES; what is inside them
      is not in the map and changes between stock states, so the drivable map
      carries no visibility information. Five cameras, no mirrored pair.
      Generated by experiments/warehouse_v2_sketches/make_world.py.
    recommended_task: fusion_overlap_primary
    spawn: {{x: 0.55, y: -7.50, z: 0.05, yaw: 1.5708}}
    planner_default: visibility_aware_efe
    camera_ids: [camera_A, camera_B, camera_C, camera_D, camera_E]
    camera_model_includes: [external_camera, external_camera_b, external_camera_c, external_camera_d, external_camera_e]
    camera_image_topics: [/external_camera/image_raw, /external_camera_b/image_raw, /external_camera_c/image_raw, /external_camera_d/image_raw, /external_camera_e/image_raw]
    occlusion_model_names: [warehouse_v2_occluders]
    collision_model_names: [warehouse_shell, warehouse_v2_occluders]
    visibility_defaults:
      visibility_map_min_x: -11.35
      visibility_map_max_x: 11.35
      visibility_map_min_y: -9.35
      visibility_map_max_y: 9.35
      visibility_map_nx: 455
      visibility_map_ny: 374
      visibility_gp_length_scale: 1.20
      visibility_gp_noise_var: 0.10
      visibility_prior_occ: 0.005
      visibility_beta: 1.0
      visibility_target_height_m: 0.35
    known_2d_regions:
      - {{name: open_floor_inside_site_boundary, type: site_boundary, xmin: {SITE[0]:.2f}, xmax: {SITE[1]:.2f}, ymin: {SITE[2]:.2f}, ymax: {SITE[3]:.2f}, note: operating field; a planner bound, deliberately NOT painted on the floor}}
{lanes}
{zones}
  warehouse_v2_shipout.world.sdf:
    <<: *warehouse_v2_peak_profile
    world_name: warehouse_v2_shipout
    stock_state: shipout
    evidence_role: paired_stock_state_transfer_only
    paired_world: warehouse_v2.world.sdf
    evidence_status: geometry_only_no_detector_data
"""
