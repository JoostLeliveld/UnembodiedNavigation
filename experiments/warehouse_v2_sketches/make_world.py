#!/usr/bin/env python3
"""Emit warehouse_v2.world.sdf from warehouse_v2.py -- edit the layout, not the SDF.

Follows the same construction as make_warehouse_full.py, which is the pattern the
rest of the toolchain expects:

  * box collisions WITH contact sensors are the operational geometry (planner,
    logger and CAD parser all read those),
  * the AWS meshes are visual-only overlays sitting exactly on those boxes,
  * floor paint is visual-only: a blue site boundary and a green keep-out
    envelope around every declared zone.

Meshes are never stretched vertically to reach a stock height. A 4.2 m rack is
one native 2.613 m ShelfD module with a second, shorter one on top; a
three-high block stack is three native pallets at 0.00 / 1.06 / 2.12 m. That
keeps the rendered shelf in the distribution the detector was trained on.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mesh_library import MESHES                      # noqa: E402
from warehouse_v2 import (APRON_TOP, A_TOP, B_TOP, C_ROWS, DOCK_DOORS, MODULE,
                          SITE, build)  # noqa: E402

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
FRAME_A = "0.10 0.30 0.55 1"     # section A: blue racking steel
FRAME_B = "0.78 0.34 0.10 1"     # section B: orange racking steel
BAY_PAINT = "0.90 0.76 0.06 1"   # section C: yellow floor bay markings
KICK_PLATE = "0.85 0.83 0.16 1"  # yellow rack-end kick plates

RACK_MESH_H = MESHES["ShelfD_01"]["size"][2]         # 2.613
PALLET = "ClutteringA_01"
PALLET_H = MESHES[PALLET]["size"][2]                 # 1.058
PALLET_SX, PALLET_SY = MESHES[PALLET]["size"][:2]    # 2.146 x 1.983


def box_link(name, cx, cy, z, sx, sy, sz, colour, *, collide=True, contact=True, visual=True):
    p = [f'      <link name="{name}"><pose>{cx:.3f} {cy:.3f} {z:.3f} 0 0 0</pose>']
    if collide:
        p.append(f'<collision name="collision"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}'
                 f'</size></box></geometry></collision>')
        if contact:
            p.append('<sensor name="contact" type="contact"><contact><collision>collision'
                     '</collision></contact><update_rate>60</update_rate></sensor>')
    if visual:
        p.append(f'<visual name="visual"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size>'
                 f'</box></geometry><material><ambient>{colour}</ambient>'
                 f'<diffuse>{colour}</diffuse></material></visual>')
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
                              3.20, 0.60, 0.014, "0.85 0.72 0.10 1", collide=False))

    # ---- storage contents ----------------------------------------------------
    boxes, meshes = [], []
    for o in fill:
        if o.zone == "DOCK_OFFICE":
            boxes.append(box_link("dock_office", o.cx, o.cy, o.h / 2, o.sx, o.sy, o.h, OFFICE_C))
            continue
        boxes.append(box_link(f"obs_{o.name}", o.cx, o.cy, o.h / 2, o.sx, o.sy, o.h,
                              WALL_C, visual=False))
        if o.zone.startswith("C"):                       # block-stacked pallets
            n = max(1, int(round(o.h / PALLET_H)))
            rot = o.sx < o.sy                            # bay turns the pallet
            yaw = math.pi / 2 if rot else 0.0
            fx = (o.sy if rot else o.sx) / PALLET_SX
            fy = (o.sx if rot else o.sy) / PALLET_SY
            for k in range(n):
                meshes.append(mesh_link(f"m_{o.name}_{k}", PALLET, o.cx, o.cy,
                                        k * PALLET_H, yaw, fx, fy, 1.0))
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
            colour, design = FRAME_A, A_TOP["full"]
        elif z.name.startswith("B"):
            colour, design = FRAME_B, B_TOP["full"]
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
                frames.append(box_link(f"post_{z.name}_{k}{'ns'[side > 0]}", cx, cy,
                                       design / 2, 0.085, 0.085, design, colour,
                                       collide=False))
        for side in (-1, 1):                              # top rail along each face
            cx = z.cx if long_x else z.cx + side * (z.sx / 2 - 0.06)
            cy = z.cy + side * (z.sy / 2 - 0.06) if long_x else z.cy
            sx = run if long_x else 0.07
            sy = 0.07 if long_x else run
            frames.append(box_link(f"rail_{z.name}_{'ns'[side > 0]}", cx, cy,
                                   design + 0.05, sx, sy, 0.07, colour, collide=False))
        for side in (-1, 1):                              # kick plate at floor level
            cx = z.cx if long_x else z.cx + side * (z.sx / 2 - 0.06)
            cy = z.cy + side * (z.sy / 2 - 0.06) if long_x else z.cy
            frames.append(box_link(f"kick_{z.name}_{'ns'[side > 0]}", cx, cy, 0.10,
                                   run if long_x else 0.10, 0.10 if long_x else run,
                                   0.20, KICK_PLATE, collide=False))
    for z in L.zones:                                     # painted floor bays in C
        if not z.name.startswith("C"):
            continue
        frames += outline(f"bay_{z.name}", z.xmin, z.xmax, z.ymin, z.ymax,
                          BAY_PAINT, pad=0.0, t=0.10)

    # ---- floor paint ---------------------------------------------------------
    marks = []
    cx, cy = 0.5 * (SITE[0] + SITE[1]), 0.5 * (SITE[2] + SITE[3])
    marks += [box_link("site_s", cx, SITE[2], 0.02, SITE[1] - SITE[0], 0.07, 0.012, BLUE, collide=False),
              box_link("site_n", cx, SITE[3], 0.02, SITE[1] - SITE[0], 0.07, 0.012, BLUE, collide=False),
              box_link("site_w", SITE[0], cy, 0.02, 0.07, SITE[3] - SITE[2], 0.012, BLUE, collide=False),
              box_link("site_e", SITE[1], cy, 0.02, 0.07, SITE[3] - SITE[2], 0.012, BLUE, collide=False)]
    for z in L.zones:
        marks += outline(f"nogo_{z.name}", z.xmin, z.xmax, z.ymin, z.ymax)

    # Six fixtures. Lamp_01 is 1.52 m tall and its geometry sits at z in
    # [12.50, 14.02] inside its own DAE, so the include pose has to be NEGATIVE:
    # -5.50 hangs the fixture at 7.00-8.52 m, clear of the 4.2 m racks and under
    # the 9.0 m roof. They never block a wall camera: a ray from 8.0 m at 44 deg
    # is already below 7.0 m one metre out, and no lamp is within 2.5 m of a wall.
    lamp_xy = [(-8.90, -1.20), (-8.90, 5.20), (-3.30, 2.00),
               (2.90, -2.20), (2.90, 5.60), (8.60, 1.00)]
    lamps = "\n".join(
        f'    <include><name>lamp_{i}</name><uri>model://aws_robomaker_warehouse_Lamp_01</uri>'
        f'<pose>{x:.2f} {y:.2f} -5.50 0 0 0</pose></include>\n'
        f'    <light type="point" name="lamp_light_{i}"><pose>{x:.2f} {y:.2f} 7.10 0 0 0</pose>'
        f'<cast_shadows>false</cast_shadows><diffuse>0.30 0.29 0.27 1</diffuse>'
        f'<specular>0.03 0.03 0.03 1</specular><attenuation><range>16</range>'
        f'<constant>1.10</constant><linear>0.08</linear><quadratic>0.010</quadratic>'
        f'</attenuation></light>' for i, (x, y) in enumerate(lamp_xy))

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
    <scene><ambient>0.42 0.42 0.44 1</ambient><background>0.74 0.78 0.82 1</background><shadows>true</shadows></scene>

    <!-- One directional light so the racking has form instead of reading as flat
         clay, plus hung fixtures over the aisles.  Lamp_01's geometry sits at
         z in [12.50, 14.02] inside its own DAE, so the include pose has to be
         NEGATIVE to hang it in this building: -5.30 puts the fixture at 7.20 m. -->
    <light type="directional" name="sun">
      <pose>0 0 12 0 0 0</pose><cast_shadows>true</cast_shadows>
      <diffuse>0.62 0.62 0.60 1</diffuse><specular>0.08 0.08 0.08 1</specular>
      <direction>-0.35 0.55 -0.80</direction>
      <attenuation><range>60</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
    </light>
{lamps}

    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>24.5 20.5</size></plane></geometry></collision>
      <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>24.5 20.5</size></plane></geometry><material><ambient>{FLOOR_C}</ambient><diffuse>{FLOOR_C}</diffuse><specular>0.04 0.04 0.04 1</specular></material></visual>
    </link></model>

    <model name="warehouse_shell"><static>true</static>
{chr(10).join(walls)}
    </model>

    <model name="warehouse_v2_occluders"><static>true</static>
{chr(10).join(boxes)}
    </model>

    <model name="warehouse_v2_mesh_visuals"><static>true</static>
{chr(10).join(meshes)}
    </model>

    <model name="warehouse_v2_rack_frames"><static>true</static>
{chr(10).join(frames)}
    </model>

    <model name="known_driveable_boundary"><static>true</static>
{chr(10).join(marks)}
    </model>

{chr(10).join(camera_include(c) for c in cams)}
    <!-- Overview camera: 19.0 m so the 24 x 20 m hall just fills a 1280x720
         frame at 90 deg, and yawed so north is up and east is right. -->
    <include><name>presentation_overview_camera</name><uri>model://presentation_overview_camera</uri><pose>0.00 0.00 19.00 0 1.5708 1.5708</pose></include>
    <!-- Checking aid: 0.90 rad at 42 m spans 40.6 x 30.5 m, so a 4.2 m rack at the
         far wall spills sideways by 1.1 m instead of 2.6 m at 19 m. -->
    <include><name>plan_view_camera</name><uri>model://plan_view_camera</uri><pose>0.00 0.00 42.00 0 1.5708 1.5708</pose></include>
  </world>
</sdf>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="A", choices=["A", "B"])
    ap.add_argument(
        "--out",
        default=None,
        help=("output SDF; defaults to the canonical filename for --state so "
              "--state B cannot overwrite the peak world"),
    )
    a = ap.parse_args()
    p = pathlib.Path(a.out) if a.out else WORLD_DIR / WORLD_FILES[a.state]
    p.write_text(build_world(a.state))
    print(f"wrote {p.relative_to(REPO)}  ({p.stat().st_size/1024:.0f} kB)")


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
    return f"""  warehouse_v2.world.sdf:
    world_name: warehouse_v2
    investigation_focus: >-
      Three-section warehouse: north-south selective racking in the west (top beam
      4.2 m), racking turned 90 degrees in the north-east (top beam 2.6 m), and
      floor block-stacking in the south-east where the same pallet is stacked one,
      two and three high. The map declares the storage ZONES; what is inside them
      is not in the map and changes between stock states, so the drivable map
      carries no visibility information. Five cameras, no mirrored pair.
      Generated by experiments/warehouse_v2_sketches/make_world.py.
    recommended_task: apron_to_north_cross_aisle
    spawn: {{x: 0.55, y: -7.50, z: 0.05, yaw: 1.5708}}
    planner_default: visibility_aware_efe
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
      - {{name: open_floor_inside_site_boundary, type: site_boundary, xmin: {SITE[0]:.2f}, xmax: {SITE[1]:.2f}, ymin: {SITE[2]:.2f}, ymax: {SITE[3]:.2f}, note: operating field; blue paint marks its edge}}
{lanes}
{zones}
"""
