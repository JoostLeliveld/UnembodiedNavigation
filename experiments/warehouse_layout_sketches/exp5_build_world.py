#!/usr/bin/env python3
"""exp5: build the Meerhoven world as a real Gazebo SDF, and verify it.

Takes the approved layout from ``exp4_meerhoven_hub`` as the single source of truth
and emits:

  1. src/sim/gazebo_worlds/worlds/warehouse_meerhoven.world.sdf
  2. the eight ADDITIONAL camera models the 12-camera network needs
     (only external_camera{,_b,_c,_d} existed; they differ from one another purely
     by model name and topic prefix, so the rest are templated)
  3. a verification figure computed FROM THE GENERATED SDF, not from the spec

Point 3 is the part that matters. Reading the world back through
``reliability.projection.camera_model_from_world`` and recomputing coverage proves
the world Gazebo will load matches the drawing that was approved. Anything that
drifted between the plan and the SDF shows up as a coverage difference.

This is a NEW world. It does not replace ``warehouse_full_4cam``, so no locked
calibration, coverage artifact or campaign is orphaned by running this. Promoting it
to the evaluation world is a separate, deliberate decision (see
``RUNBOOK_world_pivot.md`` step 0).

Outputs -> src/sim/gazebo_worlds/worlds/, src/sim/models/, and
           logs/studies/warehouse_layout_sketches/exp5_build_world/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

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
sys.path.insert(0, str(REPO / "src" / "reliability"))

import exp4_meerhoven_hub as HUB  # noqa: E402  THE layout spec
import facility_draw as fd  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_meerhoven.world.sdf"
MODELS = REPO / "src/sim/models"
OUT = REPO / "logs/studies/warehouse_layout_sketches/exp5_build_world"

WALL_GRAY = "0.62 0.62 0.62 1"
RACK_YELLOW = "0.93 0.80 0.08 1"
RAIL_BLUE = "0.14 0.34 0.75 1"
CRATE = "0.72 0.58 0.38 1"
MACHINE_GRAY = "0.42 0.50 0.58 1"
DECK_GRAY = "0.55 0.55 0.58 1"
CAGE_MESH = "0.35 0.35 0.35 1"
BLUE_MARK = "0.10 0.45 0.90 1"
DOCK_ORANGE = "0.84 0.37 0.00 1"

#: AWS shelf mesh native dimensions, as measured in make_warehouse_full.py.
MESH_L, MESH_W, MESH_H = 3.917, 0.880, 2.613

#: Camera model directory names, in the order HUB.CAMERAS lists them. The first four
#: reuse the existing models so current tooling keeps working; the rest are templated
#: from external_camera_b, which is the cleanest of the four.
CAMERA_MODEL_NAMES = [
    "external_camera", "external_camera_b", "external_camera_c", "external_camera_d",
    "external_camera_e", "external_camera_f", "external_camera_g", "external_camera_h",
    "external_camera_i", "external_camera_j", "external_camera_k", "external_camera_l",
]

CAMERA_MODEL_SDF = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <!-- Pose is set per-world via <include><pose>. Intrinsics match the deployed
         rig exactly (1280x720, 90 deg HFOV) so the detector stays in distribution
         and reliability.projection reproduces the same ground projection. -->
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <sensor name="camera" type="camera">
        <camera>
          <horizontal_fov>1.5708</horizontal_fov>
          <image><width>1280</width><height>720</height></image>
          <clip><near>0.1</near><far>100</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>5</update_rate>
        <visualize>false</visualize>
        <topic>{name}/image_raw</topic>
      </sensor>
      <sensor name="depth_camera" type="depth_camera">
        <camera>
          <horizontal_fov>1.5708</horizontal_fov>
          <image><width>1280</width><height>720</height><format>R_FLOAT32</format></image>
          <clip><near>0.1</near><far>100</far></clip>
        </camera>
        <always_on>0</always_on>
        <update_rate>5</update_rate>
        <visualize>false</visualize>
        <topic>{name}/depth</topic>
      </sensor>
      <sensor name="segmentation_camera" type="segmentation">
        <camera>
          <segmentation_type>semantic</segmentation_type>
          <horizontal_fov>1.5708</horizontal_fov>
          <image><width>1280</width><height>720</height></image>
          <clip><near>0.1</near><far>100</far></clip>
        </camera>
        <always_on>0</always_on>
        <update_rate>1</update_rate>
        <visualize>false</visualize>
        <topic>{name}/segmentation</topic>
      </sensor>
    </link>
  </model>
</sdf>
"""

CAMERA_MODEL_CONFIG = """<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>Joost Leliveld</name><email>joost.leliveld@example.com</email></author>
  <description>
    External overhead camera for BEV localization. Templated by
    experiments/warehouse_layout_sketches/exp5_build_world.py for the
    twelve-camera Meerhoven network; identical to external_camera_b apart from the
    model name and topic prefix.
  </description>
</model>
"""


# --------------------------------------------------------------- SDF primitives


def box_link(name, cx, cy, sx, sy, sz, color, *, collide=True, z=None, visual=True):
    zc = sz / 2 if z is None else z
    parts = [f'      <link name="{name}"><pose>{cx:.3f} {cy:.3f} {zc:.3f} 0 0 0</pose>']
    if collide:
        parts.append(f'<collision name="collision"><geometry><box><size>'
                     f'{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>')
    if visual:
        parts.append(f'<visual name="visual"><geometry><box><size>'
                     f'{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>'
                     f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse>'
                     f'</material></visual>')
    parts.append("</link>")
    return "".join(parts)


def rack_links(name, cx, cy, sx, sy, h, mesh):
    """Collision box (invisible) plus a scaled real AWS shelf mesh as the look."""
    out = [box_link(f"rack_{name}", cx, cy, sx, sy, h, RACK_YELLOW, visual=False)]
    along_y = sy >= sx
    length = sy if along_y else sx
    depth = sx if along_y else sy
    for side, d in (("a", -depth / 2 + 0.02), ("b", depth / 2 - 0.02)):
        if along_y:
            out.append(box_link(f"rail_{name}_{side}", cx + d, cy, 0.04, length, 0.10,
                                RAIL_BLUE, collide=False, z=h + 0.07))
        else:
            out.append(box_link(f"rail_{name}_{side}", cx, cy + d, length, 0.04, 0.10,
                                RAIL_BLUE, collide=False, z=h + 0.07))
    yaw = 1.5708 if along_y else 0.0
    scale_l, scale_w, scale_h = length / MESH_L, depth / MESH_W, h / MESH_H
    out.append(
        f'      <link name="mesh_{name}"><pose>{cx:.3f} {cy:.3f} 0 0 0 0</pose>'
        f'<visual name="visual"><pose>0 0 0 0 0 {yaw}</pose><geometry><mesh>'
        f'<uri>model://aws_robomaker_warehouse_{mesh}/meshes/'
        f'aws_robomaker_warehouse_{mesh}_visual.DAE</uri>'
        f'<scale>{scale_l:.4f} {scale_w:.4f} {scale_h:.4f}</scale>'
        f'</mesh></geometry></visual></link>')
    return out


def block_stack_links(name, cx, cy, sx, sy, h):
    """Bulk goods: three stacked crate tiers, slightly inset, so it reads as a pile."""
    out = []
    tiers = max(2, int(round(h / 0.8)))
    for i in range(tiers):
        inset = 0.06 * i
        tier_h = h / tiers
        out.append(box_link(f"{name}_t{i}", cx, cy, max(sx - 2 * inset, 0.3),
                            max(sy - 2 * inset, 0.3), tier_h, CRATE,
                            collide=(i == 0), z=tier_h * (i + 0.5)))
    return out


def cage_links(name, cx, cy, sx, sy, h):
    """Mesh cage: a solid collision box, but only posts and rails are VISIBLE, so a
    camera sees through it exactly as the plan assumes."""
    out = [box_link(f"{name}_collision", cx, cy, sx, sy, h, CAGE_MESH,
                    collide=True, visual=False)]
    for dx in (-sx / 2, sx / 2):
        for dy in (-sy / 2, sy / 2):
            out.append(box_link(f"{name}_post_{dx:+.1f}_{dy:+.1f}", cx + dx, cy + dy,
                                0.08, 0.08, h, CAGE_MESH, collide=False))
    for frac in (0.35, 0.98):
        out.append(box_link(f"{name}_rail_s{frac:.2f}", cx, cy - sy / 2, sx, 0.04, 0.04,
                            CAGE_MESH, collide=False, z=h * frac))
        out.append(box_link(f"{name}_rail_n{frac:.2f}", cx, cy + sy / 2, sx, 0.04, 0.04,
                            CAGE_MESH, collide=False, z=h * frac))
    return out


def mezzanine_links(x0, y0, x1, y1, deck_z):
    """A raised deck: the slab itself plus corner posts. The slab is what blocks the
    cameras above it, which is why pack_mezz has to hang underneath."""
    out = [box_link("mezz_deck", (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.22,
                    DECK_GRAY, collide=True, z=deck_z)]
    for px in (x0 + 0.25, x1 - 0.25):
        for py in (y0 + 0.25, y1 - 0.25):
            out.append(box_link(f"mezz_post_{px:+.0f}_{py:+.0f}", px, py, 0.18, 0.18,
                                deck_z, DECK_GRAY))
    return out


def conveyor_links(points, width, height=0.62):
    out = []
    for i, (p, q) in enumerate(zip(points[:-1], points[1:])):
        cx, cy = 0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])
        sx = abs(q[0] - p[0]) or width
        sy = abs(q[1] - p[1]) or width
        out.append(box_link(f"conveyor_{i}", cx, cy, sx, sy, 0.16, MACHINE_GRAY,
                            z=height))
        for side in (-1, 1):
            if sx > sy:
                out.append(box_link(f"conveyor_{i}_leg{side}", cx + side * sx / 3, cy,
                                    0.10, 0.10, height, MACHINE_GRAY))
            else:
                out.append(box_link(f"conveyor_{i}_leg{side}", cx, cy + side * sy / 3,
                                    0.10, 0.10, height, MACHINE_GRAY))
    return out


# ------------------------------------------------------------------- the world


def build_world_sdf() -> str:
    X0, X1, Y0, Y1 = HUB.X0, HUB.X1, HUB.Y0, HUB.Y1
    links: list[str] = []

    # shell
    for name, cx, cy, sx, sy in (
        ("wall_west", X0, 0.0, 0.20, Y1 - Y0), ("wall_east", X1, 0.0, 0.20, Y1 - Y0),
        ("wall_south", 0.0, Y0, X1 - X0, 0.20), ("wall_north", 0.0, Y1, X1 - X0, 0.20),
    ):
        links.append(box_link(name, cx, cy, sx, sy, 7.20, WALL_GRAY, z=3.60))

    # elements, straight from the approved spec
    mesh_cycle = ("ShelfD_01", "ShelfE_01", "ShelfF_01")
    for i, e in enumerate(HUB.ELEMENTS):
        n, x, y, sx, sy, h = (e["name"], e["x"], e["y"], e["sx"], e["sy"], e["height"])
        if e["kind"] == "rack":
            links += rack_links(n, x, y, sx, sy, h, mesh_cycle[i % len(mesh_cycle)])
        elif e["kind"] == "block":
            links += block_stack_links(n, x, y, sx, sy, h)
        elif e["kind"] == "cage":
            links += cage_links(n, x, y, sx, sy, h)
        elif e["kind"] in ("machine", "station"):
            links.append(box_link(n, x, y, sx, sy, h, MACHINE_GRAY))
        # 'prop' elements come in below as real AWS mesh includes

    mx0, my0, mx1, my1 = HUB.MEZZANINE
    links += mezzanine_links(mx0, my0, mx1, my1, HUB.MEZZANINE_DECK_Z)
    links += conveyor_links(HUB.CONVEYOR, 0.8)

    for cx in HUB.COLUMN_XS:
        for cy in HUB.COLUMN_YS:
            links.append(box_link(f"column_{cx:+.0f}_{cy:+.0f}", cx, cy,
                                  HUB.COLUMN_SIZE, HUB.COLUMN_SIZE, 7.00, WALL_GRAY))

    # floor markings: site boundary, dock doors, guardrails, walkway
    marks = []
    # drivable envelope, matching HUB.build_grid()'s 0.4 m wall inset
    sx0, sx1, sy0, sy1 = X0 + 0.4, X1 - 0.4, Y0 + 0.4, Y1 - 0.4
    for nm, cx, cy, mx, my in (("bnd_s", 0, sy0, sx1 - sx0, 0.05),
                               ("bnd_n", 0, sy1, sx1 - sx0, 0.05),
                               ("bnd_w", sx0, 0, 0.05, sy1 - sy0),
                               ("bnd_e", sx1, 0, 0.05, sy1 - sy0)):
        marks.append(box_link(nm, cx, cy, mx, my, 0.014, BLUE_MARK, collide=False,
                              z=0.035))
    for i, (dx0, dx1, _tag) in enumerate(HUB.DOCK_DOORS):
        marks.append(box_link(f"dock_{i}", 0.5 * (dx0 + dx1), Y0 + 0.14, dx1 - dx0,
                              0.10, 0.014, DOCK_ORANGE, collide=False, z=0.035))
    # Guardrails: posts every 2 m plus two thin rails. Rendering them as one solid
    # 0.95 m box (the first attempt) put an opaque wall across the camera views and
    # made the world disagree with the plan's occlusion model.
    for gi, pts in enumerate(HUB.GUARDRAILS):
        (gx0, gy0), (gx1, gy1) = pts[0], pts[-1]
        horizontal = abs(gx1 - gx0) >= abs(gy1 - gy0)
        span = abs(gx1 - gx0) if horizontal else abs(gy1 - gy0)
        n_posts = max(2, int(span / 2.0) + 1)
        for k in range(n_posts):
            f = k / (n_posts - 1)
            px = gx0 + f * (gx1 - gx0)
            py = gy0 + f * (gy1 - gy0)
            marks.append(box_link(f"grail_{gi}_post{k}", px, py, 0.09, 0.09, 0.95,
                                  "0.90 0.72 0.00 1"))
        for rz in (0.42, 0.90):
            marks.append(box_link(f"grail_{gi}_rail{rz:.2f}",
                                  0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1),
                                  span if horizontal else 0.05,
                                  0.05 if horizontal else span, 0.06,
                                  "0.90 0.72 0.00 1", collide=False, z=rz))
    for wi, pts in enumerate(HUB.WALKWAYS):
        (wx0, wy0), (wx1, wy1) = pts[0], pts[-1]
        marks.append(box_link(f"walkway_{wi}", 0.5 * (wx0 + wx1), 0.5 * (wy0 + wy1),
                              abs(wx1 - wx0) or 1.1, abs(wy1 - wy0) or 1.1, 0.012,
                              "0.86 0.86 0.86 1", collide=False, z=0.030))

    # AWS prop meshes, with their own collisions
    prop_model = {"pallet_jack": "PalletJackB_01", "clutter_inbound": "ClutteringA_01",
                  "trash_ne": "TrashCanC_01", "bucket_pack": "Bucket_01"}
    includes = []
    for e in HUB.ELEMENTS:
        if e["kind"] != "prop":
            continue
        model = prop_model.get(e["name"], "ClutteringD_01")
        includes.append(f'    <include><name>{e["name"]}</name>'
                        f'<uri>model://aws_robomaker_warehouse_{model}</uri>'
                        f'<pose>{e["x"]:.2f} {e["y"]:.2f} 0 0 0 0</pose></include>')
    for e in HUB.ELEMENTS:                          # a desk for the QC bench
        if e["name"] == "returns_bench_1":
            includes.append('    <include><name>qc_desk</name>'
                            '<uri>model://aws_robomaker_warehouse_DeskC_01</uri>'
                            f'<pose>{e["x"] - 1.4:.2f} {e["y"]:.2f} 0 0 0 1.5708</pose>'
                            '</include>')
    for i, (lx, ly) in enumerate(((-8.0, -5.0), (-8.0, 4.0), (8.0, -5.0), (8.0, 4.0))):
        includes.append(f'    <include><name>lamp_{i}</name>'
                        f'<uri>model://aws_robomaker_warehouse_Lamp_01</uri>'
                        f'<pose>{lx:.1f} {ly:.1f} 6.95 0 0 0</pose></include>')

    # the twelve cameras
    for (name, x, y, z, yaw, _prov, _note), model in zip(HUB.CAMERAS,
                                                         CAMERA_MODEL_NAMES):
        pitch = math.radians(HUB.CAMERA_PITCH.get(name, HUB.CAMERA_PITCH_DEG))
        includes.append(f'    <include><name>{model}</name>'
                        f'<uri>model://{model}</uri>'
                        f'<pose>{x:.3f} {y:.3f} {z:.3f} 0 {pitch:.5f} '
                        f'{math.radians(yaw):.5f}</pose></include>'
                        f'<!-- {name}: {_prov} -->')

    return f"""<?xml version="1.0" ?>
<!-- MEERHOVEN REGIONAL HUB — generated by
     experiments/warehouse_layout_sketches/exp5_build_world.py from the approved
     layout in exp4_meerhoven_hub.py. DO NOT EDIT BY HAND: regenerate instead.

     A NEW world. It does not replace warehouse_full_4cam, so no locked artifact is
     orphaned. Twelve cameras, every one bolted to a wall, a building column, or the
     mezzanine edge beam. -->
<sdf version="1.8">
  <world name="warehouse_meerhoven">
    <physics name="default_physics" type="ode"><max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <scene><ambient>0.72 0.72 0.72 1</ambient><background>0.82 0.86 0.90 1</background>
      <grid>false</grid></scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows><pose>0 0 12 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.25 0.25 0.25 1</specular>
      <direction>-0.4 0.3 -0.9</direction></light>

    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="collision"><geometry><plane><normal>0 0 1</normal>
        <size>200 200</size></plane></geometry></collision>
      <visual name="visual"><geometry><plane><normal>0 0 1</normal>
        <size>200 200</size></plane></geometry>
        <material><ambient>0.55 0.55 0.55 1</ambient>
          <diffuse>0.58 0.58 0.58 1</diffuse></material></visual>
    </link></model>

    <model name="facility"><static>true</static>
{chr(10).join(links)}
    </model>

    <model name="floor_markings"><static>true</static>
{chr(10).join(marks)}
    </model>

{chr(10).join(includes)}
  </world>
</sdf>
"""


def write_camera_models() -> list[str]:
    created = []
    for name in CAMERA_MODEL_NAMES:
        d = MODELS / name
        if d.exists():
            continue
        d.mkdir(parents=True)
        (d / "model.sdf").write_text(CAMERA_MODEL_SDF.format(name=name),
                                     encoding="utf-8")
        (d / "model.config").write_text(CAMERA_MODEL_CONFIG.format(name=name),
                                        encoding="utf-8")
        created.append(name)
    return created


# ------------------------------------------------------------- verify the SDF


def verify_from_sdf():
    """Read the world BACK and recompute coverage from it. This is the check that
    the world Gazebo will load is the world that was approved."""
    xs, ys, gx, gy, drivable = HUB.build_grid()
    counts = np.zeros(gx.shape, dtype=int)
    per_camera = {}
    for (name, _x, _y, _z, _yaw, prov, _n), model_name in zip(HUB.CAMERAS,
                                                              CAMERA_MODEL_NAMES):
        model = camera_model_from_world(WORLD, include_name=model_name)
        seen = (HUB.in_frame(model, gx, gy) & HUB.line_of_sight(model, gx, gy)
                & drivable)
        per_camera[name] = float(seen.sum() / max(drivable.sum(), 1))
        counts += seen.astype(int)
    corridor = HUB.importance_weights(gx, gy, drivable) >= 0.99
    return {
        "per_camera_floor_share": per_camera,
        "whole_floor": {
            "unseen": float(np.mean(counts[drivable] == 0)),
            "single": float(np.mean(counts[drivable] == 1)),
            "redundant": float(np.mean(counts[drivable] >= 2)),
        },
        "operating_corridor": {
            "unseen": float(np.mean(counts[corridor] == 0)),
            "single": float(np.mean(counts[corridor] == 1)),
            "three_plus": float(np.mean(counts[corridor] >= 3)),
        },
    }, (xs, ys, counts, drivable)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    created = write_camera_models()
    WORLD.parent.mkdir(parents=True, exist_ok=True)
    WORLD.write_text(build_world_sdf(), encoding="utf-8")

    root = ET.parse(WORLD).getroot()                # parses, or this raises
    n_links = len(root.findall(".//link"))
    n_includes = len(root.findall(".//include"))
    n_meshes = len(root.findall(".//mesh"))

    from_sdf, (xs, ys, counts, drivable) = verify_from_sdf()

    # compare against the plan
    plan_shares = {}
    for name, x, y, z, yaw, _prov, _n in HUB.CAMERAS:
        model = HUB.camera_model(x, y, z, yaw,
                                 pitch_deg=HUB.CAMERA_PITCH.get(name))
        gxg, gyg = np.meshgrid(xs, ys)
        seen = (HUB.in_frame(model, gxg, gyg) & HUB.line_of_sight(model, gxg, gyg)
                & drivable)
        plan_shares[name] = float(seen.sum() / max(drivable.sum(), 1))
    worst = max(abs(plan_shares[k] - from_sdf["per_camera_floor_share"][k])
                for k in plan_shares)

    payload = {"world": str(WORLD.relative_to(REPO)),
               "camera_models_created": created,
               "sdf": {"links": n_links, "includes": n_includes, "mesh_visuals": n_meshes,
                       "bytes": WORLD.stat().st_size},
               "verified_from_sdf": from_sdf,
               "plan_vs_sdf_worst_share_difference": worst}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # a figure computed FROM the SDF
    fig, ax = plt.subplots(figsize=(13.2, 8.6))
    shaded = np.where(drivable, np.minimum(counts, 4), np.nan)
    mesh = ax.pcolormesh(xs, ys, shaded, cmap="YlGnBu", vmin=0, vmax=4,
                         shading="auto", zorder=0)
    for e in HUB.ELEMENTS:
        if e["kind"] == "rack":
            fd.rack_run(ax, e["x"], e["y"], e["sx"], e["sy"],
                        tall=e.get("tall", False), bays=1)
        elif e["kind"] == "block":
            fd.block_stack(ax, e["x"], e["y"], e["sx"], e["sy"])
        elif e["kind"] == "cage":
            fd.cage(ax, e["x"], e["y"], e["sx"], e["sy"])
    fd.wall_poche(ax, HUB.X0, HUB.Y0, HUB.X1, HUB.Y1, thickness=0.45)
    fd.column_grid(ax, HUB.COLUMN_XS, HUB.COLUMN_YS, size=HUB.COLUMN_SIZE, tags=False)
    for (name, _x, _y, _z, _yaw, prov, _n), model_name in zip(HUB.CAMERAS,
                                                              CAMERA_MODEL_NAMES):
        model = camera_model_from_world(WORLD, include_name=model_name)
        cx, cy, _cz = model.cam_pos
        look = model.look_at
        yaw = math.degrees(math.atan2(look[1] - cy, look[0] - cx))
        fd.camera(ax, cx, cy, yaw, inherited=(prov == "inherited"), show_fov=False,
                  label=name, label_offset=(0, -13))
    fd.route(ax, [(m["x"], m["y"]) for m in HUB.MISSION])
    fd.style_axes(ax, HUB.X0, HUB.Y0, HUB.X1, HUB.Y1, margin=1.4)
    c = from_sdf["operating_corridor"]
    ax.set_title("warehouse_meerhoven.world.sdf — coverage recomputed FROM THE "
                 "GENERATED WORLD\n"
                 f"corridor: unseen {100 * c['unseen']:.1f} % · single "
                 f"{100 * c['single']:.1f} % · three+ {100 * c['three_plus']:.1f} %   |   "
                 f"plan-vs-SDF worst per-camera difference "
                 f"{100 * worst:.3f} pp",
                 fontsize=11.5, fontweight="bold")
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8, ticks=range(5))
    cb.set_label("cameras with line of sight")
    cb.ax.set_yticklabels(["0", "1", "2", "3", "4+"])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_w1_world_verified.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {WORLD.relative_to(REPO)}  "
          f"({WORLD.stat().st_size / 1024:.0f} KB, {n_links} links, "
          f"{n_includes} includes, {n_meshes} mesh visuals)")
    print(f"camera models created: {', '.join(created) if created else '(all existed)'}")
    print(f"SDF parses: yes")
    print(f"plan vs SDF, worst per-camera coverage difference: {100 * worst:.4f} pp")
    w, c = from_sdf["whole_floor"], from_sdf["operating_corridor"]
    print(f"from the SDF — whole floor: unseen {100 * w['unseen']:.1f}%  "
          f"single {100 * w['single']:.1f}%  redundant {100 * w['redundant']:.1f}%")
    print(f"from the SDF — corridor:    unseen {100 * c['unseen']:.1f}%  "
          f"single {100 * c['single']:.1f}%  three+ {100 * c['three_plus']:.1f}%")
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
