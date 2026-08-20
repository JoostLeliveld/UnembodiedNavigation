#!/usr/bin/env python3
"""The CURRENT world, measured the same way, so the sketches have a 'before'.

Geometry is parsed straight out of warehouse_full_4cam.world.sdf (collision
boxes with contact sensors are the operational geometry there) and the lanes
come from world_profiles.yaml, so nothing is retyped from the generator.
"""
from __future__ import annotations

import math
import pathlib
import xml.etree.ElementTree as ET

import yaml

from layouts import Camera, Lane, Layout, Obstacle, Zone

REPO = pathlib.Path(__file__).resolve().parents[2]
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
CAM_COLOURS = ["#2f80ed", "#27ae60", "#9b51e0", "#f2994a", "#e05a5a"]


def load_current() -> Layout:
    root = ET.parse(WORLD).getroot().find("world")
    L = Layout("W0", "warehouse_full_4cam (current)",
               "The world in the repo today: two mirrored four-row rack blocks "
               "either side of a 4.5 m central aisle, four cameras in two "
               "mirrored pairs on the north and south walls.",
               "Baseline for comparison. Its zones ARE its objects, so the "
               "drivable map and the occluders are the same rectangles.")

    for m in root.findall("model"):
        if m.get("name") != "warehouse_rack_occluders":
            continue
        for link in m.findall("link"):
            nm = link.get("name")
            if not (nm.startswith("rack_") or nm == "support_pillar"):
                continue
            px, py, pz = [float(v) for v in link.find("pose").text.split()[:3]]
            sx, sy, sz = [float(v) for v in link.find("collision/geometry/box/size").text.split()]
            L.zones.append(Zone(nm, px - sx / 2, px + sx / 2, py - sy / 2, py + sy / 2))
            o = Obstacle(nm, px, py, sx, sy, 0.0, sz,
                         "ShelfE_01" if sz < 5 else "WallB_01", nm)
            L.fill_a.append(o)
            L.fill_b.append(o)          # nothing about this world is reconfigurable

    prof = yaml.safe_load(PROFILE.read_text())["worlds"]["warehouse_full_4cam.world.sdf"]
    for r in prof.get("known_2d_regions", []):
        if r.get("type") == "traversable":
            L.lanes.append(Lane(r["name"], float(r["xmin"]), float(r["xmax"]),
                                float(r["ymin"]), float(r["ymax"])))

    # cameras: pose is x y z roll pitch yaw, pitch positive = tilted down
    for i, inc in enumerate(root.findall("include")):
        nm = inc.find("name").text
        if not nm.startswith("external_camera"):
            continue
        v = [float(t) for t in inc.find("pose").text.split()]
        L.cameras.append(Camera(nm.replace("external_camera", "cam").strip("_") or "camA",
                                v[0], v[1], v[2], math.degrees(v[5]), math.degrees(v[4]),
                                "wall mount", CAM_COLOURS[len(L.cameras) % 5]))
    L.notes = [
        "Cameras sit at 6.10 m but the walls are only 4.50 m tall, so they float "
        "1.60 m above the wall they are mounted on.",
        "Fill states A and B are identical here: there is nothing in this world "
        "that can be restocked without moving an obstacle the map declares.",
    ]
    return L


if __name__ == "__main__":
    L = load_current()
    print(f"{len(L.zones)} rack/pillar boxes, {len(L.lanes)} declared lanes, {len(L.cameras)} cameras")
    for c in L.cameras:
        print(f"   {c.name:<6} ({c.x:+.2f}, {c.y:+.2f}, {c.z:.2f})  yaw {c.yaw_deg:+.1f}  pitch {c.pitch_deg:.1f}")
