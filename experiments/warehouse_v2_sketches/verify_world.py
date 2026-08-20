#!/usr/bin/env python3
"""Check the generated SDF against the layout it claims to implement.

A perspective render cannot show a 20 cm error, so the geometry is checked
numerically and the renders are used only for the things numbers cannot show
(lighting, mesh choice, whether it reads as a warehouse).
"""
from __future__ import annotations

import math
import pathlib
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from warehouse_v2 import SITE, build                      # noqa: E402
from make_world import CAMERA_MODELS, WALL_H              # noqa: E402

WORLDS = {
    "A": {"file": "warehouse_v2.world.sdf", "name": "warehouse_v2"},
    "B": {"file": "warehouse_v2_shipout.world.sdf", "name": "warehouse_v2_shipout"},
}
WORLD_DIR = pathlib.Path(__file__).resolve().parents[2] / "src/sim/gazebo_worlds/worlds"
TOL = 1e-3


def boxes(world, model_name):
    out = {}
    for m in world.findall("model"):
        if m.get("name") != model_name:
            continue
        for link in m.findall("link"):
            c = link.find("collision/geometry/box/size")
            v = link.find("visual/geometry/box/size")
            size = c if c is not None else v
            if size is None:
                continue
            p = [float(t) for t in link.find("pose").text.split()]
            s = [float(t) for t in size.text.split()]
            out[link.get("name")] = (p, s)
    return out


def main(state: str = "A"):
    if state not in WORLDS:
        print(f"unknown state {state!r}; expected one of {sorted(WORLDS)}")
        return 1
    L = build()
    fill = L.fill_a if state == "A" else L.fill_b
    spec = WORLDS[state]
    root = ET.parse(WORLD_DIR / spec["file"]).getroot()
    w = root.find("world")
    fails, checks = [], 0

    # 0. Runtime namespace: filename, SDF name and world profile must agree.
    checks += 1
    if w is None:
        print(f"[state {state}: {spec['file']}] FAIL: missing <world>")
        return 1
    if w.get("name") != spec["name"]:
        fails.append(
            f"internal world name {w.get('name')!r} != {spec['name']!r}"
        )

    # 1. every fill object has a collision box at the right place and size
    occ = boxes(w, "warehouse_v2_occluders")
    for o in fill:
        key = "dock_office" if o.zone == "DOCK_OFFICE" else f"obs_{o.name}"
        checks += 1
        if key not in occ:
            fails.append(f"missing collision box for {o.name}")
            continue
        (px, py, pz, *_), (sx, sy, sz) = occ[key]
        for label, got, want in (("x", px, o.cx), ("y", py, o.cy), ("z", pz, o.h / 2),
                                 ("sx", sx, o.sx), ("sy", sy, o.sy), ("sz", sz, o.h)):
            if abs(got - want) > 2e-3:
                fails.append(f"{o.name}: {label} {got:.3f} != {want:.3f}")

    # 2. no collision box escapes its declared zone
    zmap = {z.name: z for z in L.zones}
    for o in fill:
        z = zmap[o.zone]
        checks += 1
        if (o.cx - o.sx / 2 < z.xmin - TOL or o.cx + o.sx / 2 > z.xmax + TOL
                or o.cy - o.sy / 2 < z.ymin - TOL or o.cy + o.sy / 2 > z.ymax + TOL):
            fails.append(f"{o.name} escapes zone {o.zone}")

    # 3. mesh visuals: one per tier, sitting on the box, never taller than it
    vis = {}
    for m in w.findall("model"):
        if m.get("name") != "warehouse_v2_mesh_visuals":
            continue
        for link in m.findall("link"):
            p = [float(t) for t in link.find("pose").text.split()]
            sc = [float(t) for t in link.find("visual/geometry/mesh/scale").text.split()]
            uri = link.find("visual/geometry/mesh/uri").text
            vis[link.get("name")] = (p, sc, uri)
    for name, (p, sc, uri) in vis.items():
        checks += 1
        if max(sc[0], sc[1]) > 1.001 or min(sc) <= 0:
            fails.append(f"{name}: horizontal mesh scale {sc[0]:.3f}x{sc[1]:.3f} "
                         f"is not 1:1 -- the mesh is being stretched")
        if sc[2] > 1.001:
            fails.append(f"{name}: vertical mesh scale {sc[2]:.3f} > 1 -- stretched")
    # every fill object must be dressed
    for o in fill:
        if o.zone == "DOCK_OFFICE":
            continue
        checks += 1
        if not any(k.startswith(f"m_{o.name}") for k in vis):
            fails.append(f"{o.name} has a collision box but no mesh visual")

    # 4. cameras: pose in the SDF equals the layout, and inside the building
    inc = {i.find("name").text: [float(t) for t in i.find("pose").text.split()]
           for i in w.findall("include")}
    inc_uri = {i.find("name").text: i.find("uri").text.strip()
               for i in w.findall("include")}
    checks += 1
    expected_camera_includes = set(CAMERA_MODELS.values())
    actual_camera_includes = {
        name for name, uri in inc_uri.items()
        if uri.removeprefix("model://") in expected_camera_includes
    }
    if actual_camera_includes != expected_camera_includes:
        fails.append(
            "localisation camera registry mismatch: "
            f"got {sorted(actual_camera_includes)}, expected {sorted(expected_camera_includes)}"
        )
    for c in L.cameras:
        checks += 1
        nm = CAMERA_MODELS[c.name]
        if nm not in inc:
            fails.append(f"camera {c.name} ({nm}) not included"); continue
        if inc_uri[nm] != f"model://{nm}":
            fails.append(f"camera {c.name}: URI {inc_uri[nm]!r} != 'model://{nm}'")
        x, y, z, roll, pitch, yaw = inc[nm]
        for label, got, want in (("x", x, c.x), ("y", y, c.y), ("z", z, c.z),
                                 ("pitch", pitch, math.radians(c.pitch_deg)),
                                 ("yaw", yaw, math.radians(c.yaw_deg))):
            if abs(got - want) > 1e-3:
                fails.append(f"camera {c.name}: {label} {got:.4f} != {want:.4f}")
        if not (-12 < x < 12 and -10 < y < 10):
            fails.append(f"camera {c.name} is outside the hall")
        if z > WALL_H - 0.3:
            fails.append(f"camera {c.name} at {z:.2f} m is within 0.3 m of the {WALL_H} m wall top")

    # 5. the south wall really is closed except at the doors
    shell = boxes(w, "warehouse_shell")
    seg = []
    for name, ((px, py, pz, *_), (sx, sy, sz)) in shell.items():
        if name.startswith("wall_south") or name.startswith("door_head"):
            seg.append((px - sx / 2, px + sx / 2, pz - sz / 2, pz + sz / 2, name))
    checks += 1
    for x in [v / 10 for v in range(-119, 120)]:
        covered = []
        for a, b, z0, z1, nm in seg:
            if a - 1e-6 <= x <= b + 1e-6:
                covered.append((z0, z1))
        top = max((z1 for z0, z1 in covered), default=0.0)
        low = min((z0 for z0, z1 in covered), default=WALL_H)
        if top < WALL_H - 1e-3:
            fails.append(f"south wall: gap above z={top:.2f} at x={x:.1f}")
            break
        if low > 1e-3 and not any(abs(x - d) < 1.61 for d in (-8.5, -1.5, 5.5)):
            fails.append(f"south wall: hole below z={low:.2f} at x={x:.1f} (not a door)")
            break

    # 6. lamps hang above the cameras and below the roof
    for nm, p in inc.items():
        if not nm.startswith("lamp_"):
            continue
        checks += 1
        bottom, top = p[2] + 12.50, p[2] + 14.02      # Lamp_01's own DAE extent
        if bottom < 6.0:
            fails.append(f"{nm} hangs at {bottom:.2f} m, low enough to be in a sight-line")
        if top > WALL_H:
            fails.append(f"{nm} top at {top:.2f} m pokes through the {WALL_H} m roof")

    print(f"[state {state}: {spec['file']}] {checks} checks")
    if fails:
        print(f"FAIL ({len(fails)}):")
        for f in fails[:30]:
            print("   ", f)
        if len(fails) > 30:
            print(f"    ... and {len(fails)-30} more")
        return 1
    print(f"[state {state}] PASS: the world matches the layout")
    return 0


def paired_invariants():
    """The stock-state pair may differ only in its declared fill contents.

    The building shell, map paint, permanent rack steel and all five camera
    includes are frozen byte-for-byte at the XML element level.
    """

    worlds = {}
    for state, spec in WORLDS.items():
        worlds[state] = ET.parse(WORLD_DIR / spec["file"]).getroot().find("world")

    fails, checks = [], 0
    for model_name in (
        "warehouse_shell",
        "warehouse_v2_rack_frames",
        "known_driveable_boundary",
    ):
        checks += 1
        elems = [worlds[state].find(f"model[@name='{model_name}']") for state in ("A", "B")]
        if any(elem is None for elem in elems):
            fails.append(f"paired invariant model {model_name!r} missing")
        elif ET.tostring(elems[0]) != ET.tostring(elems[1]):
            fails.append(f"paired invariant model {model_name!r} differs between states")

    for camera_model in CAMERA_MODELS.values():
        checks += 1
        elems = []
        for state in ("A", "B"):
            found = None
            for include in worlds[state].findall("include"):
                if include.findtext("name") == camera_model:
                    found = include
                    break
            elems.append(found)
        if any(elem is None for elem in elems):
            fails.append(f"paired camera include {camera_model!r} missing")
        elif ET.tostring(elems[0]) != ET.tostring(elems[1]):
            fails.append(f"paired camera include {camera_model!r} differs between states")

    print(f"[paired peak/shipout invariants] {checks} checks")
    if fails:
        print(f"FAIL ({len(fails)}):")
        for failure in fails:
            print("   ", failure)
        return 1
    print("[paired peak/shipout invariants] PASS")
    return 0


if __name__ == "__main__":
    states = sys.argv[1:] or ["A", "B"]
    results = [main(st) for st in states]
    if set(states) == set(WORLDS):
        results.append(paired_invariants())
    sys.exit(max(results))
