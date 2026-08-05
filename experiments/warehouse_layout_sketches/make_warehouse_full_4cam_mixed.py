#!/usr/bin/env python3
"""Create an asymmetric mixed-use warehouse from the canonical large world.

The derivative keeps the original shell, AWS asset library, primary rack aisles,
sensor topics, and robot interface.  Selected rack segments are deliberately
changed into transverse shelves, pallet-box bays, and a compact machine cell.
All changed visuals have matching collision/contact geometry and floor outlines.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import make_warehouse_full_4cam_staggered as common


REPO = common.REPO
SOURCE = common.SOURCE
OUTPUT = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam_mixed.world.sdf"
OUT_DIR = REPO / "logs/studies/warehouse_layout_sketches/warehouse_full_4cam_mixed_v1"

# Move two existing rack segments into the north/south apron and rotate them.
TRANSVERSE_SHELVES = {
    "W2_mid": {"old": (-6.725, 0.200), "new": (-6.45, -7.65), "yaw": math.pi / 2},
    "E4_mid": {"old": (8.825, 0.200), "new": (7.75, 7.70), "yaw": math.pi / 2},
}

# These three longitudinal segments are removed from the regular rack grid.
BOX_BAYS = {
    "W4_north": {
        "center": (-2.525, 4.950),
        "stacks": [
            (-2.525, 3.75, 1.05, 0.85, 1.15),
            (-2.525, 4.95, 0.90, 0.90, 1.85),
            (-2.525, 6.10, 1.10, 0.80, 1.35),
        ],
    },
    "E2_south": {
        "center": (4.625, -4.550),
        "stacks": [
            (4.625, -5.70, 0.90, 0.90, 1.55),
            (4.625, -4.50, 1.05, 0.82, 1.10),
            (4.625, -3.35, 0.92, 0.88, 2.00),
        ],
    },
}

MACHINE_REPLACES = "E1_mid"
MACHINE_CENTER = (2.525, 0.200)

# Existing AWS props are redistributed around the retained rack aisles.
PROP_POSES = {
    "bucket_dock": "10.70 -8.35 0 0 0 0.35",
    "trashcan_nw": "-10.70 8.55 0 0 0 0.20",
    "clutterA_sw": "-10.60 -8.35 0 0 0 0.30",
    "clutterC_ne": "10.20 7.65 0 0 0 2.65",
    "clutterD_dock": "0.70 -8.55 0 0 0 -0.25",
    "palletjack_apron": "-2.00 -8.40 0 0 0 0.75",
    "desk_qc_ne": "9.10 8.40 0 0 0 2.90",
}

# Different wall offsets, heights, pitch, and yaw emulate a retrofitted network.
CAMERA_POSES = {
    "external_camera": (-8.40, -10.00, 6.10, 0.0, 0.92, math.radians(78.0)),
    "external_camera_b": (-3.60, 10.00, 5.70, 0.0, 0.98, math.radians(-101.0)),
    "external_camera_c": (3.90, -10.00, 6.40, 0.0, 0.88, math.radians(104.0)),
    "external_camera_d": (8.60, 10.00, 5.90, 0.0, 0.95, math.radians(-79.0)),
}

RACK_COLOR = "0.93 0.80 0.08 1"
BOX_COLOR = "0.67 0.46 0.20 1"
MACHINE_COLOR = "0.20 0.42 0.58 1"
MACHINE_DARK = "0.10 0.20 0.28 1"
GREEN = "0.05 0.55 0.20 1"


def _segment_match(name: str, segment: str) -> bool:
    return bool(
        re.search(
            rf"(?:rack_|rail_|shelfbox_|mesh_|nogo_){re.escape(segment)}(?:_|$)",
            name,
        )
    )


def _segment_links(root: ET.Element, segment: str) -> list[ET.Element]:
    return [
        link
        for link in root.findall(".//link")
        if _segment_match(link.get("name", ""), segment)
    ]


def _transform_segment(
    root: ET.Element,
    segment: str,
    old_center: tuple[float, float],
    new_center: tuple[float, float],
    yaw: float,
) -> int:
    links = _segment_links(root, segment)
    if len(links) != 9:
        raise RuntimeError(f"expected 9 linked parts for {segment}, found {len(links)}")
    ox, oy = old_center
    nx, ny = new_center
    c, s = math.cos(yaw), math.sin(yaw)
    for link in links:
        pose = link.find("pose")
        if pose is None:
            raise RuntimeError(f"{link.get('name')} has no pose")
        values = common._pose_values(pose)
        rx, ry = values[0] - ox, values[1] - oy
        values[0] = nx + c * rx - s * ry
        values[1] = ny + s * rx + c * ry
        values[5] += yaw
        common._set_pose(pose, values)
    return len(links)


def _remove_segment(root: ET.Element, segment: str) -> int:
    removed = 0
    for model in root.findall(".//model"):
        for link in list(model.findall("link")):
            if _segment_match(link.get("name", ""), segment):
                model.remove(link)
                removed += 1
    if removed != 9:
        raise RuntimeError(f"expected to remove 9 linked parts for {segment}, removed {removed}")
    return removed


def _material(parent: ET.Element, color: str) -> None:
    material = ET.SubElement(parent, "material")
    ET.SubElement(material, "ambient").text = color
    ET.SubElement(material, "diffuse").text = color


def _box_link(
    name: str,
    x: float,
    y: float,
    sx: float,
    sy: float,
    sz: float,
    color: str,
    *,
    z: float | None = None,
    collide: bool = True,
    contact: bool = True,
) -> ET.Element:
    link = ET.Element("link", {"name": name})
    ET.SubElement(link, "pose").text = f"{x:.4f} {y:.4f} {(sz / 2 if z is None else z):.4f} 0 0 0"
    if collide:
        collision = ET.SubElement(link, "collision", {"name": "collision"})
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        ET.SubElement(box, "size").text = f"{sx:.4f} {sy:.4f} {sz:.4f}"
        if contact:
            sensor = ET.SubElement(link, "sensor", {"name": "contact", "type": "contact"})
            contact_el = ET.SubElement(sensor, "contact")
            ET.SubElement(contact_el, "collision").text = "collision"
            ET.SubElement(sensor, "update_rate").text = "60"
    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = f"{sx:.4f} {sy:.4f} {sz:.4f}"
    _material(visual, color)
    return link


def _outline_links(name: str, x: float, y: float, sx: float, sy: float) -> list[ET.Element]:
    pad, thickness = 0.12, 0.055
    ox, oy = sx + 2 * pad, sy + 2 * pad
    return [
        _box_link(f"{name}_outline_s", x, y - oy / 2, ox, thickness, 0.014, GREEN, z=0.035, collide=False),
        _box_link(f"{name}_outline_n", x, y + oy / 2, ox, thickness, 0.014, GREEN, z=0.035, collide=False),
        _box_link(f"{name}_outline_w", x - ox / 2, y, thickness, oy, 0.014, GREEN, z=0.035, collide=False),
        _box_link(f"{name}_outline_e", x + ox / 2, y, thickness, oy, 0.014, GREEN, z=0.035, collide=False),
    ]


def _add_box_bay(
    rack_model: ET.Element,
    marks_model: ET.Element,
    bay_name: str,
    spec: dict[str, object],
) -> None:
    stacks = spec["stacks"]
    assert isinstance(stacks, list)
    for index, (x, y, sx, sy, height) in enumerate(stacks):
        # Two visible tiers per pallet stack make the replacement clearly read as boxes.
        tier_h = height / 2
        rack_model.append(
            _box_link(
                f"{bay_name}_box_{index}_lower",
                x,
                y,
                sx,
                sy,
                tier_h,
                BOX_COLOR,
                z=tier_h / 2,
            )
        )
        rack_model.append(
            _box_link(
                f"{bay_name}_box_{index}_upper",
                x + 0.04,
                y - 0.03,
                sx * 0.91,
                sy * 0.91,
                tier_h,
                "0.76 0.58 0.34 1",
                z=tier_h + tier_h / 2,
            )
        )
    cx, cy = spec["center"]
    marks_model.extend(_outline_links(f"nogo_{bay_name}", cx, cy, 1.15, 3.65))


def _add_machine_cell(rack_model: ET.Element, marks_model: ET.Element) -> None:
    x, y = MACHINE_CENTER
    rack_model.append(_box_link("packing_machine_body", x, y, 1.35, 1.80, 1.35, MACHINE_COLOR))
    rack_model.append(
        _box_link(
            "packing_machine_head",
            x,
            y + 0.15,
            0.85,
            0.70,
            0.50,
            MACHINE_DARK,
            z=1.60,
            collide=False,
        )
    )
    rack_model.append(
        _box_link(
            "packing_machine_conveyor",
            x,
            y - 1.35,
            1.00,
            0.85,
            0.75,
            "0.42 0.50 0.56 1",
        )
    )
    marks_model.extend(_outline_links("nogo_packing_machine", x, y - 0.35, 1.55, 2.85))


def _move_prop_outlines(
    source_root: ET.Element,
    output_root: ET.Element,
    name: str,
    new_pose: list[float],
) -> None:
    source_include = common._include_map(source_root)[name]
    old_pose = [float(v) for v in source_include.findtext("pose", "").split()]
    common._translate_named_object(
        output_root,
        f"nogo_{name}",
        new_pose[0] - old_pose[0],
        new_pose[1] - old_pose[1],
    )


def build() -> dict[str, object]:
    source_root = common._parse(SOURCE).getroot()
    output_root = copy.deepcopy(source_root)
    world = output_root.find("world")
    if world is None or world.get("name") != "warehouse_full_4cam":
        raise RuntimeError("canonical source contract changed")
    world.set("name", "warehouse_full_4cam_mixed")
    output_root.insert(
        0,
        ET.Comment(
            " DERIVATIVE: retained aisles with transverse shelves, pallet-box bays, "
            "packing machine, redistributed AWS props, and asymmetric cameras. "
        ),
    )

    rack_model = output_root.find(".//model[@name='warehouse_rack_occluders']")
    marks_model = output_root.find(".//model[@name='known_driveable_boundary']")
    if rack_model is None or marks_model is None:
        raise RuntimeError("canonical rack or floor-marking model missing")

    transformed = {}
    for segment, spec in TRANSVERSE_SHELVES.items():
        transformed[segment] = _transform_segment(
            output_root, segment, spec["old"], spec["new"], spec["yaw"]
        )

    removed = {}
    for segment, spec in BOX_BAYS.items():
        removed[segment] = _remove_segment(output_root, segment)
        _add_box_bay(rack_model, marks_model, f"box_bay_{segment.lower()}", spec)
    removed[MACHINE_REPLACES] = _remove_segment(output_root, MACHINE_REPLACES)
    _add_machine_cell(rack_model, marks_model)

    source_includes = common._include_map(source_root)
    output_includes = common._include_map(output_root)
    for name, pose_text in PROP_POSES.items():
        values = [float(v) for v in pose_text.split()]
        _move_prop_outlines(source_root, output_root, name, values)
        pose = output_includes[name].find("pose")
        assert pose is not None
        common._set_pose(pose, values)
        if source_includes[name].findtext("uri") != output_includes[name].findtext("uri"):
            raise RuntimeError(f"prop URI changed for {name}")

    for name, values in CAMERA_POSES.items():
        pose = output_includes[name].find("pose")
        assert pose is not None
        common._set_pose(pose, list(values))

    # Includes and their URIs stay identical; only three shelf-mesh links are
    # intentionally removed and replaced by collision-backed primitives.
    source_includes_signature = common._canonical_signature(source_root)["includes"]
    output_includes_signature = common._canonical_signature(output_root)["includes"]
    if source_includes_signature != output_includes_signature:
        raise RuntimeError("include/model URI inventory changed")

    ET.indent(output_root, space="  ")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(
        OUTPUT, encoding="utf-8", xml_declaration=True, short_empty_elements=True
    )

    summary = {
        "source_world": str(SOURCE.relative_to(REPO)),
        "output_world": str(OUTPUT.relative_to(REPO)),
        "transverse_shelves": TRANSVERSE_SHELVES,
        "box_bays": BOX_BAYS,
        "machine_replaces": MACHINE_REPLACES,
        "machine_center": MACHINE_CENTER,
        "prop_poses": PROP_POSES,
        "camera_poses": CAMERA_POSES,
        "transformed_segment_parts": transformed,
        "removed_segment_parts": removed,
        "counts": {
            "links": len(output_root.findall(".//link")),
            "collisions": len(output_root.findall(".//collision")),
            "contacts": len(output_root.findall(".//sensor[@type='contact']")),
            "mesh_uris": len(output_root.findall(".//mesh/uri")),
            "includes": len(output_root.findall(".//include")),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _collision_rects(root: ET.Element) -> list[tuple[str, float, float, float, float, float]]:
    out = []
    for link in root.findall(".//link"):
        box = link.find("collision/geometry/box/size")
        pose = link.find("pose")
        if box is None or pose is None or not box.text:
            continue
        size = [float(v) for v in box.text.split()]
        values = common._pose_values(pose)
        if size[0] > 5 or size[1] > 8:
            continue
        out.append((link.get("name", ""), values[0], values[1], size[0], size[1], values[5]))
    return out


def render_comparison() -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/warehouse_mixed_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.transforms import Affine2D

    source_root = common._parse(SOURCE).getroot()
    output_root = common._parse(OUTPUT).getroot()
    colors = dict(zip(CAMERA_POSES, ("#2f80ed", "#27ae60", "#9b51e0", "#f2994a")))
    labels = dict(zip(CAMERA_POSES, "ABCD"))
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.3), constrained_layout=True)
    for ax, root, title in (
        (axes[0], source_root, "Canonical large world"),
        (axes[1], output_root, "Mixed-use asymmetric derivative"),
    ):
        ax.add_patch(Rectangle((-12, -10), 24, 20, facecolor="#ddd8cf", edgecolor="#343a40", lw=3))
        for name, x, y, sx, sy, yaw in _collision_rects(root):
            if name.startswith("rack_"):
                color = "#d8ae20"
            elif "box_bay" in name:
                color = "#a86832"
            elif "packing_machine" in name:
                color = "#326b8c"
            elif "stack" in name or "pillar" in name:
                color = "#8f613b"
            else:
                continue
            patch = Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor=color, edgecolor="#493b2b", lw=0.7)
            patch.set_transform(Affine2D().rotate_around(x, y, yaw) + ax.transData)
            ax.add_patch(patch)
        includes = common._include_map(root)
        for name in CAMERA_POSES:
            pose = [float(v) for v in includes[name].findtext("pose", "").split()]
            x, y, yaw = pose[0], pose[1], pose[5]
            reach, half = 14.0, math.radians(45)
            wedge = [(x, y), (x + reach * math.cos(yaw - half), y + reach * math.sin(yaw - half)), (x + reach * math.cos(yaw + half), y + reach * math.sin(yaw + half))]
            ax.add_patch(Polygon(wedge, facecolor=colors[name], edgecolor=colors[name], alpha=0.12))
            ax.scatter([x], [y], s=95, color=colors[name], edgecolor="black", zorder=5)
            ax.arrow(x, y, math.cos(yaw), math.sin(yaw), width=0.035, head_width=0.28, color=colors[name], zorder=6)
            ax.text(x, y + (0.55 if y > 0 else -0.65), labels[name], ha="center", weight="bold")
        ax.set(xlim=(-12.7, 12.7), ylim=(-10.8, 10.8), xlabel="x [m]", ylabel="y [m]")
        ax.set_aspect("equal")
        ax.set_title(title, weight="bold")
        ax.grid(alpha=0.16)
    fig.suptitle(
        "warehouse_full_4cam_mixed — retained aisles, transverse shelves, boxes, machine",
        fontsize=15,
        weight="bold",
    )
    path = OUT_DIR / "fig_layout_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    summary = build()
    figure = render_comparison()
    print(f"wrote {OUTPUT}")
    print(f"wrote {figure}")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
