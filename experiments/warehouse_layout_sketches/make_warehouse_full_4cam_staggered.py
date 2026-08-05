#!/usr/bin/env python3
"""Build a visibly asymmetric layout from the canonical large four-camera world.

This is a structural derivative, not a replacement.  It reuses every model and
mesh from ``warehouse_full_4cam.world.sdf`` while applying explicit translations
to existing rack rows and staging objects, then moves/re-aims the four existing
wall cameras.  Visuals, collisions, contact sensors, and no-go outlines belonging
to each moved object are translated together.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
OUTPUT = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam_staggered.world.sdf"
OUT_DIR = (
    REPO
    / "logs/studies/warehouse_layout_sketches/warehouse_full_4cam_staggered_v1"
)

# Longitudinal offsets break the aligned three-band rack grid while preserving
# every aisle's original lateral width.  The wall-backed W0 shelf remains fixed.
RACK_ROW_DY = {
    "W1": -0.55,
    "W2": 0.30,
    "W3": 0.70,
    "W4": -0.25,
    "E1": 0.85,
    "E2": -0.45,
    "E3": 0.40,
    "E4": -0.75,
}

# Existing collision-backed staging objects; their outlines move with them.
OBJECT_TRANSLATIONS = {
    "island_stack": (-0.45, -2.10),
    "dropped_stack_WA2": (1.20, 0.55),
}

# Existing included AWS props.  No new model types are introduced.
PROP_POSES = {
    "bucket_dock": "10.70 -8.25 0 0 0 0.35",
    "trashcan_nw": "-10.55 8.55 0 0 0 0.20",
    "clutterA_sw": "-10.55 -8.35 0 0 0 0.30",
    "clutterC_ne": "10.15 7.75 0 0 0 2.65",
    "clutterD_dock": "1.25 -8.65 0 0 0 -0.25",
    "palletjack_apron": "-2.15 -8.40 0 0 0 0.75",
    "desk_qc_ne": "8.95 8.15 0 0 0 2.90",
}

# Greater than the mild derivative, but still wall-mounted and conservative.
CAMERA_POSES = {
    "external_camera": (-8.10, -10.00, 6.10, 0.0, 0.92, math.radians(76.0)),
    "external_camera_b": (-3.70, 10.00, 6.10, 0.0, 0.92, math.radians(-103.0)),
    "external_camera_c": (4.25, -10.00, 6.10, 0.0, 0.92, math.radians(105.0)),
    "external_camera_d": (8.20, 10.00, 6.10, 0.0, 0.92, math.radians(-78.0)),
}

PROP_SIZES = {
    "bucket_dock": (0.42, 0.42),
    "trashcan_nw": (0.55, 0.55),
    "clutterA_sw": (0.90, 0.75),
    "clutterC_ne": (1.00, 0.80),
    "clutterD_dock": (0.90, 0.80),
    "palletjack_apron": (1.35, 0.65),
    "desk_qc_ne": (1.55, 0.85),
}


def _parse(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def _pose_values(node: ET.Element) -> list[float]:
    return [float(v) for v in (node.text or "").split()]


def _set_pose(node: ET.Element, values: list[float]) -> None:
    node.text = " ".join(f"{v:.4f}" for v in values)


def _translate_link(link: ET.Element, dx: float, dy: float) -> None:
    pose = link.find("pose")
    if pose is None:
        raise RuntimeError(f"link {link.get('name')} has no pose")
    values = _pose_values(pose)
    values[0] += dx
    values[1] += dy
    _set_pose(pose, values)


def _include_map(root: ET.Element) -> dict[str, ET.Element]:
    return {
        include.findtext("name", default=""): include
        for include in root.findall(".//include")
    }


def _all_links(root: ET.Element) -> list[ET.Element]:
    return list(root.findall(".//link"))


def _rack_row_for_link(name: str) -> str | None:
    match = re.search(
        r"(?:rack_|rail_|shelfbox_|mesh_|nogo_)([WE][1-4])_"
        r"(?:south|mid|north)",
        name,
    )
    return match.group(1) if match else None


def _translate_named_object(root: ET.Element, prefix: str, dx: float, dy: float) -> int:
    moved = 0
    for link in _all_links(root):
        name = link.get("name", "")
        if name.startswith(prefix) or name.startswith(f"nogo_{prefix}"):
            _translate_link(link, dx, dy)
            moved += 1
    if moved == 0:
        raise RuntimeError(f"no links matched object prefix {prefix}")
    return moved


def _move_prop_outline(root: ET.Element, name: str, old_xy: tuple[float, float], new_xy: tuple[float, float]) -> int:
    return _translate_named_object(
        root,
        f"nogo_{name}",
        new_xy[0] - old_xy[0],
        new_xy[1] - old_xy[1],
    )


def _canonical_signature(root: ET.Element) -> dict[str, object]:
    return {
        "uris": [node.text for node in root.findall(".//uri")],
        "links": [node.get("name") for node in root.findall(".//link")],
        "collisions": len(root.findall(".//collision")),
        "contacts": len(root.findall(".//sensor[@type='contact']")),
        "includes": [
            (node.findtext("name"), node.findtext("uri"))
            for node in root.findall(".//include")
        ],
    }


def build() -> dict[str, object]:
    source_tree = _parse(SOURCE)
    source_root = source_tree.getroot()
    source_world = source_root.find("world")
    if source_world is None or source_world.get("name") != "warehouse_full_4cam":
        raise RuntimeError("canonical large-world contract changed")

    output_root = copy.deepcopy(source_root)
    output_world = output_root.find("world")
    assert output_world is not None
    output_world.set("name", "warehouse_full_4cam_staggered")
    output_root.insert(
        0,
        ET.Comment(
            " DERIVATIVE: staggered rack layout and relocated existing staging/cameras; "
            "all model and mesh URIs originate in warehouse_full_4cam. "
        ),
    )

    moved_rack_links: dict[str, int] = {row: 0 for row in RACK_ROW_DY}
    for link in _all_links(output_root):
        row = _rack_row_for_link(link.get("name", ""))
        if row is None:
            continue
        _translate_link(link, 0.0, RACK_ROW_DY[row])
        moved_rack_links[row] += 1
    if any(count != 27 for count in moved_rack_links.values()):
        raise RuntimeError(f"unexpected rack-row link groups: {moved_rack_links}")

    moved_object_links = {}
    for name, (dx, dy) in OBJECT_TRANSLATIONS.items():
        moved_object_links[name] = _translate_named_object(output_root, name, dx, dy)

    source_includes = _include_map(source_root)
    output_includes = _include_map(output_root)
    moved_prop_outlines = {}
    for name, pose_text in PROP_POSES.items():
        source_include = source_includes.get(name)
        output_include = output_includes.get(name)
        if source_include is None or output_include is None:
            raise RuntimeError(f"missing canonical prop include {name}")
        source_pose = [float(v) for v in source_include.findtext("pose", "").split()]
        new_pose = [float(v) for v in pose_text.split()]
        output_pose = output_include.find("pose")
        assert output_pose is not None
        _set_pose(output_pose, new_pose)
        moved_prop_outlines[name] = _move_prop_outline(
            output_root, name, (source_pose[0], source_pose[1]), (new_pose[0], new_pose[1])
        )

    for name, pose in CAMERA_POSES.items():
        include = output_includes.get(name)
        if include is None:
            raise RuntimeError(f"missing canonical camera include {name}")
        pose_node = include.find("pose")
        assert pose_node is not None
        _set_pose(pose_node, list(pose))

    source_signature = _canonical_signature(source_root)
    output_signature = _canonical_signature(output_root)
    if source_signature != output_signature:
        raise RuntimeError("model/mesh/collision/contact inventory changed")

    ET.indent(output_root, space="  ")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(
        OUTPUT, encoding="utf-8", xml_declaration=True, short_empty_elements=True
    )

    summary = {
        "source_world": str(SOURCE.relative_to(REPO)),
        "output_world": str(OUTPUT.relative_to(REPO)),
        "world_name": "warehouse_full_4cam_staggered",
        "rack_row_dy_m": RACK_ROW_DY,
        "moved_rack_links": moved_rack_links,
        "object_translations_m": OBJECT_TRANSLATIONS,
        "moved_object_links": moved_object_links,
        "prop_poses": PROP_POSES,
        "moved_prop_outline_links": moved_prop_outlines,
        "camera_poses": {
            name: " ".join(f"{v:.4f}" for v in pose)
            for name, pose in CAMERA_POSES.items()
        },
        "preserved_counts": {
            "links": len(source_signature["links"]),
            "collisions": source_signature["collisions"],
            "contacts": source_signature["contacts"],
            "includes": len(source_signature["includes"]),
            "uris": len(source_signature["uris"]),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _rectangles(root: ET.Element) -> list[tuple[str, float, float, float, float]]:
    rectangles = []
    for link in root.findall(".//link"):
        box = link.find("collision/geometry/box/size")
        pose = link.find("pose")
        if box is None or pose is None or not box.text:
            continue
        size = [float(v) for v in box.text.split()]
        values = _pose_values(pose)
        if size[0] > 5 or size[1] > 8:  # omit shell walls from the interior plot
            continue
        rectangles.append((link.get("name", ""), values[0], values[1], size[0], size[1]))
    return rectangles


def _camera_map(root: ET.Element) -> dict[str, tuple[float, ...]]:
    includes = _include_map(root)
    return {
        name: tuple(float(v) for v in includes[name].findtext("pose", "").split())
        for name in CAMERA_POSES
    }


def render_comparison() -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/warehouse_staggered_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    source_root = _parse(SOURCE).getroot()
    output_root = _parse(OUTPUT).getroot()
    colors = dict(zip(CAMERA_POSES, ("#2f80ed", "#27ae60", "#9b51e0", "#f2994a")))
    labels = dict(zip(CAMERA_POSES, "ABCD"))
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.3), constrained_layout=True)

    for ax, root, title in (
        (axes[0], source_root, "Canonical large world"),
        (axes[1], output_root, "Staggered derivative"),
    ):
        ax.add_patch(Rectangle((-12, -10), 24, 20, facecolor="#ddd8cf", edgecolor="#343a40", lw=3))
        for name, x, y, sx, sy in _rectangles(root):
            if name.startswith("rack_"):
                color = "#d8ae20"
            elif "stack" in name or "pillar" in name:
                color = "#9a6334"
            else:
                continue
            ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor=color, edgecolor="#5b4222", lw=0.7))
        for name, pose in _camera_map(root).items():
            x, y, _z, _r, _p, yaw = pose
            reach, half = 14.0, math.radians(45)
            wedge = [
                (x, y),
                (x + reach * math.cos(yaw - half), y + reach * math.sin(yaw - half)),
                (x + reach * math.cos(yaw + half), y + reach * math.sin(yaw + half)),
            ]
            ax.add_patch(Polygon(wedge, facecolor=colors[name], edgecolor=colors[name], alpha=0.13))
            ax.scatter([x], [y], s=95, color=colors[name], edgecolor="black", zorder=5)
            ax.arrow(x, y, math.cos(yaw), math.sin(yaw), width=0.035, head_width=0.28, color=colors[name], zorder=6)
            ax.text(x, y + (0.55 if y > 0 else -0.65), labels[name], ha="center", weight="bold")
        ax.set(xlim=(-12.7, 12.7), ylim=(-10.8, 10.8), xlabel="x [m]", ylabel="y [m]")
        ax.set_aspect("equal")
        ax.set_title(title, weight="bold")
        ax.grid(alpha=0.16)

    fig.suptitle(
        "warehouse_full_4cam_staggered — same assets, changed rack/staging/camera layout",
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
