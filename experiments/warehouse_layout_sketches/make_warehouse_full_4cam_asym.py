#!/usr/bin/env python3
"""Build a mildly asymmetric four-camera variant from the canonical large world.

The warehouse geometry, meshes, collisions, props, sensor models, and topics are
copied unchanged from ``warehouse_full_4cam.world.sdf``.  Only the world name and
the four camera include poses change.  This keeps the derivative auditable and
prevents the exploratory variant from overwriting the canonical evaluation world.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
OUTPUT = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam_asym.world.sdf"
OUT_DIR = (
    REPO
    / "logs/studies/warehouse_layout_sketches/warehouse_full_4cam_asym_v1"
)

# Keep the deployed height and downward pitch.  The only physical change is a
# modest slide along the existing north/south walls plus 7--9 degrees of yaw.
# The poses are deliberately not mirror pairs, but remain conservative enough to
# keep all four views on the operating floor.
CAMERA_POSES = {
    "external_camera": (-7.20, -10.00, 6.10, 0.0, 0.92, math.radians(82.0)),
    "external_camera_b": (-4.80, 10.00, 6.10, 0.0, 0.92, math.radians(-98.0)),
    "external_camera_c": (5.40, -10.00, 6.10, 0.0, 0.92, math.radians(99.0)),
    "external_camera_d": (7.00, 10.00, 6.10, 0.0, 0.92, math.radians(-83.0)),
}


def _pose_text(pose: tuple[float, ...]) -> str:
    x, y, z, roll, pitch, yaw = pose
    return f"{x:.2f} {y:.2f} {z:.2f} {roll:.0f} {pitch:.2f} {yaw:.4f}"


def _includes(root: ET.Element) -> dict[str, ET.Element]:
    return {
        include.findtext("name", default=""): include
        for include in root.findall(".//include")
    }


def build() -> dict[str, object]:
    source_text = SOURCE.read_text(encoding="utf-8")
    source_root = ET.fromstring(source_text)
    source_world = source_root.find("world")
    if source_world is None or source_world.get("name") != "warehouse_full_4cam":
        raise RuntimeError("canonical source world contract changed")

    source_includes = _includes(source_root)
    old_poses: dict[str, str] = {}
    output_text = source_text
    for name, new_pose in CAMERA_POSES.items():
        include = source_includes.get(name)
        if include is None:
            raise RuntimeError(f"canonical source is missing camera include {name}")
        pose_node = include.find("pose")
        if pose_node is None or not pose_node.text:
            raise RuntimeError(f"canonical camera {name} has no pose")
        old_pose = pose_node.text.strip()
        old_poses[name] = old_pose
        old_fragment = f"<name>{name}</name><uri>model://{name}</uri><pose>{old_pose}</pose>"
        new_fragment = (
            f"<name>{name}</name><uri>model://{name}</uri>"
            f"<pose>{_pose_text(new_pose)}</pose>"
        )
        if output_text.count(old_fragment) != 1:
            raise RuntimeError(f"expected one exact canonical include for {name}")
        output_text = output_text.replace(old_fragment, new_fragment, 1)

    old_world_tag = '<world name="warehouse_full_4cam">'
    if output_text.count(old_world_tag) != 1:
        raise RuntimeError("expected one canonical world tag")
    output_text = output_text.replace(
        old_world_tag,
        '<world name="warehouse_full_4cam_asym">',
        1,
    )
    provenance = (
        "  <!-- DERIVATIVE: warehouse_full_4cam_asym. Geometry, meshes, collisions, "
        "props, and sensor models are unchanged from warehouse_full_4cam; only the "
        "world name and four camera include poses differ. -->\n"
    )
    output_text = output_text.replace(
        '<sdf version="1.8">\n',
        '<sdf version="1.8">\n' + provenance,
        1,
    )

    OUTPUT.write_text(output_text, encoding="utf-8")
    output_root = ET.fromstring(output_text)
    output_world = output_root.find("world")
    if output_world is None or output_world.get("name") != "warehouse_full_4cam_asym":
        raise RuntimeError("generated derivative has the wrong world name")

    output_includes = _includes(output_root)
    for name, expected in CAMERA_POSES.items():
        actual = output_includes[name].findtext("pose", default="").strip()
        if actual != _pose_text(expected):
            raise RuntimeError(f"generated camera pose mismatch for {name}")
        source_uri = source_includes[name].findtext("uri")
        output_uri = output_includes[name].findtext("uri")
        if source_uri != output_uri:
            raise RuntimeError(f"camera model URI changed for {name}")

    source_models = [m.get("name") for m in source_root.findall(".//model")]
    output_models = [m.get("name") for m in output_root.findall(".//model")]
    if source_models != output_models:
        raise RuntimeError("embedded model inventory changed")
    source_uris = [n.text for n in source_root.findall(".//uri")]
    output_uris = [n.text for n in output_root.findall(".//uri")]
    if source_uris != output_uris:
        raise RuntimeError("mesh/model URI inventory changed")
    if len(source_root.findall(".//collision")) != len(output_root.findall(".//collision")):
        raise RuntimeError("collision inventory changed")

    summary = {
        "source_world": str(SOURCE.relative_to(REPO)),
        "output_world": str(OUTPUT.relative_to(REPO)),
        "world_name": "warehouse_full_4cam_asym",
        "unchanged_contract": [
            "embedded models",
            "model and mesh URIs",
            "collision inventory",
            "camera model URIs and topics",
            "warehouse geometry and props",
        ],
        "camera_pose_changes": {
            name: {"old": old_poses[name], "new": _pose_text(pose)}
            for name, pose in CAMERA_POSES.items()
        },
        "counts": {
            "embedded_models": len(source_models),
            "includes": len(source_root.findall(".//include")),
            "mesh_uris": len(source_root.findall(".//mesh/uri")),
            "collisions": len(source_root.findall(".//collision")),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _camera_poses(root: ET.Element) -> dict[str, tuple[float, ...]]:
    poses = {}
    for name, include in _includes(root).items():
        if name not in CAMERA_POSES:
            continue
        poses[name] = tuple(float(v) for v in include.findtext("pose", "").split())
    return poses


def _collision_rectangles(root: ET.Element) -> list[tuple[float, float, float, float]]:
    rectangles = []
    for link in root.findall(".//model[@name='warehouse_rack_occluders']/link"):
        box = link.find("collision/geometry/box/size")
        if box is None or not box.text:
            continue
        pose = tuple(float(v) for v in link.findtext("pose", "0 0 0 0 0 0").split())
        size = tuple(float(v) for v in box.text.split())
        rectangles.append((pose[0], pose[1], size[0], size[1]))
    return rectangles


def render_comparison() -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/warehouse_asym_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    source_root = ET.parse(SOURCE).getroot()
    output_root = ET.parse(OUTPUT).getroot()
    obstacles = _collision_rectangles(source_root)
    labels = {
        "external_camera": "A",
        "external_camera_b": "B",
        "external_camera_c": "C",
        "external_camera_d": "D",
    }
    colors = {
        "external_camera": "#2f80ed",
        "external_camera_b": "#27ae60",
        "external_camera_c": "#9b51e0",
        "external_camera_d": "#f2994a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), constrained_layout=True)
    for ax, root, title in (
        (axes[0], source_root, "Canonical: symmetric camera grid"),
        (axes[1], output_root, "Derivative: mild wall-camera asymmetry"),
    ):
        ax.add_patch(Rectangle((-12, -10), 24, 20, facecolor="#ded9d0", edgecolor="#343a40", lw=3))
        for x, y, sx, sy in obstacles:
            ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor="#e0b62f", edgecolor="#7b6420", lw=0.6))
        for name, pose in _camera_poses(root).items():
            x, y, _z, _roll, _pitch, yaw = pose
            reach = 13.5
            half_fov = math.radians(45)
            wedge = [
                (x, y),
                (x + reach * math.cos(yaw - half_fov), y + reach * math.sin(yaw - half_fov)),
                (x + reach * math.cos(yaw + half_fov), y + reach * math.sin(yaw + half_fov)),
            ]
            ax.add_patch(Polygon(wedge, closed=True, facecolor=colors[name], edgecolor=colors[name], alpha=0.13))
            ax.scatter([x], [y], s=95, color=colors[name], edgecolor="black", zorder=5)
            ax.arrow(x, y, 1.1 * math.cos(yaw), 1.1 * math.sin(yaw), width=0.035, head_width=0.28, color=colors[name], zorder=6)
            ax.text(x + (0.35 if x < 0 else -0.35), y + (-0.55 if y < 0 else 0.55), labels[name], weight="bold", ha="left" if x < 0 else "right")
        ax.set_xlim(-12.7, 12.7)
        ax.set_ylim(-10.8, 10.8)
        ax.set_aspect("equal")
        ax.set_title(title, weight="bold")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.16)

    fig.suptitle(
        "warehouse_full_4cam_asym — same warehouse, cameras only",
        fontsize=15,
        weight="bold",
    )
    path = OUT_DIR / "fig_camera_placement_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    summary = build()
    figure = render_comparison()
    print(f"wrote {OUTPUT}")
    print(f"wrote {figure}")
    print(json.dumps(summary["camera_pose_changes"], indent=2))


if __name__ == "__main__":
    main()
