#!/usr/bin/env python3
"""Geometry verification figure for the exploratory AWS warehouse.

Reads the rack/stack collision geometry directly from the world SDF (so it reflects
any edits to box_spot_R4 / r4_high_stack_*) and the driveable/staging regions from
world_profiles.yaml. This is a diagnostic for checking geometry and driveable
regions before interpreting any AWS run.

Convention:
  - Gray rectangles: rack bodies + low crates (non-traversable, physical obstacles).
  - Orange rectangles: the R4 high-stack links (tall occluder).
  - Green outlined rectangles: traversable driveable regions from world_profiles.
  - Red hatched rectangles: non-driveable staging pads from world_profiles.

The robot MUST stay inside the green-outlined regions and NEVER enter the red-hatched
pads. The green lines are the driveable-region boundaries; outside them is forbidden.

Usage:
  python3 scripts/visibility_comparison/plot_world_geometry.py \
    --world warehouse_aws.world.sdf \
    --out logs/visibility_comparison/aws_geometry_diagnostic/world_geometry.png

If you also pass --gp <npz>, the GP is drawn only as a diagnostic underlay. It is
not paper evidence unless the GP was captured and fitted for the same final AWS
geometry.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_DIR = REPO_ROOT / "src" / "sim" / "gazebo_worlds" / "worlds"
PROFILES = REPO_ROOT / "src" / "experiments" / "config" / "world_profiles.yaml"


def parse_world_collision_boxes(world_sdf: Path) -> list[dict]:
    """Extract every <link> with a <pose> + <collision><box> from the warehouse_rack_occluders model."""
    text = world_sdf.read_text()
    # The SDF is one-long-line-per-link in this repo's style — regex is robust enough.
    link_pat = re.compile(
        r'<link name="(?P<name>[^"]+)">'
        r'<pose>(?P<pose>[^<]+)</pose>'
        r'(?P<rest>.+?)</link>',
        re.DOTALL,
    )
    size_pat = re.compile(
        r'<collision[^>]*>.*?<box><size>(?P<size>[^<]+)</size></box>',
        re.DOTALL,
    )
    visual_size_pat = re.compile(
        r'<visual[^>]*>.*?<box><size>(?P<size>[^<]+)</size></box>',
        re.DOTALL,
    )
    out = []
    for m in link_pat.finditer(text):
        name = m.group("name")
        pose_parts = m.group("pose").strip().split()
        if len(pose_parts) < 3:
            continue
        cx, cy = float(pose_parts[0]), float(pose_parts[1])
        rest = m.group("rest")
        sz_match = size_pat.search(rest)
        is_collision = sz_match is not None
        if not is_collision:
            sz_match = visual_size_pat.search(rest)
            if sz_match is None:
                continue
        size_parts = sz_match.group("size").strip().split()
        if len(size_parts) < 2:
            continue
        sx, sy = float(size_parts[0]), float(size_parts[1])
        out.append({
            "name": name,
            "cx": cx, "cy": cy,
            "xmin": cx - sx / 2.0, "xmax": cx + sx / 2.0,
            "ymin": cy - sy / 2.0, "ymax": cy + sy / 2.0,
            "is_collision": is_collision,
        })
    return out


def load_profile(world_name: str) -> dict:
    data = yaml.safe_load(PROFILES.read_text())
    return dict((data.get("worlds") or {}).get(world_name) or {})


def draw_regions(ax, profile):
    drivable_legend_done = False
    staging_legend_done = False
    for r in profile.get("known_2d_regions", []) or []:
        rtype = str(r.get("type", "")).lower()
        x0, x1 = float(r["xmin"]), float(r["xmax"])
        y0, y1 = float(r["ymin"]), float(r["ymax"])
        if "traversable" in rtype:
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="none", edgecolor="#0d8a3d",
                linewidth=1.8, linestyle="-", zorder=5,
            ))
            drivable_legend_done = True
        else:  # non_driveable_staging or unspecified
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#ffd7d7", edgecolor="#c93030",
                linewidth=1.2, hatch="///", alpha=0.6, zorder=6,
            ))
            staging_legend_done = True
    return drivable_legend_done, staging_legend_done


def draw_collision_boxes(ax, links):
    # Skip the dock_* / wall_side_* / box_spot_* visual-only links — they overlap with
    # rectangle regions we draw separately. We want the rack bodies, crates, and stack.
    for L in links:
        name = L["name"]
        if any(name.startswith(p) for p in ("dock_", "wall_", "shelf_box_R", "rail_",
                                            "box_spot_", "ground_plane")):
            continue
        if "high_stack" in name:
            face = "#cc6633"; edge = "#5e2c10"; alpha = 0.92; lw = 0.7
        elif name.startswith("rack_R"):
            face = "#888888"; edge = "#222222"; alpha = 0.85; lw = 0.5
        elif name.startswith("low_crate"):
            face = "#b9b9b9"; edge = "#444444"; alpha = 0.85; lw = 0.5
        else:
            continue
        ax.add_patch(Rectangle(
            (L["xmin"], L["ymin"]), L["xmax"] - L["xmin"], L["ymax"] - L["ymin"],
            facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha, zorder=10,
        ))


def annotate_aisles(ax):
    ax.text(-3.025, -3.35, "A1", ha="center", fontsize=11, color="#0d8a3d",
            fontweight="bold", zorder=11)
    ax.text(-0.975, -3.35, "A2", ha="center", fontsize=11, color="#0d8a3d",
            fontweight="bold", zorder=11)
    ax.text(1.075, -3.35, "A3", ha="center", fontsize=11, color="#0d8a3d",
            fontweight="bold", zorder=11)
    ax.text(3.125, -3.35, "A4", ha="center", fontsize=11, color="#0d8a3d",
            fontweight="bold", zorder=11)
    ax.text(-5.4, 0.0, "west\nservice", ha="left", fontsize=9, color="#0d8a3d",
            fontweight="bold", zorder=11)
    ax.text(5.0, 0.0, "east\nfar-corner", ha="right", fontsize=9, color="#0d8a3d",
            fontweight="bold", zorder=11)
    ax.text(0.0, 1.65, "mid cross-aisle", ha="center", fontsize=9, color="#0d8a3d",
            fontweight="bold", zorder=11)
    ax.text(0.0, 4.55, "upper cross-aisle", ha="center", fontsize=9, color="#0d8a3d",
            fontweight="bold", zorder=11)


def draw_camera(ax, profile):
    # Camera is encoded in the GP artifact, not the world profile directly; the
    # spawn block is the robot start. The AWS camera sits at (0, -4.9, 5.8).
    ax.scatter([0.0], [-4.9], marker="v", s=110, c="#111111", edgecolor="white",
               linewidth=0.8, zorder=20)
    ax.text(0.0, -4.78, "camera", ha="center", fontsize=8, color="#111111", zorder=20)


def draw_legend(ax):
    handles = [
        Patch(facecolor="#888888", edgecolor="#222222", label="rack body"),
        Patch(facecolor="#cc6633", edgecolor="#5e2c10", label="R4 high stack (occluder)"),
        Patch(facecolor="#b9b9b9", edgecolor="#444444", label="low crate"),
        Line2D([0], [0], color="#0d8a3d", lw=1.8,
               label="green driveable boundary (robot MAY drive INSIDE)"),
        Patch(facecolor="#ffd7d7", edgecolor="#c93030", hatch="///",
              label="non-driveable staging (robot MUST NOT enter)"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#111111",
               markeredgecolor="white", markersize=10, label="camera"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.92, ncol=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="warehouse_aws.world.sdf")
    ap.add_argument("--out", default=str(REPO_ROOT / "logs" / "visibility_comparison"
                                           / "aws_geometry_diagnostic" / "world_geometry.png"))
    ap.add_argument("--gp", default=None,
                    help="Optional GP .npz to underlay for diagnostic comparison.")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    world_path = WORLD_DIR / args.world
    if not world_path.is_file():
        raise SystemExit(f"world not found: {world_path}")
    profile = load_profile(args.world)
    links = parse_world_collision_boxes(world_path)

    # R4 stack position summary (sanity print)
    stack_links = [L for L in links if "high_stack" in L["name"]]
    if stack_links:
        cys = [L["cy"] for L in stack_links]
        print(f"R4 stack y centers: {cys}")

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 9.5))

    # Optional GP underlay
    if args.gp:
        d = np.load(args.gp)
        P = d["P_conservative_plan_map"]
        xs, ys = d["xs"], d["ys"]
        im = ax.imshow(P, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                       cmap="Greys", vmin=0, vmax=1.0, aspect="equal", zorder=1, alpha=0.45)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04,
                     label="P_conservative_plan diagnostic underlay")

    # Draw 5 m × 5.5 m bounding box from the SDF wall list (warehouse size 11×10).
    ax.add_patch(Rectangle((-5.5, -5.0), 11.0, 10.0, facecolor="none",
                           edgecolor="#222222", linewidth=1.5, zorder=4))

    # Driveable + staging regions from world_profiles (overlay first so it's underneath physical boxes).
    draw_regions(ax, profile)

    # Physical obstacles from the SDF
    draw_collision_boxes(ax, links)

    # Robot spawn
    spawn = profile.get("spawn", {})
    if spawn:
        ax.scatter([float(spawn["x"])], [float(spawn["y"])], marker="s", s=100,
                   c="#1976d2", edgecolor="white", linewidth=1.0, zorder=22,
                   label="default spawn")
        ax.text(float(spawn["x"]) + 0.15, float(spawn["y"]) - 0.15,
                f"spawn ({spawn['x']}, {spawn['y']})", fontsize=8, color="#1976d2", zorder=22)

    annotate_aisles(ax)
    draw_camera(ax, profile)
    draw_legend(ax)

    ax.set_xlim(-5.7, 5.7)
    ax.set_ylim(-5.2, 5.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)
    title = args.title or "AWS warehouse geometry verification\n" \
        "Green outlines = driveable regions (robot MUST stay inside).\n" \
        "Red hatches = non-driveable staging pads (robot MUST NOT enter)."
    ax.set_title(title, fontsize=11, loc="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
