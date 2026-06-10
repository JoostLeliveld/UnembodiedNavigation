#!/usr/bin/env python3
"""Top-down metric verification that the camera-image floor markings match the
PLANNER driveable/no-go geometry.

Overlays, in one equal-aspect metric plot:
  * planner driveable region  = union of the 9 prisms (blue fill + outline)
  * no-go regions             = computed rack columns (green fill + outline)
  * shelf/rack/wall footprints = SDF *collision* geometry (grey)
  * generated floor markings   = the blue/green rectangles emitted into the SDF
                                 (dashed) — must coincide with the planner region
  * start/goal + camera glyph

Save: logs/paper_figures/driveable_region_alignment.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "paper_figures"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))

from generate_driveable_overlay_sdf import (  # noqa: E402
    _load_prisms,
    build_overlay,
    load_obstacle_footprints,
)

CONFIG = REPO / "scripts/visibility_comparison/aws_f86a_camera_xy_config.yaml"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"
TASKS = REPO / "src/experiments/config/tasks.yaml"
TASK = "a0_west_to_a1_upper_blocked_mid"
OUT_PNG = REPO / "logs/paper_figures/driveable_region_alignment.png"
OUT_PDF = REPO / "logs/paper_figures/driveable_region_alignment.pdf"


def _collision_footprints(world_path: Path):
    """Axis-aligned XY footprints of every collision box in the world."""
    from unav_common.occlusion_geometry import parse_occlusion_scene_from_world
    scene = parse_occlusion_scene_from_world(str(world_path), geometry_tags=("collision",))
    return list(scene.prisms)


def _task_start_goal(tasks_path: Path):
    data = yaml.safe_load(tasks_path.read_text())
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    # `tasks` is keyed by world; each value is a list of task dicts.
    groups = tasks.values() if isinstance(tasks, dict) else [tasks]
    for group in groups:
        for t in group:
            if isinstance(t, dict) and t.get("name") == TASK:
                s, g = t["start"], t["goal"]
                return (float(s["x"]), float(s["y"])), (float(g["x"]), float(g["y"]))
    return None, None


def main() -> int:
    prisms = _load_prisms(CONFIG)
    footprints = load_obstacle_footprints(WORLD)
    _model, summary = build_overlay(prisms, footprints)
    ux0, ux1, uy0, uy1 = summary["outer_bbox"]
    nogo = summary["nogo_regions"]
    connectors = summary["connectors"]
    coll_footprints = _collision_footprints(WORLD)
    start, goal = _task_start_goal(TASKS)

    fig, ax = plt.subplots(figsize=(8.5, 8.0))

    # Planner driveable prisms (light blue fill).
    for p in prisms:
        ax.add_patch(Rectangle((float(p["xmin"]), float(p["ymin"])),
                               float(p["xmax"]) - float(p["xmin"]),
                               float(p["ymax"]) - float(p["ymin"]),
                               facecolor="#cfe2ff", edgecolor="none", alpha=0.55, zorder=1))
    # Mid-gap connectors (driveable, light blue) — so driveable + no-go tile fully.
    for (cx0, cx1, cy0, cy1) in connectors:
        ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0,
                               facecolor="#cfe2ff", edgecolor="none", alpha=0.55, zorder=1))
    # No-go columns (light green fill) — span aisle-edge to aisle-edge.
    for (gx0, gx1, gy0, gy1) in nogo:
        ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0,
                               facecolor="#cdeccd", edgecolor="none", alpha=0.7, zorder=2))
    # Collision footprints (grey): walls/racks/crates/stack.
    for fp in coll_footprints:
        name = getattr(fp, "name", "")
        c = "#9aa0a6" if "wall" in name else "#7f7f7f"
        ax.add_patch(Rectangle((fp.xmin, fp.ymin), fp.xmax - fp.xmin, fp.ymax - fp.ymin,
                               facecolor=c, edgecolor="#3a3a3a", linewidth=0.5, alpha=0.85, zorder=5))

    # Generated SDF floor markings (dashed) — must coincide with planner region.
    ax.add_patch(Rectangle((ux0, uy0), ux1 - ux0, uy1 - uy0, fill=False,
                           edgecolor="#0a3fcc", linewidth=2.2, linestyle=(0, (6, 3)),
                           zorder=8, label="floor marking: outer driveable (blue)"))
    for k, (gx0, gx1, gy0, gy1) in enumerate(nogo):
        ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                               edgecolor="#0a8a2a", linewidth=1.8, linestyle=(0, (4, 2)),
                               zorder=8, label="floor marking: no-go (green)" if k == 0 else None))

    if start:
        ax.scatter([start[0]], [start[1]], s=90, c="#2ca02c", edgecolor="black",
                   zorder=12, label="start")
    if goal:
        ax.scatter([goal[0]], [goal[1]], s=170, marker="*", c="#ffd51a", edgecolor="black",
                   zorder=12, label="goal")
    ax.scatter([0.0], [-5.5], marker="v", s=90, c="#111111", edgecolor="white", lw=0.8,
               zorder=12, label="camera")

    ax.set_aspect("equal")
    ax.set_xlim(-6.2, 5.7)
    ax.set_ylim(-5.3, 5.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Driveable-region alignment: planner geometry vs rendered floor markings\n"
                 "blue = driveable (aisles + mid-gap connectors), green = no-go region "
                 "(inter-aisle columns, R1 solid / R2-R5 split)")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")
    print(f"driveable prisms: {len(prisms)}, no-go segments: {len(nogo)}, "
          f"collision footprints: {len(coll_footprints)}, start={start}, goal={goal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
