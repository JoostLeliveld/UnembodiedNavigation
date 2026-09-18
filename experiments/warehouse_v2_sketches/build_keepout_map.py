#!/usr/bin/env python3
"""Build the warehouse_v2 non-traversable map: one box per obstacle, +MARGIN a side.

The map is one statement and nothing else:

    non-traversable = every collision footprint, grown by MARGIN on all four sides
    traversable     = everything else inside the site boundary

MARGIN is 0.10 m. It is a fixed keep-clear band around each obstacle, not a
chassis derivation: the planner already checks its own footprint, so the map
does not need to encode the robot.

Obstacles come from the world .sdf with include_names applied, so the map cannot
drift from the world and cannot miss the bin, forklift, pallet jack or pallets.

Writes the boxes into world_profiles.yaml, replacing whatever was there.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src/unav_common"))
from unav_common.occlusion_geometry import parse_collision_scene_from_world  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
PROFILES = REPO / "src/experiments/config/world_profiles.yaml"
COLLISION_INCLUDES = ("forklift_parked", "pallet_jack", "bin_office",
                      "pallet_loose_1", "pallet_loose_2")
MARGIN = 0.10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()

    scene = parse_collision_scene_from_world(
        str(WORLD), model_names=("warehouse_shell", "warehouse_v2_occluders"),
        include_names=COLLISION_INCLUDES, robot_z_range=(0.0, 0.55))
    if len(scene.prisms) != 56:
        raise RuntimeError(f"expected 56 collision prisms, got {len(scene.prisms)}")

    m = float(args.margin)
    note = f"obstacle footprint grown by {m:.2f} m a side; everything else is traversable"
    rows = [
        "      - {name: %s, type: non_driveable_obstacle, "
        "xmin: %.3f, xmax: %.3f, ymin: %.3f, ymax: %.3f, note: %s}"
        % (p.name.replace("/", ":").replace(":collision", ""),
           p.xmin - m, p.xmax + m, p.ymin - m, p.ymax + m, note)
        for p in scene.prisms
    ]

    lines = PROFILES.read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("  warehouse_v2.world.sdf:"))
    end = next(i for i, l in enumerate(lines)
               if i > start and l.startswith("  warehouse_v2_shipout.world.sdf:"))
    block = lines[start:end]
    # Drop every previous region except the site boundary, then write the boxes.
    kept = [l for l in block
            if not re.search(r"^\s+- \{name: \S+, type: (traversable|non_driveable\w*),", l)]
    site_i = next(i for i, l in enumerate(kept) if "type: site_boundary" in l)
    PROFILES.write_text("\n".join(
        lines[:start] + kept[:site_i + 1] + rows + kept[site_i + 1:] + lines[end:]))
    print(f"wrote {len(rows)} non-traversable boxes (+{m:.2f} m a side) into {PROFILES.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
