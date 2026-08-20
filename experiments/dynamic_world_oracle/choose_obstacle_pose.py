#!/usr/bin/env python3
"""Sweep an aisle and report what an obstacle there would hide, per camera.

This is how a scenario's poses get chosen instead of guessed.  Give it a lane and
an obstacle, and it prints how many visible floor cells each camera loses with the
obstacle at each position along that lane.  A good "this obstacle intersects a
SELECTED camera's rays" pose is one where a single column of the table moves and
the rest stay at zero.

It needs no simulator: the answer comes from the same CAD prisms and camera model
the oracle uses at run time, so a pose chosen here is the pose the run will
reproduce.

    python3 experiments/dynamic_world_oracle/choose_obstacle_pose.py \
        --model dyn_pallet_box --lane rack_aisle_W2 --along y

Lanes come from the four-camera world's profile (``known_2d_regions``), so the
sweep can only propose places the planner already calls driveable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "scripts" / "shared"))

import oracle as ora  # noqa: E402
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
WORLDS = REPO / "src" / "sim" / "gazebo_worlds" / "worlds"
STAGE_JSON = WORLDS / "warehouse_full_4cam_dynamic.stage.json"
PROFILE = REPO / "src" / "experiments" / "config" / "world_profiles.yaml"
BASE_WORLD_KEY = "warehouse_full_4cam.world.sdf"

SHORT = {"external_camera": "A", "external_camera_b": "B",
         "external_camera_c": "C", "external_camera_d": "D"}


def lane_rect(name: str) -> dict:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    regions = profile["worlds"][BASE_WORLD_KEY]["known_2d_regions"]
    for region in regions:
        if region.get("name") == name:
            return region
    raise SystemExit(f"no lane named {name!r}; known lanes: "
                     f"{[r['name'] for r in regions if r.get('type') == 'traversable']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="dyn_pallet_box")
    parser.add_argument("--lane", default="rack_aisle_W2")
    parser.add_argument("--along", choices=("x", "y"), default="y",
                        help="axis to sweep along; the obstacle is centred across the other")
    parser.add_argument("--yaw-deg", type=float, default=None,
                        help="obstacle yaw; default aligns its long axis with the sweep axis")
    parser.add_argument("--step-m", type=float, default=1.0)
    parser.add_argument("--target-height-m", type=float, default=0.35)
    parser.add_argument("--grid-resolution-m", type=float, default=0.25)
    args = parser.parse_args(argv)

    stage = json.loads(STAGE_JSON.read_text(encoding="utf-8"))
    catalogue = {c["model_name"]: c for c in stage["obstacle_catalogue"]}
    if args.model not in catalogue:
        raise SystemExit(f"unknown model {args.model!r}; catalogue: {sorted(catalogue)}")
    parts = ora.parts_from_model_sdf(REPO / catalogue[args.model]["model_sdf"], args.model)

    world = WORLDS / stage["world_file"]
    camera_ids = [c["camera_id"] for c in stage["cameras"]]
    scene = ora.OracleScene.from_world(world, camera_ids)
    bounds = stage["site_bounds"]
    grid = ora.FloorGrid(bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"],
                         args.grid_resolution_m)
    baseline = ora.visibility_grids(scene.cameras, grid, scene.static_prisms,
                                    target_height_m=args.target_height_m)
    base_visible = {c: int((g == ora.VISIBLE).sum()) for c, g in baseline.items()}

    lane = lane_rect(args.lane)
    yaw = (np.pi / 2 if args.along == "y" else 0.0) if args.yaw_deg is None \
        else np.deg2rad(args.yaw_deg)
    across = "x" if args.along == "y" else "y"
    centre_across = 0.5 * (lane[f"{across}min"] + lane[f"{across}max"])
    lane_width = lane[f"{across}max"] - lane[f"{across}min"]
    footprint = ora.place_obstacle("probe", args.model, parts, {"x": 0.0, "y": 0.0, "yaw": yaw})
    bound = footprint.aabb
    width_across = (bound.xmax - bound.xmin) if across == "x" else (bound.ymax - bound.ymin)

    print(f"obstacle : {args.model}  {catalogue[args.model]['description']}")
    print(f"lane     : {args.lane}  {across}[{lane[f'{across}min']:+.2f},{lane[f'{across}max']:+.2f}] "
          f"= {lane_width:.2f} m wide")
    print(f"placement: yaw {np.rad2deg(yaw):+.0f} deg, {width_across:.2f} m across the lane -> "
          f"{(lane_width - width_across) / 2:.3f} m clearance each side"
          f"{'   *** DOES NOT FIT ***' if width_across > lane_width else ''}")
    print(f"baseline : visible cells " +
          "  ".join(f"{SHORT.get(c, c)}={base_visible[c]}" for c in camera_ids))
    print()
    header = f"{args.along:>7} | " + " | ".join(f"{SHORT.get(c, c):>4}" for c in camera_ids)
    print(header + "   visible cells LOST vs the clear aisle")
    print("-" * len(header))

    lo, hi = lane[f"{args.along}min"], lane[f"{args.along}max"]
    for value in np.arange(lo, hi + 1e-9, args.step_m):
        pose = {args.along: float(value), across: centre_across, "z": 0.0, "yaw": yaw}
        box = ora.place_obstacle("probe", args.model, parts, pose)
        grids = ora.visibility_grids(scene.cameras, grid, scene.static_prisms, box.prisms,
                                     target_height_m=args.target_height_m)
        lost = [base_visible[c] - int((grids[c] == ora.VISIBLE).sum()) for c in camera_ids]
        flag = "  <- one camera only" if sum(1 for v in lost if v > 0) == 1 else ""
        print(f"{value:>7.2f} | " + " | ".join(f"{v:>4d}" for v in lost) + flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
