#!/usr/bin/env python3
"""Check the world, the declared map and the profile against OPERATING_LOCK.json.

This is NOT world_freeze.py. That verifies the world still regenerates from
make_world.py, and it currently FAILS by design: the world file was edited after
the freeze and is what every capture actually used. This checks the weaker but
operative property - that the world and map have not moved since the captures.

Exit code 0 = locked state intact, 1 = drifted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCK = HERE / "OPERATING_LOCK.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock = json.loads(LOCK.read_text())
    failures: list[str] = []

    world = REPO / lock["world"]["path"]
    if not world.is_file():
        failures.append(f"world missing: {world}")
    else:
        got = sha256(world)
        if got != lock["world"]["sha256"]:
            failures.append(f"world sha256 {got[:12]} != locked {lock['world']['sha256'][:12]}")

    profile = REPO / lock["profile"]["path"]
    if not profile.is_file():
        failures.append(f"profile missing: {profile}")
    else:
        prof = yaml.safe_load(profile.read_text())["worlds"]["warehouse_v2.world.sdf"]
        canon = hashlib.sha256(json.dumps(
            prof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if canon != lock["profile"]["canonical_sha256"]:
            failures.append(
                f"profile canonical {canon[:12]} != locked {lock['profile']['canonical_sha256'][:12]}")
        known = prof["known_2d_regions"]
        trav = sum(1 for r in known if str(r.get("type", "")).lower() == "traversable")
        nogo = sum(1 for r in known
                   if str(r.get("type", "")).lower() not in ("traversable", "site_boundary"))
        # The map is one statement: the complement of the lanes IS non-drivable.
        # A non_driveable declaration here would re-introduce the contradiction
        # where 19.66 m2 was both drivable and not.
        if trav:
            failures.append(
                f"warehouse_v2 declares {trav} traversable regions; the map is the obstacle set "
                "and its complement, so there must be none")
        for key in ("route_horizontal_centres", "route_vertical_centres"):
            if not prof.get(key):
                failures.append(f"profile is missing {key}, required for lane-graph route seeding")
        if nogo != lock["profile"]["obstacle_box_count"]:
            failures.append(f"obstacle boxes {nogo} != {lock['profile']['obstacle_box_count']}")

    # The parse contract is the part that silently breaks analysis, so check it too.
    sys.path.insert(0, str(REPO / "src/unav_common"))
    from unav_common.occlusion_geometry import parse_collision_scene_from_world
    c = lock["collision_parse_contract"]
    scene = parse_collision_scene_from_world(
        str(world), model_names=tuple(c["model_names"]),
        include_names=tuple(c["include_names"]), robot_z_range=tuple(c["robot_z_range"]))
    if len(scene.prisms) != c["prism_count_expected"]:
        failures.append(f"collision prisms {len(scene.prisms)} != {c['prism_count_expected']}")

    # The planner contract: the campaign must use keep_out over the obstacle
    # prisms, at the locked safe distance, and must not be silently handed the
    # lane union instead. Checked end-to-end rather than assumed.
    contract = lock.get("planner_contract")
    if contract:
        import numpy as np
        sys.path.insert(0, str(REPO / "src/experiments"))
        from experiments.core.world_profiles import (
            serialize_collision_geometry_from_world, serialize_driveable_geometry_from_profile)
        from unav_common.occlusion_geometry import signed_distance_to_union_xy, AxisAlignedPrism

        campaign = yaml.safe_load(
            (REPO / "experiments/thesis_pipeline_lock/stage09_navigation_campaign.yaml").read_text())
        mode = str(campaign.get("nogo_mode", "")).strip().lower()
        if mode != contract["nogo_mode"]:
            failures.append(f"campaign nogo_mode {mode!r} != {contract['nogo_mode']!r}")
        for key in ("nogo_safe_distance", "local_nogo_safe_distance"):
            got = float(campaign.get(key, -1))
            if abs(got - contract["nogo_safe_distance_m"]) > 1e-9:
                failures.append(f"campaign {key} {got} != {contract['nogo_safe_distance_m']}")

        # What geometry does the launch path actually hand the planner?
        if mode == "keep_out":
            geom = json.loads(serialize_collision_geometry_from_world(
                str(world), model_names=tuple(prof.get("collision_model_names") or ()),
                include_names=tuple(prof.get("collision_include_names") or ())))
        else:
            geom = json.loads(serialize_driveable_geometry_from_profile(prof))
        if len(geom["prisms"]) != c["prism_count_expected"]:
            failures.append(
                f"planner would receive {len(geom['prisms'])} prisms, not {c['prism_count_expected']}")
        pr = [AxisAlignedPrism(name=q["name"], xmin=q["xmin"], xmax=q["xmax"],
                               ymin=q["ymin"], ymax=q["ymax"], zmin=q["zmin"], zmax=q["zmax"])
              for q in geom["prisms"]]
        # Sample the ACTUAL drivable region: inside the site boundary and outside
        # every obstacle box. Sampling declared rectangles would be vacuous now
        # that the map has none.
        from shapely.geometry import Point, box as _box
        from shapely.ops import unary_union as _union
        site = next(r for r in prof["known_2d_regions"]
                    if str(r.get("type", "")).lower() == "site_boundary")
        boxes = _union([_box(float(r["xmin"]), float(r["ymin"]),
                             float(r["xmax"]), float(r["ymax"]))
                        for r in prof["known_2d_regions"]
                        if str(r.get("type", "")).lower() == "non_driveable_obstacle"])
        drivable = _box(float(site["xmin"]), float(site["ymin"]),
                        float(site["xmax"]), float(site["ymax"])).difference(boxes)
        rng = np.random.default_rng(0)
        pts = []
        while len(pts) < 800:
            x = rng.uniform(float(site["xmin"]), float(site["xmax"]))
            y = rng.uniform(float(site["ymin"]), float(site["ymax"]))
            if drivable.contains(Point(x, y)):
                pts.append((x, y))
        d = signed_distance_to_union_xy(pr, np.asarray(pts), keep_in=(mode == "keep_in"))
        tot = len(d)
        bad = int((d <= 0).sum())
        if bad:
            failures.append(
                f"{bad} of {tot} points in the drivable region are inside the planner's no-go "
                "union; the planner would be penalised where it may legitimately drive")

    # Route seeding must still produce routes against the keep-out map.
    if contract and not failures:
        from unav_common.lane_graph_routes import generate_route_seeds
        blob = serialize_collision_geometry_from_world(
            str(world), model_names=tuple(prof.get("collision_model_names") or ()),
            include_names=tuple(prof.get("collision_include_names") or ()), profile=prof)
        for start, goal in (((-8.1, -8.7), (9.5, 6.5)), ((0.0, -8.5), (0.0, 8.5))):
            if not generate_route_seeds(blob, start, goal):
                failures.append(f"lane-graph seeding produced no route for {start} -> {goal}")

    if failures:
        print("OPERATING LOCK: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("OPERATING LOCK: OK")
    print(f"  world    {lock['world']['sha256'][:12]}  ({lock['world']['size_bytes']} bytes)")
    print(f"  profile  {lock['profile']['canonical_sha256'][:12]}  "
          f"({lock['profile']['obstacle_box_count']} obstacle boxes + "
          f"{len(prof.get('route_horizontal_centres', []))}h/"
          f"{len(prof.get('route_vertical_centres', []))}v route axes)")
    print(f"  prisms   {len(scene.prisms)} (include_names applied)")
    if contract:
        print(f"  planner  nogo_mode={contract['nogo_mode']} "
              f"safe_distance={contract['nogo_safe_distance_m']} "
              f"| 0 of {tot} drivable-region points penalised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
