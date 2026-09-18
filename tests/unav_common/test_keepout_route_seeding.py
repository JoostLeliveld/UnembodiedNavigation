"""Route seeding must work against the keep-out (obstacle) map.

The declared map is the obstacle set: one box per collision footprint grown by a
fixed margin, everything else traversable. These tests pin the three things that
silently produced ZERO routes while being individually plausible:

  1. the payload must be tagged keepout_region, or the seeder applies keep-in
     semantics and every segment fails;
  2. corridor axes must be declared, because inferring them from obstacle boxes
     would place centre-lines inside obstacles;
  3. a segment through an obstacle must be rejected, and one down a clear aisle
     accepted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/unav_common"))
sys.path.insert(0, str(REPO / "src/experiments"))

from experiments.core.world_profiles import serialize_collision_geometry_from_world  # noqa: E402
from unav_common.lane_graph_routes import (  # noqa: E402
    KEEPOUT_MODEL, _route_centres_from_geometry, _segment_free, generate_route_seeds,
)
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
PROFILES = REPO / "src/experiments/config/world_profiles.yaml"


@pytest.fixture(scope="module")
def geometry() -> str:
    profile = yaml.safe_load(PROFILES.read_text())["worlds"]["warehouse_v2.world.sdf"]
    return serialize_collision_geometry_from_world(
        str(WORLD),
        model_names=tuple(profile["collision_model_names"]),
        include_names=tuple(profile["collision_include_names"]),
        profile=profile,
    )


def test_payload_is_tagged_keepout_and_carries_axes(geometry: str) -> None:
    payload = json.loads(geometry)
    assert payload["model_name"] == KEEPOUT_MODEL
    assert len(payload["prisms"]) == 56, "include_names must be applied"
    assert payload["route_horizontal_centres"]
    assert payload["route_vertical_centres"]


def test_keepout_geometry_without_declared_axes_is_refused() -> None:
    payload = {"model_name": KEEPOUT_MODEL, "prisms": [
        {"name": "o", "xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0,
         "zmin": 0.0, "zmax": 0.1}]}
    blob = json.dumps(payload)
    with pytest.raises(ValueError, match="must be declared"):
        _route_centres_from_geometry(blob, scene_from_json(blob).prisms)


def test_segment_rejects_obstacle_and_accepts_clear_aisle(geometry: str) -> None:
    prisms = scene_from_json(geometry).prisms
    # straight through bin_office at (-8.71, -7.45)
    assert not _segment_free(prisms, (-9.5, -7.45), (-8.0, -7.45), keep_out=True)
    # along the southern apron, clear of everything
    assert _segment_free(prisms, (-10.0, -6.25), (10.0, -6.25), keep_out=True)


@pytest.mark.parametrize(
    "start,goal",
    [((-8.1, -8.7), (9.5, 6.5)), ((-10.5, 0.0), (10.5, 0.0)),
     ((0.0, -8.5), (0.0, 8.5)), ((-10.0, 8.0), (10.0, -8.0))],
)
def test_route_seeds_exist_and_avoid_obstacles(geometry: str, start, goal) -> None:
    seeds = generate_route_seeds(geometry, start, goal)
    assert seeds, f"no route seeds for {start} -> {goal}"
    prisms = scene_from_json(geometry).prisms
    for seed in seeds:
        pts = [(float(x), float(y)) for x, y in seed["waypoints"]]
        for a, b in zip(pts, pts[1:]):
            assert _segment_free(prisms, a, b, keep_out=True), \
                f"seed {seed['name']} crosses an obstacle"
