#!/usr/bin/env python3
"""Build the maximal declared obstacle-free floor support for warehouse_v2.

The support is the site-boundary rectangle minus both kinds of registered
obstacle geometry:

* the storage/structure cores reconstructed from the explicitly documented
  0.32 m semantic keep-out envelopes in ``world_profiles.yaml``; and
* every robot-height physical collision AABB used by the runtime planner.

The orthogonal difference is emitted as a deterministic union of rectangles.
Routing-axis metadata preserves the original, condition-neutral aisle graph;
the rectangles themselves are used only as the exact keep-in support.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import yaml
from shapely.geometry import box
from shapely.ops import unary_union


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / path) for path in ("src/experiments", "src/unav_common")]

from experiments.core.world_profiles import (  # noqa: E402
    serialize_collision_geometry_from_world,
)
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402


DEFAULT_PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
DEFAULT_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
DEFAULT_OUTPUT = (
    REPO / "experiments/reference_controlled_commissioning_v1/"
    "maximal_driveable_geometry_20260915_v3.json"
)
INCLUDE_NAMES = (
    "forklift_parked", "pallet_jack", "bin_office",
    "pallet_loose_1", "pallet_loose_2",
)
SEMANTIC_KEEP_OUT_M = 0.32
PHYSICAL_RELEASE_CLEARANCE_M = 0.0


@dataclass(frozen=True, order=True)
class Rect:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def area(self) -> float:
        return max(self.xmax - self.xmin, 0.0) * max(self.ymax - self.ymin, 0.0)


def _clip(rect: Rect, boundary: Rect) -> Rect | None:
    clipped = Rect(
        max(rect.xmin, boundary.xmin), min(rect.xmax, boundary.xmax),
        max(rect.ymin, boundary.ymin), min(rect.ymax, boundary.ymax),
    )
    return clipped if clipped.area > 0.0 else None


def _expanded(rect: Rect, guard_m: float) -> Rect:
    return Rect(
        rect.xmin - guard_m, rect.xmax + guard_m,
        rect.ymin - guard_m, rect.ymax + guard_m,
    )


def _contracted(rect: Rect, margin_m: float) -> Rect:
    contracted = Rect(
        rect.xmin + margin_m, rect.xmax - margin_m,
        rect.ymin + margin_m, rect.ymax - margin_m,
    )
    if contracted.area <= 0.0:
        raise ValueError("semantic keep-out contraction erased an obstacle core")
    return contracted


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for low, high in sorted(intervals):
        if high <= low:
            continue
        if not merged or low > merged[-1][1] + 1.0e-12:
            merged.append([low, high])
        else:
            merged[-1][1] = max(merged[-1][1], high)
    return [(low, high) for low, high in merged]


def rectangular_difference(boundary: Rect, exclusions: list[Rect]) -> list[Rect]:
    """Return an exact deterministic rectangular cover of boundary minus exclusions."""
    clipped = [item for rect in exclusions if (item := _clip(rect, boundary))]
    xs = sorted({boundary.xmin, boundary.xmax, *(
        value for rect in clipped for value in (rect.xmin, rect.xmax)
    )})
    slabs: list[Rect] = []
    for xmin, xmax in zip(xs, xs[1:]):
        if xmax <= xmin:
            continue
        middle = 0.5 * (xmin + xmax)
        blocked = _merge_intervals([
            (rect.ymin, rect.ymax)
            for rect in clipped
            if rect.xmin < middle < rect.xmax
        ])
        cursor = boundary.ymin
        for ymin, ymax in blocked:
            if ymin > cursor:
                slabs.append(Rect(xmin, xmax, cursor, min(ymin, boundary.ymax)))
            cursor = max(cursor, ymax)
        if cursor < boundary.ymax:
            slabs.append(Rect(xmin, xmax, cursor, boundary.ymax))

    # Merge consecutive x-slabs with identical vertical extents. This keeps
    # the JSON and the symbolic boundary model compact without changing area.
    merged: list[Rect] = []
    for rect in sorted(slabs, key=lambda item: (item.ymin, item.ymax, item.xmin)):
        if (merged and abs(merged[-1].xmax - rect.xmin) <= 1.0e-12
                and abs(merged[-1].ymin - rect.ymin) <= 1.0e-12
                and abs(merged[-1].ymax - rect.ymax) <= 1.0e-12):
            previous = merged[-1]
            merged[-1] = Rect(previous.xmin, rect.xmax, rect.ymin, rect.ymax)
        else:
            merged.append(rect)
    return sorted(merged, key=lambda item: (item.ymin, item.xmin, item.ymax, item.xmax))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(profile_path: Path, world_path: Path, guard_m: float) -> dict:
    profiles = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile = profiles["worlds"][world_path.name]
    regions = profile["known_2d_regions"]
    site = next(region for region in regions if region.get("type") == "site_boundary")
    boundary = Rect(
        float(site["xmin"]), float(site["xmax"]),
        float(site["ymin"]), float(site["ymax"]),
    )
    semantic_envelopes = [
        Rect(float(item["xmin"]), float(item["xmax"]),
             float(item["ymin"]), float(item["ymax"]))
        for item in regions if item.get("type") == "non_driveable_obstacle"
    ]
    semantic = [
        _contracted(item, SEMANTIC_KEEP_OUT_M) for item in semantic_envelopes
    ]
    old_lanes = [item for item in regions if item.get("type") == "traversable"]
    horizontal = sorted({
        round(0.5 * (float(item["ymin"]) + float(item["ymax"])), 3)
        for item in old_lanes
        if float(item["xmax"]) - float(item["xmin"])
        > float(item["ymax"]) - float(item["ymin"])
    })
    vertical = sorted({
        round(0.5 * (float(item["xmin"]) + float(item["xmax"])), 3)
        for item in old_lanes
        if float(item["ymax"]) - float(item["ymin"])
        > float(item["xmax"]) - float(item["xmin"])
    })

    collision_json = serialize_collision_geometry_from_world(
        str(world_path),
        model_names=tuple(profile["collision_model_names"]),
        include_names=INCLUDE_NAMES,
    )
    collision_scene = scene_from_json(collision_json)
    physical = [
        Rect(item.xmin, item.xmax, item.ymin, item.ymax)
        for item in collision_scene.prisms
    ]
    exclusions = [
        *(_expanded(rect, guard_m) for rect in semantic),
        *(_expanded(rect, PHYSICAL_RELEASE_CLEARANCE_M + guard_m)
          for rect in physical),
    ]
    cells = rectangular_difference(boundary, exclusions)
    free_area = sum(item.area for item in cells)
    support_shape = unary_union([
        box(item.xmin, item.ymin, item.xmax, item.ymax) for item in cells
    ])
    semantic_shape = unary_union([
        box(item.xmin, item.ymin, item.xmax, item.ymax) for item in semantic
    ])
    physical_shape = unary_union([
        box(item.xmin, item.ymin, item.xmax, item.ymax) for item in physical
    ])
    old_shape = unary_union([
        box(float(item["xmin"]), float(item["ymin"]),
            float(item["xmax"]), float(item["ymax"]))
        for item in old_lanes
    ])
    semantic_overlap = float(support_shape.intersection(semantic_shape).area)
    physical_overlap = float(support_shape.intersection(physical_shape).area)
    if semantic_overlap > 1.0e-12 or physical_overlap > 1.0e-12:
        raise RuntimeError("constructed driveable support overlaps registered obstacles")
    component_count = (
        len(support_shape.geoms) if support_shape.geom_type == "MultiPolygon" else 1
    )
    if component_count != 1 or not support_shape.is_valid:
        raise RuntimeError("constructed driveable support is not one valid component")
    return {
        "schema": "maximal_obstacle_free_driveable_geometry.v3",
        "model_name": "driveable_region_maximal_obstacle_free_v3",
        "source_world": str(world_path.relative_to(REPO)),
        "source_world_sha256": _sha256(world_path),
        "source_profile": str(profile_path.relative_to(REPO)),
        "source_profile_sha256": _sha256(profile_path),
        "source_generator": str(Path(__file__).resolve().relative_to(REPO)),
        "source_generator_sha256": _sha256(Path(__file__).resolve()),
        "construction": "site boundary minus reconstructed storage/structure cores minus runtime physical collision AABBs",
        "numerical_guard_m": guard_m,
        "removed_legacy_semantic_keep_out_m": SEMANTIC_KEEP_OUT_M,
        "physical_release_clearance_m": PHYSICAL_RELEASE_CLEARANCE_M,
        "physical_collision_include_names": list(INCLUDE_NAMES),
        "physical_collision_geometry_sha256": hashlib.sha256(
            collision_json.encode("utf-8")
        ).hexdigest(),
        "site_boundary": boundary.__dict__,
        "route_horizontal_centres": horizontal,
        "route_vertical_centres": vertical,
        "semantic_exclusion_count": len(semantic),
        "physical_collision_prism_count": len(physical),
        "old_driveable_area_m2": float(old_shape.area),
        "free_area_m2": free_area,
        "net_area_gain_m2": float(free_area - old_shape.area),
        "validation": {
            "connected_component_count": component_count,
            "valid_geometry": bool(support_shape.is_valid),
            "semantic_obstacle_overlap_m2": semantic_overlap,
            "physical_obstacle_overlap_m2": physical_overlap,
            "minimum_semantic_obstacle_gap_m": float(
                support_shape.distance(semantic_shape)
            ),
            "minimum_physical_obstacle_gap_m": float(
                support_shape.distance(physical_shape)
            ),
        },
        "prisms": [
            {
                "name": f"free_cell_{index:03d}",
                **rect.__dict__, "zmin": 0.0, "zmax": 0.1,
            }
            for index, rect in enumerate(cells, start=1)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--guard-m", type=float, default=0.001)
    args = parser.parse_args()
    if not 0.0 < args.guard_m <= 0.01:
        raise ValueError("guard-m must be in (0, 0.01]")
    payload = build(args.profile.resolve(), args.world.resolve(), args.guard_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "cells": len(payload["prisms"]),
        "free_area_m2": payload["free_area_m2"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
