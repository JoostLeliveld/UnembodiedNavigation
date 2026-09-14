#!/usr/bin/env python3
"""V4 wrapper: add combined start- and goal-lane endpoint variants."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]


def _load_v3():
    path = REPO / "experiments/thesis_pipeline_lock/select_stage09_routes_v3.py"
    spec = importlib.util.spec_from_file_location("stage09_route_selector_v3_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v3 selector from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_v3()
ORIGINAL_CANDIDATES = BASE._candidate_routes


def _candidate_routes(driveable_json: str, start_xy, goal_xy, step_m: float):
    scene, candidates = ORIGINAL_CANDIDATES(
        driveable_json, start_xy, goal_xy, step_m
    )
    base = list(BASE.generate_route_seeds(driveable_json, start_xy, goal_xy))
    vertical_centres = sorted(
        0.5 * float(prism.xmin + prism.xmax)
        for prism in scene.prisms
        if float(prism.ymax - prism.ymin) > float(prism.xmax - prism.xmin)
    )
    for candidate in base:
        route = BASE._dedupe([start_xy, *candidate["waypoints"]])
        first_vertical = (
            len(route) >= 4 and abs(route[1][0] - route[0][0]) < 1e-9
        )
        last_vertical = (
            len(route) >= 4 and abs(route[-1][0] - route[-2][0]) < 1e-9
        )
        if not (first_vertical and last_vertical):
            continue
        for start_centre in vertical_centres:
            for goal_centre in vertical_centres:
                shifted = BASE._dedupe([
                    route[0],
                    [start_centre, route[0][1]],
                    [start_centre, route[1][1]],
                    *route[2:-2],
                    [goal_centre, route[-2][1]],
                    [goal_centre, route[-1][1]],
                    route[-1],
                ])
                if BASE._inside_union(scene, shifted, step_m):
                    candidates.append({
                        "name": (
                            f"{candidate['name']}__start_lane_{start_centre:+.3f}"
                            f"__goal_lane_{goal_centre:+.3f}"
                        ),
                        "waypoints": shifted[1:],
                    })
    unique = []
    seen = set()
    for candidate in candidates:
        route = BASE._dedupe([start_xy, *candidate["waypoints"]])
        key = tuple((round(point[0], 12), round(point[1], 12)) for point in route)
        if key in seen:
            continue
        seen.add(key)
        unique.append({"name": str(candidate["name"]), "waypoints": route[1:]})
    return scene, unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base_path = (REPO / protocol["base_selector"]["path"]).resolve()
    actual = BASE.sha256_file(base_path)
    if actual != protocol["base_selector"]["sha256"]:
        raise RuntimeError(
            f"v3 base-selector SHA-256 mismatch: expected "
            f"{protocol['base_selector']['sha256']}, got {actual}"
        )
    BASE._candidate_routes = _candidate_routes
    BASE.__file__ = str(Path(__file__).resolve())
    BASE.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
