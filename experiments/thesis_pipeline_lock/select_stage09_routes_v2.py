#!/usr/bin/env python3
"""Stage-09 route selector with a predeclared static body-clearance gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis09_mpl")
os.environ.setdefault(
    "ROS_LOG_DIR", str(Path(tempfile.gettempdir()) / "roslog_thesis09_route_selection")
)

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / path) for path in ("src/unav_common", "src/experiments")]

from unav_common.occlusion_geometry import (  # noqa: E402
    scene_from_json,
    signed_distance_to_union_xy,
)
from unav_common.preselected_route import sample_polyline, sha256_file  # noqa: E402
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402


def _load_base_selector():
    path = REPO / "experiments/thesis_pipeline_lock/select_stage09_routes.py"
    spec = importlib.util.spec_from_file_location("stage09_route_selector_v1_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base selector from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_selector()
ORIGINAL_GENERATOR = BASE.generate_route_seeds
ACTIVE_PROTOCOL: dict = {}


def _dedupe(points):
    result = []
    for point in points:
        xy = [float(point[0]), float(point[1])]
        if not result or math.hypot(xy[0] - result[-1][0], xy[1] - result[-1][1]) > 1e-12:
            result.append(xy)
    return result


def _inside_union(scene, points, step_m):
    samples = sample_polyline(points, maximum_step_m=step_m)
    signed = np.asarray(
        signed_distance_to_union_xy(scene.prisms, samples, keep_in=True), dtype=float
    )
    return bool(signed.size and np.isfinite(signed).all() and np.max(signed) <= 1e-9)


def _minimum_static_body_clearance(scene, route, *, length_m, width_m, step_m):
    body = RectangularFootprint(
        scene.prisms, float(length_m), float(width_m), keep_in=True
    )
    headings = [
        math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(route, route[1:])
    ]
    clearances = []
    for (start, end), heading in zip(zip(route, route[1:]), headings):
        for x, y in sample_polyline([start, end], maximum_step_m=step_m):
            clearances.append(body.clearance([x, y, heading]))
    for index in range(1, len(route) - 1):
        before, after = headings[index - 1], headings[index]
        delta = (after - before + math.pi) % (2.0 * math.pi) - math.pi
        for fraction in np.linspace(0.0, 1.0, 37):
            clearances.append(
                body.clearance(
                    [route[index][0], route[index][1], before + fraction * delta]
                )
            )
    return float(min(clearances)) if clearances else -math.inf


def _geometry_candidates_with_body_gate(driveable_json, start_xy, goal_xy):
    gate = ACTIVE_PROTOCOL["route_gate"]
    scene = scene_from_json(driveable_json)
    candidates = list(ORIGINAL_GENERATOR(driveable_json, start_xy, goal_xy))

    # The lane graph normally uses each task endpoint's x coordinate when a
    # vertical connection exists.  Add the centres of vertical lane rectangles
    # as geometry-only alternatives so a valid but edge-hugging direct route is
    # not the only candidate.
    for prism in scene.prisms:
        if float(prism.ymax - prism.ymin) <= float(prism.xmax - prism.xmin):
            continue
        centre_x = 0.5 * float(prism.xmin + prism.xmax)
        route = _dedupe(
            [
                start_xy,
                [centre_x, float(start_xy[1])],
                [centre_x, float(goal_xy[1])],
                goal_xy,
            ]
        )
        if len(route) < 2 or not _inside_union(
            scene, route, float(gate["sample_step_m"])
        ):
            continue
        candidates.append(
            {
                "name": f"vertical_lane_centre_{centre_x:+.3f}",
                "waypoints": route[1:],
            }
        )

    eligible = []
    seen = set()
    for candidate in candidates:
        route = _dedupe([start_xy, *candidate["waypoints"]])
        key = json.dumps(route, separators=(",", ":"), allow_nan=False)
        if key in seen:
            continue
        seen.add(key)
        clearance = _minimum_static_body_clearance(
            scene,
            route,
            length_m=float(gate["robot_length_m"]),
            width_m=float(gate["robot_width_m"]),
            step_m=float(gate["sample_step_m"]),
        )
        if clearance + 1e-9 < float(gate["minimum_static_body_clearance_m"]):
            continue
        eligible.append(candidate)
    return eligible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO
        / "experiments/thesis_pipeline_lock/stage09_route_selection_protocol_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()

    global ACTIVE_PROTOCOL
    ACTIVE_PROTOCOL = json.loads(args.protocol.read_text(encoding="utf-8"))
    base_path = REPO / ACTIVE_PROTOCOL["base_selector"]["path"]
    actual_base = sha256_file(base_path)
    expected_base = ACTIVE_PROTOCOL["base_selector"]["sha256"]
    if actual_base != expected_base:
        raise RuntimeError(
            f"base selector SHA-256 mismatch: expected {expected_base}, got {actual_base}"
        )

    BASE.generate_route_seeds = _geometry_candidates_with_body_gate
    # The base implementation binds the selected method through __file__.  For
    # this versioned run, that identity is this wrapper; the base source is bound
    # separately above and in the protocol.
    BASE.__file__ = str(Path(__file__).resolve())
    BASE.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
