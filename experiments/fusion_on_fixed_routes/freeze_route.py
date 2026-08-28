#!/usr/bin/env python3
"""Freeze the one route all six fusion arms drive, as a hash-bound artifact.

The arms must execute the coordinates chosen here, not a route repaired at launch time, so
the polyline is written once with its provenance and its sha256, and the campaign refers to
it by hash. Route choice is not the treatment in this experiment: the candidates come from
the lane geometry alone and the shortest is taken.

    python3 experiments/fusion_on_fixed_routes/freeze_route.py [--check]

Writes experiments/fusion_on_fixed_routes/route_fusion_network_traverse.json and prints the
fields the campaign config needs. Reads no detector output and starts no simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for extra in ("experiments/warehouse_v2_sketches", "src/experiments", "src/unav_common"):
    sys.path.insert(0, str(REPO / extra))
import route_tasks as RT                                                # noqa: E402
from unav_common.preselected_route import (                             # noqa: E402
    canonicalize_polyline_json, route_sha256, sha256_file,
)

#: Every route the fusion arms drive. One entry per task, frozen into its own artifact.
#:
#: Why four. One route cannot separate "how a fusion rule behaves" from "how THIS corridor
#: behaves", and a 30 m traverse cannot show whether an over-claim compounds. So:
#:   - network_traverse  the mixed corridor: 0 to 4 cameras, the original
#:   - overlap_rich      the SAME endpoints through the best-covered corridor (92% two-plus)
#:   - overlap_sparse    the SAME endpoints through the worst-covered one that still clears
#:                       the planner's margin (45% two-plus, 18% with no camera at all)
#:   - long_traverse     twice the distance, out to the east cross aisle and back west, so
#:                       the robot revisits places with a belief that has already drifted
#:
#: rich and sparse share endpoints with the original by design: same start, same goal, same
#: length to within a metre, opposite camera character. That is the control that separates the
#: corridor from the rule.
ROUTES = {
    "fusion_network_traverse": {"chain": ["dock_w", "xaisle_e"], "candidate": "shortest",
                                "note": "the mixed corridor"},
    "fusion_overlap_rich": {"chain": ["dock_w", "xaisle_e"], "candidate": 1,
                            "note": "same endpoints, the best-covered corridor"},
    "fusion_overlap_sparse": {"chain": ["dock_w", "xaisle_e"], "candidate": 3,
                              "note": "same endpoints, the worst-covered corridor that clears"},
    "fusion_long_traverse": {"chain": ["dock_w", "xaisle_e", "aisleC_s_w", "apron_w"],
                             "candidate": "shortest",
                             "note": "twice the distance, out and back"},
}
CAMPAIGN = REPO / "scripts/visibility_comparison/fusion_on_fixed_routes_campaign.yaml"
#: one hash-bound artifact per route
OUT_DIR = Path(__file__).resolve().parent / "routes"
#: What the polyline must clear along its whole length, measured the way the launch gate
#: measures it. This is the planner's own nogo_safe_distance: 0.275 m of robot half-width
#: plus a 0.075 m margin. Anything less and the planner would be told to follow a route its
#: own cost function refuses to drive.
REQUIRED_CLEARANCE_M = 0.35
#: Grid erosions to try, smallest first. The generator erodes a 0.10 m grid, but a polyline
#: joining cell centres cuts corners diagonally and loses about 0.14 m of the erosion it was
#: given -- which is why the erosion that produces the route is larger than the clearance the
#: route is required to keep. Measured, not assumed: at 0.35 m erosion this route's tightest
#: point measures 0.212 m.
EROSION_LADDER = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
ENDPOINT_TOLERANCE_M = 0.25                # the launch gate's own limit


#: Douglas-Peucker tolerance for turning the grid path into a route a controller can track.
#: The generator walks a 0.10 m grid, so its polyline is a staircase of 280-odd points. The
#: preselected-route contract executes EVERY coordinate as a waypoint -- deliberately, so the
#: driven route is the selected one -- and turn_then_go then tries to face a point 0.10 m
#: away, where a few centimetres of belief error swamp the bearing. Measured: the robot span
#: in place at 20.8 m of 30.7 until the stuck detector fired. Simplifying to the corners the
#: route actually turns at fixes the cause rather than loosening the detector.
SIMPLIFY_TOLERANCE_M = 0.20


def _simplify(points, tolerance_m: float):
    """Douglas-Peucker: keep the vertices that carry the shape, drop the staircase."""

    pts = [tuple(float(v) for v in p) for p in points]
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        a, b = np.asarray(pts[lo]), np.asarray(pts[hi])
        ab = b - a
        norm = float(np.linalg.norm(ab))
        worst, worst_i = -1.0, -1
        for i in range(lo + 1, hi):
            p = np.asarray(pts[i])
            if norm < 1.0e-12:
                dist = float(np.linalg.norm(p - a))
            else:
                t = float(np.clip(np.dot(p - a, ab) / (norm * norm), 0.0, 1.0))
                dist = float(np.linalg.norm(p - (a + t * ab)))
            if dist > worst:
                worst, worst_i = dist, i
        if worst > tolerance_m and worst_i > lo:
            keep[worst_i] = True
            stack.append((lo, worst_i))
            stack.append((worst_i, hi))
    return [p for p, k in zip(pts, keep) if k]


def _driveable_geometry_json() -> str:
    """The same driveable geometry the campaign and the launch gate measure against."""

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "campaign_runner", REPO / "scripts/visibility_comparison/run_visibility_campaign.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner._profile_driveable_geometry({"world": "warehouse_v2.world.sdf"}, CAMPAIGN)


def _task_endpoints(task: str):
    import yaml
    tasks = yaml.safe_load((REPO / "src/experiments/config/tasks.yaml").read_text())
    task_name = task
    for task in tasks["tasks"]["warehouse_v2.world.sdf"]:
        if task["name"] == task_name:
            return (np.array([float(task["start"]["x"]), float(task["start"]["y"])]),
                    np.array([float(task["goal"]["x"]), float(task["goal"]["y"])]))
    raise SystemExit(f"task {task_name} is not registered for warehouse_v2")


def _min_clearance(points, scene):
    from unav_common.occlusion_geometry import signed_distance_to_union_xy
    from unav_common.preselected_route import sample_polyline
    samples = sample_polyline([tuple(map(float, p)) for p in points], maximum_step_m=0.04)
    clearance = -np.asarray(
        signed_distance_to_union_xy(scene.prisms, samples, keep_in=True), dtype=float)
    worst = int(np.argmin(clearance))
    return float(clearance[worst]), tuple(float(v) for v in samples[worst])


def _legs(chain, erosion):
    """Concatenate the shortest path of each leg into one polyline, at this erosion."""

    RT.CLEARANCE_M = erosion
    xs, ys, mask, _ = RT.driveable()
    poly, per_leg = [], []
    for a, b in zip(chain, chain[1:]):
        ia = RT.snap(xs, ys, mask, RT.WAYPOINTS[a])
        ib = RT.snap(xs, ys, mask, RT.WAYPOINTS[b])
        candidates = RT.diverse_routes(mask, ia, ib, xs, ys)
        if not candidates:
            return None, None
        lengths = [float(np.linalg.norm(np.diff(np.asarray(c), axis=0), axis=1).sum())
                   for c in candidates]
        seg = candidates[int(np.argmin(lengths))]
        per_leg.append(candidates)
        poly.extend(seg if not poly else seg[1:])
    return poly, per_leg


def _single(chain, erosion, index):
    """One named candidate of a single-leg task, so corridors can be compared directly."""

    RT.CLEARANCE_M = erosion
    xs, ys, mask, _ = RT.driveable()
    ia = RT.snap(xs, ys, mask, RT.WAYPOINTS[chain[0]])
    ib = RT.snap(xs, ys, mask, RT.WAYPOINTS[chain[1]])
    candidates = RT.diverse_routes(mask, ia, ib, xs, ys)
    if not candidates or index >= len(candidates):
        return None, None
    return candidates[index], candidates


def build(task: str, verbose: bool = False) -> dict:
    from unav_common.occlusion_geometry import scene_from_json
    spec = ROUTES[task]
    chain, which = spec["chain"], spec["candidate"]
    scene = scene_from_json(_driveable_geometry_json())
    start_xy, goal_xy = _task_endpoints(task)
    attempts = []
    for erosion in EROSION_LADDER:
        if which == "shortest":
            raw, candidates = _legs(chain, erosion)
        else:
            raw, candidates = _single(chain, erosion, int(which))
        if raw is None:
            attempts.append({"erosion_m": erosion, "verdict": "no candidate"})
            continue
        route = _simplify(raw, SIMPLIFY_TOLERANCE_M)
        clearance, where = _min_clearance(route, scene)
        start_off = float(np.linalg.norm(np.asarray(route[0], dtype=float) - start_xy))
        goal_off = float(np.linalg.norm(np.asarray(route[-1], dtype=float) - goal_xy))
        length = float(np.linalg.norm(
            np.diff(np.asarray(route, dtype=float), axis=0), axis=1).sum())
        record = {"erosion_m": erosion, "length_m": round(length, 3),
                  "min_clearance_m": round(clearance, 3),
                  "tightest_point": [round(v, 2) for v in where],
                  "start_offset_m": round(start_off, 3), "goal_offset_m": round(goal_off, 3)}
        if max(start_off, goal_off) > ENDPOINT_TOLERANCE_M:
            record["verdict"] = "endpoints moved off the task"
            attempts.append(record)
            if verbose:
                print(f"    erosion {erosion:.2f}: endpoints moved "
                      f"{max(start_off, goal_off):.2f} m -- not this task any more")
            continue
        record["verdict"] = ("clears" if clearance + 1e-9 >= REQUIRED_CLEARANCE_M else "too tight")
        attempts.append(record)
        if verbose:
            print(f"    erosion {erosion:.2f}: {length:.1f} m, measured clearance "
                  f"{clearance:.3f} m -> {record['verdict']}")
        if clearance + 1e-9 < REQUIRED_CLEARANCE_M:
            continue
        points, canonical = canonicalize_polyline_json(
            json.dumps([[float(x), float(y)] for x, y in route]))
        return {
            "task": task,
            "note": spec["note"],
            "waypoint_chain": chain,
            "candidate": which,
            "selection_rule": (
                "the lane geometry's own candidate for this corridor, simplified to the "
                "corners it turns at, at the smallest grid erosion whose polyline measures at "
                f"least the planner's own {REQUIRED_CLEARANCE_M:.2f} m clearance along its "
                "whole length and whose endpoints stay on the registered task. Route choice is "
                "not the treatment in the fusion comparison, so the rule is stated rather than "
                "optimised."),
            "generator": "experiments/warehouse_v2_sketches/route_tasks.py",
            "generator_sha256": sha256_file(
                REPO / "experiments/warehouse_v2_sketches/route_tasks.py"),
            "grid_erosion_m": erosion,
            "required_clearance_m": REQUIRED_CLEARANCE_M,
            "measured_min_clearance_m": round(clearance, 4),
            "tightest_point_xy": [round(v, 3) for v in where],
            "grid_resolution_m": RT.RES,
            "simplify_tolerance_m": SIMPLIFY_TOLERANCE_M,
            "n_points_before_simplify": len(raw),
            "length_m": round(length, 3),
            "n_points": len(points),
            "start_xy": [points[0][0], points[0][1]],
            "goal_xy": [points[-1][0], points[-1][1]],
            "erosion_ladder": attempts,
            "polyline_canonical_json": canonical,
            "polyline_sha256": route_sha256(canonical),
        }
    raise SystemExit(f"{task}: no candidate clears the required clearance:\n"
                     + json.dumps(attempts, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*", default=None,
                        help="which routes to freeze (default: all of them)")
    parser.add_argument("--verbose", action="store_true",
                        help="show the erosion ladder as it is measured")
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare against the frozen files; change nothing")
    args = parser.parse_args()
    tasks = args.tasks or list(ROUTES)
    unknown = [t for t in tasks if t not in ROUTES]
    if unknown:
        raise SystemExit(f"unknown route(s) {unknown}; known: {list(ROUTES)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failures = 0
    for task in tasks:
        path = OUT_DIR / f"{task}.json"
        if args.check:
            if not path.exists():
                print(f"{task}: MISSING {path}")
                failures += 1
                continue
            same = json.loads(path.read_text()).get("polyline_sha256") == \
                build(task)["polyline_sha256"]
            print(f"{task}: {'MATCHES' if same else 'DIFFERS FROM'} the generator")
            failures += 0 if same else 1
            continue
        print(task)
        record = build(task, verbose=args.verbose)
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"    {record['length_m']} m, {record['n_points']} vertices, erosion "
              f"{record['grid_erosion_m']} m, measured clearance "
              f"{record['measured_min_clearance_m']} m")
        print(f"    start {[round(v, 2) for v in record['start_xy']]}  "
              f"goal {[round(v, 2) for v in record['goal_xy']]}")
        print(f"    route sha256 {record['polyline_sha256']}")
        print(f"    file  sha256 {sha256_file(path)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
