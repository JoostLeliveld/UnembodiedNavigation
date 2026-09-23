#!/usr/bin/env python3
"""Freeze one accepted route-selection probe into a hash-bound execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / path) for path in ("src/planning", "src/unav_common")]

from planning.planners.base_planner import extract_waypoints  # noqa: E402
from unav_common.occlusion_geometry import scene_from_json  # noqa: E402
from unav_common.preselected_route import (  # noqa: E402
    canonicalize_polyline_json,
    route_sha256,
    sample_polyline,
    sha256_file,
    validate_preselected_route,
)
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def angle_sweep(start: float, end: float, step: float = math.radians(5.0)) -> list[float]:
    delta = (float(end) - float(start) + math.pi) % (2.0 * math.pi) - math.pi
    count = max(1, int(math.ceil(abs(delta) / step)))
    return np.linspace(float(start), float(start) + delta, count + 1).tolist()


def body_clearance(settings: dict, route: list[list[float]], start_yaw: float) -> tuple[float, dict]:
    physical_scene = scene_from_json(settings["collision_geometry_json"])
    driveable_scene = scene_from_json(settings["driveable_geometry_json"])
    physical = RectangularFootprint(physical_scene.prisms, 0.8, 0.55, keep_in=False)
    driveable = RectangularFootprint(driveable_scene.prisms, 0.8, 0.55, keep_in=True)
    headings = [
        math.atan2(b[1] - a[1], b[0] - a[0])
        for a, b in zip(route, route[1:])
    ]
    best = {"clearance_m": math.inf, "kind": "", "index": -1, "pose": []}

    def check(model, kind: str, pose, index: int) -> None:
        value = float(model.clearance(pose))
        if value < best["clearance_m"]:
            best.update(clearance_m=value, kind=kind, index=index,
                        pose=[float(item) for item in pose])

    for index, (start, end) in enumerate(zip(route, route[1:])):
        for x, y in sample_polyline([start, end], maximum_step_m=0.04):
            pose = [x, y, headings[index]]
            check(physical, "physical_segment", pose, index)
            check(driveable, "driveable_segment", pose, index)
    rotations = [(0, route[0], start_yaw, headings[0])]
    rotations.extend(
        (index, route[index], headings[index - 1], headings[index])
        for index in range(1, len(route) - 1)
    )
    for index, point, before, after in rotations:
        for yaw in angle_sweep(before, after):
            pose = [point[0], point[1], yaw]
            check(physical, "physical_rotation", pose, index)
            check(driveable, "driveable_rotation", pose, index)
    return float(best["clearance_m"]), best


def render_map(path: Path, route: np.ndarray, settings: dict, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 6.8), constrained_layout=True)
    for prism in scene_from_json(settings["collision_geometry_json"]).prisms:
        if prism.zmin <= 0.55 and prism.zmax >= 0.0:
            axis.add_patch(Rectangle(
                (prism.xmin, prism.ymin), prism.xmax - prism.xmin,
                prism.ymax - prism.ymin, facecolor="#d6dadd", edgecolor="#9aa3a8",
                linewidth=0.35, zorder=0,
            ))
    axis.plot(route[:, 0], route[:, 1], color="#1565c0", linewidth=1.8,
              label="Frozen selected route")
    axis.plot(route[0, 0], route[0, 1], "o", color="#111111", markersize=5,
              label="Start")
    axis.plot(route[-1, 0], route[-1, 1], "*", color="#d32f2f", markersize=9,
              label="Goal")
    axis.set(xlim=(-12.3, 12.3), ylim=(-10.3, 10.3), aspect="equal",
             xlabel="World x [m]", ylabel="World y [m]", title=title)
    axis.grid(alpha=0.15)
    axis.legend(loc="best", fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("frozen route outputs are immutable; choose a new directory")
    output.mkdir(parents=True)
    record = json.loads(probe.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = record["result"]
    accepted_backends = {"casadi", "lane_graph_expected_belief"}
    if result.get("backend") not in accepted_backends or result.get("rollout_valid") is not True:
        raise RuntimeError("route probe is not a valid accepted-planner rollout")
    tolerance = float(record["settings"]["optimizer_terminal_goal_tolerance_m"])
    iteration_cap = (
        int(result.get("optimizer_status", -1)) == 1
        and int(result.get("optimizer_nit", -1)) == int(record["settings"]["optimizer_maxiter"])
        and "ITERATIONS REACHED LIMIT" in str(result.get("optimizer_message", ""))
    )
    selected_source = str(result.get("selected_source", ""))
    route_ranked_fallback = (
        selected_source.startswith(("seed:route:", "solver:route:"))
        and bool(record.get("seed_results"))
        and bool(record.get("attempts"))
    )
    # L-BFGS-B can report status 2 after a failed line search even though its
    # retained iterate is finite, goal-reaching, and passes the independent
    # exact-body rollout validator.  Do not confuse that numerical stopping
    # code with route invalidity; the goal and clearance gates below remain
    # mandatory.
    valid_abnormal_termination = (
        int(result.get("optimizer_status", -1)) == 2
        and result.get("rollout_valid") is True
    )
    if (result.get("optimizer_success") is not True and not iteration_cap
            and not route_ranked_fallback and not valid_abnormal_termination):
        raise RuntimeError("optimizer termination is neither convergence nor the declared iteration cap")

    states = np.asarray(result["states"], dtype=float)
    goal = np.asarray(record["goal"], dtype=float)
    goal_distance = np.linalg.norm(states[:, :2] - goal[None, :], axis=1)
    goal_entries = np.flatnonzero(goal_distance <= tolerance + 1.0e-12)
    if goal_entries.size == 0:
        raise RuntimeError(
            "CasADi route never enters the declared terminal tolerance; "
            f"closest approach is {float(np.min(goal_distance)):.6f} m"
        )
    # The global horizon is fixed.  Once the route enters the operational goal
    # region, any later near-zero-control tail is not part of the navigation
    # route and must not pull the frozen endpoint back outside the tolerance.
    goal_entry_index = int(goal_entries[0])
    route_states = states[:goal_entry_index + 1]
    points = [states[0, :2].tolist(), *[
        list(point) for point in extract_waypoints(
            route_states, spacing_m=0.2, include_goal=True)
    ]]
    if np.linalg.norm(np.asarray(points[-1]) - goal) > 1.0e-12:
        points.append(goal.tolist())
    deduplicated = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(np.asarray(point) - np.asarray(deduplicated[-1])) > 1.0e-12:
            deduplicated.append(point)
    _, canonical = canonicalize_polyline_json(json.dumps(deduplicated))
    route_hash = route_sha256(canonical)
    clearance, limiting = body_clearance(
        record["settings"], deduplicated, float(states[0, 2]))
    required = float(protocol["common_feasibility"]["minimum_static_body_clearance_m"])
    if clearance + 1.0e-9 < required:
        raise RuntimeError(
            f"execution polyline has {clearance:.6f} m body clearance; requires {required:.6f} m"
        )
    length = float(np.linalg.norm(np.diff(np.asarray(deduplicated), axis=0), axis=1).sum())
    source = {
        "schema_version": 1,
        "kind": "single_cell_expected_belief_route_selection",
        "task": args.task,
        "arm": args.arm,
        "route_json": canonical,
        "route_sha256": route_hash,
        "route_length_m": length,
        "minimum_static_body_clearance_m": clearance,
        "limiting_clearance": limiting,
        "terminal_goal_distance_pred_m": float(result["terminal_goal_distance_pred"]),
        "first_goal_region_state_index": goal_entry_index,
        "first_goal_region_distance_m": float(goal_distance[goal_entry_index]),
        "closest_goal_distance_m": float(np.min(goal_distance)),
        "fixed_horizon_tail_states_removed": int(len(states) - len(route_states)),
        "backend": result["backend"],
        "selected_source": result["selected_source"],
        "optimizer_success": bool(result["optimizer_success"]),
        "optimizer_status": int(result["optimizer_status"]),
        "optimizer_message": result["optimizer_message"],
        "optimizer_nit": int(result["optimizer_nit"]),
        "optimizer_nfev": int(result["optimizer_nfev"]),
        "termination_acceptance": (
            "optimizer_converged" if result.get("optimizer_success") is True
            else "optimizer_iteration_cap" if iteration_cap
            else "valid_goal_reaching_abnormal_line_search" if valid_abnormal_termination
            else "valid_goal_reaching_route_ranked_fallback"
        ),
        "solve_time_s": float(result["solve_time_s"]),
        "total_cost": float(result["total_cost"]),
        "probe_path": str(probe.relative_to(REPO)),
        "probe_sha256": sha256_file(probe),
        "protocol_path": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": sha256_file(protocol_path),
        "selector_path": str(Path(__file__).resolve().relative_to(REPO)),
    }
    source_path = output / f"{args.task}__{args.arm}.json"
    atomic_json(source_path, source)
    source_hash = sha256_file(source_path)
    validated = validate_preselected_route(
        canonical, route_hash, start_xy=states[0, :2], goal_xy=goal,
        driveable_geometry_json=record["settings"]["driveable_geometry_json"],
        declared_clearance_m=float(protocol["common_feasibility"]["centerline_clearance_m"]),
        source_path=source_path, expected_source_sha256=source_hash,
        endpoint_tolerance_m=0.0, sample_step_m=0.04,
    )
    selected = {
        args.task: {
            args.arm: {
                "status": "selected",
                "candidate": result["selected_source"],
                "preselected_route_json": canonical,
                "preselected_route_sha256": route_hash,
                "preselected_route_source_path": str(source_path.relative_to(REPO)),
                "preselected_route_source_sha256": source_hash,
                "route_length_m": validated.length_m,
                "minimum_centerline_clearance_m": validated.minimum_driveable_clearance_m,
                "minimum_static_body_clearance_m": clearance,
                "total_cost": float(result["total_cost"]),
            }
        }
    }
    manifest = {
        "schema_version": 1,
        "kind": "single_cell_expected_belief_route_selection_manifest",
        "status": "complete",
        "protocol_path": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": sha256_file(protocol_path),
        "selected": selected,
        "output_files": {source_path.name: source_hash},
    }
    manifest["selection_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output / "manifest.json", manifest)
    render_map(
        output / f"{args.task}__{args.arm}_route",
        np.asarray(deduplicated), record["settings"],
        f"Accepted route: {args.task} / {args.arm}",
    )
    print(json.dumps({
        "manifest": str((output / "manifest.json").relative_to(REPO)),
        "route_sha256": route_hash,
        "route_length_m": length,
        "minimum_static_body_clearance_m": clearance,
        "limiting_clearance": limiting,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
