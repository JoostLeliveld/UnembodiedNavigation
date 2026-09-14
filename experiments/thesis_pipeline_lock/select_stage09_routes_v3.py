#!/usr/bin/env python3
"""Select Stage-09 routes with a heading-aware commissioned-support gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis09_mpl")
os.environ.setdefault(
    "ROS_LOG_DIR", str(Path(tempfile.gettempdir()) / "roslog_thesis09_route_selection")
)

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / path)
    for path in (
        "src/planning",
        "src/reliability",
        "src/unav_common",
        "src/experiments",
        "experiments/icra_commissioning",
    )
]

from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402
from reliability.commissioned_availability import CommissionedAvailabilityModel  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402
from unav_common.lane_graph_routes import generate_route_seeds  # noqa: E402
from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy  # noqa: E402
from unav_common.preselected_route import (  # noqa: E402
    canonicalize_polyline_json,
    route_sha256,
    sample_polyline,
    sha256_file,
    validate_preselected_route,
)
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402


def _load_route_probe():
    path = REPO / "experiments/icra_commissioning/network_route_probe.py"
    spec = importlib.util.spec_from_file_location("stage09_network_route_probe_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load route-probe implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROUTE_PROBE = _load_route_probe()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    value = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _verify(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _dedupe(points) -> list[list[float]]:
    result: list[list[float]] = []
    for point in points:
        xy = [float(point[0]), float(point[1])]
        if not result or math.hypot(xy[0] - result[-1][0], xy[1] - result[-1][1]) > 1e-12:
            result.append(xy)
    return result


def _inside_union(scene, points, step_m: float) -> bool:
    samples = sample_polyline(points, maximum_step_m=step_m)
    signed = np.asarray(
        signed_distance_to_union_xy(scene.prisms, samples, keep_in=True), dtype=float
    )
    return bool(signed.size and np.isfinite(signed).all() and np.max(signed) <= 1e-9)


def _candidate_routes(driveable_json: str, start_xy, goal_xy, step_m: float):
    """Add lane-centre and endpoint-turn-safe variants to the lane graph."""
    scene = scene_from_json(driveable_json)
    base = list(generate_route_seeds(driveable_json, start_xy, goal_xy))
    candidates = list(base)
    vertical_centres = sorted(
        0.5 * float(prism.xmin + prism.xmax)
        for prism in scene.prisms
        if float(prism.ymax - prism.ymin) > float(prism.xmax - prism.xmin)
    )

    for centre_x in vertical_centres:
        route = _dedupe(
            [start_xy, [centre_x, float(start_xy[1])],
             [centre_x, float(goal_xy[1])], goal_xy]
        )
        if len(route) >= 2 and _inside_union(scene, route, step_m):
            candidates.append({
                "name": f"vertical_lane_centre_{centre_x:+.3f}",
                "waypoints": route[1:],
            })

    # A route whose first or last leg is vertical can demand an in-place turn
    # beside a pallet. Shift that endpoint leg to every valid vertical-lane
    # centre. The later swept-footprint check remains authoritative.
    variants = []
    for candidate in base:
        route = _dedupe([start_xy, *candidate["waypoints"]])
        if len(route) >= 3 and abs(route[1][0] - route[0][0]) < 1e-9:
            for centre_x in vertical_centres:
                shifted = _dedupe([
                    route[0], [centre_x, route[0][1]], [centre_x, route[1][1]],
                    *route[2:],
                ])
                if _inside_union(scene, shifted, step_m):
                    variants.append({
                        "name": f"{candidate['name']}__start_lane_{centre_x:+.3f}",
                        "waypoints": shifted[1:],
                    })
        if len(route) >= 3 and abs(route[-1][0] - route[-2][0]) < 1e-9:
            for centre_x in vertical_centres:
                shifted = _dedupe([
                    *route[:-2], [centre_x, route[-2][1]],
                    [centre_x, route[-1][1]], route[-1],
                ])
                if _inside_union(scene, shifted, step_m):
                    variants.append({
                        "name": f"{candidate['name']}__goal_lane_{centre_x:+.3f}",
                        "waypoints": shifted[1:],
                    })
    candidates.extend(variants)

    unique = []
    seen = set()
    for candidate in candidates:
        route = _dedupe([start_xy, *candidate["waypoints"]])
        key = json.dumps(route, separators=(",", ":"), allow_nan=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append({"name": str(candidate["name"]), "waypoints": route[1:]})
    return scene, unique


def _angle_sweep(first: float, second: float, maximum_step_rad: float):
    delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
    count = max(1, int(math.ceil(abs(delta) / maximum_step_rad)))
    return [first + delta * fraction for fraction in np.linspace(0.0, 1.0, count + 1)]


def _minimum_static_body_clearance(
    scene, route, *, start_yaw: float, length_m: float, width_m: float, step_m: float
) -> float:
    body = RectangularFootprint(scene.prisms, length_m, width_m, keep_in=True)
    headings = [
        math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(route, route[1:])
    ]
    clearances = []
    for (start, end), heading in zip(zip(route, route[1:]), headings):
        clearances.extend(
            body.clearance([x, y, heading])
            for x, y in sample_polyline([start, end], maximum_step_m=step_m)
        )
    rotations = [(route[0], start_yaw, headings[0])]
    rotations.extend(
        (route[index], headings[index - 1], headings[index])
        for index in range(1, len(route) - 1)
    )
    for point, before, after in rotations:
        clearances.extend(
            body.clearance([point[0], point[1], yaw])
            for yaw in _angle_sweep(before, after, math.radians(5.0))
        )
    return float(min(clearances)) if clearances else -math.inf


def _availability_support(
    route, *, start_yaw: float, q_model, cameras, threshold: float,
    step_m: float, heading_step_rad: float,
) -> dict[str, Any]:
    samples: list[tuple[float, float, float, float]] = []
    headings = [
        math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(route, route[1:])
    ]
    distance = 0.0
    for (start, end), yaw in zip(zip(route, route[1:]), headings):
        segment = sample_polyline([start, end], maximum_step_m=step_m)
        for x, y in segment:
            samples.append((distance, float(x), float(y), yaw))
        distance += math.dist(start, end)
    rotations = [(route[0], start_yaw, headings[0])]
    rotations.extend(
        (route[index], headings[index - 1], headings[index])
        for index in range(1, len(route) - 1)
    )
    for point, before, after in rotations:
        samples.extend(
            (math.nan, float(point[0]), float(point[1]), yaw)
            for yaw in _angle_sweep(before, after, heading_step_rad)
        )

    q_max = []
    for _, x, y, yaw in samples:
        q_max.append(max(
            q_model.probability(camera_id, x, y, yaw, cameras[camera_id])
            for camera_id in q_model.camera_ids
        ))
    supported = np.asarray(q_max, dtype=float) >= threshold
    return {
        "eligible": bool(supported.size and np.all(supported)),
        "minimum_max_per_camera_q": float(np.min(q_max)) if q_max else 0.0,
        "mean_max_per_camera_q": float(np.mean(q_max)) if q_max else 0.0,
        "supported_fraction": float(np.mean(supported)) if supported.size else 0.0,
        "sample_count": int(len(samples)),
    }


def run(args: argparse.Namespace) -> None:
    protocol = _read_json(args.protocol)
    if protocol.get("status") != "frozen_before_navigation_execution":
        raise RuntimeError("Stage-09 route-selection protocol is not frozen")
    if args.output.exists():
        raise RuntimeError("route-selection outputs are immutable; choose a new directory")
    args.output.mkdir(parents=True)

    config = (REPO / protocol["route_selection_config"]).resolve()
    tasks_path = (REPO / protocol["tasks_yaml"]).resolve()
    world = (REPO / protocol["world"]["path"]).resolve()
    availability_model_path = (REPO / protocol["availability_support_gate"]["model_path"]).resolve()
    master_index = (REPO / protocol["availability_support_gate"]["master_capture_index"]).resolve()
    _verify(config, protocol["route_selection_config_sha256"], "route-selection config")
    _verify(tasks_path, protocol["tasks_yaml_sha256"], "task registry")
    _verify(Path(__file__).resolve(), protocol["selector_sha256"], "route selector")
    _verify(
        (REPO / protocol["route_probe"]["path"]).resolve(),
        protocol["route_probe"]["sha256"],
        "route probe",
    )
    _verify(world, protocol["world"]["sha256"], "warehouse world")
    _verify(
        availability_model_path,
        protocol["availability_support_gate"]["model_sha256"],
        "availability model",
    )
    _verify(
        master_index,
        protocol["availability_support_gate"]["master_capture_index_sha256"],
        "master capture index",
    )
    for arm, field in protocol["availability_fields"].items():
        _verify((REPO / field["path"]).resolve(), field["sha256"], f"{arm} field")

    q_model = CommissionedAvailabilityModel(
        availability_model_path,
        world,
        expected_sha256=protocol["availability_support_gate"]["model_sha256"],
    )
    with master_index.open(newline="", encoding="utf-8") as handle:
        capture_rows = list(csv.DictReader(handle))
    cameras = {
        camera_id: camera_model_from_world(
            world,
            include_name=next(
                row["camera_model"] for row in capture_rows
                if row["camera_id"] == camera_id
            ),
        )
        for camera_id in q_model.camera_ids
    }
    initial_covariance = np.diag([
        float(protocol["nominal_route_prior"]["xy_std_m"]) ** 2,
        float(protocol["nominal_route_prior"]["xy_std_m"]) ** 2,
        math.radians(float(protocol["nominal_route_prior"]["yaw_std_deg"])) ** 2,
    ])
    route_gate = protocol["route_gate"]
    support_gate = protocol["availability_support_gate"]
    selected: dict[str, dict[str, Any]] = {}

    for task_name in protocol["task_order"]:
        selected[task_name] = {}
        for arm in protocol["arm_order"]:
            captured = ROUTE_PROBE.resolve(
                config, task_name, arm, int(protocol["selection_seed"])
            )
            settings, resolved = captured["settings"], captured["resolved"]
            state = np.asarray(
                [resolved["spawn"][axis] for axis in ("x", "y", "yaw")], dtype=float
            )
            goal = np.asarray([resolved["goal_x"], resolved["goal_y"]], dtype=float)
            scene, candidates = _candidate_routes(
                settings["driveable_geometry_json"], state[:2], goal,
                float(route_gate["sample_step_m"]),
            )
            if not candidates:
                raise RuntimeError(f"no geometry-only route candidates for {task_name}")

            settings["optimizer_initial_routes_json"] = json.dumps(candidates)
            planner = UnicyclePlannerBase(**settings)
            records: list[dict[str, Any]] = []
            terminal_tolerance = float(settings["optimizer_terminal_goal_tolerance_m"])
            for candidate in candidates:
                route_points = _dedupe([state[:2].tolist(), *candidate["waypoints"]])
                static_clearance = _minimum_static_body_clearance(
                    scene,
                    route_points,
                    start_yaw=float(state[2]),
                    length_m=float(route_gate["robot_length_m"]),
                    width_m=float(route_gate["robot_width_m"]),
                    step_m=float(route_gate["sample_step_m"]),
                )
                if static_clearance + 1e-9 < float(route_gate["minimum_static_body_clearance_m"]):
                    continue
                support = _availability_support(
                    route_points,
                    start_yaw=float(state[2]),
                    q_model=q_model,
                    cameras=cameras,
                    threshold=float(support_gate["minimum_single_camera_probability"]),
                    step_m=float(support_gate["sample_step_m"]),
                    heading_step_rad=math.radians(float(support_gate["heading_step_deg"])),
                )
                controls = planner._controls_for_waypoints(state, candidate["waypoints"])
                result = planner.evaluate_rollout_controls(
                    state, initial_covariance, goal, controls
                )
                _, canonical = canonicalize_polyline_json(
                    json.dumps(route_points, separators=(",", ":"), allow_nan=False)
                )
                records.append({
                    "name": candidate["name"],
                    "route_json": canonical,
                    "route_sha256": route_sha256(canonical),
                    "route_length_m": float(
                        np.linalg.norm(np.diff(np.asarray(route_points), axis=0), axis=1).sum()
                    ),
                    "minimum_static_body_clearance_m": static_clearance,
                    "availability_support": support,
                    "rollout_valid": bool(result["rollout_valid"]),
                    "invalid_reason": str(result["invalid_reason"]),
                    "terminal_goal_feasible": bool(
                        result["rollout_valid"]
                        and float(result["terminal_goal_distance_pred"]) <= terminal_tolerance
                    ),
                    **{
                        key: _serializable(result[key])
                        for key in (
                            "total_cost", "risk_cost", "ambiguity_cost", "control_cost",
                            "obstacle_cost", "terminal_goal_distance_pred",
                            "terminal_goal_progress_m", "min_predicted_obstacle_distance_m",
                            "mean_p_vis_plan", "mean_p_vis_plan_eff",
                            "mean_r_plan_u_std", "mean_r_plan_v_std",
                        )
                    },
                })

            eligible = [
                record for record in records
                if record["rollout_valid"] and record["terminal_goal_feasible"]
                and (arm != "P1" or record["availability_support"]["eligible"])
            ]
            winner = min(eligible, key=lambda record: (float(record["total_cost"]), record["name"])) if eligible else None
            source_payload = {
                "schema_version": 3,
                "kind": "stage09_frozen_route_selection",
                "task": task_name,
                "arm": arm,
                "selection_status": "selected" if winner is not None else "commissioning_refused",
                "selection_rule": protocol["selection_rule"],
                "selected_candidate": winner,
                "candidate_results": records,
                "nominal_initial_state": state.tolist(),
                "nominal_initial_covariance": initial_covariance.tolist(),
                "goal_xy": goal.tolist(),
                "availability_field": protocol["availability_fields"][arm],
                "availability_support_gate": support_gate,
                "route_selection_config_sha256": protocol["route_selection_config_sha256"],
                "tasks_yaml_sha256": protocol["tasks_yaml_sha256"],
                "selector_sha256": protocol["selector_sha256"],
            }
            source_path = args.output / f"{task_name}__{arm}.json"
            _atomic_json(source_path, source_payload)
            source_sha = sha256_file(source_path)
            if winner is None:
                selected[task_name][arm] = {
                    "status": "commissioning_refused",
                    "reason": "no continuously commissioned route passed all geometry and rollout gates",
                    "preselected_route_source_path": str(source_path.relative_to(REPO)),
                    "preselected_route_source_sha256": source_sha,
                }
                print(f"{task_name}/{arm}: COMMISSIONING REFUSED", flush=True)
                continue
            validated = validate_preselected_route(
                winner["route_json"],
                winner["route_sha256"],
                start_xy=state[:2],
                goal_xy=goal,
                driveable_geometry_json=settings["driveable_geometry_json"],
                declared_clearance_m=float(route_gate["centerline_clearance_m"]),
                source_path=source_path,
                expected_source_sha256=source_sha,
                endpoint_tolerance_m=float(route_gate["endpoint_tolerance_m"]),
                sample_step_m=float(route_gate["sample_step_m"]),
            )
            selected[task_name][arm] = {
                "status": "selected",
                "candidate": winner["name"],
                "preselected_route_json": winner["route_json"],
                "preselected_route_sha256": winner["route_sha256"],
                "preselected_route_source_path": str(source_path.relative_to(REPO)),
                "preselected_route_source_sha256": source_sha,
                "route_length_m": validated.length_m,
                "minimum_centerline_clearance_m": validated.minimum_driveable_clearance_m,
                "minimum_static_body_clearance_m": winner["minimum_static_body_clearance_m"],
                "availability_support": winner["availability_support"],
                "total_cost": winner["total_cost"],
                "risk_cost": winner["risk_cost"],
                "ambiguity_cost": winner["ambiguity_cost"],
                "mean_usable_detection_probability": winner["mean_p_vis_plan"],
            }
            print(
                f"{task_name}/{arm}: {winner['name']} L={validated.length_m:.2f} m "
                f"min-q={winner['availability_support']['minimum_max_per_camera_q']:.3f}",
                flush=True,
            )

    manifest = {
        "schema_version": 3,
        "kind": "stage09_route_selection_manifest",
        "status": "complete",
        "protocol_path": str(args.protocol.relative_to(REPO)),
        "protocol_sha256": sha256_file(args.protocol),
        "selected": selected,
        "output_files": {},
    }
    for path in sorted(args.output.glob("*.json")):
        manifest["output_files"][path.name] = sha256_file(path)
    identity = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["selection_digest"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    _atomic_json(args.output / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
