#!/usr/bin/env python3
"""Fail-closed validation for the prospective commissioning campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src/experiments"), str(REPO / "src/unav_common")]


class ProtocolError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_and_hash(relative: str, expected: str, label: str) -> Path:
    path = (REPO / relative).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes the repository: {relative}") from exc
    if not path.is_file():
        raise ProtocolError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise ProtocolError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return path


def _angle_difference(first: float, second: float) -> float:
    return (float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi


def _route_heading(points: list[list[float]]) -> float:
    return math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])


def validate(protocol_path: Path = HERE / "campaign.yaml") -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "reference_controlled_commissioning.v1":
        raise ProtocolError("unexpected commissioning schema")
    if protocol.get("status") not in {
        "implemented_awaiting_simulator_pilot",
        "pilot_passed_frozen_before_collection",
        "sensor_model_frozen_audit_authorized",
    }:
        raise ProtocolError("protocol is neither pilot-ready, collection-frozen, nor audit-authorized")

    world = protocol["world"]
    launch = protocol["launch"]
    detector = protocol["detector"]
    sensor_gate = protocol["sensor_gate"]
    checked = {
        "world": _path_and_hash(world["path"], world["sha256"], "world"),
        "world_profiles": _path_and_hash(
            world["profiles_path"], world["profiles_sha256"], "world profiles"
        ),
        "launch": _path_and_hash(launch["path"], launch["sha256"], "launch"),
        "tasks": _path_and_hash(
            launch["tasks_path"], launch["tasks_sha256"], "commissioning tasks"
        ),
        "detector": _path_and_hash(
            detector["checkpoint_path"], detector["checkpoint_sha256"], "detector checkpoint"
        ),
        "sensor_gate": _path_and_hash(
            sensor_gate["config_path"], sensor_gate["config_sha256"], "sensor gate"
        ),
    }
    acquisition = protocol.get("acquisition", {})
    for label, entry in acquisition.items():
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise ProtocolError(f"acquisition source {label} lacks path/sha256")
        checked[f"acquisition_{label}"] = _path_and_hash(
            entry["path"], entry["sha256"], f"acquisition source {label}"
        )
    required_acquisition_sources = {
        "protocol_validator", "drive_runner", "campaign_runner", "table_builder",
        "reference_controller", "frame_identity_recorder", "launch_common",
        "batched_detector",
    }
    if set(acquisition) != required_acquisition_sources:
        missing = sorted(required_acquisition_sources - set(acquisition))
        extra = sorted(set(acquisition) - required_acquisition_sources)
        raise ProtocolError(
            f"acquisition source registry mismatch: missing={missing}, extra={extra}"
        )

    sys.path.insert(0, str(REPO / "src/reliability"))
    from reliability.observation_gates import UsableObservationGateConfig

    gate_config = UsableObservationGateConfig.from_yaml(str(checked["sensor_gate"]))
    gate_config.assert_belief_independent()
    if sensor_gate.get("role") != "deterministic_belief_independent_pre_estimator_gate":
        raise ProtocolError("sensor gate role must remain deterministic and pre-estimator")
    if sensor_gate.get("target") != "detector_hit_and_sensor_gate_pass":
        raise ProtocolError("q_sensor target must be detector hit and sensor-gate pass")
    if sensor_gate.get("includes_nis") is not False:
        raise ProtocolError("NIS must not be included in the sensor gate or q_sensor target")
    if not math.isclose(
        gate_config.confidence_threshold,
        float(detector["confidence_threshold"]),
        abs_tol=1e-12,
    ):
        raise ProtocolError("sensor-gate and deployed-detector confidence thresholds differ")

    controller = protocol["controller"]
    cruise = float(controller["nominal_cruise_mps"])
    if not math.isclose(cruise, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ProtocolError(f"nominal commissioning cruise must be exactly 1.0 m/s, got {cruise}")
    if any(math.isclose(cruise, float(value), abs_tol=1e-12)
           for value in controller.get("forbidden_nominal_speeds_mps", [])):
        raise ProtocolError("nominal commissioning cruise is on the forbidden-speed list")
    if str(detector["device"]) != "0":
        raise ProtocolError("evidence collection must use YOLO device '0'")
    if not math.isclose(float(detector["camera_rate_hz"]), 5.0, abs_tol=1e-12):
        raise ProtocolError("camera rate must remain 5 Hz")
    if detector["camera_ids"] != [
        "camera_A", "camera_B", "camera_C", "camera_D", "camera_E"
    ]:
        raise ProtocolError("camera registry must contain the five warehouse_v2 cameras in order")

    tasks_payload = yaml.safe_load(checked["tasks"].read_text(encoding="utf-8"))
    task_list = tasks_payload.get("tasks", {}).get(world["name"], [])
    tasks = {entry["name"]: entry for entry in task_list}
    if len(tasks) != len(task_list):
        raise ProtocolError("task names are not unique")

    from experiments.core.world_profiles import load_world_profiles, serialize_driveable_geometry_from_profile
    from unav_common.occlusion_geometry import scene_from_json
    from unav_common.preselected_route import sample_polyline
    from unav_common.rectangular_footprint import RectangularFootprint

    profiles = load_world_profiles(str(checked["world_profiles"]))
    profile = profiles["worlds"][world["name"]]
    scene = scene_from_json(serialize_driveable_geometry_from_profile(profile))
    geometry = protocol["geometry"]
    body = RectangularFootprint(
        scene.prisms,
        length=float(geometry["robot_length_m"]),
        width=float(geometry["robot_width_m"]),
        keep_in=True,
    )
    clearance_by_route: dict[str, float] = {}
    anchor_clearance_by_route: dict[str, float] = {}
    route_variants: set[tuple[str, str]] = set()
    for route_id, route in protocol["routes"].items():
        points = [[float(v) for v in point] for point in route["points"]]
        if len(points) != 2:
            raise ProtocolError(
                f"{route_id}: v1 controller accepts one straight transect with two endpoints"
            )
        if math.dist(points[0], points[1]) < 2.0:
            raise ProtocolError(f"{route_id}: route is too short to commission moving observations")
        heading = _route_heading(points)
        samples = sample_polyline(points, maximum_step_m=float(geometry["route_sample_step_m"]))
        clearances = [body.clearance([float(x), float(y), heading]) for x, y in samples]
        minimum = float(min(clearances))
        required = float(geometry["minimum_static_body_clearance_m"])
        if minimum + 1e-12 < required:
            raise ProtocolError(
                f"{route_id}: body clearance {minimum:.4f} m is below {required:.4f} m"
            )
        clearance_by_route[route_id] = minimum
        anchor_cfg = controller["stationary_anchor"]
        anchor_clearances = []
        for endpoint_index, base_heading in ((0, heading), (1, heading + math.pi)):
            for offset_deg in anchor_cfg["heading_offsets_deg"]:
                anchor_clearances.append(body.clearance([
                    points[endpoint_index][0],
                    points[endpoint_index][1],
                    base_heading + math.radians(float(offset_deg)),
                ]))
        anchor_minimum = float(min(anchor_clearances))
        if anchor_minimum + 1e-12 < required:
            raise ProtocolError(
                f"{route_id}: stationary-anchor body clearance {anchor_minimum:.4f} m "
                f"is below {required:.4f} m"
            )
        anchor_clearance_by_route[route_id] = anchor_minimum
        for direction, task_key in (("forward", "forward_task"), ("reverse", "reverse_task")):
            task_name = route[task_key]
            if task_name not in tasks:
                raise ProtocolError(f"{route_id}: missing task {task_name}")
            directed = points if direction == "forward" else list(reversed(points))
            task = tasks[task_name]
            start = task["start"]
            goal = task["goal"]
            if math.dist([float(start["x"]), float(start["y"])], directed[0]) > 1e-9:
                raise ProtocolError(f"{task_name}: task start does not match route")
            if math.dist([float(goal["x"]), float(goal["y"])], directed[-1]) > 1e-9:
                raise ProtocolError(f"{task_name}: task goal does not match route")
            expected_yaw = _route_heading(directed)
            if abs(_angle_difference(float(start["yaw"]), expected_yaw)) > 1e-9:
                raise ProtocolError(f"{task_name}: start yaw does not face along route")
            route_variants.add((route_id, direction))

    drives = protocol["drives"]
    if len(drives) != 24:
        raise ProtocolError(f"expected 24 drives, got {len(drives)}")
    ids = [str(drive["id"]) for drive in drives]
    orders = [int(drive["order"]) for drive in drives]
    seeds = [int(drive["seed"]) for drive in drives]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ProtocolError("drive ids and seeds must each be globally unique")
    if sorted(orders) != list(range(1, 25)):
        raise ProtocolError("drive order must be exactly 1..24")
    expected_partitions = {"fit", "development", "audit"}
    if set(protocol["partitions"]) != expected_partitions:
        raise ProtocolError("partition registry must be fit, development, audit")
    counts: dict[str, int] = {}
    for partition in sorted(expected_partitions):
        members = [drive for drive in drives if drive["partition"] == partition]
        variants = {(drive["route"], drive["direction"]) for drive in members}
        if len(members) != 8 or variants != route_variants:
            raise ProtocolError(
                f"{partition}: must contain each of the eight route-direction variants once"
            )
        counts[partition] = len(members)
    for drive in drives:
        if drive["partition"] not in expected_partitions:
            raise ProtocolError(f"{drive['id']}: unknown partition")
        if (drive["route"], drive["direction"]) not in route_variants:
            raise ProtocolError(f"{drive['id']}: unknown route-direction variant")
        expected_prefix = str(drive["partition"]) + "_"
        if not str(drive["id"]).startswith(expected_prefix):
            raise ProtocolError(f"{drive['id']}: id does not expose its locked partition")

    if protocol["analysis_order"][-1] != "open_audit_partition_once":
        raise ProtocolError("audit partition must remain the final analysis step")
    online = protocol["online_covariance"]
    if online["primary_method"] != "one_robot_filter_consuming_each_corrected_camera_frame_once":
        raise ProtocolError("the primary estimator must consume each corrected frame once")
    if online["double_cascade_role"] != "diagnostic_comparator_only":
        raise ProtocolError("the double cascade may only be a diagnostic comparator")
    if online["prohibited_handoff"] != "accumulated_first_filter_posterior_as_fresh_independent_measurement":
        raise ProtocolError("double-cascade no-double-counting rule is missing")

    return {
        "schema": "reference_controlled_commissioning_preflight.v1",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "status": protocol["status"],
        "nominal_cruise_mps": cruise,
        "partition_counts": counts,
        "drive_count": len(drives),
        "route_minimum_static_body_clearance_m": clearance_by_route,
        "stationary_anchor_minimum_static_body_clearance_m": anchor_clearance_by_route,
        "artifact_sha256_verified": {key: sha256_file(path) for key, path in checked.items()},
        "sensor_gate_id": gate_config.gate_id,
        "sensor_gate_config_hash": gate_config.config_hash(),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "campaign.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = validate(args.protocol)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else (
        f"PASS: {report['drive_count']} drives at {report['nominal_cruise_mps']:.1f} m/s; "
        f"partitions={report['partition_counts']}"
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
