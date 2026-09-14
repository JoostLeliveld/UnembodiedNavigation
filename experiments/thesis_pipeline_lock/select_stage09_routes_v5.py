#!/usr/bin/env python3
"""Select routes from one arm-independent feasible candidate set.

Availability and covariance forecasts may change a candidate's objective value.
They never change candidate generation or eligibility.  Static body clearance,
the bounded-unicycle swept-footprint rollout and terminal-goal reachability are
computed for every arm and must agree before any route is selected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / path)
    for path in (
        "src/planning", "src/reliability", "src/unav_common", "src/experiments",
        "src/perception", "src/sim", "experiments/icra_commissioning",
    )
]


def _load_v4():
    path = REPO / "experiments/thesis_pipeline_lock/select_stage09_routes_v4.py"
    spec = importlib.util.spec_from_file_location("stage09_route_selector_v4_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v4 selector from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V4 = _load_v4()
CORE = V4.BASE


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _shared_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _shared_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shared_value(item) for item in value]
    return value


def _assert_arm_invariants(task_name: str, resolved_by_arm: dict, keys: list[str]) -> None:
    arms = list(resolved_by_arm)
    reference = resolved_by_arm[arms[0]]
    reference_pose = (
        [reference["resolved"]["spawn"][axis] for axis in ("x", "y", "yaw")],
        [reference["resolved"]["goal_x"], reference["resolved"]["goal_y"]],
    )
    reference_settings = {
        key: _shared_value(reference["settings"].get(key)) for key in keys
    }
    for arm in arms[1:]:
        current = resolved_by_arm[arm]
        current_pose = (
            [current["resolved"]["spawn"][axis] for axis in ("x", "y", "yaw")],
            [current["resolved"]["goal_x"], current["resolved"]["goal_y"]],
        )
        current_settings = {
            key: _shared_value(current["settings"].get(key)) for key in keys
        }
        if current_pose != reference_pose:
            raise RuntimeError(f"{task_name}: start or goal differs between arms")
        if current_settings != reference_settings:
            changed = [key for key in keys if current_settings[key] != reference_settings[key]]
            raise RuntimeError(
                f"{task_name}: feasibility setting differs between arms: {changed}"
            )


def run(args: argparse.Namespace) -> None:
    protocol = CORE._read_json(args.protocol)
    if protocol.get("status") != "frozen_before_navigation_execution":
        raise RuntimeError("Stage-09 v5 route-selection protocol is not frozen")
    if "availability_support_gate" in protocol:
        raise RuntimeError("availability must not be a route-eligibility gate")
    try:
        args.output.relative_to(REPO)
    except ValueError as exc:
        raise RuntimeError("route-selection output must be inside the repository") from exc
    if args.output.exists():
        raise RuntimeError("route-selection outputs are immutable; choose a new directory")
    args.output.mkdir(parents=True)

    config_path = (REPO / protocol["route_selection_config"]).resolve()
    tasks_path = (REPO / protocol["tasks_yaml"]).resolve()
    CORE._verify(config_path, protocol["route_selection_config_sha256"], "route config")
    CORE._verify(tasks_path, protocol["tasks_yaml_sha256"], "task registry")
    CORE._verify(Path(__file__).resolve(), protocol["selector_sha256"], "route selector")
    CORE._verify(
        (REPO / protocol["base_selector"]["path"]).resolve(),
        protocol["base_selector"]["sha256"], "v4 candidate generator",
    )
    CORE._verify(
        (REPO / protocol["route_probe"]["path"]).resolve(),
        protocol["route_probe"]["sha256"], "route probe",
    )
    CORE._verify(
        (REPO / protocol["world"]["path"]).resolve(),
        protocol["world"]["sha256"], "warehouse world",
    )
    for arm, field in protocol["planning_fields"].items():
        CORE._verify((REPO / field["path"]).resolve(), field["sha256"], f"{arm} field")

    prior = protocol["nominal_route_prior"]
    initial_covariance = np.diag([
        float(prior["xy_std_m"]) ** 2,
        float(prior["xy_std_m"]) ** 2,
        math.radians(float(prior["yaw_std_deg"])) ** 2,
    ])
    gate = protocol["common_feasibility"]
    selected: dict[str, dict[str, Any]] = {}

    for task_name in protocol["task_order"]:
        resolved_by_arm = {
            arm: CORE.ROUTE_PROBE.resolve(
                config_path, task_name, arm, int(protocol["selection_seed"])
            )
            for arm in protocol["arm_order"]
        }
        _assert_arm_invariants(
            task_name, resolved_by_arm, list(gate["shared_setting_keys"])
        )
        first_arm = protocol["arm_order"][0]
        first = resolved_by_arm[first_arm]
        state = np.asarray(
            [first["resolved"]["spawn"][axis] for axis in ("x", "y", "yaw")],
            dtype=float,
        )
        goal = np.asarray(
            [first["resolved"]["goal_x"], first["resolved"]["goal_y"]], dtype=float
        )
        scene, candidates = V4._candidate_routes(
            first["settings"]["driveable_geometry_json"], state[:2], goal,
            float(gate["sample_step_m"]),
        )
        if not candidates:
            raise RuntimeError(f"no geometry-only route candidates for {task_name}")

        common = []
        for candidate in candidates:
            route = CORE._dedupe([state[:2].tolist(), *candidate["waypoints"]])
            clearance = CORE._minimum_static_body_clearance(
                scene, route, start_yaw=float(state[2]),
                length_m=float(gate["robot_length_m"]),
                width_m=float(gate["robot_width_m"]),
                step_m=float(gate["sample_step_m"]),
            )
            common.append({
                "name": candidate["name"], "waypoints": candidate["waypoints"],
                "route": route, "minimum_static_body_clearance_m": clearance,
                "static_body_feasible": bool(
                    clearance + 1e-9 >= float(gate["minimum_static_body_clearance_m"])
                ),
            })
        candidate_set_digest = _digest([
            {"name": row["name"], "route": row["route"]} for row in common
        ])

        records_by_arm: dict[str, list[dict[str, Any]]] = {}
        feasibility_by_arm: dict[str, dict[str, tuple[bool, bool, str]]] = {}
        for arm in protocol["arm_order"]:
            settings = resolved_by_arm[arm]["settings"]
            settings["optimizer_initial_routes_json"] = json.dumps([
                {"name": row["name"], "waypoints": row["waypoints"]} for row in common
            ])
            planner = CORE.UnicyclePlannerBase(**settings)
            tolerance = float(settings["optimizer_terminal_goal_tolerance_m"])
            records = []
            feasibility = {}
            for candidate in common:
                controls = planner._controls_for_waypoints(state, candidate["waypoints"])
                result = planner.evaluate_rollout_controls(
                    state, initial_covariance, goal, controls
                )
                rollout_valid = bool(result["rollout_valid"])
                terminal_feasible = bool(
                    rollout_valid
                    and float(result["terminal_goal_distance_pred"]) <= tolerance
                )
                feasible = bool(
                    candidate["static_body_feasible"]
                    and rollout_valid and terminal_feasible
                )
                feasibility[candidate["name"]] = (
                    feasible, terminal_feasible, str(result["invalid_reason"])
                )
                _, canonical = CORE.canonicalize_polyline_json(
                    json.dumps(candidate["route"], separators=(",", ":"), allow_nan=False)
                )
                records.append({
                    "name": candidate["name"], "route_json": canonical,
                    "route_sha256": CORE.route_sha256(canonical),
                    "route_length_m": float(np.linalg.norm(
                        np.diff(np.asarray(candidate["route"]), axis=0), axis=1
                    ).sum()),
                    "minimum_static_body_clearance_m": candidate[
                        "minimum_static_body_clearance_m"
                    ],
                    "common_feasible": feasible,
                    "rollout_valid": rollout_valid,
                    "invalid_reason": str(result["invalid_reason"]),
                    "terminal_goal_feasible": terminal_feasible,
                    **{
                        key: CORE._serializable(result[key]) for key in (
                            "total_cost", "risk_cost", "ambiguity_cost", "control_cost",
                            "obstacle_cost", "terminal_goal_distance_pred",
                            "terminal_goal_progress_m", "min_predicted_obstacle_distance_m",
                            "mean_p_vis_plan", "mean_p_vis_plan_eff",
                            "mean_r_plan_u_std", "mean_r_plan_v_std",
                        )
                    },
                })
            records_by_arm[arm] = records
            feasibility_by_arm[arm] = feasibility

        first_feasibility = feasibility_by_arm[first_arm]
        for arm in protocol["arm_order"][1:]:
            if feasibility_by_arm[arm] != first_feasibility:
                raise RuntimeError(
                    f"{task_name}: rollout feasibility differs between {first_arm} and {arm}"
                )
        feasible_names = sorted(
            name for name, value in first_feasibility.items() if value[0]
        )
        if not feasible_names:
            raise RuntimeError(f"no common feasible candidate for {task_name}")
        feasible_set = set(feasible_names)
        feasible_digest = _digest(feasible_names)

        selected[task_name] = {}
        for arm in protocol["arm_order"]:
            records = records_by_arm[arm]
            winner = min(
                (record for record in records if record["name"] in feasible_set),
                key=lambda record: (float(record["total_cost"]), record["name"]),
            )
            source_payload = {
                "schema_version": 5,
                "kind": "stage09_shared_feasible_route_selection",
                "task": task_name, "arm": arm,
                "selection_rule": protocol["selection_rule"],
                "candidate_set_digest": candidate_set_digest,
                "common_feasible_candidate_digest": feasible_digest,
                "common_feasible_candidates": feasible_names,
                "selected_candidate": winner, "candidate_results": records,
                "nominal_initial_state": state.tolist(),
                "nominal_initial_covariance": initial_covariance.tolist(),
                "goal_xy": goal.tolist(), "planning_field": protocol["planning_fields"][arm],
                "route_selection_config_sha256": protocol["route_selection_config_sha256"],
                "tasks_yaml_sha256": protocol["tasks_yaml_sha256"],
                "selector_sha256": protocol["selector_sha256"],
            }
            source_path = args.output / f"{task_name}__{arm}.json"
            CORE._atomic_json(source_path, source_payload)
            source_sha = CORE.sha256_file(source_path)
            validated = CORE.validate_preselected_route(
                winner["route_json"], winner["route_sha256"],
                start_xy=state[:2], goal_xy=goal,
                driveable_geometry_json=first["settings"]["driveable_geometry_json"],
                declared_clearance_m=float(gate["centerline_clearance_m"]),
                source_path=source_path, expected_source_sha256=source_sha,
                endpoint_tolerance_m=float(gate["endpoint_tolerance_m"]),
                sample_step_m=float(gate["sample_step_m"]),
            )
            selected[task_name][arm] = {
                "status": "selected", "candidate": winner["name"],
                "preselected_route_json": winner["route_json"],
                "preselected_route_sha256": winner["route_sha256"],
                "preselected_route_source_path": str(source_path.relative_to(REPO)),
                "preselected_route_source_sha256": source_sha,
                "candidate_set_digest": candidate_set_digest,
                "common_feasible_candidate_digest": feasible_digest,
                "route_length_m": validated.length_m,
                "minimum_centerline_clearance_m": validated.minimum_driveable_clearance_m,
                "minimum_static_body_clearance_m": winner[
                    "minimum_static_body_clearance_m"
                ],
                "total_cost": winner["total_cost"], "risk_cost": winner["risk_cost"],
                "ambiguity_cost": winner["ambiguity_cost"],
                "mean_usable_detection_probability": winner["mean_p_vis_plan"],
            }
            print(
                f"{task_name}/{arm}: {winner['name']} L={validated.length_m:.2f} m "
                f"J={float(winner['total_cost']):.3f}", flush=True,
            )

    manifest = {
        "schema_version": 5, "kind": "stage09_route_selection_manifest",
        "status": "complete", "protocol_path": str(args.protocol.relative_to(REPO)),
        "protocol_sha256": CORE.sha256_file(args.protocol), "selected": selected,
        "output_files": {},
    }
    for path in sorted(args.output.glob("*.json")):
        manifest["output_files"][path.name] = CORE.sha256_file(path)
    manifest["selection_digest"] = _digest(manifest)
    CORE._atomic_json(args.output / "manifest.json", manifest)


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
