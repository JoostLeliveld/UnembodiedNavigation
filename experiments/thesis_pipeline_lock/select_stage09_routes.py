#!/usr/bin/env python3
"""Select and freeze one Q0/Q1 route per Stage-09 navigation task.

Candidate polylines come only from the warehouse driveable geometry.  Each
candidate is converted to a bounded unicycle rollout and evaluated with the
same metric expected-belief accounting and geometry checks as the runtime
global planner.  No final-audit labels or navigation outcomes enter selection.
"""

from __future__ import annotations

import argparse
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
from unav_common.lane_graph_routes import generate_route_seeds  # noqa: E402
from unav_common.preselected_route import (  # noqa: E402
    canonicalize_polyline_json,
    route_sha256,
    sha256_file,
    validate_preselected_route,
)


def _load_route_probe():
    path = REPO / "experiments/icra_commissioning/network_route_probe.py"
    spec = importlib.util.spec_from_file_location("stage09_network_route_probe", path)
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
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
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


def _dedupe(points: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for point in points:
        xy = [float(point[0]), float(point[1])]
        if not result or math.hypot(xy[0] - result[-1][0], xy[1] - result[-1][1]) > 1e-12:
            result.append(xy)
    return result


def _candidate_rank(record: dict[str, Any]) -> tuple[int, int, float, str]:
    # Fail closed on geometry, then require task completion, then compare the
    # exact objective.  Name is a deterministic final tie-break only.
    return (
        0 if record["rollout_valid"] else 1,
        0 if record["terminal_goal_feasible"] else 1,
        float(record["total_cost"]),
        str(record["name"]),
    )


def run(args: argparse.Namespace) -> None:
    protocol = _read_json(args.protocol)
    if args.output.exists():
        raise RuntimeError("route-selection outputs are immutable; choose a new directory")
    args.output.mkdir(parents=True)

    config = (REPO / protocol["route_selection_config"]).resolve()
    tasks_path = (REPO / protocol["tasks_yaml"]).resolve()
    _verify(config, protocol["route_selection_config_sha256"], "route-selection config")
    _verify(tasks_path, protocol["tasks_yaml_sha256"], "task registry")
    _verify(Path(__file__).resolve(), protocol["selector_sha256"], "route selector")
    _verify(
        (REPO / protocol["world"]["path"]).resolve(),
        protocol["world"]["sha256"],
        "warehouse world",
    )
    for arm, field in protocol["availability_fields"].items():
        _verify((REPO / field["path"]).resolve(), field["sha256"], f"{arm} availability field")

    initial_covariance = np.diag(
        [
            float(protocol["nominal_route_prior"]["xy_std_m"]) ** 2,
            float(protocol["nominal_route_prior"]["xy_std_m"]) ** 2,
            math.radians(float(protocol["nominal_route_prior"]["yaw_std_deg"])) ** 2,
        ]
    )
    task_names = list(protocol["task_order"])
    arm_names = list(protocol["arm_order"])
    selected: dict[str, dict[str, Any]] = {}

    for task_name in task_names:
        selected[task_name] = {}
        for arm in arm_names:
            captured = ROUTE_PROBE.resolve(
                config, task_name, arm, int(protocol["selection_seed"])
            )
            settings = captured["settings"]
            resolved = captured["resolved"]
            state = np.asarray(
                [resolved["spawn"][axis] for axis in ("x", "y", "yaw")], dtype=float
            )
            goal = np.asarray([resolved["goal_x"], resolved["goal_y"]], dtype=float)
            candidates = generate_route_seeds(
                settings["driveable_geometry_json"], state[:2], goal
            )
            if not candidates:
                raise RuntimeError(f"no geometry-only route candidates for {task_name}")

            settings["optimizer_initial_routes_json"] = json.dumps(candidates)
            planner = UnicyclePlannerBase(**settings)
            records: list[dict[str, Any]] = []
            terminal_tolerance = float(settings["optimizer_terminal_goal_tolerance_m"])
            for candidate in candidates:
                controls = planner._controls_for_waypoints(state, candidate["waypoints"])
                result = planner.evaluate_rollout_controls(
                    state, initial_covariance, goal, controls
                )
                route_points = _dedupe([state[:2].tolist(), *candidate["waypoints"]])
                _, canonical = canonicalize_polyline_json(
                    json.dumps(route_points, separators=(",", ":"), allow_nan=False)
                )
                records.append(
                    {
                        "name": candidate["name"],
                        "route_json": canonical,
                        "route_sha256": route_sha256(canonical),
                        "route_length_m": float(
                            np.linalg.norm(np.diff(np.asarray(route_points), axis=0), axis=1).sum()
                        ),
                        "rollout_valid": bool(result["rollout_valid"]),
                        "invalid_reason": str(result["invalid_reason"]),
                        "terminal_goal_feasible": bool(
                            result["rollout_valid"]
                            and float(result["terminal_goal_distance_pred"]) <= terminal_tolerance
                        ),
                        **{
                            key: _serializable(result[key])
                            for key in (
                                "total_cost",
                                "risk_cost",
                                "ambiguity_cost",
                                "control_cost",
                                "obstacle_cost",
                                "terminal_goal_distance_pred",
                                "terminal_goal_progress_m",
                                "min_predicted_obstacle_distance_m",
                                "mean_p_vis_plan",
                                "mean_p_vis_plan_eff",
                                "mean_r_plan_u_std",
                                "mean_r_plan_v_std",
                            )
                        },
                    }
                )

            winner = min(records, key=_candidate_rank)
            if not (winner["rollout_valid"] and winner["terminal_goal_feasible"]):
                raise RuntimeError(
                    f"no footprint-valid goal-reaching candidate for {task_name}/{arm}"
                )

            source_payload = {
                "schema_version": 1,
                "kind": "stage09_frozen_route_selection",
                "task": task_name,
                "arm": arm,
                "selection_rule": protocol["selection_rule"],
                "selected_candidate": winner,
                "candidate_results": records,
                "nominal_initial_state": state.tolist(),
                "nominal_initial_covariance": initial_covariance.tolist(),
                "goal_xy": goal.tolist(),
                "availability_field": protocol["availability_fields"][arm],
                "route_selection_config_sha256": protocol["route_selection_config_sha256"],
                "tasks_yaml_sha256": protocol["tasks_yaml_sha256"],
                "selector_sha256": protocol["selector_sha256"],
            }
            source_path = args.output / f"{task_name}__{arm}.json"
            _atomic_json(source_path, source_payload)
            source_sha = sha256_file(source_path)
            validated = validate_preselected_route(
                winner["route_json"],
                winner["route_sha256"],
                start_xy=state[:2],
                goal_xy=goal,
                driveable_geometry_json=settings["driveable_geometry_json"],
                declared_clearance_m=float(protocol["route_gate"]["centerline_clearance_m"]),
                source_path=source_path,
                expected_source_sha256=source_sha,
                endpoint_tolerance_m=float(protocol["route_gate"]["endpoint_tolerance_m"]),
                sample_step_m=float(protocol["route_gate"]["sample_step_m"]),
            )
            selected[task_name][arm] = {
                "candidate": winner["name"],
                "preselected_route_json": winner["route_json"],
                "preselected_route_sha256": winner["route_sha256"],
                "preselected_route_source_path": str(source_path.relative_to(REPO)),
                "preselected_route_source_sha256": source_sha,
                "route_length_m": validated.length_m,
                "minimum_centerline_clearance_m": validated.minimum_driveable_clearance_m,
                "total_cost": winner["total_cost"],
                "risk_cost": winner["risk_cost"],
                "ambiguity_cost": winner["ambiguity_cost"],
                "mean_usable_detection_probability": winner["mean_p_vis_plan"],
            }
            print(
                f"{task_name}/{arm}: {winner['name']} "
                f"L={validated.length_m:.2f} m J={winner['total_cost']:.3f}",
                flush=True,
            )

    manifest = {
        "schema_version": 1,
        "kind": "stage09_route_selection_manifest",
        "status": "complete",
        "protocol_path": str(args.protocol.relative_to(REPO)),
        "protocol_sha256": sha256_file(args.protocol),
        "selected": selected,
        "output_files": {},
    }
    for path in sorted(args.output.glob("*.json")):
        manifest["output_files"][path.name] = sha256_file(path)
    manifest_text = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["selection_digest"] = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    _atomic_json(args.output / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "experiments/thesis_pipeline_lock/stage09_route_selection_protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
