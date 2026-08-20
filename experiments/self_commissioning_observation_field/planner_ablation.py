#!/usr/bin/env python3
"""Exact four-camera hit-subset belief propagation and route ablation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import numpy as np

import field_common as C

C.add_import_paths()
from reliability.projection import camera_model_from_world  # noqa: E402


@dataclass
class RouteEvaluation:
    route_id: str
    path: np.ndarray
    length_m: float
    expected_longest_miss_steps: float
    p_only: dict
    full_field: dict


def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def exact_subset_update(
    prior_covariance: np.ndarray,
    probabilities: np.ndarray,
    measurement_covariances: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Expected covariance after all 2^C simultaneous hit/miss outcomes.

    The nominal corrected innovations are zero, so the subset-conditioned means
    coincide and the mixture covariance is the probability-weighted covariance.
    """
    prior_covariance = nearest_psd(prior_covariance)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    measurement_covariances = np.asarray(measurement_covariances, dtype=float)
    camera_count = len(probabilities)
    prior_information = np.linalg.inv(prior_covariance)
    expected = np.zeros_like(prior_covariance)
    records = []
    for mask in range(1 << camera_count):
        mass = 1.0
        information = prior_information.copy()
        hits = []
        for camera_index, probability in enumerate(probabilities):
            hit = bool(mask & (1 << camera_index))
            mass *= float(probability if hit else 1.0 - probability)
            if hit:
                covariance = nearest_psd(measurement_covariances[camera_index])
                information += np.linalg.inv(covariance)
                hits.append(camera_index)
        posterior = nearest_psd(np.linalg.inv(information))
        expected += mass * posterior
        records.append({"mask": mask, "hits": hits, "probability": mass, "covariance": posterior})
    total_mass = float(sum(item["probability"] for item in records))
    if not np.isclose(total_mass, 1.0, atol=1e-10):
        raise RuntimeError(f"subset probability mass is {total_mass}, not one")
    return nearest_psd(expected), records


def projection_jacobian(camera_model, point: np.ndarray) -> np.ndarray | None:
    u, v, _ = camera_model.world_to_pixel(float(point[0]), float(point[1]), 0.0)
    epsilon = 0.25
    up, um = camera_model.pixel_to_world(u + epsilon, v), camera_model.pixel_to_world(u - epsilon, v)
    vp, vm = camera_model.pixel_to_world(u, v + epsilon), camera_model.pixel_to_world(u, v - epsilon)
    if any(value is None for value in (up, um, vp, vm)):
        return None
    return np.column_stack([
        (np.asarray(up, dtype=float) - np.asarray(um, dtype=float)) / (2.0 * epsilon),
        (np.asarray(vp, dtype=float) - np.asarray(vm, dtype=float)) / (2.0 * epsilon),
    ])


class ObservationField:
    def __init__(self) -> None:
        artifact = C.OUT / "commissioned/observation_field.npz"
        summary = C.OUT / "commissioned/summary.json"
        if not artifact.is_file() or not summary.is_file():
            raise RuntimeError("run commission_field.py first")
        self.value = {key: np.asarray(value) for key, value in np.load(artifact).items()}
        self.xs = np.asarray(self.value["xs"], float)
        self.ys = np.asarray(self.value["ys"], float)
        ids = [str(value) for value in self.value["camera_ids"]]
        self.camera_index = {camera: ids.index(camera) for camera in ids}
        self.models = {
            camera: camera_model_from_world(C.WORLD, include_name=C.INCLUDE_NAMES[camera])
            for camera in C.CAMERAS
        }

    def cell(self, camera: str, point: np.ndarray) -> tuple[int, int, int]:
        return (
            self.camera_index[camera],
            int(np.argmin(np.abs(self.ys - float(point[1])))),
            int(np.argmin(np.abs(self.xs - float(point[0])))),
        )

    def uv_covariance(self, camera: str, point: np.ndarray, arm: str) -> np.ndarray:
        if arm == "p_only":
            return nearest_psd(
                self.value["pooled_R_uv_px2"] + self.value["pooled_bias_posterior_cov_uv_px2"]
            )
        ci, iy, ix = self.cell(camera, point)
        return nearest_psd(np.asarray([
            [self.value["R_uu_px2"][ci, iy, ix] + self.value["B_uu_px2"][ci, iy, ix],
             self.value["R_uv_px2"][ci, iy, ix] + self.value["B_uv_px2"][ci, iy, ix]],
            [self.value["R_uv_px2"][ci, iy, ix] + self.value["B_uv_px2"][ci, iy, ix],
             self.value["R_vv_px2"][ci, iy, ix] + self.value["B_vv_px2"][ci, iy, ix]],
        ]))

    def ground_covariance(self, camera: str, point: np.ndarray, arm: str) -> np.ndarray | None:
        jacobian = projection_jacobian(self.models[camera], point)
        if jacobian is None:
            return None
        return nearest_psd(jacobian @ self.uv_covariance(camera, point, arm) @ jacobian.T)


def belief_profile(
    points: np.ndarray,
    per_camera_p: dict[str, np.ndarray],
    field: ObservationField,
    arm: str,
) -> dict:
    covariance = np.eye(2) * C.INITIAL_SIGMA_M**2
    process = np.eye(2) * C.PROCESS_SIGMA_M_PER_STEP**2
    traces, major_sigma, no_hit = [], [], []
    for step, point in enumerate(points):
        predicted = nearest_psd(covariance + process)
        probabilities, measurements = [], []
        for camera in C.CAMERAS:
            measurement = field.ground_covariance(camera, point, arm)
            probability = float(per_camera_p[camera][step])
            if measurement is None or not np.isfinite(measurement).all():
                probability = 0.0
                measurement = np.eye(2) * 1e9
            probabilities.append(probability)
            measurements.append(measurement)
        covariance, _ = exact_subset_update(predicted, np.asarray(probabilities), np.asarray(measurements))
        traces.append(float(np.trace(covariance)))
        major_sigma.append(float(math.sqrt(max(np.linalg.eigvalsh(covariance)))))
        no_hit.append(float(np.prod(1.0 - np.asarray(probabilities))))
    return {
        "mean_trace_m2": float(np.mean(traces)),
        "maximum_trace_m2": float(np.max(traces)),
        "terminal_trace_m2": float(traces[-1]),
        "maximum_major_sigma_m": float(np.max(major_sigma)),
        "expected_no_hit_steps": float(np.sum(no_hit)),
        "trace_profile_m2": traces,
        "major_sigma_profile_m": major_sigma,
    }


def evaluate_route(route_id: str, path: np.ndarray, p: dict[str, np.ndarray], field: ObservationField) -> RouteEvaluation:
    points = C.resample_path(path)
    xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    per_camera = {
        camera: C.sample(np.asarray(p[f"P_{camera}_map"], float), xs, ys, points)
        for camera in C.CAMERAS
    }
    fused = 1.0 - np.prod(np.stack([1.0 - per_camera[camera] for camera in C.CAMERAS]), axis=0)
    return RouteEvaluation(
        route_id=route_id,
        path=np.asarray(path, dtype=float),
        length_m=C.path_length(path),
        expected_longest_miss_steps=C.expected_longest_miss(fused),
        p_only=belief_profile(points, per_camera, field, "p_only"),
        full_field=belief_profile(points, per_camera, field, "full_field"),
    )


def serialise_route(route: RouteEvaluation) -> dict:
    return {
        "route_id": route.route_id,
        "path": route.path.tolist(),
        "length_m": route.length_m,
        "expected_longest_miss_steps": route.expected_longest_miss_steps,
        "expected_longest_miss_s": route.expected_longest_miss_steps * C.DT_S,
        "p_only_belief": route.p_only,
        "full_field_belief": route.full_field,
    }


def solve_task(task: str, p: dict[str, np.ndarray], field: ObservationField) -> dict:
    candidates = [evaluate_route(route_id, path, p, field) for route_id, path in C.route_library()[task]]
    shortest = min(candidates, key=lambda item: (item.length_m, item.route_id))
    length_budget = shortest.length_m * C.LENGTH_BUDGET_RATIO
    eligible = [item for item in candidates if item.length_m <= length_budget + 1e-9]
    p_selected = min(eligible, key=lambda item: (round(item.expected_longest_miss_steps, 10), item.length_m, item.route_id))
    availability_tolerance = max(0.25, 0.05 * p_selected.expected_longest_miss_steps)
    availability_safe = [
        item for item in eligible
        if item.expected_longest_miss_steps <= p_selected.expected_longest_miss_steps + availability_tolerance + 1e-12
    ]
    full_selected = min(
        availability_safe,
        key=lambda item: (
            round(item.full_field["mean_trace_m2"], 12),
            round(item.full_field["maximum_trace_m2"], 12),
            round(item.expected_longest_miss_steps, 10),
            item.length_m,
            item.route_id,
        ),
    )
    return {
        "task": task,
        "shortest": serialise_route(shortest),
        "p_only_selected": serialise_route(p_selected),
        "full_field_selected": serialise_route(full_selected),
        "length_budget_m": length_budget,
        "availability_tolerance_steps": availability_tolerance,
        "candidate_count": len(candidates),
        "length_eligible_count": len(eligible),
        "availability_safe_count": len(availability_safe),
        "p_to_full_route_separation_m": C.max_route_separation(p_selected.path, full_selected.path),
        "full_field_trace_reduction_vs_p_route_fraction": float(
            (p_selected.full_field["mean_trace_m2"] - full_selected.full_field["mean_trace_m2"])
            / max(p_selected.full_field["mean_trace_m2"], 1e-12)
        ),
        "p_gap_regret_full_vs_p_steps": float(full_selected.expected_longest_miss_steps - p_selected.expected_longest_miss_steps),
    }


def main() -> None:
    C.assert_frozen()
    field = ObservationField()
    p = C.load_p()
    tasks = [solve_task(task, p, field) for task in C.TASKS]
    changed = sum(item["p_to_full_route_separation_m"] >= 0.25 for item in tasks)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "OFFLINE_ROUTE_ABLATION_COMPLETE",
        "availability": "frozen A3 monocular-depth prior plus learned GP residual",
        "full_field": "selected Bayesian conditional bias/R artifact; bias posterior covariance included in R_eff",
        "p_only": "same per-camera p_use and projection geometry, but one pooled global bias correction and calibrated pixel covariance; no spatial/camera/mode model",
        "hit_model": "all 16 four-camera hit/miss subsets enumerated exactly at every route step; conditional independence assumption",
        "belief_assumptions": {
            "initial_sigma_m": C.INITIAL_SIGMA_M,
            "process_sigma_m_per_step": C.PROCESS_SIGMA_M_PER_STEP,
            "step_s": C.DT_S,
            "nominal_corrected_innovation": 0.0,
        },
        "decision_rule": {
            "length": "at most 1.05 times the shortest candidate",
            "p_only": "minimum exact expected longest miss run",
            "full_field": "minimum mean exact belief trace among routes within max(0.25 step, 5%) of the p-only availability optimum",
            "weighted_sum": False,
        },
        "changed_task_count_at_0p25m": changed,
        "median_full_field_trace_reduction_fraction": float(np.median([
            item["full_field_trace_reduction_vs_p_route_fraction"] for item in tasks
        ])),
        "tasks": tasks,
        "scope": "offline discrimination over the frozen solved E3 route library; not executed closed-loop trajectories",
    }
    C.write_json(C.OUT / "planning/ablation.json", payload)
    print(f"p-only -> full-field changed {changed}/{len(tasks)} tasks by >=0.25 m")
    for item in tasks:
        print(
            f"  {item['task']}: separation={item['p_to_full_route_separation_m']:.2f} m, "
            f"full-field trace reduction={100*item['full_field_trace_reduction_vs_p_route_fraction']:.1f}%"
        )
    print(C.OUT / "planning/ablation.json")


if __name__ == "__main__":
    main()
