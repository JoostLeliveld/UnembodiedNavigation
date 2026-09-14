#!/usr/bin/env python3
"""Export the four q/R planner arms from the completed commissioned model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src/planning"), str(REPO / "src/reliability"),
                str(REPO / "src/unav_common")]

from planning.core.camera_network import CameraNetworkModel  # noqa: E402
from reliability.bernoulli_gp import predict_laplace  # noqa: E402
from reliability.commissioned_availability import CommissionedAvailabilityModel  # noqa: E402
from reliability.commissioned_visibility import CommissionedVisibilitySensorModel  # noqa: E402


CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def spatial_field(
    model: CommissionedVisibilitySensorModel,
    xs: np.ndarray,
    ys: np.ndarray,
    headings: np.ndarray,
) -> np.ndarray:
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    output = np.empty((len(CAMERAS), len(headings), len(ys), len(xs), 2, 2), dtype=float)
    specification = model.spatial_spec
    for camera_index, camera_id in enumerate(CAMERAS):
        selected = model.spatial_camera == camera_id
        train = model.spatial_features[selected]
        residual = model.spatial_residual[selected]
        sample_weight = model.spatial_sample_weight[selected]
        base = model.spatial_base[camera_index]
        prior = float(specification["prior_strength"])
        for heading_index, heading in enumerate(headings):
            values = np.empty((len(points), 2, 2), dtype=float)
            for start in range(0, len(points), 256):
                query = points[start:start + 256]
                distance2 = np.sum((query[:, None, :] - train[None, :, :2]) ** 2, axis=2)
                heading_term = 1.0 - np.cos(float(heading) - train[None, :, 2])
                weight = np.exp(
                    -0.5 * distance2 / float(specification["position_scale_m"]) ** 2
                    - heading_term / float(specification["heading_scale"]) ** 2
                ) * sample_weight[None, :]
                numerator = np.einsum("qn,ni,nj->qij", weight, residual, residual)
                denominator = weight.sum(axis=1)
                covariance = (numerator + prior * base[None]) / (
                    denominator[:, None, None] + prior
                )
                covariance += np.eye(2)[None] * 1.0e-6
                values[start:start + len(query)] = covariance
            output[camera_index, heading_index] = values.reshape(len(ys), len(xs), 2, 2)
    output *= float(specification["external_calibration_scale"])
    return output


def fit_availability_constants(capture_root: Path) -> np.ndarray:
    totals = {camera: 0 for camera in CAMERAS}
    admitted = {camera: 0 for camera in CAMERAS}
    execution = json.loads((capture_root / "campaign_execution.json").read_text())
    for drive_id in execution["completed_drive_ids"]:
        if not drive_id.startswith("fit_"):
            continue
        with (capture_root / drive_id / "tables/camera_opportunities.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                if row["reference_supported"] != "1":
                    continue
                camera = row["camera_id"]
                totals[camera] += 1
                admitted[camera] += row["sensor_gate_admitted"] == "1"
    return np.asarray([admitted[camera] / totals[camera] for camera in CAMERAS])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--availability-model", required=True, type=Path)
    parser.add_argument("--visibility-model", required=True, type=Path)
    parser.add_argument("--runtime-model", type=Path)
    parser.add_argument("--planning-opportunities-per-second", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-step-m", type=float, default=0.20)
    parser.add_argument("--heading-count", type=int, default=16)
    parser.add_argument("--xmin", type=float, default=-11.0)
    parser.add_argument("--xmax", type=float, default=11.0)
    parser.add_argument("--ymin", type=float, default=-9.0)
    parser.add_argument("--ymax", type=float, default=9.0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite planner-field output")
    output.mkdir(parents=True)
    capture_root = args.capture_root.resolve()
    availability_path = args.availability_model.resolve()
    visibility_path = args.visibility_model.resolve()
    runtime_path = (
        args.runtime_model.resolve(strict=True)
        if args.runtime_model is not None else visibility_path
    )
    if args.planning_opportunities_per_second < 1:
        raise ValueError("planning opportunities per second must be positive")
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    availability = CommissionedAvailabilityModel(availability_path, world)
    sensor = CommissionedVisibilitySensorModel(visibility_path)
    if tuple(availability.camera_ids) != CAMERAS or tuple(sensor.camera_order) != CAMERAS:
        raise RuntimeError("commissioned camera registries differ")
    xs = np.arange(args.xmin, args.xmax + 0.5 * args.grid_step_m, args.grid_step_m)
    ys = np.arange(args.ymin, args.ymax + 0.5 * args.grid_step_m, args.grid_step_m)
    headings = np.linspace(0.0, 2.0 * math.pi, args.heading_count + 1)
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    q1 = np.empty((len(CAMERAS), len(ys), len(xs)), dtype=float)
    for index, camera_id in enumerate(CAMERAS):
        probability, _ = predict_laplace(
            availability._gp_parameters[camera_id], points,
            uncertainty_penalty=availability._uncertainty_penalty,
        )
        q1[index] = np.clip(probability, *availability._clip).reshape(len(ys), len(xs))
    q0 = fit_availability_constants(capture_root)
    q0_field = np.broadcast_to(q0[:, None, None], q1.shape).copy()
    r1 = spatial_field(sensor, xs, ys, headings)
    r0 = np.empty((len(CAMERAS), 2, 2), dtype=float)
    for camera_index in range(len(CAMERAS)):
        r0[camera_index] = np.mean(r1[camera_index, :-1], axis=(0, 1, 2))
    r0_field = np.broadcast_to(
        r0[:, None, None, None, :, :], r1.shape
    ).copy()
    sources = {
        relative(availability_path): sha256(availability_path),
        relative(availability.parameter_path): sha256(availability.parameter_path),
        relative(visibility_path): sha256(visibility_path),
        relative(runtime_path): sha256(runtime_path),
        relative(Path(sensor.manifest["parameters"]["path"])): sensor.manifest["parameters"]["sha256"],
        relative(capture_root / "campaign_execution.json"): sha256(capture_root / "campaign_execution.json"),
        relative(world): sha256(world),
        relative(Path(__file__)): sha256(Path(__file__)),
    }
    artifacts = {}
    arms = {
        "C00": (q0_field, r0_field, "constant", "constant"),
        "C01": (q0_field, r1, "constant", "commissioned_spatial"),
        "C10": (q1, r0_field, "commissioned_position_gp", "constant"),
        "C11": (q1, r1, "commissioned_position_gp", "commissioned_spatial"),
    }
    for arm, (q_field, r_field, q_kind, r_kind) in arms.items():
        metadata = {
            "schema": "camera_network.thesis_stage09.v2",
            "reference": "robot_ground_reference_xy",
            "frame": "map_bev",
            "covariance_units": "m2",
            "score_target": "unused_in_metric_expected_belief",
            "availability_target": "stage06_admitted_localization_measurement",
            "arm": arm,
            "availability_arm": q_kind,
            "covariance_arm": r_kind,
            "future_heading_approximation": "none; R field has a periodic heading axis",
            "runtime_equivalence": (
                "all arms execute M3+M4 plus temporally inflated image-conditioned "
                "R at 5 Hz; planning uses a conservative effective update rate and "
                "only the future q/R forecast changes between arms"
            ),
            "planning_opportunities_per_second": args.planning_opportunities_per_second,
            "runtime_sensor_model_path": relative(runtime_path),
            "runtime_sensor_model_sha256": sha256(runtime_path),
            "final_audit_used_for_export_or_selection": False,
            "source_hashes": sources,
        }
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            xs=xs,
            ys=ys,
            headings=headings,
            camera_ids=np.asarray(CAMERAS),
            score=np.zeros_like(q_field),
            availability=q_field,
            R_cond_m2=r0,
            R_miss_proxy_m2=r0 + np.eye(2)[None],
            R_cond_field_m2=r_field,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        path = output / f"{arm.lower()}_planner_field.npz"
        path.write_bytes(buffer.getvalue())
        loaded = CameraNetworkModel(path, expected_camera_ids=CAMERAS)
        if loaded.metadata["arm"] != arm:
            raise RuntimeError("planner artifact round trip changed arm identity")
        artifacts[arm] = {"path": relative(path), "sha256": sha256(path)}
    manifest = {
        "schema": "commissioned_four_arm_planner_fields.v1",
        "status": "frozen_before_audit_and_navigation",
        "audit_accessed": False,
        "grid": {
            "x": [float(xs[0]), float(xs[-1]), len(xs)],
            "y": [float(ys[0]), float(ys[-1]), len(ys)],
            "heading_count_including_periodic_endpoint": len(headings),
        },
        "q0_per_camera": dict(zip(CAMERAS, map(float, q0))),
        "r0_per_camera_m2": dict(zip(CAMERAS, r0.tolist())),
        "artifacts": artifacts,
        "sources": sources,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
