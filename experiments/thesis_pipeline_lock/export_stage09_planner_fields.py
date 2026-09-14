#!/usr/bin/env python3
"""Export matched Q0/Q1 planner fields from the frozen Stage-06--08 artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from planning.core.camera_network import CameraNetworkModel
from reliability.commissioned_availability import CommissionedAvailabilityModel
from reliability.projection import camera_model_from_world


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-step-m", type=float, default=0.20)
    parser.add_argument("--heading-count", type=int, default=16)
    args = parser.parse_args()
    if not math.isfinite(args.grid_step_m) or args.grid_step_m <= 0.0:
        raise ValueError("grid step must be positive")
    if args.heading_count < 4:
        raise ValueError("at least four heading samples are required")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_final_audit_access":
        raise RuntimeError("Stage-09 protocol is not frozen")
    locked = protocol["locked_inputs"]
    availability_path = REPO / locked["availability_model"]
    measurement_path = REPO / locked["measurement_model"]
    gate_path = REPO / locked["gate_report"]
    for path, expected in (
        (availability_path, locked["availability_model_sha256"]),
        (measurement_path, locked["measurement_model_sha256"]),
        (gate_path, locked["gate_report_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"locked input drift: {path}")
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    master_index = REPO / protocol["final_audit"]["master_capture"] / "capture_index.csv"
    with master_index.open(newline="", encoding="utf-8") as handle:
        master = list(csv.DictReader(handle))
    model_names = {
        camera: next(row["camera_model"] for row in master if row["camera_id"] == camera)
        for camera in CAMERAS
    }
    cameras = {camera: camera_model_from_world(world, include_name=model_names[camera]) for camera in CAMERAS}
    q_model = CommissionedAvailabilityModel(
        availability_path, world, expected_sha256=locked["availability_model_sha256"]
    )

    admissions_path = REPO / "logs/thesis_final_pipeline_v1/stage06_detector_gate/gate_selection_v1/admission_records.csv"
    with admissions_path.open(newline="", encoding="utf-8") as handle:
        admission_rows = list(csv.DictReader(handle))
    if len(admission_rows) != 9600:
        raise RuntimeError("planner export requires all 9,600 commissioning opportunities")
    q0 = np.asarray([
        sum(row["admitted"] == "1" and row["camera_id"] == camera for row in admission_rows)
        / sum(row["camera_id"] == camera for row in admission_rows)
        for camera in CAMERAS
    ], dtype=float)

    measurement_payload = json.loads(measurement_path.read_text())
    scale = float(measurement_payload["covariance_parameters"]["calibration_scale"])
    covariance_tables = measurement_payload["covariance_parameters"]["per_camera_width_bins"]
    conditional_covariances = []
    covariance_counts = {}
    for camera in CAMERAS:
        entries = [row for row in admission_rows if row["camera_id"] == camera and row["admitted"] == "1"]
        table = covariance_tables[camera]
        edges = [float(value) for value in table["width_edges_px"]]
        matrices = [scale * np.asarray(value, dtype=float) for value in table["covariance_m2"]]
        selected = []
        bins = [0, 0, 0]
        for row in entries:
            width = float(row["best_box_x1"]) - float(row["best_box_x0"])
            index = 0 if width <= edges[0] else 1 if width <= edges[1] else 2
            selected.append(matrices[index])
            bins[index] += 1
        if not selected:
            raise RuntimeError(f"no admitted commissioning observations for {camera}")
        conditional_covariances.append(np.mean(np.asarray(selected), axis=0))
        covariance_counts[camera] = {"admitted": len(entries), "width_bin_counts": bins}
    R = np.asarray(conditional_covariances)

    x_values = [float(row["robot_x"]) for row in master]
    y_values = [float(row["robot_y"]) for row in master]
    nx = int(math.ceil((max(x_values) - min(x_values)) / args.grid_step_m)) + 1
    ny = int(math.ceil((max(y_values) - min(y_values)) / args.grid_step_m)) + 1
    xs = np.linspace(min(x_values), max(x_values), nx)
    ys = np.linspace(min(y_values), max(y_values), ny)
    headings = np.linspace(0.0, 2.0 * math.pi, args.heading_count, endpoint=False)
    availability_headings = np.asarray([0.0]) if q_model.schema.endswith(".v2") else headings
    q1 = np.empty((len(CAMERAS), len(ys), len(xs)), dtype=float)
    for camera_index, camera_id in enumerate(CAMERAS):
        camera = cameras[camera_id]
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                q1[camera_index, iy, ix] = float(np.mean([
                    q_model.probability(camera_id, float(x), float(y), float(yaw), camera)
                    for yaw in availability_headings
                ]))
    q0_grid = np.broadcast_to(q0[:, None, None], q1.shape).copy()
    output = args.output.resolve()
    if output.exists() or output.with_name(output.name + ".incomplete").exists():
        raise FileExistsError("refusing to overwrite frozen planner fields")
    staging = output.with_name(output.name + ".incomplete")
    staging.mkdir(parents=True)
    implementation = Path(__file__).resolve()
    sources = {
        rel(protocol_path): sha256(protocol_path), rel(availability_path): sha256(availability_path),
        rel(measurement_path): sha256(measurement_path), rel(gate_path): sha256(gate_path),
        rel(admissions_path): sha256(admissions_path), rel(master_index): sha256(master_index),
        rel(world): sha256(world), rel(implementation): sha256(implementation),
    }
    artifacts = {}
    for arm, grid in (("Q0", q0_grid), ("Q1", q1)):
        metadata = {
            "schema": "camera_network.thesis_stage09.v1",
            "pipeline_id": protocol["pipeline_id"], "reference": "robot_ground_reference_xy",
            "frame": "map_bev", "covariance_units": "m2",
            "score_target": "unused_in_metric_expected_belief",
            "availability_target": "stage06_admitted_localization_measurement",
            "availability_arm": arm,
            "availability_definition": (
                "per-camera commissioning constant" if arm == "Q0" else
                f"frozen commissioned availability model {q_model.model_id}"
            ),
            "future_heading_approximation": {
                "reason": (
                    "the commissioned GP is position-only" if q_model.schema.endswith(".v2") else
                    "current planner field interface is XY-only"
                ),
                "uniform_heading_samples": (
                    args.heading_count
                    if arm == "Q1" and not q_model.schema.endswith(".v2") else 0
                ),
            },
            "conditional_covariance": (
                "per-camera mean of frozen Stage-07 deployment R2C over admitted commissioning views; "
                "identical in Q0 and Q1; runtime retains width-conditioned R2C"
            ),
            "runtime_equivalence": (
                "planner arms differ only in future q; both runtime arms use the same frozen "
                "detector, gate, hull+ridge mean, R2C and Q1 runtime quality"
            ),
            "final_audit_used_for_export_or_selection": False,
            "source_hashes": sources,
        }
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, xs=xs, ys=ys, camera_ids=np.asarray(CAMERAS), score=np.zeros_like(grid),
            availability=grid, R_cond_m2=R, R_miss_proxy_m2=R + np.eye(2)[None] * 1.0,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        path = staging / f"{arm.lower()}_availability.npz"
        path.write_bytes(buffer.getvalue())
        model = CameraNetworkModel(path, expected_camera_ids=CAMERAS)
        if model.metadata["availability_arm"] != arm:
            raise RuntimeError("planner artifact round-trip changed arm identity")
        artifacts[arm] = {"path": str((output / path.name).relative_to(REPO)), "sha256": sha256(path)}
    manifest = {
        "schema": "thesis_stage09_planner_fields.v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "final_audit_accessed": False,
        "grid": {"nx": len(xs), "ny": len(ys), "step_x_m": float(xs[1] - xs[0]),
                 "step_y_m": float(ys[1] - ys[0]), "heading_samples_Q1": args.heading_count},
        "q0_per_camera": dict(zip(CAMERAS, q0.tolist())),
        "planning_R_cond_m2": dict(zip(CAMERAS, R.tolist())),
        "covariance_source_counts": covariance_counts, "artifacts": artifacts,
        "sources": sources,
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": sha256(manifest_path)}) + "\n")
    os.replace(staging, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
