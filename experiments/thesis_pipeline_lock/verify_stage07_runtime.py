#!/usr/bin/env python3
"""Check the runtime Stage-07 forward path against the frozen offline formula."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from reliability.commissioned_measurement import CommissionedMeasurementModel
from reliability.projection import camera_model_from_world


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_path = args.artifact.resolve()
    payload = json.loads(artifact_path.read_text())
    runtime = CommissionedMeasurementModel(
        artifact_path, expected_sha256=sha256(artifact_path)
    )
    include_names = dict(zip(CAMERAS, (
        "external_camera", "external_camera_b", "external_camera_c",
        "external_camera_d", "external_camera_e",
    )))
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    cameras = {
        name: camera_model_from_world(world, include_name=include)
        for name, include in include_names.items()
    }
    correction_kind = payload["correction_model"]
    coefficient = intercept = direct_mlp = None
    if correction_kind == "C2_ridge_residual":
        coefficient = np.asarray(payload["correction_parameters"]["coefficient"])
        intercept = np.asarray(payload["correction_parameters"]["intercept"])
    elif correction_kind == "C3_mlp_residual":
        direct_mlp = torch.nn.Sequential(
            torch.nn.Linear(len(payload["feature_names"]), 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
        state_path = artifact_path.parent / payload["correction_state"]
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        direct_mlp.load_state_dict({
            key.removeprefix("layers."): value for key, value in state.items()
        })
        direct_mlp.eval()
    else:
        raise ValueError(f"unsupported correction model: {correction_kind!r}")
    mean = np.asarray(payload["correction_parameters"]["scaler_mean"])
    scale = np.asarray(payload["correction_parameters"]["scaler_scale"])
    bias = {
        camera: np.asarray(value)
        for camera, value in payload["per_camera_residual_bias_xy_m"].items()
    }
    covariance_scale = float(payload["covariance_parameters"]["calibration_scale"])
    maximum_xy = 0.0
    maximum_covariance = 0.0
    checked = 0
    with args.admissions.resolve().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["admitted"] != "1":
                continue
            camera_id = row["camera_id"]
            camera = cameras[camera_id]
            box = tuple(float(row[key]) for key in (
                "best_box_x0", "best_box_y0", "best_box_x1", "best_box_y1"))
            predicted = tuple(float(row[key]) for key in (
                "expected_box_x0", "expected_box_y0", "expected_box_x1", "expected_box_y1"))
            raw = tuple(float(row[key]) for key in ("raw_ground_x", "raw_ground_y"))
            analytic = tuple(float(row[key]) for key in ("equivalent_x", "equivalent_y"))
            prior = tuple(float(row[key]) for key in ("robot_x", "robot_y"))
            yaw = float(row["robot_yaw"])
            confidence = float(row["raw_best_confidence"])
            result = runtime.apply(
                camera_id, raw, analytic, box, predicted, confidence,
                prior, yaw, camera,
            )
            if result is None:
                raise RuntimeError(f"runtime refused admitted row {checked}")
            runtime_xy, runtime_covariance = result
            width, height = box[2] - box[0], box[3] - box[1]
            predicted_width = predicted[2] - predicted[0]
            predicted_height = predicted[3] - predicted[1]
            cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
            dx, dy = raw[0] - cx, raw[1] - cy
            raw_range = math.hypot(dx, dy)
            relative = yaw - math.atan2(cy - prior[1], cx - prior[0])
            features = np.asarray([
                raw_range, 1.0 / raw_range, width, height, width / height,
                0.5 * (box[0] + box[2]) / 1280.0, box[3] / 720.0,
                confidence, width / predicted_width, height / predicted_height,
                math.sin(relative), math.cos(relative),
                *(float(camera_id == candidate) for candidate in CAMERAS),
            ])
            standardized = (features - mean) / scale
            if correction_kind == "C2_ridge_residual":
                ray_prediction = intercept + coefficient @ standardized
            else:
                with torch.no_grad():
                    ray_prediction = direct_mlp(
                        torch.tensor([standardized], dtype=torch.float32)
                    )[0].numpy()
            unit = np.asarray((dx / raw_range, dy / raw_range))
            basis = np.column_stack((unit, np.asarray((-unit[1], unit[0]))))
            expected_xy = np.asarray(analytic) + basis @ ray_prediction + bias[camera_id]
            entry = payload["covariance_parameters"]["per_camera_width_bins"][camera_id]
            bin_index = int(np.searchsorted(np.asarray(entry["width_edges_px"]), width, side="left"))
            expected_covariance = covariance_scale * np.asarray(entry["covariance_m2"][bin_index])
            maximum_xy = max(maximum_xy, float(np.max(np.abs(np.asarray(runtime_xy) - expected_xy))))
            maximum_covariance = max(
                maximum_covariance,
                float(np.max(np.abs(np.asarray(runtime_covariance) - expected_covariance))),
            )
            checked += 1
    report = {
        "schema": "thesis_stage07_runtime_verification.v1",
        "status": "pass" if maximum_xy <= 1e-12 and maximum_covariance <= 1e-12 else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_audit_accessed": False,
        "rows_checked": checked,
        "artifact_sha256": sha256(artifact_path),
        "admissions_sha256": sha256(args.admissions.resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "maximum_xy_absolute_difference_m": maximum_xy,
        "maximum_covariance_absolute_difference_m2": maximum_covariance,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
