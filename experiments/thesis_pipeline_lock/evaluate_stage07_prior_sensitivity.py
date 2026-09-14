#!/usr/bin/env python3
"""Check the frozen Stage-07 correction under deterministic runtime-prior error."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reliability.projection import camera_model_from_world
from reliability.silhouette_observation import equivalent_position_measurement
from select_stage06_gate import gate_reasons
from unav_common.robot_hull import VISUAL_HULL, silhouette_box


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def percentile(values, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    index = (len(ordered) - 1) * probability
    lower, upper = int(math.floor(index)), int(math.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def summarize(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)), "median_m": float(np.median(values)),
        "rms_m": float(np.sqrt(np.mean(values ** 2))),
        "p90_m": percentile(values, 0.90), "p95_m": percentile(values, 0.95),
        "maximum_m": float(np.max(values)), "above_0_25_m": int(np.sum(values > 0.25)),
    }


def fitted_exact_biases(admission_csv: Path) -> dict[str, dict[str, np.ndarray]]:
    values = {name: defaultdict(list) for name in ("raw", "analytic")}
    with admission_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["admitted"] != "1":
                continue
            truth = np.asarray([float(row["robot_x"]), float(row["robot_y"])])
            raw = np.asarray([float(row["raw_ground_x"]), float(row["raw_ground_y"])])
            analytic = np.asarray([float(row["equivalent_x"]), float(row["equivalent_y"])])
            values["raw"][row["camera_id"]].append(truth - raw)
            values["analytic"][row["camera_id"]].append(truth - analytic)
    return {
        name: {camera: np.mean(rows, axis=0) for camera, rows in by_camera.items()}
        for name, by_camera in values.items()
    }


def ridge_features(record: dict, camera, prior_xy, prior_yaw, predicted) -> np.ndarray:
    box = record["best_box_xyxy"]
    raw = record["raw_ground_xy"]
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    distance = math.hypot(float(raw[0]) - cx, float(raw[1]) - cy)
    width, height = float(box[2] - box[0]), float(box[3] - box[1])
    predicted_width = float(predicted[2] - predicted[0])
    predicted_height = float(predicted[3] - predicted[1])
    bearing = math.atan2(cy - prior_xy[1], cx - prior_xy[0])
    relative = prior_yaw - bearing
    return np.asarray([
        distance, 1.0 / max(distance, 1e-6), width, height,
        width / max(height, 1e-6), 0.5 * (box[0] + box[2]) / 1280.0,
        box[3] / 720.0, record["raw_best_confidence"],
        width / predicted_width, height / predicted_height,
        math.sin(relative), math.cos(relative),
        *[float(record["camera_id"] == camera_id) for camera_id in CAMERAS],
    ], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage06-inference", type=Path, required=True)
    parser.add_argument("--stage06-gate", type=Path, required=True)
    parser.add_argument("--stage07", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if "post_selection_runtime_prior_sensitivity" not in protocol:
        raise RuntimeError("Stage-07 sensitivity rule is not frozen")
    stage07 = args.stage07.resolve()
    run_manifest_path = stage07 / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    if run_manifest["selected_correction"] != "C2_ridge_residual":
        raise RuntimeError("This diagnostic expects the selected ridge residual model")
    model_path = stage07 / "measurement_model.json"
    artifact = json.loads(model_path.read_text())
    if artifact["correction_model"] != run_manifest["selected_correction"]:
        raise RuntimeError("Stage-07 correction identity mismatch")
    parameters = artifact["correction_parameters"]
    scaler_mean = np.asarray(parameters["scaler_mean"])
    scaler_scale = np.asarray(parameters["scaler_scale"])
    coefficient = np.asarray(parameters["coefficient"])
    intercept = np.asarray(parameters["intercept"])
    ridge_bias = {
        camera: np.asarray(value) for camera, value in artifact["per_camera_residual_bias_xy_m"].items()
    }

    gate_report_path = args.stage06_gate.resolve()
    gate_report = json.loads(gate_report_path.read_text())
    gate = gate_report["selected"]["candidate"]
    inference = args.stage06_inference.resolve()
    inference_manifest = json.loads((inference / "manifest.json").read_text())
    records_path = inference / inference_manifest["records"]
    if sha256(records_path) != inference_manifest["records_sha256"]:
        raise RuntimeError("Stage-06 detector records drift")
    with records_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != 9600:
        raise RuntimeError("Stage-06 population size mismatch")

    admission_csv = args.stage06_gate.resolve().parent / "admission_records.csv"
    exact_bias = fitted_exact_biases(admission_csv)
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    includes = dict(zip(CAMERAS, (
        "external_camera", "external_camera_b", "external_camera_c",
        "external_camera_d", "external_camera_e",
    )))
    cameras = {camera: camera_model_from_world(world, include_name=include) for camera, include in includes.items()}

    errors = {name: [] for name in ("raw", "analytic", "ridge")}
    by_camera = {camera: {name: [] for name in errors} for camera in CAMERAS}
    counts = Counter()
    by_draw = defaultdict(Counter)
    for record in records:
        camera_id = record["camera_id"]
        camera = cameras[camera_id]
        raw = record["raw_ground_xy"]
        for draw in range(4):
            counts["opportunities"] += 1; by_draw[draw]["opportunities"] += 1
            key = (
                f"thesis-stage06-prior-sensitivity-v1|{record['pose_id']}|"
                f"{camera_id}|{draw}"
            ).encode("ascii")
            seed = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            dx, dy = np.clip(rng.normal(0.0, 0.10, size=2), -0.20, 0.20)
            dyaw = float(np.clip(rng.normal(0.0, math.radians(5.0)), -math.radians(10.0), math.radians(10.0)))
            prior = (record["robot_x"] + float(dx), record["robot_y"] + float(dy))
            prior_yaw = record["robot_yaw"] + dyaw
            predicted = silhouette_box(camera, prior[0], prior[1], prior_yaw, VISUAL_HULL)
            equivalent = None
            if raw is not None:
                converted = equivalent_position_measurement(
                    (float(raw[0]), float(raw[1])), ((1.0, 0.0), (0.0, 1.0)),
                    camera, prior, prior_yaw,
                )
                equivalent = None if converted is None else np.asarray(converted[0])
            reasons = gate_reasons(
                record, gate, predicted_box=predicted,
                equivalent_available=equivalent is not None,
            )
            if reasons:
                continue
            counts["admitted"] += 1; by_draw[draw]["admitted"] += 1
            truth = np.asarray([record["robot_x"], record["robot_y"]])
            raw_point = np.asarray(raw) + exact_bias["raw"][camera_id]
            analytic_point = equivalent + exact_bias["analytic"][camera_id]
            features = ridge_features(record, camera, prior, prior_yaw, predicted)
            ray_residual = coefficient @ ((features - scaler_mean) / scaler_scale) + intercept
            camera_xy = np.asarray(camera.cam_pos[:2], dtype=float)
            along = np.asarray(raw) - camera_xy; along /= np.linalg.norm(along)
            across = np.asarray([-along[1], along[0]])
            ridge_point = equivalent + ray_residual[0] * along + ray_residual[1] * across + ridge_bias[camera_id]
            for name, point in (("raw", raw_point), ("analytic", analytic_point), ("ridge", ridge_point)):
                error = float(np.linalg.norm(point - truth))
                errors[name].append(error); by_camera[camera_id][name].append(error)
                by_draw[draw][f"{name}_above_0_25"] += int(error > 0.25)

    report = {
        "schema": "thesis_stage07_prior_sensitivity.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Four deterministic prior perturbations per opportunity; not independent replicates.",
        "opportunities": counts["opportunities"], "admitted": counts["admitted"],
        "models": {name: summarize(values) for name, values in errors.items()},
        "by_camera": {
            camera: {name: summarize(values) for name, values in models.items()}
            for camera, models in by_camera.items()
        },
        "by_draw_counts": {str(draw): dict(values) for draw, values in sorted(by_draw.items())},
        "advance_rule": protocol["post_selection_runtime_prior_sensitivity"]["advance_rule"],
        "advance_pass": (
            summarize(errors["ridge"])["p95_m"] <= summarize(errors["analytic"])["p95_m"]
            and summarize(errors["ridge"])["above_0_25_m"] <= summarize(errors["analytic"])["above_0_25_m"]
        ),
        "protocol": str(protocol_path.relative_to(REPO)), "protocol_sha256": sha256(protocol_path),
        "stage06_inference_manifest_sha256": sha256(inference / "manifest.json"),
        "stage06_gate_report_sha256": sha256(gate_report_path),
        "stage07_run_manifest_sha256": sha256(run_manifest_path),
        "measurement_model_sha256": sha256(model_path),
        "final_audit_accessed": False,
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["advance_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
