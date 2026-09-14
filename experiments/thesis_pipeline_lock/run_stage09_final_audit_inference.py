#!/usr/bin/env python3
"""Open the sealed final-audit images once with the already frozen detector."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from export_detector_dataset import classify
from run_stage06_commissioning_inference import clean_number, distance, finite, iou, sha256
from reliability.projection import camera_model_from_world
from reliability.silhouette_observation import equivalent_position_measurement


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_final_audit_access":
        raise RuntimeError("Stage-09 release protocol is not frozen")
    specification = protocol["final_audit"]
    if specification["authorized_role"] != "final_audit":
        raise RuntimeError("this runner opens only final_audit")
    pipeline = json.loads((REPO / "experiments/thesis_pipeline_lock/pipeline_lock.json").read_text())
    for stage_id in ("05_detector_training", "06_detector_gate", "07_correction_and_covariance", "08_availability_model"):
        stage = next(value for value in pipeline["stages"] if value["id"] == stage_id)
        if stage["status"] != "locked":
            raise RuntimeError(f"{stage_id} is not locked")
    inputs = protocol["locked_inputs"]
    weights = (REPO / inputs["detector_checkpoint"]).resolve()
    if sha256(weights) != inputs["detector_checkpoint_sha256"]:
        raise RuntimeError("detector checkpoint drift")
    capture = (REPO / specification["master_capture"]).resolve()
    capture_manifest_path = capture / "capture_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text())
    index_path = capture / "capture_index.csv"
    if capture_manifest["capture_index_sha256"] != sha256(index_path):
        raise RuntimeError("capture index drift")
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset_split"] == "final_audit"]
    if len(rows) != int(specification["views"]):
        raise RuntimeError(f"expected {specification['views']} final views, found {len(rows)}")
    images = [(capture / row["image"]).resolve() for row in rows]
    if any(not image.is_file() for image in images):
        raise RuntimeError("a final-audit image is missing")
    camera_models = {
        camera: camera_model_from_world(
            REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf",
            include_name=next(row["camera_model"] for row in rows if row["camera_id"] == camera),
        )
        for camera in CAMERAS
    }
    label_protocol_path = REPO / inputs["label_protocol"]
    if sha256(label_protocol_path) != inputs["label_protocol_sha256"]:
        raise RuntimeError("label protocol drift")
    label_contract = json.loads(label_protocol_path.read_text())["detector_label_contract"]["positive_requires_all"]
    model = YOLO(str(weights))
    batch = int(args.batch)
    if batch <= 0:
        raise ValueError("batch must be positive")

    def predictions():
        for start in range(0, len(images), batch):
            yield from model.predict(
                source=[str(path) for path in images[start:start + batch]],
                imgsz=int(specification["detector_imgsz"]),
                conf=float(specification["detector_prediction_floor"]),
                iou=float(specification["detector_iou"]),
                max_det=20, device="0", batch=batch, workers=2,
                verbose=False, save=False, stream=False,
            )

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    staging = output.with_name(output.name + ".incomplete")
    staging.mkdir(parents=True)
    records_path = staging / "records.jsonl"
    label_counts, return_counts = collections.Counter(), collections.Counter()
    start_time = time.monotonic()
    with records_path.open("w", encoding="utf-8") as handle:
        for index, (row, result) in enumerate(zip(rows, predictions(), strict=True), start=1):
            label_class, label_reasons, mask_box, label_metrics = classify(row, 1280, 720, label_contract)
            label_counts[label_class] += 1
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
            scores = result.boxes.conf.detach().cpu().numpy().astype(float)
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            valid = np.isfinite(scores) & np.all(np.isfinite(boxes), axis=1)
            valid &= (boxes[:, 0] >= 0.0) & (boxes[:, 1] >= 0.0)
            valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (classes == 0)
            indexes = np.flatnonzero(valid)
            indexes = indexes[np.argsort(-scores[indexes], kind="stable")]
            best_box, confidence = None, 0.0
            if len(indexes):
                selected = int(indexes[0])
                best_box = [float(value) for value in boxes[selected, :4]]
                confidence = float(scores[selected])
            returned = confidence >= float(specification["detector_operating_confidence"])
            return_counts["at_0.25" if returned else "miss_at_0.25"] += 1
            camera = camera_models[row["camera_id"]]
            truth = (float(row["robot_x"]), float(row["robot_y"]))
            raw, equivalent = None, None
            if best_box is not None:
                bottom = (0.5 * (best_box[0] + best_box[2]), best_box[3])
                raw = camera.pixel_to_world_at_z(bottom[0], bottom[1], 0.0)
                if raw is not None:
                    converted = equivalent_position_measurement(
                        raw, ((1.0, 0.0), (0.0, 1.0)), camera,
                        truth, float(row["robot_yaw"]),
                    )
                    if converted is not None:
                        equivalent = tuple(float(value) for value in converted[0])
            expected = [finite(row, key) for key in ("expected_x0", "expected_y0", "expected_x1", "expected_y1")]
            record = {
                "pose_id": int(row["pose_id"]), "position_id": int(row["position_id"]),
                "position_key": row["position_key"], "block_id": row["block_id"],
                "heading_id": int(row["heading_id"]), "camera_id": row["camera_id"],
                "source_batch_id": row["source_batch_id"], "image": row["image"],
                "image_sha1": row["image_sha1"], "robot_x": truth[0], "robot_y": truth[1],
                "robot_yaw": float(row["robot_yaw"]), "camera_range_m": float(row["camera_range_m"]),
                "semantic_robot_pixels": int(float(row["semantic_robot_pixels"])),
                "reference_class": label_class, "reference_reasons": label_reasons,
                "reference_metrics": {key: clean_number(value) for key, value in label_metrics.items()},
                "mask_box_xyxy": None if mask_box is None else list(mask_box),
                "expected_box_xyxy": expected if all(value is not None for value in expected) else None,
                "raw_best_confidence": confidence, "best_box_xyxy": best_box,
                "candidate_count_above_0.001": int(len(indexes)),
                "detector_return_at_0.25": returned, "iou_with_mask_box": iou(best_box, mask_box),
                "raw_ground_xy": None if raw is None else [float(raw[0]), float(raw[1])],
                "silhouette_equivalent_xy_exact_prior": None if equivalent is None else list(equivalent),
                "raw_ground_error_m": distance(raw, truth),
                "silhouette_equivalent_error_m_exact_prior": distance(equivalent, truth),
            }
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            if index % 400 == 0 or index == len(rows):
                print(f"final audit inference {index}/{len(rows)}", file=sys.stderr, flush=True)
    elapsed = time.monotonic() - start_time
    manifest = {
        "schema": "thesis_stage09_final_audit_inference.v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "authorized_role": "final_audit",
        "views": len(rows), "positions": len({row["position_id"] for row in rows}),
        "blocks": len({row["block_id"] for row in rows}), "final_audit_accessed": True,
        "protocol": str(protocol_path.relative_to(REPO)), "protocol_sha256": sha256(protocol_path),
        "capture_manifest_sha256": sha256(capture_manifest_path), "capture_index_sha256": sha256(index_path),
        "checkpoint_sha256": sha256(weights), "records": records_path.name,
        "records_sha256": sha256(records_path), "reference_class_counts": dict(sorted(label_counts.items())),
        "detector_return_counts": dict(sorted(return_counts.items())), "inference_elapsed_s": elapsed,
        "mean_wall_time_per_view_ms": 1000.0 * elapsed / len(rows),
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": sha256(manifest_path)}) + "\n")
    os.replace(staging, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
