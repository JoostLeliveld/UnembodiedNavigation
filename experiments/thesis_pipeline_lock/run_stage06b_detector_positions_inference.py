#!/usr/bin/env python3
"""Run the frozen detector once over the authorized DETECTOR-SPLIT static images.

Sibling of run_stage06_commissioning_inference.py. Every integrity check is kept:
stage-05 lock, detector checkpoint hash, capture-index hash, no-overwrite staging.
The ONLY differences are that authorized_role is a list of the two detector splits
and the audit drives are never referenced.

These positions trained the detector. Records produced here are contaminated by
construction and must be reported as such.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from export_detector_dataset import classify
from reliability.projection import camera_model_from_world
from reliability.silhouette_observation import equivalent_position_measurement


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def iou(box: list[float] | None, gt: tuple[int, int, int, int] | None) -> float | None:
    if box is None or gt is None:
        return None
    x0, y0 = max(box[0], gt[0]), max(box[1], gt[1])
    x1, y1 = min(box[2], gt[2]), min(box[3], gt[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_gt = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    return intersection / max(area_box + area_gt - intersection, 1e-12)


def distance(point: tuple[float, float] | None, truth: tuple[float, float]) -> float | None:
    if point is None:
        return None
    return float(math.hypot(point[0] - truth[0], point[1] - truth[1]))


def clean_number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_commissioning_inference":
        raise RuntimeError("Stage-06 protocol is not frozen")
    authorized_roles = protocol["inputs"]["authorized_role"]
    if not isinstance(authorized_roles, list):
        authorized_roles = [authorized_roles]
    if sorted(authorized_roles) != ["detector_fit", "detector_validation"]:
        raise RuntimeError("Only detector_fit and detector_validation are authorized here")

    pipeline_path = REPO / "experiments/thesis_pipeline_lock/pipeline_lock.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    stage05 = next(stage for stage in pipeline["stages"] if stage["id"] == "05_detector_training")
    if stage05.get("status") != "locked" or stage05.get("lock_id") != protocol["stage05_lock_id"]:
        raise RuntimeError("Stage 05 is not locked to the detector named by this protocol")
    stage05_manifest = REPO / stage05["manifest"]
    if sha256(stage05_manifest) != stage05["manifest_sha256"]:
        raise RuntimeError("Stage-05 manifest hash drift")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    capture = (REPO / protocol["inputs"]["master_capture"]).resolve()
    capture_manifest_path = capture / "capture_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    index_path = capture / "capture_index.csv"
    if capture_manifest.get("status") != "complete":
        raise RuntimeError("Master capture is incomplete")
    if capture_manifest.get("capture_index_sha256") != sha256(index_path):
        raise RuntimeError("Master capture index hash mismatch")
    with index_path.open(newline="", encoding="utf-8") as handle:
        authorized = [
            row for row in csv.DictReader(handle)
            if row["dataset_split"] in set(authorized_roles)
        ]
    expected_views = int(protocol["inputs"]["authorized_views"])
    if len(authorized) != expected_views:
        raise RuntimeError(f"Expected {expected_views} detector-split views, found {len(authorized)}")
    if {row["camera_id"] for row in authorized} != set(CAMERAS):
        raise RuntimeError("Commissioning population lacks a frozen camera")
    images = [(capture / row["image"]).resolve() for row in authorized]
    if any(not path.is_file() for path in images):
        raise RuntimeError("An authorized commissioning image is missing")

    weights = (REPO / protocol["inputs"]["detector_checkpoint"]).resolve()
    if sha256(weights) != protocol["inputs"]["detector_checkpoint_sha256"]:
        raise RuntimeError("Detector checkpoint hash mismatch")
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    camera_models = {
        camera: camera_model_from_world(world, include_name=next(
            row["camera_model"] for row in authorized if row["camera_id"] == camera
        ))
        for camera in CAMERAS
    }
    label_protocol_path = REPO / protocol["inputs"]["label_protocol"]
    label_protocol = json.loads(label_protocol_path.read_text(encoding="utf-8"))
    label_contract = label_protocol["detector_label_contract"]["positive_requires_all"]

    model = YOLO(str(weights))
    batch = int(args.batch)
    if batch <= 0:
        raise ValueError("batch must be positive")
    start_time = time.monotonic()

    def predictions():
        for start in range(0, len(images), batch):
            chunk = images[start:start + batch]
            yield from model.predict(
                source=[str(path) for path in chunk],
                imgsz=int(protocol["inputs"]["detector_imgsz"]),
                conf=float(protocol["inputs"]["detector_prediction_floor"]),
                iou=float(protocol["inputs"]["detector_iou"]),
                max_det=20,
                device="0",
                batch=batch,
                workers=2,
                verbose=False,
                save=False,
                stream=False,
            )

    label_counts: collections.Counter[str] = collections.Counter()
    return_counts: collections.Counter[str] = collections.Counter()
    records_path = staging / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for view_index, (row, result) in enumerate(
            zip(authorized, predictions(), strict=True), start=1
        ):
            label_class, label_reasons, mask_box, label_metrics = classify(
                row, 1280, 720, label_contract
            )
            label_counts[label_class] += 1
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
            scores = result.boxes.conf.detach().cpu().numpy().astype(float)
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            valid = np.isfinite(scores) & np.all(np.isfinite(boxes), axis=1)
            valid &= (boxes[:, 0] >= 0.0) & (boxes[:, 1] >= 0.0)
            valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            valid &= classes == 0
            indexes = np.flatnonzero(valid)
            indexes = indexes[np.argsort(-scores[indexes], kind="stable")]
            best_box = None
            best_confidence = 0.0
            if len(indexes):
                best = int(indexes[0])
                best_box = [float(value) for value in boxes[best, :4]]
                best_confidence = float(scores[best])
            detector_return = best_confidence >= 0.25
            return_counts["at_0.25" if detector_return else "miss_at_0.25"] += 1

            camera = camera_models[row["camera_id"]]
            truth = (float(row["robot_x"]), float(row["robot_y"]))
            raw_ground = None
            equivalent = None
            if best_box is not None:
                bottom = (0.5 * (best_box[0] + best_box[2]), best_box[3])
                raw_ground = camera.pixel_to_world_at_z(bottom[0], bottom[1], 0.0)
                if raw_ground is not None:
                    converted = equivalent_position_measurement(
                        raw_ground,
                        ((1.0, 0.0), (0.0, 1.0)),
                        camera,
                        truth,
                        float(row["robot_yaw"]),
                    )
                    if converted is not None:
                        equivalent = (float(converted[0][0]), float(converted[0][1]))

            expected = [
                finite(row, key)
                for key in ("expected_x0", "expected_y0", "expected_x1", "expected_y1")
            ]
            record = {
                "pose_id": int(row["pose_id"]),
                "position_id": int(row["position_id"]),
                "position_key": row["position_key"],
                "block_id": row["block_id"],
                "heading_id": int(row["heading_id"]),
                "camera_id": row["camera_id"],
                "source_batch_id": row["source_batch_id"],
                "image": row["image"],
                "image_sha1": row["image_sha1"],
                "robot_x": truth[0],
                "robot_y": truth[1],
                "robot_yaw": float(row["robot_yaw"]),
                "camera_range_m": float(row["camera_range_m"]),
                "semantic_robot_pixels": int(float(row["semantic_robot_pixels"])),
                "reference_class": label_class,
                "reference_reasons": label_reasons,
                "reference_metrics": {
                    key: clean_number(value) for key, value in label_metrics.items()
                },
                "mask_box_xyxy": None if mask_box is None else list(mask_box),
                "expected_box_xyxy": expected if all(value is not None for value in expected) else None,
                "raw_best_confidence": best_confidence,
                "best_box_xyxy": best_box,
                "candidate_count_above_0.001": int(len(indexes)),
                "detector_return_at_0.25": detector_return,
                "iou_with_mask_box": iou(best_box, mask_box),
                "raw_ground_xy": None if raw_ground is None else [float(raw_ground[0]), float(raw_ground[1])],
                "silhouette_equivalent_xy_exact_prior": None if equivalent is None else list(equivalent),
                "raw_ground_error_m": distance(raw_ground, truth),
                "silhouette_equivalent_error_m_exact_prior": distance(equivalent, truth),
            }
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            if view_index % 500 == 0 or view_index == len(authorized):
                print(
                    f"detector-split inference {view_index}/{len(authorized)} views",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed = time.monotonic() - start_time
    manifest = {
        "schema": "thesis_stage06_commissioning_inference.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_role": sorted(authorized_roles),
        "views": len(authorized),
        "positions": len({row["position_id"] for row in authorized}),
        "blocks": len({row["block_id"] for row in authorized}),
        "final_audit_rgb_or_mask_accessed": False,
        "protocol": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": sha256(protocol_path),
        "stage05_manifest": str(stage05_manifest.relative_to(REPO)),
        "stage05_manifest_sha256": sha256(stage05_manifest),
        "capture_manifest": str(capture_manifest_path.relative_to(REPO)),
        "capture_manifest_sha256": sha256(capture_manifest_path),
        "capture_index_sha256": sha256(index_path),
        "label_protocol_sha256": sha256(label_protocol_path),
        "checkpoint": str(weights.relative_to(REPO)),
        "checkpoint_sha256": sha256(weights),
        "records": "records.jsonl",
        "records_sha256": sha256(records_path),
        "reference_class_counts": dict(sorted(label_counts.items())),
        "detector_return_counts": dict(sorted(return_counts.items())),
        "inference_elapsed_s": elapsed,
        "mean_wall_time_per_view_ms": 1000.0 * elapsed / len(authorized),
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (staging / ".complete").write_text(
        json.dumps({"manifest_sha256": sha256(manifest_path)}) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
