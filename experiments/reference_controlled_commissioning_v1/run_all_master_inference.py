#!/usr/bin/env python3
"""Complete frozen-detector inference for all 400 master-capture positions.

Existing frozen outputs are reused by physical image identity.  Only the detector-fit
and detector-validation images that lack an inference record are evaluated.  The output
contains raw detector boxes and their ground-plane projections; no geometric visibility,
semantic silhouette, robot belief, or estimator quantity is computed or consumed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/reliability"))
sys.path.insert(0, str(REPO / "src/unav_common"))
from reliability.projection import camera_model_from_world


CAPTURE = REPO / "logs/thesis_final_pipeline_v1/master_capture"
CHECKPOINT = (
    REPO / "logs/thesis_final_pipeline_v1/stage05_detector_training"
    / "imgsz960/upper_finetune/weights/best.pt"
)
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
EXISTING = (
    REPO / "logs/thesis_final_pipeline_v1/stage06_detector_gate/commissioning_inference/records.jsonl",
    REPO / "logs/thesis_final_pipeline_v1/stage09_navigation/final_audit_inference/records.jsonl",
)
ROLES = ("detector_fit", "detector_validation", "commissioning_fit", "final_audit")
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
EXPECTED_CHECKPOINT_SHA256 = "1e99afb5361a7c2599cb3f2863ebb04fd6f65bc1ee23996b989c9bba8b004f13"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict) -> tuple[int, str]:
    return int(row["pose_id"]), str(row["camera_id"])


def sanitize(source: dict, capture_row: dict) -> dict:
    """Retain only fields used by the perception-only correction pipeline."""
    names = (
        "pose_id", "position_id", "position_key", "block_id", "heading_id",
        "camera_id", "source_batch_id", "image", "image_sha1", "robot_x",
        "robot_y", "robot_yaw", "camera_range_m", "raw_best_confidence",
        "best_box_xyxy", "candidate_count_above_0.001", "detector_return_at_0.25",
        "raw_ground_xy",
    )
    result = {name: source.get(name) for name in names}
    result["dataset_split"] = capture_row["dataset_split"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    capture_manifest_path = CAPTURE / "capture_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    index_path = CAPTURE / "capture_index.csv"
    if capture_manifest.get("status") != "complete":
        raise RuntimeError("master capture is incomplete")
    if capture_manifest.get("capture_index_sha256") != sha256(index_path):
        raise RuntimeError("master capture index hash mismatch")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen detector checkpoint hash mismatch")
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16000 or {row["dataset_split"] for row in rows} != set(ROLES):
        raise RuntimeError("unexpected master-capture population")
    by_key = {key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise RuntimeError("duplicate physical frame identity in master capture")

    existing: dict[tuple[int, str], dict] = {}
    for path in EXISTING:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    existing[key(record)] = record
    if len(existing) != 11200:
        raise RuntimeError(f"expected 11,200 reusable records, got {len(existing):,}")
    missing_rows = [row for row in rows if key(row) not in existing]
    if len(missing_rows) != 4800:
        raise RuntimeError(f"expected 4,800 missing detector-role records, got {len(missing_rows):,}")
    images = [(CAPTURE / row["image"]).resolve() for row in missing_rows]
    if any(not path.is_file() for path in images):
        raise RuntimeError("a master-capture RGB image is missing")

    camera_models = {
        camera: camera_model_from_world(
            WORLD,
            include_name=next(row["camera_model"] for row in rows if row["camera_id"] == camera),
        )
        for camera in CAMERAS
    }
    model = YOLO(str(CHECKPOINT))
    batch = int(args.batch)
    if batch <= 0:
        raise ValueError("batch must be positive")
    start_time = time.monotonic()

    def predictions():
        for start in range(0, len(images), batch):
            chunk = images[start:start + batch]
            yield from model.predict(
                source=[str(path) for path in chunk], imgsz=960, conf=0.001,
                iou=0.7, max_det=20, device=args.device, batch=batch,
                workers=2, verbose=False, save=False, stream=False,
            )

    fresh: dict[tuple[int, str], dict] = {}
    return_counts: collections.Counter[str] = collections.Counter()
    for index, (row, prediction) in enumerate(zip(missing_rows, predictions(), strict=True), start=1):
        boxes = prediction.boxes.xyxy.detach().cpu().numpy().astype(float)
        scores = prediction.boxes.conf.detach().cpu().numpy().astype(float)
        classes = prediction.boxes.cls.detach().cpu().numpy().astype(int)
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
        raw_ground = None
        if best_box is not None:
            bottom_u = 0.5 * (best_box[0] + best_box[2])
            raw_ground = camera_models[row["camera_id"]].pixel_to_world_at_z(bottom_u, best_box[3], 0.0)
        source = {
            **row,
            "raw_best_confidence": best_confidence,
            "best_box_xyxy": best_box,
            "candidate_count_above_0.001": int(len(indexes)),
            "detector_return_at_0.25": detector_return,
            "raw_ground_xy": None if raw_ground is None else [float(raw_ground[0]), float(raw_ground[1])],
        }
        fresh[key(row)] = sanitize(source, row)
        if index % 500 == 0 or index == len(missing_rows):
            print(f"all-master inference {index}/{len(missing_rows)} new views", flush=True)

    records_path = staging / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            source = existing.get(key(row)) or fresh[key(row)]
            if source.get("image_sha1") != row["image_sha1"]:
                raise RuntimeError(f"image identity mismatch for {key(row)}")
            handle.write(json.dumps(sanitize(source, row), separators=(",", ":"), allow_nan=False) + "\n")

    elapsed = time.monotonic() - start_time
    manifest = {
        "schema": "static_detector_inference_all_master.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_roles": list(ROLES),
        "views": len(rows),
        "positions": len({row["position_id"] for row in rows}),
        "blocks": len({row["block_id"] for row in rows}),
        "role_view_counts": dict(collections.Counter(row["dataset_split"] for row in rows)),
        "reused_views": len(existing),
        "newly_inferred_views": len(fresh),
        "new_detector_return_counts": dict(sorted(return_counts.items())),
        "capture_manifest": str(capture_manifest_path.relative_to(REPO)),
        "capture_manifest_sha256": sha256(capture_manifest_path),
        "capture_index_sha256": sha256(index_path),
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "records": "records.jsonl",
        "records_sha256": sha256(records_path),
        "inference_elapsed_s_for_new_views": elapsed,
        "model_inputs": ["raw_rgb"],
        "model_outputs": ["frozen_detector_box", "box_bottom_ground_projection"],
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
