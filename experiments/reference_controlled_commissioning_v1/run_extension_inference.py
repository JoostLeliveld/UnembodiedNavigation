#!/usr/bin/env python3
"""Frozen-detector inference over the v4 edge-extension capture.

A sibling of run_all_master_inference.py.  That script is specific to the v3
master capture: it asserts 16 000 views, four fixed roles, and reuses 11 200
previously inferred records.  None of that holds for the extension, so this runs
the SAME frozen checkpoint, the same gate and the same projection over the
extension's own index and writes records in the identical schema.

Nothing in the v3 capture or its records is read or modified.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/reliability"))
sys.path.insert(0, str(REPO / "src/unav_common"))
from reliability.projection import camera_model_from_world  # noqa: E402

CAPTURE = REPO / "logs/thesis_final_pipeline_v1/master_capture_v4_edge_extension"
CHECKPOINT = (
    REPO / "logs/thesis_final_pipeline_v1/stage05_detector_training"
    / "imgsz960/upper_finetune/weights/best.pt"
)
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
EXPECTED_CHECKPOINT_SHA256 = "1e99afb5361a7c2599cb3f2863ebb04fd6f65bc1ee23996b989c9bba8b004f13"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict) -> tuple[int, str]:
    return int(row["pose_id"]), str(row["camera_id"])


def sanitize(source: dict, capture_row: dict) -> dict:
    """Identical field set to run_all_master_inference.sanitize."""
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    staging = args.out.resolve()
    if staging.exists():
        raise FileExistsError(staging)

    capture_manifest_path = CAPTURE / "capture_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    index_path = CAPTURE / "capture_index.csv"
    if capture_manifest.get("status") not in ("complete", "complete_with_failed_batches"):
        raise RuntimeError(f"extension capture not finished: {capture_manifest.get('status')}")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen detector checkpoint hash mismatch")

    with index_path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    # Failed batches carry no image; they are recorded in the manifest, not inferred.
    rows = [row for row in all_rows if row["capture_status"] == "ok"]
    if not rows:
        raise RuntimeError("no successfully captured rows")
    by_key = {key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise RuntimeError("duplicate physical frame identity in extension capture")

    images = [(CAPTURE / row["image"]).resolve() for row in rows]
    if any(not path.is_file() for path in images):
        raise RuntimeError("an extension RGB image is missing")

    staging.mkdir(parents=True)
    camera_models = {
        camera: camera_model_from_world(
            WORLD,
            include_name=next(r["camera_model"] for r in rows if r["camera_id"] == camera),
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
    for index, (row, prediction) in enumerate(
        zip(rows, predictions(), strict=True), start=1
    ):
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
            raw_ground = camera_models[row["camera_id"]].pixel_to_world_at_z(
                bottom_u, best_box[3], 0.0)
        source = {
            **row,
            "raw_best_confidence": best_confidence,
            "best_box_xyxy": best_box,
            "candidate_count_above_0.001": int(len(indexes)),
            "detector_return_at_0.25": detector_return,
            "raw_ground_xy": None if raw_ground is None else [
                float(raw_ground[0]), float(raw_ground[1])],
        }
        fresh[key(row)] = sanitize(source, row)
        if index % 500 == 0 or index == len(rows):
            print(f"extension inference {index}/{len(rows)} views", flush=True)

    records_path = staging / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            source = fresh[key(row)]
            if source.get("image_sha1") != row["image_sha1"]:
                raise RuntimeError(f"image identity mismatch for {key(row)}")
            handle.write(json.dumps(sanitize(source, row),
                                    separators=(",", ":"), allow_nan=False) + "\n")

    elapsed = time.monotonic() - start_time
    manifest = {
        "schema": "static_detector_inference_extension.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("Frozen-detector inference over the v4 edge extension. Same checkpoint, gate "
                 "and projection as run_all_master_inference.py; the v3 capture and its records "
                 "are neither read nor modified. pose_id/position_id are LOCAL to the extension "
                 "and collide with v3 by design."),
        "views": len(rows),
        "skipped_failed_capture_rows": len(all_rows) - len(rows),
        "positions": len({row["position_id"] for row in rows}),
        "blocks": len({row["block_id"] for row in rows}),
        "role_view_counts": dict(collections.Counter(row["dataset_split"] for row in rows)),
        "newly_inferred_views": len(fresh),
        "detector_return_counts": dict(sorted(return_counts.items())),
        "capture": str(CAPTURE.relative_to(REPO)),
        "capture_manifest": str(capture_manifest_path.relative_to(REPO)),
        "capture_index_sha256": sha256(index_path),
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "world": str(WORLD.relative_to(REPO)),
        "world_sha256": sha256(WORLD),
        "records": str(records_path.relative_to(REPO)),
        "records_sha256": sha256(records_path),
        "elapsed_s": round(elapsed, 3),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "views", "positions", "detector_return_counts", "elapsed_s")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
