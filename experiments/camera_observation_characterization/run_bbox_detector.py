#!/usr/bin/env python3
"""Run the frozen YOLO box detector once per unique characterization image."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import io
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for rel in ('src/perception', 'src/unav_common'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from perception.core.yolo_selection import (  # noqa: E402
    select_best_detection,
    target_class_ids,
)
from ultralytics import YOLO  # noqa: E402
from unav_common.capture_integrity import (
    atomic_csv, atomic_json, capture_lock, checked_bytes, checked_image, digest,
)


FIELDS = (
    'pose_id', 'position_id', 'heading_id', 'repetition_id', 'source_batch_id', 'camera_id',
    'image', 'image_sha1', 'capture_status', 'capture_error', 'inference_status',
    'capture_session_id', 'image_id', 'image_stamp_ns', 'detected', 'detector_clipped', 'n_candidates', 'confidence',
    'x0', 'y0', 'x1', 'y1', 'u_bbox_bottom', 'v_bbox_bottom',
)
CLIP_EPSILON_PX = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument(
        '--weights', type=Path,
        default=REPO / 'logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt',
    )
    parser.add_argument('--image-size', type=int, default=960)
    parser.add_argument('--confidence-threshold', type=float, default=0.25)
    parser.add_argument('--predict-confidence-floor', type=float, default=0.001)
    parser.add_argument('--iou-threshold', type=float, default=0.45)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='')
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    with capture_lock(capture):
        return run(args, capture)


def run(args, capture: Path) -> int:
    index, manifest = capture / 'capture_index.csv', capture / 'capture_manifest.json'
    output = capture / 'bbox_observations.csv'
    output_manifest = capture / 'bbox_detector_manifest.json'
    if output.exists() or output_manifest.exists():
        raise RuntimeError('detector outputs already exist; frozen outputs are never overwritten')
    manifest_bytes = manifest.read_bytes()
    capture_meta = json.loads(manifest_bytes)
    if capture_meta.get('status') not in ('complete', 'complete_with_failed_batches'):
        raise RuntimeError('capture must be complete before inference')
    index_bytes = checked_bytes(index, capture_meta.get('capture_index_sha256'))
    rows = list(csv.DictReader(io.StringIO(index_bytes.decode())))
    keys = [(r['pose_id'], r['repetition_id'], r['camera_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError('duplicate capture opportunity keys')
    sizes = {r['camera_id']: (int(r['image_width']), int(r['image_height']))
             for r in capture_meta['cameras']}
    if len(sizes) != len(capture_meta['cameras']) or not sizes:
        raise RuntimeError('duplicate or empty capture camera registry')
    unique = {}
    for row in rows:
        if row['camera_id'] not in sizes or row['capture_status'] not in ('ok', 'failed'):
            raise RuntimeError('unknown camera or acquisition outcome')
        if row['capture_status'] == 'ok':
            checked_image(capture, row, size=sizes[row['camera_id']])
            unique.setdefault(row['image_sha1'], row)
    weights = args.weights.expanduser().resolve()
    weight_bytes = weights.read_bytes()
    weights_hash = digest(weight_bytes)
    selections = {}
    items = list(unique.items())
    # Native YOLO takes a path; a private exact-byte copy prevents path-swap races.
    with tempfile.TemporaryDirectory(prefix='frozen_capture_detector_') as name:
        loaded_weights = Path(name) / weights.name
        loaded_weights.write_bytes(weight_bytes)
        model = YOLO(str(loaded_weights))
        target_ids = target_class_ids(getattr(model, 'names', {}), 'robot', -1)
        if target_ids == set():
            raise RuntimeError('frozen detector has no robot class')
        batch_size = max(int(args.batch_size), 1)
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            images = [checked_image(capture, row, size=sizes[row['camera_id']]) for _,row in chunk]
            kwargs = dict(source=images, imgsz=int(args.image_size),
                          conf=float(args.predict_confidence_floor), iou=float(args.iou_threshold),
                          batch=len(images), stream=False, verbose=False)
            if str(args.device).strip():
                kwargs['device'] = str(args.device).strip()
            results = list(model.predict(**kwargs))
            if len(results) != len(chunk):
                raise RuntimeError('detector returned wrong number of image results')
            for (image_hash, _), result in zip(chunk, results, strict=True):
                selections[image_hash] = select_best_detection(result, target_ids=target_ids,
                    confidence_threshold=float(args.confidence_threshold), use_masks=False,
                    mask_min_area=0.0, mask_bottom_band_px=3.0)
    written = []
    detected = 0
    for source in rows:
        row = {field: source.get(field, '') for field in FIELDS}
        if source['capture_status'] == 'failed':
            row.update(inference_status='not_attempted_acquisition_failed', detected='',
                       confidence='', n_candidates='')
            written.append(row)
            continue
        selection = selections[source['image_sha1']]
        box = selection['bbox_xyxy']
        hit = bool(selection['detected'])
        width, height = sizes[source['camera_id']]
        clipped = bool(hit and box is not None and (
            float(box[0]) <= CLIP_EPSILON_PX or float(box[1]) <= CLIP_EPSILON_PX
            or float(box[2]) >= width - CLIP_EPSILON_PX or float(box[3]) >= height - CLIP_EPSILON_PX))
        row.update(inference_status='detected' if hit else 'miss', detected=int(hit),
                   confidence=float(selection['confidence']), n_candidates=int(selection['n_candidates']),
                   detector_clipped=int(clipped),
                   u_bbox_bottom=float(selection['bbox_bottom_u']) if hit else math.nan,
                   v_bbox_bottom=float(selection['bbox_bottom_v']) if hit else math.nan)
        row.update({key: float(box[i]) if box is not None else math.nan
                    for i,key in enumerate(('x0','y0','x1','y1'))})
        written.append(row)
        detected += int(hit)
    # Do not certify a source/configuration that changed while inference was running.
    checked_bytes(index, digest(index_bytes))
    checked_bytes(manifest, digest(manifest_bytes))
    checked_bytes(weights, weights_hash)
    atomic_csv(output, written, FIELDS)
    metadata = dict(status='complete', schema='bbox_characterization_detector.v3',
        created_utc=datetime.now(timezone.utc).isoformat(),
        detector_script=str(Path(__file__).resolve()), detector_script_sha256=sha256(Path(__file__)),
        selection_helper=str(REPO / 'src/perception/perception/core/yolo_selection.py'),
        selection_helper_sha256=sha256(REPO / 'src/perception/perception/core/yolo_selection.py'),
        capture_index_sha256=digest(index_bytes), capture_manifest_sha256=digest(manifest_bytes),
        weights=str(weights), weights_sha256=weights_hash,
        preprocessing='verified decoded BGR uint8 arrays in original capture dimensions; native YOLO preprocessing',
        runtime=dict(image_size=int(args.image_size), confidence_threshold=float(args.confidence_threshold),
                     predict_confidence_floor=float(args.predict_confidence_floor),
                     iou_threshold=float(args.iou_threshold), class_name='robot', selected_point='bbox_bottom_centre'),
        attempt_rows=len(rows), opportunity_rows=len(rows),
        inference_rows=sum(r['capture_status']=='ok' for r in rows),
        acquisition_failed_rows=sum(r['capture_status']=='failed' for r in rows),
        unique_images=len(items), detected_rows=detected,
        availability_target='detector return conditional on successful acquisition; acquisition failures have unknown detector outcome',
        bbox_observations_sha256=sha256(output))
    atomic_json(output_manifest, metadata)
    print(json.dumps(dict(rows=len(rows), detected=detected, output=str(output)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
