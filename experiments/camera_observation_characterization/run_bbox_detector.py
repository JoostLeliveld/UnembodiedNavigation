#!/usr/bin/env python3
"""Run the frozen YOLO box detector once per unique characterization image."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for rel in ('src/perception',):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from perception.core.yolo_selection import (  # noqa: E402
    select_best_detection,
    target_class_ids,
)
from ultralytics import YOLO  # noqa: E402


FIELDS = (
    'pose_id', 'position_id', 'heading_id', 'repetition_id', 'source_batch_id', 'camera_id',
    'image', 'image_sha1', 'detected', 'detector_clipped', 'n_candidates', 'confidence',
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
    index = capture / 'capture_index.csv'
    manifest = capture / 'capture_manifest.json'
    weights = args.weights.expanduser().resolve()
    if not index.is_file() or not manifest.is_file() or not weights.is_file():
        raise RuntimeError('Capture index, capture manifest, and detector weights must exist')
    capture_meta = json.loads(manifest.read_text(encoding='utf-8'))
    if not str(capture_meta.get('status', '')).startswith('complete'):
        raise RuntimeError(f'Capture is not complete: {capture_meta.get("status")!r}')

    rows = list(csv.DictReader(index.open(encoding='utf-8')))
    image_size_by_camera = {
        item['camera_id']: (int(item['image_width']), int(item['image_height']))
        for item in capture_meta['cameras']
    }
    usable = [row for row in rows if row['capture_status'] == 'ok' and row['image']]
    unique_by_hash = {}
    for row in usable:
        unique_by_hash.setdefault(row['image_sha1'], capture / row['image'])
    hashes = list(unique_by_hash)
    paths = [unique_by_hash[value] for value in hashes]

    model = YOLO(str(weights))
    target_ids = target_class_ids(getattr(model, 'names', {}), 'robot', -1)
    if target_ids == set():
        raise RuntimeError(f'Frozen detector has no robot class: {getattr(model, "names", {})!r}')
    selected_by_hash = {}
    batch_size = max(int(args.batch_size), 1)
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        kwargs = {
            'source': [str(path) for path in batch],
            'imgsz': int(args.image_size),
            'conf': float(args.predict_confidence_floor),
            'iou': float(args.iou_threshold),
            'batch': len(batch),
            'stream': False,
            'verbose': False,
        }
        if str(args.device).strip():
            kwargs['device'] = str(args.device).strip()
        results = list(model.predict(**kwargs))
        if len(results) != len(batch):
            raise RuntimeError(f'YOLO returned {len(results)} results for {len(batch)} images')
        for image_hash, result in zip(hashes[start:start + batch_size], results, strict=True):
            selected_by_hash[image_hash] = select_best_detection(
                result,
                target_ids=target_ids,
                confidence_threshold=float(args.confidence_threshold),
                use_masks=False,
                mask_min_area=0.0,
                mask_bottom_band_px=3.0,
            )
        done = min(start + len(batch), len(paths))
        if done % 256 == 0 or done == len(paths):
            print(f'inferred {done}/{len(paths)} unique images', flush=True)

    output = capture / 'bbox_observations.csv'
    detected = 0
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for source in usable:
            selection = selected_by_hash[source['image_sha1']]
            box = selection['bbox_xyxy']
            is_detected = bool(selection['detected'])
            width, height = image_size_by_camera[source['camera_id']]
            is_clipped = bool(
                is_detected and box is not None
                and (float(box[0]) <= CLIP_EPSILON_PX
                     or float(box[1]) <= CLIP_EPSILON_PX
                     or float(box[2]) >= width - CLIP_EPSILON_PX
                     or float(box[3]) >= height - CLIP_EPSILON_PX)
            )
            detected += int(is_detected)
            row = {
                'pose_id': source['pose_id'],
                'position_id': source['position_id'],
                'heading_id': source['heading_id'],
                'repetition_id': source['repetition_id'],
                'source_batch_id': source.get(
                    'source_batch_id',
                    f"pose_{int(source['pose_id']):06d}_r{int(source['repetition_id']):02d}",
                ),
                'camera_id': source['camera_id'],
                'image': source['image'],
                'image_sha1': source['image_sha1'],
                'detected': int(is_detected),
                'detector_clipped': int(is_clipped),
                'n_candidates': int(selection['n_candidates']),
                'confidence': float(selection['confidence']),
                'x0': float(box[0]) if box is not None else math.nan,
                'y0': float(box[1]) if box is not None else math.nan,
                'x1': float(box[2]) if box is not None else math.nan,
                'y1': float(box[3]) if box is not None else math.nan,
                'u_bbox_bottom': float(selection['bbox_bottom_u']) if is_detected else math.nan,
                'v_bbox_bottom': float(selection['bbox_bottom_v']) if is_detected else math.nan,
            }
            writer.writerow(row)

    detector_manifest = {
        'status': 'complete',
        'schema': 'bbox_characterization_detector.v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'detector_script': str(Path(__file__).resolve()),
        'detector_script_sha256': sha256(Path(__file__).resolve()),
        'selection_helper': str(
            (REPO / 'src/perception/perception/core/yolo_selection.py').resolve()
        ),
        'selection_helper_sha256': sha256(
            (REPO / 'src/perception/perception/core/yolo_selection.py').resolve()
        ),
        'capture_index_sha256': sha256(index),
        'capture_manifest_sha256': sha256(manifest),
        'weights': str(weights),
        'weights_sha256': sha256(weights),
        'runtime': {
            'image_size': int(args.image_size),
            'confidence_threshold': float(args.confidence_threshold),
            'predict_confidence_floor': float(args.predict_confidence_floor),
            'iou_threshold': float(args.iou_threshold),
            'class_name': 'robot',
            'selected_point': 'bbox_bottom_centre',
            'detector_clipped_definition': (
                'selected box touches an image boundary within 0.5 px rounding tolerance'
            ),
        },
        'attempt_rows': len(usable),
        'unique_images': len(paths),
        'detected_rows': detected,
        'bbox_observations_sha256': sha256(output),
    }
    (capture / 'bbox_detector_manifest.json').write_text(
        json.dumps(detector_manifest, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'rows': len(usable), 'unique_images': len(paths), 'detected': detected,
        'output': str(output),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
