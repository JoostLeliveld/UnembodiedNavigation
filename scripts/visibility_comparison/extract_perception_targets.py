#!/usr/bin/env python3
"""Run the paper YOLO detector over captured visibility samples.

Input is the `samples.csv` produced by `capture_visibility_samples.py`.
Output is `perception_targets.csv`, the detector-score table consumed by
`build_gp_targets.py`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import (
    CURRENT_CAPTURE_DIR,
    CURRENT_TARGETS_DIR,
    PERCEPTION_TARGET_COLUMNS,
    ensure_repo_python_paths,
    parse_float,
    read_csv_rows,
    write_csv,
    write_manifest,
)


SELECTED_PIXEL_SOURCE_CODE = {
    'none': 0,
    'bbox_bottom': 1,
    'mask_bottom': 2,
}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_sample_image(capture_dir: Path, image_value: str) -> Path:
    image_path = Path(str(image_value or '').strip())
    if image_path.is_absolute():
        return image_path
    return capture_dir / image_path


def _camera_relative_bearing_deg(row: dict[str, str], camera_pos: list[float] | None) -> str:
    if not camera_pos or len(camera_pos) < 2:
        return ''
    x = parse_float(row.get('x', ''), math.nan)
    y = parse_float(row.get('y', ''), math.nan)
    theta = parse_float(row.get('theta', ''), math.nan)
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(theta)):
        return ''
    bearing = math.atan2(float(camera_pos[1]) - y, float(camera_pos[0]) - x)
    rel = math.atan2(math.sin(bearing - theta), math.cos(bearing - theta))
    return f'{math.degrees(rel):.8f}'


def _bbox_json(selection: dict) -> str:
    bbox = selection.get('bbox_xyxy')
    if bbox is None:
        return ''
    values = np.asarray(bbox, dtype=float).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)):
        return ''
    return json.dumps([float(v) for v in values.tolist()], separators=(',', ':'))


def _format_float(value: object, default: str = '') -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(val):
        return default
    return f'{val:.8f}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', default=str(CURRENT_CAPTURE_DIR))
    parser.add_argument('--samples', default='')
    parser.add_argument('--out', default=str(CURRENT_TARGETS_DIR))
    parser.add_argument('--model', required=True, help='Path to the trained YOLO .pt model')
    parser.add_argument('--device', default='')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-threshold', type=float, default=0.10)
    parser.add_argument('--iou-threshold', type=float, default=0.45)
    parser.add_argument('--target-class', default='robot')
    parser.add_argument('--class-id', type=int, default=-1)
    parser.add_argument('--use-masks', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--min-mask-area-px', type=float, default=12.0)
    parser.add_argument('--mask-bottom-band-px', type=float, default=3.0)
    args = parser.parse_args()

    ensure_repo_python_paths()
    import cv2
    from ultralytics import YOLO
    from perception.core.yolo_selection import select_best_detection, target_class_ids

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    samples_path = Path(args.samples).expanduser().resolve() if args.samples else capture_dir / 'samples.csv'
    if not samples_path.is_file():
        raise RuntimeError(f'Samples CSV not found: {samples_path}')

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise RuntimeError(f'YOLO model not found: {model_path}')

    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_manifest = _load_json(capture_dir / 'capture_manifest.json')
    camera_pos = capture_manifest.get('camera_pos')
    model = YOLO(str(model_path))
    target_ids = target_class_ids(getattr(model, 'names', {}), args.target_class, args.class_id)
    if target_ids == set():
        raise RuntimeError(
            f'class filter does not match any YOLO class names: '
            f'target_class={args.target_class!r}, class_id={args.class_id}, '
            f'names={getattr(model, "names", {})!r}'
        )

    rows = read_csv_rows(samples_path)
    out_rows: list[dict[str, str]] = []
    raw_scores: list[float] = []
    detected_count = 0

    for row in rows:
        image_path = _resolve_sample_image(capture_dir, row.get('image_path', ''))
        if not image_path.is_file():
            raise RuntimeError(f'Captured image not found for sample {row.get("sample_id")}: {image_path}')
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f'Failed to read image for sample {row.get("sample_id")}: {image_path}')

        predict_kwargs = {
            'source': image_bgr,
            'imgsz': int(args.imgsz),
            'conf': 0.0,
            'iou': float(args.iou_threshold),
            'verbose': False,
        }
        if str(args.device).strip():
            predict_kwargs['device'] = str(args.device).strip()
        results = model.predict(**predict_kwargs)
        selection = {}
        if results:
            selection = select_best_detection(
                results[0],
                target_ids=target_ids,
                confidence_threshold=float(args.conf_threshold),
                use_masks=bool(args.use_masks),
                mask_min_area=float(args.min_mask_area_px),
                mask_bottom_band_px=float(args.mask_bottom_band_px),
            )

        raw_score = float(selection.get('raw_best_score', 0.0) or 0.0)
        selected_score = float(selection.get('selected_score', raw_score) or raw_score)
        detected = bool(selection.get('detected_after_threshold', False))
        if detected:
            detected_count += 1
        raw_scores.append(raw_score)
        source = str(selection.get('selected_pixel_source', 'none') or 'none')

        out_rows.append({
            'sample_id': str(row.get('sample_id', '')).strip(),
            'x': str(row.get('x', '')).strip(),
            'y': str(row.get('y', '')).strip(),
            'theta': str(row.get('theta', '')).strip(),
            'image_path': str(row.get('image_path', '')).strip(),
            'camera_relative_bearing_deg': _camera_relative_bearing_deg(row, camera_pos),
            'yolo_detected_after_threshold': '1' if detected else '0',
            'yolo_score_raw': f'{raw_score:.8f}',
            'yolo_raw_best_score': f'{raw_score:.8f}',
            'yolo_score_selected': f'{selected_score:.8f}',
            'yolo_selected_score': f'{selected_score:.8f}',
            'yolo_best_class_id': _format_float(selection.get('best_class_id', math.nan)),
            'yolo_selected_class_id': _format_float(selection.get('class_id', math.nan)),
            'yolo_num_target_candidates': str(int(selection.get('n_candidates', 0) or 0)),
            'yolo_mask_area': _format_float(selection.get('mask_area', math.nan)),
            'yolo_bbox_area': _format_float(
                max(float(selection.get('bbox_xyxy', [math.nan, math.nan, math.nan, math.nan])[2])
                    - float(selection.get('bbox_xyxy', [math.nan, math.nan, math.nan, math.nan])[0]), 0.0)
                * max(float(selection.get('bbox_xyxy', [math.nan, math.nan, math.nan, math.nan])[3])
                      - float(selection.get('bbox_xyxy', [math.nan, math.nan, math.nan, math.nan])[1]), 0.0)
                if selection.get('bbox_xyxy') is not None else math.nan
            ),
            'yolo_bbox_xyxy': _bbox_json(selection),
            'yolo_bottom_u': _format_float(selection.get('selected_u', math.nan)),
            'yolo_bottom_v': _format_float(selection.get('selected_v', math.nan)),
            'yolo_selected_pixel_source': source,
            'yolo_selected_pixel_source_code': str(SELECTED_PIXEL_SOURCE_CODE.get(source, 0)),
            'oracle_visible': str(row.get('oracle_visible', '')).strip(),
            'oracle_bottom_u': str(row.get('oracle_bottom_u', '')).strip(),
            'oracle_bottom_v': str(row.get('oracle_bottom_v', '')).strip(),
            'camera_frame': str(row.get('camera_frame', '')).strip(),
        })

    targets_path = output_dir / 'perception_targets.csv'
    write_csv(targets_path, PERCEPTION_TARGET_COLUMNS, out_rows)
    finite_scores = [v for v in raw_scores if math.isfinite(v)]
    manifest = {
        'samples_csv': str(samples_path),
        'capture_dir': str(capture_dir),
        'capture_metadata': capture_manifest,
        'yolo_model': str(model_path),
        'yolo_device': str(args.device).strip(),
        'yolo_imgsz': int(args.imgsz),
        'yolo_conf_threshold': float(args.conf_threshold),
        'yolo_iou_threshold': float(args.iou_threshold),
        'yolo_target_class': str(args.target_class),
        'yolo_class_id': int(args.class_id),
        'yolo_use_masks': bool(args.use_masks),
        'yolo_min_mask_area_px': float(args.min_mask_area_px),
        'yolo_mask_bottom_band_px': float(args.mask_bottom_band_px),
        'yolo_summary': {
            'sample_count': int(len(out_rows)),
            'detected_after_threshold_count': int(detected_count),
            'detection_rate': float(detected_count / len(out_rows)) if out_rows else math.nan,
            'mean_raw_score': float(np.mean(finite_scores)) if finite_scores else math.nan,
            'min_raw_score': float(np.min(finite_scores)) if finite_scores else math.nan,
            'max_raw_score': float(np.max(finite_scores)) if finite_scores else math.nan,
        },
        'perception_targets_csv': str(targets_path),
    }
    write_manifest(output_dir / 'manifest.json', manifest)
    print(f'Wrote perception targets to {targets_path}')
    print(f'Detections above threshold: {detected_count}/{len(out_rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
