#!/usr/bin/env python3
"""Run a simple visual smoke test with a local YOLO model on one image or a folder."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def _iter_images(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return [path for path in sorted(src.rglob('*')) if path.suffix.lower() in IMAGE_EXTS]


def _target_class_ids(names, class_name: str, class_id: int):
    if int(class_id) >= 0:
        return {int(class_id)}
    target = str(class_name or '').strip().lower()
    if target in ('', '*', 'any', 'all'):
        return None
    if isinstance(names, dict):
        return {int(idx) for idx, name in names.items() if str(name).strip().lower() == target}
    return set()


def _mask_bottom(points: np.ndarray, band_px: float) -> tuple[float, float]:
    v_bottom = float(np.max(points[:, 1]))
    band = points[points[:, 1] >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(points[:, 0])), v_bottom
    return float(np.mean(band[:, 0])), v_bottom


def _collect_detections(result, target_ids, confidence_threshold: float):
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    valid = np.isfinite(conf) & (conf >= float(confidence_threshold))
    if target_ids is not None:
        valid &= np.asarray([int(c) in target_ids for c in cls], dtype=bool)
    idxs = np.flatnonzero(valid)
    if idxs.size == 0:
        return []
    detections = []
    masks = getattr(result, 'masks', None)
    mask_polygons = getattr(masks, 'xy', None) if masks is not None else None
    for idx in idxs:
        idx = int(idx)
        polygon = None
        if mask_polygons is not None and idx < len(mask_polygons):
            pts = np.asarray(mask_polygons[idx], dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                polygon = pts[:, :2]
        detections.append({
            'bbox': xyxy[idx].astype(float),
            'confidence': float(conf[idx]),
            'class_id': int(cls[idx]),
            'polygon': polygon,
        })
    detections.sort(key=lambda item: float(item['confidence']), reverse=True)
    return detections


def _draw_candidate_overlay(preview: np.ndarray, detection) -> None:
    x0, y0, x1, y1 = [float(v) for v in detection['bbox']]
    cv2.rectangle(
        preview,
        (int(round(x0)), int(round(y0))),
        (int(round(x1)), int(round(y1))),
        (180, 120, 255),
        1,
    )
    polygon = detection['polygon']
    if polygon is not None:
        pts = np.asarray(polygon, dtype=float).reshape(-1, 2).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [pts], True, (200, 120, 255), 1)
    cv2.putText(
        preview,
        f'{detection["class_id"]}:{detection["confidence"]:.2f}',
        (int(round(x0)), max(14, int(round(y0)) - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 220, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_preview(image_bgr: np.ndarray, detections, *, mask_min_area: float, mask_bottom_band_px: float) -> tuple[np.ndarray, dict]:
    preview = image_bgr.copy()
    for detection in detections:
        _draw_candidate_overlay(preview, detection)

    if not detections:
        return preview, {
            'detected': 0,
            'n_candidates': 0,
            'mask_available': 0,
            'selected_pixel_source': 'none',
            'selected_u': math.nan,
            'selected_v': math.nan,
            'confidence': 0.0,
        }

    detection = detections[0]
    bbox = detection['bbox']
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cv2.rectangle(preview, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), (255, 200, 0), 2)

    polygon = detection['polygon']
    mask_available = 0
    mask_area = math.nan
    if polygon is not None:
        pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
        area = float(abs(0.5 * (np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))))
        if area >= float(mask_min_area):
            mask_available = 1
            mask_area = area
            pts_int = pts.astype(np.int32).reshape(-1, 1, 2)
            overlay = preview.copy()
            cv2.fillPoly(overlay, [pts_int], (0, 180, 0))
            preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0.0)
            cv2.polylines(preview, [pts_int], True, (0, 200, 0), 2)
            selected_u, selected_v = _mask_bottom(pts, mask_bottom_band_px)
            selected_source = 'mask_bottom'
        else:
            selected_u = float(0.5 * (x0 + x1))
            selected_v = float(y1)
            selected_source = 'bbox_bottom'
    else:
        selected_u = float(0.5 * (x0 + x1))
        selected_v = float(y1)
        selected_source = 'bbox_bottom'

    cv2.circle(preview, (int(round(selected_u)), int(round(selected_v))), 5, (0, 255, 255), -1)
    cv2.putText(
        preview,
        f'score={detection["confidence"]:.2f} source={selected_source}',
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview, {
        'detected': 1,
        'n_candidates': int(len(detections)),
        'mask_available': int(mask_available),
        'mask_area': float(mask_area),
        'selected_pixel_source': selected_source,
        'selected_u': float(selected_u),
        'selected_v': float(selected_v),
        'confidence': float(detection['confidence']),
        'bbox_xyxy': [float(x0), float(y0), float(x1), float(y1)],
        'class_id': int(detection['class_id']),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Visually test whether a local YOLO model segments the robot at all.')
    parser.add_argument('--model', required=True, help='Local YOLO .pt model path')
    parser.add_argument('--input', required=True, help='Input image or folder of images')
    parser.add_argument('--out', required=True, help='Output folder for previews and summary.json')
    parser.add_argument('--conf-threshold', type=float, default=0.10)
    parser.add_argument('--iou-threshold', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--class-name', default='any')
    parser.add_argument('--class-id', type=int, default=-1)
    parser.add_argument('--mask-min-area', type=float, default=12.0)
    parser.add_argument('--mask-bottom-band-px', type=float, default=3.0)
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise RuntimeError(f'Model path does not exist: {model_path}')
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise RuntimeError(f'Input path does not exist: {input_path}')
    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        raise RuntimeError(f'Output directory already exists: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=False)

    image_paths = _iter_images(input_path)
    if not image_paths:
        raise RuntimeError(f'No images found in {input_path}')

    model = YOLO(str(model_path))
    target_ids = _target_class_ids(getattr(model, 'names', {}), args.class_name, int(args.class_id))
    if target_ids == set():
        raise RuntimeError(f'Class filter did not match any model classes: class_name={args.class_name!r}, class_id={args.class_id}')

    rows = []
    inference_times_ms = []
    detections = 0
    masks = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        t0 = time.perf_counter()
        results = model.predict(
            source=image,
            imgsz=int(args.imgsz),
            conf=float(args.conf_threshold),
            iou=float(args.iou_threshold),
            device=str(args.device),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        inference_times_ms.append(elapsed_ms)
        detections_for_image = _collect_detections(results[0], target_ids, float(args.conf_threshold)) if results else []
        preview, row = _draw_preview(
            image,
            detections_for_image,
            mask_min_area=float(args.mask_min_area),
            mask_bottom_band_px=float(args.mask_bottom_band_px),
        )
        row['image'] = image_path.name
        row['inference_time_ms'] = float(elapsed_ms)
        rows.append(row)
        detections += int(row['detected'])
        masks += int(row['mask_available'])
        cv2.imwrite(str(out_dir / image_path.name), preview)

    summary = {
        'n_images': len(rows),
        'n_detections': int(detections),
        'n_masks': int(masks),
        'average_inference_time_ms': float(np.mean(inference_times_ms)) if inference_times_ms else math.nan,
        'model_path': str(model_path),
        'class_name': str(args.class_name),
        'class_id': int(args.class_id),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out_dir / 'per_image.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
