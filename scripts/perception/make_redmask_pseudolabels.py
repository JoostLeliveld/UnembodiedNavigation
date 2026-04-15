#!/usr/bin/env python3
"""Generate simple YOLO-seg pseudo-labels from projected bboxes and red pixels inside them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _parse_yolo_bbox(label_path: Path, img_w: int, img_h: int):
    if not label_path.is_file():
        return None
    text = label_path.read_text(encoding='utf-8').strip()
    if not text:
        return None
    values = [float(v) for v in text.split()]
    if len(values) != 5:
        return None
    _, cx, cy, bw, bh = values
    x0 = (cx - 0.5 * bw) * img_w
    y0 = (cy - 0.5 * bh) * img_h
    x1 = (cx + 0.5 * bw) * img_w
    y1 = (cy + 0.5 * bh) * img_h
    return np.asarray([x0, y0, x1, y1], dtype=float)


def _clip_bbox(bbox, img_w: int, img_h: int, pad_px: float):
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0 = float(np.clip(x0 - pad_px, 0.0, img_w - 1.0))
    y0 = float(np.clip(y0 - pad_px, 0.0, img_h - 1.0))
    x1 = float(np.clip(x1 + pad_px, 0.0, img_w - 1.0))
    y1 = float(np.clip(y1 + pad_px, 0.0, img_h - 1.0))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.asarray([x0, y0, x1, y1], dtype=float)


def _red_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 70, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))


def _polygon_from_mask(mask: np.ndarray, x0: int, y0: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0.0:
        return None
    approx = cv2.approxPolyDP(contour, max(1.0, 0.015 * cv2.arcLength(contour, True)), True)
    pts = approx.reshape(-1, 2).astype(float)
    if pts.shape[0] < 3:
        return None
    pts[:, 0] += float(x0)
    pts[:, 1] += float(y0)
    return pts


def _write_seg_label(path: Path, polygon: np.ndarray | None, img_w: int, img_h: int) -> None:
    if polygon is None or len(polygon) < 3:
        path.write_text('', encoding='utf-8')
        return
    values = ['0']
    for x, y in polygon:
        values.append(f'{np.clip(x / float(img_w), 0.0, 1.0):.8f}')
        values.append(f'{np.clip(y / float(img_h), 0.0, 1.0):.8f}')
    path.write_text(' '.join(values) + '\n', encoding='utf-8')


def _mask_bottom(polygon: np.ndarray | None, band_px: float) -> tuple[float, float]:
    if polygon is None or len(polygon) < 3:
        return math.nan, math.nan
    v_bottom = float(np.max(polygon[:, 1]))
    band = polygon[polygon[:, 1] >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(polygon[:, 0])), v_bottom
    return float(np.mean(band[:, 0])), v_bottom


def main() -> int:
    parser = argparse.ArgumentParser(description='Create simple red-mask YOLO-seg pseudo-labels from a bbox-labeled dataset.')
    parser.add_argument('--input', required=True, help='Input dataset folder or data.yaml with projected bbox labels')
    parser.add_argument('--out', required=True, help='Output YOLO segmentation dataset folder')
    parser.add_argument('--pad-px', type=float, default=8.0)
    parser.add_argument('--min-mask-area', type=float, default=20.0)
    parser.add_argument('--max-mask-bbox-ratio', type=float, default=1.25)
    parser.add_argument('--preview-count', type=int, default=80)
    parser.add_argument('--bottom-band-px', type=float, default=3.0)
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    if src.is_file():
        source_root = src.parent
    else:
        source_root = src
    if not source_root.is_dir():
        raise RuntimeError(f'Input dataset does not exist: {source_root}')
    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        raise RuntimeError(f'Output folder already exists: {out_dir}')

    data_yaml = source_root / 'data.yaml'
    if not data_yaml.is_file():
        raise RuntimeError(f'Missing data.yaml in {source_root}')
    data_cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}

    diagnostics = []
    accepted = 0
    rejected = 0
    preview_written = 0

    for split in ('train', 'val', 'test'):
        image_dir = source_root / 'images' / split
        label_dir = source_root / 'labels' / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for image_path in sorted(image_dir.glob('*')):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            bbox = _parse_yolo_bbox(label_dir / image_path.with_suffix('.txt').name, image.shape[1], image.shape[0])
            polygon = None
            reason = 'missing_bbox'
            bbox_area = 0.0
            mask_area = 0.0
            area_ratio = math.nan
            if bbox is not None:
                roi = _clip_bbox(bbox, image.shape[1], image.shape[0], float(args.pad_px))
                if roi is not None:
                    x0, y0, x1, y1 = [int(round(v)) for v in roi]
                    crop = image[y0:y1 + 1, x0:x1 + 1]
                    polygon = _polygon_from_mask(_red_mask(crop), x0, y0)
                    bbox_area = float(max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0))
                    if polygon is not None:
                        mask_area = float(abs(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))
                        area_ratio = float(mask_area / bbox_area) if bbox_area > 0.0 else math.nan
                        if mask_area < float(args.min_mask_area):
                            reason = 'small_mask'
                            polygon = None
                        elif math.isfinite(area_ratio) and area_ratio > float(args.max_mask_bbox_ratio):
                            reason = 'large_mask'
                            polygon = None
                        else:
                            reason = ''
            ok = int(polygon is not None)
            accepted += ok
            rejected += int(not ok)

            out_image = out_dir / 'images' / split / image_path.name
            out_label = out_dir / 'labels' / split / image_path.with_suffix('.txt').name
            shutil.copy2(image_path, out_image)
            _write_seg_label(out_label, polygon, image.shape[1], image.shape[0])

            mask_bottom_u, mask_bottom_v = _mask_bottom(polygon, float(args.bottom_band_px))
            bbox_bottom_u = float(0.5 * (bbox[0] + bbox[2])) if bbox is not None else math.nan
            bbox_bottom_v = float(bbox[3]) if bbox is not None else math.nan
            preview_rel = ''
            if preview_written < max(int(args.preview_count), 0):
                preview = image.copy()
                if bbox is not None:
                    cv2.rectangle(preview, (int(round(bbox[0])), int(round(bbox[1]))), (int(round(bbox[2])), int(round(bbox[3]))), (255, 200, 0), 2)
                if polygon is not None:
                    pts = polygon.astype(np.int32).reshape(-1, 1, 2)
                    overlay = preview.copy()
                    cv2.fillPoly(overlay, [pts], (0, 180, 0))
                    preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0.0)
                    cv2.polylines(preview, [pts], True, (0, 200, 0), 2)
                if math.isfinite(mask_bottom_u) and math.isfinite(mask_bottom_v):
                    cv2.circle(preview, (int(round(mask_bottom_u)), int(round(mask_bottom_v))), 5, (0, 255, 0), -1)
                elif math.isfinite(bbox_bottom_u) and math.isfinite(bbox_bottom_v):
                    cv2.circle(preview, (int(round(bbox_bottom_u)), int(round(bbox_bottom_v))), 5, (0, 255, 255), -1)
                cv2.putText(preview, f'accepted={ok} reason={reason or "ok"}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                preview_path = out_dir / 'previews' / split / image_path.name
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_path), preview)
                preview_rel = str(preview_path.relative_to(out_dir))
                preview_written += 1

            diagnostics.append({
                'split': split,
                'image': str(out_image.relative_to(out_dir)),
                'label': str(out_label.relative_to(out_dir)),
                'accepted': ok,
                'rejection_reason': reason,
                'bbox_xmin': math.nan if bbox is None else float(bbox[0]),
                'bbox_ymin': math.nan if bbox is None else float(bbox[1]),
                'bbox_xmax': math.nan if bbox is None else float(bbox[2]),
                'bbox_ymax': math.nan if bbox is None else float(bbox[3]),
                'bbox_area_px': float(bbox_area),
                'mask_area_px': float(mask_area),
                'mask_bbox_ratio': float(area_ratio),
                'mask_bottom_u': float(mask_bottom_u),
                'mask_bottom_v': float(mask_bottom_v),
                'bbox_bottom_u': float(bbox_bottom_u),
                'bbox_bottom_v': float(bbox_bottom_v),
                'preview': preview_rel,
            })

    (out_dir / 'data.yaml').write_text(yaml.safe_dump({
        'path': str(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': data_cfg.get('names', {0: 'robot'}),
        'task': 'segment',
    }, sort_keys=False), encoding='utf-8')

    with (out_dir / 'label_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        fieldnames = list(diagnostics[0].keys()) if diagnostics else [
            'split', 'image', 'label', 'accepted', 'rejection_reason',
            'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
            'bbox_area_px', 'mask_area_px', 'mask_bbox_ratio',
            'mask_bottom_u', 'mask_bottom_v', 'bbox_bottom_u', 'bbox_bottom_v', 'preview',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)

    manifest = {
        'source_dataset': str(source_root),
        'accepted_labels': int(accepted),
        'rejected_labels': int(rejected),
        'label_backend': 'redmask_pseudolabels',
        'timestamp': _timestamp(),
        'notes': [
            'Red-mask is only an offline bootstrap pseudo-label generator.',
            'It is not the runtime detector and not the thesis contribution.',
        ],
    }
    (out_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote pseudo-label dataset to {out_dir}')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
