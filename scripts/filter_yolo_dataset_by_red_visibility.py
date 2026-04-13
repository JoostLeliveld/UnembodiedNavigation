#!/usr/bin/env python3
"""Create a filtered YOLO dataset copy by blanking labels with little visible red content."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


def _label_to_bbox(label_path: Path, img_w: int, img_h: int):
    text = label_path.read_text(encoding='utf-8').strip() if label_path.is_file() else ''
    if not text:
        return None
    vals = [float(v) for v in text.splitlines()[0].split()]
    if len(vals) < 5:
        return None
    _cls, cx, cy, bw, bh = vals[:5]
    x0 = (cx - 0.5 * bw) * img_w
    y0 = (cy - 0.5 * bh) * img_h
    x1 = (cx + 0.5 * bw) * img_w
    y1 = (cy + 0.5 * bh) * img_h
    return x0, y0, x1, y1


def _red_pixels_in_bbox(image_bgr: np.ndarray, bbox) -> int:
    if bbox is None:
        return 0
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_bgr.shape[1] - 1)
    y1 = min(y1, image_bgr.shape[0] - 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    crop = image_bgr[y0:y1 + 1, x0:x1 + 1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 70, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    return int(np.count_nonzero(mask))


def _write_preview(path: Path, image_bgr: np.ndarray, bbox, kept: bool, red_px: int) -> None:
    preview = image_bgr.copy()
    if bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        color = (0, 255, 255) if kept else (0, 0, 255)
        cv2.rectangle(preview, (x0, y0), (x1, y1), color, 2)
    label = f'kept={int(kept)} red_px={red_px}'
    cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def main() -> int:
    parser = argparse.ArgumentParser(description='Filter YOLO labels by visible red pixels inside the labeled bbox.')
    parser.add_argument('dataset', help='Input YOLO dataset directory containing data.yaml')
    parser.add_argument('--output-dir', default='', help='Output dataset directory; defaults to <dataset>_redfiltered_<timestamp>')
    parser.add_argument('--min-red-px', type=int, default=20)
    parser.add_argument('--preview-count', type=int, default=80)
    args = parser.parse_args()

    src = Path(args.dataset).expanduser().resolve()
    if (src / 'data.yaml').is_file():
        data_yaml_path = src / 'data.yaml'
    elif src.is_file() and src.name.endswith(('.yaml', '.yml')):
        data_yaml_path = src
        src = src.parent
    else:
        raise RuntimeError(f'Expected YOLO dataset directory or data.yaml, got {src}')

    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else src.with_name(
        f'{src.name}_redfiltered_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    if out.exists():
        raise RuntimeError(f'Output directory already exists: {out}')
    shutil.copytree(src, out, ignore=shutil.ignore_patterns('yolo_benchmark', 'previews', '*contact_sheet*.jpg'))

    data = yaml.safe_load(data_yaml_path.read_text(encoding='utf-8')) or {}
    rows = []
    preview_written = 0
    for split in ('train', 'val'):
        for image_path in sorted((out / 'images' / split).glob('*')):
            if image_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            label_path = out / 'labels' / split / image_path.with_suffix('.txt').name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            original_label = label_path.read_text(encoding='utf-8') if label_path.is_file() else ''
            bbox = _label_to_bbox(label_path, image.shape[1], image.shape[0])
            red_px = _red_pixels_in_bbox(image, bbox)
            kept = bool(bbox is not None and red_px >= int(args.min_red_px))
            if bbox is not None and not kept:
                label_path.write_text('', encoding='utf-8')
            if preview_written < max(int(args.preview_count), 0):
                preview_path = out / 'filter_previews' / split / image_path.name
                _write_preview(preview_path, image, bbox, kept, red_px)
                preview_written += 1
            rows.append({
                'split': split,
                'image': str(image_path.relative_to(out)),
                'label': str(label_path.relative_to(out)),
                'had_label': bool(original_label.strip()),
                'kept_label': kept,
                'red_px': int(red_px),
            })

    data['path'] = str(out)
    (out / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    (out / 'red_visibility_filter_manifest.json').write_text(json.dumps({
        'source_dataset': str(src),
        'output_dataset': str(out),
        'min_red_px': int(args.min_red_px),
        'n_images': len(rows),
        'n_original_labels': sum(1 for row in rows if row['had_label']),
        'n_kept_labels': sum(1 for row in rows if row['kept_label']),
        'n_blank_labels': sum(1 for row in rows if row['had_label'] and not row['kept_label']),
        'rows': rows,
    }, indent=2), encoding='utf-8')
    print(f'Wrote red-filtered dataset to {out}')
    print(f'Kept {sum(1 for row in rows if row["kept_label"])} labels; blanked {sum(1 for row in rows if row["had_label"] and not row["kept_label"])} labels.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
