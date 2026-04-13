#!/usr/bin/env python3
"""Convert an existing YOLO bbox dataset into a visible-red-mask YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
scripts_dir = str((REPO_ROOT / 'scripts').resolve())
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from capture_yolo_seg_dataset import _largest_visible_robot_polygon, _write_preview, _write_seg_label


def _label_bbox(label_path: Path, img_w: int, img_h: int):
    if not label_path.is_file():
        return None
    text = label_path.read_text(encoding='utf-8').strip()
    if not text:
        return None
    vals = [float(v) for v in text.splitlines()[0].split()]
    if len(vals) < 5 or len(vals) > 5:
        return None
    _cls, cx, cy, bw, bh = vals
    return (
        (cx - 0.5 * bw) * img_w,
        (cy - 0.5 * bh) * img_h,
        (cx + 0.5 * bw) * img_w,
        (cy + 0.5 * bh) * img_h,
    )


def _label_path_for_image(root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(root)
    parts = list(rel.parts)
    if parts and parts[0] == 'images':
        parts[0] = 'labels'
    return root / Path(*parts).with_suffix('.txt')


def main() -> int:
    parser = argparse.ArgumentParser(description='Create a YOLO segmentation dataset from existing bbox labels and red robot pixels.')
    parser.add_argument('dataset', help='Input YOLO bbox dataset directory or data.yaml')
    parser.add_argument('--output-dir', default='', help='Output directory; defaults to logs/yolo_seg_datasets/<dataset>_seg_<timestamp>')
    parser.add_argument('--seg-roi-pad-px', type=float, default=8.0)
    parser.add_argument('--seg-morph-kernel-px', type=int, default=3)
    parser.add_argument('--min-mask-area-px', type=float, default=20.0)
    parser.add_argument('--polygon-epsilon-frac', type=float, default=0.015)
    parser.add_argument('--max-polygon-points', type=int, default=64)
    parser.add_argument('--preview-count', type=int, default=120)
    args = parser.parse_args()

    src = Path(args.dataset).expanduser().resolve()
    if src.is_file() and src.name.endswith(('.yaml', '.yml')):
        src_yaml = src
        src = src.parent
    else:
        src_yaml = src / 'data.yaml'
    if not src_yaml.is_file():
        raise RuntimeError(f'Could not find data.yaml for {src}')

    src_data = yaml.safe_load(src_yaml.read_text(encoding='utf-8')) or {}
    source_root = Path(str(src_data.get('path', src))).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (
        REPO_ROOT / 'logs' / 'yolo_seg_datasets' / f'{source_root.name}_seg_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    if out.exists():
        raise RuntimeError(f'Output directory already exists: {out}')

    rows = []
    preview_written = 0
    for split in ('train', 'val'):
        src_image_dir = source_root / 'images' / split
        out_image_dir = out / 'images' / split
        out_label_dir = out / 'labels' / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        if not src_image_dir.is_dir():
            continue

        for image_path in sorted(src_image_dir.glob('*')):
            if image_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            src_label = _label_path_for_image(source_root, image_path)
            projected_bbox = _label_bbox(src_label, image.shape[1], image.shape[0])
            polygon, roi, mask_area_px, visible_red_px = _largest_visible_robot_polygon(
                image,
                projected_bbox,
                roi_pad_px=float(args.seg_roi_pad_px),
                morph_kernel_px=int(args.seg_morph_kernel_px),
                min_mask_area_px=float(args.min_mask_area_px),
                polygon_epsilon_frac=float(args.polygon_epsilon_frac),
                max_polygon_points=int(args.max_polygon_points),
            )

            out_image = out_image_dir / image_path.name
            out_label = out_label_dir / image_path.with_suffix('.txt').name
            shutil.copy2(image_path, out_image)
            _write_seg_label(out_label, polygon, image.shape[1], image.shape[0])
            preview_path = ''
            if preview_written < max(int(args.preview_count), 0):
                preview = out / 'previews' / split / image_path.name
                _write_preview(preview, image, projected_bbox, polygon, roi, mask_area_px)
                preview_path = str(preview.relative_to(out))
                preview_written += 1
            rows.append({
                'split': split,
                'image': str(out_image.relative_to(out)),
                'label': str(out_label.relative_to(out)),
                'source_image': str(image_path.relative_to(source_root)),
                'source_label': str(src_label.relative_to(source_root)) if src_label.exists() else '',
                'source_has_bbox': projected_bbox is not None,
                'has_mask': polygon is not None,
                'polygon_points': int(len(polygon)) if polygon is not None else 0,
                'mask_area_px': float(mask_area_px),
                'visible_red_px': int(visible_red_px),
                'preview': preview_path,
            })

    data_yaml = {
        'path': str(out),
        'train': 'images/train',
        'val': 'images/val',
        'names': src_data.get('names', {0: 'robot'}),
        'task': 'segment',
    }
    (out / 'data.yaml').write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding='utf-8')
    (out / 'conversion_manifest.json').write_text(json.dumps({
        'source_dataset': str(source_root),
        'output_dataset': str(out),
        'n_images': len(rows),
        'n_source_bboxes': sum(1 for row in rows if row['source_has_bbox']),
        'n_masks': sum(1 for row in rows if row['has_mask']),
        'label_model': {
            'method': 'existing_bbox_roi_plus_visible_red_mask',
            'seg_roi_pad_px': float(args.seg_roi_pad_px),
            'seg_morph_kernel_px': int(args.seg_morph_kernel_px),
            'min_mask_area_px': float(args.min_mask_area_px),
            'polygon_epsilon_frac': float(args.polygon_epsilon_frac),
            'max_polygon_points': int(args.max_polygon_points),
        },
        'rows': rows,
    }, indent=2), encoding='utf-8')
    print(f'Wrote YOLO segmentation dataset to {out}')
    print(f'Converted {sum(1 for row in rows if row["source_has_bbox"])} source boxes into {sum(1 for row in rows if row["has_mask"])} mask labels')
    print(f'Train with: python3 scripts/train_yolo_robot.py --task segment --model yolo11n-seg.pt --data {out / "data.yaml"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
