#!/usr/bin/env python3
"""Benchmark several Ultralytics YOLO models on the same YOLO-format image set."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def _resolve_model_ref(model_ref: str, hf_filename: str) -> str:
    ref = str(model_ref or '').strip()
    if ref.startswith('hf://'):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Model reference uses hf:// but huggingface_hub is not installed. "
                "Install it with `pip install huggingface_hub` or pass a local .pt path."
            ) from exc
        parts = [p for p in ref[len('hf://'):].strip('/').split('/') if p]
        if len(parts) < 2:
            raise RuntimeError('Use hf://owner/repo or hf://owner/repo/path/to/model.pt')
        repo_id = '/'.join(parts[:2])
        filename = '/'.join(parts[2:]) if len(parts) > 2 else str(hf_filename or 'best.pt')
        return hf_hub_download(repo_id=repo_id, filename=filename)
    return ref


def _target_class_ids(names, target_class: str):
    target = str(target_class or '').strip().lower()
    if target in ('', '*', 'any', 'all'):
        return None
    if target.lstrip('-').isdigit():
        return {int(target)}
    if isinstance(names, dict):
        return {int(idx) for idx, name in names.items() if str(name).strip().lower() == target}
    return set()


def _load_dataset_images(dataset: Path, split: str):
    if dataset.is_file() and dataset.name.endswith(('.yaml', '.yml')):
        data = yaml.safe_load(dataset.read_text(encoding='utf-8')) or {}
        root = Path(str(data.get('path', dataset.parent))).expanduser()
        split_key = str(split or 'val').strip().lower()
        if split_key == 'all':
            split_names = ('val', 'train', 'test')
        else:
            split_names = (split_key,)
        image_dirs = [root / str(data.get(name, f'images/{name}')) for name in split_names]
    else:
        root = dataset
        image_dirs = [root]
    images = []
    for image_dir in image_dirs:
        if not image_dir.is_dir():
            continue
        images.extend(sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')))
    return root, images


def _label_path_for_image(root: Path, image_path: Path) -> Path | None:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts or parts[0] != 'images':
        return None
    parts[0] = 'labels'
    return root / Path(*parts).with_suffix('.txt')


def _load_first_label(root: Path, image_path: Path, img_w: int, img_h: int):
    label_path = _label_path_for_image(root, image_path)
    if label_path is None or not label_path.is_file():
        return None, None
    for line in label_path.read_text(encoding='utf-8').splitlines():
        vals = [float(v) for v in line.split()]
        if len(vals) < 5:
            continue
        if len(vals) > 5:
            pts = np.asarray(vals[1:], dtype=float).reshape(-1, 2)
            pts[:, 0] *= float(img_w)
            pts[:, 1] *= float(img_h)
            xmin = float(np.min(pts[:, 0]))
            ymin = float(np.min(pts[:, 1]))
            xmax = float(np.max(pts[:, 0]))
            ymax = float(np.max(pts[:, 1]))
            return np.array([xmin, ymin, xmax, ymax], dtype=float), pts
        _cls, cx, cy, bw, bh = vals[:5]
        xmin = (cx - 0.5 * bw) * img_w
        ymin = (cy - 0.5 * bh) * img_h
        xmax = (cx + 0.5 * bw) * img_w
        ymax = (cy + 0.5 * bh) * img_h
        return np.array([xmin, ymin, xmax, ymax], dtype=float), None
    return None, None


def _bbox_iou(a, b) -> float:
    if a is None or b is None:
        return math.nan
    ix0 = max(float(a[0]), float(b[0]))
    iy0 = max(float(a[1]), float(b[1]))
    ix1 = min(float(a[2]), float(b[2]))
    iy1 = min(float(a[3]), float(b[3]))
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    area_a = max(float(a[2] - a[0]), 0.0) * max(float(a[3] - a[1]), 0.0)
    area_b = max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else math.nan


def _shape_mask(polygon, bbox, img_w: int, img_h: int) -> np.ndarray | None:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if polygon is not None and len(polygon) >= 3:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
        return mask
    if bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0 = int(np.clip(x0, 0, img_w - 1))
        y0 = int(np.clip(y0, 0, img_h - 1))
        x1 = int(np.clip(x1, 0, img_w - 1))
        y1 = int(np.clip(y1, 0, img_h - 1))
        if x1 <= x0 or y1 <= y0:
            return None
        mask[y0:y1 + 1, x0:x1 + 1] = 1
        return mask
    return None


def _shape_iou(pred_polygon, pred_bbox, gt_polygon, gt_bbox, img_w: int, img_h: int) -> float:
    pred_mask = _shape_mask(pred_polygon, pred_bbox, img_w, img_h)
    gt_mask = _shape_mask(gt_polygon, gt_bbox, img_w, img_h)
    if pred_mask is None or gt_mask is None:
        return math.nan
    inter = float(np.sum((pred_mask > 0) & (gt_mask > 0)))
    union = float(np.sum((pred_mask > 0) | (gt_mask > 0)))
    return float(inter / union) if union > 0.0 else math.nan


def _draw_preview(path: Path, image, pred_bbox, pred_polygon, gt_bbox, gt_polygon, score: float, model_name: str) -> None:
    preview = image.copy()
    if gt_polygon is not None and len(gt_polygon) >= 3:
        pts = np.asarray(gt_polygon, dtype=np.int32).reshape(-1, 1, 2)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        preview = cv2.addWeighted(overlay, 0.22, preview, 0.78, 0.0)
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    if gt_bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in gt_bbox]
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(preview, 'gt', (x0, max(y0 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    if pred_polygon is not None and len(pred_polygon) >= 3:
        pts = np.asarray(pred_polygon, dtype=np.int32).reshape(-1, 1, 2)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        preview = cv2.addWeighted(overlay, 0.28, preview, 0.72, 0.0)
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 180, 0), thickness=2)
    if pred_bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in pred_bbox]
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 180, 0), 2)
        cv2.putText(
            preview,
            f'pred {score:.2f}',
            (x0, min(y1 + 20, preview.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
        )
    cv2.putText(preview, model_name, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def _select_detection(result, target_ids, conf_threshold: float):
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    mask = np.isfinite(conf) & (conf >= float(conf_threshold))
    if target_ids is not None:
        mask &= np.asarray([int(c) in target_ids for c in cls], dtype=bool)
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return None
    idx = int(valid[np.argmax(conf[valid])])
    pred_polygon = None
    masks = getattr(result, 'masks', None)
    mask_polygons = getattr(masks, 'xy', None) if masks is not None else None
    if mask_polygons is not None and idx < len(mask_polygons):
        pts = np.asarray(mask_polygons[idx], dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            pred_polygon = pts[:, :2]
    return xyxy[idx].astype(float), float(conf[idx]), int(cls[idx]), pred_polygon


def main() -> int:
    parser = argparse.ArgumentParser(description='Benchmark YOLO model refs on a saved YOLO dataset/images directory.')
    parser.add_argument('--dataset', required=True, help='YOLO data.yaml or image directory')
    parser.add_argument('--models', nargs='+', default=['yolo11n.pt'])
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--hf-filename', default='best.pt')
    parser.add_argument('--target-class', default='robot')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.45)
    parser.add_argument('--device', default='')
    parser.add_argument(
        '--split',
        choices=('train', 'val', 'test', 'all'),
        default='val',
        help='Dataset split to benchmark when --dataset points to a YOLO data.yaml.',
    )
    parser.add_argument('--preview-count', type=int, default=40, help='Number of prediction overlay images to save per model.')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install Ultralytics first: `pip install ultralytics`') from exc

    dataset = Path(args.dataset).expanduser().resolve()
    root, images = _load_dataset_images(dataset, str(args.split))
    if not images:
        raise RuntimeError(f'No images found for dataset {dataset}')
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / 'yolo_benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / 'predictions.csv'
    summary_path = output_dir / 'summary.csv'
    prediction_rows = []
    summary_rows = []

    for model_ref in args.models:
        resolved = _resolve_model_ref(model_ref, args.hf_filename)
        model = YOLO(resolved)
        target_ids = _target_class_ids(getattr(model, 'names', {}), args.target_class)
        scores = []
        ious = []
        bbox_ious = []
        times_ms = []
        detected_count = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        preview_written = 0
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            kwargs = {
                'source': image,
                'imgsz': int(args.imgsz),
                'conf': float(args.conf_threshold),
                'iou': float(args.iou_threshold),
                'verbose': False,
            }
            if str(args.device).strip():
                kwargs['device'] = str(args.device).strip()
            t0 = time.perf_counter()
            results = model.predict(**kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(elapsed_ms)
            det = _select_detection(results[0], target_ids, float(args.conf_threshold)) if results else None
            gt, gt_polygon = _load_first_label(root, image_path, image.shape[1], image.shape[0])
            has_gt = gt is not None
            if det is None:
                bbox = None
                score = 0.0
                cls_id = math.nan
                pred_polygon = None
            else:
                bbox, score, cls_id, pred_polygon = det
                detected_count += 1
                scores.append(score)
            if has_gt and det is not None:
                tp += 1
            elif (not has_gt) and det is not None:
                fp += 1
            elif has_gt and det is None:
                fn += 1
            else:
                tn += 1
            iou_val = _shape_iou(pred_polygon, bbox, gt_polygon, gt, image.shape[1], image.shape[0])
            bbox_iou_val = _bbox_iou(bbox, gt)
            if math.isfinite(iou_val):
                ious.append(iou_val)
            if math.isfinite(bbox_iou_val):
                bbox_ious.append(bbox_iou_val)
            preview_path = ''
            if preview_written < max(int(args.preview_count), 0):
                safe_model = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(model_ref))
                preview = output_dir / 'previews' / safe_model / image_path.name
                _draw_preview(preview, image, bbox, pred_polygon, gt, gt_polygon, float(score), str(model_ref))
                preview_path = str(preview.relative_to(output_dir))
                preview_written += 1
            prediction_rows.append({
                'model': model_ref,
                'image': str(image_path.relative_to(root) if image_path.is_relative_to(root) else image_path),
                'detected': int(det is not None),
                'score': score,
                'class_id': cls_id,
                'iou': iou_val,
                'bbox_iou': bbox_iou_val,
                'time_ms': elapsed_ms,
                'preview': preview_path,
            })

        summary_rows.append({
            'model': model_ref,
            'resolved_model': resolved,
            'split': str(args.split),
            'n_images': len(images),
            'detection_rate': detected_count / max(len(images), 1),
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'mean_iou': float(np.mean(ious)) if ious else math.nan,
            'mean_bbox_iou': float(np.mean(bbox_ious)) if bbox_ious else math.nan,
            'match_rate_iou_0_5': float(np.mean([v >= 0.5 for v in ious])) if ious else math.nan,
            'mean_time_ms': float(np.mean(times_ms)) if times_ms else math.nan,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        })

    with predictions_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()) if prediction_rows else [])
        writer.writeheader()
        writer.writerows(prediction_rows)
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f'Wrote predictions to {predictions_path}')
    print(f'Wrote summary to {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
