#!/usr/bin/env python3
"""Convert YOLO bbox labels into SAM-generated YOLO-seg pseudo-labels.

The projected geometric box tells SAM which object should become the robot
label. SAM is used offline only: it proposes a visible-object mask from the
full camera image, and this script keeps only pseudo-masks that pass simple
area and bbox-overlap validation. Runtime perception uses the trained YOLO-seg
model, not SAM or simulator ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset_paths(dataset_arg: str) -> tuple[Path, Path]:
    src = Path(dataset_arg).expanduser().resolve()
    if src.is_file() and src.suffix.lower() in ('.yaml', '.yml'):
        return src.parent, src
    data_yaml = src / 'data.yaml'
    if not data_yaml.is_file():
        raise RuntimeError(f'Could not find data.yaml for {src}')
    return src, data_yaml


def _source_root_from_yaml(data_yaml: Path) -> tuple[Path, dict]:
    data = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    root = Path(str(data.get('path', data_yaml.parent))).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    else:
        root = root.resolve()
    return root, data


def _label_path_for_image(root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(root)
    parts = list(rel.parts)
    if parts and parts[0] == 'images':
        parts[0] = 'labels'
    return root / Path(*parts).with_suffix('.txt')


def _iter_images(source_root: Path) -> Iterable[tuple[str, Path]]:
    for split in ('train', 'val', 'test'):
        image_dir = source_root / 'images' / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob('*')):
            if image_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                yield split, image_path


def _source_label_has_text(source_root: Path, image_path: Path) -> bool:
    label_path = _label_path_for_image(source_root, image_path)
    return label_path.is_file() and bool(label_path.read_text(encoding='utf-8').strip())


def _clip_xyxy(box, img_w: int, img_h: int, pad_px: float = 0.0):
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 -= float(pad_px)
    y0 -= float(pad_px)
    x1 += float(pad_px)
    y1 += float(pad_px)
    x0 = float(np.clip(x0, 0.0, float(img_w - 1)))
    y0 = float(np.clip(y0, 0.0, float(img_h - 1)))
    x1 = float(np.clip(x1, 0.0, float(img_w - 1)))
    y1 = float(np.clip(y1, 0.0, float(img_h - 1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _boxes_from_label(label_path: Path, img_w: int, img_h: int, *, pad_px: float):
    """Read YOLO detect/segment labels and return class/bbox prompts."""
    if not label_path.is_file():
        return []
    rows = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values = [float(v) for v in text.split()]
        except ValueError:
            continue
        if len(values) == 5:
            cls_id, cx, cy, bw, bh = values
            box = [
                (cx - 0.5 * bw) * img_w,
                (cy - 0.5 * bh) * img_h,
                (cx + 0.5 * bw) * img_w,
                (cy + 0.5 * bh) * img_h,
            ]
            label_kind = 'bbox'
        elif len(values) >= 7 and len(values[1:]) % 2 == 0:
            cls_id = values[0]
            xs = np.asarray(values[1::2], dtype=float) * float(img_w)
            ys = np.asarray(values[2::2], dtype=float) * float(img_h)
            if xs.size < 3 or ys.size < 3:
                continue
            box = [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))]
            label_kind = 'seg'
        else:
            continue
        if not all(math.isfinite(v) for v in box):
            continue
        clipped = _clip_xyxy(box, img_w, img_h, pad_px=pad_px)
        if clipped is None:
            continue
        rows.append({
            'class_id': int(round(cls_id)),
            'bbox_xyxy': clipped,
            'source_label_kind': label_kind,
        })
    return rows


def _polygon_area_px(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def _simplify_polygon(points_px: np.ndarray, *, epsilon_frac: float, max_points: int, min_area_px: float):
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    contour = pts.reshape(-1, 1, 2)
    arc_len = float(cv2.arcLength(contour, closed=True))
    eps = max(float(epsilon_frac) * arc_len, 0.5)
    approx = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)
    for _ in range(16):
        if int(max_points) <= 0 or approx.shape[0] <= int(max_points):
            break
        eps *= 1.35
        approx = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)
    if approx.shape[0] < 3 or _polygon_area_px(approx) < float(min_area_px):
        return None
    return approx.astype(float)


def _polygon_from_mask_data(mask_data, img_w: int, img_h: int, *, epsilon_frac: float, max_points: int, min_area_px: float):
    arr = mask_data
    try:
        if hasattr(arr, 'detach'):
            arr = arr.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.size == 0:
        return None
    mask = (arr > 0.5).astype(np.uint8)
    if mask.shape[1] != img_w or mask.shape[0] != img_h:
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return _simplify_polygon(
        contour,
        epsilon_frac=epsilon_frac,
        max_points=max_points,
        min_area_px=min_area_px,
    )


def _polygons_from_sam_result(result, img_w: int, img_h: int, *, epsilon_frac: float, max_points: int, min_area_px: float):
    """Extract polygons from SAM result."""
    masks = getattr(result, 'masks', None)
    if masks is None:
        return []
    
    polygons = []
    xyn = getattr(masks, 'xyn', None)
    if xyn is not None:
        for poly_norm in xyn:
            pts = np.asarray(poly_norm, dtype=float).reshape(-1, 2)
            if pts.shape[0] < 3:
                polygons.append(None)
                continue
            pts[:, 0] = np.clip(pts[:, 0] * float(img_w), 0.0, float(img_w - 1))
            pts[:, 1] = np.clip(pts[:, 1] * float(img_h), 0.0, float(img_h - 1))
            polygons.append(_simplify_polygon(
                pts,
                epsilon_frac=epsilon_frac,
                max_points=max_points,
                min_area_px=min_area_px,
            ))
        return polygons

    data = getattr(masks, 'data', None)
    if data is not None:
        try:
            n_masks = int(data.shape[0])
        except Exception:
            n_masks = 0
        for idx in range(n_masks):
            polygons.append(_polygon_from_mask_data(
                data[idx],
                img_w,
                img_h,
                epsilon_frac=epsilon_frac,
                max_points=max_points,
                min_area_px=min_area_px,
            ))
    return polygons


def _polygon_bbox_overlap_ratio(polygon, bbox_xyxy, img_w: int, img_h: int) -> float:
    if polygon is None:
        return 0.0

    poly_pts = np.asarray(polygon, dtype=float)
    if poly_pts.ndim != 2 or poly_pts.shape[0] < 3:
        return 0.0
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts.astype(np.int32).reshape(-1, 1, 2)], 1)
    mask_area = float(np.sum(mask))
    if mask_area <= 0.0:
        return 0.0

    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [float(v) for v in bbox_xyxy]
    y0, y1 = int(np.clip(bbox_y0, 0, img_h)), int(np.clip(bbox_y1, 0, img_h))
    x0, x1 = int(np.clip(bbox_x0, 0, img_w)), int(np.clip(bbox_x1, 0, img_w))
    if y1 <= y0 or x1 <= x0:
        return 0.0
    return float(np.sum(mask[y0:y1, x0:x1]) / mask_area)


def _polygon_mask(polygon, img_w: int, img_h: int) -> np.ndarray | None:
    if polygon is None:
        return None
    poly_pts = np.asarray(polygon, dtype=float)
    if poly_pts.ndim != 2 or poly_pts.shape[0] < 3:
        return None
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts.astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask


def _polygon_color_fraction(
    image_bgr: np.ndarray,
    polygon,
    *,
    min_saturation: float,
    min_value: float,
) -> float:
    mask = _polygon_mask(polygon, image_bgr.shape[1], image_bgr.shape[0])
    if mask is None or not np.any(mask):
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        return 0.0
    sat = pixels[:, 1].astype(float)
    val = pixels[:, 2].astype(float)
    color_like = (sat >= float(min_saturation)) & (val >= float(min_value))
    return float(np.mean(color_like))


def _polygon_bbox_area_ratio(polygon, bbox_xyxy, img_w: int, img_h: int) -> float:
    mask = _polygon_mask(polygon, img_w, img_h)
    if mask is None:
        return 0.0
    mask_area = float(np.sum(mask))
    if mask_area <= 0.0:
        return 0.0
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    bbox_area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    if bbox_area <= 0.0:
        return 0.0
    return float(mask_area / bbox_area)


def _bbox_center_and_diagonal(bbox_xyxy) -> tuple[float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    diag = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
    return cx, cy, diag


def _polygon_centroid(polygon) -> tuple[float, float]:
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        return math.nan, math.nan
    contour = pts.reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    if abs(float(moments.get('m00', 0.0))) > 1e-6:
        return float(moments['m10'] / moments['m00']), float(moments['m01'] / moments['m00'])
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _polygon_centroid_distance_norm(polygon, bbox_xyxy) -> float:
    px, py = _polygon_centroid(polygon)
    if not (math.isfinite(px) and math.isfinite(py)):
        return math.inf
    cx, cy, diag = _bbox_center_and_diagonal(bbox_xyxy)
    return float(math.hypot(px - cx, py - cy) / diag)


def _candidate_reject_reason(
    *,
    mask_area_px: float,
    in_bbox_ratio: float,
    centroid_distance_norm: float,
    bbox_area_ratio: float,
    min_mask_area_px: float,
    min_in_bbox_ratio: float,
    max_centroid_distance_norm: float,
    min_bbox_area_ratio: float,
    max_bbox_area_ratio: float,
) -> str:
    if float(mask_area_px) < float(min_mask_area_px):
        return 'small_area'
    if float(in_bbox_ratio) < float(min_in_bbox_ratio):
        return 'low_overlap'
    if float(centroid_distance_norm) > float(max_centroid_distance_norm):
        return 'centroid_distance'
    if (
        float(bbox_area_ratio) < float(min_bbox_area_ratio)
        or float(bbox_area_ratio) > float(max_bbox_area_ratio)
    ):
        return 'area_ratio'
    return ''


def _write_seg_labels(path: Path, labels, img_w: int, img_h: int) -> None:
    lines = []
    for label in labels:
        polygon = label['polygon']
        values = [str(int(label['class_id']))]
        for x, y in np.asarray(polygon, dtype=float).reshape(-1, 2):
            values.append(f'{np.clip(x / float(img_w), 0.0, 1.0):.8f}')
            values.append(f'{np.clip(y / float(img_h), 0.0, 1.0):.8f}')
        if len(values) >= 7:
            lines.append(' '.join(values))
    path.write_text(('\n'.join(lines) + '\n') if lines else '', encoding='utf-8')


def _write_preview(path: Path, image_bgr: np.ndarray, prompts, labels, summary_lines: list[str]) -> None:
    preview = image_bgr.copy()
    
    # Draw bbox prompts and point prompts
    for prompt in prompts:
        x0, y0, x1, y1 = [int(round(v)) for v in prompt['bbox_xyxy']]
        # Light bbox outline
        cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 160, 0), 1)
        # Point prompt marker (cyan circle at center)
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        cv2.circle(preview, (cx, cy), 6, (0, 255, 255), -1)
        cv2.circle(preview, (cx, cy), 6, (0, 200, 200), 1)
    
    # Draw resulting masks
    for label in labels:
        pts = np.asarray(label['polygon'], dtype=np.int32).reshape(-1, 1, 2)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green fill
        preview = cv2.addWeighted(overlay, 0.30, preview, 0.70, 0.0)
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    y = 30
    for line in summary_lines:
        cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def _load_sam(model_ref: str):
    try:
        from ultralytics import SAM
    except Exception as exc:
        raise RuntimeError(
            "Could not import Ultralytics SAM. Install/update optional YOLO dependencies first, "
            "for example `pip install ultralytics` in the environment used for this repo."
        ) from exc
    return SAM(model_ref)


def _sam_polygon_for_prompt(
    sam,
    image_bgr: np.ndarray,
    prompt: dict,
    *,
    prompt_modes: tuple[str, ...],
    device: str,
    imgsz: int,
    img_w: int,
    img_h: int,
    epsilon_frac: float,
    max_points: int,
    min_area_px: float,
    min_color_saturation: float,
    min_color_value: float,
    color_score_weight: float,
    area_penalty_weight: float,
    centroid_score_weight: float,
    min_bbox_overlap_ratio: float,
    max_centroid_distance_norm: float,
    min_bbox_area_ratio: float,
    max_bbox_area_ratio: float,
):
    """Run SAM for one robot prompt and return candidate diagnostics plus the best polygon.

    Box prompting is tried first in the default pipeline because the projected
    bbox is the offline ground-truth helper that identifies the robot instance.
    Point prompting remains a fallback because Ultralytics/SAM backends differ
    slightly in the prompt keywords they support.

    To reduce shadow-heavy pseudo-labels, we score candidates using:
    1. in-bbox overlap,
    2. fraction of sufficiently saturated/bright pixels inside the mask,
    3. a small penalty for masks that are much larger than the projected robot bbox,
    4. a penalty for masks whose centroid drifts too far from the prompt center.
    """
    x0, y0, x1, y1 = [float(v) for v in prompt['bbox_xyxy']]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    errors = []
    candidates: list[dict] = []

    for mode in prompt_modes:
        try:
            if mode == 'box':
                results = sam(
                    image_bgr,
                    bboxes=[[x0, y0, x1, y1]],
                    verbose=False,
                    save=False,
                    device=str(device),
                    imgsz=int(imgsz),
                )
            elif mode == 'point':
                results = sam(
                    image_bgr,
                    points=np.asarray([[cx, cy]], dtype=float),
                    verbose=False,
                    save=False,
                    device=str(device),
                    imgsz=int(imgsz),
                )
            else:
                continue
            polygons = _polygons_from_sam_result(
                results[0],
                img_w,
                img_h,
                epsilon_frac=epsilon_frac,
                max_points=max_points,
                min_area_px=0.0,
            ) if results else []
            for polygon in polygons:
                if polygon is None:
                    continue
                poly_pts = np.asarray(polygon, dtype=float)
                mask_area_px = _polygon_area_px(poly_pts)
                overlap_ratio = _polygon_bbox_overlap_ratio(polygon, prompt['bbox_xyxy'], img_w, img_h)
                color_fraction = _polygon_color_fraction(
                    image_bgr,
                    polygon,
                    min_saturation=min_color_saturation,
                    min_value=min_color_value,
                )
                bbox_area_ratio = _polygon_bbox_area_ratio(polygon, prompt['bbox_xyxy'], img_w, img_h)
                centroid_distance_norm = _polygon_centroid_distance_norm(polygon, prompt['bbox_xyxy'])
                score = (
                    float(overlap_ratio)
                    + float(color_score_weight) * float(color_fraction)
                    - float(area_penalty_weight) * min(float(bbox_area_ratio), 4.0)
                    - float(centroid_score_weight) * float(centroid_distance_norm)
                )
                reject_reason = _candidate_reject_reason(
                    mask_area_px=mask_area_px,
                    in_bbox_ratio=overlap_ratio,
                    centroid_distance_norm=centroid_distance_norm,
                    bbox_area_ratio=bbox_area_ratio,
                    min_mask_area_px=min_area_px,
                    min_in_bbox_ratio=min_bbox_overlap_ratio,
                    max_centroid_distance_norm=max_centroid_distance_norm,
                    min_bbox_area_ratio=min_bbox_area_ratio,
                    max_bbox_area_ratio=max_bbox_area_ratio,
                )
                candidates.append({
                    'polygon': polygon,
                    'prompt_used': mode,
                    'mask_area_px': float(mask_area_px),
                    'in_bbox_ratio': float(overlap_ratio),
                    'color_fraction': float(color_fraction),
                    'bbox_area_ratio': float(bbox_area_ratio),
                    'centroid_distance_norm': float(centroid_distance_norm),
                    'score': float(score),
                    'reject_reason': reject_reason,
                    'accepted': 0,
                })
        except Exception as exc:
            errors.append(f'{mode}: {exc}')

    viable = [candidate for candidate in candidates if not candidate['reject_reason']]
    if viable:
        best = max(
            viable,
            key=lambda candidate: (
                candidate['score'],
                candidate['in_bbox_ratio'],
                candidate['color_fraction'],
                -candidate['bbox_area_ratio'],
                -candidate['centroid_distance_norm'],
            ),
        )
        best['accepted'] = 1
        return best, candidates, '; '.join(errors)
    return None, candidates, '; '.join(errors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Use YOLO bbox labels as full-image SAM prompts to generate YOLO-seg pseudo-labels.'
    )
    parser.add_argument('dataset', help='Input YOLO bbox dataset directory, or its data.yaml')
    parser.add_argument('--output-dir', default='', help='Output directory; defaults to logs/yolo_seg_datasets/<dataset>_sam_<timestamp>')
    parser.add_argument('--sam-model', default='mobile_sam.pt', help='SAM model ref/path, e.g. mobile_sam.pt or sam_b.pt')
    parser.add_argument('--device', default='', help="Ultralytics device string, e.g. '', 'cpu', '0', or 'cuda:0'")
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument(
        '--prompt-mode',
        choices=('box', 'point', 'box_then_point'),
        default='box_then_point',
        help='SAM prompt strategy; default uses the projected bbox first and falls back to a center point.',
    )
    parser.add_argument('--prompt-pad-px', type=float, default=4.0, help='Pixel padding applied to bbox prompts before SAM.')
    parser.add_argument('--min-mask-area-px', type=float, default=50.0, help='Minimum mask area in pixels')
    parser.add_argument(
        '--min-bbox-overlap-ratio',
        type=float,
        default=0.5,
        help='Minimum in-bbox ratio: this fraction of the SAM mask pixels must lie inside the projected robot bbox.',
    )
    parser.add_argument('--polygon-epsilon-frac', type=float, default=0.015, help='Contour simplification factor')
    parser.add_argument('--max-polygon-points', type=int, default=96)
    parser.add_argument(
        '--min-color-saturation',
        type=float,
        default=60.0,
        help='HSV saturation threshold used when preferring robot-like masks over low-information floor spill.',
    )
    parser.add_argument(
        '--min-color-value',
        type=float,
        default=50.0,
        help='HSV value threshold used when preferring robot-like masks over low-information floor spill.',
    )
    parser.add_argument(
        '--color-score-weight',
        type=float,
        default=0.35,
        help='How strongly to prefer masks with colored robot evidence.',
    )
    parser.add_argument(
        '--area-penalty-weight',
        type=float,
        default=0.08,
        help='Small penalty on oversized masks relative to the projected robot bbox.',
    )
    parser.add_argument(
        '--centroid-score-weight',
        type=float,
        default=0.20,
        help='Penalty weight for masks whose centroid is far from the prompt center.',
    )
    parser.add_argument(
        '--max-centroid-distance-norm',
        type=float,
        default=0.35,
        help='Reject a candidate if its centroid is farther than this fraction of the bbox diagonal from the prompt center.',
    )
    parser.add_argument(
        '--min-bbox-area-ratio',
        type=float,
        default=0.05,
        help='Reject a candidate if its mask area is smaller than this fraction of the prompt bbox area.',
    )
    parser.add_argument(
        '--max-bbox-area-ratio',
        type=float,
        default=1.25,
        help='Reject a candidate if its mask area is larger than this multiple of the prompt bbox area.',
    )
    parser.add_argument('--preview-count', type=int, default=120)
    parser.add_argument('--limit', type=int, default=0, help='Optional image limit for quick smoke runs')
    parser.add_argument(
        '--skip-empty-source-labels',
        action='store_true',
        help='Only convert images whose source label file is non-empty; useful for quick SAM smoke checks.',
    )
    args = parser.parse_args()

    src, src_yaml = _dataset_paths(args.dataset)
    source_root, src_data = _source_root_from_yaml(src_yaml)
    out = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (
        REPO_ROOT / 'logs' / 'yolo_seg_datasets' / f'{source_root.name}_sam_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    if out.exists():
        raise RuntimeError(f'Output directory already exists: {out}')

    sam = _load_sam(str(args.sam_model))

    image_rows = list(_iter_images(source_root))
    if bool(args.skip_empty_source_labels):
        image_rows = [(split, path) for split, path in image_rows if _source_label_has_text(source_root, path)]
    if int(args.limit) > 0:
        image_rows = image_rows[: int(args.limit)]

    print(f"\n{'='*80}")
    prompt_modes = ('box', 'point') if args.prompt_mode == 'box_then_point' else (str(args.prompt_mode),)

    print(f"SAM SEGMENTATION CONVERSION (supervisor-aligned pseudo-labels)")
    print(f"{'='*80}")
    print(f"Input dataset: {source_root.name}")
    print(f"Method: full-image SAM prompts from projected robot bboxes")
    print(f"Settings:")
    print(f"  - Prompt mode: {args.prompt_mode}")
    print(f"  - Prompt padding: {args.prompt_pad_px}px")
    print(f"  - Min mask area: {args.min_mask_area_px}px²")
    print(f"  - Min bbox overlap: {args.min_bbox_overlap_ratio*100:.0f}%")
    print(f"  - Max centroid distance: {args.max_centroid_distance_norm:.2f} * bbox diagonal")
    print(f"  - Bbox area ratio window: [{args.min_bbox_area_ratio:.2f}, {args.max_bbox_area_ratio:.2f}]")
    print(f"  - Appearance prior: sat>={args.min_color_saturation:.0f}, val>={args.min_color_value:.0f}, color_weight={args.color_score_weight:.2f}, area_penalty={args.area_penalty_weight:.2f}")
    print(f"  - Centroid penalty weight: {args.centroid_score_weight:.2f}")
    print(f"  - Processing {len(image_rows)} images...")
    print()

    masks_accepted = 0
    masks_rejected = 0
    total_prompts = 0
    prompt_used_counts = {}
    rejection_counts = {
        'low_overlap': 0,
        'centroid_distance': 0,
        'area_ratio': 0,
        'small_area': 0,
        'no_viable_candidate': 0,
    }
    candidate_rows = []

    for image_idx, (split, image_path) in enumerate(image_rows):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        img_h, img_w = image_bgr.shape[:2]
        src_label = _label_path_for_image(source_root, image_path)
        prompts = _boxes_from_label(src_label, img_w, img_h, pad_px=float(args.prompt_pad_px))

        labels = []
        
        if prompts:
            total_prompts += len(prompts)

            for prompt_idx, prompt in enumerate(prompts):
                selected, candidates, prompt_error = _sam_polygon_for_prompt(
                    sam,
                    image_bgr,
                    prompt,
                    prompt_modes=prompt_modes,
                    device=str(args.device),
                    imgsz=int(args.imgsz),
                    img_w=img_w,
                    img_h=img_h,
                    epsilon_frac=float(args.polygon_epsilon_frac),
                    max_points=int(args.max_polygon_points),
                    min_area_px=float(args.min_mask_area_px),
                    min_color_saturation=float(args.min_color_saturation),
                    min_color_value=float(args.min_color_value),
                    color_score_weight=float(args.color_score_weight),
                    area_penalty_weight=float(args.area_penalty_weight),
                    centroid_score_weight=float(args.centroid_score_weight),
                    min_bbox_overlap_ratio=float(args.min_bbox_overlap_ratio),
                    max_centroid_distance_norm=float(args.max_centroid_distance_norm),
                    min_bbox_area_ratio=float(args.min_bbox_area_ratio),
                    max_bbox_area_ratio=float(args.max_bbox_area_ratio),
                )
                for candidate_idx, candidate in enumerate(candidates):
                    reject_reason = str(candidate.get('reject_reason', ''))
                    if reject_reason in rejection_counts:
                        rejection_counts[reject_reason] += 1
                    candidate_rows.append({
                        'split': split,
                        'image': image_path.name,
                        'prompt_index': int(prompt_idx),
                        'prompt_mode': str(candidate['prompt_used']),
                        'candidate_index': int(candidate_idx),
                        'accepted': int(candidate.get('accepted', 0)),
                        'reject_reason': reject_reason,
                        'in_bbox_ratio': float(candidate['in_bbox_ratio']),
                        'color_fraction': float(candidate['color_fraction']),
                        'bbox_area_ratio': float(candidate['bbox_area_ratio']),
                        'centroid_distance_norm': float(candidate['centroid_distance_norm']),
                        'mask_area_px': float(candidate['mask_area_px']),
                        'final_score': float(candidate['score']),
                        'prompt_error': str(prompt_error),
                    })

                if selected is None:
                    masks_rejected += 1
                    rejection_counts['no_viable_candidate'] += 1
                    continue

                poly_pts = np.asarray(selected['polygon'], dtype=float)
                labels.append({
                    'class_id': prompt['class_id'],
                    'polygon': selected['polygon'],
                    'mask_area_px': float(selected['mask_area_px']),
                    'prompt_used': str(selected['prompt_used']),
                    'score': float(selected['score']),
                    'in_bbox_ratio': float(selected['in_bbox_ratio']),
                    'color_fraction': float(selected['color_fraction']),
                    'bbox_area_ratio': float(selected['bbox_area_ratio']),
                    'centroid_distance_norm': float(selected['centroid_distance_norm']),
                })
                prompt_used_counts[str(selected['prompt_used'])] = int(prompt_used_counts.get(str(selected['prompt_used']), 0)) + 1
                masks_accepted += 1

        out_image_dir = out / 'images' / split
        out_label_dir = out / 'labels' / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_image = out_image_dir / image_path.name
        out_label = out_label_dir / image_path.with_suffix('.txt').name
        shutil.copy2(image_path, out_image)
        _write_seg_labels(out_label, labels, img_w, img_h)

        if (image_idx + 1) % max(1, len(image_rows) // 10) == 0 or image_idx == 0:
            status = f'{len(labels)}/{len(prompts)} masks' if prompts else '0 prompts'
            print(f"  [{image_idx + 1:3d}/{len(image_rows)}] {image_path.name}: {status}")

        if image_idx < max(int(args.preview_count), 0):
            preview = out / 'previews' / split / image_path.name
            summary_lines = [f'Full-image SAM prompts: {len(labels)}/{len(prompts)} masks']
            if labels:
                chosen = labels[0]
                summary_lines.extend([
                    f"chosen={chosen['prompt_used']} score={chosen['score']:.2f}",
                    f"in_bbox={chosen['in_bbox_ratio']:.2f} color={chosen['color_fraction']:.2f}",
                    f"area={chosen['bbox_area_ratio']:.2f} centroid={chosen['centroid_distance_norm']:.2f}",
                ])
            elif prompts:
                summary_lines.append('no viable candidate')
            _write_preview(preview, image_bgr, prompts, labels, summary_lines)

    # Write manifest
    acceptance_rate = masks_accepted / (masks_accepted + masks_rejected) if (masks_accepted + masks_rejected) > 0 else 0
    candidate_csv_path = out / 'candidate_masks.csv'
    with candidate_csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'split',
            'image',
            'prompt_index',
            'prompt_mode',
            'candidate_index',
            'accepted',
            'reject_reason',
            'in_bbox_ratio',
            'color_fraction',
            'bbox_area_ratio',
            'centroid_distance_norm',
            'mask_area_px',
            'final_score',
            'prompt_error',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)
    manifest = {
        'input_dataset': str(source_root),
        'method': 'full_image_bbox_prompted_sam_segmentation',
        'sam_model': str(args.sam_model),
        'description': 'Offline SAM pseudo-labels from full-image prompts seeded by projected robot bboxes',
        'key_improvements': [
            'Run SAM on FULL image (not cropped to bbox)',
            'Use projected robot bboxes as the primary prompt and point prompts as a compatibility fallback',
            'Validate pseudo-mask overlap with the expected robot bbox region',
            'Keep SAM out of the runtime ROS perception loop',
        ],
        'settings': {
            'prompt_mode': str(args.prompt_mode),
            'prompt_modes_tried': list(prompt_modes),
            'prompt_pad_px': float(args.prompt_pad_px),
            'min_mask_area_px': float(args.min_mask_area_px),
            'min_bbox_overlap_ratio': float(args.min_bbox_overlap_ratio),
            'min_color_saturation': float(args.min_color_saturation),
            'min_color_value': float(args.min_color_value),
            'color_score_weight': float(args.color_score_weight),
            'area_penalty_weight': float(args.area_penalty_weight),
            'centroid_score_weight': float(args.centroid_score_weight),
            'max_centroid_distance_norm': float(args.max_centroid_distance_norm),
            'min_bbox_area_ratio': float(args.min_bbox_area_ratio),
            'max_bbox_area_ratio': float(args.max_bbox_area_ratio),
        },
        'results': {
            'total_images': len(image_rows),
            'total_prompts': total_prompts,
            'total_masks_accepted': masks_accepted,
            'total_masks_rejected': masks_rejected,
            'acceptance_rate': acceptance_rate,
            'prompt_used_counts': prompt_used_counts,
            'accepted_by_box': int(prompt_used_counts.get('box', 0)),
            'accepted_by_point': int(prompt_used_counts.get('point', 0)),
            'rejected_for_low_overlap': int(rejection_counts['low_overlap']),
            'rejected_for_centroid_distance': int(rejection_counts['centroid_distance']),
            'rejected_for_area_ratio': int(rejection_counts['area_ratio']),
            'rejected_for_small_area': int(rejection_counts['small_area']),
            'rejected_for_no_viable_candidate': int(rejection_counts['no_viable_candidate']),
            'candidate_csv': str(candidate_csv_path),
        },
    }
    (out / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")
    print(f"Total masks: {masks_accepted} accepted, {masks_rejected} rejected")
    print(f"Acceptance rate: {100.0 * acceptance_rate:.1f}%")
    print(f"Candidate diagnostics: {candidate_csv_path}")
    print()

    # Copy data.yaml
    src_yaml_str = (source_root / 'data.yaml').read_text(encoding='utf-8') if (source_root / 'data.yaml').is_file() else ""
    if not src_yaml_str:
        src_yaml_str = (src_yaml).read_text(encoding='utf-8')
    data_out = yaml.safe_load(src_yaml_str) or {}
    data_out['path'] = str(out.resolve())
    data_out['task'] = 'segment'
    (out / 'data.yaml').write_text(yaml.dump(data_out))
    print(f'Train with: python3 scripts/train_yolo_robot.py --task segment --model yolo11n-seg.pt --data {out / "data.yaml"}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
