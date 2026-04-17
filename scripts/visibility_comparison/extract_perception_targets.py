#!/usr/bin/env python3
"""Build the shared perception-target table from a canonical raw capture."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import (
    CAPTURE_COLUMNS,
    CURRENT_TARGETS_DIR,
    LOGS_ROOT,
    PERCEPTION_TARGET_COLUMNS,
    bbox_bottom_center,
    choose_preview_rows,
    ensure_repo_python_paths,
    format_xyxy_text,
    parse_bool01,
    parse_float,
    read_csv_rows,
    repo_relative,
    safe_reset_generated_dir,
    write_csv,
    write_manifest,
)

ensure_repo_python_paths()

from perception.core.yolo_selection import select_best_detection, target_class_ids


def _require_capture_columns(sample_rows: list[dict[str, str]]) -> None:
    if not sample_rows:
        raise RuntimeError('Capture samples.csv is empty')
    missing = [col for col in CAPTURE_COLUMNS if col not in sample_rows[0]]
    if missing:
        raise RuntimeError(f'Capture samples.csv is missing required columns: {missing}')


def _red_mask(
    image_bgr: np.ndarray,
    *,
    red_h1_low: int,
    red_h1_high: int,
    red_h2_low: int,
    red_h2_high: int,
    red_s_min: int,
    red_v_min: int,
    morph_kernel_px: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([int(red_h1_low), int(red_s_min), int(red_v_min)], dtype=np.uint8)
    upper1 = np.array([int(red_h1_high), 255, 255], dtype=np.uint8)
    lower2 = np.array([int(red_h2_low), int(red_s_min), int(red_v_min)], dtype=np.uint8)
    upper2 = np.array([int(red_h2_high), 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel_size = max(int(morph_kernel_px), 1)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_red_blob(mask: np.ndarray, *, min_blob_area_px: float):
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    best_idx = None
    best_area = -1.0
    for idx in range(1, n_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < float(min_blob_area_px):
            continue
        if area > best_area:
            best_idx = idx
            best_area = area
    if best_idx is None:
        return None

    blob_mask = (labels == best_idx)
    ys, xs = np.where(blob_mask)
    if xs.size == 0 or ys.size == 0:
        return None
    left = float(stats[best_idx, cv2.CC_STAT_LEFT])
    top = float(stats[best_idx, cv2.CC_STAT_TOP])
    width = float(stats[best_idx, cv2.CC_STAT_WIDTH])
    height = float(stats[best_idx, cv2.CC_STAT_HEIGHT])
    return {
        'area_px': float(best_area),
        'bbox_xyxy': np.asarray([left, top, left + width - 1.0, top + height - 1.0], dtype=float),
        'blob_mask': blob_mask,
        'xs': xs.astype(float),
        'ys': ys.astype(float),
    }


def _blob_bottom(blob: dict, *, band_px: float) -> tuple[float, float]:
    xs = np.asarray(blob['xs'], dtype=float)
    ys = np.asarray(blob['ys'], dtype=float)
    if xs.size == 0 or ys.size == 0:
        return math.nan, math.nan
    v_bottom = float(np.max(ys))
    band = xs[ys >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(xs)), v_bottom
    return float(np.mean(band)), v_bottom


def _draw_yolo_preview(
    image_bgr: np.ndarray,
    *,
    candidates: list[dict],
    selected: dict,
) -> np.ndarray:
    preview = image_bgr.copy()
    for detection in candidates:
        x0, y0, x1, y1 = [float(v) for v in detection['bbox']]
        cv2.rectangle(
            preview,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (180, 120, 255),
            1,
        )
        polygon = detection.get('polygon')
        if polygon is not None:
            pts = np.asarray(polygon, dtype=float).reshape(-1, 2).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(preview, [pts], True, (200, 120, 255), 1)
        cv2.putText(
            preview,
            f'{int(detection["class_id"])}:{float(detection["confidence"]):.2f}',
            (int(round(x0)), max(14, int(round(y0)) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 220, 255),
            1,
            cv2.LINE_AA,
        )

    if selected.get('detected', False):
        bbox = selected.get('bbox_xyxy')
        if bbox is not None:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            cv2.rectangle(
                preview,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                (255, 200, 0),
                2,
            )
        detection = selected.get('detection')
        if detection is not None and detection.get('polygon') is not None and selected.get('mask_available', 0) == 1:
            pts = np.asarray(detection['polygon'], dtype=float).reshape(-1, 2).astype(np.int32).reshape(-1, 1, 2)
            overlay = preview.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0.0)
            cv2.polylines(preview, [pts], True, (0, 200, 0), 2)
        u = float(selected.get('selected_u', math.nan))
        v = float(selected.get('selected_v', math.nan))
        if math.isfinite(u) and math.isfinite(v):
            cv2.circle(preview, (int(round(u)), int(round(v))), 5, (0, 255, 255), -1)
    cv2.putText(
        preview,
        (
            f'yolo_detected={1 if selected.get("detected", False) else 0} '
            f'score={float(selected.get("confidence", 0.0)):.2f} '
            f'source={str(selected.get("selected_pixel_source", "none"))}'
        ),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def main() -> int:
    parser = argparse.ArgumentParser(description='Create the shared perception_targets.csv contract from a raw capture.')
    parser.add_argument('--capture-dir', default='', help='Capture folder with samples.csv; defaults to logs/visibility_comparison/current_capture')
    parser.add_argument('--out', default=str(CURRENT_TARGETS_DIR))
    parser.add_argument('--preview-count', type=int, default=24)
    parser.add_argument('--red-min-blob-area-px', type=float, default=8.0)
    parser.add_argument('--red-morph-kernel-px', type=int, default=3)
    parser.add_argument('--red-h1-low', type=int, default=0)
    parser.add_argument('--red-h1-high', type=int, default=12)
    parser.add_argument('--red-h2-low', type=int, default=168)
    parser.add_argument('--red-h2-high', type=int, default=180)
    parser.add_argument('--red-s-min', type=int, default=70)
    parser.add_argument('--red-v-min', type=int, default=50)
    parser.add_argument('--red-bottom-band-px', type=float, default=3.0)
    parser.add_argument('--yolo-model', default='', help='Optional local YOLO .pt path for filling YOLO perception columns')
    parser.add_argument('--yolo-device', default='')
    parser.add_argument('--yolo-imgsz', type=int, default=512)
    parser.add_argument('--yolo-conf-threshold', type=float, default=0.10)
    parser.add_argument('--yolo-iou-threshold', type=float, default=0.45)
    parser.add_argument('--yolo-class-name', default='robot')
    parser.add_argument('--yolo-class-id', type=int, default=-1)
    parser.add_argument('--yolo-use-masks', default='true')
    parser.add_argument('--yolo-mask-min-area', type=float, default=12.0)
    parser.add_argument('--yolo-mask-bottom-band-px', type=float, default=3.0)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).expanduser().resolve() if str(args.capture_dir).strip() else (LOGS_ROOT / 'current_capture').resolve()
    samples_csv = capture_dir / 'samples.csv'
    manifest_path = capture_dir / 'capture_manifest.json'
    if not samples_csv.is_file():
        raise RuntimeError(f'Missing samples.csv in capture directory: {capture_dir}')
    if not manifest_path.is_file():
        raise RuntimeError(f'Missing capture_manifest.json in capture directory: {capture_dir}')

    output_dir = safe_reset_generated_dir(Path(args.out), allowed_root=LOGS_ROOT)
    previews_dir = output_dir / 'previews' / 'raw_capture'
    previews_dir.mkdir(parents=True, exist_ok=False)
    oracle_previews_dir = output_dir / 'previews' / 'oracle_visibility'
    oracle_previews_dir.mkdir(parents=True, exist_ok=False)
    red_previews_dir = output_dir / 'previews' / 'red_binary'
    red_previews_dir.mkdir(parents=True, exist_ok=False)
    yolo_model_path = Path(args.yolo_model).expanduser().resolve() if str(args.yolo_model).strip() else None
    yolo_binary_previews_dir = None
    yolo_confidence_previews_dir = None
    yolo_model = None
    yolo_target_ids = None
    yolo_enabled = yolo_model_path is not None
    if yolo_enabled:
        if not yolo_model_path.is_file():
            raise RuntimeError(f'YOLO model path does not exist: {yolo_model_path}')
        from ultralytics import YOLO

        yolo_binary_previews_dir = output_dir / 'previews' / 'yolo_binary'
        yolo_binary_previews_dir.mkdir(parents=True, exist_ok=False)
        yolo_confidence_previews_dir = output_dir / 'previews' / 'yolo_confidence'
        yolo_confidence_previews_dir.mkdir(parents=True, exist_ok=False)
        yolo_model = YOLO(str(yolo_model_path))
        yolo_target_ids = target_class_ids(
            getattr(yolo_model, 'names', {}),
            str(args.yolo_class_name),
            int(args.yolo_class_id),
        )
        if yolo_target_ids == set():
            raise RuntimeError(
                f'YOLO class filter did not match any model classes: '
                f'class_name={args.yolo_class_name!r}, class_id={int(args.yolo_class_id)}'
            )

    sample_rows = read_csv_rows(samples_csv)
    _require_capture_columns(sample_rows)
    target_rows: list[dict[str, str]] = []
    target_rows_by_id: dict[str, dict[str, str]] = {}
    oracle_visible_count = 0
    oracle_reason_counts: dict[str, int] = {}
    red_detected_count = 0
    red_area_values = []
    yolo_detected_count = 0
    yolo_mask_count = 0
    yolo_conf_values = []
    yolo_preview_payloads: dict[str, dict] = {}
    for row in sample_rows:
        oracle_visible = int(parse_bool01(row.get('oracle_visible', '0')))
        oracle_visible_count += oracle_visible
        oracle_reason = str(row.get('oracle_occlusion_reason', '')).strip() or 'unknown'
        oracle_reason_counts[oracle_reason] = oracle_reason_counts.get(oracle_reason, 0) + 1

        image_rel = str(row.get('image_path', '')).strip()
        image_path = (capture_dir / image_rel).resolve() if image_rel else None
        red_detected = 0
        red_area = math.nan
        red_bbox_text = ''
        red_bottom_u = math.nan
        red_bottom_v = math.nan
        yolo_detected = '0' if yolo_enabled else ''
        yolo_confidence = '0.00000000' if yolo_enabled else ''
        yolo_mask_area = ''
        yolo_bbox_text = ''
        yolo_bottom_u = ''
        yolo_bottom_v = ''
        if image_path is not None and image_path.is_file():
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is not None:
                red_mask = _red_mask(
                    image_bgr,
                    red_h1_low=int(args.red_h1_low),
                    red_h1_high=int(args.red_h1_high),
                    red_h2_low=int(args.red_h2_low),
                    red_h2_high=int(args.red_h2_high),
                    red_s_min=int(args.red_s_min),
                    red_v_min=int(args.red_v_min),
                    morph_kernel_px=int(args.red_morph_kernel_px),
                )
                blob = _largest_red_blob(red_mask, min_blob_area_px=float(args.red_min_blob_area_px))
                if blob is not None:
                    red_detected = 1
                    red_area = float(blob['area_px'])
                    red_bbox_text = format_xyxy_text(blob['bbox_xyxy'])
                    red_bottom_u, red_bottom_v = bbox_bottom_center(blob['bbox_xyxy'])
                    red_detected_count += 1
                    red_area_values.append(float(red_area))
                if yolo_enabled and yolo_model is not None:
                    kwargs = {
                        'source': image_bgr,
                        'imgsz': int(args.yolo_imgsz),
                        'conf': float(args.yolo_conf_threshold),
                        'iou': float(args.yolo_iou_threshold),
                        'verbose': False,
                    }
                    if str(args.yolo_device).strip():
                        kwargs['device'] = str(args.yolo_device).strip()
                    results = yolo_model.predict(**kwargs)
                    result = results[0] if results else None
                    if result is not None:
                        selected = select_best_detection(
                            result,
                            target_ids=yolo_target_ids,
                            confidence_threshold=float(args.yolo_conf_threshold),
                            use_masks=str(args.yolo_use_masks).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on'),
                            mask_min_area=float(args.yolo_mask_min_area),
                            mask_bottom_band_px=float(args.yolo_mask_bottom_band_px),
                        )
                        if bool(selected.get('detected', False)):
                            yolo_detected = '1'
                            yolo_confidence = f'{float(selected["confidence"]):.8f}'
                            yolo_bbox_text = format_xyxy_text(selected.get('bbox_xyxy'))
                            yolo_bottom_u = f'{float(selected["selected_u"]):.8f}'
                            yolo_bottom_v = f'{float(selected["selected_v"]):.8f}'
                            yolo_detected_count += 1
                            yolo_conf_values.append(float(selected['confidence']))
                            mask_area = float(selected.get('mask_area', math.nan))
                            if math.isfinite(mask_area):
                                yolo_mask_area = f'{mask_area:.8f}'
                                yolo_mask_count += 1
                        yolo_preview_payloads[str(row.get('sample_id', '')).strip()] = {
                            'candidates': list(selected.get('candidates', [])),
                            'selected': selected,
                        }
                    else:
                        yolo_preview_payloads[str(row.get('sample_id', '')).strip()] = {
                            'candidates': [],
                            'selected': {
                                'detected': False,
                                'confidence': 0.0,
                                'selected_pixel_source': 'none',
                                'selected_u': math.nan,
                                'selected_v': math.nan,
                                'bbox_xyxy': None,
                            },
                        }

        target_row = {
            'sample_id': str(row.get('sample_id', '')).strip(),
            'x': str(row.get('x', '')).strip(),
            'y': str(row.get('y', '')).strip(),
            'theta': str(row.get('theta', '')).strip(),
            'image_path': image_rel,
            'red_detected': str(int(red_detected)),
            'red_area': '' if not math.isfinite(red_area) else f'{float(red_area):.8f}',
            'red_bbox_xyxy': red_bbox_text,
            'red_bottom_u': '' if not math.isfinite(red_bottom_u) else f'{float(red_bottom_u):.8f}',
            'red_bottom_v': '' if not math.isfinite(red_bottom_v) else f'{float(red_bottom_v):.8f}',
            'yolo_detected': yolo_detected,
            'yolo_confidence': yolo_confidence,
            'yolo_mask_area': yolo_mask_area,
            'yolo_bbox_xyxy': yolo_bbox_text,
            'yolo_bottom_u': yolo_bottom_u,
            'yolo_bottom_v': yolo_bottom_v,
            'oracle_visible': str(oracle_visible),
            'oracle_bottom_u': '' if not math.isfinite(parse_float(row.get('oracle_bottom_u', ''), math.nan)) else str(row.get('oracle_bottom_u', '')).strip(),
            'oracle_bottom_v': '' if not math.isfinite(parse_float(row.get('oracle_bottom_v', ''), math.nan)) else str(row.get('oracle_bottom_v', '')).strip(),
        }
        target_rows.append(target_row)
        target_rows_by_id[target_row['sample_id']] = target_row

    preview_rows = choose_preview_rows(sample_rows, int(args.preview_count))
    preview_index = []
    method_preview_index = []
    for row in preview_rows:
        sample_id = str(row.get('sample_id', '')).strip()
        raw_preview_rel = str(row.get('preview_path', '')).strip()
        if not raw_preview_rel:
            continue
        raw_preview = (capture_dir / raw_preview_rel).resolve()
        if not raw_preview.is_file():
            continue
        out_preview = previews_dir / raw_preview.name
        shutil.copy2(raw_preview, out_preview)
        preview_index.append({
            'sample_id': sample_id,
            'source_preview': raw_preview_rel,
            'copied_preview': repo_relative(out_preview, output_dir),
        })
        oracle_out_preview = oracle_previews_dir / raw_preview.name
        shutil.copy2(raw_preview, oracle_out_preview)
        method_preview_index.append({
            'method_id': 'oracle_visibility',
            'sample_id': sample_id,
            'source_preview': raw_preview_rel,
            'copied_preview': repo_relative(oracle_out_preview, output_dir),
        })
        image_rel = str(row.get('image_path', '')).strip()
        image_path = (capture_dir / image_rel).resolve() if image_rel else None
        target_row = target_rows_by_id.get(sample_id)
        if image_path is not None and image_path.is_file() and target_row is not None:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is not None:
                preview = image_bgr.copy()
                bbox = target_row['red_bbox_xyxy']
                if bbox:
                    parts = [float(v) for v in bbox.split()]
                    cv2.rectangle(
                        preview,
                        (int(round(parts[0])), int(round(parts[1]))),
                        (int(round(parts[2])), int(round(parts[3]))),
                        (0, 200, 255),
                        2,
                    )
                u = parse_float(target_row.get('red_bottom_u', ''), math.nan)
                v = parse_float(target_row.get('red_bottom_v', ''), math.nan)
                if math.isfinite(u) and math.isfinite(v):
                    cv2.circle(preview, (int(round(u)), int(round(v))), 5, (0, 255, 0), -1)
                cv2.putText(
                    preview,
                    f'red_detected={target_row["red_detected"]} area={target_row["red_area"] or "nan"}',
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                red_out_preview = red_previews_dir / raw_preview.name
                cv2.imwrite(str(red_out_preview), preview)
                method_preview_index.append({
                    'method_id': 'red_binary',
                    'sample_id': sample_id,
                    'source_preview': image_rel,
                    'copied_preview': repo_relative(red_out_preview, output_dir),
                })
                if yolo_enabled and yolo_binary_previews_dir is not None and yolo_confidence_previews_dir is not None:
                    payload = yolo_preview_payloads.get(sample_id, {'candidates': [], 'selected': {'detected': False, 'confidence': 0.0, 'selected_pixel_source': 'none'}})
                    yolo_preview = _draw_yolo_preview(
                        image_bgr,
                        candidates=list(payload.get('candidates', [])),
                        selected=dict(payload.get('selected', {})),
                    )
                    yolo_binary_out = yolo_binary_previews_dir / raw_preview.name
                    cv2.imwrite(str(yolo_binary_out), yolo_preview)
                    method_preview_index.append({
                        'method_id': 'yolo_binary',
                        'sample_id': sample_id,
                        'source_preview': image_rel,
                        'copied_preview': repo_relative(yolo_binary_out, output_dir),
                    })
                    yolo_conf_out = yolo_confidence_previews_dir / raw_preview.name
                    cv2.imwrite(str(yolo_conf_out), yolo_preview)
                    method_preview_index.append({
                        'method_id': 'yolo_confidence',
                        'sample_id': sample_id,
                        'source_preview': image_rel,
                        'copied_preview': repo_relative(yolo_conf_out, output_dir),
                    })

    write_csv(output_dir / 'perception_targets.csv', PERCEPTION_TARGET_COLUMNS, target_rows)
    write_csv(output_dir / 'preview_index.csv', ('sample_id', 'source_preview', 'copied_preview'), preview_index)
    write_csv(
        output_dir / 'method_preview_index.csv',
        ('method_id', 'sample_id', 'source_preview', 'copied_preview'),
        method_preview_index,
    )
    write_manifest(output_dir / 'manifest.json', {
        'capture_dir': str(capture_dir),
        'capture_manifest': str(manifest_path),
        'row_count': int(len(target_rows)),
        'preview_count': int(len(preview_index)),
        'shared_only_stage': False,
        'implemented_methods': ['oracle_visibility', 'red_binary'] + (['yolo_binary', 'yolo_confidence'] if yolo_enabled else []),
        'deferred_methods': ['red_area_corrected'] + ([] if yolo_enabled else ['yolo_binary', 'yolo_confidence']),
        'oracle_summary': {
            'visible_count': int(oracle_visible_count),
            'hidden_count': int(len(target_rows) - oracle_visible_count),
            'reason_counts': oracle_reason_counts,
        },
        'red_binary_summary': {
            'detected_count': int(red_detected_count),
            'undetected_count': int(len(target_rows) - red_detected_count),
            'mean_area_px': float(np.mean(red_area_values)) if red_area_values else math.nan,
            'detector_params': {
                'min_blob_area_px': float(args.red_min_blob_area_px),
                'morph_kernel_px': int(args.red_morph_kernel_px),
                'red_h1_low': int(args.red_h1_low),
                'red_h1_high': int(args.red_h1_high),
                'red_h2_low': int(args.red_h2_low),
                'red_h2_high': int(args.red_h2_high),
                'red_s_min': int(args.red_s_min),
                'red_v_min': int(args.red_v_min),
                'bottom_band_px': float(args.red_bottom_band_px),
                'reference_point_rule': 'bbox_bottom_center_runtime_aligned',
            },
        },
        'yolo_summary': {
            'enabled': bool(yolo_enabled),
            'detected_count': int(yolo_detected_count),
            'undetected_count': int(len(target_rows) - yolo_detected_count) if yolo_enabled else math.nan,
            'mask_count': int(yolo_mask_count),
            'mean_confidence': float(np.mean(yolo_conf_values)) if yolo_conf_values else math.nan,
            'model_path': str(yolo_model_path) if yolo_model_path is not None else '',
            'class_name': str(args.yolo_class_name),
            'class_id': int(args.yolo_class_id),
            'device': str(args.yolo_device).strip(),
            'imgsz': int(args.yolo_imgsz),
            'conf_threshold': float(args.yolo_conf_threshold),
            'iou_threshold': float(args.yolo_iou_threshold),
            'use_masks': str(args.yolo_use_masks).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on'),
            'mask_min_area': float(args.yolo_mask_min_area),
            'mask_bottom_band_px': float(args.yolo_mask_bottom_band_px),
            'confidence_calibrated': False,
        },
        'notes': [
            'Oracle/reference visibility is the first fully implemented method in the comparison backbone.',
            'Oracle columns are passed through from the raw geometry-based capture because the shared capture already computes them directly from world geometry.',
            'Red binary is now implemented using the same simple HSV + morphology + largest-blob rule as the old runtime marker detector.',
            'Offline red_bottom_u/v now use the runtime-aligned bbox-bottom-center rule; red_bottom_band_px is retained only for backward-compatible manifests.',
            'YOLO binary and YOLO confidence now reuse the same highest-confidence mask/bbox-bottom selection rule as the YOLO runtime detector.',
            'YOLO confidence is stored as an uncalibrated detector score.',
            'Red corrected area remains deferred to a later method-specific pass.',
        ],
    })
    print(f'Wrote perception targets to {output_dir}')
    print(f'Rows: {len(target_rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
