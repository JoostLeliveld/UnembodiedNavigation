from __future__ import annotations

import math

import numpy as np


def target_class_ids(names, class_name: str, class_id: int):
    if int(class_id) >= 0:
        return {int(class_id)}
    target = str(class_name or '').strip().lower()
    if target in ('', '*', 'any', 'all'):
        return None
    if isinstance(names, dict):
        return {int(idx) for idx, name in names.items() if str(name).strip().lower() == target}
    return set()


def mask_bottom(points: np.ndarray, band_px: float) -> tuple[float, float]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return math.nan, math.nan
    v_bottom = float(np.max(points[:, 1]))
    band = points[points[:, 1] >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(points[:, 0])), v_bottom
    return float(np.mean(band[:, 0])), v_bottom


def polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return math.nan
    return float(
        abs(
            0.5
            * (
                np.dot(points[:, 0], np.roll(points[:, 1], -1))
                - np.dot(points[:, 1], np.roll(points[:, 0], -1))
            )
        )
    )


def collect_candidate_detections(result, target_ids, confidence_threshold: float):
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

    masks = getattr(result, 'masks', None)
    mask_polygons = getattr(masks, 'xy', None) if masks is not None else None
    detections = []
    for idx in idxs:
        idx = int(idx)
        polygon = None
        if mask_polygons is not None and idx < len(mask_polygons):
            pts = np.asarray(mask_polygons[idx], dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                pts = pts[:, :2]
                pts = pts[np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])]
                if pts.shape[0] >= 3:
                    polygon = pts
        detections.append({
            'bbox': xyxy[idx, :4].astype(float),
            'confidence': float(conf[idx]),
            'class_id': int(cls[idx]),
            'polygon': polygon,
        })
    detections.sort(key=lambda item: float(item['confidence']), reverse=True)
    return detections


def select_best_detection(
    result,
    *,
    target_ids,
    confidence_threshold: float,
    use_masks: bool,
    mask_min_area: float,
    mask_bottom_band_px: float,
):
    detections = collect_candidate_detections(result, target_ids, confidence_threshold)
    if not detections:
        return {
            'detected': False,
            'confidence': 0.0,
            'class_id': math.nan,
            'bbox_xyxy': None,
            'mask_area': math.nan,
            'selected_u': math.nan,
            'selected_v': math.nan,
            'selected_pixel_source': 'none',
            'mask_available': 0,
            'n_candidates': 0,
            'detection': None,
            'candidates': detections,
        }

    detection = detections[0]
    bbox = np.asarray(detection['bbox'], dtype=float)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    polygon = detection['polygon']
    mask_area = math.nan
    mask_available = 0
    selected_source = 'bbox_bottom'
    selected_u = float(0.5 * (x0 + x1))
    selected_v = float(y1)
    if use_masks and polygon is not None:
        area = polygon_area(polygon)
        if math.isfinite(area) and area >= float(mask_min_area):
            mask_area = float(area)
            mask_available = 1
            selected_u, selected_v = mask_bottom(polygon, mask_bottom_band_px)
            selected_source = 'mask_bottom'

    return {
        'detected': True,
        'confidence': float(detection['confidence']),
        'class_id': int(detection['class_id']),
        'bbox_xyxy': bbox,
        'mask_area': float(mask_area) if math.isfinite(mask_area) else math.nan,
        'selected_u': float(selected_u),
        'selected_v': float(selected_v),
        'selected_pixel_source': selected_source,
        'mask_available': int(mask_available),
        'n_candidates': int(len(detections)),
        'detection': detection,
        'candidates': detections,
    }
