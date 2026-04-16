#!/usr/bin/env python3
"""Shared helpers for the visibility-comparison backbone."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / 'logs' / 'visibility_comparison'
CURRENT_CAPTURE_DIR = LOGS_ROOT / 'current_capture'
CURRENT_TARGETS_DIR = LOGS_ROOT / 'current_targets'
CURRENT_GP_DIR = LOGS_ROOT / 'current_gp'
PLANNER_RUNS_DIR = LOGS_ROOT / 'planner_runs'
REPORT_DIR = LOGS_ROOT / 'report'

ACTIVE_METHOD_IDS = (
    'red_binary',
    'red_area_corrected',
    'yolo_binary',
    'yolo_confidence',
    'oracle_visibility',
    'visibility_unaware_baseline',
)

CAPTURE_COLUMNS = (
    'sample_id',
    'image_path',
    'preview_path',
    'x',
    'y',
    'theta',
    'timestamp',
    'world',
    'camera_frame',
    'oracle_visible',
    'oracle_bottom_u',
    'oracle_bottom_v',
    'oracle_occlusion_reason',
    'segmentation_path',
    'labels_map_path',
    'robot_label',
)

PERCEPTION_TARGET_COLUMNS = (
    'sample_id',
    'x',
    'y',
    'theta',
    'image_path',
    'red_detected',
    'red_area',
    'red_bbox_xyxy',
    'red_bottom_u',
    'red_bottom_v',
    'yolo_detected',
    'yolo_confidence',
    'yolo_mask_area',
    'yolo_bbox_xyxy',
    'yolo_bottom_u',
    'yolo_bottom_v',
    'oracle_visible',
    'oracle_bottom_u',
    'oracle_bottom_v',
)

GP_TARGET_COLUMNS = (
    'sample_id',
    'x',
    'y',
    'red_binary',
    'red_area_corrected',
    'yolo_binary',
    'yolo_confidence',
    'oracle_visibility',
)


def ensure_repo_python_paths() -> None:
    for rel in ('src/experiments', 'src/perception', 'src/planning', 'src/unav_common'):
        path = str((REPO_ROOT / rel).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def repo_relative(path: Path | str, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def json_dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=False)


def write_manifest(path: Path | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload), encoding='utf-8')


def safe_reset_generated_dir(path: Path | str, *, allowed_root: Path | None = None) -> Path:
    path = Path(path).expanduser().resolve()
    root = (allowed_root or LOGS_ROOT).expanduser().resolve()
    if root not in path.parents and path != root:
        raise RuntimeError(f'Refusing to reset directory outside allowed root {root}: {path}')
    if path == root:
        raise RuntimeError(f'Refusing to reset the allowed root itself: {path}')
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path | str, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value, default: float = math.nan) -> float:
    if value in (None, ''):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def finite_or_blank(value: float) -> str:
    value = float(value)
    return '' if not math.isfinite(value) else f'{value:.8f}'


def parse_bool01(value) -> int:
    if value in (None, ''):
        return 0
    text = str(value).strip().lower()
    if text in ('1', 'true', 't', 'yes', 'y', 'on'):
        return 1
    try:
        return int(float(text) >= 0.5)
    except (TypeError, ValueError):
        return 0


def empty_string_row(columns: Sequence[str]) -> dict[str, str]:
    return {str(col): '' for col in columns}


def parse_xyxy_text(value: str) -> tuple[float, float, float, float] | None:
    raw = str(value or '').strip()
    if not raw:
        return None
    parts = [part for part in raw.replace(',', ' ').split() if part]
    if len(parts) != 4:
        return None
    try:
        vals = tuple(float(part) for part in parts)
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return vals  # type: ignore[return-value]


def format_xyxy_text(bbox: Sequence[float] | None) -> str:
    if bbox is None:
        return ''
    vals = [float(v) for v in bbox]
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
        return ''
    return ' '.join(f'{v:.3f}' for v in vals)


def clip_bbox(bbox: Sequence[float], img_w: int, img_h: int) -> np.ndarray | None:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0 = float(np.clip(x0, 0.0, img_w - 1.0))
    y0 = float(np.clip(y0, 0.0, img_h - 1.0))
    x1 = float(np.clip(x1, 0.0, img_w - 1.0))
    y1 = float(np.clip(y1, 0.0, img_h - 1.0))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.asarray([x0, y0, x1, y1], dtype=float)


def project_world_point(camera, xyz: np.ndarray) -> np.ndarray | None:
    cam_pt = camera.R @ (np.asarray(xyz, dtype=float) - camera.cam_pos)
    if cam_pt[2] <= 1e-6:
        return None
    pixel_h = camera.K @ cam_pt
    return np.asarray([pixel_h[0] / pixel_h[2], pixel_h[1] / pixel_h[2]], dtype=float)


def project_robot_bbox(
    camera,
    *,
    x: float,
    y: float,
    yaw: float,
    box_length: float,
    box_width: float,
    box_height: float,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    hx = 0.5 * float(box_length)
    hy = 0.5 * float(box_width)
    hz = float(box_height)
    local_corners = np.asarray([
        [-hx, -hy, 0.0],
        [-hx, hy, 0.0],
        [hx, -hy, 0.0],
        [hx, hy, 0.0],
        [-hx, -hy, hz],
        [-hx, hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
    ], dtype=float)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.asarray([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    world_corners = (rot @ local_corners.T).T
    world_corners[:, 0] += float(x)
    world_corners[:, 1] += float(y)

    pixels = []
    for corner in world_corners:
        uv = project_world_point(camera, corner)
        if uv is None:
            return None
        pixels.append(uv)
    pts = np.asarray(pixels, dtype=float)
    return clip_bbox(
        [float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))],
        img_w,
        img_h,
    )


def bbox_bottom_center(bbox: Sequence[float] | None) -> tuple[float, float]:
    if bbox is None:
        return math.nan, math.nan
    x0, _y0, x1, y1 = [float(v) for v in bbox]
    return float(0.5 * (x0 + x1)), float(y1)


def choose_preview_rows(rows: Sequence[dict], limit: int) -> list[dict]:
    rows = list(rows)
    if limit <= 0 or len(rows) <= limit:
        return rows
    stride = max(int(math.ceil(len(rows) / float(limit))), 1)
    return rows[::stride][:limit]
