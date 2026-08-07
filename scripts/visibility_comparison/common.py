#!/usr/bin/env python3
"""Shared helpers for the visibility-comparison backbone."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / 'logs' / 'visibility_comparison'
CURRENT_CAPTURE_DIR = LOGS_ROOT / 'current_capture'
CURRENT_TARGETS_DIR = LOGS_ROOT / 'current_targets'
CURRENT_GP_DIR = LOGS_ROOT / 'current_gp'
PLANNER_RUNS_DIR = LOGS_ROOT / 'planner_runs'
REPORT_DIR = LOGS_ROOT / 'report'
DEFAULT_WORLD_PROFILES_PATH = REPO_ROOT / 'src' / 'experiments' / 'config' / 'world_profiles.yaml'
ARTIFACT_SCHEMA_VERSION = 2
ACCEPTED_COMPLETION_REASONS = ('goal_reached', 'goal_reached_stable')
PLOTTABLE_COMPLETION_REASONS = ACCEPTED_COMPLETION_REASONS + (
    'timeout_after_first_cmd',
    'stuck',
    'interrupted',
)
PAPER_VISIBILITY_DEFAULTS = {
    'r_visible_uv': 2.5,
    'r_miss_uv': 120.0,
    'visibility_sigma_kappa': 1.0,
}

ACTIVE_METHOD_IDS = (
    'yolo_score_raw',
    'constant_R_efe',
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
    'camera_relative_bearing_deg',
    'yolo_detected_after_threshold',
    'yolo_score_raw',
    'yolo_raw_best_score',
    'yolo_score_selected',
    'yolo_selected_score',
    'yolo_best_class_id',
    'yolo_selected_class_id',
    'yolo_num_target_candidates',
    'yolo_mask_area',
    'yolo_bbox_area',
    'yolo_bbox_xyxy',
    'yolo_bottom_u',
    'yolo_bottom_v',
    'yolo_selected_pixel_source',
    'yolo_selected_pixel_source_code',
    'oracle_visible',
    'oracle_bottom_u',
    'oracle_bottom_v',
    # Appended for multicam captures: four cameras share one sample_id, so without this
    # the rows are distinguishable only by image_path. Readers use DictReader, so older
    # single-camera consumers ignore it.
    'camera_frame',
)

GP_TARGET_COLUMNS = (
    'sample_id',
    'x',
    'y',
    'yolo_score_raw',
    'yolo_raw_best_score',
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def visibility_geometry_sha256(geometry_json: str) -> str:
    return sha256_text(str(geometry_json or '').strip())


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


def accepted_completed_run(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, Mapping):
        return False
    return bool(summary.get('completed', False)) and str(summary.get('completion_reason', '')).strip() in ACCEPTED_COMPLETION_REASONS


def accepted_plotworthy_run(
    summary: Mapping[str, Any] | None,
    *,
    include_interrupted: bool = True,
) -> bool:
    if not isinstance(summary, Mapping):
        return False
    reason = str(summary.get('completion_reason', '')).strip()
    if accepted_completed_run(summary):
        return True
    return bool(include_interrupted) and reason == 'interrupted'


def run_has_usable_logs(run_dir: Path | str) -> bool:
    run_dir = Path(run_dir)
    for name in ('experiment.csv', 'perception.csv', 'plan_samples.csv'):
        path = run_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return True
    return False


def normalize_run_manifest_for_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {'run_id', 'timestamp', 'frame_sanity'}
    return {
        str(key): value
        for key, value in dict(manifest).items()
        if key not in ignored
    }


def expected_visibility_metadata(
    world: str,
    *,
    world_profiles_path: Path | str = DEFAULT_WORLD_PROFILES_PATH,
) -> dict[str, Any]:
    ensure_repo_python_paths()
    from experiments.core.world_profiles import (
        compute_look_at_from_pose,
        load_profile,
        serialize_occlusion_geometry_from_world,
    )

    profile, intrinsics, world_path, camera_pose = load_profile(str(Path(world_profiles_path).resolve()), str(world))
    cam_pos = np.asarray(camera_pose[:3], dtype=float)
    look_at = np.asarray(
        compute_look_at_from_pose(cam_pos, camera_pose[3], camera_pose[4], camera_pose[5]),
        dtype=float,
    )
    geometry_json = str(serialize_occlusion_geometry_from_world(world_path))
    return {
        'world': str(world),
        'world_name': str(profile.get('world_name', '')),
        'world_path': str(Path(world_path).resolve()),
        'camera_pose': [float(v) for v in camera_pose],
        'camera_pos': [float(v) for v in cam_pos],
        'look_at': [float(v) for v in look_at],
        'img_width': int(intrinsics['img_width']),
        'img_height': int(intrinsics['img_height']),
        'fov_h_rad': float(intrinsics['fov_h_rad']),
        'geometry_json': geometry_json,
        'geometry_sha256': visibility_geometry_sha256(geometry_json),
    }


def _compare_numeric(name: str, actual: Any, expected: Any, mismatches: list[str], *, tol: float = 1e-9) -> None:
    try:
        if abs(float(actual) - float(expected)) > float(tol):
            mismatches.append(f'{name}: actual={actual!r} expected={expected!r}')
    except (TypeError, ValueError):
        mismatches.append(f'{name}: actual={actual!r} expected={expected!r}')


def validate_visibility_metadata(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    mismatches: list[str] = []
    for key in ('world', 'world_name', 'world_path', 'geometry_sha256'):
        if str(actual.get(key, '')).strip() != str(expected.get(key, '')).strip():
            mismatches.append(f'{key}: actual={actual.get(key)!r} expected={expected.get(key)!r}')
    for key in ('img_width', 'img_height'):
        _compare_numeric(key, actual.get(key), expected.get(key), mismatches, tol=0.0)
    for key in ('fov_h_rad',):
        _compare_numeric(key, actual.get(key), expected.get(key), mismatches)
    for key in ('camera_pose', 'camera_pos', 'look_at'):
        actual_values = np.asarray(actual.get(key, []), dtype=float).reshape(-1)
        expected_values = np.asarray(expected.get(key, []), dtype=float).reshape(-1)
        if actual_values.shape != expected_values.shape:
            mismatches.append(f'{key}: actual_shape={actual_values.shape} expected_shape={expected_values.shape}')
            continue
        for idx, (actual_value, expected_value) in enumerate(zip(actual_values, expected_values)):
            if abs(float(actual_value) - float(expected_value)) > 1e-9:
                mismatches.append(f'{key}[{idx}]: actual={actual_value!r} expected={expected_value!r}')
    if mismatches:
        raise RuntimeError(f'{label} metadata does not match the requested world profile: ' + '; '.join(mismatches))


def load_visibility_artifact_metadata(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f'Visibility artifact not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        def _scalar_string(name: str) -> str:
            if name not in data.files:
                return ''
            arr = np.asarray(data[name])
            return str(arr.reshape(-1)[0]) if arr.size else ''

        def _float_list(name: str) -> list[float]:
            if name not in data.files:
                return []
            return [float(v) for v in np.asarray(data[name], dtype=float).reshape(-1)]

        return {
            'artifact_path': str(path),
            'artifact_sha256': sha256_file(path),
            'artifact_schema_version': int(np.asarray(data['artifact_schema_version']).reshape(-1)[0]) if 'artifact_schema_version' in data.files else 0,
            'world': _scalar_string('world'),
            'world_name': _scalar_string('world_name'),
            'world_path': _scalar_string('world_path'),
            'camera_pose': _float_list('camera_pose'),
            'camera_pos': _float_list('camera_pos'),
            'look_at': _float_list('look_at'),
            'img_width': int(np.asarray(data['img_width']).reshape(-1)[0]) if 'img_width' in data.files else 0,
            'img_height': int(np.asarray(data['img_height']).reshape(-1)[0]) if 'img_height' in data.files else 0,
            'fov_h_rad': float(np.asarray(data['fov_h_rad']).reshape(-1)[0]) if 'fov_h_rad' in data.files else math.nan,
            'geometry_sha256': _scalar_string('geometry_sha256'),
        }


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
