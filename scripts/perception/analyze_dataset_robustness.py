#!/usr/bin/env python3
"""Audit yaw coverage and split leakage in local perception datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOT = REPO_ROOT / 'logs'


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _parse_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _dataset_dirs(search_root: Path) -> list[Path]:
    candidates = set()
    for name in ('capture_manifest.json', 'dataset_manifest.json', 'samples.csv', 'capture_diagnostics.csv', 'label_diagnostics.csv'):
        for path in search_root.rglob(name):
            candidates.add(path.parent)
    return sorted(candidates)


def _kind_for_dir(dataset_dir: Path) -> str:
    if (dataset_dir / 'samples.csv').is_file():
        return 'visibility_capture'
    if (dataset_dir / 'capture_diagnostics.csv').is_file():
        return 'projected_bbox_dataset'
    if (dataset_dir / 'label_diagnostics.csv').is_file():
        rows = _read_csv_rows(dataset_dir / 'label_diagnostics.csv')
        if rows and 'robot_yaw' in rows[0]:
            return 'simseg_dataset'
        return 'pseudolabel_dataset'
    return 'unknown'


def _pose_rows(dataset_dir: Path, kind: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if kind == 'visibility_capture':
        for row in _read_csv_rows(dataset_dir / 'samples.csv'):
            rows.append({
                'split': 'all',
                'accepted': 1.0,
                'x': _parse_float(row.get('x')),
                'y': _parse_float(row.get('y')),
                'yaw': _parse_float(row.get('theta')),
            })
        return rows
    if kind == 'projected_bbox_dataset':
        for row in _read_csv_rows(dataset_dir / 'capture_diagnostics.csv'):
            rows.append({
                'split': str(row.get('split', '') or '').strip() or 'unspecified',
                'accepted': float(int((_parse_float(row.get('accepted')) or 0.0) > 0.5)),
                'x': _parse_float(row.get('x')),
                'y': _parse_float(row.get('y')),
                'yaw': _parse_float(row.get('yaw_rad')),
            })
        return rows
    if kind == 'simseg_dataset':
        for row in _read_csv_rows(dataset_dir / 'label_diagnostics.csv'):
            rows.append({
                'split': str(row.get('split', '') or '').strip() or 'unspecified',
                'accepted': float(int((_parse_float(row.get('accepted')) or 0.0) > 0.5)),
                'x': _parse_float(row.get('robot_x')),
                'y': _parse_float(row.get('robot_y')),
                'yaw': _parse_float(row.get('robot_yaw')),
            })
        return rows
    return rows


def _unique_sorted(values: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return finite
    finite.sort()
    unique = [float(finite[0])]
    for value in finite[1:]:
        if abs(float(value) - unique[-1]) > float(tol):
            unique.append(float(value))
    return np.asarray(unique, dtype=float)


def _grid_spacing(values: np.ndarray) -> float:
    unique = _unique_sorted(values)
    if unique.size <= 1:
        return math.nan
    diffs = np.diff(unique)
    diffs = diffs[diffs > 1e-9]
    return float(np.min(diffs)) if diffs.size else math.nan


def _min_cross_split_distance(rows: list[dict[str, float | str]]) -> float:
    train_pts = np.asarray(
        [[float(row['x']), float(row['y'])] for row in rows if str(row['split']) == 'train' and float(row['accepted']) > 0.5 and math.isfinite(float(row['x'])) and math.isfinite(float(row['y']))],
        dtype=float,
    )
    val_pts = np.asarray(
        [[float(row['x']), float(row['y'])] for row in rows if str(row['split']) == 'val' and float(row['accepted']) > 0.5 and math.isfinite(float(row['x'])) and math.isfinite(float(row['y']))],
        dtype=float,
    )
    if train_pts.size == 0 or val_pts.size == 0:
        return math.nan
    min_dist = math.inf
    for point in val_pts:
        dists = np.linalg.norm(train_pts - point, axis=1)
        min_dist = min(min_dist, float(np.min(dists)))
    return float(min_dist) if math.isfinite(min_dist) else math.nan


def analyze_dataset_dir(dataset_dir: Path) -> dict:
    kind = _kind_for_dir(dataset_dir)
    manifest = _load_json(dataset_dir / 'capture_manifest.json') or _load_json(dataset_dir / 'dataset_manifest.json')
    rows = _pose_rows(dataset_dir, kind)
    x_vals = np.asarray([float(row['x']) for row in rows], dtype=float) if rows else np.array([], dtype=float)
    y_vals = np.asarray([float(row['y']) for row in rows], dtype=float) if rows else np.array([], dtype=float)
    yaw_vals = np.asarray([float(row['yaw']) for row in rows], dtype=float) if rows else np.array([], dtype=float)
    splits = [str(row['split']) for row in rows]
    accepted = np.asarray([float(row['accepted']) for row in rows], dtype=float) if rows else np.array([], dtype=float)

    unique_yaws = _unique_sorted(np.mod(yaw_vals, 2.0 * math.pi))
    split_counts: dict[str, int] = {}
    for split in splits:
        split_counts[split] = split_counts.get(split, 0) + 1

    exact_pose_overlap = 0
    train_pose_keys = {
        (round(float(row['x']), 6), round(float(row['y']), 6), round(float((float(row['yaw']) % (2.0 * math.pi))), 6))
        for row in rows
        if str(row['split']) == 'train' and float(row['accepted']) > 0.5 and math.isfinite(float(row['x'])) and math.isfinite(float(row['y'])) and math.isfinite(float(row['yaw']))
    }
    for row in rows:
        if str(row['split']) != 'val' or float(row['accepted']) <= 0.5:
            continue
        key = (round(float(row['x']), 6), round(float(row['y']), 6), round(float((float(row['yaw']) % (2.0 * math.pi))), 6))
        exact_pose_overlap += int(key in train_pose_keys)

    x_spacing = _grid_spacing(x_vals)
    y_spacing = _grid_spacing(y_vals)
    nearest_cross_split_distance = _min_cross_split_distance(rows)

    warnings: list[str] = []
    if kind in ('visibility_capture', 'projected_bbox_dataset', 'simseg_dataset') and unique_yaws.size <= 1:
        warnings.append('Only one unique yaw/orientation is present.')
    if int(manifest.get('yaw_samples', 0) or 0) <= 1 and kind in ('projected_bbox_dataset', 'simseg_dataset'):
        warnings.append('Manifest reports yaw_samples <= 1.')
    if 'train' in split_counts and 'val' in split_counts:
        if exact_pose_overlap > 0:
            warnings.append(f'Train/val exact pose overlap detected ({exact_pose_overlap} shared pose keys).')
        spacing_ref = max(v for v in (x_spacing, y_spacing) if math.isfinite(v)) if any(math.isfinite(v) for v in (x_spacing, y_spacing)) else math.nan
        if math.isfinite(nearest_cross_split_distance) and math.isfinite(spacing_ref) and nearest_cross_split_distance <= spacing_ref + 1e-9:
            warnings.append(
                f'Nearest accepted val pose is only {nearest_cross_split_distance:.3f} m from train, comparable to grid spacing {spacing_ref:.3f} m.'
            )
    if kind == 'visibility_capture':
        duplicate_frames = int(manifest.get('duplicate_image_count', 0) or 0)
        if duplicate_frames > 0:
            warnings.append(f'Capture manifest reports {duplicate_frames} duplicate frames.')

    return {
        'dataset_dir': str(dataset_dir),
        'kind': kind,
        'sample_count': int(len(rows)),
        'accepted_count': int(np.sum(accepted > 0.5)) if accepted.size else int(len(rows)),
        'split_counts': split_counts,
        'unique_yaw_count': int(unique_yaws.size),
        'unique_yaw_values_rad': [float(v) for v in unique_yaws.tolist()],
        'manifest_yaw_samples': int(manifest.get('yaw_samples', 0) or 0) if manifest else 0,
        'manifest_split_mode': str(manifest.get('split_mode', '') or ''),
        'x_spacing_m': x_spacing,
        'y_spacing_m': y_spacing,
        'nearest_cross_split_distance_m': nearest_cross_split_distance,
        'exact_pose_overlap_count': int(exact_pose_overlap),
        'warnings': warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Audit local perception datasets for yaw coverage and split leakage.')
    parser.add_argument('--root', default=str(DEFAULT_SEARCH_ROOT), help='Search root for dataset directories.')
    parser.add_argument('--dataset-dir', action='append', default=[], help='Specific dataset directory to audit. May be repeated.')
    parser.add_argument('--out', default='', help='Optional JSON output path.')
    args = parser.parse_args()

    dataset_dirs = [Path(p).expanduser().resolve() for p in args.dataset_dir]
    if not dataset_dirs:
        dataset_dirs = _dataset_dirs(Path(args.root).expanduser().resolve())

    analyses = []
    for dataset_dir in dataset_dirs:
        kind = _kind_for_dir(dataset_dir)
        if kind == 'unknown':
            continue
        analyses.append(analyze_dataset_dir(dataset_dir))

    payload = {'dataset_count': int(len(analyses)), 'datasets': analyses}
    if str(args.out).strip():
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
