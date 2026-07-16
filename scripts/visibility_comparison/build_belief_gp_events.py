#!/usr/bin/env python3
"""Build belief-aware detector events from visibility campaign logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import LOGS_ROOT, REPO_ROOT, repo_relative, write_csv, write_manifest


DEFAULT_CAMPAIGN = LOGS_ROOT / 'honest_campaign_v1'
DEFAULT_OUT = LOGS_ROOT / 'belief_gp_events'

EVENT_COLUMNS = (
    'event_id',
    'run_dir',
    'route',
    'condition',
    'seed',
    'run_id',
    'diag_stamp',
    'log_stamp',
    'matched_experiment_stamp',
    'stamp_delta_s',
    'm_x',
    'm_y',
    'S_xx',
    'S_xy',
    'S_yy',
    'sigma_major_m',
    'sigma_minor_m',
    'trace_S_xy',
    'det_hit',
    'yolo_score_raw',
    'yolo_detected_after_threshold',
    'pixel_pose_available',
    'pixel_pose_fresh',
    'localization_error_captime_m',
    'state_source',
    'eval_gt_x',
    'eval_gt_y',
    'eval_belief_error_gt_m',
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, '')
    if raw in (None, '', 'nan', 'NaN'):
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _flag(row: dict[str, str], key: str) -> str:
    value = _f(row, key)
    if not math.isfinite(value):
        return ''
    return str(int(value >= 0.5))


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return ''
    return f'{float(value):.10g}'


def _nearest_index(stamps: np.ndarray, stamp: float) -> tuple[int, float]:
    if stamps.size == 0 or not math.isfinite(stamp):
        return -1, math.inf
    pos = int(np.searchsorted(stamps, stamp))
    candidates = []
    if 0 <= pos < stamps.size:
        candidates.append(pos)
    if 0 <= pos - 1 < stamps.size:
        candidates.append(pos - 1)
    best = min(candidates, key=lambda idx: abs(float(stamps[idx]) - stamp))
    return int(best), abs(float(stamps[best]) - stamp)


def _covariance_stats(cov_xx: float, cov_xy: float, cov_yy: float) -> tuple[float, float, float]:
    if not (math.isfinite(cov_xx) and math.isfinite(cov_yy)):
        return math.nan, math.nan, math.nan
    cov_xy = float(cov_xy) if math.isfinite(cov_xy) else 0.0
    S = np.asarray([[float(cov_xx), cov_xy], [cov_xy, float(cov_yy)]], dtype=float)
    S = 0.5 * (S + S.T)
    try:
        vals = np.linalg.eigvalsh(S)
    except np.linalg.LinAlgError:
        return math.nan, math.nan, math.nan
    vals = np.clip(vals, 0.0, None)
    sigma_minor = math.sqrt(float(vals[0]))
    sigma_major = math.sqrt(float(vals[-1]))
    return sigma_major, sigma_minor, float(vals[0] + vals[-1])


def _run_parts(campaign_dir: Path, run_dir: Path) -> tuple[str, str, str, str]:
    try:
        parts = run_dir.relative_to(campaign_dir).parts
    except ValueError:
        parts = run_dir.parts[-4:]
    route = parts[0] if len(parts) > 0 else ''
    condition = parts[1] if len(parts) > 1 else ''
    seed = parts[2] if len(parts) > 2 else ''
    run_id = parts[3] if len(parts) > 3 else run_dir.name
    return route, condition, seed, run_id


def _run_dirs(campaign_dir: Path) -> list[Path]:
    return sorted(path.parent for path in campaign_dir.glob('*/*/*/experiment_*/perception.csv'))


def _extract_run_events(
    campaign_dir: Path,
    run_dir: Path,
    *,
    event_start: int,
    stamp_key: str,
    stamp_tolerance_s: float,
    require_belief_available: bool,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    perception_path = run_dir / 'perception.csv'
    experiment_path = run_dir / 'experiment.csv'
    counts = {
        'perception_rows': 0,
        'skipped_no_binary_detection': 0,
        'skipped_bad_stamp': 0,
        'skipped_stamp_tolerance': 0,
        'skipped_missing_belief': 0,
        'skipped_missing_covariance': 0,
    }
    if not perception_path.is_file() or not experiment_path.is_file():
        return [], counts

    exp_rows = _read_csv_rows(experiment_path)
    per_rows = _read_csv_rows(perception_path)
    stamps = np.asarray([_f(row, 'stamp') for row in exp_rows], dtype=float)
    finite_order = np.argsort(stamps[np.isfinite(stamps)])
    finite_indices = np.flatnonzero(np.isfinite(stamps))[finite_order]
    stamps_sorted = stamps[finite_indices]
    route, condition, seed, run_id = _run_parts(campaign_dir, run_dir)
    rel_run_dir = repo_relative(run_dir, REPO_ROOT)

    rows: list[dict[str, str]] = []
    event_idx = int(event_start)
    for per_row in per_rows:
        counts['perception_rows'] += 1
        det_raw = str(per_row.get('detected', '')).strip()
        if det_raw not in ('0', '1'):
            counts['skipped_no_binary_detection'] += 1
            continue
        match_stamp = _f(per_row, stamp_key)
        if not math.isfinite(match_stamp):
            counts['skipped_bad_stamp'] += 1
            continue
        sorted_idx, delta_s = _nearest_index(stamps_sorted, match_stamp)
        if sorted_idx < 0:
            counts['skipped_bad_stamp'] += 1
            continue
        if delta_s > stamp_tolerance_s:
            counts['skipped_stamp_tolerance'] += 1
            continue
        exp_row = exp_rows[int(finite_indices[sorted_idx])]

        belief_available = _f(exp_row, 'planner_belief_available')
        m_x = _f(exp_row, 'planner_belief_x')
        m_y = _f(exp_row, 'planner_belief_y')
        if require_belief_available and not (math.isfinite(belief_available) and belief_available >= 0.5):
            counts['skipped_missing_belief'] += 1
            continue
        if not (math.isfinite(m_x) and math.isfinite(m_y)):
            counts['skipped_missing_belief'] += 1
            continue

        S_xx = _f(exp_row, 'planner_cov_x')
        S_xy = _f(exp_row, 'planner_cov_xy', 0.0)
        S_yy = _f(exp_row, 'planner_cov_y')
        sigma_major, sigma_minor, trace_s = _covariance_stats(S_xx, S_xy, S_yy)
        if not (math.isfinite(trace_s) and trace_s > 0.0):
            counts['skipped_missing_covariance'] += 1
            continue

        rows.append(
            {
                'event_id': f'belief_event_{event_idx:08d}',
                'run_dir': rel_run_dir,
                'route': route,
                'condition': condition,
                'seed': seed,
                'run_id': run_id,
                'diag_stamp': _fmt(_f(per_row, 'diag_stamp')),
                'log_stamp': _fmt(_f(per_row, 'log_stamp')),
                'matched_experiment_stamp': _fmt(_f(exp_row, 'stamp')),
                'stamp_delta_s': _fmt(delta_s),
                'm_x': _fmt(m_x),
                'm_y': _fmt(m_y),
                'S_xx': _fmt(S_xx),
                'S_xy': _fmt(S_xy),
                'S_yy': _fmt(S_yy),
                'sigma_major_m': _fmt(sigma_major),
                'sigma_minor_m': _fmt(sigma_minor),
                'trace_S_xy': _fmt(trace_s),
                'det_hit': det_raw,
                'yolo_score_raw': _fmt(_f(per_row, 'yolo_score_raw')),
                'yolo_detected_after_threshold': _flag(per_row, 'yolo_detected_after_threshold'),
                'pixel_pose_available': _flag(per_row, 'pixel_pose_available'),
                'pixel_pose_fresh': _flag(per_row, 'pixel_pose_fresh'),
                'localization_error_captime_m': _fmt(_f(per_row, 'localization_error_captime_m')),
                'state_source': 'BELIEF',
                'eval_gt_x': _fmt(_f(exp_row, 'gt_x')),
                'eval_gt_y': _fmt(_f(exp_row, 'gt_y')),
                'eval_belief_error_gt_m': _fmt(_f(exp_row, 'belief_error_gt_m')),
            }
        )
        event_idx += 1
    return rows, counts


def _mean_finite(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        value = _f(row, key)
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description='Extract belief/covariance detector events for belief-aware GP fitting.')
    parser.add_argument('--campaign', default=str(DEFAULT_CAMPAIGN))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--stamp-key', default='log_stamp', choices=('log_stamp', 'diag_stamp'))
    parser.add_argument('--stamp-tolerance-s', type=float, default=0.3)
    parser.add_argument('--allow-missing-belief-available', action='store_true')
    args = parser.parse_args()

    campaign_dir = Path(args.campaign).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    logs_root = LOGS_ROOT.resolve()
    if logs_root not in output_dir.parents and output_dir != logs_root:
        raise RuntimeError(f'Output directory must stay under {logs_root}: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _run_dirs(campaign_dir)
    if not run_dirs:
        raise RuntimeError(f'No campaign run perception.csv files found under {campaign_dir}')

    all_rows: list[dict[str, str]] = []
    skipped: dict[str, int] = {}
    run_event_counts: dict[str, int] = {}
    for run_dir in run_dirs:
        rows, counts = _extract_run_events(
            campaign_dir,
            run_dir,
            event_start=len(all_rows),
            stamp_key=str(args.stamp_key),
            stamp_tolerance_s=float(args.stamp_tolerance_s),
            require_belief_available=not bool(args.allow_missing_belief_available),
        )
        all_rows.extend(rows)
        run_event_counts[repo_relative(run_dir, REPO_ROOT)] = len(rows)
        for key, value in counts.items():
            skipped[key] = skipped.get(key, 0) + int(value)

    events_path = output_dir / 'events.csv'
    write_csv(events_path, EVENT_COLUMNS, all_rows)
    manifest = {
        'campaign_dir': str(campaign_dir),
        'events_csv': str(events_path),
        'run_count': int(len(run_dirs)),
        'event_count': int(len(all_rows)),
        'stamp_key': str(args.stamp_key),
        'stamp_tolerance_s': float(args.stamp_tolerance_s),
        'state_source': 'BELIEF',
        'training_input': {
            'mean': ['m_x', 'm_y'],
            'covariance': ['S_xx', 'S_xy', 'S_yy'],
            'target_default': 'det_hit',
            'score_target': 'yolo_score_raw',
        },
        'gt_fields_evaluation_only': ['eval_gt_x', 'eval_gt_y', 'eval_belief_error_gt_m'],
        'skip_counts': skipped,
        'run_event_counts': run_event_counts,
        'summary': {
            'routes': sorted({row['route'] for row in all_rows}),
            'conditions': sorted({row['condition'] for row in all_rows}),
            'detection_rate': _mean_finite(all_rows, 'det_hit'),
            'mean_yolo_score_raw': _mean_finite(all_rows, 'yolo_score_raw'),
            'mean_sigma_major_m': _mean_finite(all_rows, 'sigma_major_m'),
            'mean_trace_S_xy': _mean_finite(all_rows, 'trace_S_xy'),
            'mean_eval_belief_error_gt_m': _mean_finite(all_rows, 'eval_belief_error_gt_m'),
        },
        'notes': [
            'Each event is a detector observation paired to the nearest experiment.csv row by the selected perception stamp.',
            'Training coordinates are planner_belief_x/y with planner covariance; state_x/state_y are intentionally not used.',
            'Ground-truth columns are copied only for audit/evaluation and must not be used as GP inputs.',
        ],
    }
    write_manifest(output_dir / 'manifest.json', manifest)

    print(f'Wrote belief GP events to {events_path}')
    print(f'Runs: {len(run_dirs)}')
    print(f'Events: {len(all_rows)}')
    print(f'Skipped missing covariance: {skipped.get("skipped_missing_covariance", 0)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
