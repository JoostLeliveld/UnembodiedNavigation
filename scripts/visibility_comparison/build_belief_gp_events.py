#!/usr/bin/env python3
"""Build belief-aware detector events from visibility campaign logs.

Model inputs stay strictly operational (belief mean + covariance, detector
outcome). Alongside them the builder copies an EVALUATION-ONLY block of
``eval_`` columns, including the SIGNED localization residual
``pred_world - ground_truth`` needed to fit a per-camera bias and conditional
covariance offline. Nothing in the ``eval_`` block may ever be a GP or
deployment input.
"""

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
    # Signed localization residual, EVALUATION-ONLY (see RESIDUAL_COLUMNS).
    'eval_pred_world_x',
    'eval_pred_world_y',
    'eval_res_x',
    'eval_res_y',
    'eval_res_gt_source',
)

# The signed 2-vector residual (pred_world - ground truth) in metres. This is
# what `localization_error_captime_m` throws away: that column is a magnitude,
# so it can never identify a *direction*, i.e. a per-camera bias b_c(x).
#
# EVALUATION-ONLY, like every other `eval_` column here: they exist for audit
# and for offline bias / conditional-covariance fitting, and must never be fed
# to a GP or any other deployed model as an input feature.
RESIDUAL_COLUMNS = (
    'eval_pred_world_x',
    'eval_pred_world_y',
    'eval_res_x',
    'eval_res_y',
    'eval_res_gt_source',
)

EVALUATION_ONLY_COLUMNS = tuple(name for name in EVENT_COLUMNS if name.startswith('eval_'))


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


def _signed_residual(
    per_row: dict[str, str],
    exp_row: dict[str, str],
    *,
    detected: bool,
) -> dict[str, str]:
    """Signed localization residual ``pred_world - ground_truth`` in metres.

    EVALUATION-ONLY (see ``RESIDUAL_COLUMNS``).

    Every field is emitted EMPTY -- never ``0.0`` -- when the quantity is not
    actually measured, because a zero would read to a bias fit as "this camera
    is unbiased here" and silently drag ``b_c(x)`` toward the origin. Empty is
    produced when:

    * the camera did not detect the robot on this frame (no projected point),
    * ``pred_world_x/y`` is missing or non-finite,
    * no ground truth is available for the frame.

    Ground truth is taken from the perception row's own ``true_x/true_y`` when
    present (same row, same log time as ``pred_world_x/y``, so the residual is
    exactly the vector whose norm the logger stores as
    ``localization_error_m``). Campaign perception rows that predate the truth
    columns fall back to the joined ``experiment.csv`` ``gt_x/gt_y``, which is
    the nearest experiment sample rather than the same instant --
    ``eval_res_gt_source`` records which one was used so the two time bases are
    never silently pooled in one bias fit.
    """

    blank = {name: '' for name in RESIDUAL_COLUMNS}
    if not detected:
        return blank
    pred_x = _f(per_row, 'pred_world_x')
    pred_y = _f(per_row, 'pred_world_y')
    if not (math.isfinite(pred_x) and math.isfinite(pred_y)):
        return blank

    out = dict(blank)
    out['eval_pred_world_x'] = _fmt(pred_x)
    out['eval_pred_world_y'] = _fmt(pred_y)

    gt_x = _f(per_row, 'true_x')
    gt_y = _f(per_row, 'true_y')
    source = 'perception_true_xy'
    true_available = _f(per_row, 'true_available')
    if math.isfinite(true_available) and true_available < 0.5:
        gt_x = gt_y = math.nan
    if not (math.isfinite(gt_x) and math.isfinite(gt_y)):
        gt_x = _f(exp_row, 'gt_x')
        gt_y = _f(exp_row, 'gt_y')
        source = 'experiment_gt_xy'
        gt_available = _f(exp_row, 'gt_available')
        if math.isfinite(gt_available) and gt_available < 0.5:
            gt_x = gt_y = math.nan
    if not (math.isfinite(gt_x) and math.isfinite(gt_y)):
        return out

    out['eval_res_x'] = _fmt(pred_x - gt_x)
    out['eval_res_y'] = _fmt(pred_y - gt_y)
    out['eval_res_gt_source'] = source
    return out


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

        event_row = {
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
        event_row.update(_signed_residual(per_row, exp_row, detected=det_raw == '1'))
        rows.append(event_row)
        event_idx += 1
    return rows, counts


def _mean_finite(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        value = _f(row, key)
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


def _residual_coverage(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Audit how many events carry a usable signed residual, and from which GT."""

    finite = [
        row
        for row in rows
        if math.isfinite(_f(row, 'eval_res_x')) and math.isfinite(_f(row, 'eval_res_y'))
    ]
    detections = sum(1 for row in rows if str(row.get('det_hit', '')).strip() == '1')
    by_source: dict[str, int] = {}
    for row in finite:
        key = str(row.get('eval_res_gt_source', '')).strip() or 'unknown'
        by_source[key] = by_source.get(key, 0) + 1
    return {
        'events_with_signed_residual': int(len(finite)),
        'detection_events': int(detections),
        'gt_source_counts': dict(sorted(by_source.items())),
        'mean_eval_res_x': _mean_finite(finite, 'eval_res_x'),
        'mean_eval_res_y': _mean_finite(finite, 'eval_res_y'),
        'std_eval_res_x': float(np.std([_f(row, 'eval_res_x') for row in finite])) if finite else math.nan,
        'std_eval_res_y': float(np.std([_f(row, 'eval_res_y') for row in finite])) if finite else math.nan,
    }


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
        'residual_fields_evaluation_only': list(RESIDUAL_COLUMNS),
        'evaluation_only_columns': list(EVALUATION_ONLY_COLUMNS),
        'residual_coverage': _residual_coverage(all_rows),
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
            'Ground-truth columns are copied only for audit/evaluation and must not be used as GP inputs. '
            'This covers EVERY eval_ column, including the signed-residual block '
            '(eval_pred_world_x/y, eval_res_x/y, eval_res_gt_source): they are derived from ground truth '
            'and are evaluation-only, never a model/deployment input.',
            'eval_res_x/eval_res_y are the SIGNED residual pred_world - ground_truth in metres. '
            'localization_error_captime_m is the magnitude only and cannot identify a bias direction; '
            'the signed pair is what a per-camera bias b_c(x) and conditional covariance R_cond,c(x) are fitted from.',
            'A missing residual is written EMPTY, never 0.0: no detection, no projected world point, or no '
            'ground truth all yield blanks so an absent measurement cannot be mistaken for a zero bias.',
            'eval_res_gt_source records the ground-truth time base: perception_true_xy is the same perception '
            'row (log-time truth, so hypot(eval_res_x, eval_res_y) reproduces the logger localization_error_m); '
            'experiment_gt_xy is the nearest joined experiment.csv sample. Do not pool the two without checking.',
        ],
    }
    write_manifest(output_dir / 'manifest.json', manifest)

    coverage = manifest['residual_coverage']
    print(f'Wrote belief GP events to {events_path}')
    print(f'Runs: {len(run_dirs)}')
    print(f'Events: {len(all_rows)}')
    print(f'Skipped missing covariance: {skipped.get("skipped_missing_covariance", 0)}')
    print(
        'Signed residuals (eval-only): '
        f'{coverage["events_with_signed_residual"]}/{coverage["detection_events"]} detections, '
        f'mean=({coverage["mean_eval_res_x"]:.4f}, {coverage["mean_eval_res_y"]:.4f}) m'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
