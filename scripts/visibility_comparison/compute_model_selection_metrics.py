#!/usr/bin/env python3
"""Compute lean model-selection metrics and figure.

The script combines:
  * a fresh 36-run raw-GP anchor campaign log, and
  * the 35-run non-nominal sensitivity grid log.

It refuses mixed sensor contracts when run manifests are available: all runs
must use runtime YOLO threshold 0.10, C2/C3 nominal anchor rows must use the
reference raw-score GP, and grid rows must use the artifact recorded in their
merged_config.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
warnings.filterwarnings('ignore', message='Unable to import Axes3D.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

RHO_LOW_THRESHOLD = 0.35
EXPECTED_YOLO_CONF_THRESHOLD = 0.10
GOAL_SUCCESS_RADIUS_M = 0.20
RUN_TIMEOUT_AFTER_FIRST_CMD_S = 75.0

TASK_A = 'shadow_tradeoff_a'
TASK_B = 'shadow_tradeoff_b'
TASK_S = 'sanity_open'

FIELDNAMES = [
    'source', 'axis', 'label', 'task', 'condition', 'planner', 'seed',
    'run_dir', 'outcome', 'completion_reason',
    'clean_success', 'collision', 'timeout', 'valid_run',
    'path_length_m', 'minimum_goal_distance', 'elapsed_after_first_cmd_s',
    'reference_low_rho_exposure', 'cell_low_rho_exposure',
    'mean_rho_reference', 'mean_rho_cell',
    'mean_loc_error_m', 'mean_overconf', 'mean_cov_trace',
    'mean_r_plan_u_std', 'mean_r_plan_v_std', 'mean_p_vis_plan',
    'mean_p_vis_plan_eff',
    'mean_efe_risk', 'mean_efe_ambiguity', 'mean_efe_obstacle',
    'gp_artifact_path', 'gp_beta', 'yolo_conf_threshold', 'yolo_model',
    'r_miss_uv', 'process_noise_xy', 'ambiguity_weight',
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_for_compare(path_value: object) -> Path:
    return Path(str(path_value)).expanduser().resolve(strict=False)


def _float_close(a: object, b: object, *, tol: float = 1e-8) -> bool:
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= tol


def _pf(row: dict[str, str], key: str) -> float:
    v = row.get(key, '')
    if v in (None, '', 'nan', 'NaN'):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _format(v: object) -> str:
    if v is None:
        return ''
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        return '' if not math.isfinite(v) else f'{v:.6f}'
    return str(v)


def _load_run_csv(run_dir: Path) -> list[dict[str, str]]:
    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        return []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _find_summary_path(run_dir_str: str) -> Path | None:
    if not run_dir_str:
        return None
    run_dir = Path(run_dir_str)
    direct = run_dir / 'run_summary.json'
    if direct.is_file():
        return direct
    candidates = sorted(run_dir.rglob('run_summary.json')) if run_dir.is_dir() else []
    return candidates[-1] if candidates else None


def _load_gp(path: Path) -> tuple[RegularGridInterpolator, float]:
    with np.load(path, allow_pickle=False) as data:
        required = {'xs', 'ys', 'P_conservative_plan_map'}
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f'GP artifact missing keys {missing}: {path}')
        xs = np.asarray(data['xs'], dtype=float)
        ys = np.asarray(data['ys'], dtype=float)
        p_plan = np.asarray(data['P_conservative_plan_map'], dtype=float)
    manifest = _load_json(path.parent / 'gp_manifest.json')
    beta = float(manifest.get('beta', math.nan))
    interp = RegularGridInterpolator(
        (ys, xs), p_plan, method='linear', bounds_error=False, fill_value=np.nan
    )
    return interp, beta


def _query_rho(interp: RegularGridInterpolator | None, x: float, y: float) -> float:
    if interp is None or not (math.isfinite(x) and math.isfinite(y)):
        return math.nan
    return float(interp([[y, x]])[0])


def _run_summary_metrics(summary: dict[str, Any], outcome: str) -> dict[str, Any]:
    completion_reason = str(summary.get('completion_reason', '')) if summary else 'infra_invalid'
    collision = bool(
        summary.get('crashed', False)
        or summary.get('collision_any', False)
        or completion_reason == 'collision'
        or outcome == 'collision'
    )
    timeout = bool(completion_reason == 'timeout_after_first_cmd' or outcome == 'timeout')
    valid_run = bool(summary.get('valid_run', False)) if summary else False
    min_goal = float(summary.get('minimum_goal_distance', math.nan) or math.nan) if summary else math.nan
    elapsed = float(summary.get('elapsed_after_first_cmd_s', math.nan) or math.nan) if summary else math.nan
    clean_success = bool(
        outcome == 'goal_reached'
        and not collision
        and valid_run
        and math.isfinite(min_goal)
        and min_goal <= GOAL_SUCCESS_RADIUS_M
        and math.isfinite(elapsed)
        and 0.0 <= elapsed <= RUN_TIMEOUT_AFTER_FIRST_CMD_S
    )
    return {
        'completion_reason': completion_reason,
        'clean_success': clean_success,
        'collision': collision,
        'timeout': timeout,
        'valid_run': valid_run,
        'path_length_m': float(summary.get('path_length_m', math.nan) or math.nan) if summary else math.nan,
        'minimum_goal_distance': min_goal,
        'elapsed_after_first_cmd_s': elapsed,
        'mean_r_plan_u_std': float(summary.get('mean_r_plan_u_std', math.nan) or math.nan) if summary else math.nan,
        'mean_r_plan_v_std': float(summary.get('mean_r_plan_v_std', math.nan) or math.nan) if summary else math.nan,
        'mean_p_vis_plan': float(summary.get('mean_p_vis_plan', math.nan) or math.nan) if summary else math.nan,
        'mean_p_vis_plan_eff': float(summary.get('mean_p_vis_plan_eff', math.nan) or math.nan) if summary else math.nan,
        'mean_efe_risk': float(summary.get('mean_efe_risk', math.nan) or math.nan) if summary else math.nan,
        'mean_efe_ambiguity': float(summary.get('mean_efe_ambiguity', math.nan) or math.nan) if summary else math.nan,
        'mean_efe_obstacle': float(summary.get('mean_efe_obstacle', math.nan) or math.nan) if summary else math.nan,
    }


def _csv_path_metrics(
    rows: list[dict[str, str]],
    reference_gp,
    cell_gp,
) -> dict[str, float]:
    reference_rhos = []
    cell_rhos = []
    loc_errors = []
    overconf_values = []
    cov_traces = []

    for row in rows:
        truth_ok = _pf(row, 'truth_available') >= 0.5
        truth_x = _pf(row, 'truth_x')
        truth_y = _pf(row, 'truth_y')
        if truth_ok and math.isfinite(truth_x) and math.isfinite(truth_y):
            rho_ref = _query_rho(reference_gp, truth_x, truth_y)
            rho_cell = _query_rho(cell_gp, truth_x, truth_y)
            if math.isfinite(rho_ref):
                reference_rhos.append(rho_ref)
            if math.isfinite(rho_cell):
                cell_rhos.append(rho_cell)

        belief_ok = _pf(row, 'planner_belief_available') >= 0.5
        belief_x = _pf(row, 'planner_belief_x')
        belief_y = _pf(row, 'planner_belief_y')
        cov_x = _pf(row, 'planner_cov_x')
        cov_y = _pf(row, 'planner_cov_y')
        if (
            truth_ok and belief_ok
            and math.isfinite(truth_x) and math.isfinite(truth_y)
            and math.isfinite(belief_x) and math.isfinite(belief_y)
        ):
            loc_errors.append(math.hypot(truth_x - belief_x, truth_y - belief_y))
            if math.isfinite(cov_x) and math.isfinite(cov_y) and cov_x + cov_y > 0:
                trace = cov_x + cov_y
                cov_traces.append(trace)
                overconf_values.append(loc_errors[-1] / math.sqrt(trace))

    ref_arr = np.asarray(reference_rhos, dtype=float)
    cell_arr = np.asarray(cell_rhos, dtype=float)
    return {
        'reference_low_rho_exposure': float(np.mean(ref_arr < RHO_LOW_THRESHOLD)) if ref_arr.size else math.nan,
        'cell_low_rho_exposure': float(np.mean(cell_arr < RHO_LOW_THRESHOLD)) if cell_arr.size else math.nan,
        'mean_rho_reference': float(np.mean(ref_arr)) if ref_arr.size else math.nan,
        'mean_rho_cell': float(np.mean(cell_arr)) if cell_arr.size else math.nan,
        'mean_loc_error_m': float(np.mean(loc_errors)) if loc_errors else math.nan,
        'mean_overconf': float(np.mean(overconf_values)) if overconf_values else math.nan,
        'mean_cov_trace': float(np.mean(cov_traces)) if cov_traces else math.nan,
    }


def _expected_grid_artifact(entry: dict[str, Any]) -> str:
    merged = entry.get('merged_config') or {}
    if isinstance(merged, dict) and merged.get('gp_artifact'):
        return str(_resolve_for_compare(merged['gp_artifact']))
    overrides = entry.get('overrides') or {}
    if isinstance(overrides, dict) and overrides.get('gp_artifact'):
        return str(_resolve_for_compare(overrides['gp_artifact']))
    return ''


def _validate_manifest_contract(
    *,
    source: str,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    reference_gp_path: Path,
    yolo_models: set[str],
) -> None:
    if not manifest:
        return

    threshold = manifest.get('yolo_conf_threshold')
    if not _float_close(threshold, EXPECTED_YOLO_CONF_THRESHOLD):
        raise RuntimeError(
            f'{source} run has yolo_conf_threshold={threshold}; expected {EXPECTED_YOLO_CONF_THRESHOLD}. '
            f'Run dir: {entry.get("run_dir")}'
        )

    yolo_model = str(manifest.get('yolo_model', '') or '')
    if yolo_model:
        yolo_models.add(str(_resolve_for_compare(yolo_model)))

    planner = str(manifest.get('planner', entry.get('planner', '')))
    actual_artifact = str(manifest.get('visibility_artifact_path', '') or '')
    if source == 'anchor':
        condition = str(entry.get('condition', ''))
        if condition in ('C2', 'C3'):
            if _resolve_for_compare(actual_artifact) != reference_gp_path:
                raise RuntimeError(
                    f'Anchor {condition} uses mixed GP artifact: {actual_artifact or "<missing>"}; '
                    f'expected {reference_gp_path}. Run dir: {entry.get("run_dir")}'
                )
    elif source == 'grid':
        expected = _expected_grid_artifact(entry)
        if planner != 'constant_R_efe' and expected:
            if _resolve_for_compare(actual_artifact) != _resolve_for_compare(expected):
                raise RuntimeError(
                    f'Grid row uses mixed GP artifact: {actual_artifact or "<missing>"}; '
                    f'expected {expected}. Run dir: {entry.get("run_dir")}'
                )


def _row_from_entry(
    *,
    source: str,
    entry: dict[str, Any],
    key: str,
    reference_gp_path: Path,
    gp_cache: dict[str, tuple[RegularGridInterpolator, float]],
    yolo_models: set[str],
) -> dict[str, Any]:
    run_dir_str = str(entry.get('run_dir', '') or '')
    summary_path = _find_summary_path(run_dir_str)
    run_dir = summary_path.parent if summary_path else Path(run_dir_str) if run_dir_str else None
    summary = _load_json(summary_path) if summary_path else {}
    manifest = _load_json(run_dir / 'run_manifest.json') if run_dir else {}

    _validate_manifest_contract(
        source=source,
        entry=entry,
        manifest=manifest,
        reference_gp_path=reference_gp_path,
        yolo_models=yolo_models,
    )

    outcome = str(entry.get('outcome', '') or 'infra_invalid')
    summary_metrics = _run_summary_metrics(summary, outcome)

    gp_artifact = str(manifest.get('visibility_artifact_path', '') or '')
    if not gp_artifact:
        gp_artifact = _expected_grid_artifact(entry) if source == 'grid' else str(reference_gp_path)
    gp_path = _resolve_for_compare(gp_artifact)
    if not gp_path.is_file():
        gp_path = reference_gp_path

    ref_gp, ref_beta = gp_cache[str(reference_gp_path)]
    if str(gp_path) not in gp_cache:
        gp_cache[str(gp_path)] = _load_gp(gp_path)
    cell_gp, cell_beta = gp_cache[str(gp_path)]

    csv_metrics = _csv_path_metrics(
        _load_run_csv(run_dir) if run_dir else [],
        ref_gp,
        cell_gp,
    )

    merged = entry.get('merged_config') if isinstance(entry.get('merged_config'), dict) else {}
    return {
        'source': source,
        'axis': str(entry.get('axis', '') or ''),
        'label': str(entry.get('label', entry.get('condition', '')) or ''),
        'task': str(entry.get('task', merged.get('task', '')) or ''),
        'condition': str(entry.get('condition', '') or ''),
        'planner': str(entry.get('planner', manifest.get('planner', merged.get('planner', ''))) or ''),
        'seed': entry.get('seed', ''),
        'run_dir': str(run_dir) if run_dir else run_dir_str,
        'outcome': outcome,
        **summary_metrics,
        **csv_metrics,
        'gp_artifact_path': str(gp_path),
        'gp_beta': cell_beta if math.isfinite(cell_beta) else ref_beta,
        'yolo_conf_threshold': manifest.get('yolo_conf_threshold', math.nan),
        'yolo_model': str(manifest.get('yolo_model', '') or ''),
        'r_miss_uv': manifest.get('r_miss_uv', merged.get('r_miss_uv', math.nan)),
        'process_noise_xy': manifest.get('process_noise_xy', merged.get('process_noise_xy', math.nan)),
        'ambiguity_weight': manifest.get('ambiguity_weight', merged.get('ambiguity_weight', math.nan)),
    }


def _load_all_rows(anchor_log: Path, grid_log: Path, reference_gp: Path) -> tuple[list[dict[str, Any]], set[str]]:
    anchor = _load_json(anchor_log)
    grid = _load_json(grid_log)
    if not anchor:
        raise RuntimeError(f'Anchor log is empty or missing: {anchor_log}')
    if not grid:
        raise RuntimeError(f'Grid log is empty or missing: {grid_log}')

    gp_cache: dict[str, tuple[RegularGridInterpolator, float]] = {str(reference_gp): _load_gp(reference_gp)}
    yolo_models: set[str] = set()
    rows = []

    for key, entry in sorted(anchor.items()):
        row = _row_from_entry(
            source='anchor',
            entry=dict(entry),
            key=key,
            reference_gp_path=reference_gp,
            gp_cache=gp_cache,
            yolo_models=yolo_models,
        )
        rows.append(row)

    for key, entry in sorted(grid.items()):
        row = _row_from_entry(
            source='grid',
            entry=dict(entry),
            key=key,
            reference_gp_path=reference_gp,
            gp_cache=gp_cache,
            yolo_models=yolo_models,
        )
        rows.append(row)

    if len(yolo_models) > 1:
        raise RuntimeError('Mixed YOLO models detected:\n  ' + '\n  '.join(sorted(yolo_models)))
    return rows, yolo_models


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        try:
            v = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            v = math.nan
        if math.isfinite(v):
            vals.append(v)
    return vals


def _median(rows: list[dict[str, Any]], key: str) -> float:
    vals = _finite_values(rows, key)
    return float(np.median(vals)) if vals else math.nan


def _clean_rate(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    n = len(rows)
    clean = sum(1 for r in rows if bool(r.get('clean_success')))
    return clean, n, float(clean / n) if n else math.nan


def _select(rows: list[dict[str, Any]], **criteria: str) -> list[dict[str, Any]]:
    selected = rows
    for key, value in criteria.items():
        selected = [r for r in selected if str(r.get(key, '')) == str(value)]
    return selected


def _sensitivity_groups(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get('source') == 'grid':
            groups[(str(row.get('axis', '')), str(row.get('label', '')))].append(row)
    return dict(groups)


def _acceptance_summary(rows: list[dict[str, Any]]) -> tuple[list[str], bool, str]:
    lines = []
    failures = []

    c2_a = _select(rows, source='anchor', task=TASK_A, condition='C2')
    c2_b = _select(rows, source='anchor', task=TASK_B, condition='C2')
    c2_s = _select(rows, source='anchor', task=TASK_S, condition='C2')
    c1_a = _select(rows, source='anchor', task=TASK_A, condition='C1')
    c3_a = _select(rows, source='anchor', task=TASK_A, condition='C3')

    checks = [
        ('C2 Task A clean success >= 4/5', c2_a, 4),
        ('C2 Task B clean success >= 4/5', c2_b, 4),
        ('C2 sanity clean success >= 3/3', c2_s, 3),
    ]
    for label, subset, threshold in checks:
        clean, n, _rate = _clean_rate(subset)
        ok = clean >= threshold and n >= threshold
        lines.append(f'- {"PASS" if ok else "FAIL"}: {label} ({clean}/{n})')
        if not ok:
            failures.append(label)

    c1_low = _median(c1_a, 'reference_low_rho_exposure')
    c2_low = _median(c2_a, 'reference_low_rho_exposure')
    low_delta = c1_low - c2_low
    low_rel = low_delta / c1_low if math.isfinite(c1_low) and c1_low > 0 else math.nan
    low_ok = math.isfinite(low_delta) and (low_delta >= 0.10 or low_rel >= 0.25)
    lines.append(
        f'- {"PASS" if low_ok else "FAIL"}: C2 lowers Task-A low-rho exposure vs C1 '
        f'(C1={c1_low:.3f}, C2={c2_low:.3f}, delta={low_delta:.3f}, rel={low_rel:.1%})'
        if math.isfinite(low_rel)
        else f'- {"PASS" if low_ok else "FAIL"}: C2 lowers Task-A low-rho exposure vs C1 '
             f'(C1={c1_low:.3f}, C2={c2_low:.3f})'
    )
    if not low_ok:
        failures.append('C2 low-rho exposure vs C1')

    c2_path = _median([r for r in c2_a if r.get('clean_success')], 'path_length_m')
    c3_path = _median([r for r in c3_a if r.get('clean_success')], 'path_length_m')
    path_ratio = c2_path / c3_path if math.isfinite(c2_path) and math.isfinite(c3_path) and c3_path > 0 else math.nan
    path_ok = math.isfinite(path_ratio) and path_ratio <= 1.25
    lines.append(
        f'- {"PASS" if path_ok else "FAIL"}: C2 Task-A path length <= 1.25x C3 '
        f'(C2={c2_path:.3f}, C3={c3_path:.3f}, ratio={path_ratio:.3f})'
    )
    if not path_ok:
        failures.append('C2 path length vs C3')

    bad_axes = []
    for axis in sorted({r['axis'] for r in rows if r.get('source') == 'grid'}):
        axis_bad = False
        for (_axis, label), group in _sensitivity_groups(rows).items():
            if _axis != axis:
                continue
            clean, n, _rate = _clean_rate(group)
            if n < 5 or clean < 4:
                axis_bad = True
                lines.append(f'- WARN: sensitivity cell {axis}/{label} is below 4/5 ({clean}/{n})')
        if axis_bad:
            bad_axes.append(axis)
    sens_ok = len(bad_axes) <= 1
    lines.append(
        f'- {"PASS" if sens_ok else "FAIL"}: at most one sensitivity axis below 4/5 '
        f'({len(bad_axes)} bad axes: {", ".join(bad_axes) if bad_axes else "none"})'
    )
    if not sens_ok:
        failures.append('sensitivity robustness')

    selected = len(failures) == 0
    if selected:
        decision = 'Nominal C2 passes the locked acceptance criteria and remains the paper model.'
    else:
        candidates = []
        for (axis, label), group in _sensitivity_groups(rows).items():
            clean, n, rate = _clean_rate(group)
            candidates.append((
                -rate,
                _median(group, 'reference_low_rho_exposure'),
                _median([r for r in group if r.get('clean_success')], 'path_length_m'),
                axis,
                label,
                clean,
                n,
            ))
        candidates = [c for c in candidates if math.isfinite(c[0]) and math.isfinite(c[1])]
        if candidates:
            best = sorted(candidates)[0]
            decision = (
                'Nominal C2 fails at least one criterion. Best non-nominal sensitivity cell by the '
                f'locked order is {best[3]}/{best[4]} ({best[5]}/{best[6]} clean successes). '
                'Rerun the 36-run anchor with that setting before using it in the paper.'
            )
        else:
            decision = 'Nominal C2 fails, but no complete sensitivity candidate is available to select.'

    c2_clean, c2_n, c2_rate = _clean_rate(c2_a)
    c3_clean, c3_n, c3_rate = _clean_rate(c3_a)
    if abs(c2_rate - c3_rate) < 0.2 and abs(_median(c2_a, 'reference_low_rho_exposure') - _median(c3_a, 'reference_low_rho_exposure')) < 0.05:
        ambiguity_note = (
            'Ambiguity should be reported as secondary: C2 and C3 are similar on Task A, '
            'so route selection is mostly supported by state-dependent observation risk.'
        )
    else:
        ambiguity_note = (
            f'Ambiguity may be supportive: Task-A C2={c2_clean}/{c2_n}, C3={c3_clean}/{c3_n}; '
            'interpret only after inspecting low-rho/path differences.'
        )

    lines.append('')
    lines.append(f'**Decision:** {decision}')
    lines.append(f'**Ambiguity interpretation:** {ambiguity_note}')
    return lines, selected, decision


def _group_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        '| Source | Axis/Condition | Label/Task | N | Clean | Median low-rho | Median path (m) |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    anchor_keys = [
        ('anchor', 'C1', TASK_A),
        ('anchor', 'C2', TASK_A),
        ('anchor', 'C3', TASK_A),
        ('anchor', 'C2', TASK_B),
        ('anchor', 'C2', TASK_S),
    ]
    for source, cond, task in anchor_keys:
        subset = _select(rows, source=source, condition=cond, task=task)
        clean, n, _rate = _clean_rate(subset)
        lines.append(
            f'| anchor | {cond} | {task} | {n} | {clean} | '
            f'{_median(subset, "reference_low_rho_exposure"):.3f} | {_median(subset, "path_length_m"):.3f} |'
        )
    for (axis, label), subset in sorted(_sensitivity_groups(rows).items()):
        clean, n, _rate = _clean_rate(subset)
        lines.append(
            f'| grid | {axis} | {label} | {n} | {clean} | '
            f'{_median(subset, "reference_low_rho_exposure"):.3f} | {_median(subset, "path_length_m"):.3f} |'
        )
    return lines


def _stats(rows: list[dict[str, Any]]) -> tuple[float, float]:
    clean, _n, rate = _clean_rate(rows)
    low = _median(rows, 'reference_low_rho_exposure')
    return rate, low


def _plot_model_selection(rows: list[dict[str, Any]], out_path: Path) -> None:
    panels = [
        ('r_miss_uv', 'r_miss_uv', [
            ('60', _select(rows, source='grid', axis='r_miss_uv', label='r_miss_uv_60')),
            ('120', _select(rows, source='anchor', task=TASK_A, condition='C2')),
            ('200', _select(rows, source='grid', axis='r_miss_uv', label='r_miss_uv_200')),
        ]),
        ('beta', 'GP beta', [
            ('0.0', _select(rows, source='grid', axis='gp_beta', label='gp_beta_0p0')),
            ('0.75', _select(rows, source='anchor', task=TASK_A, condition='C2')),
            ('1.25', _select(rows, source='grid', axis='gp_beta', label='gp_beta_1p25')),
        ]),
        ('Q_xy', 'process_noise_xy', [
            ('0.0025', _select(rows, source='grid', axis='process_noise_xy', label='Q_xy_low')),
            ('0.01', _select(rows, source='anchor', task=TASK_A, condition='C2')),
            ('0.04', _select(rows, source='grid', axis='process_noise_xy', label='Q_xy_high')),
        ]),
        ('ambiguity', 'ambiguity_weight', [
            ('0', _select(rows, source='anchor', task=TASK_A, condition='C3')),
            ('3', _select(rows, source='anchor', task=TASK_A, condition='C2')),
            ('5', _select(rows, source='grid', axis='ambiguity_weight', label='ambiguity_high')),
        ]),
    ]

    c1_rate, c1_low = _stats(_select(rows, source='anchor', task=TASK_A, condition='C1'))
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), sharey=True)
    for ax, (_key, title, cells) in zip(axes, panels):
        x = np.arange(len(cells))
        rates = []
        lows = []
        for _label, subset in cells:
            rate, low = _stats(subset)
            rates.append(rate)
            lows.append(low)
        ax2 = ax.twinx()
        ax.plot(x, rates, marker='o', color='#2ca02c', linewidth=1.6, label='clean success')
        ax2.plot(x, lows, marker='s', color='#ff7f0e', linewidth=1.4, label='median low-rho')
        if math.isfinite(c1_rate):
            ax.axhline(c1_rate, color='#2ca02c', linestyle=':', linewidth=1.0, alpha=0.7)
        if math.isfinite(c1_low):
            ax2.axhline(c1_low, color='#ff7f0e', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _subset in cells], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.25, linestyle=':')
        ax2.set_ylim(-0.05, 1.05)
        if ax is axes[0]:
            ax.set_ylabel('clean success rate')
        if ax is axes[-1]:
            ax2.set_ylabel('median reference low-rho exposure')
        else:
            ax2.set_yticklabels([])
    handles = [
        plt.Line2D([0], [0], color='#2ca02c', marker='o', label='clean success'),
        plt.Line2D([0], [0], color='#ff7f0e', marker='s', label='median low-rho'),
        plt.Line2D([0], [0], color='black', linestyle=':', label='C1 Task-A reference'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.28, top=0.86, wspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)


def _write_outputs(rows: list[dict[str, Any]], out: Path, summary_out: Path, figure_out: Path, yolo_models: set[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in FIELDNAMES})

    acceptance_lines, _selected, _decision = _acceptance_summary(rows)
    summary_lines = [
        '# Model Selection Summary',
        '',
        '## Data Integrity',
        f'- Rows written: {len(rows)}',
        f'- Unique YOLO models: {", ".join(sorted(yolo_models)) if yolo_models else "none found in manifests"}',
        f'- Required runtime threshold: {EXPECTED_YOLO_CONF_THRESHOLD}',
        '',
        '## Acceptance Criteria',
        *acceptance_lines,
        '',
        '## Cell Summary',
        *_group_table(rows),
        '',
    ]
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text('\n'.join(summary_lines), encoding='utf-8')
    _plot_model_selection(rows, figure_out)


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute lean model-selection metrics.')
    parser.add_argument('--anchor-log', required=True)
    parser.add_argument('--grid-log', required=True)
    parser.add_argument('--reference-gp', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--summary-out', required=True)
    parser.add_argument('--figure-out', default='',
                        help='Output PDF for the four-panel model-selection figure.')
    args = parser.parse_args()

    anchor_log = Path(args.anchor_log).expanduser().resolve()
    grid_log = Path(args.grid_log).expanduser().resolve()
    reference_gp = Path(args.reference_gp).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    summary_out = Path(args.summary_out).expanduser().resolve()
    figure_out = (
        Path(args.figure_out).expanduser().resolve()
        if args.figure_out
        else out.with_name('model_selection_figure.pdf')
    )

    if not reference_gp.is_file():
        print(f'ERROR: reference GP not found: {reference_gp}', file=sys.stderr)
        return 1

    rows, yolo_models = _load_all_rows(anchor_log, grid_log, reference_gp)
    _write_outputs(rows, out, summary_out, figure_out, yolo_models)
    print(f'Wrote metrics: {out}')
    print(f'Wrote summary: {summary_out}')
    print(f'Wrote figure: {figure_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
