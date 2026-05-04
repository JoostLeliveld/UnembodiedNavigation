#!/usr/bin/env python3
"""Compute protocol-defined paper metrics from completed run directories.

Metrics computed per run (all from experiment.csv + run_summary.json):

Primary (main table):
  goal_reached        : bool — from run_summary.json
  collision           : bool — from run_summary.json
  path_length_m  (L)  : cumulative ground-truth path length (in summary)
  mean_loc_error (ē)  : mean ||truth - state_estimate|| using planner belief
  mean_overconf  (c̄)  : mean error / sqrt(tr(Σ_xy))  [planner belief covariance]
  f_shadow            : fraction of path steps with rho_plan < 0.35
  completion_reason   : goal_reached | timeout_after_first_cmd | collision

Secondary:
  min_goal_distance   : d_min — closest approach to goal
  path_efficiency (η) : straight-line / path_length
  mean_rho_plan       : mean rho_plan along ground-truth path
  elapsed_s           : seconds from first cmd to termination
  mean_cov_trace      : mean tr(Σ_xy) over run

Usage:
    python compute_paper_metrics.py \\
        --campaign-log logs/visibility_comparison/iwai_campaign/campaign_log.json \\
        --gp-artifact logs/visibility_comparison/current_gp/yolo_score_calibrated_gp.npz \\
        --out paper_metrics.csv

Outputs:
    paper_metrics.csv   : one row per run, all metrics
    paper_summary.txt   : mean ± std per (task, condition), copy-pasteable for paper
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

RHO_SHADOW_THRESHOLD = 0.35


def _load_gp(artifact_path: Path):
    with np.load(artifact_path, allow_pickle=False) as data:
        xs = np.asarray(data['xs'], dtype=float)
        ys = np.asarray(data['ys'], dtype=float)
        p_plan = np.asarray(data.get('P_conservative_plan_map', data.get('P_conservative_map')), dtype=float)
    interp = RegularGridInterpolator(
        (ys, xs), p_plan, method='linear', bounds_error=False, fill_value=np.nan
    )
    return interp


def _query_rho(interp, x: float, y: float) -> float:
    if not (math.isfinite(x) and math.isfinite(y)):
        return math.nan
    return float(interp([[y, x]])[0])


def _load_experiment_csv(run_dir: Path) -> list[dict]:
    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        return []
    rows = []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _pf(row: dict, key: str) -> float:
    v = row.get(key, '')
    if v in (None, '', 'nan', 'NaN'):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _compute_run_metrics(run_dir: Path, summary: dict, gp_interp, task_info: dict | None) -> dict:
    rows = _load_experiment_csv(run_dir)

    # --- From summary ---
    goal_reached = bool(summary.get('goal_reached', False)) or summary.get('completion_reason') == 'goal_reached'
    crashed = bool(summary.get('crashed', False))
    completion_reason = str(summary.get('completion_reason', ''))
    path_length_m = float(summary.get('path_length_m', math.nan) or math.nan)
    min_goal_distance = float(summary.get('minimum_goal_distance', math.nan) or math.nan)
    elapsed_s = float(summary.get('elapsed_after_first_cmd_s', math.nan) or math.nan)

    if not rows:
        return {
            'goal_reached': goal_reached, 'collision': crashed,
            'completion_reason': completion_reason,
            'path_length_m': path_length_m, 'min_goal_distance': min_goal_distance,
            'elapsed_s': elapsed_s,
            'mean_loc_error_m': math.nan, 'mean_overconf': math.nan,
            'f_shadow': math.nan, 'mean_rho_plan': math.nan,
            'path_efficiency': math.nan, 'mean_cov_trace': math.nan,
            'n_rows': 0,
        }

    # --- From CSV ---
    loc_errors = []
    overconf_values = []
    rho_values = []
    cov_traces = []

    for row in rows:
        # Use planner belief as the estimator (that's what drives control)
        truth_x = _pf(row, 'truth_x')
        truth_y = _pf(row, 'truth_y')
        truth_ok = _pf(row, 'truth_available') >= 0.5 if math.isfinite(_pf(row, 'truth_available')) else False

        belief_x = _pf(row, 'planner_belief_x')
        belief_y = _pf(row, 'planner_belief_y')
        belief_ok = _pf(row, 'planner_belief_available') >= 0.5 if math.isfinite(_pf(row, 'planner_belief_available')) else False
        cov_x = _pf(row, 'planner_cov_x')
        cov_y = _pf(row, 'planner_cov_y')

        if truth_ok and belief_ok and math.isfinite(belief_x) and math.isfinite(belief_y):
            err = math.hypot(truth_x - belief_x, truth_y - belief_y)
            loc_errors.append(err)

            if math.isfinite(cov_x) and math.isfinite(cov_y) and (cov_x + cov_y) > 0:
                cov_trace = cov_x + cov_y
                cov_traces.append(cov_trace)
                overconf = err / math.sqrt(cov_trace)
                overconf_values.append(overconf)

        if truth_ok and math.isfinite(truth_x) and math.isfinite(truth_y) and gp_interp is not None:
            rho = _query_rho(gp_interp, truth_x, truth_y)
            if math.isfinite(rho):
                rho_values.append(rho)

    mean_loc_error = float(np.mean(loc_errors)) if loc_errors else math.nan
    mean_overconf = float(np.mean(overconf_values)) if overconf_values else math.nan
    mean_cov_trace = float(np.mean(cov_traces)) if cov_traces else math.nan

    f_shadow = math.nan
    mean_rho_plan = math.nan
    if rho_values:
        f_shadow = float(np.mean(np.array(rho_values) < RHO_SHADOW_THRESHOLD))
        mean_rho_plan = float(np.mean(rho_values))

    # Path efficiency: straight-line distance / path length
    path_efficiency = math.nan
    if task_info and math.isfinite(path_length_m) and path_length_m > 0:
        start = task_info.get('start')
        goal = task_info.get('goal')
        if start and goal:
            straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
            path_efficiency = straight / path_length_m

    return {
        'goal_reached': goal_reached,
        'collision': crashed,
        'completion_reason': completion_reason,
        'path_length_m': path_length_m,
        'min_goal_distance': min_goal_distance,
        'elapsed_s': elapsed_s,
        'mean_loc_error_m': mean_loc_error,
        'mean_overconf': mean_overconf,
        'f_shadow': f_shadow,
        'mean_rho_plan': mean_rho_plan,
        'path_efficiency': path_efficiency,
        'mean_cov_trace': mean_cov_trace,
        'n_rows': len(rows),
    }


TASK_INFO = {
    'shadow_tradeoff_a': {'start': (-2.0, 0.5), 'goal': (2.0, 0.5)},
    'shadow_tradeoff_b': {'start': (-2.0, -1.0), 'goal': (2.0, 0.5)},
    'sanity_open':       {'start': (-2.0, -1.5), 'goal': (2.0, -1.5)},
}

FIELDNAMES = [
    'task', 'condition', 'seed', 'planner', 'run_dir',
    'goal_reached', 'collision', 'completion_reason',
    'path_length_m', 'min_goal_distance', 'elapsed_s',
    'mean_loc_error_m', 'mean_overconf', 'f_shadow', 'mean_rho_plan',
    'path_efficiency', 'mean_cov_trace', 'n_rows', 'outcome',
]


def _format(v) -> str:
    if v is None:
        return ''
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        return '' if not math.isfinite(v) else f'{v:.6f}'
    return str(v)


def _print_summary(rows: list[dict]) -> str:
    lines = []
    header = f'{"Task":<20} {"Cond":<6} {"N":>3} {"Goal%":>6} {"Coll%":>6} '
    header += f'{"L(m)":>9} {"ē(m)":>9} {"c̄":>9} {"f_shad":>8} {"η":>8}'
    lines.append(header)
    lines.append('-' * len(header))

    tasks = ['shadow_tradeoff_a', 'shadow_tradeoff_b', 'sanity_open']
    conditions = ['C1', 'C2', 'C3']

    for task in tasks:
        for cond in conditions:
            subset = [r for r in rows if r['task'] == task and r['condition'] == cond
                      and r['outcome'] not in ('infra_invalid', None)]
            if not subset:
                continue
            n = len(subset)
            goal_rate = 100.0 * sum(1 for r in subset if r['goal_reached']) / n
            coll_rate = 100.0 * sum(1 for r in subset if r['collision']) / n
            valid = [r for r in subset if not r['collision']]

            def _mean_std(key):
                vals = [r[key] for r in valid if math.isfinite(float(r[key] or 'nan'))]
                if not vals:
                    return 'n/a'
                return f'{np.mean(vals):.2f}±{np.std(vals):.2f}'

            l_str = _mean_std('path_length_m')
            e_str = _mean_std('mean_loc_error_m')
            c_str = _mean_std('mean_overconf')
            f_str = _mean_std('f_shadow')
            eta_str = _mean_std('path_efficiency')
            lines.append(
                f'{task:<20} {cond:<6} {n:>3} {goal_rate:>5.0f}% {coll_rate:>5.0f}% '
                f'{l_str:>9} {e_str:>9} {c_str:>9} {f_str:>8} {eta_str:>8}'
            )
        lines.append('')

    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute paper metrics from IWAI campaign runs.')
    parser.add_argument('--campaign-log', required=True,
                        help='Path to campaign_log.json from run_iwai_campaign.py.')
    parser.add_argument('--gp-artifact', default='',
                        help='Path to GP .npz artifact for rho_plan queries (optional but needed for f_shadow).')
    parser.add_argument('--out', default='paper_metrics.csv',
                        help='Output CSV path.')
    parser.add_argument('--summary-out', default='paper_summary.txt',
                        help='Output summary text path.')
    args = parser.parse_args()

    campaign_log_path = Path(args.campaign_log).expanduser().resolve()
    if not campaign_log_path.is_file():
        print(f'ERROR: campaign log not found: {campaign_log_path}', file=sys.stderr)
        return 1

    campaign_log = json.loads(campaign_log_path.read_text(encoding='utf-8'))
    print(f'Loaded {len(campaign_log)} run entries from {campaign_log_path}')

    gp_interp = None
    if args.gp_artifact:
        gp_path = Path(args.gp_artifact).expanduser().resolve()
        if gp_path.is_file():
            print(f'Loading GP artifact: {gp_path}')
            gp_interp = _load_gp(gp_path)
        else:
            print(f'WARNING: GP artifact not found: {gp_path} — f_shadow and mean_rho will be NaN')

    out_path = Path(args.out).expanduser().resolve()
    summary_path = Path(args.summary_out).expanduser().resolve()

    all_rows = []
    for key, entry in campaign_log.items():
        task = str(entry.get('task', ''))
        condition = str(entry.get('condition', ''))
        seed = entry.get('seed', '')
        planner = str(entry.get('planner', ''))
        outcome = str(entry.get('outcome', ''))
        run_dir_str = str(entry.get('run_dir', ''))

        base_row = {
            'task': task, 'condition': condition, 'seed': seed,
            'planner': planner, 'run_dir': run_dir_str, 'outcome': outcome,
        }

        if outcome == 'infra_invalid' or not run_dir_str:
            base_row.update({k: math.nan for k in FIELDNAMES if k not in base_row})
            base_row['goal_reached'] = False
            base_row['collision'] = False
            base_row['completion_reason'] = 'infra_invalid'
            all_rows.append(base_row)
            print(f'  infra_invalid: {task}/{condition}/seed{seed}')
            continue

        run_dir = Path(run_dir_str)
        summary_path_run = run_dir / 'run_summary.json'
        if not summary_path_run.is_file():
            # Try to find summary in subdirectory
            candidates = sorted(run_dir.rglob('run_summary.json'))
            if candidates:
                summary_path_run = candidates[-1]
                run_dir = summary_path_run.parent

        if not summary_path_run.is_file():
            print(f'  WARNING: no run_summary.json for {task}/{condition}/seed{seed}')
            base_row.update({k: math.nan for k in FIELDNAMES if k not in base_row})
            base_row['outcome'] = 'infra_invalid'
            all_rows.append(base_row)
            continue

        summary = json.loads(summary_path_run.read_text(encoding='utf-8'))
        task_info = TASK_INFO.get(task)
        metrics = _compute_run_metrics(run_dir, summary, gp_interp, task_info)
        base_row.update(metrics)
        all_rows.append(base_row)
        gr = 'YES' if metrics['goal_reached'] else 'no '
        print(f'  {task:<22} {condition} seed{seed}: goal={gr} '
              f'L={metrics["path_length_m"]:.2f}m  ē={metrics["mean_loc_error_m"]:.3f}m  '
              f'c̄={metrics["mean_overconf"]:.2f}  f_shadow={metrics["f_shadow"]:.2f}')

    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: _format(row.get(k)) for k in FIELDNAMES})
    print(f'\nMetrics written to: {out_path}')

    summary_text = _print_summary(all_rows)
    print('\n' + summary_text)
    summary_path.write_text(summary_text, encoding='utf-8')
    print(f'Summary written to: {summary_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
