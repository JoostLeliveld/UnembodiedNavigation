#!/usr/bin/env python3
"""Plot grouped bar charts of paper metrics from compute_paper_metrics.py output.

Usage:
    python plot_paper_metrics.py \
        --metrics-csv paper_metrics.csv \
        --out-dir figures/metrics/

Produces one multi-panel PNG (metrics_overview.png) and one PNG per metric.
Fails if any required column is missing from the CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'visibility_comparison'))
import common  # noqa: F401

# ── configuration ─────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    'task', 'condition',
    'goal_reached', 'collision',
    'path_length_m', 'final_goal_dist_m', 'mean_rho_plan', 'f_shadow',
]

# final_goal_dist_m is an alias for min_goal_distance in the metrics CSV
COLUMN_ALIASES = {
    'final_goal_dist_m': 'min_goal_distance',
}

METRIC_SPECS: list[dict] = [
    {
        'key': 'goal_reached_pct',
        'label': 'Goal reached [%]',
        'computed': True,
        'ylim': (0, 105),
    },
    {
        'key': 'collision_pct',
        'label': 'Collision rate [%]',
        'computed': True,
        'ylim': (0, 105),
    },
    {
        'key': 'path_length_m',
        'label': 'Path length [m]',
        'computed': False,
        'ylim': None,
    },
    {
        'key': 'final_goal_dist_m',
        'label': 'Final goal distance [m]',
        'computed': False,
        'ylim': None,
    },
    {
        'key': 'mean_rho_plan',
        'label': r'Mean $\rho_\mathrm{plan}$',
        'computed': False,
        'ylim': (0.0, 1.0),
    },
    {
        'key': 'f_shadow',
        'label': r'Shadow fraction $f_\mathrm{shadow}$',
        'computed': False,
        'ylim': (0.0, 1.0),
    },
]

TASKS = ['shadow_tradeoff_a', 'shadow_tradeoff_b', 'sanity_open']
TASK_LABELS = {
    'shadow_tradeoff_a': 'Task A',
    'shadow_tradeoff_b': 'Task B',
    'sanity_open':       'Task S',
}

CONDITIONS = ['C1', 'C2', 'C3']
CONDITION_COLORS = {
    'C1': '#4393c3',
    'C2': '#d6604d',
    'C3': '#74c476',
}
CONDITION_LABELS = {
    'C1': 'C1 (baseline)',
    'C2': 'C2 (EFE-full)',
    'C3': 'C3 (GP-risk)',
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-csv', required=True,
                        help='Path to paper_metrics.csv produced by compute_paper_metrics.py')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for PNG figures.')
    return parser.parse_args()


def _pf(value: str) -> float:
    if value in (None, '', 'nan', 'NaN'):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _load_metrics(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        print(f'ERROR: metrics CSV not found: {csv_path}', file=sys.stderr)
        sys.exit(1)
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f'ERROR: metrics CSV is empty: {csv_path}', file=sys.stderr)
        sys.exit(1)
    headers = list(rows[0].keys())
    # Check required columns, accounting for aliases
    missing = []
    for col in REQUIRED_COLUMNS:
        resolved = COLUMN_ALIASES.get(col, col)
        if resolved not in headers and col not in headers:
            missing.append(f'{col} (or {resolved})')
    if missing:
        print(
            f'ERROR: metrics CSV is missing required columns: {missing}\n'
            f'Columns present: {headers}',
            file=sys.stderr,
        )
        sys.exit(1)
    return rows


def _resolve_col(rows: list[dict], key: str) -> str:
    """Return the actual column name, following aliases."""
    if not rows:
        return key
    alias = COLUMN_ALIASES.get(key, key)
    if alias in rows[0]:
        return alias
    return key


def _aggregate(rows: list[dict], task: str, condition: str, metric_key: str) -> tuple[float, float, int]:
    """Return (mean, std, n) for the given (task, condition, metric_key).

    For computed percentage keys (goal_reached_pct, collision_pct), derives values
    from the raw boolean columns. Excludes infra_invalid rows.
    """
    subset = [
        r for r in rows
        if r.get('task') == task and r.get('condition') == condition
        and r.get('outcome') not in ('infra_invalid',)
    ]
    if not subset:
        return math.nan, math.nan, 0

    if metric_key == 'goal_reached_pct':
        vals = [100.0 * (1.0 if r.get('goal_reached') in ('1', 'True', 'true', 1, True) else 0.0)
                for r in subset]
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    if metric_key == 'collision_pct':
        vals = [100.0 * (1.0 if r.get('collision') in ('1', 'True', 'true', 1, True) else 0.0)
                for r in subset]
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    resolved = _resolve_col(rows, metric_key)
    vals = [_pf(r.get(resolved, '')) for r in subset]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return math.nan, math.nan, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def _draw_metric_panel(ax, rows: list[dict], metric: dict) -> None:
    """Draw one grouped bar chart for a single metric."""
    key = metric['key']
    n_tasks = len(TASKS)
    n_conds = len(CONDITIONS)
    group_width = 0.75
    bar_w = group_width / n_conds

    x_centers = np.arange(n_tasks, dtype=float)

    for ci, cond in enumerate(CONDITIONS):
        offsets = x_centers + (ci - n_conds / 2.0 + 0.5) * bar_w
        means, stds = [], []
        for task in TASKS:
            m, s, n = _aggregate(rows, task, cond, key)
            means.append(m)
            stds.append(s if math.isfinite(s) else 0.0)

        means_arr = np.array(means, dtype=float)
        stds_arr  = np.array(stds, dtype=float)
        valid = np.isfinite(means_arr)

        ax.bar(
            offsets[valid],
            means_arr[valid],
            bar_w * 0.88,
            yerr=stds_arr[valid],
            color=CONDITION_COLORS[cond],
            alpha=0.85,
            label=CONDITION_LABELS[cond],
            error_kw={'elinewidth': 0.9, 'capsize': 2.5, 'ecolor': '#333333'},
            zorder=3,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=8)
    ax.set_ylabel(metric['label'], fontsize=8)
    ax.set_title(metric['label'], fontsize=9, pad=3)
    if metric.get('ylim') is not None:
        ax.set_ylim(*metric['ylim'])
    ax.grid(True, axis='y', color='#d8d8d8', linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main() -> int:
    args = _parse_args()
    csv_path = Path(args.metrics_csv).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve()

    rows = _load_metrics(csv_path)
    print(f'Loaded {len(rows)} rows from {csv_path}')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    out_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(METRIC_SPECS)
    ncols = 3
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.0, nrows * 3.2),
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)

    for i, metric in enumerate(METRIC_SPECS):
        _draw_metric_panel(axes_flat[i], rows, metric)
        if i == 0:
            axes_flat[i].legend(
                loc='upper right', fontsize=7,
                frameon=True, framealpha=0.9, fancybox=False,
                borderpad=0.3, labelspacing=0.2,
            )

    # Hide unused axes
    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Paper metrics: C1 vs C2 vs C3 across tasks', fontsize=11, y=1.01)
    overview_path = out_dir / 'metrics_overview.png'
    fig.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {overview_path}')

    # ── one PNG per metric ─────────────────────────────────────────────────
    for metric in METRIC_SPECS:
        fig2, ax2 = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
        _draw_metric_panel(ax2, rows, metric)
        ax2.legend(
            loc='upper right', fontsize=7,
            frameon=True, framealpha=0.9, fancybox=False,
            borderpad=0.3, labelspacing=0.2,
        )
        single_path = out_dir / f'metric_{metric["key"]}.png'
        fig2.savefig(single_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f'Wrote {single_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
