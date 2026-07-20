#!/usr/bin/env python3
"""Plot EFE component time series from a single experiment run directory.

Loads experiment.csv, prints all available column names to stderr, then searches
for columns matching: 'efe', 'risk', 'ambiguity', 'control_cost', 'total'
(case-insensitive substring match). Plots each matched column as a time series.

Usage:
    python plot_efe_decomposition.py \
        --run-dir logs/experiments/experiment_20240501_120000 \
        --out figures/efe_decomposition.png

Hard-fail rules:
  - --run-dir must exist and contain experiment.csv.
  - experiment.csv must not be empty and must have a header row.
  - If none of the expected EFE component columns are found, fail with a
    clear message listing all available column names.
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

# Substring patterns to search for (case-insensitive).
EFE_PATTERNS = ['efe', 'risk', 'ambiguity', 'control_cost', 'total']

# Preferred color cycle for matched columns (will repeat if more columns found).
_COLOR_CYCLE = [
    '#e41a1c',  # red       — total EFE
    '#377eb8',  # blue      — risk
    '#4daf4a',  # green     — ambiguity
    '#ff7f00',  # orange    — control_cost
    '#984ea3',  # purple    — other
    '#a65628',  # brown
    '#f781bf',  # pink
    '#999999',  # grey
]

# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True,
                        help='Path to experiment run directory (must contain experiment.csv).')
    parser.add_argument('--out', required=True,
                        help='Output PNG path.')
    return parser.parse_args()


def _pf(value) -> float:
    if value in (None, '', 'nan', 'NaN', 'inf', '-inf'):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _match_efe_columns(headers: list[str]) -> list[str]:
    """Return headers whose names contain any EFE_PATTERNS substring (case-insensitive)."""
    matched = []
    for h in headers:
        hl = h.lower()
        if any(pat in hl for pat in EFE_PATTERNS):
            matched.append(h)
    return matched


def _load_experiment_csv(run_dir: Path) -> tuple[list[str], list[dict]]:
    """Load experiment.csv; return (headers, rows). Hard-fails on missing/empty file."""
    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        print(f'ERROR: experiment.csv not found in {run_dir}', file=sys.stderr)
        sys.exit(1)
    if csv_path.stat().st_size == 0:
        print(f'ERROR: experiment.csv is empty: {csv_path}', file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    headers: list[str] = []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        if not headers:
            print(f'ERROR: experiment.csv has no header row: {csv_path}', file=sys.stderr)
            sys.exit(1)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f'ERROR: experiment.csv has a header but no data rows: {csv_path}', file=sys.stderr)
        sys.exit(1)

    return headers, rows


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not run_dir.is_dir():
        print(f'ERROR: run directory not found: {run_dir}', file=sys.stderr)
        sys.exit(1)

    # ── load CSV ──────────────────────────────────────────────────────────────
    headers, rows = _load_experiment_csv(run_dir)

    # Print all columns to stderr so the user can see what is available.
    print('Available columns in experiment.csv:', file=sys.stderr)
    for h in headers:
        print(f'  {h}', file=sys.stderr)

    # ── find EFE component columns ────────────────────────────────────────────
    efe_cols = _match_efe_columns(headers)
    if not efe_cols:
        print(
            f'\nERROR: no EFE component columns found in experiment.csv.\n'
            f'Searched for substrings (case-insensitive): {EFE_PATTERNS}\n'
            f'All available columns:\n  ' + '\n  '.join(headers),
            file=sys.stderr,
        )
        sys.exit(1)

    print(f'Matched EFE columns ({len(efe_cols)}): {efe_cols}', file=sys.stderr)

    # ── build time axis ───────────────────────────────────────────────────────
    # Use 'time' or 'stamp' column if present; otherwise use row index.
    time_col: str | None = None
    for candidate in ('time', 'stamp', 'timestamp'):
        if candidate in headers:
            time_col = candidate
            break

    n = len(rows)

    if time_col is not None:
        raw_t = np.array([_pf(r.get(time_col, '')) for r in rows], dtype=float)
        finite_t = raw_t[np.isfinite(raw_t)]
        if finite_t.size > 1:
            # Shift to start at 0 for readability
            t = raw_t - float(np.nanmin(finite_t))
            xlabel = f'Time [s] (from first row; column: {time_col!r})'
        else:
            t = np.arange(n, dtype=float)
            xlabel = 'Row index'
    else:
        t = np.arange(n, dtype=float)
        xlabel = 'Row index'

    # ── build data arrays ─────────────────────────────────────────────────────
    series: dict[str, np.ndarray] = {}
    for col in efe_cols:
        arr = np.array([_pf(r.get(col, '')) for r in rows], dtype=float)
        series[col] = arr

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)

    for i, col in enumerate(efe_cols):
        color = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
        y = series[col]
        # Use the time axis with matching length; mask non-finite pairs
        t_plot = t[:len(y)] if len(t) >= len(y) else np.arange(len(y), dtype=float)
        mask = np.isfinite(y) & np.isfinite(t_plot)
        if not np.any(mask):
            print(f'  Skipping column {col!r}: all values are NaN/Inf', file=sys.stderr)
            continue
        ax.plot(
            t_plot[mask], y[mask],
            color=color,
            linewidth=1.4,
            label=col,
            alpha=0.88,
        )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('EFE component value', fontsize=9)
    ax.set_title(
        f'EFE decomposition — {run_dir.name}',
        fontsize=10,
    )
    ax.legend(
        loc='best',
        fontsize=7.5,
        frameon=True,
        framealpha=0.92,
        fancybox=False,
        borderpad=0.4,
        handlelength=1.8,
        labelspacing=0.3,
    )
    ax.grid(True, axis='y', color='#d8d8d8', linewidth=0.5, zorder=0)
    ax.grid(True, axis='x', color='#d8d8d8', linewidth=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
