#!/usr/bin/env python3
"""Show locked initial GP versus GP maps after each belief-aware update type."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt

from common import LOGS_ROOT, REPO_ROOT, write_csv, write_manifest


DEFAULT_BASELINE_GP = REPO_ROOT / 'paper_artifacts' / 'gp' / 'warehouse_visibility_gp_v1' / 'yolo_score_raw_gp.npz'
DEFAULT_OUT = LOGS_ROOT / 'belief_aware_gp_showcase'
DEFAULT_SCORE_ARTIFACT_DIR = LOGS_ROOT / 'belief_aware_gp_score_v1'
DEFAULT_HIT_ARTIFACT_DIR = LOGS_ROOT / 'belief_aware_gp_v1'
MODES = ('naive', 'uncertainty_weighted', 'belief_spread', 'expected_kernel')


def _artifact_dir_for_target(target: str) -> Path:
    return DEFAULT_SCORE_ARTIFACT_DIR if target == 'score' else DEFAULT_HIT_ARTIFACT_DIR


def _target_id(target: str) -> str:
    return 'yolo_score_raw' if target == 'score' else 'det_hit'


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _load_map(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f'GP artifact not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        return {
            'xs': np.asarray(data['xs'], dtype=float),
            'ys': np.asarray(data['ys'], dtype=float),
            'p_plan': np.asarray(data['P_conservative_plan_map'], dtype=float),
            'p_mean': np.asarray(data['P_mean_map'], dtype=float),
        }


def _assert_same_grid(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], *, label: str) -> None:
    if reference['xs'].shape != candidate['xs'].shape or reference['ys'].shape != candidate['ys'].shape:
        raise RuntimeError(f'{label} grid shape does not match the baseline grid')
    if not np.allclose(reference['xs'], candidate['xs']) or not np.allclose(reference['ys'], candidate['ys']):
        raise RuntimeError(f'{label} grid coordinates do not match the baseline grid')


def _mode_label(mode: str, *, baseline_label: str) -> str:
    baseline_title = str(baseline_label or '').strip()
    if not baseline_title:
        baseline_title = 'locked initial GP'
    return {
        'initial': f'Before:\n{baseline_title}',
        'naive': 'After:\nnaive mean update',
        'uncertainty_weighted': 'After:\nuncertainty-weighted',
        'belief_spread': 'After:\nbelief-spread',
        'expected_kernel': 'After:\nexpected-kernel',
    }.get(mode, mode.replace('_', '\n'))


def _summary_lookup(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {str(row.get(key, '')): row for row in rows}


def _float_text(row: dict[str, str], key: str) -> str:
    raw = str(row.get(key, '')).strip()
    if not raw:
        return ''
    try:
        return f'{float(raw):.5f}'
    except ValueError:
        return raw


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot locked initial GP and after-update GP maps side by side.')
    parser.add_argument('--target', choices=('score', 'hit'), default='score')
    parser.add_argument('--artifact-dir', default='')
    parser.add_argument('--baseline-gp', default=str(DEFAULT_BASELINE_GP))
    parser.add_argument('--baseline-label', default='locked initial GP')
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--map-key', default='P_conservative_plan_map', choices=('P_conservative_plan_map', 'P_mean_map'))
    args = parser.parse_args()

    target = str(args.target)
    target_id = _target_id(target)
    artifact_dir = Path(args.artifact_dir).expanduser().resolve() if args.artifact_dir else _artifact_dir_for_target(target).resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(args.baseline_gp).expanduser().resolve()
    baseline_label = str(args.baseline_label).strip() or 'baseline GP'

    fit_rows = _summary_lookup(_read_csv(artifact_dir / 'fit_summary.csv'), 'mode')
    validation_rows = _summary_lookup(_read_csv(artifact_dir / 'validation_summary.csv'), 'mode')

    entries: list[dict] = []
    baseline = _load_map(baseline_path)
    baseline_grid = baseline['p_plan'] if args.map_key == 'P_conservative_plan_map' else baseline['p_mean']
    entries.append(
        {
            'mode': 'initial',
            'path': baseline_path,
            'map': baseline_grid,
            'map_data': baseline,
            'fit': {},
            'validation': {},
        }
    )

    for mode in MODES:
        artifact_path = artifact_dir / f'{target_id}_{mode}_gp.npz'
        map_data = _load_map(artifact_path)
        _assert_same_grid(baseline, map_data, label=mode)
        grid = map_data['p_plan'] if args.map_key == 'P_conservative_plan_map' else map_data['p_mean']
        entries.append(
            {
                'mode': mode,
                'path': artifact_path,
                'map': grid,
                'map_data': map_data,
                'fit': fit_rows.get(mode, {}),
                'validation': validation_rows.get(mode, {}),
            }
        )

    deltas = [entry['map'] - baseline_grid for entry in entries]
    delta_abs_max = max(float(np.max(np.abs(delta))) for delta in deltas[1:])
    delta_abs_max = max(delta_abs_max, 0.05)

    xs = baseline['xs']
    ys = baseline['ys']
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    fig, axes = plt.subplots(2, len(entries), figsize=(3.45 * len(entries), 7.2), constrained_layout=True)
    fig.suptitle(f'GP before and after belief-aware updates ({target_id}, {args.map_key})', fontsize=15, fontweight='bold')

    map_images = []
    delta_images = []
    for col, entry in enumerate(entries):
        mode = str(entry['mode'])
        grid = entry['map']
        delta = grid - baseline_grid

        ax = axes[0, col]
        im = ax.imshow(grid, origin='lower', extent=extent, vmin=0.0, vmax=1.0, cmap='viridis', aspect='equal')
        map_images.append(im)
        ax.set_title(_mode_label(mode, baseline_label=baseline_label), fontsize=10.5)
        ax.set_xlabel('x [m]')
        if col == 0:
            ax.set_ylabel('y [m]')
        else:
            ax.set_yticklabels([])
        fit = entry['fit']
        validation = entry['validation']
        detail_lines = [f'mean={float(np.mean(grid)):.3f}']
        if validation:
            detail_lines.append(f'Brier={_float_text(validation, "brier_mean")}')
        if mode == 'expected_kernel':
            detail_lines.append('main method')
        ax.text(
            0.03,
            0.03,
            '\n'.join(detail_lines),
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=8,
            color='white',
            bbox={'facecolor': 'black', 'alpha': 0.52, 'edgecolor': 'none', 'pad': 3.0},
        )

        ax = axes[1, col]
        im_delta = ax.imshow(
            delta,
            origin='lower',
            extent=extent,
            vmin=-delta_abs_max,
            vmax=delta_abs_max,
            cmap='coolwarm',
            aspect='equal',
        )
        delta_images.append(im_delta)
        ax.set_title(f'delta from {baseline_label}', fontsize=10.5)
        ax.set_xlabel('x [m]')
        if col == 0:
            ax.set_ylabel('y [m]')
        else:
            ax.set_yticklabels([])
        ax.text(
            0.03,
            0.03,
            f'mean Δ={float(np.mean(delta)):+.3f}\nmin/max Δ={float(np.min(delta)):+.3f}/{float(np.max(delta)):+.3f}',
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=8,
            color='black',
            bbox={'facecolor': 'white', 'alpha': 0.72, 'edgecolor': 'none', 'pad': 3.0},
        )

    fig.colorbar(map_images[0], ax=list(axes[0, :]), shrink=0.72, label=args.map_key)
    fig.colorbar(delta_images[0], ax=list(axes[1, :]), shrink=0.72, label=f'after - {baseline_label}')

    stem = f'gp_before_after_updates_{target}'
    png_path = output_dir / f'{stem}.png'
    pdf_path = output_dir / f'{stem}.pdf'
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary_rows = []
    for entry, delta in zip(entries, deltas):
        mode = str(entry['mode'])
        validation = entry['validation']
        grid = entry['map']
        summary_rows.append(
            {
                'mode': mode,
                'artifact_path': str(entry['path']),
                'map_mean': f'{float(np.mean(grid)):.10g}',
                'map_min': f'{float(np.min(grid)):.10g}',
                'map_max': f'{float(np.max(grid)):.10g}',
                'delta_mean': f'{float(np.mean(delta)):.10g}',
                'delta_min': f'{float(np.min(delta)):.10g}',
                'delta_max': f'{float(np.max(delta)):.10g}',
                'brier_mean': str(validation.get('brier_mean', '')),
                'logloss_mean': str(validation.get('logloss_mean', '')),
            }
        )
    summary_csv = output_dir / f'{stem}_summary.csv'
    write_csv(
        summary_csv,
        (
            'mode',
            'artifact_path',
            'map_mean',
            'map_min',
            'map_max',
            'delta_mean',
            'delta_min',
            'delta_max',
            'brier_mean',
            'logloss_mean',
        ),
        summary_rows,
    )
    write_manifest(
        output_dir / f'{stem}_manifest.json',
        {
            'target': target,
            'target_id': target_id,
            'map_key': str(args.map_key),
            'baseline_gp': str(baseline_path),
            'baseline_label': baseline_label,
            'artifact_dir': str(artifact_dir),
            'figure_png': str(png_path),
            'figure_pdf': str(pdf_path),
            'summary_csv': str(summary_csv),
            'modes': [entry['mode'] for entry in entries],
            'notes': [
                f'The first column is the before-update baseline: {baseline_label}. All other columns are after-update artifacts.',
                f'The second row is the after-update map minus {baseline_label} on the same grid.',
                'For target=hit, the after maps model binary detector hits while the default locked initial GP is the yolo_score_raw baseline.',
            ],
        },
    )

    print(f'Wrote before/after GP showcase PNG: {png_path}')
    print(f'Wrote before/after GP showcase PDF: {pdf_path}')
    print(f'Wrote summary CSV: {summary_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
