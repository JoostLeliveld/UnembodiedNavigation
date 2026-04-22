#!/usr/bin/env python3
"""Analyze heading-marginal GP targets and fitted visibility maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def _grid_from_xy(rows: list[dict[str, object]], value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(sorted({_float(row['x']) for row in rows if math.isfinite(_float(row.get('x')))}), dtype=float)
    ys = np.asarray(sorted({_float(row['y']) for row in rows if math.isfinite(_float(row.get('y')))}), dtype=float)
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)
    if xs.size == 0 or ys.size == 0:
        return xs, ys, grid
    x_lookup = {round(float(x), 6): idx for idx, x in enumerate(xs)}
    y_lookup = {round(float(y), 6): idx for idx, y in enumerate(ys)}
    for row in rows:
        x = _float(row.get('x'))
        y = _float(row.get('y'))
        value = _float(row.get(value_key))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(value)):
            continue
        grid[y_lookup[round(y, 6)], x_lookup[round(x, 6)]] = value
    return xs, ys, grid


def _plot_grid(path: Path, xs: np.ndarray, ys: np.ndarray, grid: np.ndarray, title: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    if xs.size and ys.size:
        im = ax.imshow(
            grid,
            origin='lower',
            extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
            aspect='equal',
            cmap='viridis',
            vmin=0.0,
            vmax=1.0 if np.nanmax(grid) <= 1.0 else None,
        )
        fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _artifact_map(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        xs = np.asarray(data['xs'], dtype=float)
        ys = np.asarray(data['ys'], dtype=float)
        grid = np.asarray(data['P_conservative_plan_map'], dtype=float)
    return xs, ys, grid


def main() -> int:
    parser = argparse.ArgumentParser(description='Analyze GP capture targets and fitted GP maps.')
    parser.add_argument('--gp-targets', default='logs/visibility_comparison/current_targets/gp_targets.csv')
    parser.add_argument('--capture-samples', default='logs/visibility_comparison/current_capture/samples.csv')
    parser.add_argument('--method', default='yolo_score_raw')
    parser.add_argument('--out', default='logs/visibility_comparison/gp_analysis')
    parser.add_argument('--gp-artifacts', nargs='*', default=[])
    args = parser.parse_args()

    targets_path = Path(args.gp_targets).expanduser().resolve()
    capture_path = Path(args.capture_samples).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = _read_rows(targets_path)
    capture_rows = {str(row.get('sample_id', '')): row for row in _read_rows(capture_path)}
    method = str(args.method)

    grouped: dict[tuple[float, float], dict[str, object]] = {}
    for row in target_rows:
        x = _float(row.get('x'))
        y = _float(row.get('y'))
        value = _float(row.get(method))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(value)):
            continue
        key = (round(x, 6), round(y, 6))
        bucket = grouped.setdefault(key, {'x': x, 'y': y, 'values': [], 'theta': []})
        bucket['values'].append(float(value))
        cap = capture_rows.get(str(row.get('sample_id', '')))
        theta = _float(cap.get('theta')) if cap else math.nan
        if math.isfinite(theta):
            bucket['theta'].append(theta)

    rows = []
    for idx, key in enumerate(sorted(grouped.keys(), key=lambda item: (item[1], item[0]))):
        bucket = grouped[key]
        values = np.asarray(bucket['values'], dtype=float)
        thetas = np.asarray(bucket['theta'], dtype=float)
        rows.append({
            'xy_id': f'xy_{idx:06d}',
            'x': f"{float(bucket['x']):.8f}",
            'y': f"{float(bucket['y']):.8f}",
            'heading_count': int(values.size),
            'heading_unique_count': int(np.unique(np.round(thetas, 6)).size) if thetas.size else 0,
            'target_mean': f"{float(np.mean(values)):.8f}",
            'target_std': f"{float(np.std(values)):.8f}",
            'target_min': f"{float(np.min(values)):.8f}",
            'target_max': f"{float(np.max(values)):.8f}",
            'target_range': f"{float(np.max(values) - np.min(values)):.8f}",
        })

    _write_csv(
        output_dir / f'{method}_heading_variability.csv',
        ('xy_id', 'x', 'y', 'heading_count', 'heading_unique_count', 'target_mean', 'target_std', 'target_min', 'target_max', 'target_range'),
        rows,
    )
    summary = {
        'method': method,
        'target_rows': int(len(target_rows)),
        'xy_rows': int(len(rows)),
        'mean_heading_count': float(np.mean([int(row['heading_count']) for row in rows])) if rows else 0.0,
        'mean_target': float(np.mean([_float(row['target_mean']) for row in rows])) if rows else math.nan,
        'mean_heading_std': float(np.mean([_float(row['target_std']) for row in rows])) if rows else math.nan,
        'max_heading_range': float(np.max([_float(row['target_range']) for row in rows])) if rows else math.nan,
    }
    (output_dir / f'{method}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    xs, ys, mean_grid = _grid_from_xy(rows, 'target_mean')
    _plot_grid(output_dir / f'{method}_xy_mean.png', xs, ys, mean_grid, f'{method}: heading-averaged target', 'mean target')
    _xs, _ys, std_grid = _grid_from_xy(rows, 'target_std')
    _plot_grid(output_dir / f'{method}_xy_heading_std.png', xs, ys, std_grid, f'{method}: heading variability', 'std across headings')

    if args.gp_artifacts:
        n = len(args.gp_artifacts)
        fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.2), squeeze=False)
        for ax, artifact_arg in zip(axes[0], args.gp_artifacts):
            artifact = Path(artifact_arg).expanduser().resolve()
            xs_a, ys_a, grid_a = _artifact_map(artifact)
            im = ax.imshow(
                grid_a,
                origin='lower',
                extent=[float(xs_a[0]), float(xs_a[-1]), float(ys_a[0]), float(ys_a[-1])],
                aspect='equal',
                cmap='viridis',
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_title(artifact.parent.name)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            fig.colorbar(im, ax=ax, label='P conservative')
        fig.tight_layout()
        fig.savefig(output_dir / f'{method}_gp_artifact_comparison.png', dpi=160)
        plt.close(fig)

    print(f'Wrote GP capture analysis to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
