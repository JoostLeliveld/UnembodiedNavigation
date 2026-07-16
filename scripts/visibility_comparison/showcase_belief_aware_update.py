#!/usr/bin/env python3
"""Visualize how belief uncertainty changes GP detector-reliability updates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from common import LOGS_ROOT, REPO_ROOT, write_csv, write_manifest
from fit_belief_aware_gp import _aggregate_events, _load_events, _sigma_points


DEFAULT_EVENTS = LOGS_ROOT / 'belief_gp_events' / 'events.csv'
DEFAULT_SCORE_ARTIFACT_DIR = LOGS_ROOT / 'belief_aware_gp_score_v1'
DEFAULT_BASELINE_GP = REPO_ROOT / 'paper_artifacts' / 'gp' / 'warehouse_visibility_gp_v1' / 'yolo_score_raw_gp.npz'
DEFAULT_OUT = LOGS_ROOT / 'belief_aware_gp_showcase'


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _ellipse_patch(mean: np.ndarray, cov: np.ndarray, *, n_sigma: float, **kwargs) -> Ellipse:
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    vals = np.clip(vals, 0.0, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(float(vecs[1, 0]), float(vecs[0, 0])))
    width = 2.0 * float(n_sigma) * math.sqrt(float(vals[0]))
    height = 2.0 * float(n_sigma) * math.sqrt(float(vals[1]))
    return Ellipse(xy=(float(mean[0]), float(mean[1])), width=width, height=height, angle=angle, **kwargs)


def _local_grid(mean: np.ndarray, radius: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(float(mean[0] - radius), float(mean[0] + radius), int(n))
    ys = np.linspace(float(mean[1] - radius), float(mean[1] + radius), int(n))
    Xg, Yg = np.meshgrid(xs, ys)
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])
    return xs, ys, XY


def _rbf_influence(query: np.ndarray, points: np.ndarray, weights: np.ndarray, *, length_scale: float) -> np.ndarray:
    diff = query[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    bumps = np.exp(-0.5 * dist2 / max(float(length_scale) ** 2, 1e-12))
    return bumps @ np.asarray(weights, dtype=float)


def _read_validation(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _load_gp_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            'xs': np.asarray(data['xs'], dtype=float),
            'ys': np.asarray(data['ys'], dtype=float),
            'p_plan': np.asarray(data['P_conservative_plan_map'], dtype=float),
            'p_mean': np.asarray(data['P_mean_map'], dtype=float),
        }


def _line_at_y(gp_map: dict[str, np.ndarray], y: float) -> tuple[np.ndarray, np.ndarray, float]:
    ys = np.asarray(gp_map['ys'], dtype=float)
    row = int(np.argmin(np.abs(ys - float(y))))
    return np.asarray(gp_map['xs'], dtype=float), np.asarray(gp_map['p_plan'][row, :], dtype=float), float(ys[row])


def _select_aggregate(agg, *, min_count: float) -> int:
    trace = np.trace(agg.cov, axis1=1, axis2=2)
    valid = np.flatnonzero(agg.count >= float(min_count))
    if valid.size == 0:
        valid = np.arange(agg.X.shape[0])
    return int(valid[np.argmax(trace[valid])])


def _plot_influence_panel(
    ax,
    title: str,
    xs: np.ndarray,
    ys: np.ndarray,
    field: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    points: np.ndarray,
    *,
    marker_size: float,
    caption: str,
) -> None:
    ax.imshow(
        field.reshape(ys.size, xs.size),
        origin='lower',
        extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        vmin=0.0,
        vmax=1.0,
        cmap='viridis',
        aspect='equal',
    )
    ax.add_patch(_ellipse_patch(mean, cov, n_sigma=1.0, fill=False, edgecolor='white', linewidth=2.0))
    ax.scatter(points[:, 0], points[:, 1], s=marker_size, c='white', edgecolors='black', linewidths=0.8, zorder=3)
    ax.scatter([mean[0]], [mean[1]], s=35, c='#f0c419', edgecolors='black', linewidths=0.8, zorder=4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.text(
        0.02,
        0.02,
        caption,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=8.5,
        color='white',
        bbox={'facecolor': 'black', 'alpha': 0.55, 'edgecolor': 'none', 'pad': 3.0},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Create a data-backed figure explaining belief-aware GP updates.')
    parser.add_argument('--events', default=str(DEFAULT_EVENTS))
    parser.add_argument('--artifact-dir', default=str(DEFAULT_SCORE_ARTIFACT_DIR))
    parser.add_argument('--baseline-gp', default=str(DEFAULT_BASELINE_GP))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--target', choices=('score', 'hit'), default='score')
    parser.add_argument('--aggregate-resolution-m', type=float, default=0.20)
    parser.add_argument('--max-bin-weight', type=float, default=20.0)
    parser.add_argument('--min-count', type=float, default=4.0)
    parser.add_argument('--local-radius-m', type=float, default=1.15)
    parser.add_argument('--local-grid-n', type=int, default=160)
    parser.add_argument('--aggregate-index', type=int, default=-1)
    args = parser.parse_args()

    events_path = Path(args.events).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(artifact_dir / 'manifest.json')
    h = dict(manifest.get('hyperparameters') or {})
    gp_length_scale = float(h.get('gp_length_scale', 0.90))
    gp_noise_var = float(h.get('gp_noise_var', 0.05))
    pose_length_scale = float(h.get('pose_length_scale', 0.35))
    min_certainty = float(h.get('min_certainty', 0.05))
    spread_scale = float(h.get('spread_scale', 1.0))

    data = _load_events(events_path, target=str(args.target), min_prob=1e-4)
    agg = _aggregate_events(
        data,
        resolution_m=float(args.aggregate_resolution_m),
        max_bin_weight=float(args.max_bin_weight),
    )
    idx = int(args.aggregate_index)
    if idx < 0:
        idx = _select_aggregate(agg, min_count=float(args.min_count))
    if idx >= agg.X.shape[0]:
        raise RuntimeError(f'aggregate-index {idx} out of range for {agg.X.shape[0]} aggregate points')

    mean = np.asarray(agg.X[idx], dtype=float)
    cov = np.asarray(agg.cov[idx], dtype=float)
    target_value = float(agg.y[idx])
    count = float(agg.count[idx])
    trace = float(np.trace(cov))
    vals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    sigma_major = float(math.sqrt(max(float(vals[-1]), 0.0)))
    sigma_minor = float(math.sqrt(max(float(vals[0]), 0.0)))
    certainty = float(np.clip(math.exp(-0.5 * trace / max(pose_length_scale**2, 1e-12)), min_certainty, 1.0))
    alpha_naive = gp_noise_var / max(count, 1.0)
    alpha_weighted = gp_noise_var / max(count * certainty, 1e-12)
    sigma_points, sigma_weights = _sigma_points(mean, cov, scale=spread_scale)
    alpha_spread_each = gp_noise_var / max(count * 0.2, 1e-12)

    xs_local, ys_local, XY_local = _local_grid(mean, float(args.local_radius_m), int(args.local_grid_n))
    naive_field = _rbf_influence(XY_local, mean.reshape(1, 2), np.asarray([1.0]), length_scale=gp_length_scale)
    weighted_field = _rbf_influence(XY_local, mean.reshape(1, 2), np.asarray([certainty]), length_scale=gp_length_scale)
    spread_field = _rbf_influence(XY_local, sigma_points, sigma_weights, length_scale=gp_length_scale)

    target_id = 'yolo_score_raw' if str(args.target) == 'score' else 'det_hit'
    artifacts = {
        'locked original': Path(args.baseline_gp).expanduser().resolve(),
        'naive': artifact_dir / f'{target_id}_naive_gp.npz',
        'uncertainty-weighted': artifact_dir / f'{target_id}_uncertainty_weighted_gp.npz',
        'belief-spread': artifact_dir / f'{target_id}_belief_spread_gp.npz',
        'expected-kernel': artifact_dir / f'{target_id}_expected_kernel_gp.npz',
    }
    maps = {label: _load_gp_map(path) for label, path in artifacts.items() if path.is_file()}

    validation_rows = _read_validation(artifact_dir / 'validation_summary.csv')
    validation_rows = [row for row in validation_rows if row.get('mode') != 'constant_train_mean']

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True)
    fig.suptitle('Belief-aware GP update mechanics', fontsize=15, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(agg.X[:, 0], agg.X[:, 1], s=np.clip(agg.count, 3.0, 20.0), c=agg.y, cmap='cividis', alpha=0.55)
    ax.add_patch(_ellipse_patch(mean, cov, n_sigma=1.0, fill=False, edgecolor='crimson', linewidth=2.2))
    ax.add_patch(_ellipse_patch(mean, cov, n_sigma=2.0, fill=False, edgecolor='crimson', linewidth=1.2, linestyle='--'))
    ax.scatter([mean[0]], [mean[1]], s=55, c='crimson', edgecolors='black', linewidths=0.8, zorder=3)
    ax.set_title('Selected real aggregate event', fontsize=11)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.text(
        0.02,
        0.02,
        f'target={target_value:.4f}\ncount={count:.0f}\nsigma={sigma_major:.2f} / {sigma_minor:.2f} m',
        transform=ax.transAxes,
        fontsize=8.5,
        ha='left',
        va='bottom',
        bbox={'facecolor': 'white', 'alpha': 0.82, 'edgecolor': 'none', 'pad': 3.0},
    )

    _plot_influence_panel(
        axes[0, 1],
        'Naive: one full-strength point',
        xs_local,
        ys_local,
        naive_field,
        mean,
        cov,
        mean.reshape(1, 2),
        marker_size=90,
        caption=f'alpha={alpha_naive:.4f}\nrelative precision=1.00',
    )
    _plot_influence_panel(
        axes[0, 2],
        'Uncertainty-weighted: same point, less trust',
        xs_local,
        ys_local,
        weighted_field,
        mean,
        cov,
        mean.reshape(1, 2),
        marker_size=45,
        caption=f'certainty={certainty:.3f}\nalpha={alpha_weighted:.4f}',
    )
    _plot_influence_panel(
        axes[1, 0],
        'Belief-spread: update covers plausible poses',
        xs_local,
        ys_local,
        spread_field,
        mean,
        cov,
        sigma_points,
        marker_size=65,
        caption=f'5 sigma points\neach alpha={alpha_spread_each:.4f}',
    )

    ax = axes[1, 1]
    colors = {
        'locked original': '#303030',
        'naive': '#2f7ed8',
        'uncertainty-weighted': '#159a7f',
        'belief-spread': '#b15f00',
        'expected-kernel': '#7b4ab2',
    }
    y_row_used = math.nan
    for label, gp_map in maps.items():
        xs_line, p_line, y_row = _line_at_y(gp_map, float(mean[1]))
        y_row_used = y_row
        mask = (xs_line >= mean[0] - 2.2) & (xs_line <= mean[0] + 2.2)
        ax.plot(xs_line[mask], p_line[mask], label=label, linewidth=2.0, color=colors.get(label))
    ax.axvline(float(mean[0]), color='crimson', linestyle='--', linewidth=1.2)
    ax.set_title(f'Planner trust slice near y={y_row_used:.2f} m', fontsize=11)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('P_conservative_plan_map')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc='best')

    ax = axes[1, 2]
    if validation_rows:
        modes = [str(row['mode']).replace('_', '\n') for row in validation_rows]
        brier = [float(row['brier_mean']) for row in validation_rows]
        x = np.arange(len(modes))
        bar_colors = ['#5b8fd9', '#159a7f', '#b15f00', '#7b4ab2']
        bars = ax.bar(x, brier, width=0.55, color=bar_colors[: len(modes)])
        ax.set_xticks(x)
        ax.set_xticklabels(modes, fontsize=8)
        ax.set_title('Held-out Brier, run-group folds', fontsize=11)
        ax.set_ylabel('Brier score, lower is better')
        ax.grid(True, axis='y', alpha=0.25)
        ymax = max(brier) * 1.18 if brier else 1.0
        ax.set_ylim(0.0, ymax)
        for bar, value in zip(bars, brier):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ymax * 0.025,
                f'{value:.5f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No validation_summary.csv found', ha='center', va='center')

    png_path = output_dir / f'belief_aware_update_showcase_{args.target}.png'
    pdf_path = output_dir / f'belief_aware_update_showcase_{args.target}.pdf'
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary_rows = [
        {'field': 'target', 'value': str(args.target)},
        {'field': 'aggregate_index', 'value': str(idx)},
        {'field': 'event_mean_x_m', 'value': f'{mean[0]:.8f}'},
        {'field': 'event_mean_y_m', 'value': f'{mean[1]:.8f}'},
        {'field': 'target_value', 'value': f'{target_value:.10g}'},
        {'field': 'aggregate_count_capped', 'value': f'{count:.10g}'},
        {'field': 'trace_S_xy', 'value': f'{trace:.10g}'},
        {'field': 'sigma_major_m', 'value': f'{sigma_major:.10g}'},
        {'field': 'sigma_minor_m', 'value': f'{sigma_minor:.10g}'},
        {'field': 'pose_length_scale_m', 'value': f'{pose_length_scale:.10g}'},
        {'field': 'uncertainty_certainty', 'value': f'{certainty:.10g}'},
        {'field': 'alpha_naive', 'value': f'{alpha_naive:.10g}'},
        {'field': 'alpha_uncertainty_weighted', 'value': f'{alpha_weighted:.10g}'},
        {'field': 'alpha_belief_spread_each', 'value': f'{alpha_spread_each:.10g}'},
        {'field': 'relative_precision_naive', 'value': '1'},
        {'field': 'relative_precision_uncertainty_weighted', 'value': f'{certainty:.10g}'},
        {'field': 'relative_precision_belief_spread_total', 'value': '1'},
    ]
    summary_csv = output_dir / f'belief_aware_update_showcase_{args.target}_summary.csv'
    write_csv(summary_csv, ('field', 'value'), summary_rows)
    write_manifest(
        output_dir / f'belief_aware_update_showcase_{args.target}_manifest.json',
        {
            'events_csv': str(events_path),
            'artifact_dir': str(artifact_dir),
            'baseline_gp': str(Path(args.baseline_gp).expanduser().resolve()),
            'target': str(args.target),
            'aggregate_index': int(idx),
            'figure_png': str(png_path),
            'figure_pdf': str(pdf_path),
            'summary_csv': str(summary_csv),
            'selected_event': {
                'mean_xy_m': [float(mean[0]), float(mean[1])],
                'target_value': float(target_value),
                'count_capped': float(count),
                'trace_S_xy': float(trace),
                'sigma_major_m': float(sigma_major),
                'sigma_minor_m': float(sigma_minor),
                'certainty': float(certainty),
            },
            'mechanism': {
                'naive': 'Use belief mean as one observation with alpha = gp_noise_var / count.',
                'uncertainty_weighted': 'Use the same point but multiply observation precision by certainty = exp(-0.5 trace(S) / pose_length_scale^2).',
                'belief_spread': 'Replace one observation with five sigma points carrying weights 0.2 each, conserving total precision while spreading spatial support.',
            },
        },
    )

    print(f'Wrote showcase PNG: {png_path}')
    print(f'Wrote showcase PDF: {pdf_path}')
    print(f'Wrote summary CSV: {summary_csv}')
    print(f'Selected aggregate {idx}: xy=({mean[0]:.3f}, {mean[1]:.3f}), target={target_value:.4f}, certainty={certainty:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
