#!/usr/bin/env python3
"""Plot GP visibility fields and induced ambiguity maps for each method."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from common import CURRENT_GP_DIR, CURRENT_TARGETS_DIR, LOGS_ROOT, parse_float, read_csv_rows, write_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/unav_common',):
    path = str((REPO_ROOT / rel).resolve())
    if path not in __import__('sys').path:
        __import__('sys').path.insert(0, path)

from unav_common.camera_model import ObliqueCameraModel


def _load_artifact(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as data:
        payload: dict[str, np.ndarray | str] = {}
        for key in data.files:
            value = np.asarray(data[key])
            if value.dtype.kind in ('U', 'S') and value.size == 1:
                payload[key] = str(value.reshape(-1)[0])
            else:
                payload[key] = value
        return payload


def _parse_geometry_json(raw: str) -> list[dict[str, float]]:
    text = str(raw or '').strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    prisms = payload.get('prisms', []) if isinstance(payload, dict) else []
    clean = []
    for prism in prisms:
        try:
            clean.append({
                'xmin': float(prism['xmin']),
                'xmax': float(prism['xmax']),
                'ymin': float(prism['ymin']),
                'ymax': float(prism['ymax']),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return clean


def _grid_extent(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    return float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])


def _draw_geometry(ax, prisms: list[dict[str, float]]) -> None:
    for prism in prisms:
        rect = Rectangle(
            (prism['xmin'], prism['ymin']),
            prism['xmax'] - prism['xmin'],
            prism['ymax'] - prism['ymin'],
            facecolor='white',
            edgecolor='black',
            linewidth=1.2,
            alpha=0.30,
        )
        ax.add_patch(rect)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _visibility_effective_score(
    p_vis: np.ndarray,
    *,
    min_prob: float,
    visibility_power: float,
    visibility_trust_low: float,
    visibility_trust_high: float,
) -> np.ndarray:
    p_vis = np.clip(np.asarray(p_vis, dtype=float), min_prob, 1.0 - min_prob)
    shaped = np.clip(p_vis ** float(visibility_power), min_prob, 1.0 - min_prob)
    lo = float(np.clip(visibility_trust_low, min_prob, 1.0 - min_prob))
    hi = float(np.clip(visibility_trust_high, lo + 1e-6, 1.0 - min_prob))
    x = (shaped - lo) / max(hi - lo, 1e-6)
    return np.clip(_smoothstep(x), min_prob, 1.0 - min_prob)


def _blend_observation_covariance(trust: np.ndarray, *, r_visible_uv: float, r_miss_uv: float) -> np.ndarray:
    trust = np.asarray(trust, dtype=float)
    visible_std = float(r_visible_uv)
    miss_std = float(r_miss_uv)
    plan_std = trust * visible_std + (1.0 - trust) * miss_std
    return np.square(plan_std)


def _ambiguity_from_variance(var_u: np.ndarray, var_v: np.ndarray) -> np.ndarray:
    det = np.clip(np.asarray(var_u, dtype=float) * np.asarray(var_v, dtype=float), 1e-12, None)
    return 0.5 * np.log(det)


def _load_target_points(gp_targets_path: Path, method_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not gp_targets_path.is_file():
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    rows = read_csv_rows(gp_targets_path)
    xs, ys, vals = [], [], []
    for row in rows:
        raw = str(row.get(method_id, '')).strip()
        if raw == '':
            continue
        x = parse_float(row.get('x', ''), math.nan)
        y = parse_float(row.get('y', ''), math.nan)
        v = parse_float(raw, math.nan)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(v):
            xs.append(float(x))
            ys.append(float(y))
            vals.append(float(v))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(vals, dtype=float)


def _method_label(method_id: str) -> str:
    return method_id.replace('_', ' ')


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot GP visibility fields and ambiguity maps.')
    parser.add_argument('--gp-dir', default=str(CURRENT_GP_DIR))
    parser.add_argument('--gp-targets', default=str(CURRENT_TARGETS_DIR / 'gp_targets.csv'))
    parser.add_argument('--out', default='')
    parser.add_argument('--r-visible-uv', type=float, default=2.5)
    parser.add_argument('--r-miss-uv', type=float, default=140.0)
    parser.add_argument('--visibility-power', type=float, default=1.0)
    parser.add_argument('--visibility-trust-low', type=float, default=0.08)
    parser.add_argument('--visibility-trust-high', type=float, default=0.30)
    parser.add_argument('--min-prob', type=float, default=1e-4)
    args = parser.parse_args()

    gp_dir = Path(args.gp_dir).expanduser().resolve()
    if not gp_dir.is_dir():
        raise RuntimeError(f'GP directory not found: {gp_dir}')
    output_dir = Path(args.out).expanduser().resolve() if str(args.out).strip() else (gp_dir / 'plots').resolve()
    allowed_root = LOGS_ROOT.resolve()
    if allowed_root not in output_dir.parents and output_dir != allowed_root:
        raise RuntimeError(f'Plot output must stay under {allowed_root}: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = sorted(gp_dir.glob('*_gp.npz'))
    if not artifact_paths:
        raise RuntimeError(f'No GP artifacts found in {gp_dir}')
    gp_targets_path = Path(args.gp_targets).expanduser().resolve()

    combined_rows = []
    plot_rows = []
    for artifact_path in artifact_paths:
        method_id = artifact_path.name.replace('_gp.npz', '')
        artifact = _load_artifact(artifact_path)
        xs = np.asarray(artifact['xs'], dtype=float)
        ys = np.asarray(artifact['ys'], dtype=float)
        p_map = np.asarray(artifact['P_map'], dtype=float)
        extent = _grid_extent(xs, ys)
        geometry = _parse_geometry_json(str(artifact.get('geometry_json', '')))

        p_eff = _visibility_effective_score(
            p_map,
            min_prob=float(args.min_prob),
            visibility_power=float(args.visibility_power),
            visibility_trust_low=float(args.visibility_trust_low),
            visibility_trust_high=float(args.visibility_trust_high),
        )
        plan_var = _blend_observation_covariance(
            p_eff,
            r_visible_uv=float(args.r_visible_uv),
            r_miss_uv=float(args.r_miss_uv),
        )
        observation_ambiguity = _ambiguity_from_variance(plan_var, plan_var)
        diagnostic_ambiguity = observation_ambiguity.copy()

        tx, ty, tv = _load_target_points(gp_targets_path, method_id)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        sample_ax, gp_ax, obs_ax, diag_ax = axes.ravel()

        sample_ax.set_title(f'{_method_label(method_id)} samples')
        if tv.size:
            sc = sample_ax.scatter(tx, ty, c=tv, s=24, cmap='viridis', vmin=0.0, vmax=1.0)
            fig.colorbar(sc, ax=sample_ax, fraction=0.046, pad=0.04, label='target')
        _draw_geometry(sample_ax, geometry)
        sample_ax.set_xlim(extent[0], extent[1])
        sample_ax.set_ylim(extent[2], extent[3])
        sample_ax.set_aspect('equal')

        im_gp = gp_ax.imshow(p_map, origin='lower', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
        _draw_geometry(gp_ax, geometry)
        gp_ax.set_title('Predicted visibility field p_vis(x,y)')
        fig.colorbar(im_gp, ax=gp_ax, fraction=0.046, pad=0.04, label='p_vis')

        im_obs = obs_ax.imshow(observation_ambiguity, origin='lower', extent=extent, cmap='magma', aspect='equal')
        _draw_geometry(obs_ax, geometry)
        obs_ax.set_title('Observation-only ambiguity 0.5 log det R_plan')
        fig.colorbar(im_obs, ax=obs_ax, fraction=0.046, pad=0.04, label='ambiguity')

        im_diag = diag_ax.imshow(diagnostic_ambiguity, origin='lower', extent=extent, cmap='magma', aspect='equal')
        _draw_geometry(diag_ax, geometry)
        diag_ax.set_title('Diagnostic ambiguity with S = I')
        fig.colorbar(im_diag, ax=diag_ax, fraction=0.046, pad=0.04, label='ambiguity')

        for ax in axes.ravel():
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

        method_plot = output_dir / f'{method_id}_maps.png'
        fig.savefig(method_plot, dpi=160)
        plt.close(fig)

        combined_rows.append({
            'method_id': method_id,
            'artifact_path': str(artifact_path),
            'plot_path': str(method_plot),
        })
        plot_rows.append((method_id, p_map, observation_ambiguity, geometry, extent))

    if plot_rows:
        fig, axes = plt.subplots(len(plot_rows), 3, figsize=(15, 4.8 * len(plot_rows)), constrained_layout=True)
        if len(plot_rows) == 1:
            axes = np.asarray([axes])
        for row_axes, (method_id, p_map, ambiguity_map, geometry, extent) in zip(axes, plot_rows):
            tx, ty, tv = _load_target_points(gp_targets_path, method_id)
            ax0, ax1, ax2 = row_axes
            if tv.size:
                ax0.scatter(tx, ty, c=tv, s=20, cmap='viridis', vmin=0.0, vmax=1.0)
            _draw_geometry(ax0, geometry)
            ax0.set_title(f'{_method_label(method_id)} samples')
            ax0.set_xlim(extent[0], extent[1])
            ax0.set_ylim(extent[2], extent[3])
            ax0.set_aspect('equal')

            ax1.imshow(p_map, origin='lower', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
            _draw_geometry(ax1, geometry)
            ax1.set_title('p_vis')

            ax2.imshow(ambiguity_map, origin='lower', extent=extent, cmap='magma', aspect='equal')
            _draw_geometry(ax2, geometry)
            ax2.set_title('ambiguity')

            for ax in row_axes:
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')

        combined_path = output_dir / 'combined_gp_comparison.png'
        fig.savefig(combined_path, dpi=160)
        plt.close(fig)
    else:
        combined_path = output_dir / 'combined_gp_comparison.png'

    write_manifest(output_dir / 'plot_manifest.json', {
        'gp_dir': str(gp_dir),
        'gp_targets_csv': str(gp_targets_path) if gp_targets_path.is_file() else '',
        'method_count': int(len(combined_rows)),
        'method_plots': combined_rows,
        'combined_plot': str(combined_path) if combined_path.is_file() else '',
        'planner_covariance_defaults': {
            'r_visible_uv': float(args.r_visible_uv),
            'r_miss_uv': float(args.r_miss_uv),
            'visibility_power': float(args.visibility_power),
            'visibility_trust_low': float(args.visibility_trust_low),
            'visibility_trust_high': float(args.visibility_trust_high),
        },
        'notes': [
            'Observation-only ambiguity uses the same visibility-to-covariance shaping as the planner.',
            'The S = I diagnostic currently matches the observation-only map under first-order conditioning, but is still plotted explicitly for supervisor-facing comparison.',
        ],
    })
    print(f'Wrote GP/ambiguity plots to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
