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

from common import CURRENT_GP_DIR, CURRENT_TARGETS_DIR, LOGS_ROOT, PLANNER_RUNS_DIR, parse_float, read_csv_rows, write_csv, write_manifest


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


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_run_manifest(run_dir: Path) -> dict:
    for name in ('run_manifest.json', 'manifest.json'):
        payload = _load_json(run_dir / name)
        if payload:
            return payload
    return {}


def _float_from_payload(payload: dict, key: str, default: float) -> tuple[float, bool]:
    raw = payload.get(key, None)
    if raw in (None, ''):
        return float(default), False
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default), False
    return value, math.isfinite(value)


def _plot_settings_for_method(run_manifest: dict, args) -> dict[str, float | str | list[str]]:
    defaults = {
        'r_visible_uv': float(args.r_visible_uv),
        'r_miss_uv': float(args.r_miss_uv),
        'visibility_power': float(args.visibility_power),
        'visibility_trust_low': float(args.visibility_trust_low),
        'visibility_trust_high': float(args.visibility_trust_high),
        'visibility_sigma_kappa': float(getattr(args, 'visibility_sigma_kappa', 1.0)),
        'min_prob': float(args.min_prob),
    }
    used_defaults: list[str] = []
    cfg: dict[str, float | str | list[str]] = {'min_prob': defaults['min_prob']}
    for key in ('r_visible_uv', 'r_miss_uv', 'visibility_power', 'visibility_trust_low', 'visibility_trust_high', 'visibility_sigma_kappa'):
        value, ok = _float_from_payload(run_manifest, key, defaults[key])
        if not ok:
            used_defaults.append(key)
        cfg[key] = value
    cfg['source'] = 'run_manifest' if not used_defaults else 'run_manifest+args_fallback'
    cfg['used_arg_defaults'] = used_defaults
    return cfg


def _infer_method_id(run_manifest: dict, run_dir: Path) -> str:
    method = str(run_manifest.get('method', '') or '').strip()
    if method:
        return method
    visibility_artifact = str(run_manifest.get('visibility_artifact_path', '') or '').strip()
    if visibility_artifact:
        name = Path(visibility_artifact).name
        if name.endswith('_gp.npz'):
            return name[:-7]
    return run_dir.parent.name


def _find_latest_run_manifests(root: Path) -> dict[str, dict]:
    latest: dict[str, tuple[float, dict]] = {}
    if not root.is_dir():
        return {}
    for summary_path in root.rglob('run_summary.json'):
        run_dir = summary_path.parent
        run_manifest = _load_run_manifest(run_dir)
        if not run_manifest:
            continue
        method_id = _infer_method_id(run_manifest, run_dir)
        mtime = run_dir.stat().st_mtime
        current = latest.get(method_id)
        if current is None or mtime > current[0]:
            latest[method_id] = (mtime, run_manifest)
    return {method_id: manifest for method_id, (_, manifest) in latest.items()}


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
    parser.add_argument('--planner-runs-root', default=str(PLANNER_RUNS_DIR))
    parser.add_argument('--out', default='')
    parser.add_argument('--r-visible-uv', type=float, default=2.5)
    parser.add_argument('--r-miss-uv', type=float, default=140.0)
    parser.add_argument('--visibility-power', type=float, default=1.0)
    parser.add_argument('--visibility-trust-low', type=float, default=0.08)
    parser.add_argument('--visibility-trust-high', type=float, default=0.30)
    parser.add_argument('--visibility-sigma-kappa', type=float, default=1.0)
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
    latest_run_manifests = _find_latest_run_manifests(Path(args.planner_runs_root).expanduser().resolve())

    combined_rows = []
    plot_rows = []
    summary_rows = []
    plot_notes = ['Observation-only ambiguity uses the same visibility-to-covariance shaping as the planner.']
    for artifact_path in artifact_paths:
        method_id = artifact_path.name.replace('_gp.npz', '')
        artifact = _load_artifact(artifact_path)
        xs = np.asarray(artifact['xs'], dtype=float)
        ys = np.asarray(artifact['ys'], dtype=float)
        p_map = np.asarray(artifact.get('P_map', artifact.get('P_conservative_map', np.zeros(1))), dtype=float)
        p_mean = np.asarray(artifact.get('P_mean_map', p_map), dtype=float)
        p_unc = np.clip(p_mean - p_map, 0.0, None)
        extent = _grid_extent(xs, ys)
        geometry = _parse_geometry_json(str(artifact.get('geometry_json', '')))

        plot_cfg = _plot_settings_for_method(latest_run_manifests.get(method_id, {}), args)
        if plot_cfg['used_arg_defaults']:
            plot_notes.append(
                f"{method_id}: GP plot settings fell back to CLI defaults for {', '.join(plot_cfg['used_arg_defaults'])}."
            )

        p_eff = _visibility_effective_score(
            p_map,
            min_prob=float(plot_cfg['min_prob']),
            visibility_power=float(plot_cfg['visibility_power']),
            visibility_trust_low=float(plot_cfg['visibility_trust_low']),
            visibility_trust_high=float(plot_cfg['visibility_trust_high']),
        )
        plan_var = _blend_observation_covariance(
            p_eff,
            r_visible_uv=float(plot_cfg['r_visible_uv']),
            r_miss_uv=float(plot_cfg['r_miss_uv']),
        )
        observation_ambiguity = _ambiguity_from_variance(plan_var, plan_var)
        r_plan_uv_std = np.sqrt(np.clip(plan_var, 1e-12, None))

        summary_rows.append({
            'method_id': method_id,
            'p_vis_mean': float(np.mean(p_map)),
            'p_vis_min': float(np.min(p_map)),
            'p_vis_max': float(np.max(p_map)),
            'ambiguity_mean': float(np.mean(observation_ambiguity)),
            'ambiguity_min': float(np.min(observation_ambiguity)),
            'ambiguity_max': float(np.max(observation_ambiguity)),
            'r_plan_uv_std_mean': float(np.mean(r_plan_uv_std)),
            'r_plan_uv_std_min': float(np.min(r_plan_uv_std)),
            'r_plan_uv_std_max': float(np.max(r_plan_uv_std)),
            'plot_settings_source': str(plot_cfg['source']),
        })

        tx, ty, tv = _load_target_points(gp_targets_path, method_id)

        fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
        sample_ax, mean_ax, unc_ax, gp_ax, obs_ax, std_ax = axes.ravel()

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
        gp_ax.set_title('Conservative visibility P_map(x,y)')
        fig.colorbar(im_gp, ax=gp_ax, fraction=0.046, pad=0.04, label='p_vis (conservative)')
        
        im_mean = mean_ax.imshow(p_mean, origin='lower', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
        _draw_geometry(mean_ax, geometry)
        mean_ax.set_title('Predicted mean visibility P_mean_map')
        fig.colorbar(im_mean, ax=mean_ax, fraction=0.046, pad=0.04, label='p_vis (mean)')
        
        im_unc = unc_ax.imshow(p_unc, origin='lower', extent=extent, cmap='plasma', vmin=0.0, vmax=0.3, aspect='equal')
        _draw_geometry(unc_ax, geometry)
        unc_ax.set_title('GP Epistemic Uncertainty (Mean - Conservative)')
        fig.colorbar(im_unc, ax=unc_ax, fraction=0.046, pad=0.04, label='uncertainty')

        im_obs = obs_ax.imshow(observation_ambiguity, origin='lower', extent=extent, cmap='magma', aspect='equal')
        _draw_geometry(obs_ax, geometry)
        obs_ax.set_title('Observation-only ambiguity 0.5 log det R_plan')
        fig.colorbar(im_obs, ax=obs_ax, fraction=0.046, pad=0.04, label='ambiguity')

        im_std = std_ax.imshow(r_plan_uv_std, origin='lower', extent=extent, cmap='magma_r', aspect='equal')
        _draw_geometry(std_ax, geometry)
        std_ax.set_title('Planner observation noise proxy r_plan_uv_std')
        fig.colorbar(im_std, ax=std_ax, fraction=0.046, pad=0.04, label='r_plan_uv_std')

        for ax in axes.ravel():
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

        method_plot = output_dir / f'{method_id}_maps.png'
        fig.savefig(method_plot, dpi=160)
        plt.close(fig)

        region_fig, region_ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        region_im = region_ax.imshow(
            observation_ambiguity, origin='lower', extent=extent, cmap='magma', aspect='equal'
        )
        _draw_geometry(region_ax, geometry)
        if tv.size:
            region_ax.scatter(tx, ty, c='white', s=12, alpha=0.75, edgecolors='none', label='samples')
        finite_amb = observation_ambiguity[np.isfinite(observation_ambiguity)]
        contour_levels = []
        contour_labels = []
        for q, label in ((75.0, '75th pct'), (90.0, '90th pct')):
            if finite_amb.size:
                level = float(np.nanpercentile(finite_amb, q))
                contour_levels.append(level)
                contour_labels.append(label)
        if contour_levels:
            contours = region_ax.contour(
                observation_ambiguity,
                levels=contour_levels,
                origin='lower',
                extent=extent,
                colors=['cyan', 'yellow'][: len(contour_levels)],
                linewidths=1.8,
            )
            fmt = {lvl: lbl for lvl, lbl in zip(contours.levels, contour_labels)}
            region_ax.clabel(contours, inline=True, fontsize=8, fmt=fmt)
        region_ax.set_title(f'{_method_label(method_id)} ambiguity regions')
        region_ax.set_xlabel('x [m]')
        region_ax.set_ylabel('y [m]')
        region_fig.colorbar(region_im, ax=region_ax, fraction=0.046, pad=0.04, label='ambiguity')
        if tv.size:
            region_ax.legend(loc='upper right', fontsize=8)
        region_plot = output_dir / f'ambiguity_regions_{method_id}.png'
        region_fig.savefig(region_plot, dpi=160)
        plt.close(region_fig)

        combined_rows.append({
            'method_id': method_id,
            'artifact_path': str(artifact_path),
            'plot_path': str(method_plot),
            'ambiguity_regions_plot': str(region_plot),
            'plot_settings': {
                'r_visible_uv': float(plot_cfg['r_visible_uv']),
                'r_miss_uv': float(plot_cfg['r_miss_uv']),
                'visibility_power': float(plot_cfg['visibility_power']),
                'visibility_trust_low': float(plot_cfg['visibility_trust_low']),
                'visibility_trust_high': float(plot_cfg['visibility_trust_high']),
                'visibility_sigma_kappa': float(plot_cfg['visibility_sigma_kappa']),
                'source': str(plot_cfg['source']),
                'used_arg_defaults': list(plot_cfg['used_arg_defaults']),
            },
        })
        plot_rows.append((method_id, p_map, observation_ambiguity, r_plan_uv_std, geometry, extent))

    if plot_rows:
        fig, axes = plt.subplots(len(plot_rows), 4, figsize=(20, 4.8 * len(plot_rows)), constrained_layout=True)
        if len(plot_rows) == 1:
            axes = np.asarray([axes])
        for row_axes, (method_id, p_map, ambiguity_map, std_map, geometry, extent) in zip(axes, plot_rows):
            tx, ty, tv = _load_target_points(gp_targets_path, method_id)
            ax0, ax1, ax2, ax3 = row_axes
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

            ax3.imshow(std_map, origin='lower', extent=extent, cmap='magma_r', aspect='equal')
            _draw_geometry(ax3, geometry)
            ax3.set_title('r_plan_uv_std')

            for ax in row_axes:
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')

        combined_path = output_dir / 'combined_gp_comparison.png'
        fig.savefig(combined_path, dpi=160)
        plt.close(fig)
    else:
        combined_path = output_dir / 'combined_gp_comparison.png'

    write_csv(
        output_dir / 'field_method_summary.csv',
        ('method_id', 'p_vis_mean', 'p_vis_min', 'p_vis_max', 'ambiguity_mean', 'ambiguity_min', 'ambiguity_max', 'r_plan_uv_std_mean', 'r_plan_uv_std_min', 'r_plan_uv_std_max', 'plot_settings_source'),
        summary_rows
    )

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
            'visibility_sigma_kappa': float(args.visibility_sigma_kappa),
        },
        'notes': plot_notes,
    })
    print(f'Wrote GP/ambiguity plots to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
