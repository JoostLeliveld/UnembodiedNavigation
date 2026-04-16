#!/usr/bin/env python3
"""Plot planner runs over shared GP visibility and ambiguity backgrounds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from common import ACTIVE_METHOD_IDS, CURRENT_GP_DIR, LOGS_ROOT, PLANNER_RUNS_DIR, REPORT_DIR, write_manifest


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    columns: dict[str, list[float]] = {name: [] for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            raw = str(row.get(name, '') or '').strip()
            try:
                columns[name].append(float(raw))
            except ValueError:
                columns[name].append(math.nan)
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _load_plan_groups(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        return []
    groups: dict[float, list[tuple[float, float]]] = {}
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                stamp = float(row['plan_stamp'])
                x = float(row['x'])
                y = float(row['y'])
            except (KeyError, TypeError, ValueError):
                continue
            groups.setdefault(stamp, []).append((x, y))
    out = []
    for stamp in sorted(groups.keys()):
        pts = np.asarray(groups[stamp], dtype=float)
        if pts.shape[0] >= 2:
            out.append(pts)
    return out


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


def _visibility_effective_score(p_vis: np.ndarray, *, min_prob: float, visibility_power: float, visibility_trust_low: float, visibility_trust_high: float) -> np.ndarray:
    p_vis = np.clip(np.asarray(p_vis, dtype=float), min_prob, 1.0 - min_prob)
    shaped = np.clip(p_vis ** float(visibility_power), min_prob, 1.0 - min_prob)
    lo = float(np.clip(visibility_trust_low, min_prob, 1.0 - min_prob))
    hi = float(np.clip(visibility_trust_high, lo + 1e-6, 1.0 - min_prob))
    x = (shaped - lo) / max(hi - lo, 1e-6)
    return np.clip(_smoothstep(x), min_prob, 1.0 - min_prob)


def _ambiguity_map(p_map: np.ndarray, *, min_prob: float, r_visible_uv: float, r_miss_uv: float, visibility_power: float, visibility_trust_low: float, visibility_trust_high: float) -> np.ndarray:
    trust = _visibility_effective_score(
        p_map,
        min_prob=min_prob,
        visibility_power=visibility_power,
        visibility_trust_low=visibility_trust_low,
        visibility_trust_high=visibility_trust_high,
    )
    std = trust * float(r_visible_uv) + (1.0 - trust) * float(r_miss_uv)
    det = np.clip(np.square(std) * np.square(std), 1e-12, None)
    return 0.5 * np.log(det)


def _infer_method_id(run_manifest: dict, run_dir: Path) -> str:
    planner = str(run_manifest.get('planner', '') or '').strip()
    if planner == 'visibility_unaware_baseline':
        return 'visibility_unaware_baseline'

    raw_artifact = str(run_manifest.get('visibility_artifact_path', '') or '').strip()
    if raw_artifact:
        name = Path(raw_artifact).name
        if name.endswith('_gp.npz'):
            return name[:-7]

    parent_name = run_dir.parent.name
    if parent_name in ACTIVE_METHOD_IDS:
        return parent_name
    return planner or run_dir.name


def _find_latest_runs(root: Path) -> dict[str, Path]:
    run_dirs = []
    for experiment_csv in root.rglob('experiment.csv'):
        run_dir = experiment_csv.parent
        manifest = _load_json(run_dir / 'manifest.json')
        method_id = _infer_method_id(manifest, run_dir)
        run_dirs.append((method_id, run_dir))

    latest: dict[str, Path] = {}
    for method_id, run_dir in run_dirs:
        current = latest.get(method_id)
        if current is None or run_dir.stat().st_mtime > current.stat().st_mtime:
            latest[method_id] = run_dir
    return latest


def _trajectory_from_cols(cols: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(cols.get('x', np.array([], dtype=float)), dtype=float)
    y = np.asarray(cols.get('y', np.array([], dtype=float)), dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.zeros((0, 2), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return np.column_stack([x[mask], y[mask]]) if np.any(mask) else np.zeros((0, 2), dtype=float)


def _goal_from_cols(cols: dict[str, np.ndarray]) -> tuple[float, float] | None:
    gx = np.asarray(cols.get('goal_x', np.array([], dtype=float)), dtype=float)
    gy = np.asarray(cols.get('goal_y', np.array([], dtype=float)), dtype=float)
    mask = np.isfinite(gx) & np.isfinite(gy)
    if not np.any(mask):
        return None
    idx = int(np.flatnonzero(mask)[-1])
    return float(gx[idx]), float(gy[idx])


def _method_label(method_id: str) -> str:
    return method_id.replace('_', ' ')


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot planned and realized trajectories over GP fields.')
    parser.add_argument('--planner-runs-root', default=str(PLANNER_RUNS_DIR))
    parser.add_argument('--gp-dir', default=str(CURRENT_GP_DIR))
    parser.add_argument('--out', default=str(REPORT_DIR / 'path_plots'))
    parser.add_argument('--background-method', default='oracle_visibility')
    parser.add_argument('--r-visible-uv', type=float, default=2.5)
    parser.add_argument('--r-miss-uv', type=float, default=140.0)
    parser.add_argument('--visibility-power', type=float, default=1.0)
    parser.add_argument('--visibility-trust-low', type=float, default=0.08)
    parser.add_argument('--visibility-trust-high', type=float, default=0.30)
    parser.add_argument('--min-prob', type=float, default=1e-4)
    args = parser.parse_args()

    planner_runs_root = Path(args.planner_runs_root).expanduser().resolve()
    gp_dir = Path(args.gp_dir).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    allowed_root = LOGS_ROOT.resolve()
    if allowed_root not in output_dir.parents and output_dir != allowed_root:
        raise RuntimeError(f'Path-plot output must stay under {allowed_root}: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_runs = _find_latest_runs(planner_runs_root) if planner_runs_root.is_dir() else {}
    if not latest_runs:
        write_manifest(output_dir / 'plot_manifest.json', {
            'planner_runs_root': str(planner_runs_root),
            'gp_dir': str(gp_dir),
            'available_methods': [],
            'notes': ['No planner runs were found; path plotting completed as a no-op.'],
        })
        print(f'No planner runs found under {planner_runs_root}')
        return 0

    artifacts: dict[str, Path] = {}
    for path in sorted(gp_dir.glob('*_gp.npz')):
        artifacts[path.name[:-7]] = path
    background_method = str(args.background_method).strip()
    background_artifact = artifacts.get(background_method) or next(iter(artifacts.values()), None)

    method_entries = []
    combined_paths = []
    for method_id, run_dir in sorted(latest_runs.items()):
        run_cols = _read_csv_columns(run_dir / 'experiment.csv')
        realized = _trajectory_from_cols(run_cols)
        if realized.shape[0] == 0:
            continue
        latest_plan = _load_plan_groups(run_dir / 'plan_samples.csv')
        latest_plan_pts = latest_plan[-1] if latest_plan else np.zeros((0, 2), dtype=float)
        goal_xy = _goal_from_cols(run_cols)

        artifact_path = artifacts.get(method_id)
        if artifact_path is None:
            run_manifest = _load_json(run_dir / 'manifest.json')
            raw_artifact = str(run_manifest.get('visibility_artifact_path', '') or '').strip()
            if raw_artifact:
                candidate = Path(raw_artifact).expanduser().resolve()
                if candidate.is_file():
                    artifact_path = candidate
        if artifact_path is None:
            artifact_path = background_artifact
        if artifact_path is None or not artifact_path.is_file():
            continue

        artifact = _load_artifact(artifact_path)
        xs = np.asarray(artifact['xs'], dtype=float)
        ys = np.asarray(artifact['ys'], dtype=float)
        p_map = np.asarray(artifact['P_map'], dtype=float)
        ambiguity_map = _ambiguity_map(
            p_map,
            min_prob=float(args.min_prob),
            r_visible_uv=float(args.r_visible_uv),
            r_miss_uv=float(args.r_miss_uv),
            visibility_power=float(args.visibility_power),
            visibility_trust_low=float(args.visibility_trust_low),
            visibility_trust_high=float(args.visibility_trust_high),
        )
        geometry = _parse_geometry_json(str(artifact.get('geometry_json', '')))
        extent = _grid_extent(xs, ys)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        for ax, grid, title, cmap in (
            (axes[0], p_map, 'Trajectory over visibility field', 'viridis'),
            (axes[1], ambiguity_map, 'Trajectory over ambiguity map', 'magma'),
        ):
            ax.imshow(grid, origin='lower', extent=extent, cmap=cmap, aspect='equal')
            _draw_geometry(ax, geometry)
            ax.plot(realized[:, 0], realized[:, 1], color='deepskyblue', linewidth=2.2, label='realized')
            ax.scatter(realized[0, 0], realized[0, 1], c='lime', s=50, marker='o', label='start')
            if goal_xy is not None:
                ax.scatter(goal_xy[0], goal_xy[1], c='gold', s=70, marker='*', label='goal')
            if latest_plan_pts.shape[0] >= 2:
                ax.plot(latest_plan_pts[:, 0], latest_plan_pts[:, 1], color='white', linestyle='--', linewidth=1.5, label='latest plan')
            ax.set_title(f'{_method_label(method_id)}: {title}')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
        axes[0].legend(loc='upper right', fontsize=8)
        method_plot = output_dir / f'{method_id}_paths.png'
        fig.savefig(method_plot, dpi=160)
        plt.close(fig)

        method_entries.append({
            'method_id': method_id,
            'run_dir': str(run_dir),
            'artifact_path': str(artifact_path),
            'plot_path': str(method_plot),
        })
        combined_paths.append((method_id, realized, goal_xy))

    if background_artifact is not None and combined_paths:
        artifact = _load_artifact(background_artifact)
        xs = np.asarray(artifact['xs'], dtype=float)
        ys = np.asarray(artifact['ys'], dtype=float)
        p_map = np.asarray(artifact['P_map'], dtype=float)
        geometry = _parse_geometry_json(str(artifact.get('geometry_json', '')))
        extent = _grid_extent(xs, ys)

        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax.imshow(p_map, origin='lower', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
        _draw_geometry(ax, geometry)
        palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(combined_paths), 1)))
        for color, (method_id, realized, goal_xy) in zip(palette, combined_paths):
            ax.plot(realized[:, 0], realized[:, 1], linewidth=2.0, color=color, label=_method_label(method_id))
            ax.scatter(realized[0, 0], realized[0, 1], color=color, s=28)
            if goal_xy is not None:
                ax.scatter(goal_xy[0], goal_xy[1], color=color, s=60, marker='*')
        ax.set_title(f'All method paths over {args.background_method} field')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.legend(loc='upper right', fontsize=8)
        combined_path = output_dir / 'combined_method_paths.png'
        fig.savefig(combined_path, dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(len(combined_paths), 1, figsize=(8, 4.5 * len(combined_paths)), constrained_layout=True)
        if len(combined_paths) == 1:
            axes = np.asarray([axes])
        ambiguity_map = _ambiguity_map(
            p_map,
            min_prob=float(args.min_prob),
            r_visible_uv=float(args.r_visible_uv),
            r_miss_uv=float(args.r_miss_uv),
            visibility_power=float(args.visibility_power),
            visibility_trust_low=float(args.visibility_trust_low),
            visibility_trust_high=float(args.visibility_trust_high),
        )
        for ax, (method_id, realized, goal_xy) in zip(axes, combined_paths):
            ax.imshow(ambiguity_map, origin='lower', extent=extent, cmap='magma', aspect='equal')
            _draw_geometry(ax, geometry)
            ax.plot(realized[:, 0], realized[:, 1], color='deepskyblue', linewidth=2.2)
            ax.scatter(realized[0, 0], realized[0, 1], c='lime', s=40)
            if goal_xy is not None:
                ax.scatter(goal_xy[0], goal_xy[1], c='gold', s=70, marker='*')
            ax.set_title(f'{_method_label(method_id)} over common ambiguity background')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
        ambiguity_compare_path = output_dir / 'combined_ambiguity_paths.png'
        fig.savefig(ambiguity_compare_path, dpi=160)
        plt.close(fig)
    else:
        combined_path = output_dir / 'combined_method_paths.png'
        ambiguity_compare_path = output_dir / 'combined_ambiguity_paths.png'

    write_manifest(output_dir / 'plot_manifest.json', {
        'planner_runs_root': str(planner_runs_root),
        'gp_dir': str(gp_dir),
        'background_method': background_method,
        'available_methods': [entry['method_id'] for entry in method_entries],
        'method_entries': method_entries,
        'combined_path_plot': str(combined_path) if combined_path.is_file() else '',
        'combined_ambiguity_plot': str(ambiguity_compare_path) if ambiguity_compare_path.is_file() else '',
        'notes': [
            'This script selects the latest run directory per method under logs/visibility_comparison/planner_runs.',
            'The baseline method is plotted over the chosen common background field because it has no visibility artifact of its own.',
        ],
    })
    print(f'Wrote planned-path plots to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
