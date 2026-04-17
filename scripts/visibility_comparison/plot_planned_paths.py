#!/usr/bin/env python3
"""Plot planner runs over shared GP visibility and ambiguity backgrounds with temporal alignment."""

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

from common import ACTIVE_METHOD_IDS, CURRENT_GP_DIR, LOGS_ROOT, PLANNER_RUNS_DIR, REPORT_DIR, write_csv, write_manifest


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


def _r_plan_uv_std_map(p_map: np.ndarray, *, min_prob: float, r_visible_uv: float, r_miss_uv: float, visibility_power: float, visibility_trust_low: float, visibility_trust_high: float) -> np.ndarray:
    trust = _visibility_effective_score(
        p_map,
        min_prob=min_prob,
        visibility_power=visibility_power,
        visibility_trust_low=visibility_trust_low,
        visibility_trust_high=visibility_trust_high,
    )
    return trust * float(r_visible_uv) + (1.0 - trust) * float(r_miss_uv)


def _infer_method_id(run_manifest: dict, run_dir: Path) -> str:
    planner = str(run_manifest.get('planner', '') or '').strip()
    if planner == 'visibility_unaware_baseline':
        return 'visibility_unaware_baseline'
    
    comp_id = str(run_manifest.get('comparison_method_id', '') or '').strip()
    if comp_id and comp_id in ACTIVE_METHOD_IDS:
        return comp_id

    raw_artifact = str(run_manifest.get('visibility_artifact_path', '') or '').strip()
    if raw_artifact:
        name = Path(raw_artifact).name
        if name.endswith('_gp.npz'):
            return name[:-7]

    parent_name = run_dir.parent.name
    if parent_name in ACTIVE_METHOD_IDS:
        return parent_name
    return planner or run_dir.name


def _find_latest_completed_runs(root: Path) -> dict[str, Path]:
    run_dirs = []
    for summary_path in root.rglob('run_summary.json'):
        run_dir = summary_path.parent
        summary = _load_json(summary_path)
        if not summary.get('completed', False):
            continue
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


def _time_align_trajectory(cols: dict[str, np.ndarray], first_cmd_stamp: float) -> np.ndarray:
    stamps = np.asarray(cols.get('stamp', np.array([], dtype=float)), dtype=float)
    if first_cmd_stamp and math.isfinite(first_cmd_stamp):
        return stamps - first_cmd_stamp
    return stamps


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


def _sample_grid_along_path(grid: np.ndarray, extent: tuple[float, float, float, float], path: np.ndarray) -> np.ndarray:
    if path.shape[0] == 0:
        return np.array([], dtype=float)
    xmin, xmax, ymin, ymax = extent
    ny, nx = grid.shape
    dx = (xmax - xmin) / max(nx - 1, 1)
    dy = (ymax - ymin) / max(ny - 1, 1)
    
    samples = []
    for x, y in path:
        idx_x = int(round((x - xmin) / dx))
        idx_y = int(round((y - ymin) / dy))
        idx_x = np.clip(idx_x, 0, nx - 1)
        idx_y = np.clip(idx_y, 0, ny - 1)
        samples.append(grid[idx_y, idx_x])
    return np.asarray(samples, dtype=float)


def _cumulative_path_distance(path: np.ndarray) -> np.ndarray:
    if path.shape[0] <= 1:
        return np.zeros(path.shape[0], dtype=float)
    diffs = np.diff(path, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.concatenate(([0.0], np.cumsum(dists)))


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot planned and realized trajectories over GP fields with temporal analysis.')
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

    latest_runs = _find_latest_completed_runs(planner_runs_root) if planner_runs_root.is_dir() else {}
    if not latest_runs:
        write_manifest(output_dir / 'plot_manifest.json', {
            'planner_runs_root': str(planner_runs_root),
            'gp_dir': str(gp_dir),
            'available_methods': [],
            'notes': ['No valid completed planner runs were found; path plotting completed as a no-op.'],
        })
        print(f'No completed planner runs found under {planner_runs_root}')
        return 0

    artifacts: dict[str, Path] = {}
    for path in sorted(gp_dir.glob('*_gp.npz')):
        artifacts[path.name[:-7]] = path
    background_method = str(args.background_method).strip()
    background_artifact = artifacts.get(background_method) or next(iter(artifacts.values()), None)

    method_entries = []
    combined_ts = []
    combined_profiles = []
    planner_summaries = []

    for method_id, run_dir in sorted(latest_runs.items()):
        run_cols = _read_csv_columns(run_dir / 'experiment.csv')
        run_summary = _load_json(run_dir / 'run_summary.json')
        
        # Add to summary CSV rows
        planner_summaries.append({
            'method_id': method_id,
            'completed': run_summary.get('completed', False),
            'completion_reason': run_summary.get('completion_reason', ''),
            'elapsed_after_first_cmd_s': run_summary.get('elapsed_after_first_cmd_s', ''),
            'path_length_m': run_summary.get('path_length_m', ''),
            'final_goal_distance': run_summary.get('final_goal_distance', ''),
            'minimum_goal_distance': run_summary.get('minimum_goal_distance', ''),
            'mean_solve_time_ms': run_summary.get('mean_solve_time_ms', ''),
            'mean_efe_risk': run_summary.get('mean_efe_risk', ''),
            'mean_efe_ambiguity': run_summary.get('mean_efe_ambiguity', ''),
            'mean_p_vis_plan': run_summary.get('mean_p_vis_plan', ''),
            'mean_p_vis_plan_eff': run_summary.get('mean_p_vis_plan_eff', ''),
            'mean_r_plan_u_std': run_summary.get('mean_r_plan_u_std', ''),
            'run_dir': str(run_dir.resolve())
        })

        realized = _trajectory_from_cols(run_cols)
        if realized.shape[0] == 0:
            continue
        latest_plan = _load_plan_groups(run_dir / 'plan_samples.csv')
        latest_plan_pts = latest_plan[-1] if latest_plan else np.zeros((0, 2), dtype=float)
        goal_xy = _goal_from_cols(run_cols)

        first_cmd_stamp = float(run_summary.get('first_cmd_stamp', math.nan))
        t_aligned = _time_align_trajectory(run_cols, first_cmd_stamp)
        
        # Time-series values
        gd = np.asarray(run_cols.get('goal_dist', []), dtype=float)
        efe_risk = np.asarray(run_cols.get('efe_risk', []), dtype=float)
        efe_ambiguity = np.asarray(run_cols.get('efe_ambiguity', []), dtype=float)
        p_vis_plan = np.asarray(run_cols.get('p_vis_plan', []), dtype=float)
        p_vis_plan_eff = np.asarray(run_cols.get('p_vis_plan_eff', []), dtype=float)
        r_plan_u_std = np.asarray(run_cols.get('r_plan_u_std', []), dtype=float)

        combined_ts.append((method_id, t_aligned, gd, efe_risk, efe_ambiguity, p_vis_plan, p_vis_plan_eff, r_plan_u_std, realized, goal_xy))

        artifact_path = artifacts.get(method_id)
        if artifact_path is None:
            raw_artifact = str(run_summary.get('visibility_artifact_path', '') or '').strip()
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
        extent = _grid_extent(xs, ys)
        geometry = _parse_geometry_json(str(artifact.get('geometry_json', '')))
        
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
        std_map = _r_plan_uv_std_map(
            p_map,
            min_prob=float(args.min_prob),
            r_visible_uv=float(args.r_visible_uv),
            r_miss_uv=float(args.r_miss_uv),
            visibility_power=float(args.visibility_power),
            visibility_trust_low=float(args.visibility_trust_low),
            visibility_trust_high=float(args.visibility_trust_high),
        )

        # Build analysis sheet
        fig = plt.figure(figsize=(24, 12), constrained_layout=True)
        fig.suptitle(f'{_method_label(method_id)} Analysis Sheet', fontsize=16)
        gs = fig.add_gridspec(2, 3)
        ax_pvis = fig.add_subplot(gs[0, 0])
        ax_amb = fig.add_subplot(gs[0, 1])
        ax_bg = fig.add_subplot(gs[0, 2])
        ax_gd = fig.add_subplot(gs[1, 0])
        ax_efe = fig.add_subplot(gs[1, 1])
        ax_vis = fig.add_subplot(gs[1, 2])

        # Top row: spatial maps
        for ax, grid, title, cmap in (
            (ax_pvis, p_map, 'Trajectory over visibility field', 'viridis'),
            (ax_amb, ambiguity_map, 'Trajectory over ambiguity map', 'magma'),
        ):
            ax.imshow(grid, origin='lower', extent=extent, cmap=cmap, aspect='equal')
            _draw_geometry(ax, geometry)
            ax.plot(realized[:, 0], realized[:, 1], color='deepskyblue', linewidth=2.2, label='realized')
            ax.scatter(realized[0, 0], realized[0, 1], c='lime', s=50, marker='o', label='start')
            if goal_xy is not None:
                ax.scatter(goal_xy[0], goal_xy[1], c='gold', s=70, marker='*', label='goal')
            if latest_plan_pts.shape[0] >= 2:
                ax.plot(latest_plan_pts[:, 0], latest_plan_pts[:, 1], color='white', linestyle='--', linewidth=1.5, label='latest plan')
            ax.set_title(title)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
        ax_pvis.legend(loc='upper right', fontsize=8)

        if background_artifact and background_artifact.is_file():
            bg_art = _load_artifact(background_artifact)
            bg_p = np.asarray(bg_art['P_map'], dtype=float)
            bg_extent = _grid_extent(np.asarray(bg_art['xs'], dtype=float), np.asarray(bg_art['ys'], dtype=float))
            ax_bg.imshow(bg_p, origin='lower', extent=bg_extent, cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
            _draw_geometry(ax_bg, _parse_geometry_json(str(bg_art.get('geometry_json', ''))))
        
        ax_bg.plot(realized[:, 0], realized[:, 1], color='deepskyblue', linewidth=2.2)
        ax_bg.scatter(realized[0, 0], realized[0, 1], c='lime', s=50, marker='o')
        if goal_xy is not None:
            ax_bg.scatter(goal_xy[0], goal_xy[1], c='gold', s=70, marker='*')
        ax_bg.set_title(f'Trajectory over {args.background_method} (reference)')
        ax_bg.set_xlabel('x [m]')
        ax_bg.set_ylabel('y [m]')

        # Bottom row: temporal graphs
        mask_t = (t_aligned >= 0.0)
        t_plot = t_aligned[mask_t]

        ax_gd.plot(t_plot, gd[mask_t], linewidth=2, color='dodgerblue')
        ax_gd.set_title('Goal Distance over Time')
        ax_gd.set_xlabel('Time since first command [s]')
        ax_gd.set_ylabel('Distance [m]')
        ax_gd.grid(True)

        ax_efe.plot(t_plot, efe_risk[mask_t], label='efe_risk', linewidth=2, color='crimson')
        ax_efe.plot(t_plot, efe_ambiguity[mask_t], label='efe_ambiguity', linewidth=2, color='darkorange')
        ax_efe.set_title('Planner Objectives over Time')
        ax_efe.set_xlabel('Time since first command [s]')
        ax_efe.set_ylabel('Objective Value')
        ax_efe.legend()
        ax_efe.grid(True)

        ax_vis.plot(t_plot, p_vis_plan[mask_t], label='p_vis_plan', linewidth=2, color='forestgreen')
        ax_vis.plot(t_plot, p_vis_plan_eff[mask_t], label='p_vis_plan_eff', linewidth=2, color='mediumseagreen', linestyle='--')
        ax_vis2 = ax_vis.twinx()
        ax_vis2.plot(t_plot, r_plan_u_std[mask_t], label='r_plan_u_std', linewidth=2, color='purple', alpha=0.7)
        ax_vis.set_title('Visibility and Noise Proxies over Time')
        ax_vis.set_xlabel('Time since first command [s]')
        ax_vis.set_ylabel('Probability')
        ax_vis2.set_ylabel('r_plan_u_std [px/m]')
        lines_1, labels_1 = ax_vis.get_legend_handles_labels()
        lines_2, labels_2 = ax_vis2.get_legend_handles_labels()
        ax_vis.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax_vis.grid(True)

        method_plot = output_dir / f'{method_id}_analysis.png'
        fig.savefig(method_plot, dpi=160)
        plt.close(fig)

        method_entries.append({
            'method_id': method_id,
            'run_dir': str(run_dir),
            'artifact_path': str(artifact_path),
            'plot_path': str(method_plot),
        })

        # Calculate profile metrics
        cum_dist = _cumulative_path_distance(realized)
        p_vis_profile = _sample_grid_along_path(p_map, extent, realized)
        amb_profile = _sample_grid_along_path(ambiguity_map, extent, realized)
        std_profile = _sample_grid_along_path(std_map, extent, realized)
        combined_profiles.append((method_id, cum_dist, p_vis_profile, amb_profile, std_profile))

    write_csv(
        output_dir / 'planner_method_summary.csv',
        (
            'method_id', 'completed', 'completion_reason', 'elapsed_after_first_cmd_s', 'path_length_m',
            'final_goal_distance', 'minimum_goal_distance', 'mean_solve_time_ms', 'mean_efe_risk',
            'mean_efe_ambiguity', 'mean_p_vis_plan', 'mean_p_vis_plan_eff', 'mean_r_plan_u_std', 'run_dir'
        ),
        planner_summaries
    )

    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(combined_ts), 1)))

    if background_artifact is not None and combined_ts:
        artifact = _load_artifact(background_artifact)
        xs = np.asarray(artifact['xs'], dtype=float)
        ys = np.asarray(artifact['ys'], dtype=float)
        p_map = np.asarray(artifact['P_map'], dtype=float)
        geometry = _parse_geometry_json(str(artifact.get('geometry_json', '')))
        extent = _grid_extent(xs, ys)

        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax.imshow(p_map, origin='lower', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
        _draw_geometry(ax, geometry)
        for color, (method_id, t_aligned, gd, risk, amb, pv, pveff, rstd, realized, goal_xy) in zip(palette, combined_ts):
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

    # Combined time series
    for tsv_name, idx_tuple, y_label, title in (
        ('combined_goal_distance_vs_time.png', (2,), 'Distance [m]', 'Goal Distance vs Time'),
        ('combined_efe_risk_vs_time.png', (3,), 'efe_risk', 'EFE Risk vs Time'),
        ('combined_efe_ambiguity_vs_time.png', (4,), 'efe_ambiguity', 'EFE Ambiguity vs Time'),
    ):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        idx = idx_tuple[0]
        for color, pack in zip(palette, combined_ts):
            method_id = pack[0]
            t_aligned = pack[1]
            data = pack[idx]
            m = (t_aligned >= 0.0)
            ax.plot(t_aligned[m], data[m], color=color, linewidth=2, label=_method_label(method_id))
        ax.set_xlabel('Time since first command [s]')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        fig.savefig(output_dir / tsv_name, dpi=160)
        plt.close(fig)

    # Combined visibility vs time
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True, sharex=True)
    for color, pack in zip(palette, combined_ts):
        method_id = pack[0]
        t_aligned = pack[1]
        m = (t_aligned >= 0.0)
        t_m = t_aligned[m]
        p_vis, p_vis_eff, r_std = pack[5], pack[6], pack[7]
        axes[0].plot(t_m, p_vis[m], color=color, linewidth=2, label=_method_label(method_id))
        axes[1].plot(t_m, p_vis_eff[m], color=color, linewidth=2)
        axes[2].plot(t_m, r_std[m], color=color, linewidth=2)
    
    axes[0].set_ylabel('p_vis_plan')
    axes[1].set_ylabel('p_vis_plan_eff')
    axes[2].set_ylabel('r_plan_u_std')
    axes[2].set_xlabel('Time since first command [s]')
    axes[0].set_title('Visibility Proxies vs Time')
    axes[0].legend()
    for ax in axes:
        ax.grid(True)
    fig.savefig(output_dir / 'combined_visibility_vs_time.png', dpi=160)
    plt.close(fig)

    # Combined path profile
    if combined_profiles:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True, sharex=True)
        for color, (method_id, cum_dist, p_vis, amb, r_std) in zip(palette, combined_profiles):
            axes[0].plot(cum_dist, p_vis, color=color, linewidth=2, label=_method_label(method_id))
            axes[1].plot(cum_dist, amb, color=color, linewidth=2)
            axes[2].plot(cum_dist, r_std, color=color, linewidth=2)
        axes[0].set_ylabel('p_vis(grid)')
        axes[1].set_ylabel('ambiguity(grid)')
        axes[2].set_ylabel('r_plan_uv_std(grid)')
        axes[2].set_xlabel('Cumulative path distance [m]')
        axes[0].set_title('Spatially Sampled Background Fields along Trajectory')
        axes[0].legend()
        for ax in axes:
            ax.grid(True)
        fig.savefig(output_dir / 'combined_path_profile.png', dpi=160)
        plt.close(fig)

    write_manifest(output_dir / 'plot_manifest.json', {
        'planner_runs_root': str(planner_runs_root),
        'gp_dir': str(gp_dir),
        'background_method': background_method,
        'available_methods': [entry['method_id'] for entry in method_entries],
        'method_entries': method_entries,
        'notes': [
            'Only runs with a completed run_summary.json are considered.',
            'Time series are aligned to the first_cmd_stamp rather than system startup.',
        ],
    })
    print(f'Wrote planned-path analysis to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
