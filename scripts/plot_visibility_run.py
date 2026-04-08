#!/usr/bin/env python3
"""Generate visibility summary plots from a logged run."""

from __future__ import annotations

import argparse
import csv
import json
from json import JSONDecodeError
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def _load_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    cols: dict[str, list[float]] = {name: [] for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            raw = (row.get(name) or '').strip()
            try:
                cols[name].append(float(raw))
            except ValueError:
                cols[name].append(np.nan)
    return {name: np.asarray(values, dtype=float) for name, values in cols.items()}


def _load_plan_groups(path: Path, *, max_groups: int = 32) -> list[np.ndarray]:
    if not path.is_file():
        return []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        groups: dict[float, list[tuple[float, float]]] = {}
        for row in reader:
            try:
                stamp = float(row['plan_stamp'])
                x = float(row['x'])
                y = float(row['y'])
            except (KeyError, TypeError, ValueError):
                continue
            groups.setdefault(stamp, []).append((x, y))
    stamps = sorted(groups.keys())
    if not stamps:
        return []
    stride = max(int(np.ceil(len(stamps) / max(max_groups, 1))), 1)
    selected = stamps[::stride]
    return [np.asarray(groups[stamp], dtype=float) for stamp in selected if len(groups[stamp]) >= 2]


def _load_artifact(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as data:
        artifact: dict[str, np.ndarray | str] = {key: np.asarray(data[key]) for key in data.files}
    for key in ('geometry_json',):
        if key in artifact:
            artifact[key] = str(np.asarray(artifact[key]).reshape(-1)[0])
    return artifact


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding='utf-8') as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, JSONDecodeError, TypeError, ValueError):
        return {}


def _parse_prisms(geometry_json: str) -> list[dict[str, float]]:
    payload = str(geometry_json or '').strip()
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except JSONDecodeError:
        return []
    prisms = data.get('prisms', []) if isinstance(data, dict) else []
    clean = []
    for prism in prisms:
        try:
            clean.append({
                'xmin': float(prism['xmin']),
                'xmax': float(prism['xmax']),
                'ymin': float(prism['ymin']),
                'ymax': float(prism['ymax']),
                'zmin': float(prism.get('zmin', 0.0)),
                'zmax': float(prism.get('zmax', 0.0)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return clean


def _grid_extent(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    x0 = float(xs[0])
    x1 = float(xs[-1])
    y0 = float(ys[0])
    y1 = float(ys[-1])
    return (x0, x1, y0, y1)


def _draw_scene(ax, prisms: list[dict[str, float]]) -> None:
    for prism in prisms:
        rect = Rectangle(
            (prism['xmin'], prism['ymin']),
            prism['xmax'] - prism['xmin'],
            prism['ymax'] - prism['ymin'],
            facecolor='white',
            edgecolor='black',
            linewidth=1.3,
            alpha=0.28,
        )
        ax.add_patch(rect)


def _bilinear_sample(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    if xs.size < 2 or ys.size < 2:
        return np.full_like(xq, np.nan, dtype=float)
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)
    ix = np.clip(np.searchsorted(xs, xq, side='right') - 1, 0, xs.size - 2)
    iy = np.clip(np.searchsorted(ys, yq, side='right') - 1, 0, ys.size - 2)

    x0 = xs[ix]
    x1 = xs[ix + 1]
    y0 = ys[iy]
    y1 = ys[iy + 1]
    tx = np.where(np.abs(x1 - x0) < 1e-12, 0.0, (xq - x0) / (x1 - x0))
    ty = np.where(np.abs(y1 - y0) < 1e-12, 0.0, (yq - y0) / (y1 - y0))
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    z00 = grid[iy, ix]
    z10 = grid[iy, ix + 1]
    z01 = grid[iy + 1, ix]
    z11 = grid[iy + 1, ix + 1]
    z0 = (1.0 - tx) * z00 + tx * z10
    z1 = (1.0 - tx) * z01 + tx * z11
    return (1.0 - ty) * z0 + ty * z1


def _maybe_get(cols: dict[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(cols.get(name, np.array([], dtype=float)), dtype=float)


def _plot_field_panels(output_dir: Path, artifact: dict[str, np.ndarray | str] | None, manifest: dict[str, object], run_cols: dict[str, np.ndarray], plan_groups: list[np.ndarray], perception_cols: dict[str, np.ndarray]) -> Path:
    if artifact is not None:
        xs = np.asarray(artifact['xs'], dtype=float)
        ys = np.asarray(artifact['ys'], dtype=float)
        p_map = np.asarray(artifact['P_map'], dtype=float)
        geometry_json = str(artifact.get('geometry_json', ''))
        visibility_enabled_arr = np.asarray(artifact.get('use_visibility_model', np.array([1.0])), dtype=float).reshape(-1)
        visibility_enabled = bool(visibility_enabled_arr.size and visibility_enabled_arr[0] >= 0.5)
    else:
        traj_x = _maybe_get(run_cols, 'x')
        traj_y = _maybe_get(run_cols, 'y')
        if traj_x.size and traj_y.size:
            xmin = float(np.nanmin(traj_x)) - 1.0
            xmax = float(np.nanmax(traj_x)) + 1.0
            ymin = float(np.nanmin(traj_y)) - 1.0
            ymax = float(np.nanmax(traj_y)) + 1.0
        else:
            xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
        xs = np.linspace(xmin, xmax, 120)
        ys = np.linspace(ymin, ymax, 120)
        p_map = np.full((ys.size, xs.size), np.nan, dtype=float)
        geometry_json = ''
        visibility_enabled = bool(manifest.get('use_visibility_model', False))
    artifact_model = 'empirical_gp_visibility' if artifact is not None else None

    rho_mean = None if artifact is None else artifact.get('rho_mean_map')
    rho_cons = None if artifact is None else artifact.get('rho_conservative_map')
    p_mean = None if artifact is None else artifact.get('P_mean_map')
    p_cons = None if artifact is None else artifact.get('P_conservative_map')
    if rho_mean is not None:
        rho_mean = np.asarray(rho_mean, dtype=float)
    if rho_cons is not None:
        rho_cons = np.asarray(rho_cons, dtype=float)
    if p_mean is not None:
        p_mean = np.asarray(p_mean, dtype=float)
    if p_cons is not None:
        p_cons = np.asarray(p_cons, dtype=float)

    prisms = _parse_prisms(geometry_json)
    camera_pos = (
        np.asarray(artifact.get('camera_pos', np.array([np.nan, np.nan, np.nan])), dtype=float).reshape(-1)
        if artifact is not None else np.array([np.nan, np.nan, np.nan], dtype=float)
    )

    traj_x = _maybe_get(run_cols, 'x')
    traj_y = _maybe_get(run_cols, 'y')
    goal_x = _maybe_get(run_cols, 'goal_x')
    goal_y = _maybe_get(run_cols, 'goal_y')
    goal_xy = None
    if goal_x.size and goal_y.size:
        valid_goal = np.isfinite(goal_x) & np.isfinite(goal_y)
        if np.any(valid_goal):
            last_idx = np.flatnonzero(valid_goal)[-1]
            goal_xy = (float(goal_x[last_idx]), float(goal_y[last_idx]))

    miss_mask = (_maybe_get(perception_cols, 'detected') < 0.5) & (_maybe_get(perception_cols, 'true_available') > 0.5)
    miss_x = _maybe_get(perception_cols, 'true_x')[miss_mask]
    miss_y = _maybe_get(perception_cols, 'true_y')[miss_mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True, sharex=True, sharey=True)
    has_gp_field = visibility_enabled and artifact is not None
    if has_gp_field:
        panels = [
            ('GP Visibility Mean', p_mean if p_mean is not None else p_map, 'viridis'),
            ('GP Visibility Conservative', p_cons if p_cons is not None else p_map, 'magma'),
            ('Visibility Prior', p_map, 'viridis'),
        ]
    else:
        panels = [
            ('GP Occlusion Mean', rho_mean if rho_mean is not None else p_map, 'viridis'),
            ('GP Occlusion Conservative', rho_cons if rho_cons is not None else p_map, 'magma'),
            ('Visibility Prior', p_map, 'viridis'),
        ]
    extent = _grid_extent(xs, ys)

    for ax, (title, grid, cmap) in zip(axes, panels):
        im = ax.imshow(grid, extent=extent, origin='lower', cmap=cmap, vmin=0.0, vmax=1.0, aspect='equal')
        _draw_scene(ax, prisms)
        if traj_x.size and traj_y.size:
            ax.plot(traj_x, traj_y, color='black', linewidth=2.2, label='executed path')
            ax.scatter(traj_x[0], traj_y[0], color='tab:green', s=45, zorder=5)
            ax.scatter(traj_x[-1], traj_y[-1], color='tab:red', s=55, marker='x', zorder=5)
        if goal_xy is not None:
            ax.scatter(goal_xy[0], goal_xy[1], color='deepskyblue', s=90, marker='*', zorder=5)
        if camera_pos.size >= 2 and np.all(np.isfinite(camera_pos[:2])):
            ax.scatter(camera_pos[0], camera_pos[1], color='cyan', s=50, marker='^', zorder=5)
        if title == 'Visibility Prior':
            for plan_xy in plan_groups:
                ax.plot(plan_xy[:, 0], plan_xy[:, 1], color='tab:orange', alpha=0.14, linewidth=1.0)
            if miss_x.size:
                ax.scatter(miss_x, miss_y, color='crimson', s=28, marker='x', alpha=0.8, zorder=6)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    method = str(manifest.get('method', '')).strip() or str(manifest.get('planner', '')).strip() or 'unknown_method'
    planner = str(manifest.get('planner', '')).strip() or method
    enabled_note = 'uses visibility' if visibility_enabled else 'planner ignores visibility'
    fig.suptitle(
        f"{method} | planner={planner} | model={artifact_model or 'none'} | {enabled_note}",
        fontsize=12,
    )

    out_path = output_dir / 'field_story.png'
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_timeseries(output_dir: Path, artifact: dict[str, np.ndarray | str] | None, manifest: dict[str, object], run_cols: dict[str, np.ndarray], perception_cols: dict[str, np.ndarray]) -> Path:
    if artifact is not None:
        xs = np.asarray(artifact['xs'], dtype=float)
        ys = np.asarray(artifact['ys'], dtype=float)
        p_map = np.asarray(artifact['P_map'], dtype=float)
    else:
        xs = ys = np.array([], dtype=float)
        p_map = np.array([], dtype=float)

    stamp = _maybe_get(run_cols, 'stamp')
    if not stamp.size:
        raise RuntimeError('experiment.csv is empty or missing stamp column')
    diag_stamp = _maybe_get(perception_cols, 'diag_stamp')
    first_stamps = [float(v) for v in (stamp[0], diag_stamp[0] if diag_stamp.size else np.nan) if np.isfinite(v)]
    t0 = min(first_stamps)
    t = stamp - t0
    traj_x = _maybe_get(run_cols, 'x')
    traj_y = _maybe_get(run_cols, 'y')
    traj_p_vis = (
        _bilinear_sample(xs, ys, p_map, traj_x, traj_y)
        if artifact is not None and p_map.size
        else np.full_like(traj_x, np.nan, dtype=float)
    )

    goal_dist = _maybe_get(run_cols, 'goal_dist')
    plan_length = _maybe_get(run_cols, 'plan_length')
    optimizer_success = _maybe_get(run_cols, 'optimizer_success')
    optimizer_status = _maybe_get(run_cols, 'optimizer_status')
    optimizer_nit = _maybe_get(run_cols, 'optimizer_nit')
    optimizer_nfev = _maybe_get(run_cols, 'optimizer_nfev')
    measurement_available = _maybe_get(run_cols, 'measurement_available')
    belief_age_s = _maybe_get(run_cols, 'belief_age_s')
    p_vis_plan = _maybe_get(run_cols, 'p_vis_plan')
    p_vis_plan_eff = _maybe_get(run_cols, 'p_vis_plan_eff')
    r_plan_u_std = _maybe_get(run_cols, 'r_plan_u_std')
    r_plan_v_std = _maybe_get(run_cols, 'r_plan_v_std')
    plan_time_ms = _maybe_get(run_cols, 'plan_time_ms')
    solve_time_ms = _maybe_get(run_cols, 'solve_time_ms')
    cov_x = _maybe_get(run_cols, 'planner_cov_x')
    cov_y = _maybe_get(run_cols, 'planner_cov_y')
    cov_yaw = _maybe_get(run_cols, 'planner_cov_yaw')
    if (not cov_x.size) or (not np.any(np.isfinite(cov_x))):
        cov_x = _maybe_get(run_cols, 'cov_x')
    if (not cov_y.size) or (not np.any(np.isfinite(cov_y))):
        cov_y = _maybe_get(run_cols, 'cov_y')
    if (not cov_yaw.size) or (not np.any(np.isfinite(cov_yaw))):
        cov_yaw = _maybe_get(run_cols, 'cov_yaw')
    efe_risk = _maybe_get(run_cols, 'efe_risk')
    efe_ambiguity = _maybe_get(run_cols, 'efe_ambiguity')
    efe_control = _maybe_get(run_cols, 'efe_control')
    efe_visibility = _maybe_get(run_cols, 'efe_visibility')
    efe_obstacle = _maybe_get(run_cols, 'efe_obstacle')

    detected = _maybe_get(perception_cols, 'detected')
    state_err = _maybe_get(perception_cols, 'state_pos_error')
    if diag_stamp.size:
        diag_t = diag_stamp - t0
    else:
        diag_t = np.array([], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t, traj_p_vis, color='tab:green', linewidth=2.2, label='p_vis exec')
    if p_vis_plan.size:
        ax.plot(t, p_vis_plan, color='tab:blue', linewidth=1.8, label='p_vis plan')
    if p_vis_plan_eff.size:
        ax.plot(t, p_vis_plan_eff, color='tab:orange', linewidth=1.8, label='p_vis plan eff')
    if diag_t.size and detected.size:
        ax.step(diag_t, detected, where='post', color='black', linewidth=1.4, alpha=0.7, label='detected')
    ax.set_title('Visibility And Detection')
    ax.set_xlabel('time [s]')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')

    ax = axes[0, 1]
    if goal_dist.size:
        ax.plot(t, goal_dist, color='tab:blue', linewidth=2.2, label='goal distance')
    if plan_length.size:
        ax.plot(t, plan_length, color='tab:orange', linewidth=1.9, label='plan length')
    ax.set_title('Progress And Plan Length')
    ax.set_xlabel('time [s]')
    ax.legend(loc='best')

    ax = axes[0, 2]
    if plan_time_ms.size:
        ax.plot(t, plan_time_ms, linewidth=2.0, label='plan time [ms]')
    if solve_time_ms.size:
        ax.plot(t, solve_time_ms, linewidth=2.0, label='solver time [ms]')
    ax.set_title('Planner Timing')
    ax.set_xlabel('time [s]')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='best')

    ax = axes[1, 0]
    if cov_x.size:
        ax.plot(t, cov_x, linewidth=2.0, label='cov_x')
    if cov_y.size:
        ax.plot(t, cov_y, linewidth=2.0, label='cov_y')
    if cov_yaw.size:
        ax.plot(t, cov_yaw, linewidth=2.0, label='cov_yaw')
    if belief_age_s.size:
        ax.plot(t, belief_age_s, linewidth=1.8, linestyle='--', label='belief age [s]')
    if diag_t.size and state_err.size:
        ax.plot(diag_t, state_err, color='black', linewidth=1.6, alpha=0.8, label='state position error')
    ax.set_title('Planner Belief And State Error')
    ax.set_xlabel('time [s]')
    ax.legend(loc='best')

    ax = axes[1, 1]
    if efe_risk.size:
        ax.plot(t, efe_risk, linewidth=2.0, label='risk')
    if efe_ambiguity.size:
        ax.plot(t, efe_ambiguity, linewidth=2.0, label='ambiguity')
    if efe_control.size:
        ax.plot(t, efe_control, linewidth=2.0, label='control')
    if efe_visibility.size:
        ax.plot(t, efe_visibility, linewidth=2.0, label='visibility')
    if efe_obstacle.size:
        ax.plot(t, efe_obstacle, linewidth=2.0, label='obstacle')
    ax.set_title('EFE Components')
    ax.set_xlabel('time [s]')
    ax.legend(loc='best')

    ax = axes[1, 2]
    handles = []
    labels = []
    if optimizer_success.size:
        line = ax.step(t, optimizer_success, where='post', linewidth=1.8, label='success')[0]
        handles.append(line)
        labels.append('success')
    if measurement_available.size:
        line = ax.step(t, measurement_available, where='post', linewidth=1.6, label='meas avail')[0]
        handles.append(line)
        labels.append('meas avail')
    if optimizer_status.size:
        line = ax.plot(t, optimizer_status, linewidth=1.8, label='status')[0]
        handles.append(line)
        labels.append('status')
    ax.set_title('Optimizer Diagnostics')
    ax.set_xlabel('time [s]')
    ax.set_ylim(-0.1, max(1.1, float(np.nanmax(optimizer_status)) + 0.5) if optimizer_status.size else 1.1)
    ax2 = ax.twinx()
    if optimizer_nit.size:
        line = ax2.plot(t, optimizer_nit, linewidth=1.8, linestyle='--', label='nit')[0]
        handles.append(line)
        labels.append('nit')
    if optimizer_nfev.size:
        line = ax2.plot(t, optimizer_nfev, linewidth=1.8, linestyle=':', label='nfev')[0]
        handles.append(line)
        labels.append('nfev')
    if r_plan_u_std.size:
        line = ax2.plot(t, r_plan_u_std, linewidth=1.5, linestyle='-.', label='R_u std')[0]
        handles.append(line)
        labels.append('R_u std')
    if r_plan_v_std.size:
        line = ax2.plot(t, r_plan_v_std, linewidth=1.5, linestyle='-', alpha=0.6, label='R_v std')[0]
        handles.append(line)
        labels.append('R_v std')
    if handles:
        ax.legend(handles, labels, loc='best')

    method = str(manifest.get('method', '')).strip() or str(manifest.get('planner', '')).strip() or 'unknown_method'
    planner = str(manifest.get('planner', '')).strip() or method
    fig.suptitle(
        f"{method} | planner={planner}",
        fontsize=12,
    )

    out_path = output_dir / 'run_timeseries.png'
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Experiment run directory under logs/experiments')
    parser.add_argument('--artifact', type=Path, default=None, help='Optional visibility artifact path')
    parser.add_argument('--output-dir', type=Path, default=None, help='Optional output directory for figures')
    parser.add_argument('--show', action='store_true', help='Also open the generated figures interactively')
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    artifact_path = (args.artifact.expanduser().resolve() if args.artifact else run_dir / 'visibility_artifacts.npz')
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else run_dir / 'figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_csv = run_dir / 'experiment.csv'
    if not experiment_csv.is_file():
        raise SystemExit(f'Missing experiment log: {experiment_csv}')

    artifact = _load_artifact(artifact_path) if artifact_path.is_file() else None
    manifest = _load_manifest(run_dir / 'run_manifest.json')
    run_cols = _load_csv_columns(experiment_csv)
    perception_cols = _load_csv_columns(run_dir / 'perception.csv') if (run_dir / 'perception.csv').is_file() else {}
    plan_groups = _load_plan_groups(run_dir / 'plan_samples.csv')

    field_path = _plot_field_panels(output_dir, artifact, manifest, run_cols, plan_groups, perception_cols)
    ts_path = _plot_timeseries(output_dir, artifact, manifest, run_cols, perception_cols)

    print(f'Wrote {field_path}')
    print(f'Wrote {ts_path}')

    if args.show:
        img = plt.imread(field_path)
        plt.figure(figsize=(12, 4))
        plt.imshow(img)
        plt.axis('off')
        img = plt.imread(ts_path)
        plt.figure(figsize=(12, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
