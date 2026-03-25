#!/usr/bin/env python3
"""Generate notebook-style visibility summary plots from a logged run."""

from __future__ import annotations

import argparse
import csv
import json
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
            except Exception:
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
            except Exception:
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
    for key in ('geometry_json', 'visibility_model'):
        if key in artifact:
            artifact[key] = str(np.asarray(artifact[key]).reshape(-1)[0])
    return artifact


def _parse_prisms(geometry_json: str) -> list[dict[str, float]]:
    payload = str(geometry_json or '').strip()
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception:
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
        except Exception:
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


def _plot_field_panels(output_dir: Path, artifact: dict[str, np.ndarray | str], run_cols: dict[str, np.ndarray], plan_groups: list[np.ndarray], perception_cols: dict[str, np.ndarray]) -> Path:
    xs = np.asarray(artifact['xs'], dtype=float)
    ys = np.asarray(artifact['ys'], dtype=float)
    rho_mean = np.asarray(artifact['rho_mean_map'], dtype=float)
    rho_cons = np.asarray(artifact['rho_conservative_map'], dtype=float)
    p_map = np.asarray(artifact['P_map'], dtype=float)
    geometry_json = str(artifact.get('geometry_json', ''))
    prisms = _parse_prisms(geometry_json)
    camera_pos = np.asarray(artifact.get('camera_pos', np.array([np.nan, np.nan, np.nan])), dtype=float).reshape(-1)

    traj_x = _maybe_get(run_cols, 'x')
    traj_y = _maybe_get(run_cols, 'y')
    goal_x = _maybe_get(run_cols, 'goal_x')
    goal_y = _maybe_get(run_cols, 'goal_y')
    goal_xy = None
    if goal_x.size and goal_y.size and np.isfinite(goal_x[0]) and np.isfinite(goal_y[0]):
        goal_xy = (float(goal_x[0]), float(goal_y[0]))

    miss_mask = (_maybe_get(perception_cols, 'detected') < 0.5) & (_maybe_get(perception_cols, 'true_available') > 0.5)
    miss_x = _maybe_get(perception_cols, 'true_x')[miss_mask]
    miss_y = _maybe_get(perception_cols, 'true_y')[miss_mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True, sharex=True, sharey=True)
    panels = [
        ('GP Opacity Mean', rho_mean, 'viridis'),
        ('GP Opacity Conservative', rho_cons, 'magma'),
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

    out_path = output_dir / 'field_story.png'
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_timeseries(output_dir: Path, artifact: dict[str, np.ndarray | str], run_cols: dict[str, np.ndarray], perception_cols: dict[str, np.ndarray]) -> Path:
    xs = np.asarray(artifact['xs'], dtype=float)
    ys = np.asarray(artifact['ys'], dtype=float)
    p_map = np.asarray(artifact['P_map'], dtype=float)

    stamp = _maybe_get(run_cols, 'stamp')
    if not stamp.size:
        raise RuntimeError('experiment.csv is empty or missing stamp column')
    t = stamp - float(stamp[0])
    traj_x = _maybe_get(run_cols, 'x')
    traj_y = _maybe_get(run_cols, 'y')
    traj_p_vis = _bilinear_sample(xs, ys, p_map, traj_x, traj_y)

    goal_dist = _maybe_get(run_cols, 'goal_dist')
    plan_length = _maybe_get(run_cols, 'plan_length')
    cov_x = _maybe_get(run_cols, 'cov_x')
    cov_y = _maybe_get(run_cols, 'cov_y')
    cov_yaw = _maybe_get(run_cols, 'cov_yaw')
    efe_risk = _maybe_get(run_cols, 'efe_risk')
    efe_ambiguity = _maybe_get(run_cols, 'efe_ambiguity')
    efe_control = _maybe_get(run_cols, 'efe_control')
    efe_visibility = _maybe_get(run_cols, 'efe_visibility')

    diag_stamp = _maybe_get(perception_cols, 'diag_stamp')
    detected = _maybe_get(perception_cols, 'detected')
    state_err = _maybe_get(perception_cols, 'state_pos_error')
    if diag_stamp.size:
        diag_t = diag_stamp - float(stamp[0])
    else:
        diag_t = np.array([], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t, traj_p_vis, color='tab:green', linewidth=2.2, label='visibility along executed trajectory')
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

    ax = axes[1, 0]
    if cov_x.size:
        ax.plot(t, cov_x, linewidth=2.0, label='cov_x')
    if cov_y.size:
        ax.plot(t, cov_y, linewidth=2.0, label='cov_y')
    if cov_yaw.size:
        ax.plot(t, cov_yaw, linewidth=2.0, label='cov_yaw')
    if diag_t.size and state_err.size:
        ax.plot(diag_t, state_err, color='black', linewidth=1.6, alpha=0.8, label='state position error')
    ax.set_title('Belief And State Error')
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
    ax.set_title('EFE Components')
    ax.set_xlabel('time [s]')
    ax.legend(loc='best')

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

    if not artifact_path.is_file():
        raise SystemExit(f'Missing visibility artifact: {artifact_path}')

    experiment_csv = run_dir / 'experiment.csv'
    if not experiment_csv.is_file():
        raise SystemExit(f'Missing experiment log: {experiment_csv}')

    artifact = _load_artifact(artifact_path)
    run_cols = _load_csv_columns(experiment_csv)
    perception_cols = _load_csv_columns(run_dir / 'perception.csv') if (run_dir / 'perception.csv').is_file() else {}
    plan_groups = _load_plan_groups(run_dir / 'plan_samples.csv')

    field_path = _plot_field_panels(output_dir, artifact, run_cols, plan_groups, perception_cols)
    ts_path = _plot_timeseries(output_dir, artifact, run_cols, perception_cols)

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
