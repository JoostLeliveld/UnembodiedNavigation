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
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

from common import ACTIVE_METHOD_IDS, CURRENT_GP_DIR, LOGS_ROOT, PLANNER_RUNS_DIR, REPORT_DIR, write_csv, write_manifest


RUN_MANIFEST_FILENAMES = ('run_manifest.json', 'manifest.json')


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_run_manifest(run_dir: Path) -> dict:
    for name in RUN_MANIFEST_FILENAMES:
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


def _plot_settings_for_run(run_manifest: dict, args) -> dict[str, float | str | list[str]]:
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
    method = str(run_manifest.get('method', '') or '').strip()
    if method and method in ACTIVE_METHOD_IDS:
        return method
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


def _run_has_usable_logs(run_dir: Path) -> bool:
    experiment_csv = run_dir / 'experiment.csv'
    perception_csv = run_dir / 'perception.csv'
    plan_csv = run_dir / 'plan_samples.csv'
    for path in (experiment_csv, perception_csv, plan_csv):
        if path.is_file() and path.stat().st_size > 0:
            return True
    return False


def _find_latest_runs(root: Path) -> dict[str, Path]:
    run_dirs = []
    for summary_path in root.rglob('run_summary.json'):
        run_dir = summary_path.parent
        summary = _load_json(summary_path)
        if not summary:
            continue
        if not _run_has_usable_logs(run_dir):
            continue
        manifest = _load_run_manifest(run_dir)
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


def _column_with_fallback(cols: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        values = np.asarray(cols.get(name, np.array([], dtype=float)), dtype=float)
        if values.size:
            return values
    return np.array([], dtype=float)


def _trajectory_for(cols: dict[str, np.ndarray], x_names: tuple[str, ...], y_names: tuple[str, ...]) -> np.ndarray:
    xs = _column_with_fallback(cols, *x_names)
    ys = _column_with_fallback(cols, *y_names)
    if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
        return np.zeros((0, 2), dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(mask):
        return np.zeros((0, 2), dtype=float)
    return np.column_stack([xs[mask], ys[mask]])


def _aligned_series(cols: dict[str, np.ndarray], first_cmd_stamp: float, *names: str) -> tuple[np.ndarray, ...]:
    stamps = np.asarray(cols.get('stamp', np.array([], dtype=float)), dtype=float)
    t = stamps - first_cmd_stamp if math.isfinite(first_cmd_stamp) else stamps
    series = [t]
    for name in names:
        series.append(np.asarray(cols.get(name, np.full_like(t, math.nan)), dtype=float))
    return tuple(series)


def _first_finite_scalar(cols: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        arr = np.asarray(cols.get(name, np.array([], dtype=float)), dtype=float)
        if arr.size:
            return arr
    return np.array([], dtype=float)


def _build_covariance_arrays(cols: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = _column_with_fallback(cols, 'est_cov_xx', 'planner_cov_x', 'state_cov_xx', 'cov_x')
    xy = _column_with_fallback(cols, 'est_cov_xy', 'state_cov_xy')
    yy = _column_with_fallback(cols, 'est_cov_yy', 'planner_cov_y', 'state_cov_yy', 'cov_y')
    if xx.size == 0:
        xx = np.full(n, math.nan, dtype=float)
    if yy.size == 0:
        yy = np.full(n, math.nan, dtype=float)
    if xy.size == 0:
        xy = np.full(n, math.nan, dtype=float)
    xx = xx[:n] if xx.size >= n else np.pad(xx, (0, max(n - xx.size, 0)), constant_values=math.nan)
    xy = xy[:n] if xy.size >= n else np.pad(xy, (0, max(n - xy.size, 0)), constant_values=math.nan)
    yy = yy[:n] if yy.size >= n else np.pad(yy, (0, max(n - yy.size, 0)), constant_values=math.nan)
    return xx, xy, yy


def _covariance_metrics_arrays(cov_xx: np.ndarray, cov_xy: np.ndarray, cov_yy: np.ndarray):
    n = min(cov_xx.size, cov_xy.size, cov_yy.size)
    trace = np.full(n, math.nan, dtype=float)
    det = np.full(n, math.nan, dtype=float)
    sigma_major = np.full(n, math.nan, dtype=float)
    sigma_minor = np.full(n, math.nan, dtype=float)
    entropy = np.full(n, math.nan, dtype=float)
    for i in range(n):
        xx = float(cov_xx[i])
        xy = float(cov_xy[i]) if math.isfinite(cov_xy[i]) else 0.0
        yy = float(cov_yy[i])
        if not (math.isfinite(xx) and math.isfinite(yy)):
            continue
        trace[i] = xx + yy
        det_val = xx * yy - xy * xy
        det[i] = det_val
        try:
            evals = np.linalg.eigvalsh(np.array([[xx, xy], [xy, yy]], dtype=float))
            evals = np.clip(evals, 0.0, None)
            sigma_minor[i] = float(math.sqrt(evals[0]))
            sigma_major[i] = float(math.sqrt(evals[1]))
        except np.linalg.LinAlgError:
            pass
        if det_val > 0.0:
            entropy[i] = 0.5 * math.log(((2.0 * math.pi * math.e) ** 2) * det_val)
    return trace, det, sigma_major, sigma_minor, entropy


def _plot_covariance_ellipses(ax, path: np.ndarray, cov_xx: np.ndarray, cov_xy: np.ndarray, cov_yy: np.ndarray, *, every: int = 20, edgecolor: str = 'white') -> None:
    if path.shape[0] == 0:
        return
    n = min(path.shape[0], cov_xx.size, cov_xy.size, cov_yy.size)
    if n == 0:
        return
    every = max(int(every), 1)
    for idx in range(0, n, every):
        xx = float(cov_xx[idx])
        xy = float(cov_xy[idx]) if math.isfinite(cov_xy[idx]) else 0.0
        yy = float(cov_yy[idx])
        if not (math.isfinite(xx) and math.isfinite(yy)):
            continue
        try:
            evals, evecs = np.linalg.eigh(np.array([[xx, xy], [xy, yy]], dtype=float))
        except np.linalg.LinAlgError:
            continue
        evals = np.clip(evals, 0.0, None)
        width = 4.0 * math.sqrt(evals[1]) if evals.size >= 2 else 0.0
        height = 4.0 * math.sqrt(evals[0]) if evals.size >= 1 else 0.0
        if width <= 0.0 or height <= 0.0:
            continue
        angle = math.degrees(math.atan2(evecs[1, 1], evecs[0, 1]))
        ellipse = Ellipse(
            (float(path[idx, 0]), float(path[idx, 1])),
            width=width,
            height=height,
            angle=angle,
            facecolor='none',
            edgecolor=edgecolor,
            linewidth=0.9,
            alpha=0.85,
        )
        ax.add_patch(ellipse)


def _plot_time_markers(ax, path: np.ndarray, t_aligned: np.ndarray, *, spacing_s: float = 5.0, color: str = 'white') -> None:
    if path.shape[0] == 0 or t_aligned.size == 0:
        return
    n = min(path.shape[0], t_aligned.size)
    next_t = 0.0
    points = []
    for idx in range(n):
        t = float(t_aligned[idx])
        if not math.isfinite(t) or t < 0.0:
            continue
        if t >= next_t:
            points.append(path[idx])
            next_t += float(spacing_s)
    if points:
        pts = np.asarray(points, dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=20, marker='o', edgecolors='black', linewidths=0.4, zorder=6)


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return float(np.mean(values[mask])) if np.any(mask) else math.nan


def _finite_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return float(np.max(values[mask])) if np.any(mask) else math.nan


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
    parser.add_argument('--visibility-sigma-kappa', type=float, default=1.0)
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
            'notes': ['No usable planner runs were found; path plotting completed as a no-op.'],
        })
        print(f'No usable planner runs found under {planner_runs_root}')
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
    plot_notes = [
        'Usable runs are accepted even if run_summary.json marks them as interrupted.',
        'Time series are aligned to the first_cmd_stamp rather than system startup.',
    ]

    for method_id, run_dir in sorted(latest_runs.items()):
        run_cols = _read_csv_columns(run_dir / 'experiment.csv')
        run_summary = _load_json(run_dir / 'run_summary.json')
        run_manifest = _load_run_manifest(run_dir)
        plot_cfg = _plot_settings_for_run(run_manifest, args)
        if plot_cfg['used_arg_defaults']:
            plot_notes.append(
                f"{method_id}: plot settings fell back to CLI defaults for {', '.join(plot_cfg['used_arg_defaults'])}."
            )

        actual_path = _trajectory_for(run_cols, ('truth_x',), ('truth_y',))
        raw_state_path = _trajectory_for(run_cols, ('state_x',), ('state_y',))
        inferred_path = _trajectory_for(
            run_cols,
            ('planner_belief_x', 'est_x'),
            ('planner_belief_y', 'est_y'),
        )
        # Never silently substitute belief for actual. Warn if truth is absent.
        if actual_path.shape[0] == 0:
            plot_notes.append(
                f"{method_id}: truth path (truth_x/y) unavailable — actual path panel will be empty."
            )
        if actual_path.shape[0] == 0 and inferred_path.shape[0] == 0 and raw_state_path.shape[0] == 0:
            continue
        latest_plan = _load_plan_groups(run_dir / 'plan_samples.csv')
        latest_plan_pts = latest_plan[-1] if latest_plan else np.zeros((0, 2), dtype=float)
        goal_xy = _goal_from_cols(run_cols)

        first_cmd_stamp = float(run_summary.get('first_cmd_stamp', math.nan))
        t_aligned = _time_align_trajectory(run_cols, first_cmd_stamp)
        gd = np.asarray(run_cols.get('goal_dist', []), dtype=float)
        efe_risk = np.asarray(run_cols.get('efe_risk', []), dtype=float)
        efe_ambiguity = np.asarray(run_cols.get('efe_ambiguity', []), dtype=float)
        efe_control = np.asarray(run_cols.get('efe_control', []), dtype=float)
        efe_visibility = np.asarray(run_cols.get('efe_visibility', []), dtype=float)
        efe_obstacle = np.asarray(run_cols.get('efe_obstacle', []), dtype=float)
        efe_total = np.asarray(run_cols.get('efe_total', []), dtype=float)
        p_vis_plan = np.asarray(run_cols.get('p_vis_plan', []), dtype=float)
        p_vis_plan_eff = np.asarray(run_cols.get('p_vis_plan_eff', []), dtype=float)
        r_plan_u_std = np.asarray(run_cols.get('r_plan_u_std', []), dtype=float)
        # Prefer explicit unambiguous error columns; fall back to legacy state_pos_error_m
        truth_state_error_m = _column_with_fallback(run_cols, 'truth_state_error_m')
        truth_belief_error_m = _column_with_fallback(run_cols, 'truth_belief_error_m')
        state_pos_error = _column_with_fallback(run_cols, 'state_pos_error_m')
        cov_trace = _column_with_fallback(run_cols, 'state_cov_trace')
        cov_major = _column_with_fallback(run_cols, 'state_sigma_major_m')

        max_len = max(
            inferred_path.shape[0],
            raw_state_path.shape[0],
            _first_finite_scalar(run_cols, 'planner_belief_x', 'est_x', 'state_x', 'x').size,
            _first_finite_scalar(run_cols, 'stamp').size,
        )
        est_cov_xx, est_cov_xy, est_cov_yy = _build_covariance_arrays(run_cols, max_len)
        if cov_trace.size == 0 or cov_major.size == 0:
            cov_trace, _, cov_major, _, _ = _covariance_metrics_arrays(est_cov_xx, est_cov_xy, est_cov_yy)

        combined_ts.append({
            'method_id': method_id,
            't': t_aligned,
            'goal_dist': gd,
            'efe_risk': efe_risk,
            'efe_ambiguity': efe_ambiguity,
            'efe_control': efe_control,
            'efe_visibility': efe_visibility,
            'efe_obstacle': efe_obstacle,
            'efe_total': efe_total,
            'p_vis_plan': p_vis_plan,
            'p_vis_plan_eff': p_vis_plan_eff,
            'r_plan_u_std': r_plan_u_std,
            'truth_state_error_m': truth_state_error_m,
            'truth_belief_error_m': truth_belief_error_m,
            'state_pos_error_m': state_pos_error,   # legacy fallback
            'state_cov_trace': cov_trace,
            'state_sigma_major_m': cov_major,
            'actual_path': actual_path,
            'raw_state_path': raw_state_path,
            'inferred_path': inferred_path,
            'goal_xy': goal_xy,
            'plot_cfg': plot_cfg,
        })

        artifact_path = artifacts.get(method_id)
        if artifact_path is None:
            raw_artifact = str(run_manifest.get('visibility_artifact_path', '') or run_summary.get('visibility_artifact_path', '') or '').strip()
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
            min_prob=float(plot_cfg['min_prob']),
            r_visible_uv=float(plot_cfg['r_visible_uv']),
            r_miss_uv=float(plot_cfg['r_miss_uv']),
            visibility_power=float(plot_cfg['visibility_power']),
            visibility_trust_low=float(plot_cfg['visibility_trust_low']),
            visibility_trust_high=float(plot_cfg['visibility_trust_high']),
        )
        std_map = _r_plan_uv_std_map(
            p_map,
            min_prob=float(plot_cfg['min_prob']),
            r_visible_uv=float(plot_cfg['r_visible_uv']),
            r_miss_uv=float(plot_cfg['r_miss_uv']),
            visibility_power=float(plot_cfg['visibility_power']),
            visibility_trust_low=float(plot_cfg['visibility_trust_low']),
            visibility_trust_high=float(plot_cfg['visibility_trust_high']),
        )

        method_dir = output_dir / method_id
        method_dir.mkdir(parents=True, exist_ok=True)

        finite_ambiguity = ambiguity_map[np.isfinite(ambiguity_map)]
        amb75 = float(np.nanpercentile(finite_ambiguity, 75.0)) if finite_ambiguity.size else math.nan
        amb90 = float(np.nanpercentile(finite_ambiguity, 90.0)) if finite_ambiguity.size else math.nan

        # Only sample path profiles along truth path. Skip gracefully if truth absent.
        if actual_path.shape[0] > 0:
            actual_for_sampling = actual_path
            p_vis_profile = _sample_grid_along_path(p_map, extent, actual_for_sampling)
            amb_profile = _sample_grid_along_path(ambiguity_map, extent, actual_for_sampling)
            std_profile = _sample_grid_along_path(std_map, extent, actual_for_sampling)
            cum_dist = _cumulative_path_distance(actual_for_sampling)
            combined_profiles.append({
                'method_id': method_id,
                'cum_dist': cum_dist,
                'p_vis_profile': p_vis_profile,
                'amb_profile': amb_profile,
                'std_profile': std_profile,
                'state_cov_trace': cov_trace[: cum_dist.size] if cov_trace.size else np.full(cum_dist.size, math.nan),
                'truth_state_error_m': truth_state_error_m[: cum_dist.size] if truth_state_error_m.size else np.full(cum_dist.size, math.nan),
                'truth_belief_error_m': truth_belief_error_m[: cum_dist.size] if truth_belief_error_m.size else np.full(cum_dist.size, math.nan),
            })
        else:
            plot_notes.append(
                f"{method_id}: path profile skipped — truth path unavailable (truth_x/y columns absent or all NaN)."
            )

        # Actual vs inferred state over visibility map
        fig_state, ax_state = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax_state.imshow(p_map, origin='lower', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
        _draw_geometry(ax_state, geometry)
        if actual_path.shape[0]:
            ax_state.plot(actual_path[:, 0], actual_path[:, 1], color='deepskyblue', linewidth=2.3, label='actual')
            _plot_time_markers(ax_state, actual_path, t_aligned, color='deepskyblue')
            ax_state.scatter(actual_path[-1, 0], actual_path[-1, 1], c='cyan', s=42, marker='s', label='actual final')
        if raw_state_path.shape[0]:
            ax_state.plot(raw_state_path[:, 0], raw_state_path[:, 1], color='tomato', linewidth=1.4, linestyle=':', label='raw perception state')
        if inferred_path.shape[0]:
            ax_state.plot(inferred_path[:, 0], inferred_path[:, 1], color='white', linewidth=1.8, linestyle='--', label='planner belief')
            _plot_covariance_ellipses(ax_state, inferred_path, est_cov_xx, est_cov_xy, est_cov_yy, every=20, edgecolor='white')
        if goal_xy is not None:
            ax_state.scatter(goal_xy[0], goal_xy[1], c='gold', s=75, marker='*', label='goal')
        if actual_path.shape[0]:
            ax_state.scatter(actual_path[0, 0], actual_path[0, 1], c='lime', s=42, marker='o', label='start')
        ax_state.set_title('Actual, planner belief, and raw state over visibility')
        ax_state.set_xlabel('x [m]')
        ax_state.set_ylabel('y [m]')
        ax_state.legend(loc='upper right', fontsize=8)
        fig_state.savefig(method_dir / 'actual_vs_inferred_state.png', dpi=160)
        plt.close(fig_state)

        # State certainty map
        fig_cert, ax_cert = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax_cert.imshow(ambiguity_map, origin='lower', extent=extent, cmap='magma', aspect='equal')
        _draw_geometry(ax_cert, geometry)
        if actual_path.shape[0]:
            ax_cert.plot(actual_path[:, 0], actual_path[:, 1], color='black', linewidth=2.5, alpha=0.75, label='actual')
        if raw_state_path.shape[0]:
            ax_cert.plot(raw_state_path[:, 0], raw_state_path[:, 1], color='tomato', linewidth=1.2, linestyle=':', label='raw perception state')
        if inferred_path.shape[0]:
            line_color = cov_major[: inferred_path.shape[0]] if cov_major.size else np.full(inferred_path.shape[0], math.nan)
            sc = ax_cert.scatter(
                inferred_path[:, 0],
                inferred_path[:, 1],
                c=line_color,
                s=20,
                cmap='coolwarm',
                label='planner belief sigma_major',
            )
            fig_cert.colorbar(sc, ax=ax_cert, fraction=0.046, pad=0.04, label='state_sigma_major_m')
            ax_cert.plot(inferred_path[:, 0], inferred_path[:, 1], color='white', linewidth=1.2, linestyle='--')
            _plot_covariance_ellipses(ax_cert, inferred_path, est_cov_xx, est_cov_xy, est_cov_yy, every=20, edgecolor='white')
        if goal_xy is not None:
            ax_cert.scatter(goal_xy[0], goal_xy[1], c='gold', s=75, marker='*')
        ax_cert.set_title('State certainty over ambiguity field')
        ax_cert.set_xlabel('x [m]')
        ax_cert.set_ylabel('y [m]')
        ax_cert.legend(loc='upper right', fontsize=8)
        fig_cert.savefig(method_dir / 'state_certainty_map.png', dpi=160)
        plt.close(fig_cert)

        # Path over ambiguity regions
        fig_regions, ax_regions = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax_regions.imshow(ambiguity_map, origin='lower', extent=extent, cmap='magma', aspect='equal')
        _draw_geometry(ax_regions, geometry)
        levels_with_labels = []
        if math.isfinite(amb75):
            levels_with_labels.append((amb75, '75th pct'))
        if math.isfinite(amb90):
            # Matplotlib requires strictly increasing levels
            if not levels_with_labels or amb90 > levels_with_labels[-1][0]:
                levels_with_labels.append((amb90, '90th pct'))
                
        levels = [lvl for lvl, lbl in levels_with_labels]
        if levels:
            contours = ax_regions.contour(
                ambiguity_map,
                levels=levels,
                origin='lower',
                extent=extent,
                colors=['cyan', 'yellow'][: len(levels)],
                linewidths=1.8,
            )
            labels = {lvl: lbl for lvl, lbl in levels_with_labels}
            ax_regions.clabel(contours, inline=True, fontsize=8, fmt=labels)
        if actual_path.shape[0]:
            ax_regions.plot(actual_path[:, 0], actual_path[:, 1], color='deepskyblue', linewidth=2.2, label='actual')
        if raw_state_path.shape[0]:
            ax_regions.plot(raw_state_path[:, 0], raw_state_path[:, 1], color='tomato', linewidth=1.2, linestyle=':', label='raw perception state')
        if inferred_path.shape[0]:
            ax_regions.plot(inferred_path[:, 0], inferred_path[:, 1], color='white', linewidth=1.5, linestyle='--', label='planner belief')
        if latest_plan_pts.shape[0] >= 2:
            ax_regions.plot(latest_plan_pts[:, 0], latest_plan_pts[:, 1], color='lawngreen', linewidth=1.4, linestyle=':', label='latest plan')
        if goal_xy is not None:
            ax_regions.scatter(goal_xy[0], goal_xy[1], c='gold', s=75, marker='*')
        ax_regions.set_title('Path over ambiguity regions')
        ax_regions.set_xlabel('x [m]')
        ax_regions.set_ylabel('y [m]')
        ax_regions.legend(loc='upper right', fontsize=8)
        fig_regions.savefig(method_dir / 'path_over_ambiguity_regions.png', dpi=160)
        plt.close(fig_regions)

        # Uncertainty propagation sheet
        fig = plt.figure(figsize=(24, 12), constrained_layout=True)
        fig.suptitle(f'{_method_label(method_id)} uncertainty propagation', fontsize=16)
        gs = fig.add_gridspec(2, 3)
        ax_pvis = fig.add_subplot(gs[0, 0])
        ax_amb = fig.add_subplot(gs[0, 1])
        ax_bg = fig.add_subplot(gs[0, 2])
        ax_gd = fig.add_subplot(gs[1, 0])
        ax_efe = fig.add_subplot(gs[1, 1])
        ax_vis = fig.add_subplot(gs[1, 2])

        # Top row: spatial maps
        for ax, grid, title, cmap in (
            (ax_pvis, p_map, 'Actual / belief / raw state over p_vis', 'viridis'),
            (ax_amb, ambiguity_map, 'Actual / belief / raw state over ambiguity', 'magma'),
        ):
            ax.imshow(grid, origin='lower', extent=extent, cmap=cmap, aspect='equal')
            _draw_geometry(ax, geometry)
            if actual_path.shape[0]:
                ax.plot(actual_path[:, 0], actual_path[:, 1], color='deepskyblue', linewidth=2.2, label='actual')
                _plot_time_markers(ax, actual_path, t_aligned, color='deepskyblue')
                ax.scatter(actual_path[0, 0], actual_path[0, 1], c='lime', s=50, marker='o', label='start')
            if raw_state_path.shape[0]:
                ax.plot(raw_state_path[:, 0], raw_state_path[:, 1], color='tomato', linewidth=1.2, linestyle=':', label='raw perception state')
            if inferred_path.shape[0]:
                ax.plot(inferred_path[:, 0], inferred_path[:, 1], color='white', linewidth=1.5, linestyle='--', label='planner belief')
                _plot_covariance_ellipses(ax, inferred_path, est_cov_xx, est_cov_xy, est_cov_yy, every=20, edgecolor='white')
            if goal_xy is not None:
                ax.scatter(goal_xy[0], goal_xy[1], c='gold', s=70, marker='*', label='goal')
            if latest_plan_pts.shape[0] >= 2:
                ax.plot(latest_plan_pts[:, 0], latest_plan_pts[:, 1], color='lawngreen', linestyle=':', linewidth=1.5, label='latest plan')
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

        if actual_path.shape[0]:
            ax_bg.plot(actual_path[:, 0], actual_path[:, 1], color='black', linewidth=2.3, alpha=0.75, label='actual')
        if raw_state_path.shape[0]:
            ax_bg.plot(raw_state_path[:, 0], raw_state_path[:, 1], color='tomato', linewidth=1.2, linestyle=':', label='raw perception state')
        if inferred_path.shape[0]:
            sigma_series = cov_major[: inferred_path.shape[0]] if cov_major.size else np.full(inferred_path.shape[0], math.nan)
            sc_bg = ax_bg.scatter(inferred_path[:, 0], inferred_path[:, 1], c=sigma_series, s=18, cmap='coolwarm', label='planner belief')
            fig.colorbar(sc_bg, ax=ax_bg, fraction=0.046, pad=0.04, label='state_sigma_major_m')
            _plot_covariance_ellipses(ax_bg, inferred_path, est_cov_xx, est_cov_xy, est_cov_yy, every=20, edgecolor='white')
        if goal_xy is not None:
            ax_bg.scatter(goal_xy[0], goal_xy[1], c='gold', s=70, marker='*')
        ax_bg.set_title(f'State certainty over {args.background_method}')
        ax_bg.set_xlabel('x [m]')
        ax_bg.set_ylabel('y [m]')

        # Bottom row: temporal graphs
        mask_t = (t_aligned >= 0.0)
        t_plot = t_aligned[mask_t]

        ax_gd.plot(t_plot, gd[mask_t], linewidth=2, color='dodgerblue')
        ax_gd.set_title('Goal Distance over Time (truth)')
        ax_gd.set_xlabel('Time since first command [s]')
        ax_gd.set_ylabel('||truth - goal|| [m]')
        ax_gd.grid(True)

        # EFE decomposition — all six terms, raw + normalized
        efe_terms = [
            ('efe_risk',       efe_risk,       'crimson',     'risk'),
            ('efe_ambiguity',  efe_ambiguity,  'darkorange',  'ambiguity'),
            ('efe_control',    efe_control,    'steelblue',   'control'),
            ('efe_visibility', efe_visibility, 'forestgreen', 'visibility'),
            ('efe_obstacle',   efe_obstacle,   'purple',      'obstacle'),
        ]
        for term_key, term_arr, color, lbl in efe_terms:
            if term_arr.size:
                n = min(len(t_plot), term_arr[mask_t].size)
                ax_efe.plot(t_plot[:n], term_arr[mask_t][:n], label=lbl, linewidth=1.8, color=color)
        if efe_total.size:
            n = min(len(t_plot), efe_total[mask_t].size)
            ax_efe.plot(t_plot[:n], efe_total[mask_t][:n], label='total', linewidth=2.2, color='black', linestyle='--')
        ax_efe.set_title('EFE Decomposition over Time')
        ax_efe.set_xlabel('Time since first command [s]')
        ax_efe.set_ylabel('Objective value')
        ax_efe.legend(fontsize=7)
        ax_efe.grid(True)

        ax_vis.plot(t_plot, p_vis_plan[mask_t], label='p_vis_plan (raw)', linewidth=2, color='forestgreen')
        ax_vis.plot(t_plot, p_vis_plan_eff[mask_t], label='p_vis_plan_eff (shaped)', linewidth=2, color='mediumseagreen', linestyle='--')
        ax_vis2 = ax_vis.twinx()
        ax_vis2.plot(t_plot, r_plan_u_std[mask_t], label='r_plan_u_std', linewidth=2, color='purple', alpha=0.7)
        # Show explicit unambiguous error signals; fall back to legacy if new columns absent
        if truth_state_error_m.size:
            n_err = min(t_aligned.size, truth_state_error_m.size)
            err_mask = t_aligned[:n_err] >= 0.0
            ax_vis2.plot(
                t_aligned[:n_err][err_mask],
                truth_state_error_m[:n_err][err_mask],
                label='||truth - state|| [m]',
                linewidth=1.6,
                color='tomato',
                linestyle=':',
            )
        if truth_belief_error_m.size:
            n_err = min(t_aligned.size, truth_belief_error_m.size)
            err_mask = t_aligned[:n_err] >= 0.0
            ax_vis2.plot(
                t_aligned[:n_err][err_mask],
                truth_belief_error_m[:n_err][err_mask],
                label='||truth - belief|| [m]',
                linewidth=1.6,
                color='crimson',
                linestyle='-.',
            )
        elif state_pos_error.size:
            n_err = min(t_aligned.size, state_pos_error.size)
            err_mask = t_aligned[:n_err] >= 0.0
            ax_vis2.plot(
                t_aligned[:n_err][err_mask],
                state_pos_error[:n_err][err_mask],
                label='state_pos_error_m (legacy)',
                linewidth=1.6,
                color='crimson',
                linestyle=':',
            )
        if cov_trace.size:
            n_cov = min(t_aligned.size, cov_trace.size)
            cov_mask = t_aligned[:n_cov] >= 0.0
            ax_vis2.plot(
                t_aligned[:n_cov][cov_mask],
                cov_trace[:n_cov][cov_mask],
                label='state_cov_trace',
                linewidth=1.6,
                color='black',
                linestyle='-.',
            )
        ax_vis.set_title('Visibility proxies and state errors over Time')
        ax_vis.set_xlabel('Time since first command [s]')
        ax_vis.set_ylabel('p_vis probability')
        ax_vis2.set_ylabel('noise std / error [m]')
        lines_1, labels_1 = ax_vis.get_legend_handles_labels()
        lines_2, labels_2 = ax_vis2.get_legend_handles_labels()
        ax_vis.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=7)
        ax_vis.grid(True)

        method_plot = method_dir / 'uncertainty_propagation_sheet.png'
        fig.savefig(method_plot, dpi=160)
        plt.close(fig)

        alias_plot = output_dir / f'{method_id}_analysis.png'
        if method_plot.exists():
            alias_plot.write_bytes(method_plot.read_bytes())

        method_entries.append({
            'method_id': method_id,
            'run_dir': str(run_dir),
            'artifact_path': str(artifact_path),
            'plot_path': str(method_plot),
            'method_dir': str(method_dir),
            'actual_vs_inferred_state_plot': str(method_dir / 'actual_vs_inferred_state.png'),
            'state_certainty_map_plot': str(method_dir / 'state_certainty_map.png'),
            'path_over_ambiguity_regions_plot': str(method_dir / 'path_over_ambiguity_regions.png'),
            'uncertainty_propagation_sheet_plot': str(method_plot),
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

        high_amb_path_fraction = math.nan
        if amb_profile.size and math.isfinite(amb90):
            high_amb_path_fraction = float(np.mean(np.asarray(amb_profile >= amb90, dtype=float)))
        high_amb_time_fraction = math.nan
        if t_plot.size and math.isfinite(amb90):
            actual_samples = _sample_grid_along_path(ambiguity_map, extent, actual_path if actual_path.shape[0] else realized)
            if actual_samples.size:
                high_amb_time_fraction = float(np.mean(np.asarray(actual_samples[: t_plot.size] >= amb90, dtype=float)))

        planner_summaries.append({
            'method_id': method_id,
            'completed': run_summary.get('completed', False),
            'completion_reason': run_summary.get('completion_reason', ''),
            'frame_sanity_recorded': str(run_summary.get('frame_sanity', {}).get('recorded', '')),
            'frame_sanity_ok': str(run_summary.get('frame_sanity', {}).get('ok', '')),
            'frame_sanity_reason': str(run_summary.get('frame_sanity', {}).get('reason', '')),
            'frame_truth_start_error_m': run_summary.get('frame_sanity', {}).get('truth_start_error_m', ''),
            'frame_raw_start_error_m': run_summary.get('frame_sanity', {}).get('raw_start_error_m', ''),
            'elapsed_after_first_cmd_s': run_summary.get('elapsed_after_first_cmd_s', ''),
            'path_length_m': run_summary.get('path_length_m', ''),
            'final_goal_distance': run_summary.get('final_goal_distance', ''),
            'minimum_goal_distance': run_summary.get('minimum_goal_distance', ''),
            'mean_solve_time_ms': run_summary.get('mean_solve_time_ms', ''),
            'mean_efe_risk': run_summary.get('mean_efe_risk', ''),
            'mean_efe_ambiguity': run_summary.get('mean_efe_ambiguity', ''),
            'mean_efe_control': run_summary.get('mean_efe_control', ''),
            'mean_efe_visibility': run_summary.get('mean_efe_visibility', ''),
            'mean_efe_obstacle': run_summary.get('mean_efe_obstacle', ''),
            'mean_p_vis_plan': run_summary.get('mean_p_vis_plan', ''),
            'mean_p_vis_plan_eff': run_summary.get('mean_p_vis_plan_eff', ''),
            'mean_r_plan_u_std': run_summary.get('mean_r_plan_u_std', ''),
            # Explicit unambiguous error metrics from run_summary
            'mean_truth_state_error_m': run_summary.get('mean_truth_state_error_m', ''),
            'mean_truth_belief_error_m': run_summary.get('mean_truth_belief_error_m', ''),
            # Derived from CSV columns
            'mean_state_cov_trace': _finite_mean(cov_trace),
            'max_state_cov_trace': _finite_max(cov_trace),
            'mean_state_sigma_major_m': _finite_mean(cov_major),
            'max_state_sigma_major_m': _finite_max(cov_major),
            'mean_path_ambiguity': _finite_mean(amb_profile) if actual_path.shape[0] else math.nan,
            'max_path_ambiguity': _finite_max(amb_profile) if actual_path.shape[0] else math.nan,
            'mean_path_r_plan_uv_std': _finite_mean(std_profile) if actual_path.shape[0] else math.nan,
            'max_path_r_plan_uv_std': _finite_max(std_profile) if actual_path.shape[0] else math.nan,
            'fraction_path_in_high_ambiguity_region': high_amb_path_fraction,
            'fraction_time_in_high_ambiguity_region': high_amb_time_fraction,
            'run_config_hash': run_summary.get('run_config_hash', ''),
            'run_dir': str(run_dir.resolve()),
        })

    write_csv(
        output_dir / 'planner_method_summary.csv',
        (
            'method_id', 'completed', 'completion_reason',
            'frame_sanity_recorded', 'frame_sanity_ok', 'frame_sanity_reason',
            'frame_truth_start_error_m', 'frame_raw_start_error_m',
            'elapsed_after_first_cmd_s', 'path_length_m',
            'final_goal_distance', 'minimum_goal_distance', 'mean_solve_time_ms',
            'mean_efe_risk', 'mean_efe_ambiguity',
            'mean_efe_control', 'mean_efe_visibility', 'mean_efe_obstacle',
            'mean_p_vis_plan', 'mean_p_vis_plan_eff', 'mean_r_plan_u_std',
            'mean_truth_state_error_m', 'mean_truth_belief_error_m',
            'mean_state_cov_trace', 'max_state_cov_trace',
            'mean_state_sigma_major_m', 'max_state_sigma_major_m',
            'mean_path_ambiguity', 'max_path_ambiguity',
            'mean_path_r_plan_uv_std', 'max_path_r_plan_uv_std',
            'fraction_path_in_high_ambiguity_region', 'fraction_time_in_high_ambiguity_region',
            'run_config_hash',
            'run_dir'
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
        background_cfg = _plot_settings_for_run(
            _load_run_manifest(latest_runs[background_method]) if background_method in latest_runs else {},
            args,
        )

        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax.imshow(p_map, origin='lower', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
        _draw_geometry(ax, geometry)
        for color, pack in zip(palette, combined_ts):
            method_id = pack['method_id']
            actual_path = pack['actual_path']
            goal_xy = pack['goal_xy']
            realized_path = actual_path if actual_path.shape[0] else pack['inferred_path']
            if realized_path.shape[0] == 0:
                continue
            ax.plot(realized_path[:, 0], realized_path[:, 1], linewidth=2.0, color=color, label=_method_label(method_id))
            ax.scatter(realized_path[0, 0], realized_path[0, 1], color=color, s=28)
            if goal_xy is not None:
                ax.scatter(goal_xy[0], goal_xy[1], color=color, s=60, marker='*')
        ax.set_title(f'All method paths over {args.background_method} field')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.legend(loc='upper right', fontsize=8)
        combined_path = output_dir / 'combined_method_paths.png'
        fig.savefig(combined_path, dpi=160)
        plt.close(fig)

        amb_bg = _ambiguity_map(
            p_map,
            min_prob=float(background_cfg['min_prob']),
            r_visible_uv=float(background_cfg['r_visible_uv']),
            r_miss_uv=float(background_cfg['r_miss_uv']),
            visibility_power=float(background_cfg['visibility_power']),
            visibility_trust_low=float(background_cfg['visibility_trust_low']),
            visibility_trust_high=float(background_cfg['visibility_trust_high']),
        )
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax.imshow(amb_bg, origin='lower', extent=extent, cmap='magma', aspect='equal')
        _draw_geometry(ax, geometry)
        for color, pack in zip(palette, combined_ts):
            method_id = pack['method_id']
            actual_path = pack['actual_path']
            inferred_path = pack['inferred_path']
            goal_xy = pack['goal_xy']
            if actual_path.shape[0]:
                ax.plot(actual_path[:, 0], actual_path[:, 1], linewidth=2.0, color=color, label=f'{_method_label(method_id)} actual')
            if inferred_path.shape[0]:
                ax.plot(inferred_path[:, 0], inferred_path[:, 1], linewidth=1.2, color=color, linestyle='--', alpha=0.85)
            if goal_xy is not None:
                ax.scatter(goal_xy[0], goal_xy[1], color=color, s=60, marker='*')
        ax.set_title(f'All method paths over {args.background_method} ambiguity')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.legend(loc='upper right', fontsize=7)
        fig.savefig(output_dir / 'combined_ambiguity_paths.png', dpi=160)
        plt.close(fig)

    # Combined time series: goal distance + individual EFE terms
    for tsv_name, series_key, y_label, title in (
        ('combined_goal_distance_vs_time.png', 'goal_dist', '||truth - goal|| [m]', 'Goal Distance vs Time (truth)'),
        ('combined_efe_risk_vs_time.png', 'efe_risk', 'efe_risk', 'EFE Risk vs Time'),
        ('combined_efe_ambiguity_vs_time.png', 'efe_ambiguity', 'efe_ambiguity', 'EFE Ambiguity vs Time'),
    ):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for color, pack in zip(palette, combined_ts):
            method_id = pack['method_id']
            t_aligned = pack['t']
            data = np.asarray(pack[series_key], dtype=float)
            n = min(t_aligned.size, data.size)
            if n == 0:
                continue
            m = (t_aligned[:n] >= 0.0)
            ax.plot(t_aligned[:n][m], data[:n][m], color=color, linewidth=2, label=_method_label(method_id))
        ax.set_xlabel('Time since first command [s]')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        fig.savefig(output_dir / tsv_name, dpi=160)
        plt.close(fig)

    # Combined EFE full decomposition: raw terms + normalized
    _EFE_TERM_STYLES = [
        ('efe_risk',       'crimson',     '-',  'risk'),
        ('efe_ambiguity',  'darkorange',  '-',  'ambiguity'),
        ('efe_control',    'steelblue',   '-',  'control'),
        ('efe_visibility', 'forestgreen', '-',  'visibility'),
        ('efe_obstacle',   'purple',      '-',  'obstacle'),
    ]
    fig_efe, axes_efe = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True, sharex=True)
    fig_efe.suptitle('EFE Decomposition (raw + normalized)', fontsize=14)
    for color, pack in zip(palette, combined_ts):
        method_id = pack['method_id']
        t_aligned = pack['t']
        total = np.asarray(pack.get('efe_total', np.array([])), dtype=float)
        denom = np.where(np.abs(total) > 1e-9, np.abs(total), math.nan)
        m_base = t_aligned >= 0.0
        t_m = t_aligned[m_base]
        for term_key, tc, ls, lbl in _EFE_TERM_STYLES:
            arr = np.asarray(pack.get(term_key, np.array([])), dtype=float)
            n = min(arr.size, t_aligned.size)
            if n == 0:
                continue
            arr_m = arr[:n][m_base[:n]]
            t_mm = t_aligned[:n][m_base[:n]]
            # Raw
            axes_efe[0].plot(t_mm, arr_m, color=color, linewidth=1.4, linestyle=ls,
                             label=f'{_method_label(method_id)} {lbl}' if pack is combined_ts[0] else '_')
            # Normalized
            d_m = denom[:n][m_base[:n]] if denom.size >= n else np.full(arr_m.shape, math.nan)
            with np.errstate(invalid='ignore'):
                norm_m = arr_m / d_m
            axes_efe[1].plot(t_mm, norm_m, color=color, linewidth=1.4, linestyle=ls)
    axes_efe[0].set_ylabel('EFE term (raw)')
    axes_efe[1].set_ylabel('EFE term / |total|')
    axes_efe[1].set_xlabel('Time since first command [s]')
    # Annotate term names in legend using first method's colors
    if combined_ts:
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color=tc, linestyle=ls, label=lbl)
            for _, tc, ls, lbl in _EFE_TERM_STYLES
        ]
        axes_efe[0].legend(handles=handles, fontsize=8, loc='upper right')
    axes_efe[0].set_title('Raw EFE terms per method (color = method)')
    axes_efe[1].set_title('Normalized EFE terms (fraction of |total|)')
    for ax in axes_efe:
        ax.grid(True)
    fig_efe.savefig(output_dir / 'combined_efe_breakdown_vs_time.png', dpi=160)
    plt.close(fig_efe)

    # Combined visibility vs time
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True, sharex=True)
    for color, pack in zip(palette, combined_ts):
        method_id = pack['method_id']
        t_aligned = pack['t']
        p_vis = np.asarray(pack['p_vis_plan'], dtype=float)
        p_vis_eff = np.asarray(pack['p_vis_plan_eff'], dtype=float)
        r_std = np.asarray(pack['r_plan_u_std'], dtype=float)
        n = min(t_aligned.size, p_vis.size, p_vis_eff.size, r_std.size)
        if n == 0:
            continue
        m = (t_aligned[:n] >= 0.0)
        t_m = t_aligned[:n][m]
        axes[0].plot(t_m, p_vis[:n][m], color=color, linewidth=2, label=_method_label(method_id))
        axes[1].plot(t_m, p_vis_eff[:n][m], color=color, linewidth=2)
        axes[2].plot(t_m, r_std[:n][m], color=color, linewidth=2)
    
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

    # Combined path profile with uncertainty (truth path only)
    if combined_profiles:
        fig, axes = plt.subplots(5, 1, figsize=(10, 16), constrained_layout=True, sharex=True)
        for color, pack in zip(palette, combined_profiles):
            method_id = pack['method_id']
            cum_dist = pack['cum_dist']
            axes[0].plot(cum_dist, pack['p_vis_profile'], color=color, linewidth=2, label=_method_label(method_id))
            axes[1].plot(cum_dist, pack['amb_profile'], color=color, linewidth=2)
            axes[2].plot(cum_dist, pack['std_profile'], color=color, linewidth=2)
            axes[3].plot(cum_dist, pack['state_cov_trace'], color=color, linewidth=2)
            # Plot truth_state_error (solid) and truth_belief_error (dashed)
            tse = pack.get('truth_state_error_m', np.array([]))
            tbe = pack.get('truth_belief_error_m', np.array([]))
            if tse.size:
                axes[4].plot(cum_dist[:tse.size], tse, color=color, linewidth=2, label=f'{_method_label(method_id)} state')
            if tbe.size:
                axes[4].plot(cum_dist[:tbe.size], tbe, color=color, linewidth=2, linestyle='--', label=f'{_method_label(method_id)} belief')
        axes[0].set_ylabel('p_vis(truth path)')
        axes[1].set_ylabel('ambiguity(truth path)')
        axes[2].set_ylabel('r_plan_uv_std(truth path)')
        axes[3].set_ylabel('state_cov_trace')
        axes[4].set_ylabel('position error [m]')
        axes[4].set_xlabel('Cumulative truth path distance [m]')
        axes[0].set_title('Path-profile uncertainty comparison (sampled along truth path)')
        axes[0].legend(fontsize=7)
        axes[4].legend(fontsize=7)
        for ax in axes:
            ax.grid(True)
        fig.savefig(output_dir / 'combined_path_profile_uncertainty.png', dpi=160)
        plt.close(fig)

    # Combined state error / uncertainty vs time
    for name, key, ylabel, title in (
        ('combined_state_error_vs_time.png',       'truth_state_error_m',  '||truth - state|| [m]',  'State perception error vs time (truth vs /state/bev)'),
        ('combined_belief_error_vs_time.png',      'truth_belief_error_m', '||truth - belief|| [m]', 'Planner belief error vs time (truth vs /planner_belief)'),
        ('combined_state_uncertainty_vs_time.png', 'state_cov_trace',      'state_cov_trace',        'State covariance trace vs time'),
    ):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for color, pack in zip(palette, combined_ts):
            method_id = pack['method_id']
            t_aligned = pack['t']
            series = np.asarray(pack.get(key, np.array([])), dtype=float)
            # For old logs that lack truth_{state,belief}_error_m, fall back to state_pos_error_m
            if series.size == 0 and key in ('truth_state_error_m', 'truth_belief_error_m'):
                series = np.asarray(pack.get('state_pos_error_m', np.array([])), dtype=float)
            if t_aligned.size == 0 or series.size == 0:
                continue
            n = min(t_aligned.size, series.size)
            m = (t_aligned[:n] >= 0.0)
            ax.plot(t_aligned[:n][m], series[:n][m], color=color, linewidth=2, label=_method_label(method_id))
        ax.set_xlabel('Time since first command [s]')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        fig.savefig(output_dir / name, dpi=160)
        plt.close(fig)

    write_manifest(output_dir / 'plot_manifest.json', {
        'planner_runs_root': str(planner_runs_root),
        'gp_dir': str(gp_dir),
        'background_method': background_method,
        'available_methods': [entry['method_id'] for entry in method_entries],
        'method_entries': method_entries,
        'notes': plot_notes,
    })
    print(f'Wrote planned-path analysis to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
