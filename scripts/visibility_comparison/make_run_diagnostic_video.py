#!/usr/bin/env python3
"""Create an offline diagnostic video for one planner run directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

from common import CURRENT_GP_DIR, LOGS_ROOT, PAPER_VISIBILITY_DEFAULTS


RUN_MANIFEST_FILENAMES = ('run_manifest.json', 'manifest.json')


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_run_manifest(run_dir: Path) -> dict:
    for name in RUN_MANIFEST_FILENAMES:
        payload = _load_json(run_dir / name)
        if payload:
            return payload
    return {}


def _read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    cols: dict[str, list[float]] = {name: [] for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            raw = str(row.get(name, '') or '').strip()
            try:
                cols[name].append(float(raw))
            except ValueError:
                cols[name].append(math.nan)
    return {name: np.asarray(values, dtype=float) for name, values in cols.items()}


def _col(cols: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        arr = cols.get(name)
        if arr is not None and arr.size:
            return np.asarray(arr, dtype=float)
    return np.asarray([], dtype=float)


def _safe_array(arr: np.ndarray, n: int, fill: float = math.nan) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size >= n:
        return arr[:n]
    out = np.full(n, fill, dtype=float)
    if arr.size:
        out[:arr.size] = arr
    return out


def _finite_at(arr: np.ndarray, idx: int, default: float = math.nan) -> float:
    if idx < 0 or idx >= arr.size:
        return default
    value = float(arr[idx])
    return value if math.isfinite(value) else default


def _interp_to(target_t: np.ndarray, source_t: np.ndarray, values: np.ndarray, *, fill: float = math.nan) -> np.ndarray:
    target_t = np.asarray(target_t, dtype=float)
    source_t = np.asarray(source_t, dtype=float)
    values = _safe_array(np.asarray(values, dtype=float), source_t.size, fill=fill)
    finite = np.isfinite(source_t) & np.isfinite(values)
    out = np.full(target_t.shape, fill, dtype=float)
    if np.sum(finite) < 2:
        if np.sum(finite) == 1:
            out[np.isfinite(target_t)] = float(values[finite][0])
        return out
    order = np.argsort(source_t[finite])
    xs = source_t[finite][order]
    ys = values[finite][order]
    keep = np.r_[True, np.diff(xs) > 1e-9]
    xs = xs[keep]
    ys = ys[keep]
    if xs.size < 2:
        out[np.isfinite(target_t)] = float(ys[0])
        return out
    target_finite = np.isfinite(target_t)
    out[target_finite] = np.interp(target_t[target_finite], xs, ys, left=fill, right=fill)
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


def _artifact_visibility_map(artifact: dict[str, np.ndarray | str]) -> np.ndarray:
    for key in ('P_conservative_plan_map', 'P_map', 'P_conservative_map', 'P_mean_map'):
        if key in artifact:
            return np.asarray(artifact[key], dtype=float)
    raise KeyError('artifact missing planner visibility map')


def _parse_geometry_json(raw: str) -> list[dict[str, float]]:
    text = str(raw or '').strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    prisms = payload.get('prisms', []) if isinstance(payload, dict) else []
    out = []
    for prism in prisms:
        try:
            out.append({
                'xmin': float(prism['xmin']),
                'xmax': float(prism['xmax']),
                'ymin': float(prism['ymin']),
                'ymax': float(prism['ymax']),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _draw_geometry(ax, prisms: list[dict[str, float]]) -> None:
    for prism in prisms:
        ax.add_patch(Rectangle(
            (prism['xmin'], prism['ymin']),
            prism['xmax'] - prism['xmin'],
            prism['ymax'] - prism['ymin'],
            facecolor='white',
            edgecolor='black',
            linewidth=1.1,
            alpha=0.35,
        ))


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
    visibility_trust_mode: str = 'smoothstep',
) -> np.ndarray:
    p_vis = np.clip(np.asarray(p_vis, dtype=float), min_prob, 1.0 - min_prob)
    if str(visibility_trust_mode).strip().lower() in ('direct', 'identity', 'gp'):
        return p_vis
    shaped = np.clip(p_vis ** float(visibility_power), min_prob, 1.0 - min_prob)
    lo = float(np.clip(visibility_trust_low, min_prob, 1.0 - min_prob))
    hi = float(np.clip(visibility_trust_high, lo + 1e-6, 1.0 - min_prob))
    return np.clip(_smoothstep((shaped - lo) / max(hi - lo, 1e-6)), min_prob, 1.0 - min_prob)


def _ambiguity_map(
    p_map: np.ndarray,
    *,
    min_prob: float,
    r_visible_uv: float,
    r_miss_uv: float,
    visibility_power: float,
    visibility_trust_low: float,
    visibility_trust_high: float,
    visibility_trust_mode: str = 'smoothstep',
) -> np.ndarray:
    trust = _visibility_effective_score(
        p_map,
        min_prob=min_prob,
        visibility_power=visibility_power,
        visibility_trust_low=visibility_trust_low,
        visibility_trust_high=visibility_trust_high,
        visibility_trust_mode=visibility_trust_mode,
    )
    visible_var = float(r_visible_uv) ** 2
    miss_var = float(r_miss_uv) ** 2
    var = 1.0 / np.maximum(trust / max(visible_var, 1e-6) + (1.0 - trust) / max(miss_var, 1e-6), 1e-9)
    return 0.5 * np.log(np.clip(var * var, 1e-12, None))


def _load_plan_groups(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    if not path.is_file():
        return np.zeros((0,), dtype=float), []
    groups: dict[float, list[tuple[int, float, float]]] = {}
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                stamp = float(row['plan_stamp'])
                point_idx = int(float(row.get('point_idx', 0)))
                x = float(row['x'])
                y = float(row['y'])
            except (KeyError, TypeError, ValueError):
                continue
            groups.setdefault(stamp, []).append((point_idx, x, y))
    stamps = []
    plans = []
    for stamp in sorted(groups):
        ordered = sorted(groups[stamp], key=lambda item: item[0])
        pts = np.asarray([(x, y) for _idx, x, y in ordered], dtype=float)
        if pts.shape[0] >= 2:
            stamps.append(float(stamp))
            plans.append(pts)
    return np.asarray(stamps, dtype=float), plans


def _nearest_plan(plan_stamps: np.ndarray, plans: list[np.ndarray], stamp: float) -> np.ndarray:
    if plan_stamps.size == 0 or not plans:
        return np.zeros((0, 2), dtype=float)
    idx = int(np.argmin(np.abs(plan_stamps - float(stamp))))
    return plans[idx]


def _cov_ellipse_params(xx: float, xy: float, yy: float) -> tuple[float, float, float] | None:
    cov = np.asarray([[xx, xy], [xy, yy]], dtype=float)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
    vals = np.clip(vals, 0.0, None)
    if float(np.max(vals)) <= 0.0:
        return None
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
    return 2.0 * math.sqrt(float(vals[0])), 2.0 * math.sqrt(float(vals[1])), angle


def _plot_timeseries(ax, t: np.ndarray, series: list[tuple[str, np.ndarray, str]], idx: int, title: str, *, logy: bool = False) -> None:
    ax.clear()
    for label, values, color in series:
        values = _safe_array(values, t.size)
        finite = np.isfinite(t) & np.isfinite(values)
        if np.any(finite):
            ax.plot(t[finite], values[finite], label=label, color=color, linewidth=1.4)
    current_t = _finite_at(t, idx, math.nan)
    if math.isfinite(current_t):
        ax.axvline(current_t, color='black', linewidth=1.0, alpha=0.65)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    if logy:
        ax.set_yscale('symlog', linthresh=1.0)
    ax.legend(loc='upper right', fontsize=7)


def _event_ribbon(ax, t: np.ndarray, idx: int, events: list[tuple[str, np.ndarray, str]]) -> None:
    ax.clear()
    for row, (label, mask, color) in enumerate(events):
        mask = _safe_array(np.asarray(mask, dtype=float), t.size, fill=0.0) > 0.5
        finite = np.isfinite(t) & mask
        if np.any(finite):
            ax.scatter(t[finite], np.full(np.sum(finite), row), s=9, color=color, marker='s')
        ax.text(0.005, row, label, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8)
    current_t = _finite_at(t, idx, math.nan)
    if math.isfinite(current_t):
        ax.axvline(current_t, color='black', linewidth=1.0, alpha=0.75)
    ax.set_ylim(-0.8, max(len(events) - 0.2, 1.0))
    ax.set_yticks([])
    ax.set_xlabel('time after first command [s]')
    ax.grid(True, axis='x', alpha=0.2)


def _resolve_artifact(run_manifest: dict, args) -> Path:
    candidates = [
        str(args.visibility_artifact_path or ''),
        str(run_manifest.get('visibility_artifact_path', '') or ''),
    ]
    method = str(run_manifest.get('method', '') or '').strip()
    if method:
        candidates.append(str(CURRENT_GP_DIR / f'{method}_gp.npz'))
    candidates.append(str(CURRENT_GP_DIR / 'yolo_score_raw_gp.npz'))
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_file():
            return path
    raise RuntimeError('Could not resolve a visibility GP artifact for the run')


def main() -> int:
    parser = argparse.ArgumentParser(description='Create an offline diagnostic video for one run.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out', default='')
    parser.add_argument('--visibility-artifact-path', default='')
    parser.add_argument('--background', choices=('pvis', 'ambiguity'), default='pvis')
    parser.add_argument('--fps', type=float, default=8.0)
    parser.add_argument('--duration-s', type=float, default=30.0)
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--save-frames-dir', default='')
    parser.add_argument('--min-prob', type=float, default=1e-4)
    parser.add_argument('--r-visible-uv', type=float, default=PAPER_VISIBILITY_DEFAULTS['r_visible_uv'])
    parser.add_argument('--r-miss-uv', type=float, default=PAPER_VISIBILITY_DEFAULTS['r_miss_uv'])
    parser.add_argument('--visibility-power', type=float, default=PAPER_VISIBILITY_DEFAULTS['visibility_power'])
    parser.add_argument('--visibility-trust-low', type=float, default=PAPER_VISIBILITY_DEFAULTS['visibility_trust_low'])
    parser.add_argument('--visibility-trust-high', type=float, default=PAPER_VISIBILITY_DEFAULTS['visibility_trust_high'])
    parser.add_argument('--visibility-trust-mode', default='smoothstep')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f'Run directory not found: {run_dir}')
    out = Path(args.out).expanduser().resolve() if str(args.out).strip() else (run_dir / 'run_diagnostic_video.mp4').resolve()
    if LOGS_ROOT.resolve() not in out.parents and out.parent != run_dir:
        raise RuntimeError(f'Output must stay under {LOGS_ROOT.resolve()} or inside the run directory: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)

    run_manifest = _load_run_manifest(run_dir)
    run_summary = _load_json(run_dir / 'run_summary.json')
    exp = _read_csv_columns(run_dir / 'experiment.csv')
    perception = _read_csv_columns(run_dir / 'perception.csv')
    if not exp:
        raise RuntimeError(f'experiment.csv is missing or empty in {run_dir}')

    artifact_path = _resolve_artifact(run_manifest, args)
    artifact = _load_artifact(artifact_path)
    xs = np.asarray(artifact['xs'], dtype=float)
    ys = np.asarray(artifact['ys'], dtype=float)
    p_map = _artifact_visibility_map(artifact)
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    geometry = _parse_geometry_json(str(artifact.get('geometry_json', '') or run_manifest.get('visibility_geometry_json', '')))
    collision_geometry = _parse_geometry_json(str(run_manifest.get('collision_geometry_json', '') or ''))
    plot_cfg = {
        'min_prob': float(args.min_prob),
        'r_visible_uv': float(run_manifest.get('r_visible_uv', args.r_visible_uv) or args.r_visible_uv),
        'r_miss_uv': float(run_manifest.get('r_miss_uv', args.r_miss_uv) or args.r_miss_uv),
        'visibility_power': float(run_manifest.get('visibility_power', args.visibility_power) or args.visibility_power),
        'visibility_trust_low': float(run_manifest.get('visibility_trust_low', args.visibility_trust_low) or args.visibility_trust_low),
        'visibility_trust_high': float(run_manifest.get('visibility_trust_high', args.visibility_trust_high) or args.visibility_trust_high),
        'visibility_trust_mode': str(run_manifest.get('visibility_trust_mode', args.visibility_trust_mode) or args.visibility_trust_mode),
    }
    ambiguity_bg = _ambiguity_map(p_map, **plot_cfg)

    stamp = _col(exp, 'stamp')
    first_cmd = run_summary.get('first_cmd_stamp', math.nan)
    try:
        first_cmd = float(first_cmd)
    except (TypeError, ValueError):
        first_cmd = math.nan
    if not math.isfinite(first_cmd):
        finite_stamp = stamp[np.isfinite(stamp)]
        first_cmd = float(finite_stamp[0]) if finite_stamp.size else 0.0
    t = stamp - first_cmd
    finite_frames = np.where(np.isfinite(t))[0]
    if finite_frames.size == 0:
        raise RuntimeError('No finite timestamps available in experiment.csv')
    end_idx = int(finite_frames[-1])
    frame_count = min(int(args.max_frames), max(2, int(float(args.duration_s) * float(args.fps))))
    frame_indices = np.unique(np.linspace(int(finite_frames[0]), end_idx, frame_count).astype(int))

    truth_x = _safe_array(_col(exp, 'truth_x'), stamp.size)
    truth_y = _safe_array(_col(exp, 'truth_y'), stamp.size)
    belief_x = _safe_array(_col(exp, 'planner_belief_x', 'est_x'), stamp.size)
    belief_y = _safe_array(_col(exp, 'planner_belief_y', 'est_y'), stamp.size)
    state_x = _safe_array(_col(exp, 'state_x', 'x'), stamp.size)
    state_y = _safe_array(_col(exp, 'state_y', 'y'), stamp.size)
    goal_x = _col(exp, 'goal_x')
    goal_y = _col(exp, 'goal_y')
    p_vis_plan = _safe_array(_col(exp, 'p_vis_plan'), stamp.size)
    p_vis_plan_eff = _safe_array(_col(exp, 'p_vis_plan_eff'), stamp.size)
    r_plan_u_std = _safe_array(_col(exp, 'r_plan_u_std'), stamp.size)
    terminal_goal_progress = _safe_array(_col(exp, 'terminal_goal_progress_m'), stamp.size)
    goal_dist = _safe_array(_col(exp, 'goal_dist'), stamp.size)
    efe_risk = _safe_array(_col(exp, 'efe_risk'), stamp.size)
    efe_ambiguity = _safe_array(_col(exp, 'efe_ambiguity'), stamp.size)
    efe_obstacle = _safe_array(_col(exp, 'efe_obstacle'), stamp.size)
    efe_control = _safe_array(_col(exp, 'efe_control'), stamp.size)
    truth_belief_error = _safe_array(_col(exp, 'truth_belief_error_m'), stamp.size)
    truth_state_error = _safe_array(_col(exp, 'truth_state_error_m', 'state_pos_error_m'), stamp.size)
    state_cov_trace = _safe_array(_col(exp, 'state_cov_trace'), stamp.size)
    yaw_error_state_deg = np.abs(_safe_array(_col(exp, 'yaw_error_truth_state_rad'), stamp.size)) * 180.0 / math.pi
    yaw_error_belief_deg = np.abs(_safe_array(_col(exp, 'yaw_error_truth_belief_rad'), stamp.size)) * 180.0 / math.pi
    cov_xx = _safe_array(_col(exp, 'planner_cov_x', 'est_cov_xx', 'state_cov_xx'), stamp.size)
    cov_xy = _safe_array(_col(exp, 'planner_cov_xy', 'est_cov_xy', 'state_cov_xy'), stamp.size, 0.0)
    cov_yy = _safe_array(_col(exp, 'planner_cov_y', 'est_cov_yy', 'state_cov_yy'), stamp.size)
    collision_any = _safe_array(_col(exp, 'collision_any'), stamp.size, 0.0)
    wall_penetration = _safe_array(_col(exp, 'wall_penetration_m'), stamp.size, 0.0)
    obstacle_penetration = _safe_array(_col(exp, 'obstacle_penetration_m'), stamp.size, 0.0)
    high_ambiguity_fraction = _safe_array(_col(exp, 'fraction_horizon_high_ambiguity'), stamp.size, 0.0)
    plan_stamps, plans = _load_plan_groups(run_dir / 'plan_samples.csv')

    p_diag_stamp = _col(perception, 'diag_stamp')
    p_t = p_diag_stamp - first_cmd if p_diag_stamp.size else np.asarray([], dtype=float)
    yolo_raw = _col(perception, 'yolo_raw_best_score', 'yolo_score_raw')
    yolo_det = _col(perception, 'yolo_detected_after_threshold', 'detected')
    yolo_raw_interp = _interp_to(t, p_t, yolo_raw)
    yolo_det_interp = _interp_to(t, p_t, yolo_det, fill=0.0)

    low_pvis_event = p_vis_plan < 0.2
    high_amb_event = high_ambiguity_fraction > 0.5
    collision_event = collision_any > 0.5
    penetration_event = (wall_penetration > 0.0) | (obstacle_penetration > 0.0)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, width_ratios=[1.7, 1.0, 1.0], height_ratios=[1.0, 1.0, 1.0, 0.38])
    ax_map = fig.add_subplot(gs[:3, 0])
    ax_vis = fig.add_subplot(gs[0, 1:])
    ax_obj = fig.add_subplot(gs[1, 1:])
    ax_state = fig.add_subplot(gs[2, 1:])
    ax_ribbon = fig.add_subplot(gs[3, :])

    method = str(run_manifest.get('method', run_dir.name))
    completion = str(run_summary.get('completion_reason', ''))
    valid = run_summary.get('valid_run', '')
    title = f'{method} | {run_dir.name} | completion={completion} | valid={valid}'

    def draw_frame(idx: int) -> None:
        ax_map.clear()
        bg = p_map if args.background == 'pvis' else ambiguity_bg
        cmap = 'viridis' if args.background == 'pvis' else 'magma'
        im = ax_map.imshow(bg, origin='lower', extent=extent, cmap=cmap, aspect='equal', alpha=0.88)
        ax_map.contour(p_map, levels=[0.2], origin='lower', extent=extent, colors='cyan', linewidths=1.0)
        _draw_geometry(ax_map, geometry)
        _draw_geometry(ax_map, collision_geometry)
        n = idx + 1
        finite_truth = np.isfinite(truth_x[:n]) & np.isfinite(truth_y[:n])
        finite_belief = np.isfinite(belief_x[:n]) & np.isfinite(belief_y[:n])
        finite_state = np.isfinite(state_x[:n]) & np.isfinite(state_y[:n])
        if np.any(finite_truth):
            ax_map.plot(truth_x[:n][finite_truth], truth_y[:n][finite_truth], color='black', linewidth=2.0, label='truth')
            ax_map.scatter(truth_x[idx], truth_y[idx], color='black', s=35)
        if np.any(finite_belief):
            ax_map.plot(belief_x[:n][finite_belief], belief_y[:n][finite_belief], color='#1f77b4', linestyle='--', linewidth=1.7, label='belief')
            ax_map.scatter(belief_x[idx], belief_y[idx], color='#1f77b4', s=35)
        if np.any(finite_state):
            ax_map.plot(state_x[:n][finite_state], state_y[:n][finite_state], color='#ff7f0e', linestyle=':', linewidth=1.5, label='raw state')
            ax_map.scatter(state_x[idx], state_y[idx], color='#ff7f0e', s=25)
        current_plan = _nearest_plan(plan_stamps, plans, _finite_at(stamp, idx, 0.0))
        if current_plan.size:
            ax_map.plot(current_plan[:, 0], current_plan[:, 1], color='lime', linewidth=2.2, label='selected plan')
        gx = _finite_at(goal_x, idx, _finite_at(goal_x, 0, math.nan))
        gy = _finite_at(goal_y, idx, _finite_at(goal_y, 0, math.nan))
        if math.isfinite(gx) and math.isfinite(gy):
            ax_map.scatter([gx], [gy], marker='*', color='gold', edgecolor='black', s=180, label='goal')

        cov_params = _cov_ellipse_params(
            _finite_at(cov_xx, idx),
            _finite_at(cov_xy, idx, 0.0),
            _finite_at(cov_yy, idx),
        )
        bx = _finite_at(belief_x, idx)
        by = _finite_at(belief_y, idx)
        if cov_params is not None and math.isfinite(bx) and math.isfinite(by):
            width, height, angle = cov_params
            ax_map.add_patch(Ellipse((bx, by), width, height, angle=angle, edgecolor='#1f77b4', facecolor='none', linewidth=1.4, alpha=0.8))

        if _finite_at(collision_any, idx, 0.0) > 0.5 and math.isfinite(_finite_at(truth_x, idx)):
            ax_map.scatter([truth_x[idx]], [truth_y[idx]], marker='x', color='red', s=120, linewidths=3, label='collision')

        current_t = _finite_at(t, idx, 0.0)
        dominant_terms = {
            'risk': abs(_finite_at(efe_risk, idx, 0.0)),
            'ambiguity': abs(_finite_at(efe_ambiguity, idx, 0.0)),
            'obstacle': abs(_finite_at(efe_obstacle, idx, 0.0)),
            'control': abs(_finite_at(efe_control, idx, 0.0)),
        }
        dominant = max(dominant_terms, key=dominant_terms.get)
        text = (
            f't={current_t:.1f}s\n'
            f'p_vis={_finite_at(p_vis_plan, idx):.2f}, '
            f'p_eff={_finite_at(p_vis_plan_eff, idx):.2f}\n'
            f'Rstd={_finite_at(r_plan_u_std, idx):.1f}px, '
            f'goal={_finite_at(goal_dist, idx):.2f}m\n'
            f'term prog={_finite_at(terminal_goal_progress, idx):.2f}m\n'
            f'dominant={dominant}'
        )
        ax_map.text(0.02, 0.98, text, transform=ax_map.transAxes, va='top', ha='left', fontsize=9, bbox=dict(facecolor='white', alpha=0.78, edgecolor='none'))
        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])
        ax_map.set_xlabel('x [m]')
        ax_map.set_ylabel('y [m]')
        ax_map.set_title(f'Map over {args.background}')
        ax_map.legend(loc='lower left', fontsize=7)

        _plot_timeseries(ax_vis, t, [
            ('p_vis_plan', p_vis_plan, '#1f77b4'),
            ('p_vis_eff', p_vis_plan_eff, '#2ca02c'),
            ('YOLO raw', yolo_raw_interp, '#ff7f0e'),
            ('R std / 120', r_plan_u_std / 120.0, '#d62728'),
        ], idx, 'Visibility, detector score, and planner trust')

        _plot_timeseries(ax_obj, t, [
            ('risk', efe_risk, '#d62728'),
            ('ambiguity', efe_ambiguity, '#9467bd'),
            ('obstacle', efe_obstacle, '#8c564b'),
            ('control', efe_control, '#7f7f7f'),
        ], idx, 'Objective decomposition', logy=True)

        _plot_timeseries(ax_state, t, [
            ('truth-belief err', truth_belief_error, '#1f77b4'),
            ('truth-state err', truth_state_error, '#ff7f0e'),
            ('state yaw err deg', yaw_error_state_deg, '#d62728'),
            ('belief yaw err deg', yaw_error_belief_deg, '#9467bd'),
            ('cov trace', state_cov_trace, '#2ca02c'),
            ('goal dist', goal_dist, 'black'),
        ], idx, 'State quality, yaw error, and goal progress')

        _event_ribbon(ax_ribbon, t, idx, [
            ('detect', yolo_det_interp > 0.5, '#2ca02c'),
            ('low p_vis', low_pvis_event, '#17becf'),
            ('high amb', high_amb_event, '#ff7f0e'),
            ('collision', collision_event, 'red'),
            ('penetration', penetration_event, '#8c564b'),
        ])

        fig.suptitle(title, fontsize=13)

    frames_dir = Path(args.save_frames_dir).expanduser().resolve() if str(args.save_frames_dir).strip() else None
    if frames_dir is not None:
        if LOGS_ROOT.resolve() not in frames_dir.parents and frames_dir.parent != run_dir:
            raise RuntimeError(f'Frame output must stay under {LOGS_ROOT.resolve()} or inside the run directory: {frames_dir}')
        frames_dir.mkdir(parents=True, exist_ok=True)

    suffix = out.suffix.lower()
    if suffix == '.gif':
        writer = PillowWriter(fps=float(args.fps))
    else:
        writer = FFMpegWriter(fps=float(args.fps), metadata={'title': title})

    try:
        with writer.saving(fig, str(out), dpi=int(args.dpi)):
            for frame_no, idx in enumerate(frame_indices):
                draw_frame(int(idx))
                if frames_dir is not None:
                    fig.savefig(frames_dir / f'frame_{frame_no:04d}.png', dpi=int(args.dpi))
                writer.grab_frame()
    except (RuntimeError, FileNotFoundError) as exc:
        fallback_dir = frames_dir or (out.parent / f'{out.stem}_frames')
        fallback_dir.mkdir(parents=True, exist_ok=True)
        for frame_no, idx in enumerate(frame_indices):
            draw_frame(int(idx))
            fig.savefig(fallback_dir / f'frame_{frame_no:04d}.png', dpi=int(args.dpi))
        print(f'Video writer failed ({exc}); wrote frames to {fallback_dir}')
        return 0
    finally:
        plt.close(fig)

    print(f'Wrote diagnostic video to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
