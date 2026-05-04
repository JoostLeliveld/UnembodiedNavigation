#!/usr/bin/env python3
"""Offline fixed-rollout family probe for one saved planner run."""

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

from common import DEFAULT_WORLD_PROFILES_PATH, LOGS_ROOT, ensure_repo_python_paths, write_csv, write_manifest


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _safe(arr: np.ndarray, n: int, fill: float = math.nan) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size >= n:
        return arr[:n]
    out = np.full(n, fill, dtype=float)
    if arr.size:
        out[:arr.size] = arr
    return out


def _at(arr: np.ndarray, idx: int, default: float = math.nan) -> float:
    if idx < 0 or idx >= arr.size:
        return default
    value = float(arr[idx])
    return value if math.isfinite(value) else default


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


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
    try:
        payload = json.loads(str(raw or ''))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    out = []
    for prism in payload.get('prisms', []) if isinstance(payload, dict) else []:
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
            linewidth=1.0,
            alpha=0.42,
        ))


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
    if plan_stamps.size == 0:
        return np.zeros((0, 2), dtype=float)
    return plans[int(np.argmin(np.abs(plan_stamps - float(stamp))))]


def _ambiguity_map(planner, p_map: np.ndarray) -> np.ndarray:
    trust = np.clip(np.asarray(p_map, dtype=float), planner._visibility_min_prob, 1.0 - planner._visibility_min_prob)
    visible_var = float(planner.r_visible_uv) ** 2
    miss_var = float(planner.r_miss_uv) ** 2
    var = 1.0 / np.maximum(trust / max(visible_var, 1e-6) + (1.0 - trust) / max(miss_var, 1e-6), 1e-9)
    return 0.5 * np.log(np.clip(var * var, 1e-12, None))


def _resolve_world_camera_params(run_dir: Path, manifest: dict) -> dict:
    ensure_repo_python_paths()
    from experiments.core.world_profiles import compute_look_at_from_pose, load_profile

    profiles_path = run_dir / 'world_profiles.yaml'
    if not profiles_path.is_file():
        profiles_path = DEFAULT_WORLD_PROFILES_PATH
    profile, intrinsics, _world_path, camera_pose = load_profile(str(profiles_path), str(manifest.get('world', 'warehouse_occ_light.world.sdf')))
    del profile
    cam_pos = [float(camera_pose[0]), float(camera_pose[1]), float(camera_pose[2])]
    look_at = compute_look_at_from_pose(cam_pos, float(camera_pose[3]), float(camera_pose[4]), float(camera_pose[5]))
    return {
        'cam_pos': cam_pos,
        'look_at': [float(v) for v in look_at],
        'img_width': int(intrinsics['img_width']),
        'img_height': int(intrinsics['img_height']),
        'fov_h_rad': float(intrinsics['fov_h_rad']),
    }


def _build_planner(run_dir: Path, manifest: dict):
    ensure_repo_python_paths()
    from planning.planners.base_planner import UnicyclePlannerBase

    camera_params = _resolve_world_camera_params(run_dir, manifest)
    return UnicyclePlannerBase(
        horizon=int(manifest.get('horizon', 36)),
        dt=float(manifest.get('dt', 0.2)),
        v_min=float(manifest.get('v_min', 0.0)),
        v_max=float(manifest.get('v_max', 0.22)),
        w_min=float(manifest.get('w_min', -1.0)),
        w_max=float(manifest.get('w_max', 1.0)),
        control_weight=float(manifest.get('control_weight', 0.0)),
        process_noise_xy=float(manifest.get('process_noise_xy', 0.01)),
        process_noise_theta=float(manifest.get('process_noise_theta', 0.02)),
        obs_noise_uv=float(manifest.get('obs_noise_uv', 2.0)),
        goal_sigma_uv=float(manifest.get('goal_sigma_uv', 2.0)),
        risk_weight_obs=float(manifest.get('risk_weight_obs', 1.0)),
        ambiguity_weight=float(manifest.get('ambiguity_weight', 1.0)),
        optimizer_maxiter=int(manifest.get('optimizer_maxiter', 80)),
        optimizer_gtol=float(manifest.get('optimizer_gtol', 1e-4)),
        optimizer_warm_start=bool(manifest.get('optimizer_warm_start', True)),
        optimizer_maxfun=int(manifest.get('optimizer_maxfun', 500)),
        optimizer_ftol=float(manifest.get('optimizer_ftol', 1e-6)),
        approx_method='ET1',
        use_obs_risk=bool(manifest.get('use_obs_risk', True)),
        use_ambiguity=bool(manifest.get('use_ambiguity', True)),
        seed=int(manifest.get('seed', 0)),
        camera_params=camera_params,
        use_visibility_model=bool(manifest.get('use_visibility_model', True)),
        visibility_target_height_m=float(manifest.get('visibility_target_height_m', 0.0)),
        visibility_geometry_json=str(manifest.get('visibility_geometry_json', '') or ''),
        collision_geometry_json=str(manifest.get('collision_geometry_json', '') or ''),
        visibility_artifact_path=str(manifest.get('visibility_artifact_path', '') or ''),
        r_visible_uv=float(manifest.get('r_visible_uv', 2.5)),
        r_miss_uv=float(manifest.get('r_miss_uv', 120.0)),
        visibility_sigma_kappa=float(manifest.get('visibility_sigma_kappa', 1.0)),
        goal_prior_u_std_start=float(manifest.get('goal_prior_u_std_start', 80.0)),
        goal_prior_v_std_start=float(manifest.get('goal_prior_v_std_start', 80.0)),
        goal_prior_u_std_final=float(manifest.get('goal_prior_u_std_final', 18.0)),
        goal_prior_v_std_final=float(manifest.get('goal_prior_v_std_final', 18.0)),
        goal_tightening_power=float(manifest.get('goal_tightening_power', 0.45)),
        goal_progress_n_steps=int(manifest.get('goal_progress_n_steps', 90)),
        observation_risk_scale=float(manifest.get('observation_risk_scale', 1.25)),
        ambiguity_term_scale=float(manifest.get('ambiguity_term_scale', 1.0)),
        discount_gamma=float(manifest.get('discount_gamma', 0.98)),
        use_nogo_cost=bool(manifest.get('use_nogo_cost', False)),
        nogo_penalty_type=str(manifest.get('nogo_penalty_type', 'softplus')),
        nogo_weight=float(manifest.get('nogo_weight', 0.0)),
        nogo_safe_distance=float(manifest.get('nogo_safe_distance', 0.35)),
        nogo_gaussian_sigma=float(manifest.get('nogo_gaussian_sigma', 0.25)),
        nogo_softplus_scale=float(manifest.get('nogo_softplus_scale', 0.08)),
        nogo_logbarrier_scale=float(manifest.get('nogo_logbarrier_scale', 0.25)),
        nogo_logbarrier_eps=float(manifest.get('nogo_logbarrier_eps', 1e-3)),
        robot_collision_radius_m=float(manifest.get('robot_collision_radius_m', 0.125)),
        min_terminal_goal_progress_m=float(manifest.get('min_terminal_goal_progress_m', 0.0)),
        invalid_rollout_barrier_cost=float(manifest.get('invalid_rollout_barrier_cost', 1e6)),
    )


def _simulate_controls(m0: np.ndarray, horizon: int, dt: float, v: float, target_fn, v_max: float, w_min: float, w_max: float) -> np.ndarray:
    ensure_repo_python_paths()
    from planning.core.dynamics import unicycle_step

    state = np.asarray(m0, dtype=float).copy()
    controls = []
    for k in range(horizon):
        target = np.asarray(target_fn(k, state), dtype=float).reshape(2)
        desired = math.atan2(float(target[1] - state[1]), float(target[0] - state[0]))
        w = float(np.clip(1.35 * _wrap(desired - float(state[2])), w_min, w_max))
        u = np.array([float(np.clip(v, 0.0, v_max)), w], dtype=float)
        controls.append(u)
        state = unicycle_step(state, u, dt)
    return np.asarray(controls, dtype=float)


def _phase_controls(horizon: int, v: float, w_first: float, w_second: float, split_fraction: float) -> np.ndarray:
    split = int(np.clip(round(float(split_fraction) * horizon), 1, max(horizon - 1, 1)))
    w = np.full(horizon, float(w_second), dtype=float)
    w[:split] = float(w_first)
    return np.column_stack([np.full(horizon, float(v), dtype=float), w])


def _visibility_target(artifact: dict, m0: np.ndarray, goal_xy: np.ndarray, side: str) -> np.ndarray:
    try:
        xs = np.asarray(artifact['xs'], dtype=float)
        ys = np.asarray(artifact['ys'], dtype=float)
        p_map = _artifact_visibility_map(artifact)
    except (KeyError, TypeError, ValueError):
        y = 1.45 if side == 'upper' else -1.45
        return np.array([(float(m0[0]) + float(goal_xy[0])) * 0.5, y], dtype=float)

    x0 = float(m0[0])
    x1 = float(goal_xy[0])
    xmin = min(x0, x1) - 0.25
    xmax = max(x0, x1) + 0.85
    if side == 'upper':
        mask_y = ys >= max(float(m0[1]), -0.15)
    else:
        mask_y = ys <= min(float(m0[1]), 0.15)
    mask_x = (xs >= xmin) & (xs <= xmax)
    if not np.any(mask_x) or not np.any(mask_y):
        y = 1.45 if side == 'upper' else -1.45
        return np.array([(x0 + x1) * 0.5, y], dtype=float)
    sub = np.asarray(p_map[np.ix_(mask_y, mask_x)], dtype=float)
    if sub.size == 0 or not np.any(np.isfinite(sub)):
        y = 1.45 if side == 'upper' else -1.45
        return np.array([(x0 + x1) * 0.5, y], dtype=float)
    iy, ix = np.unravel_index(int(np.nanargmax(sub)), sub.shape)
    return np.array([xs[mask_x][ix], ys[mask_y][iy]], dtype=float)


def _candidate_library(planner, m0: np.ndarray, goal_xy: np.ndarray, artifact: dict | None = None) -> list[tuple[str, np.ndarray]]:
    horizon = int(planner.horizon)
    v = min(float(planner.v_max), 0.18)
    v_fast = min(float(planner.v_max), 0.22)
    v_slow = min(float(planner.v_max), 0.10)
    moderate = min(0.33, abs(float(planner.w_max)) * 0.45)
    strong = min(0.85, abs(float(planner.w_max)) * 0.90)
    gentle = min(0.18, abs(float(planner.w_max)) * 0.25)
    controls = []
    controls.append(('straight_to_goal', np.column_stack([np.full(horizon, v), np.zeros(horizon)])))
    controls.append(('straight_to_goal_fast', _simulate_controls(m0, horizon, planner.dt, v_fast, lambda _k, _s: goal_xy, planner.v_max, planner.w_min, planner.w_max)))
    controls.append(('straight_to_goal_slow', _simulate_controls(m0, horizon, planner.dt, v_slow, lambda _k, _s: goal_xy, planner.v_max, planner.w_min, planner.w_max)))
    controls.append(('wide_upper_arc', np.column_stack([np.full(horizon, v), np.full(horizon, moderate)])))
    controls.append(('wide_lower_arc', np.column_stack([np.full(horizon, v), np.full(horizon, -moderate)])))
    controls.append(('hard_upper_arc', np.column_stack([np.full(horizon, v), np.full(horizon, strong)])))
    controls.append(('hard_lower_arc', np.column_stack([np.full(horizon, v), np.full(horizon, -strong)])))
    controls.append(('turn_then_upper_commit', _phase_controls(horizon, v, strong, 0.0, 0.30)))
    controls.append(('turn_then_lower_commit', _phase_controls(horizon, v, -strong, 0.0, 0.30)))
    controls.append(('turn_then_upper_commit_fast', _phase_controls(horizon, v_fast, strong, 0.0, 0.22)))
    controls.append(('turn_then_lower_commit_fast', _phase_controls(horizon, v_fast, -strong, 0.0, 0.22)))
    controls.append(('turn_then_upper_commit_wide', _phase_controls(horizon, v_fast, strong, gentle, 0.25)))
    controls.append(('turn_then_lower_commit_wide', _phase_controls(horizon, v_fast, -strong, -gentle, 0.25)))

    upper_target = _visibility_target(artifact or {}, m0, goal_xy, 'upper')
    lower_target = _visibility_target(artifact or {}, m0, goal_xy, 'lower')
    controls.append((
        'visible_recover_upper',
        _simulate_controls(
            m0,
            horizon,
            planner.dt,
            v,
            lambda k, _s: upper_target if k < int(0.52 * horizon) else goal_xy,
            planner.v_max,
            planner.w_min,
            planner.w_max,
        ),
    ))
    controls.append((
        'visible_recover_lower',
        _simulate_controls(
            m0,
            horizon,
            planner.dt,
            v,
            lambda k, _s: lower_target if k < int(0.52 * horizon) else goal_xy,
            planner.v_max,
            planner.w_min,
            planner.w_max,
        ),
    ))
    controls.append((
        'visible_recover_upper_fast',
        _simulate_controls(
            m0,
            horizon,
            planner.dt,
            v_fast,
            lambda k, _s: upper_target if k < int(0.42 * horizon) else goal_xy,
            planner.v_max,
            planner.w_min,
            planner.w_max,
        ),
    ))
    controls.append((
        'visible_recover_lower_fast',
        _simulate_controls(
            m0,
            horizon,
            planner.dt,
            v_fast,
            lambda k, _s: lower_target if k < int(0.42 * horizon) else goal_xy,
            planner.v_max,
            planner.w_min,
            planner.w_max,
        ),
    ))
    return controls


def _probe_indices(t: np.ndarray, p_vis: np.ndarray, xs: np.ndarray, probe_times: list[float]) -> list[tuple[str, int]]:
    finite = np.where(np.isfinite(t) & (t >= 0.0))[0]
    if finite.size == 0:
        raise RuntimeError('No nonnegative run timestamps available for probing')
    if probe_times:
        return [(f't{value:.1f}s', int(finite[np.argmin(np.abs(t[finite] - value))])) for value in probe_times]
    duration = float(np.nanmax(t[finite]) - np.nanmin(t[finite]))
    idxs: list[tuple[str, int]] = []
    idxs.append(('early_split', int(finite[np.argmin(np.abs(t[finite] - max(2.0, 0.20 * duration)))])))
    if p_vis.size:
        pv = _safe(p_vis, t.size)
        valid = finite[np.isfinite(pv[finite])]
        if valid.size >= 3:
            diffs = np.diff(pv[valid])
            idxs.append(('pvis_drop', int(valid[int(np.argmin(diffs)) + 1])))
        else:
            idxs.append(('pvis_drop', int(finite[len(finite) // 2])))
    else:
        idxs.append(('pvis_drop', int(finite[len(finite) // 2])))
    valid_x = finite[np.isfinite(xs[finite])]
    commit = None
    if valid_x.size:
        crossed = valid_x[xs[valid_x] > -0.65]
        if crossed.size:
            commit = int(crossed[0])
    if commit is None:
        commit = int(finite[np.argmin(np.abs(t[finite] - 0.55 * duration))])
    idxs.append(('pre_commit', commit))
    seen = set()
    out = []
    for name, idx in idxs:
        if idx in seen:
            continue
        seen.add(idx)
        out.append((name, idx))
    return out


def _selected_summary(planner, exp: dict[str, np.ndarray], idx: int, selected_plan: np.ndarray, m0: np.ndarray, goal_xy: np.ndarray) -> dict:
    mean_p = math.nan
    mean_pe = math.nan
    mean_ru = math.nan
    mean_rv = math.nan
    if selected_plan.size:
        p_vals = []
        pe_vals = []
        ru_vals = []
        rv_vals = []
        S = np.diag([1e-6, 1e-6, 1e-6])
        theta = float(m0[2])
        for xy in selected_plan:
            diag = planner.planning_visibility_diagnostics(np.array([float(xy[0]), float(xy[1]), theta], dtype=float), S)
            p_vals.append(diag['p_vis'])
            pe_vals.append(diag['p_vis_eff'])
            ru_vals.append(diag['r_plan_u_std'])
            rv_vals.append(diag['r_plan_v_std'])
        mean_p = float(np.mean(p_vals))
        mean_pe = float(np.mean(pe_vals))
        mean_ru = float(np.mean(ru_vals))
        mean_rv = float(np.mean(rv_vals))
    return {
        'family': 'optimizer_selected',
        'selected': 1,
        'total_cost': _at(_col(exp, 'efe_total'), idx),
        'risk_cost': _at(_col(exp, 'efe_risk'), idx),
        'ambiguity_cost': _at(_col(exp, 'efe_ambiguity'), idx),
        'obstacle_cost': _at(_col(exp, 'efe_obstacle'), idx),
        'control_cost': _at(_col(exp, 'efe_control'), idx),
        'risk_mean': _at(_col(exp, 'efe_risk_mean'), idx),
        'risk_cov_trace': _at(_col(exp, 'efe_risk_cov_trace'), idx),
        'risk_cov_logdet': _at(_col(exp, 'efe_risk_cov_logdet'), idx),
        'delta_risk_visibility': _at(_col(exp, 'efe_delta_risk_visibility'), idx),
        'delta_ambiguity_visibility': _at(_col(exp, 'efe_delta_ambiguity_visibility'), idx),
        'terminal_goal_distance_pred': _at(_col(exp, 'terminal_goal_distance_pred'), idx),
        'terminal_goal_progress_m': _at(_col(exp, 'terminal_goal_progress_m'), idx),
        'fraction_horizon_low_pvis': _at(_col(exp, 'fraction_horizon_low_pvis'), idx),
        'fraction_horizon_high_ambiguity': _at(_col(exp, 'fraction_horizon_high_ambiguity'), idx),
        'min_predicted_obstacle_distance_m': _at(_col(exp, 'min_predicted_obstacle_distance_m'), idx),
        'rollout_valid': bool(_at(_col(exp, 'rollout_valid'), idx, 1.0) >= 0.5),
        'fallback_stop_applied': bool(_at(_col(exp, 'fallback_stop_applied'), idx, 0.0) >= 0.5),
        'mean_p_vis_plan': mean_p,
        'mean_p_vis_plan_eff': mean_pe,
        'mean_r_plan_u_std': mean_ru,
        'mean_r_plan_v_std': mean_rv,
        'states': selected_plan,
        'goal_progress_advantage': 0.0,
        'observability_advantage': 0.0,
    }


def _annotate_progress_bands(candidates: list[dict], tolerance: float = 0.20) -> None:
    progress_values = [
        float(item.get('terminal_goal_progress_m', math.nan))
        for item in candidates
        if math.isfinite(float(item.get('terminal_goal_progress_m', math.nan)))
    ]
    if not progress_values:
        for item in candidates:
            item['matched_progress_count'] = 0
            item['matched_best_total_rank'] = math.nan
            item['matched_best_mean_p_vis_plan'] = math.nan
            item['matched_best_family'] = ''
            item['matched_selected_wins_total'] = math.nan
        return
    for item in candidates:
        progress = float(item.get('terminal_goal_progress_m', math.nan))
        if not math.isfinite(progress):
            band = []
        else:
            band = [
                other for other in candidates
                if math.isfinite(float(other.get('terminal_goal_progress_m', math.nan)))
                and abs(float(other.get('terminal_goal_progress_m')) - progress) <= tolerance
            ]
        band_total_sorted = sorted(
            band,
            key=lambda other: float(other.get('total_cost', math.inf))
            if math.isfinite(float(other.get('total_cost', math.inf)))
            else math.inf,
        )
        band_vis_sorted = sorted(
            band,
            key=lambda other: float(other.get('mean_p_vis_plan', -math.inf))
            if math.isfinite(float(other.get('mean_p_vis_plan', -math.inf)))
            else -math.inf,
            reverse=True,
        )
        item['matched_progress_count'] = len(band)
        item['matched_best_family'] = str(band_total_sorted[0].get('family', '')) if band_total_sorted else ''
        item['matched_best_mean_p_vis_plan'] = (
            float(band_vis_sorted[0].get('mean_p_vis_plan', math.nan)) if band_vis_sorted else math.nan
        )
        item['matched_best_total_cost'] = (
            float(band_total_sorted[0].get('total_cost', math.nan)) if band_total_sorted else math.nan
        )
        rank = math.nan
        for rank_idx, other in enumerate(band_total_sorted, start=1):
            if other is item:
                rank = float(rank_idx)
                break
        item['matched_best_total_rank'] = rank
        selected = next((other for other in band if bool(other.get('selected', 0))), None)
        if selected is None or not band_total_sorted:
            item['matched_selected_wins_total'] = math.nan
        else:
            item['matched_selected_wins_total'] = 1.0 if selected is band_total_sorted[0] else 0.0


def _plot_probe(out_path: Path, title: str, artifact: dict, p_map: np.ndarray, ambiguity_bg: np.ndarray, extent, geometry, selected: dict, candidates: list[dict], m0: np.ndarray, truth_xy: np.ndarray, goal_xy: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    colors = {
        'optimizer_selected': 'black',
        'straight_to_goal': '#1f77b4',
        'wide_upper_arc': '#2ca02c',
        'wide_lower_arc': '#d62728',
        'hard_upper_arc': '#008000',
        'hard_lower_arc': '#8b0000',
        'turn_then_upper_commit': '#17becf',
        'turn_then_lower_commit': '#e377c2',
        'turn_then_upper_commit_fast': '#00a6d6',
        'turn_then_lower_commit_fast': '#d64aa6',
        'turn_then_upper_commit_wide': '#005f73',
        'turn_then_lower_commit_wide': '#9b287b',
        'visible_recover_upper': '#bcbd22',
        'visible_recover_lower': '#9467bd',
        'visible_recover_upper_fast': '#808000',
        'visible_recover_lower_fast': '#6a3d9a',
    }
    for ax, bg, cmap, name in (
        (axes[0], p_map, 'viridis', 'P_vis background'),
        (axes[1], ambiguity_bg, 'magma', 'ambiguity background'),
    ):
        ax.imshow(bg, origin='lower', extent=extent, cmap=cmap, aspect='equal', alpha=0.88)
        try:
            ax.contour(p_map, levels=[0.2], origin='lower', extent=extent, colors='cyan', linewidths=1.0)
        except ValueError:
            pass
        _draw_geometry(ax, geometry)
        for item in candidates:
            states = np.asarray(item['states'], dtype=float)
            if states.ndim != 2 or states.shape[0] < 2:
                continue
            family = item['family']
            lw = 3.2 if item.get('selected') else 1.8
            alpha = 0.95 if item.get('selected') else 0.78
            if states.shape[1] >= 3:
                xs = states[:, 0]
                ys = states[:, 1]
            else:
                xs = states[:, 0]
                ys = states[:, 1]
            ax.plot(xs, ys, color=colors.get(family, 'gray'), linewidth=lw, alpha=alpha, label=family)
        ax.scatter([m0[0]], [m0[1]], color='white', edgecolor='black', s=70, zorder=5, label='belief')
        if np.all(np.isfinite(truth_xy)):
            ax.scatter([truth_xy[0]], [truth_xy[1]], color='black', marker='x', s=80, zorder=5, label='truth')
        ax.scatter([goal_xy[0]], [goal_xy[1]], color='gold', edgecolor='black', marker='*', s=180, zorder=5, label='goal')
        ax.set_title(name)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.legend(fontsize=7, loc='lower left')

    axes[2].axis('off')
    rows = []
    selected_mean_p = float(selected.get('mean_p_vis_plan', math.nan))
    selected_progress = float(selected.get('terminal_goal_progress_m', math.nan))
    for item in candidates:
        rows.append([
            item['family'].replace('_', ' '),
            f"{item['total_cost']:.1f}" if math.isfinite(float(item['total_cost'])) else 'nan',
            f"{item['risk_cost']:.1f}" if math.isfinite(float(item['risk_cost'])) else 'nan',
            f"{item.get('risk_mean', math.nan):.1f}" if math.isfinite(float(item.get('risk_mean', math.nan))) else 'nan',
            f"{item.get('risk_cov_trace', math.nan):.1f}" if math.isfinite(float(item.get('risk_cov_trace', math.nan))) else 'nan',
            f"{item.get('risk_cov_logdet', math.nan):.1f}" if math.isfinite(float(item.get('risk_cov_logdet', math.nan))) else 'nan',
            f"{item.get('delta_risk_visibility', math.nan):.1f}" if math.isfinite(float(item.get('delta_risk_visibility', math.nan))) else 'nan',
            f"{item['ambiguity_cost']:.1f}" if math.isfinite(float(item['ambiguity_cost'])) else 'nan',
            f"{item['obstacle_cost']:.1f}" if math.isfinite(float(item['obstacle_cost'])) else 'nan',
            f"{item['terminal_goal_progress_m']:.2f}" if math.isfinite(float(item['terminal_goal_progress_m'])) else 'nan',
            f"{item['mean_p_vis_plan']:.2f}" if math.isfinite(float(item['mean_p_vis_plan'])) else 'nan',
            f"{item['fraction_horizon_low_pvis']:.2f}" if math.isfinite(float(item['fraction_horizon_low_pvis'])) else 'nan',
            f"{item.get('matched_best_total_rank', math.nan):.0f}" if math.isfinite(float(item.get('matched_best_total_rank', math.nan))) else 'nan',
            'yes' if item['rollout_valid'] else 'no',
        ])
    table = axes[2].table(
        cellText=rows,
        colLabels=['family', 'total', 'risk', 'r_mean', 'r_covT', 'r_logD', 'dRvis', 'amb', 'obs', 'prog', 'mean p', 'low p', 'band rank', 'valid'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    table.scale(1.0, 1.35)
    axes[2].set_title(
        'Rollout accounting\n'
        f"selected mean_p={selected_mean_p:.2f}, selected progress={selected_progress:.2f}"
    )
    fig.suptitle(title)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate fixed rollout families at selected decision frames.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out', default='')
    parser.add_argument('--probe-times', nargs='*', type=float, default=[])
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f'Run directory not found: {run_dir}')
    out_dir = Path(args.out).expanduser().resolve() if args.out else (run_dir / 'rollout_family_probe').resolve()
    allowed = LOGS_ROOT.resolve()
    if allowed not in out_dir.parents and run_dir not in out_dir.parents and out_dir != run_dir / 'rollout_family_probe':
        raise RuntimeError(f'Output must stay under {allowed} or inside run directory: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(run_dir / 'run_manifest.json')
    summary = _load_json(run_dir / 'run_summary.json')
    exp = _read_csv_columns(run_dir / 'experiment.csv')
    if not exp:
        raise RuntimeError(f'Missing experiment.csv in {run_dir}')

    planner = _build_planner(run_dir, manifest)
    artifact_path = Path(str(manifest.get('visibility_artifact_path', '') or '')).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = (Path.cwd() / artifact_path).resolve()
    artifact = _load_artifact(artifact_path)
    xs = np.asarray(artifact['xs'], dtype=float)
    ys = np.asarray(artifact['ys'], dtype=float)
    p_map = _artifact_visibility_map(artifact)
    ambiguity_bg = _ambiguity_map(planner, p_map)
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    geometry = _parse_geometry_json(str(manifest.get('collision_geometry_json', '') or manifest.get('visibility_geometry_json', '') or artifact.get('geometry_json', '')))

    stamp = _col(exp, 'stamp')
    first_cmd = summary.get('first_cmd_stamp', math.nan)
    try:
        first_cmd = float(first_cmd)
    except (TypeError, ValueError):
        first_cmd = math.nan
    if not math.isfinite(first_cmd):
        finite_stamp = stamp[np.isfinite(stamp)]
        first_cmd = float(finite_stamp[0]) if finite_stamp.size else 0.0
    t = stamp - first_cmd
    belief_x = _safe(_col(exp, 'planner_belief_x', 'est_x'), stamp.size)
    belief_y = _safe(_col(exp, 'planner_belief_y', 'est_y'), stamp.size)
    belief_yaw = _safe(_col(exp, 'planner_belief_yaw', 'est_yaw'), stamp.size)
    cov_x = _safe(_col(exp, 'planner_cov_x', 'est_cov_xx', 'state_cov_xx'), stamp.size, 1e-6)
    cov_xy = _safe(_col(exp, 'planner_cov_xy', 'est_cov_xy', 'state_cov_xy'), stamp.size, 0.0)
    cov_y = _safe(_col(exp, 'planner_cov_y', 'est_cov_yy', 'state_cov_yy'), stamp.size, 1e-6)
    cov_yaw = _safe(_col(exp, 'planner_cov_yaw', 'state_cov_yaw'), stamp.size, 1e-4)
    goal_x = _safe(_col(exp, 'goal_x'), stamp.size)
    goal_y = _safe(_col(exp, 'goal_y'), stamp.size)
    truth_x = _safe(_col(exp, 'truth_x'), stamp.size)
    truth_y = _safe(_col(exp, 'truth_y'), stamp.size)
    p_vis_plan = _safe(_col(exp, 'p_vis_plan'), stamp.size)
    plan_stamps, plans = _load_plan_groups(run_dir / 'plan_samples.csv')

    probe_indices = _probe_indices(t, p_vis_plan, belief_x, list(args.probe_times or []))
    rows = []
    matched_rows = []
    figures = []
    for probe_name, idx in probe_indices:
        if not (math.isfinite(belief_x[idx]) and math.isfinite(belief_y[idx]) and math.isfinite(belief_yaw[idx])):
            continue
        m0 = np.array([belief_x[idx], belief_y[idx], belief_yaw[idx]], dtype=float)
        S0 = np.array([
            [max(_at(cov_x, idx, 1e-6), 1e-9), _at(cov_xy, idx, 0.0), 0.0],
            [_at(cov_xy, idx, 0.0), max(_at(cov_y, idx, 1e-6), 1e-9), 0.0],
            [0.0, 0.0, max(_at(cov_yaw, idx, 1e-4), 1e-9)],
        ], dtype=float)
        goal_xy = np.array([_at(goal_x, idx), _at(goal_y, idx)], dtype=float)
        if not np.all(np.isfinite(goal_xy)):
            continue
        start_pose = manifest.get('task_start_pose') or {}
        start_dist = math.hypot(float(start_pose.get('x', m0[0])) - goal_xy[0], float(start_pose.get('y', m0[1])) - goal_xy[1])
        current_dist = math.hypot(float(m0[0]) - goal_xy[0], float(m0[1]) - goal_xy[1])
        progress_fraction = max(min((start_dist - current_dist) / max(start_dist, 1e-9), 1.0), 0.0)
        progress_index = progress_fraction * float(max(planner.goal_progress_n_steps, 1))
        selected_plan = _nearest_plan(plan_stamps, plans, _at(stamp, idx, 0.0))
        selected = _selected_summary(planner, exp, idx, selected_plan, m0, goal_xy)

        candidates = [selected]
        for family, controls in _candidate_library(planner, m0, goal_xy, artifact):
            item = planner.evaluate_rollout_controls(m0, S0, goal_xy, controls, progress_index=progress_index)
            item['family'] = family
            item['selected'] = 0
            item['observability_advantage'] = float(item['mean_p_vis_plan'] - selected['mean_p_vis_plan']) if math.isfinite(selected['mean_p_vis_plan']) else math.nan
            item['goal_progress_advantage'] = float(selected['terminal_goal_progress_m'] - item['terminal_goal_progress_m']) if math.isfinite(selected['terminal_goal_progress_m']) else math.nan
            candidates.append(item)
        _annotate_progress_bands(candidates, tolerance=0.20)

        for reference in candidates:
            ref_progress = float(reference.get('terminal_goal_progress_m', math.nan))
            if not math.isfinite(ref_progress):
                continue
            band = [
                other for other in candidates
                if math.isfinite(float(other.get('terminal_goal_progress_m', math.nan)))
                and abs(float(other.get('terminal_goal_progress_m')) - ref_progress) <= 0.20
            ]
            band_sorted = sorted(
                band,
                key=lambda other: float(other.get('total_cost', math.inf))
                if math.isfinite(float(other.get('total_cost', math.inf)))
                else math.inf,
            )
            for rank_idx, other in enumerate(band_sorted, start=1):
                matched_rows.append({
                    'probe_name': probe_name,
                    'reference_family': reference.get('family', ''),
                    'candidate_family': other.get('family', ''),
                    'reference_progress_m': ref_progress,
                    'candidate_progress_m': float(other.get('terminal_goal_progress_m', math.nan)),
                    'progress_delta_m': float(other.get('terminal_goal_progress_m', math.nan)) - ref_progress,
                    'rank_by_total_cost_in_band': rank_idx,
                    'total_cost': float(other.get('total_cost', math.nan)),
                    'risk_cost': float(other.get('risk_cost', math.nan)),
                    'risk_mean': float(other.get('risk_mean', math.nan)),
                    'risk_cov_trace': float(other.get('risk_cov_trace', math.nan)),
                    'risk_cov_logdet': float(other.get('risk_cov_logdet', math.nan)),
                    'delta_risk_visibility': float(other.get('delta_risk_visibility', math.nan)),
                    'ambiguity_cost': float(other.get('ambiguity_cost', math.nan)),
                    'delta_ambiguity_visibility': float(other.get('delta_ambiguity_visibility', math.nan)),
                    'obstacle_cost': float(other.get('obstacle_cost', math.nan)),
                    'mean_p_vis_plan': float(other.get('mean_p_vis_plan', math.nan)),
                    'mean_p_vis_plan_eff': float(other.get('mean_p_vis_plan_eff', math.nan)),
                    'fraction_horizon_low_pvis': float(other.get('fraction_horizon_low_pvis', math.nan)),
                    'rollout_valid': int(bool(other.get('rollout_valid', True))),
                    'selected': int(bool(other.get('selected', 0))),
                })

        for item in candidates:
            row = {
                'probe_name': probe_name,
                'frame_idx': idx,
                'stamp': _at(stamp, idx),
                'time_after_first_cmd_s': _at(t, idx),
                'belief_x': float(m0[0]),
                'belief_y': float(m0[1]),
                'belief_yaw': float(m0[2]),
                'truth_x': _at(truth_x, idx),
                'truth_y': _at(truth_y, idx),
                'goal_x': float(goal_xy[0]),
                'goal_y': float(goal_xy[1]),
                'family': item['family'],
                'selected': int(item.get('selected', 0)),
            }
            for key in (
                'total_cost', 'risk_cost', 'ambiguity_cost', 'obstacle_cost', 'control_cost',
                'risk_mean', 'risk_cov_trace', 'risk_cov_logdet',
                'delta_risk_visibility', 'delta_ambiguity_visibility',
                'terminal_goal_distance_pred', 'terminal_goal_progress_m', 'fraction_horizon_low_pvis',
                'fraction_horizon_high_ambiguity', 'min_predicted_obstacle_distance_m',
                'mean_p_vis_plan', 'mean_p_vis_plan_eff', 'mean_r_plan_u_std', 'mean_r_plan_v_std',
                'observability_advantage', 'goal_progress_advantage',
                'matched_progress_count', 'matched_best_total_rank', 'matched_best_mean_p_vis_plan',
                'matched_best_total_cost', 'matched_selected_wins_total',
            ):
                row[key] = item.get(key, math.nan)
            row['matched_best_family'] = item.get('matched_best_family', '')
            row['rollout_valid'] = int(bool(item.get('rollout_valid', True)))
            row['fallback_stop_applied'] = int(bool(item.get('fallback_stop_applied', False)))
            rows.append(row)

        fig_path = out_dir / f'{probe_name}_rollout_comparison.png'
        _plot_probe(
            fig_path,
            f'{run_dir.name} | {probe_name} | t={_at(t, idx):.1f}s',
            artifact,
            p_map,
            ambiguity_bg,
            extent,
            geometry,
            selected,
            candidates,
            m0,
            np.array([_at(truth_x, idx), _at(truth_y, idx)], dtype=float),
            goal_xy,
        )
        figures.append(str(fig_path))

    fieldnames = [
        'probe_name', 'frame_idx', 'stamp', 'time_after_first_cmd_s',
        'belief_x', 'belief_y', 'belief_yaw', 'truth_x', 'truth_y', 'goal_x', 'goal_y',
        'family', 'selected',
        'total_cost', 'risk_cost', 'ambiguity_cost', 'obstacle_cost', 'control_cost',
        'risk_mean', 'risk_cov_trace', 'risk_cov_logdet',
        'delta_risk_visibility', 'delta_ambiguity_visibility',
        'terminal_goal_distance_pred', 'terminal_goal_progress_m',
        'fraction_horizon_low_pvis', 'fraction_horizon_high_ambiguity',
        'min_predicted_obstacle_distance_m', 'mean_p_vis_plan', 'mean_p_vis_plan_eff',
        'mean_r_plan_u_std', 'mean_r_plan_v_std',
        'observability_advantage', 'goal_progress_advantage',
        'matched_progress_count', 'matched_best_total_rank', 'matched_best_family',
        'matched_best_mean_p_vis_plan', 'matched_best_total_cost', 'matched_selected_wins_total',
        'rollout_valid', 'fallback_stop_applied',
    ]
    write_csv(out_dir / 'rollout_family_probe.csv', fieldnames, rows)
    write_csv(out_dir / 'rollout_matched_comparison.csv', [
        'probe_name', 'reference_family', 'candidate_family',
        'reference_progress_m', 'candidate_progress_m', 'progress_delta_m',
        'rank_by_total_cost_in_band',
        'total_cost', 'risk_cost', 'risk_mean', 'risk_cov_trace', 'risk_cov_logdet',
        'delta_risk_visibility', 'ambiguity_cost', 'delta_ambiguity_visibility',
        'obstacle_cost', 'mean_p_vis_plan', 'mean_p_vis_plan_eff',
        'fraction_horizon_low_pvis', 'rollout_valid', 'selected',
    ], matched_rows)
    write_manifest(out_dir / 'rollout_family_probe_manifest.json', {
        'run_dir': str(run_dir),
        'artifact_path': str(artifact_path),
        'probes': [{'name': name, 'frame_idx': idx, 'time_after_first_cmd_s': _at(t, idx)} for name, idx in probe_indices],
        'families': [name for name, _controls in _candidate_library(planner, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0]), artifact)],
        'matched_progress_tolerance_m': 0.20,
        'figures': figures,
        'notes': [
            'Candidate rollouts are fixed deterministic probes; they are not optimizer seeds.',
            'Costs use the same planner evaluate_rollout_controls accounting as runtime selection.',
            'optimizer_selected row uses logged selected-plan metrics plus mean visibility computed over logged plan points.',
            'rollout_matched_comparison.csv compares candidates within +/-0.20 m terminal progress bands.',
        ],
    })
    print(f'Wrote rollout-family probe to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
