#!/usr/bin/env python3
"""Compare the notebook-simple JAX reference path against the CasADi backend."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extend_sys_path(repo_root: Path) -> None:
    for rel_path in ('src/planning', 'src/experiments', 'src/unav_common'):
        candidate = repo_root / rel_path
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    install_root = repo_root / 'install'
    if install_root.is_dir():
        prefixes = [str(p) for p in install_root.iterdir() if p.is_dir()]
        existing = [p for p in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep) if p]
        merged = prefixes + [p for p in existing if p not in prefixes]
        os.environ['AMENT_PREFIX_PATH'] = os.pathsep.join(merged)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _select_snapshot_row(rows, row_index):
    if not rows:
        raise RuntimeError("experiment.csv contains no rows")
    if row_index is not None:
        return rows[row_index], row_index

    for idx, row in enumerate(rows):
        try:
            if float(row.get('plan_length', '0') or 0.0) > 0.0:
                return row, idx
        except ValueError:
            continue
    return rows[-1], len(rows) - 1


def _float(row, key, default=math.nan):
    try:
        value = row.get(key, default)
        if value in (None, ''):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _build_launch_like_cfg(manifest, repo_root: Path):
    from experiments.core.visibility_launch_common import (
        PAPER_LAUNCH_DEFAULTS,
        VISIBILITY_FALLBACK_DEFAULTS,
        resolve_world_setup,
    )

    cfg = dict(PAPER_LAUNCH_DEFAULTS)
    cfg.update(dict(VISIBILITY_FALLBACK_DEFAULTS))
    cfg.update(dict(manifest))
    cfg['task_name'] = str(manifest['task'])
    cfg['world_profiles_path'] = str(repo_root / 'src/experiments/config/world_profiles.yaml')
    cfg['tasks_yaml'] = str(repo_root / 'src/experiments/config/tasks.yaml')
    cfg['use_sim_time'] = True
    cfg.setdefault('visibility_geometry_json', '')
    return resolve_world_setup(cfg)


def _planner_kwargs_from_cfg(cfg, *, optimizer_backend):
    planner = str(cfg['planner']).strip().lower()
    if planner == 'efe2':
        approx_method = 'ET1' if str(cfg['math_mode']).strip().lower() == 'notebook_simple' else 'ET2'
        use_ambiguity = _coerce_bool(cfg['use_ambiguity'])
        use_obs_risk = _coerce_bool(cfg['use_obs_risk'])
    elif planner == 'efer':
        approx_method = 'ET1' if str(cfg['math_mode']).strip().lower() == 'notebook_simple' else 'ET2'
        use_ambiguity = False
        use_obs_risk = True
    elif planner == 'mpc':
        approx_method = 'ET1'
        use_ambiguity = False
        use_obs_risk = True
    else:
        approx_method = 'ET1'
        use_ambiguity = _coerce_bool(cfg['use_ambiguity'])
        use_obs_risk = _coerce_bool(cfg['use_obs_risk'])

    return {
        'horizon': int(cfg['horizon']),
        'dt': float(cfg['dt']),
        'v_min': float(cfg.get('v_min', 0.0)),
        'v_max': float(cfg.get('v_max', 0.22)),
        'w_min': float(cfg.get('w_min', -1.0)),
        'w_max': float(cfg.get('w_max', 1.0)),
        'control_weight': float(cfg['control_weight']),
        'process_noise_xy': float(cfg['process_noise_xy']),
        'process_noise_theta': float(cfg['process_noise_theta']),
        'obs_noise_uv': float(cfg['obs_noise_uv']),
        'goal_sigma_xy': float(cfg.get('goal_sigma_xy', 0.25)),
        'goal_sigma_theta': float(cfg.get('goal_sigma_theta', 0.5)),
        'goal_sigma_uv': float(cfg['goal_sigma_uv']),
        'risk_weight_state': float(cfg['risk_weight_state']),
        'risk_weight_obs': float(cfg['risk_weight_obs']),
        'ambiguity_weight': float(cfg['ambiguity_weight']),
        'optimizer_maxiter': int(cfg['optimizer_maxiter']),
        'optimizer_gtol': float(cfg['optimizer_gtol']),
        'optimizer_warm_start': _coerce_bool(cfg['optimizer_warm_start']),
        'optimizer_warm_start_shift_steps': 1,
        'approx_method': approx_method,
        'use_obs_risk': use_obs_risk,
        'use_ambiguity': use_ambiguity,
        'math_mode': str(cfg['math_mode']),
        'optimizer_backend': optimizer_backend,
        'seed': int(cfg['seed']),
        'camera_params': cfg['camera_params'],
        'use_visibility_model': _coerce_bool(cfg['use_visibility_model']),
        'visibility_model': str(cfg['visibility_model']),
        'visibility_weight': float(cfg['visibility_weight']),
        'visibility_map_min_x': float(cfg['visibility_map_min_x']),
        'visibility_map_max_x': float(cfg['visibility_map_max_x']),
        'visibility_map_min_y': float(cfg['visibility_map_min_y']),
        'visibility_map_max_y': float(cfg['visibility_map_max_y']),
        'visibility_map_nx': int(cfg['visibility_map_nx']),
        'visibility_map_ny': int(cfg['visibility_map_ny']),
        'visibility_gp_length_scale': float(cfg['visibility_gp_length_scale']),
        'visibility_gp_noise_var': float(cfg['visibility_gp_noise_var']),
        'visibility_prior_occ': float(cfg['visibility_prior_occ']),
        'visibility_beta': float(cfg['visibility_beta']),
        'visibility_height_tau': float(cfg['visibility_height_tau']),
        'visibility_ray_samples': int(cfg['visibility_ray_samples']),
        'visibility_target_height_m': float(cfg['visibility_target_height_m']),
        'visibility_geometry_json': str(cfg['visibility_geometry_json']),
        'visibility_gp_seed': int(cfg['seed']),
        'visibility_r_bad_uv': float(cfg['visibility_r_bad_uv']),
        'visibility_cov_pos_scale': float(cfg['visibility_cov_pos_scale']),
        'visibility_cov_theta_scale': float(cfg['visibility_cov_theta_scale']),
        'r_visible_uv': float(cfg['r_visible_uv']),
        'r_miss_uv': float(cfg['r_miss_uv']),
        'visibility_power': float(cfg['visibility_power']),
        'visibility_sigma_kappa': float(cfg['visibility_sigma_kappa']),
        'goal_prior_u_std_start': float(cfg['goal_prior_u_std_start']),
        'goal_prior_v_std_start': float(cfg['goal_prior_v_std_start']),
        'goal_prior_u_std_final': float(cfg['goal_prior_u_std_final']),
        'goal_prior_v_std_final': float(cfg['goal_prior_v_std_final']),
        'goal_tightening_power': float(cfg['goal_tightening_power']),
        'goal_progress_n_steps': int(cfg['goal_progress_n_steps']),
        'notebook_risk_scale': float(cfg['notebook_risk_scale']),
        'notebook_ambiguity_scale': float(cfg['notebook_ambiguity_scale']),
        'discount_gamma': float(cfg.get('discount_gamma', 0.98)),
        'optimizer_maxfun': int(cfg['optimizer_maxfun']),
        'optimizer_ftol': float(cfg['optimizer_ftol']),
        'use_nogo_cost': _coerce_bool(cfg['use_nogo_cost']),
        'nogo_penalty_type': str(cfg['nogo_penalty_type']),
        'nogo_weight': float(cfg['nogo_weight']),
        'nogo_safe_distance': float(cfg['nogo_safe_distance']),
        'nogo_gaussian_sigma': float(cfg['nogo_gaussian_sigma']),
        'nogo_softplus_scale': float(cfg['nogo_softplus_scale']),
        'nogo_logbarrier_scale': float(cfg['nogo_logbarrier_scale']),
        'nogo_logbarrier_eps': float(cfg['nogo_logbarrier_eps']),
        'runtime_debug': False,
    }


def _snapshot_state(row):
    belief_x = _float(row, 'planner_belief_x')
    belief_y = _float(row, 'planner_belief_y')
    belief_yaw = _float(row, 'planner_belief_yaw')
    if not np.all(np.isfinite([belief_x, belief_y, belief_yaw])):
        belief_x = _float(row, 'x')
        belief_y = _float(row, 'y')
        belief_yaw = _float(row, 'yaw')

    cov_x = _float(row, 'planner_cov_x')
    cov_y = _float(row, 'planner_cov_y')
    cov_yaw = _float(row, 'planner_cov_yaw')
    if not np.all(np.isfinite([cov_x, cov_y, cov_yaw])):
        cov_x = _float(row, 'cov_x')
        cov_y = _float(row, 'cov_y')
        cov_yaw = _float(row, 'cov_yaw')

    m0 = np.array([belief_x, belief_y, belief_yaw], dtype=float)
    S0 = np.diag([
        max(cov_x, 1e-9),
        max(cov_y, 1e-9),
        max(cov_yaw, 1e-9),
    ]).astype(float)
    goal_xy = (_float(row, 'goal_x'), _float(row, 'goal_y'))
    return m0, S0, goal_xy


def _seed_eval(planner, backend_name, m0, S0, goal_xy, progress_index):
    (
        goal_state,
        goal_cov,
        goal_obs,
        goal_obs_cov,
        use_observation_risk,
        use_ambiguity_term,
        use_state_risk,
    ) = planner._resolve_plan_problem(m0, goal_xy)
    x0 = planner._initial_controls_flat()

    if backend_name == 'casadi':
        valgrad = planner._get_casadi_valgrad(
            goal_state,
            goal_cov,
            goal_obs,
            goal_obs_cov,
            use_observation_risk=use_observation_risk,
            use_ambiguity_term=use_ambiguity_term,
            use_state_risk=use_state_risk,
        )
        val, grad = valgrad(x0, m0, S0, goal_obs, progress_index)
        return float(val), np.asarray(grad, dtype=float).reshape(-1)

    import jax.numpy as jnp

    valgrad, _ = planner._get_jax_valgrad(
        goal_state,
        goal_cov,
        goal_obs,
        goal_obs_cov,
        use_observation_risk=use_observation_risk,
        use_ambiguity_term=use_ambiguity_term,
        use_state_risk=use_state_risk,
    )
    val, grad = valgrad(
        jnp.array(x0),
        jnp.array(m0),
        jnp.array(S0),
        jnp.array(goal_obs),
        jnp.asarray(progress_index, dtype=jnp.array(goal_obs).dtype),
    )
    return float(val), np.asarray(grad, dtype=float).reshape(-1)


def _plan_summary(result):
    return {
        'backend': str(result.backend),
        'total_cost': float(result.total_cost),
        'risk_cost': float(result.risk_cost),
        'ambiguity_cost': float(result.ambiguity_cost),
        'control_cost': float(result.control_cost),
        'visibility_cost': float(result.visibility_cost),
        'obstacle_cost': float(result.obstacle_cost),
        'optimizer_success': bool(result.optimizer_success),
        'optimizer_status': int(result.optimizer_status),
        'optimizer_nit': int(result.optimizer_nit),
        'optimizer_nfev': int(result.optimizer_nfev),
        'optimizer_message': str(result.optimizer_message),
        'solve_time_ms': 1000.0 * float(result.solve_time_s),
        'p_vis_plan': float(result.p_vis_plan),
        'p_vis_plan_eff': float(result.p_vis_plan_eff),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Experiment run directory containing run_manifest.json and experiment.csv')
    parser.add_argument('--row-index', type=int, default=None, help='CSV row index to use as the frozen planning snapshot')
    parser.add_argument('--progress-index', type=float, default=0.0, help='Planner progress index used in the notebook objective')
    parser.add_argument('--reference-backend', default='scipy', choices=('scipy', 'jax'), help='Reference notebook backend (JAX-backed)')
    parser.add_argument('--candidate-backend', default='casadi', choices=('casadi',), help='Candidate backend to compare')
    args = parser.parse_args()

    repo_root = _repo_root()
    _extend_sys_path(repo_root)

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    with (run_dir / 'experiment.csv').open() as f:
        rows = list(csv.DictReader(f))
    row, selected_index = _select_snapshot_row(rows, args.row_index)
    cfg = _build_launch_like_cfg(manifest, repo_root)

    from planning.planners.base_planner import UnicyclePlannerBase

    m0, S0, goal_xy = _snapshot_state(row)
    ref_planner = UnicyclePlannerBase(**_planner_kwargs_from_cfg(cfg, optimizer_backend=args.reference_backend))
    try:
        cand_planner = UnicyclePlannerBase(**_planner_kwargs_from_cfg(cfg, optimizer_backend=args.candidate_backend))
    except Exception as exc:
        if 'CasADi' in str(exc):
            print(json.dumps({
                'run_dir': str(run_dir),
                'candidate_backend': args.candidate_backend,
                'error': str(exc),
            }, indent=2, sort_keys=True))
            return 2
        raise

    try:
        cand_seed_val, cand_seed_grad = _seed_eval(cand_planner, args.candidate_backend, m0, S0, goal_xy, args.progress_index)
    except Exception as exc:
        if 'CasADi' in str(exc):
            print(json.dumps({
                'run_dir': str(run_dir),
                'candidate_backend': args.candidate_backend,
                'error': str(exc),
            }, indent=2, sort_keys=True))
            return 2
        raise
    ref_seed_val, ref_seed_grad = _seed_eval(ref_planner, args.reference_backend, m0, S0, goal_xy, args.progress_index)

    try:
        cand_result = cand_planner.plan(m0, S0, goal_xy, progress_index=args.progress_index)
    except Exception as exc:
        if 'CasADi' in str(exc):
            print(json.dumps({
                'run_dir': str(run_dir),
                'candidate_backend': args.candidate_backend,
                'error': str(exc),
            }, indent=2, sort_keys=True))
            return 2
        raise
    ref_result = ref_planner.plan(m0, S0, goal_xy, progress_index=args.progress_index)

    summary = {
        'run_dir': str(run_dir),
        'selected_row_index': int(selected_index),
        'selected_stamp': row.get('stamp', ''),
        'snapshot_goal_dist': _float(row, 'goal_dist'),
        'reference_backend': args.reference_backend,
        'candidate_backend': args.candidate_backend,
        'seed_objective_reference': ref_seed_val,
        'seed_objective_candidate': cand_seed_val,
        'seed_objective_abs_diff': abs(ref_seed_val - cand_seed_val),
        'seed_grad_norm_reference': float(np.linalg.norm(ref_seed_grad)),
        'seed_grad_norm_candidate': float(np.linalg.norm(cand_seed_grad)),
        'seed_grad_norm_abs_diff': abs(float(np.linalg.norm(ref_seed_grad)) - float(np.linalg.norm(cand_seed_grad))),
        'seed_grad_l2_diff': float(np.linalg.norm(ref_seed_grad - cand_seed_grad)),
        'controls_l2_diff': float(np.linalg.norm(np.asarray(ref_result.controls) - np.asarray(cand_result.controls))),
        'controls_inf_diff': float(np.max(np.abs(np.asarray(ref_result.controls) - np.asarray(cand_result.controls)))),
        'states_l2_diff': float(np.linalg.norm(np.asarray(ref_result.states) - np.asarray(cand_result.states))),
        'reference_result': _plan_summary(ref_result),
        'candidate_result': _plan_summary(cand_result),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
