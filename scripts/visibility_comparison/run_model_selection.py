#!/usr/bin/env python3
"""Run visibility-aware model-selection grids and MC comparisons.

The original use is the lean Task-A sensitivity grid. The same manifest-checked
machinery is also used for small Monte Carlo comparison configs, for example a
selected C2 cell versus the C1 constant-R baseline.

Usage:
    python3 run_model_selection.py --config model_selection_config.yaml --dry-run
    python3 run_model_selection.py --config model_selection_config.yaml --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / 'logs' / 'visibility_comparison'

PLANNER_BY_NAME = {
    'constant_R_efe': 'constant_R_efe',
    'visibility_aware_efe': 'visibility_aware_efe',
    'risk_only_ablation': 'risk_only_ablation',
}

_LAUNCH_KEY_ALIASES = {
    'gp_artifact': 'visibility_artifact_path',
}

_RUNNER_ONLY_KEYS = {
    'world',
    'launch_file',
    'task',
    'planner',
    'seeds',
    'perception_backend',
    'yolo_model',
    'expected_gp_beta',
}

_PATH_KEYS = {
    'gp_artifact',
    'visibility_artifact_path',
    'yolo_model',
}

_SIM_PROCESS_PATTERNS = [
    'gz sim',
    'ign gazebo',
    'ruby /opt/ros',
    'ros_gz_sim',
]

_MANIFEST_FLOAT_KEYS = (
    'horizon',
    'dt',
    'discount_gamma',
    'process_noise_xy',
    'process_noise_theta',
    'observation_risk_scale',
    'ambiguity_term_scale',
    'risk_weight_obs',
    'ambiguity_weight',
    'r_visible_uv',
    'r_miss_uv',
    'goal_prior_u_std_start',
    'goal_prior_v_std_start',
    'goal_prior_u_std_final',
    'goal_prior_v_std_final',
    'goal_tightening_power',
    'goal_progress_n_steps',
    'goal_sigma_uv',
    'min_terminal_goal_progress_m',
    'goal_success_radius',
    'goal_success_hold_s',
    'run_timeout_after_first_cmd_s',
    'optimizer_maxiter',
    'optimizer_maxfun',
    'optimizer_ftol',
    'optimizer_gtol',
    'yolo_conf_threshold',
)

_MANIFEST_BOOL_KEYS = (
    'auto_stop_on_goal',
    'use_command_noise',
    'optimizer_warm_start',
)

_MANIFEST_STRING_KEYS = (
    'world',
    'task',
    'planner',
)

_OWN_NODE_PATTERNS = [
    'yolo_robot_detector_node',
    'pixel_to_bev_state_node',
    'goal_mission_node',
    'goal_marker_node',
    'experiment_logger',
    'install/planning/lib/planning/efe_agent',
    'wait_for_odom',
    'reset_world',
]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f'Config is not a YAML mapping: {path}')
    return cfg


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_for_compare(path_value: object) -> Path:
    return Path(str(path_value)).expanduser().resolve(strict=False)


def _stringify(v: object) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def _as_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _float_close(a: object, b: object, *, tol: float = 1e-9) -> bool:
    fa = _as_float(a)
    fb = _as_float(b)
    return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= tol


def _load_gp_beta(artifact_path: Path) -> float:
    manifest_path = artifact_path.parent / 'gp_manifest.json'
    manifest = _load_json(manifest_path)
    if 'beta' not in manifest:
        raise RuntimeError(f'GP manifest missing beta: {manifest_path}')
    return float(manifest['beta'])


def _validate_gp_artifact(artifact_value: object, expected_beta: object, label: str) -> Path:
    artifact_path = _resolve_for_compare(artifact_value)
    if not artifact_path.is_file():
        raise RuntimeError(f'{label} GP artifact not found: {artifact_path}')

    with np.load(artifact_path, allow_pickle=False) as data:
        required = {'xs', 'ys', 'P_conservative_plan_map'}
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f'{label} GP artifact missing keys {missing}: {artifact_path}')

    actual_beta = _load_gp_beta(artifact_path)
    if not _float_close(actual_beta, expected_beta, tol=1e-8):
        raise RuntimeError(
            f'{label} GP beta mismatch for {artifact_path}: '
            f'expected {expected_beta}, manifest has {actual_beta}'
        )
    return artifact_path


def _merged_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(overrides or {})
    return merged


def _preflight_config(cfg: dict[str, Any]) -> None:
    if 'base' not in cfg or 'cells' not in cfg:
        raise RuntimeError('model-selection config must contain base and cells')
    base = cfg['base']
    cells = cfg['cells']
    if not isinstance(base, dict) or not isinstance(cells, list):
        raise RuntimeError('model-selection config has invalid base/cells types')
    if not cells:
        raise RuntimeError('model-selection config must contain at least one cell')
    expected_cells = cfg.get('expected_cells')
    if expected_cells is not None:
        expected_n = int(expected_cells)
        if len(cells) != expected_n:
            raise RuntimeError(f'Config expected {expected_n} cells, found {len(cells)}')

    required_base = (
        'world', 'launch_file', 'task', 'planner', 'seeds', 'yolo_model',
        'yolo_conf_threshold', 'gp_artifact', 'expected_gp_beta',
    )
    for key in required_base:
        if key not in base:
            raise RuntimeError(f'model-selection base missing required key: {key}')

    if str(base['planner']) not in PLANNER_BY_NAME:
        raise RuntimeError(f"unknown base planner: {base['planner']!r}")
    if not _float_close(base['yolo_conf_threshold'], 0.10, tol=1e-12):
        raise RuntimeError(
            f'model-selection yolo_conf_threshold must be 0.10, got {base["yolo_conf_threshold"]}'
        )
    yolo_model = _resolve_for_compare(base['yolo_model'])
    if not yolo_model.is_file():
        raise RuntimeError(f'YOLO model not found: {yolo_model}')

    base_uses_gp = str(base['planner']) != 'constant_R_efe'
    any_cell_uses_gp = False
    for cell in cells:
        overrides = cell.get('overrides', {}) or {}
        merged = _merged_config(base, overrides)
        if str(merged.get('planner', base['planner'])) != 'constant_R_efe':
            any_cell_uses_gp = True
            break
    if base_uses_gp or any_cell_uses_gp:
        _validate_gp_artifact(base['gp_artifact'], base['expected_gp_beta'], 'nominal')

    seen_labels: set[str] = set()
    for cell in cells:
        axis = str(cell.get('axis', '')).strip()
        label = str(cell.get('label', '')).strip()
        overrides = cell.get('overrides', {}) or {}
        if not axis or not label:
            raise RuntimeError(f'Cell missing axis/label: {cell!r}')
        if label in seen_labels:
            raise RuntimeError(f'Duplicate cell label: {label}')
        seen_labels.add(label)
        if not isinstance(overrides, dict) or not overrides:
            raise RuntimeError(f'Cell {label} must be non-nominal and contain overrides')

        merged = _merged_config(base, overrides)
        planner = str(merged.get('planner', base['planner']))
        if planner not in PLANNER_BY_NAME:
            raise RuntimeError(f'Cell {label} has unknown planner: {planner!r}')
        if planner != 'constant_R_efe':
            expected_beta = cell.get('expected_gp_beta', base['expected_gp_beta'])
            _validate_gp_artifact(merged['gp_artifact'], expected_beta, label)


def _build_launch_cmd(base: dict[str, Any], overrides: dict[str, Any], seed: int, log_dir: Path) -> list[str]:
    merged = _merged_config(base, overrides)

    planner = str(merged['planner'])
    if planner not in PLANNER_BY_NAME:
        raise RuntimeError(f"unknown planner: {planner!r}")

    cmd = [
        'ros2', 'launch', 'experiments', str(merged['launch_file']),
        f'world:={merged["world"]}',
        f'task:={merged["task"]}',
        f'planner:={planner}',
        f'seed:={seed}',
        f'log_dir:={log_dir}',
        f'perception_backend:={merged.get("perception_backend", "yolo")}',
        f'yolo_model:={_resolve_for_compare(merged["yolo_model"])}',
    ]

    for key, value in merged.items():
        if key in _RUNNER_ONLY_KEYS:
            continue
        launch_key = _LAUNCH_KEY_ALIASES.get(key, key)
        if launch_key == 'visibility_artifact_path':
            value = str(_resolve_for_compare(value))
        if str(value).startswith('[FILL'):
            raise RuntimeError(f'unfilled value for {key}: {value!r}')
        cmd.append(f'{launch_key}:={_stringify(value)}')

    if planner == 'constant_R_efe':
        cmd = [c for c in cmd if not c.startswith('visibility_artifact_path:=')]

    return cmd


def _expected_manifest_values(merged: dict[str, Any]) -> dict[str, Any]:
    expected: dict[str, Any] = {
        'yolo_model': str(_resolve_for_compare(merged['yolo_model'])),
    }
    if str(merged.get('planner', '')) != 'constant_R_efe':
        expected['visibility_artifact_path'] = str(_resolve_for_compare(merged['gp_artifact']))
    for key in _MANIFEST_STRING_KEYS + _MANIFEST_FLOAT_KEYS + _MANIFEST_BOOL_KEYS:
        if key in merged:
            expected[key] = merged[key]
    return expected


def _manifest_matches_expected(run_dir: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    manifest = _load_json(run_dir / 'run_manifest.json')
    if not manifest:
        return False, f'missing or unreadable run_manifest.json in {run_dir}'

    for key, exp in expected.items():
        if key in _PATH_KEYS or key == 'visibility_artifact_path':
            actual = str(manifest.get(key, '') or '')
            if _resolve_for_compare(actual) != _resolve_for_compare(exp):
                return False, f'{key} mismatch: run used {actual or "<missing>"}, expected {exp}'
        elif key in _MANIFEST_FLOAT_KEYS:
            if key not in manifest or not _float_close(manifest.get(key), exp, tol=1e-8):
                return False, f'{key} mismatch: run used {manifest.get(key, "<missing>")}, expected {exp}'
        elif key in _MANIFEST_BOOL_KEYS:
            if key not in manifest or _as_bool(manifest.get(key)) != _as_bool(exp):
                return False, f'{key} mismatch: run used {manifest.get(key, "<missing>")}, expected {exp}'
        elif key in _MANIFEST_STRING_KEYS:
            if str(manifest.get(key, '')) != str(exp):
                return False, f'{key} mismatch: run used {manifest.get(key, "<missing>")}, expected {exp}'
    return True, ''


def _existing_entry_matches_expected(entry: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    run_dir_str = str(entry.get('run_dir', '') or '')
    if not run_dir_str:
        return False, 'missing run_dir'
    return _manifest_matches_expected(Path(run_dir_str), expected)


def _reap_own_node_stragglers() -> None:
    for pattern in _OWN_NODE_PATTERNS:
        try:
            subprocess.run(['pkill', '-f', pattern], timeout=2,
                           capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _reap_sim_stragglers() -> None:
    for pattern in _SIM_PROCESS_PATTERNS:
        try:
            subprocess.run(['pkill', '-9', '-f', pattern], timeout=3,
                           capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _terminate_process_group(pgid: int, *, grace_s: float = 3.0) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(grace_s)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_run_summary(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    p = run_dir / 'run_summary.json'
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def _find_latest_run_dir(log_dir: Path) -> Path | None:
    if not log_dir.is_dir():
        return None
    for d in sorted(log_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / 'run_summary.json').is_file():
            return d
    return None


def _cell_key(axis: str, label: str, seed: int) -> str:
    return f'{axis}__{label}__seed{seed}'


def _load_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')


def _run_one(cmd: list[str], run_log_dir: Path, run_timeout_s: float) -> dict[str, Any]:
    print('  CMD:', ' '.join(cmd))
    proc = subprocess.Popen(cmd, start_new_session=True)
    pgid = os.getpgid(proc.pid)
    started = time.time()
    timed_out = False
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            if time.time() - started > run_timeout_s:
                print(f'  TIMEOUT after {run_timeout_s:.0f}s, killing process group')
                timed_out = True
                _terminate_process_group(pgid)
                proc.wait(timeout=5)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        _terminate_process_group(pgid)
        raise

    run_dir = _find_latest_run_dir(run_log_dir)
    summary = _read_run_summary(run_dir)
    outcome = 'infra_invalid'
    if summary:
        reason = summary.get('completion_reason')
        if reason == 'goal_reached' and not summary.get('crashed'):
            outcome = 'goal_reached'
        elif reason == 'timeout_after_first_cmd':
            outcome = 'timeout'
        elif summary.get('crashed') or reason == 'collision':
            outcome = 'collision'

    return {
        'run_dir': str(run_dir) if run_dir else '',
        'outcome': outcome,
        'completion_reason': summary.get('completion_reason') if summary else None,
        'goal_reached': bool(outcome == 'goal_reached'),
        'crashed': bool(summary.get('crashed')) if summary else False,
        'path_length_m': summary.get('path_length_m') if summary else None,
        'minimum_goal_distance': summary.get('minimum_goal_distance') if summary else None,
        'elapsed_after_first_cmd_s': summary.get('elapsed_after_first_cmd_s') if summary else None,
        'timed_out_outer': timed_out,
        'finished_at': datetime.now().isoformat(),
    }


def _build_matrix(
    base: dict[str, Any],
    cells: list[dict[str, Any]],
    *,
    only_axis: str,
    only_label: str,
) -> list[tuple[str, str, dict[str, Any], dict[str, Any], int]]:
    seeds = list(base['seeds'])
    matrix = []
    for cell in cells:
        axis = str(cell['axis'])
        label = str(cell['label'])
        if only_axis and axis != only_axis:
            continue
        if only_label and label != only_label:
            continue
        overrides = cell.get('overrides', {}) or {}
        merged = _merged_config(base, overrides)
        expected_beta = cell.get('expected_gp_beta', base.get('expected_gp_beta'))
        for seed in seeds:
            matrix.append((axis, label, overrides, {'expected_gp_beta': expected_beta, 'merged': merged}, int(seed)))
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the lean model-selection sensitivity grid.')
    parser.add_argument('--config', default='model_selection_config.yaml')
    parser.add_argument('--log-root',
                        default=str(LOGS_ROOT / 'model_selection_rawgp_v1'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-timeout', type=float, default=300.0)
    parser.add_argument('--cleanup-delay', type=float, default=8.0)
    parser.add_argument('--headless', action='store_true',
                        help='Force Gazebo server-only mode for all runs.')
    parser.add_argument('--only-axis', default='',
                        help='If set, only run cells with axis == this value')
    parser.add_argument('--only-label', default='',
                        help='If set, only run cells with label == this value')
    args = parser.parse_args()

    cfg = _load_config(Path(args.config).expanduser().resolve())
    _preflight_config(cfg)
    base = cfg['base']
    if args.headless:
        base = dict(base)
        base['headless'] = True
    cells = cfg['cells']
    matrix = _build_matrix(base, cells, only_axis=args.only_axis, only_label=args.only_label)

    log_root = Path(args.log_root).expanduser().resolve()
    grid_log_path = log_root / 'grid_log.json'

    print(f'Preflight OK: {len(cells)} cells, {len(base["seeds"])} seeds')
    print(f'Sensitivity grid: {len(matrix)} runs')
    print(f'Log root: {log_root}')
    if args.dry_run:
        print('DRY RUN - no simulator processes or cleanup commands will be started.\n')
    else:
        log_root.mkdir(parents=True, exist_ok=True)

    log = _load_log(grid_log_path) if args.resume else {}
    log = dict(log)

    if not args.dry_run:
        _reap_sim_stragglers()
        _reap_own_node_stragglers()
        time.sleep(args.cleanup_delay)

    for idx, (axis, label, overrides, meta, seed) in enumerate(matrix):
        key = _cell_key(axis, label, seed)
        merged = meta['merged']
        expected_manifest = _expected_manifest_values(merged)
        banner = f'[{idx + 1}/{len(matrix)}] axis={axis} label={label} seed={seed}'

        if args.resume and key in log and log[key].get('outcome') not in (None, 'infra_invalid'):
            matches, reason = _existing_entry_matches_expected(log[key], expected_manifest)
            if not matches:
                raise RuntimeError(f'Cannot resume stale model-selection entry for {banner}: {reason}')
            print(f'  SKIP (already done): {banner}')
            continue

        run_log_dir = log_root / axis / label / f'seed{seed}'
        cmd = _build_launch_cmd(base, overrides, seed, run_log_dir)
        print(f'\n{banner}')
        if args.dry_run:
            print('  CMD:', ' '.join(cmd))
            continue

        run_log_dir.mkdir(parents=True, exist_ok=True)
        if idx > 0:
            _reap_sim_stragglers()
            _reap_own_node_stragglers()
            time.sleep(args.cleanup_delay)

        log[key] = {
            'axis': axis,
            'label': label,
            'seed': seed,
            'overrides': overrides,
            'merged_config': merged,
            'expected_gp_beta': meta['expected_gp_beta'],
            'expected_visibility_artifact_path': str(_resolve_for_compare(merged['gp_artifact'])),
            'started_at': datetime.now().isoformat(),
            'outcome': None,
        }
        _save_log(grid_log_path, log)

        result = _run_one(cmd, run_log_dir, args.run_timeout)
        if result.get('run_dir'):
            matches, reason = _manifest_matches_expected(Path(str(result['run_dir'])), expected_manifest)
            if not matches:
                result['outcome'] = 'infra_invalid'
                result['manifest_mismatch'] = reason
                print(f'  INFRA INVALID: manifest mismatch: {reason}')
        log[key].update(result)
        _save_log(grid_log_path, log)

        path_str = result.get('path_length_m')
        path_show = f'{path_str:.2f}m' if isinstance(path_str, (int, float)) else 'n/a'
        print(f'  -> outcome={log[key]["outcome"]}  L={path_show}')

    if not args.dry_run:
        _reap_sim_stragglers()
        _reap_own_node_stragglers()
    print('\nGrid complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
