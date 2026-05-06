#!/usr/bin/env python3
"""Run the full 36-run IWAI campaign from a locked config file.

Usage:
    python run_iwai_campaign.py --config iwai_campaign_config.yaml [--dry-run] [--resume]

Each run result is written immediately to campaign_log.json so the campaign
can be interrupted and resumed with --resume (already-completed runs are skipped).
Completion reasons are exactly: goal_reached, timeout_after_first_cmd, collision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / 'logs' / 'visibility_comparison'

# Map condition ID to planner name (must match ALLOWED_PLANNERS in launch file).
CONDITION_PLANNER = {
    'C1': 'constant_R_efe',
    'C2': 'visibility_aware_efe',
    'C3': 'risk_only_ablation',
}


def _load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg, path)
    return cfg


def _validate_config(cfg: dict, path: Path) -> None:
    for key in ('world', 'launch_file', 'conditions', 'tasks', 'gp_artifact',
                'yolo_model', 'horizon', 'dt', 'goal_success_radius',
                'run_timeout_after_first_cmd_s'):
        if key not in cfg:
            raise RuntimeError(f"Campaign config {path} is missing required key: '{key}'")
    for key in ('gp_artifact', 'yolo_model'):
        if str(cfg[key]).startswith('[FILL'):
            raise RuntimeError(
                f"Campaign config {path} has unfilled placeholder for '{key}': {cfg[key]!r}\n"
                f"Fill in all [FILL] entries before running."
            )
    for key in ('observation_risk_scale', 'ambiguity_term_scale'):
        if key in cfg and str(cfg[key]).startswith('[FILL'):
            raise RuntimeError(
                f"Campaign config {path} has unfilled placeholder for '{key}': {cfg[key]!r}\n"
                f"Verify the lambda mapping against the planner source and fill it in."
            )


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


def _reap_own_node_stragglers() -> None:
    for pattern in _OWN_NODE_PATTERNS:
        try:
            subprocess.run(['pkill', '-f', pattern], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _run_key(task: str, condition: str, seed: int) -> str:
    return f'{task}__{condition}__seed{seed}'


def _load_run_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {}
    try:
        return json.loads(log_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_run_log(log_path: Path, log: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding='utf-8')


def _resolve_for_compare(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve(strict=False)


def _float_close(a, b, *, tol: float = 1e-8) -> bool:
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= tol


def _load_run_manifest(run_dir: Path) -> dict:
    p = run_dir / 'run_manifest.json'
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _existing_entry_matches_config(entry: dict, cfg: dict) -> tuple[bool, str]:
    condition_id = str(entry.get('condition', ''))
    run_dir_str = str(entry.get('run_dir', '') or '')
    if not run_dir_str:
        return False, 'missing run_dir'
    manifest = _load_run_manifest(Path(run_dir_str))
    if not manifest:
        return False, f'missing run_manifest.json in {run_dir_str}'

    expected_yolo_model = str(Path(cfg['yolo_model']).expanduser().resolve(strict=False))
    actual_yolo_model = str(manifest.get('yolo_model', '') or '')
    if _resolve_for_compare(actual_yolo_model) != _resolve_for_compare(expected_yolo_model):
        return False, f'yolo_model mismatch: run used {actual_yolo_model or "<missing>"}, config expects {expected_yolo_model}'

    numeric_keys = (
        'horizon', 'dt', 'goal_success_radius', 'goal_success_hold_s',
        'run_timeout_after_first_cmd_s', 'r_visible_uv', 'r_miss_uv',
        'process_noise_xy', 'process_noise_theta', 'risk_weight_obs',
        'ambiguity_weight', 'observation_risk_scale', 'ambiguity_term_scale',
        'control_weight', 'discount_gamma', 'yolo_conf_threshold',
    )
    for key in numeric_keys:
        if key in cfg and (key not in manifest or not _float_close(manifest.get(key), cfg[key])):
            return False, f'{key} mismatch: run used {manifest.get(key, "<missing>")}, config expects {cfg[key]}'

    if CONDITION_PLANNER.get(condition_id) != 'constant_R_efe':
        actual = str(manifest.get('visibility_artifact_path', '') or '')
        expected = str(Path(cfg['gp_artifact']).expanduser().resolve(strict=False))
        if _resolve_for_compare(actual) != _resolve_for_compare(expected):
            return (
                False,
                f'visibility_artifact_path mismatch: run used {actual or "<missing>"}, config expects {expected}',
            )
    return True, ''


def _build_run_matrix(cfg: dict) -> list[tuple[str, str, int]]:
    """Return list of (task_name, condition_id, seed) in deterministic order."""
    runs = []
    for task_name, task_cfg in cfg['tasks'].items():
        for condition_id in task_cfg['conditions']:
            for seed in task_cfg['seeds']:
                runs.append((task_name, condition_id, seed))
    return runs


def _build_launch_cmd(cfg: dict, task_name: str, condition_id: str, seed: int, log_dir: Path) -> list[str]:
    planner = CONDITION_PLANNER[condition_id]
    gp_artifact = str(Path(cfg['gp_artifact']).expanduser().resolve())
    yolo_model = str(Path(cfg['yolo_model']).expanduser().resolve())

    cmd = [
        'ros2', 'launch', 'experiments', str(cfg['launch_file']),
        f'world:={cfg["world"]}',
        f'task:={task_name}',
        f'planner:={planner}',
        f'seed:={seed}',
        f'log_dir:={log_dir}',
        f'perception_backend:={cfg.get("perception_backend", "yolo")}',
        f'horizon:={cfg["horizon"]}',
        f'dt:={cfg["dt"]}',
        f'goal_success_radius:={cfg["goal_success_radius"]}',
        f'goal_success_hold_s:={cfg.get("goal_success_hold_s", 2.0)}',
        f'run_timeout_after_first_cmd_s:={cfg["run_timeout_after_first_cmd_s"]}',
        f'auto_stop_on_goal:=true',
        f'r_visible_uv:={cfg.get("r_visible_uv", 2.5)}',
        f'r_miss_uv:={cfg.get("r_miss_uv", 120.0)}',
        f'discount_gamma:={cfg.get("discount_gamma", 0.98)}',
        f'process_noise_xy:={cfg.get("process_noise_xy", 0.01)}',
        f'process_noise_theta:={cfg.get("process_noise_theta", 0.02)}',
        f'control_weight:={cfg.get("control_weight", 0.0)}',
        f'optimizer_maxiter:={cfg.get("optimizer_maxiter", 80)}',
        f'optimizer_maxfun:={cfg.get("optimizer_maxfun", 500)}',
        f'use_command_noise:={str(cfg.get("use_command_noise", True)).lower()}',
        f'command_noise_linear_slip_mean:={cfg.get("command_noise_linear_slip_mean", 0.03)}',
        f'command_noise_linear_slip_std:={cfg.get("command_noise_linear_slip_std", 0.06)}',
        f'command_noise_angular_slip_std:={cfg.get("command_noise_angular_slip_std", 0.04)}',
        f'command_noise_linear_additive_std:={cfg.get("command_noise_linear_additive_std", 0.008)}',
        f'command_noise_angular_additive_std:={cfg.get("command_noise_angular_additive_std", 0.035)}',
        f'command_noise_correlation_alpha:={cfg.get("command_noise_correlation_alpha", 0.85)}',
        f'yolo_model:={yolo_model}',
        f'yolo_imgsz:={cfg.get("yolo_imgsz", 640)}',
        f'yolo_conf_threshold:={cfg.get("yolo_conf_threshold", 0.25)}',
        f'yolo_iou_threshold:={cfg.get("yolo_iou_threshold", 0.45)}',
        f'yolo_target_class:={cfg.get("yolo_target_class", "robot")}',
        f'yolo_class_id:={cfg.get("yolo_class_id", -1)}',
        f'yolo_use_masks:={str(cfg.get("yolo_use_masks", True)).lower()}',
        f'yolo_min_mask_area_px:={cfg.get("yolo_min_mask_area_px", 12.0)}',
        f'yolo_mask_bottom_band_px:={cfg.get("yolo_mask_bottom_band_px", 3.0)}',
    ]

    # Planner-specific args: pass GP artifact for C2/C3, locked weights
    if planner != 'constant_R_efe':
        cmd.append(f'visibility_artifact_path:={gp_artifact}')

    for key in ('observation_risk_scale', 'ambiguity_term_scale',
                'risk_weight_obs', 'ambiguity_weight',
                'belief_publish_rate'):
        if key in cfg and not str(cfg[key]).startswith('[FILL'):
            cmd.append(f'{key}:={cfg[key]}')

    return cmd


def _read_run_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / 'run_summary.json'
    if not summary_path.is_file():
        return None
    try:
        return json.loads(summary_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def _find_latest_run_dir(log_dir: Path) -> Path | None:
    if not log_dir.is_dir():
        return None
    candidates = sorted(log_dir.iterdir(), reverse=True)
    for d in candidates:
        if d.is_dir() and (d / 'run_summary.json').is_file():
            return d
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the full IWAI 36-run campaign.')
    parser.add_argument('--config', default='iwai_campaign_config.yaml',
                        help='Path to the locked campaign config YAML.')
    parser.add_argument('--log-root', default=str(LOGS_ROOT / 'iwai_campaign_rawgp_v1'),
                        help='Root directory for all run logs.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without executing.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip runs already marked completed in campaign_log.json.')
    parser.add_argument('--run-timeout', type=float, default=300.0,
                        help='Wall-clock timeout per run including simulator startup (seconds).')
    parser.add_argument('--cleanup-delay', type=float, default=8.0,
                        help='Sleep between runs for process cleanup (seconds).')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f'ERROR: config file not found: {config_path}', file=sys.stderr)
        return 1

    cfg = _load_config(config_path)
    log_root = Path(args.log_root).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    campaign_log_path = log_root / 'campaign_log.json'

    run_matrix = _build_run_matrix(cfg)
    existing_log = _load_run_log(campaign_log_path) if args.resume else {}

    print(f'Campaign: {len(run_matrix)} runs total')
    print(f'Config: {config_path}')
    print(f'Log root: {log_root}')
    print(f'Campaign log: {campaign_log_path}')
    if args.dry_run:
        print('DRY RUN — no processes will be started.\n')

    _reap_own_node_stragglers()
    if not args.dry_run:
        time.sleep(args.cleanup_delay)

    campaign_log = dict(existing_log)

    for run_idx, (task_name, condition_id, seed) in enumerate(run_matrix):
        key = _run_key(task_name, condition_id, seed)
        label = f'[{run_idx + 1}/{len(run_matrix)}] task={task_name} condition={condition_id} seed={seed}'

        if args.resume and key in campaign_log and campaign_log[key].get('outcome') not in (None, 'infra_invalid'):
            matches, reason = _existing_entry_matches_config(campaign_log[key], cfg)
            if not matches:
                raise RuntimeError(
                    f'Cannot resume campaign with stale run entry for {label}: {reason}. '
                    'Start a fresh log root or rerun the campaign without --resume.'
                )
            print(f'  SKIP (already done): {label}')
            continue

        run_log_dir = log_root / task_name / condition_id / f'seed{seed}'
        run_log_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_launch_cmd(cfg, task_name, condition_id, seed, run_log_dir)
        print(f'\n{label}')
        print('  CMD:', ' '.join(str(p) for p in cmd))

        if args.dry_run:
            continue

        if run_idx > 0:
            _reap_own_node_stragglers()
            time.sleep(args.cleanup_delay)

        run_entry: dict = {
            'task': task_name,
            'condition': condition_id,
            'seed': seed,
            'planner': CONDITION_PLANNER[condition_id],
            'run_log_dir': str(run_log_dir),
            'started_at': datetime.now().isoformat(),
            'outcome': None,
            'completion_reason': None,
            'goal_reached': None,
            'crashed': None,
            'path_length_m': None,
            'mean_truth_state_error_m': None,
            'elapsed_after_first_cmd_s': None,
            'minimum_goal_distance': None,
        }
        campaign_log[key] = run_entry
        _save_run_log(campaign_log_path, campaign_log)

        process = subprocess.Popen(cmd, start_new_session=True)
        pgid = os.getpgid(process.pid)
        timed_out = False
        try:
            process.wait(timeout=args.run_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f'  Wall-clock timeout after {args.run_timeout:.0f}s — killing.')
        finally:
            _terminate_process_group(pgid)
            _reap_own_node_stragglers()

        # Read run summary written by experiment_logger
        run_dir = _find_latest_run_dir(run_log_dir)
        summary = _read_run_summary(run_dir) if run_dir else None

        if summary is None:
            outcome = 'infra_invalid'
            completion_reason = 'no_summary'
            print(f'  INFRA INVALID: no run_summary.json found in {run_log_dir}')
        elif not summary.get('completed', False) and timed_out:
            outcome = 'infra_invalid'
            completion_reason = 'wall_clock_timeout'
            print(f'  INFRA INVALID: wall-clock timeout, logger did not complete.')
        else:
            completion_reason = str(summary.get('completion_reason', ''))
            crashed = bool(summary.get('crashed', False))
            if completion_reason == 'goal_reached':
                outcome = 'goal_reached'
            elif completion_reason == 'timeout_after_first_cmd':
                outcome = 'timeout'
            elif crashed or completion_reason not in ('goal_reached', 'timeout_after_first_cmd'):
                outcome = 'collision'
            else:
                outcome = completion_reason

        run_entry.update({
            'finished_at': datetime.now().isoformat(),
            'outcome': outcome,
            'completion_reason': completion_reason,
            'goal_reached': outcome == 'goal_reached',
            'crashed': bool(summary.get('crashed', False)) if summary else None,
            'path_length_m': summary.get('path_length_m') if summary else None,
            'mean_truth_state_error_m': summary.get('mean_truth_state_error_m') if summary else None,
            'elapsed_after_first_cmd_s': summary.get('elapsed_after_first_cmd_s') if summary else None,
            'minimum_goal_distance': summary.get('minimum_goal_distance') if summary else None,
            'run_dir': str(run_dir) if run_dir else None,
        })
        campaign_log[key] = run_entry
        _save_run_log(campaign_log_path, campaign_log)

        goal_str = 'YES' if outcome == 'goal_reached' else 'no'
        print(f'  -> outcome={outcome}, goal={goal_str}, reason={completion_reason}')

    print('\n=== Campaign complete ===')
    if not args.dry_run:
        total = len(run_matrix)
        completed = sum(1 for e in campaign_log.values() if e.get('outcome') not in (None, 'infra_invalid'))
        goals = sum(1 for e in campaign_log.values() if e.get('outcome') == 'goal_reached')
        infra = sum(1 for e in campaign_log.values() if e.get('outcome') == 'infra_invalid')
        print(f'  {completed}/{total} runs completed, {goals} goal_reached, {infra} infra_invalid')
        print(f'  Full log: {campaign_log_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
