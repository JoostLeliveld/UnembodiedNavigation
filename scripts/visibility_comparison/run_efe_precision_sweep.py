#!/usr/bin/env python3
"""Run the current EFE thesis profile."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from common import CURRENT_GP_DIR, LOGS_ROOT


PROFILE_ORDER = (
    'efe_main',
)


EFE_MAIN_PARAMS = {
    'goal_prior_u_std_start': '180.0',
    'goal_prior_v_std_start': '180.0',
    'goal_prior_u_std_final': '45.0',
    'goal_prior_v_std_final': '45.0',
    'goal_tightening_power': '1.20',
    'horizon': '50',
    'dt': '0.2',
    'discount_gamma': '0.98',
    'r_visible_uv': '2.5',
    'r_miss_uv': '100.0',
    'odom_heading_correction_mode': 'overwrite',
    'clamp_pixel_uv_theta_without_yaw': 'true',
}

PROFILES = {
    'efe_main': {
        'planner': 'efe1',
        'description': (
            'Current thesis EFE configuration: broad goal preferences, horizon 50, direct '
            'GP-to-covariance mapping, single warm-start optimizer, and odom-anchored belief yaw '
            'for the YOLO position-only pipeline.'
        ),
        'params': dict(EFE_MAIN_PARAMS),
    },
}


OWN_NODE_PATTERNS = (
    'yolo_robot_detector_node',
    'pixel_to_bev_state_node',
    'goal_mission_node',
    'goal_marker_node',
    'experiment_logger',
    'install/planning/lib/planning/efe_agent',
    'wait_for_odom',
    'reset_world',
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _reap_processes() -> None:
    for pattern in OWN_NODE_PATTERNS:
        try:
            subprocess.run(['pkill', '-f', pattern], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass
    for pattern in ('ign gazebo', 'parameter_bridge'):
        try:
            subprocess.run(['pkill', '-f', pattern], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _terminate_group(process: subprocess.Popen, *, grace_s: float = 3.0) -> None:
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(grace_s)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _latest_run_dir(root: Path, start_wall: float) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest if latest.stat().st_mtime >= start_wall - 5.0 else None


def _run_valid_for_success(summary: dict) -> bool:
    if not summary:
        return False
    if not bool(summary.get('valid_run', True)):
        return False
    if bool(summary.get('collision_any', False)):
        return False
    for key in ('max_wall_penetration_m', 'max_obstacle_penetration_m'):
        try:
            if float(summary.get(key, 0.0) or 0.0) > 0.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _command_for_profile(profile_id: str, seed: str, args, log_dir: Path) -> list[str]:
    profile = PROFILES[profile_id]
    artifact = Path(args.visibility_artifact_path).expanduser().resolve()
    yolo_model = Path(args.yolo_model).expanduser().resolve()
    cmd = [
        'ros2', 'launch', 'experiments', str(args.launch_file),
        f'world:={args.world}',
        f'task:={args.task}',
        f'seed:={seed}',
        f'planner:={profile["planner"]}',
        f'comparison_method_id:={profile_id}',
        'perception_backend:=yolo',
        f'visibility_artifact_path:={artifact}',
        f'yolo_model:={yolo_model}',
        f'yolo_imgsz:={int(args.yolo_imgsz)}',
        f'yolo_conf_threshold:={float(args.yolo_conf_threshold)}',
        f'yolo_iou_threshold:={float(args.yolo_iou_threshold)}',
        f'yolo_target_class:={args.yolo_target_class}',
        f'yolo_class_id:={int(args.yolo_class_id)}',
        f'yolo_use_masks:={args.yolo_use_masks}',
        f'yolo_min_mask_area_px:={float(args.yolo_min_mask_area_px)}',
        f'yolo_mask_bottom_band_px:={float(args.yolo_mask_bottom_band_px)}',
        f'run_timeout_after_first_cmd_s:={float(args.timeout_after_first_cmd_s)}',
        f'use_rviz:={args.use_rviz}',
        f'log_dir:={log_dir}',
    ]
    if str(args.yolo_device).strip():
        cmd.append(f'yolo_device:={str(args.yolo_device).strip()}')
    for key, value in profile['params'].items():
        if str(value).strip() == '':
            continue
        cmd.append(f'{key}:={value}')
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the current EFE thesis profile.')
    parser.add_argument('--profiles', nargs='*', default=['efe_main'], choices=PROFILE_ORDER)
    parser.add_argument('--seeds', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--task', default='main_shadow_tradeoff')
    parser.add_argument('--launch-file', default='warehouse_visibility_agent.launch.py')
    parser.add_argument('--log-root', default=str(Path('logs/experiments')))
    parser.add_argument('--visibility-artifact-path', default=str(CURRENT_GP_DIR / 'yolo_score_raw_gp.npz'))
    parser.add_argument('--yolo-model', default='logs/perception_models/yolo_simseg_smoke/model.pt')
    parser.add_argument('--yolo-device', default='')
    parser.add_argument('--yolo-imgsz', type=int, default=640)
    parser.add_argument('--yolo-conf-threshold', type=float, default=0.10)
    parser.add_argument('--yolo-iou-threshold', type=float, default=0.45)
    parser.add_argument('--yolo-target-class', default='robot')
    parser.add_argument('--yolo-class-id', type=int, default=-1)
    parser.add_argument('--yolo-use-masks', default='true')
    parser.add_argument('--yolo-min-mask-area-px', type=float, default=12.0)
    parser.add_argument('--yolo-mask-bottom-band-px', type=float, default=3.0)
    parser.add_argument('--timeout-after-first-cmd-s', type=float, default=90.0)
    parser.add_argument('--wall-timeout-s', type=float, default=1200.0)
    parser.add_argument('--cleanup-delay-s', type=float, default=3.0)
    parser.add_argument('--use-rviz', default='false')
    parser.add_argument('--continue-on-failure', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    log_root = Path(args.log_root).expanduser().resolve()
    allowed_root = LOGS_ROOT.parent.resolve()
    if allowed_root not in log_root.parents and log_root != allowed_root:
        raise RuntimeError(f'Run log root must stay under {allowed_root}: {log_root}')
    if not args.dry_run:
        log_root.mkdir(parents=True, exist_ok=True)

    artifact = Path(args.visibility_artifact_path).expanduser().resolve()
    if not artifact.is_file():
        raise RuntimeError(f'Visibility artifact not found: {artifact}')
    yolo_model = Path(args.yolo_model).expanduser().resolve()
    if not yolo_model.is_file():
        raise RuntimeError(f'YOLO model not found: {yolo_model}')

    manifest = {
        'profile_order': list(args.profiles),
        'seeds': [str(seed) for seed in args.seeds],
        'world': str(args.world),
        'task': str(args.task),
        'visibility_artifact_path': str(artifact),
        'yolo_model': str(yolo_model),
        'profiles': PROFILES,
        'notes': [
            'EFE main preserves the risk and ambiguity equations; tuning is limited to precisions, horizon, discount, and GP-to-covariance semantics.',
            'There is no direct visibility reward or route penalty in the paper EFE profile.',
            'Runs with collision or positive penetration are invalid for success claims but remain included in plots.',
        ],
    }
    if not args.dry_run:
        (log_root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    summary_rows = []
    for profile_id in args.profiles:
        for seed in [str(value) for value in args.seeds]:
            profile_log_root = log_root / profile_id / f'seed_{seed}'
            if not args.dry_run:
                profile_log_root.mkdir(parents=True, exist_ok=True)
            cmd = _command_for_profile(profile_id, seed, args, profile_log_root)
            print('\n---', profile_id, 'seed', seed, '---')
            print(' '.join(str(part) for part in cmd))
            if args.dry_run:
                continue
            _reap_processes()
            time.sleep(max(float(args.cleanup_delay_s), 0.0))
            start_wall = time.time()
            process = subprocess.Popen(cmd, start_new_session=True)
            run_dir = None
            try:
                while time.time() - start_wall < float(args.wall_timeout_s):
                    run_dir = _latest_run_dir(profile_log_root, start_wall) or run_dir
                    if run_dir and (run_dir / 'run_summary.json').is_file():
                        time.sleep(0.2)
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)
            finally:
                _terminate_group(process)
                _reap_processes()

            summary = _load_json(run_dir / 'run_summary.json') if run_dir else {}
            success_claim_valid = _run_valid_for_success(summary)
            row = {
                'profile_id': profile_id,
                'seed': seed,
                'run_dir': str(run_dir or ''),
                'completed': summary.get('completed', ''),
                'completion_reason': summary.get('completion_reason', ''),
                'success_claim_valid': success_claim_valid,
                'valid_run': summary.get('valid_run', ''),
                'invalid_reason': summary.get('invalid_reason', ''),
                'final_goal_distance': summary.get('final_goal_distance', ''),
                'minimum_goal_distance': summary.get('minimum_goal_distance', ''),
                'path_length_m': summary.get('path_length_m', ''),
                'collision_any': summary.get('collision_any', ''),
                'max_wall_penetration_m': summary.get('max_wall_penetration_m', ''),
                'max_obstacle_penetration_m': summary.get('max_obstacle_penetration_m', ''),
                'mean_solve_time_ms': summary.get('mean_solve_time_ms', ''),
                'mean_efe_risk': summary.get('mean_efe_risk', ''),
                'mean_efe_ambiguity': summary.get('mean_efe_ambiguity', ''),
                'mean_efe_obstacle': summary.get('mean_efe_obstacle', ''),
                'mean_p_vis_plan': summary.get('mean_p_vis_plan', ''),
                'mean_p_vis_plan_eff': summary.get('mean_p_vis_plan_eff', ''),
            }
            summary_rows.append(row)
            print(
                f"Result: completion={row['completion_reason']} valid={row['valid_run']} "
                f"success_claim_valid={row['success_claim_valid']} final_goal={row['final_goal_distance']} "
                f"invalid_reason={row['invalid_reason']}"
            )
            if not summary and not args.continue_on_failure:
                raise RuntimeError(f'No run_summary.json produced for {profile_id} seed {seed}')

    if summary_rows:
        import csv

        summary_path = log_root / 'run_summary.csv'
        with summary_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'Wrote run summary to {summary_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
