#!/usr/bin/env python3
"""Run planner comparisons with method-matched live perception backends."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from common import (
    ACTIVE_METHOD_IDS,
    CURRENT_GP_DIR,
    CURRENT_TARGETS_DIR,
    DEFAULT_WORLD_PROFILES_PATH,
    LOGS_ROOT,
    expected_visibility_metadata,
    load_visibility_artifact_metadata,
    validate_visibility_metadata,
)


COMPARISON_METHODS = tuple(method for method in ACTIVE_METHOD_IDS if method != 'visibility_unaware_baseline')


def _uses_live_yolo(method_id: str, args) -> bool:
    if method_id in ('yolo_binary', 'yolo_score_raw', 'yolo_score_calibrated'):
        return True
    if method_id == 'oracle_visibility':
        return str(args.oracle_backend).strip() == 'yolo'
    if method_id == 'visibility_unaware_baseline':
        return str(args.baseline_backend).strip() == 'yolo'
    return False


def _validate_yolo_runtime_matches_targets(args, methods: list[str]) -> None:
    if not any(_uses_live_yolo(method_id, args) for method_id in methods):
        return
    manifest_path = Path(args.targets_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f'YOLO methods require a valid targets manifest for config matching: {manifest_path}')
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Failed to read targets manifest {manifest_path}: {exc}') from exc
    yolo_summary = payload.get('yolo_summary', {}) if isinstance(payload, dict) else {}
    capture_metadata = payload.get('capture_metadata', {}) if isinstance(payload, dict) else {}
    if not isinstance(yolo_summary, dict) or not bool(yolo_summary.get('enabled', False)):
        raise RuntimeError(f'Targets manifest does not contain an enabled yolo_summary: {manifest_path}')
    implemented_targets = set(str(v).strip() for v in payload.get('implemented_targets', []) if str(v).strip())
    required_targets = {'yolo_binary', 'yolo_score_raw', 'yolo_score_calibrated'}
    missing_targets = sorted(required_targets - implemented_targets)
    if missing_targets:
        raise RuntimeError(
            f'Targets manifest {manifest_path} is missing required YOLO targets after schema cutover: {missing_targets}'
        )
    expected_metadata = expected_visibility_metadata(
        str(args.world),
        world_profiles_path=Path(args.world_profiles).expanduser().resolve(),
    )
    if isinstance(capture_metadata, dict) and capture_metadata:
        validate_visibility_metadata(capture_metadata, expected_metadata, label=f'YOLO targets manifest {manifest_path}')

    runtime_model = str(Path(args.yolo_model).expanduser().resolve()) if str(args.yolo_model).strip() else ''
    manifest_model = str(Path(str(yolo_summary.get('model_path', '') or '')).expanduser().resolve()) if str(yolo_summary.get('model_path', '') or '').strip() else ''
    yolo_expected = {
        'model_path': manifest_model,
        'class_name': str(yolo_summary.get('class_name', '')),
        'class_id': int(yolo_summary.get('class_id', -1)),
        'imgsz': int(yolo_summary.get('imgsz', -1)),
        'conf_threshold': float(yolo_summary.get('conf_threshold', math.nan)),
        'iou_threshold': float(yolo_summary.get('iou_threshold', math.nan)),
        'use_masks': str(yolo_summary.get('use_masks', 'true')).strip().lower(),
        'mask_min_area': float(yolo_summary.get('mask_min_area', math.nan)),
        'mask_bottom_band_px': float(yolo_summary.get('mask_bottom_band_px', math.nan)),
    }
    runtime = {
        'model_path': runtime_model,
        'class_name': str(args.yolo_target_class),
        'class_id': int(args.yolo_class_id),
        'imgsz': int(args.yolo_imgsz),
        'conf_threshold': float(args.yolo_conf_threshold),
        'iou_threshold': float(args.yolo_iou_threshold),
        'use_masks': str(args.yolo_use_masks).strip().lower(),
        'mask_min_area': float(args.yolo_min_mask_area_px),
        'mask_bottom_band_px': float(args.yolo_mask_bottom_band_px),
    }
    mismatches = []
    for key in ('model_path', 'class_name', 'class_id', 'imgsz'):
        if str(runtime[key]).strip() != str(yolo_expected[key]).strip():
            mismatches.append(f'{key}: runtime={runtime[key]!r} extractor={yolo_expected[key]!r}')
    for key in ('conf_threshold', 'iou_threshold', 'mask_min_area', 'mask_bottom_band_px'):
        if abs(float(runtime[key]) - float(yolo_expected[key])) > 1e-9:
            mismatches.append(f'{key}: runtime={runtime[key]!r} extractor={yolo_expected[key]!r}')
    if runtime['use_masks'] != yolo_expected['use_masks']:
        mismatches.append(f"use_masks: runtime={runtime['use_masks']!r} extractor={yolo_expected['use_masks']!r}")
    if mismatches:
        raise RuntimeError(
            'Runtime YOLO configuration does not match the extracted yolo_summary in '
            f'{manifest_path}: ' + '; '.join(mismatches)
        )


def _terminate_process_group(pgid: int, *, grace_s: float = 3.0) -> None:
    """Send SIGTERM to the process group we own, wait, then SIGKILL stragglers."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(grace_s)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# Last-resort: ROS 2 launch can spawn grandchildren that escape the PGID on
# some distros. Kill only this repo's own node executables by name - never
# broad patterns like 'gazebo' or 'ign' that would nuke unrelated processes.
_OWN_NODE_PATTERNS = [
    'image_marker_detector_node',
    'homography_sim_node',
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
    """Kill only repo-specific node processes that outlived their PGID teardown."""
    for pattern in _OWN_NODE_PATTERNS:
        try:
            subprocess.run(
                ['pkill', '-f', pattern],
                timeout=2, capture_output=True, check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass


def _method_launch_args(method_id: str, args, gp_dir: Path) -> list[str]:
    if method_id == 'visibility_unaware_baseline':
        launch_args = [
            f'comparison_method_id:=visibility_unaware_baseline',
            f'planner:=visibility_unaware_baseline',
            f'perception_backend:={str(args.baseline_backend).strip()}',
        ]
        if str(args.baseline_backend).strip() == 'yolo':
            if not str(args.yolo_model).strip():
                raise RuntimeError('baseline_backend=yolo requires --yolo-model')
            launch_args.extend([
                f'yolo_model:={str(Path(args.yolo_model).expanduser().resolve())}',
                f'yolo_imgsz:={int(args.yolo_imgsz)}',
                f'yolo_conf_threshold:={float(args.yolo_conf_threshold)}',
                f'yolo_iou_threshold:={float(args.yolo_iou_threshold)}',
                f'yolo_target_class:={str(args.yolo_target_class)}',
                f'yolo_class_id:={int(args.yolo_class_id)}',
                f'yolo_use_masks:={str(args.yolo_use_masks)}',
                f'yolo_min_mask_area_px:={float(args.yolo_min_mask_area_px)}',
                f'yolo_mask_bottom_band_px:={float(args.yolo_mask_bottom_band_px)}',
            ])
            if str(args.yolo_device).strip():
                launch_args.insert(1, f'yolo_device:={str(args.yolo_device).strip()}')
        return launch_args

    artifact = gp_dir / f'{method_id}_gp.npz'
    if not artifact.is_file():
        raise RuntimeError(f'GP artifact not found for method {method_id}: {artifact}')

    if method_id in ('red_binary', 'red_area_corrected'):
        perception_backend = 'image_markers'
    elif method_id in ('yolo_binary', 'yolo_score_raw', 'yolo_score_calibrated'):
        perception_backend = 'yolo'
    elif method_id == 'oracle_visibility':
        perception_backend = str(args.oracle_backend).strip()
    else:
        raise RuntimeError(f'Unsupported method id: {method_id}')

    launch_args = [
        f'comparison_method_id:={method_id}',
        f'planner:={str(args.planner).strip()}',
        f'perception_backend:={perception_backend}',
        f'visibility_artifact_path:={artifact.resolve()}',
    ]
    if perception_backend == 'yolo':
        if not str(args.yolo_model).strip():
            raise RuntimeError(f'{method_id} requires --yolo-model because it uses the live YOLO detector stack')
        launch_args.extend([
            f'yolo_model:={str(Path(args.yolo_model).expanduser().resolve())}',
            f'yolo_imgsz:={int(args.yolo_imgsz)}',
            f'yolo_conf_threshold:={float(args.yolo_conf_threshold)}',
            f'yolo_iou_threshold:={float(args.yolo_iou_threshold)}',
            f'yolo_target_class:={str(args.yolo_target_class)}',
            f'yolo_class_id:={int(args.yolo_class_id)}',
            f'yolo_use_masks:={str(args.yolo_use_masks)}',
            f'yolo_min_mask_area_px:={float(args.yolo_min_mask_area_px)}',
            f'yolo_mask_bottom_band_px:={float(args.yolo_mask_bottom_band_px)}',
        ])
        if str(args.yolo_device).strip():
            launch_args.insert(1, f'yolo_device:={str(args.yolo_device).strip()}')
    return launch_args


def main() -> int:
    parser = argparse.ArgumentParser(description='Run full detector-stack planner comparisons for visibility methods.')
    parser.add_argument('--methods', nargs='*', default=list(COMPARISON_METHODS) + ['visibility_unaware_baseline'])
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--task', default='main_shadow_tradeoff')
    parser.add_argument('--planner', default='efe1')
    parser.add_argument('--launch-file', default='warehouse_primary_comparison.launch.py')
    parser.add_argument('--gp-dir', default=str(CURRENT_GP_DIR))
    parser.add_argument('--log-root', default=str(LOGS_ROOT / 'planner_runs'))
    parser.add_argument('--seed', default='0')
    parser.add_argument('--use-rviz', default='false')
    parser.add_argument('--oracle-backend', default='image_markers', help='Live perception backend to pair with oracle_visibility')
    parser.add_argument('--baseline-backend', default='image_markers', help='Live perception backend to pair with visibility_unaware_baseline')
    parser.add_argument('--yolo-model', default='', help='Local YOLO model path for yolo_* methods and optional yolo baseline/oracle runs')
    parser.add_argument('--yolo-device', default='')
    parser.add_argument('--yolo-imgsz', type=int, default=640)
    parser.add_argument('--yolo-conf-threshold', type=float, default=0.25)
    parser.add_argument('--yolo-iou-threshold', type=float, default=0.45)
    parser.add_argument('--yolo-target-class', default='robot')
    parser.add_argument('--yolo-class-id', type=int, default=-1)
    parser.add_argument('--yolo-use-masks', default='true')
    parser.add_argument('--yolo-min-mask-area-px', type=float, default=12.0)
    parser.add_argument('--yolo-mask-bottom-band-px', type=float, default=3.0)
    parser.add_argument('--targets-manifest', default=str(CURRENT_TARGETS_DIR / 'target_manifest.json'))
    parser.add_argument('--world-profiles', default=str(DEFAULT_WORLD_PROFILES_PATH))
    parser.add_argument('--timeout', type=float, default=180.0, help='Timeout in seconds for each planner run, including simulator startup')
    parser.add_argument('--cleanup-delay', type=float, default=5.0, help='Delay in seconds between runs for cleanup')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    methods = [str(method).strip() for method in args.methods if str(method).strip()]
    invalid = [method for method in methods if method not in ACTIVE_METHOD_IDS]
    if invalid:
        raise RuntimeError(f'Unknown method ids: {invalid}. Expected subset of {ACTIVE_METHOD_IDS}')
    _validate_yolo_runtime_matches_targets(args, methods)

    gp_dir = Path(args.gp_dir).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()
    allowed_root = LOGS_ROOT.resolve()
    if allowed_root not in log_root.parents and log_root != allowed_root:
        raise RuntimeError(f'Planner log root must stay under {allowed_root}: {log_root}')
    log_root.mkdir(parents=True, exist_ok=True)

    expected_metadata = expected_visibility_metadata(
        str(args.world),
        world_profiles_path=Path(args.world_profiles).expanduser().resolve(),
    )
    for method_id in methods:
        if method_id == 'visibility_unaware_baseline':
            continue
        artifact = gp_dir / f'{method_id}_gp.npz'
        if artifact.is_file():
            validate_visibility_metadata(
                load_visibility_artifact_metadata(artifact),
                expected_metadata,
                label=f'visibility artifact {artifact}',
            )

    print('Cleaning up any stale own-node processes before first run...')
    _reap_own_node_stragglers()
    time.sleep(args.cleanup_delay)

    base_cmd = [
        'ros2', 'launch', 'experiments', str(args.launch_file),
        f'world:={str(args.world)}',
        f'task:={str(args.task)}',
        f'seed:={str(args.seed)}',
        f'use_rviz:={str(args.use_rviz)}',
    ]

    for method_id in methods:
        # Cleanup before starting a new run
        if method_id != methods[0]:
            print(f'Cleaning up between runs (waiting {args.cleanup_delay}s)...')
            _reap_own_node_stragglers()
            time.sleep(args.cleanup_delay)
        
        method_log_root = (log_root / method_id).resolve()
        method_log_root.mkdir(parents=True, exist_ok=True)
        
        # Make log_dir relative to repo root for ros2 launch
        try:
            log_dir_relative = str(method_log_root.relative_to(Path.cwd()))
        except ValueError:
            # If not relative, use absolute path
            log_dir_relative = str(method_log_root)
        
        launch_args = _method_launch_args(method_id, args, gp_dir)
        launch_args.append(f'log_dir:={log_dir_relative}')
        cmd = base_cmd + launch_args
        print('Running:', ' '.join(str(part) for part in cmd))
        if args.dry_run:
            continue
        process = subprocess.Popen(cmd, start_new_session=True)
        pgid = os.getpgid(process.pid)
        try:
            process.wait(timeout=args.timeout)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        except subprocess.TimeoutExpired:
            print(f'⏱️  Timeout after {args.timeout}s for method {method_id}. Results saved to {method_log_root}')
        finally:
            _terminate_process_group(pgid)
            _reap_own_node_stragglers()
            time.sleep(max(args.cleanup_delay, 1.0))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
