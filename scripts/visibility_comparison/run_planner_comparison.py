#!/usr/bin/env python3
"""Run planner comparisons with method-matched live perception backends."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from common import ACTIVE_METHOD_IDS, CURRENT_GP_DIR, LOGS_ROOT


COMPARISON_METHODS = tuple(method for method in ACTIVE_METHOD_IDS if method != 'visibility_unaware_baseline')


def _cleanup_gazebo_and_ros():
    """Kill any remaining gazebo and ros2 processes."""
    # Try graceful shutdown first, then forceful if needed
    cleanup_commands = [
        (['pkill', '-f', 'ign gazebo'], False),  # Ignition Gazebo
        (['pkill', '-f', 'gzserver'], False),    # Gazebo Server
        (['pkill', '-9', '-f', 'gazebo'], True), # Force kill old gazebo
        (['pkill', '-f', 'ros2 launch'], False), # ROS2 launch processes
        (['pkill', '-9', '-f', 'ign'], True),    # Force kill ign processes
    ]
    
    for cmd, force_kill in cleanup_commands:
        try:
            subprocess.run(cmd, timeout=2, capture_output=True, text=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
        except Exception:
            pass


def _method_launch_args(method_id: str, args, gp_dir: Path) -> list[str]:
    if method_id == 'visibility_unaware_baseline':
        launch_args = [
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
    elif method_id in ('yolo_binary', 'yolo_confidence'):
        perception_backend = 'yolo'
    elif method_id == 'oracle_visibility':
        perception_backend = str(args.oracle_backend).strip()
    else:
        raise RuntimeError(f'Unsupported method id: {method_id}')

    launch_args = [
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
    parser.add_argument('--task', default='E0')
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
    parser.add_argument('--yolo-use-masks', default='true')
    parser.add_argument('--yolo-min-mask-area-px', type=float, default=12.0)
    parser.add_argument('--yolo-mask-bottom-band-px', type=float, default=3.0)
    parser.add_argument('--timeout', type=float, default=60.0, help='Timeout in seconds for each planner run')
    parser.add_argument('--cleanup-delay', type=float, default=5.0, help='Delay in seconds between runs for cleanup')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    methods = [str(method).strip() for method in args.methods if str(method).strip()]
    invalid = [method for method in methods if method not in ACTIVE_METHOD_IDS]
    if invalid:
        raise RuntimeError(f'Unknown method ids: {invalid}. Expected subset of {ACTIVE_METHOD_IDS}')

    gp_dir = Path(args.gp_dir).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()
    allowed_root = LOGS_ROOT.resolve()
    if allowed_root not in log_root.parents and log_root != allowed_root:
        raise RuntimeError(f'Planner log root must stay under {allowed_root}: {log_root}')
    log_root.mkdir(parents=True, exist_ok=True)

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
            print(f'Cleaning up before next run... (waiting {args.cleanup_delay}s)')
            _cleanup_gazebo_and_ros()
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
        try:
            subprocess.run(cmd, check=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f'⏱️  Timeout after {args.timeout}s for method {method_id}. Results saved to {method_log_root}')
            _cleanup_gazebo_and_ros()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
