#!/usr/bin/env python3
"""Run planner comparisons sequentially with robust method-matched live perception backends."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from common import ACTIVE_METHOD_IDS, CURRENT_GP_DIR, LOGS_ROOT

COMPARISON_METHODS = tuple(method for method in ACTIVE_METHOD_IDS if method != 'visibility_unaware_baseline')

def _kill_process_group(pgid: int):
    """Aggressively terminate a process group."""
    try:
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(1.0)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        print(f"Error killing process group {pgid}: {e}")

def _cleanup_gazebo_and_ros():
    """Kill any remaining gazebo and ros2 processes."""
    cleanup_commands = [
        (['pkill', '-f', 'ign gazebo'], False),
        (['pkill', '-f', 'gzserver'], False),
        (['pkill', '-9', '-f', 'gazebo'], True),
        (['pkill', '-f', 'ros2 launch'], False),
        (['pkill', '-9', '-f', 'ign'], True),
    ]
    for cmd, force_kill in cleanup_commands:
        try:
            subprocess.run(cmd, timeout=2, capture_output=True, text=True)
        except Exception:
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
            f'yolo_use_masks:={str(args.yolo_use_masks)}',
            f'yolo_min_mask_area_px:={float(args.yolo_min_mask_area_px)}',
            f'yolo_mask_bottom_band_px:={float(args.yolo_mask_bottom_band_px)}',
        ])
        if str(args.yolo_device).strip():
            launch_args.insert(1, f'yolo_device:={str(args.yolo_device).strip()}')
    return launch_args

def _verify_completed_run(run_dir: Path) -> bool:
    summary_file = run_dir / 'run_summary.json'
    if not summary_file.is_file():
        return False
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return bool(data.get('completed', False)) and data.get('completion_reason') in ('goal_reached', 'stuck', 'timeout_after_first_cmd')
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
        return False

def main() -> int:
    parser = argparse.ArgumentParser(description='Run full detector-stack planner comparisons sequentially.')
    parser.add_argument('--methods', nargs='*', default=list(COMPARISON_METHODS) + ['visibility_unaware_baseline'])
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--task', default='E0')
    parser.add_argument('--planner', default='efe1')
    parser.add_argument('--launch-file', default='warehouse_primary_comparison.launch.py')
    parser.add_argument('--gp-dir', default=str(CURRENT_GP_DIR))
    parser.add_argument('--log-root', default=str(LOGS_ROOT / 'planner_runs'))
    parser.add_argument('--seed', default='0')
    parser.add_argument('--use-rviz', default='false')
    parser.add_argument('--oracle-backend', default='image_markers')
    parser.add_argument('--baseline-backend', default='image_markers')
    parser.add_argument('--yolo-model', default='')
    parser.add_argument('--yolo-device', default='')
    parser.add_argument('--yolo-imgsz', type=int, default=640)
    parser.add_argument('--yolo-conf-threshold', type=float, default=0.25)
    parser.add_argument('--yolo-iou-threshold', type=float, default=0.45)
    parser.add_argument('--yolo-target-class', default='robot')
    parser.add_argument('--yolo-use-masks', default='true')
    parser.add_argument('--yolo-min-mask-area-px', type=float, default=12.0)
    parser.add_argument('--yolo-mask-bottom-band-px', type=float, default=3.0)
    parser.add_argument('--timeout', type=float, default=150.0, help='Hard process timeout in seconds')
    parser.add_argument('--cleanup-delay', type=float, default=3.0, help='Delay in seconds between runs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    methods = [str(method).strip() for method in args.methods if str(method).strip()]
    invalid = [method for method in methods if method not in ACTIVE_METHOD_IDS]
    if invalid:
        raise RuntimeError(f'Unknown method ids: {invalid}. Expected subset of {ACTIVE_METHOD_IDS}')

    gp_dir = Path(args.gp_dir).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        'ros2', 'launch', 'experiments', str(args.launch_file),
        f'world:={str(args.world)}',
        f'task:={str(args.task)}',
        f'seed:={str(args.seed)}',
        f'use_rviz:={str(args.use_rviz)}',
    ]

    for method_id in methods:
        print(f'\n--- Starting {method_id} ---')
        _cleanup_gazebo_and_ros()
        time.sleep(args.cleanup_delay)
        
        method_log_root = (log_root / method_id).resolve()
        method_log_root.mkdir(parents=True, exist_ok=True)
        
        try:
            log_dir_relative = str(method_log_root.relative_to(Path.cwd()))
        except ValueError:
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
        except subprocess.TimeoutExpired:
            print(f'⏱️ Hard timeout reached ({args.timeout}s). The logger may have missed its own timeout.')
        finally:
            _kill_process_group(pgid)
            _cleanup_gazebo_and_ros()
            
        latest_run_dir = max([Path(d) for d in method_log_root.iterdir() if d.is_dir()], key=os.path.getmtime, default=None)
        if latest_run_dir and _verify_completed_run(latest_run_dir):
            print(f"✅ Run completed validly in {latest_run_dir}")
        else:
            print(f"❌ Run in {latest_run_dir} did NOT complete cleanly.")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
