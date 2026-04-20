#!/usr/bin/env python3
"""Run planner comparisons sequentially with robust method-matched live perception backends."""

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
    accepted_completed_run,
    expected_visibility_metadata,
    load_visibility_artifact_metadata,
    normalize_run_manifest_for_hash,
    run_has_usable_logs,
    sha256_json,
    validate_visibility_metadata,
)


COMPARISON_METHODS = tuple(method for method in ACTIVE_METHOD_IDS if method != 'visibility_unaware_baseline')


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _uses_live_yolo(method_id: str, args) -> bool:
    if method_id in ('yolo_binary', 'yolo_score_raw', 'yolo_score_calibrated'):
        return True
    if method_id == 'oracle_visibility':
        return str(args.oracle_backend).strip() == 'yolo'
    if method_id == 'visibility_unaware_baseline':
        return str(args.baseline_backend).strip() == 'yolo'
    return False


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def _validate_yolo_runtime_matches_targets(args, methods: list[str]) -> None:
    if not any(_uses_live_yolo(method_id, args) for method_id in methods):
        return
    manifest_path = Path(args.targets_manifest).expanduser().resolve()
    payload = _load_json(manifest_path)
    if not payload:
        raise RuntimeError(f'YOLO methods require a valid targets manifest for config matching: {manifest_path}')

    implemented = [str(item).strip() for item in payload.get('implemented_targets', [])]
    required_targets = {'yolo_binary', 'yolo_score_raw', 'yolo_score_calibrated'}
    if not required_targets.issubset(set(implemented)):
        raise RuntimeError(
            f'Targets manifest does not contain the required YOLO targets {sorted(required_targets)}: {manifest_path}'
        )

    expected_world = expected_visibility_metadata(str(args.world), world_profiles_path=Path(args.world_profiles))
    capture_metadata = payload.get('capture_metadata', {})
    if not isinstance(capture_metadata, dict) or not capture_metadata:
        raise RuntimeError(
            f'Targets manifest is missing capture_metadata for world/camera validation: {manifest_path}'
        )
    validate_visibility_metadata(capture_metadata, expected_world, label='targets manifest')

    yolo_summary = payload.get('yolo_summary', {})
    if not isinstance(yolo_summary, dict) or not bool(yolo_summary.get('enabled', False)):
        raise RuntimeError(f'Targets manifest does not contain an enabled yolo_summary: {manifest_path}')

    runtime_model = str(Path(args.yolo_model).expanduser().resolve()) if str(args.yolo_model).strip() else ''
    manifest_model = (
        str(Path(str(yolo_summary.get('model_path', '') or '')).expanduser().resolve())
        if str(yolo_summary.get('model_path', '') or '').strip()
        else ''
    )
    expected = {
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
        if str(runtime[key]).strip() != str(expected[key]).strip():
            mismatches.append(f'{key}: runtime={runtime[key]!r} extractor={expected[key]!r}')
    for key in ('conf_threshold', 'iou_threshold', 'mask_min_area', 'mask_bottom_band_px'):
        if abs(float(runtime[key]) - float(expected[key])) > 1e-9:
            mismatches.append(f'{key}: runtime={runtime[key]!r} extractor={expected[key]!r}')
    if runtime['use_masks'] != expected['use_masks']:
        mismatches.append(f"use_masks: runtime={runtime['use_masks']!r} extractor={expected['use_masks']!r}")
    if mismatches:
        raise RuntimeError(
            'Runtime YOLO configuration does not match the extracted yolo_summary in '
            f'{manifest_path}: ' + '; '.join(mismatches)
        )


def _artifact_path_for_method(method_id: str, gp_dir: Path) -> Path | None:
    if method_id == 'visibility_unaware_baseline':
        return None
    candidate = gp_dir / f'{method_id}_gp.npz'
    return candidate if candidate.is_file() else None


def _validate_artifacts_match_world(args, methods: list[str], gp_dir: Path) -> None:
    expected_world = expected_visibility_metadata(str(args.world), world_profiles_path=Path(args.world_profiles))
    for method_id in methods:
        artifact = _artifact_path_for_method(method_id, gp_dir)
        if artifact is None:
            if method_id == 'visibility_unaware_baseline':
                continue
            raise RuntimeError(f'GP artifact not found for method {method_id}: {gp_dir / f"{method_id}_gp.npz"}')
        metadata = load_visibility_artifact_metadata(artifact)
        if int(metadata.get('artifact_schema_version', 0)) < 2:
            raise RuntimeError(
                f'Visibility artifact {artifact} uses an outdated schema; re-run fit_visibility_gps.py before sweeping.'
            )
        validate_visibility_metadata(metadata, expected_world, label=f'{method_id} artifact')


def _compute_run_config_hash(run_dir: Path) -> tuple[str, dict]:
    run_manifest = _load_json(run_dir / 'run_manifest.json') or _load_json(run_dir / 'manifest.json')
    if not run_manifest:
        raise RuntimeError(f'Run manifest missing from {run_dir}')
    normalized_manifest = normalize_run_manifest_for_hash(run_manifest)
    artifact_signature = {}
    artifact_path = str(run_manifest.get('visibility_artifact_path', '') or '').strip()
    if artifact_path:
        artifact_signature = load_visibility_artifact_metadata(artifact_path)
    config = {
        'run_manifest': normalized_manifest,
        'artifact_signature': artifact_signature,
        'git_commit': _git_commit_hash(),
    }
    return sha256_json(config)[:16], config


def _augment_run_summary_with_hash(run_dir: Path, run_hash: str, config: dict) -> None:
    summary_path = run_dir / 'run_summary.json'
    data = _load_json(summary_path)
    if not data:
        return
    data['run_config_hash'] = run_hash
    data['run_config'] = config
    try:
        summary_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    except OSError as exc:
        print(f'Warning: could not write run config hash to {summary_path}: {exc}')


def _kill_process_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(1.0)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as exc:
        print(f'Error killing process group {pgid}: {exc}')


def _cleanup_gazebo_and_ros() -> None:
    cleanup_commands = [
        (['pkill', '-f', 'ign gazebo'], False),
        (['pkill', '-f', 'gzserver'], False),
        (['pkill', '-9', '-f', 'gazebo'], True),
        (['pkill', '-f', 'ros2 launch'], False),
        (['pkill', '-9', '-f', 'ign'], True),
    ]
    for cmd, _force_kill in cleanup_commands:
        try:
            subprocess.run(cmd, timeout=2, capture_output=True, text=True)
        except Exception:
            pass


def _method_launch_args(method_id: str, args, gp_dir: Path) -> list[str]:
    if method_id == 'visibility_unaware_baseline':
        launch_args = [
            'comparison_method_id:=visibility_unaware_baseline',
            'planner:=visibility_unaware_baseline',
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

    artifact = _artifact_path_for_method(method_id, gp_dir)
    if artifact is None:
        raise RuntimeError(f'GP artifact not found for method {method_id}: {gp_dir / f"{method_id}_gp.npz"}')

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


def _verify_completed_run(run_dir: Path) -> bool:
    summary = _load_json(run_dir / 'run_summary.json')
    return accepted_completed_run(summary) and run_has_usable_logs(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run full detector-stack planner comparisons sequentially.')
    parser.add_argument('--methods', nargs='*', default=list(COMPARISON_METHODS) + ['visibility_unaware_baseline'])
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=str(DEFAULT_WORLD_PROFILES_PATH))
    parser.add_argument('--task', default='main_shadow_tradeoff')
    parser.add_argument('--planner', default='efe1')
    parser.add_argument('--launch-file', default='warehouse_primary_comparison.launch.py')
    parser.add_argument('--gp-dir', default=str(CURRENT_GP_DIR))
    parser.add_argument('--log-root', default=str(LOGS_ROOT / 'planner_runs'))
    parser.add_argument('--seeds', nargs='+', default=['0'], help='One or multiple integer seeds (e.g., 0 1 2)')
    parser.add_argument('--use-rviz', default='false')
    parser.add_argument('--oracle-backend', default='image_markers')
    parser.add_argument('--baseline-backend', default='image_markers')
    parser.add_argument('--yolo-model', default='')
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
    parser.add_argument('--timeout', type=float, default=150.0, help='Planner timeout in SIM TIME seconds (from first command)')
    parser.add_argument('--wall-timeout', type=float, default=1200.0, help='Hard wall-clock timeout in seconds to kill the run (must be > timeout/RTF)')
    parser.add_argument('--cleanup-delay', type=float, default=3.0, help='Delay in seconds between runs')
    parser.add_argument('--continue-on-failure', action='store_true')
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

    _validate_yolo_runtime_matches_targets(args, methods)
    _validate_artifacts_match_world(args, methods, gp_dir)

    logger_timeout_s = float(args.timeout)
    print(
        f'Timeout config: sim_timeout={logger_timeout_s:.0f}s (for planner), '
        f'wall_kill={args.wall_timeout:.0f}s (process hard kill buffer)'
    )

    for method_id in methods:
        for seed_val in getattr(args, 'seeds', ['0']):
            seed_val = str(seed_val).strip()
            args.current_seed = seed_val
            base_cmd = [
                'ros2', 'launch', 'experiments', str(args.launch_file),
                f'world:={str(args.world)}',
                f'task:={str(args.task)}',
                f'seed:={seed_val}',
                f'use_rviz:={str(args.use_rviz)}',
                f'run_timeout_after_first_cmd_s:={logger_timeout_s}',
            ]

            print(f'\n--- Starting {method_id} [seed: {seed_val}] ---')
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
                start_wall = time.time()
                run_dir = None
                while time.time() - start_wall < 15.0:
                    dirs = [d for d in method_log_root.iterdir() if d.is_dir()]
                    if dirs:
                        latest = max(dirs, key=os.path.getmtime)
                        if latest.stat().st_mtime >= start_wall - 5.0:
                            run_dir = latest
                            break
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)

                if run_dir:
                    summary_file = run_dir / 'run_summary.json'
                    done = False
                    while time.time() - start_wall < args.wall_timeout:
                        if summary_file.exists():
                            time.sleep(0.2)
                            done = True
                            break
                        if process.poll() is not None:
                            break
                        time.sleep(0.5)
                    if not done and process.poll() is None:
                        print(f'⏱️ Hard wall-clock timeout reached ({args.wall_timeout}s). The run was killed.')
                else:
                    process.wait(timeout=args.wall_timeout)
            except subprocess.TimeoutExpired:
                print(f'⏱️ Hard wall-clock timeout reached ({args.wall_timeout}s). The run was killed.')
            finally:
                _kill_process_group(pgid)
                _cleanup_gazebo_and_ros()

            latest_run_dir = max([Path(d) for d in method_log_root.iterdir() if d.is_dir()], key=os.path.getmtime, default=None)
            if latest_run_dir and _verify_completed_run(latest_run_dir):
                print(f'✅ Run completed validly in {latest_run_dir}')
                run_hash, run_config = _compute_run_config_hash(latest_run_dir)
                _augment_run_summary_with_hash(latest_run_dir, run_hash, run_config)
                print(f'   run_config_hash: {run_hash}')
            else:
                print(f'❌ Run in {latest_run_dir} did NOT complete cleanly.')
                if not args.continue_on_failure:
                    raise RuntimeError(f'{method_id} (seed {seed_val}) did not complete cleanly in {latest_run_dir}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
