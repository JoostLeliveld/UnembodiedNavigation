#!/usr/bin/env python3
"""Build the supervisor-aligned segmentation perception pipeline.

Pipeline:
1. Capture geometric robot bounding boxes from the simulator.
2. Use those boxes as SAM prompts to produce YOLO-seg pseudo-labels.
3. Train YOLO11n-seg so runtime perception predicts a robot class, mask, and confidence.

The geometric box identifies which object is the robot. SAM provides the visible-object
mask pseudo-label inside that prompt. The runtime YOLO-seg node then uses the bottom of
the predicted mask as the homography pixel and logs the detector confidence/logit for
analysis. Simulator ground truth and SAM are offline helpers only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / 'scripts'


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print('\n$ ' + ' '.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _latest_dir(root: Path, *, start_time: float, prefix: str) -> Path:
    if not root.is_dir():
        raise RuntimeError(f'Expected output root does not exist: {root}')
    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and p.stat().st_mtime >= start_time - 1.0
    ]
    if not candidates:
        candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise RuntimeError(f'Could not find generated directory in {root} with prefix {prefix!r}')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _data_yaml(dataset: Path) -> Path:
    dataset = dataset.expanduser().resolve()
    if dataset.is_file() and dataset.suffix.lower() in ('.yaml', '.yml'):
        return dataset
    candidate = dataset / 'data.yaml'
    if not candidate.is_file():
        raise RuntimeError(f'Could not find data.yaml for dataset: {dataset}')
    return candidate


def _dataset_name(dataset: Path) -> str:
    dataset = dataset.expanduser().resolve()
    return dataset.parent.name if dataset.is_file() else dataset.name


def _capture_geometric_bboxes(args) -> Path | None:
    output_root = Path(args.bbox_output_root).expanduser().resolve()
    start = time.time()
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / 'capture_yolo_dataset.py'),
        '--world', str(args.world),
        '--world-profiles', str(Path(args.world_profiles).expanduser()),
        '--output-root', str(output_root),
        '--sample-nx', str(args.sample_nx),
        '--sample-ny', str(args.sample_ny),
        '--sample-limit', str(args.sample_limit),
        '--repeats', str(args.repeats),
        '--wall-margin-m', str(args.wall_margin_m),
        '--yaw-rad', str(args.yaw_rad),
        '--robot-z', str(args.robot_z),
        '--robot-half-x', str(args.robot_half_x),
        '--robot-half-y', str(args.robot_half_y),
        '--robot-height-m', str(args.robot_height_m),
        '--bbox-pad-px', str(args.bbox_pad_px),
        '--val-fraction', str(args.val_fraction),
        '--seed', str(args.seed),
        '--ready-timeout-s', str(args.ready_timeout_s),
        '--settle-s', str(args.settle_s),
        '--image-timeout-s', str(args.image_timeout_s),
        '--service-timeout-s', str(args.service_timeout_s),
        '--min-label-red-px', str(args.min_capture_red_px),
        '--preview-count', str(args.preview_count),
    ]
    _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return output_root / 'yolo_dataset_<timestamp>'
    return _latest_dir(output_root, start_time=start, prefix='yolo_dataset_')


def _sam_label_dataset(args, input_dataset: Path) -> Path | None:
    output_root = Path(args.seg_output_root).expanduser().resolve()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f'{_dataset_name(input_dataset)}_sam_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    start = time.time()
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / 'convert_yolo_labels_to_seg_by_sam.py'),
        str(input_dataset),
        '--output-dir', str(output_dir),
        '--sam-model', str(args.sam_model),
        '--device', str(args.sam_device),
        '--imgsz', str(args.sam_imgsz),
        '--prompt-mode', str(args.sam_prompt_mode),
        '--prompt-pad-px', str(args.sam_prompt_pad_px),
        '--min-mask-area-px', str(args.min_mask_area_px),
        '--min-bbox-overlap-ratio', str(args.min_bbox_overlap_ratio),
        '--polygon-epsilon-frac', str(args.polygon_epsilon_frac),
        '--max-polygon-points', str(args.max_polygon_points),
        '--preview-count', str(args.preview_count),
    ]
    if int(args.label_limit) > 0:
        cmd.extend(['--limit', str(args.label_limit)])
    if bool(args.skip_empty_source_labels):
        cmd.append('--skip-empty-source-labels')
    _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return output_dir
    if output_dir.is_dir():
        return output_dir
    return _latest_dir(output_root, start_time=start, prefix='yolo_dataset_')


def _train_yolo_seg(args, seg_dataset: Path) -> Path | None:
    project = Path(args.train_project).expanduser().resolve()
    start = time.time()
    data_yaml = (seg_dataset / 'data.yaml') if args.dry_run else _data_yaml(seg_dataset)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / 'train_yolo_robot.py'),
        '--task', 'segment',
        '--data', str(data_yaml),
        '--model', str(args.yolo_seg_model),
        '--epochs', str(args.epochs),
        '--imgsz', str(args.train_imgsz),
        '--batch', str(args.batch),
        '--project', str(project),
        '--name', str(args.train_name),
    ]
    if str(args.train_device).strip():
        cmd.extend(['--device', str(args.train_device).strip()])
    _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return project / str(args.train_name)
    candidates = [
        p for p in project.iterdir()
        if p.is_dir() and p.name.startswith(str(args.train_name)) and p.stat().st_mtime >= start - 1.0
    ]
    if not candidates:
        candidates = [p for p in project.iterdir() if p.is_dir() and p.name.startswith(str(args.train_name))]
    if not candidates:
        raise RuntimeError(f'Could not find training run in {project}')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _print_runtime_command(weights: Path | None, args) -> None:
    model = str(weights) if weights is not None else '<path-to-trained-best.pt>'
    print('\nRuntime command:')
    print(
        'ros2 launch experiments warehouse_primary_comparison.launch.py '
        'perception_backend:=yolo '
        f'yolo_model:={model} '
        'yolo_target_class:=robot '
        'yolo_use_masks:=true '
        f'yolo_imgsz:={args.train_imgsz} '
        'use_rviz:=false'
    )
    print('\nRuntime interpretation:')
    print('- YOLO-seg detects the robot class and predicts an instance mask.')
    print('- The pixel sent to /perception/pixel_pose is the bottom band of that mask.')
    print('- /state/bev uses homography for x,y and odometry-backed theta as before.')
    print('- yolo_score is the detector confidence; confidence_logit is logit(yolo_score).')
    print('- GP fitting should use --target-mode yolo_soft_score for the soft observability field.')
    print('- SAM and simulator-projected boxes are not used at runtime; they are offline training helpers only.')


def _write_summary(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f'\nWrote pipeline summary to {path}')


def main() -> int:
    stage_choices = ('capture-bbox', 'sam-label', 'train', 'all', 'guide')
    parser = argparse.ArgumentParser(description='Build geometric-box -> SAM-mask -> YOLO-seg robot perception.')
    parser.add_argument(
        'stage_pos',
        nargs='?',
        choices=stage_choices,
        help='Which stage to run; equivalent to --stage.',
    )
    parser.add_argument('--stage', dest='stage_opt', choices=stage_choices, default='', help='Which stage to run.')
    parser.add_argument('--bbox-dataset', default='', help='Existing geometric YOLO bbox dataset for sam-label/all.')
    parser.add_argument('--seg-dataset', default='', help='Existing YOLO-seg dataset for train/guide.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running them.')

    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=str(REPO_ROOT / 'src/experiments/config/world_profiles.yaml'))
    parser.add_argument('--bbox-output-root', default=str(REPO_ROOT / 'logs/yolo_datasets'))
    parser.add_argument('--seg-output-root', default=str(REPO_ROOT / 'logs/yolo_seg_datasets'))
    parser.add_argument('--sample-nx', type=int, default=15)
    parser.add_argument('--sample-ny', type=int, default=15)
    parser.add_argument('--sample-limit', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--wall-margin-m', type=float, default=0.45)
    parser.add_argument('--yaw-rad', type=float, default=0.0)
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--robot-half-x', type=float, default=0.11)
    parser.add_argument('--robot-half-y', type=float, default=0.11)
    parser.add_argument('--robot-height-m', type=float, default=0.20)
    parser.add_argument('--bbox-pad-px', type=float, default=8.0)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ready-timeout-s', type=float, default=20.0)
    parser.add_argument('--settle-s', type=float, default=0.25)
    parser.add_argument('--image-timeout-s', type=float, default=1.0)
    parser.add_argument('--service-timeout-s', type=float, default=20.0)
    parser.add_argument(
        '--min-capture-red-px',
        type=float,
        default=0.0,
        help='Optional capture-time quality gate; >0 blanks projected bbox labels with little visible red evidence.',
    )

    parser.add_argument('--sam-model', default='mobile_sam.pt')
    parser.add_argument('--sam-device', default='cpu')
    parser.add_argument('--sam-imgsz', type=int, default=1024)
    parser.add_argument(
        '--sam-prompt-mode',
        choices=('box', 'point', 'box_then_point'),
        default='box_then_point',
        help='SAM prompt strategy for converting geometric boxes into segmentation pseudo-labels.',
    )
    parser.add_argument('--sam-prompt-pad-px', type=float, default=4.0)
    parser.add_argument('--min-mask-area-px', type=float, default=20.0)
    parser.add_argument('--min-bbox-overlap-ratio', type=float, default=0.50)
    parser.add_argument('--polygon-epsilon-frac', type=float, default=0.010)
    parser.add_argument('--max-polygon-points', type=int, default=96)
    parser.add_argument('--label-limit', type=int, default=0)
    parser.add_argument('--skip-empty-source-labels', action='store_true')
    parser.add_argument('--preview-count', type=int, default=120)

    parser.add_argument('--yolo-seg-model', default='yolo11n-seg.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--train-imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--train-device', default='')
    parser.add_argument('--train-project', default=str(REPO_ROOT / 'logs/yolo_runs'))
    parser.add_argument('--train-name', default='supervisor_yolo11n_seg')
    parser.add_argument('--summary', default=str(REPO_ROOT / 'logs/yolo_seg_pipeline/last_summary.json'))
    args = parser.parse_args()
    if args.stage_pos and args.stage_opt and args.stage_pos != args.stage_opt:
        raise RuntimeError(f'Conflicting stages: positional {args.stage_pos!r} vs --stage {args.stage_opt!r}')
    args.stage = args.stage_opt or args.stage_pos
    if not args.stage:
        parser.error('choose a stage, for example `--stage all` or positional `all`')

    summary: dict[str, str] = {'stage': args.stage}
    bbox_dataset = Path(args.bbox_dataset).expanduser().resolve() if args.bbox_dataset else None
    seg_dataset = Path(args.seg_dataset).expanduser().resolve() if args.seg_dataset else None
    train_dir = None
    if bbox_dataset is not None:
        summary['bbox_dataset'] = str(bbox_dataset)
    if seg_dataset is not None:
        summary['seg_dataset'] = str(seg_dataset)

    if args.stage in ('capture-bbox', 'all') and bbox_dataset is None:
        bbox_dataset = _capture_geometric_bboxes(args)
        if bbox_dataset is not None:
            summary['bbox_dataset'] = str(bbox_dataset)

    if args.stage in ('sam-label', 'all'):
        if bbox_dataset is None:
            raise RuntimeError('sam-label/all needs --bbox-dataset or a captured bbox dataset.')
        seg_dataset = _sam_label_dataset(args, bbox_dataset)
        if seg_dataset is not None:
            summary['seg_dataset'] = str(seg_dataset)

    if args.stage in ('train', 'all'):
        if seg_dataset is None:
            raise RuntimeError('train/all needs --seg-dataset or a SAM-labeled dataset.')
        train_dir = _train_yolo_seg(args, seg_dataset)
        if train_dir is not None:
            summary['train_dir'] = str(train_dir)
            summary['best_pt'] = str(train_dir / 'weights' / 'best.pt')

    if args.stage == 'guide':
        if seg_dataset is not None:
            summary['seg_dataset'] = str(seg_dataset)

    weights = Path(summary['best_pt']) if 'best_pt' in summary else None
    _print_runtime_command(weights, args)
    if not args.dry_run:
        _write_summary(Path(args.summary).expanduser().resolve(), summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
