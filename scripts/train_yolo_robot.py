#!/usr/bin/env python3
"""Train a one-class robot detector with Ultralytics YOLO."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description='Train a YOLO robot detector from a YOLO-format data.yaml.')
    parser.add_argument('--data', required=True, help='Path to YOLO data.yaml')
    parser.add_argument('--model', default='yolo11n.pt', help='Ultralytics model name/path')
    parser.add_argument('--task', choices=('auto', 'detect', 'segment'), default='auto')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='')
    parser.add_argument('--project', default='')
    parser.add_argument('--name', default='robot_yolo11n')
    parser.add_argument('--hf-repo-id', default='', help='Optional Hugging Face repo id to upload/export later; not pushed by this script.')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install Ultralytics first: `pip install ultralytics`') from exc

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise RuntimeError(f'data.yaml not found: {data_path}')

    model = YOLO(str(args.model))
    kwargs = {
        'data': str(data_path),
        'epochs': int(args.epochs),
        'imgsz': int(args.imgsz),
        'batch': int(args.batch),
        'name': str(args.name),
    }
    if str(args.task).strip().lower() != 'auto':
        kwargs['task'] = str(args.task).strip().lower()
    if str(args.device).strip():
        kwargs['device'] = str(args.device).strip()
    if str(args.project).strip():
        kwargs['project'] = str(Path(args.project).expanduser().resolve())
    results = model.train(**kwargs)
    print(results)
    print('Training complete. Use the reported best.pt as yolo_model, or upload/store it on Hugging Face Hub manually.')
    if str(args.task).strip().lower() == 'segment' or str(args.model).endswith('-seg.pt'):
        print('Segmentation note: the runtime YOLO node will use masks automatically when yolo_use_masks:=true.')
    if args.hf_repo_id:
        print(
            f'Hugging Face repo requested ({args.hf_repo_id}), but this script does not push automatically. '
            'Keep best.pt as an explicit artifact to avoid accidental credential-dependent uploads.'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
