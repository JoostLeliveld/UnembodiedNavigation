#!/usr/bin/env python3
"""Fine-tune a YOLO11n-seg model and write one small run folder."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def main() -> int:
    parser = argparse.ArgumentParser(description='Fine-tune YOLO11n-seg on a local dataset.')
    parser.add_argument('--data', required=True, help='Path to data.yaml')
    parser.add_argument('--base-model', required=True, help='Base YOLO segmentation model, e.g. yolo11n-seg.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--out', required=True, help='Output run folder')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.is_file():
        raise RuntimeError(f'Dataset yaml does not exist: {data_yaml}')
    base_model = Path(args.base_model).expanduser().resolve()
    if not base_model.is_file():
        raise RuntimeError(f'Base model does not exist: {base_model}')
    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        raise RuntimeError(f'Output folder already exists: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=False)

    train_root = out_dir / 'ultralytics_run'
    model = YOLO(str(base_model))
    results = model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        project=str(train_root.parent),
        name=train_root.name,
    )
    run_dir = Path(getattr(results, 'save_dir', train_root)).resolve()
    best_model = run_dir / 'weights' / 'best.pt'
    if not best_model.is_file():
        raise RuntimeError(f'Ultralytics did not produce best.pt in {run_dir}')

    model_out = out_dir / 'model.pt'
    shutil.copy2(best_model, model_out)
    for filename in ('results.csv', 'results.png', 'labels.jpg', 'confusion_matrix.png'):
        src = run_dir / filename
        if src.is_file():
            shutil.copy2(src, out_dir / filename)

    manifest = {
        'base_model': str(base_model),
        'dataset_yaml': str(data_yaml),
        'epochs': int(args.epochs),
        'image_size': int(args.imgsz),
        'batch': int(args.batch),
        'device': str(args.device),
        'output_model': str(model_out),
        'training_run_dir': str(run_dir),
        'timestamp': _timestamp(),
        'confidence_calibrated': False,
        'notes': [
            'Start with 5-10 epochs for a smoke test.',
            'Use 20-30 epochs for a first usable fine-tune.',
            'Detector confidence is not calibrated.',
        ],
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote model to {model_out}')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
