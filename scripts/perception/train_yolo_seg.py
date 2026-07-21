#!/usr/bin/env python3
"""Fine-tune a YOLO segmentation model with reproducible provenance.

The output directory is immutable: a failed attempt remains as evidence and a
retry must use a new directory.  A training request manifest is written before
Ultralytics starts so interrupted/OOM runs cannot be mistaken for a finished
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _resolve_dataset_root(data_yaml: Path, data: dict) -> Path:
    raw_root = Path(str(data.get('path', data_yaml.parent)))
    if not raw_root.is_absolute():
        raw_root = data_yaml.parent / raw_root
    return raw_root.expanduser().resolve()


def _validate_dataset(data_yaml: Path) -> dict:
    """Validate image/label pairing and return an immutable dataset fingerprint."""

    data = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    root = _resolve_dataset_root(data_yaml, data)
    inventory: list[dict[str, str | int]] = []
    split_counts: dict[str, int] = {}
    for split in ('train', 'val'):
        if split not in data:
            raise RuntimeError(f'Dataset yaml is missing {split!r}: {data_yaml}')
        image_root = Path(str(data[split]))
        if not image_root.is_absolute():
            image_root = root / image_root
        image_root = image_root.resolve()
        if not image_root.is_dir():
            raise RuntimeError(f'{split} image directory does not exist: {image_root}')
        images = sorted(
            path for path in image_root.rglob('*')
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        )
        if not images:
            raise RuntimeError(f'{split} split contains no images: {image_root}')
        labels_root = root / 'labels' / split
        if not labels_root.is_dir():
            raise RuntimeError(f'{split} label directory does not exist: {labels_root}')
        for image_path in images:
            relative = image_path.relative_to(image_root)
            label_path = (labels_root / relative).with_suffix('.txt')
            if not label_path.is_file():
                raise RuntimeError(f'Missing label for {image_path}: {label_path}')
            inventory.extend([
                {
                    'path': str(image_path.relative_to(root)),
                    'bytes': int(image_path.stat().st_size),
                    'sha256': _sha256_file(image_path),
                },
                {
                    'path': str(label_path.relative_to(root)),
                    'bytes': int(label_path.stat().st_size),
                    'sha256': _sha256_file(label_path),
                },
            ])
        split_counts[split] = len(images)

    fingerprint_digest = hashlib.sha256()
    for row in sorted(inventory, key=lambda item: str(item['path'])):
        fingerprint_digest.update(
            f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode('utf-8')
        )
    manifest_path = root / 'dataset_manifest.json'
    return {
        'root': str(root),
        'data_yaml': str(data_yaml),
        'data_yaml_sha256': _sha256_file(data_yaml),
        'dataset_manifest': str(manifest_path) if manifest_path.is_file() else None,
        'dataset_manifest_sha256': _sha256_file(manifest_path) if manifest_path.is_file() else None,
        'split_counts': split_counts,
        'inventory_entries': len(inventory),
        'content_fingerprint_sha256': fingerprint_digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Fine-tune YOLO11n-seg on a local dataset.')
    parser.add_argument('--data', required=True, help='Path to data.yaml')
    parser.add_argument('--base-model', required=True, help='Base YOLO segmentation model, e.g. yolo11n-seg.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--out', required=True, help='Output run folder')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=4,
                        help='P2000-safe default; validate a smoke run before increasing.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2,
                        help='Keep low on the 15 GiB commissioning host.')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cache', action=argparse.BooleanOptionalAction, default=False,
                        help='Disabled by default to avoid exhausting host RAM/swap.')
    parser.add_argument('--close-mosaic', type=int, default=10)
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

    dataset = _validate_dataset(data_yaml)
    request = {
        'status': 'in_progress',
        'started_utc': _timestamp(),
        'base_model': str(base_model),
        'base_model_sha256': _sha256_file(base_model),
        'dataset': dataset,
        'parameters': {
            'epochs': int(args.epochs),
            'image_size': int(args.imgsz),
            'batch': int(args.batch),
            'device': str(args.device),
            'seed': int(args.seed),
            'workers': int(args.workers),
            'patience': int(args.patience),
            'amp': bool(args.amp),
            'deterministic': bool(args.deterministic),
            'cache': bool(args.cache),
            'close_mosaic': int(args.close_mosaic),
        },
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'ultralytics': str(ultralytics.__version__),
            'torch': str(torch.__version__),
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_device_count': int(torch.cuda.device_count()),
        },
    }
    request_path = out_dir / 'training_request.json'
    _write_json_atomic(request_path, request)

    train_root = out_dir / 'ultralytics_run'
    try:
        model = YOLO(str(base_model))
        results = model.train(
            data=str(data_yaml),
            epochs=int(args.epochs),
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            device=str(args.device),
            seed=int(args.seed),
            workers=max(int(args.workers), 0),
            patience=max(int(args.patience), 0),
            amp=bool(args.amp),
            deterministic=bool(args.deterministic),
            cache=bool(args.cache),
            close_mosaic=max(int(args.close_mosaic), 0),
            project=str(train_root.parent),
            name=train_root.name,
            exist_ok=False,
        )
    except BaseException as exc:
        request.update({
            'status': 'failed',
            'finished_utc': _timestamp(),
            'failure_type': type(exc).__name__,
            'failure_message': str(exc),
            'traceback': traceback.format_exc(),
        })
        _write_json_atomic(request_path, request)
        raise
    run_dir = Path(getattr(results, 'save_dir', train_root)).resolve()
    best_model = run_dir / 'weights' / 'best.pt'
    if not best_model.is_file():
        raise RuntimeError(f'Ultralytics did not produce best.pt in {run_dir}')

    model_out = out_dir / 'model.pt'
    temporary_model = out_dir / 'model.pt.tmp'
    shutil.copy2(best_model, temporary_model)
    os.replace(temporary_model, model_out)
    for filename in ('results.csv', 'results.png', 'labels.jpg', 'confusion_matrix.png'):
        src = run_dir / filename
        if src.is_file():
            shutil.copy2(src, out_dir / filename)

    manifest = {
        'base_model': str(base_model),
        'dataset_yaml': str(data_yaml),
        'dataset': dataset,
        'base_model_sha256': _sha256_file(base_model),
        'epochs': int(args.epochs),
        'image_size': int(args.imgsz),
        'batch': int(args.batch),
        'device': str(args.device),
        'seed': int(args.seed),
        'workers': int(args.workers),
        'patience': int(args.patience),
        'amp': bool(args.amp),
        'deterministic': bool(args.deterministic),
        'cache': bool(args.cache),
        'close_mosaic': int(args.close_mosaic),
        'output_model': str(model_out),
        'output_model_sha256': _sha256_file(model_out),
        'training_run_dir': str(run_dir),
        'timestamp_utc': _timestamp(),
        'confidence_calibrated': False,
        'notes': [
            'Start with 5-10 epochs for a smoke test.',
            'Use 20-30 epochs for a first usable fine-tune.',
            'Detector confidence is not calibrated.',
        ],
    }
    _write_json_atomic(out_dir / 'manifest.json', manifest)
    request.update({
        'status': 'completed',
        'finished_utc': _timestamp(),
        'output_model': str(model_out),
        'output_model_sha256': manifest['output_model_sha256'],
        'training_run_dir': str(run_dir),
    })
    _write_json_atomic(request_path, request)
    print(f'Wrote model to {model_out}')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
