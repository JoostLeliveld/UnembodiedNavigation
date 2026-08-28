#!/usr/bin/env python3
"""Derive immutable YOLO detection labels from saved semantic masks.

The simulator capture stores masks on an integer pixel grid.  An occupied pixel at
``(x, y)`` spans the half-open image cell ``[x, x+1) x [y, y+1)``.  Therefore the
bounding-box maximum is ``max_index + 1``.  Segmentation polygons made from OpenCV
contours end at ``max_index``; allowing a detector to derive its box target from those
polygons creates the one-pixel bottom-edge convention mismatch this dataset removes.

Images are linked rather than copied.  Labels and provenance are newly written, and an
existing output directory is always refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png'}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def half_open_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return ``(xmin, ymin, xmax_exclusive, ymax_exclusive)`` for a binary mask."""

    if mask.ndim != 2:
        raise ValueError(f'Expected a 2-D mask, received shape {mask.shape}')
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def yolo_detection_row(
    bbox: tuple[float, float, float, float] | None,
    *,
    width: int,
    height: int,
) -> str:
    """Serialize one class-0 half-open box in normalized YOLO detection format."""

    if bbox is None:
        return ''
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid image size {width}x{height}')
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if not (0.0 <= x0 < x1 <= float(width) and 0.0 <= y0 < y1 <= float(height)):
        raise ValueError(f'Box {bbox} is outside {width}x{height}')
    xc = 0.5 * (x0 + x1) / float(width)
    yc = 0.5 * (y0 + y1) / float(height)
    bw = (x1 - x0) / float(width)
    bh = (y1 - y0) / float(height)
    return f'0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n'


def _resolve_dataset_root(data_yaml: Path, payload: dict) -> Path:
    root = Path(str(payload.get('path', data_yaml.parent)))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.expanduser().resolve()


def _resolve_split(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve()


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(relative_target)


def build_dataset(source_yaml: Path, mask_root: Path, output: Path) -> dict:
    source_yaml = source_yaml.expanduser().resolve()
    mask_root = mask_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not source_yaml.is_file():
        raise FileNotFoundError(f'Source data.yaml does not exist: {source_yaml}')
    if not mask_root.is_dir():
        raise FileNotFoundError(f'Mask root does not exist: {mask_root}')

    source_data = yaml.safe_load(source_yaml.read_text(encoding='utf-8')) or {}
    source_root = _resolve_dataset_root(source_yaml, source_data)
    staged = output.with_name(output.name + '.incomplete')
    if staged.exists():
        raise FileExistsError(f'Staging directory already exists: {staged}')
    staged.mkdir(parents=True, exist_ok=False)

    split_summary: dict[str, dict[str, int]] = {}
    label_digest = hashlib.sha256()
    try:
        for split in ('train', 'val'):
            if split not in source_data:
                raise RuntimeError(f'Source dataset is missing split {split!r}')
            image_root = _resolve_split(source_root, source_data[split])
            images = sorted(
                path for path in image_root.rglob('*')
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not images:
                raise RuntimeError(f'No images found for {split}: {image_root}')
            positives = 0
            negatives = 0
            for image_path in images:
                relative = image_path.relative_to(image_root)
                mask_path = (mask_root / split / relative).with_suffix('.png')
                if not mask_path.is_file():
                    raise FileNotFoundError(f'Missing mask for {image_path}: {mask_path}')
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.ndim != 2 or mask.size == 0:
                    raise RuntimeError(f'Cannot decode mask: {mask_path}')
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != mask.shape:
                    raise RuntimeError(
                        f'Image/mask shape mismatch for {image_path}: '
                        f'{None if image is None else image.shape[:2]} vs {mask.shape}'
                    )
                bbox = half_open_box(mask)
                label = yolo_detection_row(
                    bbox,
                    width=int(mask.shape[1]),
                    height=int(mask.shape[0]),
                )
                positives += int(bbox is not None)
                negatives += int(bbox is None)
                destination_image = staged / 'images' / split / relative
                destination_label = (staged / 'labels' / split / relative).with_suffix('.txt')
                _link(image_path, destination_image)
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                destination_label.write_text(label, encoding='utf-8')
                label_digest.update(relative.as_posix().encode('utf-8') + b'\0')
                label_digest.update(label.encode('utf-8'))
            split_summary[split] = {
                'images': len(images),
                'positives': positives,
                'negatives': negatives,
            }

        data_yaml = {
            'path': str(output),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'robot'},
            'task': 'detect',
        }
        (staged / 'data.yaml').write_text(
            yaml.safe_dump(data_yaml, sort_keys=False), encoding='utf-8'
        )
        manifest = {
            'status': 'complete',
            'created_utc': _timestamp(),
            'task': 'detect',
            'box_convention': {
                'name': 'semantic_mask_half_open_extent',
                'definition': '[xmin, ymin, xmax_occupied+1, ymax_occupied+1]',
                'purpose': 'remove contour-extrema one-pixel box-edge mismatch',
            },
            'source': {
                'data_yaml': str(source_yaml),
                'data_yaml_sha256': _sha256_file(source_yaml),
                'dataset_root': str(source_root),
                'mask_root': str(mask_root),
            },
            'splits': split_summary,
            'label_inventory_sha256': label_digest.hexdigest(),
        }
        manifest_path = staged / 'dataset_manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
        completion = {
            'status': 'complete',
            'dataset_manifest_sha256': _sha256_file(manifest_path),
        }
        (staged / '.complete').write_text(
            json.dumps(completion, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
        os.replace(staged, output)
        return manifest
    except BaseException:
        # Preserve a failed derivation for diagnosis; never make it look complete.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', required=True, type=Path)
    parser.add_argument('--mask-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    manifest = build_dataset(args.source_data, args.mask_root, args.out)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
