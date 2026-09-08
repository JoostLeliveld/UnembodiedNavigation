#!/usr/bin/env python3
"""Score the frozen detector separately on each camera's share of its own val split.

No retraining and no new data: the existing validation images are partitioned by camera
prefix so the pooled mAP can be resolved per viewpoint. A camera-network claim needs this,
because a detector that is weaker at one viewpoint would otherwise be indistinguishable
from a camera that is geometrically worse placed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
WEIGHTS = REPO / 'logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt'
DATASET = REPO / 'logs/perception_datasets/warehouse_v2_yolo_halfopen_detect_20260825'
OUT = REPO / 'logs/studies/detector_per_camera_validation_20260907'
EXPECTED_WEIGHTS_SHA = 'efff1949c1b8cdeeb11438b36de80f6cf8daeef5f3a4682cfce8ae7dfe314f34'


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_per_camera_splits(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Symlink each camera's validation images and labels into its own dataset root."""
    images = sorted((DATASET / 'images/val').glob('*.png'))
    if not images:
        raise SystemExit(f'no validation images under {DATASET}')
    grouped: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for image in images:
        grouped[image.name.split('__')[0]].append(image)

    yamls = {}
    for camera, paths in sorted(grouped.items()):
        image_dir = root / camera / 'images/val'
        label_dir = root / camera / 'labels/val'
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image in paths:
            (image_dir / image.name).symlink_to(image.resolve())
            label = DATASET / 'labels/val' / f'{image.stem}.txt'
            if label.exists():
                (label_dir / label.name).symlink_to(label.resolve())
        yaml_path = root / camera / 'data.yaml'
        # train points at val deliberately: this script only ever evaluates.
        yaml_path.write_text(
            f'path: {root / camera}\ntrain: images/val\nval: images/val\n'
            'names:\n  0: robot\ntask: detect\n'
        )
        yamls[camera] = yaml_path
    return yamls


def main() -> None:
    actual = sha256(WEIGHTS)
    if actual != EXPECTED_WEIGHTS_SHA:
        raise SystemExit(
            f'detector weights changed: expected {EXPECTED_WEIGHTS_SHA}, found {actual}. '
            'Re-commission before reusing these numbers.'
        )

    from ultralytics import YOLO

    root = pathlib.Path(tempfile.mkdtemp(prefix='percam_val_'))
    try:
        yamls = build_per_camera_splits(root)
        results = {}
        for camera, yaml_path in yamls.items():
            box = YOLO(str(WEIGHTS)).val(
                data=str(yaml_path), split='val', device=0,
                verbose=False, plots=False, project=str(root / 'runs'), name=camera,
            ).box
            results[camera] = {
                'val_images': len(list((root / camera / 'images/val').iterdir())),
                'precision': float(box.mp),
                'recall': float(box.mr),
                'mAP50': float(box.map50),
                'mAP50_95': float(box.map),
            }
            print(camera, json.dumps(results[camera]))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        'status': 'per_camera_validation_of_frozen_detector',
        'weights': str(WEIGHTS.relative_to(REPO)),
        'weights_sha256': actual,
        'dataset': str(DATASET.relative_to(REPO)),
        'note': 'in-distribution scores on the detector own validation split; not a generalization test',
        'per_camera': results,
    }
    (OUT / 'per_camera_metrics.json').write_text(json.dumps(payload, indent=1) + '\n')
    scores = [v['mAP50_95'] for v in results.values()]
    print(f'\nmAP50-95 range {min(scores):.4f}-{max(scores):.4f} (spread {max(scores) - min(scores):.4f})')


if __name__ == '__main__':
    main()
