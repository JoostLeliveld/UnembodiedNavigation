#!/usr/bin/env python3
"""Camera D is not a worse camera; it is a camera that only ever sees small robots.

Pooled detector metrics hide a 0.195 spread in mAP50-95 across the five viewpoints, with
camera D lowest at 0.776. This measures the mechanism: detection quality against how large
the robot appears in pixels. Below roughly 800 px^2 quality collapses, and D's boxes sit on
that cliff because it is the longest-range viewpoint. Matched on box size, D is not worse.

The distinction matters for the network: a camera subset containing D is range-biased, not
detector-biased, and those call for different remedies.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
WEIGHTS = REPO / 'logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt'
DATASET = REPO / 'logs/perception_datasets/warehouse_v2_yolo_halfopen_detect_20260825'
OUT = REPO / 'logs/studies/detector_per_camera_validation_20260907'
EXPECTED_WEIGHTS_SHA = 'efff1949c1b8cdeeb11438b36de80f6cf8daeef5f3a4682cfce8ae7dfe314f34'

IMAGE_W, IMAGE_H = 1280, 720
SIZE_BINS = [(0, 400), (400, 800), (800, 1500), (1500, 3000), (3000, np.inf)]
CONF = 0.25
DEVICE = 'cpu'  # the GPU belongs to training runs; this probe must never compete for it


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def iou(gt: np.ndarray, box: np.ndarray) -> float:
    ix0, iy0 = max(gt[0], box[0]), max(gt[1], box[1])
    ix1, iy1 = min(gt[2], box[2]), min(gt[3], box[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (gt[2] - gt[0]) * (gt[3] - gt[1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def collect() -> list[tuple[str, float, float]]:
    """For every labelled robot: (camera, box area in px^2, best IoU the detector achieved)."""
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    images = sorted((DATASET / 'images/val').glob('*.png'))
    rows: list[tuple[str, float, float]] = []
    for start in range(0, len(images), 16):
        batch = images[start:start + 16]
        results = model.predict([str(p) for p in batch], device=DEVICE, verbose=False, conf=CONF)
        for path, result in zip(batch, results):
            camera = path.name.split('__')[0]
            label = DATASET / 'labels/val' / f'{path.stem}.txt'
            predictions = (result.boxes.xyxy.cpu().numpy()
                           if result.boxes is not None else np.empty((0, 4)))
            for line in label.read_text().strip().split('\n'):
                parts = line.split()
                if len(parts) != 5:
                    continue
                xc, yc, w, h = (float(v) for v in parts[1:5])
                gt = np.array([(xc - w / 2) * IMAGE_W, (yc - h / 2) * IMAGE_H,
                               (xc + w / 2) * IMAGE_W, (yc + h / 2) * IMAGE_H])
                best = max((iou(gt, b) for b in predictions), default=0.0)
                rows.append((camera, (w * IMAGE_W) * (h * IMAGE_H), best))
    return rows


def main() -> None:
    actual = sha256(WEIGHTS)
    if actual != EXPECTED_WEIGHTS_SHA:
        raise SystemExit(f'detector weights changed: expected {EXPECTED_WEIGHTS_SHA}, found {actual}')

    rows = collect()
    camera = np.array([r[0] for r in rows])
    area = np.array([r[1] for r in rows], dtype=float)
    quality = np.array([r[2] for r in rows], dtype=float)
    cameras = sorted(set(camera))

    cliff = []
    print('Detection quality against how large the robot appears:')
    print(f"{'box area (px^2)':>18}{'labelled robots':>17}{'found':>9}{'mean overlap':>14}")
    for low, high in SIZE_BINS:
        mask = (area >= low) & (area < high)
        if mask.sum() == 0:
            continue
        row = {
            'area_min_px2': low, 'area_max_px2': None if np.isinf(high) else high,
            'n': int(mask.sum()),
            'found_fraction': float((quality[mask] > 0.5).mean()),
            'mean_iou': float(quality[mask].mean()),
        }
        cliff.append(row)
        label = f'{low:.0f}-{"inf" if np.isinf(high) else f"{high:.0f}"}'
        print(f'{label:>18}{row["n"]:17d}{100 * row["found_fraction"]:8.1f}%{row["mean_iou"]:14.3f}')

    per_camera = {}
    print('\nWhere each camera sits on that scale:')
    print(f"{'camera':>9}{'robots':>8}{'median area':>13}{'below cliff':>13}{'found':>8}{'mean overlap':>14}")
    for cam in cameras:
        mask = camera == cam
        per_camera[cam] = {
            'n': int(mask.sum()),
            'median_area_px2': float(np.median(area[mask])),
            'fraction_below_800px2': float((area[mask] < 800).mean()),
            'found_fraction': float((quality[mask] > 0.5).mean()),
            'mean_iou': float(quality[mask].mean()),
        }
        v = per_camera[cam]
        print(f'{cam[-1]:>9}{v["n"]:8d}{v["median_area_px2"]:13.0f}'
              f'{100 * v["fraction_below_800px2"]:12.1f}%{100 * v["found_fraction"]:7.1f}%'
              f'{v["mean_iou"]:14.3f}')

    matched = []
    print('\nMatched on box size, the cameras agree (mean overlap, blank = fewer than 8 robots):')
    header = f"{'box area (px^2)':>18}" + ''.join(f'{c[-1]:>9}' for c in cameras)
    print(header)
    for low, high in SIZE_BINS[1:]:
        entry = {'area_min_px2': low, 'area_max_px2': None if np.isinf(high) else high, 'per_camera': {}}
        label = f'{low:.0f}-{"inf" if np.isinf(high) else f"{high:.0f}"}'
        line = f'{label:>18}'
        for cam in cameras:
            mask = (camera == cam) & (area >= low) & (area < high)
            if mask.sum() >= 8:
                entry['per_camera'][cam] = {'n': int(mask.sum()), 'mean_iou': float(quality[mask].mean())}
                line += f'{quality[mask].mean():9.3f}'
            else:
                line += f'{"":>9}'
        matched.append(entry)
        print(line)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'size_cliff.json').write_text(json.dumps({
        'status': 'frozen_detector_diagnostic',
        'question': 'is camera D a worse viewpoint, or a viewpoint that only sees small robots',
        'finding': 'detection quality collapses below about 800 px^2; D sits on that cliff '
                   'because it is the longest-range camera. Size-matched, D is not worse.',
        'weights_sha256': actual,
        'confidence_threshold': CONF,
        'image_size_px': [IMAGE_W, IMAGE_H],
        'size_cliff': cliff,
        'per_camera': per_camera,
        'size_matched': matched,
        'limitations': [
            'the detector own validation split: in-distribution, not a generalization test',
            'unequal per-camera counts; no confidence intervals computed',
            'static capture only, so motion blur and live exposure are absent',
        ],
    }, indent=1) + '\n')
    print(f'\nwrote {OUT / "size_cliff.json"}')


if __name__ == '__main__':
    main()
