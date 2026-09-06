#!/usr/bin/env python3
"""Add two learned bias updates to the frozen box interpretations.

Both new interpretations correct the SAME raw back-projected point that `raw` and `fixed`
start from.  They are therefore a ladder on one frozen YOLO box:

    raw      no correction
    fixed    one constant radial shift, the same everywhere
    learned  a per-camera linear model in detector-observable quantities
    nn       one pooled neural network in the same quantities
    hull     analytic silhouette residual around the offline reference pose (not operational)

Neither learned model is allowed to see the commanded pose, the robot heading, or the true
range.  Every input is something the runtime actually holds: the pixel box, the camera that
produced it, and the raw back-projection derived from those two.  Truth enters only as the
regression target during training, and only on training tiles.

The warehouse is split into square tiles and alternate tiles are held out.  All reported
figures are evaluated on held-out tiles, so a learned correction is being asked to work at
places it never saw.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CAMERAS = tuple(f'camera_{letter}' for letter in 'ABCDE')
SOURCE_METHODS = ('raw', 'fixed', 'hull')
NEW_METHODS = ('learned', 'nn')
QUANTITIES = ('valid', 'x', 'y', 'dx', 'dy', 'error_m', 'along_m', 'across_m')
FEATURE_NAMES = (
    'range_m', 'inv_range', 'box_w_frac', 'box_h_frac', 'box_aspect',
    'u_frac', 'v_frac', 'bearing_cos', 'bearing_sin', 'confidence',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def camera_geometry(capture_manifest: dict) -> dict[str, dict]:
    geometry = {}
    for item in capture_manifest['cameras']:
        pose = [float(value) for value in item['pose_xyz_rpy']]
        geometry[item['camera_id']] = {
            'xy': np.asarray(pose[:2], dtype=float),
            'yaw': float(pose[5]),
            'width': float(item['image_width']),
            'height': float(item['image_height']),
        }
    return geometry


def ray_frame(point: np.ndarray, camera_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector away from the camera through `point`, and its left normal."""
    ray = point - camera_xy
    norm = float(np.linalg.norm(ray))
    unit = ray / norm if norm > 1e-12 else np.array([1.0, 0.0])
    return unit, np.array([-unit[1], unit[0]])


def features(row: dict[str, str], geometry: dict) -> np.ndarray:
    raw = np.array([float(row['raw_x']), float(row['raw_y'])])
    camera_xy = geometry['xy']
    ray = raw - camera_xy
    distance = float(np.linalg.norm(ray))
    bearing = math.atan2(ray[1], ray[0]) - geometry['yaw']
    box_w = (float(row['x1']) - float(row['x0'])) / geometry['width']
    box_h = (float(row['y1']) - float(row['y0'])) / geometry['height']
    return np.array([
        distance,
        1.0 / max(distance, 1e-3),
        box_w,
        box_h,
        box_w * geometry['width'] / max(box_h * geometry['height'], 1.0),
        float(row['u_bbox_bottom']) / geometry['width'],
        float(row['v_bbox_bottom']) / geometry['height'],
        math.cos(bearing),
        math.sin(bearing),
        float(row['confidence']),
    ], dtype=float)


def target(row: dict[str, str], geometry: dict) -> np.ndarray:
    """Correction the raw point needs, in its own camera-ray frame."""
    raw = np.array([float(row['raw_x']), float(row['raw_y'])])
    truth = np.array([float(row['robot_x']), float(row['robot_y'])])
    unit, left = ray_frame(raw, geometry['xy'])
    residual = truth - raw
    return np.array([float(residual @ unit), float(residual @ left)])


def apply_correction(row: dict[str, str], geometry: dict, correction: np.ndarray) -> np.ndarray:
    raw = np.array([float(row['raw_x']), float(row['raw_y'])])
    unit, left = ray_frame(raw, geometry['xy'])
    return raw + float(correction[0]) * unit + float(correction[1]) * left


def method_values(estimate, *, truth: np.ndarray, camera_xy: np.ndarray) -> dict[str, float | int]:
    """Error bookkeeping identical to derive_interpretations.py, including its error frame."""
    if estimate is None:
        return dict.fromkeys(QUANTITIES, math.nan) | {'valid': 0}
    estimate = np.asarray(estimate, dtype=float)
    if estimate.shape != (2,) or not np.all(np.isfinite(estimate)):
        return method_values(None, truth=truth, camera_xy=camera_xy)
    error = estimate - truth
    unit, left = ray_frame(truth, camera_xy)
    return {
        'valid': 1,
        'x': float(estimate[0]), 'y': float(estimate[1]),
        'dx': float(error[0]), 'dy': float(error[1]),
        'error_m': float(np.linalg.norm(error)),
        'along_m': float(error @ unit), 'across_m': float(error @ left),
    }


def tile_split(rows: list[dict[str, str]], tile_m: float) -> dict[str, str]:
    """Checkerboard of square warehouse tiles; every heading of a position shares a side."""
    split: dict[str, str] = {}
    for row in rows:
        position = row['position_id']
        if position in split:
            continue
        tile_x = math.floor(float(row['robot_x']) / tile_m)
        tile_y = math.floor(float(row['robot_y']) / tile_m)
        split[position] = 'test' if (tile_x + tile_y) % 2 == 0 else 'train'
    return split


def summarize(errors: np.ndarray) -> dict[str, float]:
    if not errors.size:
        return {'n': 0}
    return {
        'n': int(errors.size),
        'median_m': float(np.median(errors)),
        'p90_m': float(np.quantile(errors, 0.90)),
        'rmse_m': float(np.sqrt(np.mean(errors ** 2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--tile-m', type=float, default=2.0,
                        help='Side of the square holdout tiles, in metres.')
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--hidden', type=int, nargs='+', default=(64, 64))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    table = capture / 'observation_interpretations.csv'
    interp_manifest_path = capture / 'observation_interpretations_manifest.json'
    capture_manifest_path = capture / 'capture_manifest.json'
    for path in (table, interp_manifest_path, capture_manifest_path):
        if not path.is_file():
            raise RuntimeError(f'Missing required input: {path}')
    interp_manifest = json.loads(interp_manifest_path.read_text(encoding='utf-8'))
    if interp_manifest.get('status') != 'complete':
        raise RuntimeError('observation_interpretations manifest is not complete')
    if sha256(table) != interp_manifest['observation_interpretations_sha256']:
        raise RuntimeError('observation_interpretations.csv changed after it was derived')

    geometry = camera_geometry(json.loads(capture_manifest_path.read_text(encoding='utf-8')))
    rows = list(csv.DictReader(table.open(encoding='utf-8')))
    split_of = tile_split(rows, args.tile_m)
    for row in rows:
        row['split'] = split_of[row['position_id']]

    usable = [row for row in rows if row['raw_valid'] == '1']
    design = {row['pose_id'] + ':' + row['camera_id']: None for row in usable}
    if len(design) != len(usable):
        raise RuntimeError('Duplicate (pose_id, camera_id) among usable rows')

    # --- fit -------------------------------------------------------------------------
    train = [row for row in usable if row['split'] == 'train']
    if not train:
        raise RuntimeError('No training rows')

    # A per-camera linear model in detector-observable quantities.
    linear: dict[str, object] = {}
    for camera_id in CAMERAS:
        subset = [row for row in train if row['camera_id'] == camera_id]
        if len(subset) < 20:
            linear[camera_id] = None
            continue
        x = np.stack([features(row, geometry[camera_id]) for row in subset])
        y = np.stack([target(row, geometry[camera_id]) for row in subset])
        model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        model.fit(x, y)
        linear[camera_id] = model

    # One pooled network over the same quantities plus a camera indicator.
    def with_camera(row: dict[str, str]) -> np.ndarray:
        onehot = np.zeros(len(CAMERAS))
        onehot[CAMERAS.index(row['camera_id'])] = 1.0
        return np.concatenate([features(row, geometry[row['camera_id']]), onehot])

    x_nn = np.stack([with_camera(row) for row in train])
    y_nn = np.stack([target(row, geometry[row['camera_id']]) for row in train])
    network = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=tuple(args.hidden), activation='relu', solver='adam',
                     alpha=1e-4, learning_rate_init=1e-3, max_iter=4000, early_stopping=True,
                     n_iter_no_change=40, validation_fraction=0.15, random_state=args.seed),
    )
    network.fit(x_nn, y_nn)

    # The constant shift, refit on training tiles only, for an honest ladder rung.
    train_offset = float(np.median(
        [target(row, geometry[row['camera_id']])[0] for row in train]))

    # --- score every row -------------------------------------------------------------
    base_fields = [field for field in rows[0] if field != 'split']
    fields = base_fields + ['split'] + [
        f'{method}_{quantity}' for method in NEW_METHODS for quantity in QUANTITIES
    ]
    output = capture / 'bias_update_interpretations.csv'
    counts = {'rows': len(rows), 'usable_raw': len(usable),
              'train': sum(row['split'] == 'train' for row in usable),
              'test': sum(row['split'] == 'test' for row in usable)}
    scored: list[dict[str, str]] = []
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            if row['raw_valid'] != '1':
                for method in NEW_METHODS:
                    values = method_values(None, truth=np.zeros(2), camera_xy=np.zeros(2))
                    record.update({f'{method}_{key}': value for key, value in values.items()})
                writer.writerow(record)
                scored.append(record)
                continue
            camera_id = row['camera_id']
            geo = geometry[camera_id]
            truth = np.array([float(row['robot_x']), float(row['robot_y'])])
            model = linear[camera_id]
            estimates = {
                'learned': None if model is None else apply_correction(
                    row, geo, model.predict(features(row, geo)[None, :])[0]),
                'nn': apply_correction(row, geo, network.predict(with_camera(row)[None, :])[0]),
            }
            for method, estimate in estimates.items():
                values = method_values(estimate, truth=truth, camera_xy=geo['xy'])
                record.update({f'{method}_{key}': value for key, value in values.items()})
            writer.writerow(record)
            scored.append(record)

    # --- report ----------------------------------------------------------------------
    ladder = SOURCE_METHODS[:2] + NEW_METHODS + SOURCE_METHODS[2:]
    scores = {}
    for portion in ('train', 'test'):
        subset = [row for row in scored if row['split'] == portion and row['raw_valid'] == '1']
        scores[portion] = {
            method: summarize(np.asarray([float(row[f'{method}_error_m']) for row in subset
                                          if str(row[f'{method}_valid']) == '1']))
            for method in ladder
        }

    manifest = {
        'status': 'complete',
        'schema': 'bbox_bias_update_interpretations.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'observation_interpretations_sha256': sha256(table),
        'capture_manifest_sha256': sha256(capture_manifest_path),
        'holdout': {
            'scheme': 'checkerboard of square warehouse tiles, assigned per floor position',
            'tile_m': float(args.tile_m),
            'note': 'all eight headings of a position fall on the same side of the split',
            **counts,
        },
        'features': {
            'names': list(FEATURE_NAMES),
            'source': 'YOLO box, its camera, and the raw back-projection of the box bottom-centre',
            'excluded': ['commanded pose', 'robot heading', 'true range', 'segmentation labels'],
        },
        'models': {
            'learned': f'per-camera Ridge(alpha={args.ridge_alpha}) on standardized features, '
                       '2 outputs (along-ray, across-ray correction)',
            'nn': f'pooled MLPRegressor(hidden={tuple(args.hidden)}, relu, adam, early stopping) '
                  'on the same features plus a five-way camera indicator',
            'seed': int(args.seed),
        },
        'train_refit_fixed_offset_m': train_offset,
        'scores': scores,
        'bias_update_interpretations_sha256': sha256(output),
    }
    (capture / 'bias_update_interpretations_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    print(json.dumps({'output': str(output), 'holdout': manifest['holdout'],
                      'test_scores': scores['test']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
