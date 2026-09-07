#!/usr/bin/env python3
"""Derive three position interpretations from one frozen YOLO bounding box.

Every output row retains the capture key and image hash.  Commanded pose is used only as
an offline reference for error calculation and for evaluating the forward hull residual;
it is never an input to YOLO.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/unav_common', 'experiments/measurement_commissioning'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from experiments.core.world_profiles import compute_look_at_from_pose  # noqa: E402
from observation import h as hull_h, jacobian as hull_jacobian  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402


KEY = ('pose_id', 'repetition_id', 'camera_id')
METHODS = ('raw', 'fixed', 'hull')
BASE_FIELDS = (
    'pose_id', 'position_id', 'heading_id', 'repetition_id', 'source_batch_id', 'camera_id',
    'dataset_split', 'random_draw_index',
    'image', 'image_sha1', 'robot_x', 'robot_y', 'robot_yaw', 'camera_range_m',
    'nominal_in_frame', 'detected', 'detector_clipped', 'confidence', 'n_candidates',
    'x0', 'y0', 'x1', 'y1', 'u_bbox_bottom', 'v_bbox_bottom',
)
METHOD_FIELDS = tuple(
    f'{method}_{quantity}'
    for method in METHODS
    for quantity in ('valid', 'x', 'y', 'dx', 'dy', 'error_m', 'along_m', 'across_m')
)
FIELDS = BASE_FIELDS + METHOD_FIELDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in KEY)


def camera_models(capture_manifest: dict) -> dict[str, ObliqueCameraModel]:
    profiles_path = Path(capture_manifest['world_profiles_path'])
    if sha256(profiles_path) != capture_manifest['world_profiles_sha256']:
        raise RuntimeError('The world-profile file changed after capture; refusing reconstruction')
    profiles = yaml.safe_load(profiles_path.read_text(encoding='utf-8'))
    intrinsics = profiles['camera_intrinsics']
    models = {}
    for item in capture_manifest['cameras']:
        pose = [float(value) for value in item['pose_xyz_rpy']]
        look_at = compute_look_at_from_pose(pose[:3], *pose[3:])
        models[item['camera_id']] = ObliqueCameraModel(
            cam_pos=pose[:3], look_at=look_at,
            img_width=int(item['image_width']), img_height=int(item['image_height']),
            fov_h_rad=float(intrinsics['fov_h_rad']),
        )
    return models


def method_values(
    estimate: tuple[float, float] | np.ndarray | None,
    *,
    truth: np.ndarray,
    camera_xy: np.ndarray,
) -> dict[str, float | int]:
    if estimate is None:
        return {
            'valid': 0, 'x': math.nan, 'y': math.nan, 'dx': math.nan, 'dy': math.nan,
            'error_m': math.nan, 'along_m': math.nan, 'across_m': math.nan,
        }
    estimate = np.asarray(estimate, dtype=float)
    if estimate.shape != (2,) or not np.all(np.isfinite(estimate)):
        return method_values(None, truth=truth, camera_xy=camera_xy)
    error = estimate - truth
    ray = truth - camera_xy
    norm = float(np.linalg.norm(ray))
    unit = ray / norm if norm > 1e-12 else np.array([1.0, 0.0])
    left = np.array([-unit[1], unit[0]])
    return {
        'valid': 1,
        'x': float(estimate[0]), 'y': float(estimate[1]),
        'dx': float(error[0]), 'dy': float(error[1]),
        'error_m': float(np.linalg.norm(error)),
        'along_m': float(error @ unit), 'across_m': float(error @ left),
    }


def fixed_offset(raw: np.ndarray, camera_xy: np.ndarray, offset_m: float) -> np.ndarray:
    direction = raw - camera_xy
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return raw.copy()
    return raw + float(offset_m) * direction / norm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--fixed-offset-m', type=float, default=0.309)
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    capture_index = capture / 'capture_index.csv'
    bbox_path = capture / 'bbox_observations.csv'
    capture_manifest_path = capture / 'capture_manifest.json'
    detector_manifest_path = capture / 'bbox_detector_manifest.json'
    required = (capture_index, bbox_path, capture_manifest_path, detector_manifest_path)
    if any(not path.is_file() for path in required):
        raise RuntimeError('Complete capture and detector outputs are required')
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding='utf-8'))
    detector_manifest = json.loads(detector_manifest_path.read_text(encoding='utf-8'))
    if (
        not str(capture_manifest.get('status', '')).startswith('complete')
        or detector_manifest.get('status') != 'complete'
    ):
        raise RuntimeError('Capture and detector manifests must both be complete')

    capture_rows = list(csv.DictReader(capture_index.open(encoding='utf-8')))
    bbox_rows = list(csv.DictReader(bbox_path.open(encoding='utf-8')))
    bbox_by_key = {row_key(row): row for row in bbox_rows}
    if len(bbox_by_key) != len(bbox_rows):
        raise RuntimeError('Duplicate observation keys in bbox_observations.csv')
    usable = [row for row in capture_rows if row['capture_status'] == 'ok']
    if {row_key(row) for row in usable} != set(bbox_by_key):
        raise RuntimeError('Capture and detector keys do not match exactly')

    cameras = camera_models(capture_manifest)
    output = capture / 'observation_interpretations.csv'
    counts = {'attempts': 0, 'detections': 0, 'raw_valid': 0, 'fixed_valid': 0, 'hull_valid': 0}
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for source in usable:
            box = bbox_by_key[row_key(source)]
            if source['image_sha1'] != box['image_sha1']:
                raise RuntimeError(f'Image hash mismatch at {row_key(source)}')
            row = {field: '' for field in FIELDS}
            for field in BASE_FIELDS:
                if field in source:
                    row[field] = source[field]
                if field in box:
                    row[field] = box[field]
            counts['attempts'] += 1
            if int(box['detected']) != 1:
                for method in METHODS:
                    values = method_values(None, truth=np.zeros(2), camera_xy=np.zeros(2))
                    row.update({f'{method}_{key}': value for key, value in values.items()})
                writer.writerow(row)
                continue

            counts['detections'] += 1
            cam = cameras[source['camera_id']]
            truth = np.array([float(source['robot_x']), float(source['robot_y'])])
            camera_xy = np.asarray(cam.cam_pos[:2], dtype=float)
            measured = np.array([float(box['u_bbox_bottom']), float(box['v_bbox_bottom'])])
            raw_world = cam.pixel_to_world(*measured)
            raw = None if raw_world is None else np.asarray(raw_world[:2], dtype=float)
            fixed = None if raw is None else fixed_offset(raw, camera_xy, args.fixed_offset_m)
            hull = None
            try:
                predicted = hull_h(cam, truth[0], truth[1], float(source['robot_yaw']))
                if predicted is not None:
                    jac = hull_jacobian(cam, truth[0], truth[1], float(source['robot_yaw']))
                    hull = truth + np.linalg.solve(jac, measured - np.asarray(predicted))
            except (ValueError, TypeError, np.linalg.LinAlgError):
                hull = None
            for method, estimate in (('raw', raw), ('fixed', fixed), ('hull', hull)):
                values = method_values(estimate, truth=truth, camera_xy=camera_xy)
                row.update({f'{method}_{key}': value for key, value in values.items()})
                counts[f'{method}_valid'] += int(values['valid'])
            writer.writerow(row)

    manifest = {
        'status': 'complete',
        'schema': 'bbox_observation_interpretations.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'capture_index_sha256': sha256(capture_index),
        'bbox_observations_sha256': sha256(bbox_path),
        'interpretations': {
            'raw': 'floor back-projection of YOLO bounding-box bottom-centre',
            'fixed': 'raw point shifted away from camera by one fixed distance',
            'fixed_offset_m': float(args.fixed_offset_m),
            'hull': (
                'local world residual J^-1(z-h(x_ref)) from the analytic robot hull; '
                'commanded pose is the offline reference'
            ),
        },
        'error_frame': {
            'along_m': 'positive away from camera along camera-to-robot ray',
            'across_m': 'positive left of that ray',
        },
        'counts': counts,
        'observation_interpretations_sha256': sha256(output),
    }
    (capture / 'observation_interpretations_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    print(json.dumps({'output': str(output), **counts}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
