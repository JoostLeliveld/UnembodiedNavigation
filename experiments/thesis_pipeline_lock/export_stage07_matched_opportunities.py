#!/usr/bin/env python3
"""Join all camera opportunities to whole-drive OOF matched-R predictions.

This is deliberately an export/validation boundary, not another model fitter.
The selected correction/R pipeline must first produce one covariance prediction
for every admitted opportunity while excluding that opportunity's complete drive
from both correction and covariance fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PREDICTION_SCHEMA = 'matched_runtime_covariance_predictions.v1'
OUTPUT_SCHEMA = 'matched_runtime_covariance_opportunities.v1'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boolean(value: str, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true'}:
        return True
    if normalized in {'0', 'false'}:
        return False
    raise ValueError(f'{name} must be encoded as 0/1 or true/false')


def opportunity_identity(drive_id, camera_id, capture_stamp_ns) -> tuple[str, str, int]:
    return str(drive_id), str(camera_id), int(capture_stamp_ns)


def load_predictions(path: Path) -> tuple[dict, dict[tuple[str, str, int], np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            'metadata_json', 'drive_ids', 'camera_ids', 'capture_stamp_ns',
            'runtime_covariance_m2',
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f'matched-R predictions are missing {sorted(missing)}')
        metadata = json.loads(str(archive['metadata_json'].item()))
        drive = np.asarray(archive['drive_ids']).astype(str)
        camera = np.asarray(archive['camera_ids']).astype(str)
        stamp = np.asarray(archive['capture_stamp_ns'], dtype=np.int64)
        covariance = np.asarray(archive['runtime_covariance_m2'], dtype=float)
    count = len(drive)
    if camera.shape != (count,) or stamp.shape != (count,) or covariance.shape != (count, 2, 2):
        raise ValueError('matched-R prediction arrays have inconsistent row counts')
    required_metadata = {
        'schema': PREDICTION_SCHEMA,
        'correction_prediction_is_out_of_fold': True,
        'runtime_covariance_is_out_of_fold': True,
        'final_audit_accessed': False,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f'matched-R metadata {key} must equal {expected!r}')
    unit = metadata.get('resampling_unit')
    exclusion_key = {
        'complete_drive': 'prediction_drive_excluded_from_fit',
        'complete_reference_position': 'prediction_reference_position_excluded_from_fit',
    }.get(unit)
    if exclusion_key is None or metadata.get(exclusion_key) is not True:
        raise ValueError(
            f'matched-R metadata {exclusion_key or "resampling_unit"} must prove '
            'that each row\'s complete drive or reference position was excluded')
    if not metadata.get('correction_model') or not metadata.get('runtime_covariance_model'):
        raise ValueError('matched-R metadata must identify correction and covariance models')
    if (not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-12)
            or np.linalg.eigvalsh(covariance).min() <= 0.0):
        raise ValueError('every admitted matched-R prediction must be finite SPD')
    rows: dict[tuple[str, str, int], np.ndarray] = {}
    for index in range(count):
        key = opportunity_identity(drive[index], camera[index], stamp[index])
        if key in rows:
            raise ValueError(f'duplicate matched-R prediction identity: {key}')
        rows[key] = covariance[index]
    return metadata, rows


def export(opportunities: Path, predictions: Path, output: Path) -> None:
    prediction_metadata, predicted = load_predictions(predictions)
    with opportunities.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    required = {
        'drive_id', 'camera_id', 'capture_stamp_ns', 'reference_x', 'reference_y',
        'sensor_gate_admitted',
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'opportunity table must contain {sorted(required)}')

    identities: set[tuple[str, str, int]] = set()
    covariances = np.full((len(rows), 2, 2), np.nan, dtype=float)
    admitted = np.zeros(len(rows), dtype=np.uint8)
    for index, row in enumerate(rows):
        key = opportunity_identity(
            row['drive_id'], row['camera_id'], row['capture_stamp_ns'])
        if key in identities:
            raise ValueError(f'duplicate camera opportunity identity: {key}')
        identities.add(key)
        is_admitted = boolean(row['sensor_gate_admitted'], name='sensor_gate_admitted')
        admitted[index] = int(is_admitted)
        if is_admitted:
            if key not in predicted:
                raise ValueError(f'admitted opportunity has no matched-R prediction: {key}')
            covariances[index] = predicted[key]
        elif key in predicted:
            raise ValueError(f'non-admitted opportunity unexpectedly has matched R: {key}')
    extra = set(predicted).difference(identities)
    if extra:
        raise ValueError(f'matched-R predictions contain {len(extra)} unknown opportunities')

    camera_order = prediction_metadata.get('camera_order')
    observed_cameras = sorted({row['camera_id'] for row in rows})
    if (not isinstance(camera_order, list) or not camera_order
            or set(camera_order) != set(observed_cameras)):
        raise ValueError('prediction camera order differs from the opportunity roster')
    metadata = {
        'schema': OUTPUT_SCHEMA,
        'correction_model': prediction_metadata['correction_model'],
        'runtime_covariance_model': prediction_metadata['runtime_covariance_model'],
        'correction_prediction_is_out_of_fold': True,
        'runtime_covariance_is_out_of_fold': True,
        'resampling_unit': prediction_metadata['resampling_unit'],
        'camera_order': camera_order,
        'final_audit_accessed': False,
        'source_hashes': {
            str(opportunities): sha256(opportunities),
            str(predictions): sha256(predictions),
            str(Path(__file__).resolve().relative_to(REPO)): sha256(Path(__file__)),
        },
    }
    exclusion_key = ({
        'complete_drive': 'prediction_drive_excluded_from_fit',
        'complete_reference_position': 'prediction_reference_position_excluded_from_fit',
    })[prediction_metadata['resampling_unit']]
    metadata[exclusion_key] = True
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        positions_xy_m=np.asarray([
            [float(row['reference_x']), float(row['reference_y'])] for row in rows
        ]),
        camera_ids=np.asarray([row['camera_id'] for row in rows]),
        admitted=admitted,
        runtime_covariance_m2=covariances,
        drive_ids=np.asarray([row['drive_id'] for row in rows]),
        capture_stamp_ns=np.asarray(
            [int(row['capture_stamp_ns']) for row in rows], dtype=np.int64),
        **({'headings_rad': np.asarray(
            [float(row['reference_heading_rad']) for row in rows], dtype=float)}
           if 'reference_heading_rad' in rows[0] else {}),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    incomplete = output.with_name(output.name + '.incomplete')
    if output.exists() or incomplete.exists():
        raise FileExistsError('refusing to overwrite Stage-07 planning input')
    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_bytes(buffer.getvalue())
    os.replace(incomplete, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opportunities', type=Path, required=True)
    parser.add_argument('--matched-r-predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    export(
        arguments.opportunities.resolve(),
        arguments.matched_r_predictions.resolve(),
        arguments.output.resolve(),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
