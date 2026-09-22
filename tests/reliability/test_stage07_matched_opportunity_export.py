from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'experiments/thesis_pipeline_lock/export_stage07_matched_opportunities.py'
SPEC = importlib.util.spec_from_file_location('stage07_matched_export', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_opportunities(path: Path):
    fields = [
        'drive_id', 'camera_id', 'capture_stamp_ns', 'reference_x', 'reference_y',
        'sensor_gate_admitted',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            dict(drive_id='fit_1', camera_id='camera_A', capture_stamp_ns=1,
                 reference_x=0, reference_y=1, sensor_gate_admitted=1),
            dict(drive_id='fit_1', camera_id='camera_B', capture_stamp_ns=1,
                 reference_x=0, reference_y=1, sensor_gate_admitted=0),
        ])


def write_predictions(path: Path, **metadata_overrides):
    metadata = {
        'schema': 'matched_runtime_covariance_predictions.v1',
        'correction_model': 'box_mlp',
        'runtime_covariance_model': 'spatial_residual_covariance',
        'correction_prediction_is_out_of_fold': True,
        'runtime_covariance_is_out_of_fold': True,
        'resampling_unit': 'complete_drive',
        'prediction_drive_excluded_from_fit': True,
        'camera_order': ['camera_A', 'camera_B'],
        'final_audit_accessed': False,
    }
    metadata.update(metadata_overrides)
    np.savez_compressed(
        path,
        drive_ids=np.asarray(['fit_1']),
        camera_ids=np.asarray(['camera_A']),
        capture_stamp_ns=np.asarray([1], dtype=np.int64),
        runtime_covariance_m2=np.asarray([[[0.04, 0.01], [0.01, 0.09]]]),
        metadata_json=json.dumps(metadata),
    )


def test_export_preserves_misses_and_joins_only_admitted_oof_r(tmp_path):
    opportunities = tmp_path / 'camera_opportunities.csv'
    predictions = tmp_path / 'matched_r.npz'
    output = tmp_path / 'planning_input.npz'
    write_opportunities(opportunities)
    write_predictions(predictions)
    MODULE.export(opportunities, predictions, output)
    with np.load(output, allow_pickle=False) as archive:
        assert archive['admitted'].tolist() == [1, 0]
        assert np.allclose(
            archive['runtime_covariance_m2'][0], [[0.04, 0.01], [0.01, 0.09]])
        assert np.isnan(archive['runtime_covariance_m2'][1]).all()
        stored = json.loads(str(archive['metadata_json'].item()))
    assert stored['resampling_unit'] == 'complete_drive'
    assert stored['prediction_drive_excluded_from_fit'] is True


def test_export_rejects_covariance_fit_that_saw_prediction_drive(tmp_path):
    predictions = tmp_path / 'matched_r.npz'
    write_predictions(predictions, prediction_drive_excluded_from_fit=False)
    with pytest.raises(ValueError, match='prediction_drive_excluded_from_fit'):
        MODULE.load_predictions(predictions)
