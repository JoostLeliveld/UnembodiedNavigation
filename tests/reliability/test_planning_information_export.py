import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from planning.core.camera_network import CameraNetworkModel


REPO = Path(__file__).resolve().parents[2]


def test_stage08_export_round_trip(tmp_path):
    source = tmp_path / 'opportunities.npz'
    output = tmp_path / 'planning_information.npz'
    positions = np.asarray([
        [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    ])
    cameras = np.asarray(['camera_A'] * 4 + ['camera_B'] * 4)
    admitted = np.asarray([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8)
    covariance = np.full((8, 2, 2), np.nan)
    covariance[admitted.astype(bool)] = np.diag([0.25, 0.5])
    metadata = {
        'schema': 'matched_runtime_covariance_opportunities.v1',
        'runtime_covariance_is_out_of_fold': True,
        'correction_prediction_is_out_of_fold': True,
        'resampling_unit': 'complete_drive',
        'prediction_drive_excluded_from_fit': True,
        'runtime_covariance_model': 'synthetic_R2',
        'final_audit_accessed': False,
        'camera_order': ['camera_A', 'camera_B'],
    }
    np.savez(
        source,
        positions_xy_m=positions,
        camera_ids=cameras,
        admitted=admitted,
        runtime_covariance_m2=covariance,
        metadata_json=json.dumps(metadata),
    )
    subprocess.run([
        sys.executable,
        str(REPO / 'experiments/thesis_pipeline_lock/run_stage08_planning_information.py'),
        '--opportunities', str(source),
        '--output', str(output),
        '--grid-step-m', '0.5',
        '--length-scale-m', '0.4',
        '--support-radius-m', '1.0',
        '--support-tau', '2.0',
    ], cwd=REPO, check=True, capture_output=True, text=True)
    model = CameraNetworkModel(output, expected_camera_ids=['camera_A', 'camera_B'])
    assert model.direct_information
    query = model.query(np.asarray([1.0, 0.0, 0.0]))
    assert query['expected_information'].shape == (2, 2, 2)
    assert np.linalg.eigvalsh(query['expected_information']).min() >= -1e-12
    assert query['opportunity_support'].shape == (2,)


def test_loader_accepts_position_level_cross_fitting_and_rejects_leakage(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location

    script = REPO / 'experiments/thesis_pipeline_lock/run_stage08_planning_information.py'
    spec = spec_from_file_location('stage08_direct_information', script)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    path = tmp_path / 'position_oof.npz'
    metadata = {
        'schema': 'matched_runtime_covariance_opportunities.v1',
        'runtime_covariance_is_out_of_fold': True,
        'correction_prediction_is_out_of_fold': True,
        'resampling_unit': 'complete_reference_position',
        'prediction_reference_position_excluded_from_fit': True,
        'final_audit_accessed': False,
    }
    arrays = dict(
        positions_xy_m=np.asarray([[0.0, 0.0]]), camera_ids=np.asarray(['camera_A']),
        admitted=np.asarray([1]), runtime_covariance_m2=np.asarray([np.eye(2)]),
    )
    np.savez(path, **arrays, metadata_json=json.dumps(metadata))
    loaded, _ = module.load_opportunities(path)
    assert loaded['resampling_unit'] == 'complete_reference_position'
    metadata['prediction_reference_position_excluded_from_fit'] = False
    np.savez(path, **arrays, metadata_json=json.dumps(metadata))
    with pytest.raises(ValueError, match='out-of-fold predictions'):
        module.load_opportunities(path)
