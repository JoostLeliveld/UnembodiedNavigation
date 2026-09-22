import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / 'experiments/thesis_pipeline_lock/build_stage09_camera_removal_campaign.py'
)
SPEC = importlib.util.spec_from_file_location('camera_removal_builder', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_camera_removal_conditions_change_runtime_and_planning_sets(monkeypatch):
    monkeypatch.setattr(MODULE, 'sha256', lambda _path: '1' * 64)
    config = {
        'manager_camera_ids': 'camera_A,camera_B,camera_C',
        'manager_availability_model_path': 'obsolete.json',
        'manager_availability_model_expected_sha256': '0' * 64,
        'tasks': {'task': {'conditions': ['old'], 'seeds': [1]}},
    }
    artifacts = {
        model: {
            'runtime_manifest': f'{model}_runtime.json',
            'planning_artifact': f'{model}_planning.npz',
        }
        for model in ('global', 'per_camera', 'spatial')
    }
    updated = MODULE.build_campaign(
        config,
        artifacts=artifacts,
        removed_camera_id='camera_B',
    )
    conditions = updated['conditions']
    assert tuple(conditions) == (
        'global_intact', 'global_removal', 'per_camera_intact',
        'per_camera_removal', 'spatial_intact', 'spatial_removal')
    for model in ('global', 'per_camera', 'spatial'):
        assert conditions[f'{model}_intact']['manager_camera_ids'] == (
            'camera_A,camera_B,camera_C')
        assert conditions[f'{model}_intact']['camera_network_active_camera_ids'] == (
            'camera_A,camera_B,camera_C')
        assert conditions[f'{model}_removal']['manager_camera_ids'] == 'camera_A,camera_C'
        assert conditions[f'{model}_removal']['camera_network_active_camera_ids'] == (
            'camera_A,camera_C')
        assert conditions[f'{model}_intact']['manager_visibility_sensor_model_path'] == (
            f'{model}_runtime.json')
        assert conditions[f'{model}_intact']['camera_network_artifact_path'] == (
            f'{model}_planning.npz')
    assert updated['tasks']['task']['conditions'] == list(conditions)
    assert 'manager_availability_model_path' not in updated
    assert updated['local_controller_type'] == 'ff_fb'
    assert updated['robot_length_m'] == pytest.approx(0.80)
    assert updated['robot_width_m'] == pytest.approx(0.55)
    assert updated['use_nogo_cost'] is True
    assert updated['use_belief_nogo_cost'] is True
    assert updated['nogo_mode'] == 'keep_in'


def write_artifact(path, *, schema='camera_network.matched_covariance_precision.v1'):
    np.savez_compressed(
        path,
        camera_ids=np.asarray(['camera_A', 'camera_B', 'camera_C']),
        metadata_json=json.dumps({
            'schema': schema,
            'planning_target': 'inverse_of_matched_runtime_covariance',
            'additional_planning_fit': False,
            'detector_opportunities_used': False,
            'gate_outcomes_used': False,
        }),
    )


def test_campaign_builder_rejects_legacy_q_artifact(tmp_path):
    artifact = tmp_path / 'legacy.npz'
    write_artifact(artifact, schema='camera_network.thesis_stage09.v2')
    with pytest.raises(ValueError, match='matched-covariance precision'):
        MODULE.validate_planning_artifact(
            artifact, ['camera_A', 'camera_B', 'camera_C'])


def test_campaign_builder_requires_exact_artifact_camera_order(tmp_path):
    artifact = tmp_path / 'information.npz'
    write_artifact(artifact)
    with pytest.raises(ValueError, match='camera order'):
        MODULE.validate_planning_artifact(
            artifact, ['camera_B', 'camera_A', 'camera_C'])
