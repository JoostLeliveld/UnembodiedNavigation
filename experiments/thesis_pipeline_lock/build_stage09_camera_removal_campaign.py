#!/usr/bin/env python3
"""Build the canonical R0/R1/R2 by intact/removal navigation campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_list(value) -> list[str]:
    if isinstance(value, str):
        cameras = [item.strip() for item in value.split(',') if item.strip()]
    else:
        cameras = [str(item).strip() for item in value]
    if not cameras or len(cameras) != len(set(cameras)):
        raise ValueError('camera roster must be nonempty and unique')
    return cameras


def validate_planning_artifact(path: Path, expected_cameras: list[str]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json'].item()))
        artifact_cameras = [str(value) for value in archive['camera_ids']]
    if metadata.get('schema') != 'camera_network.matched_covariance_precision.v1':
        raise ValueError('camera-removal campaign requires matched-covariance precision')
    if metadata.get('planning_target') != 'inverse_of_matched_runtime_covariance':
        raise ValueError('planning artifact has the wrong matched-covariance target')
    if metadata.get('additional_planning_fit') is not False:
        raise ValueError('planning artifact must not contain an additional fitted model')
    if metadata.get('detector_opportunities_used') is not False:
        raise ValueError('planning artifact must not use detector opportunities')
    if metadata.get('gate_outcomes_used') is not False:
        raise ValueError('planning artifact must not use gate outcomes')
    if artifact_cameras != expected_cameras:
        raise ValueError('planning artifact camera order differs from the campaign roster')


def build_campaign(config: dict, *, artifacts: dict, removed_camera_id: str) -> dict:
    result = dict(config)
    roster = camera_list(result['manager_camera_ids'])
    removed = str(removed_camera_id).strip()
    if removed not in roster:
        raise ValueError(f'removed camera {removed!r} is not in the runtime roster')
    remaining = [camera for camera in roster if camera != removed]
    if not remaining:
        raise ValueError('camera removal must leave at least one runtime camera')
    full_text = ','.join(roster)
    remaining_text = ','.join(remaining)
    expected_models = ('global', 'per_camera', 'spatial')
    if set(artifacts) != set(expected_models):
        raise ValueError(f'artifacts must contain exactly {expected_models}')
    result['study_title'] = 'Canonical matched R0/R1/R2 camera-removal campaign'
    result['study_comparison'] = (
        'Global, per-camera and spatial camera-network models, each used consistently '
        'by runtime fusion and planning, under intact and camera-removal states.'
    )
    result['planning_information_method'] = 'inverse_of_matched_runtime_covariance'
    result['planning_artifact_schema'] = 'camera_network.matched_covariance_precision.v1'
    result['conditions'] = {}
    for model in expected_models:
        entry = artifacts[model]
        if set(entry) != {'runtime_manifest', 'planning_artifact'}:
            raise ValueError(f'{model} requires runtime_manifest and planning_artifact')
        for state, active in (('intact', roster), ('removal', remaining)):
            active_text = ','.join(active)
            result['conditions'][f'{model}_{state}'] = {
                'label': f'{model}_{state}',
                'manager_visibility_sensor_model_path': str(entry['runtime_manifest']),
                'manager_visibility_sensor_model_expected_sha256': sha256(
                    Path(entry['runtime_manifest'])),
                'camera_network_artifact_path': str(entry['planning_artifact']),
                'camera_network_expected_sha256': sha256(Path(entry['planning_artifact'])),
                'manager_camera_ids': active_text,
                'camera_network_active_camera_ids': active_text,
            }
    for task in result.get('tasks', {}).values():
        task['conditions'] = list(result['conditions'])
    result.pop('manager_availability_model_path', None)
    result.pop('manager_availability_model_expected_sha256', None)
    result['removed_camera_id'] = removed
    # Canonical execution/safety closure.  Keep these explicit in every emitted
    # campaign; launch defaults are deliberately not method authority.
    result.update({
        'local_controller_type': 'ff_fb',
        'robot_length_m': 0.80,
        'robot_width_m': 0.55,
        'use_nogo_cost': True,
        'use_belief_nogo_cost': True,
        'nogo_mode': 'keep_in',
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--artifacts-json', required=True, type=Path,
        help='JSON mapping global/per_camera/spatial to matched runtime and planning artifacts')
    parser.add_argument('--removed-camera-id', required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    config = yaml.safe_load(base.read_text())
    artifacts = json.loads(args.artifacts_json.read_text(encoding='utf-8'))
    for entry in artifacts.values():
        for key in ('runtime_manifest', 'planning_artifact'):
            path = Path(entry[key]).expanduser()
            if not path.is_absolute():
                path = REPO / path
            entry[key] = str(path.resolve())
        validate_planning_artifact(
            Path(entry['planning_artifact']), camera_list(config['manager_camera_ids']))
    updated = build_campaign(
        config,
        artifacts=artifacts,
        removed_camera_id=args.removed_camera_id,
    )
    temporary = output.with_name(output.name + '.incomplete')
    if temporary.exists():
        raise FileExistsError(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(yaml.safe_dump(updated, sort_keys=False))
    os.replace(temporary, output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
