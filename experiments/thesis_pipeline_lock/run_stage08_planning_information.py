#!/usr/bin/env python3
"""Fit the frozen per-camera planning-information fields for Stage 08."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src/reliability'), str(REPO / 'src/unav_common')]
from reliability.planning_information import fit_planning_information_field  # noqa: E402


INPUT_SCHEMA = 'matched_runtime_covariance_opportunities.v1'
OUTPUT_SCHEMA = 'camera_network.thesis_stage09.v3'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return f'external_inputs/{path.name}'


def grid_axis(lower: float, upper: float, step: float) -> np.ndarray:
    count = max(2, int(math.ceil((upper - lower) / step)) + 1)
    return np.linspace(lower, upper, count)


def load_opportunities(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            'metadata_json', 'positions_xy_m', 'camera_ids', 'admitted',
            'runtime_covariance_m2',
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f'opportunity artifact is missing {sorted(missing)}')
        metadata = json.loads(str(archive['metadata_json'].item()))
        arrays = {
            key: np.asarray(archive[key]) for key in required if key != 'metadata_json'
        }
        if 'headings_rad' in archive.files:
            arrays['headings_rad'] = np.asarray(archive['headings_rad'], dtype=float)
    if metadata.get('schema') != INPUT_SCHEMA:
        raise ValueError(f'opportunity artifact must use {INPUT_SCHEMA}')
    if metadata.get('runtime_covariance_is_out_of_fold') is not True:
        raise ValueError('runtime covariance predictions must be out of fold')
    if metadata.get('correction_prediction_is_out_of_fold') is not True:
        raise ValueError('correction predictions must be out of fold')
    unit = metadata.get('resampling_unit')
    exclusion_key = {
        'complete_drive': 'prediction_drive_excluded_from_fit',
        'complete_reference_position': 'prediction_reference_position_excluded_from_fit',
    }.get(unit)
    if exclusion_key is None or metadata.get(exclusion_key) is not True:
        raise ValueError(
            'Stage 08 requires out-of-fold predictions excluding the complete '
            'drive or complete reference position containing each row')
    if metadata.get('final_audit_accessed') is not False:
        raise ValueError('Stage 08 fitting input must not include final-audit data')
    return metadata, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opportunities', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid-step-m', type=float, default=0.20)
    parser.add_argument('--length-scale-m', type=float, required=True)
    parser.add_argument('--support-radius-m', type=float, required=True)
    parser.add_argument('--support-tau', type=float, required=True)
    args = parser.parse_args()

    source = args.opportunities.resolve()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + '.incomplete')
    if output.exists() or incomplete.exists():
        raise FileExistsError('refusing to overwrite a planning-information artifact')
    if not math.isfinite(args.grid_step_m) or args.grid_step_m <= 0.0:
        raise ValueError('grid step must be positive')
    source_metadata, arrays = load_opportunities(source)
    positions = np.asarray(arrays['positions_xy_m'], dtype=float)
    cameras_per_opportunity = np.asarray(arrays['camera_ids']).astype(str)
    camera_ids = tuple(str(value) for value in source_metadata['camera_order'])
    if set(cameras_per_opportunity) != set(camera_ids):
        raise ValueError('opportunity cameras differ from the declared camera roster')
    xs = grid_axis(float(positions[:, 0].min()), float(positions[:, 0].max()),
                   args.grid_step_m)
    ys = grid_axis(float(positions[:, 1].min()), float(positions[:, 1].max()),
                   args.grid_step_m)
    field = fit_planning_information_field(
        positions,
        cameras_per_opportunity,
        np.asarray(arrays['admitted'], dtype=bool),
        np.asarray(arrays['runtime_covariance_m2'], dtype=float),
        camera_ids=camera_ids,
        xs=xs,
        ys=ys,
        length_scale_m=args.length_scale_m,
        support_radius_m=args.support_radius_m,
        support_tau=args.support_tau,
        headings_rad=arrays.get('headings_rad'),
    )

    implementation = Path(__file__).resolve()
    fitter = REPO / 'src/reliability/reliability/planning_information.py'
    sources = {
        relative(source): sha256(source),
        relative(implementation): sha256(implementation),
        relative(fitter): sha256(fitter),
    }
    metadata = {
        'schema': OUTPUT_SCHEMA,
        'pipeline_id': 'THESIS-FINAL-PIPELINE-V1',
        'reference': 'robot_ground_reference_xy',
        'frame': 'map_bev',
        'covariance_units': 'm2',
        'information_units': 'm-2',
        'planning_target': 'admitted_runtime_precision_else_zero',
        'belief_update': 'deterministic_information_approximation',
        'support_population': 'all_camera_opportunities',
        'heading_treatment': 'equal_declared_heading_pool_within_reference_position',
        'unsupported_limit': 'zero_information',
        'runtime_covariance_source': source_metadata.get('runtime_covariance_model'),
        'runtime_covariance_is_out_of_fold': True,
        'correction_prediction_is_out_of_fold': True,
        'resampling_unit': source_metadata['resampling_unit'],
        'final_audit_accessed': False,
        'fit_parameters': {
            'grid_step_m': float(args.grid_step_m),
            'length_scale_m': float(args.length_scale_m),
            'support_radius_m': float(args.support_radius_m),
            'support_tau': float(args.support_tau),
        },
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_hashes': sources,
    }
    exclusion_key = {
        'complete_drive': 'prediction_drive_excluded_from_fit',
        'complete_reference_position': 'prediction_reference_position_excluded_from_fit',
    }[source_metadata['resampling_unit']]
    metadata[exclusion_key] = True
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        xs=field.xs,
        ys=field.ys,
        camera_ids=np.asarray(field.camera_ids),
        expected_information_m2_inv=field.expected_information,
        opportunity_support=field.opportunity_support,
        raw_expected_information_m2_inv=field.raw_expected_information,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_bytes(buffer.getvalue())
    os.replace(incomplete, output)
    print(json.dumps({
        'path': relative(output),
        'sha256': sha256(output),
        'camera_ids': list(field.camera_ids),
        'grid_shape': [len(field.ys), len(field.xs)],
        'maximum_support': float(field.opportunity_support.max()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
