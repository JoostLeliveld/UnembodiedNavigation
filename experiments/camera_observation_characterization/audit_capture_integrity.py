#!/usr/bin/env python3
"""Audit all-camera capture timing, hashes, row accounting, and detector provenance."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'perception'))
from capture_yolo_dataset import _sha1_array  # noqa: E402

CAMERAS = {'camera_A', 'camera_B', 'camera_C', 'camera_D', 'camera_E'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--max-batch-span-s', type=float, default=0.05)
    parser.add_argument('--output-name', default='capture_integrity_audit.json')
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    paths = {
        name: capture / name for name in (
            'capture_manifest.json', 'capture_index.csv', 'bbox_detector_manifest.json',
            'bbox_observations.csv', 'observation_interpretations.csv',
        )
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f'Missing capture artifacts: {missing}')

    cap = json.loads(paths['capture_manifest.json'].read_text(encoding='utf-8'))
    det = json.loads(paths['bbox_detector_manifest.json'].read_text(encoding='utf-8'))
    capture_rows = list(csv.DictReader(paths['capture_index.csv'].open(encoding='utf-8')))
    detector_rows = list(csv.DictReader(paths['bbox_observations.csv'].open(encoding='utf-8')))
    interpretation_rows = list(
        csv.DictReader(paths['observation_interpretations.csv'].open(encoding='utf-8'))
    )

    checks: dict[str, bool] = {
        'capture_complete': cap.get('status') == 'complete',
        'capture_schema_v2': cap.get('schema') == 'bbox_characterization_capture.v2',
        'detector_complete': det.get('status') == 'complete',
        'detector_schema_v2': det.get('schema') == 'bbox_characterization_detector.v2',
        'transport_isolated': bool(cap.get('transport', {}).get('isolation_verified')),
        'no_failed_capture_rows': all(row['capture_status'] == 'ok' for row in capture_rows),
        'planned_row_count': len(capture_rows) == int(cap['plan']['planned_rows']),
        'detector_attempt_row_count': len(detector_rows) == len(capture_rows),
        'interpretation_attempt_row_count': len(interpretation_rows) == len(capture_rows),
        'detector_has_explicit_clipping_flag': bool(
            detector_rows and 'detector_clipped' in detector_rows[0]
        ),
    }

    batches: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in capture_rows:
        batches[row.get('source_batch_id', '')].append(row)
    checks['source_batch_id_present'] = '' not in batches
    checks['exactly_five_cameras_per_batch'] = all(
        len(rows) == 5 and {row['camera_id'] for row in rows} == CAMERAS
        for rows in batches.values()
    )
    spans = [float(row['batch_image_span_s']) for row in capture_rows]
    max_span = max(spans) if spans else math.inf
    checks['batch_span_within_limit'] = max_span <= float(args.max_batch_span_s)

    capture_keys = {(row['source_batch_id'], row['camera_id']) for row in capture_rows}
    detector_keys = {(row['source_batch_id'], row['camera_id']) for row in detector_rows}
    interpretation_keys = {
        (row['source_batch_id'], row['camera_id']) for row in interpretation_rows
    }
    checks['source_batch_ids_preserved'] = (
        detector_keys == capture_keys and interpretation_keys == capture_keys
    )

    decoded_hash_mismatches = []
    for row in capture_rows:
        image_path = capture / row['image']
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or _sha1_array(image) != row['image_sha1']:
            decoded_hash_mismatches.append(str(image_path))
    checks['decoded_image_hashes_match'] = not decoded_hash_mismatches
    checks['capture_index_hash_matches_manifest'] = (
        sha256(paths['capture_index.csv']) == cap.get('capture_index_sha256')
    )
    checks['detector_inputs_match_capture'] = (
        sha256(paths['capture_index.csv']) == det.get('capture_index_sha256')
        and sha256(paths['capture_manifest.json']) == det.get('capture_manifest_sha256')
    )
    checks['detector_output_hash_matches_manifest'] = (
        sha256(paths['bbox_observations.csv']) == det.get('bbox_observations_sha256')
    )

    hash_targets = {
        'pose_file': (cap['plan'].get('pose_file'), cap['plan'].get('pose_file_sha256')),
        'world': (cap.get('world_path'), cap.get('world_sha256')),
        'world_profiles': (
            cap.get('world_profiles_path'), cap.get('world_profiles_sha256')
        ),
        'capture_script': (
            str(REPO / 'experiments/camera_observation_characterization/capture_bbox_grid.py'),
            cap.get('capture_script_sha256'),
        ),
        'capture_helper': (
            str(REPO / 'scripts/perception/capture_yolo_dataset.py'),
            cap.get('capture_helper_sha256'),
        ),
        'detector_script': (
            det.get('detector_script'), det.get('detector_script_sha256')
        ),
        'selection_helper': (
            det.get('selection_helper'), det.get('selection_helper_sha256')
        ),
        'detector_weights': (det.get('weights'), det.get('weights_sha256')),
    }
    hash_status = {}
    for name, (path_value, expected) in hash_targets.items():
        path = Path(path_value).expanduser().resolve() if path_value else None
        ok = bool(path and path.is_file() and expected and sha256(path) == expected)
        hash_status[name] = {
            'path': str(path) if path else None, 'expected_sha256': expected, 'matches': ok,
        }
        checks[f'{name}_hash_matches'] = ok

    report = {
        'experiment': 'all_camera_capture_integrity_audit',
        'capture': str(capture),
        'passed': bool(all(checks.values())),
        'checks': checks,
        'counts': {
            'source_batches': len(batches),
            'camera_opportunities': len(capture_rows),
            'unique_decoded_images': len({row['image_sha1'] for row in capture_rows}),
            'detector_hits': sum(int(row['detected']) for row in detector_rows),
            'detector_misses': len(detector_rows) - sum(
                int(row['detected']) for row in detector_rows
            ),
            'detector_clipped_hits': sum(
                int(row['detector_clipped']) for row in detector_rows
            ),
        },
        'timing': {
            'maximum_batch_span_s': max_span,
            'required_maximum_s': float(args.max_batch_span_s),
        },
        'hash_targets': hash_status,
        'decoded_hash_mismatches': decoded_hash_mismatches,
        'artifact_sha256': {name: sha256(path) for name, path in paths.items()},
    }
    output = capture / args.output_name
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
