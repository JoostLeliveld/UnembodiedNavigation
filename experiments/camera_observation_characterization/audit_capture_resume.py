#!/usr/bin/env python3
"""Read-only preflight for resuming a stopped five-camera capture.

Run after the writer and any storage converter have exited. This does not repair indexes,
rename images, or waive provenance checks. Space is budgeted without assuming future hash
deduplication, using the largest observed PNG (an empirical estimate, not a hard bound).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_hash(image) -> str:
    digest = hashlib.sha1()  # provenance of decoded pixels, not security
    digest.update(str(image.shape).encode('ascii'))
    digest.update(str(image.dtype).encode('ascii'))
    digest.update(image.tobytes())
    return digest.hexdigest()


def audit(capture: Path, *, repo: Path = REPO, reserve_bytes: int = 256 * 1024**2) -> dict:
    capture = capture.resolve()
    index = capture / 'capture_index.csv'
    manifest_path = capture / 'capture_manifest.json'
    index_hash, manifest_hash = sha256(index), sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    with index.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    plan = manifest['plan']
    cameras = {item['camera_id'] for item in manifest['cameras']}
    batches = collections.defaultdict(list)
    for row in rows:
        batches[(int(row['pose_id']), int(row['repetition_id']))].append(row)
    repeats = int(plan['repeats'])
    pose_ids = sorted({key[0] for key in batches})
    next_pose = max(pose_ids, default=-1) + 1
    checks = {
        'running_manifest': manifest.get('status') == 'running',
        # The current capture resume implementation counts five rows per pose, not repeat.
        'resume_implementation_supports_repeats': repeats == 1,
        'all_rows_successful': bool(rows) and all(r['capture_status'] == 'ok' for r in rows),
        'complete_camera_batches': bool(batches) and all(
            len(batch) == len(cameras) and {r['camera_id'] for r in batch} == cameras
            for batch in batches.values()
        ),
        'contiguous_pose_prefix': pose_ids == list(range(next_pose)),
        'all_repetitions_present': set(batches) == {
            (pose, repetition) for pose in range(next_pose) for repetition in range(repeats)
        },
        'unfinished_plan': next_pose < int(plan['pose_count']),
        'planned_rows_consistent': int(plan['planned_rows']) == (
            int(plan['pose_count']) * repeats * len(cameras)
        ),
        'source_batch_ids_match': all(
            row['source_batch_id'] == f"pose_{int(row['pose_id']):06d}_r{int(row['repetition_id']):02d}"
            for row in rows
        ),
    }
    timing_ok = True
    limit = float(plan['batch_sync_slop_ms']) / 1000
    for batch in batches.values():
        stamps = [float(row['image_stamp_s']) for row in batch]
        spans = [float(row['batch_image_span_s']) for row in batch]
        actual = max(stamps) - min(stamps)
        timing_ok &= all(math.isfinite(x) for x in stamps + spans)
        timing_ok &= 0 <= actual <= limit + 1e-9
        timing_ok &= all(abs(span - actual) <= 1e-6 for span in spans)
    checks['capture_stamps_coherent'] = bool(timing_ok)

    targets = {
        'pose_file': (plan['pose_file'], plan['pose_file_sha256']),
        'world': (manifest['world_path'], manifest['world_sha256']),
        'profiles': (manifest['world_profiles_path'], manifest['world_profiles_sha256']),
        'capture_script': (repo / 'experiments/camera_observation_characterization/capture_bbox_grid.py',
                           manifest['capture_script_sha256']),
        'capture_helper': (repo / 'scripts/perception/capture_yolo_dataset.py',
                           manifest['capture_helper_sha256']),
    }
    for name, (path, expected) in targets.items():
        checks[name + '_unchanged'] = Path(path).is_file() and sha256(Path(path)) == expected

    expected_by_path = collections.defaultdict(set)
    for row in rows:
        expected_by_path[row['image']].add(row['image_sha1'])
    checks['one_hash_per_image'] = all(len(v) == 1 for v in expected_by_path.values())
    problems, png_sizes = [], []
    for relative, expected in sorted(expected_by_path.items()):
        path = (capture / relative).resolve()
        if not path.is_relative_to(capture) or not path.is_file():
            problems.append({'image': relative, 'reason': 'missing_or_outside_capture'})
            continue
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if pixels is None or {image_hash(pixels)} != expected:
            problems.append({'image': relative, 'reason': 'decoded_hash_mismatch'})
            continue
        # Resume writes PNG even when existing images were converted to WebP. Estimate the
        # same default PNG encoding, rather than budgeting from compressed WebP file sizes.
        if path.suffix.lower() == '.png':
            png_sizes.append(path.stat().st_size)
        else:
            ok, encoded = cv2.imencode('.png', pixels)
            if not ok:
                problems.append({'image': relative, 'reason': 'png_size_estimation_failed'})
            else:
                png_sizes.append(len(encoded))
    checks['all_indexed_images_verified'] = not problems
    remaining = max(0, int(plan['planned_rows']) - len(rows))
    estimated = math.ceil(1.10 * remaining * max(png_sizes, default=0)) + reserve_bytes
    free = shutil.disk_usage(capture).free
    checks['space_for_remaining_without_deduplication'] = bool(png_sizes) and free >= estimated
    checks['index_unchanged_during_audit'] = sha256(index) == index_hash
    checks['manifest_unchanged_during_audit'] = sha256(manifest_path) == manifest_hash
    return {
        'schema': 'capture_resume_preflight.v1', 'capture': str(capture),
        'ready': all(checks.values()), 'checks': checks,
        'rows_retained': len(rows), 'next_pose_id': next_pose,
        'remaining_camera_opportunities': remaining,
        'unique_indexed_images': len(expected_by_path),
        'image_problem_count': len(problems), 'image_problems_first_20': problems[:20],
        'space': {'free_bytes': free, 'estimated_required_bytes': estimated,
                  'reserve_bytes': reserve_bytes,
                  'basis': '110% of largest observed default PNG times remaining opportunities; no deduplication assumed'},
        'capture_index_sha256': index_hash, 'capture_manifest_sha256': manifest_hash,
        'limitation': 'Run only after capture/conversion writers exit; this audit is not a process lock.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.capture)
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded)
    return 0 if report['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
