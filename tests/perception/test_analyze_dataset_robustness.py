from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / 'scripts' / 'perception'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_dataset_robustness import analyze_dataset_dir


def test_analyze_visibility_capture_flags_single_yaw_and_duplicates(tmp_path: Path) -> None:
    rows = [
        {'sample_id': '0', 'x': '0.0', 'y': '0.0', 'theta': '0.0'},
        {'sample_id': '1', 'x': '1.0', 'y': '0.0', 'theta': '0.0'},
    ]
    with (tmp_path / 'samples.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id', 'x', 'y', 'theta'])
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / 'capture_manifest.json').write_text(
        json.dumps({'duplicate_image_count': 2}),
        encoding='utf-8',
    )

    result = analyze_dataset_dir(tmp_path)
    assert result['kind'] == 'visibility_capture'
    assert result['unique_yaw_count'] == 1
    assert any('Only one unique yaw' in warning for warning in result['warnings'])
    assert any('duplicate frames' in warning for warning in result['warnings'])


def test_analyze_projected_dataset_flags_exact_overlap(tmp_path: Path) -> None:
    with (tmp_path / 'capture_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['x', 'y', 'yaw_rad', 'accepted', 'split'])
        writer.writeheader()
        writer.writerows([
            {'x': '0.0', 'y': '0.0', 'yaw_rad': '0.0', 'accepted': '1', 'split': 'train'},
            {'x': '0.0', 'y': '0.0', 'yaw_rad': '0.0', 'accepted': '1', 'split': 'val'},
            {'x': '1.0', 'y': '0.0', 'yaw_rad': '1.57', 'accepted': '1', 'split': 'train'},
        ])
    (tmp_path / 'capture_manifest.json').write_text(
        json.dumps({'yaw_samples': 1, 'split_mode': 'cyclic'}),
        encoding='utf-8',
    )

    result = analyze_dataset_dir(tmp_path)
    assert result['kind'] == 'projected_bbox_dataset'
    assert result['exact_pose_overlap_count'] == 1
    assert any('yaw_samples <= 1' in warning for warning in result['warnings'])
    assert any('exact pose overlap' in warning for warning in result['warnings'])
