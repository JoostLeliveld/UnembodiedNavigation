"""Four-camera keypoint captures must not confuse one camera for another.

A capture in the four-camera world interleaves cameras row by row. Back-projecting
a reading through the wrong camera still yields a plausible-looking position, so
these guard the plumbing that keeps each reading with its own camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = REPO_ROOT / 'experiments' / 'keypoint_measurement'
SCRIPT_DIR = REPO_ROOT / 'scripts' / 'perception'
for path in (STUDY_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_keypoint_model import cameras_from_manifest  # noqa: E402

FOUR_CAM_MANIFEST = {
    'camera_pose': [-6.0, -10.0, 6.1, 0.0, 0.92, 1.5708],
    'cameras': {
        'external_camera': {'pose': [-6.0, -10.0, 6.1, 0.0, 0.92, 1.5708],
                            'img_width': 1280, 'img_height': 720, 'fov_h_rad': 1.5708},
        'external_camera_b': {'pose': [-6.0, 10.0, 6.1, 0.0, 0.92, -1.5708],
                              'img_width': 1280, 'img_height': 720, 'fov_h_rad': 1.5708},
        'external_camera_c': {'pose': [6.0, -10.0, 6.1, 0.0, 0.92, 1.5708],
                              'img_width': 1280, 'img_height': 720, 'fov_h_rad': 1.5708},
        'external_camera_d': {'pose': [6.0, 10.0, 6.1, 0.0, 0.92, -1.5708],
                              'img_width': 1280, 'img_height': 720, 'fov_h_rad': 1.5708},
    },
}

ONE_CAM_MANIFEST = {'camera_pose': [0.0, -5.5, 4.8, 0.0, 0.92, 1.5708]}


def test_every_camera_gets_its_own_model() -> None:
    cameras = cameras_from_manifest(FOUR_CAM_MANIFEST, 1280, 720)
    assert set(cameras) == set(FOUR_CAM_MANIFEST['cameras'])
    for name, spec in FOUR_CAM_MANIFEST['cameras'].items():
        assert list(cameras[name].cam_pos) == spec['pose'][:3]


def test_cameras_facing_opposite_ways_are_not_the_same_model() -> None:
    """b and d look south while the others look north; a mix-up would flip the
    reading to the far side of the warehouse."""
    cameras = cameras_from_manifest(FOUR_CAM_MANIFEST, 1280, 720)
    north = cameras['external_camera'].look_at
    south = cameras['external_camera_b'].look_at
    assert north[1] < FOUR_CAM_MANIFEST['cameras']['external_camera']['pose'][1] + 6.0
    assert south[1] > FOUR_CAM_MANIFEST['cameras']['external_camera_b']['pose'][1] - 6.0
    assert north[1] != south[1]


def test_a_single_camera_capture_still_loads() -> None:
    """Captures written before multi-camera support have no `cameras` key."""
    cameras = cameras_from_manifest(ONE_CAM_MANIFEST, 1280, 720)
    assert len(cameras) == 1
    assert list(next(iter(cameras.values())).cam_pos) == [0.0, -5.5, 4.8]


def test_diagnostics_record_which_camera_each_row_came_from() -> None:
    source = (SCRIPT_DIR / 'capture_projected_keypoint_dataset.py').read_text(encoding='utf-8')
    assert "'sample_idx', 'camera'," in source, 'the camera column must be written per row'


def test_all_cameras_of_one_pose_share_a_split() -> None:
    """Otherwise the same pose lands in train and val and 'held out' is a lie."""
    source = (SCRIPT_DIR / 'capture_projected_keypoint_dataset.py').read_text(encoding='utf-8')
    pose_loop = source.index('for sample_idx, (x, y, yaw) in enumerate(planned):')
    camera_loop = source.index('for camera_name in camera_names:')
    split_assign = source.index("split = str(split_labels[sample_idx])")
    assert pose_loop < split_assign < camera_loop, (
        'the split must be chosen once per pose, before the per-camera loop'
    )
