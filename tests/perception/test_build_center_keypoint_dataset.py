from __future__ import annotations

import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "perception"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_center_keypoint_dataset import pose_label, project_point


def test_projected_centre_is_in_image_for_downward_camera() -> None:
    u, v, visible = project_point(
        (0.0, 0.0, 0.35),
        camera_pose_xyz_rpy=[0.0, -5.0, 5.0, 0.0, math.pi / 4.0, math.pi / 2.0],
        image_width=1280,
        image_height=720,
        fov_h_rad=math.pi / 2.0,
    )
    assert visible
    assert abs(u - 640.0) < 1.0e-8
    assert 0.0 <= v < 720.0


def test_pose_label_expands_visible_box_to_contain_amodal_centre() -> None:
    label, expanded = pose_label(
        (10.0, 20.0, 30.0, 40.0),
        (35.0, 30.0),
        visibility=1,
        width=100,
        height=80,
    )
    fields = label.split()
    assert expanded
    assert fields[0] == "0"
    assert len(fields) == 8
    assert float(fields[5]) == 0.35
    assert float(fields[6]) == 0.375
    assert fields[7] == "1"


def test_empty_observation_writes_empty_label() -> None:
    assert pose_label(None, None, visibility=0, width=100, height=80) == ("", False)
