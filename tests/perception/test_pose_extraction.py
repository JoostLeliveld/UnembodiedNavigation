"""Unit tests for the pure-numpy pose-keypoint selection helper."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_PKG = REPO_ROOT / 'src' / 'perception'
if str(PERCEPTION_PKG) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_PKG))

from perception.core.pose_extraction import select_pose_detection


def _kp(front_uv, rear_uv, front_conf=1.0, rear_conf=1.0):
    return np.asarray([[
        [front_uv[0], front_uv[1], front_conf],
        [rear_uv[0], rear_uv[1], rear_conf],
    ]], dtype=float)


def test_select_returns_invalid_when_no_boxes() -> None:
    sel = select_pose_detection(
        boxes_xyxy=np.zeros((0, 4)), boxes_conf=np.zeros((0,)),
        boxes_cls=None, keypoints_data=np.zeros((0, 2, 3)),
        target_ids=None, confidence_threshold=0.25, min_keypoint_conf=0.5,
    )
    assert not sel.valid


def test_select_filters_by_confidence_threshold() -> None:
    sel = select_pose_detection(
        boxes_xyxy=np.asarray([[0.0, 0.0, 10.0, 10.0]]),
        boxes_conf=np.asarray([0.10]),
        boxes_cls=None,
        keypoints_data=_kp((6, 5), (4, 5)),
        target_ids=None, confidence_threshold=0.25, min_keypoint_conf=0.5,
    )
    assert not sel.valid


def test_select_filters_by_class_id() -> None:
    sel = select_pose_detection(
        boxes_xyxy=np.asarray([[0.0, 0.0, 10.0, 10.0]]),
        boxes_conf=np.asarray([0.9]),
        boxes_cls=np.asarray([3]),
        keypoints_data=_kp((6, 5), (4, 5)),
        target_ids={0}, confidence_threshold=0.25, min_keypoint_conf=0.5,
    )
    assert not sel.valid


def test_yaw_zero_when_front_right_of_rear() -> None:
    # Image y=v grows downward; front_u=10, rear_u=4, same v -> dy=0, dx=6 -> atan2(0,6)=0
    sel = select_pose_detection(
        boxes_xyxy=np.asarray([[0.0, 0.0, 100.0, 100.0]]),
        boxes_conf=np.asarray([0.9]),
        boxes_cls=np.asarray([0]),
        keypoints_data=_kp((10, 50), (4, 50)),
        target_ids={0}, confidence_threshold=0.25, min_keypoint_conf=0.5,
    )
    assert sel.valid
    assert math.isclose(sel.yaw_image, 0.0, abs_tol=1e-9)
    assert math.isclose(sel.selected_u, 7.0)
    assert math.isclose(sel.selected_v, 50.0)


def test_yaw_pi_when_front_left_of_rear() -> None:
    sel = select_pose_detection(
        boxes_xyxy=np.asarray([[0.0, 0.0, 100.0, 100.0]]),
        boxes_conf=np.asarray([0.9]),
        boxes_cls=np.asarray([0]),
        keypoints_data=_kp((4, 50), (10, 50)),
        target_ids={0}, confidence_threshold=0.25, min_keypoint_conf=0.5,
    )
    assert sel.valid
    assert math.isclose(abs(sel.yaw_image), math.pi, abs_tol=1e-9)


def test_low_keypoint_conf_marks_invalid_but_keeps_box_info() -> None:
    sel = select_pose_detection(
        boxes_xyxy=np.asarray([[0.0, 0.0, 100.0, 100.0]]),
        boxes_conf=np.asarray([0.9]),
        boxes_cls=np.asarray([0]),
        keypoints_data=_kp((10, 50), (4, 50), front_conf=0.2, rear_conf=0.9),
        target_ids={0}, confidence_threshold=0.25, min_keypoint_conf=0.5,
    )
    assert not sel.valid
    assert math.isnan(sel.yaw_image)
    assert sel.bbox_xyxy is not None
    assert sel.box_score == 0.9


def test_picks_highest_confidence_detection() -> None:
    # Two boxes; second has higher conf and front-rear sep along +y => yaw=pi/2
    keypoints = np.asarray([
        [[0.0, 0.0, 0.9], [10.0, 0.0, 0.9]],
        [[5.0, 20.0, 0.9], [5.0, 0.0, 0.9]],
    ], dtype=float)
    sel = select_pose_detection(
        boxes_xyxy=np.asarray([[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 20.0, 20.0]]),
        boxes_conf=np.asarray([0.6, 0.95]),
        boxes_cls=np.asarray([0, 0]),
        keypoints_data=keypoints,
        target_ids={0}, confidence_threshold=0.25, min_keypoint_conf=0.5,
    )
    assert sel.valid
    assert math.isclose(sel.yaw_image, math.pi / 2.0, abs_tol=1e-9)
    assert math.isclose(sel.box_score, 0.95)
