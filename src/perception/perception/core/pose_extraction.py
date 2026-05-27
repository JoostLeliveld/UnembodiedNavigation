"""Pure-numpy helpers to extract front/rear keypoints from a YOLO pose result.

These functions accept already-converted numpy arrays so they can be unit-tested
without torch / ultralytics installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from perception.core.pose_keypoints import (
    KEYPOINT_FRONT,
    KEYPOINT_REAR,
    NUM_KEYPOINTS,
)


@dataclass(frozen=True)
class PoseKeypointSelection:
    selected_u: float
    selected_v: float
    front_u: float
    front_v: float
    rear_u: float
    rear_v: float
    front_conf: float
    rear_conf: float
    yaw_image: float  # image-plane yaw, only valid as a "heading is present" sentinel
    box_score: float
    bbox_xyxy: tuple[float, float, float, float] | None
    valid: bool


def _empty_selection() -> PoseKeypointSelection:
    return PoseKeypointSelection(
        selected_u=math.nan, selected_v=math.nan,
        front_u=math.nan, front_v=math.nan,
        rear_u=math.nan, rear_v=math.nan,
        front_conf=0.0, rear_conf=0.0,
        yaw_image=math.nan, box_score=0.0,
        bbox_xyxy=None, valid=False,
    )


def select_pose_detection(
    *,
    boxes_xyxy: np.ndarray,
    boxes_conf: np.ndarray,
    boxes_cls: np.ndarray | None,
    keypoints_data: np.ndarray,
    target_ids: set[int] | None,
    confidence_threshold: float,
    min_keypoint_conf: float,
) -> PoseKeypointSelection:
    """Pick the best detection and extract the two front/rear keypoints.

    Inputs are numpy arrays:
      boxes_xyxy:      (N, 4)
      boxes_conf:      (N,)
      boxes_cls:       (N,) or None
      keypoints_data:  (N, K, 3) where the last axis is (x, y, conf).

    Returns a PoseKeypointSelection. If no detection passes the class+conf gate
    or keypoints are missing, .valid is False (other fields are NaN).
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=float)
    boxes_conf = np.asarray(boxes_conf, dtype=float)
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4 or boxes_conf.size == 0:
        return _empty_selection()

    valid = np.isfinite(boxes_conf)
    if target_ids is not None and boxes_cls is not None:
        cls_arr = np.asarray(boxes_cls, dtype=int)
        valid &= np.asarray([int(c) in target_ids for c in cls_arr], dtype=bool)
    valid &= boxes_conf >= float(confidence_threshold)
    candidate_idxs = np.flatnonzero(valid)
    if candidate_idxs.size == 0:
        return _empty_selection()

    best_local = int(np.argmax(boxes_conf[candidate_idxs]))
    best_i = int(candidate_idxs[best_local])
    score = float(boxes_conf[best_i])

    kdata = np.asarray(keypoints_data, dtype=float)
    if kdata.ndim != 3 or kdata.shape[1] < NUM_KEYPOINTS or kdata.shape[2] < 2:
        return _empty_selection()
    kp = kdata[best_i]
    front = kp[KEYPOINT_FRONT]
    rear = kp[KEYPOINT_REAR]
    front_conf = float(front[2]) if kp.shape[1] >= 3 else 1.0
    rear_conf = float(rear[2]) if kp.shape[1] >= 3 else 1.0

    bbox = tuple(float(v) for v in boxes_xyxy[best_i, :4].tolist())

    front_u, front_v = float(front[0]), float(front[1])
    rear_u, rear_v = float(rear[0]), float(rear[1])
    if (
        front_conf < float(min_keypoint_conf)
        or rear_conf < float(min_keypoint_conf)
        or not all(math.isfinite(v) for v in (front_u, front_v, rear_u, rear_v))
    ):
        # Detection exists but heading is not trustworthy.
        return PoseKeypointSelection(
            selected_u=0.5 * (front_u + rear_u) if math.isfinite(front_u) and math.isfinite(rear_u) else math.nan,
            selected_v=0.5 * (front_v + rear_v) if math.isfinite(front_v) and math.isfinite(rear_v) else math.nan,
            front_u=front_u, front_v=front_v, rear_u=rear_u, rear_v=rear_v,
            front_conf=front_conf, rear_conf=rear_conf,
            yaw_image=math.nan, box_score=score,
            bbox_xyxy=bbox, valid=False,
        )

    selected_u = 0.5 * (front_u + rear_u)
    selected_v = 0.5 * (front_v + rear_v)
    yaw_image = math.atan2(front_v - rear_v, front_u - rear_u)
    return PoseKeypointSelection(
        selected_u=selected_u, selected_v=selected_v,
        front_u=front_u, front_v=front_v, rear_u=rear_u, rear_v=rear_v,
        front_conf=front_conf, rear_conf=rear_conf,
        yaw_image=yaw_image, box_score=score,
        bbox_xyxy=bbox, valid=True,
    )


def model_exposes_keypoints(model) -> bool:
    """Heuristic: does this Ultralytics model produce result.keypoints?

    Pose models report `task == 'pose'` (or have non-empty kpt_shape).
    Falls back to False if attributes are missing so seg models stay on the
    legacy branch.
    """
    task = getattr(model, 'task', None)
    if isinstance(task, str) and task.strip().lower() == 'pose':
        return True
    overrides = getattr(model, 'overrides', None)
    if isinstance(overrides, dict):
        if str(overrides.get('task', '')).strip().lower() == 'pose':
            return True
    return False
