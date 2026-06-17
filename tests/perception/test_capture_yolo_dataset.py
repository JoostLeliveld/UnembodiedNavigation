from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / 'scripts' / 'perception'
SRC_UNAV = REPO_ROOT / 'src' / 'unav_common'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_UNAV) not in sys.path:
    sys.path.insert(0, str(SRC_UNAV))

from capture_yolo_dataset import _project_robot_bbox, validate_sample_quality
from unav_common.camera_model import ObliqueCameraModel


def _camera() -> ObliqueCameraModel:
    return ObliqueCameraModel(
        cam_pos=(0.0, -5.0, 4.0),
        look_at=(0.0, 0.0, 0.0),
        img_width=640,
        img_height=360,
        fov_h_rad=1.5708,
    )


def _base_kwargs(camera: ObliqueCameraModel) -> dict:
    return {
        'robot_label': 23,
        'epsilon_ratio': 0.01,
        'bottom_band_px': 3.0,
        'min_mask_area': 20.0,
        'min_mask_bbox_w': 4.0,
        'min_mask_bbox_h': 4.0,
        'max_mask_border_fraction': 0.0,
        'min_rgb_robot_color_fraction': 0.01,
        'disable_rgb_color_check': False,
        'camera': camera,
        'x': 0.0,
        'y': 0.0,
        'yaw': 0.0,
        'robot_z': 0.05,
        'box_length': 0.22,
        'box_width': 0.22,
        'box_height': 0.20,
        'max_expected_center_error_px': 50.0,
    }


def _draw_robot_patch(
    *,
    camera: ObliqueCameraModel,
    image_bgr: np.ndarray,
    labels: np.ndarray,
    offset_px: tuple[int, int] = (0, 0),
    draw_rgb: bool = True,
) -> tuple[int, int, int, int]:
    bbox = _project_robot_bbox(
        camera,
        x=0.0,
        y=0.0,
        yaw=0.0,
        z=0.05,
        box_length=0.22,
        box_width=0.22,
        box_height=0.20,
    )
    assert bbox is not None
    dx, dy = offset_px
    x0 = int(round(bbox[0])) + dx
    y0 = int(round(bbox[1])) + dy
    x1 = int(round(bbox[2])) + dx
    y1 = int(round(bbox[3])) + dy
    x0 = max(0, min(labels.shape[1] - 1, x0))
    y0 = max(0, min(labels.shape[0] - 1, y0))
    x1 = max(x0 + 1, min(labels.shape[1], x1))
    y1 = max(y0 + 1, min(labels.shape[0], y1))
    labels[y0:y1, x0:x1] = 23
    if draw_rgb:
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (40, 45, 220), thickness=-1)
        cv2.circle(image_bgr, ((x0 + x1) // 2, y0 + 2), 3, (220, 100, 40), thickness=-1)
    return x0, y0, x1, y1


def test_validate_sample_accepts_visible_synchronized_segmentation() -> None:
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    labels = np.zeros((360, 640), dtype=np.uint32)
    _draw_robot_patch(camera=camera, image_bgr=image, labels=labels)

    result = validate_sample_quality(image_bgr=image, labels=labels, **_base_kwargs(camera))

    assert result.accepted
    assert result.reason == ''
    assert result.mask_area_px > 20.0
    assert result.rgb_robot_color_fraction > 0.01
    assert math.isfinite(result.expected_center_error_px)
    assert result.expected_center_error_px < 10.0


def test_validate_sample_rejects_projection_mismatch() -> None:
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    labels = np.zeros((360, 640), dtype=np.uint32)
    _draw_robot_patch(camera=camera, image_bgr=image, labels=labels, offset_px=(140, 0))

    result = validate_sample_quality(image_bgr=image, labels=labels, **_base_kwargs(camera))

    assert not result.accepted
    assert result.reason == 'projection_mismatch'


def test_validate_sample_rejects_label_without_visible_rgb_robot() -> None:
    camera = _camera()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    labels = np.zeros((360, 640), dtype=np.uint32)
    _draw_robot_patch(camera=camera, image_bgr=image, labels=labels, draw_rgb=False)

    result = validate_sample_quality(image_bgr=image, labels=labels, **_base_kwargs(camera))

    assert not result.accepted
    assert result.reason == 'rgb_robot_not_visible'
