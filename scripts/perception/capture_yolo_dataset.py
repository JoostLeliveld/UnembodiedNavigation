#!/usr/bin/env python3
"""Capture a YOLO-seg dataset from Gazebo semantic segmentation.

The capture path deliberately uses simulator-native semantic labels. Each
accepted sample must pass all checks:

* the robot has been teleported and a settle interval has elapsed;
* fresh RGB and semantic-label frames arrive after that settle interval;
* RGB and label headers are synchronized;
* the semantic mask is visible and not a tiny/truncated sliver;
* the mask lands near the commanded pose under the configured camera model;
* the robot is not substantially occluded by a foreground rack/box (the visible
  silhouette height and ground-contact row must match the projected robot box);
* the RGB crop under the label contains visible robot-colored pixels;
* the RGB frame is not an exact duplicate of an already accepted sample.

The script writes the YOLO dataset plus diagnostics, overlays, a contact sheet,
and a manifest. If a dataset cannot pass the final duplicate/acceptance checks,
it fails loudly instead of producing a plausible-looking but unusable dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/perception', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset_split_utils import assign_splits, build_pose_records, evenly_spaced_yaws
from experiments.core.world_profiles import compute_look_at_from_pose, load_profile
from perception.core.ros_image import image_msg_to_bgr8
from unav_common.camera_model import ObliqueCameraModel


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _stamp_ns(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _sha1_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha1()
    h.update(str(contiguous.shape).encode('ascii'))
    h.update(str(contiguous.dtype).encode('ascii'))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def _read_uint_label_map(msg: Image) -> np.ndarray:
    encoding = str(msg.encoding or '').strip().lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    if height <= 0 or width <= 0:
        raise ValueError(f'Invalid label image shape height={height}, width={width}')

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError(f'Label image has {raw.size} bytes, expected at least {expected}')
    rows = raw[:expected].reshape(height, step)

    if encoding in ('rgb8', 'bgr8', '8uc3'):
        min_step = width * 3
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        pixels = rows[:, :min_step].reshape(height, width, 3)
        return pixels[..., 2].astype(np.uint32)

    if encoding in ('mono8', '8uc1'):
        min_step = width
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        return rows[:, :min_step].reshape(height, width).astype(np.uint32)

    if encoding in ('mono16', '16uc1', '16sc1'):
        min_step = width * 2
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        view = rows[:, :min_step].reshape(height, width, 2)
        return view.view(np.uint16).reshape(height, width).astype(np.uint32)

    if encoding in ('32sc1', '32uc1'):
        min_step = width * 4
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        view = rows[:, :min_step].reshape(height, width, 4)
        dtype = np.int32 if encoding == '32sc1' else np.uint32
        return view.view(dtype).reshape(height, width).astype(np.uint32)

    raise ValueError(
        f'Unsupported segmentation labels encoding {msg.encoding!r}; '
        'expected rgb8/bgr8/mono8/mono16/32sc1/32uc1'
    )


def _polygon_from_mask(mask_u8: np.ndarray, epsilon_ratio: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return None
    epsilon = max(1.0, float(epsilon_ratio) * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2).astype(float)
    if pts.shape[0] < 3:
        return None
    return pts


def _write_seg_label(path: Path, polygon: np.ndarray | None, img_w: int, img_h: int) -> None:
    if polygon is None or len(polygon) < 3:
        path.write_text('', encoding='utf-8')
        return
    values = ['0']
    for x, y in polygon:
        values.append(f'{np.clip(x / float(img_w), 0.0, 1.0):.8f}')
        values.append(f'{np.clip(y / float(img_h), 0.0, 1.0):.8f}')
    path.write_text(' '.join(values) + '\n', encoding='utf-8')


def _mask_bbox(mask_u8: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _mask_bottom(polygon: np.ndarray | None, band_px: float) -> tuple[float, float]:
    if polygon is None or len(polygon) < 3:
        return math.nan, math.nan
    v_bottom = float(np.max(polygon[:, 1]))
    band = polygon[polygon[:, 1] >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(polygon[:, 0])), v_bottom
    return float(np.mean(band[:, 0])), v_bottom


def _border_fraction(mask_u8: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    h, w = mask_u8.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return 1.0
    border = np.zeros_like(mask_u8, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    # Also treat bbox clipped to the image edge as border contact.
    if x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h:
        edge_contact = 1.0
    else:
        edge_contact = 0.0
    mask = mask_u8 > 0
    if not np.any(mask):
        return 1.0
    return max(float(np.count_nonzero(mask & border)) / float(np.count_nonzero(mask)), edge_contact)


def _robot_color_support_bgr(image_bgr: np.ndarray, bbox: tuple[float, float, float, float], pad_px: int = 4) -> float:
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, x0 - int(pad_px))
    y0 = max(0, y0 - int(pad_px))
    x1 = min(w, x1 + int(pad_px))
    y1 = min(h, y1 + int(pad_px))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = image_bgr[y0:y1, x0:x1]
    b = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    r = crop[:, :, 2].astype(int)
    red = (r > 135) & (g < 135) & (b < 145) & ((r - g) > 30) & ((r - b) > 20)
    blue = (b > 130) & (r < 155) & (g < 180) & ((b - r) > 20)
    return float(np.count_nonzero(red | blue)) / float(max(crop.shape[0] * crop.shape[1], 1))


def _project_world_point(camera: ObliqueCameraModel, xyz: np.ndarray) -> np.ndarray | None:
    cam_pt = camera.R @ (np.asarray(xyz, dtype=float) - camera.cam_pos)
    if cam_pt[2] <= 1e-6:
        return None
    pixel_h = camera.K @ cam_pt
    return np.asarray([pixel_h[0] / pixel_h[2], pixel_h[1] / pixel_h[2]], dtype=float)


def _project_robot_bbox(
    camera: ObliqueCameraModel,
    *,
    x: float,
    y: float,
    yaw: float,
    z: float,
    box_length: float,
    box_width: float,
    box_height: float,
) -> tuple[float, float, float, float] | None:
    hx = 0.5 * float(box_length)
    hy = 0.5 * float(box_width)
    local = np.asarray([
        [-hx, -hy, 0.0],
        [-hx, hy, 0.0],
        [hx, -hy, 0.0],
        [hx, hy, 0.0],
        [-hx, -hy, box_height],
        [-hx, hy, box_height],
        [hx, -hy, box_height],
        [hx, hy, box_height],
    ], dtype=float)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    world = (rot @ local.T).T
    world[:, 0] += float(x)
    world[:, 1] += float(y)
    world[:, 2] += float(z)
    pts = []
    for corner in world:
        uv = _project_world_point(camera, corner)
        if uv is None:
            return None
        pts.append(uv)
    arr = np.asarray(pts, dtype=float)
    return (
        float(np.min(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 0])),
        float(np.max(arr[:, 1])),
    )


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {'n': 0, 'min': None, 'median': None, 'mean': None, 'p90': None, 'max': None}
    arr = np.asarray(vals, dtype=float)
    return {
        'n': int(arr.size),
        'min': float(np.min(arr)),
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'p90': float(np.quantile(arr, 0.90)),
        'max': float(np.max(arr)),
    }


@dataclass
class FramePair:
    image_bgr: np.ndarray
    labels: np.ndarray
    image_stamp_ns: int
    label_stamp_ns: int
    image_count: int
    label_count: int
    stamp_delta_s: float
    set_pose_latency_s: float
    pair_wait_s: float


@dataclass
class LabelQuality:
    accepted: bool
    reason: str
    polygon: np.ndarray | None
    raw_mask: np.ndarray
    mask_area_px: float
    mask_bbox: tuple[float, float, float, float] | None
    mask_bbox_w: float
    mask_bbox_h: float
    mask_bottom_u: float
    mask_bottom_v: float
    border_fraction: float
    rgb_robot_color_fraction: float
    expected_bbox: tuple[float, float, float, float] | None
    expected_center_error_px: float
    visible_height_fraction: float
    bottom_occlusion_px: float
    image_sha1: str
    label_sha1: str


def validate_sample_quality(
    *,
    image_bgr: np.ndarray,
    labels: np.ndarray,
    robot_label: int,
    epsilon_ratio: float,
    bottom_band_px: float,
    min_mask_area: float,
    min_mask_bbox_w: float,
    min_mask_bbox_h: float,
    max_mask_border_fraction: float,
    min_rgb_robot_color_fraction: float,
    disable_rgb_color_check: bool,
    camera: ObliqueCameraModel,
    x: float,
    y: float,
    yaw: float,
    robot_z: float,
    box_length: float,
    box_width: float,
    box_height: float,
    max_expected_center_error_px: float,
    min_visible_height_fraction: float,
    max_bottom_occlusion_px: float,
) -> LabelQuality:
    raw_mask = (labels == int(robot_label)).astype(np.uint8) * 255
    mask_area = float(cv2.countNonZero(raw_mask))
    bbox = _mask_bbox(raw_mask)
    polygon = _polygon_from_mask(raw_mask, float(epsilon_ratio)) if mask_area > 0 else None
    mask_bottom_u, mask_bottom_v = _mask_bottom(polygon, float(bottom_band_px))
    image_sha1 = _sha1_array(image_bgr)
    label_sha1 = _sha1_array(raw_mask)
    bbox_w = float(bbox[2] - bbox[0]) if bbox is not None else 0.0
    bbox_h = float(bbox[3] - bbox[1]) if bbox is not None else 0.0
    border_fraction = _border_fraction(raw_mask, bbox) if bbox is not None else 1.0
    rgb_support = _robot_color_support_bgr(image_bgr, bbox) if bbox is not None else 0.0
    expected_bbox = _project_robot_bbox(
        camera,
        x=float(x),
        y=float(y),
        yaw=float(yaw),
        z=float(robot_z),
        box_length=float(box_length),
        box_width=float(box_width),
        box_height=float(box_height),
    )
    expected_center_error = math.nan
    # Occlusion metrics. gz semantic segmentation is rendering-based, so a robot
    # hidden behind a foreground rack/box yields a SHORTER visible silhouette than
    # the full projected bounding prism, and its visible bottom sits ABOVE where the
    # true ground contact projects. Both are measured in image (pixel) space, so they
    # are independent of the world-projection accuracy under investigation.
    #  - visible_height_fraction: visible mask height / expected projected height.
    #    Drops toward 0 as the robot is occluded.
    #  - bottom_occlusion_px: expected contact row - visible mask bottom row.
    #    Positive & large => the ground-contact (box-bottom label) is occluded.
    visible_height_fraction = math.nan
    bottom_occlusion_px = math.nan
    if bbox is not None and expected_bbox is not None:
        cx, cy = _bbox_center(bbox)
        ex, ey = _bbox_center(expected_bbox)
        expected_center_error = float(math.hypot(cx - ex, cy - ey))
        expected_h = float(expected_bbox[3] - expected_bbox[1])
        if expected_h > 1e-6:
            visible_height_fraction = float(bbox_h / expected_h)
        if math.isfinite(mask_bottom_v):
            bottom_occlusion_px = float(expected_bbox[3] - mask_bottom_v)

    reason = ''
    if mask_area < float(min_mask_area):
        reason = 'small_mask'
    elif bbox is None:
        reason = 'empty_mask'
    elif polygon is None:
        reason = 'empty_polygon'
    elif bbox_w < float(min_mask_bbox_w) or bbox_h < float(min_mask_bbox_h):
        reason = 'small_bbox'
    elif border_fraction > float(max_mask_border_fraction):
        reason = 'touches_image_border'
    elif expected_bbox is None:
        reason = 'commanded_pose_not_projectable'
    elif expected_center_error > float(max_expected_center_error_px):
        reason = 'projection_mismatch'
    elif (math.isfinite(visible_height_fraction)
          and visible_height_fraction < float(min_visible_height_fraction)):
        reason = 'occluded_low_visible_height'
    elif (math.isfinite(bottom_occlusion_px)
          and bottom_occlusion_px > float(max_bottom_occlusion_px)):
        reason = 'occluded_bottom_hidden'
    elif (not bool(disable_rgb_color_check)) and rgb_support < float(min_rgb_robot_color_fraction):
        reason = 'rgb_robot_not_visible'

    return LabelQuality(
        accepted=(reason == ''),
        reason=reason,
        polygon=polygon,
        raw_mask=raw_mask,
        mask_area_px=mask_area,
        mask_bbox=bbox,
        mask_bbox_w=bbox_w,
        mask_bbox_h=bbox_h,
        mask_bottom_u=float(mask_bottom_u),
        mask_bottom_v=float(mask_bottom_v),
        border_fraction=float(border_fraction),
        rgb_robot_color_fraction=float(rgb_support),
        expected_bbox=expected_bbox,
        expected_center_error_px=float(expected_center_error),
        visible_height_fraction=float(visible_height_fraction),
        bottom_occlusion_px=float(bottom_occlusion_px),
        image_sha1=image_sha1,
        label_sha1=label_sha1,
    )


class TeleportYoloDatasetCapture(Node):
    def __init__(
        self,
        *,
        world_name: str,
        image_topic: str,
        labels_topic: str,
        robot_z: float,
        settle_s: float,
        image_timeout_s: float,
        sync_slop_s: float,
        min_new_rgb_frames: int,
        min_new_label_frames: int,
        buffer_size: int,
        zero_cmd_topic: str,
    ):
        super().__init__('capture_yolo_dataset')
        self.world_name = str(world_name)
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.image_timeout_s = float(image_timeout_s)
        self.sync_slop_s = float(max(sync_slop_s, 0.0))
        self.min_new_rgb_frames = max(int(min_new_rgb_frames), 1)
        self.min_new_label_frames = max(int(min_new_label_frames), 1)
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)
        self.rgb_count = 0
        self.label_count = 0
        self.image_buffer = deque(maxlen=max(int(buffer_size), 4))
        self.label_buffer = deque(maxlen=max(int(buffer_size), 4))
        self.last_label_error = ''
        self.zero_cmd_pub = self.create_publisher(Twist, str(zero_cmd_topic), 1)
        self.create_subscription(Image, str(image_topic), self._image_cb, 10)
        self.create_subscription(Image, str(labels_topic), self._labels_cb, 10)

    def _image_cb(self, msg: Image) -> None:
        self.rgb_count += 1
        self.image_buffer.append((int(self.rgb_count), _stamp_ns(msg), image_msg_to_bgr8(msg)))

    def _labels_cb(self, msg: Image) -> None:
        try:
            labels = _read_uint_label_map(msg)
        except Exception as exc:
            self.last_label_error = str(exc)
            self.get_logger().error(f'Failed to decode segmentation labels image: {exc}')
            return
        self.label_count += 1
        self.label_buffer.append((int(self.label_count), _stamp_ns(msg), labels))

    def _publish_zero_cmd(self, repetitions: int = 3) -> None:
        msg = Twist()
        for _ in range(max(int(repetitions), 1)):
            self.zero_cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(float(timeout_s), 0.0)
        service_ready = False
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (not service_ready) and self.client.wait_for_service(timeout_sec=0.05):
                service_ready = True
            if service_ready and self.image_buffer and self.label_buffer:
                return
        extra = f' Last label decode error: {self.last_label_error}' if self.last_label_error else ''
        raise RuntimeError(
            f'Timed out waiting for {self.service_name}, RGB image, and segmentation labels. '
            'Launch sim bringup with bridge_segmentation:=true for dataset capture.'
            + extra
        )

    def _set_robot_pose(self, *, x: float, y: float, yaw: float) -> float:
        self._publish_zero_cmd()
        req = SetEntityPose.Request()
        req.entity = Entity(name='turtlebot3')
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(self.robot_z)
        half = 0.5 * float(yaw)
        req.pose.orientation.z = math.sin(half)
        req.pose.orientation.w = math.cos(half)
        future = self.client.call_async(req)
        start = time.monotonic()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if (time.monotonic() - start) > 10.0:
                raise RuntimeError(f'Timed out waiting for {self.service_name} response')
        result = future.result()
        latency_s = float(time.monotonic() - start)
        if result is None or not bool(getattr(result, 'success', False)):
            raise RuntimeError(
                f'Set pose request failed at x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
            )
        self._publish_zero_cmd()
        return latency_s

    def _find_synced_pair(self, before_rgb: int, before_label: int) -> FramePair | None:
        if self.rgb_count < before_rgb + self.min_new_rgb_frames:
            return None
        if self.label_count < before_label + self.min_new_label_frames:
            return None
        image_candidates = [item for item in self.image_buffer if item[0] > before_rgb]
        label_candidates = [item for item in self.label_buffer if item[0] > before_label]
        best = None
        best_key = None
        for label_count, label_stamp, labels in label_candidates:
            for image_count, image_stamp, image in image_candidates:
                delta_s = abs(image_stamp - label_stamp) * 1e-9
                if delta_s > self.sync_slop_s:
                    continue
                key = (min(image_count, label_count), -delta_s)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (image_count, image_stamp, image, label_count, label_stamp, labels, delta_s)
        if best is None:
            return None
        image_count, image_stamp, image, label_count, label_stamp, labels, delta_s = best
        return FramePair(
            image_bgr=image.copy(),
            labels=labels.copy(),
            image_stamp_ns=int(image_stamp),
            label_stamp_ns=int(label_stamp),
            image_count=int(image_count),
            label_count=int(label_count),
            stamp_delta_s=float(delta_s),
            set_pose_latency_s=math.nan,
            pair_wait_s=math.nan,
        )

    def capture_at_pose(self, *, x: float, y: float, yaw: float) -> FramePair:
        set_pose_latency_s = self._set_robot_pose(x=x, y=y, yaw=yaw)

        settle_end = time.monotonic() + max(self.settle_s, 0.0)
        while rclpy.ok() and time.monotonic() < settle_end:
            self._publish_zero_cmd(repetitions=1)
            rclpy.spin_once(self, timeout_sec=0.05)

        # The freshness baseline is after settling, not before teleport. This is
        # what prevents accepting frames rendered while the robot was still being
        # moved to the new pose.
        before_rgb = int(self.rgb_count)
        before_label = int(self.label_count)
        wait_start = time.monotonic()
        deadline = wait_start + max(self.image_timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            pair = self._find_synced_pair(before_rgb, before_label)
            if pair is not None:
                pair.set_pose_latency_s = set_pose_latency_s
                pair.pair_wait_s = float(time.monotonic() - wait_start)
                return pair
        latest_delta = math.nan
        if self.image_buffer and self.label_buffer:
            latest_delta = abs(self.image_buffer[-1][1] - self.label_buffer[-1][1]) * 1e-9
        raise RuntimeError(
            'No fresh synchronized RGB/segmentation pair after teleport settle '
            f'(rgb_count={self.rgb_count}, label_count={self.label_count}, '
            f'latest_delta={latest_delta:.3f}s, required <= {self.sync_slop_s:.3f}s).'
        )


def _in_any_region(x: float, y: float, regions: list[dict], shrink_m: float) -> bool:
    for r in regions:
        if (
            float(r['xmin']) + shrink_m <= x <= float(r['xmax']) - shrink_m
            and float(r['ymin']) + shrink_m <= y <= float(r['ymax']) - shrink_m
        ):
            return True
    return False


def _camera_from_profile(camera_pose: list[float], intrinsics: dict) -> ObliqueCameraModel:
    cam_pos = [float(v) for v in camera_pose[:3]]
    rpy = [float(v) for v in camera_pose[3:6]]
    look_at = compute_look_at_from_pose(cam_pos, *rpy)
    return ObliqueCameraModel(
        cam_pos=cam_pos,
        look_at=look_at,
        img_width=int(intrinsics['img_width']),
        img_height=int(intrinsics['img_height']),
        fov_h_rad=float(intrinsics['fov_h_rad']),
    )


def _draw_overlay(
    image_bgr: np.ndarray,
    quality: LabelQuality,
    *,
    text: str,
) -> np.ndarray:
    out = image_bgr.copy()
    if quality.polygon is not None:
        pts = quality.polygon.astype(np.int32).reshape(-1, 1, 2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0.0)
        cv2.polylines(out, [pts], True, (0, 220, 0), 2)
    if quality.mask_bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in quality.mask_bbox]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
    if quality.expected_bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in quality.expected_bbox]
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
    if math.isfinite(quality.mask_bottom_u) and math.isfinite(quality.mask_bottom_v):
        cv2.circle(out, (int(round(quality.mask_bottom_u)), int(round(quality.mask_bottom_v))), 5, (0, 255, 255), -1)
    cv2.putText(out, text[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _write_contact_sheet(paths: list[Path], output_path: Path, *, cols: int = 5, tile_w: int = 320) -> str:
    if not paths:
        return ''
    imgs = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = tile_w / float(max(w, 1))
        tile_h = max(1, int(round(h * scale)))
        imgs.append(cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA))
    if not imgs:
        return ''
    tile_h = max(img.shape[0] for img in imgs)
    rows = int(math.ceil(len(imgs) / float(cols)))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    sheet[:, :] = (24, 24, 24)
    for idx, img in enumerate(imgs):
        y = (idx // cols) * tile_h
        x = (idx % cols) * tile_w
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return str(output_path)


def _archive_existing_path(path: Path) -> Path:
    archive_root = path.parent / 'archive'
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f'{path.name}_{_timestamp()}'
    shutil.move(str(path), str(target))
    return target


def _fixed_fieldnames() -> list[str]:
    return [
        'sample_index', 'attempt', 'split', 'accepted', 'rejection_reason',
        'image', 'label', 'mask', 'preview',
        'robot_x', 'robot_y', 'robot_yaw',
        'image_stamp_s', 'label_stamp_s', 'stamp_delta_s',
        'set_pose_latency_s', 'settle_s', 'pair_wait_s',
        'image_count', 'label_count',
        'mask_area_px', 'mask_bbox_x0', 'mask_bbox_y0', 'mask_bbox_x1', 'mask_bbox_y1',
        'mask_bbox_w', 'mask_bbox_h', 'mask_bottom_u', 'mask_bottom_v',
        'border_fraction', 'rgb_robot_color_fraction',
        'expected_bbox_x0', 'expected_bbox_y0', 'expected_bbox_x1', 'expected_bbox_y1',
        'expected_center_error_px', 'visible_height_fraction', 'bottom_occlusion_px',
        'image_sha1', 'label_sha1',
    ]


def _diagnostic_row(
    *,
    sample_index: int,
    attempt: int,
    split: str,
    accepted: bool,
    reason: str,
    pair: FramePair | None,
    quality: LabelQuality | None,
    x: float,
    y: float,
    yaw: float,
    settle_s: float,
    image_rel: str = '',
    label_rel: str = '',
    mask_rel: str = '',
    preview_rel: str = '',
) -> dict:
    row = {
        'sample_index': int(sample_index),
        'attempt': int(attempt),
        'split': str(split),
        'accepted': int(bool(accepted)),
        'rejection_reason': str(reason),
        'image': image_rel,
        'label': label_rel,
        'mask': mask_rel,
        'preview': preview_rel,
        'robot_x': float(x),
        'robot_y': float(y),
        'robot_yaw': float(yaw),
        'image_stamp_s': math.nan,
        'label_stamp_s': math.nan,
        'stamp_delta_s': math.nan,
        'set_pose_latency_s': math.nan,
        'settle_s': float(settle_s),
        'pair_wait_s': math.nan,
        'image_count': math.nan,
        'label_count': math.nan,
        'mask_area_px': math.nan,
        'mask_bbox_x0': math.nan,
        'mask_bbox_y0': math.nan,
        'mask_bbox_x1': math.nan,
        'mask_bbox_y1': math.nan,
        'mask_bbox_w': math.nan,
        'mask_bbox_h': math.nan,
        'mask_bottom_u': math.nan,
        'mask_bottom_v': math.nan,
        'border_fraction': math.nan,
        'rgb_robot_color_fraction': math.nan,
        'expected_bbox_x0': math.nan,
        'expected_bbox_y0': math.nan,
        'expected_bbox_x1': math.nan,
        'expected_bbox_y1': math.nan,
        'expected_center_error_px': math.nan,
        'visible_height_fraction': math.nan,
        'bottom_occlusion_px': math.nan,
        'image_sha1': '',
        'label_sha1': '',
    }
    if pair is not None:
        row.update({
            'image_stamp_s': float(pair.image_stamp_ns) * 1e-9,
            'label_stamp_s': float(pair.label_stamp_ns) * 1e-9,
            'stamp_delta_s': float(pair.stamp_delta_s),
            'set_pose_latency_s': float(pair.set_pose_latency_s),
            'pair_wait_s': float(pair.pair_wait_s),
            'image_count': int(pair.image_count),
            'label_count': int(pair.label_count),
        })
    if quality is not None:
        bbox = quality.mask_bbox or (math.nan, math.nan, math.nan, math.nan)
        expected = quality.expected_bbox or (math.nan, math.nan, math.nan, math.nan)
        row.update({
            'mask_area_px': float(quality.mask_area_px),
            'mask_bbox_x0': float(bbox[0]),
            'mask_bbox_y0': float(bbox[1]),
            'mask_bbox_x1': float(bbox[2]),
            'mask_bbox_y1': float(bbox[3]),
            'mask_bbox_w': float(quality.mask_bbox_w),
            'mask_bbox_h': float(quality.mask_bbox_h),
            'mask_bottom_u': float(quality.mask_bottom_u),
            'mask_bottom_v': float(quality.mask_bottom_v),
            'border_fraction': float(quality.border_fraction),
            'rgb_robot_color_fraction': float(quality.rgb_robot_color_fraction),
            'expected_bbox_x0': float(expected[0]),
            'expected_bbox_y0': float(expected[1]),
            'expected_bbox_x1': float(expected[2]),
            'expected_bbox_y1': float(expected[3]),
            'expected_center_error_px': float(quality.expected_center_error_px),
            'visible_height_fraction': float(quality.visible_height_fraction),
            'bottom_occlusion_px': float(quality.bottom_occlusion_px),
            'image_sha1': quality.image_sha1,
            'label_sha1': quality.label_sha1,
        })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture a YOLO-seg dataset from Gazebo semantic segmentation labels.')
    parser.add_argument('--world', default='warehouse_aws.world.sdf')
    parser.add_argument('--world-profiles', default=str((REPO_ROOT / 'src' / 'experiments' / 'config' / 'world_profiles.yaml').resolve()))
    parser.add_argument('--out', default='', help='Output dataset folder; defaults under logs/')
    parser.add_argument('--replace', action='store_true', help='Delete an existing output folder before capture.')
    parser.add_argument('--archive-existing', action='store_true', help='Move an existing output folder under sibling archive/ before capture.')
    parser.add_argument('--sample-nx', type=int, default=16)
    parser.add_argument('--sample-ny', type=int, default=14)
    parser.add_argument('--yaw-samples', type=int, default=8)
    parser.add_argument('--wall-margin', type=float, default=0.65)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--split-mode', choices=('cyclic', 'yaw_bucket', 'spatial_cell', 'spatial_yaw_bucket'), default='spatial_yaw_bucket')
    parser.add_argument('--spatial-block-size', type=int, default=2)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--robot-box-length', type=float, default=0.22)
    parser.add_argument('--robot-box-width', type=float, default=0.22)
    parser.add_argument('--robot-box-height', type=float, default=0.20)
    parser.add_argument('--settle-s', type=float, default=0.80)
    parser.add_argument('--image-timeout-s', type=float, default=8.0)
    parser.add_argument('--sync-slop-ms', type=float, default=60.0)
    parser.add_argument('--min-new-rgb-frames', type=int, default=3)
    parser.add_argument('--min-new-label-frames', type=int, default=1)
    parser.add_argument('--max-sample-attempts', type=int, default=4)
    parser.add_argument('--buffer-size', type=int, default=90)
    parser.add_argument('--preview-count', type=int, default=160)
    parser.add_argument('--rejected-preview-count', type=int, default=80)
    parser.add_argument('--save-masks', action='store_true', help='Save accepted binary masks for provenance.')
    parser.add_argument('--min-mask-area', type=float, default=80.0)
    parser.add_argument('--min-mask-bbox-w', type=float, default=8.0)
    parser.add_argument('--min-mask-bbox-h', type=float, default=8.0)
    parser.add_argument('--max-mask-border-fraction', type=float, default=0.0)
    parser.add_argument('--bottom-band-px', type=float, default=3.0)
    parser.add_argument('--epsilon-ratio', type=float, default=0.010)
    parser.add_argument('--min-rgb-robot-color-fraction', type=float, default=0.015)
    parser.add_argument('--disable-rgb-color-check', action='store_true')
    parser.add_argument('--max-expected-center-error-px', type=float, default=90.0)
    parser.add_argument('--min-visible-height-fraction', type=float, default=0.55,
                        help='Reject occluded samples whose visible mask height is below this '
                             'fraction of the projected robot-box height. Set 0.0 to disable.')
    parser.add_argument('--max-bottom-occlusion-px', type=float, default=20.0,
                        help='Reject samples whose visible mask bottom sits more than this many '
                             'pixels above the projected ground-contact row (bottom occluded by a '
                             'foreground rack/box). Set a large value to disable.')
    parser.add_argument('--max-final-duplicate-fraction', type=float, default=0.02)
    parser.add_argument('--min-accepted-samples', type=int, default=400)
    parser.add_argument('--min-accept-fraction', type=float, default=0.25)
    parser.add_argument('--image-topic', default='/external_camera/image_raw')
    parser.add_argument('--labels-topic', default='/external_camera/segmentation/labels_map')
    parser.add_argument('--zero-cmd-topic', default='/cmd_vel')
    parser.add_argument('--robot-label', type=int, default=23)
    parser.add_argument('--skip-region-filter', action='store_true',
                        help='Disable known_2d_regions traversability filter even when the profile defines regions.')
    parser.add_argument('--region-shrink-m', type=float, default=0.05,
                        help='Shrink each traversable region boundary inward before testing grid points.')
    args = parser.parse_args()

    profile, intrinsics, _world_path, camera_pose = load_profile(str(args.world_profiles), str(args.world))
    camera = _camera_from_profile(camera_pose, intrinsics)
    vis = dict(profile.get('visibility_defaults') or {})
    xmin = float(vis.get('visibility_map_min_x', -3.0)) + float(args.wall_margin)
    xmax = float(vis.get('visibility_map_max_x', 3.0)) - float(args.wall_margin)
    ymin = float(vis.get('visibility_map_min_y', -3.0)) + float(args.wall_margin)
    ymax = float(vis.get('visibility_map_max_y', 3.0)) - float(args.wall_margin)
    if xmax <= xmin or ymax <= ymin:
        raise RuntimeError('Invalid sampling bounds after wall margin')

    xs = np.linspace(xmin, xmax, max(int(args.sample_nx), 1))
    ys = np.linspace(ymin, ymax, max(int(args.sample_ny), 1))
    yaws = np.asarray(evenly_spaced_yaws(max(int(args.yaw_samples), 1)), dtype=float)
    pose_records = build_pose_records(xs, ys, yaws)
    known_regions = list(profile.get('known_2d_regions') or [])
    if known_regions and not args.skip_region_filter:
        traversable = [r for r in known_regions if str(r.get('type', '')) == 'traversable']
        if traversable:
            pose_records = [
                rec for rec in pose_records
                if _in_any_region(float(rec['x']), float(rec['y']), traversable, float(args.region_shrink_m))
            ]
    planned = [(float(item['x']), float(item['y']), float(item['yaw'])) for item in pose_records]
    split_labels = assign_splits(
        pose_records,
        val_fraction=float(args.val_fraction),
        split_mode=str(args.split_mode),
        seed=int(args.split_seed),
        spatial_block_size=int(args.spatial_block_size),
    )

    if str(args.out).strip():
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = (REPO_ROOT / 'logs' / f'warehouse_yolo_dataset_{_timestamp()}').resolve()
    archived_to = ''
    if out_dir.exists():
        if bool(args.archive_existing):
            archived_to = str(_archive_existing_path(out_dir))
        elif bool(args.replace):
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f'Output folder already exists: {out_dir}')

    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=False)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=False)
        if bool(args.save_masks):
            (out_dir / 'masks' / split).mkdir(parents=True, exist_ok=False)
    (out_dir / 'audit' / 'accepted').mkdir(parents=True, exist_ok=True)
    (out_dir / 'audit' / 'rejected').mkdir(parents=True, exist_ok=True)

    diagnostics: list[dict] = []
    accepted_preview_paths: list[Path] = []
    rejected_preview_paths: list[Path] = []
    accepted_hashes: list[str] = []
    seen_image_hashes: set[str] = set()
    rejected_counts: Counter[str] = Counter()
    accepted = 0
    rejected_samples = 0

    rclpy.init()
    node = TeleportYoloDatasetCapture(
        world_name=str(profile['world_name']),
        image_topic=str(args.image_topic),
        labels_topic=str(args.labels_topic),
        robot_z=float(args.robot_z),
        settle_s=float(args.settle_s),
        image_timeout_s=float(args.image_timeout_s),
        sync_slop_s=float(args.sync_slop_ms) / 1000.0,
        min_new_rgb_frames=int(args.min_new_rgb_frames),
        min_new_label_frames=int(args.min_new_label_frames),
        buffer_size=int(args.buffer_size),
        zero_cmd_topic=str(args.zero_cmd_topic),
    )
    try:
        node.wait_for_ready(timeout_s=30.0)
        for sample_index, (x, y, yaw) in enumerate(planned):
            split = str(split_labels[sample_index])
            sample_accepted = False
            last_reason = 'not_attempted'
            for attempt in range(max(int(args.max_sample_attempts), 1)):
                pair: FramePair | None = None
                quality: LabelQuality | None = None
                try:
                    pair = node.capture_at_pose(x=x, y=y, yaw=yaw)
                    if pair.labels.shape[:2] != pair.image_bgr.shape[:2]:
                        raise RuntimeError(
                            f'Segmentation label map shape {pair.labels.shape[:2]} does not match RGB image shape {pair.image_bgr.shape[:2]}'
                        )
                    quality = validate_sample_quality(
                        image_bgr=pair.image_bgr,
                        labels=pair.labels,
                        robot_label=int(args.robot_label),
                        epsilon_ratio=float(args.epsilon_ratio),
                        bottom_band_px=float(args.bottom_band_px),
                        min_mask_area=float(args.min_mask_area),
                        min_mask_bbox_w=float(args.min_mask_bbox_w),
                        min_mask_bbox_h=float(args.min_mask_bbox_h),
                        max_mask_border_fraction=float(args.max_mask_border_fraction),
                        min_rgb_robot_color_fraction=float(args.min_rgb_robot_color_fraction),
                        disable_rgb_color_check=bool(args.disable_rgb_color_check),
                        camera=camera,
                        x=x,
                        y=y,
                        yaw=yaw,
                        robot_z=float(args.robot_z),
                        box_length=float(args.robot_box_length),
                        box_width=float(args.robot_box_width),
                        box_height=float(args.robot_box_height),
                        max_expected_center_error_px=float(args.max_expected_center_error_px),
                        min_visible_height_fraction=float(args.min_visible_height_fraction),
                        max_bottom_occlusion_px=float(args.max_bottom_occlusion_px),
                    )
                    last_reason = quality.reason
                    if quality.accepted and quality.image_sha1 in seen_image_hashes:
                        quality.accepted = False
                        quality.reason = 'duplicate_image'
                        last_reason = quality.reason
                    if quality.accepted:
                        stem = f'sample_{sample_index:06d}'
                        image_path = out_dir / 'images' / split / f'{stem}.png'
                        label_path = out_dir / 'labels' / split / f'{stem}.txt'
                        cv2.imwrite(str(image_path), pair.image_bgr)
                        _write_seg_label(label_path, quality.polygon, pair.image_bgr.shape[1], pair.image_bgr.shape[0])
                        mask_rel = ''
                        if bool(args.save_masks):
                            mask_path = out_dir / 'masks' / split / f'{stem}.png'
                            cv2.imwrite(str(mask_path), quality.raw_mask)
                            mask_rel = str(mask_path.relative_to(out_dir))
                        preview_rel = ''
                        if len(accepted_preview_paths) < max(int(args.preview_count), 0):
                            overlay = _draw_overlay(
                                pair.image_bgr,
                                quality,
                                text=(
                                    f'ok idx={sample_index} try={attempt} area={quality.mask_area_px:.0f} '
                                    f'err={quality.expected_center_error_px:.1f}px vh={quality.visible_height_fraction:.2f} '
                                    f'bgap={quality.bottom_occlusion_px:.0f}px rgb={quality.rgb_robot_color_fraction:.3f}'
                                ),
                            )
                            preview_path = out_dir / 'audit' / 'accepted' / f'{stem}.jpg'
                            cv2.imwrite(str(preview_path), overlay)
                            accepted_preview_paths.append(preview_path)
                            preview_rel = str(preview_path.relative_to(out_dir))
                        diagnostics.append(_diagnostic_row(
                            sample_index=sample_index,
                            attempt=attempt,
                            split=split,
                            accepted=True,
                            reason='',
                            pair=pair,
                            quality=quality,
                            x=x,
                            y=y,
                            yaw=yaw,
                            settle_s=float(args.settle_s),
                            image_rel=str(image_path.relative_to(out_dir)),
                            label_rel=str(label_path.relative_to(out_dir)),
                            mask_rel=mask_rel,
                            preview_rel=preview_rel,
                        ))
                        accepted += 1
                        accepted_hashes.append(quality.image_sha1)
                        seen_image_hashes.add(quality.image_sha1)
                        sample_accepted = True
                        break
                    rejected_counts[quality.reason] += 1
                    if len(rejected_preview_paths) < max(int(args.rejected_preview_count), 0):
                        overlay = _draw_overlay(
                            pair.image_bgr,
                            quality,
                            text=(
                                f'reject={quality.reason} idx={sample_index} try={attempt} '
                                f'area={quality.mask_area_px:.0f} err={quality.expected_center_error_px:.1f}px '
                                f'vh={quality.visible_height_fraction:.2f} bgap={quality.bottom_occlusion_px:.0f}px'
                            ),
                        )
                        preview_path = out_dir / 'audit' / 'rejected' / f'sample_{sample_index:06d}_try{attempt}.jpg'
                        cv2.imwrite(str(preview_path), overlay)
                        rejected_preview_paths.append(preview_path)
                    diagnostics.append(_diagnostic_row(
                        sample_index=sample_index,
                        attempt=attempt,
                        split=split,
                        accepted=False,
                        reason=quality.reason,
                        pair=pair,
                        quality=quality,
                        x=x,
                        y=y,
                        yaw=yaw,
                        settle_s=float(args.settle_s),
                    ))
                except Exception as exc:
                    last_reason = f'capture_error:{exc}'
                    rejected_counts['capture_error'] += 1
                    diagnostics.append(_diagnostic_row(
                        sample_index=sample_index,
                        attempt=attempt,
                        split=split,
                        accepted=False,
                        reason=last_reason,
                        pair=pair,
                        quality=quality,
                        x=x,
                        y=y,
                        yaw=yaw,
                        settle_s=float(args.settle_s),
                    ))
            if not sample_accepted:
                rejected_samples += 1
                if not last_reason:
                    rejected_counts['unknown'] += 1
            if (sample_index + 1) % 50 == 0:
                print(f'captured {sample_index + 1}/{len(planned)} planned, accepted={accepted}, rejected_samples={rejected_samples}', flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    data_yaml = {
        'path': str(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'robot'},
        'task': 'segment',
    }
    (out_dir / 'data.yaml').write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding='utf-8')

    with (out_dir / 'label_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=_fixed_fieldnames())
        writer.writeheader()
        writer.writerows(diagnostics)

    accepted_rows = [r for r in diagnostics if int(r['accepted']) == 1]
    duplicate_images = sum(count for count in Counter(accepted_hashes).values() if count > 1)
    duplicate_fraction = duplicate_images / float(max(len(accepted_hashes), 1))
    split_counts = Counter(str(r['split']) for r in accepted_rows)
    accepted_fraction = accepted / float(max(len(planned), 1))
    accepted_sheet = _write_contact_sheet(accepted_preview_paths, out_dir / 'audit' / 'accepted_contact_sheet.jpg')
    rejected_sheet = _write_contact_sheet(rejected_preview_paths, out_dir / 'audit' / 'rejected_contact_sheet.jpg')
    audit = {
        'planned_samples': int(len(planned)),
        'accepted_samples': int(accepted),
        'rejected_planned_samples': int(rejected_samples),
        'accepted_fraction': float(accepted_fraction),
        'split_counts': dict(split_counts),
        'rejection_counts_by_attempt': dict(rejected_counts),
        'duplicate_accepted_images': int(duplicate_images),
        'duplicate_accepted_fraction': float(duplicate_fraction),
        'stamp_delta_s': _stats(float(r['stamp_delta_s']) for r in accepted_rows),
        'set_pose_latency_s': _stats(float(r['set_pose_latency_s']) for r in accepted_rows),
        'pair_wait_s': _stats(float(r['pair_wait_s']) for r in accepted_rows),
        'mask_area_px': _stats(float(r['mask_area_px']) for r in accepted_rows),
        'expected_center_error_px': _stats(float(r['expected_center_error_px']) for r in accepted_rows),
        'visible_height_fraction': _stats(float(r['visible_height_fraction']) for r in accepted_rows),
        'bottom_occlusion_px': _stats(float(r['bottom_occlusion_px']) for r in accepted_rows),
        'rgb_robot_color_fraction': _stats(float(r['rgb_robot_color_fraction']) for r in accepted_rows),
        'accepted_contact_sheet': accepted_sheet,
        'rejected_contact_sheet': rejected_sheet,
    }
    manifest = {
        'world': str(args.world),
        'source': 'gazebo_semantic_segmentation',
        'archived_existing_to': archived_to,
        'robot_label': int(args.robot_label),
        'camera_pose_xyz_rpy': [float(v) for v in camera_pose],
        'camera_intrinsics': dict(intrinsics),
        'sample_nx': int(args.sample_nx),
        'sample_ny': int(args.sample_ny),
        'yaw_samples': int(args.yaw_samples),
        'yaw_values_rad': [float(v) for v in yaws.tolist()],
        'split_mode': str(args.split_mode),
        'split_seed': int(args.split_seed),
        'spatial_block_size': int(args.spatial_block_size),
        'capture_thresholds': {
            'settle_s': float(args.settle_s),
            'image_timeout_s': float(args.image_timeout_s),
            'sync_slop_ms': float(args.sync_slop_ms),
            'min_new_rgb_frames': int(args.min_new_rgb_frames),
            'min_new_label_frames': int(args.min_new_label_frames),
            'max_sample_attempts': int(args.max_sample_attempts),
            'min_mask_area': float(args.min_mask_area),
            'min_mask_bbox_w': float(args.min_mask_bbox_w),
            'min_mask_bbox_h': float(args.min_mask_bbox_h),
            'max_mask_border_fraction': float(args.max_mask_border_fraction),
            'min_rgb_robot_color_fraction': float(args.min_rgb_robot_color_fraction),
            'max_expected_center_error_px': float(args.max_expected_center_error_px),
            'min_visible_height_fraction': float(args.min_visible_height_fraction),
            'max_bottom_occlusion_px': float(args.max_bottom_occlusion_px),
            'max_final_duplicate_fraction': float(args.max_final_duplicate_fraction),
            'min_accepted_samples': int(args.min_accepted_samples),
            'min_accept_fraction': float(args.min_accept_fraction),
        },
        'audit': audit,
        'timestamp': _timestamp(),
        'notes': [
            'Semantic segmentation is used only for offline training labels.',
            'The runtime detector still consumes raw RGB images.',
            'Freshness is measured after the teleport settle interval, so pre-settle frames cannot be accepted.',
            'Accepted samples require synchronized RGB and semantic labels plus projection, visibility, and duplicate checks.',
            'Rejected samples are not written as YOLO training negatives by default.',
        ],
    }
    (out_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    failures = []
    if accepted < int(args.min_accepted_samples):
        failures.append(f'accepted {accepted} < min_accepted_samples {int(args.min_accepted_samples)}')
    if accepted_fraction < float(args.min_accept_fraction):
        failures.append(f'accepted_fraction {accepted_fraction:.3f} < {float(args.min_accept_fraction):.3f}')
    if duplicate_fraction > float(args.max_final_duplicate_fraction):
        failures.append(f'duplicate_accepted_fraction {duplicate_fraction:.3f} > {float(args.max_final_duplicate_fraction):.3f}')
    if split_counts.get('train', 0) <= 0 or split_counts.get('val', 0) <= 0:
        failures.append(f'missing train/val accepted split: {dict(split_counts)}')
    if failures:
        raise RuntimeError('Dataset failed quality gates: ' + '; '.join(failures))

    print(f'Wrote simulator-segmentation YOLO dataset to {out_dir}')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
