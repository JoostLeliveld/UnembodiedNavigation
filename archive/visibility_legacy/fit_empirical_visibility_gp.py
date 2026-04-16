#!/usr/bin/env python3
"""Collect driving-based visibility data and fit a scalar GP visibility field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import Bool, Float64MultiArray, String

REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in ('src/experiments', 'src/perception', 'src/planning', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.core.visibility_capture import (
    VISIBILITY_CAPTURE_CONFIG_TOPIC,
    VISIBILITY_CAPTURE_DONE_TOPIC,
)
from experiments.core.world_profiles import load_profile, parse_camera_pose_from_world
from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from planning.core.gp_visibility_helpers import (
    SimpleRBFGP,
    clip_prob as _clip_prob,
    logit as _logit,
    sigmoid as _sigmoid,
)


@dataclass
class CaptureSample:
    sample_idx: int
    stamp: float
    state_x: float
    state_y: float
    cov_x: float
    cov_y: float
    state_age_s: float
    state_is_fresh: int
    detected: int
    usable_label: int
    u_mid: float
    v_mid: float
    yaw_est: float
    red_area_px: float
    border_margin_px: float
    normalized_blob_area: float = math.nan
    yolo_score: float = math.nan
    confidence_logit: float = math.nan
    localization_error_m: float = math.nan
    localization_quality_soft: float = math.nan


def _default_arg_path(relative: str) -> str:
    return str((Path(__file__).resolve().parents[1] / relative).resolve())


def _safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan
    mask = np.isfinite(arr)
    if not np.any(mask):
        return math.nan
    return float(np.mean(arr[mask]))


def _safe_nanstd(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan
    mask = np.isfinite(arr)
    if not np.any(mask):
        return math.nan
    return float(np.std(arr[mask]))


def _fallback_world_path(repo_root: Path, world: str) -> Path | None:
    candidates = [
        repo_root / 'src' / 'sim' / 'gazebo_worlds' / 'worlds' / world,
        repo_root / 'install' / 'sim' / 'share' / 'sim' / 'gazebo_worlds' / 'worlds' / world,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_world_profile(world_profiles_path: Path, world: str, *, world_path_hint: str = ''):
    try:
        profile, _intrinsics, world_path, camera_pose = load_profile(str(world_profiles_path), world)
        vis = dict(profile.get('visibility_defaults') or {})
        return profile, vis, Path(world_path), camera_pose
    except Exception:
        data = yaml.safe_load(world_profiles_path.read_text(encoding='utf-8')) or {}
        profiles = dict(data.get('worlds') or {})
        if world not in profiles:
            raise
        profile = dict(profiles[world] or {})
        vis = dict(profile.get('visibility_defaults') or {})
        raw_hint = Path(str(world_path_hint or '').strip()).expanduser()
        if str(raw_hint) and raw_hint.is_file():
            world_path = raw_hint.resolve()
        else:
            fallback = _fallback_world_path(REPO_ROOT, world)
            if fallback is None:
                raise
            world_path = fallback
        camera_pose = parse_camera_pose_from_world(str(world_path))
        return profile, vis, world_path, camera_pose


def _parse_float(row: dict, key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    try:
        if value in (None, ''):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        if value in (None, ''):
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _load_samples_from_capture_dir(capture_dir: Path, usable_area_px_min: float):
    raw_csv = capture_dir / 'raw_detection_samples.csv'
    config_json = capture_dir / 'capture_config.json'
    if not raw_csv.is_file():
        raise RuntimeError(f'Capture directory does not contain raw_detection_samples.csv: {capture_dir}')

    capture_config = {}
    if config_json.is_file():
        try:
            capture_config = json.loads(config_json.read_text(encoding='utf-8'))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'Failed to read capture config from {config_json}: {exc}') from exc

    samples: List[CaptureSample] = []
    with raw_csv.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        driving_format = 'state_x' in fieldnames and 'state_y' in fieldnames
        teleport_format = 'x' in fieldnames and 'y' in fieldnames
        if not driving_format and not teleport_format:
            raise RuntimeError(
                f'Unsupported raw detection sample format in {raw_csv}; expected driving or legacy teleport columns.'
            )

        for idx, row in enumerate(reader):
            if driving_format:
                detected = _parse_int(row, 'detected', 0)
                red_area_px = _parse_float(row, 'red_area_px', math.nan)
                usable_label = _parse_int(
                    row,
                    'usable_label',
                    int(detected and math.isfinite(red_area_px) and red_area_px >= usable_area_px_min),
                )
                samples.append(CaptureSample(
                    sample_idx=_parse_int(row, 'sample_idx', idx),
                    stamp=_parse_float(row, 'stamp', float(idx)),
                    state_x=_parse_float(row, 'state_x'),
                    state_y=_parse_float(row, 'state_y'),
                    cov_x=_parse_float(row, 'cov_x', math.nan),
                    cov_y=_parse_float(row, 'cov_y', math.nan),
                    state_age_s=_parse_float(row, 'state_age_s', 0.0),
                    state_is_fresh=_parse_int(row, 'state_is_fresh', 1),
                    detected=detected,
                    usable_label=usable_label,
                    u_mid=_parse_float(row, 'u_mid', math.nan),
                    v_mid=_parse_float(row, 'v_mid', math.nan),
                    yaw_est=_parse_float(row, 'yaw_est', math.nan),
                    red_area_px=red_area_px,
                    border_margin_px=_parse_float(row, 'border_margin_px', math.nan),
                    normalized_blob_area=_parse_float(row, 'normalized_blob_area', math.nan),
                    yolo_score=_parse_float(row, 'yolo_score', math.nan),
                    confidence_logit=_parse_float(row, 'confidence_logit', math.nan),
                ))
            else:
                red_area_px = _parse_float(row, 'mean_red_area_px', math.nan)
                detected = _parse_int(
                    row,
                    'detected_label',
                    int(_parse_float(row, 'detected_rate', 0.0) >= 0.5),
                )
                usable_label = int(detected and math.isfinite(red_area_px) and red_area_px >= usable_area_px_min)
                samples.append(CaptureSample(
                    sample_idx=_parse_int(row, 'sample_idx', idx),
                    stamp=float(idx),
                    state_x=_parse_float(row, 'x'),
                    state_y=_parse_float(row, 'y'),
                    cov_x=math.nan,
                    cov_y=math.nan,
                    state_age_s=0.0,
                    state_is_fresh=1,
                    detected=detected,
                    usable_label=usable_label,
                    u_mid=_parse_float(row, 'mean_u', math.nan),
                    v_mid=_parse_float(row, 'mean_v', math.nan),
                    yaw_est=_parse_float(row, 'yaw', math.nan),
                    red_area_px=red_area_px,
                    border_margin_px=_parse_float(row, 'mean_border_margin_px', math.nan),
                    yolo_score=_parse_float(row, 'mean_yolo_score', math.nan),
                    confidence_logit=_parse_float(row, 'mean_confidence_logit', math.nan),
                ))

    return samples, capture_config


class DrivingVisibilityCapture(Node):
    def __init__(self, *, usable_area_px_min: float, max_state_age_s: float):
        super().__init__('driving_visibility_capture')
        self.usable_area_px_min = float(usable_area_px_min)
        self.max_state_age_s = float(max_state_age_s)

        self.samples: List[CaptureSample] = []
        self._latest_state = None
        self._done = False
        self._sweep_config = None
        self._skipped_no_state = 0
        self._stale_state_rows = 0

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_subscription(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._diag_cb, 10)
        self.create_subscription(Bool, VISIBILITY_CAPTURE_DONE_TOPIC, self._done_cb, latched_qos)
        self.create_subscription(String, VISIBILITY_CAPTURE_CONFIG_TOPIC, self._config_cb, latched_qos)

    @property
    def skipped_no_state(self) -> int:
        return int(self._skipped_no_state)

    @property
    def stale_state_rows(self) -> int:
        return int(self._stale_state_rows)

    @property
    def sweep_config(self) -> Dict[str, object] | None:
        return dict(self._sweep_config) if self._sweep_config is not None else None

    def _state_cb(self, msg: PoseWithCovarianceStamped) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        cov = list(msg.pose.covariance)
        self._latest_state = {
            'stamp': stamp,
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'cov_x': float(cov[0]) if len(cov) > 0 else math.nan,
            'cov_y': float(cov[7]) if len(cov) > 7 else math.nan,
        }

    def _done_cb(self, msg: Bool) -> None:
        self._done = bool(msg.data)

    def _config_cb(self, msg: String) -> None:
        try:
            self._sweep_config = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'Ignoring malformed sweep config message: {exc}')

    def _diag_cb(self, msg: Float64MultiArray) -> None:
        diag = diagnostics_from_message(msg)
        if self._latest_state is None:
            self._skipped_no_state += 1
            return

        stamp = float(diag.get('stamp', math.nan))
        if not math.isfinite(stamp):
            self._skipped_no_state += 1
            return

        state_age_s = abs(stamp - float(self._latest_state['stamp']))
        state_is_fresh = int(state_age_s <= self.max_state_age_s)
        if not state_is_fresh:
            self._stale_state_rows += 1

        detected = bool(diag.get('detected', False))
        red_area_px = float(diag.get('red_area_px', math.nan))
        yolo_score = float(diag.get('yolo_score', math.nan))
        confidence_logit = float(diag.get('confidence_logit', math.nan))
        usable_label = int(detected and math.isfinite(red_area_px) and red_area_px >= self.usable_area_px_min)

        self.samples.append(CaptureSample(
            sample_idx=len(self.samples),
            stamp=stamp,
            state_x=float(self._latest_state['x']),
            state_y=float(self._latest_state['y']),
            cov_x=float(self._latest_state['cov_x']),
            cov_y=float(self._latest_state['cov_y']),
            state_age_s=float(state_age_s),
            state_is_fresh=state_is_fresh,
            detected=int(detected),
            usable_label=usable_label,
            u_mid=float(diag.get('u_mid', math.nan)),
            v_mid=float(diag.get('v_mid', math.nan)),
            yaw_est=float(diag.get('yaw_est', math.nan)),
            red_area_px=red_area_px,
            border_margin_px=float(diag.get('border_margin_px', math.nan)),
            yolo_score=yolo_score,
            confidence_logit=confidence_logit,
        ))

        if len(self.samples) % 100 == 0:
            usable_rate = float(np.mean([s.usable_label for s in self.samples]))
            self.get_logger().info(
                f'Captured {len(self.samples)} diagnostic samples '
                f'(usable_rate={usable_rate:.3f}, stale_rows={self._stale_state_rows})'
            )

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._latest_state is not None and self._sweep_config is not None:
                return
        raise RuntimeError(
            'Timed out waiting for /state/bev and /visibility_capture/config_json. '
            'Start the driving capture launch first, for example:\n'
            '  ros2 launch experiments warehouse_visibility_capture.launch.py world:=warehouse_occ_light.world.sdf\n'
            'or reuse an existing capture with:\n'
            '  python3 scripts/fit_empirical_visibility_gp.py --from-capture-dir <capture_dir>'
        )

    def run_until_complete(self, max_capture_s: float, post_done_grace_s: float) -> None:
        start = time.monotonic()
        done_seen_at = None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.monotonic()
            if max_capture_s > 0.0 and (now - start) > max_capture_s:
                self.get_logger().warn('Capture reached max_capture_s before completion; stopping collection')
                return
            if self._done:
                if done_seen_at is None:
                    done_seen_at = now
                elif (now - done_seen_at) >= max(post_done_grace_s, 0.0):
                    return


class TeleportVisibilityCapture(Node):
    def __init__(
        self,
        *,
        world_name: str,
        usable_area_px_min: float,
        yaw_rad: float,
        robot_z: float,
        settle_s: float,
        diag_timeout_s: float,
        min_diag_messages: int,
        service_timeout_s: float = 10.0,
    ):
        super().__init__('teleport_visibility_capture')
        self.world_name = str(world_name)
        self.usable_area_px_min = float(usable_area_px_min)
        self.yaw_rad = float(yaw_rad)
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.diag_timeout_s = float(diag_timeout_s)
        self.min_diag_messages = int(max(min_diag_messages, 1))
        self.service_timeout_s = float(service_timeout_s)

        self._diag_buffer = []
        self._seen_any_diag = False
        self._collecting = False
        self.create_subscription(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._diag_cb, 10)

        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)

    def _diag_cb(self, msg: Float64MultiArray) -> None:
        try:
            diag = diagnostics_from_message(msg)
        except (TypeError, ValueError, KeyError):
            return
        self._seen_any_diag = True
        if self._collecting:
            self._diag_buffer.append(diag)

    @staticmethod
    def _yaw_to_quaternion(yaw: float):
        half = 0.5 * float(yaw)
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(timeout_s, 0.0)
        service_ready = False
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (not service_ready) and self.client.wait_for_service(timeout_sec=0.05):
                service_ready = True
            if service_ready and self._seen_any_diag:
                return
        missing = []
        if not service_ready:
            missing.append(self.service_name)
        if not self._seen_any_diag:
            missing.append(DETECTION_DIAGNOSTICS_TOPIC)
        missing_text = ', '.join(missing) if missing else 'required capture interfaces'
        raise RuntimeError(
            f'Timed out waiting for {missing_text}. '
            'Start the capture launch first, for example:\n'
            '  ros2 launch experiments warehouse_visibility_capture.launch.py world:=warehouse_occ_light.world.sdf'
        )

    def _set_robot_pose(self, x: float, y: float) -> None:
        req = SetEntityPose.Request()
        req.entity = Entity(name='turtlebot3')
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(self.robot_z)
        qx, qy, qz, qw = self._yaw_to_quaternion(self.yaw_rad)
        req.pose.orientation.x = qx
        req.pose.orientation.y = qy
        req.pose.orientation.z = qz
        req.pose.orientation.w = qw

        future = self.client.call_async(req)
        start = time.monotonic()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.service_timeout_s > 0.0 and (time.monotonic() - start) > self.service_timeout_s:
                raise RuntimeError(f'Timed out waiting for {self.service_name} response')
        if not future.done():
            raise RuntimeError(f'Set pose request to {self.service_name} did not complete')
        result = future.result()
        if result is None or not bool(getattr(result, 'success', False)):
            raise RuntimeError(
                f'Set pose request failed for ({x:.3f}, {y:.3f}) via {self.service_name}'
            )

    def _capture_at_pose(self, x: float, y: float, sample_idx: int) -> CaptureSample:
        self._set_robot_pose(x, y)

        settle_end = time.monotonic() + max(self.settle_s, 0.0)
        self._collecting = False
        self._diag_buffer = []
        while rclpy.ok() and time.monotonic() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)

        self._collecting = True
        self._diag_buffer = []
        deadline = time.monotonic() + max(self.diag_timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if len(self._diag_buffer) >= self.min_diag_messages:
                break
        self._collecting = False

        rows = list(self._diag_buffer)
        if not rows:
            self.get_logger().warn(
                f'No detection diagnostics collected at pose ({x:.3f}, {y:.3f}); recording a miss'
            )
            return CaptureSample(
                sample_idx=sample_idx,
                stamp=float(time.time()),
                state_x=float(x),
                state_y=float(y),
                cov_x=math.nan,
                cov_y=math.nan,
                state_age_s=0.0,
                state_is_fresh=1,
                detected=0,
                usable_label=0,
                u_mid=math.nan,
                v_mid=math.nan,
                yaw_est=math.nan,
                red_area_px=math.nan,
                border_margin_px=math.nan,
                yolo_score=0.0,
                confidence_logit=math.nan,
            )

        detected_flags = [1.0 if bool(r.get('detected', False)) else 0.0 for r in rows]
        mean_detected = float(np.mean(detected_flags))
        detected = int(mean_detected >= 0.5)
        red_area_px = _safe_nanmean([float(r.get('red_area_px', math.nan)) for r in rows])
        yolo_score = _safe_nanmean([float(r.get('yolo_score', math.nan)) for r in rows])
        if not math.isfinite(yolo_score):
            yolo_score = 0.0 if not detected else math.nan
        confidence_logit = _safe_nanmean([float(r.get('confidence_logit', math.nan)) for r in rows])
        usable_label = int(detected and math.isfinite(red_area_px) and red_area_px >= self.usable_area_px_min)
        stamp = _safe_nanmean([float(r.get('stamp', math.nan)) for r in rows])
        if not math.isfinite(stamp):
            stamp = float(time.time())

        return CaptureSample(
            sample_idx=sample_idx,
            stamp=stamp,
            state_x=float(x),
            state_y=float(y),
            cov_x=math.nan,
            cov_y=math.nan,
            state_age_s=0.0,
            state_is_fresh=1,
            detected=detected,
            usable_label=usable_label,
            u_mid=_safe_nanmean([float(r.get('u_mid', math.nan)) for r in rows]),
            v_mid=_safe_nanmean([float(r.get('v_mid', math.nan)) for r in rows]),
            yaw_est=_safe_nanmean([float(r.get('yaw_est', math.nan)) for r in rows]),
            red_area_px=red_area_px,
            border_margin_px=_safe_nanmean([float(r.get('border_margin_px', math.nan)) for r in rows]),
            yolo_score=yolo_score,
            confidence_logit=confidence_logit,
        )


def _compute_blob_area_ref(samples: List[CaptureSample], percentile: float, usable_area_px_min: float) -> float:
    areas = np.asarray(
        [s.red_area_px for s in samples if math.isfinite(s.red_area_px) and s.red_area_px > 0.0],
        dtype=float,
    )
    if areas.size == 0:
        return float(max(usable_area_px_min, 1.0))
    ref = float(np.percentile(areas, np.clip(percentile, 1.0, 100.0)))
    return float(max(ref, usable_area_px_min, 1.0))


def _inject_normalized_blob_area(samples: List[CaptureSample], ref_area_px: float) -> None:
    denom = float(max(ref_area_px, 1.0))
    for sample in samples:
        if math.isfinite(sample.red_area_px) and sample.red_area_px > 0.0:
            sample.normalized_blob_area = float(np.clip(sample.red_area_px / denom, 0.0, 1.0))
        else:
            sample.normalized_blob_area = 0.0


def _grid_vectors(vis: Dict[str, object]):
    xmin = float(vis.get('visibility_map_min_x', -6.0))
    xmax = float(vis.get('visibility_map_max_x', 6.0))
    ymin = float(vis.get('visibility_map_min_y', -6.0))
    ymax = float(vis.get('visibility_map_max_y', 6.0))
    nx = int(vis.get('visibility_map_nx', 160))
    ny = int(vis.get('visibility_map_ny', 160))
    xs = np.linspace(xmin, xmax, max(nx, 2))
    ys = np.linspace(ymin, ymax, max(ny, 2))
    return xs, ys


def _sample_positions(vis: Dict[str, object], *, sample_nx: int, sample_ny: int, wall_margin_m: float, sample_limit: int = 0):
    xmin = float(vis.get('visibility_map_min_x', -6.0)) + float(max(wall_margin_m, 0.0))
    xmax = float(vis.get('visibility_map_max_x', 6.0)) - float(max(wall_margin_m, 0.0))
    ymin = float(vis.get('visibility_map_min_y', -6.0)) + float(max(wall_margin_m, 0.0))
    ymax = float(vis.get('visibility_map_max_y', 6.0)) - float(max(wall_margin_m, 0.0))
    if not (xmin < xmax and ymin < ymax):
        raise RuntimeError('Invalid sample bounds after applying wall_margin_m')

    xs = np.linspace(xmin, xmax, max(int(sample_nx), 1))
    ys = np.linspace(ymin, ymax, max(int(sample_ny), 1))
    positions = []
    for row_idx, y in enumerate(ys):
        x_iter = xs if (row_idx % 2 == 0) else xs[::-1]
        for x in x_iter:
            positions.append((float(x), float(y)))
    if int(sample_limit) > 0:
        positions = positions[: int(sample_limit)]
    return positions


def _nearest_index(value: float, axis: np.ndarray) -> int:
    if axis.size <= 1:
        return 0
    step = float(axis[1] - axis[0])
    idx = int(round((value - float(axis[0])) / step))
    return int(np.clip(idx, 0, axis.size - 1))


def _target_spec(target_mode: str) -> dict:
    mode = str(target_mode or '').strip().lower()
    if mode == 'binary_usable_detection':
        return {
            'mode': mode,
            'field': 'usable_rate_mean',
            'label': 'usable detection rate',
            'title': 'Binary usable-detection GP',
            'rule': 'detected && red_area_px >= usable_area_px_min',
        }
    if mode == 'normalized_blob_area':
        return {
            'mode': mode,
            'field': 'blob_area_target_mean',
            'label': 'normalized blob-area score',
            'title': 'Blob-area GP',
            'rule': 'clip(red_area_px / blob_area_ref_px, 0, 1)',
        }
    if mode == 'yolo_soft_score':
        return {
            'mode': mode,
            'field': 'yolo_score_target_mean',
            'label': 'YOLO confidence-derived observability score',
            'title': 'YOLO soft-score GP',
            'rule': 'yolo_score if finite else 0',
        }
    if mode == 'localization_quality':
        return {
            'mode': mode,
            'field': 'localization_quality_target_mean',
            'label': 'localization quality (from homography + ground-truth error)',
            'title': 'Localization-quality GP',
            'rule': 'exp(-(error_m ^ 2) / sigma_m ^ 2)',
        }
    raise ValueError(
        f"Unsupported target mode {target_mode!r}; expected 'normalized_blob_area', "
        "'binary_usable_detection', 'yolo_soft_score', or 'localization_quality'."
    )


def _aggregate_samples(samples: List[CaptureSample], vis: Dict[str, object], *, target_mode: str):
    xs, ys = _grid_vectors(vis)
    grouped: Dict[tuple[int, int], List[CaptureSample]] = defaultdict(list)
    for sample in samples:
        if not sample.state_is_fresh:
            continue
        if not (math.isfinite(sample.state_x) and math.isfinite(sample.state_y)):
            continue
        ix = _nearest_index(sample.state_x, xs)
        iy = _nearest_index(sample.state_y, ys)
        grouped[(ix, iy)].append(sample)

    agg_rows = []
    for (ix, iy), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        detected_flags = [float(s.detected) for s in group]
        usable_flags = [float(s.usable_label) for s in group]
        blob_area_scores = [float(s.normalized_blob_area) for s in group]
        yolo_scores = [
            float(np.clip(s.yolo_score, 0.0, 1.0)) if math.isfinite(float(s.yolo_score)) else 0.0
            for s in group
        ]
        locqual_scores = [float(s.localization_quality_soft) for s in group]
        agg_rows.append({
            'ix': int(ix),
            'iy': int(iy),
            'x_center': float(xs[ix]),
            'y_center': float(ys[iy]),
            'n_samples': int(len(group)),
            'n_detected': int(sum(detected_flags)),
            'n_usable': int(sum(usable_flags)),
            'detected_rate_mean': float(np.mean(detected_flags)),
            'detected_rate_std': float(np.std(detected_flags)),
            'usable_rate_mean': float(np.mean(usable_flags)),
            'usable_rate_std': float(np.std(usable_flags)),
            'blob_area_target_mean': _safe_nanmean(blob_area_scores),
            'blob_area_target_std': _safe_nanstd(blob_area_scores),
            'yolo_score_target_mean': float(np.mean(yolo_scores)),
            'yolo_score_target_std': float(np.std(yolo_scores)),
            'localization_quality_target_mean': _safe_nanmean(locqual_scores),
            'localization_quality_target_std': _safe_nanstd(locqual_scores),
            'mean_red_area_px': _safe_nanmean([s.red_area_px for s in group]),
            'mean_yolo_score': float(np.mean(yolo_scores)),
            'mean_confidence_logit': _safe_nanmean([s.confidence_logit for s in group]),
            'mean_border_margin_px': _safe_nanmean([s.border_margin_px for s in group]),
            'mean_blob_area_norm': _safe_nanmean([s.normalized_blob_area for s in group]),
            'mean_state_age_s': _safe_nanmean([s.state_age_s for s in group]),
            'mean_cov_x': _safe_nanmean([s.cov_x for s in group]),
            'mean_cov_y': _safe_nanmean([s.cov_y for s in group]),
        })

    spec = _target_spec(target_mode)
    target_field = spec['field']
    for row in agg_rows:
        row['target_mode'] = spec['mode']
        row['target_mean'] = float(row[target_field])
        std_field = target_field.replace('_mean', '_std')
        row['target_std'] = float(row.get(std_field, math.nan))
    return agg_rows


def _fit_gp_from_aggregates(
    agg_rows,
    vis: Dict[str, object],
    camera_pose,
    *,
    target_mode: str,
    gp_length_scale: float,
    gp_noise_var: float,
    beta: float,
    min_prob: float,
):
    xs, ys = _grid_vectors(vis)
    if not agg_rows:
        raise RuntimeError('No fresh state-aligned capture samples remained for GP fitting')

    spec = _target_spec(target_mode)
    X_train = np.asarray([[r['x_center'], r['y_center']] for r in agg_rows], dtype=float)
    p_train = np.asarray([r[spec['field']] for r in agg_rows], dtype=float)
    p_train = np.clip(p_train, min_prob, 1.0 - min_prob)

    gp = SimpleRBFGP(
        length_scale=float(gp_length_scale),
        signal_var=1.0,
        noise_var=float(max(gp_noise_var, 1e-8)),
    ).fit(X_train, _logit(p_train))

    Xg, Yg = np.meshgrid(xs, ys)
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])
    mu_f, sigma_f = gp.predict_mean_std(XY)
    p_mean = _sigmoid(mu_f)
    p_cons = _sigmoid(mu_f - float(beta) * sigma_f)

    return {
        'xs': xs,
        'ys': ys,
        'X_train': X_train,
        'p_train': p_train,
        'P_mean_map': _clip_prob(p_mean.reshape(Yg.shape), min_prob).astype(float),
        'P_conservative_map': _clip_prob(p_cons.reshape(Yg.shape), min_prob).astype(float),
        'P_map': _clip_prob(p_cons.reshape(Yg.shape), min_prob).astype(float),
        'camera_pos': np.asarray(camera_pose[:3], dtype=float),
    }


def _write_raw_csv(path: Path, rows: List[CaptureSample]) -> None:
    fieldnames = [
        'sample_idx',
        'stamp',
        'state_x',
        'state_y',
        'cov_x',
        'cov_y',
        'state_age_s',
        'state_is_fresh',
        'detected',
        'usable_label',
        'u_mid',
        'v_mid',
        'yaw_est',
        'red_area_px',
        'border_margin_px',
        'normalized_blob_area',
        'yolo_score',
        'confidence_logit',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: getattr(row, key) for key in fieldnames
            })


def _write_agg_csv(path: Path, agg_rows) -> None:
    default_fields = [
        'ix', 'iy', 'x_center', 'y_center', 'n_samples', 'n_detected', 'n_usable',
        'detected_rate_mean', 'detected_rate_std', 'usable_rate_mean', 'usable_rate_std',
        'blob_area_target_mean', 'blob_area_target_std',
        'yolo_score_target_mean', 'yolo_score_target_std',
        'target_mode', 'target_mean', 'target_std',
        'mean_red_area_px', 'mean_yolo_score', 'mean_confidence_logit',
        'mean_border_margin_px', 'mean_blob_area_norm',
        'mean_state_age_s', 'mean_cov_x', 'mean_cov_y',
    ]
    fieldnames = list(agg_rows[0].keys()) if agg_rows else default_fields
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)


def _plot_fit(path: Path, fit, agg_rows, title: str, *, target_mode: str) -> None:
    import matplotlib.pyplot as plt

    spec = _target_spec(target_mode)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    extent = [fit['xs'][0], fit['xs'][-1], fit['ys'][0], fit['ys'][-1]]
    im0 = axes[0].imshow(
        fit['P_mean_map'],
        origin='lower',
        extent=extent,
        aspect='equal',
        vmin=0.0,
        vmax=1.0,
        cmap='viridis',
    )
    axes[0].set_title(f'{spec["title"]} mean')
    im1 = axes[1].imshow(
        fit['P_conservative_map'],
        origin='lower',
        extent=extent,
        aspect='equal',
        vmin=0.0,
        vmax=1.0,
        cmap='viridis',
    )
    axes[1].set_title(f'{spec["title"]} conservative')

    pts = np.asarray([[r['x_center'], r['y_center']] for r in agg_rows], dtype=float) if agg_rows else np.zeros((0, 2))
    vals = np.asarray([r['target_mean'] for r in agg_rows], dtype=float) if agg_rows else np.zeros((0,))
    if pts.size:
        for ax in axes:
            ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap='coolwarm', vmin=0.0, vmax=1.0, s=24, edgecolors='k', linewidths=0.25)
            ax.scatter([fit['camera_pos'][0]], [fit['camera_pos'][1]], c='white', s=50, marker='x')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Collect simulated visibility samples and fit a scalar GP visibility field.'
    )
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=_default_arg_path('src/experiments/config/world_profiles.yaml'))
    parser.add_argument('--output-root', default=_default_arg_path('logs/visibility_capture'))
    parser.add_argument('--publish-artifact', default='', help='Optional stable output path for the fitted GP artifact')
    parser.add_argument(
        '--from-capture-dir',
        default='',
        help='Reuse an existing capture directory and fit offline without waiting for live ROS topics.',
    )
    parser.add_argument(
        '--capture-mode',
        choices=('teleport', 'driving'),
        default='teleport',
        help='Live capture mode when not reusing an existing capture directory.',
    )
    parser.add_argument(
        '--target-mode',
        choices=('normalized_blob_area', 'binary_usable_detection', 'yolo_soft_score', 'localization_quality'),
        default='normalized_blob_area',
        help='Scalar target fitted into the planner-facing GP artifact.',
    )
    parser.add_argument('--usable-area-px-min', type=float, default=20.0)
    parser.add_argument('--max-state-age-s', type=float, default=0.50)
    parser.add_argument('--blob-area-ref-percentile', type=float, default=95.0)
    parser.add_argument('--ready-timeout-s', type=float, default=20.0)
    parser.add_argument('--max-capture-s', type=float, default=600.0)
    parser.add_argument('--post-done-grace-s', type=float, default=0.75)
    parser.add_argument('--sample-nx', type=int, default=15)
    parser.add_argument('--sample-ny', type=int, default=15)
    parser.add_argument('--sample-limit', type=int, default=0)
    parser.add_argument('--yaw-rad', type=float, default=0.0)
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--wall-margin-m', type=float, default=0.45)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--settle-s', type=float, default=0.25)
    parser.add_argument('--diag-timeout-s', type=float, default=1.6)
    parser.add_argument('--min-diag-messages', type=int, default=1)
    parser.add_argument('--gp-length-scale', type=float, default=-1.0)
    parser.add_argument('--gp-noise-var', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=-1.0)
    parser.add_argument('--min-prob', type=float, default=1e-4)
    args = parser.parse_args()

    world_profiles_path = Path(args.world_profiles).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_capture_dir = Path(str(args.from_capture_dir or '').strip()).expanduser().resolve() if str(args.from_capture_dir or '').strip() else None

    if source_capture_dir is not None:
        samples, source_meta = _load_samples_from_capture_dir(
            source_capture_dir,
            float(args.usable_area_px_min),
        )
        source_world = str(source_meta.get('world', '')).strip()
        world_name = source_world or args.world
        run_dir = output_root / f"fit_{source_capture_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        profile, vis, world_path, camera_pose = _load_world_profile(
            world_profiles_path,
            world_name,
            world_path_hint=str(source_meta.get('world_path', '')),
        )
        gp_length_scale = float(args.gp_length_scale if args.gp_length_scale > 0.0 else source_meta.get('gp_length_scale', vis.get('visibility_gp_length_scale', 1.0)))
        gp_noise_var = float(args.gp_noise_var if args.gp_noise_var > 0.0 else source_meta.get('gp_noise_var', vis.get('visibility_gp_noise_var', 0.1)))
        beta = float(args.beta if args.beta > 0.0 else source_meta.get('beta', vis.get('visibility_beta', 1.0)))
        skipped_no_state = 0
        stale_state_rows = int(sum(1 for s in samples if not s.state_is_fresh))
        sweep_config = dict(source_meta.get('sweep_config') or {})
        print(f'Loaded {len(samples)} samples from existing capture directory {source_capture_dir}')
    else:
        run_dir = output_root / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        profile, vis, world_path, camera_pose = _load_world_profile(world_profiles_path, args.world)
        gp_length_scale = float(args.gp_length_scale if args.gp_length_scale > 0.0 else vis.get('visibility_gp_length_scale', 1.0))
        gp_noise_var = float(args.gp_noise_var if args.gp_noise_var > 0.0 else vis.get('visibility_gp_noise_var', 0.1))
        beta = float(args.beta if args.beta > 0.0 else vis.get('visibility_beta', 1.0))
        capture_mode = str(args.capture_mode).strip().lower()
        rclpy.init()
        if capture_mode == 'teleport':
            node = TeleportVisibilityCapture(
                world_name=str(profile['world_name']),
                usable_area_px_min=float(args.usable_area_px_min),
                yaw_rad=float(args.yaw_rad),
                robot_z=float(args.robot_z),
                settle_s=float(args.settle_s),
                diag_timeout_s=float(args.diag_timeout_s),
                min_diag_messages=int(args.min_diag_messages),
            )
            try:
                node.get_logger().info(
                    f'Waiting for teleport capture interfaces for world {profile["world_name"]!r}'
                )
                node.wait_for_ready(timeout_s=float(args.ready_timeout_s))
                positions = _sample_positions(
                    vis,
                    sample_nx=int(args.sample_nx),
                    sample_ny=int(args.sample_ny),
                    wall_margin_m=float(args.wall_margin_m),
                    sample_limit=int(args.sample_limit),
                )
                node.get_logger().info(
                    f'Starting teleport visibility capture with {len(positions)} sampled poses '
                    f'({int(args.repeats)} repeat(s), yaw={float(args.yaw_rad):.3f} rad)'
                )
                samples = []
                sample_idx = 0
                for repeat_idx in range(max(int(args.repeats), 1)):
                    for x, y in positions:
                        samples.append(node._capture_at_pose(x, y, sample_idx=sample_idx))
                        sample_idx += 1
                        if sample_idx % 25 == 0:
                            node.get_logger().info(
                                f'Captured {sample_idx} sampled poses '
                                f'(repeat {repeat_idx + 1}/{max(int(args.repeats), 1)})'
                            )
                skipped_no_state = 0
                stale_state_rows = 0
                sweep_config = {
                    'capture_mode': 'teleport',
                    'world_name': profile['world_name'],
                    'sample_nx': int(args.sample_nx),
                    'sample_ny': int(args.sample_ny),
                    'sample_limit': int(args.sample_limit),
                    'yaw_rad': float(args.yaw_rad),
                    'robot_z': float(args.robot_z),
                    'wall_margin_m': float(args.wall_margin_m),
                    'repeats': int(args.repeats),
                    'settle_s': float(args.settle_s),
                    'diag_timeout_s': float(args.diag_timeout_s),
                    'min_diag_messages': int(args.min_diag_messages),
                    'n_unique_poses': int(len(positions)),
                }
            finally:
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
        else:
            node = DrivingVisibilityCapture(
                usable_area_px_min=float(args.usable_area_px_min),
                max_state_age_s=float(args.max_state_age_s),
            )
            try:
                node.get_logger().info(
                    f'Waiting for driving capture interfaces for world {profile["world_name"]!r}'
                )
                node.wait_for_ready(timeout_s=float(args.ready_timeout_s))
                node.get_logger().info('Starting driving-based visibility capture')
                node.run_until_complete(
                    max_capture_s=float(args.max_capture_s),
                    post_done_grace_s=float(args.post_done_grace_s),
                )
            finally:
                samples = list(node.samples)
                skipped_no_state = node.skipped_no_state
                stale_state_rows = node.stale_state_rows
                sweep_config = node.sweep_config or {}
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()

    min_prob = float(max(args.min_prob, 1e-6))

    blob_area_ref_px = _compute_blob_area_ref(samples, float(args.blob_area_ref_percentile), float(args.usable_area_px_min))
    _inject_normalized_blob_area(samples, blob_area_ref_px)
    agg_rows = _aggregate_samples(samples, vis, target_mode=str(args.target_mode))
    fit = _fit_gp_from_aggregates(
        agg_rows,
        vis,
        camera_pose,
        target_mode=str(args.target_mode),
        gp_length_scale=gp_length_scale,
        gp_noise_var=gp_noise_var,
        beta=beta,
        min_prob=min_prob,
    )

    raw_csv = run_dir / 'raw_detection_samples.csv'
    agg_csv = run_dir / 'aggregated_detection_samples.csv'
    npz_path = run_dir / 'empirical_visibility_gp.npz'
    plot_path = run_dir / 'empirical_visibility_gp.png'
    meta_path = run_dir / 'capture_config.json'

    _write_raw_csv(raw_csv, samples)
    _write_agg_csv(agg_csv, agg_rows)
    np.savez_compressed(
        npz_path,
        xs=fit['xs'],
        ys=fit['ys'],
        X_train=fit['X_train'],
        p_train=fit['p_train'],
        P_mean_map=fit['P_mean_map'],
        P_conservative_map=fit['P_conservative_map'],
        P_map=fit['P_map'],
        camera_pos=fit['camera_pos'],
    )
    capture_mode_label = {
        'teleport': 'Teleport-sampled',
        'driving': 'Driving-based',
    }.get(str(args.capture_mode).strip().lower(), 'Empirical')
    _plot_fit(
        plot_path,
        fit,
        agg_rows,
        title=f'{capture_mode_label} empirical visibility GP: {args.world}',
        target_mode=str(args.target_mode),
    )

    fresh_samples = [s for s in samples if s.state_is_fresh]
    target_spec = _target_spec(str(args.target_mode))
    meta = {
        'world': args.world,
        'world_name': profile['world_name'],
        'world_path': str(world_path),
        'world_profiles_path': str(world_profiles_path),
        'output_dir': str(run_dir),
        'capture_mode': str(source_meta.get('capture_mode', 'offline_refit')) if source_capture_dir is not None else str(args.capture_mode),
        'state_input_topic': (
            '/state/bev'
            if ((source_capture_dir is not None and str(source_meta.get('capture_mode', '')).strip().lower() in ('', 'driving', 'offline_refit'))
                or (source_capture_dir is None and str(args.capture_mode).strip().lower() == 'driving'))
            else 'sampled_robot_pose'
        ),
        'state_dimensions': ['x', 'y'],
        'label_mode': target_spec['mode'],
        'label_rule': target_spec['rule'],
        'usable_area_px_min': float(args.usable_area_px_min),
        'max_state_age_s': float(args.max_state_age_s),
        'blob_area_ref_percentile': float(args.blob_area_ref_percentile),
        'blob_area_ref_px': float(blob_area_ref_px),
        'n_raw_samples': int(len(samples)),
        'n_fresh_samples': int(len(fresh_samples)),
        'n_occupied_cells': int(len(agg_rows)),
        'skipped_no_state_samples': int(skipped_no_state),
        'stale_state_rows': int(stale_state_rows),
        'gp_length_scale': gp_length_scale,
        'gp_noise_var': gp_noise_var,
        'beta': beta,
        'mean_detected_rate_raw': float(np.mean([s.detected for s in samples])) if samples else math.nan,
        'mean_usable_rate_raw': float(np.mean([s.usable_label for s in samples])) if samples else math.nan,
        'mean_blob_area_target_raw': float(np.mean([s.normalized_blob_area for s in samples])) if samples else math.nan,
        'mean_yolo_score_raw': float(np.mean([
            np.clip(s.yolo_score, 0.0, 1.0) if math.isfinite(s.yolo_score) else 0.0
            for s in samples
        ])) if samples else math.nan,
        'mean_detected_rate_cells': float(np.mean([r['detected_rate_mean'] for r in agg_rows])) if agg_rows else math.nan,
        'mean_usable_rate_cells': float(np.mean([r['usable_rate_mean'] for r in agg_rows])) if agg_rows else math.nan,
        'mean_blob_area_target_cells': float(np.mean([r['blob_area_target_mean'] for r in agg_rows])) if agg_rows else math.nan,
        'mean_yolo_score_cells': float(np.mean([r['yolo_score_target_mean'] for r in agg_rows])) if agg_rows else math.nan,
        'mean_target_cells': float(np.mean([r['target_mean'] for r in agg_rows])) if agg_rows else math.nan,
        'sweep_config': sweep_config,
        'published_artifact_path': str(Path(args.publish_artifact).expanduser().resolve()) if str(args.publish_artifact).strip() else '',
        'source_capture_dir': str(source_capture_dir) if source_capture_dir is not None else '',
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    publish_artifact_raw = str(args.publish_artifact or '').strip()
    publish_artifact = Path(publish_artifact_raw).expanduser() if publish_artifact_raw else None
    if publish_artifact is not None:
        publish_artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npz_path, publish_artifact)
        print(f'Published empirical GP artifact to {publish_artifact.resolve()}')

    print(f'Wrote raw samples to {raw_csv}')
    print(f'Wrote aggregated samples to {agg_csv}')
    print(f'Wrote empirical GP artifact to {npz_path}')
    print(f'Wrote preview plot to {plot_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
