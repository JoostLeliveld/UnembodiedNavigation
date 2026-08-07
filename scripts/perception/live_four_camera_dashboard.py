#!/usr/bin/env python3
"""Live four-camera demo dashboard for the full warehouse simulation.

The dashboard consumes operational ROS topics plus simulator ground truth for
evaluation only. It presents camera measurements, fused belief, and ground
truth distinctly and timestamp-matches detector output to RGB. Odometry is an
internal prediction input, not a displayed localization result.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import threading
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
import yaml


CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
IMAGE_TOPICS = {
    "camera_A": "/external_camera/image_raw",
    "camera_B": "/external_camera_b/image_raw",
    "camera_C": "/external_camera_c/image_raw",
    "camera_D": "/external_camera_d/image_raw",
}
COLORS = {
    # Okabe-Ito color-blind-safe categorical palette (BGR for OpenCV).
    "camera_A": (178, 114, 0),     # blue
    "camera_B": (0, 159, 230),     # orange
    "camera_C": (115, 158, 0),     # bluish green
    "camera_D": (167, 121, 204),   # reddish purple
}
BG = (18, 16, 14)
PANEL = (31, 27, 23)
GRID = (64, 57, 50)
TEXT = (242, 239, 232)
MUTED = (172, 166, 156)
GREEN = (138, 190, 50)       # healthy / available
AMBER = (66, 225, 240)       # belief / selected estimate (yellow)
RED = (55, 94, 213)          # warning / error (vermillion)
CYAN = (233, 180, 86)        # simulator ground truth (sky blue)
UPDATE = (210, 145, 196)     # accepted camera correction
SITE_BOUNDARY = (225, 135, 45)
DRIVEABLE_EDGE = (205, 205, 195)
NOGO_EDGE = (95, 190, 105)
OBSTACLE_FILL = (35, 38, 42)
FONT = cv2.FONT_HERSHEY_DUPLEX
STREAM_LIVE_TIMEOUT_S = 10.0
POSE_WALL_SYNC_TOLERANCE_S = 0.25
NOGO_MARGIN_M = 0.32


def _text(img, text, xy, scale=0.55, color=TEXT, thickness=1):
    cv2.putText(img, str(text), xy, FONT, scale, color,
                thickness, cv2.LINE_AA)


def _legend(img, entries, xy, *, scale=0.31, gap=18):
    """Draw a compact horizontal line legend and return its right edge."""
    cursor_x, baseline_y = int(xy[0]), int(xy[1])
    for label, color in entries:
        cv2.line(
            img, (cursor_x, baseline_y - 5), (cursor_x + 16, baseline_y - 5),
            color, 3, cv2.LINE_AA,
        )
        cursor_x += 22
        _text(img, label, (cursor_x, baseline_y), scale, TEXT, 1)
        width = cv2.getTextSize(str(label), FONT, scale, 1)[0][0]
        cursor_x += width + gap
    return cursor_x


def _panel(img, rect, title, subtitle=""):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, 1)
    _text(img, title, (x + 14, y + 24), 0.58, TEXT, 2)
    if subtitle:
        _text(img, subtitle, (x + 14, y + 45), 0.42, MUTED, 1)


def _quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)


def _nearest_timestamped(rows, stamp_s: float, max_delta_s: float):
    if not rows or not math.isfinite(stamp_s):
        return None
    nearest = min(rows, key=lambda row: abs(float(row[0]) - stamp_s))
    return nearest if abs(float(nearest[0]) - stamp_s) <= max_delta_s else None


class LiveDashboardNode(Node):
    def __init__(self):
        super().__init__("live_four_camera_dashboard")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frames = {cam: None for cam in CAMERAS}
        self.frame_wall = {cam: 0.0 for cam in CAMERAS}
        self.frame_buffers = {cam: deque(maxlen=12) for cam in CAMERAS}
        self.obs = {cam: {} for cam in CAMERAS}
        self.obs_wall = {cam: 0.0 for cam in CAMERAS}
        self.history = {
            cam: deque(maxlen=900) for cam in CAMERAS
        }
        self.odom = None
        self.odom_noisy = None
        self.ground_truth = None
        self.mission_start = None
        self.ground_truth_wall = 0.0
        self.ground_truth_history = deque(maxlen=900)
        self.belief = None
        self.belief_wall = 0.0
        self.belief_history = deque(maxlen=900)
        self.state_update = None
        self.goal = None
        self.plan = []
        self.odom_trail = deque(maxlen=600)
        self.subscriptions_live = []
        for cam in CAMERAS:
            self.subscriptions_live.append(self.create_subscription(
                Image, IMAGE_TOPICS[cam],
                lambda msg, selected=cam: self._image(selected, msg),
                qos_profile_sensor_data,
            ))
            self.subscriptions_live.append(self.create_subscription(
                String, f"/perception/camera_observation/{cam}",
                lambda msg, selected=cam: self._observation(selected, msg), 10,
            ))
        self.subscriptions_live.append(
            self.create_subscription(Odometry, "/odom", self._odom, 20)
        )
        self.subscriptions_live.append(
            self.create_subscription(Odometry, "/odom_noisy", self._odom_noisy, 20)
        )
        self.subscriptions_live.append(
            self.create_subscription(TFMessage, "/ground_truth_tf", self._ground_truth, 20)
        )
        self.subscriptions_live.append(self.create_subscription(
            PoseWithCovarianceStamped, "/planner_belief", self._belief, 20
        ))
        self.subscriptions_live.append(self.create_subscription(
            PoseWithCovarianceStamped, "/state/bev", self._state_update, 20
        ))
        self.subscriptions_live.append(
            self.create_subscription(PoseStamped, "/goal_bev", self._goal, 10)
        )
        self.subscriptions_live.append(
            self.create_subscription(NavPath, "/plan", self._plan, 10)
        )

    def _image(self, cam, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"{cam} image conversion failed: {exc}")
            return
        stamp_s = _stamp_seconds(msg.header.stamp)
        wall = time.monotonic()
        with self.lock:
            self.frames[cam] = frame
            self.frame_wall[cam] = wall
            self.frame_buffers[cam].append((stamp_s, wall, frame))

    def _observation(self, cam, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        now = time.monotonic()
        score = float(payload.get("detector_score", 0.0) or 0.0)
        detected = bool(payload.get("detection_valid", False))
        latency = float(payload.get("yolo_inference_wall_ms", math.nan))
        age = float(payload.get("frame_age_at_publish_s", math.nan)) * 1000.0
        with self.lock:
            self.obs[cam] = payload
            self.obs_wall[cam] = now
            self.history[cam].append((now, score, detected, latency, age))

    def _odom(self, msg):
        p = msg.pose.pose.position
        yaw = _quat_yaw(msg.pose.pose.orientation)
        with self.lock:
            self.odom = (float(p.x), float(p.y), yaw)
            if not self.odom_trail or math.hypot(p.x - self.odom_trail[-1][0], p.y - self.odom_trail[-1][1]) > 0.03:
                self.odom_trail.append((float(p.x), float(p.y)))

    def _odom_noisy(self, msg):
        p = msg.pose.pose.position
        with self.lock:
            self.odom_noisy = (float(p.x), float(p.y), _quat_yaw(msg.pose.pose.orientation))

    def _ground_truth(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "turtlebot3":
                p = transform.transform.translation
                q = transform.transform.rotation
                stamp_s = _stamp_seconds(transform.header.stamp)
                row = (stamp_s, float(p.x), float(p.y), _quat_yaw(q))
                with self.lock:
                    self.ground_truth = row
                    if self.mission_start is None:
                        self.mission_start = (row[1], row[2], row[3])
                    self.ground_truth_wall = time.monotonic()
                    if not self.ground_truth_history or stamp_s != self.ground_truth_history[-1][0]:
                        self.ground_truth_history.append(row)
                return

    @staticmethod
    def _pose_with_covariance(msg):
        p = msg.pose.pose.position
        cov = msg.pose.covariance
        return (
            _stamp_seconds(msg.header.stamp), float(p.x), float(p.y),
            _quat_yaw(msg.pose.pose.orientation),
            np.asarray(((float(cov[0]), float(cov[1])),
                        (float(cov[6]), float(cov[7]))), dtype=float),
        )

    def _belief(self, msg):
        row = self._pose_with_covariance(msg)
        with self.lock:
            self.belief = row
            self.belief_wall = time.monotonic()
            if not self.belief_history or row[0] != self.belief_history[-1][0]:
                self.belief_history.append(row)

    def _state_update(self, msg):
        row = self._pose_with_covariance(msg)
        with self.lock:
            if self.state_update is None or row[0] != self.state_update[0]:
                self.state_update = (*row, time.monotonic())

    def _goal(self, msg):
        with self.lock:
            self.goal = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _plan(self, msg):
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in msg.poses
        ]
        with self.lock:
            self.plan = points

    def snapshot(self):
        with self.lock:
            matched_frames = {}
            frame_sync_delta = {}
            matched_wall = {}
            for cam in CAMERAS:
                payload = self.obs[cam]
                try:
                    obs_stamp = float(payload.get("timestamp_s", math.nan))
                except (TypeError, ValueError):
                    obs_stamp = math.nan
                match = _nearest_timestamped(self.frame_buffers[cam], obs_stamp, 0.20)
                if match is None:
                    frame = self.frames[cam]
                    matched_frames[cam] = None if frame is None else frame.copy()
                    matched_wall[cam] = self.frame_wall[cam]
                    frame_sync_delta[cam] = math.inf
                else:
                    matched_frames[cam] = match[2].copy()
                    # Stream health is about the latest received frame, while
                    # the displayed frame is the one aligned to the detector.
                    matched_wall[cam] = self.frame_wall[cam]
                    frame_sync_delta[cam] = abs(match[0] - obs_stamp)
            return {
                "frames": matched_frames,
                "frame_wall": matched_wall,
                "frame_sync_delta": frame_sync_delta,
                "obs": {k: dict(v) for k, v in self.obs.items()},
                "obs_wall": dict(self.obs_wall),
                "histories": {k: list(v) for k, v in self.history.items()},
                "odom": self.odom,
                "odom_noisy": self.odom_noisy,
                "odom_trail": list(self.odom_trail),
                "ground_truth": self.ground_truth,
                "ground_truth_wall": self.ground_truth_wall,
                "ground_truth_history": list(self.ground_truth_history),
                "belief": self.belief,
                "belief_wall": self.belief_wall,
                "belief_history": list(self.belief_history),
                "state_update": self.state_update,
                "mission_start": self.mission_start,
                "goal": self.goal,
                "plan": list(self.plan),
            }


class DashboardRenderer:
    def __init__(self, artifact: Path, world: Path,
                 start, goal, start_yaw, gp_root: Path | None = None,
                 world_profiles: Path | None = None):
        gp = np.load(artifact, allow_pickle=True)
        self.xs = np.asarray(gp["xs"], dtype=float)
        self.ys = np.asarray(gp["ys"], dtype=float)
        field = np.asarray(gp["P_best_4cam_map"], dtype=float)
        learned_maps = {
            cam: np.asarray(gp[f"P_{cam}_map"], dtype=float) for cam in CAMERAS
        }
        self.gp_maps = {}
        for cam in CAMERAS:
            fitted = None if gp_root is None else gp_root / cam / "det_hit_expected_kernel_gp.npz"
            if fitted is not None and fitted.is_file():
                payload = np.load(fitted, allow_pickle=True)
                candidate = np.asarray(payload["P_conservative_plan_map"], dtype=float)
                if candidate.shape == field.shape:
                    self.gp_maps[cam] = candidate
                    continue
            self.gp_maps[cam] = learned_maps[cam]
        owner = np.asarray(gp["best_camera_id"]).astype(str)
        valid = np.isfinite(field)
        normalized = np.zeros_like(field, dtype=np.uint8)
        normalized[valid] = np.clip(field[valid] * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = (30, 32, 38)
        ownership = np.zeros_like(colored)
        for cam in CAMERAS:
            mask = valid & (owner == cam)
            binary = mask.astype(np.uint8)
            boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            ownership[boundary] = COLORS[cam]
        self.gp_bgr = cv2.flip(colored, 0)
        self.gp_bgr_by_camera = {}
        for cam, camera_field in self.gp_maps.items():
            camera_valid = np.isfinite(camera_field)
            camera_norm = np.zeros_like(camera_field, dtype=np.uint8)
            camera_norm[camera_valid] = np.clip(
                camera_field[camera_valid] * 255.0, 0, 255
            ).astype(np.uint8)
            camera_color = cv2.applyColorMap(camera_norm, cv2.COLORMAP_TURBO)
            camera_color[~camera_valid] = (30, 32, 38)
            self.gp_bgr_by_camera[cam] = cv2.flip(camera_color, 0)
        self.ownership_bgr = cv2.flip(ownership, 0)
        if world_profiles is None:
            repo = Path(__file__).resolve().parents[2]
            world_profiles = repo / "src/experiments/config/world_profiles.yaml"
        (
            self.site_mask,
            self.driveable_mask,
            self.nogo_mask,
            self.obstacle_mask,
        ) = self._navigation_geometry(world, world_profiles)
        from reliability.projection import camera_model_from_world
        includes = {
            "camera_A": "external_camera",
            "camera_B": "external_camera_b",
            "camera_C": "external_camera_c",
            "camera_D": "external_camera_d",
        }
        self.camera_models = {
            cam: camera_model_from_world(world, include_name=include)
            for cam, include in includes.items()
        }
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.start_yaw = float(start_yaw)
        self.started = time.monotonic()
        self.localization_history = deque(maxlen=900)
        self.belief_error_history = deque(maxlen=900)
        self.last_localization_stamp = None
        self.last_belief_stamp = None
        self.last_localization_wall = None
        self.freshness_history = deque(maxlen=900)

    def _navigation_geometry(self, world: Path, world_profiles: Path):
        """Rasterize the planner's effective floor semantics on the GP grid.

        The displayed driveable region is the declared lane union minus the
        conservative envelopes derived from the current SDF collisions. This
        is the actual navigation contract; drawing every source rectangle
        separately would create misleading internal seams and exact-fit boxes.
        """
        xx, yy = np.meshgrid(self.xs, self.ys)
        empty = np.zeros(xx.shape, dtype=bool)
        site = empty.copy()
        lane_union = empty.copy()

        profiles = yaml.safe_load(world_profiles.read_text("utf-8"))
        profile = profiles.get("worlds", {}).get(world.name, {})
        for region in profile.get("known_2d_regions", []):
            kind = str(region.get("type", ""))
            if kind not in ("site_boundary", "traversable"):
                continue
            mask = (
                (xx >= float(region["xmin"]))
                & (xx <= float(region["xmax"]))
                & (yy >= float(region["ymin"]))
                & (yy <= float(region["ymax"]))
            )
            if kind == "site_boundary":
                site |= mask
            else:
                lane_union |= mask

        from unav_common.occlusion_geometry import parse_collision_scene_from_world

        scene = parse_collision_scene_from_world(
            str(world), model_names=("warehouse_rack_occluders",),
        )
        obstacles = empty.copy()
        nogo = empty.copy()
        for prism in scene.prisms:
            obstacle = (
                (xx >= prism.xmin) & (xx <= prism.xmax)
                & (yy >= prism.ymin) & (yy <= prism.ymax)
            )
            obstacles |= obstacle
            nogo |= (
                (xx >= prism.xmin - NOGO_MARGIN_M)
                & (xx <= prism.xmax + NOGO_MARGIN_M)
                & (yy >= prism.ymin - NOGO_MARGIN_M)
                & (yy <= prism.ymax + NOGO_MARGIN_M)
            )

        # The site mask is a hard keep-in bound. Obstacles remain visible even
        # where a broad cross-aisle rectangle reaches their safety envelope.
        effective_driveable = lane_union & site & ~nogo
        return tuple(cv2.flip(mask.astype(np.uint8), 0) for mask in (
            site, effective_driveable, nogo & site, obstacles & site,
        ))

    @staticmethod
    def _resized_boundary(mask, size, *, thickness=1):
        resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        boundary = cv2.morphologyEx(
            resized, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8),
        )
        if thickness > 1:
            boundary = cv2.dilate(
                boundary, np.ones((thickness, thickness), np.uint8), iterations=1,
            )
        return boundary.astype(bool)

    def _draw_navigation_geometry(self, canvas, plot):
        px, py, pw, ph = plot
        view = canvas[py:py + ph, px:px + pw]
        site = cv2.resize(self.site_mask, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        driveable = cv2.resize(
            self.driveable_mask, (pw, ph), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        nogo = cv2.resize(self.nogo_mask, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        obstacles = cv2.resize(
            self.obstacle_mask, (pw, ph), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # Keep the GP readable in legal corridors and subordinate it elsewhere.
        outside = ~driveable
        view[outside] = (
            0.30 * view[outside] + 0.70 * np.asarray(PANEL, dtype=float)
        ).astype(np.uint8)
        view[nogo] = (
            0.72 * view[nogo] + 0.28 * np.asarray(NOGO_EDGE, dtype=float)
        ).astype(np.uint8)
        view[obstacles] = OBSTACLE_FILL

        for mask, color, thickness in (
            (self.site_mask, SITE_BOUNDARY, 2),
            (self.driveable_mask, DRIVEABLE_EDGE, 1),
            (self.nogo_mask, NOGO_EDGE, 2),
        ):
            edge = self._resized_boundary(mask, (pw, ph), thickness=thickness)
            view[edge] = color

        # One compact key makes the three semantics explicit without recreating
        # the old maze of per-rectangle labels.
        lx, ly = px + 8, py + 8
        cv2.rectangle(canvas, (lx, ly), (lx + 208, ly + 76), (20, 21, 23), -1)
        entries = (
            ("driveable", DRIVEABLE_EDGE),
            ("safety envelope", NOGO_EDGE),
            ("collision", OBSTACLE_FILL),
            ("site boundary", SITE_BOUNDARY),
        )
        for index, (label, color) in enumerate(entries):
            yy = ly + 15 + 18 * index
            cv2.rectangle(canvas, (lx + 8, yy - 8), (lx + 22, yy + 2), color, -1)
            _text(canvas, label, (lx + 29, yy + 2), 0.31, TEXT, 1)

    def _gp_value(self, cam, xy):
        if not self._in_gp_domain(xy):
            return math.nan
        ix = int(np.argmin(np.abs(self.xs - float(xy[0]))))
        iy = int(np.argmin(np.abs(self.ys - float(xy[1]))))
        value = float(self.gp_maps[cam][iy, ix])
        return value if math.isfinite(value) else math.nan

    def _in_gp_domain(self, xy):
        return (
            xy is not None
            and self.xs[0] <= float(xy[0]) <= self.xs[-1]
            and self.ys[0] <= float(xy[1]) <= self.ys[-1]
        )

    def _best_camera(self, xy):
        values = {cam: self._gp_value(cam, xy) for cam in CAMERAS}
        finite = {cam: value for cam, value in values.items() if math.isfinite(value)}
        return max(finite, key=finite.get) if finite else None

    def _project_pixel(self, cam, u, v):
        """Inverse perspective mapping -- the same projection the runtime uses."""

        point = self.camera_models[cam].pixel_to_world(float(u), float(v))
        return None if point is None else (float(point[0]), float(point[1]))

    def _project_observation(self, cam, observation):
        if not observation.get("detection_valid", False):
            return None
        pixel = observation.get("pixel_uv")
        if not pixel:
            return None
        u, v = float(pixel[0]), float(pixel[1])
        centre = self._project_pixel(cam, u, v)
        du_plus, du_minus = self._project_pixel(cam, u + 0.5, v), self._project_pixel(cam, u - 0.5, v)
        dv_plus, dv_minus = self._project_pixel(cam, u, v + 0.5), self._project_pixel(cam, u, v - 0.5)
        if centre is None or None in (du_plus, du_minus, dv_plus, dv_minus):
            return None
        jacobian = np.array([
            [du_plus[0] - du_minus[0], dv_plus[0] - dv_minus[0]],
            [du_plus[1] - du_minus[1], dv_plus[1] - dv_minus[1]],
        ], dtype=float)
        r_uv = np.asarray(observation.get("conditional_cov_uv", ((1600.0, 0.0), (0.0, 1600.0))), dtype=float)
        r_xy = jacobian @ r_uv @ jacobian.T
        sigma_radial = math.sqrt(max(float(np.trace(r_xy)), 0.0))
        return centre, sigma_radial

    def _odom_to_world(self, pose):
        if pose is None:
            return None
        c, s = math.cos(self.start_yaw), math.sin(self.start_yaw)
        return (
            self.start[0] + c * pose[0] - s * pose[1],
            self.start[1] + s * pose[0] + c * pose[1],
            self.start_yaw + pose[2],
        )

    def _map_xy(self, xy, rect):
        x, y, w, h = rect
        px = x + int(np.clip((xy[0] - self.xs[0]) / (self.xs[-1] - self.xs[0]), 0, 1) * (w - 1))
        py = y + h - 1 - int(np.clip((xy[1] - self.ys[0]) / (self.ys[-1] - self.ys[0]), 0, 1) * (h - 1))
        return px, py

    def _camera_tile(self, canvas, rect, cam, frame, frame_age, sync_delta, obs, obs_age):
        x, y, w, h = rect
        body_y, body_h = y + 35, h - 35
        cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
        # Gazebo renders the four 1280x720 cameras sequentially and can dip
        # below 0.5 Hz while YOLO and the fullscreen dashboard share the GPU.
        # Ten seconds still catches a dead bridge without presenting
        # a healthy low-rate feed as disconnected.
        live = frame is not None and frame_age < STREAM_LIVE_TIMEOUT_S
        status = "LIVE" if live else "WAITING"
        status_color = GREEN if live else RED
        _text(canvas, cam.replace("camera_", "CAMERA "), (x + 12, y + 24), 0.56, COLORS[cam], 2)
        _text(canvas, status, (x + w - 76, y + 24), 0.44, status_color, 2)
        if frame is None:
            _text(canvas, "Waiting for RGB stream...", (x + 28, body_y + body_h // 2), 0.58, MUTED, 1)
        else:
            view = cv2.resize(frame, (w, body_h), interpolation=cv2.INTER_AREA)
            bbox = obs.get("bbox_xyxy")
            synchronized = math.isfinite(sync_delta) and sync_delta <= 0.20
            detected = (
                bool(obs.get("detection_valid", False))
                and obs_age < 1.5
                and synchronized
            )
            if detected and bbox and len(bbox) == 4:
                sx, sy = w / frame.shape[1], body_h / frame.shape[0]
                p0 = (int(float(bbox[0]) * sx), int(float(bbox[1]) * sy))
                p1 = (int(float(bbox[2]) * sx), int(float(bbox[3]) * sy))
                cv2.rectangle(view, p0, p1, GREEN, 3)
                score = float(obs.get("detector_score", 0.0) or 0.0)
                label = f"ROBOT  {score:.0%}"
                ly = max(p0[1] - 8, 22)
                cv2.rectangle(view, (p0[0], ly - 20), (p0[0] + 138, ly + 3), (20, 35, 30), -1)
                _text(view, label, (p0[0] + 5, ly - 2), 0.48, GREEN, 2)
                pixel = obs.get("pixel_uv")
                if pixel:
                    cv2.circle(view, (int(float(pixel[0]) * sx), int(float(pixel[1]) * sy)), 5, AMBER, -1)
            elif obs_age < 1.5 and synchronized:
                _text(view, "NO DETECTION", (16, 28), 0.55, AMBER, 2)
            elif obs_age < 1.5:
                _text(view, "UNSYNCED OBSERVATION - OVERLAY SUPPRESSED", (16, 28), 0.44, RED, 2)
            sync_text = "sync --" if not math.isfinite(sync_delta) else f"sync {sync_delta * 1000.0:.0f} ms"
            _text(view, sync_text, (w - 105, body_h - 10), 0.34,
                  GREEN if synchronized else RED, 1)
            canvas[body_y:body_y + body_h, x:x + w] = view
        cv2.rectangle(canvas, (x, y), (x + w, y + h), GRID, 1)

    def _gp_panel(self, canvas, rect, belief, belief_trail,
                  ground_truth, truth_trail, state_update, projected, selected,
                  plan):
        x, y, w, h = rect
        reference_xy = belief[1:3] if belief is not None else (
            ground_truth[1:3] if ground_truth is not None else None
        )
        active_cam = selected[0] if selected is not None else self._best_camera(reference_xy)
        cam_label = active_cam[-1] if active_cam else "--"
        _panel(
            canvas, rect, "OFFLINE FITTED GP RELIABILITY + CAMERA OWNERSHIP",
            f"camera {cam_label} conservative P(hit), ranking prior - NOT live visibility",
        )
        plot = (x + 18, y + 58, w - 36, h - 104)
        px, py, pw, ph = plot
        gp_source = self.gp_bgr_by_camera.get(active_cam, self.gp_bgr)
        gp = cv2.resize(gp_source, (pw, ph), interpolation=cv2.INTER_LINEAR)
        canvas[py:py + ph, px:px + pw] = gp
        ownership = cv2.resize(self.ownership_bgr, (pw, ph), interpolation=cv2.INTER_NEAREST)
        owner_mask = np.any(ownership > 0, axis=2)
        canvas[py:py + ph, px:px + pw][owner_mask] = ownership[owner_mask]
        self._draw_navigation_geometry(canvas, plot)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (115, 125, 138), 1)
        camera_xy = {
            cam: tuple(float(value) for value in model.cam_pos[:2])
            for cam, model in self.camera_models.items()
        }
        for cam, location in camera_xy.items():
            cp = self._map_xy(location, plot)
            cv2.drawMarker(canvas, cp, COLORS[cam], cv2.MARKER_TRIANGLE_UP, 18, 3)
            _text(canvas, cam[-1], (cp[0] + 9, cp[1] + (14 if location[1] > 0 else -7)),
                  0.45, COLORS[cam], 2)

        for a, b in zip(belief_trail[:-1], belief_trail[1:]):
            cv2.line(canvas, self._map_xy(a, plot), self._map_xy(b, plot), AMBER, 2, cv2.LINE_AA)
        for a, b in zip(truth_trail[:-1], truth_trail[1:]):
            cv2.line(canvas, self._map_xy(a, plot), self._map_xy(b, plot), CYAN, 2, cv2.LINE_AA)

        for a, b in zip(plan[:-1], plan[1:]):
            cv2.line(canvas, self._map_xy(a, plot), self._map_xy(b, plot),
                     (245, 245, 245), 2, cv2.LINE_AA)

        start_px = self._map_xy(self.start, plot)
        goal_px = self._map_xy(self.goal, plot)
        cv2.circle(canvas, start_px, 9, GREEN, -1)
        _text(canvas, "S", (start_px[0] - 6, start_px[1] + 5), 0.45, (15, 35, 22), 2)
        cv2.drawMarker(canvas, goal_px, TEXT, cv2.MARKER_STAR, 24, 3)
        _text(canvas, "GOAL", (goal_px[0] + 13, goal_px[1] - 9), 0.43, TEXT, 2)

        if state_update is not None:
            sp = self._map_xy(state_update[1:3], plot)
            cv2.drawMarker(canvas, sp, UPDATE, cv2.MARKER_DIAMOND, 13, 2)

        if belief is not None:
            bp = self._map_xy(belief[1:3], plot)
            cv2.circle(canvas, bp, 9, AMBER, -1)
            _text(canvas, "BELIEF", (bp[0] + 10, bp[1] - 8), 0.34, AMBER, 2)
            cov = np.asarray(belief[4], dtype=float)
            if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
                vals = np.maximum(vals, 0.0)
                order = np.argsort(vals)[::-1]
                major, minor = 2.0 * np.sqrt(vals[order])
                scale_x = pw / max(self.xs[-1] - self.xs[0], 1.0e-9)
                scale_y = ph / max(self.ys[-1] - self.ys[0], 1.0e-9)
                axes = (max(1, int(major * scale_x)), max(1, int(minor * scale_y)))
                vector = vecs[:, order[0]]
                angle = -math.degrees(math.atan2(vector[1], vector[0]))
                cv2.ellipse(canvas, bp, axes, angle, 0, 360, AMBER, 1, cv2.LINE_AA)

        truth_outside = ground_truth is not None and not self._in_gp_domain(ground_truth[1:3])
        if ground_truth is not None:
            tp = self._map_xy(ground_truth[1:3], plot)
            cv2.drawMarker(canvas, tp, RED if truth_outside else CYAN,
                           cv2.MARKER_TILTED_CROSS if truth_outside else cv2.MARKER_STAR, 20, 3)
            _text(canvas, "GT OUTSIDE GP" if truth_outside else "GT",
                  (tp[0] + 10, tp[1] + 15), 0.35, RED if truth_outside else CYAN, 2)
        if truth_outside:
            cv2.rectangle(canvas, (px + 6, py + 6), (px + 226, py + 33), (25, 28, 80), -1)
            _text(canvas, "GROUND TRUTH OUTSIDE GP DOMAIN", (px + 13, py + 26), 0.40, RED, 2)

        for cam, measurement in projected.items():
            mp = self._map_xy(measurement, plot)
            cv2.circle(canvas, mp, 6, COLORS[cam], 2)
        if selected is not None:
            selected_cam, measurement, loc_conf, gp_value, anchor = selected
            mp = self._map_xy(measurement, plot)
            if anchor is not None:
                pp = self._map_xy(anchor, plot)
                cv2.arrowedLine(canvas, pp, mp, COLORS[selected_cam], 3,
                               cv2.LINE_AA, tipLength=0.25)
                delta = math.hypot(measurement[0] - anchor[0], measurement[1] - anchor[1])
                gp_text = "out" if not math.isfinite(gp_value) else f"{gp_value:.2f}"
                _text(canvas, f"PIXEL {selected_cam[-1]} residual {delta:.2f}m | GP {gp_text}",
                      (x + 18, y + h - 35), 0.43, COLORS[selected_cam], 2)

        legend_x = x + w - 258
        for i, cam in enumerate(CAMERAS):
            lx = legend_x + i * 61
            cv2.rectangle(canvas, (lx, y + 36), (lx + 13, y + 47), COLORS[cam], -1)
            _text(canvas, cam[-1], (lx + 18, y + 47), 0.38, TEXT, 1)
        if belief is not None and ground_truth is not None:
            error = math.hypot(belief[1] - ground_truth[1], belief[2] - ground_truth[2])
            footer = (f"belief ({belief[1]:+.2f},{belief[2]:+.2f}) | "
                      f"GT ({ground_truth[1]:+.2f},{ground_truth[2]:+.2f}) | error {error:.2f}m")
        else:
            footer = "waiting for fused belief and simulator ground truth"
        _text(canvas, footer, (x + 18, y + h - 15), 0.43, MUTED, 1)

    def _line_chart(self, canvas, rect, title, subtitle, histories, column, y_max, suffix):
        _panel(canvas, rect, title, subtitle)
        x, y, w, h = rect
        plot = (x + 48, y + 58, w - 66, h - 84)
        px, py, pw, ph = plot
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = py + ph - int(frac * ph)
            cv2.line(canvas, (px, yy), (px + pw, yy), GRID, 1)
            _text(canvas, f"{frac * y_max:.0f}{suffix}", (x + 6, yy + 4), 0.34, MUTED)
        now = time.monotonic()
        for cam in CAMERAS:
            points = []
            for row in histories[cam]:
                age = now - row[0]
                value = row[column]
                if age <= 30.0 and math.isfinite(value):
                    xx = px + pw - int(age / 30.0 * pw)
                    yy = py + ph - int(np.clip(value / y_max, 0, 1) * ph)
                    points.append((xx, yy))
            if len(points) > 1:
                cv2.polylines(canvas, [np.asarray(points, np.int32)], False, COLORS[cam], 2, cv2.LINE_AA)
        _text(canvas, "-30 s", (px, py + ph + 18), 0.36, MUTED)
        _text(canvas, "now", (px + pw - 28, py + ph + 18), 0.36, MUTED)

    def _localization_chart(self, canvas, rect, *, confidence):
        title = "LOCALIZATION RELIABILITY SCORE" if confidence else "FUSED BELIEF ERROR + UNCERTAINTY"
        subtitle = (
            "detector x conservative fitted GP at predicted belief (ranking only)"
            if confidence else
            "||planner belief - simulator GT|| | 2 sigma belief major axis (white)"
        )
        _panel(canvas, rect, title, subtitle)
        x, y, w, h = rect
        if confidence:
            _legend(
                canvas,
                [(cam[-1], COLORS[cam]) for cam in CAMERAS],
                (x + 16, y + 67),
                gap=12,
            )
        else:
            _legend(
                canvas,
                [("belief error", AMBER), ("2sigma uncertainty", TEXT)],
                (x + 16, y + 67),
                gap=17,
            )
        px, py, pw, ph = x + 52, y + 82, w - 70, h - 108
        y_max = 1.0 if confidence else 1.5
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = py + ph - int(frac * ph)
            cv2.line(canvas, (px, yy), (px + pw, yy), GRID, 1)
            label = f"{frac:.2f}" if confidence else f"{frac * y_max:.1f}m"
            _text(canvas, label, (x + 6, yy + 4), 0.34, MUTED)
        now = time.monotonic()
        source = self.localization_history if confidence else self.belief_error_history
        rows = [row for row in source if now - row[0] <= 30.0]
        for a, b in zip(rows[:-1], rows[1:]):
            value_a, value_b = a[1], b[1]
            p0 = (px + pw - int((now - a[0]) / 30.0 * pw),
                  py + ph - int(np.clip(value_a / y_max, 0, 1) * ph))
            p1 = (px + pw - int((now - b[0]) / 30.0 * pw),
                  py + ph - int(np.clip(value_b / y_max, 0, 1) * ph))
            color = COLORS[a[2]] if confidence else AMBER
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
            if not confidence:
                sigma_a, sigma_b = 2.0 * a[2], 2.0 * b[2]
                s0 = (p0[0], py + ph - int(np.clip(sigma_a / y_max, 0, 1) * ph))
                s1 = (p1[0], py + ph - int(np.clip(sigma_b / y_max, 0, 1) * ph))
                cv2.line(canvas, s0, s1, TEXT, 1, cv2.LINE_AA)
        if rows:
            value = rows[-1][1]
            unit = "" if confidence else " m"
            if confidence:
                cam, gp_value, detector_score = rows[-1][2:5]
                gp_text = "out" if not math.isfinite(gp_value) else f"{gp_value:.2f}"
                label = f"NOW {value:.3f} = det {detector_score:.2f} x GP {gp_text} | cam {cam[-1]}"
                color = COLORS[cam]
            else:
                label = f"NOW {value:.3f}{unit} | 2sigma {2.0 * rows[-1][2]:.3f} m"
                color = AMBER
            _text(canvas, label, (x + w - 430, y + 68), 0.34, color, 1)
        _text(canvas, "-30 s", (px, py + ph + 18), 0.36, MUTED)
        _text(canvas, "now", (px + pw - 28, py + ph + 18), 0.36, MUTED)

    def _health_panel(self, canvas, rect, histories, obs, obs_wall):
        _panel(canvas, rect, "CAMERA HEALTH", "recent detection rate and current confidence")
        x, y, w, h = rect
        now = time.monotonic()
        for i, cam in enumerate(CAMERAS):
            yy = y + 66 + i * 48
            recent = [row for row in histories[cam] if now - row[0] <= 10.0]
            rate = sum(bool(row[2]) for row in recent) / max(len(recent), 1)
            score = float(obs[cam].get("detector_score", 0.0) or 0.0) if now - obs_wall[cam] < 2 else 0.0
            _text(canvas, cam[-1], (x + 15, yy + 14), 0.55, COLORS[cam], 2)
            bx, bw = x + 42, w - 150
            cv2.rectangle(canvas, (bx, yy), (bx + bw, yy + 16), GRID, -1)
            cv2.rectangle(canvas, (bx, yy), (bx + int(rate * bw), yy + 16), COLORS[cam], -1)
            _text(canvas, f"{rate:3.0%}", (x + w - 94, yy + 14), 0.43, TEXT, 1)
            _text(canvas, f"conf {score:.2f}", (x + w - 94, yy + 33), 0.34, MUTED, 1)

    def _freshness_panel(self, canvas, rect):
        _panel(canvas, rect, "UPDATE FRESHNESS + AGREEMENT",
               "age since accepted /state/bev correction | synchronized camera spread")
        x, y, w, h = rect
        now = time.monotonic()
        rows = [row for row in self.freshness_history if now - row[0] <= 30.0]
        _legend(
            canvas,
            [("update age", AMBER), ("camera spread", RED)],
            (x + 14, y + 68),
            scale=0.28,
            gap=12,
        )

        def spark(top, height, column, y_max, color, label, unit):
            left, width = x + 44, w - 60
            cv2.rectangle(canvas, (left, top), (left + width, top + height), GRID, 1)
            _text(canvas, label, (x + 10, top - 7), 0.37, color, 1)
            points = []
            for row in rows:
                xx = left + width - int((now - row[0]) / 30.0 * width)
                yy = top + height - int(np.clip(row[column] / y_max, 0, 1) * height)
                points.append((xx, yy))
            if len(points) > 1:
                cv2.polylines(canvas, [np.asarray(points, np.int32)], False, color, 2, cv2.LINE_AA)
            value = rows[-1][column] if rows else math.nan
            value_text = "--" if not math.isfinite(value) else f"{value:.2f}{unit}"
            _text(canvas, value_text, (x + w - 82, top - 7), 0.38, TEXT, 2)
            _text(canvas, f"{y_max:g}{unit}", (x + 4, top + 12), 0.30, MUTED)
            _text(canvas, "0", (x + 24, top + height), 0.30, MUTED)

        spark(y + 101, 88, 1, 3.0, AMBER, "TIME SINCE FUSED LOCATION UPDATE", "s")
        spark(y + 250, 88, 2, 1.5, RED, "MAX PAIRWISE CAMERA DISAGREEMENT", "m")

    def render(self, snapshot):
        frames = snapshot["frames"]
        frame_wall = snapshot["frame_wall"]
        frame_sync_delta = snapshot["frame_sync_delta"]
        obs = snapshot["obs"]
        obs_wall = snapshot["obs_wall"]
        histories = snapshot["histories"]
        belief = snapshot["belief"]
        belief_history = snapshot["belief_history"]
        ground_truth = snapshot["ground_truth"]
        truth_history = snapshot["ground_truth_history"]
        state_update = snapshot["state_update"]
        if snapshot["mission_start"] is not None:
            self.start = snapshot["mission_start"][:2]
            if len(snapshot["mission_start"]) >= 3:
                self.start_yaw = snapshot["mission_start"][2]
        if snapshot["goal"] is not None:
            self.goal = snapshot["goal"]
        plan = snapshot["plan"]
        now = time.monotonic()
        if now - snapshot["belief_wall"] > 1.0:
            belief = None
        if now - snapshot["ground_truth_wall"] > 1.0:
            ground_truth = None

        projected = {}
        projected_stamps = {}
        candidates = []
        for cam in CAMERAS:
            if now - obs_wall[cam] >= 1.5:
                continue
            projection = self._project_observation(cam, obs[cam])
            if projection is None:
                continue
            measurement, sigma = projection
            projected[cam] = measurement
            try:
                obs_stamp = float(obs[cam].get("timestamp_s", math.nan))
            except (TypeError, ValueError):
                obs_stamp = math.nan
            projected_stamps[cam] = obs_stamp
            score = float(obs[cam].get("detector_score", 0.0) or 0.0)
            predicted = _nearest_timestamped(belief_history, obs_stamp, 0.35)
            anchor = predicted[1:3] if predicted is not None else (
                belief[1:3] if belief is not None else None
            )
            gp_value = self._gp_value(cam, anchor)
            loc_score = score * gp_value if math.isfinite(gp_value) else 0.0
            rank = loc_score if math.isfinite(gp_value) else score * 0.01
            candidates.append((rank, loc_score, cam, measurement, gp_value, anchor, obs_stamp, sigma, score))
        selected = None
        if candidates:
            _, loc_score, selected_cam, measurement, gp_value, anchor, stamp, _, detector_score = max(candidates)
            selected = (selected_cam, measurement, loc_score, gp_value, anchor)
            if stamp != self.last_localization_stamp and math.isfinite(stamp):
                self.localization_history.append(
                    (now, loc_score, selected_cam, gp_value, detector_score)
                )
                self.last_localization_stamp = stamp
        if belief is not None and belief[0] != self.last_belief_stamp:
            truth_at_belief = _nearest_timestamped(truth_history, belief[0], 0.20)
            if (
                truth_at_belief is None
                and ground_truth is not None
                and abs(snapshot["belief_wall"] - snapshot["ground_truth_wall"])
                <= POSE_WALL_SYNC_TOLERANCE_S
            ):
                # Some Gazebo TF bridges expose a different header clock from
                # /planner_belief. Both callbacks still arrive from the same
                # simulator tick; wall-arrival pairing is the safe fallback.
                truth_at_belief = ground_truth
            cov = np.asarray(belief[4], dtype=float)
            if truth_at_belief is not None and cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                error = math.hypot(belief[1] - truth_at_belief[1], belief[2] - truth_at_belief[2])
                eigvals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
                sigma_major = math.sqrt(max(float(np.max(eigvals)), 0.0))
                self.belief_error_history.append((now, error, sigma_major))
                self.last_belief_stamp = belief[0]

        update_age = (
            now - state_update[-1]
            if state_update is not None else 3.0
        )
        pairwise = [
            math.hypot(a[0] - b[0], a[1] - b[1])
            for i, (cam_a, a) in enumerate(projected.items())
            for cam_b, b in list(projected.items())[i + 1:]
            if math.isfinite(projected_stamps[cam_a])
            and math.isfinite(projected_stamps[cam_b])
            and abs(projected_stamps[cam_a] - projected_stamps[cam_b]) <= 0.25
        ]
        disagreement = max(pairwise) if pairwise else 0.0
        self.freshness_history.append((now, update_age, disagreement))
        canvas = np.full((1080, 1920, 3), BG, np.uint8)
        _text(canvas, "UNEMBODIED NAVIGATION", (22, 35), 0.82, TEXT, 2)
        _text(canvas, "LIVE FOUR-CAMERA PERCEPTION + RELIABILITY", (22, 59), 0.47, MUTED, 1)
        active = sum(now - frame_wall[c] < STREAM_LIVE_TIMEOUT_S for c in CAMERAS)
        dets = sum(
            bool(obs[c].get("detection_valid", False))
            and now - obs_wall[c] < 1.5
            and math.isfinite(frame_sync_delta[c])
            and frame_sync_delta[c] <= 0.20
            for c in CAMERAS
        )
        badge = GREEN if active == 4 else AMBER
        _text(canvas, f"{active}/4 STREAMS", (1540, 35), 0.58, badge, 2)
        _text(canvas, f"{dets}/4 DETECTIONS", (1715, 35), 0.58, GREEN if dets else AMBER, 2)
        _text(canvas, time.strftime("%H:%M:%S"), (1785, 61), 0.43, MUTED, 1)

        tiles = {
            "camera_A": (20, 78, 535, 286),
            "camera_B": (566, 78, 535, 286),
            "camera_C": (20, 375, 535, 286),
            "camera_D": (566, 375, 535, 286),
        }
        for cam in CAMERAS:
            self._camera_tile(
                canvas, tiles[cam], cam, frames[cam], now - frame_wall[cam],
                frame_sync_delta[cam], obs[cam], now - obs_wall[cam],
            )
        self._gp_panel(
            canvas, (1112, 78, 788, 583),
            belief, [row[1:3] for row in belief_history],
            ground_truth, [row[1:3] for row in truth_history],
            state_update, projected, selected, plan,
        )
        self._localization_chart(canvas, (20, 675, 730, 385), confidence=True)
        self._localization_chart(canvas, (762, 675, 730, 385), confidence=False)
        self._freshness_panel(canvas, (1504, 675, 396, 385))
        return canvas


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=repo / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz")
    parser.add_argument("--world", type=Path, default=repo / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf")
    parser.add_argument(
        "--gp-root", type=Path,
        default=repo / "logs/visibility_comparison/spawn_grid_20260727/gp",
        help="Per-camera fitted GP directory (used as a ranking prior, not calibrated probability).",
    )
    parser.add_argument("--start", nargs=2, type=float, default=(-5.5, -2.5), metavar=("X", "Y"))
    parser.add_argument("--start-yaw", type=float, default=-1.5708)
    parser.add_argument("--goal", nargs=2, type=float, default=(5.5, 8.0), metavar=("X", "Y"))
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = LiveDashboardNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    renderer = DashboardRenderer(
        args.artifact, args.world,
        args.start, args.goal, args.start_yaw, args.gp_root,
    )
    window = "Four-camera navigation dashboard"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window, 1600, 900)
    try:
        while rclpy.ok():
            cv2.imshow(window, renderer.render(node.snapshot()))
            key = cv2.waitKey(33) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
