#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from perception.core.camera_config import load_camera_params
from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_message,
)
from perception.core.homography import HomographyModel


class ColorPoseDetectorNode(Node):
    """Detect red-front / blue-rear markers and publish image midpoint + yaw."""

    def __init__(self):
        super().__init__('color_pose_detector_node')

        self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        self.declare_parameter('img_width', 1280)
        self.declare_parameter('img_height', 720)
        self.declare_parameter('fov_h_rad', 1.5708)

        self.declare_parameter('image_topic', '/external_camera/image_raw')
        self.declare_parameter('red_min_r', 130)
        self.declare_parameter('red_max_g', 150)
        self.declare_parameter('red_max_b', 150)
        self.declare_parameter('red_min_margin', 20)
        self.declare_parameter('blue_min_b', 130)
        self.declare_parameter('blue_max_r', 150)
        self.declare_parameter('blue_max_g', 150)
        self.declare_parameter('blue_min_margin', 20)
        self.declare_parameter('min_marker_pixels', 60)
        self.declare_parameter('red_min_pixels', -1)
        self.declare_parameter('blue_min_pixels', -1)
        self.declare_parameter('min_marker_separation_px', 8.0)
        self.declare_parameter('log_every_n', 30)

        (
            self.cam_pos,
            self.look_at,
            self.img_width,
            self.img_height,
            self.fov_h_rad,
        ) = load_camera_params(self)
        self.model = HomographyModel(
            cam_pos=self.cam_pos,
            look_at=self.look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h_rad,
        )

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.red_min_r = int(self.get_parameter('red_min_r').value)
        self.red_max_g = int(self.get_parameter('red_max_g').value)
        self.red_max_b = int(self.get_parameter('red_max_b').value)
        self.red_min_margin = int(self.get_parameter('red_min_margin').value)
        self.blue_min_b = int(self.get_parameter('blue_min_b').value)
        self.blue_max_r = int(self.get_parameter('blue_max_r').value)
        self.blue_max_g = int(self.get_parameter('blue_max_g').value)
        self.blue_min_margin = int(self.get_parameter('blue_min_margin').value)
        self.min_marker_pixels = int(self.get_parameter('min_marker_pixels').value)
        self.red_min_pixels = int(self.get_parameter('red_min_pixels').value)
        self.blue_min_pixels = int(self.get_parameter('blue_min_pixels').value)
        if self.red_min_pixels <= 0:
            self.red_min_pixels = self.min_marker_pixels
        if self.blue_min_pixels <= 0:
            # Blue rear marker is often smaller in image; use a lower default floor.
            self.blue_min_pixels = min(self.min_marker_pixels, 20)
        self.min_marker_separation_px = float(
            self.get_parameter('min_marker_separation_px').value
        )
        self.log_every_n = max(1, int(self.get_parameter('log_every_n').value))

        self.bridge = CvBridge()
        self.pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)
        self.diag_pub = self.create_publisher(
            Float64MultiArray,
            DETECTION_DIAGNOSTICS_TOPIC,
            10,
        )
        self.create_subscription(
            Image,
            self.image_topic,
            self._image_cb,
            qos_profile_sensor_data,
        )

        self._frame_counter = 0
        self._miss_counter = 0
        self._pub_counter = 0
        self._last_yaw = 0.0

        self.get_logger().info(
            f'color_pose_detector_node started (topic={self.image_topic}, '
            f'min_pixels(red={self.red_min_pixels}, blue={self.blue_min_pixels}), '
            f'min_sep_px={self.min_marker_separation_px:.1f})'
        )

    @staticmethod
    def _mask_centroid(mask: np.ndarray, min_pixels: int) -> Optional[Tuple[float, float, int]]:
        ys, xs = np.nonzero(mask)
        count = int(xs.size)
        if count < min_pixels:
            return None
        return float(xs.mean()), float(ys.mean()), count

    def _decode_rgb(self, msg: Image) -> Optional[np.ndarray]:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warn(f'Failed to decode image: {exc}')
            return None

        encoding = str(msg.encoding).lower()
        if encoding == 'rgb8':
            rgb = image
        elif encoding == 'rgba8':
            rgb = image[..., :3]
        elif encoding == 'bgr8':
            rgb = image[..., [2, 1, 0]]
        elif encoding == 'bgra8':
            rgb = image[..., [2, 1, 0]]
        elif encoding == 'mono8':
            rgb = np.repeat(image[:, :, None], repeats=3, axis=2)
        else:
            try:
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            except Exception as exc:
                self.get_logger().warn(
                    f"Unsupported image encoding '{msg.encoding}': {exc}"
                )
                return None

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            self.get_logger().warn(
                f"Unexpected image shape {tuple(rgb.shape)} for encoding '{msg.encoding}'"
            )
            return None
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def _build_masks(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rgb_i16 = rgb.astype(np.int16, copy=False)
        red = rgb_i16[:, :, 0]
        green = rgb_i16[:, :, 1]
        blue = rgb_i16[:, :, 2]

        red_mask = (
            (red >= self.red_min_r)
            & (green <= self.red_max_g)
            & (blue <= self.red_max_b)
            & ((red - np.maximum(green, blue)) >= self.red_min_margin)
        )
        blue_mask = (
            (blue >= self.blue_min_b)
            & (red <= self.blue_max_r)
            & (green <= self.blue_max_g)
            & ((blue - np.maximum(red, green)) >= self.blue_min_margin)
        )
        return red_mask, blue_mask

    def _image_cb(self, msg: Image):
        rgb = self._decode_rgb(msg)
        if rgb is None:
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._frame_counter += 1
        red_mask, blue_mask = self._build_masks(rgb)
        red = self._mask_centroid(red_mask, self.red_min_pixels)
        blue = self._mask_centroid(blue_mask, self.blue_min_pixels)
        red_px = int(np.count_nonzero(red_mask))
        blue_px = int(np.count_nonzero(blue_mask))

        u_red = math.nan
        v_red = math.nan
        u_blue = math.nan
        v_blue = math.nan
        area_red = red_px
        area_blue = blue_px
        u_mid = math.nan
        v_mid = math.nan
        separation_px = math.nan
        border_margin_px = math.nan
        yaw = math.nan
        detected = False

        if red is None or blue is None:
            self._miss_counter += 1
            if red is not None:
                u_red, v_red, area_red = red
            if blue is not None:
                u_blue, v_blue, area_blue = blue
            if red is not None and blue is not None:
                u_mid = float(0.5 * (u_red + u_blue))
                v_mid = float(0.5 * (v_red + v_blue))
                separation_px = float(math.hypot(u_red - u_blue, v_red - v_blue))
                border_margin_px = float(
                    min(
                        u_mid,
                        v_mid,
                        self.img_width - 1.0 - u_mid,
                        self.img_height - 1.0 - v_mid,
                    )
                )
            self.diag_pub.publish(
                diagnostics_message(
                    stamp=stamp,
                    detected=False,
                    u_mid=u_mid,
                    v_mid=v_mid,
                    yaw_est=yaw,
                    u_red=u_red,
                    v_red=v_red,
                    red_area_px=area_red,
                    u_blue=u_blue,
                    v_blue=v_blue,
                    blue_area_px=area_blue,
                    separation_px=separation_px,
                    border_margin_px=border_margin_px,
                )
            )
            if self._frame_counter % self.log_every_n == 0:
                self.get_logger().info(
                    f'Color miss ({self._miss_counter}/{self._frame_counter}); '
                    f"red_px={red_px} blue_px={blue_px}"
                )
            return

        u_red, v_red, area_red = red
        u_blue, v_blue, area_blue = blue
        u_mid = float(0.5 * (u_red + u_blue))
        v_mid = float(0.5 * (v_red + v_blue))
        separation_px = float(math.hypot(u_red - u_blue, v_red - v_blue))
        border_margin_px = float(
            min(
                u_mid,
                v_mid,
                self.img_width - 1.0 - u_mid,
                self.img_height - 1.0 - v_mid,
            )
        )
        if separation_px < self.min_marker_separation_px:
            self.diag_pub.publish(
                diagnostics_message(
                    stamp=stamp,
                    detected=False,
                    u_mid=u_mid,
                    v_mid=v_mid,
                    yaw_est=yaw,
                    u_red=u_red,
                    v_red=v_red,
                    red_area_px=area_red,
                    u_blue=u_blue,
                    v_blue=v_blue,
                    blue_area_px=area_blue,
                    separation_px=separation_px,
                    border_margin_px=border_margin_px,
                )
            )
            return

        world_red = self.model.pixel_to_world(u_red, v_red)
        world_blue = self.model.pixel_to_world(u_blue, v_blue)
        if world_red is not None and world_blue is not None:
            yaw = math.atan2(
                world_red[1] - world_blue[1],
                world_red[0] - world_blue[0],
            )
            self._last_yaw = yaw
        else:
            yaw = self._last_yaw
        detected = True

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'image'
        out.pose.position.x = u_mid
        out.pose.position.y = v_mid
        out.pose.position.z = 0.0
        out.pose.orientation.x = 0.0
        out.pose.orientation.y = 0.0
        out.pose.orientation.z = math.sin(0.5 * yaw)
        out.pose.orientation.w = math.cos(0.5 * yaw)
        self.pub.publish(out)
        self.diag_pub.publish(
            diagnostics_message(
                stamp=stamp,
                detected=detected,
                u_mid=u_mid,
                v_mid=v_mid,
                yaw_est=yaw,
                u_red=u_red,
                v_red=v_red,
                red_area_px=area_red,
                u_blue=u_blue,
                v_blue=v_blue,
                blue_area_px=area_blue,
                separation_px=separation_px,
                border_margin_px=border_margin_px,
            )
        )

        self._pub_counter += 1
        if self._pub_counter % self.log_every_n == 0:
            self.get_logger().info(
                'Color pose -> '
                f'front_red=({u_red:.1f},{v_red:.1f},A={area_red}) '
                f'rear_blue=({u_blue:.1f},{v_blue:.1f},A={area_blue}) '
                f'yaw={yaw:.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ColorPoseDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
