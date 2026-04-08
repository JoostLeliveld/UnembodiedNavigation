#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from state.core.pixel_to_bev import PixelToBevTransformer
from state.core.noise import build_covariance


class PixelToBevStateNode(Node):
    """Convert homography pixel measurements into a BEV state belief."""

    def __init__(self):
        super().__init__('pixel_to_bev_state_node')

        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('transform_noise_sigma', 0.0)
        self.declare_parameter('heading_pixel_noise_sigma', 0.0)
        self.declare_parameter('yaw_noise_floor_rad', 0.01)
        self.declare_parameter('use_odom_heading_fallback', True)
        self.declare_parameter('odom_heading_timeout_s', 0.5)
        self.declare_parameter('odom_heading_sigma_rad', 0.08)
        self.declare_parameter('infer_yaw_from_motion', True)
        self.declare_parameter('motion_yaw_min_displacement_m', 0.03)
        self.declare_parameter('motion_yaw_sigma_rad', 0.35)
        self.declare_parameter('seed', 0)

        self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        self.declare_parameter('img_width', 1280)
        self.declare_parameter('img_height', 720)
        self.declare_parameter('fov_h_rad', 1.5708)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.pixel_noise_sigma = float(self.get_parameter('pixel_noise_sigma').value)
        self.transform_noise_sigma = float(self.get_parameter('transform_noise_sigma').value)
        self.heading_pixel_noise_sigma = float(self.get_parameter('heading_pixel_noise_sigma').value)
        self.yaw_noise_floor_rad = float(self.get_parameter('yaw_noise_floor_rad').value)
        self.use_odom_heading_fallback = bool(self.get_parameter('use_odom_heading_fallback').value)
        self.odom_heading_timeout_s = float(self.get_parameter('odom_heading_timeout_s').value)
        self.odom_heading_sigma_rad = float(self.get_parameter('odom_heading_sigma_rad').value)
        self.infer_yaw_from_motion = bool(self.get_parameter('infer_yaw_from_motion').value)
        self.motion_yaw_min_displacement_m = float(self.get_parameter('motion_yaw_min_displacement_m').value)
        self.motion_yaw_sigma_rad = float(self.get_parameter('motion_yaw_sigma_rad').value)
        self.seed = int(self.get_parameter('seed').value)

        cam_pos = np.array(self.get_parameter('cam_pos').value, dtype=float)
        look_at = np.array(self.get_parameter('look_at').value, dtype=float)
        img_width = int(self.get_parameter('img_width').value)
        img_height = int(self.get_parameter('img_height').value)
        fov_h = float(self.get_parameter('fov_h_rad').value)

        self._rng = np.random.default_rng(self.seed)
        self._transformer = PixelToBevTransformer(
            cam_pos=cam_pos,
            look_at=look_at,
            img_width=img_width,
            img_height=img_height,
            fov_h_rad=fov_h,
            rng=self._rng,
        )
        self._publisher = self.create_publisher(PoseWithCovarianceStamped, '/state/bev', 10)
        self.create_subscription(PoseStamped, '/perception/pixel_pose', self._pixel_callback, 10)
        self.create_subscription(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._diag_callback, 10)
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self._latest_diag = None
        self._latest_odom_yaw = None
        self._latest_odom_stamp = None
        self._last_xy = None
        self._last_yaw = 0.0

        theta_sources = ['pixel heading']
        if self.use_odom_heading_fallback:
            theta_sources.append('odom fallback')
        if self.infer_yaw_from_motion:
            theta_sources.append('motion fallback')
        self.get_logger().info(
            'Pixel->BEV state node started '
            '(/perception/pixel_pose -> /state/bev; '
            f'state sources: x,y=camera homography, theta={" -> ".join(theta_sources)})'
        )

    @staticmethod
    def _stamp_to_float(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _diag_callback(self, msg: Float64MultiArray):
        try:
            self._latest_diag = diagnostics_from_message(msg)
        except Exception:
            return

    def _odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._latest_odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._latest_odom_stamp = msg.header.stamp

    @staticmethod
    def _yaw_to_quaternion(yaw: float):
        half = 0.5 * float(yaw)
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def _matching_diag(self, stamp):
        diag = self._latest_diag
        if not diag:
            return None
        stamp_s = self._stamp_to_float(stamp)
        diag_stamp = float(diag.get('stamp', math.nan))
        if (not math.isfinite(diag_stamp)) or abs(diag_stamp - stamp_s) > 1e-3:
            return None
        return diag

    def _fresh_odom_yaw(self, stamp):
        if self._latest_odom_yaw is None or self._latest_odom_stamp is None:
            return None
        stamp_s = self._stamp_to_float(stamp)
        odom_stamp_s = self._stamp_to_float(self._latest_odom_stamp)
        if not math.isfinite(odom_stamp_s):
            return None
        if self.odom_heading_timeout_s > 0.0 and abs(odom_stamp_s - stamp_s) > self.odom_heading_timeout_s:
            return None
        return float(self._latest_odom_yaw)

    def _heading_sigma_from_diag(self, diag) -> float:
        sigma = float(max(self.yaw_noise_floor_rad, 1e-6))
        if not diag:
            return sigma
        sep = float(diag.get('separation_px', math.nan))
        if not math.isfinite(sep) or sep <= 1e-6:
            return sigma
        pixel_sigma = max(float(self.heading_pixel_noise_sigma), 1e-6)
        sigma_sep = math.sqrt(2.0) * pixel_sigma / max(sep, 1.0)
        return float(max(sigma, sigma_sep))

    def _pixel_callback(self, msg: PoseStamped):
        u = float(msg.pose.position.x)
        v = float(msg.pose.position.y)
        if self.pixel_noise_sigma > 0.0:
            u += float(self._rng.normal(0.0, self.pixel_noise_sigma))
            v += float(self._rng.normal(0.0, self.pixel_noise_sigma))

        world = self._transformer.pixel_to_world(
            u,
            v,
            transform_noise_sigma=self.transform_noise_sigma,
        )
        if world is None:
            return

        x, y = world
        sigma_x = self.transform_noise_sigma
        sigma_y = self.transform_noise_sigma
        if self.pixel_noise_sigma > 0.0:
            sigmas = self._transformer.pixel_noise_to_metric(
                u,
                v,
                self.pixel_noise_sigma,
                transform_noise_sigma=self.transform_noise_sigma,
            )
            if sigmas is not None:
                sigma_x, sigma_y = sigmas
        diag = self._matching_diag(msg.header.stamp)

        yaw_out = self._last_yaw
        sigma_yaw = float(max(self.motion_yaw_sigma_rad, self.yaw_noise_floor_rad))
        if diag and bool(diag.get('detected', False)) and math.isfinite(float(diag.get('yaw_est', math.nan))):
            q = msg.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw_out = math.atan2(siny_cosp, cosy_cosp)
            sigma_yaw = self._heading_sigma_from_diag(diag)
        else:
            odom_applied = False
            if self.use_odom_heading_fallback:
                odom_yaw = self._fresh_odom_yaw(msg.header.stamp)
                if odom_yaw is not None:
                    yaw_out = odom_yaw
                    sigma_yaw = float(max(self.odom_heading_sigma_rad, self.yaw_noise_floor_rad))
                    odom_applied = True
            if (not odom_applied) and self.infer_yaw_from_motion and self._last_xy is not None:
                dx = float(x - self._last_xy[0])
                dy = float(y - self._last_xy[1])
                disp = math.hypot(dx, dy)
                if disp >= self.motion_yaw_min_displacement_m:
                    yaw_out = math.atan2(dy, dx)
        qx, qy, qz, qw = self._yaw_to_quaternion(yaw_out)
        self._last_xy = np.array([x, y], dtype=float)
        self._last_yaw = float(yaw_out)

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.pose.covariance = build_covariance(sigma_x, sigma_y, sigma_yaw)
        self._publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PixelToBevStateNode()
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
