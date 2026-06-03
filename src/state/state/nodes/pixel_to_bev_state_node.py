#!/usr/bin/env python3
import math
import time
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


HEADING_SOURCE_CODES = {
    'unknown': 0.0,
    'pixel_heading': 1.0,
    'odom_heading_fallback': 2.0,
    'motion_heading_fallback': 3.0,
    'held_previous_heading': 4.0,
    'keypoint_bev_heading': 5.0,
}


class PixelToBevStateNode(Node):
    """Convert homography pixel measurements into a BEV state belief."""

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))

    def __init__(self):
        super().__init__('pixel_to_bev_state_node')

        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('transform_noise_sigma', 0.0)
        self.declare_parameter('heading_pixel_noise_sigma', 0.0)
        self.declare_parameter('yaw_noise_floor_rad', 0.01)
        self.declare_parameter('use_odom_heading_fallback', True)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_heading_timeout_s', 0.5)
        self.declare_parameter('odom_heading_sigma_rad', 0.08)
        self.declare_parameter('odom_yaw_offset_rad', 0.0)
        self.declare_parameter('infer_yaw_from_motion', True)
        self.declare_parameter('motion_yaw_min_displacement_m', 0.03)
        self.declare_parameter('motion_yaw_sigma_rad', 0.35)
        self.declare_parameter('seed', 0)
        self.declare_parameter('keypoint_marker_world_z', 0.0)
        self.declare_parameter('keypoint_heading_sigma_rad', 0.05)
        self.declare_parameter('diagnostics_match_tolerance_s', 1e-3)
        # Empirical y-axis calibration offset.  The oblique camera geometry
        # causes the YOLO centroid back-projection to land systematically south
        # of the robot's true ground position.  This offset corrects that bias.
        # Derived from |pred_world_y - truth_y| over N=346 held-out detections
        # (F37v3 diagnostic run): mean y-bias = -0.127 m.  Apply +0.127 m here.
        # Default 0.0 preserves the original behaviour when not set.
        self.declare_parameter('bev_y_calibration_offset_m', 0.0)

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
        self.odom_topic = str(self.get_parameter('odom_topic').value or '/odom')
        self.odom_heading_timeout_s = float(self.get_parameter('odom_heading_timeout_s').value)
        self.odom_heading_sigma_rad = float(self.get_parameter('odom_heading_sigma_rad').value)
        self.odom_yaw_offset_rad = float(self.get_parameter('odom_yaw_offset_rad').value)
        self.infer_yaw_from_motion = bool(self.get_parameter('infer_yaw_from_motion').value)
        self.motion_yaw_min_displacement_m = float(self.get_parameter('motion_yaw_min_displacement_m').value)
        self.motion_yaw_sigma_rad = float(self.get_parameter('motion_yaw_sigma_rad').value)
        self.seed = int(self.get_parameter('seed').value)
        self.keypoint_marker_world_z = float(self.get_parameter('keypoint_marker_world_z').value)
        self.keypoint_heading_sigma_rad = float(self.get_parameter('keypoint_heading_sigma_rad').value)
        self.diagnostics_match_tolerance_s = float(
            self.get_parameter('diagnostics_match_tolerance_s').value
        )
        self.bev_y_calibration_offset_m = float(
            self.get_parameter('bev_y_calibration_offset_m').value
        )

        # Baseline measurement noise (pixels)
        self.R_visible_std = 2.5
        self.R_update_miss_std = 80.0

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
        self._heading_diag_pub = self.create_publisher(Float64MultiArray, '/state/heading_diagnostics', 10)
        self.create_subscription(PoseStamped, '/perception/pixel_pose', self._pixel_callback, 10)
        self.create_subscription(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._diag_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self._latest_diag = None
        self._latest_odom_yaw = None
        self._latest_odom_stamp = None
        self._last_xy = None
        self._last_yaw = 0.0
        self._last_diag_parse_warn_wall = 0.0
        self._last_heading_source = None

        theta_sources = ['pixel heading']
        if self.use_odom_heading_fallback:
            theta_sources.append('odom fallback')
        if self.infer_yaw_from_motion:
            theta_sources.append('motion fallback')
        self.get_logger().info(
            'Pixel->BEV state node started '
            '(/perception/pixel_pose -> /state/bev; '
            f'state sources: x,y=camera homography, theta={" -> ".join(theta_sources)}, '
            f'odom_topic={self.odom_topic}, '
            f'odom_yaw_offset_rad={self.odom_yaw_offset_rad:.3f}, '
            f'keypoint_marker_world_z={self.keypoint_marker_world_z:.3f}, '
            f'diag_match_tol_s={self.diagnostics_match_tolerance_s:.3f})'
        )

    @staticmethod
    def _stamp_to_float(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _diag_callback(self, msg: Float64MultiArray):
        try:
            self._latest_diag = diagnostics_from_message(msg)
        except (TypeError, ValueError, KeyError) as exc:
            now = time.monotonic()
            if (now - self._last_diag_parse_warn_wall) > 2.0:
                self._last_diag_parse_warn_wall = now
                self.get_logger().warn(
                    f"Rejected malformed detection diagnostics message: {type(exc).__name__}: {exc}"
                )
            return

    def _set_heading_source(self, source: str) -> None:
        source = str(source or '').strip() or 'unknown'
        if source == self._last_heading_source:
            return
        self._last_heading_source = source
        self.get_logger().info(f"State heading source switched to {source}")

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
        tolerance = max(float(self.diagnostics_match_tolerance_s), 1e-6)
        if (not math.isfinite(diag_stamp)) or abs(diag_stamp - stamp_s) > tolerance:
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
        return float(self._wrap_angle(self._latest_odom_yaw + self.odom_yaw_offset_rad))

    def _odom_age_s(self, stamp) -> float:
        if self._latest_odom_stamp is None:
            return math.nan
        try:
            return float(self._stamp_to_float(stamp) - self._stamp_to_float(self._latest_odom_stamp))
        except (AttributeError, TypeError, ValueError):
            return math.nan

    def _yaw_from_keypoints(self, diag):
        """Compute BEV heading from front/rear marker pixels.

        Returns (yaw_out, sigma_yaw) or None if disabled / data not finite.
        Disabled when keypoint_marker_world_z <= 0 so legacy seg-model runs are
        unaffected.
        """
        if self.keypoint_marker_world_z <= 0.0:
            return None
        if not bool(diag.get('detected', False)):
            return None
        u_front = float(diag.get('u_red', math.nan))
        v_front = float(diag.get('v_red', math.nan))
        u_rear = float(diag.get('u_blue', math.nan))
        v_rear = float(diag.get('v_blue', math.nan))
        if not (math.isfinite(u_front) and math.isfinite(v_front)
                and math.isfinite(u_rear) and math.isfinite(v_rear)):
            return None
        front_world = self._transformer.pixel_to_world_at_z(
            u_front, v_front, self.keypoint_marker_world_z,
            transform_noise_sigma=self.transform_noise_sigma,
        )
        rear_world = self._transformer.pixel_to_world_at_z(
            u_rear, v_rear, self.keypoint_marker_world_z,
            transform_noise_sigma=self.transform_noise_sigma,
        )
        if front_world is None or rear_world is None:
            return None
        dx = float(front_world[0] - rear_world[0])
        dy = float(front_world[1] - rear_world[1])
        if math.hypot(dx, dy) < 1e-4:
            return None
        yaw_out = math.atan2(dy, dx)
        sigma_yaw = float(max(self.keypoint_heading_sigma_rad, self.yaw_noise_floor_rad))
        return yaw_out, sigma_yaw

    def _heading_sigma_from_diag(self, diag) -> float:
        sigma = float(max(self.yaw_noise_floor_rad, 1e-6))
        if not diag:
            return sigma
        sep = float(diag.get('separation_px', math.nan))
        if not math.isfinite(sep) or sep <= 1e-6:
            return sigma
        pixel_sigma = max(self.pixel_noise_sigma, 1e-6)
        sigma_sep = math.sqrt(2.0) * pixel_sigma / max(sep, 1.0)
        return float(max(sigma, sigma_sep))

    def live_detection_trust(
        self,
        *,
        detected_after_threshold: bool,
        raw_score: float,
        selected_score: float,
        mask_available: bool,
        mask_area: float,
        bbox_area: float,
        selected_pixel_source_code: float,
    ) -> float:
        del mask_available, mask_area, bbox_area, selected_pixel_source_code
        if not detected_after_threshold:
            return 0.0

        score = float(selected_score if selected_score == selected_score else raw_score)
        score = max(0.0, min(1.0, score))
        return max(1e-4, min(1.0 - 1e-4, score))

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
        # Apply empirical y-axis calibration offset to correct the systematic
        # south-bias introduced by oblique camera back-projection.
        y += self.bev_y_calibration_offset_m
        sigma_x = self.transform_noise_sigma
        sigma_y = self.transform_noise_sigma

        diag = self._matching_diag(msg.header.stamp)

        # Base pixel noise starts at CLI default
        pixel_noise_sigma = self.pixel_noise_sigma

        if diag:
            trust_update = self.live_detection_trust(
                detected_after_threshold=bool(diag.get('yolo_detected_after_threshold', 0.0) >= 0.5),
                raw_score=float(diag.get('yolo_score_raw', 0.0)),
                selected_score=float(diag.get('yolo_score_selected', 0.0)),
                mask_available=bool(diag.get('mask_used', 0.0) >= 0.5),
                mask_area=float(diag.get('mask_area_px', 0.0)),
                bbox_area=float(diag.get('bbox_area_px', 0.0)),
                selected_pixel_source_code=float(diag.get('selected_pixel_source_code', 0.0)),
            )
            # R_update blending (Precision Blending)
            # This ensures that high-trust detections drop R rapidly.
            prec_visible = 1.0 / max(self.R_visible_std**2, 1e-6)
            prec_miss = 1.0 / max(self.R_update_miss_std**2, 1e-6)
            blended_prec = trust_update * prec_visible + (1.0 - trust_update) * prec_miss
            pixel_noise_sigma = math.sqrt(1.0 / blended_prec)

        if pixel_noise_sigma > 0.0:
            sigmas = self._transformer.pixel_noise_to_metric(
                u,
                v,
                pixel_noise_sigma,
                transform_noise_sigma=self.transform_noise_sigma,
            )
            if sigmas is not None:
                sigma_x, sigma_y = sigmas

        yaw_out = self._last_yaw
        sigma_yaw = float(
            max(self.motion_yaw_sigma_rad, self.odom_heading_sigma_rad, self.yaw_noise_floor_rad)
        )
        heading_source = 'held_previous_heading'
        pixel_yaw_meas = math.nan
        odom_yaw_used = math.nan
        motion_yaw_meas = math.nan
        diag_yaw_est = float(diag.get('yaw_est', math.nan)) if diag else math.nan
        diag_detected = 1.0 if (diag and bool(diag.get('detected', False))) else 0.0
        keypoint_yaw = self._yaw_from_keypoints(diag) if diag else None
        if keypoint_yaw is not None:
            yaw_out, sigma_yaw = keypoint_yaw
            pixel_yaw_meas = float(yaw_out)
            heading_source = 'keypoint_bev_heading'
        elif diag and bool(diag.get('detected', False)) and math.isfinite(float(diag.get('yaw_est', math.nan))):
            q = msg.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw_out = math.atan2(siny_cosp, cosy_cosp)
            pixel_yaw_meas = float(yaw_out)
            sigma_yaw = self._heading_sigma_from_diag(diag)
            heading_source = 'pixel_heading'
        else:
            odom_applied = False
            if self.use_odom_heading_fallback:
                odom_yaw = self._fresh_odom_yaw(msg.header.stamp)
                if odom_yaw is not None:
                    yaw_out = odom_yaw
                    odom_yaw_used = float(odom_yaw)
                    sigma_yaw = float(max(self.odom_heading_sigma_rad, self.yaw_noise_floor_rad))
                    odom_applied = True
                    heading_source = 'odom_heading_fallback'
            if (not odom_applied) and self.infer_yaw_from_motion and self._last_xy is not None:
                dx = float(x - self._last_xy[0])
                dy = float(y - self._last_xy[1])
                disp = math.hypot(dx, dy)
                if disp >= self.motion_yaw_min_displacement_m:
                    yaw_out = math.atan2(dy, dx)
                    motion_yaw_meas = float(yaw_out)
                    sigma_yaw = float(max(self.motion_yaw_sigma_rad, self.yaw_noise_floor_rad))
                    heading_source = 'motion_heading_fallback'
        self._set_heading_source(heading_source)
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

        diag_msg = Float64MultiArray()
        diag_msg.data = [
            self._stamp_to_float(msg.header.stamp),
            float(HEADING_SOURCE_CODES.get(heading_source, HEADING_SOURCE_CODES['unknown'])),
            float(pixel_yaw_meas),
            float(odom_yaw_used),
            float(motion_yaw_meas),
            float(yaw_out),
            float(sigma_yaw),
            float(diag_yaw_est),
            float(diag_detected),
            float(self._odom_age_s(msg.header.stamp)),
        ]
        self._heading_diag_pub.publish(diag_msg)


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
