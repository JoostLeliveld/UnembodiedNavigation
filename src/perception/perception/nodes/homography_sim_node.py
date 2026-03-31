import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import Float64MultiArray

from perception.core.homography import HomographyModel
from perception.core.camera_config import load_camera_params
from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_message,
)
from unav_common.occlusion_geometry import scene_from_json, segment_occluded


class HomographySimNode(Node):
    def __init__(self):
        super().__init__('homography_sim_node',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)


        def _as_bool(value):
            if isinstance(value, bool):
                return value
            return str(value).lower() == 'true'

        if not self.has_parameter('cam_pos'):
            self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        if not self.has_parameter('look_at'):
            self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        if not self.has_parameter('img_width'):
            self.declare_parameter('img_width', 1280)
        if not self.has_parameter('img_height'):
            self.declare_parameter('img_height', 720)
        if not self.has_parameter('fov_h_rad'):
            self.declare_parameter('fov_h_rad', 1.5708)
        if not self.has_parameter('pixel_noise_sigma'):
            self.declare_parameter('pixel_noise_sigma', 0.0)
        if not self.has_parameter('seed'):
            self.declare_parameter('seed', 0)
        if not self.has_parameter('world_frame'):
            self.declare_parameter('world_frame', 'map_bev')
        if not self.has_parameter('log_noisy_pixels'):
            self.declare_parameter('log_noisy_pixels', False)
        if not self.has_parameter('use_visibility_model'):
            self.declare_parameter('use_visibility_model', False)
        if not self.has_parameter('visibility_model'):
            self.declare_parameter('visibility_model', 'raycast_25d')
        if not self.has_parameter('visibility_geometry_json'):
            self.declare_parameter('visibility_geometry_json', '')
        if not self.has_parameter('visibility_target_height_m'):
            self.declare_parameter('visibility_target_height_m', 0.0)
        if not self.has_parameter('use_geometry_occlusion'):
            self.declare_parameter('use_geometry_occlusion', True)
        if not self.has_parameter('log_camera_diagnostics'):
            self.declare_parameter('log_camera_diagnostics', False)
        if not self.has_parameter('heading_marker_separation_m'):
            self.declare_parameter('heading_marker_separation_m', 0.12)
        if not self.has_parameter('heading_marker_area_px'):
            self.declare_parameter('heading_marker_area_px', 36.0)

        self.cam_pos, self.look_at, self.img_width, self.img_height, self.fov_h_rad = load_camera_params(self)
        self.model = HomographyModel(
            cam_pos=self.cam_pos,
            look_at=self.look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h_rad,
        )

        self.pixel_noise_std = float(self.get_parameter('pixel_noise_sigma').value)
        self.seed = int(self.get_parameter('seed').value)
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.log_noisy_pixels = _as_bool(self.get_parameter('log_noisy_pixels').value)
        self.use_visibility_model = _as_bool(self.get_parameter('use_visibility_model').value)
        self.visibility_model_name = str(self.get_parameter('visibility_model').value).strip().lower()
        self.visibility_target_height_m = float(self.get_parameter('visibility_target_height_m').value)
        self.log_camera_diagnostics = _as_bool(self.get_parameter('log_camera_diagnostics').value)
        self.heading_marker_separation_m = float(self.get_parameter('heading_marker_separation_m').value)
        self.heading_marker_area_px = float(self.get_parameter('heading_marker_area_px').value)
        self.occlusion_scene = scene_from_json(str(self.get_parameter('visibility_geometry_json').value))
        self.use_geometry_occlusion = _as_bool(self.get_parameter('use_geometry_occlusion').value)
        self.rng = np.random.default_rng(self.seed)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._last_tf_warn_wall = 0.0
        if self.log_camera_diagnostics:
            self._verify_camera_setup()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)
        self.diag_pub = self.create_publisher(
            Float64MultiArray,
            DETECTION_DIAGNOSTICS_TOPIC,
            10,
        )

        self.log_counter = 0
        self.get_logger().info(
            f'Homography Sim Node started (pixel_noise_sigma={self.pixel_noise_std:.3f}, '
            f'world_frame={self.world_frame}, log_noisy_pixels={self.log_noisy_pixels}, '
            f'use_geometry_occlusion={self.use_geometry_occlusion}, '
            f'visibility_model={self.visibility_model_name}, '
            f'log_camera_diagnostics={self.log_camera_diagnostics}, '
            f'heading_marker_separation_m={self.heading_marker_separation_m:.3f}, '
            f'occluders={len(self.occlusion_scene.prisms)})'
        )

    @staticmethod
    def _yaw_to_quaternion(yaw: float):
        half = 0.5 * float(yaw)
        return 0.0, 0.0, float(np.sin(half)), float(np.cos(half))

    def _publish_detection_failure(self, stamp_msg, *, reason: str, x_true: float, y_true: float):
        self.diag_pub.publish(
            diagnostics_message(
                stamp=stamp_msg.sec + stamp_msg.nanosec * 1e-9,
                detected=False,
                u_mid=np.nan,
                v_mid=np.nan,
                yaw_est=np.nan,
                u_red=np.nan,
                v_red=np.nan,
                red_area_px=np.nan,
                u_blue=np.nan,
                v_blue=np.nan,
                blue_area_px=np.nan,
                separation_px=np.nan,
                border_margin_px=np.nan,
            )
        )
        self.log_counter += 1
        if self.log_counter % 50 == 0:
            self.get_logger().warn(
                f'Robot at ({x_true:.2f}, {y_true:.2f}) is {reason}'
            )

    def _marker_world_positions(self, x_world: float, y_world: float, yaw: float):
        half_sep = 0.5 * max(self.heading_marker_separation_m, 1e-3)
        dx = half_sep * float(np.cos(yaw))
        dy = half_sep * float(np.sin(yaw))
        red = np.array([x_world + dx, y_world + dy], dtype=float)
        blue = np.array([x_world - dx, y_world - dy], dtype=float)
        return red, blue

    def _verify_camera_setup(self):
        direction = self.look_at - self.cam_pos
        horiz_dist = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        pitch_deg = np.degrees(np.arctan2(-direction[2], horiz_dist))
        yaw_deg = np.degrees(np.arctan2(direction[1], direction[0]))

        self.get_logger().info('=' * 50)
        self.get_logger().info('Camera Configuration (Oblique View):')
        self.get_logger().info(
            f'  Position ({self.world_frame}): ({self.cam_pos[0]}, {self.cam_pos[1]}, {self.cam_pos[2]})'
        )
        self.get_logger().info(f'  Look-at point: ({self.look_at[0]}, {self.look_at[1]}, {self.look_at[2]})')
        self.get_logger().info(f'  Effective angles: Yaw={yaw_deg:.1f}°, Pitch={pitch_deg:.1f}° down')
        self.get_logger().info(f'  FOV: {np.degrees(self.fov_h_rad):.1f}° horizontal')
        self.get_logger().info(f'  Resolution: {self.img_width}x{self.img_height}')

        test_points = [
            (0.0, 0.0, 'Start (0,0)'),
            (3.0, 3.0, 'Goal (3,3)'),
            (1.5, 1.5, 'Midpoint (1.5,1.5)'),
        ]

        self.get_logger().info('Ground point projections:')
        for x, y, name in test_points:
            u, v, visible = self.model.world_to_pixel(x, y)
            status = 'IN VIEW' if visible else 'OUT OF VIEW'
            self.get_logger().info(f'  {name} -> pixel ({u:.0f}, {v:.0f}) [{status}]')

        self.get_logger().info('Round-trip accuracy test:')
        for x, y, name in test_points:
            u, v, visible = self.model.world_to_pixel(x, y)
            if visible:
                x_est, y_est = self.model.pixel_to_world(u, v)
                err = np.sqrt((x - x_est) ** 2 + (y - y_est) ** 2)
                self.get_logger().info(f'  {name}: error = {err:.6f}m')

        self.get_logger().info('=' * 50)

    def _is_occluded(self, x_world: float, y_world: float) -> bool:
        if not self.use_geometry_occlusion:
            return False
        target = np.array([float(x_world), float(y_world), float(self.visibility_target_height_m)], dtype=float)
        return bool(segment_occluded(self.occlusion_scene.prisms, np.asarray(self.cam_pos, dtype=float), target))

    def odom_callback(self, msg):
        source_frame = (msg.header.frame_id or 'odom').strip() or 'odom'
        pose_world = msg.pose.pose
        if source_frame != self.world_frame:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self.world_frame,
                    source_frame,
                    rclpy.time.Time(),
                )
                pose_world = do_transform_pose(msg.pose.pose, tf_msg)
            except Exception as exc:
                now = time.monotonic()
                if (now - self._last_tf_warn_wall) > 1.0:
                    self._last_tf_warn_wall = now
                    self.get_logger().warn(
                        f"Homography TF transform {source_frame}->{self.world_frame} unavailable: {exc}"
                    )
                return

        x_true = pose_world.position.x
        y_true = pose_world.position.y
        q = pose_world.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_true = float(np.arctan2(siny_cosp, cosy_cosp))
        red_world, blue_world = self._marker_world_positions(x_true, y_true, yaw_true)

        u_red, v_red, red_visible = self.model.world_to_pixel(float(red_world[0]), float(red_world[1]))
        u_blue, v_blue, blue_visible = self.model.world_to_pixel(float(blue_world[0]), float(blue_world[1]))
        if (not red_visible) or (not blue_visible):
            self._publish_detection_failure(
                msg.header.stamp,
                reason='OUT OF CAMERA VIEW',
                x_true=x_true,
                y_true=y_true,
            )
            return
        if self._is_occluded(float(red_world[0]), float(red_world[1])) or self._is_occluded(float(blue_world[0]), float(blue_world[1])):
            self._publish_detection_failure(
                msg.header.stamp,
                reason='OCCLUDED FROM CAMERA VIEW',
                x_true=x_true,
                y_true=y_true,
            )
            return

        u_red_pub = float(u_red)
        v_red_pub = float(v_red)
        u_blue_pub = float(u_blue)
        v_blue_pub = float(v_blue)
        if self.pixel_noise_std > 0.0:
            u_red_pub += float(self.rng.normal(0.0, self.pixel_noise_std))
            v_red_pub += float(self.rng.normal(0.0, self.pixel_noise_std))
            u_blue_pub += float(self.rng.normal(0.0, self.pixel_noise_std))
            v_blue_pub += float(self.rng.normal(0.0, self.pixel_noise_std))

        red_world_est = self.model.pixel_to_world(u_red_pub, v_red_pub)
        blue_world_est = self.model.pixel_to_world(u_blue_pub, v_blue_pub)
        if red_world_est is None or blue_world_est is None:
            self._publish_detection_failure(
                msg.header.stamp,
                reason='OUT OF HOMOGRAPHY FOOTPRINT',
                x_true=x_true,
                y_true=y_true,
            )
            return

        red_world_est = np.asarray(red_world_est, dtype=float).reshape(2)
        blue_world_est = np.asarray(blue_world_est, dtype=float).reshape(2)
        delta_world = red_world_est - blue_world_est
        if float(np.hypot(delta_world[0], delta_world[1])) <= 1e-6:
            self._publish_detection_failure(
                msg.header.stamp,
                reason='HEADING MARKERS COLLAPSED',
                x_true=x_true,
                y_true=y_true,
            )
            return

        yaw_est = float(np.arctan2(delta_world[1], delta_world[0]))
        u_pub = 0.5 * (u_red_pub + u_blue_pub)
        v_pub = 0.5 * (v_red_pub + v_blue_pub)
        separation_px = float(np.hypot(u_red_pub - u_blue_pub, v_red_pub - v_blue_pub))
        border_margin_px = float(
            min(
                u_pub,
                v_pub,
                self.img_width - 1.0 - u_pub,
                self.img_height - 1.0 - v_pub,
                u_red_pub,
                v_red_pub,
                self.img_width - 1.0 - u_red_pub,
                self.img_height - 1.0 - v_red_pub,
                u_blue_pub,
                v_blue_pub,
                self.img_width - 1.0 - u_blue_pub,
                self.img_height - 1.0 - v_blue_pub,
            )
        )

        self.diag_pub.publish(
            diagnostics_message(
                stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                detected=True,
                u_mid=u_pub,
                v_mid=v_pub,
                yaw_est=yaw_est,
                u_red=u_red_pub,
                v_red=v_red_pub,
                red_area_px=self.heading_marker_area_px,
                u_blue=u_blue_pub,
                v_blue=v_blue_pub,
                blue_area_px=self.heading_marker_area_px,
                separation_px=separation_px,
                border_margin_px=border_margin_px,
            )
        )
        out_msg = PoseStamped()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = 'image'
        out_msg.pose.position.x = u_pub
        out_msg.pose.position.y = v_pub
        out_msg.pose.position.z = 0.0
        qx, qy, qz, qw = self._yaw_to_quaternion(yaw_est)
        out_msg.pose.orientation.x = qx
        out_msg.pose.orientation.y = qy
        out_msg.pose.orientation.z = qz
        out_msg.pose.orientation.w = qw
        self.pub.publish(out_msg)

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            if self.log_noisy_pixels:
                self.get_logger().info(
                    f"True:({x_true:.2f},{y_true:.2f}) -> "
                    f"Pix_mid:({u_pub:.0f},{v_pub:.0f}) "
                    f"red=({u_red_pub:.0f},{v_red_pub:.0f}) "
                    f"blue=({u_blue_pub:.0f},{v_blue_pub:.0f}) "
                    f"yaw_est={np.degrees(yaw_est):.1f}deg"
                )
            else:
                self.get_logger().info(
                    f"True:({x_true:.2f},{y_true:.2f}) -> "
                    f"Pix_mid:({u_pub:.0f},{v_pub:.0f}) "
                    f"yaw_est={np.degrees(yaw_est):.1f}deg"
                )


def main(args=None):
    rclpy.init(args=args)
    node = HomographySimNode()
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
