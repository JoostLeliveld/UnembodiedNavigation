import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

from perception.core.homography import HomographyModel
from perception.core.camera_config import load_camera_params


class HomographySimNode(Node):
    def __init__(self):
        super().__init__('homography_sim_node',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

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
        self.rng = np.random.default_rng(self.seed)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._last_tf_warn_wall = 0.0
        self._verify_camera_setup()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)

        self.log_counter = 0
        self.get_logger().info(
            f'Homography Sim Node started (pixel_noise_sigma={self.pixel_noise_std:.3f}, '
            f'world_frame={self.world_frame})'
        )

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

        u, v, visible = self.model.world_to_pixel(x_true, y_true)
        if not visible:
            self.log_counter += 1
            if self.log_counter % 50 == 0:
                self.get_logger().warn(
                    f'Robot at ({x_true:.2f}, {y_true:.2f}) is OUT OF CAMERA VIEW'
                )
            return

        u_pub = float(u)
        v_pub = float(v)
        if self.pixel_noise_std > 0.0:
            u_pub += float(self.rng.normal(0.0, self.pixel_noise_std))
            v_pub += float(self.rng.normal(0.0, self.pixel_noise_std))

        out_msg = PoseStamped()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'image'
        out_msg.pose.position.x = u_pub
        out_msg.pose.position.y = v_pub
        out_msg.pose.position.z = 0.0
        # Preserve orientation in the same world frame used for projection.
        out_msg.pose.orientation = pose_world.orientation
        self.pub.publish(out_msg)

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            self.get_logger().info(
                f"True:({x_true:.2f},{y_true:.2f}) -> "
                f"Pix:({u:.0f},{v:.0f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = HomographySimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
