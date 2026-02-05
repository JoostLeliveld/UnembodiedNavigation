import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from perception.core.homography import HomographyModel
from perception.core.camera_config import load_camera_params


class HomographySimNode(Node):
    def __init__(self):
        super().__init__('homography_sim_node',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        self.declare_parameter('img_width', 1280)
        self.declare_parameter('img_height', 720)
        self.declare_parameter('fov_h_rad', 1.5708)

        self.cam_pos, self.look_at, self.img_width, self.img_height, self.fov_h_rad = load_camera_params(self)
        self.model = HomographyModel(
            cam_pos=self.cam_pos,
            look_at=self.look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h_rad,
        )

        self.pixel_noise_std = 0.0
        self._verify_camera_setup()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)

        self.log_counter = 0
        self.get_logger().info('Homography Sim Node started')

    def _verify_camera_setup(self):
        direction = self.look_at - self.cam_pos
        horiz_dist = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        pitch_deg = np.degrees(np.arctan2(-direction[2], horiz_dist))
        yaw_deg = np.degrees(np.arctan2(direction[1], direction[0]))

        self.get_logger().info('=' * 50)
        self.get_logger().info('Camera Configuration (Oblique View):')
        self.get_logger().info(f'  Position (odom): ({self.cam_pos[0]}, {self.cam_pos[1]}, {self.cam_pos[2]})')
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
        x_true = msg.pose.pose.position.x
        y_true = msg.pose.pose.position.y

        u, v, visible = self.model.world_to_pixel(x_true, y_true)
        if not visible:
            self.log_counter += 1
            if self.log_counter % 50 == 0:
                self.get_logger().warn(
                    f'Robot at ({x_true:.2f}, {y_true:.2f}) is OUT OF CAMERA VIEW'
                )
            return

        out_msg = PoseStamped()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'image'
        out_msg.pose.position.x = float(u)
        out_msg.pose.position.y = float(v)
        out_msg.pose.position.z = 0.0
        out_msg.pose.orientation = msg.pose.pose.orientation
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
