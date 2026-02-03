#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


class PixelToBevStateNode(Node):
    def __init__(self):
        super().__init__('pixel_to_bev_state_node')

        self.declare_parameter('state_source', 'oracle')
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('transform_noise_sigma', 0.0)
        self.declare_parameter('seed', 0)

        # Camera parameters (defaults match external_camera/model.sdf)
        self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        self.declare_parameter('img_width', 1920)
        self.declare_parameter('img_height', 1080)
        self.declare_parameter('fov_h_rad', 1.5708)

        self.state_source = self.get_parameter('state_source').value
        self.frame_id = self.get_parameter('frame_id').value
        self.pixel_noise_sigma = float(self.get_parameter('pixel_noise_sigma').value)
        self.transform_noise_sigma = float(self.get_parameter('transform_noise_sigma').value)
        self.seed = int(self.get_parameter('seed').value)

        self.cam_pos = np.array(self.get_parameter('cam_pos').value, dtype=float)
        self.look_at = np.array(self.get_parameter('look_at').value, dtype=float)
        self.img_width = int(self.get_parameter('img_width').value)
        self.img_height = int(self.get_parameter('img_height').value)
        self.fov_h = float(self.get_parameter('fov_h_rad').value)

        self.rng = np.random.default_rng(self.seed)
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, '/state/bev', 10)

        if self.state_source == 'oracle':
            self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
            self.get_logger().info('Pixel->BEV state node in ORACLE mode (/odom -> /state/bev)')
        elif self.state_source == 'pixel':
            self.create_subscription(PoseStamped, '/perception/pixel_pose', self._pixel_callback, 10)
            self.get_logger().info('Pixel->BEV state node in PIXEL mode (/perception/pixel_pose -> /state/bev)')
        else:
            self.get_logger().error(
                f"Invalid state_source '{self.state_source}'. Use 'oracle' or 'pixel'."
            )

    def _build_covariance(self, sigma_x, sigma_y, sigma_yaw=1e-6):
        cov = [0.0] * 36
        cov[0] = float(sigma_x ** 2)
        cov[7] = float(sigma_y ** 2)
        cov[35] = float(sigma_yaw ** 2)
        return cov

    def _compute_intrinsics(self):
        f = (self.img_width / 2.0) / math.tan(self.fov_h / 2.0)
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    def _compute_lookat_rotation(self, cam_pos, look_at, up_hint=None):
        if up_hint is None:
            up_hint = np.array([0.0, 0.0, 1.0])
        z_cam = look_at - cam_pos
        z_cam = z_cam / np.linalg.norm(z_cam)
        x_cam = np.cross(z_cam, up_hint)
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        return np.array([x_cam, y_cam, z_cam])

    def _pixel_to_world(self, u, v):
        cam_pos = self.cam_pos.copy()
        look_at = self.look_at.copy()
        if self.transform_noise_sigma > 0.0:
            cam_pos += self.rng.normal(0.0, self.transform_noise_sigma, size=3)
            look_at += self.rng.normal(0.0, self.transform_noise_sigma, size=3)

        K = self._compute_intrinsics()
        R = self._compute_lookat_rotation(cam_pos, look_at)
        t = -R @ cam_pos
        H = K @ np.column_stack([R[:, 0], R[:, 1], t])
        H_inv = np.linalg.inv(H)

        world_h = H_inv @ np.array([u, v, 1.0])
        if abs(world_h[2]) < 1e-9:
            return None
        return world_h[0] / world_h[2], world_h[1] / world_h[2]

    def _odom_callback(self, msg: Odometry):
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose = msg.pose.pose
        out.pose.covariance = self._build_covariance(1e-6, 1e-6, 1e-6)
        self.publisher.publish(out)

    def _pixel_callback(self, msg: PoseStamped):
        u = msg.pose.position.x
        v = msg.pose.position.y
        if self.pixel_noise_sigma > 0.0:
            u += self.rng.normal(0.0, self.pixel_noise_sigma)
            v += self.rng.normal(0.0, self.pixel_noise_sigma)

        world = self._pixel_to_world(u, v)
        if world is None:
            return

        x, y = world
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Estimate metric noise from pixel noise by local finite differences
        sigma_x = self.transform_noise_sigma
        sigma_y = self.transform_noise_sigma
        if self.pixel_noise_sigma > 0.0:
            dx = self._pixel_to_world(u + self.pixel_noise_sigma, v)
            dy = self._pixel_to_world(u, v + self.pixel_noise_sigma)
            if dx is not None and dy is not None:
                sigma_x = math.hypot(dx[0] - x, dx[1] - y)
                sigma_y = math.hypot(dy[0] - x, dy[1] - y)

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.orientation = msg.pose.orientation
        out.pose.covariance = self._build_covariance(sigma_x, sigma_y, 1e-6)
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PixelToBevStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
