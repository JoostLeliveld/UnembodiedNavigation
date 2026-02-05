#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from state.core.pixel_to_bev import PixelToBevTransformer
from state.core.noise import build_covariance


class PixelToBevStateNode(Node):
    def __init__(self):
        super().__init__('pixel_to_bev_state_node')

        self.declare_parameter('state_source', 'oracle')
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('transform_noise_sigma', 0.0)
        self.declare_parameter('seed', 0)

        # Camera parameters
        self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        self.declare_parameter('img_width', 1280)
        self.declare_parameter('img_height', 720)
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
        self.transformer = PixelToBevTransformer(
            cam_pos=self.cam_pos,
            look_at=self.look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h,
            rng=self.rng,
        )

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

    def _odom_callback(self, msg: Odometry):
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose = msg.pose.pose
        out.pose.covariance = build_covariance(1e-6, 1e-6, 1e-6)
        self.publisher.publish(out)

    def _pixel_callback(self, msg: PoseStamped):
        u = msg.pose.position.x
        v = msg.pose.position.y
        if self.pixel_noise_sigma > 0.0:
            u += self.rng.normal(0.0, self.pixel_noise_sigma)
            v += self.rng.normal(0.0, self.pixel_noise_sigma)

        world = self.transformer.pixel_to_world(
            u,
            v,
            transform_noise_sigma=self.transform_noise_sigma,
        )
        if world is None:
            return

        x, y = world
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        sigma_x = self.transform_noise_sigma
        sigma_y = self.transform_noise_sigma
        if self.pixel_noise_sigma > 0.0:
            sigmas = self.transformer.pixel_noise_to_metric(
                u,
                v,
                self.pixel_noise_sigma,
                transform_noise_sigma=self.transform_noise_sigma,
            )
            if sigmas is not None:
                sigma_x, sigma_y = sigmas

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.orientation = msg.pose.orientation
        out.pose.covariance = build_covariance(sigma_x, sigma_y, 1e-6)
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PixelToBevStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
