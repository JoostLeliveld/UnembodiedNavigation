#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from state.core.pixel_to_bev import PixelToBevTransformer
from state.core.noise import build_covariance


class PixelToBevStateNode(Node):
    """Convert homography pixel measurements into a BEV state belief."""

    def __init__(self):
        super().__init__('pixel_to_bev_state_node')

        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('transform_noise_sigma', 0.0)
        self.declare_parameter('seed', 0)

        self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        self.declare_parameter('img_width', 1280)
        self.declare_parameter('img_height', 720)
        self.declare_parameter('fov_h_rad', 1.5708)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.pixel_noise_sigma = float(self.get_parameter('pixel_noise_sigma').value)
        self.transform_noise_sigma = float(self.get_parameter('transform_noise_sigma').value)
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

        self.get_logger().info(
            'Pixel->BEV state node started (/perception/pixel_pose -> /state/bev)'
        )

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

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.orientation = msg.pose.orientation
        out.pose.covariance = build_covariance(sigma_x, sigma_y, 1e-6)
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
