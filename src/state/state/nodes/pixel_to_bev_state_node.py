#!/usr/bin/env python3
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

from state.core.pixel_to_bev import PixelToBevTransformer
from state.core.noise import build_covariance


class OracleStateSource:
    """Oracle state adapter: /odom -> /state/bev."""

    def __init__(self, node: Node, publisher, frame_id: str):
        self._node = node
        self._publisher = publisher
        self._frame_id = frame_id
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        self._last_tf_warn_wall = 0.0
        self._node.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self._node.get_logger().info(
            'Pixel->BEV state node in ORACLE mode (/odom -> TF -> /state/bev)'
        )

    def _odom_callback(self, msg: Odometry):
        source_frame = (msg.header.frame_id or 'odom').strip() or 'odom'
        pose_in = PoseStamped()
        pose_in.header = msg.header
        pose_in.header.frame_id = source_frame
        pose_in.pose = msg.pose.pose

        pose_out = pose_in.pose
        if source_frame != self._frame_id:
            try:
                # Use latest available transform; the map_bev->odom transform is static.
                tf_msg = self._tf_buffer.lookup_transform(
                    self._frame_id,
                    source_frame,
                    rclpy.time.Time(),
                )
                pose_out = do_transform_pose(pose_in.pose, tf_msg)
            except Exception as exc:
                now = time.monotonic()
                if (now - self._last_tf_warn_wall) > 1.0:
                    self._last_tf_warn_wall = now
                    self._node.get_logger().warn(
                        f"Oracle TF transform {source_frame}->{self._frame_id} unavailable: {exc}"
                    )
                return

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._frame_id
        out.pose.pose = pose_out
        out.pose.covariance = build_covariance(1e-6, 1e-6, 1e-6)
        self._publisher.publish(out)


class PixelStateSource:
    """Pixel state adapter: /perception/pixel_pose -> /state/bev."""

    def __init__(
        self,
        node: Node,
        publisher,
        frame_id: str,
        pixel_noise_sigma: float,
        transform_noise_sigma: float,
        transformer: PixelToBevTransformer,
        rng,
    ):
        self._node = node
        self._publisher = publisher
        self._frame_id = frame_id
        self._pixel_noise_sigma = float(pixel_noise_sigma)
        self._transform_noise_sigma = float(transform_noise_sigma)
        self._transformer = transformer
        self._rng = rng
        self._node.create_subscription(
            PoseStamped, '/perception/pixel_pose', self._pixel_callback, 10
        )
        self._node.get_logger().info(
            'Pixel->BEV state node in PIXEL mode (/perception/pixel_pose -> /state/bev)'
        )

    def _pixel_callback(self, msg: PoseStamped):
        u = msg.pose.position.x
        v = msg.pose.position.y
        if self._pixel_noise_sigma > 0.0:
            u += self._rng.normal(0.0, self._pixel_noise_sigma)
            v += self._rng.normal(0.0, self._pixel_noise_sigma)

        world = self._transformer.pixel_to_world(
            u,
            v,
            transform_noise_sigma=self._transform_noise_sigma,
        )
        if world is None:
            return

        x, y = world
        sigma_x = self._transform_noise_sigma
        sigma_y = self._transform_noise_sigma
        if self._pixel_noise_sigma > 0.0:
            sigmas = self._transformer.pixel_noise_to_metric(
                u,
                v,
                self._pixel_noise_sigma,
                transform_noise_sigma=self._transform_noise_sigma,
            )
            if sigmas is not None:
                sigma_x, sigma_y = sigmas

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._frame_id
        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.orientation = msg.pose.orientation
        out.pose.covariance = build_covariance(sigma_x, sigma_y, 1e-6)
        self._publisher.publish(out)


class PixelToBevStateNode(Node):
    def __init__(self):
        super().__init__('pixel_to_bev_state_node')

        self.declare_parameter('state_source', 'oracle')
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('transform_noise_sigma', 0.0)
        self.declare_parameter('seed', 0)

        # Camera parameters (used in pixel mode).
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

        self.publisher = self.create_publisher(PoseWithCovarianceStamped, '/state/bev', 10)

        if self.state_source == 'oracle':
            self.source = OracleStateSource(self, self.publisher, self.frame_id)
        elif self.state_source == 'pixel':
            cam_pos = np.array(self.get_parameter('cam_pos').value, dtype=float)
            look_at = np.array(self.get_parameter('look_at').value, dtype=float)
            img_width = int(self.get_parameter('img_width').value)
            img_height = int(self.get_parameter('img_height').value)
            fov_h = float(self.get_parameter('fov_h_rad').value)
            rng = np.random.default_rng(self.seed)
            transformer = PixelToBevTransformer(
                cam_pos=cam_pos,
                look_at=look_at,
                img_width=img_width,
                img_height=img_height,
                fov_h_rad=fov_h,
                rng=rng,
            )
            self.source = PixelStateSource(
                node=self,
                publisher=self.publisher,
                frame_id=self.frame_id,
                pixel_noise_sigma=self.pixel_noise_sigma,
                transform_noise_sigma=self.transform_noise_sigma,
                transformer=transformer,
                rng=rng,
            )
        else:
            self.get_logger().error(
                f"Invalid state_source '{self.state_source}'. Use 'oracle' or 'pixel'."
            )
            self.source = None


def main(args=None):
    rclpy.init(args=args)
    node = PixelToBevStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
