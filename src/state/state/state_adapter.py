#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry


class StateAdapter(Node):
    def __init__(self):
        super().__init__('state_adapter',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.declare_parameter('state_source', 'oracle')
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('covariance_diag', [1e-6, 1e-6, 1e-6])

        self.state_source = self.get_parameter('state_source').value
        self.frame_id = self.get_parameter('frame_id').value
        self.cov_diag = list(self.get_parameter('covariance_diag').value)
        if len(self.cov_diag) < 3:
            self.cov_diag = [1e-6, 1e-6, 1e-6]

        self.publisher = self.create_publisher(PoseWithCovarianceStamped, '/state/bev', 10)
        self.covariance = self._build_covariance(self.cov_diag)

        if self.state_source == 'oracle':
            self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
            self.get_logger().info('State adapter in ORACLE mode (/odom -> /state/bev)')
        elif self.state_source == 'vision':
            self.create_subscription(PoseStamped, '/perception/robot_pose_cam', self._vision_callback, 10)
            self.get_logger().info('State adapter in VISION mode (/perception/robot_pose_cam -> /state/bev)')
        else:
            self.get_logger().error(
                f"Invalid state_source '{self.state_source}'. Use 'oracle' or 'vision'."
            )

    def _build_covariance(self, diag):
        cov = [0.0] * 36
        cov[0] = float(diag[0])
        cov[7] = float(diag[1])
        cov[35] = float(diag[2])
        return cov

    def _odom_callback(self, msg: Odometry):
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose = msg.pose.pose
        out.pose.covariance = self.covariance
        self.publisher.publish(out)

    def _vision_callback(self, msg: PoseStamped):
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose = msg.pose
        out.pose.covariance = self.covariance
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = StateAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
