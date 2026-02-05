#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from tf_transformations import euler_from_quaternion

from control.core.pure_pursuit import PurePursuitParams, compute_control


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # Parameters
        self.declare_parameter('lookahead_distance', 0.5)  # meters
        self.declare_parameter('max_speed', 0.22)  # m/s (Turtlebot3 limit)
        self.declare_parameter('kp_angular', 2.0)
        self.declare_parameter('max_angular', 1.0)  # rad/s cap to avoid tight circles
        self.declare_parameter('turn_in_place_threshold', 0.7)  # rad; above this, rotate in place
        self.declare_parameter('slowdown_distance', 0.5)  # meters; taper speed near goal
        self.declare_parameter('goal_tolerance', 0.1)  # meters
        self.declare_parameter('state_timeout_s', 0.5)  # seconds; stale-state watchdog

        self.params = PurePursuitParams(
            lookahead_distance=self.get_parameter('lookahead_distance').value,
            max_speed=self.get_parameter('max_speed').value,
            kp_angular=self.get_parameter('kp_angular').value,
            max_angular=self.get_parameter('max_angular').value,
            turn_in_place_threshold=self.get_parameter('turn_in_place_threshold').value,
            slowdown_distance=self.get_parameter('slowdown_distance').value,
            goal_tolerance=self.get_parameter('goal_tolerance').value,
        )
        self.state_timeout_s = float(self.get_parameter('state_timeout_s').value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.current_path = None
        self.last_state_time = None
        self.last_stale_warn_time = None

        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/state/bev',
            self.pose_callback,
            10,
        )

        path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, '/plan', self.path_callback, qos_profile=path_qos)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.1, self.control_loop, clock=self.control_clock)

        self.get_logger().info('Control Node Started (Pure Pursuit)')

    def pose_callback(self, msg):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pose_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.last_state_time = self.control_clock.now()

    def path_callback(self, msg):
        path_points = []
        for pose in msg.poses:
            path_points.append([pose.pose.position.x, pose.pose.position.y])

        if len(path_points) > 0:
            self.current_path = np.array(path_points)
            self.get_logger().info(f'Received new path with {len(path_points)} points')
        else:
            self.current_path = None

    def control_loop(self):
        if self.current_path is None:
            return

        now = self.control_clock.now()
        if self.last_state_time is None:
            return
        if (now - self.last_state_time).nanoseconds > self.state_timeout_s * 1e9:
            if self.last_stale_warn_time is None or (now - self.last_stale_warn_time).nanoseconds > 2e9:
                self.get_logger().warn('State is stale; stopping until /state/bev updates')
                self.last_stale_warn_time = now
            self.stop_robot()
            return

        linear_vel, angular_vel, goal_reached, lookahead_point = compute_control(
            self.pose_x,
            self.pose_y,
            self.pose_yaw,
            self.current_path,
            self.params,
        )

        if goal_reached:
            self.stop_robot()
            self.get_logger().info('Goal Reached')
            self.current_path = None
            return

        timestamp = self.get_clock().now().seconds_nanoseconds()[0]
        if lookahead_point is not None and timestamp % 2 == 0:
            dist_to_end = float(np.linalg.norm(np.array([self.pose_x, self.pose_y]) - self.current_path[-1]))
            self.get_logger().info(
                f"Pose: ({self.pose_x:.2f}, {self.pose_y:.2f}) | "
                f"Target: ({lookahead_point[0]:.2f}, {lookahead_point[1]:.2f}) | "
                f"Dist: {dist_to_end:.2f}"
            )

        cmd = Twist()
        cmd.linear.x = float(linear_vel) if linear_vel is not None else 0.0
        cmd.angular.z = float(angular_vel) if angular_vel is not None else 0.0
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
