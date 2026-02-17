#!/usr/bin/env python3
import math
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class WaitForOdom(Node):
    def __init__(self):
        super().__init__('wait_for_odom')
        self.declare_parameter('topic', '/odom')
        self.declare_parameter('timeout_s', 0.0)
        self.declare_parameter('min_messages', 1)
        self.declare_parameter('require_pose_match', False)
        self.declare_parameter('expected_x', 0.0)
        self.declare_parameter('expected_y', 0.0)
        self.declare_parameter('expected_yaw', 0.0)
        self.declare_parameter('position_tolerance', 0.5)
        self.declare_parameter('yaw_tolerance', 0.6)

        self.topic = self.get_parameter('topic').value
        self.timeout_s = float(self.get_parameter('timeout_s').value)
        self.min_messages = max(1, int(self.get_parameter('min_messages').value))
        self.require_pose_match = bool(self.get_parameter('require_pose_match').value)
        self.expected_x = float(self.get_parameter('expected_x').value)
        self.expected_y = float(self.get_parameter('expected_y').value)
        self.expected_yaw = float(self.get_parameter('expected_yaw').value)
        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.yaw_tolerance = float(self.get_parameter('yaw_tolerance').value)
        self.received = False
        self.match_count = 0

        self.create_subscription(Odometry, self.topic, self._cb, 10)
        if self.timeout_s > 0.0:
            self.get_logger().info(f"Waiting for first message on {self.topic} (timeout {self.timeout_s:.1f}s)")
        else:
            self.get_logger().info(f"Waiting for first message on {self.topic} (no timeout)")
        if self.require_pose_match:
            self.get_logger().info(
                "Pose gate enabled: "
                f"expected=({self.expected_x:.2f}, {self.expected_y:.2f}, yaw={self.expected_yaw:.2f}) "
                f"tol=({self.position_tolerance:.2f}m, {self.yaw_tolerance:.2f}rad) "
                f"min_messages={self.min_messages}"
            )

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    def _cb(self, msg: Odometry):
        if not self.received:
            if self.require_pose_match:
                x = float(msg.pose.pose.position.x)
                y = float(msg.pose.pose.position.y)
                q = msg.pose.pose.orientation
                yaw = self._yaw_from_quaternion(
                    float(q.x), float(q.y), float(q.z), float(q.w)
                )
                pos_err = math.hypot(x - self.expected_x, y - self.expected_y)
                yaw_err = abs(self._angle_diff(yaw, self.expected_yaw))
                if pos_err <= self.position_tolerance and yaw_err <= self.yaw_tolerance:
                    self.match_count += 1
                else:
                    self.match_count = 0
                    return
            else:
                self.match_count += 1

            if self.match_count >= self.min_messages:
                self.received = True
                self.get_logger().info(
                    f"Received {self.min_messages} matching messages on {self.topic}"
                )


def main(args=None):
    rclpy.init(args=args)
    node = WaitForOdom()
    start = time.monotonic()

    def _safe_shutdown():
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.received:
                node.destroy_node()
                _safe_shutdown()
                return 0
            if node.timeout_s > 0.0 and (time.monotonic() - start) > node.timeout_s:
                node.get_logger().error(f"Timeout waiting for {node.topic}")
                node.destroy_node()
                _safe_shutdown()
                return 1
    except KeyboardInterrupt:
        node.destroy_node()
        _safe_shutdown()
        return 0


if __name__ == '__main__':
    sys.exit(main())
