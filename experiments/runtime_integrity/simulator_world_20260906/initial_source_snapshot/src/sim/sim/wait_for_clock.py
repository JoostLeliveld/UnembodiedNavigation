#!/usr/bin/env python3
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rosgraph_msgs.msg import Clock


class WaitForClock(Node):
    def __init__(self):
        super().__init__('wait_for_clock')
        self.declare_parameter('topic', '/clock')
        self.declare_parameter('timeout_s', 0.0)

        self.topic = self.get_parameter('topic').value
        self.timeout_s = float(self.get_parameter('timeout_s').value)
        self.received = False

        clock_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Clock, self.topic, self._cb, clock_qos)
        if self.timeout_s > 0.0:
            self.get_logger().info(f"Waiting for first message on {self.topic} (timeout {self.timeout_s:.1f}s)")
        else:
            self.get_logger().info(f"Waiting for first message on {self.topic} (no timeout)")

    def _cb(self, _msg: Clock):
        if not self.received:
            self.received = True
            self.get_logger().info(f"Received first message on {self.topic}")


def main(args=None):
    rclpy.init(args=args)
    node = WaitForClock()
    start = time.monotonic()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.received:
                node.destroy_node()
                rclpy.shutdown()
                return 0
            if node.timeout_s > 0.0 and (time.monotonic() - start) > node.timeout_s:
                node.get_logger().error(f"Timeout waiting for {node.topic}")
                node.destroy_node()
                rclpy.shutdown()
                return 1
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        return 0


if __name__ == '__main__':
    sys.exit(main())
