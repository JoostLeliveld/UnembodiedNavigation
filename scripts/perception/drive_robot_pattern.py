#!/usr/bin/env python3
"""Publish a simple open-loop /cmd_vel pattern for collecting moving-camera examples."""

from __future__ import annotations

import argparse
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class DriveRobotPatternNode(Node):
    def __init__(self, *, pattern: str, linear: float, angular: float, period: float, duration: float):
        super().__init__('drive_robot_pattern')
        self.pattern = str(pattern)
        self.linear = float(linear)
        self.angular = float(angular)
        self.period = max(float(period), 0.1)
        self.duration = max(float(duration), 0.1)
        self.start_time = self.get_clock().now()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f'Starting drive pattern pattern={self.pattern} linear={self.linear:.3f} '
            f'angular={self.angular:.3f} period={self.period:.3f}s duration={self.duration:.3f}s'
        )

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def _command_for_time(self, elapsed: float) -> Twist:
        msg = Twist()
        if self.pattern == 'circle':
            msg.linear.x = self.linear
            msg.angular.z = self.angular
            return msg
        if self.pattern == 'straight':
            msg.linear.x = self.linear
            msg.angular.z = 0.0
            return msg
        if self.pattern == 'sweep':
            msg.linear.x = self.linear
            msg.angular.z = self.angular * math.sin(2.0 * math.pi * elapsed / self.period)
            return msg
        if self.pattern == 'figure8':
            msg.linear.x = self.linear
            msg.angular.z = self.angular * math.sin(4.0 * math.pi * elapsed / self.period)
            return msg
        raise RuntimeError(f'Unsupported drive pattern: {self.pattern}')

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def _tick(self) -> None:
        elapsed = self._elapsed()
        if elapsed >= self.duration:
            self._publish_stop()
            self.get_logger().info('Drive pattern complete, stopping robot.')
            self.create_timer(0.1, lambda: rclpy.shutdown())
            self.timer.cancel()
            return
        self.cmd_pub.publish(self._command_for_time(elapsed))


def main() -> int:
    parser = argparse.ArgumentParser(description='Drive the robot with a small open-loop /cmd_vel pattern.')
    parser.add_argument('--pattern', choices=('circle', 'sweep', 'figure8', 'straight'), default='sweep')
    parser.add_argument('--linear', type=float, default=0.10, help='Linear velocity in m/s')
    parser.add_argument('--angular', type=float, default=0.45, help='Angular velocity amplitude in rad/s')
    parser.add_argument('--period', type=float, default=8.0, help='Oscillation period for sweep/figure8')
    parser.add_argument('--duration', type=float, default=30.0, help='Total run time in seconds')
    args = parser.parse_args()

    rclpy.init()
    node = DriveRobotPatternNode(
        pattern=str(args.pattern),
        linear=float(args.linear),
        angular=float(args.angular),
        period=float(args.period),
        duration=float(args.duration),
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
