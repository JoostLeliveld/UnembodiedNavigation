#!/usr/bin/env python3
"""Drive a deterministic lawnmower sweep for visibility-data capture."""

from __future__ import annotations

import json
import math
from typing import List, Tuple

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String

from experiments.core.visibility_capture import (
    VISIBILITY_CAPTURE_CONFIG_TOPIC,
    VISIBILITY_CAPTURE_DONE_TOPIC,
)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class VisibilitySweepControllerNode(Node):
    def __init__(self):
        super().__init__('visibility_sweep_controller')

        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('xmin', -6.0)
        self.declare_parameter('xmax', 6.0)
        self.declare_parameter('ymin', -6.0)
        self.declare_parameter('ymax', 6.0)
        self.declare_parameter('sweep_margin_m', 0.45)
        self.declare_parameter('sweep_row_spacing_m', 0.75)
        self.declare_parameter('linear_speed_mps', 0.22)
        self.declare_parameter('angular_speed_radps', 0.9)
        self.declare_parameter('waypoint_tolerance_m', 0.18)
        self.declare_parameter('heading_align_threshold_rad', 0.35)
        self.declare_parameter('state_timeout_s', 2.5)
        self.declare_parameter('turn_pause_s', 0.20)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.xmin = float(self.get_parameter('xmin').value)
        self.xmax = float(self.get_parameter('xmax').value)
        self.ymin = float(self.get_parameter('ymin').value)
        self.ymax = float(self.get_parameter('ymax').value)
        self.sweep_margin_m = float(self.get_parameter('sweep_margin_m').value)
        self.sweep_row_spacing_m = float(self.get_parameter('sweep_row_spacing_m').value)
        self.linear_speed_mps = float(self.get_parameter('linear_speed_mps').value)
        self.angular_speed_radps = float(self.get_parameter('angular_speed_radps').value)
        self.waypoint_tolerance_m = float(self.get_parameter('waypoint_tolerance_m').value)
        self.heading_align_threshold_rad = float(self.get_parameter('heading_align_threshold_rad').value)
        self.state_timeout_s = float(self.get_parameter('state_timeout_s').value)
        self.turn_pause_s = float(self.get_parameter('turn_pause_s').value)

        self._latest_state: Tuple[float, float, float] | None = None
        self._latest_state_stamp_s: float | None = None
        self._pause_until_s = 0.0
        self._completed = False
        self._waypoint_index = 0
        self._waypoints = self._build_waypoints()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.done_pub = self.create_publisher(Bool, VISIBILITY_CAPTURE_DONE_TOPIC, qos_profile=latched_qos)
        self.config_pub = self.create_publisher(String, VISIBILITY_CAPTURE_CONFIG_TOPIC, qos_profile=latched_qos)
        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_timer(0.1, self._control_timer)

        self._publish_done(False)
        self._publish_config()
        self.get_logger().info(
            f'Started visibility sweep with {len(self._waypoints)} waypoints '
            f'over x=[{self.xmin:.2f}, {self.xmax:.2f}] y=[{self.ymin:.2f}, {self.ymax:.2f}]'
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _build_waypoints(self) -> List[Tuple[float, float]]:
        x_left = self.xmin + self.sweep_margin_m
        x_right = self.xmax - self.sweep_margin_m
        y_bottom = self.ymin + self.sweep_margin_m
        y_top = self.ymax - self.sweep_margin_m
        if not (x_left < x_right and y_bottom < y_top):
            raise RuntimeError('Invalid sweep bounds after applying sweep_margin_m')

        spacing = max(self.sweep_row_spacing_m, 0.2)
        rows = [y_bottom]
        while rows[-1] + spacing < y_top:
            rows.append(rows[-1] + spacing)
        if abs(rows[-1] - y_top) > 1e-6:
            rows.append(y_top)

        waypoints: List[Tuple[float, float]] = []
        for idx, y in enumerate(rows):
            if idx % 2 == 0:
                waypoints.extend([(x_left, y), (x_right, y)])
            else:
                waypoints.extend([(x_right, y), (x_left, y)])
        return waypoints

    def _publish_done(self, done: bool) -> None:
        msg = Bool()
        msg.data = bool(done)
        self.done_pub.publish(msg)

    def _publish_config(self) -> None:
        msg = String()
        msg.data = json.dumps({
            'frame_id': self.frame_id,
            'xmin': self.xmin,
            'xmax': self.xmax,
            'ymin': self.ymin,
            'ymax': self.ymax,
            'sweep_margin_m': self.sweep_margin_m,
            'sweep_row_spacing_m': self.sweep_row_spacing_m,
            'linear_speed_mps': self.linear_speed_mps,
            'angular_speed_radps': self.angular_speed_radps,
            'waypoint_tolerance_m': self.waypoint_tolerance_m,
            'heading_align_threshold_rad': self.heading_align_threshold_rad,
            'state_timeout_s': self.state_timeout_s,
            'turn_pause_s': self.turn_pause_s,
            'waypoint_count': len(self._waypoints),
        }, sort_keys=True)
        self.config_pub.publish(msg)

    def _state_cb(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        q = pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self._latest_state = (float(pose.position.x), float(pose.position.y), float(yaw))
        stamp = msg.header.stamp
        self._latest_state_stamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def _complete(self) -> None:
        if self._completed:
            return
        self._completed = True
        self._publish_cmd(0.0, 0.0)
        self._publish_done(True)
        self.get_logger().info('Visibility sweep completed')

    def _advance_waypoint(self) -> None:
        self._waypoint_index += 1
        self._pause_until_s = self._now_s() + max(self.turn_pause_s, 0.0)
        if self._waypoint_index >= len(self._waypoints):
            self._complete()

    def _control_timer(self) -> None:
        if self._completed:
            self._publish_cmd(0.0, 0.0)
            return

        if self._latest_state is None or self._latest_state_stamp_s is None:
            self._publish_cmd(0.0, 0.0)
            return

        now_s = self._now_s()
        if self.state_timeout_s > 0.0 and (now_s - self._latest_state_stamp_s) > self.state_timeout_s:
            self._publish_cmd(0.0, 0.0)
            return

        if now_s < self._pause_until_s:
            self._publish_cmd(0.0, 0.0)
            return

        if self._waypoint_index >= len(self._waypoints):
            self._complete()
            return

        x, y, yaw = self._latest_state
        goal_x, goal_y = self._waypoints[self._waypoint_index]
        dx = goal_x - x
        dy = goal_y - y
        dist = math.hypot(dx, dy)
        if dist <= self.waypoint_tolerance_m:
            self._advance_waypoint()
            self._publish_cmd(0.0, 0.0)
            return

        heading = math.atan2(dy, dx)
        heading_err = _wrap_angle(heading - yaw)

        if abs(heading_err) > self.heading_align_threshold_rad:
            self._publish_cmd(0.0, math.copysign(self.angular_speed_radps, heading_err))
            return

        linear = min(self.linear_speed_mps, max(0.06, 0.8 * dist))
        angular = max(-self.angular_speed_radps, min(self.angular_speed_radps, 2.2 * heading_err))
        self._publish_cmd(linear, angular)


def main(args=None):
    rclpy.init(args=args)
    node = VisibilitySweepControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_cmd(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
