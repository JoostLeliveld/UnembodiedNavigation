#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


class GoalMissionNode(Node):
    """Publishes the mission goal. Single-goal by default; if ``waypoints_json``
    is a non-empty JSON list of [x,y], drives them as an ordered multi-goal tour:
    the current waypoint is published to /goal_bev, and advances to the next once
    the belief is within ``arrival_radius_m`` of it. The FINAL waypoint is held so
    the planner's goal-success/auto-stop logic ends the mission there (advancing
    intermediate goals before the success-hold completes avoids a premature stop).
    """

    def __init__(self):
        super().__init__('goal_mission_node')

        self.declare_parameter('goal_x', 3.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('delay_seconds', 3.0)
        self.declare_parameter('repeat_rate', 1.0)
        self.declare_parameter('repeat_count', 0)
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('waypoints_json', '')
        self.declare_parameter('arrival_radius_m', 0.6)

        self.delay = float(self.get_parameter('delay_seconds').value)
        self.repeat_rate = float(self.get_parameter('repeat_rate').value)
        self.repeat_count = int(self.get_parameter('repeat_count').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.arrival_radius = float(self.get_parameter('arrival_radius_m').value)

        # Build the waypoint list: explicit tour if given, else the single goal.
        raw = str(self.get_parameter('waypoints_json').value or '').strip()
        self.waypoints = []
        if raw:
            try:
                for p in json.loads(raw):
                    self.waypoints.append((float(p[0]), float(p[1])))
            except (ValueError, TypeError, IndexError) as exc:
                self.get_logger().error(f"bad waypoints_json {raw!r}: {exc}")
        if not self.waypoints:
            self.waypoints = [(float(self.get_parameter('goal_x').value),
                               float(self.get_parameter('goal_y').value))]
        self.wp_idx = 0

        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_bev', qos_profile=goal_qos)
        self.create_subscription(PoseWithCovarianceStamped, '/planner_belief', self._belief_cb, 10)
        self._belief_xy = None
        self.sent_count = 0
        self.wall_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.start_time = self.wall_clock.now()

        period = 1.0 / max(self.repeat_rate, 0.1)
        self.create_timer(period, self._send_goal, clock=self.wall_clock)
        self.get_logger().info(
            f"Mission ready. {len(self.waypoints)} waypoint(s) {self.waypoints} in frame "
            f"'{self.frame_id}', first in {self.delay}s (arrival_radius {self.arrival_radius} m)"
        )

    def _belief_cb(self, msg: PoseWithCovarianceStamped):
        self._belief_xy = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def _maybe_advance(self):
        """Advance to the next waypoint once the belief reaches the current one
        (except the last, which is held for the success/auto-stop logic)."""
        if self.wp_idx >= len(self.waypoints) - 1 or self._belief_xy is None:
            return
        gx, gy = self.waypoints[self.wp_idx]
        bx, by = self._belief_xy
        if ((bx - gx) ** 2 + (by - gy) ** 2) ** 0.5 <= self.arrival_radius:
            self.wp_idx += 1
            self.get_logger().info(
                f"reached waypoint {self.wp_idx}/{len(self.waypoints) - 1} -> "
                f"advancing to ({self.waypoints[self.wp_idx][0]:.2f}, {self.waypoints[self.wp_idx][1]:.2f})"
            )

    def _send_goal(self):
        elapsed = (self.wall_clock.now() - self.start_time).nanoseconds * 1e-9
        if elapsed < self.delay:
            return
        if self.repeat_count > 0 and self.sent_count >= self.repeat_count:
            return
        self._maybe_advance()
        gx, gy = self.waypoints[self.wp_idx]
        goal = PoseStamped()
        goal.header.stamp = self.wall_clock.now().to_msg()
        goal.header.frame_id = self.frame_id
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self.sent_count += 1
        self.get_logger().info(
            f"Goal published (wp {self.wp_idx + 1}/{len(self.waypoints)}) "
            f"at ({gx:.3f}, {gy:.3f}) frame='{self.frame_id}'"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalMissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
