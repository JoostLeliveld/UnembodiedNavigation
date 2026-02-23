#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import PoseStamped


class GoalMissionNode(Node):
    def __init__(self):
        super().__init__('goal_mission_node')

        self.declare_parameter('goal_x', 3.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('delay_seconds', 3.0)
        self.declare_parameter('repeat_rate', 1.0)
        self.declare_parameter('repeat_count', 5)
        self.declare_parameter('frame_id', 'map_bev')

        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.delay = float(self.get_parameter('delay_seconds').value)
        self.repeat_rate = float(self.get_parameter('repeat_rate').value)
        self.repeat_count = int(self.get_parameter('repeat_count').value)
        self.frame_id = self.get_parameter('frame_id').value

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_bev', 10)
        self.sent_count = 0
        self.wall_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.start_time = self.wall_clock.now()

        period = 1.0 / max(self.repeat_rate, 0.1)
        self.create_timer(period, self._send_goal, clock=self.wall_clock)
        self.get_logger().info(
            f"Mission ready. Goal ({self.goal_x}, {self.goal_y}) in frame '{self.frame_id}' in {self.delay}s"
        )

    def _send_goal(self):
        elapsed = (self.wall_clock.now() - self.start_time).nanoseconds * 1e-9
        if elapsed < self.delay:
            return
        if self.sent_count >= self.repeat_count:
            return
        goal = PoseStamped()
        goal.header.stamp = self.wall_clock.now().to_msg()
        goal.header.frame_id = self.frame_id
        goal.pose.position.x = self.goal_x
        goal.pose.position.y = self.goal_y
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self.sent_count += 1
        self.get_logger().info(
            f"Goal published ({self.sent_count}/{self.repeat_count}) "
            f"at ({self.goal_x}, {self.goal_y}) frame='{self.frame_id}'"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalMissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
