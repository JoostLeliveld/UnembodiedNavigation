#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path


class BoundaryPlanner(Node):
    def __init__(self):
        super().__init__('boundary_planner')

        self.declare_parameter('min_x', -5.0)
        self.declare_parameter('max_x', 5.0)
        self.declare_parameter('min_y', -5.0)
        self.declare_parameter('max_y', 5.0)
        self.declare_parameter('wall_margin', 0.2)
        self.declare_parameter('planning_rate', 2.0)
        self.declare_parameter('frame_id', 'map_bev')

        self.min_x = float(self.get_parameter('min_x').value)
        self.max_x = float(self.get_parameter('max_x').value)
        self.min_y = float(self.get_parameter('min_y').value)
        self.max_y = float(self.get_parameter('max_y').value)
        self.wall_margin = float(self.get_parameter('wall_margin').value)
        self.frame_id = self.get_parameter('frame_id').value

        self.state_msg = None
        self.goal_msg = None

        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_callback, 10)
        self.create_subscription(PoseStamped, '/goal_bev', self._goal_callback, 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)

        rate = float(self.get_parameter('planning_rate').value)
        self.create_timer(1.0 / max(rate, 0.1), self._plan_once)

        self.get_logger().info('Boundary planner started (map-only boundary planning)')

    def _state_callback(self, msg: PoseWithCovarianceStamped):
        self.state_msg = msg

    def _goal_callback(self, msg: PoseStamped):
        self.goal_msg = msg

    def _clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def _plan_once(self):
        if self.state_msg is None or self.goal_msg is None:
            return

        start_x = self.state_msg.pose.pose.position.x
        start_y = self.state_msg.pose.pose.position.y

        goal_x = self.goal_msg.pose.position.x
        goal_y = self.goal_msg.pose.position.y

        min_x = self.min_x + self.wall_margin
        max_x = self.max_x - self.wall_margin
        min_y = self.min_y + self.wall_margin
        max_y = self.max_y - self.wall_margin

        start_x = self._clamp(start_x, min_x, max_x)
        start_y = self._clamp(start_y, min_y, max_y)
        goal_x = self._clamp(goal_x, min_x, max_x)
        goal_y = self._clamp(goal_y, min_y, max_y)

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id

        start_pose = PoseStamped()
        start_pose.header = path.header
        start_pose.pose.position.x = start_x
        start_pose.pose.position.y = start_y
        start_pose.pose.orientation = self.state_msg.pose.pose.orientation

        goal_pose = PoseStamped()
        goal_pose.header = path.header
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.orientation = self.goal_msg.pose.orientation
        if goal_pose.pose.orientation.w == 0.0:
            goal_pose.pose.orientation.w = 1.0

        path.poses = [start_pose, goal_pose]
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = BoundaryPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
