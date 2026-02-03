#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Path


class BoundaryController(Node):
    def __init__(self):
        super().__init__('boundary_controller')

        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('max_speed', 0.22)
        self.declare_parameter('kp_angular', 2.0)
        self.declare_parameter('goal_tolerance', 0.1)

        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.kp_angular = float(self.get_parameter('kp_angular').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.current_path = []

        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_callback, 10)
        self.create_subscription(Path, '/plan', self._path_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info('Boundary controller started (Pure Pursuit)')

    def _state_callback(self, msg: PoseWithCovarianceStamped):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.pose_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _path_callback(self, msg: Path):
        self.current_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _find_lookahead(self, robot_x, robot_y):
        if not self.current_path:
            return None

        for px, py in self.current_path:
            dist = math.hypot(px - robot_x, py - robot_y)
            if dist >= self.lookahead_distance:
                return (px, py)
        return self.current_path[-1]

    def _control_loop(self):
        if not self.current_path:
            self._stop()
            return

        goal_x, goal_y = self.current_path[-1]
        dist_to_goal = math.hypot(goal_x - self.pose_x, goal_y - self.pose_y)
        if dist_to_goal < self.goal_tolerance:
            self._stop()
            self.current_path = []
            self.get_logger().info('Goal reached')
            return

        target = self._find_lookahead(self.pose_x, self.pose_y)
        if target is None:
            self._stop()
            return

        target_x, target_y = target
        desired_yaw = math.atan2(target_y - self.pose_y, target_x - self.pose_x)
        yaw_error = desired_yaw - self.pose_yaw
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        angular = self.kp_angular * yaw_error
        linear = self.max_speed if abs(yaw_error) < 0.5 else 0.1

        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = BoundaryController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
