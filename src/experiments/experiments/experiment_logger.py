#!/usr/bin/env python3
import csv
import os
import math
from datetime import datetime
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PoseStamped
from nav_msgs.msg import Path


class ExperimentLogger(Node):
    def __init__(self):
        super().__init__('experiment_logger')

        self.declare_parameter('log_dir', 'logs/experiments')
        self.declare_parameter('log_rate', 10.0)
        self.declare_parameter('seed', 0)

        log_dir = self.get_parameter('log_dir').value
        self.seed = int(self.get_parameter('seed').value)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'experiment_{timestamp}.csv')

        self.state_msg = None
        self.cmd_msg = None
        self.goal_msg = None
        self.plan_msg = None

        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(PoseStamped, '/goal_bev', self._goal_cb, 10)
        self.create_subscription(Path, '/plan', self._plan_cb, 10)

        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'stamp', 'x', 'y', 'yaw',
            'cov_x', 'cov_y', 'cov_yaw',
            'cmd_v', 'cmd_w',
            'goal_x', 'goal_y', 'goal_dist',
            'plan_points', 'plan_length',
            'seed'
        ])

        rate = float(self.get_parameter('log_rate').value)
        self.create_timer(1.0 / max(rate, 0.1), self._log_once)
        self.get_logger().info(f'Experiment logger writing to {self.log_path}')

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        self.state_msg = msg

    def _cmd_cb(self, msg: Twist):
        self.cmd_msg = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_msg = msg

    def _plan_cb(self, msg: Path):
        self.plan_msg = msg

    def _log_once(self):
        if self.state_msg is None:
            return

        q = self.state_msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        cov = self.state_msg.pose.covariance
        cov_x = cov[0] if len(cov) > 0 else 0.0
        cov_y = cov[7] if len(cov) > 7 else 0.0
        cov_yaw = cov[35] if len(cov) > 35 else 0.0

        cmd_v = self.cmd_msg.linear.x if self.cmd_msg else 0.0
        cmd_w = self.cmd_msg.angular.z if self.cmd_msg else 0.0

        goal_x = self.goal_msg.pose.position.x if self.goal_msg else 0.0
        goal_y = self.goal_msg.pose.position.y if self.goal_msg else 0.0
        goal_dist = 0.0
        if self.goal_msg:
            dx = goal_x - self.state_msg.pose.pose.position.x
            dy = goal_y - self.state_msg.pose.pose.position.y
            goal_dist = math.hypot(dx, dy)

        plan_points = 0
        plan_length = 0.0
        if self.plan_msg and self.plan_msg.poses:
            plan_points = len(self.plan_msg.poses)
            for i in range(1, plan_points):
                p0 = self.plan_msg.poses[i - 1].pose.position
                p1 = self.plan_msg.poses[i].pose.position
                plan_length += math.hypot(p1.x - p0.x, p1.y - p0.y)

        stamp = self.state_msg.header.stamp.sec + self.state_msg.header.stamp.nanosec * 1e-9
        self.writer.writerow([
            stamp,
            self.state_msg.pose.pose.position.x,
            self.state_msg.pose.pose.position.y,
            yaw,
            cov_x, cov_y, cov_yaw,
            cmd_v, cmd_w,
            goal_x, goal_y, goal_dist,
            plan_points, plan_length,
            self.seed
        ])
        self.file.flush()

    def destroy_node(self):
        try:
            self.file.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
