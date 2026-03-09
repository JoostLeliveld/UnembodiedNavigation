#!/usr/bin/env python3
import csv
import math
import os
from datetime import datetime
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray

from experiments.core.manifest import create_run_dir, snapshot_configs, write_manifest


def _find_repo_root(start_dir: str) -> str:
    current = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(current, '.git')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return start_dir
        current = parent


class ExperimentLogger(Node):
    def __init__(self):
        super().__init__('experiment_logger')

        self.declare_parameter('log_dir', 'logs/experiments')
        self.declare_parameter('log_rate', 10.0)
        self.declare_parameter('seed', 0)
        self.declare_parameter('world', '')
        self.declare_parameter('task', '')
        self.declare_parameter('planner', '')
        self.declare_parameter('state_source', '')
        self.declare_parameter('perception_backend', '')
        self.declare_parameter('obs_model', '')
        self.declare_parameter('obs_mode', '')
        self.declare_parameter('use_pixel_correction', False)
        self.declare_parameter('boundary_weight', 0.0)
        self.declare_parameter('publish_static_costmap', True)
        # Legacy manifest compatibility field (mirrors use_ambiguity in current launches).
        self.declare_parameter('add_ambiguity', False)
        self.declare_parameter('use_ambiguity', False)
        self.declare_parameter('use_obs_risk', True)
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('transform_noise_sigma', 0.0)
        self.declare_parameter('world_profiles_path', '')
        self.declare_parameter('tasks_yaml', '')
        self.declare_parameter('log_plan_samples', True)
        self.declare_parameter('auto_stop_on_goal', False)
        self.declare_parameter('goal_success_radius', 0.35)
        self.declare_parameter('goal_success_hold_s', 2.0)

        log_dir = self.get_parameter('log_dir').value
        self.seed = int(self.get_parameter('seed').value)
        self.world = self.get_parameter('world').value
        self.task = self.get_parameter('task').value
        self.planner = self.get_parameter('planner').value
        self.state_source = self.get_parameter('state_source').value
        self.perception_backend = self.get_parameter('perception_backend').value
        self.obs_model = self.get_parameter('obs_model').value
        self.obs_mode = self.get_parameter('obs_mode').value
        self.use_pixel_correction = bool(self.get_parameter('use_pixel_correction').value)
        self.boundary_weight = float(self.get_parameter('boundary_weight').value)
        self.publish_static_costmap = bool(self.get_parameter('publish_static_costmap').value)
        self.add_ambiguity = bool(self.get_parameter('add_ambiguity').value)
        self.use_ambiguity = bool(self.get_parameter('use_ambiguity').value)
        self.use_obs_risk = bool(self.get_parameter('use_obs_risk').value)
        self.pixel_noise_sigma = float(self.get_parameter('pixel_noise_sigma').value)
        self.transform_noise_sigma = float(self.get_parameter('transform_noise_sigma').value)
        self.world_profiles_path = self.get_parameter('world_profiles_path').value
        self.tasks_yaml = self.get_parameter('tasks_yaml').value
        self.log_plan_samples = bool(self.get_parameter('log_plan_samples').value)
        self.auto_stop_on_goal = bool(self.get_parameter('auto_stop_on_goal').value)
        self.goal_success_radius = float(self.get_parameter('goal_success_radius').value)
        self.goal_success_hold_s = float(self.get_parameter('goal_success_hold_s').value)

        run_info = create_run_dir(log_dir)
        self.run_id = run_info['run_id']
        self.run_dir = run_info['run_dir']

        self.log_path = os.path.join(self.run_dir, 'experiment.csv')

        repo_root = _find_repo_root(os.getcwd())
        manifest_data = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'world': self.world,
            'task': self.task,
            'planner': self.planner,
            'state_source': self.state_source,
            'perception_backend': self.perception_backend,
            'obs_model': self.obs_model,
            'obs_mode': self.obs_mode,
            'use_pixel_correction': self.use_pixel_correction,
            'boundary_weight': self.boundary_weight,
            'publish_static_costmap': self.publish_static_costmap,
            # Keep the legacy field for older analysis scripts.
            'add_ambiguity': self.add_ambiguity,
            'use_ambiguity': self.use_ambiguity,
            'use_obs_risk': self.use_obs_risk,
            'seed': self.seed,
            'pixel_noise_sigma': self.pixel_noise_sigma,
            'transform_noise_sigma': self.transform_noise_sigma,
        }
        write_manifest(self.run_dir, manifest_data, repo_root)
        snapshot_configs(self.run_dir, [self.world_profiles_path, self.tasks_yaml])

        self.state_msg = None
        self.cmd_msg = None
        self.goal_msg = None
        self.plan_msg = None
        self.efe_metrics = None
        self._goal_in_radius_since = None
        self._stop_requested = False

        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(PoseStamped, '/goal_bev', self._goal_cb, 10)
        self.create_subscription(Path, '/plan', self._plan_cb, 10)
        self.create_subscription(Float64MultiArray, '/efe/metrics', self._efe_cb, 10)

        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'stamp', 'x', 'y', 'yaw',
            'cov_x', 'cov_y', 'cov_yaw',
            'cmd_v', 'cmd_w',
            'goal_x', 'goal_y', 'goal_dist',
            'plan_points', 'plan_length',
            'efe_total', 'efe_risk', 'efe_ambiguity', 'efe_control', 'efe_boundary',
            'seed'
        ])

        self.plan_file = None
        self.plan_writer = None
        if self.log_plan_samples:
            self.plan_log_path = os.path.join(self.run_dir, 'plan_samples.csv')
            self.plan_file = open(self.plan_log_path, 'w', newline='')
            self.plan_writer = csv.writer(self.plan_file)
            self.plan_writer.writerow(['plan_stamp', 'point_idx', 'x', 'y'])

        rate = float(self.get_parameter('log_rate').value)
        self.create_timer(1.0 / max(rate, 0.1), self._log_once)
        self.get_logger().info(f'Experiment logger writing to {self.log_path}')
        if self.auto_stop_on_goal:
            self.get_logger().info(
                f"Auto-stop enabled: goal radius <= {self.goal_success_radius:.3f} m "
                f"for {self.goal_success_hold_s:.2f} s"
            )

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        self.state_msg = msg

    def _cmd_cb(self, msg: Twist):
        self.cmd_msg = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_msg = msg

    def _plan_cb(self, msg: Path):
        self.plan_msg = msg
        if self.plan_writer is None:
            return
        if not msg.poses:
            return
        plan_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for i, pose_stamped in enumerate(msg.poses):
            p = pose_stamped.pose.position
            self.plan_writer.writerow([plan_stamp, i, p.x, p.y])
        self.plan_file.flush()

    def _efe_cb(self, msg: Float64MultiArray):
        self.efe_metrics = msg

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

        efe_total = 0.0
        efe_risk = 0.0
        efe_ambiguity = 0.0
        efe_control = 0.0
        efe_boundary = 0.0
        if self.efe_metrics and self.efe_metrics.data and len(self.efe_metrics.data) >= 5:
            efe_total = float(self.efe_metrics.data[0])
            efe_risk = float(self.efe_metrics.data[1])
            efe_ambiguity = float(self.efe_metrics.data[2])
            efe_control = float(self.efe_metrics.data[3])
            efe_boundary = float(self.efe_metrics.data[4])

        self.writer.writerow([
            stamp,
            self.state_msg.pose.pose.position.x,
            self.state_msg.pose.pose.position.y,
            yaw,
            cov_x, cov_y, cov_yaw,
            cmd_v, cmd_w,
            goal_x, goal_y, goal_dist,
            plan_points, plan_length,
            efe_total, efe_risk, efe_ambiguity, efe_control, efe_boundary,
            self.seed,
        ])
        self.file.flush()

        if self.auto_stop_on_goal and self.goal_msg and not self._stop_requested:
            if goal_dist <= self.goal_success_radius:
                if self._goal_in_radius_since is None:
                    self._goal_in_radius_since = stamp
                held_s = float(stamp - self._goal_in_radius_since)
                if held_s >= self.goal_success_hold_s:
                    self._stop_requested = True
                    if self.plan_file is not None:
                        self.plan_file.flush()
                    self.get_logger().info(
                        f"Goal reached (dist={goal_dist:.3f} m <= {self.goal_success_radius:.3f} m) "
                        f"and held for {held_s:.2f} s. Ending run."
                    )
                    rclpy.shutdown()
                    return
            else:
                self._goal_in_radius_since = None

    def destroy_node(self):
        try:
            self.file.close()
            if self.plan_file is not None:
                self.plan_file.close()
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
