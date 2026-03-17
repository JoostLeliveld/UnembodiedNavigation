#!/usr/bin/env python3
import csv
import math
import os
import time
from datetime import datetime

import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tf2_geometry_msgs import do_transform_pose

from experiments.core.manifest import create_run_dir, snapshot_configs, write_manifest
from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)


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
        self.declare_parameter('add_ambiguity', False)
        self.declare_parameter('use_ambiguity', False)
        self.declare_parameter('use_obs_risk', True)
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('transform_noise_sigma', 0.0)
        self.declare_parameter('world_profiles_path', '')
        self.declare_parameter('tasks_yaml', '')
        self.declare_parameter('log_plan_samples', True)
        self.declare_parameter('log_perception_samples', True)
        self.declare_parameter('auto_stop_on_goal', False)
        self.declare_parameter('goal_success_radius', 0.35)
        self.declare_parameter('goal_success_hold_s', 2.0)
        self.declare_parameter('frame_id', 'map_bev')

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
        self.log_perception_samples = bool(self.get_parameter('log_perception_samples').value)
        self.auto_stop_on_goal = bool(self.get_parameter('auto_stop_on_goal').value)
        self.goal_success_radius = float(self.get_parameter('goal_success_radius').value)
        self.goal_success_hold_s = float(self.get_parameter('goal_success_hold_s').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

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
        self.odom_msg = None
        self.obs_msg = None
        self.perception_diag = None
        self.cmd_msg = None
        self.goal_msg = None
        self.plan_msg = None
        self.efe_metrics = None
        self._goal_in_radius_since = None
        self._stop_requested = False
        self._last_tf_warn_wall = 0.0
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_subscription(PoseStamped, '/perception/pixel_pose', self._obs_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            DETECTION_DIAGNOSTICS_TOPIC,
            self._diag_cb,
            10,
        )
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
            'efe_total', 'efe_risk', 'efe_ambiguity', 'efe_control', 'efe_boundary', 'efe_visibility',
            'seed'
        ])

        self.plan_file = None
        self.plan_writer = None
        if self.log_plan_samples:
            self.plan_log_path = os.path.join(self.run_dir, 'plan_samples.csv')
            self.plan_file = open(self.plan_log_path, 'w', newline='')
            self.plan_writer = csv.writer(self.plan_file)
            self.plan_writer.writerow(['plan_stamp', 'point_idx', 'x', 'y'])

        self.perception_file = None
        self.perception_writer = None
        if self.log_perception_samples:
            self.perception_log_path = os.path.join(self.run_dir, 'perception.csv')
            self.perception_file = open(self.perception_log_path, 'w', newline='')
            self.perception_writer = csv.writer(self.perception_file)
            self.perception_writer.writerow([
                'diag_stamp',
                'log_stamp',
                'detected',
                'true_available',
                'true_x',
                'true_y',
                'true_yaw',
                'state_available',
                'state_x',
                'state_y',
                'state_yaw',
                'state_pos_error',
                'state_yaw_error_deg',
                'obs_u',
                'obs_v',
                'obs_yaw',
                'obs_yaw_error_deg',
                'pixel_pose_available',
                'pixel_pose_stamp',
                'pixel_pose_u',
                'pixel_pose_v',
                'pixel_pose_yaw',
                'pixel_pose_age_s',
                'u_red',
                'v_red',
                'red_area_px',
                'u_blue',
                'v_blue',
                'blue_area_px',
                'separation_px',
                'border_margin_px',
                'perception_backend',
                'seed',
            ])

        rate = float(self.get_parameter('log_rate').value)
        self.create_timer(1.0 / max(rate, 0.1), self._log_once)
        self.get_logger().info(f'Experiment logger writing to {self.log_path}')
        if self.perception_file is not None:
            self.get_logger().info(f'Perception samples writing to {self.perception_log_path}')
        if self.auto_stop_on_goal:
            self.get_logger().info(
                f"Auto-stop enabled: goal radius <= {self.goal_success_radius:.3f} m "
                f"for {self.goal_success_hold_s:.2f} s"
            )

    @staticmethod
    def _stamp_to_float(stamp_msg) -> float:
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _odom_cb(self, msg: Odometry):
        self.odom_msg = msg

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        self.state_msg = msg

    def _obs_cb(self, msg: PoseStamped):
        self.obs_msg = msg

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
        plan_stamp = self._stamp_to_float(msg.header.stamp)
        for i, pose_stamped in enumerate(msg.poses):
            p = pose_stamped.pose.position
            self.plan_writer.writerow([plan_stamp, i, p.x, p.y])
        self.plan_file.flush()

    def _efe_cb(self, msg: Float64MultiArray):
        self.efe_metrics = msg

    def _diag_cb(self, msg: Float64MultiArray):
        self.perception_diag = diagnostics_from_message(msg)
        self._log_perception_sample(self.perception_diag)

    def _latest_truth_pose(self):
        if self.odom_msg is None:
            return False, math.nan, math.nan, math.nan

        source_frame = (self.odom_msg.header.frame_id or 'odom').strip() or 'odom'
        pose_world = self.odom_msg.pose.pose
        if source_frame != self.frame_id:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self.frame_id,
                    source_frame,
                    rclpy.time.Time(),
                )
                pose_world = do_transform_pose(self.odom_msg.pose.pose, tf_msg)
            except Exception as exc:
                now = time.monotonic()
                if (now - self._last_tf_warn_wall) > 1.0:
                    self._last_tf_warn_wall = now
                    self.get_logger().warn(
                        f"Experiment logger TF transform {source_frame}->{self.frame_id} unavailable: {exc}"
                    )
                return False, math.nan, math.nan, math.nan

        return (
            True,
            float(pose_world.position.x),
            float(pose_world.position.y),
            self._yaw_from_quaternion(pose_world.orientation),
        )

    def _latest_state_pose(self):
        if self.state_msg is None:
            return False, math.nan, math.nan, math.nan
        pose = self.state_msg.pose.pose
        return (
            True,
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
        )

    def _latest_pixel_pose(self):
        if self.obs_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan
        pose = self.obs_msg.pose
        return (
            True,
            self._stamp_to_float(self.obs_msg.header.stamp),
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
        )

    def _log_perception_sample(self, diag):
        if self.perception_writer is None:
            return

        true_ok, true_x, true_y, true_yaw = self._latest_truth_pose()
        state_ok, state_x, state_y, state_yaw = self._latest_state_pose()
        obs_ok, pixel_pose_stamp, pixel_pose_u, pixel_pose_v, pixel_pose_yaw = self._latest_pixel_pose()

        state_pos_error = math.nan
        state_yaw_error_deg = math.nan
        if true_ok and state_ok:
            state_pos_error = math.hypot(state_x - true_x, state_y - true_y)
            state_yaw_error_deg = math.degrees(self._wrap_angle(state_yaw - true_yaw))

        obs_yaw_error_deg = math.nan
        if true_ok and diag['detected'] and math.isfinite(diag['yaw_est']):
            obs_yaw_error_deg = math.degrees(self._wrap_angle(diag['yaw_est'] - true_yaw))

        pixel_pose_age_s = math.nan
        if obs_ok and math.isfinite(diag['stamp']):
            pixel_pose_age_s = float(diag['stamp'] - pixel_pose_stamp)

        self.perception_writer.writerow([
            diag['stamp'],
            float(self.get_clock().now().nanoseconds) * 1e-9,
            int(diag['detected']),
            int(true_ok),
            true_x,
            true_y,
            true_yaw,
            int(state_ok),
            state_x,
            state_y,
            state_yaw,
            state_pos_error,
            state_yaw_error_deg,
            diag['u_mid'],
            diag['v_mid'],
            diag['yaw_est'],
            obs_yaw_error_deg,
            int(obs_ok),
            pixel_pose_stamp,
            pixel_pose_u,
            pixel_pose_v,
            pixel_pose_yaw,
            pixel_pose_age_s,
            diag['u_red'],
            diag['v_red'],
            diag['red_area_px'],
            diag['u_blue'],
            diag['v_blue'],
            diag['blue_area_px'],
            diag['separation_px'],
            diag['border_margin_px'],
            self.perception_backend,
            self.seed,
        ])
        self.perception_file.flush()

    def _log_once(self):
        if self.state_msg is None:
            return

        yaw = self._yaw_from_quaternion(self.state_msg.pose.pose.orientation)

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

        stamp = self._stamp_to_float(self.state_msg.header.stamp)

        efe_total = 0.0
        efe_risk = 0.0
        efe_ambiguity = 0.0
        efe_control = 0.0
        efe_boundary = 0.0
        efe_visibility = 0.0
        if self.efe_metrics and self.efe_metrics.data and len(self.efe_metrics.data) >= 5:
            efe_total = float(self.efe_metrics.data[0])
            efe_risk = float(self.efe_metrics.data[1])
            efe_ambiguity = float(self.efe_metrics.data[2])
            efe_control = float(self.efe_metrics.data[3])
            efe_boundary = float(self.efe_metrics.data[4])
            if len(self.efe_metrics.data) >= 6:
                efe_visibility = float(self.efe_metrics.data[5])

        self.writer.writerow([
            stamp,
            self.state_msg.pose.pose.position.x,
            self.state_msg.pose.pose.position.y,
            yaw,
            cov_x, cov_y, cov_yaw,
            cmd_v, cmd_w,
            goal_x, goal_y, goal_dist,
            plan_points, plan_length,
            efe_total, efe_risk, efe_ambiguity, efe_control, efe_boundary, efe_visibility,
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
                    if self.perception_file is not None:
                        self.perception_file.flush()
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
            if self.perception_file is not None:
                self.perception_file.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
