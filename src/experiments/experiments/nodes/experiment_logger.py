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
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, String
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
        self.declare_parameter('method', '')
        self.declare_parameter('perception_backend', '')
        self.declare_parameter('world', '')
        self.declare_parameter('task', '')
        self.declare_parameter('planner', '')
        self.declare_parameter('state_source_x', 'unknown')
        self.declare_parameter('state_source_y', 'unknown')
        self.declare_parameter('state_source_theta', 'unknown')
        self.declare_parameter('state_estimator_mode', 'unknown')
        self.declare_parameter('use_pixel_correction', False)
        self.declare_parameter('use_ambiguity', False)
        self.declare_parameter('use_obs_risk', True)
        self.declare_parameter('world_profiles_path', '')
        self.declare_parameter('tasks_yaml', '')
        self.declare_parameter('log_plan_samples', True)
        self.declare_parameter('log_perception_samples', True)
        self.declare_parameter('auto_stop_on_goal', False)
        self.declare_parameter('goal_success_radius', 0.35)
        self.declare_parameter('goal_success_hold_s', 2.0)
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('use_visibility_model', False)
        self.declare_parameter('visibility_artifact_path', '')
        self.declare_parameter('risk_weight_obs', 1.0)
        self.declare_parameter('goal_sigma_uv', 2.0)
        self.declare_parameter('r_visible_uv', 2.5)
        self.declare_parameter('r_miss_uv', 420.0)
        self.declare_parameter('visibility_power', 3.0)
        self.declare_parameter('visibility_sigma_kappa', 1.0)
        self.declare_parameter('goal_prior_u_std_start', 80.0)
        self.declare_parameter('goal_prior_v_std_start', 80.0)
        self.declare_parameter('goal_prior_u_std_final', 18.0)
        self.declare_parameter('goal_prior_v_std_final', 18.0)
        self.declare_parameter('goal_tightening_power', 0.45)
        self.declare_parameter('goal_progress_n_steps', 90)
        self.declare_parameter('notebook_risk_scale', 1.25)
        self.declare_parameter('notebook_ambiguity_scale', 1.00)
        self.declare_parameter('visibility_weight', 0.0)
        self.declare_parameter('visibility_barrier_threshold', 0.0)
        self.declare_parameter('visibility_barrier_scale', 10.0)
        self.declare_parameter('visibility_target_height_m', 0.0)
        self.declare_parameter('perception_use_geometry_occlusion', True)
        self.declare_parameter('use_nogo_cost', False)
        self.declare_parameter('nogo_penalty_type', 'softplus')
        self.declare_parameter('nogo_weight', 0.0)
        self.declare_parameter('nogo_safe_distance', 0.35)
        self.declare_parameter('nogo_gaussian_sigma', 0.25)
        self.declare_parameter('nogo_softplus_scale', 0.08)
        self.declare_parameter('nogo_logbarrier_scale', 0.25)
        self.declare_parameter('nogo_logbarrier_eps', 1e-3)
        self.declare_parameter('run_dir_topic', '/experiment/run_dir')

        log_dir = self.get_parameter('log_dir').value
        self.seed = int(self.get_parameter('seed').value)
        self.method = str(self.get_parameter('method').value)
        self.perception_backend = str(self.get_parameter('perception_backend').value)
        self.world = self.get_parameter('world').value
        self.task = self.get_parameter('task').value
        self.planner = self.get_parameter('planner').value
        self.state_source_x = str(self.get_parameter('state_source_x').value)
        self.state_source_y = str(self.get_parameter('state_source_y').value)
        self.state_source_theta = str(self.get_parameter('state_source_theta').value)
        self.state_estimator_mode = str(self.get_parameter('state_estimator_mode').value)
        self.use_pixel_correction = bool(self.get_parameter('use_pixel_correction').value)
        self.use_ambiguity = bool(self.get_parameter('use_ambiguity').value)
        self.use_obs_risk = bool(self.get_parameter('use_obs_risk').value)
        self.world_profiles_path = self.get_parameter('world_profiles_path').value
        self.tasks_yaml = self.get_parameter('tasks_yaml').value
        self.log_plan_samples = bool(self.get_parameter('log_plan_samples').value)
        self.log_perception_samples = bool(self.get_parameter('log_perception_samples').value)
        self.auto_stop_on_goal = bool(self.get_parameter('auto_stop_on_goal').value)
        self.goal_success_radius = float(self.get_parameter('goal_success_radius').value)
        self.goal_success_hold_s = float(self.get_parameter('goal_success_hold_s').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.use_visibility_model = bool(self.get_parameter('use_visibility_model').value)
        self.visibility_artifact_path = str(self.get_parameter('visibility_artifact_path').value)
        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.goal_sigma_uv = float(self.get_parameter('goal_sigma_uv').value)
        self.r_visible_uv = float(self.get_parameter('r_visible_uv').value)
        self.r_miss_uv = float(self.get_parameter('r_miss_uv').value)
        self.visibility_power = float(self.get_parameter('visibility_power').value)
        self.visibility_sigma_kappa = float(self.get_parameter('visibility_sigma_kappa').value)
        self.goal_prior_u_std_start = float(self.get_parameter('goal_prior_u_std_start').value)
        self.goal_prior_v_std_start = float(self.get_parameter('goal_prior_v_std_start').value)
        self.goal_prior_u_std_final = float(self.get_parameter('goal_prior_u_std_final').value)
        self.goal_prior_v_std_final = float(self.get_parameter('goal_prior_v_std_final').value)
        self.goal_tightening_power = float(self.get_parameter('goal_tightening_power').value)
        self.goal_progress_n_steps = int(self.get_parameter('goal_progress_n_steps').value)
        self.notebook_risk_scale = float(self.get_parameter('notebook_risk_scale').value)
        self.notebook_ambiguity_scale = float(self.get_parameter('notebook_ambiguity_scale').value)
        self.visibility_weight = float(self.get_parameter('visibility_weight').value)
        self.visibility_target_height_m = float(self.get_parameter('visibility_target_height_m').value)
        self.perception_use_geometry_occlusion = bool(
            self.get_parameter('perception_use_geometry_occlusion').value
        )
        self.use_nogo_cost = bool(self.get_parameter('use_nogo_cost').value)
        self.nogo_penalty_type = str(self.get_parameter('nogo_penalty_type').value)
        self.nogo_weight = float(self.get_parameter('nogo_weight').value)
        self.nogo_safe_distance = float(self.get_parameter('nogo_safe_distance').value)
        self.nogo_gaussian_sigma = float(self.get_parameter('nogo_gaussian_sigma').value)
        self.nogo_softplus_scale = float(self.get_parameter('nogo_softplus_scale').value)
        self.nogo_logbarrier_scale = float(self.get_parameter('nogo_logbarrier_scale').value)
        self.nogo_logbarrier_eps = float(self.get_parameter('nogo_logbarrier_eps').value)
        self.run_dir_topic = str(self.get_parameter('run_dir_topic').value).strip() or '/experiment/run_dir'

        run_info = create_run_dir(log_dir)
        self.run_id = run_info['run_id']
        self.run_dir = run_info['run_dir']

        self.log_path = os.path.join(self.run_dir, 'experiment.csv')

        repo_root = _find_repo_root(os.getcwd())
        manifest_data = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'method': self.method or self.planner,
            'perception_backend': self.perception_backend,
            'world': self.world,
            'task': self.task,
            'planner': self.planner,
            'state_source_x': self.state_source_x,
            'state_source_y': self.state_source_y,
            'state_source_theta': self.state_source_theta,
            'state_estimator_mode': self.state_estimator_mode,
            'use_pixel_correction': self.use_pixel_correction,
            'use_ambiguity': self.use_ambiguity,
            'use_obs_risk': self.use_obs_risk,
            'use_visibility_model': self.use_visibility_model,
            'visibility_artifact_path': self.visibility_artifact_path,
            'risk_weight_obs': self.risk_weight_obs,
            'goal_sigma_uv': self.goal_sigma_uv,
            'r_visible_uv': self.r_visible_uv,
            'r_miss_uv': self.r_miss_uv,
            'visibility_power': self.visibility_power,
            'visibility_sigma_kappa': self.visibility_sigma_kappa,
            'goal_prior_u_std_start': self.goal_prior_u_std_start,
            'goal_prior_v_std_start': self.goal_prior_v_std_start,
            'goal_prior_u_std_final': self.goal_prior_u_std_final,
            'goal_prior_v_std_final': self.goal_prior_v_std_final,
            'goal_tightening_power': self.goal_tightening_power,
            'goal_progress_n_steps': self.goal_progress_n_steps,
            'notebook_risk_scale': self.notebook_risk_scale,
            'notebook_ambiguity_scale': self.notebook_ambiguity_scale,
            'visibility_weight': self.visibility_weight,
            'visibility_target_height_m': self.visibility_target_height_m,
            'perception_use_geometry_occlusion': self.perception_use_geometry_occlusion,
            'use_nogo_cost': self.use_nogo_cost,
            'nogo_penalty_type': self.nogo_penalty_type,
            'nogo_weight': self.nogo_weight,
            'nogo_safe_distance': self.nogo_safe_distance,
            'nogo_gaussian_sigma': self.nogo_gaussian_sigma,
            'nogo_softplus_scale': self.nogo_softplus_scale,
            'nogo_logbarrier_scale': self.nogo_logbarrier_scale,
            'nogo_logbarrier_eps': self.nogo_logbarrier_eps,
            'seed': self.seed,
            'state_pipeline': 'homography_to_bev',
            'observation_model': 'uv',
        }
        write_manifest(self.run_dir, manifest_data, repo_root)
        snapshot_configs(self.run_dir, [self.world_profiles_path, self.tasks_yaml])

        self.state_msg = None
        self.planner_belief_msg = None
        self.odom_msg = None
        self.obs_msg = None
        self.perception_diag = None
        self.cmd_msg = None
        self.goal_msg = None
        self.plan_msg = None
        self.planner_diag = None
        self.planner_diag_text = ''
        self.efe_metrics = None
        self._goal_in_radius_since = None
        self._stop_requested = False
        self._last_tf_warn_wall = 0.0
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        run_dir_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.run_dir_pub = self.create_publisher(String, self.run_dir_topic, qos_profile=run_dir_qos)
        run_dir_msg = String()
        run_dir_msg.data = self.run_dir
        self.run_dir_pub.publish(run_dir_msg)

        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/planner_belief', self._planner_belief_cb, 10)
        self.create_subscription(PoseStamped, '/perception/pixel_pose', self._obs_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            DETECTION_DIAGNOSTICS_TOPIC,
            self._diag_cb,
            10,
        )
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(PoseStamped, '/goal_bev', self._goal_cb, qos_profile=goal_qos)
        self.create_subscription(Path, '/plan_preview', self._plan_cb, 10)
        self.create_subscription(Float64MultiArray, '/planner/diagnostics', self._planner_diag_cb, 10)
        self.create_subscription(String, '/planner/diagnostics_text', self._planner_diag_text_cb, 10)
        self.create_subscription(Float64MultiArray, '/efe/metrics', self._efe_cb, 10)

        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'stamp', 'x', 'y', 'yaw',
            'cov_x', 'cov_y', 'cov_yaw',
            'planner_belief_x', 'planner_belief_y', 'planner_belief_yaw',
            'planner_cov_x', 'planner_cov_y', 'planner_cov_yaw',
            'cmd_v', 'cmd_w',
            'goal_x', 'goal_y', 'goal_dist',
            'plan_points', 'plan_length',
            'optimizer_success', 'optimizer_status', 'optimizer_nit', 'optimizer_nfev', 'optimizer_message',
            'plan_time_ms', 'solve_time_ms',
            'measurement_available', 'belief_age_s',
            'p_vis_plan', 'p_vis_plan_eff',
            'r_plan_u_std', 'r_plan_v_std',
            'efe_total', 'efe_risk', 'efe_ambiguity', 'efe_control', 'efe_visibility', 'efe_obstacle',
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
                'seed',
            ])

        rate = float(self.get_parameter('log_rate').value)
        self.create_timer(1.0 / max(rate, 0.1), self._log_once)
        self.get_logger().info(
            f'Experiment logger writing to {self.log_path} '
            f'(method={self.method or self.planner}, world={self.world}, task={self.task})'
        )
        self.get_logger().info(
            'State-estimator provenance: '
            f'mode={self.state_estimator_mode}, '
            f'x={self.state_source_x}, y={self.state_source_y}, theta={self.state_source_theta}'
        )
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

    def _planner_belief_cb(self, msg: PoseWithCovarianceStamped):
        self.planner_belief_msg = msg

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

    def _planner_diag_cb(self, msg: Float64MultiArray):
        self.planner_diag = msg

    def _planner_diag_text_cb(self, msg: String):
        self.planner_diag_text = str(msg.data or '')

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
            except tf2_ros.TransformException as exc:
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

    def _latest_planner_belief_pose(self):
        if self.planner_belief_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
        pose = self.planner_belief_msg.pose.pose
        cov = list(self.planner_belief_msg.pose.covariance)
        return (
            True,
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
            float(cov[0]) if len(cov) > 0 else math.nan,
            float(cov[7]) if len(cov) > 7 else math.nan,
            float(cov[35]) if len(cov) > 35 else math.nan,
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

        log_stamp = float(self.get_clock().now().nanoseconds) * 1e-9
        pixel_pose_age_s = math.nan
        if obs_ok and math.isfinite(pixel_pose_stamp):
            pixel_pose_age_s = max(log_stamp - pixel_pose_stamp, 0.0)

        self.perception_writer.writerow([
            diag['stamp'],
            log_stamp,
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
        (
            planner_belief_ok,
            planner_belief_x,
            planner_belief_y,
            planner_belief_yaw,
            planner_cov_x,
            planner_cov_y,
            planner_cov_yaw,
        ) = self._latest_planner_belief_pose()
        if not planner_belief_ok:
            planner_belief_x = planner_belief_y = planner_belief_yaw = math.nan
            planner_cov_x = planner_cov_y = planner_cov_yaw = math.nan

        cmd_v = self.cmd_msg.linear.x if self.cmd_msg else 0.0
        cmd_w = self.cmd_msg.angular.z if self.cmd_msg else 0.0

        goal_x = math.nan
        goal_y = math.nan
        goal_dist = math.nan
        if self.goal_msg:
            goal_x = float(self.goal_msg.pose.position.x)
            goal_y = float(self.goal_msg.pose.position.y)
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
        efe_visibility = 0.0
        efe_obstacle = 0.0
        optimizer_success = 0.0
        optimizer_status = 0.0
        optimizer_nit = 0.0
        optimizer_nfev = 0.0
        optimizer_message = self.planner_diag_text
        plan_time_ms = 0.0
        solve_time_ms = 0.0
        measurement_available = math.nan
        belief_age_s = math.nan
        p_vis_plan = math.nan
        p_vis_plan_eff = math.nan
        r_plan_u_std = math.nan
        r_plan_v_std = math.nan
        if self.planner_diag and self.planner_diag.data and len(self.planner_diag.data) >= 6:
            optimizer_success = float(self.planner_diag.data[0])
            optimizer_status = float(self.planner_diag.data[1])
            optimizer_nit = float(self.planner_diag.data[2])
            optimizer_nfev = float(self.planner_diag.data[3])
            plan_time_ms = float(self.planner_diag.data[4])
            solve_time_ms = float(self.planner_diag.data[5])
            if len(self.planner_diag.data) >= 12:
                p_vis_plan = float(self.planner_diag.data[6])
                p_vis_plan_eff = float(self.planner_diag.data[7])
                r_plan_u_std = float(self.planner_diag.data[8])
                r_plan_v_std = float(self.planner_diag.data[9])
                measurement_available = float(self.planner_diag.data[10])
                belief_age_s = float(self.planner_diag.data[11])
        if self.efe_metrics and self.efe_metrics.data and len(self.efe_metrics.data) >= 6:
            efe_total = float(self.efe_metrics.data[0])
            efe_risk = float(self.efe_metrics.data[1])
            efe_ambiguity = float(self.efe_metrics.data[2])
            efe_control = float(self.efe_metrics.data[3])
            efe_visibility = float(self.efe_metrics.data[4])
            if len(self.efe_metrics.data) >= 6:
                efe_obstacle = float(self.efe_metrics.data[5])

        self.writer.writerow([
            stamp,
            self.state_msg.pose.pose.position.x,
            self.state_msg.pose.pose.position.y,
            yaw,
            cov_x, cov_y, cov_yaw,
            planner_belief_x, planner_belief_y, planner_belief_yaw,
            planner_cov_x, planner_cov_y, planner_cov_yaw,
            cmd_v, cmd_w,
            goal_x, goal_y, goal_dist,
            plan_points, plan_length,
            optimizer_success, optimizer_status, optimizer_nit, optimizer_nfev, optimizer_message,
            plan_time_ms, solve_time_ms,
            measurement_available, belief_age_s,
            p_vis_plan, p_vis_plan_eff,
            r_plan_u_std, r_plan_v_std,
            efe_total, efe_risk, efe_ambiguity, efe_control, efe_visibility, efe_obstacle,
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
