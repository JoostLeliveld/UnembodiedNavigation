#!/usr/bin/env python3
import csv
import math
import os
import time
from datetime import datetime

import numpy as np
import rclpy
import tf2_ros
import yaml
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


def _load_task_start_pose(tasks_yaml_path: str, world: str, task: str):
    path = str(tasks_yaml_path or '').strip()
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return None
    tasks = payload.get('tasks')
    if not isinstance(tasks, dict):
        return None
    world_tasks = tasks.get(str(world), [])
    if not isinstance(world_tasks, list):
        return None
    for entry in world_tasks:
        if not isinstance(entry, dict):
            continue
        if str(entry.get('name', '')).strip() != str(task).strip():
            continue
        start = entry.get('start')
        if not isinstance(start, dict):
            return None
        try:
            return (
                float(start.get('x')),
                float(start.get('y')),
                float(start.get('yaw', 0.0)),
            )
        except (TypeError, ValueError):
            return None
    return None


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
        self.declare_parameter('frame_sanity_start_tolerance_m', 0.25)
        self.declare_parameter('frame_sanity_start_tolerance_yaw_rad', 0.5)
        self.declare_parameter('use_visibility_model', False)
        self.declare_parameter('visibility_artifact_path', '')
        self.declare_parameter('risk_weight_obs', 1.0)
        self.declare_parameter('goal_sigma_uv', 2.0)
        self.declare_parameter('r_visible_uv', 2.5)
        self.declare_parameter('r_miss_uv', 120.0)
        self.declare_parameter('visibility_power', 1.0)
        self.declare_parameter('visibility_trust_low', 0.15)
        self.declare_parameter('visibility_trust_high', 0.65)
        self.declare_parameter('visibility_sigma_kappa', 1.0)
        self.declare_parameter('plan_rate', 2.0)
        self.declare_parameter('horizon', 36)
        self.declare_parameter('dt', 0.2)
        self.declare_parameter('control_weight', 0.0)
        self.declare_parameter('process_noise_xy', 0.01)
        self.declare_parameter('process_noise_theta', 0.02)
        self.declare_parameter('obs_noise_uv', 2.0)
        self.declare_parameter('goal_prior_u_std_start', 80.0)
        self.declare_parameter('goal_prior_v_std_start', 80.0)
        self.declare_parameter('goal_prior_u_std_final', 18.0)
        self.declare_parameter('goal_prior_v_std_final', 18.0)
        self.declare_parameter('goal_tightening_power', 0.45)
        self.declare_parameter('goal_progress_n_steps', 90)
        self.declare_parameter('observation_risk_scale', 1.25)
        self.declare_parameter('ambiguity_term_scale', 1.00)
        self.declare_parameter('visibility_weight', 0.0)
        self.declare_parameter('visibility_barrier_threshold', 0.0)
        self.declare_parameter('visibility_barrier_scale', 10.0)
        self.declare_parameter('visibility_target_height_m', 0.0)
        self.declare_parameter('perception_use_geometry_occlusion', True)
        self.declare_parameter('optimizer_maxiter', 80)
        self.declare_parameter('optimizer_maxfun', 500)
        self.declare_parameter('optimizer_ftol', 1e-6)
        self.declare_parameter('optimizer_gtol', 1e-4)
        self.declare_parameter('optimizer_warm_start', True)
        self.declare_parameter('use_nogo_cost', False)
        self.declare_parameter('nogo_penalty_type', 'softplus')
        self.declare_parameter('nogo_weight', 0.0)
        self.declare_parameter('nogo_safe_distance', 0.35)
        self.declare_parameter('nogo_gaussian_sigma', 0.25)
        self.declare_parameter('nogo_softplus_scale', 0.08)
        self.declare_parameter('nogo_logbarrier_scale', 0.25)
        self.declare_parameter('nogo_logbarrier_eps', 1e-3)
        self.declare_parameter('yolo_model', '')
        self.declare_parameter('yolo_device', '')
        self.declare_parameter('yolo_imgsz', 640)
        self.declare_parameter('yolo_conf_threshold', 0.25)
        self.declare_parameter('yolo_iou_threshold', 0.45)
        self.declare_parameter('yolo_target_class', 'robot')
        self.declare_parameter('yolo_class_id', -1)
        self.declare_parameter('yolo_use_masks', True)
        self.declare_parameter('yolo_min_mask_area_px', 12.0)
        self.declare_parameter('yolo_mask_bottom_band_px', 3.0)
        self.declare_parameter('run_dir_topic', '/experiment/run_dir')
        self.declare_parameter('run_timeout_after_first_cmd_s', 60.0)
        self.declare_parameter('first_cmd_linear_eps', 0.02)
        self.declare_parameter('first_cmd_angular_eps', 0.10)
        self.declare_parameter('stuck_window_s', 8.0)
        self.declare_parameter('stuck_max_displacement_m', 0.08)
        self.declare_parameter('stuck_max_goal_improvement_m', 0.05)
        self.declare_parameter('stuck_cmd_fraction_min', 0.50)
        self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        self.declare_parameter('img_width', 1280)
        self.declare_parameter('img_height', 720)
        self.declare_parameter('fov_h_rad', 1.5708)

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
        self.frame_sanity_start_tolerance_m = float(self.get_parameter('frame_sanity_start_tolerance_m').value)
        self.frame_sanity_start_tolerance_yaw_rad = float(self.get_parameter('frame_sanity_start_tolerance_yaw_rad').value)
        self.use_visibility_model = bool(self.get_parameter('use_visibility_model').value)
        self.visibility_artifact_path = str(self.get_parameter('visibility_artifact_path').value)
        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.goal_sigma_uv = float(self.get_parameter('goal_sigma_uv').value)
        self.r_visible_uv = float(self.get_parameter('r_visible_uv').value)
        self.r_miss_uv = float(self.get_parameter('r_miss_uv').value)
        self.visibility_power = float(self.get_parameter('visibility_power').value)
        self.visibility_trust_low = float(self.get_parameter('visibility_trust_low').value)
        self.visibility_trust_high = float(self.get_parameter('visibility_trust_high').value)
        self.visibility_sigma_kappa = float(self.get_parameter('visibility_sigma_kappa').value)
        self.plan_rate = float(self.get_parameter('plan_rate').value)
        self.horizon = int(self.get_parameter('horizon').value)
        self.dt = float(self.get_parameter('dt').value)
        self.control_weight = float(self.get_parameter('control_weight').value)
        self.process_noise_xy = float(self.get_parameter('process_noise_xy').value)
        self.process_noise_theta = float(self.get_parameter('process_noise_theta').value)
        self.obs_noise_uv = float(self.get_parameter('obs_noise_uv').value)
        self.goal_prior_u_std_start = float(self.get_parameter('goal_prior_u_std_start').value)
        self.goal_prior_v_std_start = float(self.get_parameter('goal_prior_v_std_start').value)
        self.goal_prior_u_std_final = float(self.get_parameter('goal_prior_u_std_final').value)
        self.goal_prior_v_std_final = float(self.get_parameter('goal_prior_v_std_final').value)
        self.goal_tightening_power = float(self.get_parameter('goal_tightening_power').value)
        self.goal_progress_n_steps = int(self.get_parameter('goal_progress_n_steps').value)
        self.observation_risk_scale = float(self.get_parameter('observation_risk_scale').value)
        self.ambiguity_term_scale = float(self.get_parameter('ambiguity_term_scale').value)
        self.visibility_weight = float(self.get_parameter('visibility_weight').value)
        self.visibility_target_height_m = float(self.get_parameter('visibility_target_height_m').value)
        self.perception_use_geometry_occlusion = bool(
            self.get_parameter('perception_use_geometry_occlusion').value
        )
        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_maxfun = int(self.get_parameter('optimizer_maxfun').value)
        self.optimizer_ftol = float(self.get_parameter('optimizer_ftol').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_warm_start = bool(self.get_parameter('optimizer_warm_start').value)
        self.use_nogo_cost = bool(self.get_parameter('use_nogo_cost').value)
        self.nogo_penalty_type = str(self.get_parameter('nogo_penalty_type').value)
        self.nogo_weight = float(self.get_parameter('nogo_weight').value)
        self.nogo_safe_distance = float(self.get_parameter('nogo_safe_distance').value)
        self.nogo_gaussian_sigma = float(self.get_parameter('nogo_gaussian_sigma').value)
        self.nogo_softplus_scale = float(self.get_parameter('nogo_softplus_scale').value)
        self.nogo_logbarrier_scale = float(self.get_parameter('nogo_logbarrier_scale').value)
        self.nogo_logbarrier_eps = float(self.get_parameter('nogo_logbarrier_eps').value)
        self.yolo_model = str(self.get_parameter('yolo_model').value)
        self.yolo_device = str(self.get_parameter('yolo_device').value)
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.yolo_conf_threshold = float(self.get_parameter('yolo_conf_threshold').value)
        self.yolo_iou_threshold = float(self.get_parameter('yolo_iou_threshold').value)
        self.yolo_target_class = str(self.get_parameter('yolo_target_class').value)
        self.yolo_class_id = int(self.get_parameter('yolo_class_id').value)
        self.yolo_use_masks = bool(self.get_parameter('yolo_use_masks').value)
        self.yolo_min_mask_area_px = float(self.get_parameter('yolo_min_mask_area_px').value)
        self.yolo_mask_bottom_band_px = float(self.get_parameter('yolo_mask_bottom_band_px').value)
        self.run_dir_topic = str(self.get_parameter('run_dir_topic').value).strip() or '/experiment/run_dir'
        self.run_timeout_after_first_cmd_s = float(self.get_parameter('run_timeout_after_first_cmd_s').value)
        self.first_cmd_linear_eps = float(self.get_parameter('first_cmd_linear_eps').value)
        self.first_cmd_angular_eps = float(self.get_parameter('first_cmd_angular_eps').value)
        self.stuck_window_s = float(self.get_parameter('stuck_window_s').value)
        self.stuck_max_displacement_m = float(self.get_parameter('stuck_max_displacement_m').value)
        self.stuck_max_goal_improvement_m = float(self.get_parameter('stuck_max_goal_improvement_m').value)
        self.stuck_cmd_fraction_min = float(self.get_parameter('stuck_cmd_fraction_min').value)

        # Camera model for homography projection (pixel to world)
        try:
            from unav_common.camera_model import ObliqueCameraModel
            cam_pos = np.array(self.get_parameter('cam_pos').value, dtype=float)
            look_at = np.array(self.get_parameter('look_at').value, dtype=float)
            img_width = int(self.get_parameter('img_width').value)
            img_height = int(self.get_parameter('img_height').value)
            fov_h_rad = float(self.get_parameter('fov_h_rad').value)
            self.camera_model = ObliqueCameraModel(
                cam_pos=cam_pos,
                look_at=look_at,
                img_width=img_width,
                img_height=img_height,
                fov_h_rad=fov_h_rad,
            )
        except Exception as e:
            self.get_logger().warn(f'Failed to initialize camera model for homography: {e}')
            self.camera_model = None

        run_info = create_run_dir(log_dir)
        self.run_id = run_info['run_id']
        self.run_dir = run_info['run_dir']

        self.log_path = os.path.join(self.run_dir, 'experiment.csv')

        repo_root = _find_repo_root(os.getcwd())
        self.repo_root = repo_root
        self.task_start_pose = _load_task_start_pose(self.tasks_yaml, self.world, self.task)
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
            'visibility_trust_low': self.visibility_trust_low,
            'visibility_trust_high': self.visibility_trust_high,
            'visibility_sigma_kappa': self.visibility_sigma_kappa,
            'goal_prior_u_std_start': self.goal_prior_u_std_start,
            'goal_prior_v_std_start': self.goal_prior_v_std_start,
            'goal_prior_u_std_final': self.goal_prior_u_std_final,
            'goal_prior_v_std_final': self.goal_prior_v_std_final,
            'goal_tightening_power': self.goal_tightening_power,
            'goal_progress_n_steps': self.goal_progress_n_steps,
            'observation_risk_scale': self.observation_risk_scale,
            'ambiguity_term_scale': self.ambiguity_term_scale,
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
            'yolo_model': self.yolo_model,
            'yolo_device': self.yolo_device,
            'yolo_imgsz': self.yolo_imgsz,
            'yolo_conf_threshold': self.yolo_conf_threshold,
            'yolo_iou_threshold': self.yolo_iou_threshold,
            'yolo_target_class': self.yolo_target_class,
            'yolo_class_id': self.yolo_class_id,
            'yolo_use_masks': self.yolo_use_masks,
            'yolo_min_mask_area_px': self.yolo_min_mask_area_px,
            'yolo_mask_bottom_band_px': self.yolo_mask_bottom_band_px,
            'seed': self.seed,
            'state_pipeline': 'homography_to_bev',
            'observation_model': 'uv',
            'task_start_pose': {
                'x': float(self.task_start_pose[0]),
                'y': float(self.task_start_pose[1]),
                'yaw': float(self.task_start_pose[2]),
            } if self.task_start_pose is not None else None,
            'frame_sanity_start_tolerance_m': self.frame_sanity_start_tolerance_m,
            'frame_sanity_start_tolerance_yaw_rad': self.frame_sanity_start_tolerance_yaw_rad,
            'plan_rate': self.plan_rate,
            'horizon': self.horizon,
            'dt': self.dt,
            'control_weight': self.control_weight,
            'process_noise_xy': self.process_noise_xy,
            'process_noise_theta': self.process_noise_theta,
            'obs_noise_uv': self.obs_noise_uv,
            'optimizer_maxiter': self.optimizer_maxiter,
            'optimizer_maxfun': self.optimizer_maxfun,
            'optimizer_ftol': self.optimizer_ftol,
            'optimizer_gtol': self.optimizer_gtol,
            'optimizer_warm_start': self.optimizer_warm_start,
        }
        self._manifest_data = dict(manifest_data)
        write_manifest(self.run_dir, self._manifest_data, repo_root)
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
        self._completed = False
        self._last_tf_warn_wall = 0.0
        self._frame_sanity_logged = False
        self._frame_sanity = {
            'recorded': False,
            'ok': None,
            'reason': 'pending',
            'source_frame': '',
            'truth_stamp': math.nan,
            'raw_odom_x': math.nan,
            'raw_odom_y': math.nan,
            'raw_odom_yaw': math.nan,
            'truth_x': math.nan,
            'truth_y': math.nan,
            'truth_yaw': math.nan,
            'task_start_x': float(self.task_start_pose[0]) if self.task_start_pose is not None else math.nan,
            'task_start_y': float(self.task_start_pose[1]) if self.task_start_pose is not None else math.nan,
            'task_start_yaw': float(self.task_start_pose[2]) if self.task_start_pose is not None else math.nan,
            'truth_start_error_m': math.nan,
            'raw_start_error_m': math.nan,
            'truth_start_yaw_error_rad': math.nan,
            'raw_start_yaw_error_rad': math.nan,
            'tolerance_m': self.frame_sanity_start_tolerance_m,
            'tolerance_yaw_rad': self.frame_sanity_start_tolerance_yaw_rad,
        }
        self._rewrite_manifest()

        self._first_cmd_stamp = None
        self._motion_history = []
        self._cumulative_path_length = 0.0
        self._last_path_pose = None
        self._min_goal_distance = float('inf')

        self._efe_risk_sum = 0.0
        self._efe_ambiguity_sum = 0.0
        self._efe_control_sum = 0.0
        self._efe_visibility_sum = 0.0
        self._efe_obstacle_sum = 0.0
        self._efe_count = 0
        self._solve_time_ms_sum = 0.0
        self._solve_count = 0
        self._p_vis_plan_sum = 0.0
        self._p_vis_plan_eff_sum = 0.0
        self._r_plan_u_std_sum = 0.0
        self._r_plan_v_std_sum = 0.0
        self._p_vis_count = 0
        self._p_vis_plan_below_0_2_count = 0
        self._p_vis_plan_eff_below_0_2_count = 0
        self._max_r_plan_std = 0.0
        self._truth_state_error_sum = 0.0
        self._truth_belief_error_sum = 0.0
        self._truth_state_error_count = 0
        self._truth_belief_error_count = 0
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
            'truth_available', 'truth_stamp', 'truth_x', 'truth_y', 'truth_yaw',
            'state_available', 'state_stamp', 'state_x', 'state_y', 'state_yaw',
            'state_cov_xx', 'state_cov_xy', 'state_cov_yy', 'state_cov_yaw',
            'planner_belief_available', 'planner_belief_stamp',
            'planner_belief_x', 'planner_belief_y', 'planner_belief_yaw',
            'planner_cov_x', 'planner_cov_xy', 'planner_cov_y', 'planner_cov_yaw',
            'est_available', 'est_x', 'est_y', 'est_yaw',
            'est_cov_xx', 'est_cov_xy', 'est_cov_yy',
            'state_pos_error_m', 'state_cov_trace', 'state_cov_det',
            'state_sigma_major_m', 'state_sigma_minor_m', 'state_entropy_xy',
            # Explicit unambiguous error columns:
            # truth_state_error_m  = ||truth - /state/bev||   (perception estimate vs ground truth)
            # truth_belief_error_m = ||truth - /planner_belief||  (planner internal state vs ground truth)
            'truth_state_error_m', 'truth_belief_error_m',
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
                'pred_world_x',
                'pred_world_y',
                'localization_error_m',
                'u_red',
                'v_red',
                'red_area_px',
                'u_blue',
                'v_blue',
                'blue_area_px',
                'separation_px',
                'border_margin_px',
                'yolo_score_raw',
                'yolo_score_selected',
                'yolo_detected_after_threshold',
                'yolo_best_class_id',
                'yolo_target_candidate_count',
                'bbox_area_px',
                'bbox_xmin',
                'bbox_ymin',
                'bbox_xmax',
                'bbox_ymax',
                'logit_margin',
                'class_entropy',
                'mask_area_px',
                'mask_bottom_u',
                'mask_bottom_v',
                'mask_used',
                'mask_polygon_points',
                'confidence_logit',
                'mask_compactness',
                'mask_border_frac',
                'mask_score',
                'selected_pixel_source_code',
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

    @staticmethod
    def _covariance_metrics_2d(cov_xx: float, cov_xy: float, cov_yy: float):
        if not (math.isfinite(cov_xx) and math.isfinite(cov_yy)):
            return math.nan, math.nan, math.nan, math.nan, math.nan
        cov_xy = float(cov_xy) if math.isfinite(cov_xy) else 0.0
        trace = float(cov_xx + cov_yy)
        det = float(cov_xx * cov_yy - cov_xy * cov_xy)
        sigma_major = math.nan
        sigma_minor = math.nan
        entropy_xy = math.nan
        try:
            evals = np.linalg.eigvalsh(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float))
            evals = np.clip(np.asarray(evals, dtype=float), 0.0, None)
            sigma_minor = float(math.sqrt(evals[0]))
            sigma_major = float(math.sqrt(evals[1]))
        except np.linalg.LinAlgError:
            pass
        if det > 0.0:
            entropy_xy = float(0.5 * math.log(((2.0 * math.pi * math.e) ** 2) * det))
        return trace, det, sigma_major, sigma_minor, entropy_xy

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
        if msg.data and len(msg.data) >= 3:
            self._efe_risk_sum += float(msg.data[1])
            self._efe_ambiguity_sum += float(msg.data[2])
            if len(msg.data) >= 6:
                self._efe_control_sum += float(msg.data[3])
                self._efe_visibility_sum += float(msg.data[4])
                self._efe_obstacle_sum += float(msg.data[5])
            self._efe_count += 1

    def _planner_diag_cb(self, msg: Float64MultiArray):
        self.planner_diag = msg
        if msg.data and len(msg.data) >= 6:
            solve_time_ms = float(msg.data[5])
            self._solve_time_ms_sum += solve_time_ms
            self._solve_count += 1
        if msg.data and len(msg.data) >= 12:
            p_vis_plan = float(msg.data[6])
            p_vis_plan_eff = float(msg.data[7])
            r_plan_u_std = float(msg.data[8])
            r_plan_v_std = float(msg.data[9])
            if math.isfinite(p_vis_plan):
                self._p_vis_plan_sum += p_vis_plan
                self._p_vis_plan_eff_sum += p_vis_plan_eff
                self._r_plan_u_std_sum += r_plan_u_std
                self._r_plan_v_std_sum += r_plan_v_std
                
                if p_vis_plan < 0.2:
                    self._p_vis_plan_below_0_2_count += 1
                if p_vis_plan_eff < 0.2:
                    self._p_vis_plan_eff_below_0_2_count += 1
                
                r_std_max = max(r_plan_u_std, r_plan_v_std)
                if r_std_max > self._max_r_plan_std:
                    self._max_r_plan_std = r_std_max
                
                self._p_vis_count += 1

    def _planner_diag_text_cb(self, msg: String):
        self.planner_diag_text = str(msg.data or '')

    def _diag_cb(self, msg: Float64MultiArray):
        self.perception_diag = diagnostics_from_message(msg)
        self._log_perception_sample(self.perception_diag)

    def _latest_truth_pose(self):
        if self.odom_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan

        stamp = self._stamp_to_float(self.odom_msg.header.stamp)
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
                return False, math.nan, math.nan, math.nan, math.nan

        return (
            True,
            stamp,
            float(pose_world.position.x),
            float(pose_world.position.y),
            self._yaw_from_quaternion(pose_world.orientation),
        )

    def _latest_raw_odom_pose(self):
        if self.odom_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan, ''
        pose = self.odom_msg.pose.pose
        source_frame = (self.odom_msg.header.frame_id or 'odom').strip() or 'odom'
        return (
            True,
            self._stamp_to_float(self.odom_msg.header.stamp),
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
            source_frame,
        )

    def _rewrite_manifest(self):
        payload = dict(self._manifest_data)
        payload['frame_sanity'] = dict(self._frame_sanity)
        write_manifest(self.run_dir, payload, self.repo_root)

    def _maybe_log_frame_sanity(self, now_stamp: float, cmd_v: float, cmd_w: float):
        if self._frame_sanity_logged:
            return
        if self.task_start_pose is None:
            self._frame_sanity_logged = True
            self._frame_sanity.update({
                'recorded': False,
                'ok': None,
                'reason': 'task_start_unavailable',
            })
            self._rewrite_manifest()
            self.get_logger().warn(
                f'Frame sanity check skipped because task start pose could not be loaded '
                f'from tasks_yaml={self.tasks_yaml!r} for world={self.world!r}, task={self.task!r}.'
            )
            return
        if self._first_cmd_stamp is not None:
            self._frame_sanity_logged = True
            self._frame_sanity.update({
                'recorded': False,
                'ok': None,
                'reason': 'first_command_started_before_sanity',
            })
            self._rewrite_manifest()
            self.get_logger().warn(
                'Frame sanity check could not be recorded before the first command; '
                'treat truth-frame validation as unavailable for this run.'
            )
            return
        if abs(cmd_v) >= self.first_cmd_linear_eps or abs(cmd_w) >= self.first_cmd_angular_eps:
            return

        raw_ok, _raw_stamp, raw_x, raw_y, raw_yaw, source_frame = self._latest_raw_odom_pose()
        true_ok, truth_stamp, truth_x, truth_y, truth_yaw = self._latest_truth_pose()
        if not (raw_ok and true_ok):
            return

        start_x, start_y, start_yaw = self.task_start_pose
        truth_start_error = float(math.hypot(truth_x - start_x, truth_y - start_y))
        raw_start_error = float(math.hypot(raw_x - start_x, raw_y - start_y))
        truth_start_yaw_error = abs(self._wrap_angle(truth_yaw - start_yaw))
        raw_start_yaw_error = abs(self._wrap_angle(raw_yaw - start_yaw))
        ok = bool(
            truth_start_error <= self.frame_sanity_start_tolerance_m
            and truth_start_yaw_error <= self.frame_sanity_start_tolerance_yaw_rad
        )

        self._frame_sanity_logged = True
        self._frame_sanity.update({
            'recorded': True,
            'ok': ok,
            'reason': 'ok' if ok else 'truth_start_mismatch',
            'source_frame': source_frame,
            'truth_stamp': truth_stamp,
            'raw_odom_x': raw_x,
            'raw_odom_y': raw_y,
            'raw_odom_yaw': raw_yaw,
            'truth_x': truth_x,
            'truth_y': truth_y,
            'truth_yaw': truth_yaw,
            'task_start_x': start_x,
            'task_start_y': start_y,
            'task_start_yaw': start_yaw,
            'truth_start_error_m': truth_start_error,
            'raw_start_error_m': raw_start_error,
            'truth_start_yaw_error_rad': truth_start_yaw_error,
            'raw_start_yaw_error_rad': raw_start_yaw_error,
            'tolerance_m': self.frame_sanity_start_tolerance_m,
            'tolerance_yaw_rad': self.frame_sanity_start_tolerance_yaw_rad,
            'recorded_at_log_stamp': now_stamp,
        })
        self._rewrite_manifest()

        message = (
            'Frame sanity check '
            f'({source_frame} -> {self.frame_id}): raw odom=({raw_x:.3f}, {raw_y:.3f}), '
            f'transformed truth=({truth_x:.3f}, {truth_y:.3f}), '
            f'task start=({start_x:.3f}, {start_y:.3f}), '
            f'truth_start_error={truth_start_error:.3f} m, '
            f'truth_start_yaw_error={truth_start_yaw_error:.3f} rad'
        )
        if ok:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(
                message
                + (
                    f' exceeds tolerances {self.frame_sanity_start_tolerance_m:.3f} m and/or '
                    f'{self.frame_sanity_start_tolerance_yaw_rad:.3f} rad. '
                )
                + 'This usually means the map_bev->odom transform or odom frame assumption is wrong.'
            )

    def _latest_state_pose(self):
        if self.state_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan
        stamp = self._stamp_to_float(self.state_msg.header.stamp)
        pose = self.state_msg.pose.pose
        return (
            True,
            stamp,
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
        )

    def _latest_planner_belief_pose(self):
        if self.planner_belief_msg is None:
            return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
        stamp = self._stamp_to_float(self.planner_belief_msg.header.stamp)
        pose = self.planner_belief_msg.pose.pose
        cov = list(self.planner_belief_msg.pose.covariance)
        return (
            True,
            stamp,
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
            float(cov[0]) if len(cov) > 0 else math.nan,
            float(cov[1]) if len(cov) > 1 else math.nan,
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

        true_ok, _truth_stamp, true_x, true_y, true_yaw = self._latest_truth_pose()
        state_ok, _state_stamp, state_x, state_y, state_yaw = self._latest_state_pose()
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

        # Compute predicted world position from image coordinates using homography
        pred_world_x = math.nan
        pred_world_y = math.nan
        localization_error_m = math.nan
        if self.camera_model is not None and obs_ok and math.isfinite(pixel_pose_u) and math.isfinite(pixel_pose_v):
            world = self.camera_model.pixel_to_world(float(pixel_pose_u), float(pixel_pose_v))
            if world is not None:
                pred_world_x = float(world[0])
                pred_world_y = float(world[1])
                if true_ok:
                    localization_error_m = math.hypot(pred_world_x - true_x, pred_world_y - true_y)

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
            pred_world_x,
            pred_world_y,
            localization_error_m,
            diag['u_red'],
            diag['v_red'],
            diag['red_area_px'],
            diag['u_blue'],
            diag['v_blue'],
            diag['blue_area_px'],
            diag['separation_px'],
            diag['border_margin_px'],
            diag.get('yolo_score_raw', math.nan),
            diag.get('yolo_score_selected', math.nan),
            diag.get('yolo_detected_after_threshold', math.nan),
            diag.get('yolo_best_class_id', math.nan),
            diag.get('yolo_target_candidate_count', math.nan),
            diag.get('bbox_area_px', math.nan),
            diag.get('bbox_xmin', math.nan),
            diag.get('bbox_ymin', math.nan),
            diag.get('bbox_xmax', math.nan),
            diag.get('bbox_ymax', math.nan),
            diag.get('logit_margin', math.nan),
            diag.get('class_entropy', math.nan),
            diag.get('mask_area_px', math.nan),
            diag.get('mask_bottom_u', math.nan),
            diag.get('mask_bottom_v', math.nan),
            diag.get('mask_used', math.nan),
            diag.get('mask_polygon_points', math.nan),
            diag.get('confidence_logit', math.nan),
            diag.get('mask_compactness', math.nan),
            diag.get('mask_border_frac', math.nan),
            diag.get('mask_score', math.nan),
            diag.get('selected_pixel_source_code', math.nan),
            self.seed,
        ])
        self.perception_file.flush()

    def _log_once(self):
        now_stamp = float(self.get_clock().now().nanoseconds) * 1e-9

        state_ok, state_stamp, state_x, state_y, state_yaw = self._latest_state_pose()
        if self.state_msg is not None:
            cov = self.state_msg.pose.covariance
            cov_x = float(cov[0]) if len(cov) > 0 else math.nan
            cov_xy = float(cov[1]) if len(cov) > 1 else math.nan
            cov_y = float(cov[7]) if len(cov) > 7 else math.nan
            cov_yaw = float(cov[35]) if len(cov) > 35 else math.nan
        else:
            cov_x = cov_xy = cov_y = cov_yaw = math.nan

        true_ok, truth_stamp, true_x, true_y, true_yaw = self._latest_truth_pose()
        (
            planner_belief_ok,
            planner_belief_stamp,
            planner_belief_x,
            planner_belief_y,
            planner_belief_yaw,
            planner_cov_x,
            planner_cov_xy,
            planner_cov_y,
            planner_cov_yaw,
        ) = self._latest_planner_belief_pose()
        if not planner_belief_ok:
            planner_belief_stamp = math.nan
            planner_belief_x = planner_belief_y = planner_belief_yaw = math.nan
            planner_cov_x = planner_cov_xy = planner_cov_y = planner_cov_yaw = math.nan

        if planner_belief_ok:
            est_available = 1.0
            est_x = planner_belief_x
            est_y = planner_belief_y
            est_yaw = planner_belief_yaw
            est_cov_xx = planner_cov_x
            est_cov_xy = planner_cov_xy
            est_cov_yy = planner_cov_y
        elif state_ok:
            est_available = 1.0
            est_x = float(state_x)
            est_y = float(state_y)
            est_yaw = float(state_yaw)
            est_cov_xx = float(cov_x)
            est_cov_xy = float(cov_xy)
            est_cov_yy = float(cov_y)
        else:
            est_available = 0.0
            est_x = est_y = est_yaw = math.nan
            est_cov_xx = est_cov_xy = est_cov_yy = math.nan

        state_pos_error_m = math.nan
        if true_ok and math.isfinite(est_x) and math.isfinite(est_y):
            state_pos_error_m = float(math.hypot(true_x - est_x, true_y - est_y))
        (
            state_cov_trace,
            state_cov_det,
            state_sigma_major_m,
            state_sigma_minor_m,
            state_entropy_xy,
        ) = self._covariance_metrics_2d(est_cov_xx, est_cov_xy, est_cov_yy)

        # Explicit unambiguous error signals:
        # truth_state_error_m:  ground truth vs /state/bev (perception output)
        # truth_belief_error_m: ground truth vs /planner_belief (planner's internal belief)
        truth_state_error_m = math.nan
        if true_ok and state_ok and math.isfinite(state_x) and math.isfinite(state_y):
            truth_state_error_m = float(math.hypot(true_x - state_x, true_y - state_y))
            self._truth_state_error_sum += truth_state_error_m
            if math.isfinite(truth_state_error_m):
                self._truth_state_error_count += 1
        truth_belief_error_m = math.nan
        if true_ok and planner_belief_ok and math.isfinite(planner_belief_x) and math.isfinite(planner_belief_y):
            truth_belief_error_m = float(math.hypot(true_x - planner_belief_x, true_y - planner_belief_y))
            self._truth_belief_error_sum += truth_belief_error_m
            if math.isfinite(truth_belief_error_m):
                self._truth_belief_error_count += 1

        cmd_v = self.cmd_msg.linear.x if self.cmd_msg else 0.0
        cmd_w = self.cmd_msg.angular.z if self.cmd_msg else 0.0
        self._maybe_log_frame_sanity(now_stamp, cmd_v, cmd_w)

        goal_x = math.nan
        goal_y = math.nan
        goal_dist = math.nan
        if self.goal_msg:
            goal_x = float(self.goal_msg.pose.position.x)
            goal_y = float(self.goal_msg.pose.position.y)
            if true_ok:
                dx = goal_x - true_x
                dy = goal_y - true_y
                goal_dist = math.hypot(dx, dy)

        current_pose = None
        if true_ok and math.isfinite(true_x) and math.isfinite(true_y):
            current_pose = (true_x, true_y)
            if self._last_path_pose is not None:
                self._cumulative_path_length += math.hypot(current_pose[0] - self._last_path_pose[0], current_pose[1] - self._last_path_pose[1])
            self._last_path_pose = current_pose

        if math.isfinite(goal_dist):
            self._min_goal_distance = min(self._min_goal_distance, goal_dist)

        plan_points = 0
        plan_length = 0.0
        if self.plan_msg and self.plan_msg.poses:
            plan_points = len(self.plan_msg.poses)
            for i in range(1, plan_points):
                p0 = self.plan_msg.poses[i - 1].pose.position
                p1 = self.plan_msg.poses[i].pose.position
                plan_length += math.hypot(p1.x - p0.x, p1.y - p0.y)

        stamp = now_stamp

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

        legacy_x = true_x if true_ok else math.nan
        legacy_y = true_y if true_ok else math.nan
        legacy_yaw = true_yaw if true_ok else math.nan
        self.writer.writerow([
            stamp,
            legacy_x,
            legacy_y,
            legacy_yaw,
            cov_x, cov_y, cov_yaw,
            1.0 if true_ok else 0.0, truth_stamp, true_x, true_y, true_yaw,
            1.0 if state_ok else 0.0, state_stamp, state_x, state_y, state_yaw,
            cov_x, cov_xy, cov_y, cov_yaw,
            1.0 if planner_belief_ok else 0.0, planner_belief_stamp,
            planner_belief_x, planner_belief_y, planner_belief_yaw,
            planner_cov_x, planner_cov_xy, planner_cov_y, planner_cov_yaw,
            est_available, est_x, est_y, est_yaw,
            est_cov_xx, est_cov_xy, est_cov_yy,
            state_pos_error_m, state_cov_trace, state_cov_det,
            state_sigma_major_m, state_sigma_minor_m, state_entropy_xy,
            truth_state_error_m, truth_belief_error_m,
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

        if not self._stop_requested:
            if self._first_cmd_stamp is None:
                if abs(cmd_v) >= self.first_cmd_linear_eps or abs(cmd_w) >= self.first_cmd_angular_eps:
                    self._first_cmd_stamp = now_stamp
                    self.get_logger().info(f"First command detected. Starting {self.run_timeout_after_first_cmd_s:.1f}s timeout.")
            else:
                elapsed = now_stamp - self._first_cmd_stamp
                if elapsed >= self.run_timeout_after_first_cmd_s:
                    self._finish_run("timeout_after_first_cmd", now_stamp)
                    return

                if current_pose is not None:
                    self._motion_history.append((now_stamp, current_pose[0], current_pose[1], goal_dist, cmd_v, cmd_w))
                    while self._motion_history and (now_stamp - self._motion_history[0][0]) > self.stuck_window_s:
                        self._motion_history.pop(0)

                    if len(self._motion_history) > 1 and (now_stamp - self._motion_history[0][0]) >= (self.stuck_window_s - 0.2):
                        cmd_count = sum(1 for m in self._motion_history if abs(m[4]) >= self.first_cmd_linear_eps or abs(m[5]) >= self.first_cmd_angular_eps)
                        if cmd_count / len(self._motion_history) >= self.stuck_cmd_fraction_min:
                            oldest = self._motion_history[0]
                            disp = math.hypot(current_pose[0] - oldest[1], current_pose[1] - oldest[2])
                            goal_imp = oldest[3] - goal_dist if math.isfinite(oldest[3]) and math.isfinite(goal_dist) else 0.0
                            if disp <= self.stuck_max_displacement_m and goal_imp <= self.stuck_max_goal_improvement_m:
                                self._finish_run("stuck", now_stamp)
                                return

        if self.auto_stop_on_goal and self.goal_msg and not self._stop_requested:
            if math.isfinite(goal_dist) and goal_dist <= self.goal_success_radius:
                if self._goal_in_radius_since is None:
                    self._goal_in_radius_since = now_stamp
                held_s = float(now_stamp - self._goal_in_radius_since)
                if held_s >= self.goal_success_hold_s:
                    self.get_logger().info(
                        f"Goal reached (dist={goal_dist:.3f} m <= {self.goal_success_radius:.3f} m) "
                        f"and held for {held_s:.2f} s."
                    )
                    self._finish_run("goal_reached", now_stamp)
                    return
            else:
                self._goal_in_radius_since = None

    def _finish_run(self, reason: str, stamp: float = None):
        if self._stop_requested:
            return
        self._stop_requested = True
        self._completed = True
        
        if self.plan_file is not None:
            self.plan_file.flush()
        if self.perception_file is not None:
            self.perception_file.flush()
        self.file.flush()

        import json
        mean_efe_risk = self._efe_risk_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_efe_ambiguity = self._efe_ambiguity_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_efe_control = self._efe_control_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_efe_visibility = self._efe_visibility_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_efe_obstacle = self._efe_obstacle_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_solve_time_ms = self._solve_time_ms_sum / max(self._solve_count, 1) if self._solve_count > 0 else math.nan
        mean_p_vis_plan = self._p_vis_plan_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_p_vis_plan_eff = self._p_vis_plan_eff_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_r_plan_u_std = self._r_plan_u_std_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_r_plan_v_std = self._r_plan_v_std_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        fraction_time_p_vis_below_0_2 = self._p_vis_plan_below_0_2_count / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        fraction_time_p_vis_eff_below_0_2 = self._p_vis_plan_eff_below_0_2_count / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        max_r_plan_std = self._max_r_plan_std if self._p_vis_count > 0 else math.nan
        mean_truth_state_error_m = (
            self._truth_state_error_sum / self._truth_state_error_count
            if self._truth_state_error_count > 0 else math.nan
        )
        mean_truth_belief_error_m = (
            self._truth_belief_error_sum / self._truth_belief_error_count
            if self._truth_belief_error_count > 0 else math.nan
        )

        if stamp is None:
            stamp = float(self.get_clock().now().nanoseconds) * 1e-9

        elapsed_after_first_cmd_s = stamp - self._first_cmd_stamp if self._first_cmd_stamp is not None else 0.0

        final_goal_distance = math.nan
        true_ok, _truth_stamp, true_x, true_y, _true_yaw = self._latest_truth_pose()
        if true_ok and self.goal_msg:
            goal_x = float(self.goal_msg.pose.position.x)
            goal_y = float(self.goal_msg.pose.position.y)
            dx = goal_x - true_x
            dy = goal_y - true_y
            final_goal_distance = math.hypot(dx, dy)

        summary = {
            'completed': True,
            'completion_reason': reason,
            'first_cmd_stamp': self._first_cmd_stamp if self._first_cmd_stamp is not None else math.nan,
            'stop_stamp': stamp,
            'elapsed_after_first_cmd_s': elapsed_after_first_cmd_s,
            'path_length_m': self._cumulative_path_length,
            'final_goal_distance': final_goal_distance,
            'minimum_goal_distance': self._min_goal_distance if math.isfinite(self._min_goal_distance) else math.nan,
            'mean_solve_time_ms': mean_solve_time_ms,
            # EFE terms — all six so plots can reconstruct the full decomposition
            'mean_efe_risk': mean_efe_risk,
            'mean_efe_ambiguity': mean_efe_ambiguity,
            'mean_efe_control': mean_efe_control,
            'mean_efe_visibility': mean_efe_visibility,
            'mean_efe_obstacle': mean_efe_obstacle,
            'mean_p_vis_plan': mean_p_vis_plan,
            'mean_p_vis_plan_eff': mean_p_vis_plan_eff,
            'fraction_time_p_vis_below_0_2': fraction_time_p_vis_below_0_2,
            'fraction_time_p_vis_eff_below_0_2': fraction_time_p_vis_eff_below_0_2,
            'mean_r_plan_u_std': mean_r_plan_u_std,
            'mean_r_plan_v_std': mean_r_plan_v_std,
            'max_r_plan_std': max_r_plan_std,
            # Explicit truth vs perception / planner belief errors
            'mean_truth_state_error_m': mean_truth_state_error_m,
            'mean_truth_belief_error_m': mean_truth_belief_error_m,
            'frame_sanity': dict(self._frame_sanity),
            'run_dir': self.run_dir
        }
        
        summary_path = os.path.join(self.run_dir, 'run_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info(f"Ending run. Reason: {reason}.")
        # rclpy.shutdown() must NOT be called from inside a timer/subscription callback
        # (it deadlocks on the global executor lock). Schedule it on a background thread
        # so this callback can return cleanly first.
        import threading
        threading.Timer(0.15, _safe_shutdown).start()

    def destroy_node(self):
        try:
            if not getattr(self, '_completed', False):
                import json
                summary_path = os.path.join(self.run_dir, 'run_summary.json')
                summary = {
                    'completed': False,
                    'completion_reason': 'interrupted',
                    'frame_sanity': dict(getattr(self, '_frame_sanity', {})),
                    'run_dir': getattr(self, 'run_dir', '')
                }
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)

            if hasattr(self, 'file'):
                self.file.close()
            if self.plan_file is not None:
                self.plan_file.close()
            if self.perception_file is not None:
                self.perception_file.close()
        finally:
            super().destroy_node()


def _safe_shutdown():
    """Deferred shutdown helper — called from a background thread, not from a ROS callback.

    rclpy.shutdown() deadlocks if called from inside a timer or subscription callback
    because it tries to acquire the executor lock, which the calling callback already holds.
    Scheduling it on a Thread with a small delay allows the callback to return first.
    """
    try:
        rclpy.shutdown()
    except Exception:
        pass


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
