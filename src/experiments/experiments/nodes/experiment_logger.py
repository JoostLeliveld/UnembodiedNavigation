#!/usr/bin/env python3
import csv
import hashlib
import math
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry, Path
from ros_gz_interfaces.msg import Contacts
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, String
from tf2_geometry_msgs import do_transform_pose

from experiments.core.manifest import create_run_dir, snapshot_configs, write_manifest
from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy


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
    with open(path, 'r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
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
        return (
            float(start['x']),
            float(start['y']),
            float(start.get('yaw', 0.0)),
        )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or '').encode('utf-8')).hexdigest()


def _split_prisms_by_prefix(prisms, prefix: str):
    token = str(prefix or '').strip()
    if not token:
        return tuple()
    return tuple(prism for prism in tuple(prisms or ()) if str(prism.name).startswith(token))


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
        self.declare_parameter('pixel_timeout_s', 0.5)
        self.declare_parameter('use_ambiguity', False)
        self.declare_parameter('use_obs_risk', True)
        self.declare_parameter('world_profiles_path', '')
        self.declare_parameter('tasks_yaml', '')
        self.declare_parameter('log_plan_samples', True)
        self.declare_parameter('log_perception_samples', True)
        self.declare_parameter('auto_stop_on_goal', False)
        self.declare_parameter('goal_success_radius', 0.20)
        self.declare_parameter('goal_success_hold_s', 2.0)
        self.declare_parameter('goal_stable_radius', 0.20)
        self.declare_parameter('goal_stable_hold_s', 2.0)
        self.declare_parameter('goal_stable_max_displacement_m', 0.04)
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('frame_sanity_start_tolerance_m', 0.25)
        self.declare_parameter('frame_sanity_start_tolerance_yaw_rad', 0.5)
        self.declare_parameter('use_visibility_model', False)
        self.declare_parameter('visibility_artifact_path', '')
        self.declare_parameter('risk_weight_obs', 1.0)
        self.declare_parameter('ambiguity_weight', 1.0)
        self.declare_parameter('goal_sigma_uv', 2.0)
        self.declare_parameter('r_visible_uv', 2.5)
        self.declare_parameter('r_miss_uv', 120.0)
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
        self.declare_parameter('discount_gamma', 0.98)
        self.declare_parameter('visibility_target_height_m', 0.0)
        self.declare_parameter('perception_use_geometry_occlusion', True)
        self.declare_parameter('visibility_geometry_json', '')
        self.declare_parameter('collision_geometry_json', '')
        self.declare_parameter('robot_collision_radius_m', 0.125)
        # When False, a geometric wall/obstacle penetration is still logged but does
        # NOT terminate the run (only physical contact does). Lets a run continue past
        # a boundary graze so its natural outcome (goal / stuck / timeout) and full GT
        # trajectory are observed. Default True preserves the original behaviour.
        self.declare_parameter('terminate_on_geom_collision', True)
        self.declare_parameter('use_command_noise', True)
        self.declare_parameter('command_noise_linear_slip_mean', 0.03)
        self.declare_parameter('command_noise_linear_slip_std', 0.06)
        self.declare_parameter('command_noise_angular_slip_mean', 0.0)
        self.declare_parameter('command_noise_angular_slip_std', 0.04)
        self.declare_parameter('command_noise_linear_additive_std', 0.008)
        self.declare_parameter('command_noise_angular_additive_std', 0.035)
        self.declare_parameter('command_noise_correlation_alpha', 0.85)
        self.declare_parameter('encoder_noise_angular_slip_std', 0.03)
        self.declare_parameter('encoder_noise_angular_additive_std', 0.020)
        self.declare_parameter('optimizer_maxiter', 80)
        self.declare_parameter('optimizer_maxfun', 500)
        self.declare_parameter('optimizer_ftol', 1e-6)
        self.declare_parameter('optimizer_gtol', 1e-4)
        self.declare_parameter('optimizer_warm_start', True)
        self.declare_parameter('optimizer_multistart', False)
        self.declare_parameter('optimizer_multistart_include_direct', True)
        self.declare_parameter('optimizer_initial_routes_json', '')
        self.declare_parameter('optimizer_route_seed_mode', 'explicit')
        self.declare_parameter('use_hierarchical', False)
        self.declare_parameter('global_horizon', 60)
        self.declare_parameter('global_dt', 0.0)
        self.declare_parameter('local_horizon', 12)
        self.declare_parameter('local_plan_rate', 4.0)
        self.declare_parameter('local_optimizer_maxiter', 60)
        self.declare_parameter('global_use_ambiguity', True)
        self.declare_parameter('local_use_ambiguity', False)
        self.declare_parameter('global_optimizer_multistart', True)
        self.declare_parameter('local_optimizer_multistart', True)
        self.declare_parameter('local_use_visibility_model', False)
        self.declare_parameter('local_use_belief_nogo_cost', False)
        self.declare_parameter('local_nogo_penalty_type', '')
        self.declare_parameter('local_nogo_weight', -1.0)
        self.declare_parameter('local_nogo_safe_distance', -1.0)
        self.declare_parameter('local_goal_prior_u_std_start', -1.0)
        self.declare_parameter('local_goal_prior_v_std_start', -1.0)
        self.declare_parameter('local_goal_prior_u_std_final', -1.0)
        self.declare_parameter('local_goal_prior_v_std_final', -1.0)
        self.declare_parameter('waypoint_spacing_m', 1.0)
        self.declare_parameter('waypoint_arrival_radius_m', 0.35)
        self.declare_parameter('local_replan_min_remaining_s', 0.0)
        self.declare_parameter('local_replan_on_waypoint_change', False)
        self.declare_parameter('latency_compensate_plan_handoff', False)
        self.declare_parameter('cmd_publish_rate', 10.0)
        self.declare_parameter('heading_update_mode', 'camera_xy_only')
        self.declare_parameter('use_nogo_cost', False)
        self.declare_parameter('nogo_penalty_type', 'warning_band')
        self.declare_parameter('nogo_weight', 0.0)
        self.declare_parameter('nogo_safe_distance', 0.35)
        self.declare_parameter('nogo_logbarrier_eps', 1e-3)
        self.declare_parameter('nogo_warning_band', 0.05)
        self.declare_parameter('nogo_near_weight', 50.0)
        self.declare_parameter('use_belief_nogo_cost', False)
        self.declare_parameter('nogo_belief_kappa', 1.0)
        self.declare_parameter('nogo_mode', 'keep_out')
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
        self.declare_parameter('show_pose_markers', False)
        self.declare_parameter('diagnostics_match_tolerance_s', 1e-3)
        self.declare_parameter('bev_y_calibration_offset_m', 0.0)
        self.declare_parameter('bev_affine_calibration', '')
        self.declare_parameter('bbox_contact_z_m', 0.0)
        self.declare_parameter('pixel_correction_nis_threshold', 0.0)
        self.declare_parameter('odom_topic', '/odom_noisy')
        self.declare_parameter('run_dir_topic', '/experiment/run_dir')
        self.declare_parameter('run_timeout_after_first_cmd_s', 75.0)
        self.declare_parameter('first_cmd_linear_eps', 0.02)
        self.declare_parameter('first_cmd_angular_eps', 0.10)
        self.declare_parameter('stuck_window_s', 8.0)
        self.declare_parameter('stuck_max_displacement_m', 0.08)
        self.declare_parameter('stuck_max_goal_improvement_m', 0.05)
        self.declare_parameter('stuck_cmd_fraction_min', 0.50)
        self.declare_parameter('stuck_idle_cmd_fraction_max', 0.10)
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
        self.heading_update_mode = str(self.get_parameter('heading_update_mode').value)
        self.use_pixel_correction = bool(self.get_parameter('use_pixel_correction').value)
        self.pixel_timeout_s = float(self.get_parameter('pixel_timeout_s').value)
        self.use_ambiguity = bool(self.get_parameter('use_ambiguity').value)
        self.use_obs_risk = bool(self.get_parameter('use_obs_risk').value)
        self.world_profiles_path = self.get_parameter('world_profiles_path').value
        self.tasks_yaml = self.get_parameter('tasks_yaml').value
        self.log_plan_samples = bool(self.get_parameter('log_plan_samples').value)
        self.log_perception_samples = bool(self.get_parameter('log_perception_samples').value)
        self.auto_stop_on_goal = bool(self.get_parameter('auto_stop_on_goal').value)
        self.goal_success_radius = float(self.get_parameter('goal_success_radius').value)
        self.goal_success_hold_s = float(self.get_parameter('goal_success_hold_s').value)
        self.goal_stable_radius = float(self.get_parameter('goal_stable_radius').value)
        self.goal_stable_hold_s = float(self.get_parameter('goal_stable_hold_s').value)
        self.goal_stable_max_displacement_m = float(
            self.get_parameter('goal_stable_max_displacement_m').value
        )
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.frame_sanity_start_tolerance_m = float(self.get_parameter('frame_sanity_start_tolerance_m').value)
        self.frame_sanity_start_tolerance_yaw_rad = float(self.get_parameter('frame_sanity_start_tolerance_yaw_rad').value)
        self.use_visibility_model = bool(self.get_parameter('use_visibility_model').value)
        self.visibility_artifact_path = str(self.get_parameter('visibility_artifact_path').value)
        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.ambiguity_weight = float(self.get_parameter('ambiguity_weight').value)
        self.goal_sigma_uv = float(self.get_parameter('goal_sigma_uv').value)
        self.r_visible_uv = float(self.get_parameter('r_visible_uv').value)
        self.r_miss_uv = float(self.get_parameter('r_miss_uv').value)
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
        self.discount_gamma = float(self.get_parameter('discount_gamma').value)
        self.visibility_target_height_m = float(self.get_parameter('visibility_target_height_m').value)
        self.perception_use_geometry_occlusion = bool(
            self.get_parameter('perception_use_geometry_occlusion').value
        )
        self.visibility_geometry_json = str(self.get_parameter('visibility_geometry_json').value)
        self.collision_geometry_json = str(self.get_parameter('collision_geometry_json').value)
        self.robot_collision_radius_m = float(self.get_parameter('robot_collision_radius_m').value)
        self.terminate_on_geom_collision = bool(self.get_parameter('terminate_on_geom_collision').value)
        self.use_command_noise = bool(self.get_parameter('use_command_noise').value)
        self.command_noise_linear_slip_mean = float(self.get_parameter('command_noise_linear_slip_mean').value)
        self.command_noise_linear_slip_std = float(self.get_parameter('command_noise_linear_slip_std').value)
        self.command_noise_angular_slip_mean = float(self.get_parameter('command_noise_angular_slip_mean').value)
        self.command_noise_angular_slip_std = float(self.get_parameter('command_noise_angular_slip_std').value)
        self.command_noise_linear_additive_std = float(self.get_parameter('command_noise_linear_additive_std').value)
        self.command_noise_angular_additive_std = float(self.get_parameter('command_noise_angular_additive_std').value)
        self.command_noise_correlation_alpha = float(self.get_parameter('command_noise_correlation_alpha').value)
        self.encoder_noise_angular_slip_std = float(self.get_parameter('encoder_noise_angular_slip_std').value)
        self.encoder_noise_angular_additive_std = float(self.get_parameter('encoder_noise_angular_additive_std').value)
        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_maxfun = int(self.get_parameter('optimizer_maxfun').value)
        self.optimizer_ftol = float(self.get_parameter('optimizer_ftol').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_warm_start = bool(self.get_parameter('optimizer_warm_start').value)
        self.optimizer_multistart = bool(self.get_parameter('optimizer_multistart').value)
        self.optimizer_multistart_include_direct = bool(
            self.get_parameter('optimizer_multistart_include_direct').value
        )
        self.optimizer_initial_routes_json = str(
            self.get_parameter('optimizer_initial_routes_json').value
        )
        self.optimizer_route_seed_mode = str(
            self.get_parameter('optimizer_route_seed_mode').value or 'explicit'
        )
        self.use_hierarchical = bool(self.get_parameter('use_hierarchical').value)
        self.global_horizon = int(self.get_parameter('global_horizon').value)
        self.global_dt = float(self.get_parameter('global_dt').value)
        self.local_horizon = int(self.get_parameter('local_horizon').value)
        self.local_plan_rate = float(self.get_parameter('local_plan_rate').value)
        self.local_optimizer_maxiter = int(self.get_parameter('local_optimizer_maxiter').value)
        self.global_use_ambiguity = bool(self.get_parameter('global_use_ambiguity').value)
        self.local_use_ambiguity = bool(self.get_parameter('local_use_ambiguity').value)
        self.global_optimizer_multistart = bool(
            self.get_parameter('global_optimizer_multistart').value
        )
        self.local_optimizer_multistart = bool(
            self.get_parameter('local_optimizer_multistart').value
        )
        self.local_use_visibility_model = bool(
            self.get_parameter('local_use_visibility_model').value
        )
        self.local_use_belief_nogo_cost = bool(
            self.get_parameter('local_use_belief_nogo_cost').value
        )
        self.local_nogo_penalty_type = str(
            self.get_parameter('local_nogo_penalty_type').value or ''
        )
        self.local_nogo_weight = float(self.get_parameter('local_nogo_weight').value)
        self.local_nogo_safe_distance = float(
            self.get_parameter('local_nogo_safe_distance').value
        )
        self.local_goal_prior_u_std_start = float(
            self.get_parameter('local_goal_prior_u_std_start').value
        )
        self.local_goal_prior_v_std_start = float(
            self.get_parameter('local_goal_prior_v_std_start').value
        )
        self.local_goal_prior_u_std_final = float(
            self.get_parameter('local_goal_prior_u_std_final').value
        )
        self.local_goal_prior_v_std_final = float(
            self.get_parameter('local_goal_prior_v_std_final').value
        )
        self.waypoint_spacing_m = float(self.get_parameter('waypoint_spacing_m').value)
        self.waypoint_arrival_radius_m = float(
            self.get_parameter('waypoint_arrival_radius_m').value
        )
        self.local_replan_min_remaining_s = float(
            self.get_parameter('local_replan_min_remaining_s').value
        )
        self.local_replan_on_waypoint_change = bool(
            self.get_parameter('local_replan_on_waypoint_change').value
        )
        self.latency_compensate_plan_handoff = bool(
            self.get_parameter('latency_compensate_plan_handoff').value
        )
        self.cmd_publish_rate = float(self.get_parameter('cmd_publish_rate').value)
        self.use_nogo_cost = bool(self.get_parameter('use_nogo_cost').value)
        self.nogo_penalty_type = str(self.get_parameter('nogo_penalty_type').value)
        self.nogo_weight = float(self.get_parameter('nogo_weight').value)
        self.nogo_safe_distance = float(self.get_parameter('nogo_safe_distance').value)
        self.nogo_logbarrier_eps = float(self.get_parameter('nogo_logbarrier_eps').value)
        self.nogo_warning_band = float(self.get_parameter('nogo_warning_band').value)
        self.nogo_near_weight = float(self.get_parameter('nogo_near_weight').value)
        self.use_belief_nogo_cost = bool(self.get_parameter('use_belief_nogo_cost').value)
        self.nogo_belief_kappa = float(self.get_parameter('nogo_belief_kappa').value)
        self.nogo_mode = str(self.get_parameter('nogo_mode').value or 'keep_out')
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
        self.show_pose_markers = bool(self.get_parameter('show_pose_markers').value)
        self.diagnostics_match_tolerance_s = float(
            self.get_parameter('diagnostics_match_tolerance_s').value
        )
        self.bev_y_calibration_offset_m = float(
            self.get_parameter('bev_y_calibration_offset_m').value
        )
        self.bev_affine_calibration = str(
            self.get_parameter('bev_affine_calibration').value or ''
        ).strip()
        self._bev_affine = self._parse_bev_affine(self.bev_affine_calibration)
        self.bbox_contact_z_m = float(self.get_parameter('bbox_contact_z_m').value)
        self.pixel_correction_nis_threshold = float(
            self.get_parameter('pixel_correction_nis_threshold').value
        )
        self.odom_topic = str(self.get_parameter('odom_topic').value or '/odom_noisy')
        self.run_dir_topic = str(self.get_parameter('run_dir_topic').value).strip() or '/experiment/run_dir'
        self.run_timeout_after_first_cmd_s = float(self.get_parameter('run_timeout_after_first_cmd_s').value)
        self.first_cmd_linear_eps = float(self.get_parameter('first_cmd_linear_eps').value)
        self.first_cmd_angular_eps = float(self.get_parameter('first_cmd_angular_eps').value)
        self.stuck_window_s = float(self.get_parameter('stuck_window_s').value)
        self.stuck_max_displacement_m = float(self.get_parameter('stuck_max_displacement_m').value)
        self.stuck_max_goal_improvement_m = float(self.get_parameter('stuck_max_goal_improvement_m').value)
        self.stuck_cmd_fraction_min = float(self.get_parameter('stuck_cmd_fraction_min').value)
        self.stuck_idle_cmd_fraction_max = float(
            self.get_parameter('stuck_idle_cmd_fraction_max').value
        )

        # Camera model is built below from the world profile (the single source of
        # truth shared with the state/planner nodes), NOT from the cam_pos/look_at
        # node parameters — those are never passed by the launch, so reading them
        # silently selected stale defaults ([-3,-3,6]/[1.5,1.5,0]) and corrupted the
        # pred_world homography diagnostic. See profile-based build after load_profile.

        run_info = create_run_dir(log_dir)
        self.run_id = run_info['run_id']
        self.run_dir = run_info['run_dir']

        self.log_path = os.path.join(self.run_dir, 'experiment.csv')

        repo_root = _find_repo_root(os.getcwd())
        self.repo_root = repo_root
        self.task_start_pose = _load_task_start_pose(self.tasks_yaml, self.world, self.task)
        profile, _intrinsics, _world_path, _camera_pose = load_profile(self.world_profiles_path, self.world)

        # Build the homography camera from the profile (same source the state/planner
        # nodes use) so pred_world_x/y match the true Gazebo camera. _camera_pose is
        # [x, y, z, roll, pitch, yaw]; look_at is derived exactly as the launch does.
        from unav_common.camera_model import ObliqueCameraModel
        _cam_pos = [float(_camera_pose[0]), float(_camera_pose[1]), float(_camera_pose[2])]
        _look_at = compute_look_at_from_pose(
            _cam_pos, float(_camera_pose[3]), float(_camera_pose[4]), float(_camera_pose[5])
        )
        self.camera_model = ObliqueCameraModel(
            cam_pos=np.array(_cam_pos, dtype=float),
            look_at=np.array(_look_at, dtype=float),
            img_width=int(_intrinsics['img_width']),
            img_height=int(_intrinsics['img_height']),
            fov_h_rad=float(_intrinsics['fov_h_rad']),
        )
        self.camera_pos_xy = np.asarray(_cam_pos[:2], dtype=float).reshape(2)
        self.get_logger().info(
            f"[camera_model] profile-built cam_pos={_cam_pos} look_at={_look_at} "
            f"img=({int(_intrinsics['img_width'])}x{int(_intrinsics['img_height'])}) "
            f"fov_h={float(_intrinsics['fov_h_rad']):.4f}"
        )

        visibility_defaults = dict(profile.get('visibility_defaults') or {})
        self.world_bounds = {
            'xmin': float(visibility_defaults.get('visibility_map_min_x', math.nan)),
            'xmax': float(visibility_defaults.get('visibility_map_max_x', math.nan)),
            'ymin': float(visibility_defaults.get('visibility_map_min_y', math.nan)),
            'ymax': float(visibility_defaults.get('visibility_map_max_y', math.nan)),
        }

        collision_scene = scene_from_json(self.collision_geometry_json)
        self._collision_prisms = tuple(collision_scene.prisms)
        self._wall_prisms = _split_prisms_by_prefix(self._collision_prisms, 'warehouse_walls/')
        self._obstacle_prisms = _split_prisms_by_prefix(self._collision_prisms, 'warehouse_rack_occluders/')
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
            'heading_update_mode': self.heading_update_mode,
            'use_pixel_correction': self.use_pixel_correction,
            'pixel_timeout_s': self.pixel_timeout_s,
            'use_ambiguity': self.use_ambiguity,
            'use_obs_risk': self.use_obs_risk,
            'use_visibility_model': self.use_visibility_model,
            'visibility_artifact_path': self.visibility_artifact_path,
            'risk_weight_obs': self.risk_weight_obs,
            'ambiguity_weight': self.ambiguity_weight,
            'goal_sigma_uv': self.goal_sigma_uv,
            'r_visible_uv': self.r_visible_uv,
            'r_miss_uv': self.r_miss_uv,
            'visibility_sigma_kappa': self.visibility_sigma_kappa,
            'goal_prior_u_std_start': self.goal_prior_u_std_start,
            'goal_prior_v_std_start': self.goal_prior_v_std_start,
            'goal_prior_u_std_final': self.goal_prior_u_std_final,
            'goal_prior_v_std_final': self.goal_prior_v_std_final,
            'goal_tightening_power': self.goal_tightening_power,
            'goal_progress_n_steps': self.goal_progress_n_steps,
            'observation_risk_scale': self.observation_risk_scale,
            'ambiguity_term_scale': self.ambiguity_term_scale,
            'discount_gamma': self.discount_gamma,
            'visibility_target_height_m': self.visibility_target_height_m,
            'visibility_geometry_json': self.visibility_geometry_json,
            'visibility_geometry_sha256': _sha256_text(self.visibility_geometry_json),
            'collision_geometry_json': self.collision_geometry_json,
            'collision_geometry_sha256': _sha256_text(self.collision_geometry_json),
            'robot_collision_radius_m': self.robot_collision_radius_m,
            'use_command_noise': self.use_command_noise,
            'command_noise_linear_slip_mean': self.command_noise_linear_slip_mean,
            'command_noise_linear_slip_std': self.command_noise_linear_slip_std,
            'command_noise_angular_slip_mean': self.command_noise_angular_slip_mean,
            'command_noise_angular_slip_std': self.command_noise_angular_slip_std,
            'command_noise_linear_additive_std': self.command_noise_linear_additive_std,
            'command_noise_angular_additive_std': self.command_noise_angular_additive_std,
            'command_noise_correlation_alpha': self.command_noise_correlation_alpha,
            'encoder_noise_angular_slip_std': self.encoder_noise_angular_slip_std,
            'encoder_noise_angular_additive_std': self.encoder_noise_angular_additive_std,
            'perception_use_geometry_occlusion': self.perception_use_geometry_occlusion,
            'use_nogo_cost': self.use_nogo_cost,
            'nogo_penalty_type': self.nogo_penalty_type,
            'nogo_weight': self.nogo_weight,
            'nogo_safe_distance': self.nogo_safe_distance,
            'nogo_logbarrier_eps': self.nogo_logbarrier_eps,
            'nogo_warning_band': self.nogo_warning_band,
            'nogo_near_weight': self.nogo_near_weight,
            'use_belief_nogo_cost': self.use_belief_nogo_cost,
            'nogo_belief_kappa': self.nogo_belief_kappa,
            'nogo_mode': self.nogo_mode,
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
            'show_pose_markers': self.show_pose_markers,
            'diagnostics_match_tolerance_s': self.diagnostics_match_tolerance_s,
            'bev_y_calibration_offset_m': self.bev_y_calibration_offset_m,
            'bev_affine_calibration': self.bev_affine_calibration,
            'bbox_contact_z_m': self.bbox_contact_z_m,
            'pixel_correction_nis_threshold': self.pixel_correction_nis_threshold,
            'odom_topic': self.odom_topic,
            'seed': self.seed,
            'state_pipeline': 'homography_to_bev',
            'observation_model': 'uv',
            'world_bounds': dict(self.world_bounds),
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
            'optimizer_multistart': self.optimizer_multistart,
            'optimizer_multistart_include_direct': self.optimizer_multistart_include_direct,
            'optimizer_initial_routes_json': self.optimizer_initial_routes_json,
            'optimizer_route_seed_mode': self.optimizer_route_seed_mode,
            'use_hierarchical': self.use_hierarchical,
            'global_horizon': self.global_horizon,
            'global_dt': self.global_dt,
            'local_horizon': self.local_horizon,
            'local_plan_rate': self.local_plan_rate,
            'local_optimizer_maxiter': self.local_optimizer_maxiter,
            'global_use_ambiguity': self.global_use_ambiguity,
            'local_use_ambiguity': self.local_use_ambiguity,
            'global_optimizer_multistart': self.global_optimizer_multistart,
            'local_optimizer_multistart': self.local_optimizer_multistart,
            'local_use_visibility_model': self.local_use_visibility_model,
            'local_use_belief_nogo_cost': self.local_use_belief_nogo_cost,
            'local_nogo_penalty_type': self.local_nogo_penalty_type,
            'local_nogo_weight': self.local_nogo_weight,
            'local_nogo_safe_distance': self.local_nogo_safe_distance,
            'local_goal_prior_u_std_start': self.local_goal_prior_u_std_start,
            'local_goal_prior_v_std_start': self.local_goal_prior_v_std_start,
            'local_goal_prior_u_std_final': self.local_goal_prior_u_std_final,
            'local_goal_prior_v_std_final': self.local_goal_prior_v_std_final,
            'waypoint_spacing_m': self.waypoint_spacing_m,
            'waypoint_arrival_radius_m': self.waypoint_arrival_radius_m,
            'local_replan_min_remaining_s': self.local_replan_min_remaining_s,
            'local_replan_on_waypoint_change': self.local_replan_on_waypoint_change,
            'latency_compensate_plan_handoff': self.latency_compensate_plan_handoff,
            'cmd_publish_rate': self.cmd_publish_rate,
            'auto_stop_on_goal': self.auto_stop_on_goal,
            'goal_success_radius': self.goal_success_radius,
            'goal_success_hold_s': self.goal_success_hold_s,
            'goal_stable_radius': self.goal_stable_radius,
            'goal_stable_hold_s': self.goal_stable_hold_s,
            'goal_stable_max_displacement_m': self.goal_stable_max_displacement_m,
            'run_timeout_after_first_cmd_s': self.run_timeout_after_first_cmd_s,
            'first_cmd_linear_eps': self.first_cmd_linear_eps,
            'first_cmd_angular_eps': self.first_cmd_angular_eps,
            'stuck_window_s': self.stuck_window_s,
            'stuck_max_displacement_m': self.stuck_max_displacement_m,
            'stuck_max_goal_improvement_m': self.stuck_max_goal_improvement_m,
            'stuck_cmd_fraction_min': self.stuck_cmd_fraction_min,
            'stuck_idle_cmd_fraction_max': self.stuck_idle_cmd_fraction_max,
        }
        self._manifest_data = dict(manifest_data)
        write_manifest(self.run_dir, self._manifest_data, repo_root)
        snapshot_configs(self.run_dir, [self.world_profiles_path, self.tasks_yaml])

        self.state_msg = None
        self.planner_belief_msg = None
        self.odom_msg = None
        self.odom_noisy_msg = None
        # TRUE Gazebo pose (world frame == map_bev) from /ground_truth_tf, held as
        # latest (x, y). Lets us measure error vs GROUND TRUTH instead of vs /odom,
        # which is DiffDrive wheel odometry and itself drifts in turns.
        self._gt_xy = None
        self._gt_yaw = None  # TRUE heading from /ground_truth_tf (for GT heading error)
        # Buffer of (stamp_s, x, y, yaw) ground truth in the map_bev frame, so the
        # camera measurement / state can be compared to truth AT THEIR OWN capture
        # time (not the current log time). This separates true detector quality from
        # the ~0.5 s measurement latency.
        self._truth_buf = deque(maxlen=600)
        self.obs_msg = None
        self.perception_diag = None
        self.heading_diag = None
        self.pixel_correction_diag = None
        self.cmd_msg = None
        self.cmd_raw_msg = None
        self.cmd_noise_diag = None
        self.cmd_stamp_s = math.nan
        self.cmd_raw_stamp_s = math.nan
        self.cmd_noise_diag_stamp_s = math.nan
        self.goal_msg = None
        self.plan_msg = None
        self.planner_diag = None
        self.planner_diag_text = ''
        self.active_execution_diag = None
        self.efe_metrics = None
        self._goal_in_radius_since = None
        self._goal_stable_since = None
        self._goal_region_entered = False
        self._goal_region_first_stamp = math.nan
        self._motion_history = deque()
        self._stop_requested = False
        self._completed = False
        self._last_tf_warn_wall = 0.0
        self._frame_sanity_logged = False
        self._frame_sanity = {
            'recorded': False,
            'ok': None,
            'reason': 'pending',
            'source_frame': '',
            'odom_map_stamp': math.nan,
            'raw_odom_x': math.nan,
            'raw_odom_y': math.nan,
            'raw_odom_yaw': math.nan,
            'odom_map_x': math.nan,
            'odom_map_y': math.nan,
            'odom_map_yaw': math.nan,
            'task_start_x': float(self.task_start_pose[0]) if self.task_start_pose is not None else math.nan,
            'task_start_y': float(self.task_start_pose[1]) if self.task_start_pose is not None else math.nan,
            'task_start_yaw': float(self.task_start_pose[2]) if self.task_start_pose is not None else math.nan,
            'odom_map_start_error_m': math.nan,
            'raw_start_error_m': math.nan,
            'odom_map_start_yaw_error_rad': math.nan,
            'raw_start_yaw_error_rad': math.nan,
            'tolerance_m': self.frame_sanity_start_tolerance_m,
            'tolerance_yaw_rad': self.frame_sanity_start_tolerance_yaw_rad,
        }
        self._rewrite_manifest()

        self._first_cmd_stamp = None
        self._cumulative_path_length = 0.0
        self._last_path_pose = None
        self._min_goal_distance = float('inf')
        self._contact_collision_seen = False
        self._geom_collision_seen = False
        self._collision_reason = ''
        self._first_crash_stamp = math.nan
        self._min_wall_distance = float('inf')
        self._min_obstacle_distance = float('inf')
        self._max_wall_penetration = 0.0
        self._max_obstacle_penetration = 0.0
        self._off_map_seen = False
        self._inside_no_go_seen = False
        self._valid_run = True
        self._invalid_reason = ''

        self._efe_risk_sum = 0.0
        self._efe_ambiguity_sum = 0.0
        self._efe_control_sum = 0.0
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
        self._state_error_odom_sum = 0.0
        self._belief_error_odom_sum = 0.0
        self._state_error_odom_count = 0
        self._belief_error_odom_count = 0
        self._state_error_odom_after_first_cmd_sum = 0.0
        self._belief_error_odom_after_first_cmd_sum = 0.0
        self._state_error_odom_after_first_cmd_count = 0
        self._belief_error_odom_after_first_cmd_count = 0
        self._odom_map_vs_odom_yaw_error_sum = 0.0
        self._odom_map_vs_state_yaw_error_sum = 0.0
        self._odom_map_vs_belief_yaw_error_sum = 0.0
        self._truth_odom_yaw_error_count = 0
        self._truth_state_yaw_error_count = 0
        self._truth_belief_yaw_error_count = 0
        self._odom_map_vs_odom_yaw_error_after_first_cmd_sum = 0.0
        self._odom_map_vs_state_yaw_error_after_first_cmd_sum = 0.0
        self._odom_map_vs_belief_yaw_error_after_first_cmd_sum = 0.0
        self._odom_map_vs_odom_yaw_error_after_first_cmd_count = 0
        self._odom_map_vs_state_yaw_error_after_first_cmd_count = 0
        self._odom_map_vs_belief_yaw_error_after_first_cmd_count = 0
        # GROUND-TRUTH error means (vs the real Gazebo pose gt_x/gt_y), the honest
        # counterpart to the odom-based mean_truth_* means above.
        self._belief_error_gt_sum = 0.0
        self._belief_error_gt_count = 0
        self._state_error_gt_sum = 0.0
        self._state_error_gt_count = 0
        self._belief_error_gt_after_first_cmd_sum = 0.0
        self._belief_error_gt_after_first_cmd_count = 0
        self._state_error_gt_after_first_cmd_sum = 0.0
        self._state_error_gt_after_first_cmd_count = 0
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        run_dir_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.run_dir_pub = self.create_publisher(String, self.run_dir_topic, qos_profile=run_dir_qos)
        run_dir_msg = String()
        run_dir_msg.data = self.run_dir
        self.run_dir_pub.publish(run_dir_msg)

        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Odometry, '/odom_noisy', self._odom_noisy_cb, 10)
        self.create_subscription(TFMessage, '/ground_truth_tf', self._ground_truth_cb, 50)
        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_subscription(Float64MultiArray, '/state/heading_diagnostics', self._heading_diag_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/planner_belief', self._planner_belief_cb, 10)
        self.create_subscription(PoseStamped, '/perception/pixel_pose', self._obs_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            DETECTION_DIAGNOSTICS_TOPIC,
            self._diag_cb,
            10,
        )
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_raw', self._cmd_raw_cb, 10)
        self.create_subscription(Float64MultiArray, '/cmd_vel_noise/diagnostics', self._cmd_noise_diag_cb, 10)
        self.create_subscription(PoseStamped, '/goal_bev', self._goal_cb, qos_profile=goal_qos)
        self.create_subscription(Path, '/plan_preview', self._plan_cb, 10)
        self.create_subscription(Float64MultiArray, '/planner/diagnostics', self._planner_diag_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            '/planner/active_execution_diagnostics',
            self._active_execution_diag_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            '/planner/pixel_correction_diagnostics',
            self._pixel_correction_diag_cb,
            10,
        )
        self.create_subscription(String, '/planner/diagnostics_text', self._planner_diag_text_cb, 10)
        self.create_subscription(Float64MultiArray, '/efe/metrics', self._efe_cb, 10)
        self.create_subscription(Contacts, '/world_contacts', self._contacts_cb, 10)

        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'stamp',
            'odom_map_available', 'odom_map_stamp', 'odom_map_x', 'odom_map_y', 'odom_map_yaw',
            'state_available', 'state_stamp', 'state_x', 'state_y', 'state_yaw',
            'state_age_s', 'state_fresh',
            'state_cov_xx', 'state_cov_xy', 'state_cov_yy', 'state_cov_yaw',
            'planner_belief_available', 'planner_belief_stamp',
            'planner_belief_age_s',
            'planner_belief_x', 'planner_belief_y', 'planner_belief_yaw',
            'planner_cov_x', 'planner_cov_xy', 'planner_cov_y', 'planner_cov_yaw',
            'est_available', 'est_x', 'est_y', 'est_yaw',
            'est_cov_xx', 'est_cov_xy', 'est_cov_yy',
            'state_pos_error_m', 'state_cov_trace', 'state_cov_det',
            'state_sigma_major_m', 'state_sigma_minor_m', 'state_entropy_xy',
            # Explicit unambiguous error columns:
            # state_error_odom_m  = ||truth - /state/bev||   (perception estimate vs ground truth)
            # belief_error_odom_m = ||truth - /planner_belief||  (planner internal state vs ground truth)
            'state_error_odom_m', 'belief_error_odom_m',
            'odom_available', 'odom_stamp', 'odom_x', 'odom_y', 'odom_yaw',
            'odom_v', 'odom_w',
            'odom_noisy_available', 'odom_noisy_stamp', 'odom_noisy_x', 'odom_noisy_y',
            'odom_noisy_yaw', 'odom_noisy_v', 'odom_noisy_w',
            'yaw_error_odom_map_vs_odom_rad', 'yaw_error_odom_map_vs_state_rad', 'yaw_error_odom_map_vs_belief_rad',
            'pixel_yaw_meas', 'heading_source_code', 'heading_source',
            'heading_diag_stamp', 'heading_diag_age_s',
            'state_heading_yaw_sigma', 'state_heading_odom_age_s',
            'planner_pixel_correction_available', 'planner_pixel_correction_stamp',
            'planner_pixel_correction_age_s',
            'pixel_corr_innov_u', 'pixel_corr_innov_v',
            'pixel_corr_xy_update_norm_m', 'pixel_corr_theta_update_from_uv_rad',
            'pixel_corr_nis',
            'pixel_corr_accepted', 'pixel_corr_reject_reason_code', 'pixel_corr_reject_reason',
            'pixel_corr_apply_stamp', 'pixel_corr_belief_input_stamp',
            'pixel_corr_cmd_replay_count', 'pixel_corr_cmd_replay_duration_s',
            'pixel_corr_cmd_replay_used_fallback',
            'pixel_corr_motion_replay_source_code', 'pixel_corr_motion_replay_source',
            'pixel_corr_nis_threshold',
            'pixel_heading_correction_applied', 'pixel_heading_meas_source',
            'pixel_heading_innov_rad',
            'pixel_heading_gain_theta', 'pixel_corr_theta_update_total_rad',
            'pixel_corr_pred_x', 'pixel_corr_pred_y', 'pixel_corr_pred_yaw',
            'pixel_corr_next_x', 'pixel_corr_next_y', 'pixel_corr_next_yaw',
            'pixel_corr_expected_after_u', 'pixel_corr_expected_after_v',
            'pixel_corr_expected_after_visible',
            'cmd_v', 'cmd_w',
            'cmd_raw_v', 'cmd_raw_w',
            'cmd_stamp', 'cmd_age_s', 'cmd_raw_stamp', 'cmd_raw_age_s',
            'cmd_noise_enabled',
            'cmd_noise_linear_multiplier', 'cmd_noise_angular_multiplier',
            'cmd_noise_linear_additive', 'cmd_noise_angular_additive',
            'cmd_noise_v_error', 'cmd_noise_w_error',
            'goal_x', 'goal_y', 'goal_dist',
            'plan_points', 'plan_length',
            'optimizer_success', 'optimizer_status', 'optimizer_nit', 'optimizer_nfev', 'optimizer_message',
            'plan_time_ms', 'solve_time_ms',
            'measurement_available', 'belief_age_s',
            'p_vis_plan', 'p_vis_plan_eff',
            'r_plan_u_std', 'r_plan_v_std',
            'terminal_goal_distance_pred', 'terminal_goal_progress_m',
            'fraction_horizon_low_pvis', 'fraction_horizon_high_ambiguity',
            'min_predicted_obstacle_distance_m', 'rollout_valid',
            'efe_total', 'efe_risk', 'efe_ambiguity', 'efe_control', 'efe_obstacle',
            'efe_risk_mean', 'efe_risk_cov_trace', 'efe_risk_cov_logdet',
            'efe_delta_risk_visibility', 'efe_delta_ambiguity_visibility',
            'active_plan_age_s', 'active_plan_remaining_s', 'active_control_index',
            'active_controls_len', 'active_controls_original_len',
            'latency_skip_steps', 'latency_skip_s',
            'command_timer_period_s', 'planner_timer_period_s',
            'pending_plan_started_active_remaining_s',
            'exec_plan_age_s', 'exec_plan_remaining_s', 'exec_control_index',
            'exec_controls_len', 'exec_controls_original_len',
            'exec_cmd_v', 'exec_cmd_w', 'exec_latency_skip_steps',
            'exec_latency_skip_s',
            'exec_wp_idx', 'exec_wp_count', 'exec_wp_target_x', 'exec_wp_target_y',
            'exec_wp_dist_m', 'exec_desired_yaw', 'exec_yaw_error',
            'exec_tracking_yaw', 'exec_tracking_yaw_source',
            'collision_any', 'collision_contact', 'collision_geom', 'collision_reason', 'first_crash_stamp',
            'min_wall_distance_m', 'min_obstacle_distance_m',
            'wall_penetration_m', 'obstacle_penetration_m',
            'off_map', 'inside_no_go', 'valid_run', 'invalid_reason',
            'heading_update_mode',
            'pixel_corr_K_theta_u', 'pixel_corr_K_theta_v',
            'yaw_error_odom_noisy_vs_odom_map_rad',
            'state_bev_yaw_latest',
            'state_bev_cov_theta_theta', 'state_bev_cov_x_theta', 'state_bev_cov_y_theta',
            'planner_belief_cov_theta_theta', 'planner_belief_cov_x_theta', 'planner_belief_cov_y_theta',
            'planner_diag_prediction_source', 'planner_diag_prediction_dt',
            'planner_diag_u_pred_v', 'planner_diag_u_pred_omega', 'planner_diag_Q_theta_theta',
            'planner_diag_odom_delta_theta', 'planner_diag_cmd_delta_theta',
            'planner_diag_heading_anchor_applied', 'planner_diag_state_bev_yaw_ignored',
            # Ground-truth (vs TRUE Gazebo pose, not /odom wheel odometry):
            'gt_available', 'gt_x', 'gt_y', 'gt_yaw',
            'belief_error_gt_m', 'state_error_gt_m', 'odom_map_gt_drift_m',
            'belief_yaw_error_gt_rad',
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
                'state_age_s',
                'state_fresh',
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
                'pixel_pose_fresh',
                'pred_world_x',
                'pred_world_y',
                'localization_error_m',
                'pred_world_x_calibrated',
                'pred_world_y_calibrated',
                'localization_error_calibrated_m',
                # vs truth at the measurement's OWN capture time (latency-removed):
                # true detector quality. The _calibrated_m above is vs log-time truth
                # (latency-inflated in turns). state_error_captime_m = same for /state.
                'localization_error_captime_m',
                'state_error_captime_m',
                'bev_y_calibration_offset_m',
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
                'yolo_raw_best_score',
                'yolo_selected_score',
                'yolo_num_target_candidates',
                'yolo_selected_class_id',
                'yolo_selected_pixel_source',
                'yolo_bbox_area',
                'yolo_mask_area',
                'yolo_inference_ms',
                'detector_callback_ms',
                'yolo_receive_stamp',
                'yolo_start_stamp',
                'yolo_finish_stamp',
                'yolo_publish_stamp',
                'yolo_latency_s',
                'frame_age_at_publish_s',
                'detector_total_latency_s',
                'camera_relative_bearing_deg',
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
                f"for {self.goal_success_hold_s:.2f} s; stable/idle goal radius <= "
                f"{self.goal_stable_radius:.3f} m for {self.goal_stable_hold_s:.2f} s"
            )
        if self.stuck_window_s > 0.0:
            self.get_logger().info(
                f"Stuck-stop enabled: {self.stuck_window_s:.2f}s window, "
                f"max displacement {self.stuck_max_displacement_m:.3f} m, "
                f"max goal improvement {self.stuck_max_goal_improvement_m:.3f} m, "
                f"active cmd fraction >= {self.stuck_cmd_fraction_min:.2f}, "
                f"idle cmd fraction <= {self.stuck_idle_cmd_fraction_max:.2f}"
            )

    @staticmethod
    def _stamp_to_float(stamp_msg) -> float:
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9

    @staticmethod
    def _parse_bev_affine(raw: str):
        text = str(raw or '').strip()
        if not text:
            return None
        try:
            vals = [float(v) for v in text.replace(';', ',').split(',') if v.strip()]
        except ValueError:
            return None
        if len(vals) != 6:
            return None
        return vals

    def _apply_bev_calibration(self, x: float, y: float) -> tuple[float, float]:
        if self._bev_affine is not None:
            c = self._bev_affine
            return (
                c[0] * float(x) + c[1] * float(y) + c[2],
                c[3] * float(x) + c[4] * float(y) + c[5],
            )
        return float(x), float(y) + self.bev_y_calibration_offset_m

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

    def _command_active(self, cmd_v: float, cmd_w: float) -> bool:
        return (
            abs(float(cmd_v)) >= self.first_cmd_linear_eps
            or abs(float(cmd_w)) >= self.first_cmd_angular_eps
        )

    def _remember_motion_sample(
        self,
        stamp: float,
        true_ok: bool,
        true_x: float,
        true_y: float,
        goal_dist: float,
        cmd_v: float,
        cmd_w: float,
    ) -> None:
        # Stuck / goal-stable detection uses the TRUE Gazebo pose ONLY (no /odom
        # fallback): drifting wheel-odom would misreport displacement/progress.
        if self._gt_xy is None:
            return
        gx, gy = self._gt_xy
        if not (math.isfinite(stamp) and math.isfinite(gx) and math.isfinite(gy) and math.isfinite(goal_dist)):
            return
        self._motion_history.append((
            float(stamp),
            float(gx),
            float(gy),
            float(goal_dist),
            1.0 if self._command_active(cmd_v, cmd_w) else 0.0,
        ))
        keep_window_s = max(
            float(self.stuck_window_s),
            float(self.goal_success_hold_s),
            float(self.goal_stable_hold_s),
            1.0,
        ) + 1.0
        while self._motion_history and stamp - self._motion_history[0][0] > keep_window_s:
            self._motion_history.popleft()

    def _motion_window_stats(self, stamp: float, window_s: float):
        if window_s <= 0.0 or len(self._motion_history) < 2:
            return None
        window_start = stamp - window_s
        samples = [sample for sample in self._motion_history if sample[0] >= window_start]
        if len(samples) < 2:
            return None
        duration_s = float(samples[-1][0] - samples[0][0])
        if duration_s < min(max(0.5 * window_s, 0.5), window_s):
            return None
        displacement_m = float(math.hypot(samples[-1][1] - samples[0][1], samples[-1][2] - samples[0][2]))
        goal_improvement_m = float(samples[0][3] - samples[-1][3])
        cmd_fraction = float(sum(sample[4] for sample in samples) / len(samples))
        return {
            'duration_s': duration_s,
            'displacement_m': displacement_m,
            'goal_improvement_m': goal_improvement_m,
            'cmd_fraction': cmd_fraction,
        }

    def _update_goal_region_state(self, stamp: float, goal_dist: float) -> None:
        if not (math.isfinite(stamp) and math.isfinite(goal_dist)):
            return
        if goal_dist <= self.goal_success_radius and not self._goal_region_entered:
            self._goal_region_entered = True
            self._goal_region_first_stamp = float(stamp)

    def _goal_region_reached(self) -> bool:
        return bool(self._goal_region_entered)

    def _maybe_finish_for_goal(self, stamp: float, goal_dist: float, cmd_v: float, cmd_w: float) -> bool:
        if not (self.auto_stop_on_goal and self.goal_msg and math.isfinite(goal_dist)):
            self._goal_in_radius_since = None
            self._goal_stable_since = None
            return False

        if goal_dist <= self.goal_success_radius:
            if self._goal_in_radius_since is None:
                self._goal_in_radius_since = stamp
            held_s = float(stamp - self._goal_in_radius_since)
            if held_s >= self.goal_success_hold_s:
                self.get_logger().info(
                    f"Goal reached (dist={goal_dist:.3f} m <= {self.goal_success_radius:.3f} m) "
                    f"and held for {held_s:.2f} s."
                )
                self._finish_run("goal_reached", stamp)
                return True
        else:
            self._goal_in_radius_since = None

        if goal_dist > self.goal_stable_radius:
            self._goal_stable_since = None
            return False

        stats = self._motion_window_stats(stamp, self.goal_stable_hold_s)
        stable_at_goal = (
            stats is not None
            and stats['displacement_m'] <= self.goal_stable_max_displacement_m
        )
        idle_at_goal = not self._command_active(cmd_v, cmd_w)
        if stable_at_goal or idle_at_goal:
            if self._goal_stable_since is None:
                self._goal_stable_since = stamp
            stable_held_s = float(stamp - self._goal_stable_since)
            if stable_held_s >= self.goal_stable_hold_s:
                mode = 'stable' if stable_at_goal else 'idle'
                self.get_logger().info(
                    f"Goal reached and {mode} "
                    f"(dist={goal_dist:.3f} m <= {self.goal_stable_radius:.3f} m) "
                    f"for {stable_held_s:.2f} s."
                )
                self._finish_run("goal_reached_stable", stamp)
                return True
        else:
            self._goal_stable_since = None
        return False

    def _maybe_finish_for_stuck(self, stamp: float, goal_dist: float) -> bool:
        if self._first_cmd_stamp is None or self.stuck_window_s <= 0.0:
            return False
        elapsed_after_first_cmd = float(stamp - self._first_cmd_stamp)
        if elapsed_after_first_cmd < self.stuck_window_s:
            return False
        if math.isfinite(goal_dist) and goal_dist <= self.goal_stable_radius:
            return False
        stats = self._motion_window_stats(stamp, self.stuck_window_s)
        if stats is None:
            return False
        no_motion = (
            stats['displacement_m'] <= self.stuck_max_displacement_m
            and stats['goal_improvement_m'] <= self.stuck_max_goal_improvement_m
        )
        active_stuck = stats['cmd_fraction'] >= self.stuck_cmd_fraction_min
        idle_stuck = stats['cmd_fraction'] <= self.stuck_idle_cmd_fraction_max
        stuck = no_motion and (active_stuck or idle_stuck)
        if not stuck:
            return False
        mode = 'active' if active_stuck else 'idle'
        self.get_logger().info(
            f"Stuck termination ({mode}): "
            f"displacement={stats['displacement_m']:.3f} m <= {self.stuck_max_displacement_m:.3f} m, "
            f"goal_improvement={stats['goal_improvement_m']:.3f} m <= "
            f"{self.stuck_max_goal_improvement_m:.3f} m, "
            f"cmd_fraction={stats['cmd_fraction']:.2f}, "
            f"active_threshold={self.stuck_cmd_fraction_min:.2f}, "
            f"idle_threshold={self.stuck_idle_cmd_fraction_max:.2f}."
        )
        self._finish_run("stuck", stamp)
        return True

    def _odom_cb(self, msg: Odometry):
        self.odom_msg = msg
        ok, st, x, y, yaw = self._latest_truth_pose()
        if ok and math.isfinite(st):
            self._truth_buf.append((float(st), float(x), float(y), float(yaw)))

    def _truth_at(self, stamp):
        """Interpolate buffered map_bev-frame truth to `stamp`. Returns (ok,x,y,yaw)."""
        buf = self._truth_buf
        if not buf or not math.isfinite(stamp):
            return False, math.nan, math.nan, math.nan
        if stamp <= buf[0][0]:
            return (True,) + buf[0][1:]
        if stamp >= buf[-1][0]:
            return (True,) + buf[-1][1:]
        prev = buf[0]
        for cur in buf:
            if cur[0] >= stamp:
                s0, x0, y0, yaw0 = prev
                s1, x1, y1, yaw1 = cur
                a = (stamp - s0) / max(s1 - s0, 1e-9)
                return (
                    True,
                    x0 + a * (x1 - x0),
                    y0 + a * (y1 - y0),
                    self._wrap_angle(yaw0 + a * self._wrap_angle(yaw1 - yaw0)),
                )
            prev = cur
        return (True,) + buf[-1][1:]

    def _odom_noisy_cb(self, msg: Odometry):
        self.odom_noisy_msg = msg

    def _ground_truth_cb(self, msg: TFMessage):
        # /world/<name>/dynamic_pose/info publishes every moving entity's world
        # pose. Keep the robot's. Holds last value while stationary (true pose
        # is constant then anyway).
        for tr in msg.transforms:
            if tr.child_frame_id == 'turtlebot3':
                self._gt_xy = (
                    float(tr.transform.translation.x),
                    float(tr.transform.translation.y),
                )
                self._gt_yaw = self._yaw_from_quaternion(tr.transform.rotation)

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        self.state_msg = msg

    def _planner_belief_cb(self, msg: PoseWithCovarianceStamped):
        self.planner_belief_msg = msg

    def _obs_cb(self, msg: PoseStamped):
        self.obs_msg = msg

    def _cmd_cb(self, msg: Twist):
        self.cmd_msg = msg
        self.cmd_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9

    def _cmd_raw_cb(self, msg: Twist):
        self.cmd_raw_msg = msg
        self.cmd_raw_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9

    def _cmd_noise_diag_cb(self, msg: Float64MultiArray):
        self.cmd_noise_diag = msg
        self.cmd_noise_diag_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9

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
            if len(msg.data) >= 5:
                self._efe_control_sum += float(msg.data[3])
                self._efe_obstacle_sum += float(msg.data[4])
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

    def _active_execution_diag_cb(self, msg: Float64MultiArray):
        self.active_execution_diag = msg

    def _diag_cb(self, msg: Float64MultiArray):
        self.perception_diag = diagnostics_from_message(msg)
        self._log_perception_sample(self.perception_diag)

    def _heading_diag_cb(self, msg: Float64MultiArray):
        self.heading_diag = msg

    def _pixel_correction_diag_cb(self, msg: Float64MultiArray):
        self.pixel_correction_diag = msg

    @staticmethod
    def extract_planar_covariances(cov):
        if cov is None or len(cov) < 36:
            return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
        cov_xx = float(cov[0])
        cov_xy = float(cov[1])
        cov_yy = float(cov[7])
        cov_x_theta = float(cov[30])
        cov_y_theta = float(cov[31])
        cov_theta_theta = float(cov[35])
        return cov_xx, cov_xy, cov_yy, cov_x_theta, cov_y_theta, cov_theta_theta

    @staticmethod
    def _heading_source_name(code: float) -> str:
        try:
            value = int(round(float(code)))
        except (TypeError, ValueError):
            value = 0
        return {
            1: 'pixel_heading',
            2: 'odom_heading_fallback',
            3: 'motion_heading_fallback',
            4: 'held_previous_heading',
        }.get(value, 'unknown')

    @staticmethod
    def _pixel_correction_reject_reason_name(code: float) -> str:
        try:
            value = int(round(float(code)))
        except (TypeError, ValueError):
            value = 99
        return {
            0: 'accepted',
            1: 'stale_age',
            2: 'dt_implausible',
            3: 'missing_snapshot',
            4: 'update_failed',
            5: 'jump_too_large',
            6: 'nis_too_large',
        }.get(value, 'unknown')

    @staticmethod
    def _pixel_correction_motion_replay_source_name(code: float) -> str:
        try:
            value = int(round(float(code)))
        except (TypeError, ValueError):
            value = -1
        return {
            0: 'none',
            1: 'odom_noisy',
            2: 'command_log',
            3: 'single_fallback',
        }.get(value, 'unknown')

    def _record_invalid(self, reason: str) -> None:
        reason = str(reason or '').strip()
        if not reason:
            return
        if self._valid_run:
            self._valid_run = False
            self._invalid_reason = reason
            return
        if reason not in self._invalid_reason.split('|'):
            self._invalid_reason = f'{self._invalid_reason}|{reason}' if self._invalid_reason else reason

    def _record_collision_event(self, *, stamp: float, reason: str, contact: bool, geom: bool) -> None:
        if contact:
            self._contact_collision_seen = True
        if geom:
            self._geom_collision_seen = True
        if math.isfinite(float(stamp)) and not math.isfinite(self._first_crash_stamp):
            self._first_crash_stamp = float(stamp)
        if str(reason or '').strip():
            self._collision_reason = str(reason).strip()
            self._record_invalid(self._collision_reason)

    def _contacts_cb(self, msg: Contacts):
        try:
            stamp = self._stamp_to_float(msg.header.stamp)
        except AttributeError:
            stamp = float(self.get_clock().now().nanoseconds) * 1e-9
        for contact in list(msg.contacts or []):
            name_1 = str(getattr(getattr(contact, 'collision1', None), 'name', '') or '')
            name_2 = str(getattr(getattr(contact, 'collision2', None), 'name', '') or '')
            pair = (name_1, name_2)
            if not any('turtlebot3' in name for name in pair):
                continue
            if all('turtlebot3' in name for name in pair):
                continue
            other = name_2 if 'turtlebot3' in name_1 else name_1
            if 'ground_plane' in other:
                continue
            self._record_collision_event(
                stamp=stamp,
                reason=f'contact:{other or "unknown"}',
                contact=True,
                geom=False,
            )
            self._finish_run("collision", stamp)
            break

    def _signed_distance_from_prisms(self, prisms, x: float, y: float) -> float:
        if not prisms:
            return float('inf')
        return float(signed_distance_to_union_xy(prisms, np.array([float(x), float(y)], dtype=float))[0])

    def _geometry_safety_at_truth(self, odom_map_x: float, odom_map_y: float):
        wall_signed = self._signed_distance_from_prisms(self._wall_prisms, odom_map_x, odom_map_y)
        obstacle_signed = self._signed_distance_from_prisms(self._obstacle_prisms, odom_map_x, odom_map_y)
        wall_clearance = wall_signed - self.robot_collision_radius_m if math.isfinite(wall_signed) else math.inf
        obstacle_clearance = obstacle_signed - self.robot_collision_radius_m if math.isfinite(obstacle_signed) else math.inf
        wall_penetration = max(-wall_clearance, 0.0) if math.isfinite(wall_clearance) else 0.0
        obstacle_penetration = max(-obstacle_clearance, 0.0) if math.isfinite(obstacle_clearance) else 0.0

        bounds = dict(self.world_bounds or {})
        off_map = False
        if all(math.isfinite(float(bounds.get(key, math.nan))) for key in ('xmin', 'xmax', 'ymin', 'ymax')):
            off_map = bool(
                float(odom_map_x) < float(bounds['xmin'])
                or float(odom_map_x) > float(bounds['xmax'])
                or float(odom_map_y) < float(bounds['ymin'])
                or float(odom_map_y) > float(bounds['ymax'])
            )
        inside_no_go = bool(obstacle_penetration > 0.0)
        return {
            'min_wall_distance_m': float(wall_clearance),
            'min_obstacle_distance_m': float(obstacle_clearance),
            'wall_penetration_m': float(wall_penetration),
            'obstacle_penetration_m': float(obstacle_penetration),
            'off_map': off_map,
            'inside_no_go': inside_no_go,
        }

    def _camera_relative_bearing_deg(self, odom_map_x: float, odom_map_y: float, odom_map_yaw: float) -> float:
        vec = np.asarray(self.camera_pos_xy, dtype=float) - np.array([float(odom_map_x), float(odom_map_y)], dtype=float)
        if np.linalg.norm(vec) <= 1e-9:
            return math.nan
        bearing_world = math.atan2(float(vec[1]), float(vec[0]))
        rel = self._wrap_angle(bearing_world - float(odom_map_yaw))
        return float(abs(math.degrees(rel)))

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
                now_wall = time.monotonic()
                if now_wall - self._last_tf_warn_wall > 2.0:
                    self.get_logger().warn(
                        f"Truth pose unavailable until TF {source_frame}->{self.frame_id} exists: {exc}"
                    )
                    self._last_tf_warn_wall = now_wall
                return False, stamp, math.nan, math.nan, math.nan

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

    def _odom_record(self, msg: Odometry | None):
        if msg is None:
            return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
        pose = msg.pose.pose
        twist = msg.twist.twist
        return (
            True,
            self._stamp_to_float(msg.header.stamp),
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
            float(twist.linear.x),
            float(twist.angular.z),
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
        true_ok, odom_map_stamp, odom_map_x, odom_map_y, odom_map_yaw = self._latest_truth_pose()
        if not (raw_ok and true_ok):
            return

        start_x, start_y, start_yaw = self.task_start_pose
        truth_start_error = float(math.hypot(odom_map_x - start_x, odom_map_y - start_y))
        raw_start_error = float(math.hypot(raw_x - start_x, raw_y - start_y))
        truth_start_yaw_error = abs(self._wrap_angle(odom_map_yaw - start_yaw))
        raw_start_yaw_error = abs(self._wrap_angle(raw_yaw - start_yaw))
        ok = bool(
            truth_start_error <= self.frame_sanity_start_tolerance_m
            and truth_start_yaw_error <= self.frame_sanity_start_tolerance_yaw_rad
        )

        self._frame_sanity_logged = True
        self._frame_sanity.update({
            'recorded': True,
            'ok': ok,
            'reason': 'ok' if ok else 'odom_map_start_mismatch',
            'source_frame': source_frame,
            'odom_map_stamp': odom_map_stamp,
            'raw_odom_x': raw_x,
            'raw_odom_y': raw_y,
            'raw_odom_yaw': raw_yaw,
            'odom_map_x': odom_map_x,
            'odom_map_y': odom_map_y,
            'odom_map_yaw': odom_map_yaw,
            'task_start_x': start_x,
            'task_start_y': start_y,
            'task_start_yaw': start_yaw,
            'odom_map_start_error_m': truth_start_error,
            'raw_start_error_m': raw_start_error,
            'odom_map_start_yaw_error_rad': truth_start_yaw_error,
            'raw_start_yaw_error_rad': raw_start_yaw_error,
            'tolerance_m': self.frame_sanity_start_tolerance_m,
            'tolerance_yaw_rad': self.frame_sanity_start_tolerance_yaw_rad,
            'recorded_at_log_stamp': now_stamp,
        })
        self._rewrite_manifest()

        message = (
            'Frame sanity check '
            f'({source_frame} -> {self.frame_id}): raw odom=({raw_x:.3f}, {raw_y:.3f}), '
            f'transformed truth=({odom_map_x:.3f}, {odom_map_y:.3f}), '
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
        state_ok, state_stamp, state_x, state_y, state_yaw = self._latest_state_pose()
        obs_ok, pixel_pose_stamp, pixel_pose_u, pixel_pose_v, pixel_pose_yaw = self._latest_pixel_pose()

        state_pos_error = math.nan
        state_yaw_error_deg = math.nan
        if true_ok and state_ok:
            state_pos_error = math.hypot(state_x - true_x, state_y - true_y)
            state_yaw_error_deg = math.degrees(self._wrap_angle(state_yaw - true_yaw))

        obs_yaw_error_deg = math.nan
        if true_ok and diag['detected'] and math.isfinite(diag['yaw_est']):
            obs_yaw_error_deg = math.degrees(self._wrap_angle(diag['yaw_est'] - true_yaw))
        camera_relative_bearing_deg = math.nan
        if true_ok:
            camera_relative_bearing_deg = self._camera_relative_bearing_deg(true_x, true_y, true_yaw)

        log_stamp = float(self.get_clock().now().nanoseconds) * 1e-9
        state_age_s = math.nan
        if state_ok and math.isfinite(state_stamp):
            state_age_s = max(log_stamp - state_stamp, 0.0)
        state_fresh = bool(
            state_ok
            and math.isfinite(state_age_s)
            and state_age_s <= max(float(self.pixel_timeout_s), 0.0)
        )
        pixel_pose_age_s = math.nan
        if obs_ok and math.isfinite(pixel_pose_stamp):
            pixel_pose_age_s = max(log_stamp - pixel_pose_stamp, 0.0)
        pixel_pose_fresh = bool(
            obs_ok
            and math.isfinite(pixel_pose_age_s)
            and pixel_pose_age_s <= max(float(self.pixel_timeout_s), 0.0)
        )

        # Compute predicted world position from image coordinates using homography
        pred_world_x = math.nan
        pred_world_y = math.nan
        localization_error_m = math.nan
        pred_world_x_calibrated = math.nan
        pred_world_y_calibrated = math.nan
        localization_error_calibrated_m = math.nan
        if obs_ok and math.isfinite(pixel_pose_u) and math.isfinite(pixel_pose_v):
            if self.bbox_contact_z_m > 0.0:
                world = self.camera_model.pixel_to_world_at_z(
                    float(pixel_pose_u),
                    float(pixel_pose_v),
                    self.bbox_contact_z_m,
                )
            else:
                world = self.camera_model.pixel_to_world(float(pixel_pose_u), float(pixel_pose_v))
            if world is not None:
                pred_world_x = float(world[0])
                pred_world_y = float(world[1])
                pred_world_x_calibrated, pred_world_y_calibrated = self._apply_bev_calibration(
                    pred_world_x,
                    pred_world_y,
                )
                if true_ok:
                    localization_error_m = math.hypot(pred_world_x - true_x, pred_world_y - true_y)
                    localization_error_calibrated_m = math.hypot(
                        pred_world_x_calibrated - true_x,
                        pred_world_y_calibrated - true_y,
                    )

        # Capture-time-truth errors: compare the measurement / state to truth AT THEIR
        # OWN timestamp instead of the current log time. This removes the ~0.5 s
        # latency inflation, so it reflects the TRUE detector / projection quality
        # (e.g. ~0.04 m even in turns). The *_calibrated_m and state_pos_error columns
        # above compare to log-time truth and are therefore LATENCY-INFLATED in turns.
        localization_error_captime_m = math.nan
        if (obs_ok and math.isfinite(pred_world_x_calibrated)
                and math.isfinite(pred_world_y_calibrated) and math.isfinite(pixel_pose_stamp)):
            cok, ctx, cty, _cyaw = self._truth_at(pixel_pose_stamp)
            if cok:
                localization_error_captime_m = math.hypot(
                    pred_world_x_calibrated - ctx, pred_world_y_calibrated - cty)
        state_error_captime_m = math.nan
        if state_ok and math.isfinite(state_x) and math.isfinite(state_stamp):
            sok, stx, sty, _syaw = self._truth_at(state_stamp)
            if sok:
                state_error_captime_m = math.hypot(state_x - stx, state_y - sty)

        selected_pixel_source_code = float(diag.get('selected_pixel_source_code', math.nan))
        if selected_pixel_source_code >= 1.5:
            selected_pixel_source = 'mask_bottom'
        elif selected_pixel_source_code >= 0.5:
            selected_pixel_source = 'bbox_bottom'
        else:
            selected_pixel_source = 'none'
        yolo_raw_best_score = diag.get('yolo_score_raw', math.nan)
        yolo_selected_score = diag.get('yolo_score_selected', math.nan)
        yolo_num_target_candidates = diag.get('yolo_target_candidate_count', math.nan)
        yolo_selected_class_id = diag.get('yolo_best_class_id', math.nan)
        yolo_bbox_area = diag.get('bbox_area_px', math.nan)
        yolo_mask_area = diag.get('mask_area_px', math.nan)
        yolo_inference_ms = diag.get('yolo_inference_ms', math.nan)
        detector_callback_ms = diag.get('detector_callback_ms', math.nan)
        yolo_receive_stamp = diag.get('yolo_receive_stamp', math.nan)
        yolo_start_stamp = diag.get('yolo_start_stamp', math.nan)
        yolo_finish_stamp = diag.get('yolo_finish_stamp', math.nan)
        yolo_publish_stamp = diag.get('yolo_publish_stamp', math.nan)
        yolo_latency_s = diag.get('yolo_latency_s', math.nan)
        frame_age_at_publish_s = diag.get('frame_age_at_publish_s', math.nan)
        if math.isfinite(float(yolo_publish_stamp)):
            detector_total_latency_s = max(float(yolo_publish_stamp) - float(diag['stamp']), 0.0)
        else:
            detector_total_latency_s = (
                max(log_stamp - float(diag['stamp']), 0.0)
                if math.isfinite(float(diag.get('stamp', math.nan))) else math.nan
            )

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
            state_age_s,
            1.0 if state_fresh else 0.0,
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
            1.0 if pixel_pose_fresh else 0.0,
            pred_world_x,
            pred_world_y,
            localization_error_m,
            pred_world_x_calibrated,
            pred_world_y_calibrated,
            localization_error_calibrated_m,
            localization_error_captime_m,
            state_error_captime_m,
            self.bev_y_calibration_offset_m,
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
            selected_pixel_source_code,
            yolo_raw_best_score,
            yolo_selected_score,
            yolo_num_target_candidates,
            yolo_selected_class_id,
            selected_pixel_source,
            yolo_bbox_area,
            yolo_mask_area,
            yolo_inference_ms,
            detector_callback_ms,
            yolo_receive_stamp,
            yolo_start_stamp,
            yolo_finish_stamp,
            yolo_publish_stamp,
            yolo_latency_s,
            frame_age_at_publish_s,
            detector_total_latency_s,
            camera_relative_bearing_deg,
            self.seed,
        ])
        self.perception_file.flush()

    def _log_once(self):
        now_stamp = float(self.get_clock().now().nanoseconds) * 1e-9

        state_ok, state_stamp, state_x, state_y, state_yaw = self._latest_state_pose()
        state_age_s = math.nan
        if state_ok and math.isfinite(state_stamp):
            state_age_s = max(now_stamp - state_stamp, 0.0)
        state_fresh = bool(
            state_ok
            and math.isfinite(state_age_s)
            and state_age_s <= max(float(self.pixel_timeout_s), 0.0)
        )
        if self.state_msg is not None:
            cov = self.state_msg.pose.covariance
            cov_x = float(cov[0]) if len(cov) > 0 else math.nan
            cov_xy = float(cov[1]) if len(cov) > 1 else math.nan
            cov_y = float(cov[7]) if len(cov) > 7 else math.nan
            cov_yaw = float(cov[35]) if len(cov) > 35 else math.nan
        else:
            cov_x = cov_xy = cov_y = cov_yaw = math.nan

        true_ok, odom_map_stamp, true_x, true_y, true_yaw = self._latest_truth_pose()
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
        if planner_belief_ok and math.isfinite(planner_belief_stamp):
            planner_belief_age_s = max(0.0, now_stamp - float(planner_belief_stamp))
        else:
            planner_belief_age_s = math.nan

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

        # ODOM-as-reference error signals (NOT ground truth — true_x/y = /odom,
        # which drifts from the real pose). Kept only for odom-drift diagnostics;
        # the honest errors are belief_error_gt_m / state_error_gt_m (vs gt_x/gt_y).
        # state_error_odom_m:  /odom vs /state/bev (perception output)
        # belief_error_odom_m: /odom vs /planner_belief (planner's internal belief)
        state_error_odom_m = math.nan
        after_first_cmd = bool(self._first_cmd_stamp is not None and now_stamp >= self._first_cmd_stamp)
        if true_ok and state_ok and math.isfinite(state_x) and math.isfinite(state_y):
            state_error_odom_m = float(math.hypot(true_x - state_x, true_y - state_y))
            self._state_error_odom_sum += state_error_odom_m
            if math.isfinite(state_error_odom_m):
                self._state_error_odom_count += 1
                if after_first_cmd:
                    self._state_error_odom_after_first_cmd_sum += state_error_odom_m
                    self._state_error_odom_after_first_cmd_count += 1
        belief_error_odom_m = math.nan
        if true_ok and planner_belief_ok and math.isfinite(planner_belief_x) and math.isfinite(planner_belief_y):
            belief_error_odom_m = float(math.hypot(true_x - planner_belief_x, true_y - planner_belief_y))
            self._belief_error_odom_sum += belief_error_odom_m
            if math.isfinite(belief_error_odom_m):
                self._belief_error_odom_count += 1
                if after_first_cmd:
                    self._belief_error_odom_after_first_cmd_sum += belief_error_odom_m
                    self._belief_error_odom_after_first_cmd_count += 1

        odom_ok, odom_stamp, odom_x, odom_y, odom_yaw, odom_v, odom_w = self._odom_record(self.odom_msg)
        (
            odom_noisy_ok,
            odom_noisy_stamp,
            odom_noisy_x,
            odom_noisy_y,
            odom_noisy_yaw,
            odom_noisy_v,
            odom_noisy_w,
        ) = self._odom_record(self.odom_noisy_msg)
        yaw_error_odom_noisy_vs_odom_map_rad = math.nan
        if true_ok and odom_noisy_ok and math.isfinite(odom_noisy_yaw):
            yaw_error_odom_noisy_vs_odom_map_rad = float(self._wrap_angle(odom_noisy_yaw - true_yaw))

        state_bev_yaw_latest = math.nan
        state_bev_cov_theta_theta = math.nan
        state_bev_cov_x_theta = math.nan
        state_bev_cov_y_theta = math.nan
        if self.state_msg is not None:
            state_bev_yaw_latest = self._yaw_from_quaternion(self.state_msg.pose.pose.orientation)
            _, _, _, state_bev_cov_x_theta, state_bev_cov_y_theta, state_bev_cov_theta_theta = (
                self.extract_planar_covariances(self.state_msg.pose.covariance)
            )

        planner_belief_cov_theta_theta = math.nan
        planner_belief_cov_x_theta = math.nan
        planner_belief_cov_y_theta = math.nan
        if self.planner_belief_msg is not None:
            _, _, _, planner_belief_cov_x_theta, planner_belief_cov_y_theta, planner_belief_cov_theta_theta = (
                self.extract_planar_covariances(self.planner_belief_msg.pose.covariance)
            )

        yaw_error_odom_map_vs_odom_rad = math.nan
        yaw_error_odom_map_vs_state_rad = math.nan
        yaw_error_odom_map_vs_belief_rad = math.nan
        if true_ok and odom_ok and math.isfinite(odom_yaw):
            yaw_error_odom_map_vs_odom_rad = float(self._wrap_angle(odom_yaw - true_yaw))
            self._odom_map_vs_odom_yaw_error_sum += abs(yaw_error_odom_map_vs_odom_rad)
            self._truth_odom_yaw_error_count += 1
            if after_first_cmd:
                self._odom_map_vs_odom_yaw_error_after_first_cmd_sum += abs(yaw_error_odom_map_vs_odom_rad)
                self._odom_map_vs_odom_yaw_error_after_first_cmd_count += 1
        if true_ok and state_ok and math.isfinite(state_yaw):
            yaw_error_odom_map_vs_state_rad = float(self._wrap_angle(state_yaw - true_yaw))
            self._odom_map_vs_state_yaw_error_sum += abs(yaw_error_odom_map_vs_state_rad)
            self._truth_state_yaw_error_count += 1
            if after_first_cmd:
                self._odom_map_vs_state_yaw_error_after_first_cmd_sum += abs(yaw_error_odom_map_vs_state_rad)
                self._odom_map_vs_state_yaw_error_after_first_cmd_count += 1
        if true_ok and planner_belief_ok and math.isfinite(planner_belief_yaw):
            yaw_error_odom_map_vs_belief_rad = float(self._wrap_angle(planner_belief_yaw - true_yaw))
            self._odom_map_vs_belief_yaw_error_sum += abs(yaw_error_odom_map_vs_belief_rad)
            self._truth_belief_yaw_error_count += 1
            if after_first_cmd:
                self._odom_map_vs_belief_yaw_error_after_first_cmd_sum += abs(yaw_error_odom_map_vs_belief_rad)
                self._odom_map_vs_belief_yaw_error_after_first_cmd_count += 1

        heading_diag_stamp = math.nan
        heading_diag_age_s = math.nan
        pixel_yaw_meas = math.nan
        heading_source_code = math.nan
        heading_source = 'unknown'
        state_heading_yaw_sigma = math.nan
        state_heading_odom_age_s = math.nan
        if self.heading_diag is not None and self.heading_diag.data and len(self.heading_diag.data) >= 10:
            hdata = list(self.heading_diag.data)
            heading_diag_stamp = float(hdata[0])
            heading_source_code = float(hdata[1])
            pixel_yaw_meas = float(hdata[2])
            state_heading_yaw_sigma = float(hdata[6])
            state_heading_odom_age_s = float(hdata[9])
            heading_source = self._heading_source_name(heading_source_code)
            if math.isfinite(heading_diag_stamp):
                heading_diag_age_s = max(now_stamp - heading_diag_stamp, 0.0)

        planner_pixel_correction_available = 0.0
        planner_pixel_correction_stamp = math.nan
        planner_pixel_correction_age_s = math.nan
        pixel_corr_innov_u = math.nan
        pixel_corr_innov_v = math.nan
        pixel_corr_xy_update_norm_m = math.nan
        pixel_corr_theta_update_from_uv_rad = math.nan
        pixel_corr_nis = math.nan
        pixel_corr_accepted = math.nan
        pixel_corr_reject_reason_code = math.nan
        pixel_corr_reject_reason = 'unknown'
        pixel_corr_apply_stamp = math.nan
        pixel_corr_belief_input_stamp = math.nan
        pixel_corr_cmd_replay_count = math.nan
        pixel_corr_cmd_replay_duration_s = math.nan
        pixel_corr_cmd_replay_used_fallback = math.nan
        pixel_corr_motion_replay_source_code = math.nan
        pixel_corr_motion_replay_source = 'unknown'
        pixel_corr_nis_threshold = math.nan
        pixel_heading_correction_applied = math.nan
        pixel_heading_meas_source = math.nan
        pixel_heading_innov_rad = math.nan
        pixel_heading_gain_theta = math.nan
        pixel_corr_theta_update_total_rad = math.nan
        pixel_corr_pred_x = math.nan
        pixel_corr_pred_y = math.nan
        pixel_corr_pred_yaw = math.nan
        pixel_corr_next_x = math.nan
        pixel_corr_next_y = math.nan
        pixel_corr_next_yaw = math.nan
        pixel_corr_expected_after_u = math.nan
        pixel_corr_expected_after_v = math.nan
        pixel_corr_expected_after_visible = math.nan
        pixel_corr_K_theta_u = math.nan
        pixel_corr_K_theta_v = math.nan
        if (
            self.pixel_correction_diag is not None
            and self.pixel_correction_diag.data
            and len(self.pixel_correction_diag.data) >= 20
        ):
            cdata = list(self.pixel_correction_diag.data)
            planner_pixel_correction_available = float(cdata[1])
            planner_pixel_correction_stamp = float(cdata[0])
            if math.isfinite(planner_pixel_correction_stamp):
                planner_pixel_correction_age_s = max(now_stamp - planner_pixel_correction_stamp, 0.0)
            pixel_corr_innov_u = float(cdata[6])
            pixel_corr_innov_v = float(cdata[7])
            pixel_corr_xy_update_norm_m = float(cdata[8])
            pixel_corr_theta_update_from_uv_rad = float(cdata[9])
            pixel_heading_correction_applied = float(cdata[10])
            pixel_heading_innov_rad = float(cdata[11])
            pixel_heading_gain_theta = float(cdata[12])
            pixel_corr_theta_update_total_rad = float(cdata[13])
            pixel_corr_pred_x = float(cdata[14])
            pixel_corr_pred_y = float(cdata[15])
            pixel_corr_pred_yaw = float(cdata[16])
            pixel_corr_next_x = float(cdata[17])
            pixel_corr_next_y = float(cdata[18])
            pixel_corr_next_yaw = float(cdata[19])
            if len(cdata) >= 29:
                pixel_heading_meas_source = float(cdata[28])
            if len(cdata) >= 30:
                pixel_corr_nis = float(cdata[29])
            if len(cdata) >= 38:
                pixel_corr_accepted = float(cdata[30])
                pixel_corr_reject_reason_code = float(cdata[31])
                pixel_corr_reject_reason = self._pixel_correction_reject_reason_name(
                    pixel_corr_reject_reason_code
                )
                pixel_corr_apply_stamp = float(cdata[32])
                pixel_corr_belief_input_stamp = float(cdata[33])
                pixel_corr_cmd_replay_count = float(cdata[34])
                pixel_corr_cmd_replay_duration_s = float(cdata[35])
                pixel_corr_cmd_replay_used_fallback = float(cdata[36])
                pixel_corr_nis_threshold = float(cdata[37])
            if len(cdata) >= 41:
                pixel_corr_expected_after_u = float(cdata[38])
                pixel_corr_expected_after_v = float(cdata[39])
                pixel_corr_expected_after_visible = float(cdata[40])
            if len(cdata) >= 42:
                pixel_corr_motion_replay_source_code = float(cdata[41])
                pixel_corr_motion_replay_source = self._pixel_correction_motion_replay_source_name(
                    pixel_corr_motion_replay_source_code
                )
            if len(cdata) >= 44:
                pixel_corr_K_theta_u = float(cdata[42])
                pixel_corr_K_theta_v = float(cdata[43])

        cmd_v = self.cmd_msg.linear.x if self.cmd_msg else 0.0
        cmd_w = self.cmd_msg.angular.z if self.cmd_msg else 0.0
        cmd_raw_v = self.cmd_raw_msg.linear.x if self.cmd_raw_msg else cmd_v
        cmd_raw_w = self.cmd_raw_msg.angular.z if self.cmd_raw_msg else cmd_w
        cmd_stamp = self.cmd_stamp_s
        cmd_raw_stamp = self.cmd_raw_stamp_s
        cmd_age_s = max(now_stamp - cmd_stamp, 0.0) if math.isfinite(cmd_stamp) else math.nan
        cmd_raw_age_s = max(now_stamp - cmd_raw_stamp, 0.0) if math.isfinite(cmd_raw_stamp) else math.nan
        cmd_noise_enabled = math.nan
        cmd_noise_linear_multiplier = math.nan
        cmd_noise_angular_multiplier = math.nan
        cmd_noise_linear_additive = math.nan
        cmd_noise_angular_additive = math.nan
        if self.cmd_noise_diag is not None and self.cmd_noise_diag.data and len(self.cmd_noise_diag.data) >= 10:
            ndata = list(self.cmd_noise_diag.data)
            cmd_noise_enabled = float(ndata[1])
            cmd_raw_v = float(ndata[2])
            cmd_raw_w = float(ndata[3])
            cmd_noise_linear_multiplier = float(ndata[6])
            cmd_noise_angular_multiplier = float(ndata[7])
            cmd_noise_linear_additive = float(ndata[8])
            cmd_noise_angular_additive = float(ndata[9])
        cmd_noise_v_error = float(cmd_v - cmd_raw_v)
        cmd_noise_w_error = float(cmd_w - cmd_raw_w)
        self._maybe_log_frame_sanity(now_stamp, cmd_v, cmd_w)

        goal_x = math.nan
        goal_y = math.nan
        # Goal distance & executed path from the TRUE Gazebo pose ONLY (no /odom
        # fallback). goal_reached / min_goal_distance are outcome metrics; scoring
        # them on drifting wheel-odom would (like the collision metric) misreport
        # whether the TRUE robot reached the goal. If gt is unavailable -> NaN.
        goal_dist = math.nan
        if self.goal_msg and self._gt_xy is not None:
            goal_x = float(self.goal_msg.pose.position.x)
            goal_y = float(self.goal_msg.pose.position.y)
            goal_dist = math.hypot(goal_x - self._gt_xy[0], goal_y - self._gt_xy[1])

        current_pose = None
        if self._gt_xy is not None:
            current_pose = (float(self._gt_xy[0]), float(self._gt_xy[1]))
            if self._last_path_pose is not None:
                self._cumulative_path_length += math.hypot(current_pose[0] - self._last_path_pose[0], current_pose[1] - self._last_path_pose[1])
            self._last_path_pose = current_pose

        if math.isfinite(goal_dist):
            self._min_goal_distance = min(self._min_goal_distance, goal_dist)
            self._update_goal_region_state(now_stamp, goal_dist)

        self._remember_motion_sample(now_stamp, true_ok, true_x, true_y, goal_dist, cmd_v, cmd_w)

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
        efe_obstacle = 0.0
        efe_risk_mean = math.nan
        efe_risk_cov_trace = math.nan
        efe_risk_cov_logdet = math.nan
        efe_delta_risk_visibility = math.nan
        efe_delta_ambiguity_visibility = math.nan
        active_plan_age_s = math.nan
        active_plan_remaining_s = math.nan
        active_control_index = math.nan
        active_controls_len = math.nan
        active_controls_original_len = math.nan
        latency_skip_steps = math.nan
        latency_skip_s = math.nan
        command_timer_period_s = math.nan
        planner_timer_period_s = math.nan
        pending_plan_started_active_remaining_s = math.nan
        exec_plan_age_s = math.nan
        exec_plan_remaining_s = math.nan
        exec_control_index = math.nan
        exec_controls_len = math.nan
        exec_controls_original_len = math.nan
        exec_cmd_v = math.nan
        exec_cmd_w = math.nan
        exec_latency_skip_steps = math.nan
        exec_latency_skip_s = math.nan
        exec_wp_idx = math.nan
        exec_wp_count = math.nan
        exec_wp_target_x = math.nan
        exec_wp_target_y = math.nan
        exec_wp_dist_m = math.nan
        exec_desired_yaw = math.nan
        exec_yaw_error = math.nan
        exec_tracking_yaw = math.nan
        exec_tracking_yaw_source = math.nan
        optimizer_success = 0.0
        optimizer_status = 0.0
        optimizer_nit = 0.0
        optimizer_nfev = 0.0
        optimizer_message = self.planner_diag_text
        planner_diag_prediction_source = math.nan
        planner_diag_prediction_dt = math.nan
        planner_diag_u_pred_v = math.nan
        planner_diag_u_pred_omega = math.nan
        planner_diag_Q_theta_theta = math.nan
        planner_diag_odom_delta_theta = math.nan
        planner_diag_cmd_delta_theta = math.nan
        planner_diag_heading_anchor_applied = math.nan
        planner_diag_state_bev_yaw_ignored = math.nan
        plan_time_ms = 0.0
        solve_time_ms = 0.0
        measurement_available = math.nan
        belief_age_s = math.nan
        p_vis_plan = math.nan
        p_vis_plan_eff = math.nan
        r_plan_u_std = math.nan
        r_plan_v_std = math.nan
        terminal_goal_distance_pred = math.nan
        terminal_goal_progress_m = math.nan
        fraction_horizon_low_pvis = math.nan
        fraction_horizon_high_ambiguity = math.nan
        min_predicted_obstacle_distance_m = math.nan
        rollout_valid = math.nan
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
            if len(self.planner_diag.data) >= 18:
                terminal_goal_distance_pred = float(self.planner_diag.data[12])
                terminal_goal_progress_m = float(self.planner_diag.data[13])
                fraction_horizon_low_pvis = float(self.planner_diag.data[14])
                fraction_horizon_high_ambiguity = float(self.planner_diag.data[15])
                min_predicted_obstacle_distance_m = float(self.planner_diag.data[16])
                rollout_valid = float(self.planner_diag.data[17])
            if len(self.planner_diag.data) >= 23:
                efe_risk_mean = float(self.planner_diag.data[18])
                efe_risk_cov_trace = float(self.planner_diag.data[19])
                efe_risk_cov_logdet = float(self.planner_diag.data[20])
                efe_delta_risk_visibility = float(self.planner_diag.data[21])
                efe_delta_ambiguity_visibility = float(self.planner_diag.data[22])
            if len(self.planner_diag.data) >= 33:
                active_plan_age_s = float(self.planner_diag.data[23])
                active_plan_remaining_s = float(self.planner_diag.data[24])
                active_control_index = float(self.planner_diag.data[25])
                active_controls_len = float(self.planner_diag.data[26])
                active_controls_original_len = float(self.planner_diag.data[27])
                latency_skip_steps = float(self.planner_diag.data[28])
                latency_skip_s = float(self.planner_diag.data[29])
                command_timer_period_s = float(self.planner_diag.data[30])
                planner_timer_period_s = float(self.planner_diag.data[31])
                pending_plan_started_active_remaining_s = float(self.planner_diag.data[32])
            if len(self.planner_diag.data) >= 42:
                planner_diag_prediction_source = float(self.planner_diag.data[33])
                planner_diag_prediction_dt = float(self.planner_diag.data[34])
                planner_diag_u_pred_v = float(self.planner_diag.data[35])
                planner_diag_u_pred_omega = float(self.planner_diag.data[36])
                planner_diag_Q_theta_theta = float(self.planner_diag.data[37])
                planner_diag_odom_delta_theta = float(self.planner_diag.data[38])
                planner_diag_cmd_delta_theta = float(self.planner_diag.data[39])
                planner_diag_heading_anchor_applied = float(self.planner_diag.data[40])
                planner_diag_state_bev_yaw_ignored = float(self.planner_diag.data[41])

        if (
            self.active_execution_diag
            and self.active_execution_diag.data
            and len(self.active_execution_diag.data) >= 9
        ):
            exec_plan_age_s = float(self.active_execution_diag.data[0])
            exec_plan_remaining_s = float(self.active_execution_diag.data[1])
            exec_control_index = float(self.active_execution_diag.data[2])
            exec_controls_len = float(self.active_execution_diag.data[3])
            exec_controls_original_len = float(self.active_execution_diag.data[4])
            exec_cmd_v = float(self.active_execution_diag.data[5])
            exec_cmd_w = float(self.active_execution_diag.data[6])
            exec_latency_skip_steps = float(self.active_execution_diag.data[7])
            exec_latency_skip_s = float(self.active_execution_diag.data[8])
            if len(self.active_execution_diag.data) >= 18:
                exec_wp_idx = float(self.active_execution_diag.data[9])
                exec_wp_count = float(self.active_execution_diag.data[10])
                exec_wp_target_x = float(self.active_execution_diag.data[11])
                exec_wp_target_y = float(self.active_execution_diag.data[12])
                exec_wp_dist_m = float(self.active_execution_diag.data[13])
                exec_desired_yaw = float(self.active_execution_diag.data[14])
                exec_yaw_error = float(self.active_execution_diag.data[15])
                exec_tracking_yaw = float(self.active_execution_diag.data[16])
                exec_tracking_yaw_source = float(self.active_execution_diag.data[17])

        if self.efe_metrics and self.efe_metrics.data and len(self.efe_metrics.data) >= 5:
            efe_total = float(self.efe_metrics.data[0])
            efe_risk = float(self.efe_metrics.data[1])
            efe_ambiguity = float(self.efe_metrics.data[2])
            efe_control = float(self.efe_metrics.data[3])
            efe_obstacle = float(self.efe_metrics.data[4])
            if len(self.efe_metrics.data) >= 23:
                efe_risk_mean = float(self.efe_metrics.data[18])
                efe_risk_cov_trace = float(self.efe_metrics.data[19])
                efe_risk_cov_logdet = float(self.efe_metrics.data[20])
                efe_delta_risk_visibility = float(self.efe_metrics.data[21])
                efe_delta_ambiguity_visibility = float(self.efe_metrics.data[22])

        min_wall_distance_m = self._min_wall_distance if math.isfinite(self._min_wall_distance) else math.inf
        min_obstacle_distance_m = self._min_obstacle_distance if math.isfinite(self._min_obstacle_distance) else math.inf
        wall_penetration_m = 0.0
        obstacle_penetration_m = 0.0
        off_map = 0.0
        inside_no_go = 0.0
        # Collision / clearance from the TRUE Gazebo pose ONLY. The /odom 'truth' is
        # wheel odometry and drifts (esp. in turns), which produced FALSE geometry-
        # collisions (odom penetrates a rack while the true robot is clear). NO odom
        # fallback: if ground truth is unavailable, these stay NaN (explicit), never
        # silently computed against odom. If gt is never available the run has no
        # geometry-collision metric at all -- that is obvious, not confusing.
        _gt = self._gt_xy
        if _gt is not None:
            safety = self._geometry_safety_at_truth(_gt[0], _gt[1])
            min_wall_distance_m = float(safety['min_wall_distance_m'])
            min_obstacle_distance_m = float(safety['min_obstacle_distance_m'])
            wall_penetration_m = float(safety['wall_penetration_m'])
            obstacle_penetration_m = float(safety['obstacle_penetration_m'])
            off_map = 1.0 if safety['off_map'] else 0.0
            inside_no_go = 1.0 if safety['inside_no_go'] else 0.0

            if math.isfinite(min_wall_distance_m):
                self._min_wall_distance = min(self._min_wall_distance, min_wall_distance_m)
            if math.isfinite(min_obstacle_distance_m):
                self._min_obstacle_distance = min(self._min_obstacle_distance, min_obstacle_distance_m)
            self._max_wall_penetration = max(self._max_wall_penetration, wall_penetration_m)
            self._max_obstacle_penetration = max(self._max_obstacle_penetration, obstacle_penetration_m)
            self._off_map_seen = self._off_map_seen or bool(off_map >= 0.5)
            self._inside_no_go_seen = self._inside_no_go_seen or bool(inside_no_go >= 0.5)

            geom_reason = []
            if wall_penetration_m > 0.0:
                geom_reason.append('geometry:wall_penetration')
            if obstacle_penetration_m > 0.0:
                geom_reason.append('geometry:obstacle_penetration')
            if off_map >= 0.5:
                self._record_invalid('off_map')
            if inside_no_go >= 0.5:
                self._record_invalid('inside_no_go')
            if geom_reason:
                self._record_collision_event(
                    stamp=odom_map_stamp if math.isfinite(odom_map_stamp) else now_stamp,
                    reason='|'.join(geom_reason),
                    contact=False,
                    geom=True,
                )

        collision_contact = 1.0 if self._contact_collision_seen else 0.0
        collision_geom = 1.0 if self._geom_collision_seen else 0.0
        collision_any = 1.0 if (collision_contact >= 0.5 or collision_geom >= 0.5) else 0.0
        collision_reason = self._collision_reason
        if collision_any >= 0.5:
            self._record_invalid(collision_reason or 'collision_any')
        valid_run = 1.0 if self._valid_run else 0.0
        invalid_reason = self._invalid_reason

        # --- GROUND-TRUTH errors (vs TRUE Gazebo pose, not /odom) ---
        # /odom ("truth_*") is DiffDrive wheel odometry and drifts in turns;
        # these *_gt columns are the honest errors against the real pose.
        gt_ok = self._gt_xy is not None
        gt_x = self._gt_xy[0] if gt_ok else math.nan
        gt_y = self._gt_xy[1] if gt_ok else math.nan
        gt_yaw = self._gt_yaw if (gt_ok and self._gt_yaw is not None) else math.nan
        belief_error_gt_m = math.nan
        state_error_gt_m = math.nan
        odom_map_gt_drift_m = math.nan
        belief_yaw_error_gt_rad = math.nan
        if gt_ok:
            if math.isfinite(gt_yaw) and planner_belief_ok and math.isfinite(planner_belief_yaw):
                belief_yaw_error_gt_rad = self._wrap_angle(planner_belief_yaw - gt_yaw)
            if planner_belief_ok and math.isfinite(planner_belief_x):
                belief_error_gt_m = math.hypot(planner_belief_x - gt_x, planner_belief_y - gt_y)
            if state_ok and math.isfinite(state_x):
                state_error_gt_m = math.hypot(state_x - gt_x, state_y - gt_y)
            if true_ok and math.isfinite(true_x):
                odom_map_gt_drift_m = math.hypot(true_x - gt_x, true_y - gt_y)
            if math.isfinite(belief_error_gt_m):
                self._belief_error_gt_sum += belief_error_gt_m
                self._belief_error_gt_count += 1
                if after_first_cmd:
                    self._belief_error_gt_after_first_cmd_sum += belief_error_gt_m
                    self._belief_error_gt_after_first_cmd_count += 1
            if math.isfinite(state_error_gt_m):
                self._state_error_gt_sum += state_error_gt_m
                self._state_error_gt_count += 1
                if after_first_cmd:
                    self._state_error_gt_after_first_cmd_sum += state_error_gt_m
                    self._state_error_gt_after_first_cmd_count += 1

        self.writer.writerow([
            stamp,
            1.0 if true_ok else 0.0, odom_map_stamp, true_x, true_y, true_yaw,
            1.0 if state_ok else 0.0, state_stamp, state_x, state_y, state_yaw,
            state_age_s, 1.0 if state_fresh else 0.0,
            cov_x, cov_xy, cov_y, cov_yaw,
            1.0 if planner_belief_ok else 0.0, planner_belief_stamp,
            planner_belief_age_s,
            planner_belief_x, planner_belief_y, planner_belief_yaw,
            planner_cov_x, planner_cov_xy, planner_cov_y, planner_cov_yaw,
            est_available, est_x, est_y, est_yaw,
            est_cov_xx, est_cov_xy, est_cov_yy,
            state_pos_error_m, state_cov_trace, state_cov_det,
            state_sigma_major_m, state_sigma_minor_m, state_entropy_xy,
            state_error_odom_m, belief_error_odom_m,
            1.0 if odom_ok else 0.0, odom_stamp, odom_x, odom_y, odom_yaw,
            odom_v, odom_w,
            1.0 if odom_noisy_ok else 0.0, odom_noisy_stamp, odom_noisy_x, odom_noisy_y,
            odom_noisy_yaw, odom_noisy_v, odom_noisy_w,
            yaw_error_odom_map_vs_odom_rad, yaw_error_odom_map_vs_state_rad, yaw_error_odom_map_vs_belief_rad,
            pixel_yaw_meas, heading_source_code, heading_source,
            heading_diag_stamp, heading_diag_age_s,
            state_heading_yaw_sigma, state_heading_odom_age_s,
            planner_pixel_correction_available, planner_pixel_correction_stamp,
            planner_pixel_correction_age_s,
            pixel_corr_innov_u, pixel_corr_innov_v,
            pixel_corr_xy_update_norm_m, pixel_corr_theta_update_from_uv_rad,
            pixel_corr_nis,
            pixel_corr_accepted, pixel_corr_reject_reason_code, pixel_corr_reject_reason,
            pixel_corr_apply_stamp, pixel_corr_belief_input_stamp,
            pixel_corr_cmd_replay_count, pixel_corr_cmd_replay_duration_s,
            pixel_corr_cmd_replay_used_fallback,
            pixel_corr_motion_replay_source_code, pixel_corr_motion_replay_source,
            pixel_corr_nis_threshold,
            pixel_heading_correction_applied, pixel_heading_meas_source,
            pixel_heading_innov_rad,
            pixel_heading_gain_theta, pixel_corr_theta_update_total_rad,
            pixel_corr_pred_x, pixel_corr_pred_y, pixel_corr_pred_yaw,
            pixel_corr_next_x, pixel_corr_next_y, pixel_corr_next_yaw,
            pixel_corr_expected_after_u, pixel_corr_expected_after_v,
            pixel_corr_expected_after_visible,
            cmd_v, cmd_w,
            cmd_raw_v, cmd_raw_w,
            cmd_stamp, cmd_age_s, cmd_raw_stamp, cmd_raw_age_s,
            cmd_noise_enabled,
            cmd_noise_linear_multiplier, cmd_noise_angular_multiplier,
            cmd_noise_linear_additive, cmd_noise_angular_additive,
            cmd_noise_v_error, cmd_noise_w_error,
            goal_x, goal_y, goal_dist,
            plan_points, plan_length,
            optimizer_success, optimizer_status, optimizer_nit, optimizer_nfev, optimizer_message,
            plan_time_ms, solve_time_ms,
            measurement_available, belief_age_s,
            p_vis_plan, p_vis_plan_eff,
            r_plan_u_std, r_plan_v_std,
            terminal_goal_distance_pred, terminal_goal_progress_m,
            fraction_horizon_low_pvis, fraction_horizon_high_ambiguity,
            min_predicted_obstacle_distance_m, rollout_valid,
            efe_total, efe_risk, efe_ambiguity, efe_control, efe_obstacle,
            efe_risk_mean, efe_risk_cov_trace, efe_risk_cov_logdet,
            efe_delta_risk_visibility, efe_delta_ambiguity_visibility,
            active_plan_age_s, active_plan_remaining_s, active_control_index,
            active_controls_len, active_controls_original_len,
            latency_skip_steps, latency_skip_s,
            command_timer_period_s, planner_timer_period_s,
            pending_plan_started_active_remaining_s,
            exec_plan_age_s, exec_plan_remaining_s, exec_control_index,
            exec_controls_len, exec_controls_original_len,
            exec_cmd_v, exec_cmd_w, exec_latency_skip_steps,
            exec_latency_skip_s,
            exec_wp_idx, exec_wp_count, exec_wp_target_x, exec_wp_target_y,
            exec_wp_dist_m, exec_desired_yaw, exec_yaw_error,
            exec_tracking_yaw, exec_tracking_yaw_source,
            collision_any, collision_contact, collision_geom, collision_reason, self._first_crash_stamp,
            min_wall_distance_m, min_obstacle_distance_m,
            wall_penetration_m, obstacle_penetration_m,
            off_map, inside_no_go, valid_run, invalid_reason,
            self.heading_update_mode,
            pixel_corr_K_theta_u, pixel_corr_K_theta_v,
            yaw_error_odom_noisy_vs_odom_map_rad,
            state_bev_yaw_latest,
            state_bev_cov_theta_theta, state_bev_cov_x_theta, state_bev_cov_y_theta,
            planner_belief_cov_theta_theta, planner_belief_cov_x_theta, planner_belief_cov_y_theta,
            planner_diag_prediction_source, planner_diag_prediction_dt,
            planner_diag_u_pred_v, planner_diag_u_pred_omega, planner_diag_Q_theta_theta,
            planner_diag_odom_delta_theta, planner_diag_cmd_delta_theta,
            planner_diag_heading_anchor_applied, planner_diag_state_bev_yaw_ignored,
            1.0 if gt_ok else 0.0, gt_x, gt_y, gt_yaw,
            belief_error_gt_m, state_error_gt_m, odom_map_gt_drift_m,
            belief_yaw_error_gt_rad,
            self.seed,
        ])
        self.file.flush()

        if not self._stop_requested and (
            self._contact_collision_seen
            or (self._geom_collision_seen and self.terminate_on_geom_collision)
        ):
            self._finish_run("collision", now_stamp)
            return

        if not self._stop_requested:
            if self._first_cmd_stamp is None:
                if self._command_active(cmd_v, cmd_w):
                    self._first_cmd_stamp = now_stamp
                    self.get_logger().info(f"First command detected. Starting {self.run_timeout_after_first_cmd_s:.1f}s timeout.")
            else:
                elapsed = now_stamp - self._first_cmd_stamp
                if elapsed >= self.run_timeout_after_first_cmd_s:
                    self._finish_run("timeout_after_first_cmd", now_stamp)
                    return

        if not self._stop_requested and self._maybe_finish_for_goal(now_stamp, goal_dist, cmd_v, cmd_w):
            return

        if not self._stop_requested and self._maybe_finish_for_stuck(now_stamp, goal_dist):
            return

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
        mean_efe_obstacle = self._efe_obstacle_sum / max(self._efe_count, 1) if self._efe_count > 0 else math.nan
        mean_solve_time_ms = self._solve_time_ms_sum / max(self._solve_count, 1) if self._solve_count > 0 else math.nan
        mean_p_vis_plan = self._p_vis_plan_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_p_vis_plan_eff = self._p_vis_plan_eff_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_r_plan_u_std = self._r_plan_u_std_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        mean_r_plan_v_std = self._r_plan_v_std_sum / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        fraction_time_p_vis_below_0_2 = self._p_vis_plan_below_0_2_count / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        fraction_time_p_vis_eff_below_0_2 = self._p_vis_plan_eff_below_0_2_count / max(self._p_vis_count, 1) if self._p_vis_count > 0 else math.nan
        max_r_plan_std = self._max_r_plan_std if self._p_vis_count > 0 else math.nan
        mean_state_error_odom_m = (
            self._state_error_odom_sum / self._state_error_odom_count
            if self._state_error_odom_count > 0 else math.nan
        )
        mean_belief_error_odom_m = (
            self._belief_error_odom_sum / self._belief_error_odom_count
            if self._belief_error_odom_count > 0 else math.nan
        )
        mean_state_error_odom_after_first_cmd_m = (
            self._state_error_odom_after_first_cmd_sum
            / self._state_error_odom_after_first_cmd_count
            if self._state_error_odom_after_first_cmd_count > 0 else math.nan
        )
        mean_belief_error_odom_after_first_cmd_m = (
            self._belief_error_odom_after_first_cmd_sum
            / self._belief_error_odom_after_first_cmd_count
            if self._belief_error_odom_after_first_cmd_count > 0 else math.nan
        )
        # GROUND-TRUTH error means (the honest localization metric)
        mean_belief_error_gt_m = (
            self._belief_error_gt_sum / self._belief_error_gt_count
            if self._belief_error_gt_count > 0 else math.nan
        )
        mean_state_error_gt_m = (
            self._state_error_gt_sum / self._state_error_gt_count
            if self._state_error_gt_count > 0 else math.nan
        )
        mean_belief_error_gt_after_first_cmd_m = (
            self._belief_error_gt_after_first_cmd_sum
            / self._belief_error_gt_after_first_cmd_count
            if self._belief_error_gt_after_first_cmd_count > 0 else math.nan
        )
        mean_state_error_gt_after_first_cmd_m = (
            self._state_error_gt_after_first_cmd_sum
            / self._state_error_gt_after_first_cmd_count
            if self._state_error_gt_after_first_cmd_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_odom_yaw_error_rad = (
            self._odom_map_vs_odom_yaw_error_sum / self._truth_odom_yaw_error_count
            if self._truth_odom_yaw_error_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_state_yaw_error_rad = (
            self._odom_map_vs_state_yaw_error_sum / self._truth_state_yaw_error_count
            if self._truth_state_yaw_error_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_belief_yaw_error_rad = (
            self._odom_map_vs_belief_yaw_error_sum / self._truth_belief_yaw_error_count
            if self._truth_belief_yaw_error_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_odom_yaw_error_after_first_cmd_rad = (
            self._odom_map_vs_odom_yaw_error_after_first_cmd_sum
            / self._odom_map_vs_odom_yaw_error_after_first_cmd_count
            if self._odom_map_vs_odom_yaw_error_after_first_cmd_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_state_yaw_error_after_first_cmd_rad = (
            self._odom_map_vs_state_yaw_error_after_first_cmd_sum
            / self._odom_map_vs_state_yaw_error_after_first_cmd_count
            if self._odom_map_vs_state_yaw_error_after_first_cmd_count > 0 else math.nan
        )
        mean_abs_odom_map_vs_belief_yaw_error_after_first_cmd_rad = (
            self._odom_map_vs_belief_yaw_error_after_first_cmd_sum
            / self._odom_map_vs_belief_yaw_error_after_first_cmd_count
            if self._odom_map_vs_belief_yaw_error_after_first_cmd_count > 0 else math.nan
        )

        if stamp is None:
            stamp = float(self.get_clock().now().nanoseconds) * 1e-9

        elapsed_after_first_cmd_s = stamp - self._first_cmd_stamp if self._first_cmd_stamp is not None else 0.0
        if (
            self._first_cmd_stamp is not None
            and math.isfinite(self._goal_region_first_stamp)
        ):
            goal_region_after_first_cmd_s = float(self._goal_region_first_stamp - self._first_cmd_stamp)
        else:
            goal_region_after_first_cmd_s = math.nan

        final_goal_distance = math.nan
        true_ok, _truth_stamp, true_x, true_y, _true_yaw = self._latest_truth_pose()
        if true_ok and self.goal_msg:
            goal_x = float(self.goal_msg.pose.position.x)
            goal_y = float(self.goal_msg.pose.position.y)
            dx = goal_x - true_x
            dy = goal_y - true_y
            final_goal_distance = math.hypot(dx, dy)

        crashed = bool(self._contact_collision_seen or self._geom_collision_seen)
        min_wall_distance_m = self._min_wall_distance if math.isfinite(self._min_wall_distance) else math.inf
        min_obstacle_distance_m = self._min_obstacle_distance if math.isfinite(self._min_obstacle_distance) else math.inf
        goal_region_success = bool(
            self._goal_region_reached()
            and not crashed
            and self._valid_run
        )

        summary = {
            'completed': True,
            'completion_reason': reason,
            'first_cmd_stamp': self._first_cmd_stamp if self._first_cmd_stamp is not None else math.nan,
            'stop_stamp': stamp,
            'elapsed_after_first_cmd_s': elapsed_after_first_cmd_s,
            'path_length_m': self._cumulative_path_length,
            'final_goal_distance': final_goal_distance,
            'minimum_goal_distance': self._min_goal_distance if math.isfinite(self._min_goal_distance) else math.nan,
            'goal_success_radius': self.goal_success_radius,
            'goal_success_hold_s': self.goal_success_hold_s,
            'goal_stable_radius': self.goal_stable_radius,
            'goal_stable_hold_s': self.goal_stable_hold_s,
            'goal_stable_max_displacement_m': self.goal_stable_max_displacement_m,
            'goal_region_entered': bool(self._goal_region_entered),
            'goal_region_first_stamp': self._goal_region_first_stamp,
            'goal_region_after_first_cmd_s': goal_region_after_first_cmd_s,
            'goal_region_success': goal_region_success,
            'stuck_window_s': self.stuck_window_s,
            'stuck_max_displacement_m': self.stuck_max_displacement_m,
            'stuck_max_goal_improvement_m': self.stuck_max_goal_improvement_m,
            'stuck_cmd_fraction_min': self.stuck_cmd_fraction_min,
            'stuck_idle_cmd_fraction_max': self.stuck_idle_cmd_fraction_max,
            'mean_solve_time_ms': mean_solve_time_ms,
            # EFE terms used by the paper objective.
            'mean_efe_risk': mean_efe_risk,
            'mean_efe_ambiguity': mean_efe_ambiguity,
            'mean_efe_control': mean_efe_control,
            'mean_efe_obstacle': mean_efe_obstacle,
            'mean_p_vis_plan': mean_p_vis_plan,
            'mean_p_vis_plan_eff': mean_p_vis_plan_eff,
            'fraction_time_p_vis_below_0_2': fraction_time_p_vis_below_0_2,
            'fraction_time_p_vis_eff_below_0_2': fraction_time_p_vis_eff_below_0_2,
            'mean_r_plan_u_std': mean_r_plan_u_std,
            'mean_r_plan_v_std': mean_r_plan_v_std,
            'max_r_plan_std': max_r_plan_std,
            # Explicit truth vs perception / planner belief errors
            'mean_state_error_odom_m': mean_state_error_odom_m,
            'mean_belief_error_odom_m': mean_belief_error_odom_m,
            'mean_state_error_odom_after_first_cmd_m': mean_state_error_odom_after_first_cmd_m,
            'mean_belief_error_odom_after_first_cmd_m': mean_belief_error_odom_after_first_cmd_m,
            # GROUND-TRUTH error means (honest; use these, not the mean_truth_* above)
            'mean_belief_error_gt_m': mean_belief_error_gt_m,
            'mean_state_error_gt_m': mean_state_error_gt_m,
            'mean_belief_error_gt_after_first_cmd_m': mean_belief_error_gt_after_first_cmd_m,
            'mean_state_error_gt_after_first_cmd_m': mean_state_error_gt_after_first_cmd_m,
            'mean_abs_odom_map_vs_odom_yaw_error_rad': mean_abs_odom_map_vs_odom_yaw_error_rad,
            'mean_abs_odom_map_vs_state_yaw_error_rad': mean_abs_odom_map_vs_state_yaw_error_rad,
            'mean_abs_odom_map_vs_belief_yaw_error_rad': mean_abs_odom_map_vs_belief_yaw_error_rad,
            'mean_abs_odom_map_vs_odom_yaw_error_after_first_cmd_rad': mean_abs_odom_map_vs_odom_yaw_error_after_first_cmd_rad,
            'mean_abs_odom_map_vs_state_yaw_error_after_first_cmd_rad': mean_abs_odom_map_vs_state_yaw_error_after_first_cmd_rad,
            'mean_abs_odom_map_vs_belief_yaw_error_after_first_cmd_rad': mean_abs_odom_map_vs_belief_yaw_error_after_first_cmd_rad,
            'crashed': crashed,
            'collision_any': crashed,
            'collision_contact': bool(self._contact_collision_seen),
            'collision_geom': bool(self._geom_collision_seen),
            'collision_reason': self._collision_reason,
            'first_crash_stamp': self._first_crash_stamp,
            'min_wall_distance_m': min_wall_distance_m,
            'min_obstacle_distance_m': min_obstacle_distance_m,
            'max_wall_penetration_m': float(self._max_wall_penetration),
            'max_obstacle_penetration_m': float(self._max_obstacle_penetration),
            'off_map': bool(self._off_map_seen),
            'inside_no_go': bool(self._inside_no_go_seen),
            'valid_run': bool(self._valid_run),
            'invalid_reason': self._invalid_reason,
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
                    'valid_run': bool(getattr(self, '_valid_run', True)),
                    'invalid_reason': str(getattr(self, '_invalid_reason', '') or ''),
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
