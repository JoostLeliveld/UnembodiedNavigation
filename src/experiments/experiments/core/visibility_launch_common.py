"""Shared launch helpers for the visibility-aware thesis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from launch.actions import IncludeLaunchDescription, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


PAPER_LAUNCH_DEFAULTS: Dict[str, str] = {
    'planner': 'visibility_aware_efe',
    'world': 'warehouse_occ_light.world.sdf',
    'task': '',
    'seed': '0',
    'odom_wait_timeout_s': '60.0',
    'odom_wait_min_messages': '1',
    'odom_wait_require_pose_match': 'false',
    'use_pixel_correction': 'true',
    'pixel_timeout_s': '0.5',
    'pixel_correction_min_interval_s': '0.1',
    'pixel_correction_approx': 'AUTO',
    'skip_stale_pixel_correction': 'true',
    'min_state_cov': '1e-6',
    'plan_rate': '2.0',
    'horizon': '40',
    'dt': '0.25',
    'control_weight': '0.0',
    'risk_weight_obs': '1.0',
    'ambiguity_weight': '3.0',
    'goal_sigma_uv': '2.0',
    'use_ambiguity': 'true',
    'use_obs_risk': 'true',
    'r_visible_uv': '2.5',
    'r_miss_uv': '120.0',
    'visibility_sigma_kappa': '1.0',
    'goal_prior_u_std_start': '80.0',
    'goal_prior_v_std_start': '80.0',
    'goal_prior_u_std_final': '4.0',
    'goal_prior_v_std_final': '4.0',
    'goal_tightening_power': '0.45',
    'goal_progress_n_steps': '90',
    'observation_risk_scale': '1.25',
    'ambiguity_term_scale': '1.00',
    'discount_gamma': '0.98',
    'min_terminal_goal_progress_m': '0.0',
    'invalid_rollout_barrier_cost': '1000000.0',
    'robot_collision_radius_m': '0.125',
    'bridge_contacts': 'true',
    'use_command_noise': 'true',
    'process_noise_xy': '0.01',
    'process_noise_theta': '0.02',
    'obs_noise_uv': '2.0',
    'optimizer_maxiter': '80',
    'optimizer_maxfun': '500',
    'optimizer_ftol': '1e-6',
    'optimizer_gtol': '1e-4',
    'optimizer_warm_start': 'true',
    'odom_heading_correction_mode': 'kalman',
    'clamp_pixel_uv_theta_without_yaw': 'false',
    'debug_runtime': 'false',
    'auto_stop_on_goal': 'true',
    'goal_success_radius': '0.20',
    'goal_success_hold_s': '2.0',
    'run_timeout_after_first_cmd_s': '75.0',
    'first_cmd_linear_eps': '0.02',
    'first_cmd_angular_eps': '0.10',
    'yolo_model': '',
    'yolo_device': '',
    'yolo_imgsz': '640',
    'yolo_conf_threshold': '0.25',
    'yolo_iou_threshold': '0.45',
    'yolo_target_class': 'robot',
    'yolo_class_id': '-1',
    'yolo_use_masks': 'true',
    'yolo_min_mask_area_px': '12.0',
    'yolo_mask_bottom_band_px': '3.0',
    'log_dir': 'logs/experiments',
}

# Command noise shape — paper-locked, not user-overridable.
# These values were calibrated for the TurtleBot3 in Gazebo and must not change between runs.
_COMMAND_NOISE_LINEAR_SLIP_MEAN: float = 0.03
_COMMAND_NOISE_LINEAR_SLIP_STD: float = 0.06
_COMMAND_NOISE_ANGULAR_SLIP_MEAN: float = 0.00
_COMMAND_NOISE_ANGULAR_SLIP_STD: float = 0.04
_COMMAND_NOISE_LINEAR_ADDITIVE_STD: float = 0.008
_COMMAND_NOISE_ANGULAR_ADDITIVE_STD: float = 0.035
_COMMAND_NOISE_CORRELATION_ALPHA: float = 0.85

# Sensor pixel noise — paper-locked, not user-overridable.
_SENSOR_PIXEL_NOISE_SIGMA: float = 1.0


VISIBILITY_FALLBACK_DEFAULTS: Dict[str, object] = {
    'visibility_target_height_m': 0.0,
    'use_nogo_cost': 'true',
    'nogo_penalty_type': 'softplus',
    'nogo_weight': 40.0,
    'nogo_safe_distance': 0.35,
    'nogo_gaussian_sigma': 0.25,
    'nogo_softplus_scale': 0.08,
    'nogo_logbarrier_scale': 0.25,
    'nogo_logbarrier_eps': 1e-3,
}


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _launch_value(context, name: str, default_value: str) -> str:
    return LaunchConfiguration(name, default=default_value).perform(context)

def _matches_default(current_value: object, default_value: object) -> bool:
    if isinstance(default_value, bool):
        return bool(current_value) == default_value
    if isinstance(default_value, int):
        return int(current_value) == default_value
    if isinstance(default_value, str):
        return str(current_value).strip().lower() == default_value.strip().lower()
    return abs(float(current_value) - float(default_value)) < 1e-9


def _apply_visibility_profile_defaults(cfg: Dict[str, object], profile: Dict[str, object]) -> None:
    visibility_defaults = profile.get('visibility_defaults')
    if not isinstance(visibility_defaults, dict):
        return
    for key, fallback_value in VISIBILITY_FALLBACK_DEFAULTS.items():
        if key not in visibility_defaults:
            continue
        if key not in cfg or not _matches_default(cfg[key], fallback_value):
            continue
        if isinstance(fallback_value, int):
            cfg[key] = int(visibility_defaults[key])
        elif isinstance(fallback_value, str):
            cfg[key] = str(visibility_defaults[key])
        else:
            cfg[key] = float(visibility_defaults[key])


def _require_task_field(task, key):
    if key not in task:
        raise RuntimeError(f"Task is missing '{key}' field")
    return task[key]


def _state_estimator_metadata() -> Dict[str, str]:
    return {
        'state_source_x': 'yolo_mask_or_bbox_homography',
        'state_source_y': 'yolo_mask_or_bbox_homography',
        'state_source_theta': 'odometry_heading',
        'state_estimator_mode': 'yolo_mask_or_bbox_camera_xy_odom_theta',
    }


def parse_common_launch_config(context) -> Dict[str, object]:
    """Parse the generic visibility-aware agent launch arguments."""
    seed_value = int(LaunchConfiguration('seed').perform(context))
    world_value = LaunchConfiguration('world').perform(context)
    visibility_enabled_default = 'true'
    use_visibility_raw = _launch_value(context, 'use_visibility_model', visibility_enabled_default).strip().lower()

    cfg: Dict[str, object] = {
        'use_sim_time': True,
        'planner': _launch_value(context, 'planner', PAPER_LAUNCH_DEFAULTS['planner']).strip(),
        'world': world_value,
        'world_profiles_path': LaunchConfiguration('world_profiles').perform(context),
        'tasks_yaml': LaunchConfiguration('tasks_yaml').perform(context),
        'task_name': _launch_value(context, 'task', PAPER_LAUNCH_DEFAULTS['task']).strip(),
        'comparison_method_id': _launch_value(context, 'comparison_method_id', '').strip(),
        'log_dir': _launch_value(context, 'log_dir', PAPER_LAUNCH_DEFAULTS['log_dir']).strip(),
        'seed': seed_value,
        'perception_backend': 'yolo',
        'sensor_pixel_noise_sigma': _SENSOR_PIXEL_NOISE_SIGMA,
        'odom_wait_timeout_s': float(
            _launch_value(context, 'odom_wait_timeout_s', PAPER_LAUNCH_DEFAULTS['odom_wait_timeout_s'])
        ),
        'odom_wait_min_messages': max(1, int(float(_launch_value(context, 'odom_wait_min_messages', '1')))),
        'odom_wait_require_pose_match': _as_bool(_launch_value(context, 'odom_wait_require_pose_match', 'false')),
        'odom_wait_position_tolerance': 0.25,
        'odom_wait_yaw_tolerance': 0.5,
        'use_pixel_correction': _as_bool(_launch_value(context, 'use_pixel_correction', PAPER_LAUNCH_DEFAULTS['use_pixel_correction'])),
        'pixel_timeout_s': float(_launch_value(context, 'pixel_timeout_s', PAPER_LAUNCH_DEFAULTS['pixel_timeout_s'])),
        'pixel_correction_min_interval_s': float(_launch_value(context, 'pixel_correction_min_interval_s', '0.0')),
        'pixel_correction_approx': _launch_value(
            context,
            'pixel_correction_approx',
            PAPER_LAUNCH_DEFAULTS['pixel_correction_approx'],
        ).strip().upper(),
        'skip_stale_pixel_correction': _as_bool(_launch_value(context, 'skip_stale_pixel_correction', 'true')),
        'use_ambiguity': _as_bool(_launch_value(context, 'use_ambiguity', PAPER_LAUNCH_DEFAULTS['use_ambiguity'])),
        'use_obs_risk': _as_bool(_launch_value(context, 'use_obs_risk', PAPER_LAUNCH_DEFAULTS['use_obs_risk'])),
        'auto_stop_on_goal': _as_bool(_launch_value(context, 'auto_stop_on_goal', PAPER_LAUNCH_DEFAULTS['auto_stop_on_goal'])),
        'goal_success_radius': float(_launch_value(context, 'goal_success_radius', PAPER_LAUNCH_DEFAULTS['goal_success_radius'])),
        'goal_success_hold_s': float(_launch_value(context, 'goal_success_hold_s', '2.0')),
        'run_timeout_after_first_cmd_s': float(_launch_value(context, 'run_timeout_after_first_cmd_s', PAPER_LAUNCH_DEFAULTS['run_timeout_after_first_cmd_s'])),
        'first_cmd_linear_eps': float(_launch_value(context, 'first_cmd_linear_eps', PAPER_LAUNCH_DEFAULTS['first_cmd_linear_eps'])),
        'first_cmd_angular_eps': float(_launch_value(context, 'first_cmd_angular_eps', PAPER_LAUNCH_DEFAULTS['first_cmd_angular_eps'])),
        'stuck_window_s': 8.0,
        'stuck_max_displacement_m': 0.08,
        'stuck_max_goal_improvement_m': 0.05,
        'stuck_cmd_fraction_min': 0.50,
        'process_noise_xy': float(_launch_value(context, 'process_noise_xy', PAPER_LAUNCH_DEFAULTS['process_noise_xy'])),
        'process_noise_theta': float(_launch_value(context, 'process_noise_theta', PAPER_LAUNCH_DEFAULTS['process_noise_theta'])),
        'obs_noise_uv': float(_launch_value(context, 'obs_noise_uv', PAPER_LAUNCH_DEFAULTS['obs_noise_uv'])),
        'optimizer_maxiter': int(_launch_value(context, 'optimizer_maxiter', PAPER_LAUNCH_DEFAULTS['optimizer_maxiter'])),
        'optimizer_maxfun': int(_launch_value(context, 'optimizer_maxfun', PAPER_LAUNCH_DEFAULTS['optimizer_maxfun'])),
        'optimizer_ftol': float(_launch_value(context, 'optimizer_ftol', PAPER_LAUNCH_DEFAULTS['optimizer_ftol'])),
        'optimizer_gtol': float(_launch_value(context, 'optimizer_gtol', PAPER_LAUNCH_DEFAULTS['optimizer_gtol'])),
        'optimizer_warm_start': _as_bool(_launch_value(context, 'optimizer_warm_start', PAPER_LAUNCH_DEFAULTS['optimizer_warm_start'])),
        'odom_heading_correction_mode': _launch_value(
            context, 'odom_heading_correction_mode', PAPER_LAUNCH_DEFAULTS['odom_heading_correction_mode']
        ).strip().lower(),
        'clamp_pixel_uv_theta_without_yaw': _as_bool(
            _launch_value(context, 'clamp_pixel_uv_theta_without_yaw', PAPER_LAUNCH_DEFAULTS['clamp_pixel_uv_theta_without_yaw'])
        ),
        'plan_rate': float(_launch_value(context, 'plan_rate', PAPER_LAUNCH_DEFAULTS['plan_rate'])),
        'horizon': int(_launch_value(context, 'horizon', PAPER_LAUNCH_DEFAULTS['horizon'])),
        'dt': float(_launch_value(context, 'dt', PAPER_LAUNCH_DEFAULTS['dt'])),
        'control_weight': float(_launch_value(context, 'control_weight', PAPER_LAUNCH_DEFAULTS['control_weight'])),
        'risk_weight_obs': float(_launch_value(context, 'risk_weight_obs', '1.0')),
        'ambiguity_weight': float(_launch_value(context, 'ambiguity_weight', '1.0')),
        'r_visible_uv': float(_launch_value(context, 'r_visible_uv', PAPER_LAUNCH_DEFAULTS['r_visible_uv'])),
        'r_miss_uv': float(_launch_value(context, 'r_miss_uv', PAPER_LAUNCH_DEFAULTS['r_miss_uv'])),
        'visibility_sigma_kappa': float(_launch_value(context, 'visibility_sigma_kappa', PAPER_LAUNCH_DEFAULTS['visibility_sigma_kappa'])),
        'goal_prior_u_std_start': float(_launch_value(context, 'goal_prior_u_std_start', PAPER_LAUNCH_DEFAULTS['goal_prior_u_std_start'])),
        'goal_prior_v_std_start': float(_launch_value(context, 'goal_prior_v_std_start', PAPER_LAUNCH_DEFAULTS['goal_prior_v_std_start'])),
        'goal_prior_u_std_final': float(_launch_value(context, 'goal_prior_u_std_final', PAPER_LAUNCH_DEFAULTS['goal_prior_u_std_final'])),
        'goal_prior_v_std_final': float(_launch_value(context, 'goal_prior_v_std_final', PAPER_LAUNCH_DEFAULTS['goal_prior_v_std_final'])),
        'goal_tightening_power': float(_launch_value(context, 'goal_tightening_power', PAPER_LAUNCH_DEFAULTS['goal_tightening_power'])),
        'goal_progress_n_steps': int(_launch_value(context, 'goal_progress_n_steps', PAPER_LAUNCH_DEFAULTS['goal_progress_n_steps'])),
        'observation_risk_scale': float(_launch_value(context, 'observation_risk_scale', PAPER_LAUNCH_DEFAULTS['observation_risk_scale'])),
        'ambiguity_term_scale': float(_launch_value(context, 'ambiguity_term_scale', PAPER_LAUNCH_DEFAULTS['ambiguity_term_scale'])),
        'discount_gamma': float(_launch_value(context, 'discount_gamma', PAPER_LAUNCH_DEFAULTS['discount_gamma'])),
        'use_visibility_model': _as_bool(
            visibility_enabled_default if use_visibility_raw in ('', 'auto', 'default') else use_visibility_raw
        ),
        'perception_use_geometry_occlusion': _as_bool(
            _launch_value(context, 'perception_use_geometry_occlusion', 'true')
        ),
        'visibility_target_height_m': float(_launch_value(context, 'visibility_target_height_m', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_target_height_m']))),
        'visibility_geometry_json': _launch_value(context, 'visibility_geometry_json', ''),
        'collision_geometry_json': _launch_value(context, 'collision_geometry_json', ''),
        'visibility_artifact_path': _launch_value(context, 'visibility_artifact_path', ''),
        'use_nogo_cost': _launch_value(context, 'use_nogo_cost', str(VISIBILITY_FALLBACK_DEFAULTS['use_nogo_cost'])).strip().lower(),
        'nogo_penalty_type': _launch_value(context, 'nogo_penalty_type', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_penalty_type'])).strip().lower(),
        'nogo_weight': float(_launch_value(context, 'nogo_weight', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_weight']))),
        'nogo_safe_distance': float(_launch_value(context, 'nogo_safe_distance', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_safe_distance']))),
        'nogo_gaussian_sigma': float(_launch_value(context, 'nogo_gaussian_sigma', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_gaussian_sigma']))),
        'nogo_softplus_scale': float(_launch_value(context, 'nogo_softplus_scale', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_softplus_scale']))),
        'nogo_logbarrier_scale': float(_launch_value(context, 'nogo_logbarrier_scale', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_logbarrier_scale']))),
        'nogo_logbarrier_eps': float(_launch_value(context, 'nogo_logbarrier_eps', str(VISIBILITY_FALLBACK_DEFAULTS['nogo_logbarrier_eps']))),
        'goal_sigma_uv': float(_launch_value(context, 'goal_sigma_uv', PAPER_LAUNCH_DEFAULTS['goal_sigma_uv'])),
        'min_terminal_goal_progress_m': float(
            _launch_value(
                context,
                'min_terminal_goal_progress_m',
                PAPER_LAUNCH_DEFAULTS['min_terminal_goal_progress_m'],
            )
        ),
        'invalid_rollout_barrier_cost': float(
            _launch_value(
                context,
                'invalid_rollout_barrier_cost',
                PAPER_LAUNCH_DEFAULTS['invalid_rollout_barrier_cost'],
            )
        ),
        'robot_collision_radius_m': float(
            _launch_value(
                context,
                'robot_collision_radius_m',
                PAPER_LAUNCH_DEFAULTS['robot_collision_radius_m'],
            )
        ),
        'bridge_contacts': _as_bool(
            _launch_value(context, 'bridge_contacts', PAPER_LAUNCH_DEFAULTS['bridge_contacts'])
        ),
        'use_command_noise': _as_bool(
            _launch_value(context, 'use_command_noise', PAPER_LAUNCH_DEFAULTS['use_command_noise'])
        ),
        'command_noise_linear_slip_mean': _COMMAND_NOISE_LINEAR_SLIP_MEAN,
        'command_noise_linear_slip_std': _COMMAND_NOISE_LINEAR_SLIP_STD,
        'command_noise_angular_slip_mean': _COMMAND_NOISE_ANGULAR_SLIP_MEAN,
        'command_noise_angular_slip_std': _COMMAND_NOISE_ANGULAR_SLIP_STD,
        'command_noise_linear_additive_std': _COMMAND_NOISE_LINEAR_ADDITIVE_STD,
        'command_noise_angular_additive_std': _COMMAND_NOISE_ANGULAR_ADDITIVE_STD,
        'command_noise_correlation_alpha': _COMMAND_NOISE_CORRELATION_ALPHA,
        'min_state_cov': float(_launch_value(context, 'min_state_cov', '1e-6')),
        'debug_runtime': _as_bool(_launch_value(context, 'debug_runtime', 'false')),
        'enable_logging': _as_bool(_launch_value(context, 'enable_logging', 'true')),
        'use_rviz': _as_bool(_launch_value(context, 'use_rviz', 'false')),
        'rviz_config': _launch_value(context, 'rviz_config', ''),
        'yolo_model': _launch_value(context, 'yolo_model', PAPER_LAUNCH_DEFAULTS['yolo_model']),
        'yolo_device': _launch_value(context, 'yolo_device', PAPER_LAUNCH_DEFAULTS['yolo_device']),
        'yolo_imgsz': int(_launch_value(context, 'yolo_imgsz', PAPER_LAUNCH_DEFAULTS['yolo_imgsz'])),
        'yolo_conf_threshold': float(_launch_value(context, 'yolo_conf_threshold', PAPER_LAUNCH_DEFAULTS['yolo_conf_threshold'])),
        'yolo_iou_threshold': float(_launch_value(context, 'yolo_iou_threshold', PAPER_LAUNCH_DEFAULTS['yolo_iou_threshold'])),
        'yolo_target_class': _launch_value(context, 'yolo_target_class', PAPER_LAUNCH_DEFAULTS['yolo_target_class']),
        'yolo_class_id': int(_launch_value(context, 'yolo_class_id', PAPER_LAUNCH_DEFAULTS['yolo_class_id'])),
        'yolo_use_masks': _as_bool(_launch_value(context, 'yolo_use_masks', PAPER_LAUNCH_DEFAULTS['yolo_use_masks'])),
        'yolo_min_mask_area_px': float(_launch_value(context, 'yolo_min_mask_area_px', PAPER_LAUNCH_DEFAULTS['yolo_min_mask_area_px'])),
        'yolo_mask_bottom_band_px': float(_launch_value(context, 'yolo_mask_bottom_band_px', PAPER_LAUNCH_DEFAULTS['yolo_mask_bottom_band_px'])),
    }

    return cfg


def resolve_world_setup(cfg: Dict[str, object]) -> Dict[str, object]:
    """Resolve world profile/task and derive camera/spawn launch parameters."""
    from experiments.core.world_profiles import (
        load_profile,
        compute_camera_quaternion_from_rpy,
        compute_look_at_from_pose,
        resolve_profile_asset_path,
        serialize_collision_geometry_from_world,
        serialize_occlusion_geometry_from_world,
    )
    from experiments.core.tasks import load_tasks, select_task

    profile, _intrinsics, world_path, camera_pose = load_profile(
        cfg['world_profiles_path'], cfg['world']
    )
    _apply_visibility_profile_defaults(cfg, profile)
    tasks_by_world = load_tasks(cfg['tasks_yaml'])
    task_name = str(cfg.get('task_name', '') or '').strip()
    if not task_name:
        task_name = str(profile.get('recommended_task', '') or '').strip()
    task = select_task(tasks_by_world, cfg['world'], task_name)

    start = _require_task_field(task, 'start')
    goal = _require_task_field(task, 'goal')
    for key in ('x', 'y', 'yaw'):
        if key not in start:
            raise RuntimeError(f"Task start missing '{key}'")
    for key in ('x', 'y'):
        if key not in goal:
            raise RuntimeError(f"Task goal missing '{key}'")

    profile_spawn = profile.get('spawn', {}) if isinstance(profile.get('spawn', {}), dict) else {}
    start_z = start.get('z', profile_spawn.get('z', 0.05))

    spawn = {
        'x': float(start['x']),
        'y': float(start['y']),
        'z': float(start_z),
        'yaw': float(start['yaw']),
    }
    goal_x = float(goal['x'])
    goal_y = float(goal['y'])

    planner = str(cfg['planner'])
    if planner == 'auto':
        planner = profile['planner_default']
    if planner == 'constant_R_efe':
        cfg['use_visibility_model'] = False
        cfg['use_ambiguity'] = False
        cfg['use_obs_risk'] = True
    elif planner == 'risk_only_ablation':
        cfg['use_visibility_model'] = True
        cfg['use_ambiguity'] = False
        cfg['use_obs_risk'] = True

    visibility_artifact_path = str(cfg.get('visibility_artifact_path', '') or '').strip()
    if not visibility_artifact_path:
        raise RuntimeError(
            "visibility_artifact_path must be provided explicitly — "
            "no fallback to world profile defaults is allowed for paper runs."
        )
    visibility_artifact_path = resolve_profile_asset_path(cfg['world_profiles_path'], visibility_artifact_path)
    if not Path(visibility_artifact_path).exists():
        raise RuntimeError(f"visibility_artifact_path does not exist: {visibility_artifact_path}")

    cam_pos = [camera_pose[0], camera_pose[1], camera_pose[2]]
    roll, pitch, yaw = camera_pose[3], camera_pose[4], camera_pose[5]
    look_at = compute_look_at_from_pose(cam_pos, roll, pitch, yaw)
    quat = compute_camera_quaternion_from_rpy(roll, pitch, yaw)
    spawn_quat = compute_camera_quaternion_from_rpy(0.0, 0.0, spawn['yaw'])

    intrinsics = dict(_intrinsics)
    camera_params = {
        'cam_pos': cam_pos,
        'look_at': look_at,
        'img_width': int(intrinsics['img_width']),
        'img_height': int(intrinsics['img_height']),
        'fov_h_rad': float(intrinsics['fov_h_rad']),
    }
    tf_args = {
        'use_sim_time': 'true',
        'cam_x': str(cam_pos[0]),
        'cam_y': str(cam_pos[1]),
        'cam_z': str(cam_pos[2]),
        'cam_qx': str(quat[0]),
        'cam_qy': str(quat[1]),
        'cam_qz': str(quat[2]),
        'cam_qw': str(quat[3]),
        'odom_x': str(spawn['x']),
        'odom_y': str(spawn['y']),
        'odom_z': '0.0',
        'odom_qx': str(spawn_quat[0]),
        'odom_qy': str(spawn_quat[1]),
        'odom_qz': str(spawn_quat[2]),
        'odom_qw': str(spawn_quat[3]),
    }

    visibility_geometry_json = str(cfg.get('visibility_geometry_json', '') or '')
    collision_geometry_json = str(cfg.get('collision_geometry_json', '') or '')
    raw_use_nogo_cost = str(cfg.get('use_nogo_cost', 'auto')).strip().lower()
    nogo_geometry_needed = (
        raw_use_nogo_cost in ('1', 'true', 't', 'yes', 'y', 'on')
    )
    geometry_needed = bool(cfg.get('perception_use_geometry_occlusion', False)) or nogo_geometry_needed
    if (not visibility_geometry_json) and geometry_needed:
        visibility_geometry_json = serialize_occlusion_geometry_from_world(world_path)
    if not collision_geometry_json:
        collision_geometry_json = serialize_collision_geometry_from_world(world_path)

    cfg = dict(cfg)
    cfg.update({
        'profile': profile,
        'task': task,
        'task_name': str(task.get('name', task_name)),
        'planner': planner,
        'spawn': spawn,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'camera_params': camera_params,
        'tf_args': tf_args,
        'world_path': world_path,
        'visibility_geometry_json': visibility_geometry_json,
        'collision_geometry_json': collision_geometry_json,
        'visibility_artifact_path': visibility_artifact_path,
    })
    return cfg


def build_shared_nodes(cfg: Dict[str, object]) -> Dict[str, object]:
    """Create shared nodes/components for the thesis pipeline."""
    state_sources = _state_estimator_metadata()
    sim_pkg = FindPackageShare('sim')
    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'use_lidar': 'false',
            'bridge_scan': 'false',
            'world': cfg['world'],
            'world_name': cfg['profile']['world_name'],
            'spawn_x': str(cfg['spawn']['x']),
            'spawn_y': str(cfg['spawn']['y']),
            'spawn_z': str(cfg['spawn']['z']),
            'spawn_yaw': str(cfg['spawn']['yaw']),
            'reset_world': 'false',
            'bridge_contacts': 'true' if cfg.get('bridge_contacts', True) else 'false',
        }.items(),
    )

    perception_pkg = FindPackageShare('perception')
    tf_static = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_pkg, 'launch', 'tf_static.launch.py'])
        ),
        launch_arguments=cfg['tf_args'].items(),
    )

    wait_for_odom = Node(
        package='sim',
        executable='wait_for_odom',
        name='wait_for_odom',
        output='screen',
        parameters=[{
            'topic': '/odom',
            'timeout_s': cfg['odom_wait_timeout_s'],
            'min_messages': cfg['odom_wait_min_messages'],
            'require_pose_match': cfg['odom_wait_require_pose_match'],
            'expected_x': 0.0,
            'expected_y': 0.0,
            'expected_yaw': 0.0,
            'position_tolerance': cfg['odom_wait_position_tolerance'],
            'yaw_tolerance': cfg['odom_wait_yaw_tolerance'],
        }],
    )

    command_noise_node = None
    if cfg.get('use_command_noise', True):
        command_noise_node = Node(
            package='sim',
            executable='actuation_noise_node',
            name='actuation_noise_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'enabled': True,
                'input_topic': '/cmd_vel_raw',
                'output_topic': '/cmd_vel',
                'diagnostics_topic': '/cmd_vel_noise/diagnostics',
                'seed': cfg['seed'],
                'linear_slip_mean': _COMMAND_NOISE_LINEAR_SLIP_MEAN,
                'linear_slip_std': _COMMAND_NOISE_LINEAR_SLIP_STD,
                'angular_slip_mean': _COMMAND_NOISE_ANGULAR_SLIP_MEAN,
                'angular_slip_std': _COMMAND_NOISE_ANGULAR_SLIP_STD,
                'linear_additive_std': _COMMAND_NOISE_LINEAR_ADDITIVE_STD,
                'angular_additive_std': _COMMAND_NOISE_ANGULAR_ADDITIVE_STD,
                'correlation_alpha': _COMMAND_NOISE_CORRELATION_ALPHA,
                'linear_min': 0.0,
                'linear_max': 0.22,
                'angular_min': -1.0,
                'angular_max': 1.0,
            }],
        )

    yolo_params = {
        'pixel_noise_sigma': _SENSOR_PIXEL_NOISE_SIGMA,
        'seed': cfg['seed'],
        'model_path': cfg['yolo_model'],
        'device': cfg['yolo_device'],
        'image_size': cfg['yolo_imgsz'],
        'confidence_threshold': cfg['yolo_conf_threshold'],
        'iou_threshold': cfg['yolo_iou_threshold'],
        'class_name': cfg['yolo_target_class'],
        'class_id': cfg['yolo_class_id'],
        'use_masks': cfg['yolo_use_masks'],
        'mask_min_area': cfg['yolo_min_mask_area_px'],
        'mask_bottom_band_px': cfg['yolo_mask_bottom_band_px'],
    }
    perception_node = Node(
        package='perception',
        executable='yolo_robot_detector_node',
        name='yolo_robot_detector_node',
        output='screen',
        parameters=[yolo_params],
    )

    pixel_params = {
        'use_sim_time': True,
        'frame_id': 'map_bev',
        'pixel_noise_sigma': 0.0,
        'heading_pixel_noise_sigma': _SENSOR_PIXEL_NOISE_SIGMA,
        'transform_noise_sigma': 0.0,
        'use_odom_heading_fallback': True,
        'odom_heading_timeout_s': 0.5,
        'odom_heading_sigma_rad': 0.08,
        'odom_yaw_offset_rad': float(cfg['spawn']['yaw']),
        'infer_yaw_from_motion': False,
        'seed': cfg['seed'],
        **cfg['camera_params'],
    }
    pixel_to_bev = Node(
        package='state',
        executable='pixel_to_bev_state_node',
        name='pixel_to_bev_state_node',
        output='screen',
        parameters=[pixel_params],
    )

    mission_node = Node(
        package='experiments',
        executable='goal_mission_node',
        name='goal_mission_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'frame_id': 'map_bev',
            'goal_x': cfg['goal_x'],
            'goal_y': cfg['goal_y'],
        }],
    )

    goal_marker_node = Node(
        package='experiments',
        executable='goal_marker_node',
        name='goal_marker_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'marker_topic': '/goal_marker',
            'marker_ns': 'goal',
            'scale': 0.35,
            'z': 0.08,
            'color_r': 0.0,
            'color_g': 0.35,
            'color_b': 1.0,
            'color_a': 1.0,
        }],
    )

    logger_node = None
    if cfg.get('enable_logging', True):
        logger_node = Node(
            package='experiments',
            executable='experiment_logger',
            name='experiment_logger',
            output='screen',
            on_exit=[Shutdown(reason='experiment_logger exited')],
            parameters=[{
                'use_sim_time': True,
                'log_dir': cfg['log_dir'],
                'seed': cfg['seed'],
                'method': cfg['comparison_method_id'] or cfg['planner'],
                'perception_backend': cfg['perception_backend'],
                'world': cfg['world'],
                'task': cfg['task'].get('name', cfg['task_name'] or ''),
                'planner': cfg['planner'],
                'state_source_x': state_sources['state_source_x'],
                'state_source_y': state_sources['state_source_y'],
                'state_source_theta': state_sources['state_source_theta'],
                'state_estimator_mode': state_sources['state_estimator_mode'],
                'use_pixel_correction': cfg['use_pixel_correction'],
                'use_ambiguity': cfg['use_ambiguity'],
                'use_obs_risk': cfg['use_obs_risk'],
                'use_visibility_model': cfg['use_visibility_model'],
                'visibility_artifact_path': cfg['visibility_artifact_path'],
                'risk_weight_obs': cfg['risk_weight_obs'],
                'goal_sigma_uv': cfg['goal_sigma_uv'],
                'r_visible_uv': cfg['r_visible_uv'],
                'r_miss_uv': cfg['r_miss_uv'],
                'visibility_sigma_kappa': cfg['visibility_sigma_kappa'],
                'goal_prior_u_std_start': cfg['goal_prior_u_std_start'],
                'goal_prior_v_std_start': cfg['goal_prior_v_std_start'],
                'goal_prior_u_std_final': cfg['goal_prior_u_std_final'],
                'goal_prior_v_std_final': cfg['goal_prior_v_std_final'],
                'goal_tightening_power': cfg['goal_tightening_power'],
                'goal_progress_n_steps': cfg['goal_progress_n_steps'],
                'observation_risk_scale': cfg['observation_risk_scale'],
                'ambiguity_term_scale': cfg['ambiguity_term_scale'],
                'discount_gamma': cfg['discount_gamma'],
                'visibility_target_height_m': cfg['visibility_target_height_m'],
                'visibility_geometry_json': cfg['visibility_geometry_json'],
                'collision_geometry_json': cfg['collision_geometry_json'],
                'perception_use_geometry_occlusion': cfg['perception_use_geometry_occlusion'],
                'use_nogo_cost': cfg.get('resolved_use_nogo_cost', False),
                'nogo_penalty_type': cfg['nogo_penalty_type'],
                'nogo_weight': cfg['nogo_weight'],
                'nogo_safe_distance': cfg['nogo_safe_distance'],
                'nogo_gaussian_sigma': cfg['nogo_gaussian_sigma'],
                'nogo_softplus_scale': cfg['nogo_softplus_scale'],
                'nogo_logbarrier_scale': cfg['nogo_logbarrier_scale'],
                'nogo_logbarrier_eps': cfg['nogo_logbarrier_eps'],
                'yolo_model': cfg['yolo_model'],
                'yolo_device': cfg['yolo_device'],
                'yolo_imgsz': cfg['yolo_imgsz'],
                'yolo_conf_threshold': cfg['yolo_conf_threshold'],
                'yolo_iou_threshold': cfg['yolo_iou_threshold'],
                'yolo_target_class': cfg['yolo_target_class'],
                'yolo_class_id': cfg['yolo_class_id'],
                'yolo_use_masks': cfg['yolo_use_masks'],
                'yolo_min_mask_area_px': cfg['yolo_min_mask_area_px'],
                'yolo_mask_bottom_band_px': cfg['yolo_mask_bottom_band_px'],
                'world_profiles_path': cfg['world_profiles_path'],
                'tasks_yaml': cfg['tasks_yaml'],
                'auto_stop_on_goal': cfg['auto_stop_on_goal'],
                'goal_success_radius': cfg['goal_success_radius'],
                'goal_success_hold_s': cfg['goal_success_hold_s'],
                'plan_rate': cfg['plan_rate'],
                'horizon': cfg['horizon'],
                'dt': cfg['dt'],
                'control_weight': cfg['control_weight'],
                'process_noise_xy': cfg['process_noise_xy'],
                'process_noise_theta': cfg['process_noise_theta'],
                'obs_noise_uv': cfg['obs_noise_uv'],
                'optimizer_maxiter': cfg['optimizer_maxiter'],
                'optimizer_maxfun': cfg['optimizer_maxfun'],
                'optimizer_ftol': cfg['optimizer_ftol'],
                'optimizer_gtol': cfg['optimizer_gtol'],
                'optimizer_warm_start': cfg['optimizer_warm_start'],
                'odom_heading_correction_mode': cfg['odom_heading_correction_mode'],
                'clamp_pixel_uv_theta_without_yaw': cfg['clamp_pixel_uv_theta_without_yaw'],
                'run_timeout_after_first_cmd_s': cfg['run_timeout_after_first_cmd_s'],
                'first_cmd_linear_eps': cfg['first_cmd_linear_eps'],
                'first_cmd_angular_eps': cfg['first_cmd_angular_eps'],
                'stuck_window_s': cfg['stuck_window_s'],
                'stuck_max_displacement_m': cfg['stuck_max_displacement_m'],
                'stuck_max_goal_improvement_m': cfg['stuck_max_goal_improvement_m'],
                'stuck_cmd_fraction_min': cfg['stuck_cmd_fraction_min'],
                'robot_collision_radius_m': cfg['robot_collision_radius_m'],
                'use_command_noise': cfg['use_command_noise'],
                'command_noise_linear_slip_mean': cfg['command_noise_linear_slip_mean'],
                'command_noise_linear_slip_std': cfg['command_noise_linear_slip_std'],
                'command_noise_angular_slip_mean': cfg['command_noise_angular_slip_mean'],
                'command_noise_angular_slip_std': cfg['command_noise_angular_slip_std'],
                'command_noise_linear_additive_std': cfg['command_noise_linear_additive_std'],
                'command_noise_angular_additive_std': cfg['command_noise_angular_additive_std'],
                'command_noise_correlation_alpha': cfg['command_noise_correlation_alpha'],
                **cfg['camera_params'],
            }],
        )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', cfg['rviz_config']],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return {
        'bringup_sim': bringup_sim,
        'tf_static': tf_static,
        'wait_for_odom': wait_for_odom,
        'command_noise_node': command_noise_node,
        'perception_node': perception_node,
        'pixel_to_bev': pixel_to_bev,
        'mission_node': mission_node,
        'goal_marker_node': goal_marker_node,
        'logger_node': logger_node,
        'rviz': rviz,
    }


def build_agent_runtime_actions(cfg: Dict[str, object]) -> List[object]:
    """Create runtime actions for the visibility-aware agent launch."""
    raw_use_nogo_cost = cfg.get('use_nogo_cost', 'auto')
    if isinstance(raw_use_nogo_cost, str) and raw_use_nogo_cost in ('', 'auto', 'default'):
        resolved_use_nogo_cost = False
    else:
        resolved_use_nogo_cost = _as_bool(raw_use_nogo_cost)

    cfg = dict(cfg)
    cfg['resolved_use_nogo_cost'] = resolved_use_nogo_cost
    shared_nodes = build_shared_nodes(cfg)
    planner = cfg['planner']

    if planner not in ('visibility_aware_efe', 'constant_R_efe', 'risk_only_ablation'):
        raise RuntimeError(
            "planner must be 'visibility_aware_efe', 'constant_R_efe', or 'risk_only_ablation' for agent launch"
        )

    planner_params = {
        'visibility_aware_efe': {
            'approx_method': 'ET1',
            'use_ambiguity': cfg['use_ambiguity'],
            'use_obs_risk': cfg['use_obs_risk'],
        },
        # C3: GP-derived R_eff active, ambiguity term disabled.
        # Isolates whether the risk term alone (through R_eff) drives rerouting.
        'risk_only_ablation': {
            'approx_method': 'ET1',
            'use_ambiguity': False,
            'use_obs_risk': True,
        },
        'constant_R_efe': {
            'approx_method': 'ET1',
            'use_ambiguity': False,
            'use_obs_risk': True,
        },
    }
    planner_uses_visibility = bool(cfg['use_visibility_model']) and planner != 'constant_R_efe'
    agent_node = Node(
        package='planning',
        executable='efe_agent',
        name=f'{planner}_agent',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'cmd_topic': '/cmd_vel_raw' if cfg.get('use_command_noise', True) else '/cmd_vel',
            'plan_rate': cfg['plan_rate'],
            'horizon': cfg['horizon'],
            'dt': cfg['dt'],
            'control_weight': cfg['control_weight'],
            'use_pixel_correction': cfg['use_pixel_correction'],
            'pixel_timeout_s': cfg['pixel_timeout_s'],
            'pixel_correction_min_interval_s': cfg['pixel_correction_min_interval_s'],
            'pixel_correction_approx': cfg['pixel_correction_approx'],
            'skip_stale_pixel_correction': cfg['skip_stale_pixel_correction'],
            'heading_pixel_noise_sigma': _SENSOR_PIXEL_NOISE_SIGMA,
            'use_odom_heading_correction': True,
            'odom_heading_correction_mode': cfg['odom_heading_correction_mode'],
            'odom_heading_timeout_s': 0.75,
            'odom_heading_sigma_rad': 0.08,
            'odom_yaw_offset_rad': float(cfg['spawn']['yaw']),
            'clamp_pixel_uv_theta_without_yaw': cfg['clamp_pixel_uv_theta_without_yaw'],
            'min_state_cov': cfg['min_state_cov'],
            'debug_runtime': cfg['debug_runtime'],
            'process_noise_xy': cfg['process_noise_xy'],
            'process_noise_theta': cfg['process_noise_theta'],
            'obs_noise_uv': cfg['obs_noise_uv'],
            'goal_sigma_uv': cfg['goal_sigma_uv'],
            'risk_weight_obs': cfg['risk_weight_obs'],
            'ambiguity_weight': cfg['ambiguity_weight'],
            'r_visible_uv': cfg['r_visible_uv'],
            'r_miss_uv': cfg['r_miss_uv'],
            'visibility_sigma_kappa': cfg['visibility_sigma_kappa'],
            'goal_prior_u_std_start': cfg['goal_prior_u_std_start'],
            'goal_prior_v_std_start': cfg['goal_prior_v_std_start'],
            'goal_prior_u_std_final': cfg['goal_prior_u_std_final'],
            'goal_prior_v_std_final': cfg['goal_prior_v_std_final'],
            'goal_tightening_power': cfg['goal_tightening_power'],
            'goal_progress_n_steps': cfg['goal_progress_n_steps'],
            'observation_risk_scale': cfg['observation_risk_scale'],
            'ambiguity_term_scale': cfg['ambiguity_term_scale'],
            'discount_gamma': cfg['discount_gamma'],
            'use_visibility_model': planner_uses_visibility,
            'visibility_target_height_m': cfg['visibility_target_height_m'],
            'visibility_geometry_json': cfg['visibility_geometry_json'],
            'collision_geometry_json': cfg['collision_geometry_json'],
            'visibility_artifact_path': cfg['visibility_artifact_path'],
            'use_nogo_cost': cfg['resolved_use_nogo_cost'],
            'nogo_penalty_type': cfg['nogo_penalty_type'],
            'nogo_weight': cfg['nogo_weight'],
            'nogo_safe_distance': cfg['nogo_safe_distance'],
            'nogo_gaussian_sigma': cfg['nogo_gaussian_sigma'],
            'nogo_softplus_scale': cfg['nogo_softplus_scale'],
            'nogo_logbarrier_scale': cfg['nogo_logbarrier_scale'],
            'nogo_logbarrier_eps': cfg['nogo_logbarrier_eps'],
            'robot_collision_radius_m': cfg['robot_collision_radius_m'],
            'min_terminal_goal_progress_m': cfg['min_terminal_goal_progress_m'],
            'invalid_rollout_barrier_cost': cfg['invalid_rollout_barrier_cost'],
            'optimizer_maxiter': cfg['optimizer_maxiter'],
            'optimizer_maxfun': cfg['optimizer_maxfun'],
            'optimizer_ftol': cfg['optimizer_ftol'],
            'optimizer_gtol': cfg['optimizer_gtol'],
            'optimizer_warm_start': cfg['optimizer_warm_start'],
            **cfg['camera_params'],
            **planner_params[planner],
        }],
    )

    after_odom = [
        shared_nodes['perception_node'],
        shared_nodes['pixel_to_bev'],
        shared_nodes['mission_node'],
        shared_nodes['goal_marker_node'],
    ]
    if shared_nodes['logger_node'] is not None:
        after_odom.append(shared_nodes['logger_node'])
    if cfg['use_rviz']:
        after_odom.append(shared_nodes['rviz'])

    start_after_odom = RegisterEventHandler(
        OnProcessExit(
            target_action=shared_nodes['wait_for_odom'],
            on_exit=after_odom,
        )
    )

    runtime_actions = [
        shared_nodes['bringup_sim'],
        shared_nodes['tf_static'],
    ]
    if shared_nodes.get('command_noise_node') is not None:
        runtime_actions.append(shared_nodes['command_noise_node'])
    runtime_actions.extend([
        agent_node,
        shared_nodes['wait_for_odom'],
        start_after_odom,
    ])
    return runtime_actions
