"""Shared launch helpers for the visibility-aware thesis pipeline."""

from __future__ import annotations

from typing import Dict, List

from launch.actions import IncludeLaunchDescription, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


PAPER_LAUNCH_DEFAULTS: Dict[str, str] = {
    'planner': 'efe2',
    'world': 'warehouse_occ_light.world.sdf',
    'task': 'T2_diag_demo',
    'seed': '0',
    'sensor_pixel_noise_sigma': '0.0',
    'odom_wait_timeout_s': '25.0',
    'odom_wait_min_messages': '1',
    'odom_wait_require_pose_match': 'false',
    'use_pixel_correction': 'true',
    'pixel_timeout_s': '0.5',
    'pixel_correction_min_interval_s': '0.1',
    'pixel_correction_approx': 'AUTO',
    'skip_stale_pixel_correction': 'true',
    'min_state_cov': '1e-6',
    'plan_rate': '2.0',
    'horizon': '5',
    'dt': '0.2',
    'control_weight': '0.1',
    'risk_weight_state': '0.0',
    'risk_weight_obs': '1.0',
    'ambiguity_weight': '1.0',
    'goal_sigma_uv': '0.0',
    'use_ambiguity': 'true',
    'use_obs_risk': 'true',
    'process_noise_xy': '0.01',
    'process_noise_theta': '0.02',
    'obs_noise_uv': '2.0',
    'optimizer_maxiter': '50',
    'optimizer_gtol': '1e-4',
    'optimizer_warm_start': 'true',
    'debug_runtime': 'false',
    'auto_stop_on_goal': 'true',
    'goal_success_radius': '0.35',
    'goal_success_hold_s': '2.0',
}


VISIBILITY_FALLBACK_DEFAULTS: Dict[str, object] = {
    'visibility_weight': 4.0,
    'visibility_map_min_x': -5.0,
    'visibility_map_max_x': 5.0,
    'visibility_map_min_y': -5.0,
    'visibility_map_max_y': 5.0,
    'visibility_map_nx': 140,
    'visibility_map_ny': 120,
    'visibility_occ_center_x': -1.2,
    'visibility_occ_center_y': -1.8,
    'visibility_occ_radius': 0.9,
    'visibility_occ_tau': 0.15,
    'visibility_gp_length_scale': 1.4,
    'visibility_gp_noise_var': 0.15,
    'visibility_prior_occ': 0.005,
    'visibility_beta': 1.0,
    'visibility_height_tau': 0.08,
    'visibility_ray_samples': 120,
    'visibility_sigma_kappa': 1.0,
    'visibility_target_height_m': 0.0,
    'visibility_r_bad_uv': 28.0,
    'visibility_cov_pos_scale': 2.0,
    'visibility_cov_theta_scale': 0.8,
}


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _launch_value(context, name: str, default_value: str) -> str:
    return LaunchConfiguration(name, default=default_value).perform(context)


def _is_occlusion_world(world_file: str) -> bool:
    return '_occ_' in str(world_file)


def _matches_default(current_value: object, default_value: object) -> bool:
    if isinstance(default_value, bool):
        return bool(current_value) == default_value
    if isinstance(default_value, int):
        return int(current_value) == default_value
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
        else:
            cfg[key] = float(visibility_defaults[key])


def _require_task_field(task, key):
    if key not in task:
        raise RuntimeError(f"Task is missing '{key}' field")
    return task[key]


def parse_common_launch_config(context) -> Dict[str, object]:
    """Parse the generic visibility-aware agent launch arguments."""
    seed_value = int(LaunchConfiguration('seed').perform(context))
    world_value = LaunchConfiguration('world').perform(context)
    visibility_enabled_default = 'true' if _is_occlusion_world(str(world_value)) else 'false'
    visibility_model_default = 'raycast_25d' if _is_occlusion_world(str(world_value)) else 'fixed_gp'
    use_visibility_raw = _launch_value(context, 'use_visibility_model', visibility_enabled_default).strip().lower()
    visibility_model_raw = _launch_value(context, 'visibility_model', visibility_model_default).strip().lower()

    cfg: Dict[str, object] = {
        'use_sim_time': _as_bool(LaunchConfiguration('use_sim_time').perform(context)),
        'planner': LaunchConfiguration('planner').perform(context),
        'world': world_value,
        'world_profiles_path': LaunchConfiguration('world_profiles').perform(context),
        'tasks_yaml': LaunchConfiguration('tasks_yaml').perform(context),
        'task_name': LaunchConfiguration('task').perform(context).strip(),
        'seed': seed_value,
        'sensor_pixel_noise_sigma': float(_launch_value(context, 'sensor_pixel_noise_sigma', '0.0')),
        'odom_wait_timeout_s': float(_launch_value(context, 'odom_wait_timeout_s', '25.0')),
        'odom_wait_min_messages': max(1, int(float(_launch_value(context, 'odom_wait_min_messages', '1')))),
        'odom_wait_require_pose_match': _as_bool(_launch_value(context, 'odom_wait_require_pose_match', 'false')),
        'odom_wait_position_tolerance': 0.25,
        'odom_wait_yaw_tolerance': 0.5,
        'use_pixel_correction': _as_bool(LaunchConfiguration('use_pixel_correction').perform(context)),
        'pixel_timeout_s': float(LaunchConfiguration('pixel_timeout_s').perform(context)),
        'pixel_correction_min_interval_s': float(_launch_value(context, 'pixel_correction_min_interval_s', '0.0')),
        'pixel_correction_approx': _launch_value(context, 'pixel_correction_approx', 'AUTO').strip().upper(),
        'skip_stale_pixel_correction': _as_bool(_launch_value(context, 'skip_stale_pixel_correction', 'true')),
        'use_ambiguity': _as_bool(LaunchConfiguration('use_ambiguity').perform(context)),
        'use_obs_risk': _as_bool(LaunchConfiguration('use_obs_risk').perform(context)),
        'auto_stop_on_goal': _as_bool(_launch_value(context, 'auto_stop_on_goal', 'false')),
        'goal_success_radius': float(_launch_value(context, 'goal_success_radius', '0.35')),
        'goal_success_hold_s': float(_launch_value(context, 'goal_success_hold_s', '2.0')),
        'process_noise_xy': float(LaunchConfiguration('process_noise_xy').perform(context)),
        'process_noise_theta': float(LaunchConfiguration('process_noise_theta').perform(context)),
        'obs_noise_uv': float(LaunchConfiguration('obs_noise_uv').perform(context)),
        'optimizer_backend': str(LaunchConfiguration('optimizer_backend').perform(context)).strip().lower(),
        'optimizer_maxiter': int(LaunchConfiguration('optimizer_maxiter').perform(context)),
        'optimizer_gtol': float(LaunchConfiguration('optimizer_gtol').perform(context)),
        'optimizer_warm_start': _as_bool(LaunchConfiguration('optimizer_warm_start').perform(context)),
        'plan_rate': float(_launch_value(context, 'plan_rate', '1.0')),
        'horizon': int(_launch_value(context, 'horizon', '10')),
        'dt': float(_launch_value(context, 'dt', '0.2')),
        'control_weight': float(_launch_value(context, 'control_weight', '0.1')),
        'risk_weight_state': float(_launch_value(context, 'risk_weight_state', '1.0')),
        'risk_weight_obs': float(_launch_value(context, 'risk_weight_obs', '1.0')),
        'ambiguity_weight': float(_launch_value(context, 'ambiguity_weight', '1.0')),
        'use_visibility_model': _as_bool(
            visibility_enabled_default if use_visibility_raw in ('', 'auto', 'default') else use_visibility_raw
        ),
        'visibility_model': visibility_model_default if visibility_model_raw in ('', 'auto', 'default') else visibility_model_raw,
        'visibility_weight': float(_launch_value(context, 'visibility_weight', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_weight']))),
        'visibility_map_min_x': float(_launch_value(context, 'visibility_map_min_x', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_map_min_x']))),
        'visibility_map_max_x': float(_launch_value(context, 'visibility_map_max_x', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_map_max_x']))),
        'visibility_map_min_y': float(_launch_value(context, 'visibility_map_min_y', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_map_min_y']))),
        'visibility_map_max_y': float(_launch_value(context, 'visibility_map_max_y', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_map_max_y']))),
        'visibility_map_nx': int(_launch_value(context, 'visibility_map_nx', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_map_nx']))),
        'visibility_map_ny': int(_launch_value(context, 'visibility_map_ny', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_map_ny']))),
        'visibility_occ_center_x': float(_launch_value(context, 'visibility_occ_center_x', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_occ_center_x']))),
        'visibility_occ_center_y': float(_launch_value(context, 'visibility_occ_center_y', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_occ_center_y']))),
        'visibility_occ_radius': float(_launch_value(context, 'visibility_occ_radius', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_occ_radius']))),
        'visibility_occ_tau': float(_launch_value(context, 'visibility_occ_tau', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_occ_tau']))),
        'visibility_gp_length_scale': float(_launch_value(context, 'visibility_gp_length_scale', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_gp_length_scale']))),
        'visibility_gp_noise_var': float(_launch_value(context, 'visibility_gp_noise_var', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_gp_noise_var']))),
        'visibility_prior_occ': float(_launch_value(context, 'visibility_prior_occ', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_prior_occ']))),
        'visibility_beta': float(_launch_value(context, 'visibility_beta', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_beta']))),
        'visibility_height_tau': float(_launch_value(context, 'visibility_height_tau', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_height_tau']))),
        'visibility_ray_samples': int(float(_launch_value(context, 'visibility_ray_samples', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_ray_samples'])))),
        'visibility_sigma_kappa': float(_launch_value(context, 'visibility_sigma_kappa', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_sigma_kappa']))),
        'visibility_target_height_m': float(_launch_value(context, 'visibility_target_height_m', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_target_height_m']))),
        'visibility_geometry_json': _launch_value(context, 'visibility_geometry_json', ''),
        'visibility_gp_seed': int(float(_launch_value(context, 'visibility_gp_seed', str(seed_value)))),
        'visibility_r_bad_uv': float(_launch_value(context, 'visibility_r_bad_uv', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_r_bad_uv']))),
        'visibility_cov_pos_scale': float(_launch_value(context, 'visibility_cov_pos_scale', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_cov_pos_scale']))),
        'visibility_cov_theta_scale': float(_launch_value(context, 'visibility_cov_theta_scale', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_cov_theta_scale']))),
        'goal_sigma_uv': float(_launch_value(context, 'goal_sigma_uv', '0.0')),
        'min_state_cov': float(_launch_value(context, 'min_state_cov', '1e-6')),
        'debug_runtime': _as_bool(_launch_value(context, 'debug_runtime', 'false')),
        'use_rviz': _as_bool(LaunchConfiguration('use_rviz').perform(context)),
        'rviz_config': LaunchConfiguration('rviz_config').perform(context),
    }

    if cfg['optimizer_backend'] not in ('auto', 'jax'):
        raise RuntimeError("optimizer_backend must be 'auto' or 'jax'")

    return cfg


def resolve_world_setup(cfg: Dict[str, object]) -> Dict[str, object]:
    """Resolve world profile/task and derive camera/spawn launch parameters."""
    from experiments.core.world_profiles import (
        load_profile,
        compute_camera_quaternion_from_rpy,
        compute_look_at_from_pose,
        serialize_occlusion_geometry_from_world,
    )
    from experiments.core.tasks import load_tasks, select_task

    profile, _intrinsics, world_path, camera_pose = load_profile(
        cfg['world_profiles_path'], cfg['world']
    )
    _apply_visibility_profile_defaults(cfg, profile)
    tasks_by_world = load_tasks(cfg['tasks_yaml'])
    task = select_task(tasks_by_world, cfg['world'], cfg['task_name'])

    start = _require_task_field(task, 'start')
    goal = _require_task_field(task, 'goal')
    for key in ('x', 'y', 'z', 'yaw'):
        if key not in start:
            raise RuntimeError(f"Task start missing '{key}'")
    for key in ('x', 'y'):
        if key not in goal:
            raise RuntimeError(f"Task goal missing '{key}'")

    spawn = {
        'x': float(start['x']),
        'y': float(start['y']),
        'z': float(start['z']),
        'yaw': float(start['yaw']),
    }
    goal_x = float(goal['x'])
    goal_y = float(goal['y'])

    planner = str(cfg['planner'])
    if planner == 'auto':
        planner = profile['planner_default']

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
        'use_sim_time': 'true' if cfg['use_sim_time'] else 'false',
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
    if (not visibility_geometry_json) and str(cfg.get('visibility_model', '')).lower() == 'raycast_25d':
        visibility_geometry_json = serialize_occlusion_geometry_from_world(world_path)

    cfg = dict(cfg)
    cfg.update({
        'profile': profile,
        'task': task,
        'planner': planner,
        'spawn': spawn,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'camera_params': camera_params,
        'tf_args': tf_args,
        'world_path': world_path,
        'visibility_geometry_json': visibility_geometry_json,
    })
    return cfg


def build_shared_nodes(cfg: Dict[str, object]) -> Dict[str, object]:
    """Create shared nodes/components for the thesis pipeline."""
    sim_pkg = FindPackageShare('sim')
    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': 'true' if cfg['use_sim_time'] else 'false',
            'use_lidar': 'false',
            'bridge_scan': 'false',
            'world': cfg['world'],
            'world_name': cfg['profile']['world_name'],
            'spawn_x': str(cfg['spawn']['x']),
            'spawn_y': str(cfg['spawn']['y']),
            'spawn_z': str(cfg['spawn']['z']),
            'spawn_yaw': str(cfg['spawn']['yaw']),
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

    homography_params = {'use_sim_time': cfg['use_sim_time']}
    homography_params.update(cfg['camera_params'])
    homography_params.update({
        'pixel_noise_sigma': cfg['sensor_pixel_noise_sigma'],
        'seed': cfg['seed'],
        'world_frame': 'map_bev',
        'use_visibility_model': cfg['use_visibility_model'],
        'visibility_model': cfg['visibility_model'],
        'visibility_geometry_json': cfg['visibility_geometry_json'],
        'visibility_target_height_m': cfg['visibility_target_height_m'],
    })
    homography_sim = Node(
        package='perception',
        executable='homography_sim_node',
        name='homography_sim_node',
        output='screen',
        parameters=[homography_params],
    )

    pixel_params = {
        'use_sim_time': cfg['use_sim_time'],
        'frame_id': 'map_bev',
        'pixel_noise_sigma': 0.0,
        'transform_noise_sigma': 0.0,
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
            'use_sim_time': cfg['use_sim_time'],
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
            'use_sim_time': cfg['use_sim_time'],
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

    logger_node = Node(
        package='experiments',
        executable='experiment_logger',
        name='experiment_logger',
        output='screen',
        on_exit=[Shutdown(reason='experiment_logger exited')],
        parameters=[{
            'use_sim_time': cfg['use_sim_time'],
            'seed': cfg['seed'],
            'world': cfg['world'],
            'task': cfg['task'].get('name', cfg['task_name'] or ''),
            'planner': cfg['planner'],
            'use_pixel_correction': cfg['use_pixel_correction'],
            'use_ambiguity': cfg['use_ambiguity'],
            'use_obs_risk': cfg['use_obs_risk'],
            'use_visibility_model': cfg['use_visibility_model'],
            'visibility_model': cfg['visibility_model'],
            'visibility_target_height_m': cfg['visibility_target_height_m'],
            'world_profiles_path': cfg['world_profiles_path'],
            'tasks_yaml': cfg['tasks_yaml'],
            'auto_stop_on_goal': cfg['auto_stop_on_goal'],
            'goal_success_radius': cfg['goal_success_radius'],
            'goal_success_hold_s': cfg['goal_success_hold_s'],
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', cfg['rviz_config']],
        output='screen',
        parameters=[{'use_sim_time': cfg['use_sim_time']}],
    )

    return {
        'bringup_sim': bringup_sim,
        'tf_static': tf_static,
        'wait_for_odom': wait_for_odom,
        'homography_sim': homography_sim,
        'pixel_to_bev': pixel_to_bev,
        'mission_node': mission_node,
        'goal_marker_node': goal_marker_node,
        'logger_node': logger_node,
        'rviz': rviz,
    }


def build_agent_runtime_actions(cfg: Dict[str, object]) -> List[object]:
    """Create runtime actions for the visibility-aware agent launch."""
    shared_nodes = build_shared_nodes(cfg)
    planner = cfg['planner']

    if planner not in ('efe1', 'efe2', 'mpc', 'efer'):
        raise RuntimeError("planner must be 'efe1', 'efe2', 'mpc', or 'efer' for agent launch")

    planner_params = {
        'efe1': {
            'approx_method': 'ET1',
            'use_ambiguity': cfg['use_ambiguity'],
            'use_obs_risk': cfg['use_obs_risk'],
        },
        'efe2': {
            'approx_method': 'ET2',
            'use_ambiguity': cfg['use_ambiguity'],
            'use_obs_risk': cfg['use_obs_risk'],
        },
        'mpc': {
            'approx_method': 'ET1',
            'use_ambiguity': False,
            'use_obs_risk': True,
        },
        'efer': {
            'approx_method': 'ET2',
            'use_ambiguity': False,
            'use_obs_risk': True,
        },
    }
    agent_node = Node(
        package='planning',
        executable='efe_agent',
        name=f'{planner}_agent',
        output='screen',
        additional_env={'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'},
        parameters=[{
            'use_sim_time': cfg['use_sim_time'],
            'plan_rate': cfg['plan_rate'],
            'horizon': cfg['horizon'],
            'dt': cfg['dt'],
            'control_weight': cfg['control_weight'],
            'use_pixel_correction': cfg['use_pixel_correction'],
            'pixel_timeout_s': cfg['pixel_timeout_s'],
            'pixel_correction_min_interval_s': cfg['pixel_correction_min_interval_s'],
            'pixel_correction_approx': cfg['pixel_correction_approx'],
            'skip_stale_pixel_correction': cfg['skip_stale_pixel_correction'],
            'min_state_cov': cfg['min_state_cov'],
            'debug_runtime': cfg['debug_runtime'],
            'process_noise_xy': cfg['process_noise_xy'],
            'process_noise_theta': cfg['process_noise_theta'],
            'obs_noise_uv': cfg['obs_noise_uv'],
            'goal_sigma_uv': cfg['goal_sigma_uv'],
            'risk_weight_state': cfg['risk_weight_state'],
            'risk_weight_obs': cfg['risk_weight_obs'],
            'ambiguity_weight': cfg['ambiguity_weight'],
            'use_visibility_model': cfg['use_visibility_model'],
            'visibility_model': cfg['visibility_model'],
            'visibility_weight': cfg['visibility_weight'],
            'visibility_map_min_x': cfg['visibility_map_min_x'],
            'visibility_map_max_x': cfg['visibility_map_max_x'],
            'visibility_map_min_y': cfg['visibility_map_min_y'],
            'visibility_map_max_y': cfg['visibility_map_max_y'],
            'visibility_map_nx': cfg['visibility_map_nx'],
            'visibility_map_ny': cfg['visibility_map_ny'],
            'visibility_occ_center_x': cfg['visibility_occ_center_x'],
            'visibility_occ_center_y': cfg['visibility_occ_center_y'],
            'visibility_occ_radius': cfg['visibility_occ_radius'],
            'visibility_occ_tau': cfg['visibility_occ_tau'],
            'visibility_gp_length_scale': cfg['visibility_gp_length_scale'],
            'visibility_gp_noise_var': cfg['visibility_gp_noise_var'],
            'visibility_prior_occ': cfg['visibility_prior_occ'],
            'visibility_beta': cfg['visibility_beta'],
            'visibility_height_tau': cfg['visibility_height_tau'],
            'visibility_ray_samples': cfg['visibility_ray_samples'],
            'visibility_sigma_kappa': cfg['visibility_sigma_kappa'],
            'visibility_target_height_m': cfg['visibility_target_height_m'],
            'visibility_geometry_json': cfg['visibility_geometry_json'],
            'visibility_gp_seed': cfg['visibility_gp_seed'],
            'visibility_r_bad_uv': cfg['visibility_r_bad_uv'],
            'visibility_cov_pos_scale': cfg['visibility_cov_pos_scale'],
            'visibility_cov_theta_scale': cfg['visibility_cov_theta_scale'],
            'optimizer_backend': cfg['optimizer_backend'],
            'optimizer_maxiter': cfg['optimizer_maxiter'],
            'optimizer_gtol': cfg['optimizer_gtol'],
            'optimizer_warm_start': cfg['optimizer_warm_start'],
            'jax_warmup_use_goal_hint': True,
            'jax_warmup_goal_x': cfg['goal_x'],
            'jax_warmup_goal_y': cfg['goal_y'],
            'jax_warmup_goal_frame_id': 'map_bev',
            **cfg['camera_params'],
            **planner_params[planner],
        }],
    )

    after_odom = [
        shared_nodes['homography_sim'],
        shared_nodes['pixel_to_bev'],
        agent_node,
        shared_nodes['mission_node'],
        shared_nodes['goal_marker_node'],
        shared_nodes['logger_node'],
    ]
    if cfg['use_rviz']:
        after_odom.append(shared_nodes['rviz'])

    start_after_odom = RegisterEventHandler(
        OnProcessExit(
            target_action=shared_nodes['wait_for_odom'],
            on_exit=after_odom,
        )
    )

    return [
        shared_nodes['bringup_sim'],
        shared_nodes['tf_static'],
        shared_nodes['wait_for_odom'],
        start_after_odom,
    ]
