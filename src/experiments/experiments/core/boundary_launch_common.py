"""Shared launch helpers for boundary-only experiment pipelines.

This module intentionally centralizes repetitive setup mechanics while keeping
top-level launch files readable and mode-specific.
"""

from __future__ import annotations

from typing import Dict, List

from launch.actions import IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Paper-facing defaults used by `investigative_agent.launch.py`.
# Node-level defaults remain as development fallbacks; paper experiments should
# be reproducible from launch arguments and manifests.
PAPER_LAUNCH_DEFAULTS: Dict[str, str] = {
    'planner': 'efe2',
    'world': 'warehouse_occ_light.world.sdf',
    'task': 'T2_diag_far',
    'state_source': 'pixel',
    'perception_backend': 'color',
    'seed': '0',
    'sensor_pixel_noise_sigma': '0.0',
    'pixel_noise_sigma': '0.0',
    'transform_noise_sigma': '0.0',
    'odom_wait_timeout_s': '25.0',
    'odom_wait_min_messages': '1',
    'odom_wait_require_pose_match': 'false',
    'use_pixel_correction': 'true',
    'pixel_timeout_s': '0.5',
    'pixel_correction_approx': 'AUTO',
    'skip_stale_pixel_correction': 'true',
    'min_state_cov': '1e-6',
    'obs_mode': 'uv',
    'plan_rate': '2.0',
    'horizon': '5',
    'dt': '0.2',
    'control_weight': '0.1',
    'risk_weight_state': '0.0',
    'risk_weight_obs': '1.0',
    'ambiguity_weight': '1.0',
    'goal_sigma_uv': '0.0',
    'goal_sigma_yaw': '100.0',
    'use_ambiguity': 'true',
    'use_obs_risk': 'true',
    'process_noise_xy': '0.01',
    'process_noise_theta': '0.02',
    'obs_noise_uv': '2.0',
    'obs_noise_yaw': '0.05',
    'optimizer_maxiter': '50',
    'optimizer_gtol': '1e-4',
    'optimizer_warm_start': 'true',
    'debug_runtime': 'false',
    'aruco_dict': 'DICT_4X4_50',
    'target_marker_id': '0',
    'publish_yaw_from_marker': 'true',
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
}


def _as_bool(value: str) -> bool:
    return str(value).lower() == 'true'


def _launch_value(context, name: str, default_value: str) -> str:
    """Read launch arg with a local default (works even if arg isn't declared)."""
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
    """Parse and validate common launch arguments used by both pipelines."""
    seed_value = int(LaunchConfiguration('seed').perform(context))
    cfg: Dict[str, object] = {
        'use_sim_time': _as_bool(LaunchConfiguration('use_sim_time').perform(context)),
        'state_source': LaunchConfiguration('state_source').perform(context),
        'planner': LaunchConfiguration('planner').perform(context),
        'world': LaunchConfiguration('world').perform(context),
        'world_profiles_path': LaunchConfiguration('world_profiles').perform(context),
        'tasks_yaml': LaunchConfiguration('tasks_yaml').perform(context),
        'task_name': LaunchConfiguration('task').perform(context).strip(),
        'perception_backend': LaunchConfiguration('perception_backend').perform(context).strip().lower(),
        'seed': seed_value,
        'pixel_noise_sigma': float(LaunchConfiguration('pixel_noise_sigma').perform(context)),
        'transform_noise_sigma': float(LaunchConfiguration('transform_noise_sigma').perform(context)),
        'sensor_pixel_noise_sigma': float(_launch_value(context, 'sensor_pixel_noise_sigma', '0.0')),
        'odom_wait_timeout_s': float(_launch_value(context, 'odom_wait_timeout_s', '25.0')),
        'odom_wait_min_messages': max(1, int(float(_launch_value(context, 'odom_wait_min_messages', '1')))),
        'odom_wait_require_pose_match': _as_bool(_launch_value(context, 'odom_wait_require_pose_match', 'false')),
        # Keep startup pose-match tolerances fixed to reduce launch-surface clutter.
        'odom_wait_position_tolerance': 0.25,
        'odom_wait_yaw_tolerance': 0.5,
        'use_pixel_correction': _as_bool(LaunchConfiguration('use_pixel_correction').perform(context)),
        'pixel_timeout_s': float(LaunchConfiguration('pixel_timeout_s').perform(context)),
        'pixel_correction_approx': _launch_value(context, 'pixel_correction_approx', 'AUTO').strip().upper(),
        'skip_stale_pixel_correction': _as_bool(_launch_value(context, 'skip_stale_pixel_correction', 'true')),
        'use_ambiguity': _as_bool(LaunchConfiguration('use_ambiguity').perform(context)),
        'use_obs_risk': _as_bool(LaunchConfiguration('use_obs_risk').perform(context)),
        'boundary_weight': float(LaunchConfiguration('boundary_weight').perform(context)),
        'publish_static_costmap': _as_bool(_launch_value(context, 'publish_static_costmap', 'true')),
        'auto_stop_on_goal': _as_bool(_launch_value(context, 'auto_stop_on_goal', 'false')),
        'goal_success_radius': float(_launch_value(context, 'goal_success_radius', '0.35')),
        'goal_success_hold_s': float(_launch_value(context, 'goal_success_hold_s', '2.0')),
        'costmap_min_x': float(LaunchConfiguration('costmap_min_x').perform(context)),
        'costmap_max_x': float(LaunchConfiguration('costmap_max_x').perform(context)),
        'costmap_min_y': float(LaunchConfiguration('costmap_min_y').perform(context)),
        'costmap_max_y': float(LaunchConfiguration('costmap_max_y').perform(context)),
        'costmap_wall_margin': float(LaunchConfiguration('costmap_wall_margin').perform(context)),
        'costmap_obstacle_enabled': _as_bool(LaunchConfiguration('costmap_obstacle_enabled').perform(context)),
        'costmap_obstacle_center_x': float(LaunchConfiguration('costmap_obstacle_center_x').perform(context)),
        'costmap_obstacle_center_y': float(LaunchConfiguration('costmap_obstacle_center_y').perform(context)),
        'costmap_obstacle_radius': float(LaunchConfiguration('costmap_obstacle_radius').perform(context)),
        'costmap_obstacle_value': int(LaunchConfiguration('costmap_obstacle_value').perform(context)),
        'obs_mode': LaunchConfiguration('obs_mode').perform(context),
        'process_noise_xy': float(LaunchConfiguration('process_noise_xy').perform(context)),
        'process_noise_theta': float(LaunchConfiguration('process_noise_theta').perform(context)),
        'obs_noise_uv': float(LaunchConfiguration('obs_noise_uv').perform(context)),
        'obs_noise_yaw': float(LaunchConfiguration('obs_noise_yaw').perform(context)),
        'optimizer_backend': LaunchConfiguration('optimizer_backend').perform(context),
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
        'use_visibility_model': _as_bool(_launch_value(context, 'use_visibility_model', 'false')),
        'visibility_model': _launch_value(context, 'visibility_model', 'fixed_gp').strip().lower(),
        'visibility_weight': float(_launch_value(context, 'visibility_weight', '4.0')),
        'visibility_map_min_x': float(_launch_value(context, 'visibility_map_min_x', '-5.0')),
        'visibility_map_max_x': float(_launch_value(context, 'visibility_map_max_x', '5.0')),
        'visibility_map_min_y': float(_launch_value(context, 'visibility_map_min_y', '-5.0')),
        'visibility_map_max_y': float(_launch_value(context, 'visibility_map_max_y', '5.0')),
        'visibility_map_nx': int(_launch_value(context, 'visibility_map_nx', '140')),
        'visibility_map_ny': int(_launch_value(context, 'visibility_map_ny', '120')),
        'visibility_occ_center_x': float(_launch_value(context, 'visibility_occ_center_x', '-1.2')),
        'visibility_occ_center_y': float(_launch_value(context, 'visibility_occ_center_y', '-1.8')),
        'visibility_occ_radius': float(_launch_value(context, 'visibility_occ_radius', '0.9')),
        'visibility_occ_tau': float(_launch_value(context, 'visibility_occ_tau', '0.15')),
        'visibility_gp_length_scale': float(_launch_value(context, 'visibility_gp_length_scale', '1.4')),
        'visibility_gp_noise_var': float(_launch_value(context, 'visibility_gp_noise_var', '0.15')),
        'visibility_gp_seed': int(float(_launch_value(context, 'visibility_gp_seed', str(seed_value)))),
        'visibility_r_bad_uv': float(_launch_value(context, 'visibility_r_bad_uv', '28.0')),
        'visibility_r_bad_yaw': float(_launch_value(context, 'visibility_r_bad_yaw', '1.2')),
        'visibility_cov_pos_scale': float(_launch_value(context, 'visibility_cov_pos_scale', '2.0')),
        'visibility_cov_theta_scale': float(_launch_value(context, 'visibility_cov_theta_scale', '0.8')),
        'goal_sigma_uv': float(_launch_value(context, 'goal_sigma_uv', '0.0')),
        'goal_sigma_yaw': float(_launch_value(context, 'goal_sigma_yaw', '0.0')),
        'min_state_cov': float(_launch_value(context, 'min_state_cov', '1e-6')),
        'debug_runtime': _as_bool(_launch_value(context, 'debug_runtime', 'false')),
        'use_rviz': _as_bool(LaunchConfiguration('use_rviz').perform(context)),
        'rviz_config': LaunchConfiguration('rviz_config').perform(context),
        'aruco_dict': LaunchConfiguration('aruco_dict').perform(context),
        'target_marker_id': int(LaunchConfiguration('target_marker_id').perform(context)),
        'publish_yaw_from_marker': _as_bool(LaunchConfiguration('publish_yaw_from_marker').perform(context)),
    }

    if cfg['state_source'] not in ('oracle', 'pixel'):
        raise RuntimeError("state_source must be 'oracle' or 'pixel'")
    if cfg['perception_backend'] not in ('homography', 'aruco', 'color'):
        raise RuntimeError("perception_backend must be 'homography', 'aruco', or 'color'")
    if cfg['state_source'] == 'pixel' and cfg['perception_backend'] not in ('homography', 'aruco', 'color'):
        raise RuntimeError("state_source='pixel' requires perception_backend 'homography', 'aruco', or 'color'")

    # Occlusion benchmark worlds are visual-only stress tests; force-disable
    # obstacle-costmap behavior so agents don't treat this as obstacle avoidance.
    if _is_occlusion_world(str(cfg['world'])):
        cfg['boundary_weight'] = 0.0
        cfg['publish_static_costmap'] = False
        cfg['costmap_obstacle_enabled'] = False

    return cfg


def resolve_world_setup(cfg: Dict[str, object]) -> Dict[str, object]:
    """Resolve world profile/task and derive camera/spawn launch parameters."""
    from experiments.core.world_profiles import (
        load_profile,
        compute_camera_quaternion_from_rpy,
        compute_look_at_from_pose,
    )
    from experiments.core.tasks import load_tasks, select_task

    profile, _intrinsics, _world_path, camera_pose = load_profile(
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
    })
    return cfg


def build_shared_nodes(cfg: Dict[str, object]) -> Dict[str, object]:
    """Create shared nodes/components common to planner and agent launches."""
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
    })
    homography_sim = Node(
        package='perception',
        executable='homography_sim_node',
        name='homography_sim_node',
        output='screen',
        parameters=[homography_params],
    )
    aruco_detector = Node(
        package='perception',
        executable='aruco_detector_node',
        name='aruco_detector_node',
        output='screen',
        parameters=[{
            'use_sim_time': cfg['use_sim_time'],
            'aruco_dict': cfg['aruco_dict'],
            'target_marker_id': cfg['target_marker_id'],
            'publish_yaw_from_marker': cfg['publish_yaw_from_marker'],
            **cfg['camera_params'],
        }],
    )
    color_pose_detector = Node(
        package='perception',
        executable='color_pose_detector_node',
        name='color_pose_detector_node',
        output='screen',
        parameters=[{
            'use_sim_time': cfg['use_sim_time'],
            **cfg['camera_params'],
        }],
    )

    pixel_params = {
        'use_sim_time': cfg['use_sim_time'],
        'state_source': cfg['state_source'],
        'frame_id': 'map_bev',
        'pixel_noise_sigma': cfg['pixel_noise_sigma'],
        'transform_noise_sigma': cfg['transform_noise_sigma'],
        'seed': cfg['seed'],
    }
    pixel_params.update(cfg['camera_params'])
    pixel_to_bev = Node(
        package='state',
        executable='pixel_to_bev_state_node',
        name='pixel_to_bev_state_node',
        output='screen',
        parameters=[pixel_params],
    )

    boundary_cost_node = Node(
        package='experiments',
        executable='boundary_cost_node',
        name='boundary_cost_node',
        output='screen',
        parameters=[{
            'use_sim_time': cfg['use_sim_time'],
            'frame_id': 'map_bev',
            'min_x': cfg['costmap_min_x'],
            'max_x': cfg['costmap_max_x'],
            'min_y': cfg['costmap_min_y'],
            'max_y': cfg['costmap_max_y'],
            'wall_margin': cfg['costmap_wall_margin'],
            'obstacle_enabled': cfg['costmap_obstacle_enabled'],
            'obstacle_center_x': cfg['costmap_obstacle_center_x'],
            'obstacle_center_y': cfg['costmap_obstacle_center_y'],
            'obstacle_radius': cfg['costmap_obstacle_radius'],
            'obstacle_value': cfg['costmap_obstacle_value'],
        }],
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
            'state_source': cfg['state_source'],
            'perception_backend': cfg['perception_backend'],
            'obs_model': 'camera',
            'obs_mode': cfg['obs_mode'],
            'use_pixel_correction': cfg['use_pixel_correction'],
            'boundary_weight': cfg['boundary_weight'],
            'publish_static_costmap': cfg['publish_static_costmap'],
            # Keep legacy manifest field for compatibility; it mirrors the single switch.
            'add_ambiguity': cfg['use_ambiguity'],
            'use_ambiguity': cfg['use_ambiguity'],
            'use_obs_risk': cfg['use_obs_risk'],
            'pixel_noise_sigma': cfg['pixel_noise_sigma'],
            'transform_noise_sigma': cfg['transform_noise_sigma'],
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
        'aruco_detector': aruco_detector,
        'color_pose_detector': color_pose_detector,
        'pixel_to_bev': pixel_to_bev,
        'boundary_cost_node': boundary_cost_node,
        'mission_node': mission_node,
        'logger_node': logger_node,
        'rviz': rviz,
    }


def select_perception_nodes_for_mode(cfg: Dict[str, object], shared: Dict[str, object]) -> List[object]:
    """Return the perception nodes needed before the core pipeline starts."""
    if cfg['state_source'] == 'pixel':
        if cfg['perception_backend'] == 'homography':
            return [shared['homography_sim']]
        if cfg['perception_backend'] == 'color':
            return [shared['color_pose_detector']]
        return [shared['aruco_detector']]
    return []
