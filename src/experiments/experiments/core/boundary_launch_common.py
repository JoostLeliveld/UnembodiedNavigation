"""Shared launch helpers for boundary-only experiment pipelines.

This module intentionally centralizes repetitive setup mechanics while keeping
top-level launch files readable and mode-specific.
"""

from __future__ import annotations

from typing import Dict, List

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _as_bool(value: str) -> bool:
    return str(value).lower() == 'true'


def _launch_value(context, name: str, default_value: str) -> str:
    """Read launch arg with a local default (works even if arg isn't declared)."""
    return LaunchConfiguration(name, default=default_value).perform(context)


def _require_task_field(task, key):
    if key not in task:
        raise RuntimeError(f"Task is missing '{key}' field")
    return task[key]


def parse_common_launch_config(context) -> Dict[str, object]:
    """Parse and validate common launch arguments used by both pipelines."""
    cfg: Dict[str, object] = {
        'use_sim_time': _as_bool(LaunchConfiguration('use_sim_time').perform(context)),
        'state_source': LaunchConfiguration('state_source').perform(context),
        'planner': LaunchConfiguration('planner').perform(context),
        'world': LaunchConfiguration('world').perform(context),
        'world_profiles_path': LaunchConfiguration('world_profiles').perform(context),
        'tasks_yaml': LaunchConfiguration('tasks_yaml').perform(context),
        'task_name': LaunchConfiguration('task').perform(context).strip(),
        'perception_backend': LaunchConfiguration('perception_backend').perform(context).strip().lower(),
        'seed': int(LaunchConfiguration('seed').perform(context)),
        'pixel_noise_sigma': float(LaunchConfiguration('pixel_noise_sigma').perform(context)),
        'transform_noise_sigma': float(LaunchConfiguration('transform_noise_sigma').perform(context)),
        'sensor_pixel_noise_sigma': float(_launch_value(context, 'sensor_pixel_noise_sigma', '0.0')),
        'odom_wait_timeout_s': float(_launch_value(context, 'odom_wait_timeout_s', '25.0')),
        'odom_wait_min_messages': max(1, int(float(_launch_value(context, 'odom_wait_min_messages', '1')))),
        'odom_wait_require_pose_match': _as_bool(_launch_value(context, 'odom_wait_require_pose_match', 'false')),
        'odom_wait_position_tolerance': float(_launch_value(context, 'odom_wait_position_tolerance', '0.25')),
        'odom_wait_yaw_tolerance': float(_launch_value(context, 'odom_wait_yaw_tolerance', '0.5')),
        'use_pixel_correction': _as_bool(LaunchConfiguration('use_pixel_correction').perform(context)),
        'pixel_timeout_s': float(LaunchConfiguration('pixel_timeout_s').perform(context)),
        'add_ambiguity': _as_bool(LaunchConfiguration('add_ambiguity').perform(context)),
        'use_ambiguity': _as_bool(LaunchConfiguration('use_ambiguity').perform(context)),
        'use_obs_risk': _as_bool(LaunchConfiguration('use_obs_risk').perform(context)),
        'boundary_weight': float(LaunchConfiguration('boundary_weight').perform(context)),
        'publish_static_costmap': _as_bool(_launch_value(context, 'publish_static_costmap', 'true')),
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
        'goal_sigma_uv': float(_launch_value(context, 'goal_sigma_uv', '0.0')),
        'goal_sigma_yaw': float(_launch_value(context, 'goal_sigma_yaw', '0.0')),
        'min_state_cov': float(_launch_value(context, 'min_state_cov', '1e-6')),
        'use_rviz': _as_bool(LaunchConfiguration('use_rviz').perform(context)),
        'rviz_config': LaunchConfiguration('rviz_config').perform(context),
        'aruco_dict': LaunchConfiguration('aruco_dict').perform(context),
        'target_marker_id': int(LaunchConfiguration('target_marker_id').perform(context)),
        'publish_yaw_from_marker': _as_bool(LaunchConfiguration('publish_yaw_from_marker').perform(context)),
    }

    if cfg['state_source'] not in ('oracle', 'pixel'):
        raise RuntimeError("state_source must be 'oracle' or 'pixel'")
    if cfg['perception_backend'] not in ('homography', 'aruco'):
        raise RuntimeError("perception_backend must be 'homography' or 'aruco'")
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
        parameters=[{
            'use_sim_time': cfg['use_sim_time'],
            'seed': cfg['seed'],
            'world': cfg['world'],
            'task': cfg['task'].get('name', cfg['task_name'] or ''),
            'planner': cfg['planner'],
            'state_source': cfg['state_source'],
            'perception_backend': cfg['perception_backend'],
            'obs_mode': cfg['obs_mode'],
            'use_pixel_correction': cfg['use_pixel_correction'],
            'boundary_weight': cfg['boundary_weight'],
            'publish_static_costmap': cfg['publish_static_costmap'],
            'add_ambiguity': cfg['add_ambiguity'],
            'use_ambiguity': cfg['use_ambiguity'],
            'use_obs_risk': cfg['use_obs_risk'],
            'pixel_noise_sigma': cfg['pixel_noise_sigma'],
            'transform_noise_sigma': cfg['transform_noise_sigma'],
            'world_profiles_path': cfg['world_profiles_path'],
            'tasks_yaml': cfg['tasks_yaml'],
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
        'pixel_to_bev': pixel_to_bev,
        'boundary_cost_node': boundary_cost_node,
        'mission_node': mission_node,
        'logger_node': logger_node,
        'rviz': rviz,
    }


def select_perception_nodes_for_mode(cfg: Dict[str, object], shared: Dict[str, object]) -> List[object]:
    """Return the perception nodes needed before the core pipeline starts."""
    if cfg['state_source'] != 'pixel':
        return []
    if cfg['perception_backend'] == 'homography':
        return [shared['homography_sim']]
    return [shared['aruco_detector']]
