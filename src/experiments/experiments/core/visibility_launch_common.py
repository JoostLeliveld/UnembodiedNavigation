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
    'planner': 'efe1',
    'world': 'warehouse_occ_light.world.sdf',
    'task': '',
    'seed': '0',
    'perception_backend': 'image_markers',
    'sensor_pixel_noise_sigma': '1.0',
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
    'horizon': '24',
    'dt': '0.2',
    'control_weight': '0.0',
    'risk_weight_obs': '1.0',
    'ambiguity_weight': '1.0',
    'goal_sigma_uv': '2.0',
    'use_ambiguity': 'true',
    'use_obs_risk': 'true',
    'r_visible_uv': '2.5',
    'r_miss_uv': '140.0',
    'visibility_power': '3.0',
    'visibility_sigma_kappa': '1.0',
    'goal_prior_u_std_start': '80.0',
    'goal_prior_v_std_start': '80.0',
    'goal_prior_u_std_final': '18.0',
    'goal_prior_v_std_final': '18.0',
    'goal_tightening_power': '0.45',
    'goal_progress_n_steps': '90',
    'observation_risk_scale': '1.25',
    'ambiguity_term_scale': '1.00',
    'visibility_weight': '12.0',
    'visibility_barrier_threshold': '0.90',
    'visibility_barrier_scale': '25.0',
    'process_noise_xy': '0.01',
    'process_noise_theta': '0.02',
    'obs_noise_uv': '2.0',
    'optimizer_maxiter': '80',
    'optimizer_maxfun': '500',
    'optimizer_ftol': '1e-6',
    'optimizer_gtol': '1e-4',
    'optimizer_warm_start': 'true',
    'debug_runtime': 'false',
    'auto_stop_on_goal': 'true',
    'goal_success_radius': '0.35',
    'goal_success_hold_s': '2.0',
}


VISIBILITY_FALLBACK_DEFAULTS: Dict[str, object] = {
    'visibility_weight': 4.0,
    'visibility_barrier_threshold': 0.0,
    'visibility_barrier_scale': 10.0,
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


def _state_estimator_metadata(perception_backend: str) -> Dict[str, str]:
    if str(perception_backend).strip().lower() == 'homography':
        return {
            'state_source_x': 'camera_homography',
            'state_source_y': 'camera_homography',
            'state_source_theta': 'visual_heading_else_odom',
            'state_estimator_mode': 'camera_xytheta_with_odom_fallback',
        }
    return {
        'state_source_x': 'camera_homography',
        'state_source_y': 'camera_homography',
        'state_source_theta': 'odometry_heading',
        'state_estimator_mode': 'camera_xy_odom_theta',
    }


def parse_common_launch_config(context) -> Dict[str, object]:
    """Parse the generic visibility-aware agent launch arguments."""
    seed_value = int(LaunchConfiguration('seed').perform(context))
    world_value = LaunchConfiguration('world').perform(context)
    visibility_enabled_default = 'true'
    use_visibility_raw = _launch_value(context, 'use_visibility_model', visibility_enabled_default).strip().lower()

    cfg: Dict[str, object] = {
        'use_sim_time': _as_bool(LaunchConfiguration('use_sim_time').perform(context)),
        'planner': _launch_value(context, 'planner', PAPER_LAUNCH_DEFAULTS['planner']).strip(),
        'world': world_value,
        'world_profiles_path': LaunchConfiguration('world_profiles').perform(context),
        'tasks_yaml': LaunchConfiguration('tasks_yaml').perform(context),
        'task_name': _launch_value(context, 'task', PAPER_LAUNCH_DEFAULTS['task']).strip(),
        'seed': seed_value,
        'perception_backend': _launch_value(context, 'perception_backend', PAPER_LAUNCH_DEFAULTS['perception_backend']).strip().lower(),
        'sensor_pixel_noise_sigma': float(_launch_value(context, 'sensor_pixel_noise_sigma', '0.0')),
        'odom_wait_timeout_s': float(_launch_value(context, 'odom_wait_timeout_s', '25.0')),
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
        'auto_stop_on_goal': _as_bool(_launch_value(context, 'auto_stop_on_goal', 'false')),
        'goal_success_radius': float(_launch_value(context, 'goal_success_radius', '0.35')),
        'goal_success_hold_s': float(_launch_value(context, 'goal_success_hold_s', '2.0')),
        'process_noise_xy': float(_launch_value(context, 'process_noise_xy', PAPER_LAUNCH_DEFAULTS['process_noise_xy'])),
        'process_noise_theta': float(_launch_value(context, 'process_noise_theta', PAPER_LAUNCH_DEFAULTS['process_noise_theta'])),
        'obs_noise_uv': float(_launch_value(context, 'obs_noise_uv', PAPER_LAUNCH_DEFAULTS['obs_noise_uv'])),
        'optimizer_maxiter': int(_launch_value(context, 'optimizer_maxiter', PAPER_LAUNCH_DEFAULTS['optimizer_maxiter'])),
        'optimizer_maxfun': int(_launch_value(context, 'optimizer_maxfun', PAPER_LAUNCH_DEFAULTS['optimizer_maxfun'])),
        'optimizer_ftol': float(_launch_value(context, 'optimizer_ftol', PAPER_LAUNCH_DEFAULTS['optimizer_ftol'])),
        'optimizer_gtol': float(_launch_value(context, 'optimizer_gtol', PAPER_LAUNCH_DEFAULTS['optimizer_gtol'])),
        'optimizer_warm_start': _as_bool(_launch_value(context, 'optimizer_warm_start', PAPER_LAUNCH_DEFAULTS['optimizer_warm_start'])),
        'plan_rate': float(_launch_value(context, 'plan_rate', PAPER_LAUNCH_DEFAULTS['plan_rate'])),
        'horizon': int(_launch_value(context, 'horizon', PAPER_LAUNCH_DEFAULTS['horizon'])),
        'dt': float(_launch_value(context, 'dt', '0.2')),
        'control_weight': float(_launch_value(context, 'control_weight', PAPER_LAUNCH_DEFAULTS['control_weight'])),
        'risk_weight_obs': float(_launch_value(context, 'risk_weight_obs', '1.0')),
        'ambiguity_weight': float(_launch_value(context, 'ambiguity_weight', '1.0')),
        'r_visible_uv': float(_launch_value(context, 'r_visible_uv', PAPER_LAUNCH_DEFAULTS['r_visible_uv'])),
        'r_miss_uv': float(_launch_value(context, 'r_miss_uv', PAPER_LAUNCH_DEFAULTS['r_miss_uv'])),
        'visibility_power': float(_launch_value(context, 'visibility_power', PAPER_LAUNCH_DEFAULTS['visibility_power'])),
        'visibility_sigma_kappa': float(_launch_value(context, 'visibility_sigma_kappa', PAPER_LAUNCH_DEFAULTS['visibility_sigma_kappa'])),
        'goal_prior_u_std_start': float(_launch_value(context, 'goal_prior_u_std_start', PAPER_LAUNCH_DEFAULTS['goal_prior_u_std_start'])),
        'goal_prior_v_std_start': float(_launch_value(context, 'goal_prior_v_std_start', PAPER_LAUNCH_DEFAULTS['goal_prior_v_std_start'])),
        'goal_prior_u_std_final': float(_launch_value(context, 'goal_prior_u_std_final', PAPER_LAUNCH_DEFAULTS['goal_prior_u_std_final'])),
        'goal_prior_v_std_final': float(_launch_value(context, 'goal_prior_v_std_final', PAPER_LAUNCH_DEFAULTS['goal_prior_v_std_final'])),
        'goal_tightening_power': float(_launch_value(context, 'goal_tightening_power', PAPER_LAUNCH_DEFAULTS['goal_tightening_power'])),
        'goal_progress_n_steps': int(_launch_value(context, 'goal_progress_n_steps', PAPER_LAUNCH_DEFAULTS['goal_progress_n_steps'])),
        'observation_risk_scale': float(_launch_value(context, 'observation_risk_scale', PAPER_LAUNCH_DEFAULTS['observation_risk_scale'])),
        'ambiguity_term_scale': float(_launch_value(context, 'ambiguity_term_scale', PAPER_LAUNCH_DEFAULTS['ambiguity_term_scale'])),
        'visibility_barrier_threshold': float(
            _launch_value(
                context,
                'visibility_barrier_threshold',
                PAPER_LAUNCH_DEFAULTS.get(
                    'visibility_barrier_threshold',
                    str(VISIBILITY_FALLBACK_DEFAULTS['visibility_barrier_threshold']),
                ),
            )
        ),
        'visibility_barrier_scale': float(
            _launch_value(
                context,
                'visibility_barrier_scale',
                PAPER_LAUNCH_DEFAULTS.get(
                    'visibility_barrier_scale',
                    str(VISIBILITY_FALLBACK_DEFAULTS['visibility_barrier_scale']),
                ),
            )
        ),
        'use_visibility_model': _as_bool(
            visibility_enabled_default if use_visibility_raw in ('', 'auto', 'default') else use_visibility_raw
        ),
        'perception_use_geometry_occlusion': _as_bool(
            _launch_value(context, 'perception_use_geometry_occlusion', 'true')
        ),
        'visibility_weight': float(
            _launch_value(
                context,
                'visibility_weight',
                PAPER_LAUNCH_DEFAULTS.get('visibility_weight', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_weight'])),
            )
        ),
        'visibility_target_height_m': float(_launch_value(context, 'visibility_target_height_m', str(VISIBILITY_FALLBACK_DEFAULTS['visibility_target_height_m']))),
        'visibility_geometry_json': _launch_value(context, 'visibility_geometry_json', ''),
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
        'min_state_cov': float(_launch_value(context, 'min_state_cov', '1e-6')),
        'debug_runtime': _as_bool(_launch_value(context, 'debug_runtime', 'false')),
        'enable_logging': _as_bool(_launch_value(context, 'enable_logging', 'true')),
        'use_rviz': _as_bool(_launch_value(context, 'use_rviz', 'false')),
        'rviz_config': _launch_value(context, 'rviz_config', ''),
    }

    if cfg['perception_backend'] not in ('homography', 'image_markers'):
        raise RuntimeError("perception_backend must be 'homography' or 'image_markers'")

    return cfg


def resolve_world_setup(cfg: Dict[str, object]) -> Dict[str, object]:
    """Resolve world profile/task and derive camera/spawn launch parameters."""
    from experiments.core.world_profiles import (
        load_profile,
        compute_camera_quaternion_from_rpy,
        compute_look_at_from_pose,
        resolve_profile_asset_path,
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
    if planner == 'visibility_unaware_baseline':
        cfg['use_visibility_model'] = False
        cfg['use_ambiguity'] = False
        cfg['use_obs_risk'] = True

    profile_visibility_artifact = resolve_profile_asset_path(
        cfg['world_profiles_path'],
        str(profile.get('visibility_artifact', '') or ''),
    )
    visibility_artifact_path = resolve_profile_asset_path(
        cfg['world_profiles_path'],
        str(cfg.get('visibility_artifact_path', '') or ''),
    ) or profile_visibility_artifact

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
    if cfg.get('use_visibility_model', False) and not visibility_artifact_path:
        raise RuntimeError(
            f"World '{cfg['world']}' requires a visibility_artifact path for GP visibility planning."
        )
    raw_use_nogo_cost = str(cfg.get('use_nogo_cost', 'auto')).strip().lower()
    nogo_geometry_needed = (
        raw_use_nogo_cost in ('1', 'true', 't', 'yes', 'y', 'on')
        or (raw_use_nogo_cost in ('', 'auto', 'default') and str(cfg.get('planner', '')).strip().lower() == 'mpc')
    )
    geometry_needed = bool(cfg.get('perception_use_geometry_occlusion', False)) or nogo_geometry_needed
    if (not visibility_geometry_json) and geometry_needed:
        visibility_geometry_json = serialize_occlusion_geometry_from_world(world_path)

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
        'visibility_artifact_path': visibility_artifact_path,
    })
    return cfg


def build_shared_nodes(cfg: Dict[str, object]) -> Dict[str, object]:
    """Create shared nodes/components for the thesis pipeline."""
    state_sources = _state_estimator_metadata(str(cfg['perception_backend']))
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
        'use_geometry_occlusion': cfg['perception_use_geometry_occlusion'],
        'visibility_geometry_json': cfg['visibility_geometry_json'],
        'visibility_target_height_m': cfg['visibility_target_height_m'],
    })
    if cfg['perception_backend'] == 'image_markers':
        perception_node = Node(
            package='perception',
            executable='image_marker_detector_node',
            name='image_marker_detector_node',
            output='screen',
            parameters=[homography_params],
        )
    else:
        perception_node = Node(
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
        'heading_pixel_noise_sigma': cfg['sensor_pixel_noise_sigma'],
        'transform_noise_sigma': 0.0,
        'use_odom_heading_fallback': True,
        'odom_heading_timeout_s': 0.5,
        'odom_heading_sigma_rad': 0.08,
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

    logger_node = None
    if cfg.get('enable_logging', True):
        logger_node = Node(
            package='experiments',
            executable='experiment_logger',
            name='experiment_logger',
            output='screen',
            on_exit=[Shutdown(reason='experiment_logger exited')],
            parameters=[{
                'use_sim_time': cfg['use_sim_time'],
                'seed': cfg['seed'],
                'method': cfg['planner'],
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
                'visibility_power': cfg['visibility_power'],
                'visibility_sigma_kappa': cfg['visibility_sigma_kappa'],
                'goal_prior_u_std_start': cfg['goal_prior_u_std_start'],
                'goal_prior_v_std_start': cfg['goal_prior_v_std_start'],
                'goal_prior_u_std_final': cfg['goal_prior_u_std_final'],
                'goal_prior_v_std_final': cfg['goal_prior_v_std_final'],
                'goal_tightening_power': cfg['goal_tightening_power'],
                'goal_progress_n_steps': cfg['goal_progress_n_steps'],
                'observation_risk_scale': cfg['observation_risk_scale'],
                'ambiguity_term_scale': cfg['ambiguity_term_scale'],
                'visibility_weight': cfg['visibility_weight'],
                'visibility_barrier_threshold': cfg['visibility_barrier_threshold'],
                'visibility_barrier_scale': cfg['visibility_barrier_scale'],
                'visibility_target_height_m': cfg['visibility_target_height_m'],
                'perception_use_geometry_occlusion': cfg['perception_use_geometry_occlusion'],
                'use_nogo_cost': cfg.get('resolved_use_nogo_cost', False),
                'nogo_penalty_type': cfg['nogo_penalty_type'],
                'nogo_weight': cfg['nogo_weight'],
                'nogo_safe_distance': cfg['nogo_safe_distance'],
                'nogo_gaussian_sigma': cfg['nogo_gaussian_sigma'],
                'nogo_softplus_scale': cfg['nogo_softplus_scale'],
                'nogo_logbarrier_scale': cfg['nogo_logbarrier_scale'],
                'nogo_logbarrier_eps': cfg['nogo_logbarrier_eps'],
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
        'homography_sim': perception_node,
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
        resolved_use_nogo_cost = cfg['planner'] == 'mpc'
    else:
        resolved_use_nogo_cost = _as_bool(raw_use_nogo_cost)

    cfg = dict(cfg)
    cfg['resolved_use_nogo_cost'] = resolved_use_nogo_cost
    shared_nodes = build_shared_nodes(cfg)
    planner = cfg['planner']

    if planner not in ('efe1', 'efe2', 'efer', 'mpc', 'visibility_unaware_baseline'):
        raise RuntimeError(
            "planner must be 'efe1', 'efe2', 'efer', 'mpc', or 'visibility_unaware_baseline' for agent launch"
        )

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
        'efer': {
            'approx_method': 'ET2',
            'use_ambiguity': False,
            'use_obs_risk': True,
        },
        'mpc': {
            'approx_method': 'ET1',
            'use_ambiguity': False,
            'use_obs_risk': True,
        },
        'visibility_unaware_baseline': {
            'approx_method': 'ET1',
            'use_ambiguity': False,
            'use_obs_risk': True,
        },
    }
    planner_uses_visibility = bool(cfg['use_visibility_model']) and planner != 'visibility_unaware_baseline'
    agent_node = Node(
        package='planning',
        executable='efe_agent',
        name=f'{planner}_agent',
        output='screen',
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
            'heading_pixel_noise_sigma': cfg['sensor_pixel_noise_sigma'],
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
            'visibility_power': cfg['visibility_power'],
            'visibility_sigma_kappa': cfg['visibility_sigma_kappa'],
            'goal_prior_u_std_start': cfg['goal_prior_u_std_start'],
            'goal_prior_v_std_start': cfg['goal_prior_v_std_start'],
            'goal_prior_u_std_final': cfg['goal_prior_u_std_final'],
            'goal_prior_v_std_final': cfg['goal_prior_v_std_final'],
            'goal_tightening_power': cfg['goal_tightening_power'],
            'goal_progress_n_steps': cfg['goal_progress_n_steps'],
            'observation_risk_scale': cfg['observation_risk_scale'],
            'ambiguity_term_scale': cfg['ambiguity_term_scale'],
            'use_visibility_model': planner_uses_visibility,
            'visibility_weight': cfg['visibility_weight'],
            'visibility_barrier_threshold': cfg['visibility_barrier_threshold'],
            'visibility_barrier_scale': cfg['visibility_barrier_scale'],
            'visibility_target_height_m': cfg['visibility_target_height_m'],
            'visibility_geometry_json': cfg['visibility_geometry_json'],
            'visibility_artifact_path': cfg['visibility_artifact_path'],
            'use_nogo_cost': cfg['resolved_use_nogo_cost'],
            'nogo_penalty_type': cfg['nogo_penalty_type'],
            'nogo_weight': cfg['nogo_weight'],
            'nogo_safe_distance': cfg['nogo_safe_distance'],
            'nogo_gaussian_sigma': cfg['nogo_gaussian_sigma'],
            'nogo_softplus_scale': cfg['nogo_softplus_scale'],
            'nogo_logbarrier_scale': cfg['nogo_logbarrier_scale'],
            'nogo_logbarrier_eps': cfg['nogo_logbarrier_eps'],
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
        shared_nodes['homography_sim'],
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

    return [
        shared_nodes['bringup_sim'],
        shared_nodes['tf_static'],
        agent_node,
        shared_nodes['wait_for_odom'],
        start_after_odom,
    ]
