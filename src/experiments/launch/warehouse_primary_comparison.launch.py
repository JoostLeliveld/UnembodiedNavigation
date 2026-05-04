from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


DEFAULT_PLANNER = 'efe1'
ALLOWED_PLANNERS = ('efe1', 'gp_risk_only', 'visibility_unaware_baseline')
PLANNER_DESCRIPTION = 'Primary thesis comparison: efe1 | gp_risk_only | visibility_unaware_baseline'


def _planner_precision_arguments():
    return [
        DeclareLaunchArgument('horizon', default_value='40'),
        DeclareLaunchArgument('dt', default_value='0.25'),
        DeclareLaunchArgument('discount_gamma', default_value='0.98'),
        DeclareLaunchArgument('goal_prior_u_std_start', default_value='80.0'),
        DeclareLaunchArgument('goal_prior_v_std_start', default_value='80.0'),
        DeclareLaunchArgument('goal_prior_u_std_final', default_value='4.0'),
        DeclareLaunchArgument('goal_prior_v_std_final', default_value='4.0'),
        DeclareLaunchArgument('goal_tightening_power', default_value='0.45'),
        DeclareLaunchArgument('r_visible_uv', default_value='2.5'),
        DeclareLaunchArgument('r_miss_uv', default_value='120.0'),
        DeclareLaunchArgument('odom_heading_correction_mode', default_value='kalman'),
        DeclareLaunchArgument('clamp_pixel_uv_theta_without_yaw', default_value='false'),
    ]


def _launch_setup(context, *args, **kwargs):
    from experiments.core.visibility_launch_common import (
        build_agent_runtime_actions,
        parse_common_launch_config,
        resolve_world_setup,
    )

    cfg = parse_common_launch_config(context)
    planner = str(cfg.get('planner', DEFAULT_PLANNER) or DEFAULT_PLANNER).strip().lower()
    if planner not in ALLOWED_PLANNERS:
        raise RuntimeError(f"planner must be one of: {', '.join(ALLOWED_PLANNERS)}")

    cfg['planner'] = planner
    cfg['use_rviz'] = bool(cfg.get('use_rviz', False))

    if planner == 'visibility_unaware_baseline':
        cfg['use_visibility_model'] = False
        cfg['use_ambiguity'] = False
        cfg['use_obs_risk'] = True
    elif planner == 'gp_risk_only':
        cfg['use_visibility_model'] = True
        cfg['use_ambiguity'] = False
        cfg['use_obs_risk'] = True
    else:
        cfg['use_visibility_model'] = True

    cfg = resolve_world_setup(cfg)
    return build_agent_runtime_actions(cfg)


def generate_launch_description():
    world_profiles_default = PathJoinSubstitution([
        FindPackageShare('experiments'), 'config', 'world_profiles.yaml',
    ])
    tasks_default = PathJoinSubstitution([
        FindPackageShare('experiments'), 'config', 'tasks.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('planner', default_value=DEFAULT_PLANNER, description=PLANNER_DESCRIPTION),
        DeclareLaunchArgument('world', default_value='warehouse_occ_light.world.sdf'),
        DeclareLaunchArgument('world_profiles', default_value=world_profiles_default, description='World profile YAML'),
        DeclareLaunchArgument('tasks_yaml', default_value=tasks_default, description='Task YAML'),
        DeclareLaunchArgument('task', default_value='', description='Task name; empty uses the world profile recommended_task'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('comparison_method_id', default_value=''),
        DeclareLaunchArgument('run_timeout_after_first_cmd_s', default_value='75.0'),
        DeclareLaunchArgument('first_cmd_linear_eps', default_value='0.02'),
        DeclareLaunchArgument('first_cmd_angular_eps', default_value='0.10'),
        DeclareLaunchArgument('stuck_window_s', default_value='8.0'),
        DeclareLaunchArgument('stuck_max_displacement_m', default_value='0.08'),
        DeclareLaunchArgument('stuck_max_goal_improvement_m', default_value='0.05'),
        DeclareLaunchArgument('stuck_cmd_fraction_min', default_value='0.50'),
        DeclareLaunchArgument('use_command_noise', default_value='true'),
        DeclareLaunchArgument('command_noise_linear_slip_mean', default_value='0.03'),
        DeclareLaunchArgument('command_noise_linear_slip_std', default_value='0.06'),
        DeclareLaunchArgument('command_noise_angular_slip_mean', default_value='0.00'),
        DeclareLaunchArgument('command_noise_angular_slip_std', default_value='0.04'),
        DeclareLaunchArgument('command_noise_linear_additive_std', default_value='0.008'),
        DeclareLaunchArgument('command_noise_angular_additive_std', default_value='0.035'),
        DeclareLaunchArgument('command_noise_correlation_alpha', default_value='0.85'),
        DeclareLaunchArgument('perception_backend', default_value='image_markers', description='image_markers, yolo, or homography'),
        DeclareLaunchArgument('sensor_pixel_noise_sigma', default_value='1.0'),
        DeclareLaunchArgument('yolo_model', default_value='', description='Local path to a trained YOLO .pt model'),
        DeclareLaunchArgument('yolo_device', default_value='', description='Ultralytics device string; empty lets Ultralytics choose'),
        DeclareLaunchArgument('yolo_imgsz', default_value='640'),
        DeclareLaunchArgument('yolo_conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('yolo_iou_threshold', default_value='0.45'),
        DeclareLaunchArgument('yolo_target_class', default_value='robot'),
        DeclareLaunchArgument('yolo_class_id', default_value='-1'),
        DeclareLaunchArgument('yolo_use_masks', default_value='true', description='Use YOLO segmentation masks for pixel reference when available'),
        DeclareLaunchArgument('yolo_min_mask_area_px', default_value='12.0'),
        DeclareLaunchArgument('yolo_mask_bottom_band_px', default_value='3.0'),
        *_planner_precision_arguments(),
        DeclareLaunchArgument('enable_logging', default_value='true'),
        DeclareLaunchArgument('log_dir', default_value='logs/experiments'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        OpaqueFunction(function=_launch_setup),
    ])
