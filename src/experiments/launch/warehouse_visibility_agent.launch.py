from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


DEFAULT_PLANNER = 'visibility_aware_efe'
ALLOWED_PLANNERS = ('visibility_aware_efe', 'risk_only_ablation', 'constant_R_efe')
PLANNER_DESCRIPTION = 'Planner: visibility_aware_efe | risk_only_ablation | constant_R_efe'


def _planner_precision_arguments():
    return [
        DeclareLaunchArgument('horizon', default_value='36'),
        DeclareLaunchArgument('dt', default_value='0.2'),
        DeclareLaunchArgument('v_max', default_value='0.22'),
        DeclareLaunchArgument('discount_gamma', default_value='0.98'),
        DeclareLaunchArgument('goal_prior_u_std_start', default_value='80.0'),
        DeclareLaunchArgument('goal_prior_v_std_start', default_value='80.0'),
        DeclareLaunchArgument('goal_prior_u_std_final', default_value='4.0'),
        DeclareLaunchArgument('goal_prior_v_std_final', default_value='4.0'),
        DeclareLaunchArgument('goal_tightening_power', default_value='0.45'),
        DeclareLaunchArgument('r_visible_uv', default_value='2.5'),
        DeclareLaunchArgument('r_miss_uv', default_value='120.0'),
        DeclareLaunchArgument('pixel_correction_nis_threshold', default_value='0.0',
                              description='Reject pixel corrections with 2D NIS above this threshold; 0 disables NIS gating.'),
        DeclareLaunchArgument('odom_heading_correction_mode', default_value='kalman'),
        DeclareLaunchArgument('clamp_pixel_uv_theta_without_yaw', default_value='false'),
        DeclareLaunchArgument('use_nogo_cost', default_value='auto'),
        DeclareLaunchArgument('nogo_penalty_type', default_value='softplus'),
        DeclareLaunchArgument('nogo_weight', default_value='40.0'),
        DeclareLaunchArgument('nogo_safe_distance', default_value='0.35'),
        DeclareLaunchArgument('nogo_gaussian_sigma', default_value='0.25'),
        DeclareLaunchArgument('nogo_softplus_scale', default_value='0.08'),
        DeclareLaunchArgument('nogo_logbarrier_scale', default_value='0.25'),
        DeclareLaunchArgument('nogo_logbarrier_eps', default_value='0.001'),
        DeclareLaunchArgument('use_belief_nogo_cost', default_value='false'),
        DeclareLaunchArgument('nogo_belief_kappa', default_value='1.0'),
        DeclareLaunchArgument('use_odom_for_predict', default_value='true'),
        DeclareLaunchArgument('odom_topic', default_value='/odom_noisy'),
        DeclareLaunchArgument('optimizer_multistart', default_value='false',
                              description='Offer multiple optimizer-init seeds; lowest-EFE candidate wins. Same seeds for all conditions.'),
        DeclareLaunchArgument('optimizer_multistart_include_direct', default_value='true',
                              description='Include a steer-to-goal seed when multistart is on.'),
        DeclareLaunchArgument('optimizer_multistart_lateral_offsets', default_value='',
                              description='Comma/JSON list of perpendicular offsets (m) for detour init seeds.'),
        DeclareLaunchArgument('optimizer_initial_routes_json', default_value='',
                              description='Optional JSON list of named routes used only as optimizer seeds.'),
        DeclareLaunchArgument('use_hierarchical', default_value='false'),
        DeclareLaunchArgument('global_horizon', default_value='60'),
        DeclareLaunchArgument('local_horizon', default_value='12'),
        DeclareLaunchArgument('local_plan_rate', default_value='4.0'),
        DeclareLaunchArgument('local_optimizer_maxiter', default_value='60'),
        DeclareLaunchArgument('global_use_ambiguity', default_value='true'),
        DeclareLaunchArgument('local_use_ambiguity', default_value='false'),
        DeclareLaunchArgument('global_optimizer_multistart', default_value='true'),
        DeclareLaunchArgument('local_optimizer_multistart', default_value='true'),
        DeclareLaunchArgument('local_use_visibility_model', default_value='false'),
        DeclareLaunchArgument('local_use_belief_nogo_cost', default_value='false'),
        DeclareLaunchArgument('local_nogo_penalty_type', default_value=''),
        DeclareLaunchArgument('local_nogo_weight', default_value='-1.0'),
        DeclareLaunchArgument('local_nogo_safe_distance', default_value='-1.0'),
        DeclareLaunchArgument('local_goal_prior_u_std_start', default_value='-1.0'),
        DeclareLaunchArgument('local_goal_prior_v_std_start', default_value='-1.0'),
        DeclareLaunchArgument('local_goal_prior_u_std_final', default_value='-1.0'),
        DeclareLaunchArgument('local_goal_prior_v_std_final', default_value='-1.0'),
        DeclareLaunchArgument('waypoint_spacing_m', default_value='1.0'),
        DeclareLaunchArgument('waypoint_arrival_radius_m', default_value='0.35'),
        DeclareLaunchArgument('local_replan_min_remaining_s', default_value='0.0'),
        DeclareLaunchArgument('local_replan_on_waypoint_change', default_value='false'),
        DeclareLaunchArgument('latency_compensate_plan_handoff', default_value='false'),
        DeclareLaunchArgument('use_simple_local_controller', default_value='false'),
        DeclareLaunchArgument('local_tracking_use_odom_yaw', default_value='false'),
        DeclareLaunchArgument('cmd_publish_rate', default_value='10.0'),
    ]


def _launch_setup(context, *args, **kwargs):
    from experiments.core.visibility_launch_common import (
        build_agent_runtime_actions,
        parse_common_launch_config,
        resolve_world_setup,
    )

    cfg = parse_common_launch_config(context)
    planner = str(cfg.get('planner', DEFAULT_PLANNER) or DEFAULT_PLANNER).strip()
    if planner not in ALLOWED_PLANNERS:
        raise RuntimeError(f"planner must be one of: {', '.join(ALLOWED_PLANNERS)}")

    cfg['planner'] = planner
    cfg['use_rviz'] = bool(cfg.get('use_rviz', False))

    if planner == 'constant_R_efe':
        cfg['use_visibility_model'] = False
        # C1 is constant-observability EFE: risk and ambiguity remain active,
        # but the observation covariance is spatially uniform instead of GP-based.
        cfg['use_ambiguity'] = True
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
        DeclareLaunchArgument('auto_stop_on_goal', default_value='true'),
        DeclareLaunchArgument('goal_success_radius', default_value='0.20'),
        DeclareLaunchArgument('goal_success_hold_s', default_value='2.0'),
        DeclareLaunchArgument('goal_stable_radius', default_value='0.20'),
        DeclareLaunchArgument('goal_stable_hold_s', default_value='2.0'),
        DeclareLaunchArgument('goal_stable_max_displacement_m', default_value='0.04'),
        DeclareLaunchArgument('run_timeout_after_first_cmd_s', default_value='60.0'),
        DeclareLaunchArgument('first_cmd_linear_eps', default_value='0.02'),
        DeclareLaunchArgument('first_cmd_angular_eps', default_value='0.10'),
        DeclareLaunchArgument('stuck_window_s', default_value='8.0'),
        DeclareLaunchArgument('stuck_max_displacement_m', default_value='0.08'),
        DeclareLaunchArgument('stuck_max_goal_improvement_m', default_value='0.05'),
        DeclareLaunchArgument('stuck_cmd_fraction_min', default_value='0.50'),
        DeclareLaunchArgument('stuck_idle_cmd_fraction_max', default_value='0.10'),
        DeclareLaunchArgument('use_command_noise', default_value='true'),
        DeclareLaunchArgument('use_encoder_noise', default_value='true'),
        DeclareLaunchArgument('command_noise_linear_slip_mean', default_value='0.03'),
        DeclareLaunchArgument('command_noise_linear_slip_std', default_value='0.06'),
        DeclareLaunchArgument('command_noise_angular_slip_mean', default_value='0.00'),
        DeclareLaunchArgument('command_noise_angular_slip_std', default_value='0.04'),
        DeclareLaunchArgument('command_noise_linear_additive_std', default_value='0.008'),
        DeclareLaunchArgument('command_noise_angular_additive_std', default_value='0.035'),
        DeclareLaunchArgument('command_noise_correlation_alpha', default_value='0.85'),
        DeclareLaunchArgument('perception_backend', default_value='yolo', description='Paper runtime perception backend: yolo'),
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
        DeclareLaunchArgument('use_pixel_correction', default_value='true'),
        DeclareLaunchArgument('pixel_correction_approx', default_value='AUTO'),
        *_planner_precision_arguments(),
        DeclareLaunchArgument('enable_logging', default_value='true'),
        DeclareLaunchArgument('log_dir', default_value='logs/experiments'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        OpaqueFunction(function=_launch_setup),
    ])
