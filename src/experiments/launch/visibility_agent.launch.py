from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *args, **kwargs):
    from experiments.core.visibility_launch_common import (
        build_agent_runtime_actions,
        parse_common_launch_config,
        resolve_world_setup,
    )

    cfg = resolve_world_setup(parse_common_launch_config(context))
    return build_agent_runtime_actions(cfg)


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for the visibility-aware agent pipeline',
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('visualization'), 'rviz', 'visibility_camera.rviz'
        ]),
        description='RViz config file to load when use_rviz=true',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true',
    )
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='efe2',
        description='Planner: efe1 | efe2 | mpc | efer',
    )
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.world.sdf',
        description='World file under sim/gazebo_worlds/worlds',
    )
    world_profiles_arg = DeclareLaunchArgument(
        'world_profiles',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'world_profiles.yaml'
        ]),
        description='YAML file describing per-world profiles',
    )
    tasks_yaml_arg = DeclareLaunchArgument(
        'tasks_yaml',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'tasks.yaml'
        ]),
        description='YAML file describing per-world tasks',
    )
    task_arg = DeclareLaunchArgument(
        'task',
        default_value='',
        description='Task name; defaults to first task in tasks.yaml for the world',
    )
    seed_arg = DeclareLaunchArgument('seed', default_value='0')
    sensor_pixel_noise_arg = DeclareLaunchArgument(
        'sensor_pixel_noise_sigma',
        default_value='0.0',
        description='Primary noise knob at the homography measurement source',
    )
    odom_wait_timeout_arg = DeclareLaunchArgument(
        'odom_wait_timeout_s',
        default_value='25.0',
        description='Timeout for startup /odom gate in seconds (0 disables timeout)',
    )
    odom_wait_min_messages_arg = DeclareLaunchArgument(
        'odom_wait_min_messages',
        default_value='1',
        description='Number of /odom messages required to open startup gate',
    )
    odom_wait_require_pose_match_arg = DeclareLaunchArgument(
        'odom_wait_require_pose_match',
        default_value='false',
        description='Require startup /odom pose to match expected (usually false for robustness)',
    )
    use_pixel_correction_arg = DeclareLaunchArgument(
        'use_pixel_correction',
        default_value='true',
        description='Apply pixel-space correction in the EFE agent',
    )
    pixel_timeout_arg = DeclareLaunchArgument('pixel_timeout_s', default_value='0.5')
    pixel_correction_min_interval_arg = DeclareLaunchArgument(
        'pixel_correction_min_interval_s',
        default_value='0.1',
        description='Minimum simulated time between expensive planner pixel-correction updates',
    )
    pixel_correction_approx_arg = DeclareLaunchArgument(
        'pixel_correction_approx',
        default_value='AUTO',
        description='Pixel correction approximation: AUTO | ET1 | ET2 | UT',
    )
    skip_stale_pixel_correction_arg = DeclareLaunchArgument(
        'skip_stale_pixel_correction',
        default_value='true',
        description='Skip correction when pixel measurement age exceeds pixel_timeout_s',
    )
    min_state_cov_arg = DeclareLaunchArgument(
        'min_state_cov',
        default_value='1e-6',
        description='Minimum diagonal covariance floor in planner belief',
    )
    use_ambiguity_arg = DeclareLaunchArgument(
        'use_ambiguity',
        default_value='true',
        description='Enable ambiguity term in the EFE objective',
    )
    use_obs_risk_arg = DeclareLaunchArgument(
        'use_obs_risk',
        default_value='true',
        description='Enable observation-space risk term',
    )
    plan_rate_arg = DeclareLaunchArgument(
        'plan_rate',
        default_value='5.0',
        description='Replanning rate in Hz',
    )
    horizon_arg = DeclareLaunchArgument(
        'horizon',
        default_value='10',
        description='Planning horizon length in steps',
    )
    dt_arg = DeclareLaunchArgument(
        'dt',
        default_value='0.2',
        description='Planner discretization step in seconds',
    )
    control_weight_arg = DeclareLaunchArgument(
        'control_weight',
        default_value='0.1',
        description='Quadratic control penalty weight',
    )
    risk_weight_state_arg = DeclareLaunchArgument(
        'risk_weight_state',
        default_value='0.0',
        description='State-space risk weight',
    )
    risk_weight_obs_arg = DeclareLaunchArgument(
        'risk_weight_obs',
        default_value='1.0',
        description='Observation-space risk weight',
    )
    ambiguity_weight_arg = DeclareLaunchArgument(
        'ambiguity_weight',
        default_value='1.0',
        description='Ambiguity term weight',
    )
    goal_sigma_uv_arg = DeclareLaunchArgument(
        'goal_sigma_uv',
        default_value='0.0',
        description='Goal observation std in pixel u/v (0 means use obs_noise_uv)',
    )
    process_noise_xy_arg = DeclareLaunchArgument(
        'process_noise_xy',
        default_value='0.01',
        description='Process noise std for x/y state dynamics',
    )
    process_noise_theta_arg = DeclareLaunchArgument(
        'process_noise_theta',
        default_value='0.02',
        description='Process noise std for yaw state dynamics',
    )
    obs_noise_uv_arg = DeclareLaunchArgument(
        'obs_noise_uv',
        default_value='2.0',
        description='Observation noise std in pixel u/v',
    )
    use_visibility_model_arg = DeclareLaunchArgument(
        'use_visibility_model',
        default_value='auto',
        description='Visibility model enable switch: auto | true | false',
    )
    visibility_model_arg = DeclareLaunchArgument(
        'visibility_model',
        default_value='auto',
        description='Visibility backend: auto | fixed_gp | raycast_25d',
    )
    visibility_weight_arg = DeclareLaunchArgument(
        'visibility_weight',
        default_value='4.0',
        description='Visibility penalty weight w_vis*(1-p_vis)',
    )
    visibility_map_min_x_arg = DeclareLaunchArgument('visibility_map_min_x', default_value='-5.0')
    visibility_map_max_x_arg = DeclareLaunchArgument('visibility_map_max_x', default_value='5.0')
    visibility_map_min_y_arg = DeclareLaunchArgument('visibility_map_min_y', default_value='-5.0')
    visibility_map_max_y_arg = DeclareLaunchArgument('visibility_map_max_y', default_value='5.0')
    visibility_map_nx_arg = DeclareLaunchArgument('visibility_map_nx', default_value='140')
    visibility_map_ny_arg = DeclareLaunchArgument('visibility_map_ny', default_value='120')
    visibility_gp_length_scale_arg = DeclareLaunchArgument('visibility_gp_length_scale', default_value='1.4')
    visibility_gp_noise_var_arg = DeclareLaunchArgument('visibility_gp_noise_var', default_value='0.15')
    visibility_prior_occ_arg = DeclareLaunchArgument('visibility_prior_occ', default_value='0.005')
    visibility_beta_arg = DeclareLaunchArgument('visibility_beta', default_value='1.0')
    visibility_height_tau_arg = DeclareLaunchArgument('visibility_height_tau', default_value='0.08')
    visibility_ray_samples_arg = DeclareLaunchArgument('visibility_ray_samples', default_value='120')
    visibility_sigma_kappa_arg = DeclareLaunchArgument('visibility_sigma_kappa', default_value='1.0')
    visibility_target_height_m_arg = DeclareLaunchArgument('visibility_target_height_m', default_value='0.0')
    visibility_geometry_json_arg = DeclareLaunchArgument('visibility_geometry_json', default_value='')
    visibility_gp_seed_arg = DeclareLaunchArgument('visibility_gp_seed', default_value='0')
    visibility_r_bad_uv_arg = DeclareLaunchArgument('visibility_r_bad_uv', default_value='28.0')
    visibility_cov_pos_scale_arg = DeclareLaunchArgument('visibility_cov_pos_scale', default_value='2.0')
    visibility_cov_theta_scale_arg = DeclareLaunchArgument('visibility_cov_theta_scale', default_value='0.8')
    optimizer_backend_arg = DeclareLaunchArgument(
        'optimizer_backend',
        default_value='auto',
        description="Optimizer backend: 'auto' or 'jax'",
    )
    optimizer_maxiter_arg = DeclareLaunchArgument(
        'optimizer_maxiter',
        default_value='50',
        description='Maximum optimizer iterations for the agent planner',
    )
    optimizer_gtol_arg = DeclareLaunchArgument(
        'optimizer_gtol',
        default_value='1e-4',
        description='Optimizer gradient tolerance for the agent planner',
    )
    optimizer_warm_start_arg = DeclareLaunchArgument(
        'optimizer_warm_start',
        default_value='true',
        description='Use warm-start controls between planning cycles',
    )
    debug_runtime_arg = DeclareLaunchArgument(
        'debug_runtime',
        default_value='false',
        description='Enable periodic runtime and optimizer debug logs',
    )
    auto_stop_on_goal_arg = DeclareLaunchArgument(
        'auto_stop_on_goal',
        default_value='false',
        description='Stop the run automatically when the goal has been held',
    )
    goal_success_radius_arg = DeclareLaunchArgument('goal_success_radius', default_value='0.35')
    goal_success_hold_s_arg = DeclareLaunchArgument('goal_success_hold_s', default_value='2.0')

    return LaunchDescription([
        use_rviz_arg,
        rviz_config_arg,
        use_sim_time_arg,
        planner_arg,
        world_arg,
        world_profiles_arg,
        tasks_yaml_arg,
        task_arg,
        seed_arg,
        sensor_pixel_noise_arg,
        odom_wait_timeout_arg,
        odom_wait_min_messages_arg,
        odom_wait_require_pose_match_arg,
        use_pixel_correction_arg,
        pixel_timeout_arg,
        pixel_correction_min_interval_arg,
        pixel_correction_approx_arg,
        skip_stale_pixel_correction_arg,
        min_state_cov_arg,
        use_ambiguity_arg,
        use_obs_risk_arg,
        plan_rate_arg,
        horizon_arg,
        dt_arg,
        control_weight_arg,
        risk_weight_state_arg,
        risk_weight_obs_arg,
        ambiguity_weight_arg,
        goal_sigma_uv_arg,
        process_noise_xy_arg,
        process_noise_theta_arg,
        obs_noise_uv_arg,
        use_visibility_model_arg,
        visibility_model_arg,
        visibility_weight_arg,
        visibility_map_min_x_arg,
        visibility_map_max_x_arg,
        visibility_map_min_y_arg,
        visibility_map_max_y_arg,
        visibility_map_nx_arg,
        visibility_map_ny_arg,
        visibility_gp_length_scale_arg,
        visibility_gp_noise_var_arg,
        visibility_prior_occ_arg,
        visibility_beta_arg,
        visibility_height_tau_arg,
        visibility_ray_samples_arg,
        visibility_sigma_kappa_arg,
        visibility_target_height_m_arg,
        visibility_geometry_json_arg,
        visibility_gp_seed_arg,
        visibility_r_bad_uv_arg,
        visibility_cov_pos_scale_arg,
        visibility_cov_theta_scale_arg,
        optimizer_backend_arg,
        optimizer_maxiter_arg,
        optimizer_gtol_arg,
        optimizer_warm_start_arg,
        debug_runtime_arg,
        auto_stop_on_goal_arg,
        goal_success_radius_arg,
        goal_success_hold_s_arg,
        OpaqueFunction(function=_launch_setup),
    ])
