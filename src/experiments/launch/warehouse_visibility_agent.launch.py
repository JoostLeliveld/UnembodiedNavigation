from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from experiments.core.visibility_launch_common import PAPER_LAUNCH_DEFAULTS, VISIBILITY_FALLBACK_DEFAULTS


def _launch_setup(context, *args, **kwargs):
    from experiments.core.visibility_launch_common import (
        apply_study_variant,
        build_agent_runtime_actions,
        parse_common_launch_config,
        resolve_world_setup,
    )

    cfg = parse_common_launch_config(context)
    cfg = apply_study_variant(cfg)
    cfg['auto_stop_on_goal'] = str(PAPER_LAUNCH_DEFAULTS['auto_stop_on_goal']).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
    cfg['goal_success_radius'] = float(PAPER_LAUNCH_DEFAULTS['goal_success_radius'])
    cfg['goal_success_hold_s'] = float(PAPER_LAUNCH_DEFAULTS['goal_success_hold_s'])
    cfg = resolve_world_setup(cfg)
    return build_agent_runtime_actions(cfg)


def generate_launch_description():
    """Warehouse visibility thesis launch with optional lower-compile autodiff defaults.

    This is the thesis-facing launch surface for the warehouse visibility-aware agent.
    It hides generic development controls and fixes a few study assumptions:
    - fixed thesis-facing state/perception path: pixel state from homography
    - fixed UV observation model
    - a single canonical warehouse world/task baseline
    - a paper profile that uses the notebook-style SciPy backend by default
    - a fast profile that trims iterations while staying close to the paper defaults
    - a dev profile that strips compile-heavy pieces for rapid iteration
    """

    defaults = PAPER_LAUNCH_DEFAULTS


    profile_arg = DeclareLaunchArgument(
        'profile',
        default_value='paper',
        description="Preset defaults: 'paper', 'fast', or 'dev'"
    )
    study_variant_arg = DeclareLaunchArgument(
        'study_variant',
        default_value='gp_visibility',
        description="Comparison variant: 'projection_only', 'raycast_25d', or 'gp_visibility'"
    )
    planner_default = PythonExpression([
        "'", defaults['planner'], "'"
    ])
    optimizer_backend_default = PythonExpression([
        "'scipy'"
    ])
    optimizer_maxiter_default = PythonExpression([
        "'20' if '", LaunchConfiguration('profile'), "' in ['fast', 'dev'] else '", defaults['optimizer_maxiter'], "'"
    ])
    use_ambiguity_default = PythonExpression([
        "'", defaults['use_ambiguity'], "'"
    ])
    jax_warmup_enabled_default = PythonExpression([
        "'false' if '", LaunchConfiguration('profile'), "' == 'dev' else 'true'"
    ])

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz (off by default for study runs)'
    )
    use_live_dashboard_arg = DeclareLaunchArgument(
        'use_live_dashboard',
        default_value='false',
        description='Launch the live matplotlib sanity dashboard'
    )
    dashboard_history_arg = DeclareLaunchArgument(
        'dashboard_history_s',
        default_value='60.0',
        description='History window in seconds for the live dashboard'
    )
    dashboard_redraw_arg = DeclareLaunchArgument(
        'dashboard_redraw_hz',
        default_value='4.0',
        description='Redraw rate for the live dashboard'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('visualization'), 'rviz', 'warehouse_visibility_story.rviz'
        ]),
        description='RViz config file when use_rviz=true'
    )
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value=planner_default,
        description='Planner: efe1 | efe2 | efer | mpc'
    )
    world_arg = DeclareLaunchArgument('world', default_value=defaults['world'])
    world_profiles_arg = DeclareLaunchArgument(
        'world_profiles',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'world_profiles.yaml'
        ]),
        description='World profile YAML'
    )
    tasks_yaml_arg = DeclareLaunchArgument(
        'tasks_yaml',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'tasks.yaml'
        ]),
        description='Task YAML'
    )
    task_arg = DeclareLaunchArgument('task', default_value=defaults['task'])

    seed_arg = DeclareLaunchArgument('seed', default_value=defaults['seed'])

    sensor_pixel_noise_arg = DeclareLaunchArgument(
        'sensor_pixel_noise_sigma',
        default_value=defaults['sensor_pixel_noise_sigma'],
        description='Primary study noise knob (homography source pixel noise std)'
    )
    perception_use_geometry_occlusion_arg = DeclareLaunchArgument(
        'perception_use_geometry_occlusion',
        default_value='true',
        description='Keep perception occlusion fixed to scene geometry for all comparison runs'
    )
    use_nogo_cost_arg = DeclareLaunchArgument(
        'use_nogo_cost',
        default_value=str(VISIBILITY_FALLBACK_DEFAULTS['use_nogo_cost']),
        description="Enable geometry-based no-go-zone obstacle cost ('auto' enables it for planner:=mpc)"
    )
    nogo_penalty_type_arg = DeclareLaunchArgument(
        'nogo_penalty_type',
        default_value='softplus',
        description="No-go penalty type: 'gaussian', 'softplus', or 'log_barrier'"
    )
    nogo_weight_arg = DeclareLaunchArgument(
        'nogo_weight',
        default_value='40.0',
        description='Weight of the no-go-zone obstacle penalty'
    )
    nogo_safe_distance_arg = DeclareLaunchArgument(
        'nogo_safe_distance',
        default_value='0.35',
        description='Desired obstacle clearance in meters before penalties grow sharply'
    )
    nogo_gaussian_sigma_arg = DeclareLaunchArgument(
        'nogo_gaussian_sigma',
        default_value='0.25',
        description='Gaussian decay scale for no-go penalties'
    )
    nogo_softplus_scale_arg = DeclareLaunchArgument(
        'nogo_softplus_scale',
        default_value='0.08',
        description='Softplus slope scale for no-go penalties'
    )
    nogo_logbarrier_scale_arg = DeclareLaunchArgument(
        'nogo_logbarrier_scale',
        default_value='0.25',
        description='Log-barrier decay scale outside the no-go boundary'
    )
    nogo_logbarrier_eps_arg = DeclareLaunchArgument(
        'nogo_logbarrier_eps',
        default_value='1e-3',
        description='Log-barrier clamp epsilon used inside or on the no-go boundary'
    )

    odom_wait_timeout_arg = DeclareLaunchArgument('odom_wait_timeout_s', default_value=defaults['odom_wait_timeout_s'])
    odom_wait_min_messages_arg = DeclareLaunchArgument('odom_wait_min_messages', default_value=defaults['odom_wait_min_messages'])
    odom_wait_require_pose_match_arg = DeclareLaunchArgument('odom_wait_require_pose_match', default_value=defaults['odom_wait_require_pose_match'])


    use_pixel_correction_arg = DeclareLaunchArgument('use_pixel_correction', default_value=defaults['use_pixel_correction'])
    pixel_timeout_arg = DeclareLaunchArgument('pixel_timeout_s', default_value=defaults['pixel_timeout_s'])
    pixel_correction_min_interval_arg = DeclareLaunchArgument(
        'pixel_correction_min_interval_s',
        default_value=defaults['pixel_correction_min_interval_s'],
    )
    pixel_correction_approx_arg = DeclareLaunchArgument(
        'pixel_correction_approx',
        default_value=defaults['pixel_correction_approx'],
        description='AUTO | ET1 | ET2 | UT'
    )
    skip_stale_pixel_correction_arg = DeclareLaunchArgument(
        'skip_stale_pixel_correction',
        default_value=defaults['skip_stale_pixel_correction']
    )
    min_state_cov_arg = DeclareLaunchArgument('min_state_cov', default_value=defaults['min_state_cov'])

    plan_rate_arg = DeclareLaunchArgument('plan_rate', default_value=defaults['plan_rate'])
    horizon_arg = DeclareLaunchArgument('horizon', default_value=defaults['horizon'])
    dt_arg = DeclareLaunchArgument('dt', default_value=defaults['dt'])
    control_weight_arg = DeclareLaunchArgument('control_weight', default_value=defaults['control_weight'])
    math_mode_arg = DeclareLaunchArgument('math_mode', default_value=defaults['math_mode'])
    risk_weight_state_arg = DeclareLaunchArgument('risk_weight_state', default_value=defaults['risk_weight_state'])
    risk_weight_obs_arg = DeclareLaunchArgument('risk_weight_obs', default_value=defaults['risk_weight_obs'])
    ambiguity_weight_arg = DeclareLaunchArgument('ambiguity_weight', default_value=defaults['ambiguity_weight'])
    goal_sigma_uv_arg = DeclareLaunchArgument('goal_sigma_uv', default_value=defaults['goal_sigma_uv'])
    r_visible_uv_arg = DeclareLaunchArgument('r_visible_uv', default_value=defaults['r_visible_uv'])
    r_miss_uv_arg = DeclareLaunchArgument('r_miss_uv', default_value=defaults['r_miss_uv'])
    visibility_power_arg = DeclareLaunchArgument('visibility_power', default_value=defaults['visibility_power'])
    visibility_sigma_kappa_arg = DeclareLaunchArgument('visibility_sigma_kappa', default_value=defaults['visibility_sigma_kappa'])
    goal_prior_u_std_start_arg = DeclareLaunchArgument('goal_prior_u_std_start', default_value=defaults['goal_prior_u_std_start'])
    goal_prior_v_std_start_arg = DeclareLaunchArgument('goal_prior_v_std_start', default_value=defaults['goal_prior_v_std_start'])
    goal_prior_u_std_final_arg = DeclareLaunchArgument('goal_prior_u_std_final', default_value=defaults['goal_prior_u_std_final'])
    goal_prior_v_std_final_arg = DeclareLaunchArgument('goal_prior_v_std_final', default_value=defaults['goal_prior_v_std_final'])
    goal_tightening_power_arg = DeclareLaunchArgument('goal_tightening_power', default_value=defaults['goal_tightening_power'])
    goal_progress_n_steps_arg = DeclareLaunchArgument('goal_progress_n_steps', default_value=defaults['goal_progress_n_steps'])
    notebook_risk_scale_arg = DeclareLaunchArgument('notebook_risk_scale', default_value=defaults['notebook_risk_scale'])
    notebook_ambiguity_scale_arg = DeclareLaunchArgument('notebook_ambiguity_scale', default_value=defaults['notebook_ambiguity_scale'])
    visibility_weight_arg = DeclareLaunchArgument(
        'visibility_weight',
        default_value=str(VISIBILITY_FALLBACK_DEFAULTS['visibility_weight'])
    )
    use_ambiguity_arg = DeclareLaunchArgument(
        'use_ambiguity',
        default_value=use_ambiguity_default,
        description='Enable ambiguity term for EFE planners (EFE-R/MPC presets still disable it)'
    )
    use_obs_risk_arg = DeclareLaunchArgument(
        'use_obs_risk',
        default_value=defaults['use_obs_risk'],
        description='Enable observation-space risk term'
    )

    process_noise_xy_arg = DeclareLaunchArgument('process_noise_xy', default_value=defaults['process_noise_xy'])
    process_noise_theta_arg = DeclareLaunchArgument('process_noise_theta', default_value=defaults['process_noise_theta'])
    obs_noise_uv_arg = DeclareLaunchArgument('obs_noise_uv', default_value=defaults['obs_noise_uv'])

    optimizer_backend_arg = DeclareLaunchArgument(
        'optimizer_backend',
        default_value=optimizer_backend_default,
        description="Optimizer backend: 'scipy', 'jax', or 'auto'"
    )
    optimizer_maxiter_arg = DeclareLaunchArgument('optimizer_maxiter', default_value=optimizer_maxiter_default)
    optimizer_maxfun_arg = DeclareLaunchArgument('optimizer_maxfun', default_value=defaults['optimizer_maxfun'])
    optimizer_ftol_arg = DeclareLaunchArgument('optimizer_ftol', default_value=defaults['optimizer_ftol'])
    optimizer_gtol_arg = DeclareLaunchArgument('optimizer_gtol', default_value=defaults['optimizer_gtol'])
    optimizer_warm_start_arg = DeclareLaunchArgument('optimizer_warm_start', default_value=defaults['optimizer_warm_start'])
    jax_warmup_enabled_arg = DeclareLaunchArgument(
        'jax_warmup_enabled',
        default_value=jax_warmup_enabled_default,
        description='Precompile the JAX planner before planning starts'
    )

    debug_runtime_arg = DeclareLaunchArgument(
        'debug_runtime',
        default_value=defaults['debug_runtime'],
        description='Enable runtime debug logs from planner node'
    )

    return LaunchDescription([
        profile_arg,
        study_variant_arg,
        use_rviz_arg,
        use_live_dashboard_arg,
        dashboard_history_arg,
        dashboard_redraw_arg,
        rviz_config_arg,
        use_sim_time_arg,
        planner_arg,
        world_arg,
        world_profiles_arg,
        tasks_yaml_arg,
        task_arg,
        seed_arg,
        sensor_pixel_noise_arg,
        perception_use_geometry_occlusion_arg,
        use_nogo_cost_arg,
        nogo_penalty_type_arg,
        nogo_weight_arg,
        nogo_safe_distance_arg,
        nogo_gaussian_sigma_arg,
        nogo_softplus_scale_arg,
        nogo_logbarrier_scale_arg,
        nogo_logbarrier_eps_arg,
        odom_wait_timeout_arg,
        odom_wait_min_messages_arg,
        odom_wait_require_pose_match_arg,
        use_pixel_correction_arg,
        pixel_timeout_arg,
        pixel_correction_min_interval_arg,
        pixel_correction_approx_arg,
        skip_stale_pixel_correction_arg,
        min_state_cov_arg,
        plan_rate_arg,
        horizon_arg,
        dt_arg,
        control_weight_arg,
        math_mode_arg,
        risk_weight_state_arg,
        risk_weight_obs_arg,
        ambiguity_weight_arg,
        goal_sigma_uv_arg,
        r_visible_uv_arg,
        r_miss_uv_arg,
        visibility_power_arg,
        visibility_sigma_kappa_arg,
        goal_prior_u_std_start_arg,
        goal_prior_v_std_start_arg,
        goal_prior_u_std_final_arg,
        goal_prior_v_std_final_arg,
        goal_tightening_power_arg,
        goal_progress_n_steps_arg,
        notebook_risk_scale_arg,
        notebook_ambiguity_scale_arg,
        visibility_weight_arg,
        use_ambiguity_arg,
        use_obs_risk_arg,
        process_noise_xy_arg,
        process_noise_theta_arg,
        obs_noise_uv_arg,
        optimizer_backend_arg,
        optimizer_maxiter_arg,
        optimizer_maxfun_arg,
        optimizer_ftol_arg,
        optimizer_gtol_arg,
        optimizer_warm_start_arg,
        jax_warmup_enabled_arg,
        debug_runtime_arg,
        OpaqueFunction(function=_launch_setup),
    ])
