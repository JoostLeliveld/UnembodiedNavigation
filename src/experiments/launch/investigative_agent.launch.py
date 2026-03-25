from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from experiments.core.visibility_launch_common import PAPER_LAUNCH_DEFAULTS


def _launch_setup(context, *args, **kwargs):
    from experiments.core.visibility_launch_common import (
        build_agent_runtime_actions,
        parse_common_launch_config,
        resolve_world_setup,
    )

    cfg = parse_common_launch_config(context)
    cfg['auto_stop_on_goal'] = str(PAPER_LAUNCH_DEFAULTS['auto_stop_on_goal']).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
    cfg['goal_success_radius'] = float(PAPER_LAUNCH_DEFAULTS['goal_success_radius'])
    cfg['goal_success_hold_s'] = float(PAPER_LAUNCH_DEFAULTS['goal_success_hold_s'])
    cfg = resolve_world_setup(cfg)
    return build_agent_runtime_actions(cfg)


def generate_launch_description():
    """Paper-focused agent launch with optional lower-compile autodiff defaults.

    This is the thesis-facing launch surface for the visibility-aware agent.
    It hides generic development controls and fixes a few study assumptions:
    - fixed thesis-facing state/perception path: pixel state from homography
    - fixed UV observation model
    - a paper profile that keeps the JAX backend by default
    - a fast profile that keeps autodiff but prefers lower compile cost for iteration
    """

    defaults = PAPER_LAUNCH_DEFAULTS


    profile_arg = DeclareLaunchArgument(
        'profile',
        default_value='paper',
        description="Preset defaults: 'paper' or 'fast'"
    )
    planner_default = PythonExpression([
        "'efe1' if '", LaunchConfiguration('profile'), "' == 'fast' else '", defaults['planner'], "'"
    ])
    optimizer_backend_default = PythonExpression([
        "'jax'"
    ])
    optimizer_maxiter_default = PythonExpression([
        "'20' if '", LaunchConfiguration('profile'), "' == 'fast' else '", defaults['optimizer_maxiter'], "'"
    ])

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz (off by default for study runs)'
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
    risk_weight_state_arg = DeclareLaunchArgument('risk_weight_state', default_value=defaults['risk_weight_state'])
    risk_weight_obs_arg = DeclareLaunchArgument('risk_weight_obs', default_value=defaults['risk_weight_obs'])
    ambiguity_weight_arg = DeclareLaunchArgument('ambiguity_weight', default_value=defaults['ambiguity_weight'])
    goal_sigma_uv_arg = DeclareLaunchArgument('goal_sigma_uv', default_value=defaults['goal_sigma_uv'])
    use_ambiguity_arg = DeclareLaunchArgument(
        'use_ambiguity',
        default_value=defaults['use_ambiguity'],
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
        description="Optimizer backend: 'jax' or 'auto'"
    )
    optimizer_maxiter_arg = DeclareLaunchArgument('optimizer_maxiter', default_value=optimizer_maxiter_default)
    optimizer_gtol_arg = DeclareLaunchArgument('optimizer_gtol', default_value=defaults['optimizer_gtol'])
    optimizer_warm_start_arg = DeclareLaunchArgument('optimizer_warm_start', default_value=defaults['optimizer_warm_start'])

    debug_runtime_arg = DeclareLaunchArgument(
        'debug_runtime',
        default_value=defaults['debug_runtime'],
        description='Enable runtime debug logs from planner node'
    )

    return LaunchDescription([
        profile_arg,
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
        plan_rate_arg,
        horizon_arg,
        dt_arg,
        control_weight_arg,
        risk_weight_state_arg,
        risk_weight_obs_arg,
        ambiguity_weight_arg,
        goal_sigma_uv_arg,
        use_ambiguity_arg,
        use_obs_risk_arg,
        process_noise_xy_arg,
        process_noise_theta_arg,
        obs_noise_uv_arg,
        optimizer_backend_arg,
        optimizer_maxiter_arg,
        optimizer_gtol_arg,
        optimizer_warm_start_arg,
        debug_runtime_arg,
        OpaqueFunction(function=_launch_setup),
    ])
