from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from experiments.core.boundary_launch_common import PAPER_LAUNCH_DEFAULTS


def generate_launch_description():
    """Paper-focused agent launch (no-costmap, JAX backend).

    This is a thin wrapper around `boundary_only_agent.launch.py` that hides
    boundary/costmap controls and fixes a few study assumptions:
    - no costmap / no boundary penalty
    - JAX optimizer backend
    """

    defaults = PAPER_LAUNCH_DEFAULTS

    boundary_agent_launch = PathJoinSubstitution([
        FindPackageShare('experiments'),
        'launch',
        'boundary_only_agent.launch.py',
    ])

    # Study-facing arguments only (trimmed interface).
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz (off by default for study runs)'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('visualization'), 'rviz', 'boundary_only_camera.rviz'
        ]),
        description='RViz config file when use_rviz=true'
    )
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value=defaults['planner'],
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

    state_source_arg = DeclareLaunchArgument(
        'state_source',
        default_value=defaults['state_source'],
        description='State source: oracle | pixel'
    )
    perception_backend_arg = DeclareLaunchArgument(
        'perception_backend',
        default_value=defaults['perception_backend'],
        description='Perception backend in pixel mode: homography | aruco'
    )
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

    obs_mode_arg = DeclareLaunchArgument('obs_mode', default_value=defaults['obs_mode'])
    plan_rate_arg = DeclareLaunchArgument('plan_rate', default_value=defaults['plan_rate'])
    horizon_arg = DeclareLaunchArgument('horizon', default_value=defaults['horizon'])
    dt_arg = DeclareLaunchArgument('dt', default_value=defaults['dt'])
    control_weight_arg = DeclareLaunchArgument('control_weight', default_value=defaults['control_weight'])
    risk_weight_state_arg = DeclareLaunchArgument('risk_weight_state', default_value=defaults['risk_weight_state'])
    risk_weight_obs_arg = DeclareLaunchArgument('risk_weight_obs', default_value=defaults['risk_weight_obs'])
    ambiguity_weight_arg = DeclareLaunchArgument('ambiguity_weight', default_value=defaults['ambiguity_weight'])
    goal_sigma_uv_arg = DeclareLaunchArgument('goal_sigma_uv', default_value=defaults['goal_sigma_uv'])
    goal_sigma_yaw_arg = DeclareLaunchArgument('goal_sigma_yaw', default_value=defaults['goal_sigma_yaw'])
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
    obs_noise_yaw_arg = DeclareLaunchArgument('obs_noise_yaw', default_value=defaults['obs_noise_yaw'])

    optimizer_maxiter_arg = DeclareLaunchArgument('optimizer_maxiter', default_value=defaults['optimizer_maxiter'])
    optimizer_gtol_arg = DeclareLaunchArgument('optimizer_gtol', default_value=defaults['optimizer_gtol'])
    optimizer_warm_start_arg = DeclareLaunchArgument('optimizer_warm_start', default_value=defaults['optimizer_warm_start'])

    debug_runtime_arg = DeclareLaunchArgument(
        'debug_runtime',
        default_value=defaults['debug_runtime'],
        description='Enable runtime debug logs from planner node'
    )

    aruco_dict_arg = DeclareLaunchArgument('aruco_dict', default_value=defaults['aruco_dict'])
    target_marker_id_arg = DeclareLaunchArgument('target_marker_id', default_value=defaults['target_marker_id'])
    publish_yaw_from_marker_arg = DeclareLaunchArgument('publish_yaw_from_marker', default_value=defaults['publish_yaw_from_marker'])

    forwarded = {
        'use_rviz': LaunchConfiguration('use_rviz'),
        'rviz_config': LaunchConfiguration('rviz_config'),
        'use_sim_time': LaunchConfiguration('use_sim_time'),
        'planner': LaunchConfiguration('planner'),
        'world': LaunchConfiguration('world'),
        'world_profiles': LaunchConfiguration('world_profiles'),
        'tasks_yaml': LaunchConfiguration('tasks_yaml'),
        'task': LaunchConfiguration('task'),
        'state_source': LaunchConfiguration('state_source'),
        'perception_backend': LaunchConfiguration('perception_backend'),
        'seed': LaunchConfiguration('seed'),
        'sensor_pixel_noise_sigma': LaunchConfiguration('sensor_pixel_noise_sigma'),
        'pixel_noise_sigma': '0.0',
        'transform_noise_sigma': '0.0',
        'odom_wait_timeout_s': LaunchConfiguration('odom_wait_timeout_s'),
        'odom_wait_min_messages': LaunchConfiguration('odom_wait_min_messages'),
        'odom_wait_require_pose_match': LaunchConfiguration('odom_wait_require_pose_match'),
        'use_pixel_correction': LaunchConfiguration('use_pixel_correction'),
        'pixel_timeout_s': LaunchConfiguration('pixel_timeout_s'),
        'pixel_correction_approx': LaunchConfiguration('pixel_correction_approx'),
        'skip_stale_pixel_correction': LaunchConfiguration('skip_stale_pixel_correction'),
        'min_state_cov': LaunchConfiguration('min_state_cov'),
        'obs_mode': LaunchConfiguration('obs_mode'),
        'plan_rate': LaunchConfiguration('plan_rate'),
        'horizon': LaunchConfiguration('horizon'),
        'dt': LaunchConfiguration('dt'),
        'control_weight': LaunchConfiguration('control_weight'),
        'risk_weight_state': LaunchConfiguration('risk_weight_state'),
        'risk_weight_obs': LaunchConfiguration('risk_weight_obs'),
        'ambiguity_weight': LaunchConfiguration('ambiguity_weight'),
        'goal_sigma_uv': LaunchConfiguration('goal_sigma_uv'),
        'goal_sigma_yaw': LaunchConfiguration('goal_sigma_yaw'),
        'use_ambiguity': LaunchConfiguration('use_ambiguity'),
        'use_obs_risk': LaunchConfiguration('use_obs_risk'),
        'process_noise_xy': LaunchConfiguration('process_noise_xy'),
        'process_noise_theta': LaunchConfiguration('process_noise_theta'),
        'obs_noise_uv': LaunchConfiguration('obs_noise_uv'),
        'obs_noise_yaw': LaunchConfiguration('obs_noise_yaw'),
        'optimizer_maxiter': LaunchConfiguration('optimizer_maxiter'),
        'optimizer_gtol': LaunchConfiguration('optimizer_gtol'),
        'optimizer_warm_start': LaunchConfiguration('optimizer_warm_start'),
        'debug_runtime': LaunchConfiguration('debug_runtime'),
        'aruco_dict': LaunchConfiguration('aruco_dict'),
        'target_marker_id': LaunchConfiguration('target_marker_id'),
        'publish_yaw_from_marker': LaunchConfiguration('publish_yaw_from_marker'),
        'optimizer_backend': 'jax',
        'boundary_weight': '0.0',
        'publish_static_costmap': 'false',
    }

    include_boundary_agent = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(boundary_agent_launch),
        launch_arguments=forwarded.items(),
    )

    return LaunchDescription([
        use_rviz_arg,
        rviz_config_arg,
        use_sim_time_arg,
        planner_arg,
        world_arg,
        world_profiles_arg,
        tasks_yaml_arg,
        task_arg,
        state_source_arg,
        perception_backend_arg,
        seed_arg,
        sensor_pixel_noise_arg,
        odom_wait_timeout_arg,
        odom_wait_min_messages_arg,
        odom_wait_require_pose_match_arg,
        use_pixel_correction_arg,
        pixel_timeout_arg,
        pixel_correction_approx_arg,
        skip_stale_pixel_correction_arg,
        min_state_cov_arg,
        obs_mode_arg,
        plan_rate_arg,
        horizon_arg,
        dt_arg,
        control_weight_arg,
        risk_weight_state_arg,
        risk_weight_obs_arg,
        ambiguity_weight_arg,
        goal_sigma_uv_arg,
        goal_sigma_yaw_arg,
        use_ambiguity_arg,
        use_obs_risk_arg,
        process_noise_xy_arg,
        process_noise_theta_arg,
        obs_noise_uv_arg,
        obs_noise_yaw_arg,
        optimizer_maxiter_arg,
        optimizer_gtol_arg,
        optimizer_warm_start_arg,
        debug_runtime_arg,
        aruco_dict_arg,
        target_marker_id_arg,
        publish_yaw_from_marker_arg,
        include_boundary_agent,
    ])
