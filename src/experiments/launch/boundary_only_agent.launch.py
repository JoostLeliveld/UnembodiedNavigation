from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit

# General-purpose agent launch.
# This remains the broad superset interface for development/debugging.
# Paper experiments should use `investigative_agent.launch.py`.


def _launch_setup(context, *args, **kwargs):
    from experiments.core.boundary_launch_common import (
        parse_common_launch_config,
        resolve_world_setup,
        build_shared_nodes,
        select_perception_nodes_for_mode,
    )

    cfg = resolve_world_setup(parse_common_launch_config(context))
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
        parameters=[{
            'use_sim_time': cfg['use_sim_time'],
            'plan_rate': cfg['plan_rate'],
            'horizon': cfg['horizon'],
            'dt': cfg['dt'],
            'control_weight': cfg['control_weight'],
            'use_pixel_correction': cfg['use_pixel_correction'],
            'pixel_timeout_s': cfg['pixel_timeout_s'],
            'pixel_correction_approx': cfg['pixel_correction_approx'],
            'skip_stale_pixel_correction': cfg['skip_stale_pixel_correction'],
            'min_state_cov': cfg['min_state_cov'],
            'debug_runtime': cfg['debug_runtime'],
            'boundary_weight': cfg['boundary_weight'],
            'obs_mode': cfg['obs_mode'],
            'process_noise_xy': cfg['process_noise_xy'],
            'process_noise_theta': cfg['process_noise_theta'],
            'obs_noise_uv': cfg['obs_noise_uv'],
            'obs_noise_yaw': cfg['obs_noise_yaw'],
            'goal_sigma_uv': cfg['goal_sigma_uv'],
            'goal_sigma_yaw': cfg['goal_sigma_yaw'],
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
            'visibility_gp_seed': cfg['visibility_gp_seed'],
            'visibility_r_bad_uv': cfg['visibility_r_bad_uv'],
            'visibility_r_bad_yaw': cfg['visibility_r_bad_yaw'],
            'visibility_cov_pos_scale': cfg['visibility_cov_pos_scale'],
            'visibility_cov_theta_scale': cfg['visibility_cov_theta_scale'],
            'optimizer_backend': cfg['optimizer_backend'],
            'optimizer_maxiter': cfg['optimizer_maxiter'],
            'optimizer_gtol': cfg['optimizer_gtol'],
            'optimizer_warm_start': cfg['optimizer_warm_start'],
            **cfg['camera_params'],
            **planner_params[planner],
        }],
    )

    after_odom = []
    after_odom.extend(select_perception_nodes_for_mode(cfg, shared_nodes))
    after_odom.append(shared_nodes['pixel_to_bev'])
    if cfg['publish_static_costmap']:
        after_odom.append(shared_nodes['boundary_cost_node'])
    after_odom.extend([agent_node, shared_nodes['mission_node'], shared_nodes['logger_node']])
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


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for boundary-only pipeline'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('visualization'), 'rviz', 'boundary_only_camera.rviz'
        ]),
        description='RViz config file to load when use_rviz=true'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    state_source_arg = DeclareLaunchArgument(
        'state_source',
        default_value='oracle',
        description='State source: oracle | pixel'
    )
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='efe2',
        description='Planner: efe1 | efe2 | mpc | efer'
    )
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.world.sdf',
        description='World file under sim/gazebo_worlds/worlds'
    )
    world_profiles_arg = DeclareLaunchArgument(
        'world_profiles',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'world_profiles.yaml'
        ]),
        description='YAML file describing per-world profiles'
    )
    tasks_yaml_arg = DeclareLaunchArgument(
        'tasks_yaml',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'tasks.yaml'
        ]),
        description='YAML file describing per-world tasks'
    )
    task_arg = DeclareLaunchArgument(
        'task',
        default_value='',
        description='Task name; defaults to first task in tasks.yaml for the world'
    )
    perception_backend_arg = DeclareLaunchArgument(
        'perception_backend',
        default_value='homography',
        description='Perception backend in pixel mode: homography | aruco | color'
    )
    seed_arg = DeclareLaunchArgument('seed', default_value='0')
    pixel_noise_arg = DeclareLaunchArgument(
        'pixel_noise_sigma',
        default_value='0.0',
        description='Advanced: extra noise in pixel_to_bev_state_node (keep 0.0 for study runs; prefer sensor_pixel_noise_sigma)'
    )
    transform_noise_arg = DeclareLaunchArgument(
        'transform_noise_sigma',
        default_value='0.0',
        description='Advanced: extra metric noise after pixel->BEV transform (keep 0.0 for study runs)'
    )
    sensor_pixel_noise_arg = DeclareLaunchArgument(
        'sensor_pixel_noise_sigma',
        default_value='0.0',
        description='Primary study noise knob: pixel noise std injected at homography measurement source (/perception/pixel_pose)'
    )
    odom_wait_timeout_arg = DeclareLaunchArgument(
        'odom_wait_timeout_s',
        default_value='25.0',
        description='Timeout for startup /odom gate in seconds (0 disables timeout)'
    )
    odom_wait_min_messages_arg = DeclareLaunchArgument(
        'odom_wait_min_messages',
        default_value='1',
        description='Number of /odom messages required to open startup gate'
    )
    odom_wait_require_pose_match_arg = DeclareLaunchArgument(
        'odom_wait_require_pose_match',
        default_value='false',
        description='Require startup /odom pose to match expected (usually false for robustness)'
    )
    use_pixel_correction_arg = DeclareLaunchArgument(
        'use_pixel_correction',
        default_value='true',
        description='Apply pixel-space correction in EFE agent'
    )
    pixel_timeout_arg = DeclareLaunchArgument('pixel_timeout_s', default_value='0.5')
    pixel_correction_approx_arg = DeclareLaunchArgument(
        'pixel_correction_approx',
        default_value='AUTO',
        description='Pixel correction moment approximation: AUTO | ET1 | ET2 | UT (AUTO matches planner approx)'
    )
    skip_stale_pixel_correction_arg = DeclareLaunchArgument(
        'skip_stale_pixel_correction',
        default_value='true',
        description='Skip correction when pixel measurement age exceeds pixel_timeout_s'
    )
    use_ambiguity_arg = DeclareLaunchArgument(
        'use_ambiguity',
        default_value='true',
        description='Enable ambiguity term in the EFE objective'
    )
    use_obs_risk_arg = DeclareLaunchArgument(
        'use_obs_risk',
        default_value='true',
        description='Enable observation-space risk term'
    )
    boundary_weight_arg = DeclareLaunchArgument(
        'boundary_weight',
        default_value='1.0',
        description='Boundary/costmap penalty weight for EFE agent'
    )
    publish_static_costmap_arg = DeclareLaunchArgument(
        'publish_static_costmap',
        default_value='true',
        description='Publish static /costmap from boundary_cost_node'
    )
    costmap_min_x_arg = DeclareLaunchArgument(
        'costmap_min_x',
        default_value='-5.0',
        description='Costmap lower x bound in map_bev frame'
    )
    costmap_max_x_arg = DeclareLaunchArgument(
        'costmap_max_x',
        default_value='5.0',
        description='Costmap upper x bound in map_bev frame'
    )
    costmap_min_y_arg = DeclareLaunchArgument(
        'costmap_min_y',
        default_value='-5.0',
        description='Costmap lower y bound in map_bev frame'
    )
    costmap_max_y_arg = DeclareLaunchArgument(
        'costmap_max_y',
        default_value='5.0',
        description='Costmap upper y bound in map_bev frame'
    )
    costmap_wall_margin_arg = DeclareLaunchArgument(
        'costmap_wall_margin',
        default_value='0.2',
        description='Lethal wall margin inside costmap bounds'
    )
    costmap_obstacle_enabled_arg = DeclareLaunchArgument(
        'costmap_obstacle_enabled',
        default_value='true',
        description='Enable static circular obstacle in boundary costmap'
    )
    costmap_obstacle_center_x_arg = DeclareLaunchArgument(
        'costmap_obstacle_center_x',
        default_value='1.5',
        description='Static obstacle center x in map_bev'
    )
    costmap_obstacle_center_y_arg = DeclareLaunchArgument(
        'costmap_obstacle_center_y',
        default_value='1.5',
        description='Static obstacle center y in map_bev'
    )
    costmap_obstacle_radius_arg = DeclareLaunchArgument(
        'costmap_obstacle_radius',
        default_value='0.4',
        description='Static obstacle radius in meters'
    )
    costmap_obstacle_value_arg = DeclareLaunchArgument(
        'costmap_obstacle_value',
        default_value='100',
        description='Static obstacle occupancy value'
    )
    obs_mode_arg = DeclareLaunchArgument(
        'obs_mode',
        default_value='uv',
        description="Observation mode for EFE: 'uv' or 'uvt'"
    )
    plan_rate_arg = DeclareLaunchArgument(
        'plan_rate',
        default_value='5.0',
        description='Replanning rate in Hz'
    )
    horizon_arg = DeclareLaunchArgument(
        'horizon',
        default_value='10',
        description='Planning horizon length in steps'
    )
    dt_arg = DeclareLaunchArgument(
        'dt',
        default_value='0.2',
        description='Planner discretization step in seconds'
    )
    control_weight_arg = DeclareLaunchArgument(
        'control_weight',
        default_value='0.1',
        description='Quadratic control penalty weight'
    )
    risk_weight_state_arg = DeclareLaunchArgument(
        'risk_weight_state',
        default_value='0.0',
        description='State-space risk weight'
    )
    risk_weight_obs_arg = DeclareLaunchArgument(
        'risk_weight_obs',
        default_value='1.0',
        description='Observation-space risk weight'
    )
    ambiguity_weight_arg = DeclareLaunchArgument(
        'ambiguity_weight',
        default_value='1.0',
        description='Ambiguity term weight'
    )
    use_visibility_model_arg = DeclareLaunchArgument(
        'use_visibility_model',
        default_value='false',
        description='Enable fixed GP visibility model in planner objective/correction'
    )
    visibility_weight_arg = DeclareLaunchArgument(
        'visibility_weight',
        default_value='4.0',
        description='Visibility penalty weight w_vis*(1-p_vis)'
    )
    visibility_occ_center_x_arg = DeclareLaunchArgument('visibility_occ_center_x', default_value='-1.2')
    visibility_occ_center_y_arg = DeclareLaunchArgument('visibility_occ_center_y', default_value='-1.8')
    visibility_occ_radius_arg = DeclareLaunchArgument('visibility_occ_radius', default_value='0.9')
    visibility_occ_tau_arg = DeclareLaunchArgument('visibility_occ_tau', default_value='0.15')
    visibility_gp_length_scale_arg = DeclareLaunchArgument('visibility_gp_length_scale', default_value='1.4')
    visibility_gp_noise_var_arg = DeclareLaunchArgument('visibility_gp_noise_var', default_value='0.15')
    visibility_gp_seed_arg = DeclareLaunchArgument('visibility_gp_seed', default_value='0')
    goal_sigma_uv_arg = DeclareLaunchArgument(
        'goal_sigma_uv',
        default_value='0.0',
        description='Goal observation std in pixel u/v (0 means use obs_noise_uv)'
    )
    goal_sigma_yaw_arg = DeclareLaunchArgument(
        'goal_sigma_yaw',
        default_value='100.0',
        description='Goal observation std for yaw (uvt mode); large values effectively disable yaw goal pressure'
    )
    min_state_cov_arg = DeclareLaunchArgument(
        'min_state_cov',
        default_value='1e-6',
        description='Minimum diagonal covariance floor in planner belief'
    )
    debug_runtime_arg = DeclareLaunchArgument(
        'debug_runtime',
        default_value='false',
        description='Enable periodic runtime and optimizer debug logs'
    )
    process_noise_xy_arg = DeclareLaunchArgument(
        'process_noise_xy',
        default_value='0.01',
        description='Process noise std for x/y state dynamics'
    )
    process_noise_theta_arg = DeclareLaunchArgument(
        'process_noise_theta',
        default_value='0.02',
        description='Process noise std for yaw state dynamics'
    )
    obs_noise_uv_arg = DeclareLaunchArgument(
        'obs_noise_uv',
        default_value='2.0',
        description='Observation noise std in pixel u/v'
    )
    obs_noise_yaw_arg = DeclareLaunchArgument(
        'obs_noise_yaw',
        default_value='0.05',
        description='Observation noise std for yaw observation (uvt mode)'
    )
    optimizer_backend_arg = DeclareLaunchArgument(
        'optimizer_backend',
        default_value='auto',
        description="Optimizer backend: 'auto', 'jax', or 'scipy'"
    )
    optimizer_maxiter_arg = DeclareLaunchArgument(
        'optimizer_maxiter',
        default_value='50',
        description='Maximum optimizer iterations for agent planner'
    )
    optimizer_gtol_arg = DeclareLaunchArgument(
        'optimizer_gtol',
        default_value='1e-4',
        description='Optimizer gradient tolerance for agent planner'
    )
    optimizer_warm_start_arg = DeclareLaunchArgument(
        'optimizer_warm_start',
        default_value='true',
        description='Use warm-start controls between planning cycles'
    )
    aruco_dict_arg = DeclareLaunchArgument(
        'aruco_dict',
        default_value='DICT_4X4_50',
        description='ArUco/AprilTag dictionary for aruco perception backend'
    )
    target_marker_id_arg = DeclareLaunchArgument(
        'target_marker_id',
        default_value='0',
        description='Marker id to track; use -1 to track the largest detected marker'
    )
    publish_yaw_from_marker_arg = DeclareLaunchArgument(
        'publish_yaw_from_marker',
        default_value='true',
        description='Estimate and publish yaw from marker corners (aruco backend)'
    )

    return LaunchDescription([
        use_rviz_arg,
        rviz_config_arg,
        use_sim_time_arg,
        state_source_arg,
        planner_arg,
        world_arg,
        world_profiles_arg,
        tasks_yaml_arg,
        task_arg,
        perception_backend_arg,
        seed_arg,
        pixel_noise_arg,
        transform_noise_arg,
        sensor_pixel_noise_arg,
        odom_wait_timeout_arg,
        odom_wait_min_messages_arg,
        odom_wait_require_pose_match_arg,
        use_pixel_correction_arg,
        pixel_timeout_arg,
        pixel_correction_approx_arg,
        skip_stale_pixel_correction_arg,
        use_ambiguity_arg,
        use_obs_risk_arg,
        boundary_weight_arg,
        publish_static_costmap_arg,
        costmap_min_x_arg,
        costmap_max_x_arg,
        costmap_min_y_arg,
        costmap_max_y_arg,
        costmap_wall_margin_arg,
        costmap_obstacle_enabled_arg,
        costmap_obstacle_center_x_arg,
        costmap_obstacle_center_y_arg,
        costmap_obstacle_radius_arg,
        costmap_obstacle_value_arg,
        obs_mode_arg,
        plan_rate_arg,
        horizon_arg,
        dt_arg,
        control_weight_arg,
        risk_weight_state_arg,
        risk_weight_obs_arg,
        ambiguity_weight_arg,
        use_visibility_model_arg,
        visibility_weight_arg,
        visibility_occ_center_x_arg,
        visibility_occ_center_y_arg,
        visibility_occ_radius_arg,
        visibility_occ_tau_arg,
        visibility_gp_length_scale_arg,
        visibility_gp_noise_var_arg,
        visibility_gp_seed_arg,
        goal_sigma_uv_arg,
        goal_sigma_yaw_arg,
        min_state_cov_arg,
        debug_runtime_arg,
        process_noise_xy_arg,
        process_noise_theta_arg,
        obs_noise_uv_arg,
        obs_noise_yaw_arg,
        optimizer_backend_arg,
        optimizer_maxiter_arg,
        optimizer_gtol_arg,
        optimizer_warm_start_arg,
        aruco_dict_arg,
        target_marker_id_arg,
        publish_yaw_from_marker_arg,
        OpaqueFunction(function=_launch_setup),
    ])
