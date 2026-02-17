from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit


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

    if planner == 'astar':
        planner_node = Node(
            package='planning',
            executable='astar_planner',
            name='astar_planner',
            output='screen',
            parameters=[{'use_sim_time': cfg['use_sim_time']}],
        )
    elif planner in ('efe1', 'efe2', 'mpc', 'efer'):
        planner_modes = {
            'efe1': {
                'executable': 'efe_planner',
                'name': 'efe1_planner',
                'params': {
                    'approx_method': 'ET1',
                    'add_ambiguity': cfg['add_ambiguity'],
                    'use_ambiguity': cfg['use_ambiguity'],
                    'use_obs_risk': cfg['use_obs_risk'],
                },
            },
            'efe2': {
                'executable': 'efe_planner',
                'name': 'efe2_planner',
                'params': {
                    'approx_method': 'ET2',
                    'add_ambiguity': cfg['add_ambiguity'],
                    'use_ambiguity': cfg['use_ambiguity'],
                    'use_obs_risk': cfg['use_obs_risk'],
                },
            },
            'mpc': {
                'executable': 'mpc_planner',
                'name': 'mpc_planner',
                'params': {
                    'approx_method': 'ET1',
                    'add_ambiguity': False,
                    'use_ambiguity': False,
                    'use_obs_risk': True,
                },
            },
            'efer': {
                'executable': 'efer_planner',
                'name': 'efer_planner',
                'params': {
                    'approx_method': 'ET2',
                    'add_ambiguity': False,
                    'use_ambiguity': False,
                    'use_obs_risk': True,
                },
            },
        }
        mode = planner_modes[planner]
        planner_params = {
            'use_sim_time': cfg['use_sim_time'],
            'use_pixel_correction': cfg['use_pixel_correction'],
            'pixel_timeout_s': cfg['pixel_timeout_s'],
            'boundary_weight': cfg['boundary_weight'],
            'obs_mode': cfg['obs_mode'],
            'process_noise_xy': cfg['process_noise_xy'],
            'process_noise_theta': cfg['process_noise_theta'],
            'obs_noise_uv': cfg['obs_noise_uv'],
            'obs_noise_yaw': cfg['obs_noise_yaw'],
            'optimizer_backend': cfg['optimizer_backend'],
            'optimizer_maxiter': cfg['optimizer_maxiter'],
            'optimizer_gtol': cfg['optimizer_gtol'],
            'optimizer_warm_start': cfg['optimizer_warm_start'],
            **cfg['camera_params'],
            **mode['params'],
        }
        planner_node = Node(
            package='planning',
            executable=mode['executable'],
            name=mode['name'],
            output='screen',
            parameters=[planner_params],
        )
    else:
        raise RuntimeError("planner must be 'astar', 'efe1', 'efe2', 'mpc', 'efer', or 'auto'")

    control_node = Node(
        package='control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{'use_sim_time': cfg['use_sim_time']}],
    )

    after_odom = []
    after_odom.extend(select_perception_nodes_for_mode(cfg, shared_nodes))
    after_odom.extend([
        shared_nodes['pixel_to_bev'],
        shared_nodes['boundary_cost_node'],
        planner_node,
        control_node,
        shared_nodes['mission_node'],
        shared_nodes['logger_node'],
    ])
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
        default_value='astar',
        description='Planner: astar | efe1 | efe2 | mpc | efer | auto'
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
        description='Perception backend in pixel mode: homography | aruco'
    )
    seed_arg = DeclareLaunchArgument('seed', default_value='0')
    pixel_noise_arg = DeclareLaunchArgument('pixel_noise_sigma', default_value='0.0')
    transform_noise_arg = DeclareLaunchArgument('transform_noise_sigma', default_value='0.0')
    use_pixel_correction_arg = DeclareLaunchArgument(
        'use_pixel_correction',
        default_value='true',
        description='Apply pixel-space correction in EFE planner'
    )
    pixel_timeout_arg = DeclareLaunchArgument('pixel_timeout_s', default_value='0.5')
    add_ambiguity_arg = DeclareLaunchArgument(
        'add_ambiguity',
        default_value='true',
        description='Include ambiguity term in EFE objective'
    )
    use_ambiguity_arg = DeclareLaunchArgument(
        'use_ambiguity',
        default_value='true',
        description='Enable ambiguity computation in planner'
    )
    use_obs_risk_arg = DeclareLaunchArgument(
        'use_obs_risk',
        default_value='true',
        description='Enable observation-space risk term'
    )
    boundary_weight_arg = DeclareLaunchArgument(
        'boundary_weight',
        default_value='1.0',
        description='Boundary/costmap penalty weight for EFE planner'
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
        description="Observation mode for EFE planner: 'uv' or 'uvt'"
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
        description='Maximum optimizer iterations for planner'
    )
    optimizer_gtol_arg = DeclareLaunchArgument(
        'optimizer_gtol',
        default_value='1e-4',
        description='Optimizer gradient tolerance for planner'
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
        use_pixel_correction_arg,
        pixel_timeout_arg,
        add_ambiguity_arg,
        use_ambiguity_arg,
        use_obs_risk_arg,
        boundary_weight_arg,
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
