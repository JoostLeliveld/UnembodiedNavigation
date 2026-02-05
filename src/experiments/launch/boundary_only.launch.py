from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit


def _as_bool(value: str) -> bool:
    return str(value).lower() == 'true'


def _launch_setup(context, *args, **kwargs):
    from experiments.world_profiles import load_profile, compute_camera_quaternion

    use_sim_time_value = LaunchConfiguration('use_sim_time').perform(context)
    use_sim_time = _as_bool(use_sim_time_value)
    state_source = LaunchConfiguration('state_source').perform(context)
    planner = LaunchConfiguration('planner').perform(context)
    world = LaunchConfiguration('world').perform(context)
    world_profiles_path = LaunchConfiguration('world_profiles').perform(context)
    goal_x = float(LaunchConfiguration('goal_x').perform(context))
    goal_y = float(LaunchConfiguration('goal_y').perform(context))
    seed = int(LaunchConfiguration('seed').perform(context))
    pixel_noise_sigma = float(LaunchConfiguration('pixel_noise_sigma').perform(context))
    transform_noise_sigma = float(LaunchConfiguration('transform_noise_sigma').perform(context))
    use_pixel_correction = _as_bool(LaunchConfiguration('use_pixel_correction').perform(context))
    pixel_timeout_s = float(LaunchConfiguration('pixel_timeout_s').perform(context))
    use_rviz = _as_bool(LaunchConfiguration('use_rviz').perform(context))

    if state_source not in ('oracle', 'pixel'):
        raise RuntimeError("state_source must be 'oracle' or 'pixel'")

    profile = load_profile(world_profiles_path, world)
    spawn = profile['spawn']
    camera = profile['camera']

    if planner == 'auto':
        planner = profile['planner_default']

    sim_pkg = FindPackageShare('sim')
    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': 'true' if use_sim_time else 'false',
            'use_lidar': 'false',
            'bridge_scan': 'false',
            'world': world,
            'world_name': profile['world_name'],
            'spawn_x': str(spawn['x']),
            'spawn_y': str(spawn['y']),
            'spawn_z': str(spawn['z']),
            'spawn_yaw': str(spawn['yaw']),
        }.items(),
    )

    perception_pkg = FindPackageShare('perception')
    tf_args = {'use_sim_time': 'true' if use_sim_time else 'false'}
    cam_pos = camera['cam_pos']
    look_at = camera['look_at']
    img_width = int(camera['img_width'])
    img_height = int(camera['img_height'])
    fov_h_rad = float(camera['fov_h_rad'])
    quat = compute_camera_quaternion(cam_pos, look_at)
    tf_args.update({
        'cam_x': str(cam_pos[0]),
        'cam_y': str(cam_pos[1]),
        'cam_z': str(cam_pos[2]),
        'cam_qx': str(quat[0]),
        'cam_qy': str(quat[1]),
        'cam_qz': str(quat[2]),
        'cam_qw': str(quat[3]),
    })
    camera_params = {
        'cam_pos': cam_pos,
        'look_at': look_at,
        'img_width': img_width,
        'img_height': img_height,
        'fov_h_rad': fov_h_rad,
    }

    tf_static = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_pkg, 'launch', 'tf_static.launch.py'])
        ),
        launch_arguments=tf_args.items(),
    )

    wait_for_odom = Node(
        package='sim',
        executable='wait_for_odom',
        name='wait_for_odom',
        output='screen',
        parameters=[{
            'topic': '/odom',
            'timeout_s': 0.0,
        }]
    )

    homography_params = {'use_sim_time': use_sim_time}
    homography_params.update(camera_params)
    homography_sim = Node(
        package='perception',
        executable='homography_sim_node',
        name='homography_sim_node',
        output='screen',
        parameters=[homography_params],
    )

    pixel_params = {
        'use_sim_time': use_sim_time,
        'state_source': state_source,
        'frame_id': 'map_bev',
        'pixel_noise_sigma': pixel_noise_sigma,
        'transform_noise_sigma': transform_noise_sigma,
        'seed': seed,
    }
    pixel_params.update(camera_params)
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
            'use_sim_time': use_sim_time,
            'frame_id': 'map_bev',
        }]
    )

    if planner == 'astar':
        planner_node = Node(
            package='planning',
            executable='astar_planner',
            name='astar_planner',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        )
    elif planner == 'efe1':
        planner_node = Node(
            package='planning',
            executable='efe_planner',
            name='efe_planner',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'planner_mode': 'efe1',
                'use_pixel_correction': use_pixel_correction,
                'pixel_timeout_s': pixel_timeout_s,
            }],
        )
    elif planner == 'efe2':
        planner_node = Node(
            package='planning',
            executable='efe_planner',
            name='efe_planner',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'planner_mode': 'efe2',
                'use_pixel_correction': use_pixel_correction,
                'pixel_timeout_s': pixel_timeout_s,
            }],
        )
    else:
        raise RuntimeError("planner must be 'astar', 'efe1', 'efe2', or 'auto'")

    control_node = Node(
        package='control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    mission_node = Node(
        package='experiments',
        executable='goal_mission_node',
        name='goal_mission_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'frame_id': 'map_bev',
            'goal_x': goal_x,
            'goal_y': goal_y,
        }]
    )

    logger_node = Node(
        package='experiments',
        executable='experiment_logger',
        name='experiment_logger',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'seed': seed}],
    )

    viz_pkg = FindPackageShare('visualization')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([viz_pkg, 'rviz', 'boundary_only.rviz'])],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    after_odom = []
    if state_source == 'pixel':
        after_odom.append(homography_sim)
    after_odom.extend([
        pixel_to_bev,
        boundary_cost_node,
        planner_node,
        control_node,
        mission_node,
        logger_node,
    ])
    if use_rviz:
        after_odom.append(rviz)

    start_after_odom = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_odom,
            on_exit=after_odom,
        )
    )

    return [
        bringup_sim,
        tf_static,
        wait_for_odom,
        start_after_odom,
    ]


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for boundary-only pipeline'
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
        description='Planner: astar | efe1 | efe2 | auto'
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
    goal_x_arg = DeclareLaunchArgument('goal_x', default_value='3.0')
    goal_y_arg = DeclareLaunchArgument('goal_y', default_value='3.0')
    seed_arg = DeclareLaunchArgument('seed', default_value='0')
    pixel_noise_arg = DeclareLaunchArgument('pixel_noise_sigma', default_value='0.0')
    transform_noise_arg = DeclareLaunchArgument('transform_noise_sigma', default_value='0.0')
    use_pixel_correction_arg = DeclareLaunchArgument(
        'use_pixel_correction',
        default_value='true',
        description='Apply pixel-space correction in EFE planner'
    )
    pixel_timeout_arg = DeclareLaunchArgument('pixel_timeout_s', default_value='0.5')

    return LaunchDescription([
        use_rviz_arg,
        use_sim_time_arg,
        state_source_arg,
        planner_arg,
        world_arg,
        world_profiles_arg,
        goal_x_arg,
        goal_y_arg,
        seed_arg,
        pixel_noise_arg,
        transform_noise_arg,
        use_pixel_correction_arg,
        pixel_timeout_arg,
        OpaqueFunction(function=_launch_setup),
    ])
