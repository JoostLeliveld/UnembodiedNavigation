from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import LaunchConfigurationEquals
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit


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
        description='Planner: astar | efe1 | efe2'
    )
    goal_x_arg = DeclareLaunchArgument('goal_x', default_value='3.0')
    goal_y_arg = DeclareLaunchArgument('goal_y', default_value='3.0')
    seed_arg = DeclareLaunchArgument('seed', default_value='0')
    pixel_noise_arg = DeclareLaunchArgument('pixel_noise_sigma', default_value='0.0')
    transform_noise_arg = DeclareLaunchArgument('transform_noise_sigma', default_value='0.0')

    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    state_source = LaunchConfiguration('state_source')
    planner = LaunchConfiguration('planner')
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')
    seed = LaunchConfiguration('seed')
    pixel_noise_sigma = LaunchConfiguration('pixel_noise_sigma')
    transform_noise_sigma = LaunchConfiguration('transform_noise_sigma')

    sim_pkg = FindPackageShare('sim')
    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_lidar': 'false',
            'bridge_scan': 'false',
        }.items(),
    )

    perception_pkg = FindPackageShare('perception')
    tf_static = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_pkg, 'launch', 'tf_static.launch.py'])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
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

    homography_sim = Node(
        package='perception',
        executable='homography_sim_node',
        name='homography_sim_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=LaunchConfigurationEquals('state_source', 'pixel')
    )

    pixel_to_bev = Node(
        package='state',
        executable='pixel_to_bev_state_node',
        name='pixel_to_bev_state_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'state_source': state_source,
            'frame_id': 'map_bev',
            'pixel_noise_sigma': pixel_noise_sigma,
            'transform_noise_sigma': transform_noise_sigma,
            'seed': seed,
        }]
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

    astar_planner = Node(
        package='planning',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=LaunchConfigurationEquals('planner', 'astar')
    )

    efe_planner_efe1 = Node(
        package='planning',
        executable='efe_planner',
        name='efe_planner',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'planner_mode': 'efe1',
        }],
        condition=LaunchConfigurationEquals('planner', 'efe1')
    )

    efe_planner_efe2 = Node(
        package='planning',
        executable='efe_planner',
        name='efe_planner',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'planner_mode': 'efe2',
        }],
        condition=LaunchConfigurationEquals('planner', 'efe2')
    )

    control_node = Node(
        package='control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
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
        parameters=[{'use_sim_time': use_sim_time, 'seed': seed}]
    )

    viz_pkg = FindPackageShare('visualization')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([viz_pkg, 'rviz', 'boundary_only.rviz'])],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=LaunchConfigurationEquals('use_rviz', 'true')
    )

    start_after_odom = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_odom,
            on_exit=[
                homography_sim,
                pixel_to_bev,
                boundary_cost_node,
                astar_planner,
                efe_planner_efe1,
                efe_planner_efe2,
                control_node,
                mission_node,
                logger_node,
                rviz,
            ]
        )
    )

    return LaunchDescription([
        use_rviz_arg,
        use_sim_time_arg,
        state_source_arg,
        goal_x_arg,
        goal_y_arg,
        seed_arg,
        pixel_noise_arg,
        transform_noise_arg,
        bringup_sim,
        tf_static,
        wait_for_odom,
        start_after_odom,
    ])
