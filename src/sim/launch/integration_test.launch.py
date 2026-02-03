from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to control RViz'
    )
    use_rviz = LaunchConfiguration('use_rviz')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_lidar_arg = DeclareLaunchArgument(
        'use_lidar',
        default_value='true',
        description='Enable LiDAR sensor plugin in the URDF'
    )
    use_lidar = LaunchConfiguration('use_lidar')

    sim_pkg = FindPackageShare('sim')
    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_lidar': use_lidar,
        }.items(),
    )


    viz_pkg = FindPackageShare('visualization')
    viz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([viz_pkg, 'launch', 'viz.launch.py'])
        ),
        condition=IfCondition(use_rviz),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    perception_pkg = FindPackageShare('perception')
    tf_static = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_pkg, 'launch', 'tf_static.launch.py']),
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    vision_pose_node = Node(
        package='perception',
        executable='homography_sim_node', 
        name='homography_sim_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    occupancy_mapper = Node(
        package='mapping',
        executable='occupancy_mapper',
        name='occupancy_mapper',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    costmap_node = Node(
        package='mapping',
        executable='costmap_node',
        name='costmap_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    astar_planner = Node(
        package='planning',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    control_node = Node(
        package='control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    mission_node = Node(
        package='perception',
        executable='mission_node',
        name='mission_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    nodes_group = TimerAction(
        period=5.0,
        actions=[vision_pose_node, occupancy_mapper, costmap_node, astar_planner, control_node, mission_node]
    )

    return LaunchDescription([
        use_rviz_arg,
        use_sim_time_arg,
        use_lidar_arg,
        bringup_sim,
        tf_static,
        viz_launch,
        nodes_group
    ])
