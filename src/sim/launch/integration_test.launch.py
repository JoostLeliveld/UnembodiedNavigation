from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # Arguments
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to control RViz'
    )
    use_rviz = LaunchConfiguration('use_rviz')

    # 1. Start Simulation (Gazebo + Robot State Publisher + Bridge)
    sim_pkg = FindPackageShare('sim')
    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        )
    )

    # 1.2 Start Visualization
    viz_pkg = FindPackageShare('visualization')
    viz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([viz_pkg, 'launch', 'viz.launch.py'])
        ),
        condition=IfCondition(use_rviz)
    )

    # 1.1 Start Perception (Static TF + Vision Pose)
    perception_pkg = FindPackageShare('perception')
    tf_static = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_pkg, 'launch', 'tf_static.launch.py'])
        )
    )

    vision_pose_node = Node(
        package='perception',
        executable='vision_pose_node',
        name='vision_pose_node',
        output='screen'
    )

    # 2. Start Mapping
    occupancy_mapper = Node(
        package='mapping',
        executable='occupancy_mapper',
        name='occupancy_mapper',
        output='screen'
    )

    costmap_node = Node(
        package='mapping',
        executable='costmap_node',
        name='costmap_node',
        output='screen'
    )

    # 3. Start Planning
    astar_planner = Node(
        package='planning',
        executable='astar_planner',
        name='astar_planner',
        output='screen'
    )

    # 4. Start Control
    control_node = Node(
        package='control',
        executable='control_node',
        name='control_node',
        output='screen'
    )

    # Delayed start to ensure Gazebo is ready (optional but good practice)
    nodes_group = TimerAction(
        period=5.0,
        actions=[vision_pose_node, occupancy_mapper, costmap_node, astar_planner, control_node]
    )

    return LaunchDescription([
        use_rviz_arg,
        bringup_sim,
        tf_static,
        viz_launch,
        nodes_group
    ])
