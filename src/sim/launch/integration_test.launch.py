from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # 1. Start Simulation (Gazebo + Robot State Publisher + Bridge)
    sim_pkg = FindPackageShare('sim')
    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        )
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

    # Delayed start to ensure Gazebo is ready (optional but good practice)
    nodes_group = TimerAction(
        period=5.0,
        actions=[occupancy_mapper, costmap_node, astar_planner]
    )

    return LaunchDescription([
        bringup_sim,
        nodes_group
    ])
