from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    sim_pkg = FindPackageShare('sim')
    
    # 1. Launch Simulation (Gazebo + Robot + Bridge)
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        )
    )
    
    # 2. Nodes to launch
    # Delayed launch to ensure sim is ready (optional but good practice)
    nodes_group = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='estimation',
                executable='estimation_node',
                name='estimation_node',
                output='screen'
            ),
            Node(
                package='perception',
                executable='perception_node',
                name='perception_node',
                output='screen'
            ),
            Node(
                package='mapping',
                executable='mapping_node',
                name='mapping_node',
                output='screen'
            ),
            Node(
                package='planning',
                executable='astar_planner',
                name='astar_planner',
                output='screen'
            ),
            Node(
                package='control',
                executable='control_node',
                name='control_node',
                output='screen'
            ),
        ]
    )

    return LaunchDescription([
        sim_launch,
        nodes_group
    ])
