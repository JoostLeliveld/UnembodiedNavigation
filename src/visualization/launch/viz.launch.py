from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare('visualization'), 'rviz', 'unembodied_nav.rviz']
    )

    return LaunchDescription([
        use_sim_time_arg,
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{"use_sim_time": use_sim_time}],
            output='screen'
        )
    ])
