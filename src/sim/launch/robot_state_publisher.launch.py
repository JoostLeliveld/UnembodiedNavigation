from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('sim')
    xacro_file = os.path.join(pkg_share, 'robot_description', 'urdf', 'turtlebot3_burger.urdf.xacro')

    robot_description_cmd = Command(['xacro ', xacro_file])

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_cmd}]
    )

    return LaunchDescription([rsp_node])