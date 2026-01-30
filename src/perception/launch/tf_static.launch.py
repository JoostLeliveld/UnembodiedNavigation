from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_cam_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map_cam', 'odom']
        )
    ])
