from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    return LaunchDescription([
        use_sim_time,
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_bev_to_odom',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                '--frame-id', 'map_bev',
                '--child-frame-id', 'odom',
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_bev_to_camera',
            # Camera at (-3, -3, 6) in map_bev/odom frame, looking at (1.5, 1.5, 0)
            # Yaw: 0.7854 rad (45°) = looking toward +X+Y diagonal
            # Pitch: -0.756 rad (-43.3°) = looking ~45° down from horizontal
            # Quaternion from roll=0, pitch=-0.756, yaw=0.7854
            arguments=[
                '--x', '-3', '--y', '-3', '--z', '6.0',
                '--qx', '0.141234', '--qy', '-0.340969', '--qz', '0.355669', '--qw', '0.858658',
                '--frame-id', 'map_bev',
                '--child-frame-id', 'external_camera/link/camera',
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        )
    ])
