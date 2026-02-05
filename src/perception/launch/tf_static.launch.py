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
    cam_x_arg = DeclareLaunchArgument('cam_x', default_value='-3')
    cam_y_arg = DeclareLaunchArgument('cam_y', default_value='-3')
    cam_z_arg = DeclareLaunchArgument('cam_z', default_value='6.0')
    cam_qx_arg = DeclareLaunchArgument('cam_qx', default_value='0.141234')
    cam_qy_arg = DeclareLaunchArgument('cam_qy', default_value='-0.340969')
    cam_qz_arg = DeclareLaunchArgument('cam_qz', default_value='0.355669')
    cam_qw_arg = DeclareLaunchArgument('cam_qw', default_value='0.858658')

    return LaunchDescription([
        use_sim_time,
        cam_x_arg,
        cam_y_arg,
        cam_z_arg,
        cam_qx_arg,
        cam_qy_arg,
        cam_qz_arg,
        cam_qw_arg,
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
                '--x', LaunchConfiguration('cam_x'),
                '--y', LaunchConfiguration('cam_y'),
                '--z', LaunchConfiguration('cam_z'),
                '--qx', LaunchConfiguration('cam_qx'),
                '--qy', LaunchConfiguration('cam_qy'),
                '--qz', LaunchConfiguration('cam_qz'),
                '--qw', LaunchConfiguration('cam_qw'),
                '--frame-id', 'map_bev',
                '--child-frame-id', 'external_camera/link/camera',
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        )
    ])
