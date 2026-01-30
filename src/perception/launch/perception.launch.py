from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="perception",
            executable="camera_passthrough",
            name="camera_passthrough",
            output="screen",
            parameters=[
                {
                    "input_topic": "/external_camera/image_raw",
                    "output_topic": "/perception/image_raw",
                }
            ],
        )
    ])
