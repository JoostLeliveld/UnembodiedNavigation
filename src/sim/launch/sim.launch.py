from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node



def generate_launch_description():

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("sim"),
                "launch",
                "gazebo.launch.py"
            ])
        )
    )

    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("sim"),
                "launch",
                "robot_description.launch.py"
            ])
        )
    )

    spawn = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-name", "turtlebot3",
                    "-topic", "robot_description"
                ],
                output="screen"
            )
        ]
    )
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/external_camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image",
        ],
        output="screen",
    )
    return LaunchDescription([
        gazebo,
        robot_description,
        spawn,
        ros_gz_bridge,
    ])
