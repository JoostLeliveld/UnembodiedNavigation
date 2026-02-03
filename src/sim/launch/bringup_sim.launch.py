from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.conditions import IfCondition



def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_lidar_arg = DeclareLaunchArgument(
        "use_lidar",
        default_value="true",
        description="Enable LiDAR sensor plugin in the URDF",
    )
    use_lidar = LaunchConfiguration("use_lidar")
    bridge_scan_arg = DeclareLaunchArgument(
        "bridge_scan",
        default_value="true",
        description="Bridge /scan between Gazebo and ROS",
    )
    bridge_scan = LaunchConfiguration("bridge_scan")

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
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "use_lidar": use_lidar,
        }.items(),
    )

    spawn = TimerAction(
        period=5.0,   
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-name", "turtlebot3",
                    "-topic", "robot_description",
                    "-x", "0.0",
                    "-y", "0.0",
                    "-z", "0.05"
                ],
                output="screen"
            )
        ]
    )



    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/model/turtlebot3/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/model/turtlebot3/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/model/turtlebot3/odometry_tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/model/turtlebot3/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/external_camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image",
            "/external_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock",
        ],
        remappings=[
            ("/model/turtlebot3/cmd_vel", "/cmd_vel"),
            ("/model/turtlebot3/odometry", "/odom"),
            ("/model/turtlebot3/odometry_tf", "/tf"),
            ("/model/turtlebot3/joint_states", "/joint_states"),
        ],
        output="screen",
    )
    ros_gz_scan_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
        ],
        output="screen",
        condition=IfCondition(bridge_scan),
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_lidar_arg,
        bridge_scan_arg,
        gazebo,
        robot_description,
        spawn,
        ros_gz_bridge,
        ros_gz_scan_bridge,
    ])
