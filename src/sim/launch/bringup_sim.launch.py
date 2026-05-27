from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit



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
    bridge_contacts_arg = DeclareLaunchArgument(
        "bridge_contacts",
        default_value="true",
        description="Bridge Gazebo world contact events to /world_contacts",
    )
    bridge_contacts = LaunchConfiguration("bridge_contacts")
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="warehouse_occ_light.world.sdf",
        description="World file under sim/gazebo_worlds/worlds",
    )
    world = LaunchConfiguration("world")
    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value=PythonExpression([
            "'",
            LaunchConfiguration("world"),
            "'.replace('.world.sdf','').replace('.sdf','')"
        ]),
        description="World name (used for /world/<name>/clock bridging)",
    )
    world_name = LaunchConfiguration("world_name")
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo server-only if true",
    )
    headless = LaunchConfiguration("headless")
    reset_world_arg = DeclareLaunchArgument(
        "reset_world",
        default_value="true",
        description="Reset the Gazebo world on launch",
    )
    reset_world = LaunchConfiguration("reset_world")
    spawn_x_arg = DeclareLaunchArgument(
        "spawn_x",
        default_value="0.0",
        description="Robot spawn x position",
    )
    spawn_y_arg = DeclareLaunchArgument(
        "spawn_y",
        default_value="0.0",
        description="Robot spawn y position",
    )
    spawn_z_arg = DeclareLaunchArgument(
        "spawn_z",
        default_value="0.05",
        description="Robot spawn z position",
    )
    spawn_yaw_arg = DeclareLaunchArgument(
        "spawn_yaw",
        default_value="0.0",
        description="Robot spawn yaw (radians)",
    )
    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_z = LaunchConfiguration("spawn_z")
    spawn_yaw = LaunchConfiguration("spawn_yaw")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("sim"),
                "launch",
                "gazebo.launch.py"
            ])
        ),
        launch_arguments={
            "world": world,
            "headless": headless,
        }.items(),
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

    wait_for_clock = Node(
        package="sim",
        executable="wait_for_clock",
        name="wait_for_clock",
        output="screen",
        parameters=[{
            "topic": "/clock",
            "timeout_s": 0.0,
        }],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "turtlebot3",
            "-topic", "robot_description",
            "-x", spawn_x,
            "-y", spawn_y,
            "-z", spawn_z,
            "-Y", spawn_yaw,
        ],
        output="screen"
    )

    reset_world_node = Node(
        package="sim",
        executable="reset_world",
        name="reset_world",
        parameters=[{
            "world_name": world_name,
            "reset_all": True,
        }],
        output="screen",
        condition=IfCondition(reset_world),
    )

    reset_after_clock = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_clock,
            on_exit=[reset_world_node]
        ),
        condition=IfCondition(reset_world),
    )

    spawn_after_reset = RegisterEventHandler(
        OnProcessExit(
            target_action=reset_world_node,
            on_exit=[spawn]
        ),
        condition=IfCondition(reset_world),
    )

    spawn_after_clock = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_clock,
            on_exit=[spawn]
        ),
        condition=UnlessCondition(reset_world),
    )



    clock_arg = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'"
    ])
    set_pose_service_arg = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/set_pose@ros_gz_interfaces/srv/SetEntityPose@gz.msgs.Pose@gz.msgs.Boolean'"
    ])
    control_service_arg = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/control@ros_gz_interfaces/srv/ControlWorld@gz.msgs.WorldControl@gz.msgs.Boolean'"
    ])
    clock_remap_src = PythonExpression([
        "'/world/' + '",
        world_name,
        "' + '/clock'"
    ])

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/model/turtlebot3/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/turtlebot3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/model/turtlebot3/odometry_tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/model/turtlebot3/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/external_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/external_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/external_camera/segmentation/colored_map@sensor_msgs/msg/Image[gz.msgs.Image",
            "/external_camera/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image",
            clock_arg,
            set_pose_service_arg,
            control_service_arg,
        ],
        remappings=[
            ("/model/turtlebot3/cmd_vel", "/cmd_vel"),
            ("/model/turtlebot3/odometry", "/odom"),
            ("/model/turtlebot3/odometry_tf", "/tf"),
            ("/model/turtlebot3/joint_states", "/joint_states"),
            (clock_remap_src, "/clock"),
        ],
        output="screen",
    )
    ros_gz_contact_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            PythonExpression([
                "'/world/' + '",
                world_name,
                "' + '/physics/contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'",
            ]),
        ],
        remappings=[
            (PythonExpression(["'/world/' + '", world_name, "' + '/physics/contacts'"]), "/world_contacts"),
        ],
        output="screen",
        condition=IfCondition(bridge_contacts),
    )
    ros_gz_scan_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        ],
        output="screen",
        condition=IfCondition(bridge_scan),
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_lidar_arg,
        bridge_scan_arg,
        world_arg,
        world_name_arg,
        headless_arg,
        bridge_contacts_arg,
        reset_world_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        spawn_yaw_arg,
        gazebo,
        robot_description,
        wait_for_clock,
        reset_after_clock,
        spawn_after_reset,
        spawn_after_clock,
        ros_gz_bridge,
        ros_gz_contact_bridge,
        ros_gz_scan_bridge,
    ])
