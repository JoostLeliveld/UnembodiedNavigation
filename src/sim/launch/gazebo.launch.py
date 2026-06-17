from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    sim_pkg_share = get_package_share_directory("sim")
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="warehouse_aws.world.sdf",
        description="World file under sim/gazebo_worlds/worlds",
    )
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo server-only if true",
    )
    world_path = PathJoinSubstitution([
        FindPackageShare("sim"),
        "gazebo_worlds",
        "worlds",
        LaunchConfiguration("world"),
    ])
    sim_pkg_share_parent = os.path.dirname(sim_pkg_share)

    gz_resource_paths = [
        sim_pkg_share_parent,
        os.path.join(sim_pkg_share, "models"),
        os.path.join(sim_pkg_share, "gazebo_worlds", "models"),
        os.path.join(sim_pkg_share, "gazebo_worlds"),
        os.path.join(sim_pkg_share, "robot_description"),
        sim_pkg_share
    ]

    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=":".join(gz_resource_paths)
    )


    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={
            "gz_args": [
                PythonExpression([
                    "'-r -s ' if '",
                    LaunchConfiguration("headless"),
                    "'.strip().lower() in ('1', 'true', 'yes', 'on') else '-r '",
                ]),
                world_path,
            ],
        }.items()
    )

    return LaunchDescription([
        world_arg,
        headless_arg,
        set_gz_resource_path,
        gazebo,
    ])
