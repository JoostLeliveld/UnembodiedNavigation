from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    sim_pkg = FindPackageShare("sim")

    world = PathJoinSubstitution([
        sim_pkg,
        "gazebo_worlds",
        "worlds",
        "empty.world.sdf"
    ])

    sim_pkg_share = get_package_share_directory("sim")

    gz_resource_paths = [
        os.path.join(sim_pkg_share, "models"),
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
            "gz_args": ["-r ", world],
        }.items()
    )

    return LaunchDescription([
        set_gz_resource_path,
        gazebo,
    ])
