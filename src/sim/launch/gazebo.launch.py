from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import SetEnvironmentVariable


def generate_launch_description():
    sim_pkg = FindPackageShare("sim")

    world = PathJoinSubstitution([
        sim_pkg,
        "gazebo_worlds",
        "worlds",
        "empty",
        "world.sdf"
    ])

    # IMPORTANT: allow Gazebo to find meshes via package://sim/...
    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[sim_pkg]
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
