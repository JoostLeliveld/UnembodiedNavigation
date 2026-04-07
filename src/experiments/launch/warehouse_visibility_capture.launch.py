from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *args, **kwargs):
    from experiments.core.world_profiles import load_profile, compute_look_at_from_pose

    world = LaunchConfiguration('world').perform(context)
    world_profiles = LaunchConfiguration('world_profiles').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
    pixel_noise_sigma = float(LaunchConfiguration('sensor_pixel_noise_sigma').perform(context))
    seed = int(LaunchConfiguration('seed').perform(context))

    profile, intrinsics, _world_path, camera_pose = load_profile(world_profiles, world)
    spawn = profile['spawn']

    cam_pos = [camera_pose[0], camera_pose[1], camera_pose[2]]
    roll, pitch, yaw = camera_pose[3], camera_pose[4], camera_pose[5]
    look_at = compute_look_at_from_pose(cam_pos, roll, pitch, yaw)

    sim_pkg = FindPackageShare('sim')
    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_pkg, 'launch', 'bringup_sim.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': 'true' if use_sim_time else 'false',
            'use_lidar': 'false',
            'bridge_scan': 'false',
            'world': world,
            'world_name': profile['world_name'],
            'spawn_x': str(float(spawn['x'])),
            'spawn_y': str(float(spawn['y'])),
            'spawn_z': str(float(spawn['z'])),
            'spawn_yaw': str(float(spawn['yaw'])),
        }.items(),
    )

    set_pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='set_pose_bridge',
        output='screen',
        arguments=[
            f"/world/{profile['world_name']}/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            f"/world/{profile['world_name']}/control@ros_gz_interfaces/srv/ControlWorld",
        ],
    )

    detector = Node(
        package='perception',
        executable='image_marker_detector_node',
        name='image_marker_detector_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'cam_pos': cam_pos,
            'look_at': look_at,
            'img_width': int(intrinsics['img_width']),
            'img_height': int(intrinsics['img_height']),
            'fov_h_rad': float(intrinsics['fov_h_rad']),
            'pixel_noise_sigma': pixel_noise_sigma,
            'seed': seed,
        }],
    )

    return [bringup_sim, set_pose_bridge, detector]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value='warehouse_occ_light.world.sdf'),
        DeclareLaunchArgument(
            'world_profiles',
            default_value=PathJoinSubstitution([
                FindPackageShare('experiments'), 'config', 'world_profiles.yaml'
            ]),
            description='World profile YAML',
        ),
        DeclareLaunchArgument(
            'sensor_pixel_noise_sigma',
            default_value='0.0',
            description='Optional synthetic pixel noise on the detector reference point',
        ),
        DeclareLaunchArgument('seed', default_value='0'),
        OpaqueFunction(function=_launch_setup),
    ])
