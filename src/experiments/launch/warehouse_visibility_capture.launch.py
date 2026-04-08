from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
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
    sweep_margin_m = float(LaunchConfiguration('sweep_margin_m').perform(context))
    sweep_row_spacing_m = float(LaunchConfiguration('sweep_row_spacing_m').perform(context))
    sweep_linear_speed_mps = float(LaunchConfiguration('sweep_linear_speed_mps').perform(context))
    sweep_angular_speed_radps = float(LaunchConfiguration('sweep_angular_speed_radps').perform(context))
    sweep_waypoint_tolerance_m = float(LaunchConfiguration('sweep_waypoint_tolerance_m').perform(context))
    sweep_turn_pause_s = float(LaunchConfiguration('sweep_turn_pause_s').perform(context))

    profile, intrinsics, _world_path, camera_pose = load_profile(world_profiles, world)
    spawn = profile['spawn']
    vis = dict(profile.get('visibility_defaults') or {})

    cam_pos = [camera_pose[0], camera_pose[1], camera_pose[2]]
    roll, pitch, yaw = camera_pose[3], camera_pose[4], camera_pose[5]
    look_at = compute_look_at_from_pose(cam_pos, roll, pitch, yaw)

    xmin = float(vis.get('visibility_map_min_x', -6.0))
    xmax = float(vis.get('visibility_map_max_x', 6.0))
    ymin = float(vis.get('visibility_map_min_y', -6.0))
    ymax = float(vis.get('visibility_map_max_y', 6.0))

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

    wait_for_odom = Node(
        package='sim',
        executable='wait_for_odom',
        name='wait_for_odom',
        output='screen',
        parameters=[{
            'topic': '/odom',
            'timeout_s': 15.0,
            'min_messages': 3,
            'require_pose_match': False,
        }],
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

    state_node = Node(
        package='state',
        executable='pixel_to_bev_state_node',
        name='pixel_to_bev_state_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'frame_id': 'map_bev',
            'pixel_noise_sigma': 0.0,
            'heading_pixel_noise_sigma': pixel_noise_sigma,
            'transform_noise_sigma': 0.0,
            'use_odom_heading_fallback': True,
            'odom_heading_timeout_s': 0.5,
            'odom_heading_sigma_rad': 0.08,
            'infer_yaw_from_motion': False,
            'seed': seed,
            'cam_pos': cam_pos,
            'look_at': look_at,
            'img_width': int(intrinsics['img_width']),
            'img_height': int(intrinsics['img_height']),
            'fov_h_rad': float(intrinsics['fov_h_rad']),
        }],
    )

    sweep_controller = Node(
        package='experiments',
        executable='visibility_sweep_controller_node',
        name='visibility_sweep_controller',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'frame_id': 'map_bev',
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'sweep_margin_m': sweep_margin_m,
            'sweep_row_spacing_m': sweep_row_spacing_m,
            'linear_speed_mps': sweep_linear_speed_mps,
            'angular_speed_radps': sweep_angular_speed_radps,
            'waypoint_tolerance_m': sweep_waypoint_tolerance_m,
            'turn_pause_s': sweep_turn_pause_s,
        }],
    )

    return [bringup_sim, wait_for_odom, detector, state_node, sweep_controller]


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
            default_value='1.0',
            description='Synthetic detector pixel noise applied in the image-marker capture path',
        ),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('sweep_margin_m', default_value='0.45'),
        DeclareLaunchArgument('sweep_row_spacing_m', default_value='0.75'),
        DeclareLaunchArgument('sweep_linear_speed_mps', default_value='0.22'),
        DeclareLaunchArgument('sweep_angular_speed_radps', default_value='0.90'),
        DeclareLaunchArgument('sweep_waypoint_tolerance_m', default_value='0.18'),
        DeclareLaunchArgument('sweep_turn_pause_s', default_value='0.20'),
        OpaqueFunction(function=_launch_setup),
    ])
