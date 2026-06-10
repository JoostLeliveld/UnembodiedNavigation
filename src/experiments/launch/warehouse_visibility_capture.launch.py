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
    perception_backend = LaunchConfiguration('perception_backend').perform(context).strip().lower()
    yolo_model = LaunchConfiguration('yolo_model').perform(context)
    yolo_device = LaunchConfiguration('yolo_device').perform(context)
    yolo_imgsz = int(LaunchConfiguration('yolo_imgsz').perform(context))
    yolo_conf_threshold = float(LaunchConfiguration('yolo_conf_threshold').perform(context))
    yolo_iou_threshold = float(LaunchConfiguration('yolo_iou_threshold').perform(context))
    yolo_target_class = LaunchConfiguration('yolo_target_class').perform(context)
    yolo_class_id = int(LaunchConfiguration('yolo_class_id').perform(context))
    yolo_use_masks = LaunchConfiguration('yolo_use_masks').perform(context)
    yolo_min_mask_area_px = float(LaunchConfiguration('yolo_min_mask_area_px').perform(context))
    yolo_mask_bottom_band_px = float(LaunchConfiguration('yolo_mask_bottom_band_px').perform(context))

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
            'headless': LaunchConfiguration('headless').perform(context),
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

    if perception_backend != 'yolo':
        raise RuntimeError("Visibility capture supports only perception_backend:=yolo")
    yolo_detector_params = {
        'pixel_noise_sigma': pixel_noise_sigma,
        'seed': seed,
        'model_path': yolo_model,
        'device': yolo_device,
        'image_size': yolo_imgsz,
        'confidence_threshold': yolo_conf_threshold,
        'iou_threshold': yolo_iou_threshold,
        'class_name': yolo_target_class,
        'class_id': yolo_class_id,
        'use_masks': yolo_use_masks.strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on'),
        'mask_min_area': yolo_min_mask_area_px,
        'mask_bottom_band_px': yolo_mask_bottom_band_px,
    }
    detector = Node(
        package='perception',
        executable='yolo_robot_detector_node',
        name='yolo_robot_detector_node',
        output='screen',
        parameters=[yolo_detector_params],
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

    return [bringup_sim, wait_for_odom, detector, state_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('headless', default_value='true', description='Run Gazebo server-only (batch capture default)'),
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
            description='Synthetic detector pixel noise applied before state conversion',
        ),
        DeclareLaunchArgument('perception_backend', default_value='yolo', description='Capture perception backend: yolo'),
        DeclareLaunchArgument('yolo_model', default_value='', description='Local path to a trained YOLO .pt model'),
        DeclareLaunchArgument('yolo_device', default_value='', description='Ultralytics device string; empty lets Ultralytics choose'),
        DeclareLaunchArgument('yolo_imgsz', default_value='640'),
        DeclareLaunchArgument('yolo_conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('yolo_iou_threshold', default_value='0.45'),
        DeclareLaunchArgument('yolo_target_class', default_value='robot'),
        DeclareLaunchArgument('yolo_class_id', default_value='-1'),
        DeclareLaunchArgument('yolo_use_masks', default_value='true', description='Use YOLO segmentation masks for pixel reference when available'),
        DeclareLaunchArgument('yolo_min_mask_area_px', default_value='12.0'),
        DeclareLaunchArgument('yolo_mask_bottom_band_px', default_value='3.0'),
        DeclareLaunchArgument('seed', default_value='0'),
        OpaqueFunction(function=_launch_setup),
    ])
