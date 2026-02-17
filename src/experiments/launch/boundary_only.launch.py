from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit


def _as_bool(value: str) -> bool:
    return str(value).lower() == 'true'


def _require_task_field(task, key):
    if key not in task:
        raise RuntimeError(f"Task is missing '{key}' field")
    return task[key]


def _launch_setup(context, *args, **kwargs):
    from experiments.core.world_profiles import (
        load_profile,
        compute_camera_quaternion_from_rpy,
        compute_look_at_from_pose,
    )
    from experiments.core.tasks import load_tasks, select_task

    use_sim_time_value = LaunchConfiguration('use_sim_time').perform(context)
    use_sim_time = _as_bool(use_sim_time_value)
    state_source = LaunchConfiguration('state_source').perform(context)
    planner = LaunchConfiguration('planner').perform(context)
    world = LaunchConfiguration('world').perform(context)
    world_profiles_path = LaunchConfiguration('world_profiles').perform(context)
    tasks_yaml = LaunchConfiguration('tasks_yaml').perform(context)
    task_name = LaunchConfiguration('task').perform(context).strip()
    perception_backend = LaunchConfiguration('perception_backend').perform(context).strip().lower()
    seed = int(LaunchConfiguration('seed').perform(context))
    pixel_noise_sigma = float(LaunchConfiguration('pixel_noise_sigma').perform(context))
    transform_noise_sigma = float(LaunchConfiguration('transform_noise_sigma').perform(context))
    use_pixel_correction = _as_bool(LaunchConfiguration('use_pixel_correction').perform(context))
    pixel_timeout_s = float(LaunchConfiguration('pixel_timeout_s').perform(context))
    add_ambiguity = _as_bool(LaunchConfiguration('add_ambiguity').perform(context))
    use_ambiguity = _as_bool(LaunchConfiguration('use_ambiguity').perform(context))
    use_obs_risk = _as_bool(LaunchConfiguration('use_obs_risk').perform(context))
    boundary_weight = float(LaunchConfiguration('boundary_weight').perform(context))
    costmap_min_x = float(LaunchConfiguration('costmap_min_x').perform(context))
    costmap_max_x = float(LaunchConfiguration('costmap_max_x').perform(context))
    costmap_min_y = float(LaunchConfiguration('costmap_min_y').perform(context))
    costmap_max_y = float(LaunchConfiguration('costmap_max_y').perform(context))
    costmap_wall_margin = float(LaunchConfiguration('costmap_wall_margin').perform(context))
    costmap_obstacle_enabled = _as_bool(LaunchConfiguration('costmap_obstacle_enabled').perform(context))
    costmap_obstacle_center_x = float(LaunchConfiguration('costmap_obstacle_center_x').perform(context))
    costmap_obstacle_center_y = float(LaunchConfiguration('costmap_obstacle_center_y').perform(context))
    costmap_obstacle_radius = float(LaunchConfiguration('costmap_obstacle_radius').perform(context))
    costmap_obstacle_value = int(LaunchConfiguration('costmap_obstacle_value').perform(context))
    obs_mode = LaunchConfiguration('obs_mode').perform(context)
    process_noise_xy = float(LaunchConfiguration('process_noise_xy').perform(context))
    process_noise_theta = float(LaunchConfiguration('process_noise_theta').perform(context))
    obs_noise_uv = float(LaunchConfiguration('obs_noise_uv').perform(context))
    obs_noise_yaw = float(LaunchConfiguration('obs_noise_yaw').perform(context))
    optimizer_backend = LaunchConfiguration('optimizer_backend').perform(context)
    optimizer_maxiter = int(LaunchConfiguration('optimizer_maxiter').perform(context))
    optimizer_gtol = float(LaunchConfiguration('optimizer_gtol').perform(context))
    optimizer_warm_start = _as_bool(LaunchConfiguration('optimizer_warm_start').perform(context))
    use_rviz = _as_bool(LaunchConfiguration('use_rviz').perform(context))
    rviz_config = LaunchConfiguration('rviz_config').perform(context)
    aruco_dict = LaunchConfiguration('aruco_dict').perform(context)
    target_marker_id = int(LaunchConfiguration('target_marker_id').perform(context))
    publish_yaw_from_marker = _as_bool(
        LaunchConfiguration('publish_yaw_from_marker').perform(context)
    )

    if state_source not in ('oracle', 'pixel'):
        raise RuntimeError("state_source must be 'oracle' or 'pixel'")
    if perception_backend not in ('homography', 'aruco'):
        raise RuntimeError("perception_backend must be 'homography' or 'aruco'")

    profile, intrinsics, world_path, camera_pose = load_profile(world_profiles_path, world)
    tasks_by_world = load_tasks(tasks_yaml)
    task = select_task(tasks_by_world, world, task_name)

    start = _require_task_field(task, 'start')
    goal = _require_task_field(task, 'goal')
    for key in ('x', 'y', 'z', 'yaw'):
        if key not in start:
            raise RuntimeError(f"Task start missing '{key}'")
    for key in ('x', 'y'):
        if key not in goal:
            raise RuntimeError(f"Task goal missing '{key}'")

    spawn = {
        'x': float(start['x']),
        'y': float(start['y']),
        'z': float(start['z']),
        'yaw': float(start['yaw']),
    }
    goal_x = float(goal['x'])
    goal_y = float(goal['y'])

    if planner == 'auto':
        planner = profile['planner_default']

    cam_pos = [camera_pose[0], camera_pose[1], camera_pose[2]]
    roll, pitch, yaw = camera_pose[3], camera_pose[4], camera_pose[5]
    look_at = compute_look_at_from_pose(cam_pos, roll, pitch, yaw)
    quat = compute_camera_quaternion_from_rpy(roll, pitch, yaw)
    spawn_quat = compute_camera_quaternion_from_rpy(0.0, 0.0, spawn['yaw'])

    img_width = int(intrinsics['img_width'])
    img_height = int(intrinsics['img_height'])
    fov_h_rad = float(intrinsics['fov_h_rad'])

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
            'spawn_x': str(spawn['x']),
            'spawn_y': str(spawn['y']),
            'spawn_z': str(spawn['z']),
            'spawn_yaw': str(spawn['yaw']),
        }.items(),
    )

    perception_pkg = FindPackageShare('perception')
    tf_args = {
        'use_sim_time': 'true' if use_sim_time else 'false',
        'cam_x': str(cam_pos[0]),
        'cam_y': str(cam_pos[1]),
        'cam_z': str(cam_pos[2]),
        'cam_qx': str(quat[0]),
        'cam_qy': str(quat[1]),
        'cam_qz': str(quat[2]),
        'cam_qw': str(quat[3]),
        # Keep map_bev aligned with world coordinates while odom starts at spawn.
        'odom_x': str(spawn['x']),
        'odom_y': str(spawn['y']),
        'odom_z': '0.0',
        'odom_qx': str(spawn_quat[0]),
        'odom_qy': str(spawn_quat[1]),
        'odom_qz': str(spawn_quat[2]),
        'odom_qw': str(spawn_quat[3]),
    }
    camera_params = {
        'cam_pos': cam_pos,
        'look_at': look_at,
        'img_width': img_width,
        'img_height': img_height,
        'fov_h_rad': fov_h_rad,
    }

    tf_static = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_pkg, 'launch', 'tf_static.launch.py'])
        ),
        launch_arguments=tf_args.items(),
    )

    wait_for_odom = Node(
        package='sim',
        executable='wait_for_odom',
        name='wait_for_odom',
        output='screen',
        parameters=[{
            'topic': '/odom',
            'timeout_s': 0.0,
            'min_messages': 5,
            'require_pose_match': True,
            'expected_x': 0.0,
            'expected_y': 0.0,
            'expected_yaw': 0.0,
            'position_tolerance': 0.25,
            'yaw_tolerance': 0.5,
        }]
    )

    homography_params = {'use_sim_time': use_sim_time}
    homography_params.update(camera_params)
    homography_sim = Node(
        package='perception',
        executable='homography_sim_node',
        name='homography_sim_node',
        output='screen',
        parameters=[homography_params],
    )
    aruco_detector = Node(
        package='perception',
        executable='aruco_detector_node',
        name='aruco_detector_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'aruco_dict': aruco_dict,
            'target_marker_id': target_marker_id,
            'publish_yaw_from_marker': publish_yaw_from_marker,
            **camera_params,
        }],
    )

    pixel_params = {
        'use_sim_time': use_sim_time,
        'state_source': state_source,
        'frame_id': 'map_bev',
        'pixel_noise_sigma': pixel_noise_sigma,
        'transform_noise_sigma': transform_noise_sigma,
        'seed': seed,
    }
    pixel_params.update(camera_params)
    pixel_to_bev = Node(
        package='state',
        executable='pixel_to_bev_state_node',
        name='pixel_to_bev_state_node',
        output='screen',
        parameters=[pixel_params],
    )

    boundary_cost_node = Node(
        package='experiments',
        executable='boundary_cost_node',
        name='boundary_cost_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'frame_id': 'map_bev',
            'min_x': costmap_min_x,
            'max_x': costmap_max_x,
            'min_y': costmap_min_y,
            'max_y': costmap_max_y,
            'wall_margin': costmap_wall_margin,
            'obstacle_enabled': costmap_obstacle_enabled,
            'obstacle_center_x': costmap_obstacle_center_x,
            'obstacle_center_y': costmap_obstacle_center_y,
            'obstacle_radius': costmap_obstacle_radius,
            'obstacle_value': costmap_obstacle_value,
        }]
    )

    if planner == 'astar':
        planner_node = Node(
            package='planning',
            executable='astar_planner',
            name='astar_planner',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        )
    elif planner in ('efe1', 'efe2', 'mpc', 'efer'):
        planner_modes = {
            'efe1': {
                'executable': 'efe_planner',
                'name': 'efe1_planner',
                'params': {
                    'approx_method': 'ET1',
                    'add_ambiguity': add_ambiguity,
                    'use_ambiguity': use_ambiguity,
                    'use_obs_risk': use_obs_risk,
                },
            },
            'efe2': {
                'executable': 'efe_planner',
                'name': 'efe2_planner',
                'params': {
                    'approx_method': 'ET2',
                    'add_ambiguity': add_ambiguity,
                    'use_ambiguity': use_ambiguity,
                    'use_obs_risk': use_obs_risk,
                },
            },
            'mpc': {
                'executable': 'mpc_planner',
                'name': 'mpc_planner',
                'params': {
                    'approx_method': 'ET1',
                    'add_ambiguity': False,
                    'use_ambiguity': False,
                    'use_obs_risk': True,
                },
            },
            'efer': {
                'executable': 'efer_planner',
                'name': 'efer_planner',
                'params': {
                    'approx_method': 'ET2',
                    'add_ambiguity': False,
                    'use_ambiguity': False,
                    'use_obs_risk': True,
                },
            },
        }
        mode = planner_modes[planner]
        planner_params = {
            'use_sim_time': use_sim_time,
            'use_pixel_correction': use_pixel_correction,
            'pixel_timeout_s': pixel_timeout_s,
            'boundary_weight': boundary_weight,
            'obs_mode': obs_mode,
            'process_noise_xy': process_noise_xy,
            'process_noise_theta': process_noise_theta,
            'obs_noise_uv': obs_noise_uv,
            'obs_noise_yaw': obs_noise_yaw,
            'optimizer_backend': optimizer_backend,
            'optimizer_maxiter': optimizer_maxiter,
            'optimizer_gtol': optimizer_gtol,
            'optimizer_warm_start': optimizer_warm_start,
            **camera_params,
            **mode['params'],
        }
        planner_node = Node(
            package='planning',
            executable=mode['executable'],
            name=mode['name'],
            output='screen',
            parameters=[planner_params],
        )
    else:
        raise RuntimeError("planner must be 'astar', 'efe1', 'efe2', 'mpc', 'efer', or 'auto'")

    control_node = Node(
        package='control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    mission_node = Node(
        package='experiments',
        executable='goal_mission_node',
        name='goal_mission_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'frame_id': 'map_bev',
            'goal_x': goal_x,
            'goal_y': goal_y,
        }]
    )

    logger_node = Node(
        package='experiments',
        executable='experiment_logger',
        name='experiment_logger',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'seed': seed,
            'world': world,
            'task': task.get('name', task_name or ''),
            'planner': planner,
            'state_source': state_source,
            'pixel_noise_sigma': pixel_noise_sigma,
            'transform_noise_sigma': transform_noise_sigma,
            'world_profiles_path': world_profiles_path,
            'tasks_yaml': tasks_yaml,
        }],
    )

    viz_pkg = FindPackageShare('visualization')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    after_odom = []
    if state_source == 'pixel':
        if perception_backend == 'homography':
            after_odom.append(homography_sim)
        else:
            after_odom.append(aruco_detector)
    after_odom.extend([
        pixel_to_bev,
        boundary_cost_node,
        planner_node,
        control_node,
        mission_node,
        logger_node,
    ])
    if use_rviz:
        after_odom.append(rviz)

    start_after_odom = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_odom,
            on_exit=after_odom,
        )
    )

    return [
        bringup_sim,
        tf_static,
        wait_for_odom,
        start_after_odom,
    ]


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for boundary-only pipeline'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('visualization'), 'rviz', 'boundary_only_camera.rviz'
        ]),
        description='RViz config file to load when use_rviz=true'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    state_source_arg = DeclareLaunchArgument(
        'state_source',
        default_value='oracle',
        description='State source: oracle | pixel'
    )
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='astar',
        description='Planner: astar | efe1 | efe2 | mpc | efer | auto'
    )
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.world.sdf',
        description='World file under sim/gazebo_worlds/worlds'
    )
    world_profiles_arg = DeclareLaunchArgument(
        'world_profiles',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'world_profiles.yaml'
        ]),
        description='YAML file describing per-world profiles'
    )
    tasks_yaml_arg = DeclareLaunchArgument(
        'tasks_yaml',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'tasks.yaml'
        ]),
        description='YAML file describing per-world tasks'
    )
    task_arg = DeclareLaunchArgument(
        'task',
        default_value='',
        description='Task name; defaults to first task in tasks.yaml for the world'
    )
    perception_backend_arg = DeclareLaunchArgument(
        'perception_backend',
        default_value='homography',
        description='Perception backend in pixel mode: homography | aruco'
    )
    seed_arg = DeclareLaunchArgument('seed', default_value='0')
    pixel_noise_arg = DeclareLaunchArgument('pixel_noise_sigma', default_value='0.0')
    transform_noise_arg = DeclareLaunchArgument('transform_noise_sigma', default_value='0.0')
    use_pixel_correction_arg = DeclareLaunchArgument(
        'use_pixel_correction',
        default_value='true',
        description='Apply pixel-space correction in EFE planner'
    )
    pixel_timeout_arg = DeclareLaunchArgument('pixel_timeout_s', default_value='0.5')
    add_ambiguity_arg = DeclareLaunchArgument(
        'add_ambiguity',
        default_value='true',
        description='Include ambiguity term in EFE objective'
    )
    use_ambiguity_arg = DeclareLaunchArgument(
        'use_ambiguity',
        default_value='true',
        description='Enable ambiguity computation in planner'
    )
    use_obs_risk_arg = DeclareLaunchArgument(
        'use_obs_risk',
        default_value='true',
        description='Enable observation-space risk term'
    )
    boundary_weight_arg = DeclareLaunchArgument(
        'boundary_weight',
        default_value='1.0',
        description='Boundary/costmap penalty weight for EFE planner'
    )
    costmap_min_x_arg = DeclareLaunchArgument(
        'costmap_min_x',
        default_value='-5.0',
        description='Costmap lower x bound in map_bev frame'
    )
    costmap_max_x_arg = DeclareLaunchArgument(
        'costmap_max_x',
        default_value='5.0',
        description='Costmap upper x bound in map_bev frame'
    )
    costmap_min_y_arg = DeclareLaunchArgument(
        'costmap_min_y',
        default_value='-5.0',
        description='Costmap lower y bound in map_bev frame'
    )
    costmap_max_y_arg = DeclareLaunchArgument(
        'costmap_max_y',
        default_value='5.0',
        description='Costmap upper y bound in map_bev frame'
    )
    costmap_wall_margin_arg = DeclareLaunchArgument(
        'costmap_wall_margin',
        default_value='0.2',
        description='Lethal wall margin inside costmap bounds'
    )
    costmap_obstacle_enabled_arg = DeclareLaunchArgument(
        'costmap_obstacle_enabled',
        default_value='true',
        description='Enable static circular obstacle in boundary costmap'
    )
    costmap_obstacle_center_x_arg = DeclareLaunchArgument(
        'costmap_obstacle_center_x',
        default_value='1.5',
        description='Static obstacle center x in map_bev'
    )
    costmap_obstacle_center_y_arg = DeclareLaunchArgument(
        'costmap_obstacle_center_y',
        default_value='1.5',
        description='Static obstacle center y in map_bev'
    )
    costmap_obstacle_radius_arg = DeclareLaunchArgument(
        'costmap_obstacle_radius',
        default_value='0.4',
        description='Static obstacle radius in meters'
    )
    costmap_obstacle_value_arg = DeclareLaunchArgument(
        'costmap_obstacle_value',
        default_value='100',
        description='Static obstacle occupancy value'
    )
    obs_mode_arg = DeclareLaunchArgument(
        'obs_mode',
        default_value='uv',
        description="Observation mode for EFE planner: 'uv' or 'uvt'"
    )
    process_noise_xy_arg = DeclareLaunchArgument(
        'process_noise_xy',
        default_value='0.01',
        description='Process noise std for x/y state dynamics'
    )
    process_noise_theta_arg = DeclareLaunchArgument(
        'process_noise_theta',
        default_value='0.02',
        description='Process noise std for yaw state dynamics'
    )
    obs_noise_uv_arg = DeclareLaunchArgument(
        'obs_noise_uv',
        default_value='2.0',
        description='Observation noise std in pixel u/v'
    )
    obs_noise_yaw_arg = DeclareLaunchArgument(
        'obs_noise_yaw',
        default_value='0.05',
        description='Observation noise std for yaw observation (uvt mode)'
    )
    optimizer_backend_arg = DeclareLaunchArgument(
        'optimizer_backend',
        default_value='auto',
        description="Optimizer backend: 'auto', 'jax', or 'scipy'"
    )
    optimizer_maxiter_arg = DeclareLaunchArgument(
        'optimizer_maxiter',
        default_value='50',
        description='Maximum optimizer iterations for planner'
    )
    optimizer_gtol_arg = DeclareLaunchArgument(
        'optimizer_gtol',
        default_value='1e-4',
        description='Optimizer gradient tolerance for planner'
    )
    optimizer_warm_start_arg = DeclareLaunchArgument(
        'optimizer_warm_start',
        default_value='true',
        description='Use warm-start controls between planning cycles'
    )
    aruco_dict_arg = DeclareLaunchArgument(
        'aruco_dict',
        default_value='DICT_4X4_50',
        description='ArUco/AprilTag dictionary for aruco perception backend'
    )
    target_marker_id_arg = DeclareLaunchArgument(
        'target_marker_id',
        default_value='0',
        description='Marker id to track; use -1 to track the largest detected marker'
    )
    publish_yaw_from_marker_arg = DeclareLaunchArgument(
        'publish_yaw_from_marker',
        default_value='true',
        description='Estimate and publish yaw from marker corners (aruco backend)'
    )

    return LaunchDescription([
        use_rviz_arg,
        rviz_config_arg,
        use_sim_time_arg,
        state_source_arg,
        planner_arg,
        world_arg,
        world_profiles_arg,
        tasks_yaml_arg,
        task_arg,
        perception_backend_arg,
        seed_arg,
        pixel_noise_arg,
        transform_noise_arg,
        use_pixel_correction_arg,
        pixel_timeout_arg,
        add_ambiguity_arg,
        use_ambiguity_arg,
        use_obs_risk_arg,
        boundary_weight_arg,
        costmap_min_x_arg,
        costmap_max_x_arg,
        costmap_min_y_arg,
        costmap_max_y_arg,
        costmap_wall_margin_arg,
        costmap_obstacle_enabled_arg,
        costmap_obstacle_center_x_arg,
        costmap_obstacle_center_y_arg,
        costmap_obstacle_radius_arg,
        costmap_obstacle_value_arg,
        obs_mode_arg,
        process_noise_xy_arg,
        process_noise_theta_arg,
        obs_noise_uv_arg,
        obs_noise_yaw_arg,
        optimizer_backend_arg,
        optimizer_maxiter_arg,
        optimizer_gtol_arg,
        optimizer_warm_start_arg,
        aruco_dict_arg,
        target_marker_id_arg,
        publish_yaw_from_marker_arg,
        OpaqueFunction(function=_launch_setup),
    ])
