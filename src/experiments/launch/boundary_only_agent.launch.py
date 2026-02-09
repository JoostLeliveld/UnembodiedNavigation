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
    seed = int(LaunchConfiguration('seed').perform(context))
    pixel_noise_sigma = float(LaunchConfiguration('pixel_noise_sigma').perform(context))
    transform_noise_sigma = float(LaunchConfiguration('transform_noise_sigma').perform(context))
    use_pixel_correction = _as_bool(LaunchConfiguration('use_pixel_correction').perform(context))
    pixel_timeout_s = float(LaunchConfiguration('pixel_timeout_s').perform(context))
    use_rviz = _as_bool(LaunchConfiguration('use_rviz').perform(context))
    rviz_config = LaunchConfiguration('rviz_config').perform(context)

    if state_source not in ('oracle', 'pixel'):
        raise RuntimeError("state_source must be 'oracle' or 'pixel'")

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

    if planner not in ('efe1', 'efe2'):
        raise RuntimeError("planner must be 'efe1', 'efe2', or 'auto' for agent launch")

    cam_pos = [camera_pose[0], camera_pose[1], camera_pose[2]]
    roll, pitch, yaw = camera_pose[3], camera_pose[4], camera_pose[5]
    look_at = compute_look_at_from_pose(cam_pos, roll, pitch, yaw)
    quat = compute_camera_quaternion_from_rpy(roll, pitch, yaw)

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
        }]
    )

    approx_method = 'ET1' if planner == 'efe1' else 'ET2'
    agent_node = Node(
        package='planning',
        executable='efe_agent',
        name='efe_agent',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'use_pixel_correction': use_pixel_correction,
            'pixel_timeout_s': pixel_timeout_s,
            'approx_method': approx_method,
            **camera_params,
        }],
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
        after_odom.append(homography_sim)
    after_odom.extend([
        pixel_to_bev,
        boundary_cost_node,
        agent_node,
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
        default_value='efe2',
        description='Planner: efe1 | efe2 | auto'
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
    seed_arg = DeclareLaunchArgument('seed', default_value='0')
    pixel_noise_arg = DeclareLaunchArgument('pixel_noise_sigma', default_value='0.0')
    transform_noise_arg = DeclareLaunchArgument('transform_noise_sigma', default_value='0.0')
    use_pixel_correction_arg = DeclareLaunchArgument(
        'use_pixel_correction',
        default_value='true',
        description='Apply pixel-space correction in EFE agent'
    )
    pixel_timeout_arg = DeclareLaunchArgument('pixel_timeout_s', default_value='0.5')

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
        seed_arg,
        pixel_noise_arg,
        transform_noise_arg,
        use_pixel_correction_arg,
        pixel_timeout_arg,
        OpaqueFunction(function=_launch_setup),
    ])
