import os

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _as_bool_str(value) -> str:
    return 'true' if str(value).lower() in ('1', 'true', 'yes', 'on') else 'false'


def _require_field(exp: dict, key: str):
    if key not in exp:
        raise RuntimeError(f"Experiment is missing required field '{key}'")
    return exp[key]


def _launch_setup(context, *args, **kwargs):
    experiment_name = LaunchConfiguration('experiment').perform(context).strip()
    experiments_yaml = LaunchConfiguration('experiments_yaml').perform(context)

    if not experiment_name:
        raise RuntimeError("experiment name must be provided")
    if not os.path.isfile(experiments_yaml):
        raise RuntimeError(f"experiments.yaml not found: {experiments_yaml}")

    with open(experiments_yaml, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    experiments = data.get('experiments')
    if not isinstance(experiments, dict) or not experiments:
        raise RuntimeError("experiments.yaml must contain a non-empty 'experiments' mapping")
    if experiment_name not in experiments:
        known = ", ".join(sorted(experiments.keys()))
        raise RuntimeError(
            f"Experiment '{experiment_name}' not found. Available: {known or 'none'}"
        )

    exp = experiments[experiment_name]
    if not isinstance(exp, dict):
        raise RuntimeError(f"Experiment '{experiment_name}' must be a mapping")

    world = _require_field(exp, 'world')
    task = _require_field(exp, 'task')
    planner = _require_field(exp, 'planner')
    state_source = _require_field(exp, 'state_source')

    seed = exp.get('seed', 0)
    pixel_noise_sigma = exp.get('pixel_noise_sigma', 0.0)
    transform_noise_sigma = exp.get('transform_noise_sigma', 0.0)

    args = {
        'world': str(world),
        'task': str(task),
        'planner': str(planner),
        'state_source': str(state_source),
        'seed': str(seed),
        'pixel_noise_sigma': str(pixel_noise_sigma),
        'transform_noise_sigma': str(transform_noise_sigma),
    }

    if 'use_rviz' in exp:
        args['use_rviz'] = _as_bool_str(exp.get('use_rviz'))
    if 'use_pixel_correction' in exp:
        args['use_pixel_correction'] = _as_bool_str(exp.get('use_pixel_correction'))
    if 'pixel_timeout_s' in exp:
        args['pixel_timeout_s'] = str(exp.get('pixel_timeout_s'))
    if 'use_sim_time' in exp:
        args['use_sim_time'] = _as_bool_str(exp.get('use_sim_time'))
    if 'world_profiles' in exp:
        args['world_profiles'] = str(exp.get('world_profiles'))
    if 'tasks_yaml' in exp:
        args['tasks_yaml'] = str(exp.get('tasks_yaml'))

    experiments_pkg = FindPackageShare('experiments')
    boundary_only = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([experiments_pkg, 'launch', 'boundary_only.launch.py'])
        ),
        launch_arguments=args.items(),
    )

    return [boundary_only]


def generate_launch_description():
    experiment_arg = DeclareLaunchArgument(
        'experiment',
        default_value='',
        description='Experiment name defined in experiments.yaml',
    )
    experiments_yaml_arg = DeclareLaunchArgument(
        'experiments_yaml',
        default_value=PathJoinSubstitution([
            FindPackageShare('experiments'), 'config', 'experiments.yaml'
        ]),
        description='YAML file describing named experiments',
    )

    return LaunchDescription([
        experiment_arg,
        experiments_yaml_arg,
        OpaqueFunction(function=_launch_setup),
    ])
