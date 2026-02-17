from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _stringify(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_override_args(raw: str):
    out = {}
    if not raw:
        return out
    text = raw.strip()
    if not text:
        return out
    parts = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        for inner in chunk.split(";"):
            inner = inner.strip()
            if inner:
                parts.append(inner)
    for token in parts:
        if "=" not in token:
            raise RuntimeError(
                f"Invalid override token '{token}'. Expected key=value pairs."
            )
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise RuntimeError(f"Invalid override token '{token}'. Empty key.")
        out[key] = value
    return out


def _launch_setup(context, *args, **kwargs):
    from experiments.core.experiment_presets import (
        load_experiment_presets,
        select_experiment,
    )

    experiment_name = LaunchConfiguration("experiment").perform(context).strip()
    presets_yaml = LaunchConfiguration("experiments_yaml").perform(context)
    override_args_raw = LaunchConfiguration("override_args").perform(context)

    presets = load_experiment_presets(presets_yaml)
    selected = select_experiment(presets, experiment_name)
    launch_name = selected["launch"]
    args_map = dict(selected["args"])
    args_map.update(_parse_override_args(override_args_raw))

    launch_file = "boundary_only.launch.py"
    if launch_name == "boundary_only_agent":
        launch_file = "boundary_only_agent.launch.py"

    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("experiments"), "launch", launch_file])
        ),
        launch_arguments={k: _stringify(v) for k, v in args_map.items()}.items(),
    )

    return [include]


def generate_launch_description():
    experiment_arg = DeclareLaunchArgument(
        "experiment",
        default_value="",
        description="Experiment preset key in experiments.yaml",
    )
    experiments_yaml_arg = DeclareLaunchArgument(
        "experiments_yaml",
        default_value=PathJoinSubstitution(
            [FindPackageShare("experiments"), "config", "experiments.yaml"]
        ),
        description="Path to experiment preset YAML",
    )
    override_args_arg = DeclareLaunchArgument(
        "override_args",
        default_value="",
        description=(
            "Optional key=value overrides separated by ',' or ';' "
            "(example: seed=3,use_rviz=false)"
        ),
    )

    return LaunchDescription(
        [
            experiment_arg,
            experiments_yaml_arg,
            override_args_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )
