"""Compact thesis-facing launch builders for the retained planner stack."""

from __future__ import annotations

from typing import Iterable

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _build_launch_setup(default_planner: str, allowed_planners: tuple[str, ...]):
    def _launch_setup(context, *args, **kwargs):
        from experiments.core.visibility_launch_common import (
            build_agent_runtime_actions,
            parse_common_launch_config,
            resolve_world_setup,
        )

        cfg = parse_common_launch_config(context)
        planner = str(cfg.get("planner", default_planner) or default_planner).strip().lower()
        if planner not in allowed_planners:
            allowed_text = ", ".join(allowed_planners)
            raise RuntimeError(f"planner must be one of: {allowed_text}")

        cfg["planner"] = planner
        cfg["use_live_dashboard"] = False
        cfg["use_rviz"] = bool(cfg.get("use_rviz", False))

        if planner == "geometric_baseline":
            cfg["use_visibility_model"] = False
            cfg["use_ambiguity"] = False
            cfg["use_obs_risk"] = True
            cfg["visibility_model"] = "gp_visibility"
        else:
            cfg["use_visibility_model"] = True
            cfg["visibility_model"] = "gp_visibility"

        cfg = resolve_world_setup(cfg)
        return build_agent_runtime_actions(cfg)

    return _launch_setup


def make_thesis_launch_description(
    *,
    default_planner: str,
    allowed_planners: Iterable[str],
    planner_description: str,
) -> LaunchDescription:
    allowed = tuple(str(name).strip().lower() for name in allowed_planners)
    if default_planner not in allowed:
        raise ValueError("default_planner must be included in allowed_planners")

    world_profiles_default = PathJoinSubstitution([
        FindPackageShare("experiments"), "config", "world_profiles.yaml",
    ])
    tasks_default = PathJoinSubstitution([
        FindPackageShare("experiments"), "config", "tasks.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("planner", default_value=default_planner, description=planner_description),
        DeclareLaunchArgument("world", default_value="warehouse_occ_light.world.sdf"),
        DeclareLaunchArgument("world_profiles", default_value=world_profiles_default, description="World profile YAML"),
        DeclareLaunchArgument("tasks_yaml", default_value=tasks_default, description="Task YAML"),
        DeclareLaunchArgument("task", default_value="T1_shadow_cross"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("perception_backend", default_value="image_markers", description="image_markers or homography"),
        DeclareLaunchArgument("sensor_pixel_noise_sigma", default_value="1.0"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        OpaqueFunction(function=_build_launch_setup(default_planner, allowed)),
    ])
