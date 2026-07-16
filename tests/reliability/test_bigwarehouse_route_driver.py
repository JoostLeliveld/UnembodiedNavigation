from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools" / "drive_study_route.py"
STUDY = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "study.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("bigwarehouse_study_route_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_route_offsets_are_normal_to_direction_of_travel() -> None:
    module = _module()

    _config, northbound = module.load_route(STUDY, route_name="south_to_north_handover", lateral_offset_m=0.5)
    _config, southbound = module.load_route(STUDY, route_name="north_to_south_handover", lateral_offset_m=0.5)

    assert northbound == pytest.approx([(-2.0, -7.2), (-2.0, 7.2)])
    assert southbound == pytest.approx([(2.0, 7.2), (2.0, -7.2)])


def test_route_manifest_is_operational_only_and_tied_to_study_config() -> None:
    module = _module()
    _config, points = module.load_route(STUDY, route_name="central_overlap_sweep", lateral_offset_m=0.0)
    spawn = module.route_spawn_pose(STUDY, route_name="central_overlap_sweep", lateral_offset_m=0.0)
    local_points = module.world_to_spawn_local(
        points,
        spawn_x_m=spawn[0],
        spawn_y_m=spawn[1],
        spawn_yaw_rad=spawn[2],
    )
    manifest = module.route_manifest(
        study_path=STUDY,
        route_name="central_overlap_sweep",
        lateral_offset_m=0.0,
        speed_mps=0.12,
        odom_topic="/odom",
        command_topic="/cmd_vel",
        world_waypoints=points,
        local_waypoints=local_points,
        spawn_pose_warehouse=spawn,
    )

    assert manifest["contains_ground_truth"] is False
    assert len(manifest["study_config_sha256"]) == 64
    assert manifest["waypoints_warehouse_xy_m"] == [{"x": -1.8, "y": -0.3}, {"x": 1.8, "y": -0.3}]
    assert manifest["waypoints_spawn_local_xy_m"] == [{"x": 0.0, "y": 0.0}, {"x": 3.6, "y": 0.0}]


def test_northbound_route_is_rotated_into_spawn_local_odom_coordinates() -> None:
    module = _module()
    _config, points = module.load_route(STUDY, route_name="south_to_north_handover", lateral_offset_m=0.0)
    spawn = module.route_spawn_pose(STUDY, route_name="south_to_north_handover", lateral_offset_m=0.0)

    local_points = module.world_to_spawn_local(
        points,
        spawn_x_m=spawn[0],
        spawn_y_m=spawn[1],
        spawn_yaw_rad=spawn[2],
    )

    assert local_points[0] == pytest.approx((0.0, 0.0), abs=1e-3)
    assert local_points[1][0] == pytest.approx(14.4, abs=1e-3)
    assert local_points[1][1] == pytest.approx(0.0, abs=1e-3)
