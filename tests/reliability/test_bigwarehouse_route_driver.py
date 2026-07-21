from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools" / "drive_study_route.py"
STUDY = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "study.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("bigwarehouse_study_route_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
    assert manifest["waypoints_warehouse_xy_m"] == [{"x": -1.8, "y": 0.3}, {"x": 1.8, "y": 0.3}]
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


def test_long_slow_route_default_deadlines_include_hold_and_manoeuvre_margin() -> None:
    module = _module()
    waypoints = [(0.0, 0.0), (14.4, 0.0)]

    max_sim_s, max_wall_s = module.default_runtime_limits_s(
        waypoints,
        speed_mps=0.12,
        start_hold_s=2.0,
    )

    assert max_sim_s == pytest.approx(152.0)
    assert max_wall_s == pytest.approx(608.0)


def test_watchdog_covers_missing_and_stale_odom_and_both_runtime_clocks() -> None:
    module = _module()
    limits = module.RouteSafetyLimits(
        initial_odom_timeout_s=10.0,
        odom_freshness_timeout_s=0.5,
        max_wall_runtime_s=100.0,
        max_sim_runtime_s=50.0,
    )

    assert module.route_watchdog_failure(
        limits,
        wall_elapsed_s=9.9,
        sim_elapsed_s=None,
        odom_wall_age_s=None,
        odom_sim_age_s=None,
    ) == ""
    assert module.route_watchdog_failure(
        limits,
        wall_elapsed_s=10.1,
        sim_elapsed_s=None,
        odom_wall_age_s=None,
        odom_sim_age_s=None,
    ) == "initial_odometry_timeout"
    assert module.route_watchdog_failure(
        limits,
        wall_elapsed_s=20.0,
        sim_elapsed_s=20.0,
        odom_wall_age_s=0.51,
        odom_sim_age_s=0.1,
    ) == "odometry_stale_wall"
    assert module.route_watchdog_failure(
        limits,
        wall_elapsed_s=20.0,
        sim_elapsed_s=20.0,
        odom_wall_age_s=0.1,
        odom_sim_age_s=0.51,
    ) == "odometry_stale_sim"
    assert module.route_watchdog_failure(
        limits,
        wall_elapsed_s=20.0,
        sim_elapsed_s=50.1,
        odom_wall_age_s=0.1,
        odom_sim_age_s=0.1,
    ) == "max_sim_runtime_exceeded"
    assert module.route_watchdog_failure(
        limits,
        wall_elapsed_s=100.1,
        sim_elapsed_s=20.0,
        odom_wall_age_s=0.1,
        odom_sim_age_s=0.1,
    ) == "max_wall_runtime_exceeded"


def test_atomic_json_publication_never_clobbers_a_prior_attempt(tmp_path: Path) -> None:
    module = _module()
    destination = tmp_path / "route_completion.json"

    module.write_json_atomic_new(destination, {"status": "success", "attempt": 1})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "success",
        "attempt": 1,
    }
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        module.write_json_atomic_new(destination, {"status": "success", "attempt": 2})
    assert json.loads(destination.read_text(encoding="utf-8"))["attempt"] == 1


def test_terminal_shutdown_publishes_the_configured_stop_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    limits = module.RouteSafetyLimits(
        initial_odom_timeout_s=10.0,
        odom_freshness_timeout_s=0.5,
        max_wall_runtime_s=100.0,
        max_sim_runtime_s=50.0,
        terminal_stop_publish_count=5,
        terminal_stop_period_s=0.01,
    )
    fake_driver = SimpleNamespace(safety_limits=limits, _terminal_stops_published=1)
    publishes: list[int] = []
    sleeps: list[float] = []

    def publish() -> None:
        fake_driver._terminal_stops_published += 1
        publishes.append(fake_driver._terminal_stops_published)

    fake_driver._publish_terminal_stop = publish
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    module.StudyRouteDriver.publish_remaining_terminal_stops(fake_driver)

    assert fake_driver._terminal_stops_published == 5
    assert publishes == [2, 3, 4, 5]
    assert sleeps == [0.01, 0.01, 0.01]

    timed_driver = SimpleNamespace(
        safety_limits=limits,
        _terminal_stops_published=1,
        terminating=True,
        finished=False,
    )
    timed_publishes: list[int] = []
    finalized: list[bool] = []

    def timed_publish() -> None:
        timed_driver._terminal_stops_published += 1
        timed_publishes.append(timed_driver._terminal_stops_published)

    timed_driver._publish_terminal_stop = timed_publish
    timed_driver._finalize_termination = lambda: finalized.append(True)
    for _ in range(4):
        module.StudyRouteDriver._terminal_stop_tick(timed_driver)

    assert timed_publishes == [2, 3, 4, 5]
    assert finalized == [True]


def test_failure_finalization_never_creates_a_success_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "rclpy", None)
    completion_path = tmp_path / "route_completion.json"

    class Logger:
        def info(self, _message: str) -> None:
            pass

        def error(self, _message: str) -> None:
            pass

    class Timer:
        def cancel(self) -> None:
            pass

    fake_driver = SimpleNamespace(
        finished=False,
        succeeded=False,
        termination_reason="odometry_stale_wall",
        safety_limits=module.RouteSafetyLimits(
            initial_odom_timeout_s=10.0,
            odom_freshness_timeout_s=0.5,
            max_wall_runtime_s=100.0,
            max_sim_runtime_s=50.0,
        ),
        _terminal_stops_published=5,
        completion_path=completion_path,
        wall_watchdog_timer=Timer(),
        terminal_stop_timer=Timer(),
        get_logger=lambda: Logger(),
    )

    module.StudyRouteDriver._finalize_termination(fake_driver)

    assert fake_driver.finished is True
    assert completion_path.exists() is False


def test_success_is_published_only_after_the_full_stop_burst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "rclpy", None)
    limits = module.RouteSafetyLimits(
        initial_odom_timeout_s=10.0,
        odom_freshness_timeout_s=0.5,
        max_wall_runtime_s=100.0,
        max_sim_runtime_s=50.0,
        terminal_stop_publish_count=5,
    )

    class Logger:
        def info(self, _message: str) -> None:
            pass

        def error(self, _message: str) -> None:
            pass

    class Timer:
        def cancel(self) -> None:
            pass

    payload_driver = SimpleNamespace(
        _termination_sim_time_s=20.0,
        started_at_s=5.0,
        latest_pose=(1.0, 2.0, 0.3),
        completion_context={"route": "central_overlap_sweep"},
        _termination_wall_elapsed_s=16.0,
        waypoints=[(0.0, 0.0), (3.6, 0.0)],
        waypoint_index=2,
        _terminal_stops_published=5,
        safety_limits=limits,
    )
    payload = module.StudyRouteDriver._completion_payload(payload_driver)
    assert payload["status"] == "completed"
    assert payload["route_complete"] is True
    assert payload["end_reason"] == "route_complete"
    assert payload["route_elapsed_sim_s"] == pytest.approx(15.0)
    assert payload["terminal_stop_publish_count"] == 5

    incomplete_path = tmp_path / "incomplete.json"
    incomplete = SimpleNamespace(
        finished=False,
        succeeded=True,
        termination_reason="route_complete",
        safety_limits=limits,
        _terminal_stops_published=4,
        completion_path=incomplete_path,
        wall_watchdog_timer=Timer(),
        terminal_stop_timer=Timer(),
        get_logger=lambda: Logger(),
        _completion_payload=lambda: {"status": "completed"},
    )
    module.StudyRouteDriver._finalize_termination(incomplete)
    assert incomplete.succeeded is False
    assert incomplete.termination_reason == "terminal_stop_sequence_incomplete"
    assert incomplete_path.exists() is False

    complete_path = tmp_path / "complete.json"
    complete = SimpleNamespace(
        finished=False,
        succeeded=True,
        termination_reason="route_complete",
        safety_limits=limits,
        _terminal_stops_published=5,
        completion_path=complete_path,
        wall_watchdog_timer=Timer(),
        terminal_stop_timer=Timer(),
        get_logger=lambda: Logger(),
        _completion_payload=lambda: {
            "status": "completed",
            "route_complete": True,
            "end_reason": "route_complete",
        },
    )
    module.StudyRouteDriver._finalize_termination(complete)
    assert complete.succeeded is True
    assert json.loads(complete_path.read_text(encoding="utf-8"))["route_complete"] is True
