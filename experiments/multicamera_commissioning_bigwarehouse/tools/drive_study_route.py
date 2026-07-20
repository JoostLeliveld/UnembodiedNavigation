#!/usr/bin/env python3
"""Follow one passive big-warehouse commissioning route from ``study.yaml``.

The driver is deliberately simple and is for data collection only.  It uses
Gazebo's control odometry to track the fixed commissioning waypoints, while
the recorder independently saves ``/odom_noisy`` for operational replay.  It
does not subscribe to any ground-truth topic and does not alter the camera
manager, planner, obstacle, or safety layers.

Run it in a third terminal after the study launch and operational recorder:

    source install/setup.bash
    python3 experiments/multicamera_commissioning_bigwarehouse/tools/drive_study_route.py \
      --route south_to_north_handover --speed-mps 0.12 \
      --manifest-path logs/multicamera_commissioning_bigwarehouse/run_001/raw/route_manifest.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import yaml

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.parameter import Parameter
except ImportError:  # pragma: no cover - enables pure helper tests without ROS.
    rclpy = None
    Twist = Any
    Odometry = Any
    Clock = Any
    ClockType = Any
    Node = object
    Parameter = Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_STUDY = REPO / "experiments/multicamera_commissioning_bigwarehouse/config/study.yaml"


@dataclass(frozen=True)
class RouteSafetyLimits:
    """Independent wall/simulation safety limits for one route pass."""

    initial_odom_timeout_s: float
    odom_freshness_timeout_s: float
    max_wall_runtime_s: float
    max_sim_runtime_s: float
    terminal_stop_publish_count: int = 5
    terminal_stop_period_s: float = 0.1

    def __post_init__(self) -> None:
        positive_fields = (
            "initial_odom_timeout_s",
            "odom_freshness_timeout_s",
            "max_wall_runtime_s",
            "max_sim_runtime_s",
            "terminal_stop_period_s",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        count = self.terminal_stop_publish_count
        if isinstance(count, bool) or int(count) != count or int(count) < 2:
            raise ValueError("terminal_stop_publish_count must be an integer >= 2")
        object.__setattr__(self, "terminal_stop_publish_count", int(count))


def route_distance_m(waypoints: list[tuple[float, float]]) -> float:
    """Return the polyline length of a route's spawn-local waypoints."""

    return float(
        sum(
            math.hypot(x1 - x0, y1 - y0)
            for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:])
        )
    )


def default_runtime_limits_s(
    waypoints: list[tuple[float, float]],
    *,
    speed_mps: float,
    start_hold_s: float,
) -> tuple[float, float]:
    """Return conservative simulation and wall deadlines for a route.

    Simulation time allows the ideal drive time plus at least 30 seconds for
    heading alignment and terminal settling.  Wall time permits an average
    real-time factor as low as 0.25 while still bounding a paused simulator.
    """

    speed = float(speed_mps)
    hold = float(start_hold_s)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("speed_mps must be finite and positive")
    if not math.isfinite(hold) or hold < 0.0:
        raise ValueError("start_hold_s must be finite and non-negative")
    ideal_drive_s = route_distance_m(waypoints) / speed
    sim_limit_s = hold + ideal_drive_s + max(30.0, 0.25 * ideal_drive_s)
    wall_limit_s = max(4.0 * sim_limit_s, sim_limit_s + 120.0)
    return float(sim_limit_s), float(wall_limit_s)


def route_watchdog_failure(
    limits: RouteSafetyLimits,
    *,
    wall_elapsed_s: float,
    sim_elapsed_s: float | None,
    odom_wall_age_s: float | None,
    odom_sim_age_s: float | None,
) -> str:
    """Return a stable failure reason, or an empty string while safe."""

    if wall_elapsed_s > limits.max_wall_runtime_s:
        return "max_wall_runtime_exceeded"
    if odom_wall_age_s is None:
        if wall_elapsed_s > limits.initial_odom_timeout_s:
            return "initial_odometry_timeout"
    elif odom_wall_age_s > limits.odom_freshness_timeout_s:
        return "odometry_stale_wall"
    if sim_elapsed_s is not None and sim_elapsed_s > limits.max_sim_runtime_s:
        return "max_sim_runtime_exceeded"
    if odom_sim_age_s is not None and odom_sim_age_s > limits.odom_freshness_timeout_s:
        return "odometry_stale_sim"
    return ""


def write_json_atomic_new(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically publish a complete JSON file without replacing prior data."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-linking a fully flushed same-directory temporary file publishes
        # it atomically and fails rather than clobbering a stale run artifact.
        os.link(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def _wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _yaw_from_quaternion(quaternion: Any) -> float:
    siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _message_stamp_s(message: Any) -> float | None:
    """Return a finite ROS header stamp when the message provides one."""

    try:
        stamp = message.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_route(
    study_path: str | Path,
    *,
    route_name: str,
    lateral_offset_m: float = 0.0,
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """Load a named straight route and shift it left of travel by an offset.

    The sign of ``lateral_offset_m`` is defined in the direction of travel:
    positive is left of the start-to-goal vector.  This makes the three offset
    values in the study configuration work identically in both directions.
    """

    path = Path(study_path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    collection = dict(config.get("collection") or {})
    selected = next(
        (dict(route) for route in collection.get("routes", []) if route.get("name") == route_name),
        None,
    )
    if selected is None:
        available = ", ".join(str(route.get("name")) for route in collection.get("routes", []))
        raise ValueError(f"Unknown route {route_name!r}; choose one of: {available}")

    start = dict(selected.get("start") or {})
    goal = dict(selected.get("goal") or {})
    start_xy = (float(start["x"]), float(start["y"]))
    goal_xy = (float(goal["x"]), float(goal["y"]))
    dx = goal_xy[0] - start_xy[0]
    dy = goal_xy[1] - start_xy[1]
    length = math.hypot(dx, dy)
    if length <= 1.0e-9:
        raise ValueError(f"Route {route_name!r} has coincident start and goal")
    left_normal = (-dy / length, dx / length)
    offset = float(lateral_offset_m)
    points = [
        (start_xy[0] + offset * left_normal[0], start_xy[1] + offset * left_normal[1]),
        (goal_xy[0] + offset * left_normal[0], goal_xy[1] + offset * left_normal[1]),
    ]
    return config, points


def route_start_yaw(study_path: str | Path, *, route_name: str) -> float:
    """Return the documented spawn yaw for a named study route."""

    config, _ = load_route(study_path, route_name=route_name)
    for route in config["collection"]["routes"]:
        if route["name"] == route_name:
            return float(route["start"].get("yaw", 0.0))
    raise AssertionError("load_route accepted a route that was not in the study")


def route_spawn_pose(
    study_path: str | Path,
    *,
    route_name: str,
    lateral_offset_m: float = 0.0,
) -> tuple[float, float, float]:
    """Return the documented warehouse-frame spawn pose for a route variant."""

    _config, world_waypoints = load_route(
        study_path,
        route_name=route_name,
        lateral_offset_m=lateral_offset_m,
    )
    x, y = world_waypoints[0]
    return x, y, route_start_yaw(study_path, route_name=route_name)


def world_to_spawn_local(
    world_waypoints: list[tuple[float, float]],
    *,
    spawn_x_m: float,
    spawn_y_m: float,
    spawn_yaw_rad: float,
) -> list[tuple[float, float]]:
    """Express warehouse waypoints in Gazebo DiffDrive's spawn-local odom frame."""

    cos_yaw = math.cos(spawn_yaw_rad)
    sin_yaw = math.sin(spawn_yaw_rad)
    local_points: list[tuple[float, float]] = []
    for world_x, world_y in world_waypoints:
        dx = float(world_x) - float(spawn_x_m)
        dy = float(world_y) - float(spawn_y_m)
        local_points.append((cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy))
    return local_points


def route_manifest(
    *,
    study_path: str | Path,
    route_name: str,
    lateral_offset_m: float,
    speed_mps: float,
    odom_topic: str,
    command_topic: str,
    world_waypoints: list[tuple[float, float]],
    local_waypoints: list[tuple[float, float]],
    spawn_pose_warehouse: tuple[float, float, float],
) -> dict[str, Any]:
    """Return the metadata that ties a collection log to its fixed route."""

    path = Path(study_path).expanduser().resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "study_config": str(path),
        "study_config_sha256": digest,
        "route": route_name,
        "lateral_offset_m": float(lateral_offset_m),
        "speed_mps": float(speed_mps),
        "control_odom_topic": str(odom_topic),
        "command_topic": str(command_topic),
        "spawn_pose_warehouse": {
            "x_m": float(spawn_pose_warehouse[0]),
            "y_m": float(spawn_pose_warehouse[1]),
            "yaw_rad": float(spawn_pose_warehouse[2]),
            "source": "documented_launch_spawn_not_ground_truth",
        },
        "waypoints_warehouse_xy_m": [{"x": float(x), "y": float(y)} for x, y in world_waypoints],
        "waypoints_spawn_local_xy_m": [{"x": float(x), "y": float(y)} for x, y in local_waypoints],
        "contains_ground_truth": False,
        "created_wall_time_s": time.time(),
    }


class StudyRouteDriver(Node):
    """Small proportional waypoint follower for a single commissioning pass."""

    def __init__(
        self,
        *,
        waypoints: list[tuple[float, float]],
        odom_topic: str,
        command_topic: str,
        speed_mps: float,
        max_angular_radps: float,
        heading_gain: float,
        arrival_radius_m: float,
        start_hold_s: float,
        use_sim_time: bool,
        safety_limits: RouteSafetyLimits,
        completion_path: str | Path | None = None,
        completion_context: dict[str, Any] | None = None,
        wall_time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            "bigwarehouse_study_route_driver",
            parameter_overrides=[Parameter("use_sim_time", value=bool(use_sim_time))],
        )
        if not waypoints:
            raise ValueError("At least one waypoint is required")
        self.waypoints = [(float(x), float(y)) for x, y in waypoints]
        self.speed_mps = float(speed_mps)
        self.max_angular_radps = float(max_angular_radps)
        self.heading_gain = float(heading_gain)
        self.arrival_radius_m = float(arrival_radius_m)
        self.start_hold_s = float(start_hold_s)
        self.safety_limits = safety_limits
        self.completion_path = (
            Path(completion_path).expanduser().resolve() if completion_path is not None else None
        )
        self.completion_context = dict(completion_context or {})
        self._wall_time_fn = wall_time_fn
        self.wall_started_at_s = self._wall_now_s()
        self.waypoint_index = 0
        self.latest_pose: tuple[float, float, float] | None = None
        self.started_at_s: float | None = None
        self.latest_odom_sim_stamp_s: float | None = None
        self.latest_odom_received_wall_s: float | None = None
        self._last_watchdog_sim_time_s: float | None = None
        self.terminating = False
        self.finished = False
        self.succeeded = False
        self.termination_reason = ""
        self._termination_sim_time_s: float | None = None
        self._termination_wall_elapsed_s: float | None = None
        self._terminal_stops_published = 0
        self.command_publisher = self.create_publisher(Twist, command_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_callback, 20)
        self.timer = self.create_timer(0.1, self._tick)
        # A ROS-time timer cannot enforce a wall deadman when /clock pauses.
        # Separate steady-clock timers own the watchdog and terminal-stop burst.
        steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.wall_watchdog_timer = self.create_timer(
            0.1,
            self._wall_watchdog_tick,
            clock=steady_clock,
        )
        self.terminal_stop_timer = self.create_timer(
            self.safety_limits.terminal_stop_period_s,
            self._terminal_stop_tick,
            clock=steady_clock,
        )
        self.get_logger().info(
            f"following {len(self.waypoints)} waypoint(s) from {odom_topic} on {command_topic}; "
            f"speed={self.speed_mps:.3f} m/s, odom_timeout="
            f"{self.safety_limits.odom_freshness_timeout_s:.2f} s, "
            f"sim_deadline={self.safety_limits.max_sim_runtime_s:.1f} s, "
            f"wall_deadline={self.safety_limits.max_wall_runtime_s:.1f} s"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _wall_now_s(self) -> float:
        return float(self._wall_time_fn())

    def _odom_callback(self, message: Odometry) -> None:
        received_sim_s = self._now_s()
        received_wall_s = self._wall_now_s()
        position = message.pose.pose.position
        pose = (
            float(position.x),
            float(position.y),
            _yaw_from_quaternion(message.pose.pose.orientation),
        )
        if not all(math.isfinite(value) for value in pose):
            self.get_logger().error("Ignoring non-finite control odometry; freshness watchdog remains armed.")
            return
        message_stamp_s = _message_stamp_s(message)
        odom_sim_stamp_s = received_sim_s if message_stamp_s is None else message_stamp_s
        if (
            self.latest_odom_sim_stamp_s is not None
            and odom_sim_stamp_s <= self.latest_odom_sim_stamp_s
        ):
            # Repeated or out-of-order messages must not keep a frozen pose
            # artificially fresh on the wall-clock watchdog.
            return
        self.latest_pose = pose
        self.latest_odom_sim_stamp_s = odom_sim_stamp_s
        self.latest_odom_received_wall_s = received_wall_s
        if self.started_at_s is None:
            self.started_at_s = received_sim_s

    def _publish_stop(self) -> None:
        self.command_publisher.publish(Twist())

    def _publish_terminal_stop(self) -> None:
        self._publish_stop()
        self._terminal_stops_published += 1

    def _watchdog_failure(self) -> str:
        now_wall_s = self._wall_now_s()
        now_sim_s = self._now_s()
        if (
            self._last_watchdog_sim_time_s is not None
            and now_sim_s < self._last_watchdog_sim_time_s - 1.0e-6
        ):
            return "simulation_time_reversed"
        self._last_watchdog_sim_time_s = now_sim_s
        wall_elapsed_s = max(0.0, now_wall_s - self.wall_started_at_s)
        sim_elapsed_s = (
            None if self.started_at_s is None else max(0.0, now_sim_s - self.started_at_s)
        )
        odom_wall_age_s = (
            None
            if self.latest_odom_received_wall_s is None
            else max(0.0, now_wall_s - self.latest_odom_received_wall_s)
        )
        odom_sim_age_s = (
            None
            if self.latest_odom_sim_stamp_s is None
            else max(0.0, now_sim_s - self.latest_odom_sim_stamp_s)
        )
        return route_watchdog_failure(
            self.safety_limits,
            wall_elapsed_s=wall_elapsed_s,
            sim_elapsed_s=sim_elapsed_s,
            odom_wall_age_s=odom_wall_age_s,
            odom_sim_age_s=odom_sim_age_s,
        )

    def _finish(self) -> None:
        self._begin_termination("route_complete", succeeded=True)

    def _begin_termination(self, reason: str, *, succeeded: bool) -> None:
        if self.terminating or self.finished:
            return
        self.terminating = True
        self.succeeded = bool(succeeded)
        self.termination_reason = str(reason)
        self._termination_sim_time_s = self._now_s()
        self._termination_wall_elapsed_s = max(0.0, self._wall_now_s() - self.wall_started_at_s)
        self.timer.cancel()
        self._publish_terminal_stop()
        level = self.get_logger().info if succeeded else self.get_logger().error
        level(
            f"Route terminating ({self.termination_reason}); publishing "
            f"{self.safety_limits.terminal_stop_publish_count} terminal stop commands."
        )
        if self._terminal_stops_published >= self.safety_limits.terminal_stop_publish_count:
            self._finalize_termination()

    def _completion_payload(self) -> dict[str, Any]:
        now_sim_s = self._termination_sim_time_s
        elapsed_sim_s = (
            None
            if now_sim_s is None or self.started_at_s is None
            else max(0.0, now_sim_s - self.started_at_s)
        )
        final_pose = None
        if self.latest_pose is not None:
            final_pose = {
                "x_m": float(self.latest_pose[0]),
                "y_m": float(self.latest_pose[1]),
                "yaw_rad": float(self.latest_pose[2]),
            }
        return {
            "schema_version": "bigwarehouse_route_completion_v1",
            "status": "completed",
            "route_complete": True,
            "end_reason": "route_complete",
            "completion_reason": "route_complete",
            "contains_ground_truth": False,
            "route_identity": dict(self.completion_context),
            "completed_wall_time_s": time.time(),
            "route_elapsed_wall_s": self._termination_wall_elapsed_s,
            "route_started_sim_time_s": self.started_at_s,
            "route_completed_sim_time_s": now_sim_s,
            "route_elapsed_sim_s": elapsed_sim_s,
            "waypoint_count": len(self.waypoints),
            "waypoints_reached": self.waypoint_index,
            "final_pose_spawn_local": final_pose,
            "terminal_stop_publish_count": self._terminal_stops_published,
            "safety_limits": asdict(self.safety_limits),
        }

    def _finalize_termination(self) -> None:
        if self.finished:
            return
        if (
            self.succeeded
            and self._terminal_stops_published
            < self.safety_limits.terminal_stop_publish_count
        ):
            self.succeeded = False
            self.termination_reason = "terminal_stop_sequence_incomplete"
        if self.succeeded and self.completion_path is not None:
            try:
                write_json_atomic_new(self.completion_path, self._completion_payload())
            except Exception as exc:
                self.succeeded = False
                self.termination_reason = "completion_artifact_write_failed"
                self.get_logger().error(
                    f"Route reached its goal but success artifact could not be published: {exc}"
                )
        self.finished = True
        self.wall_watchdog_timer.cancel()
        self.terminal_stop_timer.cancel()
        if self.succeeded:
            self.get_logger().info("Commissioning route complete; repeated stop sequence published.")
        else:
            self.get_logger().error(
                f"Commissioning route aborted safely: {self.termination_reason}. No success artifact was written."
            )
        # Stopped timers do not end ``rclpy.spin``. Shutting down the local
        # context makes success and watchdog failures return to the caller.
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()

    def publish_remaining_terminal_stops(self) -> None:
        """Synchronously finish the stop burst during an external interrupt."""

        while self._terminal_stops_published < self.safety_limits.terminal_stop_publish_count:
            self._publish_terminal_stop()
            if self._terminal_stops_published < self.safety_limits.terminal_stop_publish_count:
                time.sleep(self.safety_limits.terminal_stop_period_s)

    def _wall_watchdog_tick(self) -> None:
        if self.finished:
            return
        if self.terminating:
            return
        reason = self._watchdog_failure()
        if reason:
            self._begin_termination(reason, succeeded=False)
            return
        if self.latest_pose is None:
            # Assert zero command while waiting for the first control odometry.
            self._publish_stop()

    def _terminal_stop_tick(self) -> None:
        if self.finished or not self.terminating:
            return
        if self._terminal_stops_published < self.safety_limits.terminal_stop_publish_count:
            self._publish_terminal_stop()
        if self._terminal_stops_published >= self.safety_limits.terminal_stop_publish_count:
            self._finalize_termination()

    def _tick(self) -> None:
        if self.finished or self.terminating:
            return
        reason = self._watchdog_failure()
        if reason:
            self._begin_termination(reason, succeeded=False)
            return
        if self.latest_pose is None or self.started_at_s is None:
            self._publish_stop()
            return
        if self._now_s() - self.started_at_s < self.start_hold_s:
            self._publish_stop()
            return

        x, y, yaw = self.latest_pose
        target_x, target_y = self.waypoints[self.waypoint_index]
        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        if distance <= self.arrival_radius_m:
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                self._finish()
                return
            target_x, target_y = self.waypoints[self.waypoint_index]
            dx = target_x - x
            dy = target_y - y
            distance = math.hypot(dx, dy)
            self.get_logger().info(
                f"Reached waypoint {self.waypoint_index}; heading to ({target_x:.2f}, {target_y:.2f})"
            )

        desired_yaw = math.atan2(dy, dx)
        heading_error = _wrap_angle(desired_yaw - yaw)
        command = Twist()
        command.angular.z = max(
            -self.max_angular_radps,
            min(self.max_angular_radps, self.heading_gain * heading_error),
        )
        # Do not drive forwards into a large heading error.  The cosine taper
        # keeps straight collection passes straight without abrupt turns.
        if abs(heading_error) < math.pi / 2.0:
            command.linear.x = self.speed_mps * max(0.0, math.cos(heading_error))
        self.command_publisher.publish(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--route", required=True)
    parser.add_argument("--lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--speed-mps", type=float, required=True)
    parser.add_argument("--odom-topic", default="")
    parser.add_argument("--command-topic", default="")
    parser.add_argument("--arrival-radius-m", type=float, default=None)
    parser.add_argument("--start-hold-s", type=float, default=None)
    parser.add_argument("--max-angular-radps", type=float, default=0.8)
    parser.add_argument("--heading-gain", type=float, default=1.8)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument(
        "--completion-path",
        type=Path,
        default=None,
        help="Success-only JSON path (default: route_completion.json beside --manifest-path)",
    )
    parser.add_argument("--initial-odom-timeout-s", type=float, default=10.0)
    parser.add_argument("--odom-freshness-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--max-sim-runtime-s",
        type=float,
        default=None,
        help="Simulation-time deadline; default derives from route length and speed",
    )
    parser.add_argument(
        "--max-wall-runtime-s",
        type=float,
        default=None,
        help="Steady-wall-clock deadman; default permits average RTF down to 0.25",
    )
    parser.add_argument("--terminal-stop-publish-count", type=int, default=5)
    parser.add_argument("--terminal-stop-period-s", type=float, default=0.1)
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if rclpy is None:
        raise RuntimeError("rclpy is required; source the ROS workspace before running this driver")
    positive_controls = (args.speed_mps, args.max_angular_radps, args.heading_gain)
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_controls):
        parser.error("--speed-mps, --max-angular-radps, and --heading-gain must be positive")
    if not math.isfinite(args.lateral_offset_m):
        parser.error("--lateral-offset-m must be finite")

    study_path = args.study.expanduser().resolve()
    config, world_waypoints = load_route(
        study_path,
        route_name=str(args.route),
        lateral_offset_m=float(args.lateral_offset_m),
    )
    spawn_pose = route_spawn_pose(
        study_path,
        route_name=str(args.route),
        lateral_offset_m=float(args.lateral_offset_m),
    )
    local_waypoints = world_to_spawn_local(
        world_waypoints,
        spawn_x_m=spawn_pose[0],
        spawn_y_m=spawn_pose[1],
        spawn_yaw_rad=spawn_pose[2],
    )
    collection = dict(config["collection"])
    odom_topic = str(args.odom_topic or collection.get("control_odom_topic", "/odom"))
    command_topic = str(args.command_topic or collection.get("command_topic", "/cmd_vel"))
    arrival_radius_m = float(
        args.arrival_radius_m
        if args.arrival_radius_m is not None
        else collection.get("arrival_radius_m", 0.20)
    )
    start_hold_s = float(args.start_hold_s if args.start_hold_s is not None else collection.get("start_hold_s", 2.0))
    if (
        not math.isfinite(arrival_radius_m)
        or arrival_radius_m <= 0.0
        or not math.isfinite(start_hold_s)
        or start_hold_s < 0.0
    ):
        parser.error("--arrival-radius-m must be positive and --start-hold-s cannot be negative")

    default_sim_runtime_s, default_wall_runtime_s = default_runtime_limits_s(
        local_waypoints,
        speed_mps=float(args.speed_mps),
        start_hold_s=start_hold_s,
    )
    try:
        safety_limits = RouteSafetyLimits(
            initial_odom_timeout_s=float(args.initial_odom_timeout_s),
            odom_freshness_timeout_s=float(args.odom_freshness_timeout_s),
            max_sim_runtime_s=float(
                args.max_sim_runtime_s
                if args.max_sim_runtime_s is not None
                else default_sim_runtime_s
            ),
            max_wall_runtime_s=float(
                args.max_wall_runtime_s
                if args.max_wall_runtime_s is not None
                else default_wall_runtime_s
            ),
            terminal_stop_publish_count=int(args.terminal_stop_publish_count),
            terminal_stop_period_s=float(args.terminal_stop_period_s),
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    manifest_path = args.manifest_path.expanduser().resolve() if args.manifest_path is not None else None
    if args.completion_path is not None:
        completion_path = args.completion_path.expanduser().resolve()
    elif manifest_path is not None:
        completion_path = manifest_path.with_name("route_completion.json")
    else:
        completion_path = None
    if manifest_path is not None and completion_path == manifest_path:
        parser.error("--completion-path must be separate from --manifest-path")
    for label, output_path in (("manifest", manifest_path), ("completion", completion_path)):
        if output_path is not None and output_path.exists():
            parser.error(
                f"Refusing to reuse existing {label} path {output_path}; each route attempt needs new artifacts"
            )

    manifest = route_manifest(
        study_path=study_path,
        route_name=str(args.route),
        lateral_offset_m=float(args.lateral_offset_m),
        speed_mps=float(args.speed_mps),
        odom_topic=odom_topic,
        command_topic=command_topic,
        world_waypoints=world_waypoints,
        local_waypoints=local_waypoints,
        spawn_pose_warehouse=spawn_pose,
    )
    manifest.update(
        {
            "schema_version": "bigwarehouse_route_manifest_v2",
            "manifest_phase": "pre_run",
            "expected_route_distance_m": route_distance_m(local_waypoints),
            "safety_limits": asdict(safety_limits),
            "completion_artifact": str(completion_path) if completion_path is not None else None,
            "completion_status": "pending_route_success",
        }
    )
    manifest_sha256 = None
    if manifest_path is not None:
        try:
            write_json_atomic_new(manifest_path, manifest)
        except OSError as exc:
            parser.error(f"Could not publish pre-run manifest {manifest_path}: {exc}")
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    completion_context = {
        "route": str(args.route),
        "study_config_sha256": manifest["study_config_sha256"],
        "route_manifest": str(manifest_path) if manifest_path is not None else None,
        "route_manifest_sha256": manifest_sha256,
        "lateral_offset_m": float(args.lateral_offset_m),
        "speed_mps": float(args.speed_mps),
    }

    rclpy.init()
    node = StudyRouteDriver(
        waypoints=local_waypoints,
        odom_topic=odom_topic,
        command_topic=command_topic,
        speed_mps=float(args.speed_mps),
        max_angular_radps=float(args.max_angular_radps),
        heading_gain=float(args.heading_gain),
        arrival_radius_m=arrival_radius_m,
        start_hold_s=start_hold_s,
        use_sim_time=bool(args.use_sim_time),
        safety_limits=safety_limits,
        completion_path=completion_path,
        completion_context=completion_context,
    )
    interrupted = False
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        interrupted = True
        node._begin_termination("keyboard_interrupt", succeeded=False)
    finally:
        if not node.finished:
            if not node.terminating:
                node._begin_termination("spin_ended_before_route_complete", succeeded=False)
            node.publish_remaining_terminal_stops()
            node._finalize_termination()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if node.succeeded:
        return 0
    return 130 if interrupted else 2


if __name__ == "__main__":
    raise SystemExit(main())
