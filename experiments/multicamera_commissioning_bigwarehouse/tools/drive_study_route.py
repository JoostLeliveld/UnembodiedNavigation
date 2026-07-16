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
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import yaml

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
except ImportError:  # pragma: no cover - enables pure helper tests without ROS.
    rclpy = None
    Twist = Any
    Odometry = Any
    Node = object
    Parameter = Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_STUDY = REPO / "experiments/multicamera_commissioning_bigwarehouse/config/study.yaml"


def _wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _yaw_from_quaternion(quaternion: Any) -> float:
    siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


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
        self.waypoint_index = 0
        self.latest_pose: tuple[float, float, float] | None = None
        self.started_at_s: float | None = None
        self.finished = False
        self.command_publisher = self.create_publisher(Twist, command_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_callback, 20)
        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"following {len(self.waypoints)} waypoint(s) from {odom_topic} on {command_topic}; "
            f"speed={self.speed_mps:.3f} m/s"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _odom_callback(self, message: Odometry) -> None:
        position = message.pose.pose.position
        self.latest_pose = (
            float(position.x),
            float(position.y),
            _yaw_from_quaternion(message.pose.pose.orientation),
        )
        if self.started_at_s is None:
            self.started_at_s = self._now_s()

    def _publish_stop(self) -> None:
        self.command_publisher.publish(Twist())

    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self._publish_stop()
        self.timer.cancel()
        self.get_logger().info("Commissioning route complete; robot stopped.")
        # A stopped timer does not end ``rclpy.spin`` by itself.  Shutting down
        # the local ROS context makes a completed collection command exit
        # cleanly instead of relying on a caller to interrupt it.
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()

    def _tick(self) -> None:
        if self.finished or self.latest_pose is None or self.started_at_s is None:
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
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if rclpy is None:
        raise RuntimeError("rclpy is required; source the ROS workspace before running this driver")
    if args.speed_mps <= 0.0 or args.max_angular_radps <= 0.0 or args.heading_gain <= 0.0:
        parser.error("--speed-mps, --max-angular-radps, and --heading-gain must be positive")

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
    arrival_radius_m = float(args.arrival_radius_m or collection.get("arrival_radius_m", 0.20))
    start_hold_s = float(args.start_hold_s if args.start_hold_s is not None else collection.get("start_hold_s", 2.0))
    if arrival_radius_m <= 0.0 or start_hold_s < 0.0:
        parser.error("--arrival-radius-m must be positive and --start-hold-s cannot be negative")

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
    if args.manifest_path is not None:
        manifest_path = args.manifest_path.expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
