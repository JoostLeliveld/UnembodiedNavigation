#!/usr/bin/env python3
"""Ground-truth path follower for reference-controlled commissioning drives.

Ground truth is used only by this acquisition controller and its trace. It is not
published as a sensor measurement and must not be consumed by perception, support,
the estimator, or the planner.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import signal
from typing import Any, Sequence


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


@dataclass(frozen=True)
class ControlDecision:
    linear_mps: float
    angular_radps: float
    distance_to_goal_m: float
    heading_error_rad: float
    complete: bool


def project_to_path(
    pose: Sequence[float],
    points: Sequence[Sequence[float]],
    start_leg: int,
    max_advance: int = 1,
) -> tuple[int, float, float]:
    """Closest point on the polyline at or just after `start_leg`.

    Returns the leg index, the fraction along that leg, and the signed cross-track error.

    A route that revisits a corridor -- an outbound and a return sweep along the same
    apron, say -- puts two legs within millimetres of the robot, and an unrestricted
    nearest-point search will latch onto the later one and silently skip everything in
    between.  Advancing at most `max_advance` legs per call keeps progress monotone and
    local, so the robot follows the path it is on rather than the nearest piece of tape.
    """
    x, y = float(pose[0]), float(pose[1])
    best = (start_leg, 0.0, math.inf, 0.0)
    last_leg = min(start_leg + int(max_advance), len(points) - 2)
    for leg in range(start_leg, last_leg + 1):
        ax, ay = float(points[leg][0]), float(points[leg][1])
        bx, by = float(points[leg + 1][0]), float(points[leg + 1][1])
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        if span < 1e-12:
            continue
        fraction = clamp(((x - ax) * dx + (y - ay) * dy) / span, 0.0, 1.0)
        px, py = ax + dx * fraction, ay + dy * fraction
        distance = math.hypot(x - px, y - py)
        if distance < best[2]:
            cross = (dx * (y - ay) - dy * (x - ax)) / math.sqrt(span)
            best = (leg, fraction, distance, cross)
    return best[0], best[1], best[3]


def lookahead_point(
    points: Sequence[Sequence[float]],
    leg: int,
    fraction: float,
    lookahead_m: float,
) -> tuple[list[float], bool]:
    """Walk `lookahead_m` further along the path from (leg, fraction)."""
    remaining = float(lookahead_m)
    current_leg, current_fraction = leg, fraction
    while current_leg < len(points) - 1:
        ax, ay = float(points[current_leg][0]), float(points[current_leg][1])
        bx, by = float(points[current_leg + 1][0]), float(points[current_leg + 1][1])
        length = math.hypot(bx - ax, by - ay)
        left = length * (1.0 - current_fraction)
        if left >= remaining:
            advanced = current_fraction + remaining / length if length > 0 else 1.0
            return [ax + (bx - ax) * advanced, ay + (by - ay) * advanced], False
        remaining -= left
        current_leg += 1
        current_fraction = 0.0
    return [float(points[-1][0]), float(points[-1][1])], True


def control_to_goal(
    pose: Sequence[float],
    goal_xy: Sequence[float],
    *,
    nominal_cruise_mps: float,
    maximum_angular_speed_radps: float,
    heading_gain: float,
    heading_stop_rad: float,
    slowdown_distance_m: float,
    arrival_radius_m: float,
    is_final: bool = True,
) -> ControlDecision:
    """Drive toward one goal.

    On a multi-leg route the goal is a steering aim point that slides along the path, so it
    never reports arrival and is never decelerated into: the drive stays continuous through
    a corner.  Only the final goal uses the arrival radius and the slowdown ramp.
    """
    x, y, yaw = (float(value) for value in pose)
    gx, gy = (float(value) for value in goal_xy)
    distance = math.hypot(gx - x, gy - y)
    desired = math.atan2(gy - y, gx - x)
    error = wrap_angle(desired - yaw)
    if is_final and distance <= arrival_radius_m:
        return ControlDecision(0.0, 0.0, distance, error, True)
    angular = clamp(
        heading_gain * error,
        -maximum_angular_speed_radps,
        maximum_angular_speed_radps,
    )
    if abs(error) >= heading_stop_rad:
        linear = 0.0
    else:
        distance_scale = clamp(distance / slowdown_distance_m, 0.0, 1.0) if is_final else 1.0
        heading_scale = max(math.cos(error), 0.0)
        linear = nominal_cruise_mps * distance_scale * heading_scale
    return ControlDecision(linear, angular, distance, error, False)


def _load_drive(protocol_path: Path, drive_id: str) -> tuple[dict[str, Any], dict[str, Any], list[list[float]]]:
    import yaml

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    matches = [drive for drive in protocol["drives"] if drive["id"] == drive_id]
    if len(matches) != 1:
        raise ValueError(f"drive id must identify exactly one drive, got {drive_id!r}")
    drive = matches[0]
    points = [list(map(float, point)) for point in protocol["routes"][drive["route"]]["points"]]
    if len(points) < 2:
        raise ValueError(f"route {drive['route']!r} needs at least two points")
    if drive["direction"] == "reverse":
        points.reverse()
    return protocol, drive, points


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--drive-id", required=True)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    args = parser.parse_args()

    protocol, drive, points = _load_drive(args.protocol.resolve(), args.drive_id)
    cfg = protocol["controller"]
    if float(cfg["nominal_cruise_mps"]) != 1.0:
        raise RuntimeError("commissioning controller is locked to a 1.0 m/s nominal cruise")
    for forbidden in cfg.get("forbidden_nominal_speeds_mps", []):
        if math.isclose(float(cfg["nominal_cruise_mps"]), float(forbidden), abs_tol=1e-12):
            raise RuntimeError("nominal cruise is explicitly forbidden by the protocol")

    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from tf2_msgs.msg import TFMessage

    class ReferencePoseController(Node):
        def __init__(self) -> None:
            super().__init__(
                "reference_pose_commissioning_controller",
                parameter_overrides=[Parameter("use_sim_time", value=True)],
            )
            self.pose: tuple[float, float, float] | None = None
            self.pose_receipt_s = -math.inf
            self.complete = False
            self.failed = False
            self.stop_ticks = 0
            self.trace = args.trace_jsonl.open("x", encoding="utf-8")
            self.publisher = self.create_publisher(Twist, str(cfg["command_topic"]), 10)
            self.subscription = self.create_subscription(
                TFMessage, str(cfg["reference_topic"]), self._reference_cb, 50
            )
            self.timer = self.create_timer(1.0 / float(cfg["control_rate_hz"]), self._tick)

        @staticmethod
        def _yaw(rotation) -> float:
            return math.atan2(
                2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
                1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
            )

        def _reference_cb(self, message: TFMessage) -> None:
            now = float(self.get_clock().now().nanoseconds) * 1e-9
            for transform in message.transforms:
                if transform.child_frame_id == str(cfg["robot_child_frame_id"]):
                    self.pose = (
                        float(transform.transform.translation.x),
                        float(transform.transform.translation.y),
                        self._yaw(transform.transform.rotation),
                    )
                    self.pose_receipt_s = now
                    return

        def _publish(self, linear: float, angular: float) -> None:
            message = Twist()
            message.linear.x = float(linear)
            message.angular.z = float(angular)
            self.publisher.publish(message)

        def _record(
            self,
            state: str,
            decision: ControlDecision | None,
            extra: dict[str, Any] | None = None,
        ) -> None:
            now_ns = int(self.get_clock().now().nanoseconds)
            row: dict[str, Any] = {
                "schema": "reference_pose_controller_trace.v1",
                "drive_id": drive["id"],
                "stamp_ns": now_ns,
                "state": state,
                "pose": list(self.pose) if self.pose is not None else None,
            }
            if decision is not None:
                row.update(asdict(decision))
            if extra:
                row.update(extra)
            self.trace.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self.trace.flush()

        def _finish(self, *, failed: bool, reason: str, decision: ControlDecision | None = None) -> None:
            self.failed = bool(failed)
            self.complete = not failed
            self._publish(0.0, 0.0)
            self._record(reason, decision)
            _atomic_json(args.status_json, {
                "schema": "reference_pose_controller_status.v1",
                "drive_id": drive["id"],
                "complete": self.complete,
                "failed": self.failed,
                "reason": reason,
                "final_pose": list(self.pose) if self.pose is not None else None,
                "route_goal": points[-1],
                "route_waypoint_count": len(points),
                "legs_completed": int(getattr(self, "leg_index", 1)),
                "nominal_cruise_mps": float(cfg["nominal_cruise_mps"]),
                "stationary_anchor_complete": bool(
                    getattr(self, "stationary_anchor_complete", False)
                ),
            })

        def _anchor_tick(self, now: float) -> bool:
            anchor = cfg.get("stationary_anchor", {})
            if not bool(anchor.get("enabled", False)):
                self.stationary_anchor_complete = True
                return True
            offsets = [float(value) for value in anchor["heading_offsets_deg"]]
            if not hasattr(self, "anchor_index"):
                self.anchor_index = 0
                self.anchor_hold_start = None
                self.route_heading = math.atan2(
                    points[1][1] - points[0][1], points[1][0] - points[0][0]
                )
            if self.anchor_index >= len(offsets):
                self.stationary_anchor_complete = True
                return True
            target = wrap_angle(self.route_heading + math.radians(offsets[self.anchor_index]))
            error = wrap_angle(target - self.pose[2])
            tolerance = float(anchor["heading_tolerance_rad"])
            if abs(error) > tolerance:
                self.anchor_hold_start = None
                angular = clamp(
                    float(cfg["heading_gain"]) * error,
                    -float(cfg["maximum_angular_speed_radps"]),
                    float(cfg["maximum_angular_speed_radps"]),
                )
                self._publish(0.0, angular)
                self._record("stationary_anchor_turn", None, {
                    "anchor_index": self.anchor_index,
                    "anchor_target_yaw_rad": target,
                    "anchor_heading_error_rad": error,
                    "linear_mps": 0.0,
                    "angular_radps": angular,
                })
                return False
            self._publish(0.0, 0.0)
            if self.anchor_hold_start is None:
                self.anchor_hold_start = now
            held_s = max(now - self.anchor_hold_start, 0.0)
            self._record("stationary_anchor_hold", None, {
                "anchor_index": self.anchor_index,
                "anchor_target_yaw_rad": target,
                "anchor_heading_error_rad": error,
                "anchor_held_s": held_s,
                "linear_mps": 0.0,
                "angular_radps": 0.0,
            })
            if held_s >= float(anchor["hold_s"]):
                self.anchor_index += 1
                self.anchor_hold_start = None
            return False

        def _tick(self) -> None:
            if self.complete or self.failed:
                self._publish(0.0, 0.0)
                self.stop_ticks += 1
                self._record("terminal_hold", None, {
                    "linear_mps": 0.0,
                    "angular_radps": 0.0,
                })
                if self.failed and self.stop_ticks >= 5:
                    rclpy.shutdown()
                return
            now = float(self.get_clock().now().nanoseconds) * 1e-9
            if self.pose is None:
                self._publish(0.0, 0.0)
                return
            if now - self.pose_receipt_s > float(cfg["reference_timeout_s"]):
                self._finish(failed=True, reason="reference_timeout")
                return
            contact_topic = str(cfg.get("contact_ready_topic", "/world_contacts"))
            required = int(cfg.get("contact_ready_min_publishers", 1))
            if not getattr(self, "contact_ready", False):
                try:
                    publishers = int(self.count_publishers(contact_topic))
                except Exception:
                    publishers = -1
                if publishers < required:
                    if not hasattr(self, "contact_wait_start"):
                        self.contact_wait_start = now
                    waited = now - self.contact_wait_start
                    if waited > float(cfg.get("contact_ready_timeout_s", 60.0)):
                        self._finish(failed=True, reason="contact_channel_never_ready")
                        return
                    self._publish(0.0, 0.0)
                    self._record("waiting_for_contact_channel", None, {
                        "contact_topic": contact_topic,
                        "contact_topic_publishers": publishers,
                        "waited_s": waited,
                    })
                    return
                if not hasattr(self, "contact_seen_at"):
                    self.contact_seen_at = now
                settle = float(cfg.get("contact_ready_settle_s", 1.0))
                if now - self.contact_seen_at < settle:
                    self._publish(0.0, 0.0)
                    self._record("contact_channel_settling", None, {
                        "contact_topic": contact_topic,
                        "contact_topic_publishers": publishers,
                        "settled_s": now - self.contact_seen_at,
                    })
                    return
                self.contact_ready = True
                self._record("contact_channel_ready", None, {
                    "contact_topic": contact_topic,
                    "contact_topic_publishers": publishers,
                    "waited_s": now - getattr(self, "contact_wait_start", now),
                })
            if not hasattr(self, "start_checked"):
                self.start_checked = True
                start_error = math.hypot(self.pose[0] - points[0][0], self.pose[1] - points[0][1])
                if start_error > float(cfg["start_tolerance_m"]):
                    self._finish(failed=True, reason="start_pose_outside_tolerance")
                    return
            if not self._anchor_tick(now):
                return
            # Follow the polyline as a path, the way a driver tracks a painted line: aim at a
            # point a fixed distance ahead ON the path rather than at the next corner, so
            # corners are rounded smoothly and the robot returns to the line after each one.
            if not hasattr(self, "leg_index"):
                self.leg_index = 0
            single_leg = len(points) == 2
            lookahead = float(cfg.get("lookahead_m", 0.45))
            leg, fraction, cross_track = project_to_path(
                self.pose, points, self.leg_index,
                max_advance=int(cfg.get("path_max_leg_advance", 1)),
            )
            self.leg_index = leg
            target, at_end = lookahead_point(points, leg, fraction, lookahead)
            remaining = math.hypot(points[-1][0] - self.pose[0], points[-1][1] - self.pose[1])
            is_final = at_end or (leg >= len(points) - 2 and remaining <= lookahead)
            decision = control_to_goal(
                self.pose,
                points[-1] if is_final else target,
                nominal_cruise_mps=float(cfg["nominal_cruise_mps"]),
                maximum_angular_speed_radps=float(cfg["maximum_angular_speed_radps"]),
                heading_gain=float(cfg["heading_gain"]),
                heading_stop_rad=float(cfg["heading_stop_rad"]),
                slowdown_distance_m=float(cfg["slowdown_distance_m"]),
                arrival_radius_m=float(cfg["arrival_radius_m"]),
                is_final=is_final or single_leg,
            )
            if decision.complete and is_final:
                self._finish(failed=False, reason="route_complete", decision=decision)
                return
            self._publish(decision.linear_mps, decision.angular_radps)
            self._record("tracking", decision, {
                "leg_index": leg,
                "leg_count": len(points) - 1,
                "cross_track_error_m": cross_track,
            })

        def close(self) -> None:
            if rclpy.ok():
                try:
                    self._publish(0.0, 0.0)
                except Exception:
                    pass
            if not self.trace.closed:
                self.trace.flush()
                self.trace.close()

    rclpy.init()
    node = ReferencePoseController()
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if node.complete and not node.failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
