#!/usr/bin/env python3
"""Health-gated measurement filter (Tier-1 B2) — a PoseStamped interposer.

The health-aware rejection response: pass the camera measurement through while
localization is healthy, but once the online health monitor latches DEGRADED
(`/reliability/localization_degraded`), STOP fusing that camera — drop the
measurement so the planner's EKF rides dead-reckoning (odom) instead of a biased
camera. This is the correct response to a *calibration* fault (reject the bad
sensor, keep moving), as opposed to a blunt safe-STOP of the robot.

Contrast with the fixed per-frame NIS gate: the NIS gate tests each frame in
isolation and lets a slow/moderate consistent drift through (each frame's NIS is
marginal) until the bias is large; the health monitor integrates the innovation
bias over time (EWMA), so it catches the accumulating drift the per-frame gate
misses. Latched: once the camera is judged degraded it stays rejected (conservative;
a real system would require an explicit re-commission / recovery).

  in : /perception/pixel_pose_faulted            (PoseStamped, the camera measurement)
  in : /reliability/localization_degraded         (Bool, from the health monitor)
  out: /perception/pixel_pose_health_gated        (PoseStamped; planner pixel_topic points here)

Offline check: python3 health_measurement_gate.py --selftest
"""
from __future__ import annotations

import argparse


def pass_measurement(degraded_latched):
    """Pure core: forward the measurement iff not (latched) degraded."""
    return not degraded_latched


def _selftest():
    assert pass_measurement(False) is True     # healthy -> forward
    assert pass_measurement(True) is False     # degraded -> drop (reject the bad camera)
    print("SELFTEST PASS (forward while healthy; drop once latched DEGRADED)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in-topic", default="/perception/pixel_pose_faulted")
    ap.add_argument("--out-topic", default="/perception/pixel_pose_health_gated")
    ap.add_argument("--degraded-topic", default="/reliability/localization_degraded")
    ap.add_argument("--no-latch", action="store_true",
                    help="follow the live degraded flag instead of latching (default latches)")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool

    class Gate(Node):
        def __init__(self):
            super().__init__("health_measurement_gate")
            self.degraded = False
            self.latched = False
            self.n_fwd = 0
            self.n_drop = 0
            self.create_subscription(Bool, args.degraded_topic, self._deg, 10)
            self.create_subscription(PoseStamped, args.in_topic, self._meas, 20)
            self.pub = self.create_publisher(PoseStamped, args.out_topic, 10)
            self.get_logger().info(
                f"health measurement gate: {args.in_topic} -> {args.out_topic}; latch={not args.no_latch}")

        def _deg(self, msg):
            if msg.data and not self.degraded:
                self.get_logger().warn("localization DEGRADED -> rejecting camera (drive on odom)")
            self.degraded = bool(msg.data)
            if self.degraded and not args.no_latch:
                self.latched = True

        def _meas(self, msg):
            reject = self.latched or (self.degraded if args.no_latch else self.latched)
            if pass_measurement(reject):
                self.pub.publish(msg)
                self.n_fwd += 1
            else:
                self.n_drop += 1

    rclpy.init()
    node = Gate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
