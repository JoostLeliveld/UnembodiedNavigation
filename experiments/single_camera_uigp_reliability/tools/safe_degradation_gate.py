#!/usr/bin/env python3
"""Safe-degradation actuation gate (Tier-1 N3, minimal) — a /cmd_vel interposer.

The minimal safe-degradation response: when the localization-health monitor reports
DEGRADED (`/reliability/localization_degraded` = True), the robot must not keep
driving blind on a bad external-camera pose. This gate sits between the local
tracker and the sim: it passes the tracker's command through while HEALTHY and
scales it toward a stop while DEGRADED (safe-stop / slow). No planner-core surgery
— the tracker publishes to `--in-topic`, this gate republishes to `--out-topic`
(= the sim's /cmd_vel) via a launch remap.

This is the *minimal* N3 actuation (safe-stop). The richer response — inflating
R_plan so the EFE planner re-routes toward observable regions — is a separate,
deferred hook into the EFE global solve.

Offline check: python3 safe_degradation_gate.py --selftest
"""
from __future__ import annotations

import argparse


def gate_scale(degraded, *, slow_factor):
    """Pure core: degraded flag -> velocity scale in [0, 1]."""
    return slow_factor if degraded else 1.0


def _selftest():
    assert gate_scale(False, slow_factor=0.0) == 1.0
    assert gate_scale(True, slow_factor=0.0) == 0.0      # full safe-stop
    assert gate_scale(True, slow_factor=0.25) == 0.25    # slow, not stop
    print("SELFTEST PASS (passthrough healthy; scale/stop on DEGRADED)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in-topic", default="/cmd_vel_tracker")
    ap.add_argument("--out-topic", default="/cmd_vel")
    ap.add_argument("--degraded-topic", default="/reliability/localization_degraded")
    ap.add_argument("--slow-factor", type=float, default=0.0, help="velocity scale while DEGRADED (0=stop)")
    ap.add_argument("--latch-stop", action="store_true",
                    help="once DEGRADED, stay stopped even if health recovers (conservative)")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool

    class Gate(Node):
        def __init__(self):
            super().__init__("safe_degradation_gate")
            self.degraded = False
            self.latched = False
            self.create_subscription(Bool, args.degraded_topic, self._deg, 10)
            self.create_subscription(Twist, args.in_topic, self._cmd, 20)
            self.pub = self.create_publisher(Twist, args.out_topic, 10)
            self.get_logger().info(
                f"safe-degradation gate: {args.in_topic} -> {args.out_topic}; "
                f"slow_factor={args.slow_factor} latch={args.latch_stop}")

        def _deg(self, msg):
            if msg.data and not self.degraded:
                self.get_logger().warn("localization DEGRADED -> safe-stop/slow engaged")
            self.degraded = bool(msg.data)
            if self.degraded and args.latch_stop:
                self.latched = True

        def _cmd(self, msg):
            active = self.latched or self.degraded
            s = gate_scale(active, slow_factor=args.slow_factor)
            out = Twist()
            out.linear.x = msg.linear.x * s
            out.linear.y = msg.linear.y * s
            out.angular.z = msg.angular.z * s
            self.pub.publish(out)

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
