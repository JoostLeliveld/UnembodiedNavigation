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

Two trigger modes (both latched):
  - DEGRADED (default): drop once /reliability/localization_degraded fires (debounced,
    latest, 0 false alarms) — the B2 condition.
  - EARLY health-threshold (--trigger-health-below X): drop as soon as the continuous
    health h drops below X (e.g. 0.5), well before the debounced DEGRADED, to attack the
    detection-latency transient — the predictive B2p condition. The envelope showed benign
    drifts keep h > 0.55, so a 0.5 threshold does not false-trigger.

  in : /perception/pixel_pose_faulted            (PoseStamped, the camera measurement)
  in : /reliability/localization_degraded         (Bool)  OR
       /reliability/localization_health           (Float64MultiArray [h, state, ...]) if --trigger-health-below
  out: /perception/pixel_pose_health_gated        (PoseStamped; planner pixel_topic points here)

Offline check: python3 health_measurement_gate.py --selftest
"""
from __future__ import annotations

import argparse

IDX_HEALTH = 0  # /reliability/localization_health layout: [h, state_code, nis_ewma, bias, policy]


def pass_measurement(reject_latched):
    """Pure core: forward the measurement iff not (latched) rejected."""
    return not reject_latched


def health_triggers(h, *, below):
    """Pure core: does continuous health h cross the early-reject threshold?"""
    return h < below


def _selftest():
    assert pass_measurement(False) is True     # healthy -> forward
    assert pass_measurement(True) is False     # rejected -> drop (reject the bad camera)
    assert health_triggers(0.9, below=0.5) is False   # healthy h -> no early trigger
    assert health_triggers(0.3, below=0.5) is True    # degrading h -> early trigger
    assert health_triggers(0.56, below=0.5) is False  # benign-floor h stays above -> no false trigger
    print("SELFTEST PASS (forward while healthy; drop once latched; early health<X trigger)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in-topic", default="/perception/pixel_pose_faulted")
    ap.add_argument("--out-topic", default="/perception/pixel_pose_health_gated")
    ap.add_argument("--degraded-topic", default="/reliability/localization_degraded")
    ap.add_argument("--health-topic", default="/reliability/localization_health")
    ap.add_argument("--trigger-health-below", type=float, default=0.0,
                    help="EARLY predictive trigger: reject once continuous health h < this "
                         "(>0 enables; uses --health-topic instead of the debounced DEGRADED flag)")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool, Float64MultiArray

    early = args.trigger_health_below > 0.0

    class Gate(Node):
        def __init__(self):
            super().__init__("health_measurement_gate")
            self.latched = False
            self.n_fwd = 0
            self.n_drop = 0
            if early:
                self.create_subscription(Float64MultiArray, args.health_topic, self._health, 10)
            else:
                self.create_subscription(Bool, args.degraded_topic, self._deg, 10)
            self.create_subscription(PoseStamped, args.in_topic, self._meas, 20)
            self.pub = self.create_publisher(PoseStamped, args.out_topic, 10)
            self.get_logger().info(
                f"health measurement gate: {args.in_topic} -> {args.out_topic}; "
                + (f"EARLY trigger h<{args.trigger_health_below}" if early else "trigger=DEGRADED"))

        def _deg(self, msg):
            if bool(msg.data) and not self.latched:
                self.get_logger().warn("localization DEGRADED -> rejecting camera (drive on odom)")
                self.latched = True

        def _health(self, msg):
            if len(msg.data) > IDX_HEALTH and not self.latched:
                if health_triggers(float(msg.data[IDX_HEALTH]), below=args.trigger_health_below):
                    self.get_logger().warn(
                        f"health {msg.data[IDX_HEALTH]:.2f} < {args.trigger_health_below} "
                        f"-> EARLY reject camera (drive on odom)")
                    self.latched = True

        def _meas(self, msg):
            if pass_measurement(self.latched):
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
