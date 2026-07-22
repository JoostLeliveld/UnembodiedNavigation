#!/usr/bin/env python3
"""Single-camera fault injector (Tier-1) — a param-driven PoseStamped interposer.

Subscribes to the real camera measurement (`/perception/pixel_pose`, PoseStamped =
the pixel-derived world position), injects a controlled fault after a trigger time,
and republishes to a faulted topic. The planner is launched with
`pixel_topic:=<faulted topic>`, so NO planner-core surgery is needed. The fault is
CONTROLLED ABLATION (labelled): a drifting calibration, a partial occlusion, or a
stream dropout — the deployment failure modes static calibration can't detect.

Faults:
  drift     : add a ramping position bias b = min(rate*t_since_onset, max) along a fixed
              direction (a camera whose calibration has drifted -> biased world position)
  occlusion : drop a fraction (--drop-prob) of measurements (partial view blockage)
  dropout   : stop republishing entirely after onset (camera/stream death)

Offline check: python3 single_cam_fault_injector.py --selftest
"""
from __future__ import annotations

import argparse
import math
import random


def apply_fault(xy, *, t_since_onset, fault, rate, max_bias, direction_rad, drop_prob, rng):
    """Pure core: (x,y), seconds-since-onset -> faulted (x,y) or None if dropped.

    t_since_onset < 0 means the fault is not yet active (healthy passthrough).
    """
    if t_since_onset < 0.0 or fault == "none":
        return xy
    if fault == "dropout":
        return None
    if fault == "occlusion":
        return None if rng.random() < drop_prob else xy
    if fault == "drift":
        b = min(rate * t_since_onset, max_bias)
        return (xy[0] + b * math.cos(direction_rad), xy[1] + b * math.sin(direction_rad))
    raise ValueError(f"unknown fault {fault!r}")


def _selftest():
    rng = random.Random(0)
    # healthy passthrough before onset
    assert apply_fault((1.0, 2.0), t_since_onset=-1.0, fault="drift", rate=0.1, max_bias=1.0,
                       direction_rad=0.0, drop_prob=0.0, rng=rng) == (1.0, 2.0)
    # drift ramps then saturates
    d5 = apply_fault((0.0, 0.0), t_since_onset=5.0, fault="drift", rate=0.1, max_bias=1.0,
                     direction_rad=0.0, drop_prob=0.0, rng=rng)
    d50 = apply_fault((0.0, 0.0), t_since_onset=50.0, fault="drift", rate=0.1, max_bias=1.0,
                      direction_rad=0.0, drop_prob=0.0, rng=rng)
    assert abs(d5[0] - 0.5) < 1e-9 and abs(d50[0] - 1.0) < 1e-9, (d5, d50)
    # dropout = None after onset
    assert apply_fault((1.0, 1.0), t_since_onset=1.0, fault="dropout", rate=0, max_bias=0,
                       direction_rad=0, drop_prob=0, rng=rng) is None
    # occlusion drops ~drop_prob fraction
    kept = sum(1 for _ in range(2000) if apply_fault((0, 0), t_since_onset=1.0, fault="occlusion",
               rate=0, max_bias=0, direction_rad=0, drop_prob=0.5, rng=rng) is not None)
    assert 850 < kept < 1150, kept
    print("SELFTEST PASS (passthrough, drift ramp+saturate, dropout, occlusion fraction)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in-topic", default="/perception/pixel_pose")
    ap.add_argument("--out-topic", default="/perception/pixel_pose_faulted")
    ap.add_argument("--fault", choices=("none", "drift", "occlusion", "dropout"), default="drift")
    ap.add_argument("--fault-after-s", type=float, default=30.0, help="wall seconds after first msg")
    ap.add_argument("--drift-rate", type=float, default=0.10, help="m/s bias ramp")
    ap.add_argument("--drift-max-m", type=float, default=1.5)
    ap.add_argument("--drift-dir-deg", type=float, default=0.0)
    ap.add_argument("--drop-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())

    import time
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped

    rng = random.Random(args.seed)
    direction_rad = math.radians(args.drift_dir_deg)

    class Injector(Node):
        def __init__(self):
            super().__init__("single_cam_fault_injector")
            self.t0 = None
            self.onset_logged = False
            self.create_subscription(PoseStamped, args.in_topic, self._cb, 20)
            self.pub = self.create_publisher(PoseStamped, args.out_topic, 10)
            self.get_logger().info(
                f"fault injector: {args.in_topic} -> {args.out_topic}; fault={args.fault} "
                f"after {args.fault_after_s}s (drift {args.drift_rate} m/s cap {args.drift_max_m} m)")

        def _cb(self, msg):
            now = time.time()
            if self.t0 is None:
                self.t0 = now
            t_since_onset = (now - self.t0) - args.fault_after_s
            if t_since_onset >= 0.0 and not self.onset_logged:
                self.get_logger().warn(f"FAULT ONSET ({args.fault}) at t={now - self.t0:.1f}s after first msg")
                self.onset_logged = True
            xy = apply_fault((msg.pose.position.x, msg.pose.position.y), t_since_onset=t_since_onset,
                             fault=args.fault, rate=args.drift_rate, max_bias=args.drift_max_m,
                             direction_rad=direction_rad, drop_prob=args.drop_prob, rng=rng)
            if xy is None:
                return  # dropped (dropout / occlusion)
            msg.pose.position.x, msg.pose.position.y = float(xy[0]), float(xy[1])
            self.pub.publish(msg)

    rclpy.init()
    node = Injector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
