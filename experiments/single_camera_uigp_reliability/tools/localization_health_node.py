#!/usr/bin/env python3
"""Live localization-health monitor node (Tier-1 N3 core).

Subscribes to the planner's per-correction diagnostics (`/planner/pixel_correction_
diagnostics`, the exact NIS + innovation stream WP5 used offline), runs the online
innovation-health monitor (`reliability.health_ewma`), and publishes a continuous
health `h in (0,1)` + a debounced HEALTHY/SUSPECT/DEGRADED/RECOVERING state that the
planner adapter (N3) consumes to inflate R_plan / trigger safe degradation.

Ground-truth-free: consumes only operational innovation/NIS. No gt_* touched.

Topics:
  in : /planner/pixel_correction_diagnostics   std_msgs/Float64MultiArray
  out: /reliability/localization_health          Float64MultiArray [h, state_code, nis_ewma, bias_norm, policy_code]
       /reliability/localization_degraded        std_msgs/Bool

Offline check (no ROS): python3 localization_health_node.py --selftest
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "reliability"))
from reliability.health_ewma import (  # noqa: E402
    CalibrationHealthState,
    HealthDebouncer,
    HealthDebouncerConfig,
    InnovationHealthConfig,
    InnovationHealthMonitor,
)

# Float64MultiArray field indices (unicycle_planner_node._publish_pixel_correction_diagnostics)
IDX_INNOV_U, IDX_INNOV_V, IDX_NIS, IDX_ACCEPTED = 6, 7, 29, 30

POLICY = {
    CalibrationHealthState.HEALTHY: "accept",
    CalibrationHealthState.SUSPECT: "inflate",
    CalibrationHealthState.DEGRADED: "reject",
    CalibrationHealthState.RECOVERING: "slow_reentry",
}
STATE_CODE = {
    CalibrationHealthState.HEALTHY: 0.0,
    CalibrationHealthState.SUSPECT: 1.0,
    CalibrationHealthState.DEGRADED: 2.0,
    CalibrationHealthState.RECOVERING: 3.0,
}
POLICY_CODE = {"accept": 0.0, "inflate": 1.0, "reject": 2.0, "slow_reentry": 3.0}


def step_health(monitor, debouncer, *, nis, innov_uv, accepted, inflate_h):
    """Pure core: one correction -> (health, state, policy). Testable without ROS."""
    h = monitor.update(
        nis=(nis if accepted and math.isfinite(nis) else None),
        innovation_uv=(innov_uv if accepted and all(math.isfinite(v) for v in innov_uv) else None),
        dropped=(not accepted),
        cross_disagreement=None,
    )
    state = debouncer.step(h >= inflate_h)
    return h, state, POLICY[state]


def _selftest():
    """Feed a synthetic healthy->drift NIS/innovation stream and assert the response."""
    import random
    rng = random.Random(0)
    mon = InnovationHealthMonitor(config=InnovationHealthConfig())
    deb = HealthDebouncer(config=HealthDebouncerConfig())
    inflate_h = 0.5
    healthy_states, drift_states, hs = [], [], []
    # 120 healthy frames: NIS ~ chi2_2 mean 2, tiny zero-mean innovation
    for _ in range(120):
        nis = max(0.0, 2.0 + rng.gauss(0, 1.0))
        innov = (rng.gauss(0, 0.5), rng.gauss(0, 0.5))
        h, st, _ = step_health(mon, deb, nis=nis, innov_uv=innov, accepted=True, inflate_h=inflate_h)
        healthy_states.append(st); hs.append(h)
    # 60 drift frames: NIS ramps up, innovation gains a persistent direction (bias)
    for k in range(60):
        nis = 2.0 + 0.5 * k
        innov = (1.5 + rng.gauss(0, 0.3), 0.8 + rng.gauss(0, 0.3))
        h, st, _ = step_health(mon, deb, nis=nis, innov_uv=innov, accepted=True, inflate_h=inflate_h)
        drift_states.append(st); hs.append(h)
    healthy_far = healthy_states[20:]           # after burn-in
    n_healthy_bad = sum(1 for s in healthy_far if s != CalibrationHealthState.HEALTHY)
    reached_degraded = any(s == CalibrationHealthState.DEGRADED for s in drift_states)
    print(f"healthy phase: HEALTHY for {healthy_far.count(CalibrationHealthState.HEALTHY)}/{len(healthy_far)} "
          f"(false-alarm states {n_healthy_bad})")
    print(f"drift phase: reached DEGRADED = {reached_degraded}; "
          f"health {hs[100]:.2f} (healthy) -> {hs[-1]:.2f} (drifted)")
    ok = (n_healthy_bad == 0) and reached_degraded and (hs[-1] < hs[100])
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inflate-h", type=float, default=0.5, help="health below this = inconsistent frame")
    ap.add_argument("--diag-topic", default="/planner/pixel_correction_diagnostics")
    ap.add_argument("--log-csv", default="", help="if set, append t_wall,h,state_code,nis,nis_ewma,bias,accepted,degraded per frame")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float64MultiArray

    class HealthNode(Node):
        def __init__(self):
            super().__init__("localization_health_monitor")
            self.mon = InnovationHealthMonitor(config=InnovationHealthConfig())
            self.deb = HealthDebouncer(config=HealthDebouncerConfig())
            self.inflate_h = float(args.inflate_h)
            self.create_subscription(Float64MultiArray, args.diag_topic, self._cb, 20)
            self.pub_h = self.create_publisher(Float64MultiArray, "/reliability/localization_health", 10)
            self.pub_d = self.create_publisher(Bool, "/reliability/localization_degraded", 10)
            self._csv = None
            if args.log_csv:
                self._csv = open(args.log_csv, "w", buffering=1)
                self._csv.write("t_wall,h,state_code,nis,nis_ewma,bias,accepted,degraded\n")
            self.get_logger().info(f"health monitor up; diag={args.diag_topic} inflate_h={self.inflate_h}"
                                   + (f" log_csv={args.log_csv}" if args.log_csv else ""))

        def _cb(self, msg):
            d = msg.data
            if len(d) <= IDX_ACCEPTED:
                return
            nis = float(d[IDX_NIS]); innov = (float(d[IDX_INNOV_U]), float(d[IDX_INNOV_V]))
            accepted = float(d[IDX_ACCEPTED]) >= 0.5
            h, state, policy = step_health(self.mon, self.deb, nis=nis, innov_uv=innov,
                                           accepted=accepted, inflate_h=self.inflate_h)
            bias = math.hypot(*self.mon.bias_ewma)
            degraded = (state == CalibrationHealthState.DEGRADED)
            out = Float64MultiArray()
            out.data = [float(h), STATE_CODE[state], float(self.mon.nis_ewma), float(bias), POLICY_CODE[policy]]
            self.pub_h.publish(out)
            self.pub_d.publish(Bool(data=degraded))
            if self._csv is not None:
                self._csv.write(f"{time.time():.3f},{h:.4f},{STATE_CODE[state]:.0f},"
                                f"{nis if math.isfinite(nis) else float('nan'):.4f},"
                                f"{self.mon.nis_ewma:.4f},{bias:.4f},{int(accepted)},{int(degraded)}\n")

    rclpy.init()
    node = HealthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
