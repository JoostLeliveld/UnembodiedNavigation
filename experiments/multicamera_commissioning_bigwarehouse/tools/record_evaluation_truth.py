#!/usr/bin/env python3
"""Record EVALUATION-ONLY simulation ground truth during a commissioning run.

This recorder is deliberately a separate process and a separate output file
from ``record_operational_logs.py``.  It subscribes only to ``/ground_truth_tf``
(the unconditional ``ros_gz`` bridge of Gazebo's SceneBroadcaster
``dynamic_pose/info`` stream, see ``src/sim/launch/bringup_sim.launch.py``) and
writes ``ground_truth.csv``.  Nothing here may ever feed the operational
pipeline; the output exists so ``attach_evaluation_truth.py`` can populate the
``evaluation_only/`` half of the export split.

Example (run alongside the operational recorder):

    source install/setup.bash
    python3 experiments/multicamera_commissioning_bigwarehouse/tools/record_evaluation_truth.py \
      --out-dir logs/multicamera_commissioning_bigwarehouse/run_001/evaluation_only \
      --duration-s 120

"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from tf2_msgs.msg import TFMessage
except ImportError:  # pragma: no cover - only used in non-ROS tooling contexts
    rclpy = None
    Node = object
    Parameter = Any
    TFMessage = Any


TRUTH_FIELDS = ("stamp", "gt_x", "gt_y", "gt_yaw")


def _stamp_s(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _csv_float(value: float) -> str:
    number = float(value)
    return f"{number:.9f}" if math.isfinite(number) else ""


def _yaw_from_quaternion(rotation: Any) -> float:
    x = float(rotation.x)
    y = float(rotation.y)
    z = float(rotation.z)
    w = float(rotation.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class EvaluationTruthRecorder(Node):
    """Write the robot's true world pose stream to an evaluation-only CSV."""

    def __init__(
        self,
        *,
        out_dir: Path,
        topic: str,
        child_frame_id: str,
        min_interval_s: float,
        use_sim_time: bool,
    ) -> None:
        super().__init__("evaluation_truth_recorder")
        if use_sim_time:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.child_frame_id = child_frame_id
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.first_stamp: float | None = None
        self.last_stamp: float | None = None
        self._last_written_stamp: float | None = None
        self.count = 0

        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "ground_truth.csv"
        self.handle = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=TRUTH_FIELDS)
        self.writer.writeheader()
        self.create_subscription(TFMessage, topic, self._truth_callback, 50)

    def _truth_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if transform.child_frame_id != self.child_frame_id:
                continue
            stamp = _stamp_s(transform.header.stamp)
            if stamp <= 0.0:
                stamp = self.get_clock().now().nanoseconds * 1.0e-9
            if self.first_stamp is None:
                self.first_stamp = stamp
            self.last_stamp = stamp
            if (
                self._last_written_stamp is not None
                and stamp - self._last_written_stamp < self.min_interval_s
            ):
                return
            self._last_written_stamp = stamp
            self.writer.writerow(
                {
                    "stamp": _csv_float(stamp),
                    "gt_x": _csv_float(transform.transform.translation.x),
                    "gt_y": _csv_float(transform.transform.translation.y),
                    "gt_yaw": _csv_float(_yaw_from_quaternion(transform.transform.rotation)),
                }
            )
            self.count += 1
            if self.count % 50 == 0:
                self.handle.flush()
            return

    def elapsed_s(self) -> float:
        if self.first_stamp is None or self.last_stamp is None:
            return 0.0
        return self.last_stamp - self.first_stamp

    def close(self) -> None:
        self.handle.close()
        self.get_logger().info(f"wrote {self.count} ground-truth samples to {self.csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--topic", default="/ground_truth_tf")
    parser.add_argument("--child-frame-id", default="turtlebot3")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--min-interval-s", type=float, default=0.05)
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if rclpy is None:
        raise SystemExit("rclpy is required to record ground truth")

    rclpy.init()
    recorder = EvaluationTruthRecorder(
        out_dir=args.out_dir,
        topic=args.topic,
        child_frame_id=args.child_frame_id,
        min_interval_s=args.min_interval_s,
        use_sim_time=args.use_sim_time,
    )
    wall_start = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(recorder, timeout_sec=0.1)
            if args.duration_s > 0.0 and recorder.elapsed_s() >= args.duration_s:
                break
            # Wall-clock backstop so a paused simulation cannot hang the tool
            # forever: allow generous 10x real-time slack.
            if args.duration_s > 0.0 and time.monotonic() - wall_start > 10.0 * args.duration_s + 60.0:
                recorder.get_logger().warn("wall-clock backstop reached before sim duration")
                break
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        manifest = {
            "contains_ground_truth": True,
            "evaluation_only": True,
            "topic": args.topic,
            "child_frame_id": args.child_frame_id,
            "min_interval_s": float(args.min_interval_s),
            "sample_count": recorder.count,
            "first_stamp_s": recorder.first_stamp,
            "last_stamp_s": recorder.last_stamp,
            "note": (
                "Simulation truth for evaluation_only exports and calibration audits. "
                "Never an operational or model input (leakage firewall)."
            ),
        }
        (args.out_dir / "evaluation_truth_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        recorder.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
