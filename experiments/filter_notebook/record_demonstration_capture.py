#!/usr/bin/env python3
"""Record a DEMONSTRATION capture: observations, odometry and truth, nothing more.

`record_operational_logs.py` is the sanctioned recorder and it requires a campaign
ledger row, frozen-config hashes and a projection calibration, because its output
is campaign evidence. A notebook capture is not evidence, and inventing a ledger
row to get past those guards would defeat their purpose. So this records the same
three streams under an explicit `demonstration` label.

It writes exactly the columns `experiments/operational_residual_rcond/rcond_common.py`
consumes, so the capture loads through the ordinary reader:

    raw/experiment.csv                stamp, odom_noisy_x/y, odom_noisy_cov_xx/xy/yy
    raw/camera_<X>_perception.csv     diag_stamp, detected, obs_u, obs_v
    evaluation_only/ground_truth.csv  stamp, gt_x, gt_y, gt_yaw

Two things it deliberately copies from the sanctioned recorder:

* **odometry is rotated into warehouse coordinates.** `/odom_noisy` starts at the
  origin of its own frame at the spawn pose; the reader expects warehouse metres.
  The same `--odom-origin-*` triple is applied here, to the mean and to the
  covariance.
* **it does not subscribe to truth for any operational purpose.** The truth stream
  is written to `evaluation_only/`, which the reader loads only through
  `load_truth_table`, and no filter in this repo may read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.parameter import Parameter
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

# The four-camera world's cameras. A capture in the single-camera AWS world passes
# `--cameras camera_A`, so it does not create three empty CSVs that later look like a
# camera that saw nothing rather than a camera that does not exist.
DEFAULT_CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class DemonstrationRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("demonstration_capture_recorder")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.args = args
        self.cos = math.cos(args.odom_origin_yaw_rad)
        self.sin = math.sin(args.odom_origin_yaw_rad)

        raw = args.out_dir / "raw"
        truth_dir = args.out_dir / "evaluation_only"
        raw.mkdir(parents=True, exist_ok=True)
        truth_dir.mkdir(parents=True, exist_ok=True)

        self._odom_handle = (raw / "experiment.csv").open("w", newline="", encoding="utf-8")
        self._odom = csv.writer(self._odom_handle)
        self._odom.writerow(["stamp", "odom_noisy_x", "odom_noisy_y",
                             "odom_noisy_cov_xx", "odom_noisy_cov_xy",
                             "odom_noisy_cov_yy"])

        self._truth_handle = (truth_dir / "ground_truth.csv").open(
            "w", newline="", encoding="utf-8")
        self._truth = csv.writer(self._truth_handle)
        self._truth.writerow(["stamp", "gt_x", "gt_y", "gt_yaw"])

        self.cameras = tuple(args.cameras)
        self._perception_handles, self._perception = {}, {}
        for camera in self.cameras:
            handle = (raw / f"{camera}_perception.csv").open(
                "w", newline="", encoding="utf-8")
            self._perception_handles[camera] = handle
            writer = csv.writer(handle)
            writer.writerow(["diag_stamp", "detected", "obs_u", "obs_v"])
            self._perception[camera] = writer

        self.counts = {c: 0 for c in self.cameras}
        self.counts["odom"] = 0
        self.counts["truth"] = 0

        volatile = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, args.odom_topic, self._on_odom, volatile)
        self.create_subscription(TFMessage, args.truth_topic, self._on_truth, volatile)
        for camera in self.cameras:
            self.create_subscription(
                String, f"{args.observation_prefix}/{camera}",
                lambda msg, _c=camera: self._on_observation(msg, _c), volatile)

        self._started = time.time()
        self.create_timer(5.0, self._report)

    # ---------------------------------------------------------------- callbacks

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # odom frame -> warehouse frame, exactly as the sanctioned recorder does
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        x = self.args.odom_origin_x_m + self.cos * ox - self.sin * oy
        y = self.args.odom_origin_y_m + self.sin * ox + self.cos * oy
        cov = msg.pose.covariance
        cxx, cxy, cyy = cov[0], cov[1], cov[7]
        # rotate the planar block: C_world = R C R^T
        rxx = self.cos * cxx - self.sin * cxy
        rxy = self.cos * cxy - self.sin * cyy
        ryx = self.sin * cxx + self.cos * cxy
        ryy = self.sin * cxy + self.cos * cyy
        wxx = rxx * self.cos - rxy * self.sin
        wxy = rxx * self.sin + rxy * self.cos
        wyy = ryx * self.sin + ryy * self.cos
        self._odom.writerow([f"{stamp:.6f}", f"{x:.6f}", f"{y:.6f}",
                             f"{wxx:.9f}", f"{wxy:.9f}", f"{wyy:.9f}"])
        self.counts["odom"] += 1

    def _on_truth(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.child_frame_id != self.args.child_frame_id:
                continue
            stamp = (transform.header.stamp.sec
                     + transform.header.stamp.nanosec * 1e-9)
            # The ground-truth bridge publishes transforms with a zero header
            # stamp, so every truth row landed at t=0 and nothing could be joined
            # to it. `record_evaluation_truth.py` substitutes the node clock for
            # exactly this reason; do the same, or the capture has no truth at all.
            if stamp <= 0.0:
                stamp = self.get_clock().now().nanoseconds * 1.0e-9
            t = transform.transform.translation
            r = transform.transform.rotation
            self._truth.writerow([f"{stamp:.6f}", f"{t.x:.6f}", f"{t.y:.6f}",
                                  f"{_yaw_from_quaternion(r.x, r.y, r.z, r.w):.6f}"])
            self.counts["truth"] += 1

    def _on_observation(self, msg: String, camera: str) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        stamp = payload.get("timestamp_s")
        pixel = payload.get("pixel_uv")
        valid = bool(payload.get("detection_valid", False))
        if stamp is None:
            return
        if valid and pixel and len(pixel) == 2 and all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in pixel):
            self._perception[camera].writerow(
                [f"{float(stamp):.6f}", 1, f"{float(pixel[0]):.4f}",
                 f"{float(pixel[1]):.4f}"])
            self.counts[camera] += 1
        else:
            self._perception[camera].writerow([f"{float(stamp):.6f}", 0, "", ""])

    # ------------------------------------------------------------------ housekeeping

    def _report(self) -> None:
        elapsed = time.time() - self._started
        detections = ", ".join(f"{c.replace('camera_', '')}={self.counts[c]}"
                               for c in self.cameras)
        self.get_logger().info(
            f"[{elapsed:6.1f}s] odom={self.counts['odom']} truth={self.counts['truth']} "
            f"detections {detections}")
        for handle in (self._odom_handle, self._truth_handle,
                       *self._perception_handles.values()):
            handle.flush()

    def close(self) -> None:
        for handle in (self._odom_handle, self._truth_handle,
                       *self._perception_handles.values()):
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=240.0)
    parser.add_argument("--odom-topic", default="/odom_noisy")
    parser.add_argument("--truth-topic", default="/ground_truth_tf")
    parser.add_argument("--child-frame-id", default="turtlebot3")
    parser.add_argument("--observation-prefix",
                        default="/perception/camera_observation")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS),
                        help="which cameras this world actually has")
    parser.add_argument("--odom-origin-x-m", type=float, required=True)
    parser.add_argument("--odom-origin-y-m", type=float, required=True)
    parser.add_argument("--odom-origin-yaw-rad", type=float, required=True)
    args = parser.parse_args()

    rclpy.init()
    node = DemonstrationRecorder(args)
    deadline = time.time() + args.duration_s
    try:
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node._report()
        node.close()
        node.destroy_node()
        rclpy.shutdown()

    total = sum(node.counts[c] for c in node.cameras)
    print(f"detections written: {total}; odometry rows: {node.counts['odom']}; "
          f"truth rows: {node.counts['truth']}")
    (args.out_dir / "CAPTURE_ROLE.md").write_text(
        "# demonstration capture\n\n"
        "Recorded by `experiments/filter_notebook/record_demonstration_capture.py`.\n"
        "**Not campaign evidence.** No ledger row, no frozen-config hashes, no\n"
        "projection-calibration provenance. It exists so the notebook can show the\n"
        "perception stage on frames captured beside the detections. Use\n"
        "`record_operational_logs.py` for anything that must be citable.\n",
        encoding="utf-8")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
