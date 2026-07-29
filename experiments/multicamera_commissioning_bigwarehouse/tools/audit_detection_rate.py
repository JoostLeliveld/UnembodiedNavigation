#!/usr/bin/env python3
"""M0.1a config audit: per-camera YOLO detection rate over a teleport grid.

Measures whether the CURRENT detector is out-of-distribution on the four
warehouse_full_4cam viewpoints, at whatever imgsz the running stack was launched
with. The default mode is a transport/configuration smoke audit only: it reads
the detector observations while teleporting the robot over a coarse drivable
grid. Its all-grid rate must *not* be called recall because a robot can be out
of a camera's field of view.

Pass ``--semantic-evaluation`` with the four semantic-label bridges enabled to
compute an evaluation-only, opportunity-conditioned recall and false-positive
rate. Semantic labels are never sent to the detector, planner, or operational
logs. This keeps the simulator oracle on the evaluation side of the firewall.

Run it against a live stack:
  ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
      world:=warehouse_full_4cam.world.sdf \
      yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt \
      yolo_runtime_backend:=native yolo_imgsz:=640 headless:=true
then, in a second shell (env sourced):
  python3 experiments/multicamera_commissioning_bigwarehouse/tools/audit_detection_rate.py \
      --imgsz 640 --nx 8 --ny 7 --out logs/studies/fourcam_detector_audit/imgsz640

For the final detector evaluation, enable the four segmentation bridges in the
launch and append ``--semantic-evaluation``. The resulting semantic labels are
evaluation-only and must not be recorded by the operational logger.

  ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
      world:=warehouse_full_4cam.world.sdf \
      bridge_segmentation:=true bridge_segmentation_b:=true \
      bridge_segmentation_c:=true bridge_segmentation_d:=true

Default mode uses no ground truth (positions are commanded teleport poses; det
comes from YOLO). ``--semantic-evaluation`` is explicitly evaluation-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import String
from sensor_msgs.msg import Image

CAMERAS = ["camera_A", "camera_B", "camera_C", "camera_D"]
LABEL_TOPICS = {
    "camera_A": "/external_camera/segmentation/labels_map",
    "camera_B": "/external_camera_b/segmentation/labels_map",
    "camera_C": "/external_camera_c/segmentation/labels_map",
    "camera_D": "/external_camera_d/segmentation/labels_map",
}
# approximate wall-mounted camera positions (m) for range binning only
CAM_XYZ = {"camera_A": (-6.0, -10.0, 6.1), "camera_B": (-6.0, 10.0, 6.1),
           "camera_C": (6.0, -10.0, 6.1), "camera_D": (6.0, 10.0, 6.1)}
TRAV = dict(xmin=-11.20, xmax=11.50, ymin=-8.60, ymax=8.60)
OBST = [  # xmin, xmax, ymin, ymax  (from world_profiles known_2d_regions)
    (-11.91, -11.36, -6.50, 6.90), (-0.25, 0.25, -1.15, -0.65),
    (-6.125, -5.225, -4.90, -4.30), (9.75, 10.65, 0.45, 1.15)]


def in_obstacle(x, y, clear=0.45):
    return any(xm - clear <= x <= xM + clear and ym - clear <= y <= yM + clear
               for xm, xM, ym, yM in OBST)


def build_grid(nx, ny, margin=0.7):
    xs = np.linspace(TRAV["xmin"] + margin, TRAV["xmax"] - margin, nx)
    ys = np.linspace(TRAV["ymin"] + margin, TRAV["ymax"] - margin, ny)
    return [(float(x), float(y)) for x in xs for y in ys if not in_obstacle(x, y)]


def nearest_cam_range(x, y):
    return min(math.dist((x, y, 0.05), c) for c in CAM_XYZ.values())


def as_detected(payload):
    for k in ("detection_valid", "detected_after_threshold", "detected"):
        if k in payload:
            return bool(payload[k])
    return False


def as_score(payload):
    for k in ("detector_score_raw", "detector_score", "detection_confidence",
              "selected_score", "raw_best_score", "confidence"):
        if k in payload and payload[k] is not None:
            try:
                return float(payload[k])
            except (TypeError, ValueError):
                pass
    return 0.0


def stamp_ns(message: Image) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def semantic_labels_from_image(message: Image) -> np.ndarray:
    """Decode Gazebo's semantic label-map transport without cv_bridge.

    Gazebo 6 transports ``labels_map`` as ``rgb8``/``bgr8`` even though it is a
    semantic-ID image, not a visualisation: its final byte is the label ID.
    This is deliberately identical to the established dataset-capture decoder
    in ``scripts/perception/capture_yolo_dataset.py``.  Do not generalise this
    rule to arbitrary RGB images.
    """

    encoding = str(message.encoding or "").strip().lower()
    dtype_by_encoding = {
        "mono8": np.dtype(np.uint8),
        "8uc1": np.dtype(np.uint8),
        "mono16": np.dtype(np.uint16),
        "16uc1": np.dtype(np.uint16),
        "16sc1": np.dtype(np.int16),
        "32uc1": np.dtype(np.uint32),
        "32sc1": np.dtype(np.int32),
    }
    height, width, step = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid semantic image geometry {width}x{height}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("Semantic image data are shorter than its declared geometry")
    rows = raw[: height * step].reshape(height, step)

    if encoding in {"rgb8", "bgr8", "8uc3"}:
        if step < width * 3:
            raise ValueError(f"Invalid RGB semantic image step {step} for width {width}")
        return np.ascontiguousarray(rows[:, : width * 3].reshape(height, width, 3)[..., 2].astype(np.int64))

    dtype = dtype_by_encoding.get(encoding)
    if dtype is None:
        raise ValueError(
            f"Unsupported semantic label encoding {message.encoding!r}; expected scalar integer image"
        )
    if int(message.is_bigendian):
        dtype = dtype.newbyteorder(">")
    if height <= 0 or width <= 0 or step < width * dtype.itemsize:
        raise ValueError(f"Invalid semantic image geometry {width}x{height}, step={step}")
    labels = rows[:, : width * dtype.itemsize].view(dtype).reshape(height, width)
    return np.ascontiguousarray(labels.astype(np.int64, copy=False))


def opportunity_summary(rows):
    """Summarize only synchronized semantic-oracle rows, per camera/range."""

    summary = {}
    for camera in CAMERAS:
        camera_rows = [row for row in rows if row["camera"] == camera]
        paired = [row for row in camera_rows if row["semantic_pair"]]
        opportunities = [row for row in paired if row["opportunity"]]
        negatives = [row for row in paired if not row["opportunity"]]
        summary[camera] = {
            "all_grid_detection_rate": round(float(np.mean([row["detected"] for row in camera_rows])), 3)
            if camera_rows else None,
            "semantic_pairs": len(paired),
            "unpaired_or_missing_semantic": len(camera_rows) - len(paired),
            "opportunities": len(opportunities),
            "opportunity_conditioned_recall": round(
                float(np.mean([row["detected"] for row in opportunities])), 3
            ) if opportunities else None,
            "no_opportunity_frames": len(negatives),
            "false_positive_rate_no_opportunity": round(
                float(np.mean([row["detected"] for row in negatives])), 3
            ) if negatives else None,
        }
    return summary


class Probe(Node):
    def __init__(self, world, *, semantic_evaluation: bool, robot_label: int,
                 semantic_sync_slop_s: float):
        super().__init__("detection_rate_audit")
        self.count = {c: 0 for c in CAMERAS}
        self.latest = {c: None for c in CAMERAS}
        self.semantic_evaluation = bool(semantic_evaluation)
        self.robot_label = int(robot_label)
        self.semantic_sync_slop_s = float(max(semantic_sync_slop_s, 0.0))
        self.latest_labels = {c: None for c in CAMERAS}
        self.label_count = {c: 0 for c in CAMERAS}
        # Keep enough history to match an inference-delayed detector result to
        # the 1 Hz semantic render captured at the same simulation timestamp.
        self.observation_buffer = {c: deque(maxlen=256) for c in CAMERAS}
        self.label_buffer = {c: deque(maxlen=32) for c in CAMERAS}
        for c in CAMERAS:
            self.create_subscription(String, f"/perception/camera_observation/{c}",
                                     self._cb(c), 20)
            if self.semantic_evaluation:
                self.create_subscription(Image, LABEL_TOPICS[c], self._label_cb(c), 20)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.cli = self.create_client(SetEntityPose, f"/world/{world}/set_pose")

    def _cb(self, cam):
        def handler(msg):
            try:
                d = json.loads(msg.data)
            except (ValueError, TypeError):
                return
            self.count[cam] += 1
            stamp = d.get("timestamp_s", d.get("stamp", math.nan))
            try:
                stamp = float(stamp)
            except (TypeError, ValueError):
                stamp = math.nan
            self.latest[cam] = (self.count[cam], as_detected(d), as_score(d), stamp)
            self.observation_buffer[cam].append(self.latest[cam])
        return handler

    def _label_cb(self, cam):
        def handler(msg):
            try:
                self.label_count[cam] += 1
                self.latest_labels[cam] = (stamp_ns(msg) * 1.0e-9, semantic_labels_from_image(msg))
                self.label_buffer[cam].append((self.label_count[cam], *self.latest_labels[cam]))
            except (TypeError, ValueError) as exc:
                self.get_logger().warn(f"{cam}: ignored malformed semantic image: {exc}")
        return handler

    def wait_ready(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout and not self.cli.service_is_ready():
            rclpy.spin_once(self, timeout_sec=0.2)
        got = 0
        t0 = time.time()
        while time.time() - t0 < timeout and got < len(CAMERAS):
            rclpy.spin_once(self, timeout_sec=0.2)
            got = sum(
                1 for c in CAMERAS
                if self.count[c] > 0 and (not self.semantic_evaluation or self.latest_labels[c] is not None)
            )
        return self.cli.service_is_ready() and got == len(CAMERAS)

    def teleport(self, x, y, yaw=1.5708):
        req = SetEntityPose.Request()
        req.entity = Entity(name="turtlebot3", type=Entity.MODEL)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = x, y, 0.05
        req.pose.orientation.z = math.sin(yaw / 2.0)
        req.pose.orientation.w = math.cos(yaw / 2.0)
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)

    def _fresh_synced_pair(self, camera, *, after_observation, after_label):
        """Return closest fresh detector/semantic pair, or ``None``.

        Semantic rendering is intentionally 1 Hz while detector results can
        arrive later than their capture stamp. Pairing only the latest messages
        biases the audit toward stale or unrelated poses.
        """
        best = None
        for obs_count, detected, score, obs_stamp in self.observation_buffer[camera]:
            if obs_count <= after_observation or not math.isfinite(obs_stamp):
                continue
            for label_count, label_stamp, labels in self.label_buffer[camera]:
                if label_count <= after_label:
                    continue
                delta = abs(obs_stamp - label_stamp)
                if delta <= self.semantic_sync_slop_s and (best is None or delta < best[0]):
                    best = (delta, detected, score, obs_stamp, label_stamp, labels)
        if best is None:
            return None
        _, detected, score, obs_stamp, label_stamp, labels = best
        return detected, score, obs_stamp, label_stamp, int(np.count_nonzero(labels == self.robot_label))

    def sample(self, settle_frames=2, timeout=8.0):
        base = {c: self.count[c] for c in CAMERAS}
        base_labels = {c: self.label_count[c] for c in CAMERAS}
        t0 = time.time()
        synced = {c: None for c in CAMERAS}
        while time.time() - t0 < timeout:
            self.cmd.publish(Twist())  # hold still
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.semantic_evaluation:
                for camera in CAMERAS:
                    if synced[camera] is None:
                        synced[camera] = self._fresh_synced_pair(
                            camera,
                            after_observation=base[camera],
                            after_label=base_labels[camera],
                        )
                if all(synced.values()):
                    break
            elif all(self.count[c] - base[c] >= settle_frames for c in CAMERAS):
                break
        out = {}
        for camera in CAMERAS:
            if self.semantic_evaluation and synced[camera] is not None:
                out[camera] = synced[camera]
                continue
            observation = self.latest[camera] if self.count[camera] > base[camera] else None
            if observation is None:
                out[camera] = None
                continue
            _, detected, score, observation_stamp_s = observation
            label = self.latest_labels[camera]
            if self.semantic_evaluation and label is not None and math.isfinite(observation_stamp_s):
                label_stamp_s, labels = label
                out[camera] = (detected, score, observation_stamp_s, label_stamp_s,
                               int(np.count_nonzero(labels == self.robot_label)))
            else:
                out[camera] = (detected, score, observation_stamp_s, math.nan, -1)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="warehouse_full_4cam")
    ap.add_argument("--imgsz", type=int, default=0, help="label only (stack sets the real imgsz)")
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=7)
    ap.add_argument("--settle-frames", type=int, default=2)
    ap.add_argument("--sample-timeout", type=float, default=8.0)
    ap.add_argument("--semantic-evaluation", action="store_true",
                    help="Use synchronized semantic labels for evaluation-only opportunity metrics.")
    ap.add_argument("--robot-label", type=int, default=23)
    ap.add_argument("--semantic-sync-slop-ms", type=float, default=60.0)
    ap.add_argument("--out", default="logs/studies/fourcam_detector_audit/run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Probe(
        args.world,
        semantic_evaluation=args.semantic_evaluation,
        robot_label=args.robot_label,
        semantic_sync_slop_s=float(args.semantic_sync_slop_ms) * 1.0e-3,
    )
    try:
        if not node.wait_ready():
            required = "observations and semantic labels" if args.semantic_evaluation else "observations"
            print(f"ERROR: stack not ready (set_pose service or all 4 camera {required} missing).")
            return 2
        grid = build_grid(args.nx, args.ny)
        print(f"probing {len(grid)} drivable poses on {args.world} (imgsz label {args.imgsz})")
        rows = []
        for i, (x, y) in enumerate(grid):
            node.teleport(x, y)
            res = node.sample(settle_frames=args.settle_frames, timeout=args.sample_timeout)
            for c in CAMERAS:
                det, sc, obs_stamp, label_stamp, semantic_pixels = (
                    res[c] if res[c] else (False, 0.0, math.nan, math.nan, -1)
                )
                semantic_pair = bool(
                    args.semantic_evaluation
                    and semantic_pixels >= 0
                    and math.isfinite(obs_stamp)
                    and math.isfinite(label_stamp)
                    and abs(obs_stamp - label_stamp) <= args.semantic_sync_slop_ms * 1.0e-3
                )
                rows.append(dict(i=i, x=round(x, 3), y=round(y, 3), camera=c,
                                 detected=int(det), score=round(sc, 4),
                                 camera_range_m=round(math.dist((x, y, 0.05), CAM_XYZ[c]), 2),
                                 observation_stamp_s=obs_stamp, label_stamp_s=label_stamp,
                                 semantic_robot_pixels=semantic_pixels,
                                 semantic_pair=int(semantic_pair),
                                 opportunity=int(semantic_pair and semantic_pixels > 0),
                                 stale=int(res[c] is None)))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(grid)} poses")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    with open(out / "detections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # The all-grid rate is a smoke metric only. A per-camera semantic visibility
    # oracle is required before interpreting it as detector recall.
    summary = {"imgsz": args.imgsz, "world": args.world, "n_poses": len(rows) // len(CAMERAS),
               "semantic_evaluation": bool(args.semantic_evaluation), "per_camera": {}, "by_range": {}}
    print(f"\n=== detection rate (imgsz {args.imgsz}), {summary['n_poses']} poses ===")
    for c in CAMERAS:
        cr = [r for r in rows if r["camera"] == c]
        rate = float(np.mean([r["detected"] for r in cr]))
        msc = float(np.mean([r["score"] for r in cr if r["detected"]])) if any(r["detected"] for r in cr) else 0.0
        summary["per_camera"][c] = dict(mean_score_when_detected=round(msc, 3), stale=sum(r["stale"] for r in cr))
        if args.semantic_evaluation:
            summary["per_camera"][c].update(opportunity_summary(rows)[c])
            report = summary["per_camera"][c]
            print(f"  {c}: recall={report['opportunity_conditioned_recall']} "
                  f"fp_no_opp={report['false_positive_rate_no_opportunity']} "
                  f"opportunities={report['opportunities']} pairs={report['semantic_pairs']}")
        else:
            summary["per_camera"][c]["all_grid_detection_rate_not_recall"] = round(rate, 3)
            print(f"  {c}: all_grid_rate={rate:.3f} (not recall)  mean_score={msc:.3f}  stale={sum(r['stale'] for r in cr)}")
    for lo, hi in [(0, 12), (12, 16), (16, 99)]:
        rr = [r for r in rows if lo <= r["camera_range_m"] < hi]
        if rr:
            rate = float(np.mean([r["detected"] for r in rr]))
            item = dict(all_grid_detection_rate_not_recall=round(rate, 3), n=len(rr))
            if args.semantic_evaluation:
                opp = [r for r in rr if r["opportunity"]]
                item["opportunity_conditioned_recall"] = round(
                    float(np.mean([r["detected"] for r in opp])), 3
                ) if opp else None
                item["opportunities"] = len(opp)
            summary["by_range"][f"{lo}-{hi}m"] = item
            print(f"  range {lo}-{hi}m per-camera: {item}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/detections.csv and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
