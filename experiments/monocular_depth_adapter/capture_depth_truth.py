#!/usr/bin/env python3
"""Capture matched RGB + real Gazebo depth from the fixed external camera.

The depth is EVALUATION-ONLY. It never reaches the adapter, and it is stored
outside the frozen set the adapter reads. Its only job is to answer "were the
models right", which nothing else in this study can answer.

Why re-capture the RGB instead of reusing the frozen frames: the depth has to
correspond to the same scene state, pixel for pixel. Rendering both in the same
teleport is the only way to guarantee that, and it lets the run assert the two
match rather than assume it.

Run this AFTER launching the simulator, in two terminals:

    # 1. simulator (check nothing else is running first!)
    pgrep -a "ros2 launch|ign gazebo|gz sim|run_visibility_campaign"
    ros2 launch sim bringup_sim.launch.py headless:=true use_lidar:=false bridge_scan:=false

    # 2. the depth topic is not in the shared launch, so bridge it alongside
    ros2 run ros_gz_bridge parameter_bridge \
        /external_camera/depth@sensor_msgs/msg/Image[gz.msgs.Image

    # 3. this script
    python3 experiments/monocular_depth_adapter/capture_depth_truth.py

The camera's depth sensor is declared ``always_on=0`` in
``src/sim/models/external_camera/model.sdf``, so Gazebo only renders it while
something subscribes. That is what the bridge in step 2 is for; without it the
script waits and times out rather than silently capturing nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import frozen_set as fs  # noqa: E402

OUT_DIR = fs.REPO / "logs/studies/monocular_depth_adapter/depth_truth_warehouse_aws"
DEFAULT_WORLD = "warehouse_aws"
DEFAULT_ROBOT = "turtlebot3"


def _poses_for_frozen_frames(role: str = "method_development") -> list[dict]:
    """The robot poses behind the frozen development frames, in manifest order."""
    frames = fs.load_frames(role=role)
    with open(fs.AWS_CAPTURE / "samples.csv", newline="", encoding="utf-8") as handle:
        rows = {r["sample_id"]: r for r in csv.DictReader(handle)}
    out = []
    for frame in frames:
        row = rows[frame.source_sample_id]
        out.append({
            "frame_id": frame.frame_id,
            "sample_id": frame.source_sample_id,
            "x": float(row["x"]), "y": float(row["y"]), "theta": float(row["theta"]),
            "reference_image": frame.image_path,
            "reference_sha256": frame.sha256,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default=DEFAULT_WORLD)
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    parser.add_argument("--rgb-topic", default="/external_camera/image_raw")
    parser.add_argument("--depth-topic", default="/external_camera/depth")
    # 0.05 is the campaign's spawn height. Measured 2026-08-11: at 0.01 the robot
    # sinks on the first teleport and never re-appears in either camera, which
    # looks exactly like a working capture right up until you diff two frames.
    parser.add_argument("--robot-z", type=float, default=0.05)
    parser.add_argument("--settle-s", type=float, default=2.0,
                        help="seconds after a teleport before a frame is accepted")
    parser.add_argument("--min-new-frames", type=int, default=4,
                        help="fresh renders required after settling, so no stale frame is kept")
    parser.add_argument("--stale-retries", type=int, default=4,
                        help="re-waits allowed when a frame comes back identical to the last one")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    import rclpy
    from geometry_msgs.msg import Pose
    from rclpy.node import Node
    from ros_gz_interfaces.msg import Entity
    from ros_gz_interfaces.srv import SetEntityPose
    from sensor_msgs.msg import Image

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = _poses_for_frozen_frames()
    print(f"capturing {len(poses)} poses -> {out_dir}")

    class Capturer(Node):
        def __init__(self):
            super().__init__("monodepth_capture_depth_truth")
            self.rgb = None
            self.depth = None
            self.rgb_n = 0
            self.depth_n = 0
            self.create_subscription(Image, args.rgb_topic, self._on_rgb, 10)
            self.create_subscription(Image, args.depth_topic, self._on_depth, 10)
            self.client = self.create_client(SetEntityPose, f"/world/{args.world}/set_pose")

        def _on_rgb(self, msg: Image) -> None:
            if msg.encoding not in ("rgb8", "bgr8"):
                self.get_logger().warn(f"unexpected rgb encoding {msg.encoding}")
                return
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            self.rgb = arr[:, :, ::-1].copy() if msg.encoding == "bgr8" else arr.copy()
            self.rgb_n += 1

        def _on_depth(self, msg: Image) -> None:
            if msg.encoding not in ("32FC1", "TYPE_32FC1"):
                self.get_logger().warn(f"unexpected depth encoding {msg.encoding}")
                return
            self.depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
            self.depth_n += 1

        def wait_ready(self, timeout_s: float) -> None:
            end = time.monotonic() + timeout_s
            ready = False
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
                if not ready and self.client.wait_for_service(timeout_sec=0.05):
                    ready = True
                if ready and self.rgb is not None and self.depth is not None:
                    return
            raise RuntimeError(
                f"timed out waiting for set_pose, {args.rgb_topic} and {args.depth_topic}. "
                "Is the simulator up, and is the depth topic bridged? "
                "(the depth sensor is always_on=0, so it only renders while subscribed)"
            )

        def teleport(self, x: float, y: float, yaw: float) -> None:
            req = SetEntityPose.Request()
            req.entity = Entity(name=args.robot)
            req.entity.type = Entity.MODEL
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = x, y, args.robot_z
            pose.orientation.z = math.sin(0.5 * yaw)
            pose.orientation.w = math.cos(0.5 * yaw)
            req.pose = pose
            future = self.client.call_async(req)
            start = time.monotonic()
            while rclpy.ok() and not future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                if time.monotonic() - start > 10.0:
                    raise RuntimeError("set_pose did not respond")

        def fresh_pair(self, settle_s: float, min_new: int, timeout_s: float):
            """Wait out the settle time, then require N new renders of BOTH streams."""
            time.sleep(settle_s)
            rgb0, depth0 = self.rgb_n, self.depth_n
            end = time.monotonic() + timeout_s
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.rgb_n - rgb0 >= min_new and self.depth_n - depth0 >= min_new:
                    return self.rgb.copy(), self.depth.copy()  # type: ignore[union-attr]
            raise RuntimeError(
                f"only {self.rgb_n - rgb0} rgb / {self.depth_n - depth0} depth frames "
                f"arrived in {timeout_s}s; needed {min_new} of each"
            )

    rclpy.init()
    node = Capturer()
    records = []
    stale_pairs = []
    previous_rgb = None
    try:
        node.wait_ready(args.timeout_s)
        print(f"simulator ready (rgb {node.rgb.shape}, depth {node.depth.shape})")  # type: ignore[union-attr]
        for i, pose in enumerate(poses):
            node.teleport(pose["x"], pose["y"], pose["theta"])
            rgb, depth = node.fresh_pair(args.settle_s, args.min_new_frames, args.timeout_s)

            # "N new messages arrived" is not the same as "the renderer has seen
            # the new pose" — at ~2.5 Hz the camera occasionally hands back the
            # previous render. Waiting again is enough; accepting it is not.
            for attempt in range(args.stale_retries):
                if previous_rgb is None or not np.array_equal(previous_rgb, rgb):
                    break
                print(f"      frame identical to the previous one, re-waiting "
                      f"({attempt + 1}/{args.stale_retries})", flush=True)
                rgb, depth = node.fresh_pair(args.settle_s, args.min_new_frames, args.timeout_s)

            from PIL import Image as PILImage

            rgb_path = out_dir / f"{pose['frame_id']}_rgb.png"
            depth_path = out_dir / f"{pose['frame_id']}_depth.npy"
            PILImage.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth)

            finite = depth[np.isfinite(depth)]
            record = {
                **pose,
                "rgb_file": rgb_path.name,
                "depth_file": depth_path.name,
                "height": int(depth.shape[0]), "width": int(depth.shape[1]),
                "depth_finite_fraction": float(np.isfinite(depth).mean()),
                "depth_min_m": float(finite.min()) if finite.size else float("nan"),
                "depth_max_m": float(finite.max()) if finite.size else float("nan"),
            }
            # A teleport that silently failed still yields a well-formed capture
            # with plausible depth statistics. The only thing that gives it away
            # is that consecutive frames are byte-identical, so check for that
            # here rather than discovering it during analysis.
            if previous_rgb is not None and np.array_equal(previous_rgb, rgb):
                stale_pairs.append(pose["frame_id"])
            previous_rgb = rgb
            record["rgb_differs_from_previous"] = pose["frame_id"] not in stale_pairs

            records.append(record)
            print(f"  [{i+1}/{len(poses)}] {pose['frame_id']} "
                  f"depth {record['depth_min_m']:.2f}-{record['depth_max_m']:.2f} m, "
                  f"{100*record['depth_finite_fraction']:.1f}% finite"
                  f"{'  *** IDENTICAL TO PREVIOUS FRAME ***' if not record['rgb_differs_from_previous'] else ''}",
                  flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    manifest = {
        "schema_version": 1,
        "world": args.world,
        "purpose": "EVALUATION-ONLY Gazebo depth for the monocular depth benchmark",
        "contract": (
            "This depth is ground truth. It is never an adapter input, never used to "
            "anchor a prediction, and never read by anything under monodepth/."
        ),
        "rgb_topic": args.rgb_topic,
        "depth_topic": args.depth_topic,
        "sensor": "external_camera/depth_camera (R_FLOAT32, same pose+FOV as the RGB camera)",
        "settle_s": args.settle_s,
        "min_new_frames": args.min_new_frames,
        "robot_z": args.robot_z,
        "stale_frames": stale_pairs,
        "frames": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    print(f"\nwrote {(out_dir / 'manifest.json').relative_to(fs.REPO)} ({len(records)} frames)")
    if stale_pairs:
        # Two frames really can be identical: the robot is the only thing that
        # moves, and behind the racks it is invisible from this camera, so
        # consecutive occluded poses render the same picture. That is a valid
        # capture. A teleport that silently failed looks the same locally, so the
        # distinction is made by how MANY frames repeat: a sunk robot repeats
        # essentially all of them (measured: 11 of 12 at robot_z=0.01).
        share = len(stale_pairs) / max(len(records), 1)
        print(f"\n{len(stale_pairs)} frame(s) identical to the one before: {stale_pairs}")
        if share >= 0.5:
            print(f"CAPTURE IS NOT USABLE: {100*share:.0f}% of frames repeat, so the robot is "
                  "not being placed. Check --robot-z and that the model name is right.")
            return 1
        print("  Below the half-the-capture threshold; expected when the robot is occluded "
              "at consecutive poses. Verify with the analytic floor check before relying on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
