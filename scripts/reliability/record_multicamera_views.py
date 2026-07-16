#!/usr/bin/env python3
"""Record synchronized external-camera image topics to frames and MP4 clips.

Example:
  python3 scripts/reliability/record_multicamera_views.py \
    --out-dir logs/warehouse_full_4cam/videos/live_camera_views \
    --camera camera_A=/external_camera/image_raw \
    --camera camera_B=/external_camera_b/image_raw \
    --camera camera_C=/external_camera_c/image_raw \
    --camera camera_D=/external_camera_d/image_raw \
    --duration-s 30 \
    --every 2 \
    --write-mp4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Any

import cv2  # type: ignore
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class MultiCameraRecorder(Node):
    def __init__(
        self,
        cameras: dict[str, str],
        out_dir: Path,
        *,
        every: int,
    ) -> None:
        super().__init__("multicamera_view_recorder")
        self.bridge = CvBridge()
        self.out_dir = out_dir
        self.every = max(1, int(every))
        self.counts = {camera_id: 0 for camera_id in cameras}
        self.saved = {camera_id: 0 for camera_id in cameras}
        self.index_handles: dict[str, Any] = {}
        self.index_writers: dict[str, csv.writer] = {}
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        for camera_id, topic in cameras.items():
            camera_dir = out_dir / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            handle = (camera_dir / "index.csv").open("w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(["stamp", "file"])
            self.index_handles[camera_id] = handle
            self.index_writers[camera_id] = writer
            self.create_subscription(Image, topic, self._callback_for(camera_id, camera_dir), qos)
            self.get_logger().info(f"recording {camera_id}: {topic} -> {camera_dir}")

    def _callback_for(self, camera_id: str, camera_dir: Path):
        def _cb(msg: Image) -> None:
            self.counts[camera_id] += 1
            if self.counts[camera_id] % self.every:
                return
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"{camera_id}: cv_bridge failed: {exc}")
                return
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
            name = f"frame_{stamp:010.3f}.png"
            cv2.imwrite(str(camera_dir / name), frame)
            self.index_writers[camera_id].writerow([f"{stamp:.3f}", name])
            self.index_handles[camera_id].flush()
            self.saved[camera_id] += 1

        return _cb

    def close(self) -> None:
        for camera_id, handle in self.index_handles.items():
            self.get_logger().info(f"{camera_id}: recorded {self.saved[camera_id]} frames")
            handle.close()


def parse_camera_arg(values: list[str]) -> dict[str, str]:
    cameras: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--camera must be NAME=/topic, got {value!r}")
        name, topic = value.split("=", 1)
        name = name.strip()
        topic = topic.strip()
        if not name or not topic.startswith("/"):
            raise SystemExit(f"--camera must be NAME=/topic, got {value!r}")
        cameras[name] = topic
    if not cameras:
        raise SystemExit("at least one --camera is required")
    return cameras


def write_mp4s(out_dir: Path, *, fps: float) -> None:
    for camera_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
        frames = sorted(camera_dir.glob("frame_*.png"))
        if not frames:
            continue
        first = cv2.imread(str(frames[0]))
        if first is None:
            continue
        height, width = first.shape[:2]
        video_path = out_dir / f"{camera_dir.name}_view.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        writer.release()
        print(f"wrote {video_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--write-mp4", action="store_true")
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    cameras = parse_camera_arg(args.camera)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = MultiCameraRecorder(
        cameras,
        args.out_dir,
        every=int(args.every),
    )
    try:
        if float(args.duration_s) > 0.0:
            deadline = time.monotonic() + float(args.duration_s)
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.10)
            node.get_logger().info("duration elapsed; stopping recorder")
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if args.write_mp4:
        write_mp4s(args.out_dir, fps=float(args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
