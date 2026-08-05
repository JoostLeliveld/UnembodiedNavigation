#!/usr/bin/env python3
"""Capture one fresh RGB frame from each canonical A--D camera topic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
PERCEPTION_SRC = REPO / "src/perception"
if str(PERCEPTION_SRC) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_SRC))

from perception.core.ros_image import image_msg_to_bgr8  # noqa: E402


CAMERAS = [
    ("camera_A", "external_camera", "south-west wall"),
    ("camera_B", "external_camera_b", "north-west wall"),
    ("camera_C", "external_camera_c", "south-east wall"),
    ("camera_D", "external_camera_d", "north-east wall"),
]


class SnapshotNode(Node):
    def __init__(self) -> None:
        super().__init__("fourcam_snapshot")
        self.frames: dict[str, np.ndarray] = {}
        self.stamps: dict[str, float] = {}
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        for camera_id, model, _role in CAMERAS:
            self.create_subscription(
                Image,
                f"/{model}/image_raw",
                self._callback(camera_id),
                qos,
            )

    def _callback(self, camera_id: str):
        def callback(msg: Image) -> None:
            if camera_id in self.frames:
                return
            self.frames[camera_id] = image_msg_to_bgr8(msg)
            self.stamps[camera_id] = (
                float(msg.header.stamp.sec)
                + 1.0e-9 * float(msg.header.stamp.nanosec)
            )
            self.get_logger().info(f"captured {camera_id} ({len(self.frames)}/4)")

        return callback


def render_sheet(frames: dict[str, np.ndarray], out_path: Path) -> None:
    panel_w, panel_h, header_h = 640, 360, 42
    sheet = np.full((2 * (panel_h + header_h), 2 * panel_w, 3), 245, np.uint8)
    for index, (camera_id, _model, role) in enumerate(CAMERAS):
        row, col = divmod(index, 2)
        x0, y0 = col * panel_w, row * (panel_h + header_h)
        frame = cv2.resize(
            frames[camera_id], (panel_w, panel_h), interpolation=cv2.INTER_AREA
        )
        sheet[y0 + header_h : y0 + header_h + panel_h, x0 : x0 + panel_w] = frame
        cv2.rectangle(
            sheet,
            (x0, y0),
            (x0 + panel_w - 1, y0 + header_h - 1),
            (31, 42, 55),
            thickness=-1,
        )
        cv2.putText(
            sheet,
            f"{camera_id} | {role}",
            (x0 + 12, y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(out_path), sheet):
        raise RuntimeError(f"could not write {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)

    rclpy.init()
    node = SnapshotNode()
    deadline = time.monotonic() + args.timeout_s
    try:
        while rclpy.ok() and time.monotonic() < deadline and len(node.frames) < 4:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    missing = [camera_id for camera_id, _model, _role in CAMERAS if camera_id not in node.frames]
    if missing:
        raise RuntimeError(f"timed out waiting for: {', '.join(missing)}")

    for camera_id, model, _role in CAMERAS:
        if not cv2.imwrite(str(out_dir / f"{model}_image_raw.png"), node.frames[camera_id]):
            raise RuntimeError(f"could not write frame for {camera_id}")
    sheet = out_dir / "fig_fourcam_views.png"
    render_sheet(node.frames, sheet)
    manifest = {
        "world": args.world,
        "simulator": "Gazebo Sim 6 / Fortress",
        "cameras": [
            {
                "camera_id": camera_id,
                "model_include": model,
                "topic": f"/{model}/image_raw",
                "stamp_s": node.stamps[camera_id],
                "frame_file": f"{model}_image_raw.png",
            }
            for camera_id, model, _role in CAMERAS
        ],
        "sheet": sheet.name,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"captured four fresh Gazebo views: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
