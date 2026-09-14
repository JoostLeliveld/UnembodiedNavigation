#!/usr/bin/env python3
"""Record a hash identity for every commissioning RGB frame without using truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal


CAMERA_TOPICS = {
    "camera_A": "/external_camera/image_raw",
    "camera_B": "/external_camera_b/image_raw",
    "camera_C": "/external_camera_c/image_raw",
    "camera_D": "/external_camera_d/image_raw",
    "camera_E": "/external_camera_e/image_raw",
}


def image_contract_sha256(message) -> str:
    digest = hashlib.sha256()
    fields = (
        int(message.header.stamp.sec),
        int(message.header.stamp.nanosec),
        str(message.header.frame_id),
        int(message.height),
        int(message.width),
        str(message.encoding),
        int(message.is_bigendian),
        int(message.step),
    )
    digest.update(json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    digest.update(b"\0")
    digest.update(bytes(message.data))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-id", required=True)
    parser.add_argument("--index-jsonl", type=Path, required=True)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("commissioning_raw_frame_identity_recorder")
            args.index_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._output_handle = args.index_jsonl.open("x", encoding="utf-8")
            self.counts = {camera_id: 0 for camera_id in CAMERA_TOPICS}
            self._subscriptions = [
                self.create_subscription(
                    Image,
                    topic,
                    lambda message, camera_id=camera_id, topic=topic: self._record(
                        camera_id, topic, message
                    ),
                    20,
                )
                for camera_id, topic in CAMERA_TOPICS.items()
            ]

        def _record(self, camera_id: str, topic: str, message: Image) -> None:
            stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
            row = {
                "schema": "commissioning_frame_identity.v1",
                "drive_id": args.drive_id,
                "camera_id": camera_id,
                "topic": topic,
                "capture_stamp_ns": stamp_ns,
                "source_frame_id": str(message.header.frame_id),
                "height": int(message.height),
                "width": int(message.width),
                "encoding": str(message.encoding),
                "step": int(message.step),
                "image_contract_sha256": image_contract_sha256(message),
            }
            self._output_handle.write(json.dumps(row, sort_keys=True) + "\n")
            self._output_handle.flush()
            self.counts[camera_id] += 1

        def close(self) -> None:
            if not self._output_handle.closed:
                self._output_handle.flush()
                self._output_handle.close()

    rclpy.init()
    node = Recorder()
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
