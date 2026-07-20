#!/usr/bin/env python3
"""Capture a small folder of external-camera images for visual detector testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_SRC = REPO_ROOT / 'src' / 'perception'
if str(PERCEPTION_SRC) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_SRC))

from perception.core.ros_image import image_msg_to_bgr8


class ImageCaptureNode(Node):
    def __init__(self, *, out_dir: Path, max_images: int, stride: int, topic: str):
        super().__init__('capture_external_camera_images')
        self.out_dir = out_dir
        self.max_images = int(max_images)
        self.stride = max(int(stride), 1)
        self.topic = str(topic)
        self.frame_index = 0
        self.saved = 0
        self.saved_paths: list[str] = []
        self.create_subscription(Image, self.topic, self._image_cb, 10)
        self.get_logger().info(
            f'Capturing up to {self.max_images} images from {self.topic} into {self.out_dir}'
        )

    def _image_cb(self, msg: Image) -> None:
        self.frame_index += 1
        if self.saved >= self.max_images:
            return
        if (self.frame_index - 1) % self.stride != 0:
            return

        image = image_msg_to_bgr8(msg)
        path = self.out_dir / f'{self.saved:06d}.jpg'
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f'Failed to write image to {path}')
        self.saved_paths.append(str(path.name))
        self.saved += 1
        self.get_logger().info(f'Saved {path.name} ({self.saved}/{self.max_images})')
        if self.saved >= self.max_images:
            self.get_logger().info('Capture complete, shutting down.')
            self.create_timer(0.1, lambda: rclpy.shutdown())


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture a small folder of /external_camera/image_raw frames.')
    parser.add_argument('--out', required=True, help='Output folder for captured JPG images')
    parser.add_argument('--max-images', type=int, default=60, help='Number of images to save')
    parser.add_argument('--stride', type=int, default=10, help='Save every Nth incoming frame')
    parser.add_argument('--topic', default='/external_camera/image_raw', help='ROS image topic to subscribe to')
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        raise RuntimeError(f'Output folder already exists: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=False)

    rclpy.init()
    node = ImageCaptureNode(
        out_dir=out_dir,
        max_images=int(args.max_images),
        stride=int(args.stride),
        topic=str(args.topic),
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        manifest = {
            'topic': str(args.topic),
            'max_images': int(args.max_images),
            'stride': int(args.stride),
            'saved_images': int(node.saved),
            'images': list(node.saved_paths),
        }
        (out_dir / 'capture_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
