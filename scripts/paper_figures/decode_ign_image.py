#!/usr/bin/env python3
"""Decode one `ign topic --json-output` image capture into a PNG.

The overhead plan camera is not bridged to ROS, so capture_overview_frame.py (which
subscribes through rclpy) cannot see it. Read it off the Gazebo transport instead:

    ros2 launch sim bringup_sim.launch.py world:=warehouse_v2.world.sdf headless:=true
    ign topic -e -t /plan_view_camera/image_raw -n 1 --json-output > plan.json
    python3 scripts/paper_figures/decode_ign_image.py plan.json out.png

Check that nothing else is using the simulator before launching, and stop it after.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib

import numpy as np
from PIL import Image


def decode(payload: dict) -> np.ndarray:
    """The message's RGB pixels, honouring its own row stride."""
    width, height, step = int(payload['width']), int(payload['height']), int(payload['step'])
    raw = base64.b64decode(payload['data'])
    if len(raw) != height * step:
        raise SystemExit(f'expected {height * step} bytes for {width}x{height}, got {len(raw)}')
    channels = step // width
    if channels < 3:
        raise SystemExit(f'{channels} channels per pixel; expected at least 3 (RGB)')
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
    return rows[:, :width * channels].reshape(height, width, channels)[:, :, :3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=pathlib.Path, help='the --json-output capture')
    parser.add_argument('output', type=pathlib.Path, help='PNG to write')
    args = parser.parse_args()

    image = decode(json.loads(args.source.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(f'wrote {args.output} ({image.shape[1]}x{image.shape[0]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
