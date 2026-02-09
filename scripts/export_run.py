#!/usr/bin/env python3
"""Export side-by-side figures from a rosbag2 run.

Left pane: raw /external_camera/image_raw (no overlays)
Right pane: BEV view with costmap + plan + state + goal
"""

import argparse
import math
import os
from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


IMAGE_TOPIC = "/external_camera/image_raw"
COSTMAP_TOPIC = "/costmap"
PLAN_TOPIC = "/plan"
STATE_TOPIC = "/state/bev"
GOAL_TOPIC = "/goal_bev"


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _image_to_rgb(msg) -> Optional[np.ndarray]:
    enc = msg.encoding.lower()
    h = msg.height
    w = msg.width
    step = msg.step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if h == 0 or w == 0:
        return None

    if step < w:
        return None

    rows = data.reshape((h, step))

    if enc in ("rgb8", "bgr8"):
        row_bytes = w * 3
        rgb = rows[:, :row_bytes].reshape((h, w, 3))
        if enc == "bgr8":
            rgb = rgb[:, :, ::-1]
        return rgb
    if enc in ("rgba8", "bgra8"):
        row_bytes = w * 4
        rgba = rows[:, :row_bytes].reshape((h, w, 4))
        if enc == "bgra8":
            rgba = rgba[:, :, [2, 1, 0, 3]]
        return rgba[:, :, :3]
    if enc in ("mono8",):
        row_bytes = w
        mono = rows[:, :row_bytes].reshape((h, w, 1))
        return np.repeat(mono, 3, axis=2)

    print(f"[WARN] Unsupported image encoding: {msg.encoding}")
    return None


def _render_bev(costmap_msg, plan_msg, state_msg, goal_msg, height_px: int) -> np.ndarray:
    dpi = 100
    width_px = height_px  # square panel

    fig_w = max(width_px / dpi, 1.0)
    fig_h = max(height_px / dpi, 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_aspect('equal')

    extent = None
    if costmap_msg is not None:
        width = costmap_msg.info.width
        height = costmap_msg.info.height
        res = costmap_msg.info.resolution
        origin_x = costmap_msg.info.origin.position.x
        origin_y = costmap_msg.info.origin.position.y
        extent = [origin_x, origin_x + width * res, origin_y, origin_y + height * res]

        data = np.array(costmap_msg.data, dtype=float).reshape(height, width)
        grid = np.full_like(data, 0.5, dtype=float)
        known = data >= 0
        grid[known] = 1.0 - np.clip(data[known] / 100.0, 0.0, 1.0)
        ax.imshow(grid, origin='lower', extent=extent, cmap='gray', vmin=0.0, vmax=1.0)

    if plan_msg is not None and plan_msg.poses:
        xs = [p.pose.position.x for p in plan_msg.poses]
        ys = [p.pose.position.y for p in plan_msg.poses]
        ax.plot(xs, ys, color='lime', linewidth=2.0, label='plan')

    if state_msg is not None:
        px = state_msg.pose.pose.position.x
        py = state_msg.pose.pose.position.y
        yaw = _yaw_from_quaternion(state_msg.pose.pose.orientation)
        ax.arrow(px, py, 0.3 * math.cos(yaw), 0.3 * math.sin(yaw),
                 head_width=0.15, head_length=0.2, fc='red', ec='red')
        ax.plot(px, py, 'ro', markersize=4)

    if goal_msg is not None:
        gx = goal_msg.pose.position.x
        gy = goal_msg.pose.position.y
        ax.plot(gx, gy, 'bo', markersize=6)

    if extent is None:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("BEV")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    bev = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bev = bev.reshape((int(fig.canvas.get_width_height()[1]), int(fig.canvas.get_width_height()[0]), 3))
    plt.close(fig)
    return bev


def _save_image(path: str, img: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except Exception:
        plt.imsave(path, img)


def _topic_type_map(reader) -> dict:
    topics = reader.get_all_topics_and_types()
    return {t.name: t.type for t in topics}


def export_frames(bag_dir: str, out_dir: str, fps: float) -> None:
    os.makedirs(out_dir, exist_ok=True)

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    type_map = _topic_type_map(reader)

    last_costmap = None
    last_plan = None
    last_state = None
    last_goal = None

    last_export_ns = None
    interval_ns = int(1e9 / max(fps, 1e-6))
    frame_idx = 0

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in type_map:
            continue
        if topic not in (IMAGE_TOPIC, COSTMAP_TOPIC, PLAN_TOPIC, STATE_TOPIC, GOAL_TOPIC):
            continue

        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if topic == COSTMAP_TOPIC:
            last_costmap = msg
        elif topic == PLAN_TOPIC:
            last_plan = msg
        elif topic == STATE_TOPIC:
            last_state = msg
        elif topic == GOAL_TOPIC:
            last_goal = msg
        elif topic == IMAGE_TOPIC:
            if last_export_ns is not None and (t - last_export_ns) < interval_ns:
                continue
            rgb = _image_to_rgb(msg)
            if rgb is None:
                continue
            bev = _render_bev(last_costmap, last_plan, last_state, last_goal, rgb.shape[0])
            if bev.shape[0] != rgb.shape[0]:
                bev = bev[:rgb.shape[0], :, :]
            combined = np.concatenate([rgb, bev], axis=1)
            frame_idx += 1
            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}_{t}.png")
            _save_image(out_path, combined)
            last_export_ns = t

    print(f"Exported {frame_idx} frames to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export side-by-side camera + BEV figures from rosbag2.")
    parser.add_argument("--bag", required=True, help="Path to rosbag2 directory")
    parser.add_argument("--out", required=True, help="Output directory for PNG frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Export rate in frames per second")
    args = parser.parse_args()

    export_frames(args.bag, args.out, args.fps)


if __name__ == "__main__":
    main()
