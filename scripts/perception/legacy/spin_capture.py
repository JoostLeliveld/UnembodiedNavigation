#!/usr/bin/env python3
"""Controlled in-place-spin capture: teleport the robot to a FIXED (x,y) and
sweep yaw, grabbing one external-camera frame per pose. The truth (x,y) is held
constant, so any movement of the robot in the image as yaw varies isolates a
truth/base-frame-vs-visible-body effect (vs a sensing/timing one).

Requires bringup_sim already running (provides /external_camera/image_raw and
/world/warehouse_aws/set_pose). Writes frames + poses.csv to --out.
"""
import argparse, csv, math, os, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
import sys
sys.path.insert(0, os.path.dirname(__file__))


def img_to_bgr(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        a = arr.reshape(msg.height, msg.width, 3)
        return a[..., ::-1] if msg.encoding == "rgb8" else a
    raise RuntimeError(f"unhandled encoding {msg.encoding}")


class Spin(Node):
    def __init__(self, world, z):
        super().__init__("spin_capture")
        self.z = z
        self.latest = None
        self.cnt = 0
        self.cli = self.create_client(SetEntityPose, f"/world/{world}/set_pose")
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_subscription(Image, "/external_camera/image_raw", self._cb, 10)

    def _cb(self, msg):
        self.cnt += 1
        self.latest = (self.cnt, img_to_bgr(msg).copy())

    def ready(self, timeout=90.0):
        end = time.monotonic() + timeout
        svc = False
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if not svc and self.cli.wait_for_service(timeout_sec=0.05):
                svc = True
            if svc and self.latest is not None:
                return
        raise RuntimeError("timed out waiting for set_pose + camera image")

    def teleport(self, x, y, yaw):
        for _ in range(3):
            self.cmd.publish(Twist()); rclpy.spin_once(self, timeout_sec=0.01)
        req = SetEntityPose.Request()
        req.entity = Entity(name="turtlebot3", type=Entity.MODEL)
        req.pose.position.x = float(x); req.pose.position.y = float(y); req.pose.position.z = float(self.z)
        req.pose.orientation.z = math.sin(0.5 * yaw); req.pose.orientation.w = math.cos(0.5 * yaw)
        fut = self.cli.call_async(req)
        t0 = time.monotonic()
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - t0 > 10:
                raise RuntimeError("set_pose timeout")
        for _ in range(3):
            self.cmd.publish(Twist()); rclpy.spin_once(self, timeout_sec=0.01)

    def grab_after(self, settle=1.2):
        before = self.cnt
        end = time.monotonic() + 5.0
        time.sleep(settle)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.cnt > before + 3:
                break
        return self.latest[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="warehouse_aws")
    ap.add_argument("--out", default="/tmp/spin_test_frames")
    ap.add_argument("--z", type=float, default=0.05)
    args = ap.parse_args()
    import cv2
    os.makedirs(args.out, exist_ok=True)
    positions = [(-0.9, -0.45, "highErr_west"), (1.0, 0.0, "central_A3"), (3.3, -1.0, "start_east")]
    yaws = list(range(0, 360, 30))
    rclpy.init()
    node = Spin(args.world, args.z)
    node.ready()
    rows = []
    for (x, y, tag) in positions:
        for yd in yaws:
            yaw = math.radians(yd)
            node.teleport(x, y, yaw)
            im = node.grab_after()
            fn = f"{args.out}/spin_{tag}_{yd:03d}.png"
            cv2.imwrite(fn, im)
            rows.append({"tag": tag, "x": x, "y": y, "yaw_deg": yd, "frame": fn})
            node.get_logger().info(f"captured {tag} yaw={yd} -> {os.path.basename(fn)}")
    with open(f"{args.out}/poses.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tag", "x", "y", "yaw_deg", "frame"]); w.writeheader(); w.writerows(rows)
    print(f"captured {len(rows)} frames -> {args.out}")
    node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
