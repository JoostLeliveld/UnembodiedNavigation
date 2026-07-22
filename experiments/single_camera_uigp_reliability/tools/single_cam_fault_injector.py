#!/usr/bin/env python3
"""Single-camera fault injector (Tier-1) — a param-driven PIXEL interposer that
injects a *calibration* fault, not a synthetic output offset.

The detector publishes the raw robot pixel (`/perception/pixel_pose`, PoseStamped,
frame_id='image', x=u, y=v); the planner back-projects THAT pixel through its own
`ObliqueCameraModel` calibration to run the EKF correction (unicycle_planner_node
`_pixel_cb` -> `camera.pixel_to_world` / `world_to_pixel`, innovation in pixels).

So a faithful fault does NOT add a made-up bias to the output. It perturbs the
camera's *calibration* and RE-PROJECTS the same detection through it — a bumped
mount, a sagging bracket, a drifting pan — so the same robot lands at a DIFFERENT
pixel. The resulting error is pose-dependent by construction (small near the image
centre, larger at grazing/peripheral views), exactly as a real mis-calibration is,
and it is the failure mode static per-camera calibration cannot self-detect. This
is the pre-registered E6 fault model (`reliability.calibration_perturbation`),
realised online via the SAME projection geometry the planner uses.

Mechanism (calib_drift):
  1. recover the robot's true ground point from the reported pixel using the TRUE
     calibration (identical to the planner's): world = true_cam.pixel_to_world(u,v)
  2. re-image that world point through a DRIFTED-extrinsics camera (cam centre +
     look-at perturbed, ramping in after onset): (u',v') = drift_cam.world_to_pixel(world)
  3. publish (u',v'); if the drift pushes the robot out of the drifted frame -> a miss
At zero drift the two cameras are identical and the round-trip is the identity, so
the injector is a perfect pass-through until onset.

Other faults (physically faithful as-is):
  occlusion : drop a fraction (--drop-prob) of detections (partial view blockage)
  dropout   : stop republishing entirely after onset (camera/stream death)

The planner is launched with `pixel_topic:=<out topic>` — NO planner-core surgery.

Offline check: python3 single_cam_fault_injector.py --selftest
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402


def ramp_frac(t_since_onset, ramp_s):
    """Linear 0->1 ramp of the calibration drift over ramp_s after onset, then hold."""
    if t_since_onset < 0.0:
        return 0.0
    if ramp_s <= 0.0:
        return 1.0
    return min(1.0, t_since_onset / ramp_s)


def build_drift_cam(true_cam, frac, campos_drift, lookat_drift):
    """A camera whose extrinsics are the true ones plus frac * the drift vectors.

    campos_drift = metres added to the camera centre (bump / sag);
    lookat_drift = metres the aim point slides (pan / tilt). Intrinsics unchanged.
    """
    return ObliqueCameraModel(
        cam_pos=[true_cam.cam_pos[i] + frac * campos_drift[i] for i in range(3)],
        look_at=[true_cam.look_at[i] + frac * lookat_drift[i] for i in range(3)],
        img_width=true_cam.img_width,
        img_height=true_cam.img_height,
        fov_h_rad=true_cam.fov_h_rad,
    )


def reproject_through(uv, true_cam, drift_cam):
    """Recover ground point via true calibration, re-image via drifted calibration.

    Returns the faulted pixel (u', v'), or None if the drifted camera cannot see
    the point (robot pushed out of frame by the drift = a miss).
    """
    world = true_cam.pixel_to_world(uv[0], uv[1])
    if world is None:
        return uv
    u2, v2, vis = drift_cam.world_to_pixel(world[0], world[1], 0.0)
    if not vis:
        return None
    return (float(u2), float(v2))


def _selftest():
    cam = ObliqueCameraModel(cam_pos=[-3.0, -3.0, 6.0], look_at=[1.5, 1.5, 0.0],
                             img_width=1280, img_height=720, fov_h_rad=1.5708)
    lookat_drift = [0.8, 0.8, 0.0]   # ~5 deg pan drift of the aim point
    campos_drift = [0.0, 0.0, 0.0]

    # (1) zero drift => identity pass-through (round-trip pixel->world->pixel)
    dcam0 = build_drift_cam(cam, 0.0, campos_drift, lookat_drift)
    for uv in [(400.0, 300.0), (900.0, 500.0)]:
        out = reproject_through(uv, cam, dcam0)
        assert out is not None and abs(out[0] - uv[0]) < 1e-6 and abs(out[1] - uv[1]) < 1e-6, (uv, out)

    # (2) full drift => nonzero, pose-dependent WORLD error that grows with range.
    dcam1 = build_drift_cam(cam, 1.0, campos_drift, lookat_drift)
    errs = {}
    for tag, world_xy in {"near": (1.5, 1.5), "far": (4.0, 4.0)}.items():
        u, v, vis = cam.world_to_pixel(*world_xy, 0.0)
        assert vis, (tag, world_xy)
        faulted_uv = reproject_through((u, v), cam, dcam1)
        assert faulted_uv is not None
        w2 = cam.pixel_to_world(*faulted_uv)          # world the planner will infer
        errs[tag] = math.hypot(w2[0] - world_xy[0], w2[1] - world_xy[1])
    assert errs["near"] > 1e-3, errs
    assert errs["far"] > errs["near"], errs           # pose-dependent, grows with range

    # (3) ramp monotonic 0->1
    assert ramp_frac(-1.0, 10.0) == 0.0 and ramp_frac(5.0, 10.0) == 0.5 and ramp_frac(99.0, 10.0) == 1.0

    print(f"SELFTEST PASS (identity@0; calib-drift world-err near={errs['near']:.3f}m "
          f"far={errs['far']:.3f}m grows with range; occlusion/dropout=stream faults)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in-topic", default="/perception/pixel_pose")
    ap.add_argument("--out-topic", default="/perception/pixel_pose_faulted")
    ap.add_argument("--fault", choices=("none", "calib_drift", "occlusion", "dropout"),
                    default="calib_drift")
    ap.add_argument("--fault-after-s", type=float, default=30.0, help="wall seconds after first msg")
    # calibration-drift geometry (the true calibration must match the planner's)
    ap.add_argument("--cam-pos", default="-3.0,-3.0,6.0")
    ap.add_argument("--look-at", default="1.5,1.5,0.0")
    ap.add_argument("--img-width", type=int, default=1280)
    ap.add_argument("--img-height", type=int, default=720)
    ap.add_argument("--fov-h-rad", type=float, default=1.5708)
    ap.add_argument("--calib-campos-drift", default="0.0,0.0,0.0",
                    help="metres added to camera centre at full drift (bump/sag)")
    ap.add_argument("--calib-lookat-drift", default="0.8,0.8,0.0",
                    help="metres the aim point slides at full drift (pan/tilt)")
    ap.add_argument("--drift-ramp-s", type=float, default=20.0,
                    help="seconds from onset to full calibration drift")
    # stream faults
    ap.add_argument("--drop-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())

    import time
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped

    def _vec3(s):
        parts = [float(x) for x in str(s).replace(";", ",").split(",") if x.strip() != ""]
        if len(parts) != 3:
            raise ValueError(f"expected 3 comma-separated values, got {s!r}")
        return parts

    rng = random.Random(args.seed)
    true_cam = ObliqueCameraModel(cam_pos=_vec3(args.cam_pos), look_at=_vec3(args.look_at),
                                  img_width=args.img_width, img_height=args.img_height,
                                  fov_h_rad=args.fov_h_rad)
    campos_drift = _vec3(args.calib_campos_drift)
    lookat_drift = _vec3(args.calib_lookat_drift)

    class Injector(Node):
        def __init__(self):
            super().__init__("single_cam_fault_injector")
            self.t0 = None
            self.onset_logged = False
            self.create_subscription(PoseStamped, args.in_topic, self._cb, 20)
            self.pub = self.create_publisher(PoseStamped, args.out_topic, 10)
            self.get_logger().info(
                f"fault injector: {args.in_topic} -> {args.out_topic}; fault={args.fault} "
                f"after {args.fault_after_s}s (calib drift lookat={lookat_drift} campos={campos_drift} "
                f"over {args.drift_ramp_s}s)")

        def _cb(self, msg):
            now = time.time()
            if self.t0 is None:
                self.t0 = now
            t_since_onset = (now - self.t0) - args.fault_after_s
            active = t_since_onset >= 0.0
            if active and not self.onset_logged:
                self.get_logger().warn(f"FAULT ONSET ({args.fault}) at t={now - self.t0:.1f}s after first msg")
                self.onset_logged = True

            if not active or args.fault == "none":
                self.pub.publish(msg)
                return
            if args.fault == "dropout":
                return
            if args.fault == "occlusion":
                if rng.random() < args.drop_prob:
                    return
                self.pub.publish(msg)
                return
            # calib_drift: re-project through drifted calibration
            frac = ramp_frac(t_since_onset, args.drift_ramp_s)
            drift_cam = build_drift_cam(true_cam, frac, campos_drift, lookat_drift)
            uv = reproject_through((msg.pose.position.x, msg.pose.position.y), true_cam, drift_cam)
            if uv is None:
                return  # drifted out of frame -> miss
            msg.pose.position.x, msg.pose.position.y = float(uv[0]), float(uv[1])
            self.pub.publish(msg)

    rclpy.init()
    node = Injector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
