#!/usr/bin/env python3
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MODEL_SDF = Path('src/sim/models/external_camera/model.sdf')
LOOK_AT = np.array([1.5, 1.5, 0.0])
EPS_METERS = 0.02
SAMPLES = 50


def parse_camera_sdf(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()

    pose_text = root.find('.//model/pose').text.strip()
    pose_vals = [float(v) for v in pose_text.split()]
    if len(pose_vals) != 6:
        raise ValueError('Unexpected pose format in model.sdf')

    cam_elem = root.find('.//sensor[@type="camera"]/camera')
    h_fov = float(cam_elem.find('horizontal_fov').text.strip())
    width = int(cam_elem.find('image/width').text.strip())
    height = int(cam_elem.find('image/height').text.strip())

    return pose_vals, h_fov, width, height


def compute_intrinsics(width, height, fov_h):
    f = (width / 2.0) / math.tan(fov_h / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def compute_lookat_rotation(cam_pos, look_at, up_hint=None):
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])
    z_cam = look_at - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)
    x_cam = np.cross(z_cam, up_hint)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)
    return np.array([x_cam, y_cam, z_cam])


def world_to_pixel(H, R, t, width, height, x, y):
    world_pt = np.array([x, y, 1.0])
    pix = H @ world_pt
    if abs(pix[2]) < 1e-9:
        return None
    u = pix[0] / pix[2]
    v = pix[1] / pix[2]
    cam_pt = R @ np.array([x, y, 0.0]) + t
    if cam_pt[2] <= 0:
        return None
    if not (0 <= u < width and 0 <= v < height):
        return None
    return u, v


def pixel_to_world(H_inv, u, v):
    world_h = H_inv @ np.array([u, v, 1.0])
    if abs(world_h[2]) < 1e-9:
        return None
    x = world_h[0] / world_h[2]
    y = world_h[1] / world_h[2]
    return x, y


def main():
    if not MODEL_SDF.exists():
        raise SystemExit(f'Missing camera SDF: {MODEL_SDF}')

    (x, y, z, roll, pitch, yaw), fov_h, width, height = parse_camera_sdf(MODEL_SDF)
    cam_pos = np.array([x, y, z])

    K = compute_intrinsics(width, height, fov_h)
    R = compute_lookat_rotation(cam_pos, LOOK_AT)
    t = -R @ cam_pos
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H_inv = np.linalg.inv(H)

    direction = LOOK_AT - cam_pos
    horiz = math.hypot(direction[0], direction[1])
    pitch_deg = math.degrees(math.atan2(-direction[2], horiz))
    yaw_deg = math.degrees(math.atan2(direction[1], direction[0]))
    print('Camera pose from SDF:', cam_pos, 'rpy=', (roll, pitch, yaw))
    print('Look-at implied yaw/pitch (deg):', yaw_deg, pitch_deg)

    max_err = 0.0
    for _ in range(SAMPLES):
        rx = random.uniform(-1.0, 4.0)
        ry = random.uniform(-1.0, 4.0)
        pix = world_to_pixel(H, R, t, width, height, rx, ry)
        if pix is None:
            continue
        u, v = pix
        back = pixel_to_world(H_inv, u, v)
        if back is None:
            continue
        bx, by = back
        err = math.hypot(rx - bx, ry - by)
        max_err = max(max_err, err)

    print(f'Max round-trip error: {max_err:.6f} m')
    if max_err > EPS_METERS:
        raise SystemExit(f'ERROR: Homography error exceeds {EPS_METERS} m')
    print('OK: Homography consistency within tolerance')


if __name__ == '__main__':
    main()
