"""Homography helpers built on the shared camera model."""

import numpy as np

from unav_common.camera_model import ObliqueCameraModel


def build_camera(cam_pos, look_at, img_width, img_height, fov_h_rad):
    return ObliqueCameraModel(
        cam_pos=cam_pos,
        look_at=look_at,
        img_width=img_width,
        img_height=img_height,
        fov_h_rad=fov_h_rad,
    )


def world_to_pixel(cam_pos, look_at, img_width, img_height, fov_h_rad, x, y, z=0.0):
    camera = build_camera(cam_pos, look_at, img_width, img_height, fov_h_rad)
    return camera.world_to_pixel(x, y, z)


def pixel_to_world(cam_pos, look_at, img_width, img_height, fov_h_rad, u, v):
    camera = build_camera(cam_pos, look_at, img_width, img_height, fov_h_rad)
    return camera.pixel_to_world(u, v)


def compute_homography(cam_pos, look_at, img_width, img_height, fov_h_rad):
    camera = build_camera(cam_pos, look_at, img_width, img_height, fov_h_rad)
    return camera.H.copy(), camera.H_inv.copy()
