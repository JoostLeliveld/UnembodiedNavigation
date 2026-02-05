import numpy as np

from unav_common.camera_model import ObliqueCameraModel


class HomographyModel:
    def __init__(self, cam_pos, look_at, img_width, img_height, fov_h_rad):
        self.camera = ObliqueCameraModel(
            cam_pos=cam_pos,
            look_at=look_at,
            img_width=img_width,
            img_height=img_height,
            fov_h_rad=fov_h_rad,
        )

    @property
    def H(self):
        return self.camera.H

    @property
    def H_inv(self):
        return self.camera.H_inv

    def world_to_pixel(self, x, y, z=0.0):
        return self.camera.world_to_pixel(x, y, z)

    def pixel_to_world(self, u, v):
        return self.camera.pixel_to_world(u, v)

    @staticmethod
    def lookat_rotation(cam_pos, look_at, up_hint=None):
        camera = ObliqueCameraModel(cam_pos=cam_pos, look_at=look_at)
        return camera.R
