import math
import numpy as np

from unav_common.camera_model import ObliqueCameraModel


class PixelToBevTransformer:
    def __init__(self, cam_pos, look_at, img_width, img_height, fov_h_rad, rng=None):
        self.cam_pos = np.array(cam_pos, dtype=float)
        self.look_at = np.array(look_at, dtype=float)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.fov_h_rad = float(fov_h_rad)
        self.rng = rng

    def _make_camera(self, cam_pos, look_at):
        return ObliqueCameraModel(
            cam_pos=cam_pos,
            look_at=look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h_rad,
        )

    def pixel_to_world(self, u, v, transform_noise_sigma=0.0):
        cam_pos = self.cam_pos.copy()
        look_at = self.look_at.copy()
        if transform_noise_sigma > 0.0 and self.rng is not None:
            cam_pos += self.rng.normal(0.0, transform_noise_sigma, size=3)
            look_at += self.rng.normal(0.0, transform_noise_sigma, size=3)

        camera = self._make_camera(cam_pos, look_at)
        return camera.pixel_to_world(u, v)

    def pixel_noise_to_metric(self, u, v, pixel_noise_sigma, transform_noise_sigma=0.0):
        world = self.pixel_to_world(u, v, transform_noise_sigma=transform_noise_sigma)
        if world is None:
            return None
        x, y = world
        dx = self.pixel_to_world(u + pixel_noise_sigma, v, transform_noise_sigma=transform_noise_sigma)
        dy = self.pixel_to_world(u, v + pixel_noise_sigma, transform_noise_sigma=transform_noise_sigma)
        if dx is None or dy is None:
            return None
        sigma_x = math.hypot(dx[0] - x, dx[1] - y)
        sigma_y = math.hypot(dy[0] - x, dy[1] - y)
        return sigma_x, sigma_y
