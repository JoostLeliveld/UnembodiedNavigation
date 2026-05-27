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

    def pixel_to_world_at_z(self, u, v, z_plane, transform_noise_sigma=0.0):
        """Back-project a pixel to the world plane z=z_plane.

        The default ``pixel_to_world`` uses the planar homography for z=0; for
        elevated keypoints (e.g. front/rear marker discs at z ~= 0.215) using
        the ground-plane homography produces a biased BEV position. Here we
        intersect the camera ray with z=z_plane analytically.

        Returns (x, y) in world coordinates or None if the ray is parallel to
        the plane / behind the camera.
        """
        cam_pos = self.cam_pos.copy()
        look_at = self.look_at.copy()
        if transform_noise_sigma > 0.0 and self.rng is not None:
            cam_pos = cam_pos + self.rng.normal(0.0, transform_noise_sigma, size=3)
            look_at = look_at + self.rng.normal(0.0, transform_noise_sigma, size=3)
        camera = self._make_camera(cam_pos, look_at)
        K_inv = np.linalg.inv(camera.K)
        ray_cam = K_inv @ np.array([float(u), float(v), 1.0], dtype=float)
        ray_world = camera.R.T @ ray_cam
        if abs(ray_world[2]) < 1e-9:
            return None
        t = (float(z_plane) - cam_pos[2]) / ray_world[2]
        if t <= 0.0:
            return None
        x = cam_pos[0] + t * ray_world[0]
        y = cam_pos[1] + t * ray_world[1]
        return float(x), float(y)
