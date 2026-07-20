import math
import numpy as np


class ObliqueCameraModel:
    def __init__(
        self,
        cam_pos=(-3.0, -3.0, 6.0),
        look_at=(1.5, 1.5, 0.0),
        img_width=1920,
        img_height=1080,
        fov_h_rad=1.5708,
        up_hint=(0.0, 0.0, 1.0),
    ):
        self.cam_pos = np.array(cam_pos, dtype=float)
        self.look_at = np.array(look_at, dtype=float)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.fov_h_rad = float(fov_h_rad)
        self.up_hint = np.array(up_hint, dtype=float)

        self.K = self._compute_intrinsics()
        self.R = self._compute_lookat_rotation(self.cam_pos, self.look_at, self.up_hint)
        self.t = -self.R @ self.cam_pos
        RT_planar = np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        self.H = self.K @ RT_planar
        self.H_inv = np.linalg.inv(self.H)

    def _compute_intrinsics(self):
        f = (self.img_width / 2.0) / math.tan(self.fov_h_rad / 2.0)
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    def _compute_lookat_rotation(self, cam_pos, look_at, up_hint):
        z_cam = look_at - cam_pos
        z_cam = z_cam / np.linalg.norm(z_cam)
        x_cam = np.cross(z_cam, up_hint)
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        return np.array([x_cam, y_cam, z_cam])

    def world_to_pixel(self, x, y, z=0.0):
        world_3d = np.array([x, y, z], dtype=float)
        cam_pt = self.R @ (world_3d - self.cam_pos)
        if abs(cam_pt[2]) < 1e-10:
            return 0.0, 0.0, False
        pixel_h = self.K @ cam_pt
        u = pixel_h[0] / pixel_h[2]
        v = pixel_h[1] / pixel_h[2]
        visible = (0 <= u < self.img_width) and (0 <= v < self.img_height)
        if cam_pt[2] <= 0:
            visible = False
        return float(u), float(v), bool(visible)

    def pixel_to_world(self, u, v):
        pixel_pt = np.array([u, v, 1.0])
        world_h = self.H_inv @ pixel_pt
        if abs(world_h[2]) < 1e-10:
            return None
        return float(world_h[0] / world_h[2]), float(world_h[1] / world_h[2])

    def pixel_to_world_at_z(self, u, v, z_plane):
        """Back-project a pixel to the world plane z=z_plane."""
        ray_cam = np.linalg.inv(self.K) @ np.array([float(u), float(v), 1.0], dtype=float)
        ray_world = self.R.T @ ray_cam
        if abs(ray_world[2]) < 1e-10:
            return None
        t = (float(z_plane) - self.cam_pos[2]) / ray_world[2]
        if t <= 0.0:
            return None
        world = self.cam_pos + t * ray_world
        return float(world[0]), float(world[1])

    def g(self, state):
        x, y, theta = state
        uv = self.g_uv(state)
        return np.array([uv[0], uv[1], theta], dtype=float)

    def g_uv(self, state):
        x, y = state[0], state[1]
        u, v, _ = self.world_to_pixel(x, y, 0.0)
        return np.array([u, v], dtype=float)
