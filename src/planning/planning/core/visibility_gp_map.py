"""Empirical GP visibility-map model loaded from an offline artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from planning.core.visibility_raycast_25d import _clip_prob


@dataclass
class GPVisibilityMapConfig:
    artifact_path: str = ""
    map_xmin: float = -5.0
    map_xmax: float = 5.0
    map_ymin: float = -5.0
    map_ymax: float = 5.0
    map_nx: int = 140
    map_ny: int = 120
    gp_length_scale: float = 1.2
    gp_noise_var: float = 0.1
    prior_occ: float = 0.005
    beta: float = 1.0
    geometry_json: str = ""
    camera_pos: tuple[float, float, float] = (-3.0, -3.0, 6.0)
    target_height_m: float = 0.0
    n_train_total: int = 0
    n_train_boundary: int = 0
    seed: int = 0
    min_prob: float = 1e-4
    cache_enabled: bool = False
    cache_dir: str = ""


class GPVisibilityMapModel:
    """Load a fixed GP visibility field used by the thesis experiments."""

    def __init__(self, cfg: GPVisibilityMapConfig):
        self.cfg = cfg
        self.min_prob = float(max(cfg.min_prob, 1e-6))
        self.artifact_path = self._resolve_artifact_path(cfg.artifact_path)
        self._load_artifact()
        self._prob_state_casadi = None

    @staticmethod
    def _resolve_artifact_path(raw_path: str) -> Path:
        path = Path(str(raw_path or "").strip()).expanduser()
        if not str(path):
            raise RuntimeError(
                "GP visibility planning requires 'visibility_artifact_path' to point to an empirical GP artifact."
            )
        if not path.is_file():
            raise RuntimeError(f"Empirical GP visibility artifact not found: {path}")
        return path.resolve()

    def _load_artifact(self) -> None:
        try:
            with np.load(self.artifact_path, allow_pickle=False) as data:
                xs = np.asarray(data["xs"], dtype=float)
                ys = np.asarray(data["ys"], dtype=float)
                p_map = np.asarray(data["P_map"], dtype=float)
                p_mean = np.asarray(data["P_mean_map"], dtype=float) if "P_mean_map" in data.files else p_map
                p_cons = np.asarray(data["P_conservative_map"], dtype=float) if "P_conservative_map" in data.files else p_map
                camera_pos = (
                    np.asarray(data["camera_pos"], dtype=float).reshape(-1)
                    if "camera_pos" in data.files
                    else np.asarray(self.cfg.camera_pos, dtype=float).reshape(-1)
                )
                target_height = (
                    float(np.asarray(data["target_height"], dtype=float).reshape(-1)[0])
                    if "target_height" in data.files
                    else float(self.cfg.target_height_m)
                )
        except KeyError as exc:
            raise RuntimeError(
                f"Empirical GP visibility artifact {self.artifact_path} is missing required field {exc!s}."
            ) from exc

        expected_shape = (ys.shape[0], xs.shape[0])
        if xs.ndim != 1 or ys.ndim != 1 or xs.size < 2 or ys.size < 2:
            raise RuntimeError(
                f"Empirical GP visibility artifact {self.artifact_path} has an invalid grid."
            )
        for field_name, grid in (
            ("P_map", p_map),
            ("P_mean_map", p_mean),
            ("P_conservative_map", p_cons),
        ):
            if grid.shape != expected_shape:
                raise RuntimeError(
                    f"Empirical GP visibility artifact {self.artifact_path} has {field_name} shape {grid.shape}, "
                    f"expected {expected_shape}."
                )

        self.xs = xs
        self.ys = ys
        self.P_mean_map = _clip_prob(p_mean, self.min_prob).astype(float)
        self.P_conservative_map = _clip_prob(p_cons, self.min_prob).astype(float)
        self.P_map = _clip_prob(p_map, self.min_prob).astype(float)
        self.camera_pos = np.asarray(camera_pos, dtype=float).reshape(3)
        self.target_height = float(target_height)

        self.cfg.map_xmin = float(xs[0])
        self.cfg.map_xmax = float(xs[-1])
        self.cfg.map_ymin = float(ys[0])
        self.cfg.map_ymax = float(ys[-1])
        self.cfg.map_nx = int(xs.size)
        self.cfg.map_ny = int(ys.size)

    @property
    def signature(self) -> tuple:
        stat = self.artifact_path.stat()
        digest = hashlib.sha256(str(self.artifact_path).encode("utf-8")).hexdigest()[:12]
        return (
            "empirical_gp_visibility",
            digest,
            str(self.artifact_path),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            int(self.xs.size),
            int(self.ys.size),
            round(float(self.xs[0]), 6),
            round(float(self.xs[-1]), 6),
            round(float(self.ys[0]), 6),
            round(float(self.ys[-1]), 6),
        )

    def _bilinear_map_np(self, field: np.ndarray, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        x = pts[:, 0]
        y = pts[:, 1]

        ix = np.searchsorted(self.xs, x, side="right") - 1
        iy = np.searchsorted(self.ys, y, side="right") - 1
        ix = np.clip(ix, 0, self.xs.shape[0] - 2)
        iy = np.clip(iy, 0, self.ys.shape[0] - 2)

        x0 = self.xs[ix]
        x1 = self.xs[ix + 1]
        y0 = self.ys[iy]
        y1 = self.ys[iy + 1]
        tx = np.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
        ty = np.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))
        tx = np.clip(tx, 0.0, 1.0)
        ty = np.clip(ty, 0.0, 1.0)

        z00 = field[iy, ix]
        z10 = field[iy, ix + 1]
        z01 = field[iy + 1, ix]
        z11 = field[iy + 1, ix + 1]
        z0 = (1.0 - tx) * z00 + tx * z10
        z1 = (1.0 - tx) * z01 + tx * z11
        return (1.0 - ty) * z0 + ty * z1

    def prob_state_np(self, m) -> float:
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        p = self._bilinear_map_np(self.P_map, xy)[0]
        return float(_clip_prob(p, self.min_prob))

    def make_prob_state_jax(self):
        try:
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("JAX is not available for visibility model") from exc

        xs_j = jnp.asarray(self.xs)
        ys_j = jnp.asarray(self.ys)
        p_j = jnp.asarray(self.P_map)
        eps = float(self.min_prob)

        def p_vis_j(m):
            x = m[0]
            y = m[1]
            ix = jnp.clip(jnp.searchsorted(xs_j, x, side="right") - 1, 0, xs_j.shape[0] - 2)
            iy = jnp.clip(jnp.searchsorted(ys_j, y, side="right") - 1, 0, ys_j.shape[0] - 2)
            x0 = xs_j[ix]
            x1 = xs_j[ix + 1]
            y0 = ys_j[iy]
            y1 = ys_j[iy + 1]
            tx = jnp.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
            ty = jnp.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))
            tx = jnp.clip(tx, 0.0, 1.0)
            ty = jnp.clip(ty, 0.0, 1.0)
            z00 = p_j[iy, ix]
            z10 = p_j[iy, ix + 1]
            z01 = p_j[iy + 1, ix]
            z11 = p_j[iy + 1, ix + 1]
            z0 = (1.0 - tx) * z00 + tx * z10
            z1 = (1.0 - tx) * z01 + tx * z11
            z = (1.0 - ty) * z0 + ty * z1
            return jnp.clip(z, eps, 1.0 - eps)

        return p_vis_j

    def make_prob_state_casadi(self):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CasADi is not available for visibility model") from exc

        if self._prob_state_casadi is not None:
            return self._prob_state_casadi

        values = np.asarray(self.P_map.T, dtype=float).ravel(order="F")
        interp = ca.interpolant(
            f"empirical_gp_visibility_{hashlib.sha1(str(self.artifact_path).encode('utf-8')).hexdigest()[:10]}",
            "linear",
            [self.xs.tolist(), self.ys.tolist()],
            values,
        )
        eps = float(self.min_prob)

        def p_vis_ca(m):
            z = interp(ca.vertcat(m[0], m[1]))
            return ca.fmin(ca.fmax(z, eps), 1.0 - eps)

        self._prob_state_casadi = p_vis_ca
        return p_vis_ca
