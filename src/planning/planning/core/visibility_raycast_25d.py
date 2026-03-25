"""2.5D geometry-driven visibility model for warehouse occlusion scenes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from planning.core.visibility_gp import SimpleRBFGP, _clip_prob
from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy, top_heights_for_xy


def _sigmoid(x):
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-arr))


def _logit(p):
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


@dataclass
class Raycast25DVisibilityConfig:
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
    height_tau: float = 0.08
    ray_samples: int = 120
    target_height_m: float = 0.0
    geometry_json: str = ''
    camera_pos: tuple[float, float, float] = (-3.0, -3.0, 6.0)
    transition_band: float = 0.16
    n_uniform: int = 480
    n_focus: int = 560
    n_inside: int = 320
    ray_batch_size: int = 768
    seed: int = 0
    min_prob: float = 1e-4
    cache_enabled: bool = True
    cache_dir: str = ''


class Raycast25DVisibilityModel:
    """Shared latent opacity field with height-aware ray integration."""

    def __init__(self, cfg: Raycast25DVisibilityConfig):
        self.cfg = cfg
        self.min_prob = float(max(cfg.min_prob, 1e-6))
        self.xs = np.linspace(float(cfg.map_xmin), float(cfg.map_xmax), int(max(cfg.map_nx, 4)))
        self.ys = np.linspace(float(cfg.map_ymin), float(cfg.map_ymax), int(max(cfg.map_ny, 4)))
        self.camera_pos = np.asarray(cfg.camera_pos, dtype=float).reshape(3)
        self.target_height = float(cfg.target_height_m)
        self.scene = scene_from_json(cfg.geometry_json)
        self.prisms = tuple(self.scene.prisms)

        self.rho_mean_map = np.full((self.ys.shape[0], self.xs.shape[0]), float(cfg.prior_occ), dtype=float)
        self.rho_conservative_map = self.rho_mean_map.copy()

        if not self._load_cached_maps():
            if self.prisms:
                self._fit_latent_opacity_field(np.random.default_rng(int(cfg.seed)))
            self.P_map = self._precompute_visibility_map()
            self._store_cached_maps()

    @property
    def signature(self) -> tuple:
        c = self.cfg
        scene_sig = []
        for prism in self.prisms:
            scene_sig.extend([
                round(float(prism.xmin), 4),
                round(float(prism.xmax), 4),
                round(float(prism.ymin), 4),
                round(float(prism.ymax), 4),
                round(float(prism.zmin), 4),
                round(float(prism.zmax), 4),
            ])
        return (
            'raycast_25d',
            round(float(c.map_xmin), 4),
            round(float(c.map_xmax), 4),
            round(float(c.map_ymin), 4),
            round(float(c.map_ymax), 4),
            int(c.map_nx),
            int(c.map_ny),
            round(float(c.gp_length_scale), 4),
            round(float(c.gp_noise_var), 4),
            round(float(c.prior_occ), 6),
            round(float(c.beta), 4),
            round(float(c.height_tau), 4),
            round(float(c.transition_band), 4),
            int(c.ray_samples),
            int(c.n_uniform),
            int(c.n_focus),
            int(c.n_inside),
            int(c.ray_batch_size),
            round(float(c.target_height_m), 4),
            round(float(c.min_prob), 8),
            round(float(self.camera_pos[0]), 4),
            round(float(self.camera_pos[1]), 4),
            round(float(self.camera_pos[2]), 4),
            int(c.seed),
            len(self.prisms),
            *scene_sig,
        )

    def _fit_latent_opacity_field(self, rng: np.random.Generator) -> None:
        X_train, rho_train = self._build_training_data(rng)
        gp = SimpleRBFGP(
            length_scale=float(self.cfg.gp_length_scale),
            signal_var=1.0,
            noise_var=float(max(self.cfg.gp_noise_var, 1e-8)),
        ).fit(X_train, _logit(rho_train))

        Xg, Yg = np.meshgrid(self.xs, self.ys)
        XY = np.column_stack([Xg.ravel(), Yg.ravel()])
        mu_f, sigma_f = gp.predict_mean_std(XY)
        rho_mean = _sigmoid(mu_f)
        rho_cons = _sigmoid(mu_f + float(self.cfg.beta) * sigma_f)
        self.rho_mean_map = _clip_prob(rho_mean.reshape(Yg.shape), self.min_prob).astype(float)
        self.rho_conservative_map = _clip_prob(rho_cons.reshape(Yg.shape), self.min_prob).astype(float)

    def _cache_file_path(self) -> Path | None:
        if not bool(self.cfg.cache_enabled):
            return None
        raw_dir = str(self.cfg.cache_dir or '').strip()
        if raw_dir:
            cache_root = Path(os.path.expanduser(raw_dir))
        else:
            cache_root = Path(os.path.expanduser('~/.cache/unembodied_navigation/visibility_maps'))
        signature_json = json.dumps(self.signature, separators=(',', ':'), ensure_ascii=True)
        cache_key = hashlib.sha256(signature_json.encode('utf-8')).hexdigest()
        return cache_root / f'{cache_key}.npz'

    def _load_cached_maps(self) -> bool:
        cache_path = self._cache_file_path()
        if cache_path is None or (not cache_path.is_file()):
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                rho_mean = np.asarray(data['rho_mean_map'], dtype=float)
                rho_cons = np.asarray(data['rho_conservative_map'], dtype=float)
                p_map = np.asarray(data['P_map'], dtype=float)
            expected_shape = (self.ys.shape[0], self.xs.shape[0])
            if rho_mean.shape != expected_shape or rho_cons.shape != expected_shape or p_map.shape != expected_shape:
                return False
            self.rho_mean_map = _clip_prob(rho_mean, self.min_prob).astype(float)
            self.rho_conservative_map = _clip_prob(rho_cons, self.min_prob).astype(float)
            self.P_map = _clip_prob(p_map, self.min_prob).astype(float)
            return True
        except Exception:
            return False

    def _store_cached_maps(self) -> None:
        cache_path = self._cache_file_path()
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                rho_mean_map=np.asarray(self.rho_mean_map, dtype=float),
                rho_conservative_map=np.asarray(self.rho_conservative_map, dtype=float),
                P_map=np.asarray(self.P_map, dtype=float),
            )
        except Exception:
            return

    def _build_training_data(self, rng: np.random.Generator):
        c = self.cfg
        prisms = self.prisms
        n_u = int(max(c.n_uniform, 128))
        n_f = int(max(c.n_focus, max(8 * len(prisms), 128)))
        n_i = int(max(c.n_inside, max(6 * len(prisms), 96)))

        X_uniform = np.column_stack([
            rng.uniform(float(c.map_xmin), float(c.map_xmax), n_u),
            rng.uniform(float(c.map_ymin), float(c.map_ymax), n_u),
        ])

        prism_ids_focus = rng.integers(0, len(prisms), size=n_f)
        centers = np.vstack([prisms[idx].center_xy for idx in prism_ids_focus])
        scales = np.vstack([np.maximum(prisms[idx].size_xy, 0.08) for idx in prism_ids_focus])
        X_focus = centers + rng.normal(size=(n_f, 2)) * (0.75 * scales)
        X_focus[:, 0] = np.clip(X_focus[:, 0], float(c.map_xmin), float(c.map_xmax))
        X_focus[:, 1] = np.clip(X_focus[:, 1], float(c.map_ymin), float(c.map_ymax))

        prism_ids_inside = rng.integers(0, len(prisms), size=n_i)
        X_inside = np.zeros((n_i, 2), dtype=float)
        for row, prism_idx in enumerate(prism_ids_inside):
            prism = prisms[prism_idx]
            pad_x = min(0.04, 0.2 * max(prism.xmax - prism.xmin, 1e-6))
            pad_y = min(0.04, 0.2 * max(prism.ymax - prism.ymin, 1e-6))
            x_lo = min(prism.xmin + pad_x, prism.xmax)
            x_hi = max(prism.xmax - pad_x, prism.xmin)
            y_lo = min(prism.ymin + pad_y, prism.ymax)
            y_hi = max(prism.ymax - pad_y, prism.ymin)
            X_inside[row, 0] = rng.uniform(x_lo, x_hi)
            X_inside[row, 1] = rng.uniform(y_lo, y_hi)

        X_centers = np.vstack([prism.center_xy for prism in prisms])
        X = np.vstack([X_uniform, X_focus, X_inside, X_centers])
        signed = signed_distance_to_union_xy(prisms, X)
        rho = self._labels_from_signed_distance(signed)
        return X, rho

    def _labels_from_signed_distance(self, signed: np.ndarray) -> np.ndarray:
        band = float(max(self.cfg.transition_band, 1e-3))
        inner_band = 0.5 * band
        clear = float(np.clip(self.cfg.prior_occ, self.min_prob, 0.25))
        outer = float(np.clip(max(0.15, 8.0 * clear), clear, 0.3))
        inner = 0.60
        occupied = 1.0 - self.min_prob

        labels = np.full_like(signed, clear, dtype=float)
        deep_inside = signed <= -inner_band
        shallow_inside = (signed > -inner_band) & (signed < 0.0)
        near_outside = (signed >= 0.0) & (signed < band)

        labels[deep_inside] = occupied
        if np.any(shallow_inside):
            alpha = (-signed[shallow_inside]) / inner_band
            labels[shallow_inside] = inner + alpha * (occupied - inner)
        if np.any(near_outside):
            alpha = signed[near_outside] / band
            labels[near_outside] = outer + alpha * (clear - outer)
        return np.clip(labels, self.min_prob, 1.0 - self.min_prob)

    def _bilinear_map_np(self, field: np.ndarray, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        xs = self.xs
        ys = self.ys

        ix = np.searchsorted(xs, x, side='right') - 1
        iy = np.searchsorted(ys, y, side='right') - 1
        ix = np.clip(ix, 0, xs.shape[0] - 2)
        iy = np.clip(iy, 0, ys.shape[0] - 2)

        x0 = xs[ix]
        x1 = xs[ix + 1]
        y0 = ys[iy]
        y1 = ys[iy + 1]
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

    def _precompute_visibility_map(self) -> np.ndarray:
        Xg, Yg = np.meshgrid(self.xs, self.ys)
        if not self.prisms:
            return np.full_like(Xg, 1.0 - self.min_prob, dtype=float)

        targets = np.column_stack([Xg.ravel(), Yg.ravel()])
        vis = np.empty(targets.shape[0], dtype=float)
        ns = int(max(self.cfg.ray_samples, 8))
        fractions = (np.arange(ns, dtype=float) + 0.5) / float(ns)
        cam_xy = self.camera_pos[:2]
        cam_z = float(self.camera_pos[2])
        target_z = float(self.target_height)
        batch_size = int(max(self.cfg.ray_batch_size, 64))

        for start in range(0, targets.shape[0], batch_size):
            stop = min(start + batch_size, targets.shape[0])
            target_batch = targets[start:stop]
            ray_xy = cam_xy[None, None, :] + fractions[None, :, None] * (target_batch[:, None, :] - cam_xy[None, None, :])
            ray_z = cam_z + fractions[None, :] * (target_z - cam_z)
            rho = self._bilinear_map_np(self.rho_conservative_map, ray_xy.reshape(-1, 2)).reshape(target_batch.shape[0], ns)
            z_top = top_heights_for_xy(self.prisms, ray_xy.reshape(-1, 2)).reshape(target_batch.shape[0], ns)
            gate = _sigmoid((z_top - ray_z) / float(max(self.cfg.height_tau, 1e-3)))
            total_dist = np.sqrt(
                np.sum((target_batch - cam_xy[None, :]) ** 2, axis=1)
                + (target_z - cam_z) ** 2
            )
            ds = total_dist / float(ns)
            opacity = np.sum(rho * gate, axis=1) * ds
            vis[start:stop] = np.exp(-opacity)

        return _clip_prob(vis.reshape(Yg.shape), self.min_prob).astype(float)

    def prob_state_np(self, m) -> float:
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        p = self._bilinear_map_np(self.P_map, xy)[0]
        return float(_clip_prob(p, self.min_prob))

    def make_prob_state_jax(self):
        try:
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('JAX is not available for visibility model') from exc

        xs_j = jnp.asarray(self.xs)
        ys_j = jnp.asarray(self.ys)
        P_j = jnp.asarray(self.P_map)
        eps = float(self.min_prob)

        def p_vis_j(m):
            x = m[0]
            y = m[1]
            ix = jnp.clip(jnp.searchsorted(xs_j, x, side='right') - 1, 0, xs_j.shape[0] - 2)
            iy = jnp.clip(jnp.searchsorted(ys_j, y, side='right') - 1, 0, ys_j.shape[0] - 2)
            x0 = xs_j[ix]
            x1 = xs_j[ix + 1]
            y0 = ys_j[iy]
            y1 = ys_j[iy + 1]
            tx = jnp.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
            ty = jnp.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))
            tx = jnp.clip(tx, 0.0, 1.0)
            ty = jnp.clip(ty, 0.0, 1.0)
            z00 = P_j[iy, ix]
            z10 = P_j[iy, ix + 1]
            z01 = P_j[iy + 1, ix]
            z11 = P_j[iy + 1, ix + 1]
            z0 = (1.0 - tx) * z00 + tx * z10
            z1 = (1.0 - tx) * z01 + tx * z11
            z = (1.0 - ty) * z0 + ty * z1
            return jnp.clip(z, eps, 1.0 - eps)

        return p_vis_j
