"""Direct GP visibility-map model for external-camera observability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from planning.core.visibility_raycast_25d import SimpleRBFGP, _clip_prob, _logit, _sigmoid
from unav_common.occlusion_geometry import scene_from_json, segment_occluded


@dataclass
class GPVisibilityMapConfig:
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
    geometry_json: str = ''
    camera_pos: tuple[float, float, float] = (-3.0, -3.0, 6.0)
    target_height_m: float = 0.0
    n_train_total: int = 1400
    n_train_boundary: int = 320
    seed: int = 0
    min_prob: float = 1e-4
    cache_enabled: bool = True
    cache_dir: str = ''


class GPVisibilityMapModel:
    """Direct GP fit of visibility probability over (x, y)."""

    def __init__(self, cfg: GPVisibilityMapConfig):
        self.cfg = cfg
        self.min_prob = float(max(cfg.min_prob, 1e-6))
        self.xs = np.linspace(float(cfg.map_xmin), float(cfg.map_xmax), int(max(cfg.map_nx, 4)))
        self.ys = np.linspace(float(cfg.map_ymin), float(cfg.map_ymax), int(max(cfg.map_ny, 4)))
        self.camera_pos = np.asarray(cfg.camera_pos, dtype=float).reshape(3)
        self.target_height = float(cfg.target_height_m)
        self.scene = scene_from_json(cfg.geometry_json)
        self.prisms = tuple(self.scene.prisms)

        prior_vis = float(np.clip(1.0 - float(cfg.prior_occ), self.min_prob, 1.0 - self.min_prob))
        shape = (self.ys.shape[0], self.xs.shape[0])
        self.P_mean_map = np.full(shape, prior_vis, dtype=float)
        self.P_conservative_map = self.P_mean_map.copy()
        self.P_map = self.P_conservative_map.copy()

        if not self._load_cached_maps():
            self._fit_visibility_field(np.random.default_rng(int(cfg.seed)))
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
            'gp_visibility',
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
            round(float(c.target_height_m), 4),
            int(c.n_train_total),
            int(c.n_train_boundary),
            round(float(c.min_prob), 8),
            round(float(self.camera_pos[0]), 4),
            round(float(self.camera_pos[1]), 4),
            round(float(self.camera_pos[2]), 4),
            int(c.seed),
            len(self.prisms),
            *scene_sig,
        )

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
                p_mean = np.asarray(data['P_mean_map'], dtype=float)
                p_cons = np.asarray(data['P_conservative_map'], dtype=float)
                p_map = np.asarray(data['P_map'], dtype=float)
            expected_shape = (self.ys.shape[0], self.xs.shape[0])
            if p_mean.shape != expected_shape or p_cons.shape != expected_shape or p_map.shape != expected_shape:
                return False
            self.P_mean_map = _clip_prob(p_mean, self.min_prob).astype(float)
            self.P_conservative_map = _clip_prob(p_cons, self.min_prob).astype(float)
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
                P_mean_map=np.asarray(self.P_mean_map, dtype=float),
                P_conservative_map=np.asarray(self.P_conservative_map, dtype=float),
                P_map=np.asarray(self.P_map, dtype=float),
            )
        except Exception:
            return

    def _oracle_visibility_map(self) -> np.ndarray:
        Xg, Yg = np.meshgrid(self.xs, self.ys)
        xy = np.column_stack([Xg.ravel(), Yg.ravel()])
        vis = np.full(xy.shape[0], 1.0 - self.min_prob, dtype=float)
        if not self.prisms:
            return vis.reshape(Yg.shape)

        target_z = float(self.target_height)
        for idx, pt in enumerate(xy):
            target = np.array([float(pt[0]), float(pt[1]), target_z], dtype=float)
            blocked = segment_occluded(self.prisms, self.camera_pos, target)
            vis[idx] = self.min_prob if blocked else (1.0 - self.min_prob)
        return vis.reshape(Yg.shape)

    def _boundary_mask(self, oracle_map: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(oracle_map, dtype=bool)
        if oracle_map.shape[1] > 1:
            delta_x = np.abs(np.diff(oracle_map, axis=1))
            mask[:, :-1] |= delta_x > 0.1
            mask[:, 1:] |= delta_x > 0.1
        if oracle_map.shape[0] > 1:
            delta_y = np.abs(np.diff(oracle_map, axis=0))
            mask[:-1, :] |= delta_y > 0.1
            mask[1:, :] |= delta_y > 0.1
        return mask

    def _sample_training_set(self, oracle_map: np.ndarray, rng: np.random.Generator):
        Xg, Yg = np.meshgrid(self.xs, self.ys)
        XY = np.column_stack([Xg.ravel(), Yg.ravel()])
        labels = oracle_map.ravel()
        boundary = self._boundary_mask(oracle_map).ravel()
        visible = labels > 0.5
        occluded = ~visible

        all_idx = np.arange(labels.shape[0], dtype=int)
        boundary_idx = all_idx[boundary]
        visible_idx = all_idx[visible & (~boundary)]
        occluded_idx = all_idx[occluded & (~boundary)]

        take_boundary = min(int(max(self.cfg.n_train_boundary, 0)), boundary_idx.shape[0])
        chosen = []
        if take_boundary > 0:
            chosen.extend(rng.choice(boundary_idx, size=take_boundary, replace=False).tolist())

        remaining = max(int(max(self.cfg.n_train_total, 64)) - len(chosen), 0)
        take_visible = min(remaining // 2, visible_idx.shape[0])
        take_occluded = min(remaining - take_visible, occluded_idx.shape[0])

        if take_visible > 0:
            chosen.extend(rng.choice(visible_idx, size=take_visible, replace=False).tolist())
        if take_occluded > 0:
            chosen.extend(rng.choice(occluded_idx, size=take_occluded, replace=False).tolist())

        if len(chosen) < int(max(self.cfg.n_train_total, 64)):
            remaining_pool = np.setdiff1d(all_idx, np.asarray(chosen, dtype=int), assume_unique=False)
            extra = min(int(max(self.cfg.n_train_total, 64)) - len(chosen), remaining_pool.shape[0])
            if extra > 0:
                chosen.extend(rng.choice(remaining_pool, size=extra, replace=False).tolist())

        idx = np.asarray(sorted(set(chosen)), dtype=int)
        return XY[idx], labels[idx]

    def _fit_visibility_field(self, rng: np.random.Generator) -> None:
        oracle_map = self._oracle_visibility_map()
        X_train, y_train = self._sample_training_set(oracle_map, rng)
        prior_vis = float(np.clip(1.0 - float(self.cfg.prior_occ), self.min_prob, 1.0 - self.min_prob))
        gp = SimpleRBFGP(
            length_scale=float(self.cfg.gp_length_scale),
            signal_var=1.0,
            noise_var=float(max(self.cfg.gp_noise_var, 1e-8)),
        ).fit(X_train, _logit(np.clip(y_train, self.min_prob, 1.0 - self.min_prob)))

        Xg, Yg = np.meshgrid(self.xs, self.ys)
        XY = np.column_stack([Xg.ravel(), Yg.ravel()])
        mu_f, sigma_f = gp.predict_mean_std(XY)
        p_mean = _sigmoid(mu_f)
        conservative_latent = mu_f - float(self.cfg.beta) * sigma_f
        p_cons = _sigmoid(conservative_latent)

        self.P_mean_map = _clip_prob(p_mean.reshape(Yg.shape), self.min_prob).astype(float)
        self.P_conservative_map = _clip_prob(p_cons.reshape(Yg.shape), self.min_prob).astype(float)
        self.P_map = self.P_conservative_map.copy()

        # Keep a few aliases for generic visualization/export paths.
        self.prior_vis = prior_vis

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

    def make_prob_state_casadi(self):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('CasADi is not available for visibility model') from exc

        cached = getattr(self, '_prob_state_casadi', None)
        if cached is not None:
            return cached

        signature_json = json.dumps(self.signature, separators=(',', ':'), ensure_ascii=True)
        suffix = hashlib.sha256(signature_json.encode('utf-8')).hexdigest()[:10]
        values = np.asarray(self.P_map.T, dtype=float).ravel(order='F')
        interp = ca.interpolant(
            f'gp_visibility_prob_{suffix}',
            'linear',
            [self.xs.tolist(), self.ys.tolist()],
            values,
        )
        eps = float(self.min_prob)

        def p_vis_ca(m):
            z = interp(ca.vertcat(m[0], m[1]))
            return ca.fmin(ca.fmax(z, eps), 1.0 - eps)

        self._prob_state_casadi = p_vis_ca
        return p_vis_ca
