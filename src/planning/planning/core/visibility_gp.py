"""Fixed GP visibility field utilities (pre-known model, no online learning)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip_prob(p: np.ndarray | float, eps: float) -> np.ndarray | float:
    return np.clip(p, eps, 1.0 - eps)


class SimpleRBFGP:
    """Lightweight RBF GP regressor used for fixed visibility maps."""

    def __init__(self, *, length_scale=1.5, signal_var=1.0, noise_var=5e-2, jitter=1e-8):
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)
        self.noise_var = float(noise_var)
        self.jitter = float(jitter)
        self.X_train = None
        self.y_mean = None
        self.L = None
        self.alpha = None

    def _kernel(self, Xa, Xb):
        Xa = np.asarray(Xa, dtype=float)
        Xb = np.asarray(Xb, dtype=float)
        d2 = np.sum((Xa[:, None, :] - Xb[None, :, :]) ** 2, axis=2)
        ls2 = max(self.length_scale ** 2, 1e-12)
        return self.signal_var * np.exp(-0.5 * d2 / ls2)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.X_train = X
        self.y_mean = float(y.mean())
        y0 = y - self.y_mean

        K = self._kernel(X, X)
        K = K + (self.noise_var + self.jitter) * np.eye(X.shape[0])
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y0))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Ks = self._kernel(X, self.X_train)
        return self.y_mean + Ks @ self.alpha


@dataclass
class FixedGPVisibilityConfig:
    map_xmin: float = -5.0
    map_xmax: float = 5.0
    map_ymin: float = -5.0
    map_ymax: float = 5.0
    map_nx: int = 140
    map_ny: int = 120
    occ_center_x: float = -1.2
    occ_center_y: float = -1.8
    occ_radius: float = 0.9
    occ_tau: float = 0.15
    gp_length_scale: float = 1.4
    gp_noise_var: float = 0.15
    n_uniform: int = 320
    n_focus: int = 260
    n_inside: int = 220
    seed: int = 0
    min_prob: float = 1e-4


class FixedGPVisibilityModel:
    """Pre-baked GP visibility field from synthetic occlusion labels."""

    def __init__(self, cfg: FixedGPVisibilityConfig):
        self.cfg = cfg
        self.min_prob = float(max(cfg.min_prob, 1e-6))
        self.xs = np.linspace(float(cfg.map_xmin), float(cfg.map_xmax), int(max(cfg.map_nx, 4)))
        self.ys = np.linspace(float(cfg.map_ymin), float(cfg.map_ymax), int(max(cfg.map_ny, 4)))
        self._occ_center = np.array([float(cfg.occ_center_x), float(cfg.occ_center_y)], dtype=float)
        self._occ_radius = float(cfg.occ_radius)
        self._occ_tau = float(max(cfg.occ_tau, 1e-3))

        rng = np.random.default_rng(int(cfg.seed))
        X_train, y_train = self._build_training_data(rng)

        gp = SimpleRBFGP(
            length_scale=float(cfg.gp_length_scale),
            signal_var=1.0,
            noise_var=float(max(cfg.gp_noise_var, 1e-8)),
        ).fit(X_train, y_train)

        Xg, Yg = np.meshgrid(self.xs, self.ys)
        XY = np.column_stack([Xg.ravel(), Yg.ravel()])
        P = gp.predict(XY).reshape(Xg.shape)
        self.P_map = _clip_prob(P, self.min_prob).astype(float)

    @property
    def signature(self) -> tuple:
        c = self.cfg
        return (
            round(float(c.map_xmin), 6),
            round(float(c.map_xmax), 6),
            round(float(c.map_ymin), 6),
            round(float(c.map_ymax), 6),
            int(c.map_nx),
            int(c.map_ny),
            round(float(c.occ_center_x), 6),
            round(float(c.occ_center_y), 6),
            round(float(c.occ_radius), 6),
            round(float(c.occ_tau), 6),
            round(float(c.gp_length_scale), 6),
            round(float(c.gp_noise_var), 6),
            int(c.seed),
        )

    def _p_true(self, xy):
        arr = np.asarray(xy, dtype=float)
        d = np.linalg.norm(arr - self._occ_center, axis=-1) - self._occ_radius
        p = 1.0 / (1.0 + np.exp(-d / self._occ_tau))
        return _clip_prob(p, self.min_prob)

    def _build_training_data(self, rng: np.random.Generator):
        c = self.cfg
        n_u = int(max(c.n_uniform, 32))
        n_f = int(max(c.n_focus, 32))
        n_i = int(max(c.n_inside, 32))

        X_uniform = np.column_stack([
            rng.uniform(float(c.map_xmin), float(c.map_xmax), n_u),
            rng.uniform(float(c.map_ymin), float(c.map_ymax), n_u),
        ])

        X_focus = self._occ_center + rng.normal(scale=[0.9, 0.9], size=(n_f, 2))
        X_focus[:, 0] = np.clip(X_focus[:, 0], float(c.map_xmin), float(c.map_xmax))
        X_focus[:, 1] = np.clip(X_focus[:, 1], float(c.map_ymin), float(c.map_ymax))

        ang = rng.uniform(0.0, 2.0 * np.pi, n_i)
        rad = self._occ_radius * np.sqrt(rng.uniform(0.0, 1.0, n_i)) * 0.98
        X_inside = np.column_stack([
            self._occ_center[0] + rad * np.cos(ang),
            self._occ_center[1] + rad * np.sin(ang),
        ])

        X = np.vstack([X_uniform, X_focus, X_inside])
        p = self._p_true(X)
        y = (rng.random(X.shape[0]) < p).astype(float)
        return X, y

    def _bilinear_np(self, x: float, y: float) -> float:
        xs = self.xs
        ys = self.ys
        P = self.P_map

        ix = int(np.searchsorted(xs, x, side='right') - 1)
        iy = int(np.searchsorted(ys, y, side='right') - 1)
        ix = max(0, min(ix, xs.shape[0] - 2))
        iy = max(0, min(iy, ys.shape[0] - 2))

        x0, x1 = xs[ix], xs[ix + 1]
        y0, y1 = ys[iy], ys[iy + 1]

        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        tx = float(np.clip(tx, 0.0, 1.0))
        ty = float(np.clip(ty, 0.0, 1.0))

        z00 = P[iy, ix]
        z10 = P[iy, ix + 1]
        z01 = P[iy + 1, ix]
        z11 = P[iy + 1, ix + 1]

        z0 = (1.0 - tx) * z00 + tx * z10
        z1 = (1.0 - tx) * z01 + tx * z11
        return float((1.0 - ty) * z0 + ty * z1)

    def prob_state_np(self, m) -> float:
        x = float(m[0])
        y = float(m[1])
        p = self._bilinear_np(x, y)
        return float(_clip_prob(p, self.min_prob))

    def make_prob_state_jax(self):
        try:
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("JAX is not available for visibility model") from exc

        xs_j = jnp.asarray(self.xs)
        ys_j = jnp.asarray(self.ys)
        P_j = jnp.asarray(self.P_map)
        eps = float(self.min_prob)

        def p_vis_j(m):
            x = m[0]
            y = m[1]

            ix = jnp.clip(jnp.searchsorted(xs_j, x, side='right') - 1, 0, xs_j.shape[0] - 2)
            iy = jnp.clip(jnp.searchsorted(ys_j, y, side='right') - 1, 0, ys_j.shape[0] - 2)

            x0, x1 = xs_j[ix], xs_j[ix + 1]
            y0, y1 = ys_j[iy], ys_j[iy + 1]
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

