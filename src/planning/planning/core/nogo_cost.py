"""Smooth no-go-zone obstacle penalties derived from warehouse prism geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy


VALID_NOGO_PENALTIES = ('gaussian', 'softplus', 'log_barrier')


@dataclass(frozen=True)
class NogoCostConfig:
    penalty_type: str = 'softplus'
    weight: float = 0.0
    safe_distance: float = 0.35
    gaussian_sigma: float = 0.25
    softplus_scale: float = 0.08
    logbarrier_scale: float = 0.25
    logbarrier_eps: float = 1e-3
    geometry_json: str = ''


class NogoZoneCostModel:
    """Geometry-based no-go-zone penalty around obstacle footprints."""

    def __init__(self, cfg: NogoCostConfig):
        penalty_type = str(cfg.penalty_type or '').strip().lower()
        if penalty_type not in VALID_NOGO_PENALTIES:
            raise ValueError(
                f"penalty_type must be one of: {', '.join(VALID_NOGO_PENALTIES)}"
            )
        self.cfg = cfg
        self.penalty_type = penalty_type
        self.weight = float(max(cfg.weight, 0.0))
        self.safe_distance = float(max(cfg.safe_distance, 0.0))
        self.gaussian_sigma = float(max(cfg.gaussian_sigma, 1e-6))
        self.softplus_scale = float(max(cfg.softplus_scale, 1e-6))
        self.logbarrier_scale = float(max(cfg.logbarrier_scale, 1e-6))
        self.logbarrier_eps = float(max(cfg.logbarrier_eps, 1e-6))

        self.scene = scene_from_json(cfg.geometry_json)
        self.prisms = tuple(self.scene.prisms)

        self._xmins = np.asarray([float(p.xmin) for p in self.prisms], dtype=float)
        self._xmaxs = np.asarray([float(p.xmax) for p in self.prisms], dtype=float)
        self._ymins = np.asarray([float(p.ymin) for p in self.prisms], dtype=float)
        self._ymaxs = np.asarray([float(p.ymax) for p in self.prisms], dtype=float)

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 and bool(self.prisms)

    @property
    def signature(self) -> tuple:
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
            'nogo_cost',
            self.penalty_type,
            round(self.weight, 6),
            round(self.safe_distance, 6),
            round(self.gaussian_sigma, 6),
            round(self.softplus_scale, 6),
            round(self.logbarrier_scale, 6),
            round(self.logbarrier_eps, 8),
            len(self.prisms),
            *scene_sig,
        )

    def _clearance_np(self, xy: np.ndarray) -> float:
        signed_d = float(signed_distance_to_union_xy(self.prisms, np.asarray(xy, dtype=float))[0])
        return signed_d - self.safe_distance

    def _penalty_from_clearance_np(self, clearance: float) -> float:
        if not self.enabled:
            return 0.0

        if self.penalty_type == 'gaussian':
            outside = float(np.exp(-0.5 * (max(clearance, 0.0) / self.gaussian_sigma) ** 2))
            inside_extra = max(-clearance, 0.0) / self.gaussian_sigma
            return self.weight * (outside + inside_extra)

        if self.penalty_type == 'softplus':
            z = float(np.clip(-clearance / self.softplus_scale, -60.0, 60.0))
            return self.weight * float(np.log1p(np.exp(z)))

        denom = max(clearance, self.logbarrier_eps)
        return self.weight * float(np.log1p(self.logbarrier_scale / denom))

    def penalty_state_np(self, m) -> float:
        if not self.enabled:
            return 0.0
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        clearance = self._clearance_np(xy)
        return self._penalty_from_clearance_np(clearance)

    def make_penalty_state_jax(self):
        try:
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('JAX is not available for no-go-zone cost') from exc

        if (not self.prisms) or self.weight <= 0.0:
            def zero_penalty(_m):
                return jnp.array(0.0)
            return zero_penalty

        xmins = jnp.asarray(self._xmins)
        xmaxs = jnp.asarray(self._xmaxs)
        ymins = jnp.asarray(self._ymins)
        ymaxs = jnp.asarray(self._ymaxs)
        weight = float(self.weight)
        safe_distance = float(self.safe_distance)
        gaussian_sigma = float(self.gaussian_sigma)
        softplus_scale = float(self.softplus_scale)
        logbarrier_scale = float(self.logbarrier_scale)
        logbarrier_eps = float(self.logbarrier_eps)
        penalty_type = self.penalty_type

        def signed_distance_xy(x, y):
            dx = jnp.maximum(jnp.maximum(xmins - x, 0.0), x - xmaxs)
            dy = jnp.maximum(jnp.maximum(ymins - y, 0.0), y - ymaxs)
            outside = jnp.sqrt(dx * dx + dy * dy)

            inside_x = jnp.minimum(x - xmins, xmaxs - x)
            inside_y = jnp.minimum(y - ymins, ymaxs - y)
            inside_depth = jnp.minimum(inside_x, inside_y)
            inside = (dx <= 0.0) & (dy <= 0.0)
            signed = jnp.where(inside, -inside_depth, outside)
            return jnp.min(signed)

        def penalty_state_jax(m):
            signed_d = signed_distance_xy(m[0], m[1])
            clearance = signed_d - safe_distance

            if penalty_type == 'gaussian':
                outside = jnp.exp(-0.5 * (jnp.maximum(clearance, 0.0) / gaussian_sigma) ** 2)
                inside_extra = jnp.maximum(-clearance, 0.0) / gaussian_sigma
                base = outside + inside_extra
            elif penalty_type == 'softplus':
                z = jnp.clip(-clearance / softplus_scale, -60.0, 60.0)
                base = jnp.log1p(jnp.exp(z))
            else:
                denom = jnp.maximum(clearance, logbarrier_eps)
                base = jnp.log1p(logbarrier_scale / denom)
            return weight * base

        return penalty_state_jax
