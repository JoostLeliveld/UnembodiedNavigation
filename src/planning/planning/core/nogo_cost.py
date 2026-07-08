"""Smooth no-go-zone obstacle penalties derived from warehouse prism geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy, _get_union_boundary_segments


VALID_NOGO_PENALTIES = ('warning_band',)


@dataclass(frozen=True)
class NogoCostConfig:
    penalty_type: str = 'warning_band'
    weight: float = 0.0
    safe_distance: float = 0.35
    logbarrier_eps: float = 1e-3
    # warning_band: hinged-log warning penalty parameters (penalty_type ==
    # 'warning_band'). `warning_band` (b) is the clearance width over which the
    # soft warning ramps in; `near_weight` (w_near) scales the warning term. The
    # violation term reuses `weight` (w_viol) and `logbarrier_eps` (eps). The
    # penalty is exactly zero for valid interior states with clearance >= b, so
    # raising `weight` to crush violations does not bias the choice between two
    # fully valid routes (e.g. a narrow vs a wide aisle).
    warning_band: float = 0.05
    near_weight: float = 50.0
    geometry_json: str = ''
    # 'keep_out': penalise being inside/near the prisms (obstacle footprints).
    # 'keep_in':  penalise leaving the prism union (driveable region);
    #             safe_distance is the required mean clearance from the
    #             driveable-region boundary. When belief no-go is enabled,
    #             kappa additionally expands this by kappa * sigma_max.
    mode: str = 'keep_out'


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
        self.mode = str(getattr(cfg, 'mode', 'keep_out') or 'keep_out').strip().lower()
        if self.mode not in ('keep_out', 'keep_in'):
            raise ValueError("mode must be 'keep_out' or 'keep_in'")
        self.weight = float(max(cfg.weight, 0.0))
        self.safe_distance = float(max(cfg.safe_distance, 0.0))
        self.logbarrier_eps = float(max(cfg.logbarrier_eps, 1e-6))
        self.warning_band = float(max(getattr(cfg, 'warning_band', 0.05), 1e-6))
        self.near_weight = float(max(getattr(cfg, 'near_weight', 50.0), 0.0))

        self.scene = scene_from_json(cfg.geometry_json)
        self.prisms = tuple(self.scene.prisms)

        self._xmins = np.asarray([float(p.xmin) for p in self.prisms], dtype=float)
        self._xmaxs = np.asarray([float(p.xmax) for p in self.prisms], dtype=float)
        self._ymins = np.asarray([float(p.ymin) for p in self.prisms], dtype=float)
        self._ymaxs = np.asarray([float(p.ymax) for p in self.prisms], dtype=float)
        self.union_boundary_segments = _get_union_boundary_segments(self.prisms)

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
            self.mode,
            self.penalty_type,
            round(self.weight, 6),
            round(self.safe_distance, 6),
            round(self.logbarrier_eps, 8),
            round(self.warning_band, 6),
            round(self.near_weight, 6),
            len(self.prisms),
            *scene_sig,
        )

    def _clearance_np(self, xy: np.ndarray) -> float:
        keep_in_flag = (self.mode == 'keep_in')
        signed_d = float(signed_distance_to_union_xy(self.prisms, np.asarray(xy, dtype=float), keep_in=keep_in_flag)[0])
        if self.mode == 'keep_in':
            # signed_d <= 0 inside the driveable union. Positive clearance
            # means the mean state is safely inside the known driveable floor.
            return -signed_d - self.safe_distance
        return signed_d - self.safe_distance

    def signed_distance_state_np(self, m) -> float:
        if not self.prisms:
            return float('inf')
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        keep_in_flag = (self.mode == 'keep_in')
        signed_d = float(signed_distance_to_union_xy(self.prisms, xy, keep_in=keep_in_flag)[0])
        if self.mode == 'keep_in':
            # Positive means inside the known driveable union; negative means
            # outside it. This keeps penetration/inside diagnostics aligned with
            # "violation depth" instead of reporting valid lanes as no-go.
            return -signed_d
        return signed_d

    def penetration_depth_state_np(self, m) -> float:
        signed_d = self.signed_distance_state_np(m)
        return float(max(-signed_d, 0.0)) if np.isfinite(signed_d) else 0.0

    def clearance_state_np(self, m) -> float:
        if not self.enabled:
            return float('inf')
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        return float(self._clearance_np(xy))

    @staticmethod
    def _sigma_max_xy_np(S) -> float:
        cov_xy = np.asarray(S, dtype=float)[:2, :2]
        cov_xy = 0.5 * (cov_xy + cov_xy.T)
        try:
            eigvals = np.linalg.eigvalsh(cov_xy)
        except np.linalg.LinAlgError:
            return 0.0
        return float(math.sqrt(max(float(np.max(eigvals)), 0.0)))

    def clearance_belief_tube_np(self, m, S, *, kappa: float = 2.0) -> float:
        """Clearance of the mean plus a kappa-sigma xy belief tube.

        For keep_in, positive means the predicted belief tube remains inside
        the known driveable region after the configured mean clearance margin.
        For keep_out, this conservatively shrinks the obstacle clearance by the
        same covariance margin.
        """
        if not self.enabled:
            return float('inf')
        margin = max(float(kappa), 0.0) * self._sigma_max_xy_np(S)
        return float(self.clearance_state_np(m) - margin)

    def inside_state_np(self, m) -> bool:
        return bool(self.penetration_depth_state_np(m) > 0.0)

    def _penalty_from_clearance_np(self, clearance: float) -> float:
        if not self.enabled:
            return 0.0

        # Hinged-log warning + quadratic violation. Exactly zero for valid
        # interior states (clearance >= warning_band), so raising `weight`
        # crushes violations without biasing the choice between two valid
        # routes (e.g. narrow vs wide aisle). Keeps a log-like shape inside
        # the thin warning band near the boundary.
        band_excess = max(self.warning_band - clearance, 0.0) / self.warning_band
        warn = self.near_weight * float(np.log1p(band_excess * band_excess))
        viol = max(-clearance, 0.0) / self.logbarrier_eps
        return warn + self.weight * float(viol * viol)

    def penalty_state_np(self, m) -> float:
        if not self.enabled:
            return 0.0
        xy = np.array([float(m[0]), float(m[1])], dtype=float)
        clearance = self._clearance_np(xy)
        return self._penalty_from_clearance_np(clearance)

    def penalty_belief_np(self, m, S, *, kappa: float = 1.0) -> float:
        """Expected no-go penalty under the current xy belief covariance."""
        if not self.enabled:
            return 0.0
        if self.mode == 'keep_in':
            clearance = self.clearance_belief_tube_np(m, S, kappa=kappa)
            return self._penalty_from_clearance_np(clearance)

        mean_xy = np.asarray([float(m[0]), float(m[1])], dtype=float)
        cov_xy = np.asarray(S, dtype=float)[:2, :2]
        cov_xy = 0.5 * (cov_xy + cov_xy.T)
        kappa = max(float(kappa), 1e-6)
        try:
            chol = np.linalg.cholesky(cov_xy + 1e-9 * np.eye(2))
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov_xy)
            chol = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-9)))
        spread = math.sqrt(2.0 + kappa) * chol
        sigma_points = (
            mean_xy,
            mean_xy + spread[:, 0],
            mean_xy - spread[:, 0],
            mean_xy + spread[:, 1],
            mean_xy - spread[:, 1],
        )
        weights = (
            kappa / (2.0 + kappa),
            1.0 / (2.0 * (2.0 + kappa)),
            1.0 / (2.0 * (2.0 + kappa)),
            1.0 / (2.0 * (2.0 + kappa)),
            1.0 / (2.0 * (2.0 + kappa)),
        )
        return float(sum(w * self.penalty_state_np([p[0], p[1], 0.0]) for p, w in zip(sigma_points, weights)))

    def make_penalty_state_casadi(self):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('CasADi is not available for no-go-zone cost') from exc

        if (not self.prisms) or self.weight <= 0.0:
            def zero_penalty(_m):
                return 0.0
            return zero_penalty

        xmins = ca.DM(self._xmins)
        xmaxs = ca.DM(self._xmaxs)
        ymins = ca.DM(self._ymins)
        ymaxs = ca.DM(self._ymaxs)
        weight = float(self.weight)
        safe_distance = float(self.safe_distance)
        logbarrier_eps = float(self.logbarrier_eps)
        warning_band = float(self.warning_band)
        near_weight = float(self.near_weight)
        mode = self.mode

        def signed_distance_xy(x, y):
            if mode == 'keep_in' and self.union_boundary_segments:
                q = ca.vertcat(x, y)
                dists = []
                for p1, p2 in self.union_boundary_segments:
                    p1_dm = ca.DM(p1)
                    p2_dm = ca.DM(p2)
                    v = p2_dm - p1_dm
                    w = q - p1_dm
                    v_len_sq = ca.sumsqr(v)
                    t = ca.if_else(v_len_sq < 1e-9, 0.0, ca.fmin(ca.fmax(ca.dot(w, v) / v_len_sq, 0.0), 1.0))
                    closest = p1_dm + t * v
                    dists.append(ca.norm_2(q - closest))
                min_dist = ca.mmin(ca.vertcat(*dists))

                is_inside = False
                for p in self.prisms:
                    dx = ca.fmax(ca.fmax(p.xmin - x, 0.0), x - p.xmax)
                    dy = ca.fmax(ca.fmax(p.ymin - y, 0.0), y - p.ymax)
                    is_inside = ca.logic_or(is_inside, ca.logic_and(dx <= 0.0, dy <= 0.0))
                return ca.if_else(is_inside, -min_dist, min_dist)
            else:
                dx = ca.fmax(ca.fmax(xmins - x, 0.0), x - xmaxs)
                dy = ca.fmax(ca.fmax(ymins - y, 0.0), y - ymaxs)
                outside = ca.sqrt(ca.power(dx, 2) + ca.power(dy, 2))

                inside_x = ca.fmin(x - xmins, xmaxs - x)
                inside_y = ca.fmin(y - ymins, ymaxs - y)
                inside_depth = ca.fmin(inside_x, inside_y)
                inside = ca.logic_and(dx <= 0.0, dy <= 0.0)
                signed = ca.if_else(inside, -inside_depth, outside)
                return ca.mmin(signed)

        def penalty_state_casadi(m):
            signed_d = signed_distance_xy(m[0], m[1])
            if mode == 'keep_in':
                clearance = -signed_d - safe_distance
            else:
                clearance = signed_d - safe_distance

            band_excess = ca.fmax(warning_band - clearance, 0.0) / warning_band
            warn = near_weight * ca.log(1.0 + ca.power(band_excess, 2))
            viol = ca.fmax(-clearance, 0.0) / logbarrier_eps
            return warn + weight * ca.power(viol, 2)

        return penalty_state_casadi

    def make_penalty_belief_casadi(self, *, kappa: float = 1.0):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('CasADi is not available for no-go-zone cost') from exc

        if (not self.prisms) or self.weight <= 0.0:
            def zero_penalty(_m, _S):
                return 0.0
            return zero_penalty

        xmins = ca.DM(self._xmins)
        xmaxs = ca.DM(self._xmaxs)
        ymins = ca.DM(self._ymins)
        ymaxs = ca.DM(self._ymaxs)
        weight = float(self.weight)
        safe_distance = float(self.safe_distance)
        logbarrier_eps = float(self.logbarrier_eps)
        warning_band = float(self.warning_band)
        near_weight = float(self.near_weight)
        kappa = max(float(kappa), 1e-6)
        mode = self.mode

        def chol_2x2(M, eps=1e-9):
            M = 0.5 * (M + M.T)
            a = ca.fmax(M[0, 0], eps)
            l11 = ca.sqrt(a)
            l21 = M[1, 0] / l11
            diag22 = ca.fmax(M[1, 1] - l21 * l21, eps)
            l22 = ca.sqrt(diag22)
            return ca.vertcat(
                ca.horzcat(l11, 0.0),
                ca.horzcat(l21, l22),
            )

        def signed_distance_xy(x, y):
            if mode == 'keep_in' and self.union_boundary_segments:
                q = ca.vertcat(x, y)
                dists = []
                for p1, p2 in self.union_boundary_segments:
                    p1_dm = ca.DM(p1)
                    p2_dm = ca.DM(p2)
                    v = p2_dm - p1_dm
                    w = q - p1_dm
                    v_len_sq = ca.sumsqr(v)
                    t = ca.if_else(v_len_sq < 1e-9, 0.0, ca.fmin(ca.fmax(ca.dot(w, v) / v_len_sq, 0.0), 1.0))
                    closest = p1_dm + t * v
                    dists.append(ca.norm_2(q - closest))
                min_dist = ca.mmin(ca.vertcat(*dists))
                
                is_inside = False
                for p in self.prisms:
                    dx = ca.fmax(ca.fmax(p.xmin - x, 0.0), x - p.xmax)
                    dy = ca.fmax(ca.fmax(p.ymin - y, 0.0), y - p.ymax)
                    is_inside = ca.logic_or(is_inside, ca.logic_and(dx <= 0.0, dy <= 0.0))
                return ca.if_else(is_inside, -min_dist, min_dist)
            else:
                dx = ca.fmax(ca.fmax(xmins - x, 0.0), x - xmaxs)
                dy = ca.fmax(ca.fmax(ymins - y, 0.0), y - ymaxs)
                outside = ca.sqrt(ca.power(dx, 2) + ca.power(dy, 2))

                inside_x = ca.fmin(x - xmins, xmaxs - x)
                inside_y = ca.fmin(y - ymins, ymaxs - y)
                inside_depth = ca.fmin(inside_x, inside_y)
                inside = ca.logic_and(dx <= 0.0, dy <= 0.0)
                signed = ca.if_else(inside, -inside_depth, outside)
                return ca.mmin(signed)

        def penalty_xy(x, y):
            signed_d = signed_distance_xy(x, y)
            if mode == 'keep_in':
                clearance = -signed_d - safe_distance
            else:
                clearance = signed_d - safe_distance

            band_excess = ca.fmax(warning_band - clearance, 0.0) / warning_band
            warn = near_weight * ca.log(1.0 + ca.power(band_excess, 2))
            viol = ca.fmax(-clearance, 0.0) / logbarrier_eps
            return warn + weight * ca.power(viol, 2)

        def penalty_belief_casadi(m, S):
            if mode == 'keep_in':
                signed_d = signed_distance_xy(m[0], m[1])
                cov_xy = 0.5 * (S[:2, :2] + S[:2, :2].T)
                trace = cov_xy[0, 0] + cov_xy[1, 1]
                det = cov_xy[0, 0] * cov_xy[1, 1] - cov_xy[0, 1] * cov_xy[1, 0]
                disc = ca.sqrt(ca.fmax(ca.power(trace, 2) - 4.0 * det, 0.0))
                lambda_max = ca.fmax(0.5 * (trace + disc), 0.0)
                sigma_margin = kappa * ca.sqrt(lambda_max + 1e-9)
                clearance = -signed_d - safe_distance - sigma_margin
                band_excess = ca.fmax(warning_band - clearance, 0.0) / warning_band
                warn = near_weight * ca.log(1.0 + ca.power(band_excess, 2))
                viol = ca.fmax(-clearance, 0.0) / logbarrier_eps
                return warn + weight * ca.power(viol, 2)

            mean_xy = ca.reshape(m[:2], 2, 1)
            cov_xy = 0.5 * (S[:2, :2] + S[:2, :2].T)
            spread = math.sqrt(2.0 + kappa) * chol_2x2(cov_xy + 1e-9 * ca.DM.eye(2))
            sigma_points = (
                mean_xy,
                mean_xy + spread[:, 0],
                mean_xy - spread[:, 0],
                mean_xy + spread[:, 1],
                mean_xy - spread[:, 1],
            )
            weights = (
                kappa / (2.0 + kappa),
                1.0 / (2.0 * (2.0 + kappa)),
                1.0 / (2.0 * (2.0 + kappa)),
                1.0 / (2.0 * (2.0 + kappa)),
                1.0 / (2.0 * (2.0 + kappa)),
            )
            total = 0
            for sigma_xy, sigma_weight in zip(sigma_points, weights):
                total += float(sigma_weight) * penalty_xy(sigma_xy[0], sigma_xy[1])
            return total

        return penalty_belief_casadi

    def make_signed_distance_state_casadi(self):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('CasADi is not available for no-go-zone signed distance') from exc

        if not self.prisms:
            def inf_distance(_m):
                return ca.DM.inf()
            return inf_distance

        xmins = ca.DM(self._xmins)
        xmaxs = ca.DM(self._xmaxs)
        ymins = ca.DM(self._ymins)
        ymaxs = ca.DM(self._ymaxs)
        mode = self.mode

        def signed_distance_xy(x, y):
            if mode == 'keep_in' and self.union_boundary_segments:
                q = ca.vertcat(x, y)
                dists = []
                for p1, p2 in self.union_boundary_segments:
                    p1_dm = ca.DM(p1)
                    p2_dm = ca.DM(p2)
                    v = p2_dm - p1_dm
                    w = q - p1_dm
                    v_len_sq = ca.sumsqr(v)
                    t = ca.if_else(v_len_sq < 1e-9, 0.0, ca.fmin(ca.fmax(ca.dot(w, v) / v_len_sq, 0.0), 1.0))
                    closest = p1_dm + t * v
                    dists.append(ca.norm_2(q - closest))
                min_dist = ca.mmin(ca.vertcat(*dists))
                
                is_inside = False
                for p in self.prisms:
                    dx = ca.fmax(ca.fmax(p.xmin - x, 0.0), x - p.xmax)
                    dy = ca.fmax(ca.fmax(p.ymin - y, 0.0), y - p.ymax)
                    is_inside = ca.logic_or(is_inside, ca.logic_and(dx <= 0.0, dy <= 0.0))
                return ca.if_else(is_inside, -min_dist, min_dist)
            else:
                dx = ca.fmax(ca.fmax(xmins - x, 0.0), x - xmaxs)
                dy = ca.fmax(ca.fmax(ymins - y, 0.0), y - ymaxs)
                outside = ca.sqrt(ca.power(dx, 2) + ca.power(dy, 2))

                inside_x = ca.fmin(x - xmins, xmaxs - x)
                inside_y = ca.fmin(y - ymins, ymaxs - y)
                inside_depth = ca.fmin(inside_x, inside_y)
                inside = ca.logic_and(dx <= 0.0, dy <= 0.0)
                signed = ca.if_else(inside, -inside_depth, outside)
                return ca.mmin(signed)

        def signed_distance_state_casadi(m):
            signed_d = signed_distance_xy(m[0], m[1])
            if mode == 'keep_in':
                return -signed_d
            return signed_d

        return signed_distance_state_casadi
