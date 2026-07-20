"""Empirical GP visibility-map model loaded from an offline artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import scipy.interpolate


def _clip_prob(p, eps: float):
    return np.clip(p, eps, 1.0 - eps)


@dataclass
class GPVisibilityMapConfig:
    artifact_path: str = ""
    camera_pos: tuple[float, float, float] = (-3.0, -3.0, 6.0)
    target_height_m: float = 0.0
    min_prob: float = 1e-4


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
                if "P_conservative_plan_map" not in data.files:
                    raise RuntimeError(
                        f"Empirical GP visibility artifact {self.artifact_path} uses an outdated schema "
                        "(missing P_conservative_plan_map). Re-fit the visibility GPs."
                    )
                p_cons = np.asarray(data["P_conservative_plan_map"], dtype=float)
                p_mean = np.asarray(data["P_mean_map"], dtype=float)
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
            ("P_mean_map", p_mean),
            ("P_conservative_plan_map", p_cons),
        ):
            if grid.shape != expected_shape:
                raise RuntimeError(
                    f"Empirical GP visibility artifact {self.artifact_path} has {field_name} shape {grid.shape}, "
                    f"expected {expected_shape}."
                )

        self.xs = xs
        self.ys = ys
        self.P_mean_map = _clip_prob(p_mean, self.min_prob).astype(float)
        self.P_conservative_plan_map = _clip_prob(p_cons, self.min_prob).astype(float)
        self.camera_pos = np.asarray(camera_pos, dtype=float).reshape(3)
        self.target_height = float(target_height)
        self.x_min = float(self.xs[0])
        self.x_max = float(self.xs[-1])
        self.y_min = float(self.ys[0])
        self.y_max = float(self.ys[-1])
        self._prob_interp = scipy.interpolate.RegularGridInterpolator(
            (self.ys, self.xs), 
            self.P_conservative_plan_map, 
            method='linear', 
            bounds_error=True,
            fill_value=None,
        )

    def contains_xy_np(self, x: float, y: float) -> bool:
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        eps = 1e-9
        return bool(
            self.x_min - eps <= float(x) <= self.x_max + eps
            and self.y_min - eps <= float(y) <= self.y_max + eps
        )

    def _require_xy_in_support_np(self, x: float, y: float) -> tuple[float, float]:
        x = float(x)
        y = float(y)
        if not self.contains_xy_np(x, y):
            raise RuntimeError(
                "GP visibility query outside artifact support: "
                f"x={x:.3f}, y={y:.3f}, "
                f"support=[{self.x_min:.3f},{self.x_max:.3f}]x"
                f"[{self.y_min:.3f},{self.y_max:.3f}]."
            )
        return x, y

    @property
    def signature(self) -> tuple:
        stat = self.artifact_path.stat()
        digest = hashlib.sha256(str(self.artifact_path).encode("utf-8")).hexdigest()[:12]
        return (
            "empirical_gp_visibility",
            digest,
            str(self.artifact_path),
            int(stat.st_size),
            int(self.xs.size),
            int(self.ys.size),
            round(float(np.mean(self.P_conservative_plan_map)), 6),
            round(float(self.xs[0]), 6),
            round(float(self.xs[-1]), 6),
            round(float(self.ys[0]), 6),
            round(float(self.ys[-1]), 6),
        )

    def prob_state_np(self, m) -> float:
        if len(m) < 2:
            raise ValueError(f"State vector m must have length >= 2, got {len(m)}")
        x, y = self._require_xy_in_support_np(float(m[0]), float(m[1]))
        p = self._prob_interp((y, x))
        return float(_clip_prob(p, self.min_prob))

    def make_prob_state_casadi(self):
        try:
            import casadi as ca
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CasADi is not available for visibility model") from exc

        if self._prob_state_casadi is not None:
            return self._prob_state_casadi

        values = np.asarray(self.P_conservative_plan_map.T, dtype=float).ravel(order="F")
        interp = ca.interpolant(
            f"empirical_gp_visibility_{hashlib.sha1(str(self.artifact_path).encode('utf-8')).hexdigest()[:10]}",
            "linear",
            [self.xs.tolist(), self.ys.tolist()],
            values,
        )
        eps = float(self.min_prob)
        x_min = float(self.x_min)
        x_max = float(self.x_max)
        y_min = float(self.y_min)
        y_max = float(self.y_max)

        def p_vis_ca(m):
            x = ca.fmin(ca.fmax(m[0], x_min), x_max)
            y = ca.fmin(ca.fmax(m[1], y_min), y_max)
            z = interp(ca.vertcat(x, y))
            return ca.fmin(ca.fmax(z, eps), 1.0 - eps)

        self._prob_state_casadi = p_vis_ca
        return p_vis_ca
