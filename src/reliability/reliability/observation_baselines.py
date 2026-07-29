"""Per-camera observability baselines (P3) + leave-one-route-out evaluation.

These are the pre-GP baselines the learned model (P4) must beat or meaningfully complement.
All predict a probability in [0, 1] for a binary target (``detection_label`` or
``usable_label``) from the operational state ``s = (x, y)`` — never GT.

    B0  GlobalConstant        p(s) = training-set positive rate
    B1  DistanceLogistic      logistic on ground-plane distance to the camera
    B2  FovRangeLogistic      logistic on calibration FOV membership + range features
    B3  GridFrequency         smoothed empirical rate on a spatial grid (sparse cells flagged)

Camera calibration is the warehouse_aws external_camera from the world SDF
(pose 0 -5.5 4.8, pitch 0.92, yaw 1.5708, 90 deg HFOV, 1280x720). Calibration is an
operational asset (known), not GT.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np

# warehouse_aws external_camera (src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf:174)
AWS_CAM_POS = (0.0, -5.5, 4.8)
AWS_CAM_FOV_H_RAD = 1.5708
AWS_IMG_W, AWS_IMG_H = 1280, 720


def aws_camera_model():
    """ObliqueCameraModel for the warehouse_aws external camera (look_at from ground hit)."""
    from unav_common.camera_model import ObliqueCameraModel

    # optical axis from Gazebo (roll=0, pitch=0.92, yaw=1.5708) applied to body +X, then
    # intersect the ground plane z=0 to get look_at.
    roll, pitch, yaw = 0.0, 0.92, 1.5708
    dx = math.cos(yaw) * math.cos(pitch)
    dy = math.sin(yaw) * math.cos(pitch)
    dz = -math.sin(pitch)
    cx, cy, cz = AWS_CAM_POS
    t = cz / max(-dz, 1e-6)
    look_at = (cx + t * dx, cy + t * dy, 0.0)
    return ObliqueCameraModel(
        cam_pos=AWS_CAM_POS, look_at=look_at,
        img_width=AWS_IMG_W, img_height=AWS_IMG_H, fov_h_rad=AWS_CAM_FOV_H_RAD,
    )


class ObservabilityBaseline(Protocol):
    name: str

    def fit(self, xy: np.ndarray, y: np.ndarray) -> "ObservabilityBaseline": ...
    def predict_proba(self, xy: np.ndarray) -> np.ndarray: ...


def _clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-4, 1.0 - 1e-4)


@dataclass
class GlobalConstant:
    name: str = "B0_constant"
    _rate: float = 0.5

    def fit(self, xy: np.ndarray, y: np.ndarray) -> "GlobalConstant":
        self._rate = float(np.mean(y)) if len(y) else 0.5
        return self

    def predict_proba(self, xy: np.ndarray) -> np.ndarray:
        return _clip01(np.full(len(xy), self._rate))


@dataclass
class DistanceLogistic:
    name: str = "B1_distance_logistic"
    cam_xy: tuple[float, float] = (AWS_CAM_POS[0], AWS_CAM_POS[1])

    def __post_init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        self._clf = LogisticRegression(max_iter=1000)
        self._fitted = False

    def _features(self, xy: np.ndarray) -> np.ndarray:
        d = np.hypot(xy[:, 0] - self.cam_xy[0], xy[:, 1] - self.cam_xy[1])
        return np.column_stack([d, d * d])

    def fit(self, xy: np.ndarray, y: np.ndarray) -> "DistanceLogistic":
        if len(np.unique(y)) < 2:
            self._const = float(np.mean(y))
            self._fitted = False
            return self
        self._clf.fit(self._features(xy), y)
        self._fitted = True
        return self

    def predict_proba(self, xy: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return _clip01(np.full(len(xy), getattr(self, "_const", 0.5)))
        return _clip01(self._clf.predict_proba(self._features(xy))[:, 1])


@dataclass
class FovRangeLogistic:
    name: str = "B2_fov_range_logistic"

    def __post_init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        self._cam = aws_camera_model()
        self._clf = LogisticRegression(max_iter=1000)
        self._fitted = False

    def _features(self, xy: np.ndarray) -> np.ndarray:
        rows = []
        cx, cy, cz = AWS_CAM_POS
        for x, y in xy:
            u, v, visible = self._cam.world_to_pixel(float(x), float(y), 0.0)
            rng = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + cz ** 2)
            rows.append([1.0 if visible else 0.0, rng, 1.0 / max(rng, 1e-3)])
        return np.asarray(rows, dtype=float)

    def fit(self, xy: np.ndarray, y: np.ndarray) -> "FovRangeLogistic":
        if len(np.unique(y)) < 2:
            self._const = float(np.mean(y))
            self._fitted = False
            return self
        self._clf.fit(self._features(xy), y)
        self._fitted = True
        return self

    def predict_proba(self, xy: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return _clip01(np.full(len(xy), getattr(self, "_const", 0.5)))
        return _clip01(self._clf.predict_proba(self._features(xy))[:, 1])


@dataclass
class GridFrequency:
    """Smoothed empirical rate on a spatial grid. Beta(a,b) (Laplace-like) smoothing;
    cells with < ``min_count`` training samples fall back to the global rate and are flagged."""

    name: str = "B3_grid_frequency"
    cell_m: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0
    min_count: int = 5

    def fit(self, xy: np.ndarray, y: np.ndarray) -> "GridFrequency":
        self._x0, self._y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
        self._global = float(np.mean(y)) if len(y) else 0.5
        ix = np.floor((xy[:, 0] - self._x0) / self.cell_m).astype(int)
        iy = np.floor((xy[:, 1] - self._y0) / self.cell_m).astype(int)
        self._pos: dict[tuple[int, int], float] = {}
        self._cnt: dict[tuple[int, int], int] = {}
        for cx, cy, yi in zip(ix, iy, y):
            key = (int(cx), int(cy))
            self._pos[key] = self._pos.get(key, 0.0) + float(yi)
            self._cnt[key] = self._cnt.get(key, 0) + 1
        return self

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor((x - self._x0) / self.cell_m)), int(math.floor((y - self._y0) / self.cell_m)))

    def predict_proba(self, xy: np.ndarray) -> np.ndarray:
        out = np.empty(len(xy))
        for i, (x, y) in enumerate(xy):
            key = self._cell(float(x), float(y))
            n = self._cnt.get(key, 0)
            if n < self.min_count:
                out[i] = self._global
            else:
                pos = self._pos.get(key, 0.0)
                out[i] = (pos + self.alpha) / (n + self.alpha + self.beta)
        return _clip01(out)

    def sparse_fraction(self, xy: np.ndarray) -> float:
        if not len(xy):
            return float("nan")
        sparse = sum(self._cnt.get(self._cell(float(x), float(y)), 0) < self.min_count for x, y in xy)
        return sparse / len(xy)


def make_baselines() -> list[ObservabilityBaseline]:
    return [GlobalConstant(), DistanceLogistic(), FovRangeLogistic(), GridFrequency()]


# ----------------------------------------------------------------------------------------
# Leave-one-route-out evaluation
# ----------------------------------------------------------------------------------------

def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    import sys, pathlib

    root = str(pathlib.Path(__file__).resolve().parents[3] / "scripts" / "shared")
    if root not in sys.path:
        sys.path.insert(0, root)
    import metrics as M  # canonical scoring; never hand-rolled

    out = {
        "brier": M.brier(y, p),
        "nll": M.logloss(y, p),
        "ece": M.ece(y, p),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = M.auroc(y, p)
        out["auprc"] = M.auprc(y, p)
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def leave_one_route_out(
    df, baseline_factory, target: str
) -> dict[str, Any]:
    """Fit on all-but-one route, predict held-out route; pool predictions over folds.

    ``df`` needs columns state_x, state_y, route_id, run_id, and ``target`` (0/1).
    Returns pooled metrics + per-route metrics + out-of-fold predictions.
    """
    routes = sorted(df["route_id"].unique())
    oof_y = np.full(len(df), np.nan)
    oof_p = np.full(len(df), np.nan)
    per_route: dict[str, dict[str, float]] = {}
    idx = {r: np.where(df["route_id"].to_numpy() == r)[0] for r in routes}

    for held in routes:
        train_mask = df["route_id"].to_numpy() != held
        test_idx = idx[held]
        xy_tr = df.loc[train_mask, ["state_x", "state_y"]].to_numpy()
        y_tr = df.loc[train_mask, target].to_numpy().astype(float)
        xy_te = df.iloc[test_idx][["state_x", "state_y"]].to_numpy()
        y_te = df.iloc[test_idx][target].to_numpy().astype(float)
        model = baseline_factory().fit(xy_tr, y_tr)
        p_te = model.predict_proba(xy_te)
        oof_y[test_idx] = y_te
        oof_p[test_idx] = p_te
        per_route[held] = _metrics(y_te, p_te)

    valid = ~np.isnan(oof_p)
    pooled = _metrics(oof_y[valid], oof_p[valid])
    return {"pooled": pooled, "per_route": per_route, "oof_y": oof_y, "oof_p": oof_p}


def bootstrap_ci_by_run(df, oof_y, oof_p, metric: str = "brier", n_boot: int = 400, seed: int = 0) -> dict[str, float]:
    """Bootstrap a metric resampling whole RUNS (not frames)."""
    import sys, pathlib

    root = str(pathlib.Path(__file__).resolve().parents[3] / "scripts" / "shared")
    if root not in sys.path:
        sys.path.insert(0, root)
    import metrics as M

    fn = {"brier": M.brier, "nll": M.logloss}[metric]
    rng = np.random.default_rng(seed)
    runs = df["run_id"].to_numpy()
    unique_runs = np.array(sorted(set(runs)))
    run_to_rows = {r: np.where((runs == r) & ~np.isnan(oof_p))[0] for r in unique_runs}
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(unique_runs, size=len(unique_runs), replace=True)
        rows = np.concatenate([run_to_rows[r] for r in picked])
        if len(rows) == 0:
            continue
        vals.append(fn(oof_y[rows], oof_p[rows]))
    vals = np.array(vals)
    return {"mean": float(vals.mean()), "lo95": float(np.percentile(vals, 2.5)), "hi95": float(np.percentile(vals, 97.5))}
