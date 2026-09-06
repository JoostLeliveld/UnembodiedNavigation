"""Learned observability GP (P4): direct p_use / p_det and two-stage p_det*p_qual.

Reuses the canonical belief-aware GP (`scripts/visibility_comparison/fit_belief_aware_gp.py`,
per CLAUDE.md — import it, do not reimplement): aggregate events on a spatial grid, apply a
symmetric Beta prior, then fit a latent RBF GP in logit space. Predictions are the sigmoid of
the latent mean. The model conforms to the `ObservabilityBaseline` protocol so it plugs into the
P3 leave-one-route-out harness unchanged.

State is the operational belief (x, y); no Gazebo GT enters fitting or prediction. The
belief-aware (expected-kernel) variant is available but off by default because the exporter
currently records no per-sample belief covariance (a documented model-only extension).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
import sys

import numpy as np


def _fbag():
    root = str(pathlib.Path(__file__).resolve().parents[3] / "scripts" / "visibility_comparison")
    if root not in sys.path:
        sys.path.insert(0, root)
    import fit_belief_aware_gp as fbag  # canonical GP; do not reimplement
    return fbag


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@dataclass
class ObservabilityGP:
    """Latent RBF GP over aggregated, Beta-smoothed binary observability labels."""

    name: str = "GP_direct"
    resolution_m: float = 0.5
    length_scale: float = 1.0
    noise_var: float = 0.05
    pseudocount: float = 2.0
    belief_aware: bool = False

    _Xb: np.ndarray = field(default=None, repr=False)
    _yb: np.ndarray = field(default=None, repr=False)
    _covb: np.ndarray = field(default=None, repr=False)
    _alpha: np.ndarray = field(default=None, repr=False)
    _const: float = field(default=0.5, repr=False)
    _degenerate: bool = field(default=False, repr=False)

    def fit(self, xy: np.ndarray, y: np.ndarray, cov: np.ndarray | None = None) -> "ObservabilityGP":
        fbag = _fbag()
        xy = np.asarray(xy, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) < 2:
            self._degenerate = True
            self._const = float(np.mean(y)) if len(y) else 0.5
            return self
        self._degenerate = False
        n = len(xy)
        cov_arr = np.asarray(cov, dtype=float) if cov is not None else np.zeros((n, 2, 2), dtype=float)
        data = fbag.EventData(X=xy, y=y, cov=cov_arr, run_ids=np.zeros(n), rows_used=n, target_id=self.name)
        agg = fbag._aggregate_events(data, resolution_m=self.resolution_m, max_bin_weight=1.0e9)
        agg = fbag._smooth_binary_aggregate(agg, total_pseudocount=self.pseudocount)
        self._Xb = agg.X
        self._yb = np.clip(agg.y, 1e-4, 1.0 - 1e-4)
        self._covb = agg.cov
        self._alpha = self.noise_var / np.clip(agg.count, 1.0, None)
        self._predictor = None if self.belief_aware else fbag._fit_latent_gp_model(
            self._Xb, fbag._logit(self._yb), self._alpha, length_scale=self.length_scale)
        return self

    def predict_proba(self, xy: np.ndarray) -> np.ndarray:
        mu, _sigma = self.predict_latent(xy)
        return np.clip(_sigmoid(mu), 1e-4, 1.0 - 1e-4)

    def predict_latent(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Retain map-function uncertainty; this is not camera measurement R.

        The legacy probability remains sigmoid(mu). Callers that integrate the
        link can compare that approximation without changing existing behavior.
        """
        xy = np.asarray(xy, dtype=float)
        if self._degenerate:
            p = float(np.clip(self._const, 1e-4, 1.0 - 1e-4))
            return np.full(len(xy), np.log(p/(1-p))), np.zeros(len(xy))
        fbag = _fbag()
        if self.belief_aware:
            mu, sigma, _jit = fbag._fit_predict_expected_kernel_gp(
                self._Xb, self._yb, self._covb, self._alpha, xy,
                query_cov=None, length_scale=self.length_scale,
            )
        else:
            predictor = getattr(self, "_predictor", None)
            if predictor is None:  # compatibility with previously serialized wrappers
                mu, sigma = fbag._fit_predict_gp(
                    self._Xb, self._yb, self._alpha, xy, length_scale=self.length_scale)
            else:
                mu, sigma = predictor.predict(xy, return_std=True)
        return mu, sigma


@dataclass
class TwoStageGP:
    """p_use_product(s) = p_det(s) * p_qual(s), each a latent RBF GP.

    p_qual is trained only on detection_label==1 rows (conditional). On a corpus where quality
    is saturated the product collapses to p_det — a diagnostic, not a defect.
    """

    name: str = "GP_two_stage_product"
    resolution_m: float = 0.5
    length_scale: float = 1.0
    noise_var: float = 0.05
    pseudocount: float = 2.0
    _det: ObservabilityGP = field(default=None, repr=False)
    _qual: ObservabilityGP = field(default=None, repr=False)

    def fit(self, xy: np.ndarray, det: np.ndarray, qual: np.ndarray) -> "TwoStageGP":
        xy = np.asarray(xy, dtype=float)
        det = np.asarray(det, dtype=float)
        qual = np.asarray(qual, dtype=float)
        kw = dict(resolution_m=self.resolution_m, length_scale=self.length_scale,
                  noise_var=self.noise_var, pseudocount=self.pseudocount)
        self._det = ObservabilityGP(name="GP_p_det", **kw).fit(xy, det)
        mask = det == 1
        self._qual = ObservabilityGP(name="GP_p_qual", **kw).fit(xy[mask], qual[mask])
        return self

    def predict_proba(self, xy: np.ndarray) -> np.ndarray:
        p_det = self._det.predict_proba(xy)
        p_qual = self._qual.predict_proba(xy)
        return np.clip(p_det * p_qual, 1e-4, 1.0 - 1e-4)

    def predict_components(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        return {"p_det": self._det.predict_proba(xy), "p_qual": self._qual.predict_proba(xy)}
