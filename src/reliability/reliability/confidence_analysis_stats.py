"""Pure statistics for the YOLO-confidence critique (§10). Array-only, GT-agnostic.

These helpers see only numeric arrays — they carry no notion of ground truth. The
evaluation-only residual data is assembled and labelled in the CLI
(`scripts/reliability/analyze_confidence.py`); nothing here reads a log or a GT column.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(np.asarray(a, dtype=float))
    return mask


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = _finite_mask(x, y)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(np.asarray(x)[m], np.asarray(y)[m]).correlation)


def spearman_ci_by_run(
    x: np.ndarray, y: np.ndarray, run_ids: Sequence, *, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Spearman rho with a 95% CI from resampling whole runs (not rows)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    runs = np.asarray(run_ids)
    point = spearman(x, y)
    uniq = np.array(sorted(set(runs.tolist())))
    rows_by_run = {r: np.where(runs == r)[0] for r in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_run[r] for r in picked])
        rho = spearman(x[idx], y[idx])
        if np.isfinite(rho):
            vals.append(rho)
    vals = np.asarray(vals)
    return {
        "rho": point,
        "lo95": float(np.percentile(vals, 2.5)) if len(vals) else float("nan"),
        "hi95": float(np.percentile(vals, 97.5)) if len(vals) else float("nan"),
        "n": int(_finite_mask(x, y).sum()),
    }


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Spearman(x, y | controls): rank-transform, linearly residualise on rank(controls),
    correlate the residuals. controls is (n, k)."""
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    m = _finite_mask(x, y) & np.all(np.isfinite(controls), axis=1)
    if m.sum() < controls.shape[1] + 3:
        return float("nan")
    rx = rankdata(np.asarray(x)[m]); ry = rankdata(np.asarray(y)[m])
    rc = np.column_stack([rankdata(controls[m, j]) for j in range(controls.shape[1])])
    rc = np.column_stack([np.ones(len(rc)), rc])  # intercept
    def _resid(target):
        beta, *_ = np.linalg.lstsq(rc, target, rcond=None)
        return target - rc @ beta
    ex, ey = _resid(rx), _resid(ry)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def loro_predictive_delta(
    X_a: np.ndarray, X_b: np.ndarray, y: np.ndarray, groups: Sequence, *, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Leave-one-group-out logistic: feature set A vs B (B usually = A + confidence).
    Returns pooled out-of-fold Brier / NLL / AUPRC for each set. GT-agnostic."""
    import sys, pathlib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    root = str(pathlib.Path(__file__).resolve().parents[3] / "scripts" / "shared")
    if root not in sys.path:
        sys.path.insert(0, root)
    import metrics as M

    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    uniq = sorted(set(groups.tolist()))

    def _oof(X):
        X = np.asarray(X, dtype=float)
        p = np.full(len(y), np.nan)
        for g in uniq:
            tr = groups != g
            te = groups == g
            if len(np.unique(y[tr])) < 2:
                p[te] = float(np.mean(y[tr]))
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        return p

    out = {}
    for name, X in (("geometry", X_a), ("geometry+confidence", X_b)):
        p = _oof(X)
        v = ~np.isnan(p)
        out[name] = {
            "brier": M.brier(y[v], p[v]),
            "nll": M.logloss(y[v], p[v]),
            "auprc": M.auprc(y[v], p[v]) if len(np.unique(y[v])) == 2 else float("nan"),
        }
    return out
