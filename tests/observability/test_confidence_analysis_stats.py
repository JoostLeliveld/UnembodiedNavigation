"""Tests for the GT-agnostic confidence-critique statistics (§10)."""

from __future__ import annotations

import numpy as np

from reliability.confidence_analysis_stats import (
    loro_predictive_delta,
    partial_spearman,
    spearman,
    spearman_ci_by_run,
)


def test_spearman_monotonic():
    x = np.arange(50.0)
    assert abs(spearman(x, 2 * x + 1) - 1.0) < 1e-9
    assert abs(spearman(x, -x) + 1.0) < 1e-9


def test_spearman_ci_by_run_brackets_point():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = 0.7 * x + rng.normal(scale=0.5, size=300)
    runs = np.repeat(np.arange(30), 10)
    ci = spearman_ci_by_run(x, y, runs, n_boot=200)
    assert ci["lo95"] <= ci["rho"] <= ci["hi95"]
    assert ci["rho"] > 0.4


def test_partial_spearman_removes_confound():
    rng = np.random.default_rng(1)
    c = rng.normal(size=500)          # control
    x = c + 0.01 * rng.normal(size=500)  # x is basically the control
    y = c + 0.01 * rng.normal(size=500)  # y is basically the control
    # x,y strongly correlate marginally, but only through c -> partial ~ 0
    assert spearman(x, y) > 0.9
    assert abs(partial_spearman(x, y, c)) < 0.2


def test_partial_spearman_keeps_true_signal():
    rng = np.random.default_rng(2)
    c = rng.normal(size=500)
    x = rng.normal(size=500)
    y = 0.8 * x + 0.5 * c + 0.1 * rng.normal(size=500)  # genuine x->y beyond c
    assert partial_spearman(x, y, c) > 0.4


def test_loro_predictive_delta_shape():
    rng = np.random.default_rng(3)
    n = 400
    Xg = rng.normal(size=(n, 2))
    extra = rng.normal(size=(n, 1))
    y = (Xg[:, 0] + extra[:, 0] > 0).astype(float)
    Xb = np.column_stack([Xg, extra])
    groups = np.repeat(np.arange(4), n // 4)
    out = loro_predictive_delta(Xg, Xb, y, groups)
    assert set(out) == {"geometry", "geometry+confidence"}
    for v in out.values():
        assert set(v) == {"brier", "nll", "auprc"}
    # the extra informative feature should help
    assert out["geometry+confidence"]["auprc"] >= out["geometry"]["auprc"] - 0.05
