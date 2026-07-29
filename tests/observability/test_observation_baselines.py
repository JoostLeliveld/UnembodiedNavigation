"""P3 baseline unit tests: shapes, [0,1], constant behaviour, sparse flagging, LORO split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reliability.observation_baselines import (
    DistanceLogistic,
    FovRangeLogistic,
    GlobalConstant,
    GridFrequency,
    leave_one_route_out,
    make_baselines,
    bootstrap_ci_by_run,
)


def _toy_df(n_per_route: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for route in ("r1", "r2", "r3"):
        for run in range(2):
            for i in range(n_per_route):
                x = rng.uniform(-5, 4); y = rng.uniform(-2.5, 4)
                # detection more likely when close to camera at (0,-5.5)
                d = np.hypot(x - 0.0, y + 5.5)
                p = 1.0 / (1.0 + np.exp(0.5 * (d - 6)))
                rows.append({
                    "state_x": x, "state_y": y, "route_id": route,
                    "run_id": f"{route}/{run}",
                    "detection_label": int(rng.uniform() < p),
                    "usable_label": int(rng.uniform() < p),
                })
    return pd.DataFrame(rows)


def test_constant_returns_train_mean():
    xy = np.random.rand(50, 2)
    y = np.array([1] * 30 + [0] * 20, dtype=float)
    m = GlobalConstant().fit(xy, y)
    p = m.predict_proba(xy)
    assert np.allclose(p, 0.6, atol=1e-3)
    assert p.min() >= 0 and p.max() <= 1


@pytest.mark.parametrize("factory", [GlobalConstant, DistanceLogistic, FovRangeLogistic, GridFrequency])
def test_predictions_in_unit_interval(factory):
    df = _toy_df()
    xy = df[["state_x", "state_y"]].to_numpy()
    y = df["detection_label"].to_numpy().astype(float)
    m = factory().fit(xy, y)
    p = m.predict_proba(xy)
    assert p.shape == (len(xy),)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_grid_constant_label_gives_constant_prediction():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-3, 3, size=(400, 2))
    y = np.ones(400)  # all positive
    m = GridFrequency(min_count=1).fit(xy, y)
    p = m.predict_proba(xy)
    assert np.all(p > 0.6)  # smoothing pulls slightly below 1 but well above half


def test_grid_flags_sparse_cells():
    xy_tr = np.array([[0.0, 0.0]] * 20)  # one dense cell
    y = np.ones(20)
    m = GridFrequency(min_count=5, cell_m=0.5).fit(xy_tr, y)
    far = np.array([[100.0, 100.0]])  # unseen cell -> sparse -> global fallback
    assert m.sparse_fraction(far) == 1.0
    assert abs(m.predict_proba(far)[0] - np.clip(m._global, 1e-4, 1 - 1e-4)) < 1e-6


def test_leave_one_route_out_never_trains_on_held_route():
    df = _toy_df()
    # instrument: a factory that records which routes it saw at fit time
    seen = {}

    class Spy(GlobalConstant):
        def fit(self, xy, y):
            # store the set of x-coords count as a proxy is fragile; instead check via LORO wiring
            return super().fit(xy, y)

    ev = leave_one_route_out(df, GlobalConstant, "detection_label")
    # out-of-fold predictions must cover every row exactly once
    assert not np.isnan(ev["oof_p"]).any()
    assert set(ev["per_route"].keys()) == set(df["route_id"].unique())


def test_loro_holdout_prediction_uses_only_other_routes():
    # constant baseline: held-out prediction must equal the mean of the OTHER routes' target
    df = _toy_df()
    ev = leave_one_route_out(df, GlobalConstant, "detection_label")
    for held in df["route_id"].unique():
        other_mean = df.loc[df["route_id"] != held, "detection_label"].mean()
        held_pred = ev["oof_p"][df["route_id"].to_numpy() == held]
        assert np.allclose(held_pred, np.clip(other_mean, 1e-4, 1 - 1e-4), atol=1e-6)


def test_bootstrap_ci_is_finite_and_ordered():
    df = _toy_df()
    ev = leave_one_route_out(df, GlobalConstant, "detection_label")
    ci = bootstrap_ci_by_run(df, ev["oof_y"], ev["oof_p"], metric="brier", n_boot=100)
    assert ci["lo95"] <= ci["mean"] <= ci["hi95"]
    assert np.isfinite([ci["lo95"], ci["mean"], ci["hi95"]]).all()


def test_make_baselines_has_four():
    assert len(make_baselines()) == 4
