"""P4 GP unit tests: [0,1], degenerate handling, protocol conformance, two-stage product."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reliability.observation_baselines import leave_one_route_out
from reliability.observation_gp import ObservabilityGP, TwoStageGP


def _spatial_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for route in ("r1", "r2", "r3"):
        for run in range(2):
            for _ in range(120):
                x = rng.uniform(-5, 4); y = rng.uniform(-2.5, 4)
                p = 1.0 / (1.0 + np.exp(0.6 * (np.hypot(x, y + 5.5) - 6)))
                det = int(rng.uniform() < p)
                rows.append({"state_x": x, "state_y": y, "route_id": route, "run_id": f"{route}/{run}",
                             "detection_label": det, "quality_label": det if rng.uniform() < 0.99 else 0,
                             "usable_label": det if rng.uniform() < 0.99 else 0})
    return pd.DataFrame(rows)


def test_gp_predictions_in_unit_interval():
    df = _spatial_df()
    xy = df[["state_x", "state_y"]].to_numpy()
    m = ObservabilityGP(length_scale=1.0).fit(xy, df["detection_label"].to_numpy().astype(float))
    p = m.predict_proba(xy)
    assert p.shape == (len(xy),)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_gp_degenerate_constant_label():
    xy = np.random.rand(200, 2)
    m = ObservabilityGP().fit(xy, np.ones(200))
    p = m.predict_proba(xy)
    assert np.allclose(p, 1.0 - 1e-4, atol=1e-3)


def test_gp_recovers_spatial_gradient():
    # positives near camera at (0,-5.5); GP prob should be higher close than far
    df = _spatial_df()
    xy = df[["state_x", "state_y"]].to_numpy()
    m = ObservabilityGP(length_scale=1.0).fit(xy, df["detection_label"].to_numpy().astype(float))
    near = m.predict_proba(np.array([[0.0, -2.0]]))[0]
    far = m.predict_proba(np.array([[3.5, 3.5]]))[0]
    assert near > far


def test_gp_plugs_into_loro_harness():
    df = _spatial_df()
    ev = leave_one_route_out(df, lambda: ObservabilityGP(length_scale=1.0), "detection_label")
    assert not np.isnan(ev["oof_p"]).any()
    assert 0.0 <= ev["pooled"]["brier"] <= 1.0


def test_two_stage_product_equals_components():
    df = _spatial_df()
    xy = df[["state_x", "state_y"]].to_numpy()
    m = TwoStageGP(length_scale=1.0).fit(
        xy, df["detection_label"].to_numpy().astype(float), df["quality_label"].to_numpy().astype(float))
    comp = m.predict_components(xy)
    prod = m.predict_proba(xy)
    assert np.allclose(prod, np.clip(comp["p_det"] * comp["p_qual"], 1e-4, 1 - 1e-4))
