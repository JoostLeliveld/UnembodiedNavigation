"""Gate 5 + §13 sigma-point tests, driving the REAL planner adapter.

Only the p_use field differs between conditions; the adapter (GPVisibilityMapModel,
expected_visibility_ca, precision blend) is the shared, unchanged code exercised here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from planning.core.casadi_efe import expected_visibility_ca  # noqa: E402
from planning.core.visibility_gp_map import GPVisibilityMapConfig, GPVisibilityMapModel  # noqa: E402
from reliability.observation_planner_artifact import FieldGrid, write_planner_artifact  # noqa: E402

XS = np.linspace(-5.0, 3.0, 33)
YS = np.linspace(-2.5, 4.5, 29)


def _model(tmp_path, field, name="f") -> GPVisibilityMapModel:
    grid = FieldGrid(xs=XS, ys=YS)
    path = tmp_path / f"{name}.npz"
    write_planner_artifact(str(path), grid, field, camera_pos=(0.0, -5.5, 4.8),
                           source="gp", provenance={})
    return GPVisibilityMapModel(GPVisibilityMapConfig(artifact_path=str(path)))


def _expected_vis(model, x, y, cov_diag=(0.04, 0.04, 0.01)) -> float:
    ps = model.make_prob_state_casadi()
    mean = ca.DM([x, y, 0.0])
    cov = ca.DM(np.diag(cov_diag))
    return float(expected_visibility_ca(mean, cov, ps))


def _const_field(c: float) -> np.ndarray:
    return np.full((YS.size, XS.size), c)


def test_constant_field_returns_constant(tmp_path):
    model = _model(tmp_path, _const_field(0.42))
    for x, y in [(-2.0, 0.0), (1.0, 2.0), (0.0, -1.0)]:
        assert abs(_expected_vis(model, x, y) - 0.42) < 1e-3


def test_expected_visibility_in_unit_interval_and_finite(tmp_path):
    rng = np.random.default_rng(0)
    field = np.clip(rng.uniform(0, 1, size=(YS.size, XS.size)), 1e-4, 1 - 1e-4)
    model = _model(tmp_path, field)
    for _ in range(40):
        x = rng.uniform(XS[0], XS[-1]); y = rng.uniform(YS[0], YS[-1])
        p = _expected_vis(model, x, y)
        assert np.isfinite(p) and 0.0 <= p <= 1.0


def test_boundary_crossing_gives_intermediate_value(tmp_path):
    # sharp step: 0.1 for x<0, 0.9 for x>=0
    field = np.where(XS[None, :] >= 0.0, 0.9, 0.1) * np.ones((YS.size, 1))
    model = _model(tmp_path, field)
    # belief centred on the boundary with spread that straddles it
    p = _expected_vis(model, 0.0, 0.0, cov_diag=(0.5, 0.5, 0.01))
    assert 0.1 + 1e-3 < p < 0.9 - 1e-3


def test_monotone_in_field(tmp_path):
    rng = np.random.default_rng(1)
    base = np.clip(rng.uniform(0.1, 0.6, size=(YS.size, XS.size)), 1e-4, 1 - 1e-4)
    higher = np.clip(base + 0.25, 1e-4, 1 - 1e-4)
    m_lo = _model(tmp_path, base, "lo")
    m_hi = _model(tmp_path, higher, "hi")
    for _ in range(20):
        x = rng.uniform(XS[0] + 0.5, XS[-1] - 0.5); y = rng.uniform(YS[0] + 0.5, YS[-1] - 0.5)
        assert _expected_vis(m_hi, x, y) >= _expected_vis(m_lo, x, y) - 1e-6


def test_precision_blend_is_monotone_decreasing_R(tmp_path):
    # the frozen adapter: higher p_vis -> higher precision -> smaller R_plan variance
    from planning.core.casadi_efe import _blend_observation_covariance_ca

    class _P:
        R_visible = ca.DM(np.diag([2.5**2, 2.5**2]))
        R_miss = ca.DM(np.diag([120.0**2, 120.0**2]))

    r_lo = float(_blend_observation_covariance_ca(ca.DM(0.2), _P())[0, 0])
    r_hi = float(_blend_observation_covariance_ca(ca.DM(0.9), _P())[0, 0])
    assert r_hi < r_lo  # more visibility -> tighter effective covariance
