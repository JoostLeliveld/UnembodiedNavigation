from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
# scripts/geometry_visibility is a PEP-420 namespace package under scripts/, so
# putting scripts/ on the path lets `from geometry_visibility import
# geometry_visibility` resolve the offline module for the equivalence test.
sys.path.insert(0, str(ROOT / "scripts"))

from reliability.contracts import ContractValidationError, UpdateCovariance  # noqa: E402
from reliability.covariance_mapping import (  # noqa: E402
    MODE_BOUNDED,
    MODE_PRECISION,
    MappingDecision,
    MissEndpointPolicy,
    gate_decision,
    trust_to_update_covariance,
)
from reliability.single_camera_adapter import precision_blend_covariance  # noqa: E402

from geometry_visibility import geometry_visibility as gv  # noqa: E402


R_VISIBLE_UV = 2.5
R_MISS_UV = 40.0


def _reconciled_policy(gamma: float = 1.0, r_miss: float = R_MISS_UV) -> MissEndpointPolicy:
    return MissEndpointPolicy(
        r_visible_uv=R_VISIBLE_UV,
        gamma=gamma,
        reconciled=True,
        chosen_r_miss_uv=r_miss,
        provenance="test-fixture",
    )


def _r_cond(var: float = R_VISIBLE_UV ** 2) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((var, 0.0), (0.0, var))


def _trace(update: UpdateCovariance) -> float:
    m = update.matrix_px2
    return m[0][0] + m[1][1]


def _is_psd_2x2(m) -> bool:
    a, b = m[0]
    c, d = m[1]
    # symmetric within roundoff, both leading minors non-negative
    assert abs(b - c) < 1e-9
    return a >= -1e-12 and (a * d - b * c) >= -1e-9


# --------------------------------------------------------------------------- #
# Monotonicity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", [MODE_BOUNDED, MODE_PRECISION])
def test_trace_non_increasing_as_tau_rises(mode):
    policy = _reconciled_policy()
    grid = [i / 20.0 for i in range(21)]
    traces = [
        _trace(trust_to_update_covariance(t, 1.0, _r_cond(), policy, mode=mode)[0])
        for t in grid
    ]
    for lo, hi in zip(traces, traces[1:]):
        # higher tau -> equal or smaller trace (lower tau never shrinks R)
        assert hi <= lo + 1e-9, f"{mode}: trace increased as tau rose: {lo} -> {hi}"
    # and strictly larger at the low-trust end than the high-trust end
    assert traces[0] > traces[-1]


@pytest.mark.parametrize("mode", [MODE_BOUNDED, MODE_PRECISION])
def test_gamma_variants_still_monotone(mode):
    for gamma in (0.5, 1.0, 2.0):
        policy = _reconciled_policy(gamma=gamma)
        grid = [i / 10.0 for i in range(11)]
        traces = [
            _trace(trust_to_update_covariance(t, 1.0, _r_cond(), policy, mode=mode)[0])
            for t in grid
        ]
        for lo, hi in zip(traces, traces[1:]):
            assert hi <= lo + 1e-9


# --------------------------------------------------------------------------- #
# PSD + exactly 2x2 + finite endpoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", [MODE_BOUNDED, MODE_PRECISION])
@pytest.mark.parametrize("tau", [0.0, 1.0])
def test_psd_2x2_finite_endpoints(mode, tau):
    import math

    policy = _reconciled_policy()
    update, decision = trust_to_update_covariance(tau, 1.0, _r_cond(), policy, mode=mode)
    m = update.matrix_px2
    assert len(m) == 2 and len(m[0]) == 2 and len(m[1]) == 2
    for row in m:
        for value in row:
            assert math.isfinite(value)
    assert _is_psd_2x2(m)
    assert update.units == "px^2"
    assert isinstance(decision, MappingDecision)


def test_bounded_interpolation_endpoints_match_endpoints():
    # tau=1 -> R_good (conditional); tau=0 -> R_bad (miss endpoint).
    policy = _reconciled_policy()
    cond = _r_cond()
    hit, _ = trust_to_update_covariance(1.0, 1.0, cond, policy, mode=MODE_BOUNDED)
    miss, _ = trust_to_update_covariance(0.0, 1.0, cond, policy, mode=MODE_BOUNDED)
    assert hit.matrix_px2[0][0] == pytest.approx(cond[0][0], abs=1e-9)
    # miss endpoint per-axis std reaches the reconciled r_miss
    assert miss.matrix_px2[0][0] == pytest.approx(R_MISS_UV ** 2, rel=1e-9)


# --------------------------------------------------------------------------- #
# FORMULA EQUIVALENCE: adapter precision-blend == offline trust_to_r_plan
# --------------------------------------------------------------------------- #
def test_formula_equivalence_adapter_vs_geometry_visibility():
    """Same (r_visible, r_miss) -> identical variance for both code paths.

    Proves the ONLY historical divergence between the offline and runtime paths
    was the miss-endpoint constant, not the mapping math.
    """

    grid = [i / 100.0 for i in range(101)]
    for r_miss in (40.0, 120.0):  # both candidate endpoints
        for t in grid:
            adapter = precision_blend_covariance(
                t,
                r_visible_uv=R_VISIBLE_UV,
                r_miss_uv=r_miss,
                min_probability=0.0,  # match trust_to_r_plan's [0,1] clip exactly
            )
            _std, var = gv.trust_to_r_plan(t, R_VISIBLE_UV, r_miss)
            assert adapter[0][0] == pytest.approx(float(var), abs=1e-9, rel=1e-9)
            assert adapter[1][1] == pytest.approx(float(var), abs=1e-9, rel=1e-9)
            # off-diagonal is exactly zero (isotropic pixel blend)
            assert adapter[0][1] == 0.0 and adapter[1][0] == 0.0


# --------------------------------------------------------------------------- #
# require_reconciled()
# --------------------------------------------------------------------------- #
def test_require_reconciled_raises_when_unreconciled():
    policy = MissEndpointPolicy()  # defaults: reconciled=False
    with pytest.raises(ContractValidationError) as excinfo:
        policy.require_reconciled()
    msg = str(excinfo.value)
    assert "DECISION_r_miss.md" in msg
    assert "residual tails" in msg


def test_require_reconciled_returns_chosen_value():
    policy = _reconciled_policy(r_miss=77.0)
    assert policy.require_reconciled() == pytest.approx(77.0)


def test_unreconciled_policy_blocks_mapping():
    policy = MissEndpointPolicy()
    with pytest.raises(ContractValidationError):
        trust_to_update_covariance(0.5, 1.0, _r_cond(), policy)


def test_policy_validation_rejects_bad_endpoints():
    with pytest.raises(ContractValidationError):
        MissEndpointPolicy(r_visible_uv=0.0)
    with pytest.raises(ContractValidationError):
        MissEndpointPolicy(r_visible_uv=50.0, r_miss_uv_offline=40.0)  # miss < visible
    with pytest.raises(ContractValidationError):
        MissEndpointPolicy(gamma=0.0)


# --------------------------------------------------------------------------- #
# gate_decision truth table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tau,health,expected",
    [
        (1.0, 1.0, "accept"),          # both clear
        (0.4, 0.4, "accept"),          # exactly on threshold -> accept
        (0.3, 1.0, "inflate"),         # tau below tau_min but above 0.5*tau_min
        (1.0, 0.3, "inflate"),         # health below h_min but above 0.5*h_min
        (0.2, 1.0, "inflate"),         # tau exactly 0.5*tau_min -> not reject
        (0.19, 1.0, "reject"),         # tau below 0.5*tau_min -> reject
        (1.0, 0.1, "reject"),          # health collapsed
        (0.0, 0.0, "reject"),          # both collapsed
    ],
)
def test_gate_decision_truth_table(tau, health, expected):
    assert gate_decision(tau, health, tau_min=0.4, h_min=0.4) == expected


def test_gate_flows_into_update_available():
    policy = _reconciled_policy()
    reject_update, reject_decision = trust_to_update_covariance(
        0.05, 0.05, _r_cond(), policy
    )
    assert reject_decision.gate == "reject"
    assert reject_update.available is False
    accept_update, accept_decision = trust_to_update_covariance(
        0.95, 0.95, _r_cond(), policy
    )
    assert accept_decision.gate == "accept"
    assert accept_update.available is True


# --------------------------------------------------------------------------- #
# MappingDecision decomposition
# --------------------------------------------------------------------------- #
def test_mapping_decision_carries_factors_and_reason():
    policy = _reconciled_policy()
    _update, decision = trust_to_update_covariance(
        0.6,
        0.8,
        _r_cond(),
        policy,
        spatial_quality=0.7,
        calibrated_confidence=0.9,
    )
    assert decision.reason  # non-empty
    for key in ("spatial_quality", "calibrated_confidence", "health", "tau", "trace_px2"):
        assert key in decision.factors
    assert decision.factors["spatial_quality"] == pytest.approx(0.7)
    assert decision.factors["calibrated_confidence"] == pytest.approx(0.9)
    assert decision.factors["health"] == pytest.approx(0.8)
    assert decision.mode == MODE_BOUNDED
