from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
# Evaluators are STUDY code (firewall quarantines reliability.evaluation from the
# planner-facing package), so they live under the study tools/ directory.
sys.path.insert(0, str(ROOT / "experiments" / "multicamera_fusion_extension" / "tools"))

from experiment_evaluators import (  # noqa: E402
    NOMINAL_COVERAGE,
    compare_covariance_methods,
    evaluate_covariance_calibration,
    evaluate_reliability_prediction,
    reliability_prediction_gate,
)


# True residual covariance R = [[4, 1], [1, 2]] (SPD); Cholesky L below.
_R_TRUE = ((4.0, 1.0), (1.0, 2.0))
_L00 = 2.0
_L10 = 0.5
_L11 = math.sqrt(2.0 - 0.25)


def _sample_residuals(n: int, *, seed: int, bias=(0.0, 0.0)):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        z0 = rng.gauss(0.0, 1.0)
        z1 = rng.gauss(0.0, 1.0)
        ex = _L00 * z0 + bias[0]
        ey = _L10 * z0 + _L11 * z1 + bias[1]
        out.append((ex, ey))
    return out


def _scaled(matrix, s):
    return tuple(tuple(s * v for v in row) for row in matrix)


def test_matched_covariance_hits_nominal_coverage_and_zero_bias() -> None:
    res = _sample_residuals(5000, seed=1)
    cov = [_R_TRUE] * len(res)
    r = evaluate_covariance_calibration("matched", res, cov)
    # eᵀR⁻¹e ~ χ²₂ when R is correct → ~95% coverage at the 95% quantile.
    assert r.coverage_c95 == pytest.approx(NOMINAL_COVERAGE, abs=0.02)
    assert r.bias_x == pytest.approx(0.0, abs=0.06)
    assert r.bias_y == pytest.approx(0.0, abs=0.06)
    # sharpness = mean log|R| = log det R = log 7.
    assert r.sharpness == pytest.approx(math.log(7.0), abs=1e-9)
    assert r.n == 5000


def test_overconfident_covariance_undercovers() -> None:
    res = _sample_residuals(5000, seed=2)
    tight = [_scaled(_R_TRUE, 0.25)] * len(res)
    matched = [_R_TRUE] * len(res)
    tight_r = evaluate_covariance_calibration("tight", res, tight)
    matched_r = evaluate_covariance_calibration("matched", res, matched)
    assert tight_r.coverage_c95 < 0.85  # too-small R → residuals fall outside
    # Gaussian NLL is minimised at the true covariance.
    assert matched_r.mnll < tight_r.mnll


def test_overinflated_covariance_flagged_by_comparison() -> None:
    res = _sample_residuals(4000, seed=3)
    matched = evaluate_covariance_calibration("matched", res, [_R_TRUE] * len(res))
    tight = evaluate_covariance_calibration("tight", res, [_scaled(_R_TRUE, 0.25)] * len(res))
    huge = evaluate_covariance_calibration("huge", res, [_scaled(_R_TRUE, 25.0)] * len(res))
    assert huge.coverage_c95 > 0.99          # trivially high coverage...
    assert huge.sharpness > matched.sharpness  # ...bought with useless width
    comparison = compare_covariance_methods([matched, tight, huge])
    assert "huge" in comparison.overinflated_methods
    assert "matched" not in comparison.overinflated_methods
    # MNLL still correctly prefers the matched covariance.
    assert comparison.best_mnll_method == "matched"


def test_bias_is_detected() -> None:
    res = _sample_residuals(5000, seed=4, bias=(0.5, -0.3))
    r = evaluate_covariance_calibration("biased", res, [_R_TRUE] * len(res))
    assert r.bias_x == pytest.approx(0.5, abs=0.08)
    assert r.bias_y == pytest.approx(-0.3, abs=0.08)


def test_grouped_bootstrap_ci_brackets_point_and_is_deterministic() -> None:
    res = _sample_residuals(600, seed=5)
    cov = [_R_TRUE] * len(res)
    groups = [f"run{ i % 6 }" for i in range(len(res))]
    a = evaluate_covariance_calibration("m", res, cov, groups=groups, seed=0, n_boot=500)
    b = evaluate_covariance_calibration("m", res, cov, groups=groups, seed=0, n_boot=500)
    assert a.mnll_ci_low is not None and a.mnll_ci_high is not None
    assert a.mnll_ci_low <= a.mnll <= a.mnll_ci_high
    assert (a.mnll_ci_low, a.mnll_ci_high) == (b.mnll_ci_low, b.mnll_ci_high)


def test_as_row_and_markdown_carry_coverage_with_sharpness() -> None:
    res = _sample_residuals(200, seed=6)
    r = evaluate_covariance_calibration("m", res, [_R_TRUE] * len(res))
    row = r.as_row()
    # Coverage and sharpness must both be present in every reporting row (§E2).
    assert "coverage_c95" in row and "sharpness_logdetR" in row
    md = compare_covariance_methods([r]).render_markdown()
    assert "coverage" in md and "sharpness" in md


def test_guards() -> None:
    res = _sample_residuals(4, seed=7)
    with pytest.raises(ValueError):
        evaluate_covariance_calibration("m", res, [_R_TRUE] * 3)  # length mismatch
    with pytest.raises(ValueError):
        evaluate_covariance_calibration("m", res[:1], [_R_TRUE])  # < 2 samples
    with pytest.raises(ValueError):
        evaluate_covariance_calibration("m", res, [_R_TRUE] * 4, groups=["a", "b"])
    with pytest.raises(ValueError):
        compare_covariance_methods([])


# --------------------------------------------------------------------------- #
# E1 — reliability prediction
# --------------------------------------------------------------------------- #
def _reliability_dataset(n: int, *, seed: int):
    """Bernoulli labels from a known reliability field p_true; return the
    dataset plus perfectly-informative, shrunk, and noisy predictions."""
    rng = random.Random(seed)
    p_true, y, combined, gp_only, conf_only = [], [], [], [], []
    for _ in range(n):
        pt = 0.05 + 0.9 * rng.random()  # in (0.05, 0.95), both classes appear
        label = 1.0 if rng.random() < pt else 0.0
        p_true.append(pt)
        y.append(label)
        combined.append(pt)                      # oracle-calibrated (best proper score)
        gp_only.append(0.5 * pt + 0.25)          # shrunk toward 0.5 → less informative
        noisy = pt + rng.uniform(-0.25, 0.25)    # noisy
        conf_only.append(min(0.999, max(0.001, noisy)))
    return y, combined, gp_only, conf_only


def test_reliability_prediction_combined_beats_both_and_gate_passes() -> None:
    y, combined, gp_only, conf_only = _reliability_dataset(4000, seed=11)
    c = evaluate_reliability_prediction("combined", combined, y)
    g = evaluate_reliability_prediction("gp_only", gp_only, y)
    f = evaluate_reliability_prediction("conf_only", conf_only, y)
    # A proper scoring rule is minimised by the true probabilities.
    assert c.brier < g.brier and c.brier < f.brier
    assert c.nll < g.nll and c.nll < f.nll
    assert 0.0 <= c.auroc <= 1.0 and 0.0 <= c.auprc <= 1.0
    assert set(c.false_trust) == {0.8, 0.9, 0.95}
    verdict = reliability_prediction_gate(c, g, f)
    assert verdict.passed is True


def test_reliability_prediction_gate_fails_when_no_improvement() -> None:
    y, combined, gp_only, conf_only = _reliability_dataset(1500, seed=12)
    # Combined identical to gp_only → cannot strictly beat it → gate fails.
    c = evaluate_reliability_prediction("combined", gp_only, y)
    g = evaluate_reliability_prediction("gp_only", gp_only, y)
    f = evaluate_reliability_prediction("conf_only", conf_only, y)
    verdict = reliability_prediction_gate(c, g, f)
    assert verdict.passed is False
    assert verdict.criteria["beats_gp_only_brier"] is False


def test_reliability_prediction_grouped_ci() -> None:
    y, combined, _g, _f = _reliability_dataset(600, seed=13)
    groups = [f"run{i % 6}" for i in range(len(y))]
    a = evaluate_reliability_prediction("c", combined, y, groups=groups, seed=0, n_boot=400)
    b = evaluate_reliability_prediction("c", combined, y, groups=groups, seed=0, n_boot=400)
    assert a.brier_ci_low is not None and a.brier_ci_high is not None
    assert a.brier_ci_low <= a.brier <= a.brier_ci_high
    assert (a.brier_ci_low, a.brier_ci_high) == (b.brier_ci_low, b.brier_ci_high)


def test_reliability_prediction_guards() -> None:
    y, combined, _g, _f = _reliability_dataset(10, seed=14)
    with pytest.raises(ValueError):
        evaluate_reliability_prediction("m", combined, y[:5])          # length mismatch
    with pytest.raises(ValueError):
        evaluate_reliability_prediction("m", combined[:1], y[:1])      # < 2 samples
    with pytest.raises(ValueError):
        evaluate_reliability_prediction("m", [0.5, 1.2], [0.0, 1.0])   # prob out of range
    with pytest.raises(ValueError):
        evaluate_reliability_prediction("m", [0.5, 0.5], [0.0, 0.5])   # non-binary label
    with pytest.raises(ValueError):
        evaluate_reliability_prediction("m", combined, y, groups=["a", "b"])  # misaligned
