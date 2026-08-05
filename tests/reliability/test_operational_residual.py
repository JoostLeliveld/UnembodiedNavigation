"""Tests for operational (GT-free) residuals and the state-corrected R_cond.

Covers the M3/M5 checks from ``experiments/operational_residual_rcond/PLAN.md``
that survived adaptation -- residual availability, PSD, state-uncertainty
subtraction (M5.5, the one the naive ``r r^T`` estimator fails), synthetic
recovery, no-leakage-from-misses -- plus Gate R3 (circularity) and the frame
contract, neither of which the incoming spec had.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
sys.path.insert(0, str(ROOT / "src" / "state"))

from reliability.conditional_covariance import chi2_coverage, matrix_nll, sharpness  # noqa: E402
from reliability.contracts import ContractValidationError  # noqa: E402
from reliability.operational_residual import (  # noqa: E402
    ALLOWED_FRAMES,
    OperationalResidual,
    build_operational_residuals,
    circularity_factor,
    pooled_target,
    shrink_summary,
    summarize_residuals,
    total_covariances,
)
from state.core import trajectory_smoother as ts  # noqa: E402


def _trajectory(n: int = 60, belief_var: float = 0.02**2):
    """A flat smoothed track with a fixed, isotropic belief covariance."""
    mean = np.column_stack([np.arange(n) * 0.1, np.zeros(n)])
    cov = np.repeat((belief_var * np.eye(2))[None, :, :], n, axis=0)
    return mean, cov


def _measurements(mean, indices, noise_std=0.0, bias=(0.0, 0.0), source="camera_A", seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in indices:
        z = mean[k] + np.asarray(bias, dtype=float)
        if noise_std:
            z = z + rng.normal(0.0, noise_std, 2)
        out.append(
            ts.Measurement(
                index=int(k),
                z=(float(z[0]), float(z[1])),
                covariance=((0.05**2, 0.0), (0.0, 0.05**2)),
                source=source,
            )
        )
    return out


def _residuals(noise_std=0.05, bias=(0.0, 0.0), belief_var=0.02**2, n=60, held_out=True, seed=0):
    mean, cov = _trajectory(n, belief_var=belief_var)
    indices = range(0, n, 2)
    meas = _measurements(mean, indices, noise_std=noise_std, bias=bias, seed=seed)
    anchored_by = ("camera_B",) if held_out else ("camera_A", "camera_B")
    return build_operational_residuals(
        smoothed_mean=mean,
        smoothed_cov=cov,
        measurements=meas,
        camera_id="camera_A",
        frame="xy",
        anchored_by=anchored_by,
    )


# --------------------------------------------------------------------------- #
# record construction
# --------------------------------------------------------------------------- #


def test_residual_is_measurement_minus_smoothed_prediction():
    mean, cov = _trajectory(10)
    meas = _measurements(mean, [4], bias=(0.3, -0.2))
    records = build_operational_residuals(
        smoothed_mean=mean, smoothed_cov=cov, measurements=meas,
        camera_id="camera_A", anchored_by=("camera_B",),
    )
    assert len(records) == 1
    np.testing.assert_allclose(records[0].residual, (0.3, -0.2), atol=1e-12)
    np.testing.assert_allclose(records[0].predicted, mean[4], atol=1e-12)


def test_only_the_named_camera_contributes_records():
    """M5.3-adjacent: a residual must never be attributed to the wrong camera."""
    mean, cov = _trajectory(20)
    meas = _measurements(mean, [3, 5], source="camera_A") + _measurements(
        mean, [7], source="camera_C"
    )
    records = build_operational_residuals(
        smoothed_mean=mean, smoothed_cov=cov, measurements=meas,
        camera_id="camera_A", anchored_by=("camera_B",),
    )
    assert [r.index for r in records] == [3, 5]
    assert {r.camera_id for r in records} == {"camera_A"}


def test_misses_contribute_no_residual():
    """M5.3 -- only accepted detections carry a residual target.

    A miss has no ``z``, so it cannot produce a Measurement and therefore cannot
    reach this builder. Asserted by construction: a non-finite z is refused.
    """
    with pytest.raises(ts.SmootherError):
        ts.Measurement(
            index=0, z=(float("nan"), 0.0), covariance=((1.0, 0.0), (0.0, 1.0)), source="camera_A"
        )


def test_held_out_is_derived_from_the_anchor_set_not_asserted():
    mean, cov = _trajectory(10)
    meas = _measurements(mean, [4])
    kwargs = dict(smoothed_mean=mean, smoothed_cov=cov, measurements=meas, camera_id="camera_A")
    assert build_operational_residuals(anchored_by=("camera_B",), **kwargs)[0].held_out is True
    assert build_operational_residuals(anchored_by=("camera_A", "camera_B"), **kwargs)[0].held_out is False


def test_state_projection_is_H_P_Ht():
    mean, cov = _trajectory(10, belief_var=0.09)
    records = build_operational_residuals(
        smoothed_mean=mean, smoothed_cov=cov, measurements=_measurements(mean, [2]),
        camera_id="camera_A", anchored_by=("camera_B",),
    )
    np.testing.assert_allclose(records[0].state_projection, 0.09 * np.eye(2), atol=1e-12)


def test_non_identity_jacobian_is_applied():
    mean, cov = _trajectory(10, belief_var=0.04)
    H = [[2.0, 0.0], [0.0, 3.0]]
    records = build_operational_residuals(
        smoothed_mean=mean, smoothed_cov=cov, measurements=_measurements(mean, [2]),
        camera_id="camera_A", anchored_by=("camera_B",), frame="uv",
        observation_jacobian=H,
    )
    expected = np.asarray(H) @ (0.04 * np.eye(2)) @ np.asarray(H).T
    np.testing.assert_allclose(records[0].state_projection, expected, atol=1e-12)
    assert records[0].frame == "uv"


def test_frame_must_be_registered():
    mean, cov = _trajectory(10)
    with pytest.raises(ContractValidationError):
        build_operational_residuals(
            smoothed_mean=mean, smoothed_cov=cov, measurements=_measurements(mean, [2]),
            camera_id="camera_A", frame="world_metres",
        )
    assert set(ALLOWED_FRAMES) == {"uv", "xy"}


def test_index_past_the_trajectory_is_refused():
    mean, cov = _trajectory(10)
    meas = [ts.Measurement(index=5, z=(0.0, 0.0), covariance=((1.0, 0.0), (0.0, 1.0)), source="camera_A")]
    with pytest.raises(ContractValidationError):
        build_operational_residuals(
            smoothed_mean=mean[:3], smoothed_cov=cov[:3], measurements=meas, camera_id="camera_A",
        )


# --------------------------------------------------------------------------- #
# the state-corrected estimator (M5.5 is the load-bearing one)
# --------------------------------------------------------------------------- #


def test_state_uncertainty_is_subtracted_not_absorbed():
    """M5.5 -- inflating P^s with the residuals held fixed must NOT inflate R_cond.

    This is the test the naive ``r r^T`` estimator fails, and the reason this
    module exists.
    """
    tight = summarize_residuals(_residuals(belief_var=0.01**2, seed=5))
    loose = summarize_residuals(_residuals(belief_var=0.10**2, seed=5))

    # Same residuals either way (the measurement values do not depend on P^s).
    np.testing.assert_allclose(tight.raw_second_moment, loose.raw_second_moment, atol=1e-12)
    # The naive estimator would report the same R for both; the corrected one must not.
    assert np.trace(loose.mean_state_projection) > np.trace(tight.mean_state_projection)
    assert np.trace(loose.state_corrected) < np.trace(tight.state_corrected)


def _self_consistent_residuals(cam_sigma, belief_sigma, n=4000, seed=17):
    """Residuals that actually obey ``E[r r^T] = R_cond + H P^s H^T``.

    The smoothed mean is displaced from truth by a draw from its own stated
    covariance, and the measurement is generated from *truth*. So
    ``r = z - mu^s = cam_noise - state_error`` and the second moment is the sum of
    the two -- which is the model the estimator inverts. (A fixture with an exact
    ``mu^s`` would put no state error in the residual and the subtraction would
    correctly overshoot.)
    """
    rng = np.random.default_rng(seed)
    truth = np.column_stack([np.arange(n) * 0.01, np.zeros(n)])
    mean = truth + rng.normal(0.0, belief_sigma, truth.shape)
    cov = np.repeat((belief_sigma**2 * np.eye(2))[None, :, :], n, axis=0)
    meas = [
        ts.Measurement(
            index=k,
            z=tuple(truth[k] + rng.normal(0.0, cam_sigma, 2)),
            covariance=((cam_sigma**2, 0.0), (0.0, cam_sigma**2)),
            source="camera_A",
        )
        for k in range(n)
    ]
    return build_operational_residuals(
        smoothed_mean=mean, smoothed_cov=cov, measurements=meas,
        camera_id="camera_A", frame="xy", anchored_by=("camera_B",),
    )


def test_state_corrected_recovers_the_camera_noise_it_was_given():
    """M5.4/M5.2 -- with known camera sigma and known belief, recover sigma."""
    cam_sigma, belief_sigma = 0.08, 0.03
    summary = summarize_residuals(_self_consistent_residuals(cam_sigma, belief_sigma))

    naive = np.trace(np.asarray(summary.raw_second_moment)) / 2.0
    corrected = np.trace(np.asarray(summary.state_corrected)) / 2.0
    assert corrected == pytest.approx(cam_sigma**2, rel=0.10)
    # The naive r r^T estimator overstates the camera by the belief's contribution.
    assert naive == pytest.approx(cam_sigma**2 + belief_sigma**2, rel=0.10)
    assert naive > corrected


def test_bias_is_reported_separately_and_not_folded_into_covariance():
    """Gate M5: 'do not inflate covariance to hide a deterministic bias'."""
    bias = (0.0, 0.078)  # camera C's measured lateral bias
    biased = summarize_residuals(_residuals(noise_std=0.02, bias=bias, n=2000, seed=23))
    unbiased = summarize_residuals(_residuals(noise_std=0.02, bias=(0.0, 0.0), n=2000, seed=23))

    np.testing.assert_allclose(biased.mean_residual, bias, atol=0.005)
    assert biased.bias_norm == pytest.approx(0.078, abs=0.005)
    # Covariance is about the mean, so the bias must not enlarge it.
    np.testing.assert_allclose(
        biased.state_corrected, unbiased.state_corrected, atol=1e-9
    )


def test_state_corrected_is_always_psd_and_flags_the_projection():
    """A camera sharper than the belief drives the subtraction indefinite; the
    result must still be PSD and must say that it was projected."""
    records = _residuals(noise_std=0.001, belief_var=0.20**2, n=200, seed=31)
    summary = summarize_residuals(records)
    eigenvalues = np.linalg.eigvalsh(np.asarray(summary.state_corrected))
    assert float(np.min(eigenvalues)) > 0.0
    assert summary.psd_projection_applied is True


def test_summary_refuses_mixed_provenance():
    a = _residuals(held_out=True, seed=1)
    b = _residuals(held_out=False, seed=1)
    with pytest.raises(ContractValidationError):
        summarize_residuals(a + b)  # mixed held-out
    with pytest.raises(ContractValidationError):
        summarize_residuals(a[:1])  # fewer than 2


def test_summary_refuses_mixed_cameras_and_frames():
    mean, cov = _trajectory(20)
    a = build_operational_residuals(
        smoothed_mean=mean, smoothed_cov=cov,
        measurements=_measurements(mean, [2, 4], source="camera_A"),
        camera_id="camera_A", anchored_by=("camera_B",),
    )
    c = build_operational_residuals(
        smoothed_mean=mean, smoothed_cov=cov,
        measurements=_measurements(mean, [6, 8], source="camera_C"),
        camera_id="camera_C", anchored_by=("camera_B",),
    )
    with pytest.raises(ContractValidationError):
        summarize_residuals(a + c)

    a_uv = [OperationalResidual(**{**vars(r), "frame": "uv"}) for r in a]
    with pytest.raises(ContractValidationError):
        summarize_residuals(a + a_uv)


# --------------------------------------------------------------------------- #
# Gate R3 -- circularity
# --------------------------------------------------------------------------- #


def test_circularity_factor_exceeds_one_when_the_camera_anchored_its_own_reference():
    """End-to-end through the real smoother, not a stub.

    Camera A's fixes carry noise. Anchored, the smoother absorbs some of it and
    the residual shrinks; held out, it survives. The factor reports the gap.
    """
    n = 200
    track = np.column_stack([np.arange(n) * 0.05, np.zeros(n)])
    rng = np.random.default_rng(41)
    a_idx = list(range(4, n, 4))
    b_idx = list(range(6, n, 12))
    meas_a = [
        ts.Measurement(index=k, z=tuple(track[k] + rng.normal(0, 0.10, 2)),
                       covariance=((0.10**2, 0.0), (0.0, 0.10**2)), source="camera_A")
        for k in a_idx
    ]
    meas_b = [
        ts.Measurement(index=k, z=tuple(track[k] + rng.normal(0, 0.05, 2)),
                       covariance=((0.05**2, 0.0), (0.0, 0.05**2)), source="camera_B")
        for k in b_idx
    ]
    u = ts.increments_from_track(track)

    anchored_traj = ts.smooth_trajectory(u, meas_a + meas_b, initial_mean=track[0])
    heldout_traj = ts.smooth_trajectory(
        u, ts.without_source(meas_a + meas_b, "camera_A"), initial_mean=track[0]
    )

    def summarise(traj):
        return summarize_residuals(
            build_operational_residuals(
                smoothed_mean=traj.smoothed_mean, smoothed_cov=traj.smoothed_cov,
                measurements=meas_a, camera_id="camera_A", anchored_by=traj.sources,
            )
        )

    anchored = summarise(anchored_traj)
    held_out = summarise(heldout_traj)
    assert anchored.held_out is False and held_out.held_out is True

    factor = circularity_factor(held_out, anchored)
    assert factor > 1.0, "anchored estimate should be optimistic"
    # The held-out estimate should land near the true camera sigma.
    per_axis = np.trace(np.asarray(held_out.state_corrected)) / 2.0
    assert float(np.sqrt(per_axis)) == pytest.approx(0.10, rel=0.35)


def test_circularity_factor_argument_order_is_enforced():
    held_out = summarize_residuals(_residuals(held_out=True, seed=3))
    anchored = summarize_residuals(_residuals(held_out=False, seed=3))
    with pytest.raises(ContractValidationError):
        circularity_factor(anchored, held_out)  # swapped


# --------------------------------------------------------------------------- #
# scoring interoperability
# --------------------------------------------------------------------------- #


def test_total_covariance_includes_the_state_term():
    records = _residuals(seed=7)
    R = ((0.01, 0.0), (0.0, 0.01))
    C = total_covariances(records, R)
    expected = np.asarray(R) + np.asarray(records[0].state_projection)
    np.testing.assert_allclose(C[0], expected, atol=1e-12)


def test_total_covariances_accepts_a_per_camera_mapping():
    records = _residuals(seed=7)
    with pytest.raises(ContractValidationError):
        total_covariances(records, {"camera_Z": ((0.01, 0.0), (0.0, 0.01))})
    C = total_covariances(records, {"camera_A": ((0.02, 0.0), (0.0, 0.02))})
    assert len(C) == len(records)


def test_scored_against_canonical_metrics_the_corrected_R_beats_a_wrong_one():
    """M5.2 -- held-out MNLL must prefer the right scale.

    Uses the canonical MNLL/coverage/sharpness from ``conditional_covariance``;
    these are never hand-rolled here.
    """
    cam_sigma = 0.08
    records = _residuals(noise_std=cam_sigma, belief_var=0.03**2, n=3000, seed=53)
    summary = summarize_residuals(records)
    r = [rec.residual for rec in records]

    fitted = summary.state_corrected
    too_small = ((1e-4, 0.0), (0.0, 1e-4))
    too_large = ((1.0, 0.0), (0.0, 1.0))

    nll_fitted = matrix_nll(r, total_covariances(records, fitted))
    assert nll_fitted < matrix_nll(r, total_covariances(records, too_small))
    assert nll_fitted < matrix_nll(r, total_covariances(records, too_large))

    coverage = chi2_coverage(r, total_covariances(records, fitted), q=0.95)
    assert 0.90 < coverage < 0.99
    # Sharpness is always quoted next to coverage: a huge R passes coverage trivially.
    assert sharpness(total_covariances(records, too_large)) > sharpness(
        total_covariances(records, fitted)
    )


def test_shrinkage_limits_and_pooled_target():
    summary = summarize_residuals(_residuals(n=40, seed=61))
    target = ((0.25, 0.0), (0.0, 0.25))
    at_zero = shrink_summary(summary, target, shrinkage_lambda=0.0)
    at_one = shrink_summary(summary, target, shrinkage_lambda=1.0)
    np.testing.assert_allclose(at_zero.covariance, summary.state_corrected, atol=1e-12)
    np.testing.assert_allclose(at_one.covariance, target, atol=1e-12)
    assert at_zero.frame == summary.frame == "xy"

    default = shrink_summary(summary, target)
    assert 0.0 < default.shrinkage_lambda < 1.0
    assert default.sample_count == summary.sample_count


def test_pooled_target_is_count_weighted():
    small = summarize_residuals(_residuals(noise_std=0.30, n=6, seed=71))
    large = summarize_residuals(_residuals(noise_std=0.03, n=600, seed=73))
    pooled = np.asarray(pooled_target([small, large]))
    small_trace = np.trace(np.asarray(small.state_corrected))
    large_trace = np.trace(np.asarray(large.state_corrected))
    # The 600-sample camera must dominate the 6-sample one.
    assert abs(np.trace(pooled) - large_trace) < abs(np.trace(pooled) - small_trace)


def test_shrinkage_target_must_be_positive_definite():
    summary = summarize_residuals(_residuals(seed=79))
    with pytest.raises(ContractValidationError):
        shrink_summary(summary, ((0.0, 0.0), (0.0, 1.0)))


# --------------------------------------------------------------------------- #
# ground-truth firewall
# --------------------------------------------------------------------------- #


def test_no_function_here_accepts_ground_truth():
    from reliability.contracts import EVALUATION_ONLY_FIELD_NAMES

    functions = (
        build_operational_residuals,
        summarize_residuals,
        shrink_summary,
        circularity_factor,
        total_covariances,
        pooled_target,
    )
    for fn in functions:
        params = set(inspect.signature(fn).parameters)
        leaked = params & set(EVALUATION_ONLY_FIELD_NAMES)
        assert not leaked, f"{fn.__name__} accepts ground truth: {sorted(leaked)}"


def test_residual_record_carries_no_truth_field():
    record = _residuals(seed=83)[0]
    from reliability.contracts import EVALUATION_ONLY_FIELD_NAMES

    fields = set(vars(record))
    assert not (fields & set(EVALUATION_ONLY_FIELD_NAMES))
    assert not any(name.startswith("eval_") for name in fields)
