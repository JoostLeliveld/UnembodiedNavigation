"""Tests for the offline KF+RTS trajectory smoother.

Covers the M2 battery from ``experiments/operational_residual_rcond/PLAN.md``
(terminal equality, no-future-information, linear-Gaussian reference,
``P^s <= P^+``, PSD, GT-free API) plus the two checks the incoming spec lacked:
leave-one-source-out support and the ground-truth firewall on the API itself.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "state"))
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from state.core import trajectory_smoother as ts  # noqa: E402


def _straight_track(n: int = 40, step: float = 0.10) -> np.ndarray:
    return np.column_stack([np.arange(n) * step, np.zeros(n)])


def _meas(index: int, z, var: float = 0.15**2, source: str = "camera_A") -> ts.Measurement:
    return ts.Measurement(index=index, z=tuple(z), covariance=((var, 0.0), (0.0, var)), source=source)


# --------------------------------------------------------------------------- #
# contract validation
# --------------------------------------------------------------------------- #


def test_measurement_rejects_non_psd_covariance():
    with pytest.raises(ts.SmootherError):
        ts.Measurement(index=0, z=(0.0, 0.0), covariance=((0.0, 0.0), (0.0, 1.0)), source="camera_A")


def test_measurement_rejects_asymmetric_covariance():
    with pytest.raises(ts.SmootherError):
        ts.Measurement(index=0, z=(0.0, 0.0), covariance=((1.0, 0.5), (0.1, 1.0)), source="camera_A")


def test_measurement_requires_a_source_id():
    """A residual with no camera id could be attributed to the wrong camera."""
    with pytest.raises(ts.SmootherError):
        ts.Measurement(index=0, z=(0.0, 0.0), covariance=((1.0, 0.0), (0.0, 1.0)), source="")


def test_measurement_index_past_track_end_is_rejected():
    with pytest.raises(ts.SmootherError):
        ts.filter_forward(
            ts.increments_from_track(_straight_track(5)),
            [_meas(9, (1.0, 0.0))],
            initial_mean=(0.0, 0.0),
        )


def test_increments_reject_non_finite_track():
    track = _straight_track(10)
    track[4, 0] = np.nan
    with pytest.raises(ts.SmootherError):
        ts.increments_from_track(track)


def test_increments_are_differences_with_zero_first():
    track = _straight_track(6, step=0.25)
    u = ts.increments_from_track(track)
    assert np.allclose(u[0], (0.0, 0.0))
    assert np.allclose(u[1:, 0], 0.25)


# --------------------------------------------------------------------------- #
# M1-equivalent: forward pass
# --------------------------------------------------------------------------- #


def test_zero_noise_propagation_matches_deterministic_integration():
    """M1.1 -- with Q=0, P0=0 and no measurements the mean is dead reckoning."""
    track = _straight_track(25, step=0.2)
    cfg = ts.SmootherConfig(q_base_m=0.0, q_per_metre=0.0, initial_position_std_m=0.0)
    fwd = ts.filter_forward(
        ts.increments_from_track(track),
        [],
        initial_mean=track[0],
        config=cfg,
    )
    np.testing.assert_allclose(fwd.filtered_mean, track, atol=1e-12)


def test_covariance_grows_without_measurements():
    """M1.2 -- trace(P) is non-decreasing while the camera says nothing."""
    fwd = ts.filter_forward(
        ts.increments_from_track(_straight_track(30)),
        [],
        initial_mean=(0.0, 0.0),
    )
    traces = np.trace(fwd.filtered_cov, axis1=1, axis2=2)
    assert np.all(np.diff(traces) > -1e-12)


def test_accepted_measurement_contracts_covariance_and_moves_mean_toward_z():
    """M1.4 / M1.5 -- an update must not inflate P, and must pull toward z."""
    u = ts.increments_from_track(_straight_track(20))
    fwd = ts.filter_forward(u, [_meas(10, (0.5, 0.9))], initial_mean=(0.0, 0.0))
    assert np.trace(fwd.filtered_cov[10]) < np.trace(fwd.predicted_cov[10])
    # predicted y is 0; the measurement says 0.9, so the update must raise y.
    assert fwd.filtered_mean[10][1] > fwd.predicted_mean[10][1]
    assert fwd.filtered_mean[10][1] < 0.9  # and must not overshoot the measurement


def test_forward_covariance_stays_symmetric_and_psd():
    """M1.4 -- every step, filtered and predicted."""
    rng = np.random.default_rng(7)
    track = _straight_track(60)
    meas = [_meas(k, track[k] + rng.normal(0, 0.15, 2)) for k in range(0, 60, 5)]
    fwd = ts.filter_forward(ts.increments_from_track(track), meas, initial_mean=track[0])
    for name in ("filtered_cov", "predicted_cov"):
        stack = getattr(fwd, name)
        for k in range(stack.shape[0]):
            np.testing.assert_allclose(stack[k], stack[k].T, atol=1e-12)
            assert float(np.min(np.linalg.eigvalsh(stack[k]))) >= -1e-12, f"{name}[{k}] not PSD"


def test_per_step_process_noise_is_rate_dependent():
    """Why the rate-invariant form exists: the same path logged twice as densely
    accumulates ~2x the drift variance under exp5's per-step Q."""
    cfg = ts.SmootherConfig(process_model=ts.PROCESS_PER_STEP, initial_position_std_m=0.0)
    coarse = ts.filter_forward(
        ts.increments_from_track(_straight_track(50, step=0.02)), [],
        initial_mean=(0.0, 0.0), config=cfg,
    )
    fine = ts.filter_forward(
        ts.increments_from_track(_straight_track(100, step=0.01)), [],
        initial_mean=(0.0, 0.0), config=cfg,
    )
    # Same 0.98 m of travel, but the denser log ends far more uncertain.
    assert np.trace(fine.filtered_cov[-1]) > 1.5 * np.trace(coarse.filtered_cov[-1])


def test_rate_invariant_process_noise_is_rate_independent():
    """The same trajectory at two logging rates must end at the same covariance."""
    cfg = ts.SmootherConfig(process_model=ts.PROCESS_RATE_INVARIANT, initial_position_std_m=0.0)
    duration = 1.0
    coarse = ts.filter_forward(
        ts.increments_from_track(_straight_track(51, step=0.02)), [],
        initial_mean=(0.0, 0.0), config=cfg, dt=np.full(51, duration / 50),
    )
    fine = ts.filter_forward(
        ts.increments_from_track(_straight_track(101, step=0.01)), [],
        initial_mean=(0.0, 0.0), config=cfg, dt=np.full(101, duration / 100),
    )
    np.testing.assert_allclose(
        np.trace(fine.filtered_cov[-1]), np.trace(coarse.filtered_cov[-1]), rtol=1e-9
    )


def test_rate_invariant_variance_is_linear_in_time_and_distance():
    cfg = ts.SmootherConfig(
        process_model=ts.PROCESS_RATE_INVARIANT,
        sigma_per_sqrt_s=0.1, sigma_per_sqrt_m=0.2,
    )
    time_only = cfg.process_covariance(np.array([0.0, 0.0]), dt=4.0)
    np.testing.assert_allclose(time_only, 0.01 * 4.0 * np.eye(2), atol=1e-15)
    both = cfg.process_covariance(np.array([3.0, 4.0]), dt=2.0)  # |u| = 5
    np.testing.assert_allclose(both, (0.01 * 2.0 + 0.04 * 5.0) * np.eye(2), atol=1e-15)


def test_rate_invariant_model_requires_dt():
    cfg = ts.SmootherConfig(process_model=ts.PROCESS_RATE_INVARIANT)
    with pytest.raises(ts.SmootherError):
        ts.filter_forward(ts.increments_from_track(_straight_track(10)), [], initial_mean=(0.0, 0.0), config=cfg)


def test_dt_length_must_match_the_track():
    cfg = ts.SmootherConfig(process_model=ts.PROCESS_RATE_INVARIANT)
    with pytest.raises(ts.SmootherError):
        ts.filter_forward(
            ts.increments_from_track(_straight_track(10)), [],
            initial_mean=(0.0, 0.0), config=cfg, dt=np.full(5, 0.02),
        )


def test_unknown_process_model_is_refused():
    with pytest.raises(ts.SmootherError):
        ts.SmootherConfig(process_model="magic")


def test_soft_gate_downweights_an_outlier_instead_of_rejecting_it():
    """The gate must inflate R, never hard-reject -- hard rejection is what caused
    the gate-runaway chain already diagnosed in this repo."""
    u = ts.increments_from_track(_straight_track(20))
    outlier = _meas(10, (0.9, 8.0))  # ~8 m off in y

    gated = ts.filter_forward(u, [outlier], initial_mean=(0.0, 0.0))
    ungated = ts.filter_forward(
        u, [outlier], initial_mean=(0.0, 0.0), config=ts.SmootherConfig(nis_soft_gate=0.0)
    )
    # It still moves (not rejected) but by strictly less than the ungated update.
    assert gated.filtered_mean[10][1] > gated.predicted_mean[10][1]
    assert gated.filtered_mean[10][1] < ungated.filtered_mean[10][1]
    assert gated.update_counts[10] == 1
    assert np.isfinite(gated.nis[10]) and gated.nis[10] > ts.CHI2_2DOF_99


def test_simultaneous_measurements_are_applied_sequentially():
    u = ts.increments_from_track(_straight_track(15))
    both = ts.filter_forward(
        u,
        [_meas(7, (0.7, 0.4), source="camera_A"), _meas(7, (0.7, 0.4), source="camera_B")],
        initial_mean=(0.0, 0.0),
    )
    single = ts.filter_forward(u, [_meas(7, (0.7, 0.4), source="camera_A")], initial_mean=(0.0, 0.0))
    assert both.update_counts[7] == 2
    # Two independent fixes are more informative than one.
    assert np.trace(both.filtered_cov[7]) < np.trace(single.filtered_cov[7])


# --------------------------------------------------------------------------- #
# M2: smoother
# --------------------------------------------------------------------------- #


def test_terminal_smoothed_equals_terminal_filtered():
    """M2.1 -- mu_T^s = mu_T^+, P_T^s = P_T^+."""
    track = _straight_track(30)
    out = ts.smooth_trajectory(
        ts.increments_from_track(track),
        [_meas(k, track[k]) for k in (5, 12, 25)],
        initial_mean=track[0],
    )
    np.testing.assert_allclose(out.smoothed_mean[-1], out.filtered_mean[-1], atol=1e-12)
    np.testing.assert_allclose(out.smoothed_cov[-1], out.filtered_cov[-1], atol=1e-12)


def test_no_future_information_leaves_the_tail_unchanged():
    """M2.2 -- after the last measurement there is nothing to smooth with."""
    track = _straight_track(40)
    last = 20
    out = ts.smooth_trajectory(
        ts.increments_from_track(track),
        [_meas(10, track[10]), _meas(last, track[last])],
        initial_mean=track[0],
    )
    np.testing.assert_allclose(
        out.smoothed_mean[last:], out.filtered_mean[last:], atol=1e-9
    )
    np.testing.assert_allclose(out.smoothed_cov[last:], out.filtered_cov[last:], atol=1e-9)


def test_future_measurement_revises_an_earlier_belief():
    """The point of smoothing: a later fix must move an earlier mean."""
    track = _straight_track(40)
    # Odometry is drift-free here, so put the fix off-track and check the past moves.
    out = ts.smooth_trajectory(
        ts.increments_from_track(track), [_meas(30, (3.0, 0.6))], initial_mean=track[0]
    )
    assert out.smoothed_mean[5][1] > out.filtered_mean[5][1] + 1e-6
    assert np.trace(out.smoothed_cov[5]) < np.trace(out.filtered_cov[5])


def test_smoothed_covariance_is_no_larger_than_filtered():
    """M2.4 -- P^s <= P^+ in the PSD sense: lambda_max(P^s - P^+) <= eps."""
    rng = np.random.default_rng(11)
    track = _straight_track(80)
    meas = [_meas(k, track[k] + rng.normal(0, 0.15, 2)) for k in range(0, 80, 7)]
    out = ts.smooth_trajectory(ts.increments_from_track(track), meas, initial_mean=track[0])
    for k in range(out.n_steps):
        delta = out.smoothed_cov[k] - out.filtered_cov[k]
        assert float(np.max(np.linalg.eigvalsh(delta))) <= 1e-9, f"P^s exceeded P^+ at {k}"


def test_smoothed_covariance_stays_symmetric_and_psd():
    rng = np.random.default_rng(13)
    track = _straight_track(50)
    meas = [_meas(k, track[k] + rng.normal(0, 0.2, 2)) for k in range(0, 50, 4)]
    out = ts.smooth_trajectory(ts.increments_from_track(track), meas, initial_mean=track[0])
    for k in range(out.n_steps):
        np.testing.assert_allclose(out.smoothed_cov[k], out.smoothed_cov[k].T, atol=1e-12)
        assert float(np.min(np.linalg.eigvalsh(out.smoothed_cov[k]))) >= -1e-12


def test_matches_a_closed_form_two_step_kalman_smoother():
    """M2.3 -- linear-Gaussian reference computed by hand.

    Two steps, scalar-isotropic, one measurement at t=1. Hand values:
      P0 = p0 I, Q = q I  ->  P1^- = (p0 + q) I
      P1^+ = (p0+q)r/(p0+q+r) I,  mu1^+ = mu1^- + (p0+q)/(p0+q+r) (z - mu1^-)
      G0 = p0 / (p0 + q)
      mu0^s = mu0 + G0 (mu1^s - mu1^-),  P0^s = p0 + G0^2 (P1^+ - P1^-)
    """
    p0, q, r = 0.09, 0.04, 0.0225
    cfg = ts.SmootherConfig(
        q_base_m=float(np.sqrt(q)), q_per_metre=0.0, initial_position_std_m=float(np.sqrt(p0)),
        nis_soft_gate=0.0,
    )
    u = np.array([[0.0, 0.0], [1.0, 0.0]])
    z = np.array([1.5, 0.0])
    out = ts.smooth_trajectory(
        u, [_meas(1, z, var=r)], initial_mean=(0.0, 0.0), config=cfg
    )

    p1_pred = p0 + q
    gain = p1_pred / (p1_pred + r)
    mu1_pred = np.array([1.0, 0.0])
    mu1_post = mu1_pred + gain * (z - mu1_pred)
    p1_post = p1_pred * r / (p1_pred + r)
    g0 = p0 / p1_pred
    mu0_s = np.array([0.0, 0.0]) + g0 * (mu1_post - mu1_pred)
    p0_s = p0 + g0**2 * (p1_post - p1_pred)

    np.testing.assert_allclose(out.smoothed_mean[1], mu1_post, atol=1e-12)
    np.testing.assert_allclose(out.smoothed_cov[1], p1_post * np.eye(2), atol=1e-12)
    np.testing.assert_allclose(out.smoothed_mean[0], mu0_s, atol=1e-12)
    np.testing.assert_allclose(out.smoothed_cov[0], p0_s * np.eye(2), atol=1e-12)


def test_smoothing_is_deterministic():
    """M7.4-style determinism: identical inputs, identical trace."""
    track = _straight_track(35)
    meas = [_meas(k, track[k] + 0.05) for k in (4, 15, 28)]
    kwargs = dict(initial_mean=track[0])
    a = ts.smooth_trajectory(ts.increments_from_track(track), meas, **kwargs)
    b = ts.smooth_trajectory(ts.increments_from_track(track), meas, **kwargs)
    np.testing.assert_array_equal(a.smoothed_mean, b.smoothed_mean)
    np.testing.assert_array_equal(a.smoothed_cov, b.smoothed_cov)


# --------------------------------------------------------------------------- #
# leave-one-source-out (PLAN.md R3) -- the incoming spec had no equivalent
# --------------------------------------------------------------------------- #


def test_without_source_drops_only_that_camera():
    meas = [
        _meas(1, (0.1, 0.0), source="camera_A"),
        _meas(2, (0.2, 0.0), source="camera_B"),
        _meas(3, (0.3, 0.0), source="camera_A"),
    ]
    kept = ts.without_source(meas, "camera_A")
    assert [m.source for m in kept] == ["camera_B"]


def test_without_source_requires_an_id():
    with pytest.raises(ts.SmootherError):
        ts.without_source([], "")


def test_holding_a_camera_out_leaves_a_larger_residual_at_its_own_fixes():
    """The circularity Gate R3 exists to expose.

    Camera A's fix is biased; when A anchors the smoother the trajectory absorbs
    that bias and A's residual shrinks. Held out, the residual survives.
    """
    track = _straight_track(60)
    bias = 0.30
    a_indices = list(range(10, 50, 4))
    meas_a = [_meas(k, (track[k][0], track[k][1] + bias), source="camera_A") for k in a_indices]
    meas_b = [_meas(k, track[k], source="camera_B") for k in range(12, 50, 4)]

    with_a = ts.smooth_trajectory(
        ts.increments_from_track(track), meas_a + meas_b, initial_mean=track[0]
    )
    without_a = ts.smooth_trajectory(
        ts.increments_from_track(track),
        ts.without_source(meas_a + meas_b, "camera_A"),
        initial_mean=track[0],
    )

    def mean_abs_residual_y(traj):
        return float(np.mean([abs(m.z[1] - traj.smoothed_mean[m.index][1]) for m in meas_a]))

    assert mean_abs_residual_y(without_a) > mean_abs_residual_y(with_a)
    # And held out, the residual recovers most of the true bias.
    assert mean_abs_residual_y(without_a) > 0.5 * bias
    assert without_a.sources == ("camera_B",)


# --------------------------------------------------------------------------- #
# ground-truth firewall (M2.5)
# --------------------------------------------------------------------------- #


def test_no_public_inference_function_accepts_ground_truth():
    """M2.5 -- the smoother API must be structurally incapable of taking truth."""
    inference = (
        ts.filter_forward,
        ts.rts_backward,
        ts.smooth_trajectory,
        ts.increments_from_track,
        ts.without_source,
    )
    for fn in inference:
        params = set(inspect.signature(fn).parameters)
        leaked = params & ts.EVALUATION_ONLY_ARGUMENT_NAMES
        assert not leaked, f"{fn.__name__} accepts ground truth: {sorted(leaked)}"


def test_firewall_list_covers_the_repo_wide_evaluation_only_names():
    """Keep this module's local list in step with the canonical firewall.

    ``reliability.contracts`` is the repo's source of truth; a name added there
    must not silently become an acceptable smoother argument.
    """
    from reliability.contracts import EVALUATION_ONLY_FIELD_NAMES

    # The canonical list carries derived/scoring names that are not plausible
    # smoother arguments; assert coverage of the pose/residual names that are.
    must_cover = {"gt_x", "gt_y", "gt_yaw", "gt_available", "true_x", "true_y", "true_yaw",
                  "true_available", "state_pos_error", "ground_truth_pose"}
    assert must_cover <= EVALUATION_ONLY_FIELD_NAMES, "canonical firewall changed unexpectedly"
    assert must_cover <= ts.EVALUATION_ONLY_ARGUMENT_NAMES


def test_nees_is_calibrated_on_synthetic_gaussian_errors():
    """The scoring helper itself: NEES ~ chi2_2 has mean 2."""
    rng = np.random.default_rng(3)
    n = 20000
    sigma = 0.2
    truth = np.zeros((n, 2))
    mean = rng.normal(0.0, sigma, (n, 2))
    cov = np.repeat((sigma**2 * np.eye(2))[None, :, :], n, axis=0)
    values = ts.nees(mean, cov, truth)
    assert 1.9 < float(np.mean(values)) < 2.1


def test_nees_reports_nan_where_the_reference_is_missing():
    mean = np.zeros((3, 2))
    cov = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    ref = np.array([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]])
    values = ts.nees(mean, cov, ref)
    assert np.isnan(values[1]) and np.isfinite(values[0]) and np.isfinite(values[2])
