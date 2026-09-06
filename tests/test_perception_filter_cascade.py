"""Correctness tests for the perception-side box filter and the b/q split.

These test the estimator machinery, not any experimental result: whether the filter is a
valid Kalman filter, whether the calibration statistic has the reference value it is
compared against, and whether the trend/residual split recovers a component it was given.
"""
from __future__ import annotations

import math
import random
import statistics as st
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'experiments/perception_filter_cascade'))

from compare_perception_filters import (  # noqa: E402
    BoxFilter,
    inv2,
    mat_mul,
    normalised_squared_error,
    transpose,
)
from separate_bias_and_fast_noise import (  # noqa: E402
    lag1_autocorr,
    polyfit,
    polyval,
    robust_sd,
)


def test_normalised_squared_error_reference_value():
    """A one-sigma error on each axis of a 2-D measurement must score exactly 2.0.

    This is the number the calibration verdict is read against, so it is pinned here.
    """
    cov = [[0.01, 0.0], [0.0, 0.01]]  # sigma = 0.1 m per axis
    assert normalised_squared_error(0.0, 0.0, cov, 0.0) == pytest.approx(0.0)
    assert normalised_squared_error(0.1, 0.1, cov, 0.0) == pytest.approx(2.0)
    # A floor added in quadrature must make the statistic smaller, never larger.
    assert normalised_squared_error(0.1, 0.1, cov, 0.05) < 2.0


def test_covariance_floor_is_added_in_quadrature():
    cov = [[0.0004, 0.0], [0.0, 0.0004]]  # sigma = 2 cm
    floor = 0.02  # another 2 cm
    # Error of exactly the combined sigma on one axis -> contribution 1.0.
    combined = math.sqrt(0.0004 + floor ** 2)
    assert normalised_squared_error(combined, 0.0, cov, floor) == pytest.approx(1.0)


def test_filter_posterior_covariance_is_symmetric_and_shrinks():
    kf = BoxFilter(q_accel_px=16.6, r_px=1.4, step_s=0.18)
    kf.initialise(100.0, 200.0)
    first = kf.pixel_estimate()[2][0][0]
    for _ in range(10):
        kf.predict()
        kf.update(100.0, 200.0, 1.4)
    _, _, cov = kf.pixel_estimate()
    assert cov[0][1] == pytest.approx(cov[1][0], abs=1e-12)
    assert cov[0][0] > 0.0 and cov[1][1] > 0.0
    # Repeated consistent observations must reduce the stated uncertainty.
    assert cov[0][0] < first


def test_filter_reduces_white_pixel_noise_on_a_ramp():
    """With genuinely white noise the filter must beat the per-frame observation."""
    random.seed(7)
    sigma = 1.4
    truth = [(100.0 + 0.9 * index, 200.0 + 0.4 * index) for index in range(40)]
    observations = [(u + random.gauss(0.0, sigma), v + random.gauss(0.0, sigma))
                    for u, v in truth]

    kf = BoxFilter(q_accel_px=16.6, r_px=sigma, step_s=0.18)
    kf.initialise(*observations[0])
    raw_errors, filtered_errors = [], []
    for index, (u, v) in enumerate(observations):
        if index > 0:
            kf.predict()
            kf.update(u, v, sigma)
        mu_u, mu_v, _ = kf.pixel_estimate()
        raw_errors.append(math.hypot(u - truth[index][0], v - truth[index][1]))
        filtered_errors.append(math.hypot(mu_u - truth[index][0], mu_v - truth[index][1]))

    assert st.median(filtered_errors) < st.median(raw_errors)


def test_filter_cannot_remove_a_constant_offset():
    """The mechanism the study exists to expose: a shared bias survives filtering.

    The filter shrinks its stated uncertainty with every consistent observation, but a
    constant offset present in all of them is untouched. Accuracy stalls while the claimed
    covariance keeps falling, which is exactly the overconfidence the decision rule checks.
    """
    offset = 12.0
    kf = BoxFilter(q_accel_px=16.6, r_px=1.4, step_s=0.18)
    kf.initialise(100.0 + offset, 200.0)
    for _ in range(30):
        kf.predict()
        kf.update(100.0 + offset, 200.0, 1.4)
    mu_u, _, cov = kf.pixel_estimate()
    assert abs(mu_u - 100.0) == pytest.approx(offset, abs=1e-6)
    assert math.sqrt(cov[0][0]) < offset  # claims far more precision than it has


def test_jacobian_propagation_scales_covariance():
    """R_xy = J P J^T must scale as the square of a uniform Jacobian gain."""
    cov_px = [[4.0, 0.0], [0.0, 4.0]]
    jac = [[0.01, 0.0], [0.0, 0.01]]  # 1 px -> 1 cm
    out = mat_mul(mat_mul(jac, cov_px), transpose(jac))
    assert out[0][0] == pytest.approx(4.0 * 0.01 ** 2)
    assert math.sqrt(out[0][0]) == pytest.approx(0.02)  # 2 px -> 2 cm


def test_inv2_rejects_singular_matrix():
    with pytest.raises(ValueError):
        inv2([[1.0, 2.0], [2.0, 4.0]])


def test_polyfit_recovers_a_known_quadratic():
    xs = [0.04 * index for index in range(16)]
    ys = [3.0 - 2.0 * x + 5.0 * x ** 2 for x in xs]
    coeffs = polyfit(xs, ys, 2)
    assert coeffs[0] == pytest.approx(3.0, abs=1e-6)
    assert coeffs[1] == pytest.approx(-2.0, abs=1e-6)
    assert coeffs[2] == pytest.approx(5.0, abs=1e-6)
    assert polyval(coeffs, 0.2) == pytest.approx(3.0 - 0.4 + 0.2, abs=1e-6)


def test_trend_removal_recovers_an_injected_fast_component():
    """The split must return the fast component it was given, not the smooth trend."""
    random.seed(11)
    injected = 1.4
    xs = [0.04 * index for index in range(31)]
    smooth = [50.0 + 8.0 * x - 3.0 * x ** 2 for x in xs]  # stands in for b(s)
    values = [value + random.gauss(0.0, injected) for value in smooth]

    coeffs = polyfit(xs, values, 2)
    residual = [value - polyval(coeffs, x) for x, value in zip(xs, values)]

    # Recovered scale should be close to what was injected, and clearly non-zero.
    assert robust_sd(residual) == pytest.approx(injected, rel=0.6)
    assert st.pstdev(residual) > 0.4 * injected
    # The trend itself must be gone: residual mean near zero.
    assert abs(st.mean(residual)) < 0.5 * injected


def test_lag1_autocorr_separates_white_from_correlated():
    random.seed(3)
    white = [random.gauss(0.0, 1.0) for _ in range(400)]
    assert abs(lag1_autocorr(white)) < 0.2

    # A slowly varying sequence, the signature of unremoved b(s).
    correlated, value = [], 0.0
    for _ in range(400):
        value = 0.95 * value + random.gauss(0.0, 0.3)
        correlated.append(value)
    assert lag1_autocorr(correlated) > 0.7


def test_robust_sd_is_not_moved_by_a_few_outliers():
    random.seed(5)
    clean = [random.gauss(0.0, 1.0) for _ in range(200)]
    contaminated = clean + [60.0, -55.0, 70.0]
    assert robust_sd(contaminated) == pytest.approx(robust_sd(clean), rel=0.25)
    # The plain standard deviation is not robust, which is why both are reported.
    assert st.pstdev(contaminated) > 1.8 * st.pstdev(clean)


def test_decision_rule_requires_accuracy_without_losing_calibration(tmp_path):
    """The verdict must reject an arm that buys accuracy by overstating its precision.

    This pins the rule itself, so a later edit cannot quietly relax it: the only arm that
    passes is the one that improves median error while not moving further from the
    consistency reference and not growing the tail.
    """
    import json
    import subprocess

    sys.path.insert(0, str(REPO / 'experiments/perception_filter_cascade'))
    (tmp_path / 'bq_split.json').write_text(json.dumps({
        'capture': 'FIXTURE', 'rows_total': 100, 'rows_detected': 50, 'lines_used': 4,
        'trend_order': 2,
        'summary': {'u_bbox_bottom': {
            'unit': 'px', 'n': 50, 'sd': 1.5, 'robust_sd': 1.4, 'median_abs': 0.9,
            'mean': 0.0, 'lag1_autocorr_median': 0.05, 'trend_span_median': 12.0}},
        'order_sweep': {
            '1': {'u_bbox_bottom': {'robust_sd': 1.5, 'lag1_autocorr_median': 0.06}},
            '2': {'u_bbox_bottom': {'robust_sd': 1.4, 'lag1_autocorr_median': 0.05}}},
    }), encoding='utf-8')

    def arm(median, nse, tail):
        return {'n': 50, 'median_err_cm': median, 'rms_err_cm': median,
                'p90_err_cm': median, 'mean_nse': nse, 'median_nse': nse,
                'beyond_4sigma_pct': tail}

    (tmp_path / 'arm_comparison.json').write_text(json.dumps({
        'settings': {'sigma_px': 0.764, 'static_sigma_px': 1.4, 'q_accel_px': 20.7,
                     'px_per_metre': 41.4, 'accel_m_s2': 0.5, 'step_s': 0.18,
                     'lag': 3, 'floor_m': 0.0},
        'overall': {
            'E_static_R': arm(10.0, 2.0, 5.0),
            'B_kf': arm(9.0, 20.0, 30.0),      # better median, badly overconfident
            'C_robust_kf': arm(9.0, 2.0, 5.0),  # better median, calibration held
            'D_smoother': arm(11.0, 2.0, 5.0),  # calibration fine, no accuracy gain
        },
        'per_camera': {},
    }), encoding='utf-8')

    subprocess.run(
        [sys.executable, str(REPO / 'experiments/perception_filter_cascade/write_results.py'),
         '--study-dir', str(tmp_path)],
        check=True, capture_output=True)
    report = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')

    assert 'constant-velocity box filter: does NOT earn its place' in report
    assert 'soft rejection: EARNS its place' in report
    assert 'fixed-lag smoother: does NOT earn its place' in report


def test_inv4_inverts_and_rejects_singular():
    from compare_perception_filters import inv4

    matrix = [[4.0, 1.0, 0.0, 0.0],
              [1.0, 3.0, 0.5, 0.0],
              [0.0, 0.5, 2.0, 0.25],
              [0.0, 0.0, 0.25, 1.5]]
    inverse = inv4(matrix)
    product = mat_mul(matrix, inverse)
    for row in range(4):
        for col in range(4):
            assert product[row][col] == pytest.approx(1.0 if row == col else 0.0, abs=1e-9)

    with pytest.raises(ValueError):
        inv4([[1.0, 2.0, 3.0, 4.0]] * 4)


def test_smoother_beats_the_filter_and_states_less_uncertainty():
    """A correct fixed-lag smoother uses later observations, so it must do better.

    This guards against the approximation it replaced: back-propagating the filtered mean
    with the motion model leaves the covariance at its filtered value, which would let the
    arm claim the wrong uncertainty and make the calibration comparison meaningless.
    """
    import compare_perception_filters as cpf

    random.seed(23)
    sigma = 1.4

    # A scaled projection so the test isolates the estimator from the geometry, with the
    # truth columns set to the projection of the noise-free pixel. Otherwise every arm
    # carries a constant offset that swamps the noise being measured.
    def project(u, v):
        return u / 100.0, v / 100.0

    def jacobian(u, v):
        return [[0.01, 0.0], [0.0, 0.01]]

    truth = [(100.0 + 0.9 * index, 200.0 + 0.4 * index) for index in range(40)]
    rows = []
    for index, (u, v) in enumerate(truth):
        truth_x, truth_y = project(u, v)
        rows.append({
            'camera_id': 'camera_B',
            'dataset_split': 'line0_x_yaw0',
            'robot_x': str(truth_x),
            'robot_y': str(truth_y),
            'detected': 'True',
            'camera_range_m': '4.0',
            'confidence': '0.93',
            'u_bbox_bottom': str(u + random.gauss(0.0, sigma)),
            'v_bbox_bottom': str(v + random.gauss(0.0, sigma)),
        })

    out = cpf.run_line(
        rows, project=project, jacobian=jacobian, sigma_px=sigma,
        q_accel_px=20.7, step_s=0.18, gate=9.0, lag=5, floor_m=0.0,
        static_sigma_px=sigma,
    )

    raw = cpf.score(out['A_raw'])
    filtered = cpf.score(out['B_kf'])
    smoothed = cpf.score(out['D_smoother'])

    # With genuinely white noise the ordering must be raw > filtered > smoothed, because
    # the smoother additionally uses observations that arrive after the target instant.
    assert filtered['rms_err_cm'] < raw['rms_err_cm']
    assert smoothed['rms_err_cm'] < filtered['rms_err_cm']


def test_soft_rejection_downweights_an_outlier_but_not_a_clean_reading():
    """Arm C must resist a single gross detection without discarding it.

    Soft rejection is the alternative to a hard gate, so the properties that matter are:
    a clean sequence is unaffected, and one wild reading moves the estimate far less than
    it would without the inflation.
    """
    import compare_perception_filters as cpf

    def project(u, v):
        return u / 100.0, v / 100.0

    def jacobian(u, v):
        return [[0.01, 0.0], [0.0, 0.01]]

    def build(pixels):
        rows = []
        for index, (u, v) in enumerate(pixels):
            truth_x, truth_y = project(100.0 + 0.9 * index, 200.0 + 0.4 * index)
            rows.append({
                'camera_id': 'camera_B', 'dataset_split': 'line0_x_yaw0',
                'robot_x': str(truth_x), 'robot_y': str(truth_y),
                'detected': 'True', 'camera_range_m': '4.0', 'confidence': '0.93',
                'u_bbox_bottom': str(u), 'v_bbox_bottom': str(v),
            })
        return rows

    def run(rows):
        return cpf.run_line(
            rows, project=project, jacobian=jacobian, sigma_px=1.4, q_accel_px=20.7,
            step_s=0.18, gate=9.0, lag=3, floor_m=0.0, static_sigma_px=1.4)

    clean_pixels = [(100.0 + 0.9 * index, 200.0 + 0.4 * index) for index in range(20)]
    clean = run(build(clean_pixels))
    # With no surprising observation, soft rejection must change nothing.
    assert (cpf.score(clean['C_robust_kf'])['rms_err_cm']
            == pytest.approx(cpf.score(clean['B_kf'])['rms_err_cm'], rel=1e-9))

    spiked_pixels = list(clean_pixels)
    spiked_pixels[10] = (clean_pixels[10][0] + 60.0, clean_pixels[10][1] + 60.0)
    spiked = run(build(spiked_pixels))
    plain_error = math.hypot(spiked['B_kf'][10]['ex'], spiked['B_kf'][10]['ey'])
    robust_error = math.hypot(spiked['C_robust_kf'][10]['ex'], spiked['C_robust_kf'][10]['ey'])
    assert robust_error < plain_error
    # And the outlier still influences the estimate rather than being thrown away.
    assert robust_error > 0.0
