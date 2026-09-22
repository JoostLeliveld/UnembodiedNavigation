import numpy as np
import pytest

from reliability.planning_information import (
    effective_sample_size,
    fit_planning_information_field,
    realized_information,
)


def test_realized_information_uses_runtime_precision_and_zero_for_miss():
    covariance = np.asarray([
        [[0.25, 0.0], [0.0, 1.0]],
        [[np.nan, np.nan], [np.nan, np.nan]],
    ])
    actual = realized_information([True, False], covariance)
    np.testing.assert_allclose(actual[0], [[4.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(actual[1], np.zeros((2, 2)))


def test_known_bad_and_unsupported_regions_are_distinct():
    positions = np.asarray([[0.0, 0.0]] * 12 + [[1.0, 0.0]] * 12)
    admitted = np.asarray([False] * 12 + [True] * 12)
    covariance = np.full((24, 2, 2), np.nan)
    covariance[admitted] = np.asarray([[0.5, 0.0], [0.0, 0.25]])
    field = fit_planning_information_field(
        positions,
        ['camera_A'] * len(positions),
        admitted,
        covariance,
        camera_ids=['camera_A'],
        xs=[0.0, 1.0, 3.0],
        ys=[0.0, 1.0],
        length_scale_m=0.1,
        support_radius_m=0.25,
        support_tau=2.0,
    )
    # Known poor: many opportunities, supported zero information.
    assert field.opportunity_support[0, 0, 0] == pytest.approx(12.0)
    np.testing.assert_array_equal(field.expected_information[0, 0, 0], np.zeros((2, 2)))
    # Known useful: the runtime precision is retained, with finite-support shrinkage.
    alpha = 12.0 / 14.0
    np.testing.assert_allclose(
        field.expected_information[0, 0, 1], alpha * np.diag([2.0, 4.0]))
    # Unsupported: no kernel support and therefore zero deployed information.
    assert field.opportunity_support[0, 0, 2] == 0.0
    np.testing.assert_array_equal(field.expected_information[0, 0, 2], np.zeros((2, 2)))


def test_effective_sample_size_respects_unequal_kernel_weights():
    assert effective_sample_size(np.asarray([1.0, 1.0, 1.0])) == pytest.approx(3.0)
    assert effective_sample_size(np.asarray([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert effective_sample_size(np.zeros(3)) == 0.0


def test_admitted_runtime_covariance_must_be_spd():
    with pytest.raises(ValueError, match='positive definite'):
        realized_information([True], np.asarray([[[1.0, 2.0], [2.0, 1.0]]]))


def test_declared_headings_are_pooled_equally_and_support_counts_physical_positions():
    covariance = np.asarray([
        np.diag([0.25, 1.0]), np.diag([0.25, 1.0]),
        np.diag([0.25, 1.0]), np.diag([1.0, 0.25]),
    ])
    field = fit_planning_information_field(
        np.zeros((4, 2)), ['camera_A'] * 4, [True] * 4, covariance,
        camera_ids=['camera_A'], xs=[0.0, 1.0], ys=[0.0, 1.0],
        length_scale_m=0.2, support_radius_m=0.5, support_tau=0.0,
        headings_rad=[0.0, 0.0, 0.0, np.pi],
    )
    np.testing.assert_allclose(field.expected_information[0, 0, 0], 2.5 * np.eye(2))
    assert field.opportunity_support[0, 0, 0] == pytest.approx(1.0)


def test_kernel_mean_and_shrinkage_preserve_psd():
    covariance = np.asarray([
        [[0.3, 0.2], [0.2, 0.4]],
        [[0.5, -0.3], [-0.3, 0.6]],
    ])
    field = fit_planning_information_field(
        np.asarray([[0.0, 0.0], [1.0, 0.0]]), ['camera_A'] * 2,
        [True, True], covariance, camera_ids=['camera_A'],
        xs=[0.0, 0.5, 1.0], ys=[0.0, 1.0], length_scale_m=1.0,
        support_radius_m=2.0, support_tau=3.0,
    )
    assert np.linalg.eigvalsh(field.raw_expected_information).min() >= -1e-12
    assert np.linalg.eigvalsh(field.expected_information).min() >= -1e-12
