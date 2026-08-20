from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "experiments/self_commissioning_observation_field"
sys.path.insert(0, str(DIRECTORY))

import commission_field as M  # noqa: E402
import field_common as C  # noqa: E402
import planner_ablation as P  # noqa: E402


def test_visibility_strata_are_operational_probability_bins():
    assert C.visibility_mode(0.7999) == "marginal"
    assert C.visibility_mode(0.8) == "clear"
    np.testing.assert_array_equal(
        C.visibility_mode(np.asarray([0.1, 0.9])), np.asarray(["marginal", "clear"])
    )


def test_niw_keeps_bias_and_measurement_uncertainty_separate():
    error = np.asarray([[1.0, -2.0], [1.1, -1.9], [0.9, -2.1]])
    group = M.niw_group(error)
    assert group["R"].shape == (2, 2)
    assert group["bias_cov"].shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(group["R"]) > 0.0)
    assert np.trace(group["bias_cov"]) < np.trace(group["R"])


def test_exact_four_camera_subset_mass_and_extremes():
    prior = np.eye(2)
    measurements = np.repeat((np.eye(2) * 0.25)[None, :, :], 4, axis=0)
    unchanged, misses = P.exact_subset_update(prior, np.zeros(4), measurements)
    np.testing.assert_allclose(unchanged, prior)
    assert abs(sum(item["probability"] for item in misses) - 1.0) < 1e-12

    updated, hits = P.exact_subset_update(prior, np.ones(4), measurements)
    expected = np.linalg.inv(np.linalg.inv(prior) + 4.0 * np.linalg.inv(measurements[0]))
    np.testing.assert_allclose(updated, expected)
    assert len(hits) == 16


def test_expected_longest_miss_small_exact_cases():
    assert C.expected_longest_miss(np.asarray([1.0, 1.0])) == 0.0
    assert C.expected_longest_miss(np.asarray([0.0, 0.0])) == 2.0
    assert abs(C.expected_longest_miss(np.asarray([0.5, 0.5])) - 1.0) < 1e-12
