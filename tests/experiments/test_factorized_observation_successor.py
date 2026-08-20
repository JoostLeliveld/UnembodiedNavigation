from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/factorized_observation_successor"))
import common as C  # noqa: E402


def test_expected_longest_miss_exact_small_cases():
    assert C.expected_longest_miss(np.asarray([1.0, 1.0])) == 0.0
    assert C.expected_longest_miss(np.asarray([0.0, 0.0])) == 2.0
    assert abs(C.expected_longest_miss(np.asarray([0.5, 0.5])) - 1.0) < 1e-12


def test_resampled_path_keeps_endpoints_and_length():
    path = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    sampled = C.resample_path(path, 0.2)
    np.testing.assert_allclose(sampled[0], path[0])
    np.testing.assert_allclose(sampled[-1], path[-1])
    assert C.path_length(path) == 2.0


def test_holdout_pair_was_not_in_old_two_camera_sweep():
    old_pairs = {("camera_A", "camera_B"), ("camera_A", "camera_C")}
    assert C.HOLDOUT_CAMERAS not in old_pairs
