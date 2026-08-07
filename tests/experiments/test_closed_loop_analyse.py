from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import numpy as np
import pytest


CLOSED_LOOP = Path(__file__).resolve().parents[2] / "experiments" / "closed_loop_calibration"

# ``analyse`` prepends scripts/geometry_visibility to sys.path, where geometry_visibility.py
# shadows the package of the same name. Snapshot and restore, or importing this test module
# breaks collection of every test that imports the package form.
_SAVED_SYS_PATH = list(sys.path)
sys.path.insert(0, str(CLOSED_LOOP))
try:
    from analyse import (  # noqa: E402
        CHI2_2_95,
        belief_honesty,
        bootstrap_deltas,
        check_matrix,
        correction_path,
    )
finally:
    sys.path[:] = _SAVED_SYS_PATH


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_nees_matches_hand_computed_value(tmp_path: Path) -> None:
    """Unit-covariance belief 1 m off in x must score NEES exactly 1."""
    rows = [
        {"planner_cov_x": 1.0, "planner_cov_xy": 0.0, "planner_cov_y": 1.0},
        {"planner_cov_x": 1.0, "planner_cov_xy": 0.0, "planner_cov_y": 1.0},
    ]
    csv_path = tmp_path / "experiment.csv"
    _write_csv(csv_path, rows)
    run = {
        "belief_x": np.array([1.0, 3.0]),
        "belief_y": np.array([0.0, 0.0]),
        "truth_x": np.array([0.0, 0.0]),
        "truth_y": np.array([0.0, 0.0]),
    }

    out = belief_honesty(csv_path, run)

    assert out["nees_steps"] == 2
    assert out["nees_median"] == pytest.approx(5.0)  # median of {1, 9}
    assert out["outside_95_ellipse_rate"] == pytest.approx(0.5)  # only 9 > 5.991
    assert CHI2_2_95 == pytest.approx(5.9914645, abs=1e-6)


def test_correlated_covariance_is_not_treated_as_diagonal(tmp_path: Path) -> None:
    """A strong xy correlation must change NEES; ignoring it would return 2.0."""
    csv_path = tmp_path / "experiment.csv"
    _write_csv(csv_path, [{"planner_cov_x": 1.0, "planner_cov_xy": 0.8, "planner_cov_y": 1.0}])
    run = {
        "belief_x": np.array([1.0]),
        "belief_y": np.array([1.0]),
        "truth_x": np.array([0.0]),
        "truth_y": np.array([0.0]),
    }

    out = belief_honesty(csv_path, run)

    # e=(1,1), P=[[1,.8],[.8,1]] -> e'P^-1 e = 2/1.8
    assert out["nees_median"] == pytest.approx(2.0 / 1.8)


def test_singular_covariance_rows_are_dropped_not_scored(tmp_path: Path) -> None:
    csv_path = tmp_path / "experiment.csv"
    _write_csv(
        csv_path,
        [
            {"planner_cov_x": 0.0, "planner_cov_xy": 0.0, "planner_cov_y": 0.0},
            {"planner_cov_x": 1.0, "planner_cov_xy": 0.0, "planner_cov_y": 1.0},
        ],
    )
    run = {
        "belief_x": np.array([1.0, 1.0]),
        "belief_y": np.array([0.0, 0.0]),
        "truth_x": np.array([0.0, 0.0]),
        "truth_y": np.array([0.0, 0.0]),
    }

    out = belief_honesty(csv_path, run)

    assert out["nees_steps"] == 1
    assert out["nondefinite_cov_fraction"] == pytest.approx(0.5)


def test_misaligned_diagnostic_rows_fail_loudly(tmp_path: Path) -> None:
    csv_path = tmp_path / "experiment.csv"
    _write_csv(csv_path, [{"planner_cov_x": 1.0, "planner_cov_xy": 0.0, "planner_cov_y": 1.0}])
    run = {
        "belief_x": np.array([1.0, 2.0]),
        "belief_y": np.array([0.0, 0.0]),
        "truth_x": np.array([0.0, 0.0]),
        "truth_y": np.array([0.0, 0.0]),
    }

    with pytest.raises(AssertionError, match="cannot be aligned"):
        belief_honesty(csv_path, run)


def test_correction_path_counts_only_attempted_corrections(tmp_path: Path) -> None:
    csv_path = tmp_path / "experiment.csv"
    _write_csv(
        csv_path,
        [
            {"pixel_corr_nis": "", "pixel_corr_accepted": "", "planner_pixel_correction_age_s": "",
             "pixel_corr_nis_threshold": ""},
            {"pixel_corr_nis": 1.0, "pixel_corr_accepted": 1, "planner_pixel_correction_age_s": 0.2,
             "pixel_corr_nis_threshold": 9.21},
            {"pixel_corr_nis": 30.0, "pixel_corr_accepted": 0, "planner_pixel_correction_age_s": 0.6,
             "pixel_corr_nis_threshold": 9.21},
        ],
    )

    out = correction_path(csv_path)

    assert out["corrections_attempted"] == 2  # the blank row is not a rejected correction
    assert out["correction_accept_rate"] == pytest.approx(0.5)
    assert out["nis_threshold_observed"] == [9.21]


def test_matrix_gate_blocks_incomplete_and_mixed_threshold_matrices() -> None:
    comparison = {"n_pairs": 3, "unmatched_in_v2_only": [], "unmatched_in_v3_only": ["r/seed4"]}
    arms = {
        "clv2": {("r", 0): {"nis_threshold_observed": [9.21]}},
        "clv3": {("r", 0): {"nis_threshold_observed": [5.991]}},
    }

    problems = check_matrix(comparison, arms, expect_pairs=15)

    assert any("expected 15 matched pairs" in p for p in problems)
    assert any("unmatched_in_v3_only" in p for p in problems)
    assert any("NIS gate threshold" in p for p in problems)


def test_matrix_gate_passes_on_a_complete_matrix() -> None:
    comparison = {"n_pairs": 2, "unmatched_in_v2_only": [], "unmatched_in_v3_only": []}
    arms = {
        name: {
            ("r", 0): {"nis_threshold_observed": [9.21]},
            ("r", 1): {"nis_threshold_observed": [9.21]},
        }
        for name in ("clv2", "clv3")
    }

    assert check_matrix(comparison, arms, expect_pairs=2) == []


def test_bootstrap_interval_is_deterministic_and_brackets_the_point() -> None:
    pairs = [
        {"task": "r1", "seed": s, "d_belief_error_median_m": d,
         "d_belief_error_p95_m": math.nan, "d_final_goal_distance": math.nan,
         "d_nees_median": math.nan, "d_outside_95_ellipse_rate": math.nan,
         "d_correction_accept_rate": math.nan}
        for s, d in enumerate([-0.05, -0.03, -0.04, -0.06, -0.02])
    ]

    first = bootstrap_deltas(pairs, seed=0, n_boot=200)
    again = bootstrap_deltas(pairs, seed=0, n_boot=200)

    entry = first["belief_error_median_m"]
    assert entry["proportion_v3_better"] == pytest.approx(1.0)
    assert entry["ci_low"] <= entry["mean_delta"] <= entry["ci_high"]
    assert entry == again["belief_error_median_m"]
    # fields that are entirely NaN must be dropped, not reported as an empty interval
    assert "nees_median" not in first
