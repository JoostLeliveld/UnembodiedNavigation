"""Regression guard for the canonical ``auprc`` added to scripts/shared/metrics.py.

AUPRC (average precision) is the E1 rare-positive scorer. It must match the
sklearn ``average_precision_score`` step definition, including tie handling.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "shared"))

import metrics as M  # noqa: E402


def test_auprc_matches_reference_value() -> None:
    # sklearn average_precision_score([0,0,1,1], [0.1,0.4,0.35,0.8]) == 0.8333...
    assert M.auprc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(5.0 / 6.0, abs=1e-9)


def test_auprc_perfect_ranking_is_one() -> None:
    assert M.auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auprc_all_ties_equals_base_rate() -> None:
    # With all scores equal, precision is the base rate everywhere.
    assert M.auprc([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert M.auprc([0, 0, 0, 1], [0.3, 0.3, 0.3, 0.3]) == pytest.approx(0.25)


def test_auprc_nan_without_positives() -> None:
    assert math.isnan(M.auprc([0, 0, 0], [0.2, 0.5, 0.9]))


def test_auprc_registered_in_all() -> None:
    assert "auprc" in M.__all__
