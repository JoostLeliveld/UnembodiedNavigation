"""Real-data smoke test for the Paper-2 containment harness.

`experiments/multicamera_fusion_extension/tools/run_containment_pilot.py` wires the
validated bridge to the E4 subset sweep + Δ_fault fault-injection. This runs it
END-TO-END on a real capture fixture and asserts the harness produces a finite
fusion gain and structured, finite Δ_fault values — the plumbing that lets the
Paper-2 R3/R4 tables fill from a real handover capture.

Numbers here are detector-limited PLUMBING, not evidence. Skips cleanly when no
real capture fixture is on disk (never fabricates data).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
sys.path.insert(0, str(ROOT / "experiments" / "multicamera_fusion_extension" / "tools"))

import run_containment_pilot as pilot  # noqa: E402

_CANDIDATES = (
    "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke2_20260716",
    "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716",
)


def _fixture() -> Path | None:
    for rel in _CANDIDATES:
        cand = ROOT / rel
        if (cand / "raw" / "experiment.csv").is_file() and (
            cand / "evaluation_only" / "ground_truth.csv"
        ).is_file():
            return cand
    return None


FIX = _fixture()
requires_fixture = pytest.mark.skipif(FIX is None, reason="no real commissioning capture fixture on disk")


@pytest.fixture(scope="module")
def res():
    return pilot.run_pilot(FIX, bias_levels_m=(0.2, 0.5))


@requires_fixture
def test_subset_sweep_runs_and_reports_gain(res):
    sub = res["subset"]
    assert len(res["cameras"]) >= 2
    assert math.isfinite(sub["best_single_p95_m"])
    assert math.isfinite(sub["full_set_p95_m"])
    assert math.isfinite(sub["fusion_gain_p95_m"])
    # at least every single-camera subset scored; full set present
    assert len(sub["per_subset_p95_m"]) >= len(res["cameras"])
    full_key = "|".join(sub["full_set"])
    assert full_key in sub["per_subset_p95_m"]


@requires_fixture
def test_delta_fault_structured_and_finite(res):
    assert set(res["delta_fault"]) == set(res["cameras"])
    for _cam, rows in res["delta_fault"].items():
        assert len(rows) == 2  # two bias levels requested
        for row in rows:
            assert math.isfinite(row["p95_dropped_m"])
            assert math.isfinite(row["p95_with_bad_m"])
            assert math.isfinite(row["delta_fault_m"])
            # dropped baseline is bias-independent -> identical across a camera's rows
        dropped = {row["p95_dropped_m"] for row in rows}
        assert len(dropped) == 1


@requires_fixture
def test_healthy_full_set_error_is_sane(res):
    # real belief-vs-GT error is small (campaign-metrics: p95 ~0.13 m); a heavy tail
    # would signal the bridge grabbed a stale/wrong position column.
    assert 0.0 < res["p95_healthy_full_m"] < 1.0
