"""Contract tests for the longitudinal ground-anchor evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "experiments/usable_observation/supervisor_comparison"
    / "10_monocular_depth_results/temporal_anchor_sequence.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_temporal_anchor_sequence_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_frozen_sequence_is_exactly_21_updates_by_four_cameras():
    module = _load_module()
    records = module._records()
    assert len(records) == 84
    assert sorted({row["camera_id"] for row in records}) == sorted(module.CAMERAS)
    assert tuple(sorted({float(row["timestamp"]) for row in records})) == module.SOURCE_TIMES


def test_dropout_schedule_is_predeclared_and_isolated():
    module = _load_module()
    indices = [module.SOURCE_TIMES.index(timestamp) for timestamp in module.DROPOUT_TIMES]
    assert indices == [5, 10, 15, 20]
    assert all(right - left >= 5 for left, right in zip(indices, indices[1:]))
    assert module.UPDATE_PERIOD_S == 10.0


def test_visibility_metric_ignores_outside_and_occupied_cells():
    module = _load_module()
    oracle = np.array([[1, 0, 2, 3]], dtype=np.uint8)
    probability = np.array([[0.9, 0.1, 0.9, 0.9]], dtype=float)
    metrics = module._visibility_metrics(probability, oracle)
    assert metrics["n_cells"] == 2
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["visible_iou"] == 1.0


def test_zero_denominator_ratio_is_explicitly_undefined():
    module = _load_module()
    assert module._safe_ratio(1.0, 0.0) is None
    assert module._safe_ratio(1.0, 2.0) == 0.5
