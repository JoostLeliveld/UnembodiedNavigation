from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INFERENCE = _load(
    "reconfiguration_e1_inference",
    "experiments/reconfiguration_holdout/e1_reconfiguration_holdout/summarize_inference.py",
)
TRANSFER = _load(
    "reconfiguration_gp_transfer",
    "experiments/reconfiguration_holdout/gp_transfer_refit.py",
)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    assert INFERENCE.holm([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_raw_brier_interaction_uses_paired_arm_degradations(tmp_path: Path) -> None:
    path = tmp_path / "units.csv"
    rows = []
    # GP degrades by +.10 and mono by +.02 in every unit: interaction +.08.
    for camera in ("A", "B"):
        for fold in range(3):
            for arm, l0, l1 in (("gp", 0.10, 0.20), ("mono_depth", 0.08, 0.10),
                                ("hybrid", 0.09, 0.15)):
                rows.extend([
                    {"arm": arm, "environment": "L0", "camera": camera,
                     "fold": fold, "brier": l0},
                    {"arm": arm, "environment": "L1", "camera": camera,
                     "fold": fold, "brier": l1},
                ])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = INFERENCE.summarize(path, "L1")
    assert len(result) == 2
    assert result[0]["comparison"].startswith("F1_")
    assert result[0]["mean_interaction"] == pytest.approx(0.08)
    assert result[1]["mean_interaction"] == pytest.approx(0.04)


def _valid_membership_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    blocks = [0, 0, 1, 1, 2, 2]
    for event_index, block in enumerate(blocks):
        rows.append({
            "environment": "L0", "camera": "A", "outer_fold": block,
            "block": block, "role": "test", "event_index": event_index,
        })
        rows.append({
            "environment": "L1", "camera": "A", "outer_fold": block,
            "block": block, "role": "test", "event_index": event_index,
        })
        for outer in range(3):
            if outer != block:
                rows.append({
                    "environment": "L0", "camera": "A", "outer_fold": outer,
                    "block": block, "role": "train_oos", "event_index": event_index,
                })
    return rows


def test_transfer_membership_requires_each_test_once_and_each_link_event_n_minus_one() -> None:
    TRANSFER.validate_camera_rows(
        _valid_membership_rows(), camera="A", event_counts={"L0": 6, "L1": 6},
        n_blocks=3)


def test_transfer_membership_rejects_outer_block_in_link_training() -> None:
    rows = _valid_membership_rows()
    link_row = next(row for row in rows if row["role"] == "train_oos")
    link_row["outer_fold"] = link_row["block"]
    with pytest.raises(RuntimeError, match="test-block event entered link training"):
        TRANSFER.validate_camera_rows(
            rows, camera="A", event_counts={"L0": 6, "L1": 6}, n_blocks=3)
