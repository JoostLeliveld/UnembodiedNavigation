"""Signed-localization-residual columns in the belief GP event builder.

The pipeline used to carry only ``localization_error_captime_m``, a scalar
MAGNITUDE. A magnitude cannot identify a bias direction, so a per-camera bias
b_c(x) was unfittable. These tests pin the signed 2-vector that replaced that
gap, its missing-data policy, and the fact that no pre-existing column moved.

All fixtures below are hand-written test data, not captures.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts" / "visibility_comparison" / "build_belief_gp_events.py"


def _builder():
    # The script imports its sibling `common` module by flat name.
    sys.path.insert(0, str(BUILDER.parent))
    spec = importlib.util.spec_from_file_location("build_belief_gp_events", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Column list as it stood before signed residuals were added. Nothing here may
# move, be renamed, or change meaning: locked artifacts and campaign CSVs
# parse against it.
PRE_RESIDUAL_COLUMNS = (
    "event_id",
    "run_dir",
    "route",
    "condition",
    "seed",
    "run_id",
    "diag_stamp",
    "log_stamp",
    "matched_experiment_stamp",
    "stamp_delta_s",
    "m_x",
    "m_y",
    "S_xx",
    "S_xy",
    "S_yy",
    "sigma_major_m",
    "sigma_minor_m",
    "trace_S_xy",
    "det_hit",
    "yolo_score_raw",
    "yolo_detected_after_threshold",
    "pixel_pose_available",
    "pixel_pose_fresh",
    "localization_error_captime_m",
    "state_source",
    "eval_gt_x",
    "eval_gt_y",
    "eval_belief_error_gt_m",
)

EXPERIMENT_COLUMNS = (
    "stamp",
    "planner_belief_available",
    "planner_belief_x",
    "planner_belief_y",
    "planner_cov_x",
    "planner_cov_xy",
    "planner_cov_y",
    "gt_available",
    "gt_x",
    "gt_y",
    "belief_error_gt_m",
)

PERCEPTION_COLUMNS = (
    "diag_stamp",
    "log_stamp",
    "detected",
    "true_available",
    "true_x",
    "true_y",
    "pred_world_x",
    "pred_world_y",
    "localization_error_m",
    "localization_error_captime_m",
    "yolo_score_raw",
    "yolo_detected_after_threshold",
    "pixel_pose_available",
    "pixel_pose_fresh",
)


def _experiment_row(stamp: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "stamp": stamp,
        "planner_belief_available": 1,
        "planner_belief_x": 2.0,
        "planner_belief_y": 3.0,
        "planner_cov_x": 0.04,
        "planner_cov_xy": 0.0,
        "planner_cov_y": 0.09,
        "gt_available": 1,
        "gt_x": 2.5,
        "gt_y": 3.5,
        "belief_error_gt_m": 0.7071067812,
    }
    row.update(overrides)
    return row


def _perception_row(stamp: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "diag_stamp": stamp,
        "log_stamp": stamp,
        "detected": 1,
        "true_available": 1,
        "true_x": 1.0,
        "true_y": -2.0,
        "pred_world_x": 1.25,
        "pred_world_y": -2.10,
        "localization_error_m": math.hypot(0.25, -0.10),
        "localization_error_captime_m": 0.2,
        "yolo_score_raw": 0.8,
        "yolo_detected_after_threshold": 1,
        "pixel_pose_available": 1,
        "pixel_pose_fresh": 1,
    }
    row.update(overrides)
    return row


def _write(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_tree(
    tmp_path: Path,
    per_rows: list[dict[str, object]],
    exp_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    campaign = tmp_path / "fixture_campaign"
    run_dir = campaign / "route_x" / "C2" / "seed0" / "experiment_00000000_000000"
    _write(run_dir / "perception.csv", PERCEPTION_COLUMNS, per_rows)
    _write(run_dir / "experiment.csv", EXPERIMENT_COLUMNS, exp_rows)
    return campaign, run_dir


def _extract(tmp_path: Path, per_rows, exp_rows):
    module = _builder()
    # The builder records run dirs relative to the repo root; the fixture tree
    # lives in tmp_path, so point that anchor at it.
    module.REPO_ROOT = tmp_path
    campaign, run_dir = _run_tree(tmp_path, per_rows, exp_rows)
    return module, module._extract_run_events(
        campaign,
        run_dir,
        event_start=0,
        stamp_key="log_stamp",
        stamp_tolerance_s=0.3,
        require_belief_available=True,
    )


# --------------------------------------------------------------------- schema


def test_pre_existing_columns_are_untouched_and_still_lead_the_schema() -> None:
    module = _builder()
    assert module.EVENT_COLUMNS[: len(PRE_RESIDUAL_COLUMNS)] == PRE_RESIDUAL_COLUMNS
    assert module.EVENT_COLUMNS[len(PRE_RESIDUAL_COLUMNS) :] == module.RESIDUAL_COLUMNS


def test_residual_columns_are_declared_evaluation_only() -> None:
    module = _builder()
    assert module.RESIDUAL_COLUMNS == (
        "eval_pred_world_x",
        "eval_pred_world_y",
        "eval_res_x",
        "eval_res_y",
        "eval_res_gt_source",
    )
    # The eval_ prefix is the repo's evaluation-only marker; every residual
    # column must carry it, and the derived audit list must agree.
    assert all(name.startswith("eval_") for name in module.RESIDUAL_COLUMNS)
    assert set(module.RESIDUAL_COLUMNS) <= set(module.EVALUATION_ONLY_COLUMNS)
    assert module.EVALUATION_ONLY_COLUMNS == tuple(
        name for name in module.EVENT_COLUMNS if name.startswith("eval_")
    )


def test_new_columns_are_not_gp_inputs_anywhere_in_the_fitter_or_manifest() -> None:
    module = _builder()
    fitter = (ROOT / "scripts" / "visibility_comparison" / "fit_belief_aware_gp.py").read_text(
        encoding="utf-8"
    )
    for name in module.RESIDUAL_COLUMNS:
        assert name not in fitter, f"canonical GP fitter references eval-only column {name}"


def test_every_eval_column_is_refused_as_a_model_feature_by_the_firewall() -> None:
    sys.path.insert(0, str(ROOT / "src" / "reliability"))
    from reliability.contracts import LeakageError
    from reliability.firewall import validate_feature_columns

    module = _builder()
    operational = [
        name for name in module.EVENT_COLUMNS
        if not name.startswith("eval_") and "localization_error" not in name
    ]
    validate_feature_columns(operational)
    for name in module.EVALUATION_ONLY_COLUMNS:
        with pytest.raises(LeakageError, match=name):
            validate_feature_columns([*operational, name])


# ------------------------------------------------------------- residual signs


def test_signed_residual_keeps_the_sign_of_each_axis(tmp_path: Path) -> None:
    module, (rows, _counts) = _extract(
        tmp_path,
        [_perception_row(10.0)],
        [_experiment_row(10.0)],
    )
    assert len(rows) == 1
    row = rows[0]
    # pred (1.25, -2.10) - true (1.00, -2.00) = (+0.25, -0.10): the axes carry
    # OPPOSITE signs, which the old scalar magnitude erased entirely.
    assert float(row["eval_pred_world_x"]) == pytest.approx(1.25)
    assert float(row["eval_pred_world_y"]) == pytest.approx(-2.10)
    assert float(row["eval_res_x"]) == pytest.approx(0.25)
    assert float(row["eval_res_y"]) == pytest.approx(-0.10)
    assert row["eval_res_gt_source"] == "perception_true_xy"


def test_residual_norm_reproduces_the_logged_scalar_error(tmp_path: Path) -> None:
    """hypot(signed residual) must equal the magnitude the logger already wrote."""
    per = _perception_row(10.0)
    _module, (rows, _counts) = _extract(tmp_path, [per], [_experiment_row(10.0)])
    row = rows[0]
    norm = math.hypot(float(row["eval_res_x"]), float(row["eval_res_y"]))
    assert norm == pytest.approx(float(per["localization_error_m"]), abs=1e-9)


def test_residual_sign_flips_with_the_prediction(tmp_path: Path) -> None:
    _module, (rows, _counts) = _extract(
        tmp_path,
        [_perception_row(10.0, pred_world_x=0.75, pred_world_y=-1.90)],
        [_experiment_row(10.0)],
    )
    assert float(rows[0]["eval_res_x"]) == pytest.approx(-0.25)
    assert float(rows[0]["eval_res_y"]) == pytest.approx(0.10)


def test_experiment_gt_is_the_labelled_fallback_when_perception_truth_is_absent(
    tmp_path: Path,
) -> None:
    _module, (rows, _counts) = _extract(
        tmp_path,
        [_perception_row(10.0, true_x="", true_y="", true_available="")],
        [_experiment_row(10.0, gt_x=1.0, gt_y=-2.0)],
    )
    row = rows[0]
    assert float(row["eval_res_x"]) == pytest.approx(0.25)
    assert float(row["eval_res_y"]) == pytest.approx(-0.10)
    # The two GT time bases must stay distinguishable: perception truth is
    # same-row, experiment GT is a nearest-stamp join.
    assert row["eval_res_gt_source"] == "experiment_gt_xy"


# ---------------------------------------------------------------- missing data


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"detected": 0}, id="no_detection"),
        pytest.param({"pred_world_x": "", "pred_world_y": ""}, id="pred_world_missing"),
        pytest.param({"pred_world_x": "nan", "pred_world_y": "nan"}, id="pred_world_nan"),
        pytest.param({"pred_world_y": ""}, id="pred_world_half_missing"),
    ],
)
def test_missing_prediction_blanks_every_residual_column_never_zero(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    module, (rows, _counts) = _extract(
        tmp_path,
        [_perception_row(10.0, **overrides)],
        [_experiment_row(10.0)],
    )
    row = rows[0]
    for name in module.RESIDUAL_COLUMNS:
        assert row[name] == "", f"{name} must be empty, not {row[name]!r}"
        # A 0.0 here would read to a bias fit as "measured, and unbiased".
        assert row[name] != "0" and row[name] != "0.0"


@pytest.mark.parametrize(
    "per_overrides, exp_overrides",
    [
        pytest.param(
            {"true_x": "", "true_y": ""}, {"gt_x": "", "gt_y": ""}, id="no_truth_anywhere"
        ),
        pytest.param(
            {"true_available": 0}, {"gt_available": 0}, id="truth_flagged_unavailable"
        ),
        pytest.param(
            {"true_x": "nan", "true_y": "nan"}, {"gt_x": "nan", "gt_y": "nan"}, id="truth_nan"
        ),
    ],
)
def test_missing_ground_truth_keeps_the_prediction_but_blanks_the_residual(
    tmp_path: Path, per_overrides: dict[str, object], exp_overrides: dict[str, object]
) -> None:
    _module, (rows, _counts) = _extract(
        tmp_path,
        [_perception_row(10.0, **per_overrides)],
        [_experiment_row(10.0, **exp_overrides)],
    )
    row = rows[0]
    # pred_world is an operational measurement, so it survives...
    assert float(row["eval_pred_world_x"]) == pytest.approx(1.25)
    assert float(row["eval_pred_world_y"]) == pytest.approx(-2.10)
    # ...but with no truth there is no residual, and it must not be faked.
    assert row["eval_res_x"] == ""
    assert row["eval_res_y"] == ""
    assert row["eval_res_gt_source"] == ""


def test_a_true_zero_residual_is_still_written(tmp_path: Path) -> None:
    """Empty means "not measured"; a genuinely zero residual stays numeric."""
    _module, (rows, _counts) = _extract(
        tmp_path,
        [_perception_row(10.0, pred_world_x=1.0, pred_world_y=-2.0)],
        [_experiment_row(10.0)],
    )
    row = rows[0]
    assert row["eval_res_x"] != ""
    assert float(row["eval_res_x"]) == pytest.approx(0.0)
    assert float(row["eval_res_y"]) == pytest.approx(0.0)


# ------------------------------------------------------- unchanged behaviour


def test_existing_column_values_and_skip_bookkeeping_are_unchanged(tmp_path: Path) -> None:
    per_rows = [
        _perception_row(10.0),
        _perception_row(11.0, detected=0, pred_world_x="", pred_world_y=""),
        _perception_row(12.0, detected=""),          # non-binary -> skipped
        _perception_row(13.0, log_stamp="nan"),      # bad stamp  -> skipped
        _perception_row(99.0),                       # outside tolerance -> skipped
    ]
    exp_rows = [_experiment_row(10.0), _experiment_row(11.0), _experiment_row(12.0), _experiment_row(13.0)]
    _module, (rows, counts) = _extract(tmp_path, per_rows, exp_rows)

    assert counts == {
        "perception_rows": 5,
        "skipped_no_binary_detection": 1,
        "skipped_bad_stamp": 1,
        "skipped_stamp_tolerance": 1,
        "skipped_missing_belief": 0,
        "skipped_missing_covariance": 0,
    }
    assert [row["event_id"] for row in rows] == ["belief_event_00000000", "belief_event_00000001"]
    hit = rows[0]
    assert hit["route"] == "route_x"
    assert hit["condition"] == "C2"
    assert hit["seed"] == "seed0"
    assert hit["det_hit"] == "1"
    assert hit["state_source"] == "BELIEF"
    assert float(hit["m_x"]) == pytest.approx(2.0)
    assert float(hit["m_y"]) == pytest.approx(3.0)
    assert float(hit["S_xx"]) == pytest.approx(0.04)
    assert float(hit["S_yy"]) == pytest.approx(0.09)
    assert float(hit["localization_error_captime_m"]) == pytest.approx(0.2)
    assert float(hit["eval_gt_x"]) == pytest.approx(2.5)
    assert float(hit["eval_gt_y"]) == pytest.approx(3.5)
    # And the non-detection row still yields an event, just without residuals.
    assert rows[1]["det_hit"] == "0"
    assert rows[1]["eval_res_x"] == ""


def test_missing_belief_and_covariance_skips_still_fire(tmp_path: Path) -> None:
    per_rows = [_perception_row(10.0), _perception_row(11.0)]
    exp_rows = [
        _experiment_row(10.0, planner_belief_available=0),
        _experiment_row(11.0, planner_cov_x=0.0, planner_cov_y=0.0),
    ]
    _module, (rows, counts) = _extract(tmp_path, per_rows, exp_rows)
    assert rows == []
    assert counts["skipped_missing_belief"] == 1
    assert counts["skipped_missing_covariance"] == 1


def test_residual_coverage_audit_counts_only_finite_pairs(tmp_path: Path) -> None:
    module, (rows, _counts) = _extract(
        tmp_path,
        [
            _perception_row(10.0),
            _perception_row(11.0, detected=0, pred_world_x="", pred_world_y=""),
            _perception_row(12.0, true_x="", true_y=""),
        ],
        [_experiment_row(10.0), _experiment_row(11.0), _experiment_row(12.0)],
    )
    coverage = module._residual_coverage(rows)
    assert coverage["detection_events"] == 2
    assert coverage["events_with_signed_residual"] == 2
    assert set(coverage["gt_source_counts"]) == {"perception_true_xy", "experiment_gt_xy"}
    assert math.isfinite(coverage["mean_eval_res_x"])
