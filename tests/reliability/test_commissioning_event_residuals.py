"""Signed-localization-residual columns in the commissioning event adapter.

``build_actual_commissioning_inputs.py`` writes the per-camera GP event tables
(the spawn-grid schema). It used to carry no residual at all, so a per-camera
bias b_c(x) could not be fitted from its output. These tests pin the signed
2-vector that was added, the "empty, never 0.0" missing-data policy, and the
invariant that ground truth stays out of every model input.

All fixtures below are hand-written test data, not captures.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = (
    ROOT
    / "experiments"
    / "multicamera_commissioning_bigwarehouse"
    / "tools"
    / "build_actual_commissioning_inputs.py"
)
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import LeakageError  # noqa: E402
from reliability.firewall import validate_feature_columns  # noqa: E402


def _builder():
    spec = importlib.util.spec_from_file_location("build_actual_commissioning_inputs", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Schema as it stood before signed residuals were added. Existing artifacts
# parse against these names in this order; none may move or change meaning.
PRE_RESIDUAL_FIELDS = (
    "m_x",
    "m_y",
    "S_xx",
    "S_xy",
    "S_yy",
    "det_hit",
    "yolo_score_raw",
    "run_id",
    "camera_id",
    "observation_stamp_s",
    "odom_stamp_s",
    "odom_alignment_age_s",
    "state_covariance_source",
)

ODOM_COLUMNS = (
    "stamp",
    "odom_noisy_x",
    "odom_noisy_y",
    "odom_noisy_cov_xx",
    "odom_noisy_cov_xy",
    "odom_noisy_cov_yy",
)

PERCEPTION_COLUMNS = (
    "diag_stamp",
    "detected",
    "yolo_score_raw",
    "pred_world_x",
    "pred_world_y",
    "camera_id",
)
TRUTH_COLUMNS = PERCEPTION_COLUMNS + ("true_x", "true_y", "true_yaw")


def _write(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _perception_rows() -> list[dict[str, object]]:
    return [
        # detection with truth -> residual (+0.25, -0.10)
        {
            "diag_stamp": "10.000000000",
            "detected": 1,
            "yolo_score_raw": 0.9,
            "pred_world_x": 1.25,
            "pred_world_y": -2.10,
            "camera_id": "camera_A",
        },
        # no detection -> no projected point at all
        {
            "diag_stamp": "10.100000000",
            "detected": 0,
            "yolo_score_raw": 0.0,
            "pred_world_x": "",
            "pred_world_y": "",
            "camera_id": "camera_A",
        },
        # detection whose frame has no attached truth row
        {
            "diag_stamp": "10.200000000",
            "detected": 1,
            "yolo_score_raw": 0.7,
            "pred_world_x": 3.0,
            "pred_world_y": 4.0,
            "camera_id": "camera_A",
        },
    ]


def _truth_rows() -> list[dict[str, object]]:
    rows = []
    for row in _perception_rows()[:2]:
        out = dict(row)
        out["true_x"] = 1.0 if row["detected"] == 1 else ""
        out["true_y"] = -2.0 if row["detected"] == 1 else ""
        out["true_yaw"] = 0.0 if row["detected"] == 1 else ""
        rows.append(out)
    return rows


def _run_root(tmp_path: Path, *, attach_truth: bool = True) -> Path:
    run_root = tmp_path / "fixture_run_root"
    raw = run_root / "01_fixture_route" / "raw"
    _write(
        raw / "experiment.csv",
        ODOM_COLUMNS,
        [
            {
                "stamp": f"{10.0 + 0.05 * i:.9f}",
                "odom_noisy_x": 5.0,
                "odom_noisy_y": 6.0,
                "odom_noisy_cov_xx": 0.01,
                "odom_noisy_cov_xy": 0.0,
                "odom_noisy_cov_yy": 0.02,
            }
            for i in range(8)
        ],
    )
    _write(raw / "camera_A_perception.csv", PERCEPTION_COLUMNS, _perception_rows())
    if attach_truth:
        _write(
            run_root / "01_fixture_route" / "evaluation_inputs" / "camera_A_perception.csv",
            TRUTH_COLUMNS,
            _truth_rows(),
        )
    return run_root


def _events(tmp_path: Path, **kwargs):
    module = _builder()
    run_root = _run_root(tmp_path, attach_truth=kwargs.pop("attach_truth", True))
    raw_dir = run_root / "01_fixture_route" / "raw"
    return module, module._events_from_run(
        raw_dir, sigma_floor_m=0.02, max_alignment_age_s=0.15, **kwargs
    )


# --------------------------------------------------------------------- schema


def test_pre_existing_fields_are_untouched_and_still_lead_the_schema() -> None:
    module = _builder()
    assert module.EVENT_FIELDS[: len(PRE_RESIDUAL_FIELDS)] == PRE_RESIDUAL_FIELDS
    assert module.EVENT_FIELDS[len(PRE_RESIDUAL_FIELDS) :] == module.RESIDUAL_FIELDS
    assert module.RESIDUAL_FIELDS == (
        "eval_pred_world_x",
        "eval_pred_world_y",
        "eval_res_x",
        "eval_res_y",
        "eval_res_gt_source",
    )


def test_residual_fields_are_rejected_as_model_features_by_the_firewall() -> None:
    module = _builder()
    # The operational half of the schema is legitimate model input...
    validate_feature_columns(PRE_RESIDUAL_FIELDS)
    # ...and every residual column must be refused as one.
    for name in module.RESIDUAL_FIELDS:
        with pytest.raises(LeakageError, match=name):
            validate_feature_columns([*PRE_RESIDUAL_FIELDS, name])


# ------------------------------------------------------------- residual signs


def test_signed_residual_keeps_the_sign_of_each_axis(tmp_path: Path) -> None:
    _module, events = _events(tmp_path)
    rows = events["camera_A"]
    assert len(rows) == 3
    hit = rows[0]
    # pred (1.25, -2.10) - true (1.00, -2.00) = (+0.25, -0.10): opposite signs
    # on the two axes, which a scalar magnitude erases.
    assert hit["eval_pred_world_x"] == pytest.approx(1.25)
    assert hit["eval_pred_world_y"] == pytest.approx(-2.10)
    assert hit["eval_res_x"] == pytest.approx(0.25)
    assert hit["eval_res_y"] == pytest.approx(-0.10)
    assert hit["eval_res_gt_source"] == "attached_evaluation_true_xy"


def test_truth_join_is_exact_on_the_diag_stamp_text(tmp_path: Path) -> None:
    module = _builder()
    run_root = _run_root(tmp_path)
    truth = module._load_truth_by_stamp(
        run_root / "01_fixture_route" / "evaluation_inputs" / "camera_A_perception.csv"
    )
    # Only frames with finite truth are keyed; the blank non-detection row is
    # dropped rather than keyed to (0, 0).
    assert truth == {"10.000000000": (1.0, -2.0)}


# ---------------------------------------------------------------- missing data


def test_non_detection_leaves_every_residual_column_empty(tmp_path: Path) -> None:
    module, events = _events(tmp_path)
    miss = events["camera_A"][1]
    assert miss["det_hit"] == 0
    for name in module.RESIDUAL_FIELDS:
        assert miss[name] == "", f"{name} must be empty, not {miss[name]!r}"
        assert miss[name] != 0.0


def test_detection_without_attached_truth_keeps_pred_but_blanks_the_residual(
    tmp_path: Path,
) -> None:
    _module, events = _events(tmp_path)
    unmatched = events["camera_A"][2]
    assert unmatched["det_hit"] == 1
    assert unmatched["eval_pred_world_x"] == pytest.approx(3.0)
    assert unmatched["eval_pred_world_y"] == pytest.approx(4.0)
    assert unmatched["eval_res_x"] == ""
    assert unmatched["eval_res_y"] == ""
    assert unmatched["eval_res_gt_source"] == ""


def test_operational_only_run_stays_ground_truth_free(tmp_path: Path) -> None:
    """No attached-truth directory -> the columns exist but stay blank."""
    module, events = _events(tmp_path, attach_truth=False)
    for row in events["camera_A"]:
        for name in module.RESIDUAL_FIELDS:
            if name.startswith("eval_res"):
                assert row[name] == ""
    counts = module._residual_counts(events)
    assert counts["events_with_signed_residual"] == 0


def test_truth_lookup_can_be_disabled_explicitly(tmp_path: Path) -> None:
    module, events = _events(tmp_path, evaluation_truth_subdir="")
    for row in events["camera_A"]:
        assert row["eval_res_x"] == ""
    assert module._residual_counts(events)["events_with_signed_residual"] == 0


def test_a_true_zero_residual_is_still_written(tmp_path: Path) -> None:
    module = _builder()
    fields = module._residual_fields(
        {"diag_stamp": "10.0", "pred_world_x": "1.0", "pred_world_y": "-2.0"},
        {"10.0": (1.0, -2.0)},
        detected=True,
    )
    assert fields["eval_res_x"] == pytest.approx(0.0)
    assert fields["eval_res_y"] == pytest.approx(0.0)
    assert fields["eval_res_x"] != ""


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"diag_stamp": "10.0", "pred_world_x": "", "pred_world_y": ""}, id="empty"),
        pytest.param(
            {"diag_stamp": "10.0", "pred_world_x": "nan", "pred_world_y": "nan"}, id="nan"
        ),
        pytest.param({"diag_stamp": "10.0", "pred_world_y": "-2.0"}, id="half_missing"),
    ],
)
def test_non_finite_prediction_blanks_everything(row: dict[str, str]) -> None:
    module = _builder()
    fields = module._residual_fields(row, {"10.0": (1.0, -2.0)}, detected=True)
    assert fields == {name: "" for name in module.RESIDUAL_FIELDS}


# ------------------------------------------------------- unchanged behaviour


def test_operational_event_values_are_unchanged(tmp_path: Path) -> None:
    _module, events = _events(tmp_path)
    hit = events["camera_A"][0]
    assert hit["m_x"] == pytest.approx(5.0)
    assert hit["m_y"] == pytest.approx(6.0)
    assert hit["det_hit"] == 1
    assert hit["yolo_score_raw"] == pytest.approx(0.9)
    assert hit["camera_id"] == "camera_A"
    assert hit["run_id"] == "01_fixture_route"
    assert hit["observation_stamp_s"] == pytest.approx(10.0)
    assert hit["state_covariance_source"] == "propagated_odom_covariance_plus_floor_and_alignment"
    # Covariance = recorded + floor^2 + (0.10 * alignment age)^2, untouched by
    # anything the residual block does.
    assert hit["S_xx"] == pytest.approx(0.01 + 0.02**2)
    assert hit["S_yy"] == pytest.approx(0.02 + 0.02**2)
    assert hit["S_xy"] == pytest.approx(0.0)


def test_written_csv_carries_the_new_columns_and_blanks_not_zeros(tmp_path: Path) -> None:
    module, events = _events(tmp_path)
    out = tmp_path / "camera_A_events.csv"
    module._write_events(out, events["camera_A"])
    with out.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(module.EVENT_FIELDS)
        rows = list(reader)
    assert float(rows[0]["eval_res_x"]) == pytest.approx(0.25)
    assert float(rows[0]["eval_res_y"]) == pytest.approx(-0.10)
    assert rows[1]["eval_res_x"] == ""
    assert rows[2]["eval_res_x"] == ""


def test_existing_twelve_column_artifacts_still_parse() -> None:
    """Locked spawn-grid tables predate these columns; readers must tolerate that."""
    module = _builder()
    legacy_header = PRE_RESIDUAL_FIELDS[:-1]  # no state_covariance_source either
    for name in legacy_header:
        assert name in module.EVENT_FIELDS
    legacy_row = dict.fromkeys(legacy_header, "0")
    fields = module._residual_fields(legacy_row, {}, detected=False)
    assert fields == {name: "" for name in module.RESIDUAL_FIELDS}


def test_residual_counts_audit(tmp_path: Path) -> None:
    module, events = _events(tmp_path)
    counts = module._residual_counts(events)
    assert counts["detection_events"] == 2
    assert counts["events_with_signed_residual"] == 1
    assert counts["by_camera"]["camera_A"] == 1
    assert counts["by_camera"]["camera_B"] == 0
