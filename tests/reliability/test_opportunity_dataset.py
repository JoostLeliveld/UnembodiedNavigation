from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import LeakageError, OperationalReliabilitySample  # noqa: E402
from reliability.opportunity import (  # noqa: E402
    LOOReference,
    OpportunityConfig,
    OpportunityPrediction,
    build_opportunity_row,
    label_loo_usability,
)


def _sample(*, detected: bool = True, age_s: float = 0.02) -> OperationalReliabilitySample:
    return OperationalReliabilitySample(
        sample_id="run:camera_A:000001",
        timestamp_s=10.0,
        measurement_age_s=age_s,
        selected_pixel=(320.0, 240.0) if detected else None,
        detector_result={"detected": detected, "raw_score": 0.7},
        run_id="run",
        belief={"x_m": 1.0, "y_m": 0.0, "covariance_xy_m2": [[0.04, 0.0], [0.0, 0.04]]},
        metadata={"camera_id": "camera_A"},
    )


def _prediction(**overrides: object) -> OpportunityPrediction:
    payload = {
        "sample_id": "run:camera_A:000001",
        "camera_id": "camera_A",
        "predicted_uv": (320.0, 240.0),
        "predicted_cov_uv": ((4.0, 0.0), (0.0, 4.0)),
        "predicted_height_px": 32.0,
        "stream_live": True,
        "association_delta_s": 0.01,
    }
    payload.update(overrides)
    return OpportunityPrediction(**payload)


def _config() -> OpportunityConfig:
    return OpportunityConfig(image_width_px=640, image_height_px=480)


def test_opportunity_emits_detection_and_miss_rows_without_truth() -> None:
    hit = build_opportunity_row(_sample(detected=True), _prediction(), config=_config())
    miss = build_opportunity_row(_sample(detected=False), _prediction(), config=_config())

    assert hit is not None and hit.availability_label == 1 and hit.association_valid
    assert miss is not None and miss.availability_label == 0 and not miss.association_valid
    assert "gt" not in hit.to_dict()


def test_robot_outside_valid_support_is_not_an_availability_failure() -> None:
    result = build_opportunity_row(
        _sample(detected=False),
        _prediction(predicted_uv=(900.0, 240.0)),
        config=_config(),
    )

    assert result is None


def test_stale_stream_and_low_scale_are_not_opportunities() -> None:
    assert build_opportunity_row(_sample(age_s=2.0), _prediction(), config=_config()) is None
    assert build_opportunity_row(_sample(), _prediction(predicted_height_px=4.0), config=_config()) is None


def test_loo_label_requires_reference_that_excludes_labelled_camera() -> None:
    row = build_opportunity_row(_sample(), _prediction(), config=_config())
    assert row is not None
    good = LOOReference(
        sample_id=row.sample_id,
        excluded_camera_id="camera_A",
        reference_uv=(322.0, 240.0),
        reference_cov_uv=((1.0, 0.0), (0.0, 1.0)),
    )
    labelled = label_loo_usability(row, good, max_residual_px=5.0)
    assert labelled["usable_label"] == 1
    assert labelled["e_loo_uv"] == [-2.0, 0.0]

    wrong = LOOReference(
        sample_id=row.sample_id,
        excluded_camera_id="camera_B",
        reference_uv=(322.0, 240.0),
        reference_cov_uv=((1.0, 0.0), (0.0, 1.0)),
    )
    with pytest.raises(LeakageError, match="does not exclude"):
        label_loo_usability(row, wrong, max_residual_px=5.0)


def test_sidecars_reject_evaluation_only_fields() -> None:
    with pytest.raises(LeakageError, match="ground_truth"):
        OpportunityPrediction.from_dict({
            **_prediction().to_dict(),
            "ground_truth_projected_pixel": [1.0, 2.0],
        })


def test_tools_build_and_label_operational_csv(tmp_path: Path) -> None:
    def load_tool(name: str):
        path = ROOT / "experiments" / "multicamera_fusion_extension" / "tools" / name
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    opportunity_tool = load_tool("build_opportunity_dataset.py")
    loo_tool = load_tool("build_loo_labels.py")
    gp_tool = load_tool("train_factorized_gp.py")
    operational = tmp_path / "operational_reliability.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    operational.write_text(json.dumps(_sample().to_dict()) + "\n", encoding="utf-8")
    predictions.write_text(json.dumps(_prediction().to_dict()) + "\n", encoding="utf-8")
    opportunities = tmp_path / "opportunities.csv"
    summary = opportunity_tool.build_dataset(
        operational_path=operational,
        prediction_path=predictions,
        output_path=opportunities,
        config=_config(),
    )
    assert summary == {"operational_samples": 1, "opportunities": 1}

    refs = tmp_path / "references.jsonl"
    refs.write_text(json.dumps(LOOReference(
        sample_id="run:camera_A:000001",
        excluded_camera_id="camera_A",
        reference_uv=(321.0, 240.0),
        reference_cov_uv=((1.0, 0.0), (0.0, 1.0)),
    ).to_dict()) + "\n", encoding="utf-8")
    labelled = tmp_path / "labelled.csv"
    assert loo_tool.label_dataset(
        opportunity_csv=opportunities,
        reference_jsonl=refs,
        output_csv=labelled,
        max_residual_px=5.0,
    ) == {"opportunities": 1, "usable": 1}
    with labelled.open(newline="", encoding="utf-8") as handle:
        labelled_row = list(csv.DictReader(handle))[0]
    assert labelled_row["usable_label"] == "1"
    assert gp_tool.prepare_events([labelled_row], camera_id="camera_A", kind="availability") == [{
        "m_x": 1.0,
        "m_y": 0.0,
        "S_xx": 0.04,
        "S_xy": 0.0,
        "S_yy": 0.04,
        "det_hit": 1,
        "yolo_score_raw": 0.7,
        "run_id": "run",
        "camera_id": "camera_A",
    }]
