from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import LeakageError  # noqa: E402
from reliability.firewall import (  # noqa: E402
    assert_operational_feature_table,
    scan_import_targets,
    validate_config_sources,
    validate_feature_columns,
    validate_planner_facing_import_paths,
    validate_planner_facing_imports,
    validate_training_loader_sources,
)


def test_feature_columns_reject_ground_truth_and_outcome_fields() -> None:
    validate_feature_columns(
        [
            "detected",
            "selected_score",
            "pixel_pose_age_s",
            "planner_belief_x",
            "camera_relative_range_m",
        ]
    )

    with pytest.raises(LeakageError, match="gt_x"):
        validate_feature_columns(["detected", "gt_x"])
    with pytest.raises(LeakageError, match="collision_any"):
        validate_feature_columns(["selected_score", "collision_any"])
    with pytest.raises(LeakageError, match="localization_error"):
        validate_feature_columns(["state_localization_error_m"])


def test_training_loader_sources_reject_gt_topics_and_paths() -> None:
    validate_training_loader_sources(
        [
            "/perception/pixel_pose",
            "/perception/detection_diagnostics",
            "logs/experiments/experiment_00123/perception.csv",
        ]
    )

    with pytest.raises(LeakageError, match="/ground_truth_tf"):
        validate_training_loader_sources(["/perception/pixel_pose", "/ground_truth_tf"])
    with pytest.raises(LeakageError, match="ground_truth"):
        validate_training_loader_sources(["logs/export/ground_truth_labels.csv"])


def test_evaluation_contexts_may_name_oracle_sources() -> None:
    validate_training_loader_sources(
        ["/ground_truth_tf", "logs/export/oracle_visibility.csv"],
        context="evaluation",
    )


def test_operational_feature_table_combines_column_and_source_checks() -> None:
    assert_operational_feature_table(
        ["detected", "selected_score", "measurement_age_s"],
        ["logs/export/operational_features.csv"],
    )

    with pytest.raises(LeakageError, match="oracle_visible"):
        assert_operational_feature_table(["detected", "oracle_visible"], ["logs/export/features.csv"])
    with pytest.raises(LeakageError, match="ground_truth"):
        assert_operational_feature_table(["detected"], ["logs/export/ground_truth_features.csv"])


def test_planner_facing_imports_reject_evaluation_modules(tmp_path: Path) -> None:
    validate_planner_facing_imports(["reliability.contracts", "reliability.firewall"])

    with pytest.raises(LeakageError, match="evaluation"):
        validate_planner_facing_imports(["reliability.evaluation.loader"])
    with pytest.raises(LeakageError, match="reliability.oracle"):
        validate_planner_facing_imports(["reliability.oracle"])

    module = tmp_path / "provider.py"
    module.write_text(
        "from reliability.contracts import PlanningCovariance\n"
        "import reliability.evaluation_only\n",
        encoding="utf-8",
    )

    assert "reliability.evaluation_only" in scan_import_targets([module])
    with pytest.raises(LeakageError, match="evaluation_only"):
        validate_planner_facing_import_paths([module])


def test_new_reliability_package_has_no_planner_facing_eval_imports() -> None:
    package_files = sorted((ROOT / "src" / "reliability" / "reliability").glob("*.py"))
    validate_planner_facing_import_paths(package_files)


def test_normal_runtime_config_rejects_truth_state_or_reliability_sources() -> None:
    validate_config_sources(
        {
            "use_truth_localization": False,
            "state_source_x": "yolo_mask_or_bbox_homography",
            "state_source_y": "yolo_mask_or_bbox_homography",
            "reliability_source": "yolo_diagnostics",
        }
    )
    validate_config_sources({"use_truth_localization": "false"})

    with pytest.raises(LeakageError, match="use_truth_localization"):
        validate_config_sources({"use_truth_localization": True})
    with pytest.raises(LeakageError, match="state_source_x"):
        validate_config_sources({"state_source_x": "ground_truth"})
    with pytest.raises(LeakageError, match="reliability_source"):
        validate_config_sources({"reliability_source": "oracle_visibility"})


def test_active_warehouse_visibility_campaign_is_gt_free_for_runtime_sources() -> None:
    config_path = ROOT / "scripts" / "visibility_comparison" / "warehouse_visibility_campaign.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    validate_config_sources(payload)
    assert payload.get("yolo_use_masks") is False
    assert payload["perception_backend"] == "yolo"
