from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_campaign_is_exact_six_by_two_by_five():
    campaign = yaml.safe_load((
        REPO / "pipeline/route_planning_template.yaml"
    ).read_text())
    assert campaign["scheduled_rate_hz"] == 4.0
    assert campaign["pixel_timeout_s"] == campaign["manager_max_measurement_age_s"] == 1.25
    assert campaign["pixel_timeout_s"] < campaign["state_max_predict_dt_s"]
    assert campaign["waypoint_spacing_m"] == 0.1
    assert campaign["manager_decision_rate_hz"] == 4.0
    assert campaign["manager_correction_timestamp_compensation"] is False
    conditions = [
        "global_intact", "global_removal",
        "per_camera_intact", "per_camera_removal",
        "spatial_intact", "spatial_removal",
    ]
    assert list(campaign["conditions"]) == conditions
    assert list(campaign["tasks"]) == [
        "thesis09_parallel_aisles_west",
        "thesis09_parallel_aisles_central",
    ]
    assert sum(
        len(task["conditions"]) * len(task["seeds"])
        for task in campaign["tasks"].values()
    ) == 60
    for task in campaign["tasks"].values():
        assert task["conditions"] == conditions
        assert task["seeds"] == [91500, 91501, 91502, 91503, 91504]
    for name, condition in campaign["conditions"].items():
        expected = "camera_A,camera_B,camera_C,camera_D,camera_E"
        if name.endswith("_removal"):
            expected = "camera_A,camera_C,camera_D,camera_E"
        assert condition["manager_camera_ids"] == expected
        assert condition["camera_network_active_camera_ids"] == expected
    assert campaign["removed_camera_id"] == "camera_B"


def test_runtime_and_planning_artifacts_are_model_matched_and_hash_bound():
    campaign = yaml.safe_load((
        REPO / "pipeline/route_planning_template.yaml"
    ).read_text())
    expected = {
        "global": ("R0_global_full", "M0"),
        "per_camera": ("R1_per_camera_full", "M1"),
        "spatial": ("R2_spatial_residual", "M2"),
    }
    for condition_name, condition in campaign["conditions"].items():
        model = condition_name.removesuffix("_intact").removesuffix("_removal")
        runtime_name, planning_name = expected[model]
        runtime = Path(condition["manager_visibility_sensor_model_path"])
        planning = Path(condition["camera_network_artifact_path"])
        assert runtime.stem == runtime_name
        with np.load(planning, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata["planner_model"] == planning_name
        assert condition["manager_visibility_sensor_model_expected_sha256"]
        assert condition["camera_network_expected_sha256"]


def test_analyzer_requires_preselected_route_only_for_legacy_mode(tmp_path):
    analyzer = load_script(
        "stage09_analyzer",
        "pipeline/analyze_campaign.py",
    )
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"global_planner_mode": "efe"}))
    assert "preselected_route.json" not in analyzer.required_base_artifacts(tmp_path)
    manifest.write_text(json.dumps({"global_planner_mode": "preselected_route"}))
    assert "preselected_route.json" in analyzer.required_base_artifacts(tmp_path)


def test_runner_maps_all_six_conditions_to_visibility_aware_efe():
    runner = load_script(
        "stage09_runner", "scripts/visibility_comparison/run_visibility_campaign.py"
    )
    assert set(runner.CONDITION_PLANNER) == {
        "global_intact", "global_removal",
        "per_camera_intact", "per_camera_removal",
        "spatial_intact", "spatial_removal",
    }
    assert set(runner.CONDITION_PLANNER.values()) == {"visibility_aware_efe"}
