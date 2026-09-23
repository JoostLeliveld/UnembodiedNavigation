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


CONDITIONS = ["global_intact", "global_removal", "per_camera_intact", "per_camera_removal",
              "spatial_intact", "spatial_removal"]
DROPPED = {
    "thesis10_camera_a_western_dock_detour": "camera_A",
    "thesis10_camera_b_cross_warehouse_detour": "camera_B",
    "thesis10_camera_c_inner_warehouse_detour": "camera_C",
    "thesis10_camera_e_eastern_detour": "camera_E",
    "thesis10_camera_e_long_cross_warehouse_detour": "camera_E",
}
ALL_CAMERAS = ["camera_A", "camera_B", "camera_C", "camera_D", "camera_E"]


def test_campaign_is_five_tasks_by_six_conditions_by_three_seeds():
    """The amended campaign: 90 runs, one declared dropped camera per task."""
    for template in ("route_planning_template.yaml", "execution_template.yaml"):
        campaign = yaml.safe_load((REPO / "pipeline" / template).read_text())
        assert campaign["world"] == "warehouse_v2.world.sdf"
        assert campaign["manager_sensor_gate_config_path"] == "config/sensor_gate.yaml"
        assert campaign["scheduled_rate_hz"] == campaign["manager_decision_rate_hz"] == 5.0
        assert list(campaign["conditions"]) == CONDITIONS
        assert dict.fromkeys(campaign["tasks"]) == dict.fromkeys(DROPPED)
        runs = 0
        for name, task in campaign["tasks"].items():
            assert task["conditions"] == CONDITIONS
            assert task["seeds"] == [91500, 91501, 91502]
            runs += len(task["conditions"]) * len(task["seeds"])
            for condition in CONDITIONS:
                effective = {**campaign["conditions"][condition],
                             **task.get("condition_overrides", {}).get(condition, {})}
                expected = ALL_CAMERAS
                if condition.endswith("_removal"):
                    assert effective["removed_camera_id"] == DROPPED[name]
                    expected = [c for c in ALL_CAMERAS if c != DROPPED[name]]
                assert effective["manager_camera_ids"] == ",".join(expected)
                assert effective["camera_network_active_camera_ids"] == ",".join(expected)
        assert runs == 90


def test_execution_runs_in_lockstep():
    campaign = yaml.safe_load((REPO / "pipeline/execution_template.yaml").read_text())
    assert campaign["lockstep"] is True
    assert campaign["lockstep_control_step_iterations"] == 100
    assert campaign["lockstep_camera_every_control_steps"] == 2


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
        "stage09_runner", "pipeline/campaign_runner.py"
    )
    assert set(runner.CONDITION_PLANNER) == {
        "global_intact", "global_removal",
        "per_camera_intact", "per_camera_removal",
        "spatial_intact", "spatial_removal",
    }
    assert set(runner.CONDITION_PLANNER.values()) == {"visibility_aware_efe"}
