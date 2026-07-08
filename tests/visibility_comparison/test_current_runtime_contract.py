from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.visibility_comparison import run_visibility_campaign as campaign_runner


ROOT = Path(__file__).resolve().parents[2]

ACTIVE_RUNTIME_FILES = (
    "src/experiments/launch/warehouse_primary_comparison.launch.py",
    "src/experiments/experiments/core/visibility_launch_common.py",
    "src/planning/planning/nodes/unicycle_planner_node.py",
    "src/planning/planning/nodes/efe_agent_node.py",
    "scripts/visibility_comparison/run_visibility_campaign.py",
    "scripts/visibility_comparison/warehouse_visibility_campaign.yaml",
    "docs/paper_vs_current/current/warehouse_visibility_campaign.yaml",
)

RETIRED_ACTIVE_SURFACES = (
    "risk_only_ablation",
    "warehouse_visibility_agent",
    "keypoint_marker_world_z",
    "use_state_bev_heading_correction",
    "use_state_bev_yaw",
    "use_displacement_heading",
    "yolo_min_keypoint_conf",
    "keypoint_heading_sigma_rad",
    "use_odom_heading_correction",
    "heading_min_displacement_m",
    "heading_bev_noise_sigma_m",
    "heading_max_displacement_m",
    "odom_heading_correction_mode",
    "clamp_pixel_uv_theta_without_yaw",
    "local_tracking_use_odom_yaw",
    "use_pixel_heading_correction",
    "_fresh_odom_heading_locked",
    "_apply_heading_measurement",
    "_select_heading_measurement_locked",
)


def _load_yaml(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def test_current_runtime_contract_matches_campaign_config() -> None:
    campaign = _load_yaml("scripts/visibility_comparison/warehouse_visibility_campaign.yaml")
    contract = _load_yaml("docs/current_runtime_contract.yaml")["current_runtime_contract"]
    locked = contract["locked_runtime"]

    assert contract["source_config"] == (
        "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
    )
    assert locked["world"] == campaign["world"]
    assert locked["launch_file"] == campaign["launch_file"]
    assert locked["tasks"] == list(campaign["tasks"].keys())
    assert locked["seeds"] == [0, 1, 2, 3, 4]
    for task in locked["tasks"]:
        assert campaign["tasks"][task]["conditions"] == ["C1", "C2"]
        assert campaign["tasks"][task]["seeds"] == locked["seeds"]

    assert locked["yolo_model"] == campaign["yolo_model"]
    assert locked["yolo_inference_imgsz"] == campaign["yolo_imgsz"]
    assert locked["yolo_conf_threshold"] == campaign["yolo_conf_threshold"]
    assert locked["yolo_iou_threshold"] == campaign["yolo_iou_threshold"]
    assert locked["yolo_use_masks"] == campaign["yolo_use_masks"]
    assert locked["gp_artifact"] == campaign["gp_artifact"]

    assert locked["heading_update_mode"] == campaign["heading_update_mode"]
    assert locked["no_keypoint_or_visual_heading"] is True
    for retired_key in (
        "keypoint_marker_world_z",
        "use_state_bev_heading_correction",
        "use_state_bev_yaw",
        "use_displacement_heading",
        "use_odom_heading_correction",
        "heading_min_displacement_m",
        "heading_bev_noise_sigma_m",
        "heading_max_displacement_m",
        "odom_heading_correction_mode",
        "clamp_pixel_uv_theta_without_yaw",
        "local_tracking_use_odom_yaw",
        "use_pixel_heading_correction",
    ):
        assert retired_key not in campaign
    assert locked["global_horizon"] == campaign["global_horizon"]
    assert locked["global_dt"] == campaign["global_dt"]
    assert locked["lookahead_s"] == campaign["global_horizon"] * campaign["global_dt"]
    assert locked["pixel_correction_nis_threshold"] == campaign["pixel_correction_nis_threshold"]
    assert locked["use_belief_nogo_cost"] == campaign["use_belief_nogo_cost"]
    assert locked["nogo_mode"] == campaign["nogo_mode"]
    assert locked["nogo_penalty_type"] == campaign["nogo_penalty_type"]
    assert locked["nogo_weight"] == campaign["nogo_weight"]
    assert locked["nogo_belief_kappa"] == campaign["nogo_belief_kappa"]


def test_current_contract_exposes_only_final_conditions() -> None:
    contract = _load_yaml("docs/current_runtime_contract.yaml")["current_runtime_contract"]
    campaign = _load_yaml("scripts/visibility_comparison/warehouse_visibility_campaign.yaml")

    assert set(contract["active_conditions"].keys()) == {"C1", "C2"}
    assert "risk_only_ablation" in contract["retired_active_conditions"]
    assert set(campaign["conditions"].keys()) == {"C1", "C2"}


def test_active_runtime_files_do_not_expose_retired_surfaces() -> None:
    for relative in ACTIVE_RUNTIME_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for retired_surface in RETIRED_ACTIVE_SURFACES:
            assert retired_surface not in text, f"{relative} still exposes {retired_surface}"


def test_campaign_runner_rejects_retired_active_condition() -> None:
    campaign = _load_yaml("scripts/visibility_comparison/warehouse_visibility_campaign.yaml")
    campaign["conditions"] = dict(campaign["conditions"])
    campaign["conditions"]["risk_only_ablation"] = {"planner": "visibility_aware_efe"}

    with pytest.raises(RuntimeError, match="unsupported active condition"):
        campaign_runner._validate_config(campaign, ROOT / "synthetic.yaml")


def test_campaign_launch_command_uses_only_final_heading_surface(tmp_path: Path) -> None:
    campaign = _load_yaml("scripts/visibility_comparison/warehouse_visibility_campaign.yaml")
    task_name = next(iter(campaign["tasks"]))
    command = campaign_runner._build_launch_cmd(campaign, task_name, "C2", 0, tmp_path)
    command_text = " ".join(str(part) for part in command)

    assert "heading_update_mode:=camera_xy_only" in command_text
    assert "visibility_artifact_path:=" in command_text
    for retired_surface in RETIRED_ACTIVE_SURFACES:
        assert retired_surface not in command_text
