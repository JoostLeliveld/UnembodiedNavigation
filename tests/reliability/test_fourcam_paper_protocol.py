from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools" / "paper_campaign.py"
STUDY = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "study.yaml"
PROTOCOL = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "paper_protocol.yaml"
ANALYSIS = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "paper_analysis_plan.yaml"
READINESS = TOOL.with_name("paper_readiness.py")
RECORDER = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools" / "record_operational_logs.py"
ENCODER = ROOT / "src" / "sim" / "sim" / "encoder_noise_node.py"
sys.path.insert(0, str(ROOT / "src" / "reliability"))


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_protocol_is_route_disjoint_and_materialises_all_evidence_phases() -> None:
    tool = _module(TOOL, "fourcam_paper_campaign")
    study = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))

    plan = tool.build_plan(study, protocol)
    phases = {row["phase"] for row in plan}

    assert set(protocol["route_disjoint_mapping"]["train_routes"]).isdisjoint(
        protocol["route_disjoint_mapping"]["heldout_routes"]
    )
    assert {"D1_mapping_train", "D1_mapping_heldout", "D3_paired_replay"}.issubset(phases)
    assert {"D2_overlap_camera_A_camera_C", "D2_overlap_camera_B_camera_D", "D2_overlap_camera_C_camera_D"}.issubset(phases)
    assert len([row for row in plan if row["phase"] == "D1_mapping_train"]) == 120
    assert len([row for row in plan if row["phase"] == "D1_mapping_heldout"]) == 60
    assert len([row for row in plan if row["phase"] == "D3_paired_replay"]) == 600
    assert len([row for row in plan if row["phase"] == "D4_closed_loop_confirmation"]) == 60


def test_empty_audit_never_unlocks_closed_loop() -> None:
    tool = _module(TOOL, "fourcam_paper_campaign_empty")
    study = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))

    status = tool.qualification_status(study, protocol, [])

    assert status["mapping_route_disjointness"]["disjoint"] is True
    assert status["gates"]["mapping_collection_complete"] is False
    assert status["gates"]["overlap_complete"] is False
    assert status["gates"]["closed_loop_permitted"] is False


def test_readiness_requires_a_validated_ledger_and_matches_recorder_method_freeze() -> None:
    readiness = _module(READINESS, "fourcam_paper_readiness_integrity")
    recorder = _module(RECORDER, "fourcam_recorder_method_freeze")
    analysis = yaml.safe_load(ANALYSIS.read_text(encoding="utf-8"))

    assert recorder._method_freeze(ANALYSIS) == readiness._snapshot(analysis, ANALYSIS)
    runs, integrity = readiness._validated_campaign_runs(
        None,
        study_path=STUDY,
        protocol_path=PROTOCOL,
        analysis_path=ANALYSIS,
        method_snapshot=readiness._snapshot(analysis, ANALYSIS),
    )
    assert runs == []
    assert integrity["pass"] is False
    assert "campaign_ledger.json" in integrity["failures"][0]


def test_recorder_rotates_propagated_planar_covariance_into_warehouse_frame() -> None:
    module = _module(RECORDER, "fourcam_operational_recorder_covariance")
    recorder = object.__new__(module.OperationalLogRecorder)
    recorder._origin_cos = 0.0
    recorder._origin_sin = 1.0

    xx, xy, yy = recorder._odom_covariance_to_warehouse(0.04, 0.01, 0.09)

    assert xx == 0.09
    assert xy == -0.01
    assert yy == 0.04


def test_multicamera_benchmark_passes_each_frozen_gp_to_selection_policies() -> None:
    from reliability import (  # imported after the local package path is added
        FixedCameraReliabilityProvider,
        ReplayMode,
        default_replay_benchmark_conditions,
    )

    providers = {
        "camera_A": FixedCameraReliabilityProvider(camera_id="camera_A", p_available=0.8),
        "camera_B": FixedCameraReliabilityProvider(camera_id="camera_B", p_available=0.7),
        "camera_C": FixedCameraReliabilityProvider(camera_id="camera_C", p_available=0.6),
        "camera_D": FixedCameraReliabilityProvider(camera_id="camera_D", p_available=0.5),
    }
    conditions = default_replay_benchmark_conditions(
        include_multicamera=True,
        quality_providers=providers,
    )
    provider_by_mode = {
        condition.config.mode: condition.config.quality_providers
        for condition in conditions
    }

    for mode in (
        ReplayMode.CURRENT_GP_R,
        ReplayMode.CONSERVATIVE_SELECTION,
        ReplayMode.HANDOVER_AWARE_SELECTION,
        ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
    ):
        assert set(provider_by_mode[mode]) == set(providers)


def test_encoder_noise_propagates_a_finite_planar_process_covariance() -> None:
    module = _module(ENCODER, "fourcam_encoder_noise_covariance")
    node = object.__new__(module.EncoderNoiseNode)
    node._pose_cov = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
    node.linear_slip_std = 0.05
    node.angular_slip_std = 0.03
    node.linear_additive_std = 0.004
    node.angular_additive_std = 0.020
    node.correlation_alpha = 0.80
    node.linear_scale_bias_std = 0.02
    node._linear_scale_jacobian = [0.0, 0.0, 0.0]
    node.covariance_floor_m2 = 1.0e-8
    node.covariance_floor_yaw_rad2 = 1.0e-8

    var_v, var_w = node._propagate_pose_covariance(theta=0.4, v_true=0.2, w_true=0.1, dt=0.1)

    assert var_v > 0.0
    assert var_w > 0.0
    assert node._pose_cov[0][0] > 0.0
    assert node._pose_cov[1][1] > 0.0
    assert node._pose_cov[2][2] > 0.0
    published = node._published_pose_covariance()
    assert published[0][0] > node._pose_cov[0][0]
