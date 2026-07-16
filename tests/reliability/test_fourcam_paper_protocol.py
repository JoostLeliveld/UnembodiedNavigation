from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools" / "paper_campaign.py"
STUDY = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "study.yaml"
PROTOCOL = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "paper_protocol.yaml"
RECORDER = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools" / "record_operational_logs.py"


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


def test_empty_audit_never_unlocks_closed_loop() -> None:
    tool = _module(TOOL, "fourcam_paper_campaign_empty")
    study = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))

    status = tool.qualification_status(study, protocol, [])

    assert status["mapping_route_disjointness"]["disjoint"] is True
    assert status["gates"]["mapping_collection_complete"] is False
    assert status["gates"]["overlap_complete"] is False
    assert status["gates"]["closed_loop_permitted"] is False


def test_recorder_rotates_propagated_planar_covariance_into_warehouse_frame() -> None:
    module = _module(RECORDER, "fourcam_operational_recorder_covariance")
    recorder = object.__new__(module.OperationalLogRecorder)
    recorder._origin_cos = 0.0
    recorder._origin_sin = 1.0

    xx, xy, yy = recorder._odom_covariance_to_warehouse(0.04, 0.01, 0.09)

    assert xx == 0.09
    assert xy == -0.01
    assert yy == 0.04
