from __future__ import annotations

import copy
import importlib.util
import json
import numpy as np
from pathlib import Path
import sys

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments/reference_controlled_commissioning_v1"
sys.path.insert(0, str(HERE))


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = _module("commissioning_protocol_validator", "validate_protocol.py")
controller = _module("commissioning_reference_controller", "reference_pose_controller.py")
tables = _module("commissioning_table_builder", "build_tables.py")
plots = _module("commissioning_point_plots", "plot_points.py")


def test_protocol_preflight_passes_and_has_drive_level_balance():
    report = validator.validate(HERE / "campaign.yaml")
    assert report["passed"] is True
    assert report["nominal_cruise_mps"] == 1.0
    assert report["partition_counts"] == {"audit": 8, "development": 8, "fit": 8}
    assert min(report["route_minimum_static_body_clearance_m"].values()) >= 0.02
    assert min(report["stationary_anchor_minimum_static_body_clearance_m"].values()) >= 0.02


def test_commissioning_crop_argument_reaches_detector_runtime():
    common = (
        REPO / "src/experiments/experiments/core/visibility_launch_common.py"
    ).read_text(encoding="utf-8")
    assert "'yolo_debug_crop_dir': _launch_value" in common
    # One mapping configures the single-camera detector and the other configures
    # the batched five-camera detector used by commissioning.
    assert common.count("'debug_crop_dir': cfg.get('yolo_debug_crop_dir', '')") == 2


def test_each_drive_uses_an_isolated_gazebo_transport_partition():
    runner = (HERE / "run_drive.py").read_text(encoding="utf-8")
    assert 'environment["IGN_PARTITION"] = transport_partition' in runner
    assert 'environment["GZ_PARTITION"] = transport_partition' in runner


def test_retired_nominal_speed_fails_closed(tmp_path: Path):
    protocol = yaml.safe_load((HERE / "campaign.yaml").read_text(encoding="utf-8"))
    protocol = copy.deepcopy(protocol)
    protocol["controller"]["nominal_cruise_mps"] = 0.22
    path = tmp_path / "invalid_campaign.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    with pytest.raises(validator.ProtocolError, match="exactly 1.0 m/s"):
        validator.validate(path)


def test_table_builder_rejects_drive_from_another_protocol_hash(tmp_path: Path):
    run_dir = tmp_path / "old_drive"
    run_dir.mkdir()
    (run_dir / "drive_manifest.json").write_text(json.dumps({
        "protocol_sha256": "0" * 64,
        "drive": {
            "id": "fit_old", "partition": "fit",
            "route": "west_spine", "direction": "forward",
        },
    }) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="protocol SHA-256"):
        tables.build_tables(run_dir, HERE / "campaign.yaml")


def test_controller_reaches_full_cruise_on_aligned_straight():
    decision = controller.control_to_goal(
        (0.0, 0.0, 0.0), (10.0, 0.0),
        nominal_cruise_mps=1.0,
        maximum_angular_speed_radps=1.0,
        heading_gain=2.5,
        heading_stop_rad=0.35,
        slowdown_distance_m=1.0,
        arrival_radius_m=0.12,
    )
    assert decision.linear_mps == pytest.approx(1.0)
    assert decision.angular_radps == pytest.approx(0.0)
    assert decision.complete is False


def test_controller_stops_translation_for_large_heading_error():
    decision = controller.control_to_goal(
        (0.0, 0.0, 1.0), (10.0, 0.0),
        nominal_cruise_mps=1.0,
        maximum_angular_speed_radps=1.0,
        heading_gain=2.5,
        heading_stop_rad=0.35,
        slowdown_distance_m=1.0,
        arrival_radius_m=0.12,
    )
    assert decision.linear_mps == 0.0
    assert decision.angular_radps < 0.0


def test_controller_declares_arrival_without_command():
    decision = controller.control_to_goal(
        (9.95, 0.0, 0.0), (10.0, 0.0),
        nominal_cruise_mps=1.0,
        maximum_angular_speed_radps=1.0,
        heading_gain=2.5,
        heading_stop_rad=0.35,
        slowdown_distance_m=1.0,
        arrival_radius_m=0.12,
    )
    assert decision.complete is True
    assert decision.linear_mps == 0.0
    assert decision.angular_radps == 0.0


def test_synchronized_tables_count_each_physical_frame_once(tmp_path: Path):
    run_dir = tmp_path / "synthetic_drive"
    experiment = run_dir / "experiment_logs/experiment_synthetic"
    experiment.mkdir(parents=True)
    (run_dir / "drive_manifest.json").write_text(json.dumps({
        "protocol_sha256": validator.sha256_file(HERE / "campaign.yaml"),
        "drive": {
            "id": "fit_synthetic", "partition": "fit",
            "route": "west_spine", "direction": "forward",
        },
    }) + "\n", encoding="utf-8")
    batch = "synthetic_batch"
    stamp_ns = 1_000_000_000
    opportunity_rows = []
    manager_rows = []
    identity_rows = []
    for index, camera_id in enumerate(("camera_A", "camera_B", "camera_C", "camera_D", "camera_E")):
        content_hash = f"{index + 1:064x}"
        source = f"frame:epoch:{camera_id}:{stamp_ns}:{content_hash}"
        observation = {
            "camera_id": camera_id,
            "capture_stamp_ns": stamp_ns,
            "source_frame_id": source,
            "source_batch_id": batch,
            "detection_valid": True,
            "detector_score": 0.9,
            "detector_score_raw": 0.9,
            "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
        }
        opportunity_rows.append({
            "valid_contract": True,
            "duplicate": False,
            "observation": observation,
        })
        manager_rows.append({
            "status": "camera_mapping",
            "camera_id": camera_id,
            "capture_stamp_ns": stamp_ns,
            "source_frame_id": source,
            "disposition": "mapped",
            "reason": "",
            "capture_observation": {"xy_m": [-6.9, -4.4]},
        })
        identity_rows.append({
            "camera_id": camera_id,
            "capture_stamp_ns": stamp_ns,
            "image_contract_sha256": f"{index + 11:064x}",
        })

    def write_jsonl(path: Path, rows):
        import json
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    write_jsonl(experiment / "camera_opportunities.jsonl", opportunity_rows)
    write_jsonl(run_dir / "manager_outcomes.jsonl", manager_rows)
    write_jsonl(run_dir / "frame_identity.jsonl", identity_rows)
    write_jsonl(run_dir / "reference_controller_trace.jsonl", [
        {"stamp_ns": 950_000_000, "pose": [-6.95, -4.5, 1.5707963268],
         "linear_mps": 1.0, "angular_radps": 0.0, "state": "tracking"},
        {"stamp_ns": 1_050_000_000, "pose": [-6.95, -4.4, 1.5707963268],
         "linear_mps": 1.0, "angular_radps": 0.0, "state": "tracking"},
    ])
    crop_dir = run_dir / "selected_crops"
    crop_dir.mkdir()
    for index, camera_id in enumerate(("camera_A", "camera_B", "camera_C", "camera_D", "camera_E")):
        source = f"frame:epoch:{camera_id}:{stamp_ns}:{index + 1:064x}"
        np.savez_compressed(
            crop_dir / f"{camera_id}_{stamp_ns:019d}.npz",
            source_frame_id=np.asarray(source),
            capture_stamp_ns=np.asarray(stamp_ns, dtype=np.int64),
        )
    report = tables.build_tables(run_dir, HERE / "campaign.yaml")
    assert report["passed"] is True
    assert report["opportunity_rows"] == 5
    assert report["unique_physical_frame_rows"] == 5
    assert report["network_round_rows"] == 1
    assert report["selected_crop_present_fraction"] == 1.0
    assert report["selected_crop_identity_match_fraction"] == 1.0

    plot_output = run_dir / "point_plots"
    plot_output.mkdir()
    rows = plots._read([run_dir / "tables/camera_opportunities.csv"])
    plots.plot_opportunities(rows, plot_output)
    plots.plot_raw_residuals(rows, plot_output)
    assert (plot_output / "commissioning_opportunity_points.pdf").is_file()
    assert (plot_output / "commissioning_raw_residual_points.png").is_file()
