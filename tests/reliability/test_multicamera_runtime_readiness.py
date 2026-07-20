from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
TOOL = (
    ROOT
    / "experiments/multicamera_commissioning_bigwarehouse/tools/runtime_readiness.py"
)
PERCEPTION_SRC = ROOT / "src" / "perception"
if str(PERCEPTION_SRC) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_SRC))

from perception.core.four_camera_runtime_contract import (  # noqa: E402
    REQUIRED_SOURCE_KEYS,
    build_batched_runtime_contract,
    runtime_contract_json,
    runtime_contract_sha256,
)


def _module():
    spec = importlib.util.spec_from_file_location("multicamera_runtime_readiness", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(module, *, mode: str) -> object:
    return module.ReadinessConfig(
        mode=mode,
        timeout_s=10.0,
        min_clock_messages=2,
        min_odom_messages=2,
        min_odom_noisy_messages=2,
        min_ground_truth_messages=2,
        min_camera_messages=2,
        min_unique_camera_stamps=2,
        min_clock_sim_span_s=1.0,
        max_stream_wall_age_s=1.0,
        max_frame_age_s=0.15,
        min_fresh_fraction=1.0,
        sample_window=2,
    )


def _runtime_contract() -> str:
    return runtime_contract_json(
        build_batched_runtime_contract(
            model_path="/tmp/frozen-model.pt",
            model_sha256="a" * 64,
            imgsz=640,
            confidence_threshold=0.05,
            iou_threshold=0.45,
            use_masks=False,
            shared_device="0",
            cpu_threads=2,
            interop_threads=1,
            opencv_threads=1,
            max_batch_stamp_skew_s=0.10,
            max_pending_wall_s=0.50,
            source_hashes={key: "b" * 64 for key in REQUIRED_SOURCE_KEYS},
        )
    )


def _complete_state(
    module,
    *,
    frame_ages: tuple[float, float],
    compare_with_frozen_contract: bool = False,
) -> object:
    expected = json.loads(_runtime_contract()) if compare_with_frozen_contract else None
    state = module.RuntimeReadinessState(
        camera_ids=tuple(module.CAMERA_TOPICS),
        started_wall_s=0.0,
        expected_detector_contract=expected,
    )
    state.observe_detector_runtime_contract(_runtime_contract(), wall_s=0.0)
    for wall_s, sim_s in ((0.0, 5.0), (2.0, 7.0)):
        state.observe_clock(wall_s=wall_s, sim_stamp_s=sim_s)
        state.observe_odom("odom", wall_s=wall_s, sim_stamp_s=sim_s)
        state.observe_odom("odom_noisy", wall_s=wall_s, sim_stamp_s=sim_s)
        state.observe_ground_truth_heartbeat(wall_s=wall_s)
    for camera_id in module.CAMERA_TOPICS:
        for index, (wall_s, sim_s) in enumerate(((1.0, 6.0), (2.0, 7.0))):
            state.observe_camera_raw(camera_id, wall_s=wall_s)
            state.observe_camera(
                camera_id,
                module.CameraTimingSample(
                    timestamp_s=sim_s,
                    wall_received_s=wall_s,
                    frame_age_s=frame_ages[index],
                    inference_wall_ms=10.0 + 10.0 * index,
                    callback_wall_ms=15.0 + 10.0 * index,
                ),
            )
    return state


def test_pilot_reports_age_failure_without_enforcing_it_and_summarizes_timing() -> None:
    module = _module()
    state = _complete_state(module, frame_ages=(0.10, 0.20))

    pilot = state.evaluate(_config(module, mode="pilot"), now_wall_s=2.1)
    enforce = state.evaluate(_config(module, mode="enforce"), now_wall_s=2.1)

    assert pilot["pass"] is True
    assert pilot["evidence_eligible"] is False
    assert pilot["would_pass_age_enforcement"] is False
    assert pilot["age_enforced"] is False
    assert any("pilot age diagnostic" in warning for warning in pilot["warnings"])
    assert enforce["pass"] is False
    assert enforce["evidence_eligible"] is False
    assert enforce["age_enforced"] is True
    assert any("fresh fraction" in failure for failure in enforce["failures"])

    camera_a = pilot["cameras"]["camera_A"]
    assert camera_a["wall_hz"] == pytest.approx(1.0)
    assert camera_a["sim_hz"] == pytest.approx(1.0)
    assert camera_a["frame_age_s"]["p50"] == pytest.approx(0.15)
    assert camera_a["frame_age_s"]["p90"] == pytest.approx(0.19)
    assert camera_a["frame_age_s"]["fresh_fraction"] == pytest.approx(0.5)
    assert camera_a["inference_wall_ms"]["p50"] == pytest.approx(15.0)
    assert camera_a["inference_wall_ms"]["p90"] == pytest.approx(19.0)
    assert camera_a["callback_wall_ms"]["p50"] == pytest.approx(20.0)
    assert pilot["simulation"]["real_time_factor"] == pytest.approx(1.0)


def test_retained_detector_contract_is_a_required_pre_route_readiness_gate() -> None:
    module = _module()
    state = _complete_state(
        module,
        frame_ages=(0.10, 0.10),
        compare_with_frozen_contract=True,
    )

    report = state.evaluate(_config(module, mode="enforce"), now_wall_s=2.1)
    observed = report["detector_runtime_contract"]
    assert report["pass"] is True
    assert observed["validated"] is True
    assert observed["contract_sha256"] == observed["contract"]["contract_sha256"]
    assert observed["qos"] == "reliable_transient_local_keep_last_1"
    assert observed["contract"]["model_sha256"] == "a" * 64
    assert observed["expected_contract_sha256"] == observed["contract_sha256"]
    assert observed["expected_contract"] == observed["contract"]
    assert (
        observed["validation_scope"]
        == "schema_self_hash_and_frozen_locked_config_model_bytes_source_hashes"
    )

    state.detector_runtime_contract = None
    state.detector_runtime_contract_sha256 = None
    missing = state.evaluate(_config(module, mode="enforce"), now_wall_s=2.1)
    assert missing["pass"] is False
    assert any("missing a validated retained detector runtime contract" in item for item in missing["failures"])


def test_frozen_contract_mismatch_cannot_arm_readiness_or_start_timing() -> None:
    module = _module()
    expected = json.loads(_runtime_contract())
    state = module.RuntimeReadinessState(
        camera_ids=tuple(module.CAMERA_TOPICS),
        started_wall_s=0.0,
        expected_detector_contract=expected,
    )
    wrong = dict(expected)
    wrong["imgsz"] = 800
    wrong["contract_sha256"] = runtime_contract_sha256(wrong)

    state.observe_detector_runtime_contract(json.dumps(wrong), wall_s=1.0)
    state.observe_clock(wall_s=2.0, sim_stamp_s=2.0)
    report = state.evaluate(_config(module, mode="enforce"), now_wall_s=2.1)

    assert state.detector_runtime_contract_armed is False
    assert state.core["clock"].message_count == 0
    assert state.precontract_core_messages["clock"] == 1
    assert report["pass"] is False
    assert "imgsz" in report["detector_runtime_contract"]["error"]


def test_timing_samples_before_contract_receipt_are_discarded() -> None:
    module = _module()
    state = module.RuntimeReadinessState(
        camera_ids=tuple(module.CAMERA_TOPICS),
        started_wall_s=0.0,
    )
    state.observe_clock(wall_s=1.0, sim_stamp_s=1.0)
    state.observe_odom("odom", wall_s=1.0, sim_stamp_s=1.0)
    state.observe_odom("odom_noisy", wall_s=1.0, sim_stamp_s=1.0)
    state.observe_ground_truth_heartbeat(wall_s=1.0)
    state.observe_camera_raw("camera_A", wall_s=1.0)

    assert state.core["clock"].message_count == 0
    assert state.cameras["camera_A"].raw_message_count == 0
    assert state.precontract_core_messages["clock"] == 1
    assert state.precontract_camera_messages["camera_A"] == 1

    state.observe_detector_runtime_contract(_runtime_contract(), wall_s=2.0)
    state.observe_clock(wall_s=3.0, sim_stamp_s=3.0)
    report = state.evaluate(_config(module, mode="pilot"), now_wall_s=3.1)

    assert report["streams"]["clock"]["message_count"] == 1
    assert report["detector_runtime_contract"]["precontract_messages_discarded"][
        "core"
    ]["clock"] == 1
    assert report["detector_runtime_contract"][
        "post_contract_window_started_wall_elapsed_s"
    ] == 2.0


def test_bad_or_changed_retained_detector_contract_fails_closed() -> None:
    module = _module()
    state = _complete_state(module, frame_ages=(0.10, 0.10))

    state.observe_detector_runtime_contract("not JSON", wall_s=2.0)
    bad = state.evaluate(_config(module, mode="pilot"), now_wall_s=2.1)
    assert bad["pass"] is False
    assert "detector runtime contract rejected" in bad["detector_runtime_contract"]["error"]

    other = _complete_state(module, frame_ages=(0.10, 0.10))
    changed = json.loads(_runtime_contract())
    changed["imgsz"] = 800
    # A valid but different second retained contract must make the readiness
    # barrier fail; it cannot silently switch models after readiness cleared.
    changed["contract_sha256"] = runtime_contract_sha256(changed)
    other.observe_detector_runtime_contract(json.dumps(changed), wall_s=2.0)
    assert other.evaluate(_config(module, mode="pilot"), now_wall_s=2.1)["pass"] is False


def test_campaign_preflight_binds_readiness_to_one_transport_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    study = tmp_path / "study.yaml"
    protocol = tmp_path / "protocol.yaml"
    model = tmp_path / "model.pt"
    calibration = tmp_path / "calibration.json"
    runtime_config = tmp_path / "runtime.yaml"
    analysis = (
        ROOT
        / "experiments"
        / "multicamera_commissioning_bigwarehouse"
        / "config"
        / "paper_analysis_plan.yaml"
    )
    study.write_text(
        yaml.safe_dump(
            {
                "study_id": "fixture",
                "collection": {
                    "routes": [
                        {
                            "name": "fixture_route",
                            "start": {"x": 0.0, "y": 0.0},
                            "goal": {"x": 1.0, "y": 0.0},
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    protocol.write_text(
        yaml.safe_dump(
            {
                "protocol_id": "fixture",
                "randomization": {"seed_values": [0]},
                "route_disjoint_mapping": {
                    "train_routes": ["fixture_route"],
                    "heldout_routes": [],
                    "repeats_per_route": 1,
                    "lateral_offsets_m": [0.0],
                    "speeds_mps": [0.1],
                },
                "overlap_qualification": {
                    "repeats_per_route": 1,
                    "lateral_offsets_m": [0.0],
                    "speeds_mps": [0.1],
                    "edges": [],
                },
            }
        ),
        encoding="utf-8",
    )
    model.write_bytes(b"model")
    calibration.write_text("{}\n", encoding="utf-8")
    runtime_config.write_text("runtime: fixture\n", encoding="utf-8")
    desired = module.campaign_ledger.build_ledger(
        study_path=study,
        protocol_path=protocol,
        model_path=model,
        calibration_path=calibration,
        config_paths=[analysis, runtime_config],
        attempt_slots=1,
    )
    campaign_root = tmp_path / "campaign"
    ledger = module.campaign_ledger.initialize_campaign(campaign_root, desired)
    row = ledger["rows"][0]
    attempt = row["attempts"][0]
    for name, value in {
        "ROS_LOCALHOST_ONLY": "1",
        "IGN_IP": "127.0.0.1",
        "GZ_IP": "127.0.0.1",
        "IGN_PARTITION": attempt["ign_partition"],
        "ROS_DOMAIN_ID": str(attempt["ros_domain_id"]),
    }.items():
        monkeypatch.setenv(name, value)
    output = campaign_root / attempt["run_dir"] / "runtime_readiness.json"
    args = argparse.Namespace(
        campaign_ledger=campaign_root / "campaign_ledger.json",
        plan_row_id=row["row_id"],
        attempt_id=attempt["attempt_id"],
        seed=row["row_tuple"]["seed"],
        evidence_role=row["expected_evidence_role"],
        analysis_split=row["expected_analysis_split"],
        study_config=study,
        protocol=protocol,
        analysis_plan=analysis,
        detector_runtime_config=runtime_config,
        detector_imgsz=640,
        detector_model=model,
        projection_calibration=calibration,
        frozen_config=[analysis, runtime_config],
    )

    campaign_contract, method_freeze, paths, frozen_configs = module._campaign_preflight(
        args, output=output
    )

    assert campaign_contract["attempt_id"] == attempt["attempt_id"]
    assert campaign_contract["transport_environment"]["ROS_DOMAIN_ID"] == str(
        attempt["ros_domain_id"]
    )
    assert paths["model"] == model.resolve()
    assert frozen_configs == [analysis.resolve(), runtime_config.resolve()]
    assert method_freeze["analysis_plan_sha256"]

    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    with pytest.raises(RuntimeError, match="Transport environment"):
        module._campaign_preflight(args, output=output)


def test_enforce_requires_a_locked_runtime_successor_and_derives_exact_contract(
    tmp_path: Path,
) -> None:
    module = _module()
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    config_path = tmp_path / "locked_detector.yaml"
    analysis = (
        ROOT
        / "experiments"
        / "multicamera_commissioning_bigwarehouse"
        / "config"
        / "paper_analysis_plan.yaml"
    )
    runtime = {
        "evidence_selection": {
            "permitted": True,
            "selected_image_size": 640,
        },
        "image_sizes": [640],
        "model": str(model.resolve()),
        "confidence_threshold": 0.05,
        "iou_threshold": 0.45,
        "detector_mode": "batched_four_camera",
        "runtime_executable": "batched_four_camera_yolo_node",
        "launch_switch": "yolo_batched_four_camera",
        "model_format": "native_ultralytics",
        "model_instances": 1,
        "batch_size": 4,
        "camera_order": ["camera_A", "camera_B", "camera_C", "camera_D"],
        "device": "0",
        "cpu_threads": 2,
        "cpu_interop_threads": 1,
        "opencv_threads": 1,
        "max_batch_stamp_skew_s": 0.10,
        "max_pending_wall_s": 0.50,
        "frame_policy": "strict_new_unique_stamp_latest_only_no_reuse",
        "inference_timing_semantics": "full_batch_wall_ms_repeated_per_camera_result",
        "fault_policy": "fatal_process_exit_and_launch_shutdown_no_synthetic_miss",
        "synchronization_miss_policy": "no_output_detected_by_liveness_gate",
        "masks": False,
    }
    config_path.write_text(
        yaml.safe_dump(
            {
                "lock_status": "locked_before_evidence",
                "artifact_id": "fixture_v2_640",
                "runtime_pilot": runtime,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    method_freeze = module._method_freeze(analysis)

    expected, metadata = module._expected_contract_from_runtime_config(
        config_path=config_path,
        detector_model=model,
        detector_imgsz=640,
        method_freeze=method_freeze,
        analysis_path=analysis,
        mode="enforce",
    )

    assert expected["model_path"] == str(model.resolve())
    assert expected["imgsz"] == 640
    assert expected["contract_sha256"] == runtime_contract_sha256(expected)
    assert metadata["lock_status"] == "locked_before_evidence"
    assert metadata["selected_imgsz"] == 640

    v1 = (
        ROOT
        / "experiments"
        / "multicamera_commissioning_bigwarehouse"
        / "config"
        / "detector_4cam_v1.yaml"
    )
    with pytest.raises(RuntimeError, match="evidence_selection.permitted"):
        module._expected_contract_from_runtime_config(
            config_path=v1,
            detector_model=model,
            detector_imgsz=640,
            method_freeze=method_freeze,
            analysis_path=analysis,
            mode="enforce",
        )


def test_duplicate_camera_stamps_do_not_satisfy_unique_frame_barrier() -> None:
    module = _module()
    state = _complete_state(module, frame_ages=(0.10, 0.10))
    camera = state.cameras["camera_A"]
    camera.samples.clear()
    camera._seen_stamps.clear()
    camera.valid_message_count = 0
    camera.duplicate_stamp_count = 0

    state.observe_camera(
        "camera_A",
        module.CameraTimingSample(7.0, 1.0, 0.10, 10.0, 15.0),
    )
    state.observe_camera(
        "camera_A",
        module.CameraTimingSample(7.0, 2.0, 0.10, 10.0, 15.0),
    )
    report = state.evaluate(_config(module, mode="pilot"), now_wall_s=2.1)

    assert report["pass"] is False
    assert report["cameras"]["camera_A"]["valid_message_count"] == 2
    assert report["cameras"]["camera_A"]["unique_observation_stamp_count"] == 1
    assert report["cameras"]["camera_A"]["duplicate_stamp_count"] == 1
    assert any("unique stamps" in failure for failure in report["failures"])


def test_missing_or_wall_stale_stream_fails_even_in_pilot_mode() -> None:
    module = _module()
    state = _complete_state(module, frame_ages=(0.10, 0.10))

    stale = state.evaluate(_config(module, mode="pilot"), now_wall_s=4.0)
    state.core["odom_noisy"] = module.StreamSeries()
    missing = state.evaluate(_config(module, mode="pilot"), now_wall_s=2.1)

    assert stale["pass"] is False
    assert any("missing or stale" in failure for failure in stale["failures"])
    assert missing["pass"] is False
    assert any("/odom_noisy" in failure for failure in missing["failures"])


def test_ground_truth_is_only_a_counted_heartbeat() -> None:
    module = _module()
    state = _complete_state(module, frame_ages=(0.10, 0.10))

    report = state.evaluate(_config(module, mode="pilot"), now_wall_s=2.1)

    assert report["ground_truth_firewall"] == {
        "topic": "/ground_truth_tf",
        "message_count": 2,
        "values_read": False,
        "purpose": "stream existence/count heartbeat only",
    }
    source = TOOL.read_text(encoding="utf-8")
    callback = source.split("def _ground_truth_heartbeat_callback", 1)[1].split(
        "def _camera_callback", 1
    )[0]
    assert ".transforms" not in callback
    assert ".translation" not in callback
    assert ".rotation" not in callback


def test_atomic_readiness_report_does_not_replace_prior_evidence(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "runtime_readiness.json"

    module.write_json_atomic_new(output, {"pass": True, "attempt": 1})

    assert json.loads(output.read_text(encoding="utf-8"))["attempt"] == 1
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        module.write_json_atomic_new(output, {"pass": True, "attempt": 2})
    assert json.loads(output.read_text(encoding="utf-8"))["attempt"] == 1


def test_recent_window_can_clear_startup_age_outliers() -> None:
    module = _module()
    state = _complete_state(module, frame_ages=(0.50, 0.50))
    for camera_id in module.CAMERA_TOPICS:
        for wall_s, sim_s in ((3.0, 8.0), (4.0, 9.0)):
            state.observe_camera_raw(camera_id, wall_s=wall_s)
            state.observe_camera(
                camera_id,
                module.CameraTimingSample(sim_s, wall_s, 0.10, 10.0, 15.0),
            )
    state.observe_clock(wall_s=4.0, sim_stamp_s=9.0)
    state.observe_odom("odom", wall_s=4.0, sim_stamp_s=9.0)
    state.observe_odom("odom_noisy", wall_s=4.0, sim_stamp_s=9.0)
    state.observe_ground_truth_heartbeat(wall_s=4.0)

    report = state.evaluate(_config(module, mode="enforce"), now_wall_s=4.1)

    assert report["pass"] is True
    assert report["evidence_eligible"] is True
    assert report["would_pass_age_enforcement"] is True
    assert report["cameras"]["camera_A"]["summary_window_count"] == 2
    assert report["cameras"]["camera_A"]["frame_age_s"]["fresh_fraction"] == 1.0
