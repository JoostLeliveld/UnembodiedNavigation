from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = (
    ROOT
    / "experiments/multicamera_commissioning_bigwarehouse/tools/runtime_readiness.py"
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


def _complete_state(module, *, frame_ages: tuple[float, float]) -> object:
    state = module.RuntimeReadinessState(
        camera_ids=tuple(module.CAMERA_TOPICS),
        started_wall_s=0.0,
    )
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
