from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TOOL = (
    ROOT
    / "experiments/multicamera_commissioning_bigwarehouse/tools/experiment_preflight.py"
)
STUDY = ROOT / "experiments/multicamera_commissioning_bigwarehouse/config/study.yaml"
WORLD = ROOT / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"


def _module():
    spec = importlib.util.spec_from_file_location("multicamera_experiment_preflight", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_frozen_route_variant_has_robot_clearance() -> None:
    module = _module()
    study = yaml.safe_load(STUDY.read_text(encoding="utf-8"))

    rows = module._route_clearances(study, WORLD)

    assert len(rows) == 3 * len(study["collection"]["routes"])
    assert min(row["minimum_collision_clearance_m"] for row in rows) >= 0.20
    central = [row for row in rows if row["route"] == "central_overlap_sweep"]
    assert min(row["minimum_collision_clearance_m"] for row in central) >= 0.44


def test_preflight_config_uses_one_gpu_model_with_safe_separate_fallback() -> None:
    config = yaml.safe_load(
        (ROOT / "experiments/multicamera_commissioning_bigwarehouse/config/detector_4cam_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    runtime = config["runtime_pilot"]
    devices = runtime["separate_process_fallback"]["devices"]

    assert runtime["detector_mode"] == "batched_four_camera"
    assert runtime["runtime_executable"] == "batched_four_camera_yolo_node"
    assert runtime["model_format"] == "native_ultralytics"
    assert runtime["model_instances"] == 1
    assert runtime["batch_size"] == 4
    assert runtime["camera_order"] == ["camera_A", "camera_B", "camera_C", "camera_D"]
    assert runtime["device"] == "0"
    assert runtime["inference_timing_semantics"] == (
        "full_batch_wall_ms_repeated_per_camera_result"
    )
    assert runtime["fault_policy"] == (
        "fatal_process_exit_and_launch_shutdown_no_synthetic_miss"
    )
    assert sum(str(value).lower() != "cpu" for value in devices.values()) == 3
    assert devices["camera_A"] == "cpu"


def test_preflight_detects_a_stale_batched_detector_process() -> None:
    module = _module()

    assert "batched_four_camera_yolo_node" in module.STALE_PATTERNS


def test_swap_activity_measures_live_io_not_stale_occupancy() -> None:
    module = _module()
    page_samples = iter(((100, 200), (356, 328)))
    clock_samples = iter((10.0, 12.0))

    result = module._swap_activity(
        2.0,
        read_pages=lambda: next(page_samples),
        sleep=lambda _seconds: None,
        clock=lambda: next(clock_samples),
        page_size_bytes=4096,
    )

    assert result["available"] is True
    assert result["pages_in"] == 256
    assert result["pages_out"] == 128
    assert result["mib_per_s_in"] == 0.5
    assert result["mib_per_s_out"] == 0.25
    assert result["mib_per_s_total"] == 0.75


def test_vmstat_swap_counter_parser_fails_closed(tmp_path: Path) -> None:
    module = _module()
    vmstat = tmp_path / "vmstat"
    vmstat.write_text("pswpin 12\npgfault 99\n", encoding="utf-8")

    try:
        module._vmstat_swap_pages(vmstat)
    except ValueError as exc:
        assert "pswpin/pswpout" in str(exc)
    else:
        raise AssertionError("missing swap counter was accepted")
