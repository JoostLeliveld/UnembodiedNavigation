from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = (
    ROOT
    / "experiments/multicamera_commissioning_bigwarehouse/tools/record_runtime_resources.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("runtime_resource_monitor", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self, hook: Callable[[float], None] | None = None) -> None:
        self.value = 0.0
        self.hook = hook

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration_s: float) -> None:
        assert duration_s >= 0.0
        self.value += duration_s
        if self.hook is not None:
            self.hook(self.value)


def _inputs(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    model = tmp_path / "model.pt"
    model.write_bytes(b"immutable detector weights")
    study = tmp_path / "study.yaml"
    study.write_text("protocol_id: p1\n", encoding="utf-8")
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("image_size: 640\n", encoding="utf-8")
    return model, (study, runtime)


def _config(module, tmp_path: Path, **overrides: object):
    model, frozen = _inputs(tmp_path)
    values = {
        "out_dir": tmp_path / "resources",
        "run_id": "pilot_001",
        "detector_mode": "camera_A_cpu_BCD_gpu",
        "detector_image_size": 640,
        "detector_model": model,
        "frozen_configs": frozen,
        "interval_s": 1.0,
        "duration_s": 2.0,
        "completion_manifest": None,
        "wall_timeout_s": 0.0,
        "min_samples": 2,
        "nvidia_timeout_s": 3.0,
    }
    values.update(overrides)
    return module.MonitorConfig(**values)


def _sample(
    module,
    *,
    memory_used: float,
    utilization: float,
    available: float,
    swap_used: float,
    errors: tuple[str, ...] = (),
):
    return module.ProbeSample(
        sampled_utc="2026-07-17T12:00:00+00:00",
        host=module.HostSample(available, 1024.0, swap_used, 1.0, 0.5, 0.25),
        gpus=(
            module.GpuSample(
                0,
                "GPU-test",
                "Mock GPU",
                4096.0,
                memory_used,
                utilization,
                55.0,
            ),
        ),
        errors=errors,
    )


def test_parsers_and_probe_are_ros_free_and_mockable(tmp_path: Path) -> None:
    module = _module()
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemAvailable: 1048576 kB\nSwapTotal: 524288 kB\nSwapFree: 393216 kB\n",
        encoding="utf-8",
    )
    loadavg = tmp_path / "loadavg"
    loadavg.write_text("1.25 0.75 0.50 1/100 123\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 2.0
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='0, GPU-abc, "Mock, GPU", 4096, 256, 37, 58\n',
            stderr="",
        )

    sample = module.collect_probe_sample(
        meminfo_path=meminfo,
        loadavg_path=loadavg,
        nvidia_timeout_s=2.0,
        run_command=fake_run,
    )

    assert sample.errors == ()
    assert sample.host.mem_available_mib == pytest.approx(1024.0)
    assert sample.host.swap_used_mib == pytest.approx(128.0)
    assert sample.host.load_1m == pytest.approx(1.25)
    assert sample.gpus[0].name == "Mock, GPU"
    assert sample.gpus[0].memory_used_mib == pytest.approx(256.0)
    source = TOOL.read_text(encoding="utf-8")
    assert "rclpy" not in source
    assert "create_subscription" not in source
    assert "/ground_truth" not in source


def test_success_atomically_finalizes_csv_and_summary(tmp_path: Path) -> None:
    module = _module()
    config = _config(module, tmp_path)
    samples = iter(
        (
            _sample(
                module,
                memory_used=100.0,
                utilization=10.0,
                available=900.0,
                swap_used=50.0,
            ),
            _sample(
                module,
                memory_used=300.0,
                utilization=30.0,
                available=800.0,
                swap_used=70.0,
            ),
        )
    )
    clock = FakeClock()

    manifest = module.run_monitor(
        config,
        sampler=lambda: next(samples),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    out_dir = config.out_dir
    csv_path = out_dir / module.FINAL_CSV_NAME
    final_manifest = out_dir / module.FINAL_MANIFEST_NAME
    assert manifest["status"] == "completed"
    assert manifest["stop_reason"] == "steady_duration_reached"
    assert manifest["summary"]["gpu"]["peak_memory_used_mib"] == pytest.approx(300.0)
    assert manifest["summary"]["gpu"]["utilization_percent"]["p50"] == pytest.approx(20.0)
    assert manifest["summary"]["gpu"]["utilization_percent"]["p90"] == pytest.approx(28.0)
    assert manifest["summary"]["host"]["minimum_mem_available_mib"] == pytest.approx(800.0)
    assert manifest["summary"]["host"]["swap_used_mib"]["delta"] == pytest.approx(20.0)
    assert csv_path.is_file()
    assert final_manifest.is_file()
    assert not (csv_path.with_suffix(csv_path.suffix + ".part")).exists()
    assert not (out_dir / module.PROGRESS_MANIFEST_NAME).exists()
    assert not (out_dir / module.FAILED_MANIFEST_NAME).exists()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_index"] for row in rows] == ["0", "1"]
    assert all(row["sample_status"] == "ok" for row in rows)
    persisted = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert persisted["provenance"]["detector_model"]["sha256"]
    assert len(persisted["provenance"]["frozen_configs"]) == 2
    assert persisted["contains_ground_truth"] is False
    assert persisted["subscribes_to_ros"] is False


def test_probe_error_is_quarantined_with_failed_manifest(tmp_path: Path) -> None:
    module = _module()
    config = _config(module, tmp_path, duration_s=1.0, min_samples=1)
    clock = FakeClock()

    with pytest.raises(module.ResourceMonitorError, match="sample attempt"):
        module.run_monitor(
            config,
            sampler=lambda: _sample(
                module,
                memory_used=200.0,
                utilization=20.0,
                available=700.0,
                swap_used=80.0,
                errors=("gpu_probe: RuntimeError: timeout",),
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    out_dir = config.out_dir
    failed = json.loads((out_dir / module.FAILED_MANIFEST_NAME).read_text(encoding="utf-8"))
    final_csv = out_dir / module.FINAL_CSV_NAME
    part_csv = final_csv.with_suffix(final_csv.suffix + ".part")
    assert failed["status"] == "failed"
    assert failed["summary"]["sample_failure_count"] == 1
    assert "timeout" in failed["summary"]["sample_failures"][0]["errors"][0]
    assert part_csv.is_file()
    assert not final_csv.exists()
    assert not (out_dir / module.FINAL_MANIFEST_NAME).exists()
    assert not (out_dir / module.PROGRESS_MANIFEST_NAME).exists()


def test_completion_manifest_can_bound_monitor_with_steady_deadman(tmp_path: Path) -> None:
    module = _module()
    completion = tmp_path / "route_completion.json"

    def publish_completion(now_s: float) -> None:
        if now_s >= 0.2 and not completion.exists():
            completion.write_text(
                json.dumps({"status": "completed", "route_complete": True}) + "\n",
                encoding="utf-8",
            )

    config = _config(
        module,
        tmp_path,
        duration_s=0.0,
        completion_manifest=completion,
        wall_timeout_s=5.0,
        min_samples=1,
    )
    clock = FakeClock(hook=publish_completion)
    manifest = module.run_monitor(
        config,
        sampler=lambda: _sample(
            module,
            memory_used=100.0,
            utilization=10.0,
            available=900.0,
            swap_used=50.0,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert manifest["status"] == "completed"
    assert manifest["stop_reason"] == "completion_manifest"
    assert manifest["summary"]["sample_attempts"] == 1
    assert manifest["completion_manifest"]["sha256"]


def test_completion_only_without_deadman_fails_before_sampling(tmp_path: Path) -> None:
    module = _module()
    config = _config(
        module,
        tmp_path,
        duration_s=0.0,
        completion_manifest=tmp_path / "route_completion.json",
        wall_timeout_s=0.0,
    )

    with pytest.raises(module.ResourceMonitorError, match="wall_timeout"):
        module.run_monitor(config, sampler=lambda: pytest.fail("must not sample"))

    failed = json.loads(
        (config.out_dir / module.FAILED_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert failed["status"] == "failed"
    assert failed["summary"]["sample_attempts"] == 0
    assert "wall_timeout" in failed["failure_message"]


def test_completion_deadman_uses_steady_clock_and_quarantines_samples(tmp_path: Path) -> None:
    module = _module()
    config = _config(
        module,
        tmp_path,
        duration_s=0.0,
        completion_manifest=tmp_path / "route_completion.json",
        wall_timeout_s=0.25,
        min_samples=1,
    )
    clock = FakeClock()

    with pytest.raises(module.ResourceMonitorError, match="deadman"):
        module.run_monitor(
            config,
            sampler=lambda: _sample(
                module,
                memory_used=100.0,
                utilization=10.0,
                available=900.0,
                swap_used=50.0,
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    failed = json.loads(
        (config.out_dir / module.FAILED_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert failed["stop_reason"] == "steady_clock_deadman"
    assert failed["summary"]["sample_attempts"] == 1
    assert (config.out_dir / (module.FINAL_CSV_NAME + ".part")).is_file()


def test_completed_output_is_never_replaced(tmp_path: Path) -> None:
    module = _module()
    config = _config(module, tmp_path, duration_s=1.0, min_samples=1)
    clock = FakeClock()
    sample = lambda: _sample(
        module,
        memory_used=100.0,
        utilization=10.0,
        available=900.0,
        swap_used=50.0,
    )
    module.run_monitor(
        config, sampler=sample, monotonic=clock.monotonic, sleep=clock.sleep
    )
    final_manifest = config.out_dir / module.FINAL_MANIFEST_NAME
    before = final_manifest.read_bytes()

    with pytest.raises(module.ResourceMonitorError, match="refusing to reuse"):
        module.run_monitor(config, sampler=sample)

    assert final_manifest.read_bytes() == before


def test_nvidia_smi_nonzero_exit_becomes_sample_error(tmp_path: Path) -> None:
    module = _module()
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemAvailable: 1024 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
        encoding="utf-8",
    )
    loadavg = tmp_path / "loadavg"
    loadavg.write_text("0 0 0 1/1 1\n", encoding="utf-8")

    def failed_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="driver failure")

    sample = module.collect_probe_sample(
        meminfo_path=meminfo,
        loadavg_path=loadavg,
        run_command=failed_run,
    )

    assert sample.host is not None
    assert sample.gpus == ()
    assert sample.errors == ("gpu_probe: RuntimeError: exit 9: driver failure",)
