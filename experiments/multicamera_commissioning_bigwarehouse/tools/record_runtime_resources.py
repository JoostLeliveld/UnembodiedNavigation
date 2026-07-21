#!/usr/bin/env python3
"""Record fail-closed host/GPU resource evidence for detector pilots.

This recorder is deliberately ROS-free and never reads simulator or ground-
truth topics.  It samples host memory/load from ``/proc`` and GPU telemetry
from ``nvidia-smi`` on a steady-clock schedule.  A requested steady-clock
duration and/or a route completion manifest bounds every run.

Completed evidence is immutable: samples are first written to ``.csv.part``
and atomically linked into place only when every sample succeeded.  Any probe,
provenance, lifecycle, or publication error leaves partial data quarantined
and publishes ``resource_monitor_manifest.failed.json``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


CSV_FIELDS = (
    "run_id",
    "sample_index",
    "sampled_utc",
    "steady_elapsed_s",
    "sample_status",
    "sample_error",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "gpu_memory_total_mib",
    "gpu_memory_used_mib",
    "gpu_utilization_percent",
    "gpu_temperature_c",
    "mem_available_mib",
    "swap_total_mib",
    "swap_used_mib",
    "load_1m",
    "load_5m",
    "load_15m",
)

GPU_QUERY = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
)

FINAL_CSV_NAME = "runtime_resource_samples.csv"
FINAL_MANIFEST_NAME = "resource_monitor_manifest.json"
PROGRESS_MANIFEST_NAME = "resource_monitor_manifest.in_progress.json"
FAILED_MANIFEST_NAME = "resource_monitor_manifest.failed.json"


class ResourceMonitorError(RuntimeError):
    """Raised after a failed manifest has been published when possible."""


@dataclass(frozen=True)
class GpuSample:
    index: int
    uuid: str
    name: str
    memory_total_mib: float
    memory_used_mib: float
    utilization_percent: float
    temperature_c: float


@dataclass(frozen=True)
class HostSample:
    mem_available_mib: float
    swap_total_mib: float
    swap_used_mib: float
    load_1m: float
    load_5m: float
    load_15m: float


@dataclass(frozen=True)
class ProbeSample:
    """One probe attempt; partial host/GPU values are retained on failure."""

    sampled_utc: str
    host: HostSample | None
    gpus: tuple[GpuSample, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class MonitorConfig:
    out_dir: Path
    run_id: str
    detector_mode: str
    detector_image_size: int
    detector_model: Path
    frozen_configs: tuple[Path, ...]
    interval_s: float = 1.0
    duration_s: float = 0.0
    completion_manifest: Path | None = None
    wall_timeout_s: float = 0.0
    min_samples: int = 1
    nvidia_timeout_s: float = 5.0


@dataclass
class SummaryAccumulator:
    attempts: int = 0
    successful_attempts: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    host_samples: list[HostSample] = field(default_factory=list)
    gpu_samples: list[GpuSample] = field(default_factory=list)

    def add(self, *, index: int, elapsed_s: float, sample: ProbeSample) -> None:
        self.attempts += 1
        if sample.host is not None:
            self.host_samples.append(sample.host)
        self.gpu_samples.extend(sample.gpus)
        if sample.errors:
            self.failures.append(
                {
                    "sample_index": int(index),
                    "steady_elapsed_s": float(elapsed_s),
                    "errors": list(sample.errors),
                }
            )
        else:
            self.successful_attempts += 1

    def report(self) -> dict[str, Any]:
        hosts = self.host_samples
        gpus = self.gpu_samples
        per_gpu: dict[int, list[GpuSample]] = {}
        for sample in gpus:
            per_gpu.setdefault(sample.index, []).append(sample)

        first_swap = hosts[0].swap_used_mib if hosts else None
        last_swap = hosts[-1].swap_used_mib if hosts else None
        swap_delta = (
            None if first_swap is None or last_swap is None else last_swap - first_swap
        )
        return {
            "sample_attempts": self.attempts,
            "successful_sample_attempts": self.successful_attempts,
            "sample_failure_count": len(self.failures),
            "sample_failures": list(self.failures),
            "host": {
                "observation_count": len(hosts),
                "minimum_mem_available_mib": _minimum(
                    [sample.mem_available_mib for sample in hosts]
                ),
                "swap_used_mib": {
                    "first": first_swap,
                    "last": last_swap,
                    "delta": swap_delta,
                    "peak": _maximum([sample.swap_used_mib for sample in hosts]),
                },
                "load_1m": _distribution([sample.load_1m for sample in hosts]),
            },
            "gpu": {
                "observation_count": len(gpus),
                "device_indices": sorted(per_gpu),
                "peak_memory_used_mib": _maximum(
                    [sample.memory_used_mib for sample in gpus]
                ),
                "utilization_percent": _distribution(
                    [sample.utilization_percent for sample in gpus]
                ),
                "peak_temperature_c": _maximum(
                    [sample.temperature_c for sample in gpus]
                ),
                "per_device": {
                    str(index): {
                        "uuid": samples[0].uuid,
                        "name": samples[0].name,
                        "observation_count": len(samples),
                        "memory_total_mib": samples[0].memory_total_mib,
                        "peak_memory_used_mib": _maximum(
                            [sample.memory_used_mib for sample in samples]
                        ),
                        "utilization_percent": _distribution(
                            [sample.utilization_percent for sample in samples]
                        ),
                        "peak_temperature_c": _maximum(
                            [sample.temperature_c for sample in samples]
                        ),
                    }
                    for index, samples in sorted(per_gpu.items())
                },
            },
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(raw: str, *, field_name: str) -> float:
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field_name} is not finite: {raw!r}")
    return value


def _minimum(values: Sequence[float]) -> float | None:
    return min(values) if values else None


def _maximum(values: Sequence[float]) -> float | None:
    return max(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    """Linear percentile matching NumPy's default for deterministic reports."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": _percentile(values, 50.0),
        "p90": _percentile(values, 90.0),
        "maximum": _maximum(values),
    }


def parse_meminfo(text: str) -> tuple[float, float, float]:
    """Return MemAvailable, SwapTotal, and SwapUsed in MiB."""

    values_kib: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        tokens = raw.strip().split()
        if not tokens:
            continue
        values_kib[key] = _finite_float(tokens[0], field_name=key)
    missing = sorted({"MemAvailable", "SwapTotal", "SwapFree"} - values_kib.keys())
    if missing:
        raise ValueError("/proc/meminfo missing fields: " + ", ".join(missing))
    mem_available = values_kib["MemAvailable"] / 1024.0
    swap_total = values_kib["SwapTotal"] / 1024.0
    swap_free = values_kib["SwapFree"] / 1024.0
    if mem_available < 0.0 or swap_total < 0.0 or swap_free < 0.0:
        raise ValueError("/proc/meminfo contains negative memory values")
    if swap_free > swap_total + 1.0e-9:
        raise ValueError("/proc/meminfo SwapFree exceeds SwapTotal")
    return mem_available, swap_total, swap_total - swap_free


def parse_loadavg(text: str) -> tuple[float, float, float]:
    tokens = text.split()
    if len(tokens) < 3:
        raise ValueError("/proc/loadavg has fewer than three fields")
    values = tuple(
        _finite_float(tokens[index], field_name=f"load_{index}") for index in range(3)
    )
    if any(value < 0.0 for value in values):
        raise ValueError("/proc/loadavg contains negative load")
    return values  # type: ignore[return-value]


def parse_nvidia_smi(text: str) -> tuple[GpuSample, ...]:
    rows = [row for row in csv.reader(text.splitlines(), skipinitialspace=True) if row]
    if not rows:
        raise ValueError("nvidia-smi returned no GPU rows")
    samples: list[GpuSample] = []
    seen_indices: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        if len(row) != len(GPU_QUERY):
            raise ValueError(
                f"nvidia-smi row {row_number} has {len(row)} fields; expected {len(GPU_QUERY)}"
            )
        try:
            index = int(row[0].strip())
        except ValueError as exc:
            raise ValueError(f"nvidia-smi row {row_number} has invalid GPU index") from exc
        if index < 0 or index in seen_indices:
            raise ValueError(f"nvidia-smi row {row_number} has duplicate/invalid GPU index {index}")
        seen_indices.add(index)
        uuid = row[1].strip()
        name = row[2].strip()
        if not uuid or not name:
            raise ValueError(f"nvidia-smi row {row_number} lacks GPU identity")
        memory_total = _finite_float(row[3], field_name="memory.total")
        memory_used = _finite_float(row[4], field_name="memory.used")
        utilization = _finite_float(row[5], field_name="utilization.gpu")
        temperature = _finite_float(row[6], field_name="temperature.gpu")
        if memory_total <= 0.0 or not 0.0 <= memory_used <= memory_total:
            raise ValueError(f"nvidia-smi row {row_number} has invalid memory values")
        if not 0.0 <= utilization <= 100.0:
            raise ValueError(f"nvidia-smi row {row_number} has invalid utilization")
        if not -50.0 <= temperature <= 200.0:
            raise ValueError(f"nvidia-smi row {row_number} has invalid temperature")
        samples.append(
            GpuSample(
                index=index,
                uuid=uuid,
                name=name,
                memory_total_mib=memory_total,
                memory_used_mib=memory_used,
                utilization_percent=utilization,
                temperature_c=temperature,
            )
        )
    return tuple(sorted(samples, key=lambda sample: sample.index))


def collect_probe_sample(
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    loadavg_path: Path = Path("/proc/loadavg"),
    nvidia_timeout_s: float = 5.0,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ProbeSample:
    """Collect one probe attempt while preserving independent stage errors."""

    errors: list[str] = []
    host: HostSample | None = None
    gpus: tuple[GpuSample, ...] = ()
    try:
        mem_available, swap_total, swap_used = parse_meminfo(
            meminfo_path.read_text(encoding="utf-8")
        )
        loads = parse_loadavg(loadavg_path.read_text(encoding="utf-8"))
        host = HostSample(mem_available, swap_total, swap_used, *loads)
    except (OSError, ValueError) as exc:
        errors.append(f"host_probe: {type(exc).__name__}: {exc}")

    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(GPU_QUERY),
        "--format=csv,noheader,nounits",
    ]
    try:
        result = run_command(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(nvidia_timeout_s),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"exit {result.returncode}: {detail or 'no diagnostic'}")
        gpus = parse_nvidia_smi(result.stdout)
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
        errors.append(f"gpu_probe: {type(exc).__name__}: {exc}")
    return ProbeSample(_utc_now(), host, gpus, tuple(errors))


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_part_new(part_path: Path, final_path: Path) -> None:
    """Atomically publish a flushed part file without overwriting evidence."""

    os.link(part_path, final_path)
    part_path.unlink()
    directory_fd = os.open(final_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_config(config: MonitorConfig) -> None:
    if not config.run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not config.detector_mode.strip():
        raise ValueError("detector_mode must be non-empty")
    if config.detector_image_size <= 0:
        raise ValueError("detector_image_size must be positive")
    if config.interval_s <= 0.0 or not math.isfinite(config.interval_s):
        raise ValueError("interval_s must be finite and positive")
    if config.duration_s < 0.0 or not math.isfinite(config.duration_s):
        raise ValueError("duration_s must be finite and non-negative")
    if config.wall_timeout_s < 0.0 or not math.isfinite(config.wall_timeout_s):
        raise ValueError("wall_timeout_s must be finite and non-negative")
    if config.nvidia_timeout_s <= 0.0 or not math.isfinite(config.nvidia_timeout_s):
        raise ValueError("nvidia_timeout_s must be finite and positive")
    if config.min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if config.duration_s <= 0.0 and config.completion_manifest is None:
        raise ValueError("set duration_s or completion_manifest; an unbounded monitor is forbidden")
    if (
        config.duration_s <= 0.0
        and config.completion_manifest is not None
        and config.wall_timeout_s <= 0.0
    ):
        raise ValueError("completion-only monitoring requires a positive wall_timeout_s deadman")
    resolved_configs = [path.expanduser().resolve() for path in config.frozen_configs]
    if not resolved_configs:
        raise ValueError("at least one frozen config is required")
    if len(set(resolved_configs)) != len(resolved_configs):
        raise ValueError("frozen config paths must be unique")


def _provenance(config: MonitorConfig) -> dict[str, Any]:
    model = config.detector_model.expanduser().resolve()
    frozen = [path.expanduser().resolve() for path in config.frozen_configs]
    missing = [str(path) for path in (model, *frozen) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing provenance inputs: " + ", ".join(missing))
    return {
        "detector_model": {"path": str(model), "sha256": _sha256_file(model)},
        "frozen_configs": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in frozen
        ],
    }


def _completion_reached(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid completion manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"completion manifest is not a JSON object: {path}")
    status = str(payload.get("status", ""))
    if status == "completed":
        if payload.get("route_complete") is False:
            raise RuntimeError("completion manifest says completed but route_complete is false")
        return True
    if status in {"failed", "interrupted", "aborted"}:
        raise RuntimeError(f"completion manifest reports terminal failure: {status}")
    if status not in {"", "pending", "in_progress", "running"}:
        raise RuntimeError(f"completion manifest has unknown status: {status!r}")
    return False


def _csv_rows(
    *, run_id: str, index: int, elapsed_s: float, sample: ProbeSample
) -> list[dict[str, Any]]:
    host = sample.host
    gpu_rows: Sequence[GpuSample | None] = sample.gpus if sample.gpus else (None,)
    rows: list[dict[str, Any]] = []
    for gpu in gpu_rows:
        rows.append(
            {
                "run_id": run_id,
                "sample_index": index,
                "sampled_utc": sample.sampled_utc,
                "steady_elapsed_s": f"{elapsed_s:.9f}",
                "sample_status": "error" if sample.errors else "ok",
                "sample_error": " | ".join(sample.errors),
                "gpu_index": "" if gpu is None else gpu.index,
                "gpu_uuid": "" if gpu is None else gpu.uuid,
                "gpu_name": "" if gpu is None else gpu.name,
                "gpu_memory_total_mib": "" if gpu is None else gpu.memory_total_mib,
                "gpu_memory_used_mib": "" if gpu is None else gpu.memory_used_mib,
                "gpu_utilization_percent": "" if gpu is None else gpu.utilization_percent,
                "gpu_temperature_c": "" if gpu is None else gpu.temperature_c,
                "mem_available_mib": "" if host is None else host.mem_available_mib,
                "swap_total_mib": "" if host is None else host.swap_total_mib,
                "swap_used_mib": "" if host is None else host.swap_used_mib,
                "load_1m": "" if host is None else host.load_1m,
                "load_5m": "" if host is None else host.load_5m,
                "load_15m": "" if host is None else host.load_15m,
            }
        )
    return rows


def run_monitor(
    config: MonitorConfig,
    *,
    sampler: Callable[[], ProbeSample] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run the bounded monitor and return its completed immutable manifest."""

    out_dir = config.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_csv = out_dir / FINAL_CSV_NAME
    part_csv = final_csv.with_suffix(final_csv.suffix + ".part")
    final_manifest = out_dir / FINAL_MANIFEST_NAME
    progress_manifest = out_dir / PROGRESS_MANIFEST_NAME
    failed_manifest = out_dir / FAILED_MANIFEST_NAME
    artifacts = (final_csv, part_csv, final_manifest, progress_manifest, failed_manifest)
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise ResourceMonitorError(
            "refusing to reuse resource-monitor output: " + ", ".join(existing)
        )

    started_utc = _utc_now()
    accumulator = SummaryAccumulator()
    csv_handle: Any = None
    base_manifest: dict[str, Any] = {
        "schema_version": "bigwarehouse_runtime_resource_monitor_v1",
        "status": "initializing",
        "started_utc": started_utc,
        "run_id": str(config.run_id),
        "contains_ground_truth": False,
        "subscribes_to_ros": False,
        "detector_runtime": {
            "mode": str(config.detector_mode),
            "image_size": int(config.detector_image_size),
        },
        "sampling": {
            "clock": "steady_monotonic",
            "interval_s": float(config.interval_s),
            "duration_s": float(config.duration_s),
            "wall_timeout_s": float(config.wall_timeout_s),
            "min_samples": int(config.min_samples),
            "nvidia_timeout_s": float(config.nvidia_timeout_s),
        },
    }
    start_s: float | None = None
    stop_reason = "initialization_error"
    try:
        _validate_config(config)
        base_manifest["provenance"] = _provenance(config)
        if config.completion_manifest is not None:
            base_manifest["completion_manifest_path"] = str(
                config.completion_manifest.expanduser().resolve()
            )
        base_manifest["status"] = "in_progress"
        _atomic_json_new(progress_manifest, base_manifest)

        csv_handle = part_csv.open("x", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        start_s = monotonic()
        next_sample_s = start_s
        sample_index = 0
        sampler_fn = sampler or (
            lambda: collect_probe_sample(nvidia_timeout_s=config.nvidia_timeout_s)
        )

        while True:
            now_s = monotonic()
            elapsed_s = max(0.0, now_s - start_s)
            if sample_index > 0:
                try:
                    completion_reached = _completion_reached(config.completion_manifest)
                except BaseException:
                    stop_reason = "completion_manifest_error"
                    raise
                if completion_reached:
                    stop_reason = "completion_manifest"
                    break
            if sample_index > 0 and config.duration_s > 0.0 and elapsed_s >= config.duration_s:
                stop_reason = "steady_duration_reached"
                break
            if config.wall_timeout_s > 0.0 and elapsed_s >= config.wall_timeout_s:
                stop_reason = "steady_clock_deadman"
                raise RuntimeError("steady-clock deadman reached before requested completion")
            if now_s < next_sample_s:
                sleep_for = next_sample_s - now_s
                if config.completion_manifest is not None:
                    sleep_for = min(sleep_for, 0.1)
                sleep(sleep_for)
                continue

            sample_elapsed_s = max(0.0, monotonic() - start_s)
            try:
                sample = sampler_fn()
            except BaseException:
                stop_reason = "sampler_exception"
                raise
            if not isinstance(sample, ProbeSample):
                raise TypeError("sampler must return ProbeSample")
            for row in _csv_rows(
                run_id=config.run_id,
                index=sample_index,
                elapsed_s=sample_elapsed_s,
                sample=sample,
            ):
                writer.writerow(row)
            csv_handle.flush()
            accumulator.add(index=sample_index, elapsed_s=sample_elapsed_s, sample=sample)
            sample_index += 1
            next_sample_s = start_s + sample_index * config.interval_s

        if accumulator.attempts < config.min_samples:
            stop_reason = "minimum_samples_not_met"
            raise RuntimeError(
                f"resource samples {accumulator.attempts} < required {config.min_samples}"
            )
        if accumulator.failures:
            stop_reason = "resource_sample_failure"
            raise RuntimeError(
                f"{len(accumulator.failures)} resource sample attempt(s) failed"
            )
        if csv_handle is None:
            raise RuntimeError("resource CSV was not initialized")
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        csv_handle.close()
        csv_handle = None
        _finalize_part_new(part_csv, final_csv)

        completion_reference = None
        if config.completion_manifest is not None and config.completion_manifest.is_file():
            completion = config.completion_manifest.expanduser().resolve()
            completion_reference = {"path": str(completion), "sha256": _sha256_file(completion)}
        finished = {
            **base_manifest,
            "status": "completed",
            "finished_utc": _utc_now(),
            "stop_reason": stop_reason,
            "steady_elapsed_s": max(0.0, monotonic() - start_s),
            "summary": accumulator.report(),
            "raw_csv": {"path": str(final_csv), "sha256": _sha256_file(final_csv)},
            "completion_manifest": completion_reference,
        }
        progress_manifest.unlink(missing_ok=True)
        _atomic_json_new(final_manifest, finished)
        return finished
    except BaseException as exc:
        if csv_handle is not None:
            try:
                csv_handle.flush()
                os.fsync(csv_handle.fileno())
            finally:
                csv_handle.close()
        progress_manifest.unlink(missing_ok=True)
        failed = {
            **base_manifest,
            "status": "failed",
            "finished_utc": _utc_now(),
            "stop_reason": stop_reason,
            "failure_message": f"{type(exc).__name__}: {exc}",
            "steady_elapsed_s": (
                None if start_s is None else max(0.0, monotonic() - start_s)
            ),
            "summary": accumulator.report(),
            "raw_csv_part": str(part_csv) if part_csv.exists() else None,
        }
        try:
            _atomic_json_new(failed_manifest, failed)
        except BaseException as manifest_exc:
            raise ResourceMonitorError(
                f"resource monitor failed ({exc}); failed manifest publication also failed: "
                f"{manifest_exc}"
            ) from exc
        raise ResourceMonitorError(f"resource monitor failed: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", "--detector-mode", dest="detector_mode", required=True)
    parser.add_argument(
        "--image-size",
        "--detector-image-size",
        dest="detector_image_size",
        type=int,
        required=True,
    )
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument(
        "--config",
        "--frozen-config",
        dest="frozen_config",
        type=Path,
        action="append",
        required=True,
        help="Immutable runtime/study config; repeat for every frozen config.",
    )
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--completion-manifest", type=Path, default=None)
    parser.add_argument("--wall-timeout-s", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--nvidia-timeout-s", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = MonitorConfig(
        out_dir=args.out_dir,
        run_id=str(args.run_id),
        detector_mode=str(args.detector_mode),
        detector_image_size=int(args.detector_image_size),
        detector_model=args.detector_model,
        frozen_configs=tuple(args.frozen_config),
        interval_s=float(args.interval_s),
        duration_s=float(args.duration_s),
        completion_manifest=args.completion_manifest,
        wall_timeout_s=float(args.wall_timeout_s),
        min_samples=int(args.min_samples),
        nvidia_timeout_s=float(args.nvidia_timeout_s),
    )
    try:
        manifest = run_monitor(config)
    except ResourceMonitorError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
