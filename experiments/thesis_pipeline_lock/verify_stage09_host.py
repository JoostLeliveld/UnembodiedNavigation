#!/usr/bin/env python3
"""Fail-fast host check for the frozen Stage-09 GPU rate qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPECTED_FAST_WORLD_SHA256 = "4bb432f83303c12b987a1d54178b42a9202b7ea6a6d31c394fbbcdc1ac8d119c"
EXPECTED_PROTOCOL_SHA256 = "6642127a5afef5410db0e3a9d34bc6c997cd25889b623ef25b33d878640530d7"
FORBIDDEN_PROCESSES = (
    "ign gazebo", "ros_gz_bridge", "efe_agent", "yolo_robot_detector",
    "run_visibility_campaign.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks: dict[str, object] = {}
    failures: list[str] = []

    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2_fast.world.sdf"
    protocol = REPO / "experiments/thesis_pipeline_lock/stage09_rate_qualification_protocol.json"
    checks["fast_world_sha256"] = sha256(world)
    checks["rate_protocol_sha256"] = sha256(protocol)
    if checks["fast_world_sha256"] != EXPECTED_FAST_WORLD_SHA256:
        failures.append("derived fast-world hash drift")
    if checks["rate_protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        failures.append("rate-protocol hash drift")
    derive = command([sys.executable, "experiments/warehouse_v2_sketches/derive_fast_world.py", "--check"])
    checks["fast_world_derivation"] = {"returncode": derive.returncode,
                                        "stdout": derive.stdout.strip(), "stderr": derive.stderr.strip()}
    if derive.returncode:
        failures.append("fast world no longer derives exactly from canonical")

    process = command(["ps", "-eo", "pid=,args="])
    live = []
    for line in process.stdout.splitlines():
        if any(pattern in line for pattern in FORBIDDEN_PROCESSES) and "verify_stage09_host.py" not in line:
            live.append(line.strip())
    checks["conflicting_processes"] = live
    if live:
        failures.append("conflicting simulator/campaign processes are live")

    required_nodes = [Path("/dev/nvidia0"), Path("/dev/nvidiactl"), Path("/dev/nvidia-uvm")]
    checks["nvidia_device_nodes"] = {str(path): path.exists() for path in required_nodes}
    if not all(path.exists() for path in required_nodes):
        failures.append("NVIDIA device nodes are missing; use administrator-managed driver recovery or reboot")

    smi = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                   "--format=csv,noheader,nounits"])
    checks["nvidia_smi"] = {"returncode": smi.returncode, "stdout": smi.stdout.strip(),
                             "stderr": smi.stderr.strip()}
    if smi.returncode:
        failures.append("nvidia-smi cannot communicate with the driver")

    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        checks["torch"] = {"version": torch.__version__, "cuda_available": cuda,
                           "device_count": int(torch.cuda.device_count())}
        if cuda:
            start = time.perf_counter()
            matrix = torch.randn((4096, 4096), device="cuda")
            result = matrix @ matrix
            torch.cuda.synchronize()
            checks["torch"]["matmul_4096_s"] = time.perf_counter() - start
            checks["torch"]["device_name"] = torch.cuda.get_device_name(0)
            checks["torch"]["finite_probe"] = bool(torch.isfinite(result[0, 0]).item())
        else:
            failures.append("PyTorch CUDA is unavailable")
    except Exception as exc:  # host diagnostic must preserve the exact exception
        checks["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("CUDA compute probe raised an exception")

    report = {
        "schema": "thesis_stage09_host_preflight.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "final_audit_accessed": False,
        "checks": checks,
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
