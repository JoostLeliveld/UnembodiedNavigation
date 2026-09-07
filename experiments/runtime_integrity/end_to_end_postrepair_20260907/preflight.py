"""Fail-closed readiness check for the successor acceptance runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

RUNTIME_OWNERS = {
    "coordinator_ledger", "01_state", "02_timing", "03_commands", "04_acquisition",
    "05_geometry", "06_admission", "07_manager", "08_planner", "09_mission",
    "10_logger", "11_offline", "12_capture", "13_campaign", "14_physical",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_handoff(owner: str, record: dict, problems: list[str]) -> None:
    relative = record.get("handoff")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        problems.append(f"ready owner has no repository-relative handoff: {owner}")
        return
    path = ROOT / relative
    if not path.is_file():
        problems.append(f"owner handoff absent: {owner}: {relative}")
        return
    try:
        handoff = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        problems.append(f"owner handoff unreadable: {owner}: {exc}")
        return
    if handoff.get("owner") != owner or handoff.get("ready") is not True:
        problems.append(f"owner handoff identity/readiness mismatch: {owner}")
    if handoff.get("exit_code") != 0 or handoff.get("source_stable_during_tests") is not True:
        problems.append(f"owner handoff lacks passing stable-source execution: {owner}")
    hashes = handoff.get("source_sha256", handoff.get("source_hashes"))
    if not isinstance(hashes, dict) or not hashes:
        problems.append(f"owner handoff has no source hashes: {owner}")
        return
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).is_absolute() or not isinstance(expected, str):
            problems.append(f"owner handoff has invalid hash entry: {owner}")
            continue
        source = ROOT / name
        if not source.is_file():
            problems.append(f"owner handoff source absent: {owner}: {name}")
        elif _sha256(source) != expected:
            problems.append(f"owner handoff source changed: {owner}: {name}")

def check(stage: str) -> dict:
    handoffs = json.loads((HERE / "owner_handoffs.json").read_text())
    commands = json.loads((HERE / "acceptance_commands.json").read_text())
    problems: list[str] = []

    for group in commands["desired_invariant_groups"]:
        for item in group["command"]:
            if item.startswith("tests/") and not (ROOT / item).exists():
                problems.append(f"missing desired-invariant test: {item}")
    for item in commands["recorded_cross_chain"]["command"]:
        if item.endswith(".py") and not (ROOT / item).exists():
            problems.append(f"missing recorded cross-chain test: {item}")

    required = set(RUNTIME_OWNERS)
    if stage == "final":
        # This stage supplies the evidence that can make 15_cross_chain ready;
        # requiring its completed handoff here would be circular.
        required.add("documentation_I14")
    for owner in sorted(required):
        record = handoffs["owners"].get(owner)
        if not record or not record.get("ready"):
            problems.append(f"owner not ready: {owner}")
        else:
            _check_handoff(owner, record, problems)

    ready_path = HERE / "READY.json"
    ready = None
    if not ready_path.exists():
        problems.append("READY.json absent")
    else:
        ready = json.loads(ready_path.read_text())
        for key in ("source_identity", "resolved_config_identity", "schema_version", "owner_handoff_snapshot",
                    "source_batch_identity_contract"):
            if not ready.get(key):
                problems.append(f"READY.json missing {key}")
        expected_handoffs = ready.get("owner_handoff_snapshot")
        actual_handoffs = _sha256(HERE / "owner_handoffs.json")
        if expected_handoffs and expected_handoffs != actual_handoffs:
            problems.append("READY.json owner_handoff_snapshot does not match owner_handoffs.json")

        if stage == "final":
            fixture = ready.get(commands["recorded_cross_chain"]["fixture_from_ready_key"])
            if not fixture:
                problems.append("READY.json missing cross_chain_fixture")
            elif not (ROOT / fixture).is_file():
                problems.append(f"cross-chain fixture absent: {fixture}")

    if stage in {"physical", "final"} and commands["physical_integration"].get("command") is None:
        problems.append("owner-14 physical command absent")

    return {"stage": stage, "ready": not problems, "problems": problems, "ready_record": ready}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("deterministic", "physical", "final"), default="deterministic")
    args = parser.parse_args()
    result = check(args.stage)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 2)
