#!/usr/bin/env python3
"""Assemble a complete commissioning root from original and amended recovery drives.

The assembler never copies or rewrites a drive.  It validates each selected drive and
creates a provenance-recorded directory of symlinks.  A recovery drive is selected only
when the original drive is absent or failed closed and the frozen amendment permits the
original and amended protocol hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def valid_drive(path: Path, *, audit: bool) -> tuple[bool, dict | None]:
    result_path = path / "drive_result.json"
    if not result_path.is_file():
        return False, None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required = (
        result.get("passed") is True
        and result.get("valid_run") is True
        and result.get("runner_wall_timeout") is False
        and result.get("correction_ledger_valid") is True
        and result.get("terminal_stop_verified") is True
        and result.get("audit_outcomes_sealed") is audit
    )
    if not audit:
        integrity_path = path / "tables" / "integrity.json"
        required = required and integrity_path.is_file()
        if integrity_path.is_file():
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            required = required and integrity.get("passed") is True
    return bool(required), result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    amendment_path = args.amendment.resolve()
    original_root = args.original_root.resolve()
    recovery_root = args.recovery_root.resolve()
    output_root = args.output_root.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amended_hash = sha256(protocol_path)
    if amendment.get("status") != "frozen_before_recovery_capture":
        raise RuntimeError("recovery amendment was not frozen before capture")
    if amendment.get("amended_protocol_sha256") != amended_hash:
        raise RuntimeError("current protocol does not match the frozen amended hash")
    allowed_hashes = {
        amendment.get("superseded_protocol_sha256"),
        amendment.get("amended_protocol_sha256"),
    }
    if None in allowed_hashes or len(allowed_hashes) != 2:
        raise RuntimeError("amendment must name distinct original and amended hashes")

    selected: list[dict] = []
    for drive in sorted(protocol["drives"], key=lambda item: int(item["order"])):
        drive_id = str(drive["id"])
        audit = drive["partition"] == "audit"
        original_ok, _ = valid_drive(original_root / drive_id, audit=audit)
        recovery_ok, _ = valid_drive(recovery_root / drive_id, audit=audit)
        if original_ok:
            source_root = original_root
        elif recovery_ok:
            source_root = recovery_root
        else:
            raise RuntimeError(f"no valid source for {drive_id}")
        source = source_root / drive_id
        manifest_path = source / "drive_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        captured_hash = manifest.get("protocol_sha256")
        if captured_hash not in allowed_hashes:
            raise RuntimeError(f"{drive_id}: protocol hash is outside the amendment")
        if manifest.get("drive", {}).get("id") != drive_id:
            raise RuntimeError(f"{drive_id}: drive manifest identity mismatch")
        selected.append({
            "drive_id": drive_id,
            "partition": drive["partition"],
            "source": str(source),
            "source_kind": "recovery" if source_root == recovery_root else "original",
            "captured_protocol_sha256": captured_hash,
            "drive_manifest_sha256": sha256(manifest_path),
            "drive_result_sha256": sha256(source / "drive_result.json"),
        })

    if args.dry_run:
        for item in selected:
            print(f"{item['drive_id']} <- {item['source_kind']} ({item['captured_protocol_sha256'][:12]})")
        return 0
    output_root.mkdir(parents=True, exist_ok=False)
    for item in selected:
        os.symlink(item["source"], output_root / item["drive_id"], target_is_directory=True)
    now = time.time()
    execution = {
        "schema": "reference_controlled_commissioning_execution.v1",
        "assembly_schema": "reference_controlled_commissioning_recovery_assembly.v1",
        "protocol": str(protocol_path),
        "protocol_sha256": amended_hash,
        "timeout_amendment": str(amendment_path),
        "timeout_amendment_sha256": sha256(amendment_path),
        "original_root": str(original_root),
        "recovery_root": str(recovery_root),
        "output_root": str(output_root),
        "started_wall_unix_s": now,
        "finished_wall_unix_s": now,
        "status": "collection_complete_audit_sealed",
        "completed_drive_ids": [item["drive_id"] for item in selected],
        "failed_drive_id": None,
        "current_drive_id": None,
        "audit_analysis_permitted": False,
        "drive_sources": selected,
    }
    atomic_json(output_root / "campaign_execution.json", execution)
    print(json.dumps({
        "output_root": str(output_root),
        "status": execution["status"],
        "drive_count": len(selected),
        "original_drives": sum(item["source_kind"] == "original" for item in selected),
        "recovery_drives": sum(item["source_kind"] == "recovery" for item in selected),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
