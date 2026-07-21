#!/usr/bin/env python3
"""Atomically finalize one immutable D1/D2 campaign run.

This is the only study-local tool that publishes ``completion_manifest.json``.
It does so only after the route driver, operational recorder, and evaluation-
only truth recorder have all published completed, mutually consistent
artifacts.  It never repairs, replaces, or adopts a partial attempt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import campaign_ledger  # noqa: E402


class FinalizationError(RuntimeError):
    """The run is incomplete, inconsistent, or unsafe to finalize."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one fsynced JSON object without replacing prior evidence."""

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


def _select_row(ledger: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    matches = [row for row in ledger["rows"] if row.get("row_id") == row_id]
    if len(matches) != 1:
        raise FinalizationError(f"campaign contains no unique row {row_id!r}")
    return dict(matches[0])


def _select_attempt(row: Mapping[str, Any], attempt_id: str) -> dict[str, Any]:
    matches = [
        attempt for attempt in row["attempts"] if attempt.get("attempt_id") == attempt_id
    ]
    if len(matches) != 1:
        raise FinalizationError(f"row contains no unique attempt {attempt_id!r}")
    return dict(matches[0])


def _unfinished_artifacts(run_dir: Path) -> list[str]:
    patterns = (
        "raw/*.part",
        "evaluation_only/*.part",
        "raw/*.in_progress.json",
        "evaluation_only/*.in_progress.json",
        "raw/*.failed.json",
        "evaluation_only/*.failed.json",
    )
    return sorted(
        str(path.relative_to(run_dir))
        for pattern in patterns
        for path in run_dir.glob(pattern)
        if path.is_file()
    )


def build_completion_payload(
    campaign_root: Path,
    ledger: Mapping[str, Any],
    row: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and fully validate a completion payload without publishing it."""

    run_dir = campaign_root / str(attempt["run_dir"])
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise FinalizationError(f"expected immutable run directory is absent or a symlink: {run_dir}")
    unfinished = _unfinished_artifacts(run_dir)
    if unfinished:
        raise FinalizationError(
            "partial or failed recorder artifacts remain: " + ", ".join(unfinished)
        )
    missing = [
        relative
        for relative in campaign_ledger.REQUIRED_ARTIFACTS
        if not (run_dir / relative).is_file()
    ]
    if missing:
        raise FinalizationError("required artifacts are missing: " + ", ".join(missing))

    route_completion = campaign_ledger._load_json(run_dir / "raw/route_completion.json")
    operational = campaign_ledger._load_json(
        run_dir / "raw/operational_recording_manifest.json"
    )
    evaluation = campaign_ledger._load_json(
        run_dir / "evaluation_only/evaluation_truth_manifest.json"
    )
    artifacts_sha256 = {
        relative: campaign_ledger._sha256(run_dir / relative)
        for relative in campaign_ledger.REQUIRED_ARTIFACTS
    }
    runtime_readiness, route_camera_health, readiness_errors = (
        campaign_ledger.runtime_readiness_completion_evidence(run_dir, ledger, row)
    )
    if readiness_errors or runtime_readiness is None or route_camera_health is None:
        details = list(readiness_errors)
        if runtime_readiness is None:
            details.append("runtime readiness evidence could not be reconstructed")
        if route_camera_health is None:
            details.append("route-wide camera health evidence could not be reconstructed")
        raise FinalizationError(
            "runtime readiness or route-wide camera health is ineligible:\n- "
            + "\n- ".join(sorted(set(details)))
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "row_id": row["row_id"],
        "attempt_id": attempt["attempt_id"],
        "row_tuple": row["row_tuple"],
        "input_sha256": campaign_ledger.input_sha256(ledger["provenance"]),
        "method_freeze": campaign_ledger.method_freeze_snapshot(ledger),
        "method_freeze_sha256": campaign_ledger.method_freeze_sha256(ledger),
        # Embed the exact driver artifact.  Ledger validation also requires and
        # hashes the original file, so neither representation can be forged or
        # silently changed independently.
        "route_completion": route_completion,
        # The startup barrier is a separately immutable artifact.  This binding
        # freezes its hash together with the actual detector runtime/model and
        # frozen-config hashes declared by the operational recorder.
        "runtime_readiness": runtime_readiness,
        # Startup health alone is not enough: this payload is recomputed from
        # every raw camera stream on ledger refresh and covers liveness,
        # freshness, ordering, and maximum recorder-wall-time gaps over the
        # completed route interval.
        "route_camera_health": route_camera_health,
        "run_identity": {
            "run_id": operational.get("run_id"),
            "plan_row_id": operational.get("plan_row_id"),
            "attempt_id": operational.get("attempt_id"),
            "seed": operational.get("seed"),
            "truth_run_id": evaluation.get("run_id"),
            "truth_plan_row_id": evaluation.get("plan_row_id"),
            "truth_attempt_id": evaluation.get("attempt_id"),
            "truth_seed": evaluation.get("seed"),
        },
        "artifacts_sha256": artifacts_sha256,
    }
    errors = campaign_ledger.validate_completion_payload(run_dir, ledger, row, payload)
    if errors:
        raise FinalizationError("run is not eligible for completion:\n- " + "\n- ".join(errors))
    return payload


def finalize_campaign_run(
    campaign_root: str | Path, row_id: str, attempt_id: str = "attempt_001"
) -> dict[str, Any]:
    """Validate and atomically publish one row's completion manifest."""

    root = Path(campaign_root).expanduser().resolve()
    completion_path: Path | None = None
    with campaign_ledger._campaign_lock(root):
        ledger_path = root / campaign_ledger.DEFAULT_LEDGER_NAME
        if not ledger_path.is_file():
            raise FinalizationError(f"campaign ledger does not exist: {ledger_path}")
        ledger = campaign_ledger._load_json(ledger_path)
        campaign_ledger.validate_ledger_contract(ledger)
        campaign_ledger.verify_frozen_inputs(ledger)
        row = _select_row(ledger, str(row_id))
        attempt = _select_attempt(row, str(attempt_id))
        run_dir = root / str(attempt["run_dir"])
        completion_path = run_dir / "completion_manifest.json"
        failure_path = run_dir / "failure_manifest.json"
        if completion_path.exists():
            raise FinalizationError(f"completion manifest already exists: {completion_path}")
        if failure_path.exists():
            raise FinalizationError(f"failure manifest already exists: {failure_path}")
        conflicts = campaign_ledger.active_campaign_attempts(
            root, ledger, exclude_run_dir=str(attempt["run_dir"])
        )
        if conflicts:
            labels = ", ".join(
                f"{item['row_id']}/{item['attempt_id']}" for item in conflicts
            )
            raise FinalizationError(
                "another campaign attempt is active; refuse to finalize while the "
                f"shared ROS domain namespace is occupied ({labels})"
            )
        payload = build_completion_payload(root, ledger, row, attempt)
        try:
            _atomic_json_new(completion_path, payload)
        except FileExistsError as exc:
            raise FinalizationError(
                f"completion manifest appeared concurrently: {completion_path}"
            ) from exc
        # Re-read through the independent ledger validator while the campaign
        # lock is still held.  A failure is reported loudly; the immutable
        # artifact is never hidden or replaced.
        validation_errors = campaign_ledger.validate_completed_attempt(
            root, ledger, row, attempt
        )
        if validation_errors:
            raise FinalizationError(
                "published completion failed independent validation:\n- "
                + "\n- ".join(validation_errors)
            )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    try:
        payload = finalize_campaign_run(args.campaign_root, args.row_id, args.attempt_id)
    except (FinalizationError, campaign_ledger.LedgerError) as exc:
        parser.exit(2, f"campaign finalization error: {exc}\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
