#!/usr/bin/env python3
"""Losslessly compress bulky JSONL evidence after a campaign validates each run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def completed_attempts(campaign_root: Path) -> tuple[set[Path], bool]:
    ledger_path = campaign_root / "campaign_log.json"
    if not ledger_path.is_file():
        return set(), False
    ledger = json.loads(ledger_path.read_text())
    attempts = {
        Path(entry["run_log_dir"]).resolve()
        for entry in ledger.values()
        if entry.get("finished_at") and entry.get("attempt_evidence_complete") is True
        and entry.get("run_log_dir")
    }
    complete = bool(ledger) and all(entry.get("finished_at") for entry in ledger.values())
    return attempts, complete


def compress(path: Path) -> bool:
    target = path.with_name(path.name + ".zst")
    if target.is_file() or not path.is_file():
        return False
    temporary = target.with_name(target.name + ".partial")
    subprocess.run(
        ["zstd", "-q", "-T0", "-3", "-f", str(path), "-o", str(temporary)],
        check=True,
    )
    subprocess.run(["zstd", "-q", "-t", str(temporary)], check=True)
    temporary.replace(target)
    path.unlink()
    return True


def compress_attempt(attempt: Path) -> int:
    changed = 0
    for name in ("detector_outcomes.jsonl", "manager_outcomes.jsonl"):
        changed += int(compress(attempt / name))
    for run in attempt.glob("experiment_*"):
        for name in (
            "runtime_event_deliveries.jsonl",
            "camera_opportunities.jsonl",
            "belief_predictions.jsonl",
        ):
            changed += int(compress(run / name))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-s", type=float, default=20.0)
    args = parser.parse_args()
    seen: set[Path] = set()
    while True:
        attempts, campaign_complete = completed_attempts(args.campaign_root.resolve())
        for attempt in sorted(attempts - seen):
            compress_attempt(attempt)
            seen.add(attempt)
        if not args.watch or campaign_complete:
            return 0
        time.sleep(max(args.interval_s, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
