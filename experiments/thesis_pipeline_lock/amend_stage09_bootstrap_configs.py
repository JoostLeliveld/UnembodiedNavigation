#!/usr/bin/env python3
"""Create immutable Stage-09 configs with explicit start-prior bootstrap support."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN_V5 = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_frozen_routes_v5.yaml"
PREFLIGHT_V8 = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_runtime_preflight_v8.yaml"
EXPECTED = {
    CAMPAIGN_V5: "77224ce3eb7bff310047c0f1fac3acf86271c6e0875791dfdef69a83b4b46815",
    PREFLIGHT_V8: "b2a844d8d3cfb8c677688f35faf5a7d3e3a744e2c69d7fb46397fe0300fd05d5",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _amend(source: Path, destination: Path) -> None:
    if _sha(source) != EXPECTED[source]:
        raise RuntimeError(f"frozen source config changed: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable config: {destination}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload.get("manager_use_task_start_as_bootstrap_prior") is not True:
        raise RuntimeError("start-prior bootstrap was not enabled in source config")
    if int(payload.get("manager_bootstrap_min_cameras", 0)) != 2:
        raise RuntimeError("source config does not state the two-support bootstrap rule")
    payload["manager_bootstrap_prior_counts_as_support"] = True
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, width=1000), encoding="utf-8"
    )
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    args = parser.parse_args()
    _amend(CAMPAIGN_V5, args.campaign_output.resolve())
    _amend(PREFLIGHT_V8, args.preflight_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
