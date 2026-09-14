#!/usr/bin/env python3
"""Create immutable Stage-09 configs that keep yaw on the encoder-odometry channel."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN_V6 = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_frozen_routes_v6.yaml"
PREFLIGHT_V9 = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_runtime_preflight_v9.yaml"
EXPECTED = {
    CAMPAIGN_V6: "5303e3b41556deb6dbe509187b39ca8bdee9ab19036e7f052b79ed943b2c765f",
    PREFLIGHT_V9: "8e9699c2a8015bd0e3546d5725d3731c001e9e738553bf9464f60128b9313b9c",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _amend(source: Path, destination: Path, *, preflight: bool) -> None:
    if _sha(source) != EXPECTED[source]:
        raise RuntimeError(f"frozen source config changed: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable config: {destination}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload.get("heading_update_mode") != "coupled":
        raise RuntimeError("source config is not the failed coupled-heading preflight regime")
    payload["heading_update_mode"] = "camera_xy_only"
    if preflight:
        payload["study_title"] = (
            "Stage-09 supported-route 960-pixel camera-xy-only runtime preflight"
        )
        payload["study_comparison"] = (
            "Non-inferential execution preflight. Camera observations update map x/y; "
            "heading remains the independently measured map-frame encoder-odometry yaw."
        )
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
    _amend(CAMPAIGN_V6, args.campaign_output.resolve(), preflight=False)
    _amend(PREFLIGHT_V9, args.preflight_output.resolve(), preflight=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
