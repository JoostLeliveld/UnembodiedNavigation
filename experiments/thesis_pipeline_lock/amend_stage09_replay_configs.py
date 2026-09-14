#!/usr/bin/env python3
"""Version Stage-09 configs after the supported-motion replay correction."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN_V7 = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_frozen_routes_v7.yaml"
PREFLIGHT_V10 = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_runtime_preflight_v10.yaml"
EXPECTED = {
    CAMPAIGN_V7: "c873a08ab6aacdba50cd064d61690b032d52b0b8a444d36f85b30fdacda38002",
    PREFLIGHT_V10: "dcbc6f52b9d876ccaf0de804db728f4eff331c1b78f7d9854fb24a09764df1b7",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def amend(source: Path, destination: Path, *, preflight: bool) -> None:
    if sha256(source) != EXPECTED[source]:
        raise RuntimeError(f"frozen source config changed: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable config: {destination}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload.get("heading_update_mode") != "camera_xy_only":
        raise RuntimeError("source is not the frozen camera-xy-only regime")
    if float(payload.get("state_max_predict_dt_s", -1)) != 1.5:
        raise RuntimeError("source motion-sample gap bound changed")
    suffix = (
        " Timestamped odometry may bridge a longer camera outage only when the "
        "entire replay interval has motion samples with no gap above 1.5 s; "
        "unsupported intervals remain refused."
    )
    payload["study_comparison"] = str(payload["study_comparison"]).rstrip() + suffix
    if preflight:
        payload["study_title"] = (
            "Stage-09 supported-route 960-pixel supported-replay runtime preflight"
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
    amend(CAMPAIGN_V7, args.campaign_output.resolve(), preflight=False)
    amend(PREFLIGHT_V10, args.preflight_output.resolve(), preflight=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
