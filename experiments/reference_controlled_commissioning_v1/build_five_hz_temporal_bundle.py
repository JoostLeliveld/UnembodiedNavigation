#!/usr/bin/env python3
"""Export a conservative five-Hz view of the commissioned M3+M4/R4 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inflation", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("temporal bundle outputs are immutable; choose a new directory")
    if args.inflation < 1.0:
        raise ValueError("temporal covariance inflation must be at least one")
    args.output.mkdir(parents=True)
    base_path = args.base.resolve(strict=True)
    model = json.loads(base_path.read_text(encoding="utf-8"))
    if model.get("mean_model") != "M4_visibility_patch_residual":
        raise ValueError("base bundle is not the commissioned M3+M4 mean chain")
    if model.get("runtime_covariance_model") != "R4_image_conditioned_scale":
        raise ValueError("base bundle is not R4")
    if float(model.get("runtime_query", {}).get("fusion_rate_hz", 0.0)) != 1.0:
        raise ValueError("base bundle must be the one-Hz commissioned release")

    model["image_scale"]["external_calibration_scale"] *= args.inflation
    model["spatial"]["external_calibration_scale"] *= args.inflation
    model["runtime_query"]["fusion_rate_hz"] = 5
    model["runtime_query"]["temporal_covariance_inflation"] = args.inflation
    model["temporal_dependence"] = {
        "method": "conservative_information_thinning",
        "covariance_multiplier_per_five_hz_frame": args.inflation,
        "information_ceiling": (
            "five successive frames contribute no more measurement information "
            "than one uninflated commissioned update"
        ),
        "observed_five_hz_lag1_median_absolute": 0.5505110860407658,
        "observed_stride_five_lag1_median_absolute": 0.10080566984856379,
        "choice": (
            "Multiplier five is more conservative than an AR(1)-derived finite-window "
            "variance inflation and preserves the validated one-Hz information budget."
        ),
    }
    model["derived_from"] = {"path": str(base_path), "sha256": sha256(base_path)}
    runtime_path = args.output / "commissioned_visibility_runtime_model.json"
    write_json(runtime_path, model)
    report = {
        "schema": "commissioned_visibility_five_hz_temporal_export.v1",
        "status": "frozen_before_route_selection_and_navigation",
        "runtime_model": {"path": runtime_path.name, "sha256": sha256(runtime_path)},
        "base_model": {"path": str(base_path), "sha256": sha256(base_path)},
        "covariance_inflation": args.inflation,
        "detector_rate_hz": 5,
        "fusion_rate_hz": 5,
        "mean_chain": "M3_box_mlp plus gated M4_visibility_patch_residual",
        "covariance_chain": "R4_image_conditioned_scale times temporal inflation",
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
    }
    write_json(args.output / "manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
