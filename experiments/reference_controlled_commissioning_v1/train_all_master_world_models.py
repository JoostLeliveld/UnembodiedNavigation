#!/usr/bin/env python3
"""Fit both perception-only world-position models on all 400 master positions.

This is a deployment fit, not a static evaluation.  Its accuracy summaries are explicitly
training diagnostics.  Independent continuous navigation drives are the evaluation unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import train_static_world_models as models


ROLES = ("detector_fit", "detector_validation", "commissioning_fit", "final_audit")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-epochs", type=int, default=200)
    parser.add_argument("--rgb-epochs", type=int, default=100)
    parser.add_argument("--rgb-warmup-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(2)
    print(f"loading all-master admitted crops on {device}", flush=True)
    data = models.load_dataset(
        args.inference.resolve(), args.gate.resolve(), args.capture.resolve(),
        authorized_roles=ROLES,
    )
    n = len(data["target"])
    indexes = np.arange(n)
    np.savez_compressed(
        staging / "dataset_contract.npz",
        feature=data["feature"], target=data["target"], raw=data["raw"], truth=data["truth"],
        basis=data["basis"], camera=data["camera"], role=data["role"], pose_id=data["pose_id"],
        position_id=data["position_id"], heading_id=data["heading_id"], block_id=data["block_id"],
        image_sha1=data["image_sha1"],
    )

    reports = {}
    histories = {}
    predictions = {}
    specifications = {
        "box_spatial_mlp": (args.box_epochs, 0),
        "rgb_gaussian": (args.rgb_epochs, args.rgb_warmup_epochs),
    }
    for kind, (epochs, warmup) in specifications.items():
        seed = 20266914 + (1 if kind == "rgb_gaussian" else 0)
        model, feature_mean, feature_std, history = models.fit_model(
            kind, data, indexes, epochs=epochs, warmup_epochs=warmup,
            seed=seed, batch_size=args.batch_size, device=device,
        )
        mean, covariance = models.predict(
            kind, model, data, indexes, feature_mean, feature_std,
            batch_size=args.batch_size, device=device,
        )
        torch.save(
            models.checkpoint_payload(
                kind, model, feature_mean, feature_std, epochs=epochs, seed=seed,
            ),
            staging / f"{kind}_deployment.pt",
        )
        report = {
            "label": "training_fit_diagnostic_not_evaluation",
            "pooled": models.mean_summary(mean, data["target"]),
            "by_camera": {
                camera: models.mean_summary(
                    mean[data["camera"] == camera], data["target"][data["camera"] == camera]
                )
                for camera in models.CAMERAS
            },
        }
        if covariance is not None:
            report["covariance_training_diagnostic"] = models.covariance_summary(
                mean, data["target"], covariance
            )
        reports[kind] = report
        histories[kind] = history
        predictions[kind] = (mean, covariance)

    np.savez_compressed(
        staging / "training_predictions.npz",
        target_ray=data["target"], basis=data["basis"], raw_world=data["raw"],
        truth_world=data["truth"], camera=data["camera"], role=data["role"],
        box_spatial_mean=predictions["box_spatial_mlp"][0],
        rgb_gaussian_mean=predictions["rgb_gaussian"][0],
        rgb_gaussian_covariance=predictions["rgb_gaussian"][1],
    )
    (staging / "training_history.json").write_text(
        json.dumps(histories, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "schema": "static_perception_world_models_all_master.v1",
        "status": "deployment_fit_complete_navigation_evaluation_pending",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "population": {
            "roles": list(ROLES),
            "opportunities": int(data["opportunities"][0]),
            "admitted": n,
            "positions_with_admitted_observation": int(len(set(data["position_id"].tolist()))),
            "master_positions": 400,
            "static_holdout": "none",
            "evaluation": "independent continuous navigation drives",
        },
        "sensor_gate": "commissioning_sensor_gate_v2",
        "features": list(models.FEATURE_NAMES),
        "image_input": {
            "shape": [4, models.IMAGE_SIZE, models.IMAGE_SIZE],
            "channels": ["red", "green", "blue", "valid_pixel_mask"],
            "context_fraction_each_side": models.CONTEXT_FRACTION,
        },
        "target": "reference robot-centre world XY minus raw box-bottom projection, ray frame metres",
        "public_output": "corrected world XY",
        "reports": reports,
        "reporting_boundary": (
            "All static accuracy and covariance summaries are in-sample training diagnostics. "
            "They are not thesis evaluation results."
        ),
        "training": {
            "box_epochs": args.box_epochs, "rgb_epochs": args.rgb_epochs,
            "rgb_mean_warmup_epochs": args.rgb_warmup_epochs,
            "batch_size": args.batch_size, "fit": "all admitted observations from all 400 master positions",
        },
        "inputs": {
            "inference_manifest": sha256(args.inference.resolve() / "manifest.json"),
            "inference_records": sha256(args.inference.resolve() / "records.jsonl"),
            "capture_manifest": sha256(args.capture.resolve() / "capture_manifest.json"),
            "capture_index": sha256(args.capture.resolve() / "capture_index.csv"),
            "gate": sha256(args.gate.resolve()),
            "implementation": sha256(Path(__file__)),
            "shared_model_implementation": sha256(Path(models.__file__)),
        },
    }
    result_path = staging / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {path.name: sha256(path) for path in staging.iterdir() if path.is_file()}
    (staging / "manifest.json").write_text(
        json.dumps({"schema": "artifact_manifest.v1", "status": "complete", "artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
