#!/usr/bin/env python3
"""Evaluate saved mean-only world-position ensembles on a separate capture.

Model and ensemble selection remain fixed by the historical validation split.  The
transfer capture is read only after those choices and is reported as development
evidence, never as an untouched final test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_world_position as training


ARMS = ("scalar_balanced", "rgb_uniform", "rgb_balanced")


def load_prediction(payload_path: Path, data: dict[str, np.ndarray],
                    device: torch.device, batch: int) -> np.ndarray:
    payload = torch.load(payload_path, map_location=device, weights_only=False)
    if tuple(payload["feature_names"]) != training.FEATURE_NAMES:
        raise RuntimeError(f"feature contract mismatch: {payload_path}")
    centre = np.asarray(payload["feature_mean"], dtype=np.float32)
    spread = np.asarray(payload["feature_std"], dtype=np.float32)
    features = torch.from_numpy(((data["features"] - centre) / spread).astype(np.float32))
    model = training.WorldResidualNet(len(training.FEATURE_NAMES), payload["use_image"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    indices = np.flatnonzero(data["eligible"])
    return training.infer(model, data["images"], features, indices, device, batch)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--transfer-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = training.prepare()
    base["split"] = np.where(base["split"] == "development",
                             "historical_development", base["split"])
    transfer = training.prepare_capture(args.transfer_capture, role_override="development")
    data = training.combine_datasets([base, transfer])

    raw = np.zeros_like(data["target_ray"], dtype=np.float32)
    result: dict[str, object] = {
        "schema": "image_world_transfer_evaluation.v1",
        "status": "independent_spatial_development_not_final_test",
        "device": str(device),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "transfer_capture": str(args.transfer_capture.resolve()),
        "census": training.census(data),
        "raw": training.score_prediction(data, raw, "development"),
        "arms": {},
        "protocol": {
            "selection": "ensemble aggregation and gates selected on historical validation only",
            "reported_role": "development",
            "predicts_R": False,
            "output": "world-frame robot-centre XY",
        },
    }
    for arm in ARMS:
        paths = sorted(args.checkpoint_dir.glob(f"{arm}_seed*.pt"))
        if not paths:
            raise FileNotFoundError(f"no checkpoints for {arm} in {args.checkpoint_dir}")
        stack = np.stack([load_prediction(path, data, device, args.batch) for path in paths])
        ensemble, aggregation = training.choose_seed_aggregation(data, stack)
        arm_result: dict[str, object] = {
            "checkpoint_count": len(paths),
            "aggregation": aggregation,
            "validation": training.score_prediction(data, ensemble, "validation"),
            "development": training.score_prediction(data, ensemble, "development"),
        }
        if arm.startswith("rgb_"):
            arm_result["gates"] = training.choose_simple_gates(data, stack, ensemble)
        result["arms"][arm] = arm_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
