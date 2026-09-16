#!/usr/bin/env python3
"""Unified gate-v2 correction comparison: 12 whole drives fit, 6 held out."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_visibility_patch_rgb as visibility  # noqa: E402
import train_all_drive_world_models as drive_data  # noqa: E402
import train_static_world_models as world_models  # noqa: E402

FEATURE_NAMES = ("raw_range_m", "inverse_raw_range", "ray_bearing_sin",
                 "ray_bearing_cos", "bbox_width_fraction", "bbox_height_fraction",
                 "bbox_aspect", "bbox_bottom_u_fraction", "bbox_bottom_v_fraction",
                 "confidence", *(f"is_camera_{x}" for x in "ABCDE"))

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def metric_block(target, prediction, drives, index):
    residual = prediction[index] - target[index]
    error = np.linalg.norm(residual, axis=1)
    selected = drives[index]
    rows = {}
    for drive in sorted(set(selected.tolist())):
        mask = selected == drive; values = error[mask]
        rows[drive] = {"n": int(mask.sum()), "mean_error_m": float(values.mean()),
            "median_error_m": float(np.median(values)),
            "rmse_m": float(np.sqrt(np.mean(values ** 2))),
            "p90_error_m": float(np.quantile(values, .90)),
            "p95_error_m": float(np.quantile(values, .95)),
            "signed_bias_ray_m": residual[mask].mean(axis=0).tolist()}
    values = list(rows.values())
    aggregate = {"drive_count": len(values)}
    for key in ("mean_error_m", "median_error_m", "rmse_m", "p90_error_m", "p95_error_m"):
        aggregate["equal_drive_" + key] = float(np.mean([r[key] for r in values]))
    aggregate["equal_drive_signed_bias_ray_m"] = np.mean(
        [r["signed_bias_ray_m"] for r in values], axis=0).tolist()
    return {"n": int(len(index)), "equal_drive": aggregate, "per_drive": rows}

def metrics(target, prediction, drives, cameras, index):
    result = metric_block(target, prediction, drives, index)
    result["aggregation"] = "statistic within complete drive, then arithmetic mean over drives"
    result["per_camera_equal_drive"] = {}
    for camera in sorted(set(cameras[index].tolist())):
        members = index[cameras[index] == camera]
        result["per_camera_equal_drive"][camera] = metric_block(
            target, prediction, drives, members)["equal_drive"] | {"n": int(len(members))}
    return result

def fit_tabular(feature, target, train, family, seed):
    model = Ridge(alpha=1.0) if family == "linear_ridge" else MLPRegressor(
        hidden_layer_sizes=(96, 64), max_iter=1200, random_state=seed,
        early_stopping=True, n_iter_no_change=25)
    return make_pipeline(StandardScaler(), model).fit(feature[train], target[train])

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True, type=Path); p.add_argument("--gate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path); p.add_argument("--seed", type=int, default=260915)
    p.add_argument("--visibility-epochs", type=int, default=80); p.add_argument("--rgb-epochs", type=int, default=100)
    p.add_argument("--rgb-warmup-epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); args = p.parse_args()
    output = args.output.resolve(); staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists(): raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    data, source_manifest = drive_data.load_all_drives(args.campaign.resolve(), args.gate.resolve())
    if data["feature"].shape[1] != 15: raise RuntimeError("structured feature contract is not 15")
    partition, drives = data["partition"], data["drive"]
    train = np.flatnonzero(np.isin(partition, ["fit", "development"])); held = np.flatnonzero(partition == "audit")
    train_drives, held_drives = sorted(set(drives[train])), sorted(set(drives[held]))
    if len(train_drives) != 12 or len(held_drives) != 6 or set(train_drives) & set(held_drives):
        raise RuntimeError(f"whole-drive 12/6 split failed: {len(train_drives)}/{len(held_drives)}")
    target, feature = data["target"], data["feature"]
    predictions = {"raw_box": np.zeros_like(target)}; artifacts: dict[str, Any] = {}
    for offset, family in enumerate(("linear_ridge", "box_mlp")):
        fitted = fit_tabular(feature, target, train, family, args.seed + offset)
        predictions[family] = fitted.predict(feature)
        path = staging / f"{family}_train12.joblib"; joblib.dump(fitted, path)
        artifacts[family] = {"path": path.name, "sha256": sha256(path)}
    base = predictions["box_mlp"]; grids = data["visibility_grid"]
    fitted_vis = visibility.fit_visibility_residual(grids, feature, base, target, train,
        seed=args.seed + 100, epochs=args.visibility_epochs, batch_size=args.batch_size, device=device)
    predictions["box_mlp_visibility_residual"], _ = fitted_vis.predict(
        grids, feature, base, np.arange(len(target)), batch_size=args.batch_size)
    path = staging / "box_mlp_visibility_residual_train12.pt"
    torch.save({"schema": "visibility_residual_15feature_12_6.v1", "feature_names": FEATURE_NAMES,
        "state_dict": fitted_vis.model.state_dict(), "feature_mean": fitted_vis.feature_mean.tolist(),
        "feature_sd": fitted_vis.feature_sd.tolist()}, path)
    artifacts["box_mlp_visibility_residual"] = {"path": path.name, "sha256": sha256(path)}
    rgb_model, rgb_mean, rgb_sd, rgb_history = world_models.fit_model("rgb_gaussian", data, train,
        epochs=args.rgb_epochs, warmup_epochs=args.rgb_warmup_epochs, seed=args.seed + 200,
        batch_size=args.batch_size, device=device)
    predictions["rgb_crop_conditioned"], _ = world_models.predict("rgb_gaussian", rgb_model, data,
        np.arange(len(target)), rgb_mean, rgb_sd, batch_size=args.batch_size, device=device)
    path = staging / "rgb_crop_conditioned_train12.pt"
    torch.save({"schema": "rgb_crop_conditioned_mean_15feature_12_6.v1", "model_kind": "rgb_gaussian",
        "feature_names": FEATURE_NAMES, "state_dict": rgb_model.state_dict(),
        "feature_mean": rgb_mean.tolist(), "feature_std": rgb_sd.tolist(), "image_size": 96}, path)
    artifacts["rgb_crop_conditioned"] = {"path": path.name, "sha256": sha256(path)}
    model_reports = {name: {"train_12": metrics(target, pred, drives, data["camera"], train),
        "held_out_6": metrics(target, pred, drives, data["camera"], held)} for name, pred in predictions.items()}
    selected = min(model_reports, key=lambda n: model_reports[n]["held_out_6"]["equal_drive"]["equal_drive_rmse_m"])
    split = {"train_partitions": ["fit", "development"], "held_out_partition": "audit",
        "train_drive_ids": train_drives, "held_out_drive_ids": held_drives,
        "train_rows": int(len(train)), "held_out_rows": int(len(held)), "leakage_check": "passed"}
    pred_path = staging / "candidate_predictions.npz"
    np.savez_compressed(pred_path, target_ray_m=target, drive=drives, historical_partition=partition,
        camera=data["camera"], source_frame_id=data["source_frame_id"], stamp_ns=data["stamp_ns"],
        **{f"{k}_prediction_ray_m": v for k, v in predictions.items()})
    artifacts["predictions"] = {"path": pred_path.name, "sha256": sha256(pred_path)}
    (staging / "split_manifest.json").write_text(json.dumps(split, indent=2) + "\n")
    (staging / "input_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    report = {"schema": "matched_correction_ladder_v3_unified_12_6", "status": "held_out_correction_comparison_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "device": str(device), "sensor_gate_id": "commissioning_sensor_gate_v2",
        "feature_names": FEATURE_NAMES, "forbidden_features_absent": ["raw_x_m", "raw_y_m"],
        "image_contract": {"stored_crop_shape": [3, 96, 96], "model_input_shape": [4, 96, 96],
            "source": "per-drive selected_crops archive", "identity": "camera + capture_stamp_ns + source_frame_id",
            "context_fraction_each_side": .5}, "split": split, "models": model_reports,
        "selected_correction": selected, "selection_rule": "minimum equal-held-out-drive RMSE over the same six drives",
        "training": {"seed": args.seed, "visibility_epochs": args.visibility_epochs, "rgb_epochs": args.rgb_epochs,
            "rgb_warmup_epochs": args.rgb_warmup_epochs, "batch_size": args.batch_size, "rgb_history": rgb_history},
        "artifacts": artifacts, "source_hashes": {Path(__file__).name: sha256(Path(__file__)),
            Path(drive_data.__file__).name: sha256(Path(drive_data.__file__)), Path(world_models.__file__).name: sha256(Path(world_models.__file__))}}
    (staging / "correction_comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(staging, output)
    print(json.dumps({"status": report["status"], "split": split, "selected_correction": selected,
        "held_out_6": {k: v["held_out_6"]["equal_drive"] for k, v in model_reports.items()}}, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
