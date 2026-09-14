#!/usr/bin/env python3
"""Compare a 16x16 visibility-patch residual with the tabular box MLP.

The box MLP remains the base correction.  A small convolutional model receives a
runtime-observable soft target-blue visibility grid and predicts only a gated
residual in the camera-ray frame.  Fit-drive predictions are leave-one-drive-out;
development predictions use models trained on all fit drives.  Audit drives are
never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.nn import functional as F

from analyze_commissioned_model import (
    fit_mean,
    load_capture,
    mean_features,
    predict_mean,
)
from visibility_patch_rgb import (
    GRID_SIZE,
    VisibilityPatchResidualNet,
    visibility_grid_from_saved_context,
    visibility_summaries,
)


MODEL_ID = "box_mlp_visibility_residual"
BASE_MODEL_ID = "box_mlp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_visibility(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    grids: list[np.ndarray] = []
    scores: list[float] = []
    sources: list[dict] = []
    for record in records:
        path = Path(record["crop_path"]).resolve()
        if not path.is_file():
            raise RuntimeError(f"missing selected crop {path}")
        with np.load(path, allow_pickle=False) as archive:
            crop = np.asarray(archive["crop"])
            image_shape = np.asarray(archive["image_shape"], dtype=int)
            stamp_ns = int(archive["capture_stamp_ns"])
            source_frame_id = str(archive["source_frame_id"])
            stored_bbox = np.asarray(archive["bbox_xyxy"], dtype=float)
        if stamp_ns != record["stamp_ns"] or source_frame_id != record["source_frame_id"]:
            raise RuntimeError(f"crop identity differs from measurement row: {path}")
        if not np.allclose(stored_bbox, record["bbox_xyxy"], atol=1e-6, rtol=0.0):
            raise RuntimeError(f"crop bbox differs from measurement row: {path}")
        grid = visibility_grid_from_saved_context(
            crop, record["bbox_xyxy"], image_shape, grid_size=GRID_SIZE
        )
        summary = visibility_summaries(grid)
        grids.append(grid)
        scores.append(
            0.5 * summary["visible_blue_fraction"]
            + 0.5 * summary["lower_quarter_blue_fraction"]
        )
        sources.append({
            "path": str(path),
            "sha256": sha256(path),
            "source_frame_id": source_frame_id,
            "capture_stamp_ns": stamp_ns,
            **summary,
        })
    return np.stack(grids), np.asarray(scores, dtype=float), sources


def balanced_tail_weights(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Balance four correction-residual strata defined on the training fold only."""

    magnitude = np.linalg.norm(target - base, axis=1)
    edges = np.unique(np.quantile(magnitude, (0.50, 0.80, 0.95)))
    bins = np.searchsorted(edges, magnitude, side="right")
    counts = np.bincount(bins, minlength=len(edges) + 1).astype(float)
    weights = np.asarray([len(bins) / max(counts[value], 1.0) for value in bins])
    weights /= weights.mean()
    return np.clip(weights, 0.25, 6.0).astype(np.float32)


class FittedVisibilityResidual:
    def __init__(
        self,
        model: VisibilityPatchResidualNet,
        feature_mean: np.ndarray,
        feature_sd: np.ndarray,
        device: torch.device,
    ) -> None:
        self.model = model
        self.feature_mean = feature_mean
        self.feature_sd = feature_sd
        self.device = device

    def predict(
        self,
        grids: np.ndarray,
        features: np.ndarray,
        base: np.ndarray,
        index: np.ndarray,
        *,
        batch_size: int,
        override_grids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = grids if override_grids is None else override_grids
        output = np.empty((len(index), 2), dtype=np.float32)
        gates = np.empty(len(index), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(index), batch_size):
                part = index[start : start + batch_size]
                grid = torch.from_numpy(source[part]).to(self.device)
                standardized = torch.from_numpy(
                    ((features[part] - self.feature_mean) / self.feature_sd).astype(np.float32)
                ).to(self.device)
                baseline = torch.from_numpy(base[part].astype(np.float32)).to(self.device)
                corrected, _, gate = self.model(grid, standardized, baseline)
                output[start : start + len(part)] = corrected.cpu().numpy()
                gates[start : start + len(part)] = gate[:, 0].cpu().numpy()
        return output.astype(float), gates.astype(float)


def fit_visibility_residual(
    grids: np.ndarray,
    features: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> FittedVisibilityResidual:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    feature_mean = features[train].mean(axis=0)
    feature_sd = np.maximum(features[train].std(axis=0), 1e-3)
    model = VisibilityPatchResidualNet(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    weights = balanced_tail_weights(target[train], base[train])
    generator = np.random.default_rng(seed + 1709)
    for _ in range(epochs):
        model.train()
        order = generator.permutation(len(train))
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            part = train[local]
            grid = torch.from_numpy(grids[part]).to(device)
            standardized = torch.from_numpy(
                ((features[part] - feature_mean) / feature_sd).astype(np.float32)
            ).to(device)
            baseline = torch.from_numpy(base[part].astype(np.float32)).to(device)
            truth = torch.from_numpy(target[part].astype(np.float32)).to(device)
            sample_weight = torch.from_numpy(weights[local]).to(device)
            prediction, residual, gate = model(grid, standardized, baseline)
            loss_per_axis = F.smooth_l1_loss(
                prediction, truth, reduction="none", beta=0.02
            )
            data_loss = (loss_per_axis.mean(dim=1) * sample_weight).mean()
            # Keep RGB adjustments conservative unless the data repeatedly support them.
            regularizer = 2e-4 * (gate * residual).square().mean()
            loss = data_loss + regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return FittedVisibilityResidual(model, feature_mean, feature_sd, device)


def metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    drives: np.ndarray,
    index: np.ndarray,
) -> dict[str, Any]:
    error = np.linalg.norm(target[index] - prediction[index], axis=1)
    per_drive = {
        drive: {
            "n": int(np.sum(drives[index] == drive)),
            "mse_m2": float(np.mean(error[drives[index] == drive] ** 2)),
            "rmse_m": float(np.sqrt(np.mean(error[drives[index] == drive] ** 2))),
        }
        for drive in sorted(set(drives[index].tolist()))
    }
    return {
        "n": int(len(index)),
        "drive_count": len(per_drive),
        "equal_drive_mse_m2": float(np.mean([row["mse_m2"] for row in per_drive.values()])),
        "equal_drive_rmse_m": float(np.sqrt(np.mean([row["mse_m2"] for row in per_drive.values()]))),
        "median_error_m": float(np.median(error)),
        "p90_error_m": float(np.percentile(error, 90)),
        "p95_error_m": float(np.percentile(error, 95)),
        "above_1m_count": int(np.sum(error > 1.0)),
        "per_drive": per_drive,
    }


def paired_drive_comparison(base: dict, candidate: dict) -> dict[str, float]:
    drives = sorted(set(base["per_drive"]) & set(candidate["per_drive"]))
    difference = np.asarray([
        candidate["per_drive"][drive]["mse_m2"] - base["per_drive"][drive]["mse_m2"]
        for drive in drives
    ])
    se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else math.inf
    return {
        "drive_count": len(drives),
        "mean_candidate_minus_base_mse_m2": float(difference.mean()),
        "paired_se_m2": se,
        "improves_beyond_one_paired_se": bool(difference.mean() < -se),
    }


def subset_indices(index: np.ndarray, score: np.ndarray, threshold: float) -> np.ndarray:
    return index[score[index] <= threshold]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument(
        "--protocol-snapshot", type=Path,
        help=("Immutable protocol snapshot embedded in every completed drive. Required "
              "when an older capture's absolute live-protocol path has since changed."),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260913)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke-per-drive", type=int, default=0,
        help="Non-inferential code-path check using at most this many records per drive.",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.smoke_per_drive < 0:
        raise ValueError("epochs and batch size must be positive; smoke limit cannot be negative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty candidate output directory")
    args.output.mkdir(parents=True, exist_ok=True)

    execution, records, _ = load_capture(
        args.capture_root.resolve(),
        None if args.protocol_snapshot is None else args.protocol_snapshot.resolve(),
    )
    if execution.get("incomplete_note") and not args.smoke_per_drive:
        raise RuntimeError("inferential comparison requires a complete campaign")
    partition = np.asarray([record["partition"] for record in records])
    drives = np.asarray([record["drive"] for record in records])
    if args.smoke_per_drive:
        selected = []
        for drive in sorted(set(drives.tolist())):
            selected.extend(np.flatnonzero(drives == drive)[: args.smoke_per_drive].tolist())
        records = [records[index] for index in selected]
        partition = np.asarray([record["partition"] for record in records])
        drives = np.asarray([record["drive"] for record in records])

    fit_index = np.flatnonzero(partition == "fit")
    development_index = np.flatnonzero(partition == "development")
    fit_drives = sorted(set(drives[fit_index].tolist()))
    development_drives = sorted(set(drives[development_index].tolist()))
    if len(fit_drives) < 2 or not development_drives:
        raise RuntimeError("comparison needs at least two fit drives and one development drive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    target = np.stack([record["target_ray"] for record in records])
    features = mean_features(records, np.arange(len(records)))
    grids, visibility_score, crop_sources = load_visibility(records)
    base_prediction = np.full((len(records), 2), np.nan, dtype=float)
    candidate_prediction = np.full((len(records), 2), np.nan, dtype=float)
    candidate_gate = np.full(len(records), np.nan, dtype=float)

    for fold, held_drive in enumerate(fit_drives):
        train = fit_index[drives[fit_index] != held_drive]
        test = fit_index[drives[fit_index] == held_drive]
        base_model = fit_mean(records, train, BASE_MODEL_ID, args.seed + fold)
        fold_base = np.full((len(records), 2), np.nan, dtype=float)
        fold_base[train] = predict_mean(records, train, BASE_MODEL_ID, base_model)
        fold_base[test] = predict_mean(records, test, BASE_MODEL_ID, base_model)
        base_prediction[test] = fold_base[test]
        fitted = fit_visibility_residual(
            grids, features, fold_base, target, train,
            seed=args.seed + fold, epochs=args.epochs,
            batch_size=args.batch_size, device=device,
        )
        candidate_prediction[test], candidate_gate[test] = fitted.predict(
            grids, features, fold_base, test, batch_size=args.batch_size
        )

    final_base = fit_mean(records, fit_index, BASE_MODEL_ID, args.seed)
    base_prediction[development_index] = predict_mean(
        records, development_index, BASE_MODEL_ID, final_base
    )
    final_base_prediction = base_prediction.copy()
    final_base_prediction[fit_index] = predict_mean(
        records, fit_index, BASE_MODEL_ID, final_base
    )
    final_rgb = fit_visibility_residual(
        grids, features, final_base_prediction, target, fit_index,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, device=device,
    )
    candidate_prediction[development_index], candidate_gate[development_index] = final_rgb.predict(
        grids, features, final_base_prediction, development_index, batch_size=args.batch_size
    )

    shuffled_grids = grids.copy()
    rng = np.random.default_rng(args.seed + 913)
    cameras = np.asarray([record["camera"] for record in records])
    for camera in sorted(set(cameras[development_index].tolist())):
        members = development_index[cameras[development_index] == camera]
        shuffled_grids[members] = grids[rng.permutation(members)]
    shuffled_prediction = np.full((len(records), 2), np.nan, dtype=float)
    shuffled_prediction[development_index], _ = final_rgb.predict(
        grids, features, final_base_prediction, development_index,
        batch_size=args.batch_size, override_grids=shuffled_grids,
    )

    fit_visibility_threshold = float(np.quantile(visibility_score[fit_index], 0.25))
    development_visibility_threshold = float(
        np.quantile(visibility_score[development_index], 0.25)
    )
    stressed_development = subset_indices(
        development_index, visibility_score, development_visibility_threshold
    )
    base_development = metrics(target, base_prediction, drives, development_index)
    candidate_development = metrics(target, candidate_prediction, drives, development_index)
    shuffled_development = metrics(target, shuffled_prediction, drives, development_index)
    base_stressed = metrics(target, base_prediction, drives, stressed_development)
    candidate_stressed = metrics(target, candidate_prediction, drives, stressed_development)
    base_fit_oof = metrics(target, base_prediction, drives, fit_index)
    candidate_fit_oof = metrics(target, candidate_prediction, drives, fit_index)
    paired = paired_drive_comparison(base_development, candidate_development)
    passes = {
        "full_drive_mse_improves_beyond_one_se": paired["improves_beyond_one_paired_se"],
        "visibility_stressed_p90_improves": (
            candidate_stressed["p90_error_m"] < base_stressed["p90_error_m"]
        ),
        "catastrophic_tail_does_not_increase": (
            candidate_development["above_1m_count"] <= base_development["above_1m_count"]
        ),
        "real_grid_beats_shuffled_grid": (
            candidate_development["equal_drive_mse_m2"]
            < shuffled_development["equal_drive_mse_m2"]
        ),
    }
    selected = bool(all(passes.values())) and not args.smoke_per_drive

    joblib.dump(final_base, args.output / "box_mlp_fit_only.joblib")
    final_rgb.model.cpu()
    checkpoint = {
        "schema": "visibility_patch_residual.v1",
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "state_dict": final_rgb.model.state_dict(),
        "feature_mean": final_rgb.feature_mean.tolist(),
        "feature_sd": final_rgb.feature_sd.tolist(),
        "grid_size": GRID_SIZE,
        "target": "reference-minus-raw correction in camera-ray coordinates",
        "visibility_input": "soft target-blue evidence pooled inside detected bbox",
        "forbidden_runtime_inputs": ["reference", "belief", "innovation", "NIS"],
    }
    torch.save(checkpoint, args.output / "box_mlp_visibility_residual_fit_only.pt")
    final_rgb.model.to(device)
    np.savez_compressed(
        args.output / "candidate_predictions.npz",
        drive_id=drives,
        partition=partition,
        camera_id=np.asarray([record["camera"] for record in records]),
        capture_stamp_ns=np.asarray([record["stamp_ns"] for record in records], dtype=np.int64),
        source_frame_id=np.asarray([record["source_frame_id"] for record in records]),
        target_ray_m=target,
        base_prediction_ray_m=base_prediction,
        candidate_prediction_ray_m=candidate_prediction,
        candidate_gate=candidate_gate,
        visibility_score=visibility_score,
    )

    report = {
        "schema": "visibility_patch_candidate_comparison.v1",
        "status": "smoke_only" if args.smoke_per_drive else "development_selection_complete",
        "audit_analysis_permitted": False,
        "capture_root": str(args.capture_root.resolve()),
        "campaign_execution_sha256": sha256(args.capture_root.resolve() / "campaign_execution.json"),
        "declared_protocol": execution["declared_protocol"],
        "effective_protocol": execution["effective_protocol"],
        "effective_protocol_sha256": execution["effective_protocol_sha256"],
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "fit_drives": fit_drives,
        "development_drives": development_drives,
        "records": len(records),
        "crop_source_count": len(crop_sources),
        "crop_source_digest": hashlib.sha256(
            json.dumps(crop_sources, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "fit_visibility_stress_threshold": fit_visibility_threshold,
        "development_visibility_stress_threshold": development_visibility_threshold,
        "visibility_stress_definition": (
            "lowest runtime-observable visibility-score quartile within each scored partition; "
            "the threshold uses no correction target or error"
        ),
        "development_visibility_stressed_count": int(len(stressed_development)),
        "models": {
            BASE_MODEL_ID: {
                "fit_out_of_fold": base_fit_oof,
                "development": base_development,
                "stressed": base_stressed,
            },
            MODEL_ID: {
                "fit_out_of_fold": candidate_fit_oof,
                "development": candidate_development,
                "stressed": candidate_stressed,
            },
            "shuffled_visibility_control": {"development": shuffled_development},
        },
        "paired_comparison": paired,
        "selection_gates": passes,
        "selected": selected,
        "selection_rule": (
            "Select the visibility-residual candidate only when all predeclared gates pass; "
            "otherwise retain the box MLP. "
            "A smoke run is never selectable."
        ),
        "artifacts": {
            "base_model": {
                "path": "box_mlp_fit_only.joblib",
                "sha256": sha256(args.output / "box_mlp_fit_only.joblib"),
            },
            "candidate_model": {
                "path": "box_mlp_visibility_residual_fit_only.pt",
                "sha256": sha256(args.output / "box_mlp_visibility_residual_fit_only.pt"),
            },
            "predictions": {
                "path": "candidate_predictions.npz",
                "sha256": sha256(args.output / "candidate_predictions.npz"),
            },
        },
    }
    atomic_json(args.output / "visibility_patch_comparison.json", report)
    print(json.dumps({
        "status": report["status"],
        "device": str(device),
        "development_rmse_cm": {
            BASE_MODEL_ID: round(100 * base_development["equal_drive_rmse_m"], 3),
            MODEL_ID: round(100 * candidate_development["equal_drive_rmse_m"], 3),
            "shuffled": round(100 * shuffled_development["equal_drive_rmse_m"], 3),
        },
        "selection_gates": passes,
        "selected": selected,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
