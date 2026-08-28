#!/usr/bin/env python3
"""Train a paired shape-blind or shape-conditioned candidate-state update MLP."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from shape_conditioned_update_model import feature_names, feature_vector, make_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays(rows, *, use_shape):
    x = np.stack([feature_vector(row, use_shape=use_shape) for row in rows])
    y = np.asarray([[float(row["target_dx"]), float(row["target_dy"])] for row in rows], dtype=np.float32)
    return x, y


def train(dataset: Path, output: Path, *, use_shape: bool, epochs: int, patience: int,
          batch: int, learning_rate: float, seed: int, device: str) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset, output = dataset.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    output.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with (dataset / "records.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["gate_pass"] == "1"]
    fit = [row for row in rows if row["residual_split"] == "fit"]
    calibration = [row for row in rows if row["residual_split"] == "calibration"]
    x_fit, y_fit = _arrays(fit, use_shape=use_shape)
    x_cal, y_cal = _arrays(calibration, use_shape=use_shape)
    x_mean, x_std = x_fit.mean(0), x_fit.std(0); x_std[x_std < 1e-6] = 1.0
    y_mean, y_std = y_fit.mean(0), y_fit.std(0); y_std[y_std < 1e-6] = 1.0
    x_fit = (x_fit - x_mean) / x_std; x_cal = (x_cal - x_mean) / x_std
    y_fit_n = (y_fit - y_mean) / y_std
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_model(x_fit.shape[1]).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit_n)),
                        batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    x_cal_t = torch.from_numpy(x_cal).to(torch_device)
    best, best_rmse, best_epoch, stale, history = None, float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(torch_device), yb.to(torch_device)
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): pred = model(x_cal_t).cpu().numpy() * y_std + y_mean
        errors = pred - y_cal
        rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
        history.append({"epoch": epoch, "fit_loss": np.mean(losses), "calibration_rmse_m": rmse})
        if rmse < best_rmse - 1e-6:
            best, best_rmse, best_epoch, stale = copy.deepcopy(model.state_dict()), rmse, epoch, 0
        else:
            stale += 1
        if stale >= patience: break
    artifact = {
        "state_dict": best, "use_shape": use_shape,
        "feature_names": list(feature_names(use_shape)),
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
    }
    model_path = output / "model.pt"; torch.save(artifact, model_path)
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    payload = {
        "status": "trained_provisional_shape_update_model", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
        "model": str(model_path), "model_sha256": _sha256(model_path), "use_shape": use_shape,
        "n_fit_pairs": len(fit), "n_calibration_pairs": len(calibration),
        "best_epoch": best_epoch, "epochs_completed": len(history), "best_calibration_rmse_m": best_rmse,
        "selection": "minimum paired calibration correction-vector RMSE; source val untouched",
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path); parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--use-shape", action="store_true"); parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50); parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto"); args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.out, use_shape=args.use_shape, epochs=args.epochs,
                           patience=args.patience, batch=args.batch, learning_rate=args.learning_rate,
                           seed=args.seed, device=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
