#!/usr/bin/env python3
"""Train a small MLP to correct detector bottom-centre pixel bias."""

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

from residual_bias_model import FEATURE_NAMES, feature_vector, make_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays(rows: list[dict[str, str]], *, disable_heading: bool):
    x = np.stack([feature_vector(row, disable_heading=disable_heading) for row in rows])
    y = np.asarray([[float(row["target_du"]), float(row["target_dv"])] for row in rows], dtype=np.float32)
    return x, y


def train(
    dataset: Path, output: Path, *, disable_heading: bool, epochs: int,
    patience: int, batch: int, learning_rate: float, seed: int, device: str,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = dataset.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    if not (dataset / ".complete").is_file():
        raise FileNotFoundError(f"dataset is incomplete: {dataset}")
    output.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with (dataset / "records.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["detected"] == "1"]
    fit_rows = [row for row in rows if row["residual_split"] == "fit"]
    calibration_rows = [row for row in rows if row["residual_split"] == "calibration"]
    if not fit_rows or not calibration_rows:
        raise RuntimeError("fit/calibration split is empty")
    x_fit, y_fit = _arrays(fit_rows, disable_heading=disable_heading)
    x_cal, y_cal = _arrays(calibration_rows, disable_heading=disable_heading)
    x_mean, x_std = x_fit.mean(axis=0), x_fit.std(axis=0)
    x_std[x_std < 1.0e-6] = 1.0
    y_mean, y_std = y_fit.mean(axis=0), y_fit.std(axis=0)
    y_std[y_std < 1.0e-6] = 1.0
    x_fit_n = (x_fit - x_mean) / x_std
    x_cal_n = (x_cal - x_mean) / x_std
    y_fit_n = (y_fit - y_mean) / y_std

    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_model(len(FEATURE_NAMES)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit_n), torch.from_numpy(y_fit_n)),
        batch_size=batch, shuffle=True, generator=generator,
    )
    x_cal_tensor = torch.from_numpy(x_cal_n).to(torch_device)
    history = []
    best_state = None
    best_cal_rmse = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(torch_device), y_batch.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            predicted_n = model(x_cal_tensor).cpu().numpy()
        predicted = predicted_n * y_std + y_mean
        residual = predicted - y_cal
        cal_rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        cal_bias = np.mean(residual, axis=0)
        history.append({
            "epoch": epoch, "fit_loss": float(np.mean(losses)),
            "calibration_pixel_rmse": cal_rmse,
            "calibration_bias_u": float(cal_bias[0]), "calibration_bias_v": float(cal_bias[1]),
        })
        if cal_rmse < best_cal_rmse - 1.0e-5:
            best_cal_rmse, best_epoch = cal_rmse, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    artifact = {
        "state_dict": best_state,
        "feature_names": list(FEATURE_NAMES),
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        "disable_heading": disable_heading,
        "architecture": "13-64-64-32-2 SiLU MLP",
    }
    model_path = output / "model.pt"
    torch.save(artifact, model_path)
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    payload = {
        "status": "trained_provisional_not_runtime_commissioned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
        "model": str(model_path), "model_sha256": _sha256(model_path),
        "online_inputs": [
            "detector box geometry/confidence", "fixed camera identity",
            *( [] if disable_heading else ["heading relative to detected camera bearing"] ),
        ],
        "heading_conditioned": not disable_heading,
        "n_fit": len(fit_rows), "n_calibration": len(calibration_rows),
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_calibration_pixel_rmse": best_cal_rmse,
        "seed": seed, "device": str(torch_device),
        "selection": "minimum grouped calibration pixel-vector RMSE; source val untouched",
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--disable-heading", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(train(
        args.dataset, args.out, disable_heading=args.disable_heading, epochs=args.epochs,
        patience=args.patience, batch=args.batch, learning_rate=args.learning_rate,
        seed=args.seed, device=args.device,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
