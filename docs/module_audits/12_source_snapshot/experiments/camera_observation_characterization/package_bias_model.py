#!/usr/bin/env python3
"""Reproduce, verify, and package the current box-feature bias correction models.

This is deliberately tied to ``fit_bias_updates.py``: it rebuilds the per-camera ridge
models and pooled MLP with the exact frozen split/configuration, verifies every reproduced
prediction against ``bias_update_interpretations.csv``, and only then writes a model artifact,
model card, machine-readable summaries, and training/evaluation figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO / "experiments/deck_figures") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments/deck_figures"))

import fit_bias_updates as F  # noqa: E402
import style as D  # noqa: E402

DEFAULT_OUTPUT = REPO / "logs/perception_models/box_feature_bias_correction_20260831"
METHODS = (
    ("raw", "Raw box → floor", "#eb6834"),
    ("fixed", "Fixed 30.9 cm", "#2a78d6"),
    ("learned", "Linear correction", "#4a3aa7"),
    ("nn", "Neural correction", "#d4267b"),
    ("hull", "Offline hull reference", "#1baf7a"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def scores(rows: list[dict[str, str]], method: str) -> dict[str, float | int]:
    values = np.asarray([float(row[f"{method}_error_m"]) for row in rows
                         if row[f"{method}_valid"] == "1"], dtype=float)
    return {
        "n": int(values.size),
        "median_cm": float(100 * np.median(values)),
        "mean_cm": float(100 * np.mean(values)),
        "p90_cm": float(100 * np.quantile(values, 0.90)),
        "p95_cm": float(100 * np.quantile(values, 0.95)),
        "rmse_cm": float(100 * np.sqrt(np.mean(values ** 2))),
    }


def draw_learning_curve(network, output: Path) -> None:
    model = network.named_steps["mlpregressor"]
    loss = np.asarray(model.loss_curve_, dtype=float)
    validation = np.asarray(getattr(model, "validation_scores_", []), dtype=float)
    epochs = np.arange(1, len(loss) + 1)
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    ax.plot(epochs, loss, color="#d4267b", lw=2.4, label="training squared-error loss")
    ax.set(xlabel="MLP iteration", ylabel="Training loss")
    ax.grid(color="#e4e2dc", lw=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if validation.size:
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(validation) + 1), validation, color="#2a78d6", lw=2.0,
                 label="internal early-stopping validation score")
        ax2.set_ylabel("Internal validation R²")
        extra_h, extra_l = ax2.get_legend_handles_labels()
        handles += extra_h; labels += extra_l
    ax.legend(handles, labels, loc="best", frameon=True)
    ax.set_title(
        f"Current box-feature MLP training history — stopped after {model.n_iter_} iterations\n"
        "Internal validation is sampled only within TRAIN tiles; held-out tiles select nothing",
        fontsize=15.5, fontweight="bold",
    )
    save(fig, output / "01_training_curve.png")


def draw_generalisation(rows: list[dict[str, str]], output: Path) -> None:
    metrics = ("median_cm", "p90_cm", "rmse_cm")
    titles = ("Median error", "90th-percentile error", "RMS error")
    table = {(split, method): scores([row for row in rows if row["split"] == split], method)
             for split in ("train", "test") for method, _label, _colour in METHODS}
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.4), constrained_layout=True)
    for ax, metric, title in zip(axes, metrics, titles):
        train = [table[("train", method)][metric] for method, _label, _colour in METHODS]
        test = [table[("test", method)][metric] for method, _label, _colour in METHODS]
        for index in x:
            ax.plot([index, index], [train[index], test[index]], color="#cac8c1", lw=2)
        ax.scatter(x, train, s=100, facecolor="white", edgecolor=D.MUTED, linewidth=2,
                   label="TRAIN tiles")
        ax.scatter(x, test, s=100, color=D.INK, label="held-out tiles")
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace(" ", "\n", 1) for _m, label, _c in METHODS])
        ax.set(ylabel="Position error (cm)", title=title)
        ax.grid(axis="y", color="#e4e2dc", lw=0.8)
    axes[0].legend(frameon=True)
    fig.suptitle(
        "Current correction model: fitted tiles versus spatially held-out tiles\n"
        "All methods use the same returned YOLO boxes; no post-detection gate",
        fontsize=17.5, fontweight="bold",
    )
    save(fig, output / "02_train_vs_heldout.png")


def draw_cdf(rows: list[dict[str, str]], output: Path) -> None:
    test = [row for row in rows if row["split"] == "test"]
    fig, ax = plt.subplots(figsize=(11.0, 6.5), constrained_layout=True)
    for method, label, colour in METHODS:
        values = np.sort(np.asarray([float(row[f"{method}_error_m"]) for row in test
                                     if row[f"{method}_valid"] == "1"], dtype=float))
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(100 * values, y, color=colour, lw=2.3, label=label)
    ax.set(xlabel="Camera-reading position error (cm)", ylabel="Fraction of held-out readings",
           xlim=(0, 60), ylim=(0, 1.0))
    ax.grid(color="#e4e2dc", lw=0.8)
    ax.legend(loc="lower right", frameon=True)
    ax.set_title(
        "Held-out error distributions for every interpretation of the same YOLO box\n"
        "The axis stops at 60 cm; larger errors remain included in the fractions",
        fontsize=15.5, fontweight="bold",
    )
    save(fig, output / "03_heldout_error_cdf.png")


def draw_by_camera(rows: list[dict[str, str]], output: Path) -> None:
    test = [row for row in rows if row["split"] == "test" and row["nn_valid"] == "1"]
    cameras = list(F.CAMERAS)
    medians, p90s, means_along, means_across = [], [], [], []
    for camera in cameras:
        subset = [row for row in test if row["camera_id"] == camera]
        error = np.asarray([float(row["nn_error_m"]) for row in subset])
        medians.append(100 * float(np.median(error)))
        p90s.append(100 * float(np.quantile(error, 0.90)))
        means_along.append(100 * float(np.mean([float(row["nn_along_m"]) for row in subset])))
        means_across.append(100 * float(np.mean([float(row["nn_across_m"]) for row in subset])))
    x = np.arange(len(cameras)); width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), constrained_layout=True)
    axes[0].bar(x - width / 2, medians, width, color="#d4267b", label="median")
    axes[0].bar(x + width / 2, p90s, width, color="#6e205f", label="p90")
    axes[0].set(ylabel="Position error (cm)", title="Distance error")
    axes[0].legend(frameon=True)
    axes[1].bar(x - width / 2, means_along, width, color="#2a78d6", label="along ray")
    axes[1].bar(x + width / 2, means_across, width, color="#eb6834", label="across ray")
    axes[1].axhline(0, color=D.INK, lw=1.0)
    axes[1].set(ylabel="Signed mean error (cm)", title="Residual centring")
    axes[1].legend(frameon=True)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([f"Camera {camera[-1]}" for camera in cameras])
        ax.grid(axis="y", color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)
    fig.suptitle(
        "Current neural correction on spatially held-out tiles, separated by camera",
        fontsize=17.0, fontweight="bold",
    )
    save(fig, output / "04_heldout_by_camera.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    output = args.output.expanduser().resolve()
    table = capture / "bias_update_interpretations.csv"
    bias_manifest_path = capture / "bias_update_interpretations_manifest.json"
    capture_manifest_path = capture / "capture_manifest.json"
    for path in (table, bias_manifest_path, capture_manifest_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required source: {path}")
    bias_manifest = json.loads(bias_manifest_path.read_text(encoding="utf-8"))
    if sha256(table) != bias_manifest["bias_update_interpretations_sha256"]:
        raise RuntimeError("bias-update table no longer matches its manifest")
    if output.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {output}; pass --overwrite to refresh")
    output.mkdir(parents=True, exist_ok=True)

    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    geometry = F.camera_geometry(capture_manifest)
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    train = [row for row in rows if row["split"] == "train" and row["raw_valid"] == "1"]
    test = [row for row in rows if row["split"] == "test" and row["raw_valid"] == "1"]

    linear = {}
    for camera in F.CAMERAS:
        subset = [row for row in train if row["camera_id"] == camera]
        x = np.stack([F.features(row, geometry[camera]) for row in subset])
        y = np.stack([F.target(row, geometry[camera]) for row in subset])
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x, y)
        linear[camera] = model

    def with_camera(row: dict[str, str]) -> np.ndarray:
        onehot = np.zeros(len(F.CAMERAS))
        onehot[F.CAMERAS.index(row["camera_id"])] = 1.0
        return np.concatenate([F.features(row, geometry[row["camera_id"]]), onehot])

    x_nn = np.stack([with_camera(row) for row in train])
    y_nn = np.stack([F.target(row, geometry[row["camera_id"]]) for row in train])
    network = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
                     alpha=1e-4, learning_rate_init=1e-3, max_iter=4000,
                     early_stopping=True, n_iter_no_change=40, validation_fraction=0.15,
                     random_state=args.seed),
    )
    network.fit(x_nn, y_nn)

    max_abs = {"linear": 0.0, "neural": 0.0}
    for row in train + test:
        camera = row["camera_id"]
        predicted_linear = F.apply_correction(
            row, geometry[camera],
            linear[camera].predict(F.features(row, geometry[camera])[None, :])[0],
        )
        predicted_nn = F.apply_correction(
            row, geometry[camera], network.predict(with_camera(row)[None, :])[0],
        )
        stored_linear = np.asarray([float(row["learned_x"]), float(row["learned_y"])])
        stored_nn = np.asarray([float(row["nn_x"]), float(row["nn_y"])])
        max_abs["linear"] = max(max_abs["linear"], float(np.max(np.abs(predicted_linear - stored_linear))))
        max_abs["neural"] = max(max_abs["neural"], float(np.max(np.abs(predicted_nn - stored_nn))))
    if max(max_abs.values()) > 1e-10:
        raise RuntimeError(f"Rebuilt models do not reproduce frozen predictions: {max_abs}")

    artifact = {
        "schema": "box_feature_bias_correction.joblib.v1",
        "feature_names": list(F.FEATURE_NAMES) + [f"is_{camera}" for camera in F.CAMERAS],
        "linear_feature_names": list(F.FEATURE_NAMES),
        "camera_ids": list(F.CAMERAS),
        "linear_models": linear,
        "neural_model": network,
        "camera_geometry": geometry,
        "target": "truth minus raw projection in along/across camera-ray coordinates",
    }
    artifact_path = output / "models.joblib"
    joblib.dump(artifact, artifact_path)

    split_scores = {
        split: {method: scores([row for row in rows if row["split"] == split], method)
                for method, _label, _colour in METHODS}
        for split in ("train", "test")
    }
    summary = {
        "status": "complete_current_box_feature_model",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "pooled StandardScaler + sklearn MLPRegressor(64,64), ReLU, Adam",
        "online_inputs": list(F.FEATURE_NAMES) + ["five-way camera identity"],
        "excluded_inputs": ["RGB image", "robot position", "robot heading", "true range",
                            "ground truth at inference", "semantic segmentation"],
        "training": {
            "samples": len(train), "heldout_samples": len(test), "seed": args.seed,
            "spatial_split": bias_manifest["holdout"],
            "iterations": int(network.named_steps["mlpregressor"].n_iter_),
            "reproduction_max_abs_m": max_abs,
        },
        "scores": split_scores,
        "limitations": [
            "accuracy is conditional on a YOLO box being returned",
            "one image per camera-position-heading cell cannot separate bias from repeat noise",
            "held-out tiles have now informed model assessment and are development validation",
            "the model has not been deployed in a frozen replicated closed-loop campaign",
        ],
        "source_hashes": {
            "bias_update_interpretations": sha256(table),
            "bias_update_manifest": sha256(bias_manifest_path),
            "capture_manifest": sha256(capture_manifest_path),
        },
        "artifact_sha256": sha256(artifact_path),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    draw_learning_curve(network, output)
    draw_generalisation(rows, output)
    draw_cdf(rows, output)
    draw_by_camera(rows, output)
    model_card = [
        "# Current box-feature bias correction", "",
        "This is the only learned bias-correction model retained in the workspace.", "",
        "It was trained on **3,249 returned YOLO boxes** from frozen spatial TRAIN tiles and "
        "evaluated on **3,163 returned boxes** from held-out tiles. It receives 10 box/projection "
        "features plus a five-way camera identity; it receives no RGB image or robot heading.", "",
        "| Held-out camera-reading method | Median | Mean | p90 | p95 | RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label, _colour in METHODS:
        score = split_scores["test"][method]
        model_card.append(
            f"| {label} | {score['median_cm']:.2f} cm | {score['mean_cm']:.2f} cm | "
            f"{score['p90_cm']:.2f} cm | {score['p95_cm']:.2f} cm | {score['rmse_cm']:.2f} cm |"
        )
    model_card += ["", "The hull row is an offline reference-pose interpretation, not a learned model.", "",
                   "See `summary.json` for hashes, exclusions, limitations, and exact scores.", ""]
    (output / "README.md").write_text("\n".join(model_card), encoding="utf-8")
    print(json.dumps({"output": str(output), "artifact": str(artifact_path),
                      "reproduction_max_abs_m": max_abs, "figures": 4}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
