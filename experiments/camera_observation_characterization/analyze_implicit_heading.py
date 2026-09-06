#!/usr/bin/env python3
"""Test whether the heading-unaware box-feature MLP can infer heading implicitly.

The diagnostic never changes the frozen correction model. It asks three separate questions:

1. Can a probe decode the eight commanded headings from the model's runtime inputs on
   spatially held-out tiles?
2. Does the frozen correction itself vary with commanded heading in the direction required
   by the raw projection error?
3. Which box-shape features visibly change with heading for each camera?

Ground-truth heading is used only as an offline diagnostic label.
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
from matplotlib.lines import Line2D
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
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

DEFAULT_MODEL = REPO / "logs/perception_models/box_feature_bias_correction_20260831"
CAMERA_COLOURS = {
    "camera_A": "#2a78d6", "camera_B": "#eb6834", "camera_C": "#1baf7a",
    "camera_D": "#4a3aa7", "camera_E": "#e84c4c",
}
FEATURE_GROUPS = {
    "camera identity only": tuple(range(10, 15)),
    "box shape only": (2, 3, 4),
    "box shape + confidence + camera": (2, 3, 4, 9, 10, 11, 12, 13, 14),
    "projection/location + camera": (0, 1, 5, 6, 7, 8, 10, 11, 12, 13, 14),
    "all correction inputs": tuple(range(15)),
}


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


def input_vector(row: dict[str, str], geometry: dict) -> np.ndarray:
    onehot = np.zeros(len(F.CAMERAS))
    onehot[F.CAMERAS.index(row["camera_id"])] = 1.0
    return np.concatenate([F.features(row, geometry[row["camera_id"]]), onehot])


def heading_degrees(rows: list[dict[str, str]]) -> dict[int, float]:
    result = {}
    for row in rows:
        heading = int(row["heading_id"])
        result.setdefault(heading, math.degrees(float(row["robot_yaw"])) % 360.0)
    return result


def build_records(rows: list[dict[str, str]], geometry: dict, network) -> list[dict]:
    records = []
    for row in rows:
        if row["raw_valid"] != "1":
            continue
        inputs = input_vector(row, geometry)
        predicted = network.predict(inputs[None, :])[0]
        required = F.target(row, geometry[row["camera_id"]])
        records.append({
            "row": row,
            "inputs": inputs,
            "heading": int(row["heading_id"]),
            "camera": row["camera_id"],
            "predicted": np.asarray(predicted, dtype=float),
            "required": np.asarray(required, dtype=float),
        })
    return records


def fit_probes(train: list[dict], test: list[dict]) -> tuple[list[dict], np.ndarray]:
    x_train = np.stack([record["inputs"] for record in train])
    y_train = np.asarray([record["heading"] for record in train], dtype=int)
    x_test = np.stack([record["inputs"] for record in test])
    y_test = np.asarray([record["heading"] for record in test], dtype=int)
    results = []
    all_prediction = None
    for group, indices in FEATURE_GROUPS.items():
        index = np.asarray(indices, dtype=int)
        classifiers = {
            "linear probe": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=5000, random_state=0)
            ),
            "nonlinear probe": make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(32, 32), activation="relu", solver="adam",
                              alpha=1e-4, max_iter=1500, early_stopping=True,
                              n_iter_no_change=30, validation_fraction=0.15, random_state=0),
            ),
        }
        for classifier_name, classifier in classifiers.items():
            classifier.fit(x_train[:, index], y_train)
            prediction = classifier.predict(x_test[:, index])
            accuracy = float(np.mean(prediction == y_test))
            axial_accuracy = float(np.mean((prediction % 4) == (y_test % 4)))
            results.append({
                "feature_group": group,
                "classifier": classifier_name,
                "heldout_accuracy": accuracy,
                "heldout_axial_accuracy_modulo_180": axial_accuracy,
                "correct": int(np.sum(prediction == y_test)),
                "n": int(len(y_test)),
            })
            if group == "all correction inputs" and classifier_name == "nonlinear probe":
                all_prediction = prediction
    if all_prediction is None:
        raise RuntimeError("All-input nonlinear heading probe was not fitted")
    return results, confusion_matrix(y_test, all_prediction, labels=range(8))


def draw_probe(results: list[dict], confusion: np.ndarray, degrees: dict[int, float],
               output: Path) -> None:
    groups = list(FEATURE_GROUPS)
    classifiers = ("linear probe", "nonlinear probe")
    values = {(row["feature_group"], row["classifier"]): row["heldout_accuracy"]
              for row in results}
    all_nonlinear = next(
        row for row in results
        if row["feature_group"] == "all correction inputs" and row["classifier"] == "nonlinear probe"
    )
    x = np.arange(len(groups)); width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.25, 1.0)})
    for offset, classifier, colour in ((-width / 2, classifiers[0], "#2a78d6"),
                                        (width / 2, classifiers[1], "#d4267b")):
        bars = axes[0].bar(x + offset, [100 * values[(group, classifier)] for group in groups],
                           width, color=colour, label=classifier)
        axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].axhline(12.5, color=D.INK, ls="--", lw=1.4, label="8-way chance = 12.5%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([group.replace(" + ", "\n+ ") for group in groups], rotation=10,
                            ha="right")
    axes[0].set(ylabel="Heading classification accuracy on held-out tiles (%)",
                ylim=(0, max(30, 1.18 * 100 * max(values.values()))))
    axes[0].grid(axis="y", color="#e4e2dc", lw=0.8)
    axes[0].legend(frameon=True)

    row_total = confusion.sum(axis=1, keepdims=True)
    normalized = confusion / np.maximum(row_total, 1)
    image = axes[1].imshow(normalized, cmap="magma", vmin=0, vmax=1, aspect="equal")
    for row in range(8):
        for column in range(8):
            value = normalized[row, column]
            axes[1].text(column, row, f"{100 * value:.0f}", ha="center", va="center",
                         color="white" if value > 0.45 else D.INK, fontsize=8.5)
    labels = [f"h{heading}\n{degrees[heading]:.0f}°" for heading in range(8)]
    axes[1].set_xticks(range(8)); axes[1].set_xticklabels(labels)
    axes[1].set_yticks(range(8)); axes[1].set_yticklabels(labels)
    axes[1].set(
        xlabel="Decoded heading", ylabel="True commanded heading",
        title=("All-input nonlinear probe confusion (%)\n"
               f"exact: {100 * all_nonlinear['heldout_accuracy']:.1f}% · "
               f"modulo 180°: {100 * all_nonlinear['heldout_axial_accuracy_modulo_180']:.1f}%"),
    )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    fig.suptitle(
        "Can robot heading be decoded from the heading-unaware correction inputs?\n"
        "Probes train on frozen TRAIN tiles and score only spatially held-out returned boxes",
        fontsize=18, fontweight="bold",
    )
    save(fig, output / "05_heading_signal_probe.png")


def cell_quantiles(records: list[dict], camera: str, heading: int, component: int,
                   field: str) -> tuple[float, float, float]:
    values = np.asarray([record[field][component] for record in records
                         if record["camera"] == camera and record["heading"] == heading])
    return tuple(float(value) for value in np.quantile(values, (0.25, 0.50, 0.75)))


def draw_correction_by_heading(records: list[dict], degrees: dict[int, float],
                               output: Path) -> list[dict]:
    headings = sorted(degrees)
    x = np.asarray([degrees[heading] for heading in headings])
    fig, axes = plt.subplots(2, 5, figsize=(22.0, 9.2), sharex=True, sharey="row",
                             constrained_layout=True)
    cells = []
    for column, camera in enumerate(F.CAMERAS):
        for component, label in enumerate(("along camera ray", "across camera ray")):
            ax = axes[component, column]
            required = np.asarray([cell_quantiles(records, camera, heading, component, "required")
                                   for heading in headings])
            predicted = np.asarray([cell_quantiles(records, camera, heading, component, "predicted")
                                    for heading in headings])
            ax.fill_between(x, 100 * required[:, 0], 100 * required[:, 2], color="#77736d",
                            alpha=0.16)
            ax.plot(x, 100 * required[:, 1], color=D.INK, marker="o", lw=2.1,
                    label="required correction")
            ax.fill_between(x, 100 * predicted[:, 0], 100 * predicted[:, 2], color="#d4267b",
                            alpha=0.18)
            ax.plot(x, 100 * predicted[:, 1], color="#d4267b", marker="o", lw=2.1,
                    label="NN prediction")
            ax.axhline(0, color="#aaa69f", lw=1.0)
            ax.grid(color="#e4e2dc", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels([f"{value:.0f}°" for value in x])
            if component == 0:
                ax.set_title(f"Camera {camera[-1]}", fontsize=14, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"Correction {label} (cm)")
            if component == 1:
                ax.set_xlabel("True heading (offline label)")
        for heading in headings:
            subset = [record for record in records
                      if record["camera"] == camera and record["heading"] == heading]
            for component, component_name in enumerate(("along", "across")):
                target = np.asarray([record["required"][component] for record in subset])
                prediction = np.asarray([record["predicted"][component] for record in subset])
                cells.append({
                    "camera_id": camera, "heading_id": heading,
                    "heading_deg": degrees[heading], "component": component_name,
                    "n": len(subset), "required_median_cm": float(100 * np.median(target)),
                    "predicted_median_cm": float(100 * np.median(prediction)),
                    "median_absolute_prediction_miss_cm": float(100 * np.median(np.abs(prediction - target))),
                })
    handles = [
        Line2D([0], [0], color=D.INK, marker="o", lw=2.1, label="required correction median"),
        Line2D([0], [0], color="#d4267b", marker="o", lw=2.1, label="NN prediction median"),
        Line2D([0], [0], color="#77736d", lw=8, alpha=0.16, label="middle 50%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle(
        "The frozen NN changes its correction with heading despite receiving no heading input\n"
        "Held-out tiles only; black is truth-derived required correction, magenta is box-feature prediction",
        fontsize=18.5, fontweight="bold",
    )
    save(fig, output / "06_correction_by_heading.png")
    return cells


def draw_shape_by_heading(records: list[dict], degrees: dict[int, float], output: Path) -> None:
    headings = sorted(degrees); x = np.asarray([degrees[heading] for heading in headings])
    features = ((2, "Box width / image width"), (3, "Box height / image height"),
                (4, "Box aspect ratio"))
    fig, axes = plt.subplots(3, 5, figsize=(21.0, 12.0), sharex=True,
                             constrained_layout=True)
    for column, camera in enumerate(F.CAMERAS):
        for row_index, (feature_index, label) in enumerate(features):
            ax = axes[row_index, column]
            quantiles = []
            for heading in headings:
                values = np.asarray([record["inputs"][feature_index] for record in records
                                     if record["camera"] == camera and record["heading"] == heading])
                quantiles.append(np.quantile(values, (0.25, 0.50, 0.75)))
            quantiles = np.asarray(quantiles)
            colour = CAMERA_COLOURS[camera]
            ax.fill_between(x, quantiles[:, 0], quantiles[:, 2], color=colour, alpha=0.20)
            ax.plot(x, quantiles[:, 1], color=colour, marker="o", lw=2.1)
            ax.grid(color="#e4e2dc", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels([f"{value:.0f}°" for value in x])
            if row_index == 0:
                ax.set_title(f"Camera {camera[-1]}", fontsize=14, fontweight="bold")
            if column == 0:
                ax.set_ylabel(label)
            if row_index == len(features) - 1:
                ax.set_xlabel("True heading (offline label)")
    fig.suptitle(
        "Box shape changes with heading, but also with which locations remain visible\n"
        "Held-out returned boxes; line = median, band = middle 50%; no causal heading claim from this plot alone",
        fontsize=18.5, fontweight="bold",
    )
    save(fig, output / "07_box_shape_by_heading.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    capture = args.capture.expanduser().resolve()
    model_dir = args.model.expanduser().resolve()
    table = capture / "bias_update_interpretations.csv"
    artifact_path = model_dir / "models.joblib"
    summary_path = model_dir / "summary.json"
    for path in (table, artifact_path, summary_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required input: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256(artifact_path) != summary["artifact_sha256"]:
        raise RuntimeError("Current model artifact no longer matches its summary")
    targets = [model_dir / name for name in (
        "05_heading_signal_probe.png", "06_correction_by_heading.png",
        "07_box_shape_by_heading.png", "heading_signal_summary.json",
    )]
    if any(path.exists() for path in targets) and not args.overwrite:
        raise RuntimeError("Heading diagnostics already exist; pass --overwrite to refresh")

    artifact = joblib.load(artifact_path)
    geometry = artifact["camera_geometry"]
    network = artifact["neural_model"]
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    train = build_records([row for row in rows if row["split"] == "train"], geometry, network)
    test = build_records([row for row in rows if row["split"] == "test"], geometry, network)
    degrees = heading_degrees(rows)
    probe_results, confusion = fit_probes(train, test)
    draw_probe(probe_results, confusion, degrees, model_dir)
    correction_cells = draw_correction_by_heading(test, degrees, model_dir)
    draw_shape_by_heading(test, degrees, model_dir)
    payload = {
        "status": "complete_offline_heading_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Can the heading-unaware correction infer heading from box/projection features?",
        "train_readings": len(train), "heldout_readings": len(test),
        "chance_accuracy": 0.125,
        "chance_axial_accuracy_modulo_180": 0.25,
        "probe_results": probe_results,
        "all_input_nonlinear_confusion_counts": confusion.tolist(),
        "correction_by_camera_heading_component": correction_cells,
        "interpretation": (
            "Above-chance held-out decoding establishes that inputs carry heading information; "
            "variation of frozen NN corrections with heading establishes association, not that "
            "the NN internally reconstructs a unique physical heading variable."
        ),
        "confounds": [
            "accuracy is conditional on a returned box",
            "visible locations and ranges differ across camera-heading subsets",
            "one image per state cannot separate shape variation from sampling noise",
            "ground-truth heading is an offline diagnostic label and never a correction input",
        ],
        "source_hashes": {"table": sha256(table), "model_artifact": sha256(artifact_path)},
        "figures": [path.name for path in targets[:3]],
    }
    targets[3].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(model_dir), "probe_results": probe_results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
