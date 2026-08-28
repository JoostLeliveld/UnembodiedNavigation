#!/usr/bin/env python3
"""Paired held-out comparison of analytic, shape-blind, and shape-aware updates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from shape_conditioned_update_model import feature_vector, load_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(rows, xkey, ykey):
    error = np.asarray([[float(row[xkey]) - float(row["robot_x"]),
                         float(row[ykey]) - float(row["robot_y"])] for row in rows])
    mag = np.linalg.norm(error, axis=1); mean = error.mean(0)
    return {
        "n_pairs": len(rows), "n_images": len({(r["camera"], r["sample_index"]) for r in rows}),
        "mean_euclidean_error_m": float(mag.mean()), "median_euclidean_error_m": float(np.median(mag)),
        "p95_euclidean_error_m": float(np.percentile(mag, 95)), "rmse_m": float(np.sqrt(np.mean(mag ** 2))),
        "bias_x_m": float(mean[0]), "bias_y_m": float(mean[1]), "bias_m": float(np.linalg.norm(mean)),
    }


def _paired_comparison(rows, candidate_keys, reference_keys, *, seed, bootstrap_samples=5000):
    candidate_error = np.linalg.norm(np.asarray([
        [float(row[candidate_keys[0]]) - float(row["robot_x"]),
         float(row[candidate_keys[1]]) - float(row["robot_y"])] for row in rows
    ]), axis=1)
    reference_error = np.linalg.norm(np.asarray([
        [float(row[reference_keys[0]]) - float(row["robot_x"]),
         float(row[reference_keys[1]]) - float(row["robot_y"])] for row in rows
    ]), axis=1)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((row["camera"], row["sample_index"]), []).append(index)
    grouped_indices = [np.asarray(indices) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    mean_deltas, rmse_deltas = [], []
    for _ in range(bootstrap_samples):
        sampled_groups = rng.integers(0, len(grouped_indices), size=len(grouped_indices))
        indices = np.concatenate([grouped_indices[index] for index in sampled_groups])
        candidate, reference = candidate_error[indices], reference_error[indices]
        mean_deltas.append(float(np.mean(candidate - reference)))
        rmse_deltas.append(float(np.sqrt(np.mean(candidate ** 2)) - np.sqrt(np.mean(reference ** 2))))
    mean_delta = candidate_error - reference_error
    return {
        "candidate_minus_reference_mean_error_m": float(mean_delta.mean()),
        "candidate_minus_reference_mean_error_clustered_bootstrap_95ci_m": [
            float(np.percentile(mean_deltas, 2.5)), float(np.percentile(mean_deltas, 97.5))
        ],
        "candidate_minus_reference_rmse_m": float(
            np.sqrt(np.mean(candidate_error ** 2)) - np.sqrt(np.mean(reference_error ** 2))
        ),
        "candidate_minus_reference_rmse_clustered_bootstrap_95ci_m": [
            float(np.percentile(rmse_deltas, 2.5)), float(np.percentile(rmse_deltas, 97.5))
        ],
        "candidate_pair_win_fraction": float(np.mean(candidate_error < reference_error)),
        "n_pairs": len(rows), "n_image_clusters": len(groups),
        "bootstrap_samples": bootstrap_samples,
    }


def _predict(rows, path, *, device):
    import torch
    model, artifact = load_artifact(path, device=device)
    x = np.stack([feature_vector(row, use_shape=artifact["use_shape"]) for row in rows])
    x = (x - np.asarray(artifact["x_mean"], np.float32)) / np.asarray(artifact["x_std"], np.float32)
    with torch.no_grad(): pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    pred = pred * np.asarray(artifact["y_std"]) + np.asarray(artifact["y_mean"])
    return pred


def evaluate(dataset: Path, blind_model: Path, shape_model: Path, output: Path, *, device: str):
    dataset, blind_model = dataset.resolve(), blind_model.resolve()
    shape_model, output = shape_model.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"non-empty output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (dataset / "records.csv").open(newline="", encoding="utf-8") as handle:
        test = [row for row in csv.DictReader(handle) if row["residual_split"] == "test"]
    kept = [row for row in test if row["gate_pass"] == "1"]
    blind, shape = _predict(kept, blind_model, device=device), _predict(kept, shape_model, device=device)
    for row, blind_delta, shape_delta in zip(kept, blind, shape):
        row["blind_x"] = float(row["prior_x"]) + float(blind_delta[0]); row["blind_y"] = float(row["prior_y"]) + float(blind_delta[1])
        row["shape_x"] = float(row["prior_x"]) + float(shape_delta[0]); row["shape_y"] = float(row["prior_y"]) + float(shape_delta[1])
    arms = {
        "prior": ("prior_x", "prior_y"), "raw_floor_backprojection": ("raw_ground_x", "raw_ground_y"),
        "analytic_hull": ("analytic_x", "analytic_y"), "mlp_without_shape": ("blind_x", "blind_y"),
        "mlp_with_shape": ("shape_x", "shape_y"), "detection_only_floor": ("detection_only_x", "detection_only_y"),
    }
    exact = [r for r in kept if r["exact_prior"] == "1"]
    perturbed = [r for r in kept if r["exact_prior"] != "1"]
    payload = {
        "status": "provisional_paired_shape_conditioning_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric_object": "camera_measurement_equivalent_position_update",
        "reference": "commanded_ground_truth_xy", "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
        "models": {
            "mlp_without_shape": {"path": str(blind_model), "sha256": _sha256(blind_model)},
            "mlp_with_shape": {"path": str(shape_model), "sha256": _sha256(shape_model)},
        },
        "projection_runtime": "bbox_bottom_floor_ipm; analytic visual-hull general-H equivalent update; learned candidate-state updates",
        "experimental_unit": "frozen detection/candidate-prior pair; image counts also reported",
        "online_inputs": ["detector box/confidence", "fixed camera", "candidate prior xy", "exact commanded heading"],
        "evaluation_only_inputs": ["commanded GT xy", "semantic mask for detection-only floor"],
        "admission": "shared confidence>=0.25 plus runtime hull plausibility gate",
        "availability": {"test_pairs": len(test), "kept_pairs": len(kept), "rejected_pairs": len(test)-len(kept)},
        "all_kept": {name: _stats(kept, *keys) for name, keys in arms.items()},
        "exact_prior": {name: _stats(exact, *keys) for name, keys in arms.items()},
        "perturbed_prior": {name: _stats(perturbed, *keys) for name, keys in arms.items()},
        "paired_comparisons": {
            "exact_prior": {
                "shape_vs_blind": _paired_comparison(
                    exact, arms["mlp_with_shape"], arms["mlp_without_shape"], seed=101),
                "shape_vs_analytic": _paired_comparison(
                    exact, arms["mlp_with_shape"], arms["analytic_hull"], seed=102),
            },
            "perturbed_prior": {
                "shape_vs_blind": _paired_comparison(
                    perturbed, arms["mlp_with_shape"], arms["mlp_without_shape"], seed=201),
                "shape_vs_analytic": _paired_comparison(
                    perturbed, arms["mlp_with_shape"], arms["analytic_hull"], seed=202),
            },
        },
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({k for r in kept for k in r}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(kept)
    lines = ["# Robot-shape conditioning — paired held-out comparison", "",
             "Status: provisional, spatially held-out set-pose study with oracle commanded heading.", "",
             f"Admission: {len(kept)}/{len(test)} candidate-prior pairs passed the shared detector/hull gate."]
    for cohort_name, title in (("perturbed_prior", "Perturbed candidate priors"),
                               ("exact_prior", "Exact candidate priors")):
        lines.extend(["", f"## {title}", "", "| Arm | Mean | Median | p95 | RMSE | Bias |",
                      "|---|---:|---:|---:|---:|---:|"])
        for name, s in payload[cohort_name].items():
            lines.append(f"| {name} | {100*s['mean_euclidean_error_m']:.2f} cm | {100*s['median_euclidean_error_m']:.2f} cm | {100*s['p95_euclidean_error_m']:.2f} cm | {100*s['rmse_m']:.2f} cm | {100*s['bias_m']:.2f} cm |")
    comparison = payload["paired_comparisons"]["perturbed_prior"]["shape_vs_blind"]
    mean_ci = comparison["candidate_minus_reference_mean_error_clustered_bootstrap_95ci_m"]
    rmse_ci = comparison["candidate_minus_reference_rmse_clustered_bootstrap_95ci_m"]
    lines.extend(["", "## Shape contribution on perturbed priors", "",
                  f"Shape minus blind mean-error delta: {100*comparison['candidate_minus_reference_mean_error_m']:.2f} cm "
                  f"(95% image-clustered bootstrap CI {100*mean_ci[0]:.2f} to {100*mean_ci[1]:.2f} cm).",
                  f"Shape minus blind RMSE delta: {100*comparison['candidate_minus_reference_rmse_m']:.2f} cm "
                  f"(95% image-clustered bootstrap CI {100*rmse_ci[0]:.2f} to {100*rmse_ci[1]:.2f} cm)."])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--blind-model", required=True, type=Path); p.add_argument("--shape-model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path); p.add_argument("--device", default="cpu"); a = p.parse_args()
    result = evaluate(a.dataset, a.blind_model, a.shape_model, a.out, device=a.device)
    print(json.dumps({"availability": result["availability"], "exact_prior": result["exact_prior"],
                      "perturbed_prior": result["perturbed_prior"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
