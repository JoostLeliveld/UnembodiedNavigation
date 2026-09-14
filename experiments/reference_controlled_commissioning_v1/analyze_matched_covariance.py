#!/usr/bin/env python3
"""Fit matched covariance candidates to gate-v2 whole-drive bias residuals.

Only the unsealed fit and development partitions are opened.  Covariance candidates
are fitted to leave-one-drive-out fit residuals and scored on complete development
drives whose mean predictions came from a model fitted on the fit partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln


MODELS = ("B0_raw", "B1_camera_constant", "B2_smooth_geometry", "B3_box_mlp", "B4_rgb_context")
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CHI2 = {0.50: 1.38629436112, 0.90: 4.60517018599, 0.95: 5.99146454711, 0.99: 9.21034037198}
FLOOR_M2 = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def camera_positions(campaign_root: Path) -> dict[str, np.ndarray]:
    execution = json.loads((campaign_root / "campaign_execution.json").read_text(encoding="utf-8"))
    protocol_path = Path(execution["protocol"])
    import sys
    import yaml

    repo = protocol_path.parents[2]
    sys.path[:0] = [str(repo / "experiments/reference_controlled_commissioning_v1"),
                    str(repo / "src/reliability"), str(repo / "src/unav_common")]
    from build_tables import _camera_positions

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=float) for key, value in _camera_positions(protocol).items()}


def enrich_yaw(bias_root: Path) -> dict[tuple[str, str, int], float]:
    manifest = json.loads((bias_root / "input_manifest.json").read_text(encoding="utf-8"))
    result: dict[tuple[str, str, int], float] = {}
    for source in manifest["input_drives"]:
        if source["partition"] not in {"fit", "development"}:
            raise RuntimeError("covariance analysis received an audit source")
        table = Path(source["camera_measurements"])
        if sha256(table) != source["camera_measurements_sha256"]:
            raise RuntimeError(f"measurement table changed: {table}")
        for row in read_csv(table):
            key = (row["drive_id"], row["camera_id"], int(row["capture_stamp_ns"]))
            result[key] = float(row["reference_yaw"])
    return result


def load_predictions(path: Path, yaw: dict[tuple[str, str, int], float], cameras: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = read_csv(path)
    count = len(rows)
    drive = np.asarray([row["drive_id"] for row in rows])
    camera = np.asarray([row["camera_id"] for row in rows])
    stamp = np.asarray([int(row["capture_stamp_ns"]) for row in rows], dtype=np.int64)
    raw = np.asarray([[float(row["raw_x"]), float(row["raw_y"])] for row in rows])
    reference = np.asarray([[float(row["reference_x"]), float(row["reference_y"])] for row in rows])
    heading = np.asarray([yaw[(row["drive_id"], row["camera_id"], int(row["capture_stamp_ns"]))] for row in rows])
    basis = np.zeros((count, 2, 2), dtype=float)
    features = np.zeros((count, 4), dtype=float)
    for index in range(count):
        ray = raw[index] - cameras[camera[index]]
        distance = np.linalg.norm(ray)
        along = ray / distance
        basis[index] = np.column_stack((along, (-along[1], along[0])))
        bearing = math.atan2(along[1], along[0])
        features[index] = (reference[index, 0], reference[index, 1], heading[index], distance)
    target = np.einsum("nij,nj->ni", np.transpose(basis, (0, 2, 1)), reference - raw)
    residuals = {}
    for model in MODELS:
        prediction = np.asarray([[float(row[f"{model}_correction_along_m"]),
                                  float(row[f"{model}_correction_across_m"])] for row in rows])
        residuals[model] = prediction - target
    return {"rows": rows, "drive": drive, "camera": camera, "stamp": stamp,
            "raw": raw, "reference": reference, "heading": heading, "basis": basis,
            "features": features, "residuals": residuals}


def balanced_weights(drives: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(drives), dtype=float)
    unique = sorted(set(drives))
    for drive in unique:
        selected = drives == drive
        weights[selected] = 1.0 / (len(unique) * selected.sum())
    return weights / weights.sum()


def empirical_covariance(residual: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight = weight / weight.sum()
    covariance = np.einsum("n,ni,nj->ij", weight, residual, residual)
    covariance = 0.5 * (covariance + covariance.T) + np.eye(2) * FLOOR_M2
    return covariance


class ConstantCovariance:
    def __init__(self, per_camera: bool) -> None:
        self.per_camera = per_camera
        self.values: dict[str, np.ndarray] = {}

    def fit(self, data: dict[str, Any], residual: np.ndarray) -> "ConstantCovariance":
        groups = CAMERAS if self.per_camera else ("all",)
        for group in groups:
            selected = np.ones(len(residual), dtype=bool) if group == "all" else data["camera"] == group
            self.values[group] = empirical_covariance(residual[selected], balanced_weights(data["drive"][selected]))
        return self

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        if not self.per_camera:
            return np.repeat(self.values["all"][None, :, :], len(data["drive"]), axis=0)
        return np.stack([self.values[value] for value in data["camera"]])


class StratifiedCovariance:
    def __init__(self, prior_strength: float) -> None:
        self.prior_strength = prior_strength

    def fit(self, data: dict[str, Any], residual: np.ndarray) -> "StratifiedCovariance":
        self.base = ConstantCovariance(per_camera=True).fit(data, residual)
        self.edges: dict[str, np.ndarray] = {}
        self.values: dict[tuple[str, int], np.ndarray] = {}
        for camera in CAMERAS:
            selected = data["camera"] == camera
            ranges = data["features"][selected, 3]
            edges = np.quantile(ranges, [1 / 3, 2 / 3])
            self.edges[camera] = edges
            bins = np.digitize(ranges, edges)
            local_residual = residual[selected]
            local_drive = data["drive"][selected]
            for bin_id in range(3):
                use = bins == bin_id
                local = empirical_covariance(local_residual[use], balanced_weights(local_drive[use]))
                count = float(use.sum())
                self.values[(camera, bin_id)] = (
                    count * local + self.prior_strength * self.base.values[camera]
                ) / (count + self.prior_strength)
        return self

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        result = []
        for camera, feature in zip(data["camera"], data["features"], strict=True):
            bin_id = int(np.digitize(feature[3], self.edges[camera]))
            result.append(self.values[(camera, bin_id)])
        return np.stack(result)


class SmoothCovariance:
    def __init__(self, position_scale: float, heading_scale: float, prior_strength: float) -> None:
        self.position_scale = position_scale
        self.heading_scale = heading_scale
        self.prior_strength = prior_strength

    def fit(self, data: dict[str, Any], residual: np.ndarray) -> "SmoothCovariance":
        self.data = data
        self.residual = residual
        self.base = ConstantCovariance(per_camera=True).fit(data, residual)
        self.sample_weight = balanced_weights(data["drive"]) * len(data["drive"])
        return self

    def predict(self, query: dict[str, Any]) -> np.ndarray:
        result = np.empty((len(query["drive"]), 2, 2), dtype=float)
        for camera in CAMERAS:
            train = self.data["camera"] == camera
            test_indices = np.flatnonzero(query["camera"] == camera)
            p_train = self.data["features"][train, :2]
            h_train = self.data["features"][train, 2]
            residual = self.residual[train]
            sample_weight = self.sample_weight[train]
            base = self.base.values[camera]
            for start in range(0, len(test_indices), 256):
                part = test_indices[start:start + 256]
                p = query["features"][part, :2]
                heading = query["features"][part, 2]
                distance2 = np.sum((p[:, None, :] - p_train[None, :, :]) ** 2, axis=2)
                heading_term = 1.0 - np.cos(heading[:, None] - h_train[None, :])
                weight = np.exp(-0.5 * distance2 / self.position_scale ** 2
                                - heading_term / self.heading_scale ** 2)
                weight *= sample_weight[None, :]
                numerator = np.einsum("qn,ni,nj->qij", weight, residual, residual)
                denominator = weight.sum(axis=1)
                result[part] = (
                    numerator + self.prior_strength * base[None, :, :]
                ) / (denominator[:, None, None] + self.prior_strength)
                result[part] += np.eye(2)[None, :, :] * FLOOR_M2
        return result


def gaussian_metrics(data: dict[str, Any], residual: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    inverse = np.linalg.inv(covariance)
    mahalanobis = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    logdet = np.linalg.slogdet(covariance)[1]
    nll = 0.5 * (2.0 * math.log(2.0 * math.pi) + logdet + mahalanobis)
    area95 = math.pi * CHI2[0.95] * np.sqrt(np.linalg.det(covariance)) * 1.0e4
    by_drive = {drive: float(nll[data["drive"] == drive].mean()) for drive in sorted(set(data["drive"]))}
    return {
        "n": len(residual),
        "pooled_nll": float(nll.mean()),
        "equal_drive_nll": float(np.mean(list(by_drive.values()))),
        "mean_95_ellipse_area_cm2": float(area95.mean()),
        "median_95_ellipse_area_cm2": float(np.median(area95)),
        "mean_mahalanobis_d2": float(mahalanobis.mean()),
        "coverage": {str(level): float(np.mean(mahalanobis <= threshold)) for level, threshold in CHI2.items()},
        "tail_over_99": float(np.mean(mahalanobis > CHI2[0.99])),
        "by_drive_nll": by_drive,
    }


def student_t_nll(residual: np.ndarray, covariance: np.ndarray, degrees: float) -> float:
    scale = covariance * (degrees - 2.0) / degrees
    inverse = np.linalg.inv(scale)
    mahalanobis = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    logdet = np.linalg.slogdet(scale)[1]
    dimension = 2.0
    nll = (
        -gammaln((degrees + dimension) / 2.0) + gammaln(degrees / 2.0)
        + 0.5 * (dimension * math.log(degrees * math.pi) + logdet)
        + 0.5 * (degrees + dimension) * np.log1p(mahalanobis / degrees)
    )
    return float(nll.mean())


def dependence_diagnostics(data: dict[str, Any], residual: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    standardized = np.empty_like(residual)
    for index, (value, cov) in enumerate(zip(residual, covariance, strict=True)):
        standardized[index] = np.linalg.solve(np.linalg.cholesky(cov), value)
    lag_values = []
    by_series = {}
    for drive in sorted(set(data["drive"])):
        for camera in CAMERAS:
            selected = np.flatnonzero((data["drive"] == drive) & (data["camera"] == camera))
            if len(selected) < 4:
                continue
            selected = selected[np.argsort(data["stamp"][selected])]
            values = []
            for axis in range(2):
                correlation = float(np.corrcoef(standardized[selected[:-1], axis], standardized[selected[1:], axis])[0, 1])
                if math.isfinite(correlation):
                    values.append(correlation)
                    lag_values.append(correlation)
            by_series[f"{drive}:{camera}"] = values
    simultaneous = []
    world = np.einsum("nij,nj->ni", data["basis"], residual)
    for drive in sorted(set(data["drive"])):
        use_drive = data["drive"] == drive
        for stamp in sorted(set(data["stamp"][use_drive])):
            selected = np.flatnonzero(use_drive & (data["stamp"] == stamp))
            if len(selected) >= 2:
                # Mean pairwise cosine records common signed error without treating
                # camera pairs from different times as simultaneous.
                for first in range(len(selected)):
                    for second in range(first + 1, len(selected)):
                        a, b = world[selected[first]], world[selected[second]]
                        norm = np.linalg.norm(a) * np.linalg.norm(b)
                        if norm > 1e-12:
                            simultaneous.append(float(np.dot(a, b) / norm))
    return {
        "lag1_median": float(np.median(lag_values)) if lag_values else math.nan,
        "lag1_median_absolute": float(np.median(np.abs(lag_values))) if lag_values else math.nan,
        "lag1_series": by_series,
        "simultaneous_pair_cosine_mean": float(np.mean(simultaneous)) if simultaneous else math.nan,
        "simultaneous_pair_count": len(simultaneous),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bias_root = args.bias_root.resolve()
    campaign_root = args.campaign_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    execution = json.loads((campaign_root / "campaign_execution.json").read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed" or execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("campaign must be complete with audit sealed")
    bias_manifest = json.loads((bias_root / "input_manifest.json").read_text(encoding="utf-8"))
    if bias_manifest.get("sensor_gate_id") != "commissioning_sensor_gate_v2":
        raise RuntimeError("matched covariance requires gate-v2 bias predictions")
    completion = json.loads((bias_root / "completion.json").read_text(encoding="utf-8"))
    for name in ("fit_leave_one_drive_out_predictions.csv", "development_predictions.csv"):
        if sha256(bias_root / name) != completion["artifacts"][name]:
            raise RuntimeError(f"bias prediction artifact changed: {name}")

    yaw = enrich_yaw(bias_root)
    cameras = camera_positions(campaign_root)
    fit = load_predictions(bias_root / "fit_leave_one_drive_out_predictions.csv", yaw, cameras)
    development = load_predictions(bias_root / "development_predictions.csv", yaw, cameras)

    report: dict[str, Any] = {
        "schema": "reference_controlled_matched_covariance.v1",
        "status": "complete_unsealed_inputs_only",
        "audit_opened": False,
        "campaign_root": str(campaign_root),
        "campaign_protocol_sha256": execution["protocol_sha256"],
        "bias_root": str(bias_root),
        "bias_completion_sha256": sha256(bias_root / "completion.json"),
        "sensor_gate_id": bias_manifest["sensor_gate_id"],
        "fit_rows": len(fit["drive"]),
        "development_rows": len(development["drive"]),
        "models": {},
    }

    for model in MODELS:
        fit_residual = fit["residuals"][model]
        dev_residual = development["residuals"][model]
        candidates: dict[str, tuple[Any, np.ndarray, dict[str, Any]]] = {}

        r0 = ConstantCovariance(False).fit(fit, fit_residual)
        cov = r0.predict(development)
        candidates["R0_global_full"] = (r0, cov, gaussian_metrics(development, dev_residual, cov))

        r1 = ConstantCovariance(True).fit(fit, fit_residual)
        cov = r1.predict(development)
        candidates["R1_per_camera_full"] = (r1, cov, gaussian_metrics(development, dev_residual, cov))

        r2_options = []
        for prior in (10.0, 30.0, 100.0):
            candidate = StratifiedCovariance(prior).fit(fit, fit_residual)
            prediction = candidate.predict(development)
            metrics = gaussian_metrics(development, dev_residual, prediction)
            r2_options.append((metrics["equal_drive_nll"], prior, candidate, prediction, metrics))
        _, prior, r2, cov, metrics = min(r2_options, key=lambda item: item[0])
        metrics["prior_strength"] = prior
        candidates["R2_range_strata"] = (r2, cov, metrics)

        r3_options = []
        for position_scale in (0.75, 1.5, 3.0):
            for heading_scale in (0.75, 1.5, 100.0):
                for prior in (10.0, 30.0, 100.0):
                    candidate = SmoothCovariance(position_scale, heading_scale, prior).fit(fit, fit_residual)
                    prediction = candidate.predict(development)
                    metrics = gaussian_metrics(development, dev_residual, prediction)
                    r3_options.append((metrics["equal_drive_nll"], position_scale, heading_scale,
                                       prior, candidate, prediction, metrics))
        _, pscale, hscale, prior, r3, cov, metrics = min(r3_options, key=lambda item: item[0])
        metrics.update({"position_scale_m": pscale, "heading_scale": hscale, "prior_strength": prior})
        candidates["R3_smooth_spatial_heading"] = (r3, cov, metrics)

        best_gaussian = min(candidates, key=lambda name: candidates[name][2]["equal_drive_nll"])
        best_covariance = candidates[best_gaussian][1]
        student = []
        for degrees in (3.0, 5.0, 8.0, 15.0, 30.0):
            student.append((student_t_nll(dev_residual, best_covariance, degrees), degrees))
        student_nll, degrees = min(student)
        candidates_report = {name: item[2] for name, item in candidates.items()}
        candidates_report["R4_student_t_diagnostic"] = {
            "base_covariance": best_gaussian,
            "degrees_of_freedom": degrees,
            "pooled_student_t_nll": student_nll,
            "eligible_for_gaussian_ekf": False,
        }
        dependence = dependence_diagnostics(development, dev_residual, best_covariance)
        r5_required = bool(dependence["lag1_median_absolute"] > 0.30
                           or abs(dependence["simultaneous_pair_cosine_mean"]) > 0.20)
        candidates_report["R5_persistent_bias_diagnostic"] = {
            "required_by_predeclared_threshold": r5_required,
            "thresholds": {"lag1_median_absolute": 0.30, "simultaneous_pair_cosine_absolute_mean": 0.20},
            "dependence": dependence,
        }
        report["models"][model] = {
            "gaussian_candidates": candidates_report,
            "best_gaussian_by_equal_drive_nll": best_gaussian,
            "R5_required": r5_required,
        }
        print(model, best_gaussian, candidates[best_gaussian][2]["equal_drive_nll"],
              "R5", r5_required, flush=True)

    # Pair selection is deliberately based on the held-out proper score, then the
    # simpler correction/covariance pair within one drive-level standard error.
    pair_rows = []
    for model in MODELS:
        candidate = report["models"][model]["best_gaussian_by_equal_drive_nll"]
        metrics = report["models"][model]["gaussian_candidates"][candidate]
        values = np.asarray(list(metrics["by_drive_nll"].values()))
        pair_rows.append((float(values.mean()), model, candidate, values))
    best_mean, best_model, best_r, best_values = min(pair_rows, key=lambda item: item[0])
    eligible = []
    comparisons = {}
    for mean, model, covariance, values in pair_rows:
        difference = values - best_values
        se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
        within = bool(difference.mean() <= se + 1e-15)
        comparisons[f"{model}+{covariance}"] = {
            "mean_difference_to_best": float(difference.mean()), "paired_se": se,
            "within_one_paired_se": within,
        }
        if within:
            eligible.append((MODELS.index(model), model, covariance))
    _, selected_model, selected_r = min(eligible)
    report["selection"] = {
        "primary": "equal-development-drive Gaussian negative log likelihood",
        "rule": "simplest mean model whose best matched Gaussian covariance is within one paired SE of the best pair",
        "best_pair": {"mean": best_model, "covariance": best_r},
        "selected_pair": {"mean": selected_model, "covariance": selected_r},
        "comparisons": comparisons,
        "audit_opened": False,
    }

    atomic_json(output / "matched_covariance_report.json", report)
    with (output / "pair_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ["mean_model", "covariance_model", "equal_drive_nll", "coverage_50",
                  "coverage_90", "coverage_95", "coverage_99", "mean_95_ellipse_area_cm2",
                  "mean_mahalanobis_d2", "tail_over_99"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            for covariance in ("R0_global_full", "R1_per_camera_full", "R2_range_strata", "R3_smooth_spatial_heading"):
                metrics = report["models"][model]["gaussian_candidates"][covariance]
                writer.writerow({
                    "mean_model": model, "covariance_model": covariance,
                    "equal_drive_nll": metrics["equal_drive_nll"],
                    "coverage_50": metrics["coverage"]["0.5"],
                    "coverage_90": metrics["coverage"]["0.9"],
                    "coverage_95": metrics["coverage"]["0.95"],
                    "coverage_99": metrics["coverage"]["0.99"],
                    "mean_95_ellipse_area_cm2": metrics["mean_95_ellipse_area_cm2"],
                    "mean_mahalanobis_d2": metrics["mean_mahalanobis_d2"],
                    "tail_over_99": metrics["tail_over_99"],
                })

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    labels = []
    for model in MODELS:
        candidate = report["models"][model]["best_gaussian_by_equal_drive_nll"]
        metrics = report["models"][model]["gaussian_candidates"][candidate]
        labels.append(model.split("_", 1)[0])
        axes[0].scatter(metrics["mean_95_ellipse_area_cm2"], metrics["equal_drive_nll"], s=52, label=labels[-1])
    axes[0].set(xlabel=r"mean 95% ellipse area [cm$^2$]", ylabel="equal-drive Gaussian NLL")
    axes[0].grid(alpha=0.2); axes[0].legend(title="mean")
    levels = list(CHI2)
    axes[1].plot(levels, levels, color="#222222", linestyle="--", label="ideal")
    for model in MODELS:
        candidate = report["models"][model]["best_gaussian_by_equal_drive_nll"]
        coverage = report["models"][model]["gaussian_candidates"][candidate]["coverage"]
        axes[1].plot(levels, [coverage[str(level)] for level in levels], marker="o", label=model.split("_", 1)[0])
    axes[1].set(xlabel="nominal ellipse probability", ylabel="observed containment", xlim=(0.47, 1.0), ylim=(0.47, 1.0))
    axes[1].grid(alpha=0.2); axes[1].legend(title="mean")
    fig.savefig(output / "matched_covariance_selection.png", dpi=220)
    plt.close(fig)

    atomic_json(output / "completion.json", {
        "schema": "reference_controlled_matched_covariance_completion.v1",
        "status": "complete_unsealed_inputs_only",
        "audit_opened": False,
        "selected_pair": report["selection"]["selected_pair"],
        "best_pair": report["selection"]["best_pair"],
        "artifacts": {name: sha256(output / name) for name in
                      ("matched_covariance_report.json", "pair_summary.csv", "matched_covariance_selection.png")},
        "source_sha256": sha256(Path(__file__)),
    })
    print(json.dumps(report["selection"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
