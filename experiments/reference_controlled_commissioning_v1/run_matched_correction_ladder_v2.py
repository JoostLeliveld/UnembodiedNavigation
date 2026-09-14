#!/usr/bin/env python3
"""Run the matched raw/ridge/MLP/visibility correction comparison with gate v2.

The existing visibility-patch implementation is reused without changing its frozen
source.  This entry point restricts the recorded v1 measurement table to the stricter
v2 full-box gate, adds raw and ridge predictions on the identical rows, and selects
one correction by equal-development-drive RMSE.  Audit tables are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import analyze_commissioned_model as commissioned  # noqa: E402
import analyze_visibility_patch_rgb as visibility  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_capture_v2(root: Path, protocol_override: Path | None = None):
    execution, records, rounds = commissioned.load_capture(root, protocol_override)
    retained = []
    for record in records:
        x0, y0, x1, y1 = record["bbox_xyxy"]
        edge = min(float(x0), float(y0), 1280.0 - float(x1), 720.0 - float(y1))
        if edge >= 5.0:
            retained.append(record)
    execution = dict(execution)
    execution["sensor_gate_id"] = "commissioning_sensor_gate_v2"
    execution["v1_admitted_reference_supported_rows"] = len(records)
    execution["v2_admitted_reference_supported_rows"] = len(retained)
    return execution, retained, rounds


def fit_ridge(records: list[dict], train: np.ndarray):
    features = commissioned.mean_features(records, train)
    target = np.stack([records[index]["target_ray"] for index in train])
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(features, target)


def drive_aggregated_metrics(
    target: np.ndarray, prediction: np.ndarray, drives: np.ndarray, index: np.ndarray
) -> dict:
    """Compute every statistic within drive before equal-drive aggregation."""

    error = np.linalg.norm(target[index] - prediction[index], axis=1)
    selected_drives = drives[index]
    per_drive = {}
    for drive in sorted(set(selected_drives.tolist())):
        values = error[selected_drives == drive]
        per_drive[drive] = {
            "n": int(len(values)),
            "mean_error_m": float(np.mean(values)),
            "median_error_m": float(np.median(values)),
            "mse_m2": float(np.mean(values ** 2)),
            "rmse_m": float(np.sqrt(np.mean(values ** 2))),
            "p90_error_m": float(np.quantile(values, 0.90)),
            "p95_error_m": float(np.quantile(values, 0.95)),
            "above_1m_count": int(np.sum(values > 1.0)),
        }
    summaries = list(per_drive.values())
    return {
        "n": int(len(index)),
        "drive_count": int(len(summaries)),
        "aggregation": "statistic within complete drive, then arithmetic mean over drives",
        "equal_drive_mean_error_m": float(np.mean([row["mean_error_m"] for row in summaries])),
        "equal_drive_median_error_m": float(np.mean([row["median_error_m"] for row in summaries])),
        "equal_drive_mse_m2": float(np.mean([row["mse_m2"] for row in summaries])),
        "equal_drive_rmse_m": float(np.mean([row["rmse_m"] for row in summaries])),
        "equal_drive_p90_error_m": float(np.mean([row["p90_error_m"] for row in summaries])),
        "equal_drive_p95_error_m": float(np.mean([row["p95_error_m"] for row in summaries])),
        "above_1m_count": int(sum(row["above_1m_count"] for row in summaries)),
        "per_drive": per_drive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--gate-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260915)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--finalize-existing", action="store_true",
        help="Recompute drive-first summaries from an existing completed prediction bundle.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.finalize_existing:
        raise FileExistsError(f"refusing nonempty output directory {output}")
    gate_report_path = args.gate_report.resolve()
    gate_report = json.loads(gate_report_path.read_text(encoding="utf-8"))
    if gate_report.get("gate_id") != "commissioning_sensor_gate_v2":
        raise RuntimeError("gate report is not the v2 replay")
    if gate_report.get("audit_analysis_permitted") is not False:
        raise RuntimeError("gate report does not keep audit sealed")

    if not args.finalize_existing:
        visibility.load_capture = load_capture_v2
        saved_argv = sys.argv
        try:
            sys.argv = [
                str(Path(__file__)),
                "--capture-root", str(args.capture_root.resolve()),
                "--output", str(output),
                "--seed", str(args.seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--device", str(args.device),
            ]
            visibility.main()
        finally:
            sys.argv = saved_argv

    execution, records, _ = load_capture_v2(args.capture_root.resolve())
    partition = np.asarray([record["partition"] for record in records])
    drives = np.asarray([record["drive"] for record in records])
    fit_index = np.flatnonzero(partition == "fit")
    development_index = np.flatnonzero(partition == "development")
    fit_drives = sorted(set(drives[fit_index].tolist()))
    target = np.stack([record["target_ray"] for record in records])

    prediction_path = output / "candidate_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: np.asarray(archive[key]) for key in archive.files}
    if "shuffled_prediction_ray_m" not in prediction:
        grids, _, _ = visibility.load_visibility(records)
        features = commissioned.mean_features(records, np.arange(len(records)))
        checkpoint = torch.load(
            output / "box_mlp_visibility_residual_fit_only.pt",
            map_location="cpu",
        )
        model = visibility.VisibilityPatchResidualNet(features.shape[1])
        model.load_state_dict(checkpoint["state_dict"])
        fitted_visibility = visibility.FittedVisibilityResidual(
            model,
            np.asarray(checkpoint["feature_mean"], dtype=float),
            np.asarray(checkpoint["feature_sd"], dtype=float),
            torch.device("cpu"),
        )
        shuffled_grids = grids.copy()
        rng = np.random.default_rng(args.seed + 913)
        cameras = np.asarray([record["camera"] for record in records])
        for camera in sorted(set(cameras[development_index].tolist())):
            members = development_index[cameras[development_index] == camera]
            shuffled_grids[members] = grids[rng.permutation(members)]
        shuffled_prediction = np.full_like(target, np.nan)
        shuffled_prediction[development_index], _ = fitted_visibility.predict(
            grids,
            features,
            prediction["base_prediction_ray_m"],
            development_index,
            batch_size=args.batch_size,
            override_grids=shuffled_grids,
        )
        prediction["shuffled_prediction_ray_m"] = shuffled_prediction
    raw_prediction = np.zeros_like(target)
    if "ridge_prediction_ray_m" in prediction:
        ridge_prediction = prediction["ridge_prediction_ray_m"]
        final_ridge = joblib.load(output / "linear_ridge_fit_only.joblib")
    else:
        ridge_prediction = np.full_like(target, np.nan)
        for fold, held_drive in enumerate(fit_drives):
            train = fit_index[drives[fit_index] != held_drive]
            test = fit_index[drives[fit_index] == held_drive]
            model = fit_ridge(records, train)
            ridge_prediction[test] = model.predict(
                commissioned.mean_features(records, test)
            )
        final_ridge = fit_ridge(records, fit_index)
        ridge_prediction[development_index] = final_ridge.predict(
            commissioned.mean_features(records, development_index)
        )
        if not np.isfinite(ridge_prediction[np.r_[fit_index, development_index]]).all():
            raise RuntimeError("ridge predictions are incomplete")
    if len(prediction["drive_id"]) != len(records):
        raise RuntimeError("prediction row count differs from v2 input rows")
    prediction["raw_prediction_ray_m"] = raw_prediction
    prediction["ridge_prediction_ray_m"] = ridge_prediction
    np.savez_compressed(prediction_path, **prediction)

    def summaries(values: np.ndarray) -> dict:
        return {
            "fit_out_of_fold": drive_aggregated_metrics(target, values, drives, fit_index),
            "development": drive_aggregated_metrics(
                target, values, drives, development_index
            ),
        }

    report_path = output / "visibility_patch_comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema"] = "matched_correction_ladder_v2"
    report["sensor_gate_id"] = "commissioning_sensor_gate_v2"
    report["gate_report"] = {
        "path": str(gate_report_path),
        "sha256": sha256(gate_report_path),
    }
    report["models"]["raw_box"] = summaries(raw_prediction)
    report["models"]["linear_ridge"] = summaries(ridge_prediction)
    report["models"]["box_mlp"].update(summaries(prediction["base_prediction_ray_m"]))
    report["models"]["box_mlp_visibility_residual"].update(
        summaries(prediction["candidate_prediction_ray_m"])
    )
    report["models"]["shuffled_visibility_control"]["development"] = (
        drive_aggregated_metrics(
            target, prediction["shuffled_prediction_ray_m"], drives, development_index
        )
    )
    eligible = ["raw_box", "linear_ridge", "box_mlp"]
    if report.get("selected") is True:
        eligible.append("box_mlp_visibility_residual")
    selected = min(
        eligible,
        key=lambda name: report["models"][name]["development"]["equal_drive_rmse_m"],
    )
    report["selected_correction"] = selected
    report["correction_selection_rule"] = (
        "minimum equal-development-drive RMSE; visibility residual is eligible only "
        "when every predeclared visibility gate passes"
    )
    report["audit_drive_ids_not_opened"] = gate_report["provenance"][
        "audit_drive_ids_not_opened"
    ]
    report["v1_admitted_reference_supported_rows"] = execution[
        "v1_admitted_reference_supported_rows"
    ]
    report["v2_admitted_reference_supported_rows"] = execution[
        "v2_admitted_reference_supported_rows"
    ]
    joblib.dump(final_ridge, output / "linear_ridge_fit_only.joblib")
    report["artifacts"]["ridge_model"] = {
        "path": "linear_ridge_fit_only.joblib",
        "sha256": sha256(output / "linear_ridge_fit_only.joblib"),
    }
    report["artifacts"]["predictions"]["sha256"] = sha256(prediction_path)
    report["source_hashes"] = {
        Path(__file__).name: sha256(Path(__file__)),
        "analyze_visibility_patch_rgb.py": sha256(
            HERE / "analyze_visibility_patch_rgb.py"
        ),
        "commissioning_sensor_gate_v2.yaml": sha256(
            REPO / "config/commissioning_sensor_gate_v2.yaml"
        ),
    }
    visibility.atomic_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "fit_rows": int(len(fit_index)),
        "development_rows": int(len(development_index)),
        "selected_correction": selected,
        "development": {
            name: report["models"][name]["development"] for name in eligible
        },
        "audit_drive_ids_not_opened": report["audit_drive_ids_not_opened"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
