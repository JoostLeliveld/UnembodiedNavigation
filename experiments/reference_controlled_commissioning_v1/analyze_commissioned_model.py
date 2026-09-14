#!/usr/bin/env python3
"""Fit the commissioned mean correction and its matching covariance on one capture.

Stage 1 fits the mean correction on the fit drives, leave-one-drive-out, so every fit
residual is out of fold, and applies the all-fit-drive model to the development drives.
Stage 2 fits the covariance ladder on those out-of-fold residuals and scores it on the
development drives.  Stage 3 replays a position belief along each development drive.
Stage 4 measures dependence between simultaneous cameras.

Audit drives are never opened.  The complete drive is the replication unit throughout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE)]

from build_tables import _camera_positions  # noqa: E402

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CHI2_2 = {"50": 1.38629436112, "90": 4.60517018599, "95": 5.99146454711, "99": 9.21034037198}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def spd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def sample_covariance(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.eye(2) * 1e-4
    centered = values - values.mean(axis=0)
    return spd(centered.T @ centered / (len(values) - 1))


def ray_basis(camera_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    ray = raw_xy - camera_xy
    norm = float(np.linalg.norm(ray))
    if norm < 1e-6:
        raise RuntimeError("degenerate camera ray")
    along = ray / norm
    return np.stack([along, np.asarray([-along[1], along[0]])], axis=1)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_capture(
    root: Path, protocol_override: Path | None = None
) -> tuple[dict, list[dict], list[dict]]:
    execution = json.loads((root / "campaign_execution.json").read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed":
        raise RuntimeError("capture is not complete with the audit sealed")
    if execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("expected audit_analysis_permitted=false")
    declared_protocol_path = Path(execution["protocol"]).resolve()
    protocol_path = (
        declared_protocol_path if protocol_override is None else protocol_override.resolve()
    )
    if protocol_override is not None:
        # Older completed captures recorded an absolute path to the live protocol file.
        # Permit an immutable per-drive snapshot only when every completed drive carries
        # an identical copy.  This avoids silently interpreting old data with a protocol
        # that was edited for a later campaign.
        expected_hash = sha256(protocol_path)
        for drive_id in execution.get("completed_drive_ids", []):
            matches = sorted((root / drive_id / "experiment_logs").glob(
                f"*/{protocol_path.name}"
            ))
            if len(matches) != 1 or sha256(matches[0]) != expected_hash:
                raise RuntimeError(
                    f"{drive_id}: protocol snapshot is missing, ambiguous, or differs"
                )
    import yaml

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    camera_xy = {key: np.asarray(value, dtype=float)
                 for key, value in _camera_positions(protocol).items()}

    records: list[dict] = []
    rounds: list[dict] = []
    missing: list[str] = []
    for drive_id in sorted(d["id"] for d in protocol["drives"]):
        if drive_id.startswith("audit_"):
            continue
        run = root / drive_id
        if not run.is_dir():
            # A capture may legitimately be short of drives -- the host filesystem filled
            # during collection here -- but only when the campaign says so explicitly and
            # names the drive as not collected.  A drive that is merely absent is an error.
            if drive_id in set(execution.get("completed_drive_ids", [])):
                raise RuntimeError(f"{drive_id} is marked completed but its directory is gone")
            if not execution.get("incomplete_note"):
                raise RuntimeError(f"missing drive directory {run}")
            missing.append(drive_id)
            continue
        integrity = json.loads((run / "tables/integrity.json").read_text(encoding="utf-8"))
        if integrity.get("passed") is not True:
            raise RuntimeError(f"{drive_id}: integrity did not pass")
        partition = integrity["partition"]
        if partition not in {"fit", "development"}:
            raise RuntimeError(f"{drive_id}: unexpected partition {partition}")

        with (run / "tables/camera_measurements.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["sensor_gate_admitted"] != "1" or row["reference_supported"] != "1":
                    raise RuntimeError("non-admitted row in camera_measurements")
                if str(row["controller_state"]).startswith("terminal"):
                    raise RuntimeError("parked row reached the measurement table")
                camera = row["camera_id"]
                raw = np.asarray([float(row["raw_projected_x"]), float(row["raw_projected_y"])])
                reference = np.asarray([float(row["reference_x"]), float(row["reference_y"])])
                basis = ray_basis(camera_xy[camera], raw)
                ray = raw - camera_xy[camera]
                width = float(row["bbox_xmax"]) - float(row["bbox_xmin"])
                height = float(row["bbox_ymax"]) - float(row["bbox_ymin"])
                records.append({
                    "drive": drive_id, "partition": partition, "camera": camera,
                    "stamp_ns": int(row["capture_stamp_ns"]),
                    "source_frame_id": row["source_frame_id"],
                    "crop_path": row["crop_path"],
                    "bbox_xyxy": np.asarray([
                        float(row["bbox_xmin"]), float(row["bbox_ymin"]),
                        float(row["bbox_xmax"]), float(row["bbox_ymax"]),
                    ]),
                    "raw": raw, "reference": reference, "basis": basis,
                    "target_ray": basis.T @ (reference - raw),
                    "range_m": float(np.linalg.norm(ray)),
                    "bearing_rad": math.atan2(ray[1], ray[0]),
                    "bbox_w": width, "bbox_h": height,
                    "confidence": float(row["detector_score"]),
                    "bottom_v": float(row["bbox_ymax"]),
                    "bottom_u": 0.5 * (float(row["bbox_xmin"]) + float(row["bbox_xmax"])),
                })

        with (run / "tables/network_rounds.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["reference_supported"] != "1":
                    continue
                rounds.append({
                    "drive": drive_id, "partition": partition,
                    "stamp_ns": int(row["capture_stamp_ns"]),
                    "reference": np.asarray([float(row["reference_x"]), float(row["reference_y"])]),
                    "reference_yaw": float(row["reference_yaw"]),
                    "reference_speed_mps": float(row["reference_speed_mps"]),
                })
    records.sort(key=lambda item: (item["drive"], item["stamp_ns"], item["camera"]))
    rounds.sort(key=lambda item: (item["drive"], item["stamp_ns"]))
    execution = dict(execution)
    execution["uncollected_drives"] = missing
    execution["declared_protocol"] = str(declared_protocol_path)
    execution["effective_protocol"] = str(protocol_path)
    execution["effective_protocol_sha256"] = sha256(protocol_path)
    return execution, records, rounds


# --------------------------------------------------------------------------------------
# Stage 1: mean correction, leave one drive out
# --------------------------------------------------------------------------------------

def mean_features(records: list[dict], index: np.ndarray) -> np.ndarray:
    rows = []
    for i in index:
        r = records[i]
        rows.append([
            r["range_m"], 1.0 / max(r["range_m"], 1e-6),
            math.sin(r["bearing_rad"]), math.cos(r["bearing_rad"]),
            r["bbox_w"] / 1280.0, r["bbox_h"] / 720.0,
            r["bbox_w"] / max(r["bbox_h"], 1e-6),
            r["bottom_u"] / 1280.0, r["bottom_v"] / 720.0, r["confidence"],
            *[float(r["camera"] == c) for c in CAMERAS],
        ])
    return np.asarray(rows, dtype=float)


def fit_mean(records: list[dict], train: np.ndarray, model: str, seed: int):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    target = np.stack([records[i]["target_ray"] for i in train])
    if model == "raw_box":
        return None
    features = mean_features(records, train)
    if model == "box_mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(96, 64), max_iter=1200, random_state=seed,
            early_stopping=True, n_iter_no_change=25)).fit(features, target)
    raise ValueError(model)


def predict_mean(records: list[dict], index: np.ndarray, model: str, fitted) -> np.ndarray:
    if model == "raw_box":
        return np.zeros((len(index), 2))
    return fitted.predict(mean_features(records, index))


# --------------------------------------------------------------------------------------
# covariance ladder
# --------------------------------------------------------------------------------------

def covariance_scores(covariances: np.ndarray, residuals: np.ndarray,
                      drives: np.ndarray) -> dict:
    nll, d2, areas = [], [], []
    for matrix, residual in zip(covariances, residuals):
        inverse = np.linalg.inv(matrix)
        determinant = max(float(np.linalg.det(matrix)), 1e-14)
        distance = float(residual @ inverse @ residual)
        d2.append(distance)
        nll.append(0.5 * (2 * math.log(2 * math.pi) + math.log(determinant) + distance))
        areas.append(math.pi * CHI2_2["95"] * math.sqrt(determinant))
    nll, d2, areas = np.asarray(nll), np.asarray(d2), np.asarray(areas)
    per_drive = {d: float(np.mean(nll[drives == d])) for d in sorted(set(drives.tolist()))}
    return {
        "equal_drive_mean_nll": float(np.mean(list(per_drive.values()))),
        "per_drive_nll": per_drive,
        "containment": {k: float(np.mean(d2 <= v)) for k, v in CHI2_2.items()},
        "median_95_ellipse_area_m2": float(np.median(areas)),
        "above_99_fraction": float(np.mean(d2 > CHI2_2["99"])),
        "n": int(len(nll)),
    }


def fit_covariance(records: list[dict], index: np.ndarray, residual: np.ndarray) -> dict:
    camera = np.asarray([records[i]["camera"] for i in index])
    width = np.asarray([records[i]["bbox_w"] for i in index])
    models: dict[str, Any] = {"R0": sample_covariance(residual).tolist(), "R1": {}, "R2": {}}
    for c in CAMERAS:
        mask = camera == c
        models["R1"][c] = (sample_covariance(residual[mask]) if mask.sum() >= 5
                           else sample_covariance(residual)).tolist()
        if mask.sum() >= 30:
            edges = np.quantile(width[mask], [1 / 3, 2 / 3])
            entries = []
            for b in range(3):
                low = -math.inf if b == 0 else edges[b - 1]
                high = math.inf if b == 2 else edges[b]
                sub = mask & (width > low) & (width <= high)
                n = int(sub.sum())
                weight = n / (n + 20.0)
                entries.append(spd(weight * sample_covariance(residual[sub])
                                   + (1 - weight) * np.asarray(models["R1"][c])).tolist())
            models["R2"][c] = {"edges": edges.tolist(), "covariances": entries}
        else:
            models["R2"][c] = None
    return models


def predict_covariance(models: dict, name: str, records: list[dict],
                       index: np.ndarray, scale: float = 1.0) -> np.ndarray:
    out = np.empty((len(index), 2, 2))
    for position, i in enumerate(index):
        camera = records[i]["camera"]
        if name == "R0":
            out[position] = np.asarray(models["R0"])
        elif name == "R1":
            out[position] = np.asarray(models["R1"][camera])
        else:
            entry = models["R2"][camera]
            if entry is None:
                out[position] = np.asarray(models["R1"][camera])
            else:
                b = int(np.searchsorted(entry["edges"], records[i]["bbox_w"], side="left"))
                out[position] = np.asarray(entry["covariances"][b])
    return np.stack([spd(scale * m) for m in out])


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260911)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    execution, records, rounds = load_capture(arguments.capture_root.resolve())
    partition = np.asarray([r["partition"] for r in records])
    drive_of = np.asarray([r["drive"] for r in records])
    fit_index = np.flatnonzero(partition == "fit")
    dev_index = np.flatnonzero(partition == "development")
    fit_drives = sorted(set(drive_of[fit_index].tolist()))
    dev_drives = sorted(set(drive_of[dev_index].tolist()))
    if len(fit_drives) < 2:
        raise RuntimeError("leave-one-drive-out needs at least two fit drives")

    target = np.stack([r["target_ray"] for r in records])
    MODELS = ("raw_box", "box_mlp")

    # ---- Stage 1: mean correction ----------------------------------------------------
    oof = {name: np.zeros((len(records), 2)) for name in MODELS}
    for name in MODELS:
        for held in fit_drives:
            inner = fit_index[drive_of[fit_index] != held]
            outer = fit_index[drive_of[fit_index] == held]
            fitted = fit_mean(records, inner, name, arguments.seed)
            oof[name][outer] = predict_mean(records, outer, name, fitted)
        fitted_all = fit_mean(records, fit_index, name, arguments.seed)
        oof[name][dev_index] = predict_mean(records, dev_index, name, fitted_all)

    stage1 = {}
    for name in MODELS:
        entry = {}
        for label, index in (("fit_out_of_fold", fit_index), ("development", dev_index)):
            error = np.linalg.norm(target[index] - oof[name][index], axis=1)
            per_drive = {d: float(np.sqrt(np.mean(error[drive_of[index] == d] ** 2)))
                         for d in sorted(set(drive_of[index].tolist()))}
            entry[label] = {
                "equal_drive_rmse_m": float(np.mean(list(per_drive.values()))),
                "median_error_m": float(np.median(error)),
                "p95_error_m": float(np.percentile(error, 95)),
                "per_drive_rmse_m": per_drive,
            }
        stage1[name] = entry
    selected_mean = min(MODELS, key=lambda n: stage1[n]["development"]["equal_drive_rmse_m"])

    # ---- Stage 2: covariance ladder on the selected mean ------------------------------
    residual_all = target - oof[selected_mean]
    covariance_models = fit_covariance(records, fit_index, residual_all[fit_index])
    stage2 = {}
    for name in ("R0", "R1", "R2"):
        covariances = predict_covariance(covariance_models, name, records, dev_index)
        stage2[name] = covariance_scores(covariances, residual_all[dev_index],
                                         drive_of[dev_index])
    # one scalar calibration fitted on the fit drives only
    fit_cov = predict_covariance(covariance_models, "R2", records, fit_index)
    d2 = np.asarray([residual_all[i] @ np.linalg.inv(m) @ residual_all[i]
                     for i, m in zip(fit_index, fit_cov)])
    scale = max(1.0, float(np.quantile(d2, 0.90) / CHI2_2["90"]),
                float(np.quantile(d2, 0.95) / CHI2_2["95"]))
    stage2["R2C"] = covariance_scores(
        predict_covariance(covariance_models, "R2", records, dev_index, scale),
        residual_all[dev_index], drive_of[dev_index])
    stage2["R2C"]["calibration_scale"] = scale
    selected_cov = min(("R0", "R1", "R2", "R2C"),
                       key=lambda n: abs(stage2[n]["containment"]["90"] - 0.90)
                       + abs(stage2[n]["containment"]["95"] - 0.95))

    # ---- Stage 4: cross-camera dependence --------------------------------------------
    chosen = predict_covariance(covariance_models, selected_cov.rstrip("C"), records,
                                dev_index, stage2.get("R2C", {}).get("calibration_scale", 1.0)
                                if selected_cov == "R2C" else 1.0)
    by_round: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    for position, i in enumerate(dev_index):
        factor = np.linalg.cholesky(chosen[position])
        by_round.setdefault((records[i]["drive"], records[i]["stamp_ns"]), []).append(
            (records[i]["camera"], np.linalg.solve(factor, residual_all[i])))
    pairs: dict[str, list[float]] = {}
    for members in by_round.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                key = "|".join(sorted([members[a][0], members[b][0]]))
                pairs.setdefault(key, []).append(float(np.dot(members[a][1], members[b][1]) / 2.0))
    stage4 = {"pairwise": {}, "rounds_with_multiple_cameras":
              int(sum(1 for m in by_round.values() if len(m) > 1))}
    for key, values in sorted(pairs.items()):
        if len(values) < 10:
            continue
        v = np.asarray(values)
        stage4["pairwise"][key] = {
            "n": int(len(v)), "mean_correlation": float(v.mean()),
            "se": float(v.std(ddof=1) / math.sqrt(len(v))),
        }

    report = {
        "schema": "commissioned_model_report.v1",
        "capture_root": str(arguments.capture_root),
        "audit_analysis_permitted": False,
        "fit_drives": fit_drives,
        "development_drives": dev_drives,
        "counts": {"fit_measurements": int(len(fit_index)),
                   "development_measurements": int(len(dev_index)),
                   "rounds": len(rounds)},
        "stage1_mean_correction": stage1,
        "selected_mean_model": selected_mean,
        "stage2_covariance_ladder": stage2,
        "selected_covariance_model": selected_cov,
        "stage4_cross_camera": stage4,
    }
    atomic_json(arguments.output / "commissioned_model_report.json", report)
    print(json.dumps({
        "fit_drives": len(fit_drives), "development_drives": len(dev_drives),
        "mean_development_rmse_cm": {n: round(stage1[n]["development"]["equal_drive_rmse_m"] * 100, 2)
                                     for n in MODELS},
        "selected_mean": selected_mean,
        "covariance_nll": {n: round(stage2[n]["equal_drive_mean_nll"], 3) for n in stage2},
        "covariance_cov95": {n: round(stage2[n]["containment"]["95"], 3) for n in stage2},
        "selected_covariance": selected_cov,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
