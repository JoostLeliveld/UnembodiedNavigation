#!/usr/bin/env python3
"""Fit the commissioned Bernoulli-GP availability field q_i(p).

Availability is the probability that camera i turns an opportunity into a usable
localization measurement: the frozen detector returns a box and that box passes the
deterministic, belief-independent sensor gate.  Every opportunity contributes, detector
misses and gate refusals as negatives, so this is the Bernoulli event the planner needs
and not a summary of the readings that happened to succeed.

What the planner may condition on is the binding constraint.  At planning time there is no
image, so the only inputs available are the predicted pose and the fixed camera geometry.
The candidate ladder therefore runs from a per-camera constant to Gaussian-process fields
over position, and nothing in it may read the detector output.

The replication unit is the route: replicates of one lap share their geometry, so a model
held out on a replicate is scored on a route it has effectively memorised. Fit routes are
held out one at a time for comparison. The development routes finalize the spatial length
scale before the audit is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src/reliability")]

import analyze_commissioned_model as base  # noqa: E402
from reliability.bernoulli_gp import (  # noqa: E402
    aggregate_binomial,
    fit_laplace_binomial,
    predict_laplace,
)

CAMERAS = base.CAMERAS
GP_IMPLEMENTATION = REPO / "src/reliability/reliability/bernoulli_gp.py"
PROBABILITY_CLIP = (0.001, 0.999)
CELL_SIZE_M = 0.20
LATENT_VARIANCE = 4.0
DEPLOYMENT_MODEL = "A2_gp_position_ls1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_of(drive_id: str) -> str:
    return drive_id.rsplit("_r", 1)[0]


def load_opportunities(root: Path) -> tuple[dict, list[dict]]:
    execution = json.loads((root / "campaign_execution.json").read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed":
        raise RuntimeError("capture is not complete with the audit sealed")
    if execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("expected audit_analysis_permitted=false")
    import yaml

    protocol = yaml.safe_load(Path(execution["protocol"]).resolve().read_text(encoding="utf-8"))
    camera_xy = {k: np.asarray(v, dtype=float)
                 for k, v in base._camera_positions(protocol).items()}

    rows: list[dict] = []
    for drive in sorted(d["id"] for d in protocol["drives"]):
        if drive.startswith("audit_"):
            continue
        run = root / drive
        if not run.is_dir():
            if not execution.get("incomplete_note"):
                raise RuntimeError(f"missing drive directory {run}")
            continue
        integrity = json.loads((run / "tables/integrity.json").read_text(encoding="utf-8"))
        if integrity.get("passed") is not True:
            raise RuntimeError(f"{drive}: integrity did not pass")
        with (run / "tables/camera_opportunities.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["reference_supported"] != "1":
                    continue
                camera = row["camera_id"]
                position = np.asarray([float(row["reference_x"]), float(row["reference_y"])])
                offset = position - camera_xy[camera]
                rows.append({
                    "drive": drive, "route": route_of(drive),
                    "partition": integrity["partition"], "camera": camera,
                    "x": position[0], "y": position[1],
                    "yaw": float(row["reference_yaw"]),
                    "range_m": float(np.linalg.norm(offset)),
                    "bearing_rad": math.atan2(offset[1], offset[0]),
                    "admitted": 1 if row["sensor_gate_admitted"] == "1" else 0,
                })
    return execution, rows


def scores(label: np.ndarray, probability: np.ndarray, routes: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    per_route = {}
    for route in sorted(set(routes.tolist())):
        mask = routes == route
        if mask.sum() < 50 or len(set(label[mask].tolist())) < 2:
            continue
        per_route[route] = {
            "brier": float(brier_score_loss(label[mask], probability[mask])),
            "auc": float(roc_auc_score(label[mask], probability[mask])),
            "log_loss": float(-np.mean(label[mask] * np.log(probability[mask])
                                       + (1 - label[mask]) * np.log(1 - probability[mask]))),
        }
    values = list(per_route.values())
    # reliability: predicted against observed rate in ten probability bins
    bins = np.clip((probability * 10).astype(int), 0, 9)
    reliability = []
    for b in range(10):
        mask = bins == b
        if mask.sum() >= 30:
            reliability.append({"bin": b / 10.0, "n": int(mask.sum()),
                                "predicted": float(probability[mask].mean()),
                                "observed": float(label[mask].mean())})
    calibration_error = float(sum(e["n"] * abs(e["predicted"] - e["observed"])
                                  for e in reliability) / max(len(label), 1))
    return {
        "equal_route_brier": float(np.mean([v["brier"] for v in values])) if values else math.nan,
        "equal_route_auc": float(np.mean([v["auc"] for v in values])) if values else math.nan,
        "equal_route_log_loss": float(np.mean([v["log_loss"] for v in values])) if values else math.nan,
        "per_route": per_route,
        "expected_calibration_error": calibration_error,
        "reliability": reliability,
        "base_rate": float(label.mean()),
        "n": int(len(label)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260912)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    execution, rows = load_opportunities(arguments.capture_root.resolve())
    partition = np.asarray([r["partition"] for r in rows])
    camera = np.asarray([r["camera"] for r in rows])
    route = np.asarray([r["route"] for r in rows])
    label = np.asarray([r["admitted"] for r in rows])
    fit_index = np.flatnonzero(partition == "fit")
    dev_index = np.flatnonzero(partition == "development")

    CANDIDATES = {
        "A0_camera_constant": None,
        "A1_gp_position_ls0.8": {"length_scale_m": 0.8, "uncertainty_penalty": 0.0},
        "A2_gp_position_ls1.0": {"length_scale_m": 1.0, "uncertainty_penalty": 0.0},
        "A3_gp_position_ls1.6": {"length_scale_m": 1.6, "uncertainty_penalty": 0.0},
        "A4_gp_position_ls3.2": {"length_scale_m": 3.2, "uncertainty_penalty": 0.0},
        "A5_gp_position_ls0.8_lcb0.5": {"length_scale_m": 0.8, "uncertainty_penalty": 0.5},
        "A6_gp_position_ls1.0_lcb0.5": {"length_scale_m": 1.0, "uncertainty_penalty": 0.5},
        "A7_gp_position_ls1.6_lcb0.5": {"length_scale_m": 1.6, "uncertainty_penalty": 0.5},
        "A8_gp_position_ls3.2_lcb0.5": {"length_scale_m": 3.2, "uncertainty_penalty": 0.5},
    }
    xy = np.asarray([[r["x"], r["y"]] for r in rows], dtype=float)
    model_cache: dict[tuple, dict] = {}

    def predict(name: str, train: np.ndarray, test: np.ndarray) -> np.ndarray:
        """Fit one position-only Bernoulli GP per camera and predict test rows."""
        out = np.empty(len(test))
        for value in CAMERAS:
            train_mask = train[camera[train] == value]
            test_mask = np.flatnonzero(camera[test] == value)
            if len(test_mask) == 0:
                continue
            if len(train_mask) < 50 or len(set(label[train_mask].tolist())) < 2:
                out[test_mask] = label[train_mask].mean() if len(train_mask) else label[train].mean()
                continue
            if CANDIDATES[name] is None:
                out[test_mask] = label[train_mask].mean()
                continue
            specification = CANDIDATES[name]
            key = (value, hashlib.sha256(train_mask.tobytes()).hexdigest(),
                   specification["length_scale_m"])
            model = model_cache.get(key)
            if model is None:
                centres, success, total = aggregate_binomial(
                    xy[train_mask], label[train_mask], cell_size_m=CELL_SIZE_M
                )
                model = fit_laplace_binomial(
                    centres, success, total,
                    length_scale_m=specification["length_scale_m"],
                    latent_variance=LATENT_VARIANCE,
                    prior_probability=float(label[train_mask].mean()),
                )
                model_cache[key] = model
            probability, _standard_deviation = predict_laplace(
                model, xy[test[test_mask]],
                uncertainty_penalty=specification["uncertainty_penalty"],
            )
            out[test_mask] = probability
        return out

    # ---- model choice on held-out fit routes ------------------------------------------
    selection = {}
    fit_routes = sorted(set(route[fit_index].tolist()))
    for name in CANDIDATES:
        predicted = np.zeros(len(fit_index))
        for held in fit_routes:
            inner = fit_index[route[fit_index] != held]
            outer_positions = np.flatnonzero(route[fit_index] == held)
            predicted[outer_positions] = predict(name, inner, fit_index[outer_positions])
        selection[name] = scores(label[fit_index], predicted, route[fit_index])

    gp_candidates = [name for name in CANDIDATES if name != "A0_camera_constant"]
    # ---- report on the development routes, which the fit never saw --------------------
    development = {}
    for name in CANDIDATES:
        development[name] = scores(label[dev_index], predict(name, fit_index, dev_index),
                                   route[dev_index])

    if DEPLOYMENT_MODEL not in gp_candidates:
        raise RuntimeError(f"unknown deployment model {DEPLOYMENT_MODEL}")
    chosen = DEPLOYMENT_MODEL

    per_camera = {}
    chosen_dev = predict(chosen, fit_index, dev_index)
    for value in CAMERAS:
        mask = camera[dev_index] == value
        if mask.sum() >= 50 and len(set(label[dev_index][mask].tolist())) > 1:
            per_camera[value] = scores(label[dev_index][mask], chosen_dev[mask],
                                       route[dev_index][mask])

    report = {
        "schema": "commissioned_availability_selection.v2",
        "capture_root": str(arguments.capture_root),
        "audit_analysis_permitted": False,
        "event": "frozen detector returned a box and the box passed the deterministic sensor gate",
        "planner_inputs_only": ["camera identity", "predicted position"],
        "fit_routes": fit_routes,
        "development_routes": sorted(set(route[dev_index].tolist())),
        "counts": {"fit_opportunities": int(len(fit_index)),
                   "development_opportunities": int(len(dev_index)),
                   "overall_admitted_rate": float(label.mean())},
        "selection_on_held_out_fit_routes": selection,
        "selected_model": chosen,
        "selection_basis": {
            "stage": "development",
            "reason": "1.0 m spatial length scale chosen during development review",
            "audit_opened": False,
        },
        "development_routes_unseen": development,
        "selected_per_camera_development": per_camera,
        "gp_implementation": {
            "likelihood": "binomial aggregation of every Bernoulli opportunity",
            "inference": "Laplace approximation",
            "kernel": "isotropic squared exponential over world x,y",
            "spatial_cell_size_m": CELL_SIZE_M,
            "latent_variance": LATENT_VARIANCE,
            "candidate_hyperparameters": CANDIDATES,
        },
    }

    selected_specification = CANDIDATES[chosen]
    deployment_models = {}
    arrays = {}
    for value in CAMERAS:
        camera_fit = fit_index[camera[fit_index] == value]
        centres, success, total = aggregate_binomial(
            xy[camera_fit], label[camera_fit], cell_size_m=CELL_SIZE_M
        )
        model = fit_laplace_binomial(
            centres, success, total,
            length_scale_m=selected_specification["length_scale_m"],
            latent_variance=LATENT_VARIANCE,
            prior_probability=float(label[camera_fit].mean()),
        )
        prefix = value.replace("camera_", "camera_")
        for field in ("centres_xy_m", "alpha", "sqrt_weight", "cholesky"):
            arrays[f"{prefix}__{field}"] = np.asarray(model[field], dtype=float)
        deployment_models[value] = {
            "array_prefix": prefix,
            "prior_mean_logit": model["prior_mean_logit"],
            "length_scale_m": model["length_scale_m"],
            "latent_variance": model["latent_variance"],
            "jitter": model["jitter"],
            "training_cells": model["training_cells"],
            "training_opportunities": model["training_opportunities"],
            "iterations": model["iterations"],
            "converged": model["converged"],
        }

    parameter_path = arguments.output / "availability_gp_parameters.npz"
    parameter_buffer = io.BytesIO()
    np.savez_compressed(parameter_buffer, **arrays)
    parameter_path.write_bytes(parameter_buffer.getvalue())
    protocol_path = Path(execution["protocol"]).resolve()
    protocol = __import__("yaml").safe_load(protocol_path.read_text(encoding="utf-8"))
    world_path = REPO / protocol["world"]["path"]
    if sha256(world_path) != protocol["world"]["sha256"]:
        raise RuntimeError("commissioning world differs from the frozen protocol")
    runtime_artifact = {
        "schema": "thesis_commissioned_availability_model.v2",
        "model": chosen,
        "final_audit_accessed": False,
        "world_sha256": sha256(world_path),
        "camera_order": list(CAMERAS),
        "feature_names": ["robot_x_m", "robot_y_m"],
        "probability_clip": list(PROBABILITY_CLIP),
        "likelihood": "Bernoulli",
        "inference": "Laplace",
        "kernel": "isotropic_rbf",
        "spatial_cell_size_m": CELL_SIZE_M,
        "uncertainty_penalty": selected_specification["uncertainty_penalty"],
        "parameter_file": parameter_path.name,
        "parameter_file_sha256": sha256(parameter_path),
        "parameters": deployment_models,
        "training_population": "all fitting-route camera opportunities; audit sealed",
        "source_hashes": {
            "campaign_execution.json": sha256(arguments.capture_root / "campaign_execution.json"),
            str(protocol_path.relative_to(REPO)): sha256(protocol_path),
            str(Path(__file__).resolve().relative_to(REPO)): sha256(Path(__file__).resolve()),
            str(GP_IMPLEMENTATION.relative_to(REPO)): sha256(GP_IMPLEMENTATION),
        },
    }
    report["deployment_artifact"] = {
        "path": "availability_runtime_model.json",
        "parameter_file": parameter_path.name,
        "parameter_file_sha256": runtime_artifact["parameter_file_sha256"],
    }
    base.atomic_json(arguments.output / "availability_model.json", report)
    base.atomic_json(arguments.output / "availability_runtime_model.json", runtime_artifact)
    print(json.dumps({
        "selected": chosen,
        "held_out_fit_brier": {k: round(v["equal_route_brier"], 4) for k, v in selection.items()},
        "development_brier": {k: round(v["equal_route_brier"], 4) for k, v in development.items()},
        "development_auc": {k: round(v["equal_route_auc"], 3) for k, v in development.items()},
        "development_calibration_error": {k: round(v["expected_calibration_error"], 4)
                                          for k, v in development.items()},
        "base_rate": round(float(label.mean()), 4),
        "per_camera_auc": {k: round(v["equal_route_auc"], 3) for k, v in per_camera.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
