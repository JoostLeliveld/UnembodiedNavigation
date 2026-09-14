#!/usr/bin/env python3
"""Compare camera-error covariance models on unsealed reference-controlled drives.

The frozen B3 mean correction from the bias study is held fixed.  This script fits the
static covariance ladder (S0--S2b), the dynamic camera-bias model (D1), the shared
cross-camera latent (X1) and replays each through matched estimators (E0--E4).  Audit
drives are never opened.

Stage A  static covariance ladder      -- held-out drive NLL, containment, ellipse area
Stage B  dynamic residual test         -- sequential predictive likelihood, whiteness
Stage C  estimator architecture        -- NEES, NIS, belief coverage, RMSE on held-out drives
Stage D  cross-camera dependence       -- simultaneous residual correlation, shared latent
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
sys.path.insert(0, str(HERE))

from build_tables import _camera_positions  # noqa: E402

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CHI2_2 = {"50": 1.38629436112, "90": 4.60517018599, "95": 5.99146454711, "99": 9.21034037198}
FROZEN_MEAN_MODEL = "B3_box_mlp"
EPS = 1e-9


# --------------------------------------------------------------------------------------
# integrity helpers
# --------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {key}")
    return value


def ray_basis(camera_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    """Columns are the along-ray and across-ray unit vectors in world coordinates."""
    ray = raw_xy - camera_xy
    norm = float(np.linalg.norm(ray))
    if norm < 1e-6:
        raise RuntimeError("degenerate camera ray")
    along = ray / norm
    across = np.asarray([-along[1], along[0]])
    return np.stack([along, across], axis=1)


def spd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return vectors @ np.diag(values) @ vectors.T


def sample_covariance(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    if len(values) < 3:
        return np.eye(2) * 1e-4
    centered = values - values.mean(axis=0)
    return spd(centered.T @ centered / (len(values) - 1), floor)


# --------------------------------------------------------------------------------------
# data loading: frozen B3 residuals plus synchronized network rounds
# --------------------------------------------------------------------------------------

def load_campaign(campaign_root: Path) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    execution_path = campaign_root / "campaign_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed":
        raise RuntimeError("campaign is not complete with audit sealed")
    if execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("expected audit_analysis_permitted=false")

    protocol_path = Path(execution["protocol"]).resolve()
    if sha256(protocol_path) != execution["protocol_sha256"]:
        raise RuntimeError("campaign protocol hash changed")
    import yaml

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    camera_xy = {key: np.asarray(value, dtype=float)
                 for key, value in _camera_positions(protocol).items()}

    selected = execution["selected_run_dirs"]
    audit_ids = sorted(key for key in selected if key.startswith("audit_"))
    usable_ids = sorted(key for key in selected if key.startswith(("fit_", "development_")))
    if len(audit_ids) != 8 or len(usable_ids) != 16:
        raise RuntimeError("expected eight sealed audit and sixteen unsealed drives")
    return execution, camera_xy, [{"drive_id": d, "run": Path(selected[d]).resolve()}
                                  for d in usable_ids]


def load_rounds(drives: list[dict]) -> list[dict]:
    """One record per synchronized capture round, with every admitted camera in it."""
    rounds: list[dict] = []
    for entry in drives:
        drive_id, run = entry["drive_id"], entry["run"]
        if "audit" in run.name or any(part.startswith("audit_") for part in run.parts):
            raise RuntimeError(f"refusing audit path {run}")
        integrity = json.loads((run / "tables/integrity.json").read_text(encoding="utf-8"))
        if integrity.get("passed") is not True or integrity.get("drive_id") != drive_id:
            raise RuntimeError(f"invalid table integrity for {drive_id}")
        partition = integrity["partition"]
        if partition not in {"fit", "development"}:
            raise RuntimeError("unexpected partition")

        with (run / "tables/network_rounds.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["drive_id"] != drive_id or row["partition"] != partition:
                    raise RuntimeError("round row crossed the declared partition")
                if row["reference_supported"] != "1":
                    continue
                rounds.append({
                    "drive": drive_id,
                    "partition": partition,
                    "route": row.get("route", drive_id.split("_", 1)[1]),
                    "stamp_ns": int(row["capture_stamp_ns"]),
                    "reference": np.asarray([require_float(row, "reference_x"),
                                             require_float(row, "reference_y")]),
                    "reference_yaw": require_float(row, "reference_yaw"),
                    "reference_speed_mps": require_float(row, "reference_speed_mps"),
                    "admitted": json.loads(row["sensor_gate_admitted_subset_json"]),
                })
    rounds.sort(key=lambda item: (item["drive"], item["stamp_ns"]))
    return rounds


def load_measurements(drives: list[dict], camera_xy: dict[str, np.ndarray],
                      corrections: dict[tuple[str, str, int], np.ndarray]) -> list[dict]:
    """Admitted camera readings with the frozen B3 corrected position and ray basis."""
    records: list[dict] = []
    for entry in drives:
        drive_id, run = entry["drive_id"], entry["run"]
        integrity = json.loads((run / "tables/integrity.json").read_text(encoding="utf-8"))
        partition = integrity["partition"]
        table_path = run / "tables/camera_measurements.csv"
        with table_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(integrity["measurement_rows"]):
            raise RuntimeError(f"measurement-row count changed for {drive_id}")
        for row in rows:
            if row["sensor_gate_admitted"] != "1" or row["reference_supported"] != "1":
                raise RuntimeError("non-admitted row in camera_measurements")
            camera = row["camera_id"]
            stamp = int(row["capture_stamp_ns"])
            key = (drive_id, camera, stamp)
            if key not in corrections:
                raise RuntimeError(f"no frozen {FROZEN_MEAN_MODEL} correction for {key}")
            raw = np.asarray([require_float(row, "raw_projected_x"),
                              require_float(row, "raw_projected_y")])
            reference = np.asarray([require_float(row, "reference_x"),
                                    require_float(row, "reference_y")])
            basis = ray_basis(camera_xy[camera], raw)
            corrected = corrections[key]
            residual_world = corrected - reference
            ray = raw - camera_xy[camera]
            records.append({
                "drive": drive_id,
                "partition": partition,
                "route": row["route"],
                "camera": camera,
                "stamp_ns": stamp,
                "raw": raw,
                "corrected": corrected,
                "reference": reference,
                "basis": basis,
                "residual_world": residual_world,
                "residual_ray": basis.T @ residual_world,
                "range_m": float(np.linalg.norm(ray)),
                "bearing_rad": math.atan2(ray[1], ray[0]),
                "stationary": str(row["controller_state"]).startswith("stationary"),
                "reference_speed_mps": require_float(row, "reference_speed_mps"),
            })
    records.sort(key=lambda item: (item["drive"], item["stamp_ns"], item["camera"]))
    return records


def load_frozen_corrections(prediction_files: list[Path]) -> dict[tuple[str, str, int], np.ndarray]:
    """Read the frozen B3 corrected positions produced by the bias study."""
    corrections: dict[tuple[str, str, int], np.ndarray] = {}
    x_key = f"{FROZEN_MEAN_MODEL}_corrected_x"
    y_key = f"{FROZEN_MEAN_MODEL}_corrected_y"
    for path in prediction_files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["drive_id"], row["camera_id"], int(row["capture_stamp_ns"]))
                value = np.asarray([float(row[x_key]), float(row[y_key])])
                if key in corrections and not np.allclose(corrections[key], value):
                    raise RuntimeError(f"conflicting frozen correction for {key}")
                corrections[key] = value
    if not corrections:
        raise RuntimeError("no frozen corrections loaded")
    return corrections


# --------------------------------------------------------------------------------------
# Stage A: static covariance ladder, fitted in the ray frame
# --------------------------------------------------------------------------------------

def cholesky_from_parameters(l1: float, l2: float, c: float) -> np.ndarray:
    factor = np.asarray([[math.exp(np.clip(l1, -12.0, 4.0)), 0.0],
                         [c, math.exp(np.clip(l2, -12.0, 4.0))]])
    return factor @ factor.T


def fit_geometry_covariance(residual_ray: np.ndarray, basis_features: np.ndarray) -> dict:
    """Cholesky parameters linear in the supplied basis, by maximum likelihood."""
    from scipy.optimize import minimize

    k = basis_features.shape[1]

    def negative_log_likelihood(theta: np.ndarray) -> float:
        a, b, c = theta[:k], theta[k:2 * k], theta[2 * k:]
        l1 = np.clip(basis_features @ a, -12.0, 4.0)
        l2 = np.clip(basis_features @ b, -12.0, 4.0)
        off = basis_features @ c
        s1, s2 = np.exp(l1), np.exp(l2)
        # R = L L^T with L = [[s1, 0], [off, s2]]
        r11 = s1 ** 2
        r12 = s1 * off
        r22 = off ** 2 + s2 ** 2
        determinant = np.maximum(r11 * r22 - r12 ** 2, 1e-14)
        e1, e2 = residual_ray[:, 0], residual_ray[:, 1]
        quadratic = (r22 * e1 ** 2 - 2 * r12 * e1 * e2 + r11 * e2 ** 2) / determinant
        return float(np.mean(0.5 * (np.log(determinant) + quadratic)))

    start = np.zeros(3 * k)
    marginal = sample_covariance(residual_ray)
    factor = np.linalg.cholesky(marginal)
    start[0] = math.log(max(factor[0, 0], 1e-6))
    start[k] = math.log(max(factor[1, 1], 1e-6))
    start[2 * k] = factor[1, 0]
    result = minimize(negative_log_likelihood, start, method="L-BFGS-B",
                      options={"maxiter": 800, "ftol": 1e-12})
    return {"theta": result.x.tolist(), "k": k, "success": bool(result.success)}


def predict_geometry_covariance(model: dict, basis_features: np.ndarray) -> np.ndarray:
    theta = np.asarray(model["theta"])
    k = model["k"]
    a, b, c = theta[:k], theta[k:2 * k], theta[2 * k:]
    s1 = np.exp(np.clip(basis_features @ a, -12.0, 4.0))
    s2 = np.exp(np.clip(basis_features @ b, -12.0, 4.0))
    off = basis_features @ c
    out = np.empty((len(basis_features), 2, 2))
    out[:, 0, 0] = s1 ** 2
    out[:, 0, 1] = out[:, 1, 0] = s1 * off
    out[:, 1, 1] = off ** 2 + s2 ** 2
    return out


def range_basis(range_m: np.ndarray, scale: float) -> np.ndarray:
    r = range_m / scale
    return np.stack([np.ones_like(r), r, r ** 2], axis=1)


def range_bearing_basis(range_m: np.ndarray, bearing: np.ndarray, scale: float) -> np.ndarray:
    r = range_m / scale
    return np.stack([np.ones_like(r), r, r ** 2,
                     np.sin(bearing), np.cos(bearing),
                     r * np.sin(bearing), r * np.cos(bearing)], axis=1)


def fit_static_ladder(records: list[dict], train: np.ndarray, scale: float) -> dict:
    """Fit S0, S1, S2a, S2b on the training drives.  All fits are in the ray frame."""
    residual = np.stack([records[i]["residual_ray"] for i in train])
    camera = np.asarray([records[i]["camera"] for i in train])
    range_m = np.asarray([records[i]["range_m"] for i in train])
    bearing = np.asarray([records[i]["bearing_rad"] for i in train])

    models: dict[str, Any] = {}
    models["S0"] = {"pooled": sample_covariance(residual).tolist(),
                    "mean": residual.mean(axis=0).tolist()}
    models["S1"] = {"per_camera": {}, "mean": {}}
    for value in CAMERAS:
        mask = camera == value
        if mask.sum() < 5:
            models["S1"]["per_camera"][value] = models["S0"]["pooled"]
            models["S1"]["mean"][value] = [0.0, 0.0]
            continue
        models["S1"]["per_camera"][value] = sample_covariance(residual[mask]).tolist()
        models["S1"]["mean"][value] = residual[mask].mean(axis=0).tolist()

    models["S2a"] = {"per_camera": {}}
    models["S2b"] = {"per_camera": {}}
    for value in CAMERAS:
        mask = camera == value
        if mask.sum() < 30:
            models["S2a"]["per_camera"][value] = None
            models["S2b"]["per_camera"][value] = None
            continue
        centered = residual[mask] - residual[mask].mean(axis=0)
        models["S2a"]["per_camera"][value] = fit_geometry_covariance(
            centered, range_basis(range_m[mask], scale))
        models["S2b"]["per_camera"][value] = fit_geometry_covariance(
            centered, range_bearing_basis(range_m[mask], bearing[mask], scale))
    return models


def predict_static(models: dict, name: str, records: list[dict], index: np.ndarray,
                   scale: float) -> np.ndarray:
    out = np.empty((len(index), 2, 2))
    if name == "S0":
        out[:] = np.asarray(models["S0"]["pooled"])
        return out
    for position, i in enumerate(index):
        camera = records[i]["camera"]
        if name == "S1":
            out[position] = np.asarray(models["S1"]["per_camera"][camera])
        else:
            entry = models[name]["per_camera"][camera]
            if entry is None:
                out[position] = np.asarray(models["S1"]["per_camera"][camera])
            else:
                if name == "S2a":
                    basis = range_basis(np.asarray([records[i]["range_m"]]), scale)
                else:
                    basis = range_bearing_basis(np.asarray([records[i]["range_m"]]),
                                                np.asarray([records[i]["bearing_rad"]]), scale)
                out[position] = predict_geometry_covariance(entry, basis)[0]
    return np.stack([spd(matrix) for matrix in out])


def static_mean(models: dict, name: str, records: list[dict], index: np.ndarray) -> np.ndarray:
    """Residual mean removed alongside each covariance model."""
    if name == "S0":
        base = np.asarray(models["S0"]["mean"])
        return np.tile(base, (len(index), 1))
    return np.stack([np.asarray(models["S1"]["mean"][records[i]["camera"]]) for i in index])


def covariance_scores(covariances: np.ndarray, residuals: np.ndarray,
                      drives: np.ndarray) -> dict:
    nll, d2, areas = [], [], []
    for matrix, residual in zip(covariances, residuals):
        inverse = np.linalg.inv(matrix)
        determinant = float(np.linalg.det(matrix))
        distance = float(residual @ inverse @ residual)
        d2.append(distance)
        nll.append(0.5 * (2 * math.log(2 * math.pi) + math.log(max(determinant, 1e-14)) + distance))
        areas.append(math.pi * CHI2_2["95"] * math.sqrt(max(determinant, 1e-14)))
    nll, d2, areas = np.asarray(nll), np.asarray(d2), np.asarray(areas)
    per_drive = [float(np.mean(nll[drives == value])) for value in sorted(set(drives.tolist()))]
    return {
        "mean_nll": float(np.mean(nll)),
        "equal_drive_mean_nll": float(np.mean(per_drive)),
        "median_drive_nll": float(np.median(per_drive)),
        "per_drive_nll": {value: float(np.mean(nll[drives == value]))
                          for value in sorted(set(drives.tolist()))},
        "containment": {key: float(np.mean(d2 <= value)) for key, value in CHI2_2.items()},
        "mean_95_ellipse_area_m2": float(np.mean(areas)),
        "median_95_ellipse_area_m2": float(np.median(areas)),
        "above_99_fraction": float(np.mean(d2 > CHI2_2["99"])),
        "n": int(len(nll)),
    }


def whiteness(records: list[dict], index: np.ndarray, covariances: np.ndarray,
              means: np.ndarray) -> dict:
    """Lag-1 and lag-5 autocorrelation of normalized residuals within (drive, camera)."""
    grouped: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for position, i in enumerate(index):
        factor = np.linalg.cholesky(covariances[position])
        normalized = np.linalg.solve(factor, records[i]["residual_ray"] - means[position])
        grouped.setdefault((records[i]["drive"], records[i]["camera"]), []).append(
            (records[i]["stamp_ns"], normalized))
    result = {}
    for lag in (1, 5):
        numerator, denominator = 0.0, 0.0
        for series in grouped.values():
            if len(series) <= lag + 2:
                continue
            series.sort(key=lambda item: item[0])
            values = np.stack([item[1] for item in series])
            values = values - values.mean(axis=0)
            numerator += float((values[:-lag] * values[lag:]).sum())
            denominator += float((values * values).sum())
        result[f"lag{lag}_acf"] = float(numerator / denominator) if denominator > 0 else math.nan
    normalized_all = np.concatenate([np.stack([item[1] for item in series])
                                     for series in grouped.values()])
    result["normalized_variance"] = normalized_all.var(axis=0).tolist()
    return result


# --------------------------------------------------------------------------------------
# Stage B: D1 dynamic camera-bias model, fitted by sequential predictive likelihood
# --------------------------------------------------------------------------------------

def d1_sequential_nll(records: list[dict], index: np.ndarray, tau: float,
                      fast: dict[str, np.ndarray], bias_sigma: dict[str, np.ndarray],
                      return_innovations: bool = False):
    """Run a per-camera Gauss-Markov bias filter over each drive and score innovations."""
    grouped: dict[tuple[str, str], list[int]] = {}
    for i in index:
        grouped.setdefault((records[i]["drive"], records[i]["camera"]), []).append(i)

    total, per_drive = [], {}
    innovations: list[tuple[int, str, str, np.ndarray, np.ndarray]] = []
    for (drive, camera), members in grouped.items():
        members.sort(key=lambda i: records[i]["stamp_ns"])
        stationary_covariance = bias_sigma[camera]
        bias = np.zeros(2)
        covariance = stationary_covariance.copy()
        previous_stamp = None
        for i in members:
            stamp = records[i]["stamp_ns"]
            if previous_stamp is not None:
                dt = (stamp - previous_stamp) / 1e9
                decay = math.exp(-dt / tau)
                bias = decay * bias
                covariance = (decay ** 2) * covariance + (1.0 - decay ** 2) * stationary_covariance
            previous_stamp = stamp
            innovation = records[i]["residual_ray"] - bias
            innovation_covariance = spd(covariance + fast[camera])
            inverse = np.linalg.inv(innovation_covariance)
            determinant = float(np.linalg.det(innovation_covariance))
            distance = float(innovation @ inverse @ innovation)
            value = 0.5 * (2 * math.log(2 * math.pi) + math.log(max(determinant, 1e-14)) + distance)
            total.append(value)
            per_drive.setdefault(drive, []).append(value)
            if return_innovations:
                innovations.append((stamp, drive, camera, innovation, innovation_covariance))
            gain = covariance @ inverse
            bias = bias + gain @ innovation
            covariance = spd(covariance - gain @ covariance)
    scores = {
        "mean_nll": float(np.mean(total)),
        "equal_drive_mean_nll": float(np.mean([np.mean(v) for v in per_drive.values()])),
        "per_drive_nll": {key: float(np.mean(value)) for key, value in per_drive.items()},
        "n": int(len(total)),
    }
    if return_innovations:
        return scores, innovations
    return scores


def fit_d1(records: list[dict], train: np.ndarray, tau_grid: list[float]) -> dict:
    """Split the marginal per-camera covariance into a persistent and a fast part."""
    camera = np.asarray([records[i]["camera"] for i in train])
    residual = np.stack([records[i]["residual_ray"] for i in train])

    marginal = {}
    for value in CAMERAS:
        mask = camera == value
        marginal[value] = sample_covariance(residual[mask]) if mask.sum() >= 5 \
            else sample_covariance(residual)

    best: dict[str, Any] = {}
    for tau in tau_grid:
        for share in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            fast = {key: spd((1.0 - share) * value) for key, value in marginal.items()}
            bias_sigma = {key: spd(share * value) for key, value in marginal.items()}
            score = d1_sequential_nll(records, train, tau, fast, bias_sigma)
            if not best or score["equal_drive_mean_nll"] < best["score"]["equal_drive_mean_nll"]:
                best = {"tau_s": tau, "bias_share": share, "score": score,
                        "fast": {k: v.tolist() for k, v in fast.items()},
                        "bias_sigma": {k: v.tolist() for k, v in bias_sigma.items()}}
    if not best:
        raise RuntimeError("D1 grid search produced no candidate")
    return best


# --------------------------------------------------------------------------------------
# Stage D: cross-camera dependence at synchronized rounds
# --------------------------------------------------------------------------------------

def cross_camera_dependence(records: list[dict], index: np.ndarray,
                            covariances: np.ndarray, means: np.ndarray) -> dict:
    """Correlation of simultaneous normalized residuals, in the world frame."""
    by_round: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    for position, i in enumerate(index):
        factor = np.linalg.cholesky(covariances[position])
        normalized = np.linalg.solve(factor, records[i]["residual_ray"] - means[position])
        by_round.setdefault((records[i]["drive"], records[i]["stamp_ns"]), []).append(
            (records[i]["camera"], normalized))

    pairs: dict[str, list[float]] = {}
    shared_fraction: list[float] = []
    round_sizes: list[int] = []
    for members in by_round.values():
        if len(members) < 2:
            continue
        round_sizes.append(len(members))
        stacked = np.stack([item[1] for item in members])
        # leading principal component share of simultaneous normalized residuals
        centered = stacked.reshape(len(members), -1)
        singular = np.linalg.svd(centered, compute_uv=False)
        shared_fraction.append(float(singular[0] ** 2 / max((singular ** 2).sum(), EPS)))
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                key = "|".join(sorted([members[a][0], members[b][0]]))
                pairs.setdefault(key, []).append(float(np.dot(members[a][1], members[b][1]) / 2.0))
    summary = {}
    for key, values in sorted(pairs.items()):
        values = np.asarray(values)
        if len(values) < 10:
            continue
        summary[key] = {
            "n": int(len(values)),
            "mean_correlation": float(values.mean()),
            "se": float(values.std(ddof=1) / math.sqrt(len(values))),
        }
    # The leading-component share must be compared with its independence null, which is far
    # above 0.5 for the two-to-four-camera rounds this network produces.  Simulate that null
    # at the observed round sizes rather than quoting a fixed reference.
    generator = np.random.default_rng(0)
    null_by_size: dict[int, float] = {}
    for size in sorted(set(round_sizes)):
        draws = []
        for _ in range(4000):
            sample = generator.standard_normal((size, 2))
            singular = np.linalg.svd(sample, compute_uv=False)
            draws.append(float(singular[0] ** 2 / max((singular ** 2).sum(), EPS)))
        null_by_size[size] = float(np.mean(draws))
    expected_null = float(np.mean([null_by_size[size] for size in round_sizes])) \
        if round_sizes else math.nan
    return {
        "pairwise": summary,
        "rounds_with_multiple_cameras": int(len(shared_fraction)),
        "mean_leading_component_share": float(np.mean(shared_fraction)) if shared_fraction else math.nan,
        "independence_null_leading_share": expected_null,
        "independence_null_by_round_size": null_by_size,
    }


# --------------------------------------------------------------------------------------
# Stage C: estimator replay on held-out drives
# --------------------------------------------------------------------------------------

def replay_drive(records: list[dict], rounds: list[dict], drive: str, arm: str,
                 static_models: dict, static_name: str, d1: dict | None,
                 cross_scale: float, scale: float,
                 process_noise: float, nis_threshold: float) -> dict:
    """Propagate a 2-D position belief along one drive and fuse admitted camera readings."""
    drive_rounds = [r for r in rounds if r["drive"] == drive]
    drive_rounds.sort(key=lambda item: item["stamp_ns"])
    by_stamp: dict[int, list[int]] = {}
    for i, record in enumerate(records):
        if record["drive"] == drive:
            by_stamp.setdefault(record["stamp_ns"], []).append(i)

    if not drive_rounds:
        raise RuntimeError(f"no rounds for {drive}")

    mean = drive_rounds[0]["reference"].copy()
    covariance = np.eye(2) * 0.25
    bias_state: dict[str, np.ndarray] = {c: np.zeros(2) for c in CAMERAS}
    bias_covariance: dict[str, np.ndarray] = {}
    if d1 is not None:
        bias_covariance = {c: np.asarray(d1["bias_sigma"][c]) for c in CAMERAS}

    previous_stamp = None
    errors, nees, accepted, rejected, nis_values = [], [], 0, 0, []
    for entry in drive_rounds:
        stamp = entry["stamp_ns"]
        if previous_stamp is not None:
            dt = (stamp - previous_stamp) / 1e9
            covariance = covariance + np.eye(2) * (process_noise ** 2) * dt
            speed = entry["reference_speed_mps"]
            heading = entry["reference_yaw"]
            mean = mean + np.asarray([math.cos(heading), math.sin(heading)]) * speed * dt
            if d1 is not None:
                for camera in CAMERAS:
                    decay = math.exp(-dt / d1["tau_s"])
                    bias_state[camera] = decay * bias_state[camera]
                    stationary = np.asarray(d1["bias_sigma"][camera])
                    bias_covariance[camera] = (decay ** 2) * bias_covariance[camera] + \
                        (1.0 - decay ** 2) * stationary
        previous_stamp = stamp

        members = by_stamp.get(stamp, [])
        if members:
            stacked_innovation, blocks, jacobians, camera_ids, bases = [], [], [], [], []
            for i in members:
                record = records[i]
                basis = record["basis"]
                index = np.asarray([i])
                if arm in ("E0", "E1"):
                    matrix_ray = predict_static(static_models, static_name, records, index, scale)[0]
                    mean_ray = static_mean(static_models, static_name, records, index)[0]
                    effective_ray = matrix_ray
                else:
                    camera = record["camera"]
                    mean_ray = np.zeros(2)
                    fast = np.asarray(d1["fast"][camera])
                    if arm == "E2":  # covariance-aware cascade
                        effective_ray = spd(fast + bias_covariance[camera])
                    else:  # E3 joint: fast noise only, bias carried in the state
                        effective_ray = fast
                predicted = record["corrected"] - basis @ mean_ray
                if arm in ("E2", "E3"):
                    predicted = predicted - basis @ bias_state[record["camera"]]
                stacked_innovation.append(predicted - mean)
                blocks.append(basis @ effective_ray @ basis.T)
                jacobians.append(np.eye(2))
                camera_ids.append(record["camera"])
                bases.append(basis)

            count = len(members)
            innovation = np.concatenate(stacked_innovation)
            jacobian = np.vstack(jacobians)
            noise = np.zeros((2 * count, 2 * count))
            for position, block in enumerate(blocks):
                noise[2 * position:2 * position + 2, 2 * position:2 * position + 2] = block
            if cross_scale > 0.0 and count > 1:
                # Conservative fusion for unknown cross-correlation.  Covariance
                # intersection divides each reading's weight by its mixing coefficient, so
                # simultaneous readings can never be counted as independent evidence.  With
                # equal weights this scales each block by the number of readings, which is
                # the standard guard against double-counting shared error.
                weight = 1.0 + cross_scale * (count - 1)
                for position in range(count):
                    noise[2 * position:2 * position + 2, 2 * position:2 * position + 2] *= weight
            noise = spd(noise)
            innovation_covariance = jacobian @ covariance @ jacobian.T + noise
            innovation_covariance = spd(innovation_covariance)
            inverse = np.linalg.inv(innovation_covariance)
            distance = float(innovation @ inverse @ innovation)
            nis_values.append(distance / (2 * count))
            if distance <= nis_threshold * 2 * count:
                gain = covariance @ jacobian.T @ inverse
                if arm == "E3":
                    for position, camera in enumerate(camera_ids):
                        local = bases[position].T @ innovation[2 * position:2 * position + 2]
                        weight = bias_covariance[camera] @ np.linalg.inv(
                            bias_covariance[camera] + np.asarray(d1["fast"][camera]) +
                            bases[position].T @ covariance @ bases[position])
                        bias_state[camera] = bias_state[camera] + weight @ local
                        bias_covariance[camera] = spd(
                            bias_covariance[camera] - weight @ bias_covariance[camera])
                mean = mean + gain @ innovation
                identity = np.eye(2) - gain @ jacobian
                covariance = spd(identity @ covariance @ identity.T + gain @ noise @ gain.T)
                accepted += 1
            else:
                rejected += 1

        error = mean - entry["reference"]
        errors.append(float(np.linalg.norm(error)))
        nees.append(float(error @ np.linalg.inv(spd(covariance)) @ error))

    errors, nees = np.asarray(errors), np.asarray(nees)
    return {
        "drive": drive,
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "median_error_m": float(np.median(errors)),
        "p95_error_m": float(np.percentile(errors, 95)),
        "mean_nees": float(np.mean(nees)),
        "median_nees": float(np.median(nees)),
        "coverage_95": float(np.mean(nees <= CHI2_2["95"])),
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "mean_normalized_nis": float(np.mean(nis_values)) if nis_values else math.nan,
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--bias-study", required=True, type=Path,
                        help="directory holding the frozen B3 prediction tables")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--process-noise", type=float, default=0.10,
                        help="odometry process noise, m per sqrt(s)")
    parser.add_argument("--nis-threshold", type=float, default=CHI2_2["99"] / 2.0,
                        help="per-degree-of-freedom NIS acceptance threshold")
    arguments = parser.parse_args()

    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    execution, camera_xy, drives = load_campaign(arguments.campaign_root)
    # The fit partition uses the leave-one-drive-out predictions so every fit residual is
    # out-of-fold; the development partition uses the fit-only model's predictions.  The
    # leave-one-route-out table is a separate diagnostic scheme and is never mixed in.
    prediction_files = [arguments.bias_study / "fit_leave_one_drive_out_predictions.csv",
                        arguments.bias_study / "development_predictions.csv"]
    missing = [p for p in prediction_files if not p.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen prediction tables: {missing}")
    corrections = load_frozen_corrections(prediction_files)
    records = load_measurements(drives, camera_xy, corrections)
    rounds = load_rounds(drives)

    partition = np.asarray([record["partition"] for record in records])
    drive_of = np.asarray([record["drive"] for record in records])
    fit_index = np.flatnonzero(partition == "fit")
    development_index = np.flatnonzero(partition == "development")
    scale = float(np.mean([record["range_m"] for record in records]))

    manifest = {
        "schema": "reference_controlled_r_model_inputs.v1",
        "campaign_root": str(arguments.campaign_root),
        "campaign_execution_sha256": sha256(arguments.campaign_root / "campaign_execution.json"),
        "audit_analysis_permitted": False,
        "frozen_mean_model": FROZEN_MEAN_MODEL,
        "bias_study": str(arguments.bias_study),
        "frozen_prediction_tables": {p.name: sha256(p) for p in prediction_files},
        "counts": {
            "fit_measurements": int(len(fit_index)),
            "development_measurements": int(len(development_index)),
            "fit_drives": int(len(set(drive_of[fit_index].tolist()))),
            "development_drives": int(len(set(drive_of[development_index].tolist()))),
            "rounds": int(len(rounds)),
        },
        "range_scale_m": scale,
        "process_noise_m_per_sqrt_s": arguments.process_noise,
        "nis_threshold_per_dof": arguments.nis_threshold,
    }
    atomic_json(output / "input_manifest.json", manifest)

    # ---- Stage A ---------------------------------------------------------------------
    static_models = fit_static_ladder(records, fit_index, scale)
    development_residual = np.stack([records[i]["residual_ray"] for i in development_index])
    development_drives = drive_of[development_index]
    stage_a = {}
    for name in ("S0", "S1", "S2a", "S2b"):
        covariances = predict_static(static_models, name, records, development_index, scale)
        means = static_mean(static_models, name, records, development_index)
        stage_a[name] = covariance_scores(covariances, development_residual - means,
                                          development_drives)
        stage_a[name]["whiteness"] = whiteness(records, development_index, covariances, means)

    # ---- Stage B ---------------------------------------------------------------------
    tau_grid = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0]
    d1 = fit_d1(records, fit_index, tau_grid)
    d1_development = d1_sequential_nll(
        records, development_index, d1["tau_s"],
        {k: np.asarray(v) for k, v in d1["fast"].items()},
        {k: np.asarray(v) for k, v in d1["bias_sigma"].items()})
    stage_b = {
        "selected_tau_s": d1["tau_s"],
        "selected_bias_share": d1["bias_share"],
        "fit_score": d1["score"],
        "development_score": d1_development,
        "static_reference_equal_drive_nll": {
            name: stage_a[name]["equal_drive_mean_nll"] for name in ("S0", "S1", "S2a", "S2b")},
    }

    # ---- Stage D ---------------------------------------------------------------------
    best_static = min(("S0", "S1", "S2a", "S2b"),
                      key=lambda name: stage_a[name]["equal_drive_mean_nll"])
    best_covariances = predict_static(static_models, best_static, records,
                                      development_index, scale)
    best_means = static_mean(static_models, best_static, records, development_index)
    stage_d = cross_camera_dependence(records, development_index, best_covariances, best_means)

    # ---- Stage C ---------------------------------------------------------------------
    cross_scale = 0.0
    correlations = [entry["mean_correlation"] for entry in stage_d["pairwise"].values()]
    measured_cross = float(np.mean(correlations)) if correlations else 0.0
    arms = {
        "E0": {"arm": "E0", "static": "S1", "d1": None, "cross": 0.0},
        "E1": {"arm": "E1", "static": best_static, "d1": None, "cross": 0.0},
        "E2": {"arm": "E2", "static": best_static, "d1": d1, "cross": 0.0},
        "E3": {"arm": "E3", "static": best_static, "d1": d1, "cross": 0.0},
        # E4 is the conservative-fusion arm the literature recommends when cross-camera
        # dependence is unknown.  It is scored to show what that conservatism costs here,
        # not because the measurement supports it.
        "E4": {"arm": "E1", "static": best_static, "d1": None, "cross": 1.0},
    }
    stage_c = {}
    held_out_drives = sorted(set(drive_of[development_index].tolist()))
    for label, configuration in arms.items():
        per_drive = [replay_drive(records, rounds, drive, configuration["arm"], static_models,
                                  configuration["static"], configuration["d1"],
                                  configuration["cross"], scale,
                                  arguments.process_noise, arguments.nis_threshold)
                     for drive in held_out_drives]
        stage_c[label] = {
            "configuration": {k: (v if k != "d1" else (None if v is None else "D1"))
                              for k, v in configuration.items()},
            "equal_drive_rmse_m": float(np.mean([d["rmse_m"] for d in per_drive])),
            "equal_drive_median_error_m": float(np.mean([d["median_error_m"] for d in per_drive])),
            "equal_drive_p95_error_m": float(np.mean([d["p95_error_m"] for d in per_drive])),
            "equal_drive_mean_nees": float(np.mean([d["mean_nees"] for d in per_drive])),
            "equal_drive_coverage_95": float(np.mean([d["coverage_95"] for d in per_drive])),
            "total_accepted": int(sum(d["accepted_updates"] for d in per_drive)),
            "total_rejected": int(sum(d["rejected_updates"] for d in per_drive)),
            "per_drive": per_drive,
        }

    report = {
        "schema": "reference_controlled_r_model_comparison.v1",
        "status": "complete_audit_sealed",
        "frozen_mean_model": FROZEN_MEAN_MODEL,
        "stage_a_static_ladder": stage_a,
        "stage_b_dynamic": stage_b,
        "stage_c_estimator_replay": stage_c,
        "stage_d_cross_camera": stage_d,
        "best_static_by_held_out_nll": best_static,
        "measured_mean_cross_camera_correlation": measured_cross,
    }
    atomic_json(output / "r_model_comparison.json", report)
    atomic_json(output / "static_models_fit_only.json", static_models)
    atomic_json(output / "d1_model_fit_only.json", d1)
    print(json.dumps({
        "stage_a_equal_drive_nll": {k: v["equal_drive_mean_nll"] for k, v in stage_a.items()},
        "stage_a_coverage_95": {k: v["containment"]["95"] for k, v in stage_a.items()},
        "stage_b": {"tau_s": d1["tau_s"], "bias_share": d1["bias_share"],
                    "development_equal_drive_nll": d1_development["equal_drive_mean_nll"]},
        "stage_c": {k: {"rmse_m": v["equal_drive_rmse_m"],
                        "mean_nees": v["equal_drive_mean_nees"],
                        "coverage_95": v["equal_drive_coverage_95"]}
                    for k, v in stage_c.items()},
        "stage_d_mean_cross_correlation": measured_cross,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
