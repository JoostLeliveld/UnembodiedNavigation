"""Does YOLO confidence predict conditional pixel-residual covariance?

This is an evaluation-only audit of the frozen warehouse_v2 commissioning capture.
It operates after the observation geometry and the frozen residual-offset correction:

    residual = [du_px, dv_px] - b(range)

The 20 positions used to fit ``b(range)`` are excluded.  Every reported covariance score
therefore uses the 3,151 held-out sightings only.  Models are fit in six spatial-block
folds and scored out of fold with Gaussian NLL and nominal 95% ellipse coverage.

The previous-method curve is the historical offline image-space precision blend
``1/var = q/2.5^2 + (1-q)/40^2``.  It is a reference, not a current endpoint claim.

Run:
    python3 experiments/measurement_commissioning/confidence_covariance.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "logs/studies/measurement_commissioning"
DATA = REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
READINGS = "detector_readings_halfopen_detect_20260825.csv"
OUT = STUDY / "confidence_covariance.json"
BIN_CSV = STUDY / "confidence_covariance_bins.csv"
SEED = 260826
N_BOOT = 2000
N_BINS = 10
OLD_R_VISIBLE_PX = 2.5
OLD_R_MISS_PX = 40.0
CHI2_95_2D = 5.991464547107979


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def offset(range_m: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(range_m)), range_m, range_m**2])
    return design @ coeff


def load() -> dict[str, np.ndarray]:
    frozen = json.loads((STUDY / "calibration.json").read_text())
    cal = frozen["calibration"]
    coeff = np.column_stack([cal["coefficients_du"], cal["coefficients_dv"]])
    fit_positions = {
        int(r["position_id"])
        for r in csv.DictReader((STUDY / "offset_positions.csv").open())
    }

    rows = list(csv.DictReader((STUDY / "sightings.csv").open()))
    rows = [r for r in rows if int(r["position_id"]) not in fit_positions]
    if len(rows) != int(cal["held_out_sightings"]):
        raise RuntimeError(
            f"expected {cal['held_out_sightings']} held-out sightings, found {len(rows)}"
        )
    position = np.array([int(r["position_id"]) for r in rows], dtype=int)
    camera = np.array([r["camera"] for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    q = np.array([float(r["confidence"]) for r in rows])
    range_m = np.array([float(r["range_m"]) for r in rows])
    raw = np.array([[float(r["du_px"]), float(r["dv_px"])] for r in rows])
    residual = raw - offset(range_m, coeff)
    return {
        "position": position,
        "camera": camera,
        "x": x,
        "y": y,
        "q": q,
        "range_m": range_m,
        "residual": residual,
    }


def quantile_edges(q: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(q, np.linspace(0.0, 1.0, n_bins + 1))
    # The detector writes four decimals, so adjacent quantiles can tie.  A strictly
    # increasing edge vector keeps every observation assigned deterministically.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bin_index(q: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges[1:-1], q, side="right"), 0, len(edges) - 2)


def cluster_bootstrap_stat(
    values: np.ndarray,
    clusters: np.ndarray,
    stat,
    *,
    draws: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    unique = np.unique(clusters)
    members = {g: np.flatnonzero(clusters == g) for g in unique}
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for b in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[g] for g in sampled])
        out[b] = stat(values[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def empirical_bins(data: dict[str, np.ndarray]) -> list[dict]:
    q, r, pos = data["q"], data["residual"], data["position"]
    edges = quantile_edges(q, N_BINS)
    bins = bin_index(q, edges)
    out = []
    for b in range(N_BINS):
        m = bins == b
        rb, pb = r[m], pos[m]
        pooled = math.sqrt(float(np.mean(np.sum(rb**2, axis=1))) / 2.0)
        lo, hi = cluster_bootstrap_stat(
            rb,
            pb,
            lambda z: math.sqrt(float(np.mean(np.sum(z**2, axis=1))) / 2.0),
            seed=SEED + b,
        )
        out.append({
            "bin": b + 1,
            "n": int(m.sum()),
            "n_positions": int(len(np.unique(pb))),
            "q_min": float(np.min(q[m])),
            "q_max": float(np.max(q[m])),
            "q_mean": float(np.mean(q[m])),
            "sigma_pooled_px": pooled,
            "sigma_pooled_ci95_px": [lo, hi],
            "sigma_u_px": float(math.sqrt(np.mean(rb[:, 0] ** 2))),
            "sigma_v_px": float(math.sqrt(np.mean(rb[:, 1] ** 2))),
            "old_precision_blend_sigma_px": float(old_sigma(np.mean(q[m]))),
        })
    return out


def old_sigma(q: np.ndarray | float) -> np.ndarray | float:
    q = np.clip(q, 0.0, 1.0)
    precision = q / OLD_R_VISIBLE_PX**2 + (1.0 - q) / OLD_R_MISS_PX**2
    return np.sqrt(1.0 / precision)


def spatial_folds(data: dict[str, np.ndarray]) -> np.ndarray:
    # Six contiguous-ish blocks: three x bands by two y bands.  Edges are based on
    # unique floor positions rather than duplicated camera/heading observations.
    positions = {}
    for p, x, y in zip(data["position"], data["x"], data["y"]):
        positions[int(p)] = (float(x), float(y))
    xy = np.array(list(positions.values()))
    x_edges = np.quantile(xy[:, 0], [1 / 3, 2 / 3])
    y_edge = float(np.median(xy[:, 1]))
    lookup = {
        p: int(np.searchsorted(x_edges, x, side="right") + 3 * (y > y_edge))
        for p, (x, y) in positions.items()
    }
    folds = np.array([lookup[int(p)] for p in data["position"]], dtype=int)
    if set(folds.tolist()) != set(range(6)):
        raise RuntimeError(f"spatial split did not produce six folds: {sorted(set(folds))}")
    return folds


def _fit_loglinear(
    q: np.ndarray,
    energy: np.ndarray,
    camera: np.ndarray | None = None,
) -> dict:
    q_mean = float(np.mean(q))
    q_scale = float(np.std(q))
    z = (q - q_mean) / max(q_scale, 1e-9)

    if camera is None:
        groups = np.zeros(len(q), dtype=int)
        names = ["pooled"]
    else:
        names = sorted(np.unique(camera).tolist())
        index = {name: i for i, name in enumerate(names)}
        groups = np.array([index[c] for c in camera], dtype=int)

    def intercepts(slope: float) -> np.ndarray:
        ans = []
        for g in range(len(names)):
            m = groups == g
            ans.append(math.log(max(float(np.mean(energy[m] * np.exp(-slope * z[m]))), 1e-12)))
        return np.array(ans)

    def objective(slope: float) -> float:
        a = intercepts(slope)
        variance = np.exp(a[groups] + slope * z)
        return float(np.mean(energy / variance + np.log(variance)))

    opt = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    slope = float(opt.x)
    return {
        "q_mean": q_mean,
        "q_scale": q_scale,
        "slope_per_q_sd": slope,
        "intercepts": {name: float(a) for name, a in zip(names, intercepts(slope))},
    }


def _predict_loglinear(model: dict, q: np.ndarray, camera: np.ndarray | None = None) -> np.ndarray:
    z = (q - model["q_mean"]) / max(model["q_scale"], 1e-9)
    intercepts = model["intercepts"]
    if camera is None:
        a = np.full(len(q), intercepts["pooled"])
    else:
        fallback = float(np.mean(list(intercepts.values())))
        a = np.array([intercepts.get(c, fallback) for c in camera])
    return np.exp(a + model["slope_per_q_sd"] * z)


def _fit_camera_bin_additive(
    energy: np.ndarray,
    camera: np.ndarray,
    q_bin: np.ndarray,
    camera_names: list[str],
    n_bins: int,
) -> dict:
    """Fit log variance = global + camera effect + confidence-bin effect.

    The first camera and first confidence bin are references.  A tiny ridge on the
    non-global effects stabilizes sparse camera/bin combinations without driving the fit.
    """
    cam_index = {name: i for i, name in enumerate(camera_names)}
    ci = np.array([cam_index[c] for c in camera], dtype=int)
    n_cam_effects = len(camera_names) - 1
    n_bin_effects = n_bins - 1
    ridge = 1e-3

    def unpack(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        global_ = float(theta[0])
        cam_effect = np.r_[0.0, theta[1:1 + n_cam_effects]]
        bin_effect = np.r_[0.0, theta[1 + n_cam_effects:]]
        return global_, cam_effect, bin_effect

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        global_, cam_effect, bin_effect = unpack(theta)
        eta = global_ + cam_effect[ci] + bin_effect[q_bin]
        ratio = energy * np.exp(-eta)
        loss = float(np.mean(ratio + eta) + ridge * np.sum(theta[1:] ** 2))
        base_grad = (1.0 - ratio) / len(energy)
        grad = np.zeros_like(theta)
        grad[0] = np.sum(base_grad)
        for c in range(1, len(camera_names)):
            grad[c] = np.sum(base_grad[ci == c])
        start = 1 + n_cam_effects
        for b in range(1, n_bins):
            grad[start + b - 1] = np.sum(base_grad[q_bin == b])
        grad[1:] += 2.0 * ridge * theta[1:]
        return loss, grad

    initial = np.zeros(1 + n_cam_effects + n_bin_effects)
    initial[0] = math.log(max(float(np.mean(energy)), 1e-12))
    opt = minimize(lambda z: objective(z), initial, method="L-BFGS-B", jac=True)
    if not opt.success:
        raise RuntimeError(f"camera+confidence additive fit failed: {opt.message}")
    global_, cam_effect, bin_effect = unpack(opt.x)
    return {
        "global_log_variance": global_,
        "camera_effect": {name: float(v) for name, v in zip(camera_names, cam_effect)},
        "confidence_bin_effect": [float(v) for v in bin_effect],
    }


def _predict_camera_bin_additive(
    model: dict,
    camera: np.ndarray,
    q_bin: np.ndarray,
) -> np.ndarray:
    cam_effect = model["camera_effect"]
    fallback = float(np.mean(list(cam_effect.values())))
    ce = np.array([cam_effect.get(c, fallback) for c in camera])
    be = np.asarray(model["confidence_bin_effect"])[q_bin]
    return np.exp(float(model["global_log_variance"]) + ce + be)


def score_models(data: dict[str, np.ndarray]) -> tuple[dict, dict]:
    q, r, camera, pos = data["q"], data["residual"], data["camera"], data["position"]
    energy = np.sum(r**2, axis=1) / 2.0
    folds = spatial_folds(data)
    names = ["constant", "camera_only", "confidence_binned",
             "camera_plus_confidence_binned", "previous_offline_mapping"]
    predicted = {name: np.full(len(q), np.nan) for name in names}
    fit_details = []

    for fold in range(6):
        train, test = folds != fold, folds == fold
        predicted["constant"][test] = float(np.mean(energy[train]))

        cam_var = {
            c: float(np.mean(energy[train & (camera == c)]))
            for c in np.unique(camera[train])
        }
        fallback = float(np.mean(energy[train]))
        predicted["camera_only"][test] = np.array([cam_var.get(c, fallback) for c in camera[test]])

        edges = quantile_edges(q[train], 8)
        tr_bin = bin_index(q[train], edges)
        bin_var = np.array([
            float(np.mean(energy[train][tr_bin == b])) if np.any(tr_bin == b) else fallback
            for b in range(8)
        ])
        predicted["confidence_binned"][test] = bin_var[bin_index(q[test], edges)]

        camera_names = sorted(np.unique(camera[train]).tolist())
        model = _fit_camera_bin_additive(
            energy[train], camera[train], tr_bin, camera_names, 8
        )
        predicted["camera_plus_confidence_binned"][test] = _predict_camera_bin_additive(
            model, camera[test], bin_index(q[test], edges)
        )
        predicted["previous_offline_mapping"][test] = np.asarray(old_sigma(q[test])) ** 2
        fit_details.append({
            "fold": fold,
            "n_fit": int(train.sum()),
            "n_test": int(test.sum()),
            "camera_plus_confidence_binned": model,
        })

    metrics = {}
    nll_by_model = {}
    for name, variance in predicted.items():
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            raise RuntimeError(f"invalid OOF variance for {name}")
        nis = np.sum(r**2, axis=1) / variance
        nll = 0.5 * (nis + 2.0 * np.log(variance) + 2.0 * math.log(2.0 * math.pi))
        nll_by_model[name] = nll
        metrics[name] = {
            "mean_gaussian_nll_px": float(np.mean(nll)),
            "coverage_95": float(np.mean(nis <= CHI2_95_2D)),
            "mean_normalized_squared_error": float(np.mean(nis)),
        }

    base = nll_by_model["constant"]
    for i, name in enumerate(names):
        diff = nll_by_model[name] - base
        lo, hi = cluster_bootstrap_stat(diff, pos, np.mean, seed=SEED + 100 + i)
        metrics[name]["delta_nll_vs_constant"] = float(np.mean(diff))
        metrics[name]["delta_nll_vs_constant_ci95"] = [lo, hi]

    diff = nll_by_model["camera_plus_confidence_binned"] - nll_by_model["camera_only"]
    lo, hi = cluster_bootstrap_stat(diff, pos, np.mean, seed=SEED + 200)
    comparison = {
        "camera_plus_confidence_binned_minus_camera_only_nll": float(np.mean(diff)),
        "ci95": [lo, hi],
    }
    return {"metrics": metrics, "fit_details": fit_details, "fold": folds.tolist()}, comparison


def spearman_cluster_ci(data: dict[str, np.ndarray]) -> dict:
    q = data["q"]
    energy = np.sum(data["residual"] ** 2, axis=1) / 2.0
    rho = float(spearmanr(q, energy).statistic)
    pairs = np.column_stack([q, energy])
    lo, hi = cluster_bootstrap_stat(
        pairs,
        data["position"],
        lambda z: float(spearmanr(z[:, 0], z[:, 1]).statistic),
        seed=SEED + 300,
    )
    return {"rho_confidence_vs_residual_energy": rho, "cluster_bootstrap_ci95": [lo, hi]}


def admission_bins() -> list[dict]:
    confidence = {}
    for path in sorted(DATA.glob(f"camera_*/{READINGS}")):
        for row in csv.DictReader(path.open()):
            if row["detected"] == "1":
                confidence[(path.parent.name, row["image"])] = float(row["confidence"])

    rows = []
    for row in csv.DictReader((STUDY / "availability.csv").open()):
        if not row["image"]:
            continue
        key = (row["camera"], row["image"])
        if key not in confidence:
            continue
        rows.append((confidence[key], int(row["usable"]), int(row["position_id"])))
    q = np.array([r[0] for r in rows])
    usable = np.array([r[1] for r in rows], dtype=float)
    position = np.array([r[2] for r in rows], dtype=int)
    edges = quantile_edges(q, N_BINS)
    idx = bin_index(q, edges)
    out = []
    for b in range(N_BINS):
        m = idx == b
        values = usable[m]
        lo, hi = cluster_bootstrap_stat(values, position[m], np.mean, seed=SEED + 400 + b)
        out.append({
            "bin": b + 1,
            "n_detected": int(m.sum()),
            "n_positions": int(len(np.unique(position[m]))),
            "q_mean": float(np.mean(q[m])),
            "admission_rate": float(np.mean(values)),
            "admission_rate_ci95": [lo, hi],
        })
    return out


def verdict(models: dict, comparison: dict) -> str:
    q_delta = models["confidence_binned"]["delta_nll_vs_constant"]
    q_ci = models["confidence_binned"]["delta_nll_vs_constant_ci95"]
    add_delta = comparison["camera_plus_confidence_binned_minus_camera_only_nll"]
    add_ci = comparison["ci95"]
    q_wins = q_delta < -0.01 and q_ci[1] < 0.0
    adds = add_delta < -0.01 and add_ci[1] < 0.0
    if q_wins and adds:
        return "confidence_predicts_variance_and_adds_beyond_camera"
    if q_wins and not adds:
        return "confidence_predicts_some_pooled_variance_but_not_beyond_camera"
    if not q_wins and adds:
        return "confidence_adds_only_after_camera_conditioning"
    return "confidence_does_not_materially_improve_conditional_pixel_covariance"


def main() -> int:
    data = load()
    bins = empirical_bins(data)
    cv, camera_comparison = score_models(data)
    association = spearman_cluster_ci(data)
    admission = admission_bins()
    result = {
        "schema_version": 1,
        "analysis_id": "WHV2-CONFIDENCE-RCOND-DIAGNOSTIC",
        "status": "diagnostic_pending_registry",
        "metric_object": "camera_measurement_conditional_pixel_covariance",
        "metric_name": "post_geometry_pixel_residual_variance",
        "reference": "commanded_set_pose_ground_truth_used_only_to_evaluate_h_c",
        "projection_runtime": "warehouse_v2_observation_model_half_open_bbox_current_capture",
        "dataset_or_campaign": "warehouse_v2_yolo_shared_20260822",
        "detector": "warehouse_v2_yolo_detect_halfopen_20260825_r1",
        "experimental_unit": "usable_sighting_cluster_bootstrap_by_floor_position",
        "n_sightings": int(len(data["q"])),
        "n_positions": int(len(np.unique(data["position"]))),
        "n_cameras": int(len(np.unique(data["camera"]))),
        "inclusion": "usable sightings not used to fit the frozen residual-offset model",
        "online_inputs": ["detector_confidence", "camera_id", "camera_geometry", "robot_pose_estimate_at_runtime"],
        "evaluation_only_inputs": ["commanded_set_pose_truth", "pixel_residual", "position_id"],
        "confidence_distribution": {
            "min": float(np.min(data["q"])),
            "p05": float(np.percentile(data["q"], 5)),
            "median": float(np.median(data["q"])),
            "p95": float(np.percentile(data["q"], 95)),
            "max": float(np.max(data["q"])),
        },
        "residual_definition": "[du_px,dv_px] minus frozen b(range); observation model h_c already accounts for geometry and bbox convention",
        "binning": {"method": "10 equal-count confidence bins", "bootstrap": f"{N_BOOT} floor-position cluster draws", "bins": bins},
        "association": association,
        "held_out_design": "six spatial blocks (three x bands by two y bands); every covariance prediction scored out of fold",
        "models": cv["metrics"],
        "camera_incremental_comparison": camera_comparison,
        "previous_mapping": {
            "status": "historical_offline_reference_only",
            "formula": "1/var = q/2.5^2 + (1-q)/40^2",
            "r_visible_px": OLD_R_VISIBLE_PX,
            "r_miss_px": OLD_R_MISS_PX,
            "caveat": "40 px was the previous offline endpoint; the repository documents an unreconciled 40 vs 120 px runtime mismatch",
        },
        "admission_given_detection": admission,
        "verdict": verdict(cv["metrics"], camera_comparison),
        "limits": [
            "one detector checkpoint, one simulated stock state",
            "admission already conditions on agreement with the predicted box",
            "six robot headings per floor position",
            "metric registry tracked at HEAD predates this warehouse_v2 capture; result remains diagnostic until registered",
        ],
        "input_sha256": {
            "sightings.csv": sha256(STUDY / "sightings.csv"),
            "availability.csv": sha256(STUDY / "availability.csv"),
            "calibration.json": sha256(STUDY / "calibration.json"),
            "offset_positions.csv": sha256(STUDY / "offset_positions.csv"),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with BIN_CSV.open("w", newline="") as handle:
        fields = ["bin", "n", "n_positions", "q_min", "q_max", "q_mean",
                  "sigma_pooled_px", "sigma_ci95_lo_px", "sigma_ci95_hi_px",
                  "sigma_u_px", "sigma_v_px", "old_precision_blend_sigma_px"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in bins:
            writer.writerow({
                **{k: row[k] for k in fields if k in row},
                "sigma_ci95_lo_px": row["sigma_pooled_ci95_px"][0],
                "sigma_ci95_hi_px": row["sigma_pooled_ci95_px"][1],
            })
    print(json.dumps({
        "n_sightings": result["n_sightings"],
        "n_positions": result["n_positions"],
        "association": association,
        "models": cv["metrics"],
        "camera_incremental_comparison": camera_comparison,
        "verdict": result["verdict"],
        "output": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
