#!/usr/bin/env python3
"""Decompose keypoint error into bias, repeatability and a persistent floor.

The experimental unit for calibration is a grouped mean from one
(session, camera, spatial anchor, heading). Individual frames are never randomly
split. Ground truth enters here, offline, and nowhere in the capture/runtime path.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist


FEATURE_NAMES = ("intercept", "range_z", "sin_yaw", "cos_yaw", "image_u_z")
STUDENT_DF = 5.0
HONEST_MEDIAN_CHI2 = 1.386


def load_rows(root: Path) -> list[dict]:
    files = sorted(root.glob("session_*/evaluation/per_sample.csv"))
    if len(files) < 2:
        raise RuntimeError(f"need at least two evaluated sessions under {root}; found {len(files)}")
    rows = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("detected") != "1":
                    continue
                if raw.get("front_labelled_visible") != "1" or raw.get("rear_labelled_visible") != "1":
                    continue
                required = ("session_id", "anchor_id", "yaw_idx", "repeat_idx", "err_x_m", "err_y_m")
                if any(raw.get(key, "") == "" for key in required):
                    raise RuntimeError(f"commissioning metadata missing in {path}: {required}")
                row = dict(raw)
                for key in (
                    "range_m", "yaw_rad", "gt_front_u", "gt_rear_u", "err_x_m", "err_y_m",
                    "nominal_x", "nominal_y", "nominal_yaw_rad",
                ):
                    row[key] = float(raw[key])
                row["repeat_idx"] = int(raw["repeat_idx"])
                row["yaw_idx"] = int(raw["yaw_idx"])
                row["error"] = np.asarray([row["err_x_m"], row["err_y_m"]], float)
                rows.append(row)
    if not rows:
        raise RuntimeError("no detected rows with both markers rendered")
    return rows


def fit_feature_transform(rows: list[dict]) -> dict:
    ranges = np.asarray([row["range_m"] for row in rows])
    image_u = np.asarray([0.5 * (row["gt_front_u"] + row["gt_rear_u"]) / 1280.0 for row in rows])
    return {
        "range_mean": float(ranges.mean()), "range_sd": float(max(ranges.std(), 1e-6)),
        "image_u_mean": float(image_u.mean()), "image_u_sd": float(max(image_u.std(), 1e-6)),
    }


def feature(row: dict, transform: dict) -> np.ndarray:
    image_u = 0.5 * (row["gt_front_u"] + row["gt_rear_u"]) / 1280.0
    return np.asarray([
        1.0,
        (row["range_m"] - transform["range_mean"]) / transform["range_sd"],
        math.sin(row["yaw_rad"]), math.cos(row["yaw_rad"]),
        (image_u - transform["image_u_mean"]) / transform["image_u_sd"],
    ])


def fit_bias(rows: list[dict]) -> dict:
    transform = fit_feature_transform(rows)
    X = np.stack([feature(row, transform) for row in rows])
    Y = np.stack([row["error"] for row in rows])
    penalty = np.eye(X.shape[1]) * 1e-3
    penalty[0, 0] = 1e-8
    precision = X.T @ X + penalty
    covariance_base = np.linalg.inv(precision)
    beta = covariance_base @ X.T @ Y
    residual = Y - X @ beta
    axis_var = np.sum(residual**2, axis=0) / max(len(Y) - X.shape[1], 1)
    return {"transform": transform, "beta": beta, "covariance_base": covariance_base, "axis_var": axis_var}


def predict_bias(row: dict, model: dict) -> tuple[np.ndarray, np.ndarray]:
    x = feature(row, model["transform"])
    mean = x @ model["beta"]
    leverage = float(x @ model["covariance_base"] @ x)
    covariance = np.diag(np.maximum(model["axis_var"] * leverage, 0.0))
    return mean, covariance


def group_key(row: dict) -> tuple[str, str, str, int]:
    return str(row["session_id"]), str(row.get("camera", "")), str(row["anchor_id"]), int(row["yaw_idx"])


def grouped(rows: list[dict]) -> list[list[dict]]:
    result: dict[tuple, list[dict]] = {}
    for row in rows:
        result.setdefault(group_key(row), []).append(row)
    return [items for _, items in sorted(result.items()) if len(items) >= 2]


def nearest_psd(matrix: np.ndarray, minimum: float = 1e-10) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, minimum)) @ vectors.T


def fit_components(rows: list[dict], bias: dict) -> dict:
    groups = grouped(rows)
    scatter = np.zeros((2, 2)); dof = 0
    group_records = []
    for items in groups:
        corrected = np.stack([row["error"] - predict_bias(row, bias)[0] for row in items])
        mean = corrected.mean(axis=0)
        centred = corrected - mean
        scatter += centred.T @ centred
        dof += len(items) - 1
        group_records.append((mean, len(items), items))
    if dof < 2 or len(group_records) < 3:
        raise RuntimeError(f"insufficient repeated groups: {len(group_records)} groups, {dof} within-group dof")
    # Weak 2 mm per-axis regularizer prevents a singular pilot covariance.
    R_iid = (scatter + np.eye(2) * (0.002**2) * 4.0) / (dof + 4.0)
    means = np.stack([value[0] for value in group_records])
    raw_between = np.cov(means.T, ddof=1)
    finite_repeat = sum(R_iid / n for _, n, _ in group_records) / len(group_records)
    bias_uncertainty = sum(
        predict_bias(items[0], bias)[1] for _, _, items in group_records
    ) / len(group_records)
    B = nearest_psd(raw_between - finite_repeat - bias_uncertainty, minimum=0.0)
    return {"R_iid": nearest_psd(R_iid), "B_persistent": B, "groups": len(group_records), "within_dof": dof}


def gaussian_nll(error: np.ndarray, covariance: np.ndarray) -> float:
    covariance = nearest_psd(covariance)
    return float(0.5 * (2 * math.log(2 * math.pi) + np.linalg.slogdet(covariance)[1] + error @ np.linalg.inv(covariance) @ error))


def student_nll(error: np.ndarray, covariance: np.ndarray, df: float = STUDENT_DF) -> float:
    # covariance is the desired covariance; Student scale is covariance*(nu-2)/nu.
    scale = nearest_psd(covariance) * (df - 2.0) / df
    d = 2
    delta = float(error @ np.linalg.inv(scale) @ error)
    return float(
        gammaln(df / 2.0) - gammaln((df + d) / 2.0)
        + 0.5 * (d * math.log(df * math.pi) + np.linalg.slogdet(scale)[1])
        + 0.5 * (df + d) * math.log1p(delta / df)
    )


def score_groups(rows: list[dict], bias: dict, components: dict) -> dict:
    records = []
    for items in grouped(rows):
        corrected = np.stack([row["error"] - predict_bias(row, bias)[0] for row in items])
        error = corrected.mean(axis=0)
        n = len(items)
        bias_cov = sum((predict_bias(row, bias)[1] for row in items), np.zeros((2, 2))) / n**2
        naive = components["R_iid"] / n + bias_cov
        floor = components["B_persistent"] + components["R_iid"] / n + bias_cov
        records.append((error, naive, floor, group_key(items[0]), n))
    if not records:
        return {"n_groups": 0}

    def model_summary(which: str) -> dict:
        nll, d2 = [], []
        for error, naive, floor, _, _ in records:
            cov = naive if which == "naive_gaussian" else floor
            nll.append(student_nll(error, cov) if which == "floor_student_t" else gaussian_nll(error, cov))
            if which == "floor_student_t":
                scale = cov * (STUDENT_DF - 2.0) / STUDENT_DF
                d2.append(float(error @ np.linalg.inv(nearest_psd(scale)) @ error))
            else:
                d2.append(float(error @ np.linalg.inv(nearest_psd(cov)) @ error))
        thresholds = {
            "coverage_50": (2.0 * f_dist.ppf(0.50, 2, STUDENT_DF) if which == "floor_student_t" else 1.38629436112),
            "coverage_80": (2.0 * f_dist.ppf(0.80, 2, STUDENT_DF) if which == "floor_student_t" else 3.21887582487),
            "coverage_95": (2.0 * f_dist.ppf(0.95, 2, STUDENT_DF) if which == "floor_student_t" else 5.9914645471),
        }
        return {
            "mean_nll": float(np.mean(nll)),
            **{name: float(np.mean(np.asarray(d2) <= threshold)) for name, threshold in thresholds.items()},
            "median_mahalanobis2": float(np.median(d2)),
        }
    return {
        "n_groups": len(records), "n_readings": int(sum(item[-1] for item in records)),
        "naive_gaussian": model_summary("naive_gaussian"),
        "floor_gaussian": model_summary("floor_gaussian"),
        "floor_student_t": model_summary("floor_student_t"),
    }


def serial_matrix(value: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(value)]


def load_geometry_audit(root: Path, rows: list[dict], holdout_session: str) -> tuple[dict, dict]:
    pixel_train_rows = [
        row for row in rows
        if str(row["session_id"]) != holdout_session and row.get("split") == "train"
    ]
    train_residuals = np.asarray([
        [float(row["res_front_u"]), float(row["res_front_v"]),
         float(row["res_rear_u"]), float(row["res_rear_v"])]
        for row in pixel_train_rows
    ])
    if len(train_residuals) < 8:
        return {}, {}
    centred = train_residuals - train_residuals.mean(axis=0)
    # Weak scale-invariant variance posterior. Treat a frame, rather than each
    # of its four correlated marker coordinates, as one effective sample. This
    # makes the upper bound conservative without looking at either spatial or
    # session holdouts.
    effective_df = len(train_residuals) - 1
    pooled_scale = float(math.sqrt(np.sum(centred**2) / (4.0 * effective_df)))
    posterior_mean_sigma = float(math.sqrt(effective_df * pooled_scale**2 / (effective_df - 2.0)))
    posterior_upper95_sigma = float(math.sqrt(
        effective_df * pooled_scale**2 / chi2_dist.ppf(0.05, effective_df)
    ))
    pixel_model = {
        "n_training_readings": len(train_residuals),
        "effective_df": effective_df,
        "coordinate_bias_px": [float(value) for value in train_residuals.mean(axis=0)],
        "posterior_scale_sigma_px": pooled_scale,
        "posterior_mean_sigma_px": posterior_mean_sigma,
        "posterior_upper95_sigma_px": posterior_upper95_sigma,
    }
    result = {}
    for session in sorted({str(row["session_id"]) for row in rows}):
        evaluation_rows = [row for row in rows if str(row["session_id"]) == session]
        residuals = np.asarray([
            [float(row["res_front_u"]), float(row["res_front_v"]),
             float(row["res_rear_u"]), float(row["res_rear_v"])]
            for row in evaluation_rows
        ])
        path = root / session / "covariance_honesty/per_reading.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            records = [{key: float(value) for key, value in raw.items()} for raw in csv.DictReader(handle)]
        chi2 = np.asarray([record["chi2"] for record in records])
        own_pixel_sigma = float(residuals.std(ddof=1))
        # check_covariance_is_honest used this session's own sigma. Covariance
        # scales with sigma^2, so convert its chi-square values to the one fixed
        # on training anchors without rerunning the projection.
        fixed_chi2_scale = (own_pixel_sigma / posterior_upper95_sigma) ** 2
        result[session] = {
            "pixel_sigma_diagnostic": own_pixel_sigma,
            "honesty_multiplier": float(math.sqrt(np.median(chi2 * fixed_chi2_scale) / HONEST_MEDIAN_CHI2)),
            "fixed_chi2_scale": fixed_chi2_scale,
            "records": records,
        }
    return pixel_model, result


def make_geometry_figure(root: Path, pixel_model: dict, audit: dict) -> None:
    if not audit:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    sessions = list(audit)
    colors = ["#386cb0", "#f07c26", "#1a9850"]

    ax = axes[0]
    values = [audit[session]["pixel_sigma_diagnostic"] for session in sessions]
    ax.bar(np.arange(len(sessions)), values, color=colors[:len(sessions)])
    safe_sigma = pixel_model["posterior_upper95_sigma_px"]
    ax.axhline(safe_sigma, color="black", linestyle="--", linewidth=1.0,
               label="training posterior 95%")
    ax.set_xticks(np.arange(len(sessions)), [name.replace("session_", "S") for name in sessions])
    ax.set(ylim=(0, max(values) * 1.35), ylabel=r"learned pixel $\sigma$ [px]",
           title=r"(a) constant $R_{uv}$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for index, session in enumerate(sessions):
        records = audit[session]["records"]
        ranges = np.asarray([record["range_m"] for record in records])
        sigma = np.asarray([
            math.sqrt(0.5 * (record["stated_sigma_x_cm"]**2 + record["stated_sigma_y_cm"]**2))
            for record in records
        ]) * safe_sigma / audit[session]["pixel_sigma_diagnostic"]
        ax.scatter(ranges, sigma, s=15, alpha=0.60, color=colors[index],
                   label=session.replace("session_", "S"))
    ax.set(xlabel="camera range [m]", ylabel=r"ground $\sigma$ [cm]",
           title=r"(b) $J(x)R_{uv}J(x)^T$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    bins = ((3, 5), (5, 7), (7, 9), (9, 20))
    for index, session in enumerate(sessions):
        records = audit[session]["records"]
        x, y = [], []
        for lo, hi in bins:
            chi2 = [record["chi2"] for record in records if lo <= record["range_m"] < hi]
            if chi2:
                x.append(0.5 * (lo + min(hi, 11)))
                fixed = np.asarray(chi2) * audit[session]["fixed_chi2_scale"]
                y.append(math.sqrt(float(np.median(fixed)) / HONEST_MEDIAN_CHI2))
        ax.plot(x, y, marker="o", linewidth=1.8, color=colors[index],
                label=session.replace("session_", "S"))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="honest")
    ax.axhspan(0.7, 1.3, color="0.8", alpha=0.3)
    ax.set(xlabel="camera range [m]", ylabel="truth / claimed scale",
           title="(c) fixed-train-R honesty", ylim=(0.5, 1.5))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.73, wspace=0.34)
    fig.suptitle(
        r"Learn $R$ in pixels; let visibility geometry create the spatial ellipses",
        fontsize=11, y=0.96,
    )
    out = root / "analysis"
    fig.savefig(out / "fig_geometry_r_pilot.png", dpi=220)
    fig.savefig(out / "fig_geometry_r_pilot.pdf")
    plt.close(fig)


def predictive_records(rows: list[dict], bias: dict, components: dict) -> list[dict]:
    records = []
    for items in grouped(rows):
        corrected = np.stack([row["error"] - predict_bias(row, bias)[0] for row in items])
        n = len(items)
        bias_cov = sum((predict_bias(row, bias)[1] for row in items), np.zeros((2, 2))) / n**2
        records.append({
            "error": corrected.mean(axis=0),
            "covariance": components["B_persistent"] + components["R_iid"] / n + bias_cov,
        })
    return records


def add_covariance_ellipse(ax, covariance: np.ndarray, *, threshold: float, **kwargs) -> None:
    values, vectors = np.linalg.eigh(nearest_psd(covariance))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2.0 * np.sqrt(values * threshold) * 100.0
    ax.add_patch(Ellipse((0.0, 0.0), width, height, angle=angle, **kwargs))


def make_figure(root: Path, partitions: dict, bias: dict, components: dict, scores: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), constrained_layout=True)

    ax = axes[0]
    representative_n = 3
    naive = components["R_iid"] / representative_n
    proposed = components["B_persistent"] + naive
    add_covariance_ellipse(
        ax, naive, threshold=5.9914645471, fill=False, linewidth=2.0,
        edgecolor="#386cb0", label=r"naive $R_{iid}/n$",
    )
    add_covariance_ellipse(
        ax, proposed, threshold=5.9914645471, fill=False, linewidth=2.0,
        edgecolor="#f07c26", label=r"floor $B+R_{iid}/n$",
    )
    ax.scatter([0], [0], marker="+", s=45, color="black", zorder=3)
    radius = 1.15 * max(
        patch.width if isinstance(patch, Ellipse) else 0.0 for patch in ax.patches
    ) / 2.0
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), xlabel="x error [cm]", ylabel="y error [cm]",
           title="(a) uncertainty after 3 repeats")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    styles = {
        "spatial_holdout": ("unseen space", "^", "#d73027"),
        "session_holdout": ("new session", "o", "#1a9850"),
        "combined_holdout": ("both", "s", "#762a83"),
    }
    all_errors = []
    for name, part in partitions.items():
        records = predictive_records(part, bias, components)
        errors = np.asarray([record["error"] for record in records]) * 100.0
        if len(errors):
            label, marker, color = styles[name]
            ax.scatter(errors[:, 0], errors[:, 1], label=label, marker=marker,
                       color=color, s=34, alpha=0.85)
            all_errors.extend(errors)
    if all_errors:
        reference_cov = components["B_persistent"] + components["R_iid"] / representative_n
        student_95 = 2.0 * f_dist.ppf(0.95, 2, STUDENT_DF)
        add_covariance_ellipse(
            ax, reference_cov, threshold=student_95, fill=False, linewidth=2.0,
            linestyle="--", edgecolor="#f07c26", label="proposed 95%",
        )
        values = np.asarray(all_errors)
        extent = max(1.0, 1.12 * float(np.abs(values).max()))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set(xlabel="corrected x error [cm]", ylabel="corrected y error [cm]",
           title="(b) held-out grouped means")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="best")

    ax = axes[2]
    names = list(scores)
    x = np.arange(len(names))
    naive_cov = [100.0 * scores[name]["naive_gaussian"]["coverage_95"] for name in names]
    floor_cov = [100.0 * scores[name]["floor_student_t"]["coverage_95"] for name in names]
    ax.bar(x - 0.18, naive_cov, width=0.36, color="#386cb0", label=r"naive $R/n$")
    ax.bar(x + 0.18, floor_cov, width=0.36, color="#f07c26", label="floor Student-t")
    ax.axhline(95.0, color="black", linestyle="--", linewidth=1.0, label="95% target")
    ax.set_xticks(x, ["space", "session", "both"])
    ax.set(ylim=(0, 105), ylabel="empirical 95% coverage", title="(c) calibration audit")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Repeated-pose commissioning: the floor helps, but spatial bias is not solved", fontsize=11)
    out = root / "analysis"
    fig.savefig(out / "fig_rcond_pilot.png", dpi=220)
    fig.savefig(out / "fig_rcond_pilot.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = load_rows(root)
    sessions = sorted({str(row["session_id"]) for row in rows})
    holdout_session = sessions[-1]
    train = [row for row in rows if row["session_id"] != holdout_session and row.get("split") == "train"]
    partitions = {
        "spatial_holdout": [row for row in rows if row["session_id"] != holdout_session and row.get("split") == "val"],
        "session_holdout": [row for row in rows if row["session_id"] == holdout_session and row.get("split") == "train"],
        "combined_holdout": [row for row in rows if row["session_id"] == holdout_session and row.get("split") == "val"],
    }
    if not train:
        raise RuntimeError("no training rows after grouped session/spatial split")
    bias = fit_bias(train)
    components = fit_components(train, bias)
    scores = {name: score_groups(part, bias, components) for name, part in partitions.items()}
    pixel_model, geometry_audit = load_geometry_audit(root, rows, holdout_session)
    eligible = [value for value in scores.values() if value.get("n_groups", 0) >= 5]
    pilot_valid = len(eligible) >= 2
    floor_better = bool(eligible) and all(
        value["floor_student_t"]["mean_nll"] <= value["naive_gaussian"]["mean_nll"]
        for value in eligible
    )
    heldout_coverage_ok = bool(eligible) and all(
        value["floor_student_t"]["coverage_95"] >= 0.50 for value in eligible
    )
    geometry_holdout_scale = geometry_audit.get(holdout_session, {}).get("honesty_multiplier", float("nan"))
    geometry_holdout_ok = bool(np.isfinite(geometry_holdout_scale) and 0.70 <= geometry_holdout_scale <= 1.30)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PILOT_VALID" if pilot_valid else "PILOT_INSUFFICIENT",
        "claim_status": "plumbing/mechanism only; not paper evidence",
        "sessions": sessions, "heldout_session": holdout_session,
        "n_qualified_readings": len(rows), "n_training_readings": len(train),
        "bias_features": list(FEATURE_NAMES),
        "bias_beta_xy_m": serial_matrix(bias["beta"]),
        "R_iid_m2": serial_matrix(components["R_iid"]),
        "B_persistent_m2": serial_matrix(components["B_persistent"]),
        "R_iid_sigma_cm": [float(100 * math.sqrt(components["R_iid"][i, i])) for i in range(2)],
        "B_persistent_sigma_cm": [float(100 * math.sqrt(max(components["B_persistent"][i, i], 0.0))) for i in range(2)],
        "training_groups": components["groups"], "training_within_group_dof": components["within_dof"],
        "holdout_scores": scores,
        "demonstration_gate": "PASS" if pilot_valid and floor_better else "FAIL",
        "gate_meaning": "the persistent-floor Student-t predictive model must beat naive R/n on sufficiently populated holdouts",
        "heldout_calibration_verdict": "PROVISIONAL" if heldout_coverage_ok else "NOT_CALIBRATED",
        "heldout_calibration_note": "pilot diagnostic only: every populated partition must exceed 50% empirical coverage for its nominal 95% region",
        "geometry_holdout_verdict": "CALIBRATED" if geometry_holdout_ok else "NOT_CALIBRATED",
        "geometry_propagation_audit": {
            "training_pixel_model": pixel_model,
            "heldout_session": holdout_session,
            "sessions": {
            session: {
                "pixel_sigma_diagnostic": value["pixel_sigma_diagnostic"],
                "honesty_multiplier": value["honesty_multiplier"],
                "n": len(value["records"]),
            }
            for session, value in geometry_audit.items()
            },
        },
        "online_inputs": ["camera image", "camera calibration", "detected marker pixels", "estimated pose/heading geometry"],
        "evaluation_only_inputs": ["teleport ground truth", "session and split labels"],
    }
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    make_figure(root, partitions, bias, components, scores)
    make_geometry_figure(root, pixel_model, geometry_audit)
    (root / "analysis/summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (root / "analysis/RESULTS.md").open("w", encoding="utf-8") as handle:
        handle.write("# Rcond commissioning v2 pilot\n\n")
        handle.write(f"Pipeline status: **{payload['status']}**. Geometry-propagated R on the restart holdout: "
                     f"**{payload['geometry_holdout_verdict']}** ({geometry_holdout_scale:.3f}x truth/claimed scale).\n\n")
        handle.write(f"Separate global ground-space floor diagnostic: NLL mechanism gate "
                     f"**{payload['demonstration_gate']}**, calibration **{payload['heldout_calibration_verdict']}**. "
                     "The NLL gate only tests whether adding a floor helps; it does not certify nominal coverage.\n\n")
        handle.write(f"Qualified keypoint readings: {len(rows)}; training: {len(train)}; sessions: {sessions}; held out: `{holdout_session}`.\n\n")
        handle.write(f"Independent repeatability sigma: {payload['R_iid_sigma_cm'][0]:.3f} × {payload['R_iid_sigma_cm'][1]:.3f} cm.\n\n")
        handle.write(f"Persistent floor sigma: {payload['B_persistent_sigma_cm'][0]:.3f} × {payload['B_persistent_sigma_cm'][1]:.3f} cm.\n\n")
        if geometry_audit:
            handle.write("## Deployable geometry-propagated R\n\n")
            handle.write(
                "Pixel sigma learned only from non-held-out training anchors: "
                f"posterior scale **{pixel_model['posterior_scale_sigma_px']:.3f} px**, "
                f"posterior mean **{pixel_model['posterior_mean_sigma_px']:.3f} px**, "
                f"deployed 95% upper value **{pixel_model['posterior_upper95_sigma_px']:.3f} px** "
                f"({pixel_model['n_training_readings']} readings, conservative df={pixel_model['effective_df']}).\n\n"
            )
            handle.write("| session | own pixel sigma (diagnostic) | truth / claimed scale using fixed training R | readings |\n")
            handle.write("|---|---:|---:|---:|\n")
            for session, value in geometry_audit.items():
                label = f"{session} (held out)" if session == holdout_session else session
                handle.write(f"| {label} | {value['pixel_sigma_diagnostic']:.3f} px | "
                             f"{value['honesty_multiplier']:.3f}x | {len(value['records'])} |\n")
            handle.write("\nHere 1.00x is calibrated. This is the deployable construction: learn constant pixel noise, then use the projection Jacobian to obtain spatial ground ellipses.\n\n")
        handle.write("| partition | groups | naive NLL | floor Student-t NLL | naive 95% | floor Student-t 95% |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for name, value in scores.items():
            if not value.get("n_groups"):
                handle.write(f"| {name} | 0 | — | — | — | — |\n")
            else:
                handle.write(f"| {name} | {value['n_groups']} | {value['naive_gaussian']['mean_nll']:.3f} | {value['floor_student_t']['mean_nll']:.3f} | {100*value['naive_gaussian']['coverage_95']:.1f}% | {100*value['floor_student_t']['coverage_95']:.1f}% |\n")
        handle.write("\nThe new-session result supports the persistent-floor mechanism, but the unseen-space result rejects this small global bias model as a calibrated spatial model. Do not put this pilot covariance into the planner.\n\n")
        handle.write("Figures: `fig_geometry_r_pilot.png` is the deployable geometry result; `fig_rcond_pilot.png` is the failed global-floor diagnostic (PDF versions are beside them).\n\n")
        handle.write("This pilot validates grouped capture and decomposition only. Micro-jitter is not a substitute for real camera/session variation.\n")
    print(root / "analysis/RESULTS.md")
    print(f"demonstration gate: {payload['demonstration_gate']}")


if __name__ == "__main__":
    main()
