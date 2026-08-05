#!/usr/bin/env python3
"""exp2: fit the missing degree of freedom in the per-camera projection bias.

Three independent lines of evidence converged on the same next step, so this is
that step:

1. exp1 (this study) — the deployed `projection_calibration_v2` removes the
   along-bearing bias essentially perfectly and leaves the CROSS-bearing bias
   untouched, because it has only that one degree of freedom. Camera C keeps
   +0.078 m lateral. Under blocked CV a constant 2-D world offset already beats
   the deployed model on A, C and D.
2. `logs/studies/projection_amplification/exp1_geometry_vs_detector` — every
   variance model on this data is limited by BIAS TRANSFER, not by its variance
   form: scoring with train-fold centring instead of oracle centring costs camera
   A 9-47 nats and drops empirical 90 % coverage to 13 %.
3. `logs/studies/operational_residual_rcond/exp2_operational_rcond` — belief NEES
   is 8.5-10.8 at detection instants (calibrated 1.39) because updates contract P
   toward measurements carrying 3-8 cm of systematic offset. `R_cond` is
   bias-bound, not data-bound.

So: extend the correction to two degrees of freedom in the bearing frame, select
among candidates by held-out performance, and then check whether the thing that
was actually blocked — the conditional covariance — becomes estimable.

Model ladder (all per-camera, all fitted on the TRAIN fold only):

    M0_raw               no correction
    MD_deployed          the frozen deployed constants (incumbent; never refit)
    M2_bearing_const     along = a0                               (1 param)
    M3_bearing_affine    along = a0 + a1 d                        (2 params, the
                                                                   deployed FORM)
    M1_world_const       (bx, by) constant in world frame         (2 params)
    M4_bearing_2dof      along = a0, cross = c0                   (2 params)  NEW
    M5_bearing_2dof_aff  along = a0 + a1 d, cross = c0 + c1 d     (4 params)  NEW
    M6_world_affine      b = A [x, y] + t                         (6 params)  NEW

M4 is the load-bearing comparison: it has the SAME parameter count as the
deployed form (M3) and as M1, so if it wins it is because it points in the right
directions, not because it has more freedom. M6 is included as an
overfitting control — a richer model that leave-region-out folds should punish.

Two fold schemes are reported side by side, because the choice is not innocent:
time-blocked (what exp1 used) and leave-region-out (the harder, and for a SPATIAL
bias claim the relevant, generalisation test).

Scope: this is per-camera commissioning calibration of the same kind as the
deployed `projection_calibration_v2` — which was itself fitted from these same
4-camera captures by `tools/fit_projection_calibration.py`. It is not method
development in the evaluation world. Nothing here is wired into the runtime: the
runtime projection has one along-bearing degree of freedom, so deploying a 2-DOF
correction is a separate, deliberate change (see RESULTS.md "What would have to
change to deploy this").

Ground truth is EVALUATION-ONLY: it measures residuals and selects among models.
It never enters a projection or a Jacobian.

Outputs -> logs/studies/external_camera_bias_model/exp2_two_dof_bias/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(REPO / "experiments" / "projection_amplification"))

from reliability.conditional_covariance import chi2_coverage, matrix_nll  # noqa: E402
from reliability.projection import (  # noqa: E402
    _project_pixel_to_world,
    camera_model_from_world,
    load_projection_calibration,
)

import residual_audit as RA  # noqa: E402  (owns the study constants + fold scheme)
# The variance-model machinery is owned by the projection-amplification study;
# imported so the loop closure is scored by exactly the same code.
import exp1_geometry_vs_detector as GEO  # noqa: E402

RESIDUALS_CSV = (
    REPO / "logs/studies/external_camera_bias_model/exp1_residual_characterization"
    / "residuals.csv"
)
OUT = REPO / "logs/studies/external_camera_bias_model/exp2_two_dof_bias"

CAMERAS = RA.CAMERAS
MIN_CAMERA_SAMPLES = 60
BOOT = 2000
RNG = np.random.default_rng(20260804)

C_DEPLOYED = "#D55E00"
C_2DOF = "#0072B2"
C_RAW = "#999999"
C_OTHER = "#E69F00"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


# ---------------------------------------------------------------------- loading


def load_rows() -> dict[str, list[dict]]:
    """Per-camera rows in capture/stamp order (fold schemes assume that order)."""

    per_camera: dict[str, list[dict]] = {camera: [] for camera in CAMERAS}
    with RESIDUALS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["camera"] not in per_camera:
                continue
            per_camera[row["camera"]].append({
                key: (row[key] if key in ("camera", "capture") else float(row[key]))
                for key in ("camera", "capture", "stamp", "u", "v", "true_x", "true_y",
                            "range_m", "raw_px", "raw_py", "raw_ex", "raw_ey")
            })
    for camera in per_camera:
        per_camera[camera].sort(key=lambda r: (r["capture"], r["stamp"]))
    return per_camera


def bearing_frame(rows: list[dict], camera_model) -> dict[str, np.ndarray]:
    """Bearing-frame decomposition of the RAW projection error, per sample.

    ``e_along`` is positive away from the camera and ``e_cross`` is positive to the
    left of the bearing. The basis is built from the RAW projected point, which is
    exactly what ``reliability.projection._project_pixel_to_world`` uses when it
    applies the deployed correction — so a correction expressed here and one passed
    through the runtime are the same operation, not an approximation of it.
    """

    cam_x, cam_y = float(camera_model.cam_pos[0]), float(camera_model.cam_pos[1])
    unit_along, unit_cross, distance, error = [], [], [], []
    for row in rows:
        bx, by = row["raw_px"] - cam_x, row["raw_py"] - cam_y
        d = math.hypot(bx, by)
        unit_along.append((bx / d, by / d))
        unit_cross.append((-by / d, bx / d))
        distance.append(d)
        error.append((row["raw_ex"], row["raw_ey"]))
    unit_along = np.asarray(unit_along)
    unit_cross = np.asarray(unit_cross)
    error = np.asarray(error)
    return {
        "unit_along": unit_along,
        "unit_cross": unit_cross,
        "distance": np.asarray(distance),
        "error": error,
        "e_along": np.sum(error * unit_along, axis=1),
        "e_cross": np.sum(error * unit_cross, axis=1),
        "xy": np.asarray([[r["raw_px"], r["raw_py"]] for r in rows]),
    }


# ------------------------------------------------------------------ the ladder


def _fit_line(distance: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """along/cross = a + b*d with the deployed fitter's slope gating."""

    return RA._fit_line(distance, values)


def predicted_error(name: str, frame: dict, train: np.ndarray, test: np.ndarray,
                    deployed: dict) -> np.ndarray:
    """Predicted 2-D world-frame error for ``test``, from a model fitted on ``train``.

    The returned vector is what the correction will SUBTRACT from the raw projected
    point, so the corrected residual is ``error[test] - predicted``.
    """

    along_hat = np.zeros(test.size)
    cross_hat = np.zeros(test.size)
    if name == "M0_raw":
        return np.zeros((test.size, 2))
    if name == "MD_deployed":
        # Frozen constants, never refit. The deployed correction ADDS
        # (intercept + slope*d) along the bearing, i.e. it predicts an error of
        # the opposite sign.
        along_hat = -(deployed["intercept_m"] + deployed["slope_per_m"] * frame["distance"][test])
    elif name == "M2_bearing_const":
        along_hat = np.full(test.size, float(frame["e_along"][train].mean()))
    elif name == "M3_bearing_affine":
        a, b = _fit_line(frame["distance"][train], frame["e_along"][train])
        along_hat = a + b * frame["distance"][test]
    elif name == "M1_world_const":
        return np.tile(frame["error"][train].mean(axis=0), (test.size, 1))
    elif name == "M4_bearing_2dof":
        along_hat = np.full(test.size, float(frame["e_along"][train].mean()))
        cross_hat = np.full(test.size, float(frame["e_cross"][train].mean()))
    elif name == "M5_bearing_2dof_aff":
        a, b = _fit_line(frame["distance"][train], frame["e_along"][train])
        c, e = _fit_line(frame["distance"][train], frame["e_cross"][train])
        along_hat = a + b * frame["distance"][test]
        cross_hat = c + e * frame["distance"][test]
    elif name == "M6_world_affine":
        design = np.column_stack([frame["xy"][train], np.ones(train.size)])
        target = frame["error"][train]
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return np.column_stack([frame["xy"][test], np.ones(test.size)]) @ coefficients
    elif name in ("MD_plus_cross_const", "MD_plus_cross_affine", "MD_plus_world_const"):
        # Keep the FROZEN deployed along-bearing correction, which exp1 showed works,
        # and add only what it structurally cannot represent. Because the deployed
        # correction moves the point along e_along, the cross component of the
        # residual is untouched by it, so the cross term is fitted directly on
        # e_cross with no re-derivation.
        along_hat = -(deployed["intercept_m"] + deployed["slope_per_m"] * frame["distance"][test])
        if name == "MD_plus_cross_const":
            cross_hat = np.full(test.size, float(frame["e_cross"][train].mean()))
        elif name == "MD_plus_cross_affine":
            c, e = _fit_line(frame["distance"][train], frame["e_cross"][train])
            cross_hat = c + e * frame["distance"][test]
        else:
            # study B's implicit baseline: deployed, then a world-frame constant
            # refit on the training fold.
            deployed_only = predicted_error("MD_deployed", frame, train, train, deployed)
            offset = (frame["error"][train] - deployed_only).mean(axis=0)
            return (predicted_error("MD_deployed", frame, train, test, deployed)
                    + np.tile(offset, (test.size, 1)))
    else:
        raise ValueError(f"unknown model {name!r}")
    return (along_hat[:, None] * frame["unit_along"][test]
            + cross_hat[:, None] * frame["unit_cross"][test])


MODELS = ("M0_raw", "MD_deployed", "M2_bearing_const", "M3_bearing_affine",
          "M1_world_const", "M4_bearing_2dof", "M5_bearing_2dof_aff", "M6_world_affine",
          "MD_plus_cross_const", "MD_plus_cross_affine", "MD_plus_world_const")
#: Parameters fitted PER FOLD. ``MD_*`` models inherit two frozen deployed constants
#: that were fitted once, on all of this data — see the RESULTS.md caveat.
PARAMETERS = {"M0_raw": 0, "MD_deployed": 0, "M2_bearing_const": 1, "M3_bearing_affine": 2,
              "M1_world_const": 2, "M4_bearing_2dof": 2, "M5_bearing_2dof_aff": 4,
              "M6_world_affine": 6, "MD_plus_cross_const": 1, "MD_plus_cross_affine": 2,
              "MD_plus_world_const": 2}


def verify_runtime_equivalence(rows: list[dict], camera_model, frame: dict,
                               deployed: dict) -> float:
    """Max discrepancy between a bearing-frame correction and the runtime path.

    Guards the central assumption of this script: applying ``a`` along the bearing
    in residual space equals passing ``a`` through
    ``_project_pixel_to_world(along_bearing_offset_m=...)``. If this ever exceeds
    machine noise, every number below is an approximation and must be relabelled.
    """

    worst = 0.0
    for index, row in enumerate(rows):
        offset = deployed["intercept_m"] + deployed["slope_per_m"] * frame["distance"][index]
        runtime = _project_pixel_to_world(
            row["u"], row["v"], camera_model,
            contact_z_m=RA.CONTACT_Z_M,
            along_bearing_offset_m=deployed["intercept_m"],
            along_bearing_slope_per_m=deployed["slope_per_m"],
        )
        mine = frame["xy"][index] + offset * frame["unit_along"][index]
        worst = max(worst, float(np.hypot(runtime[0] - mine[0], runtime[1] - mine[1])))
    return worst


# ------------------------------------------------------------------ evaluation


def fold_schemes(rows: list[dict], frame: dict):
    n = len(rows)
    yield "time_blocked", list(RA.blocked_folds(n, 5))
    spatial = []
    for train_mask, test_mask in GEO.spatial_folds(frame["xy"]):
        spatial.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
    yield "leave_region_out", spatial


def evaluate(per_camera: dict, models_geo: dict, calib: dict) -> dict:
    results: dict[str, dict] = {}
    for camera in CAMERAS:
        rows = per_camera[camera]
        if len(rows) < MIN_CAMERA_SAMPLES:
            results[camera] = {"status": "insufficient_samples", "n": len(rows)}
            continue
        camera_model = models_geo[camera]
        frame = bearing_frame(rows, camera_model)
        deployed = calib.get(camera, {"intercept_m": 0.0, "slope_per_m": 0.0})
        entry = {
            "status": "ok",
            "n": len(rows),
            "runtime_equivalence_max_m": verify_runtime_equivalence(
                rows, camera_model, frame, deployed
            ),
            "deployed_constants": deployed,
            "raw_bias_along_m": float(frame["e_along"].mean()),
            "raw_bias_cross_m": float(frame["e_cross"].mean()),
            "sigma_along_m": float(frame["e_along"].std()),
            "sigma_cross_m": float(frame["e_cross"].std()),
            # Whether the cross-bearing systematic is even resolvable against this
            # camera's own scatter. This is the quantity that predicts whether
            # fitting a cross term helps or hurts (see fig_x3).
            "cross_bias_to_sigma": abs(float(frame["e_cross"].mean()))
            / max(float(frame["e_cross"].std()), 1e-9),
            "along_bias_to_sigma": abs(float(frame["e_along"].mean()))
            / max(float(frame["e_along"].std()), 1e-9),
        }
        for scheme, folds in fold_schemes(rows, frame):
            per_model: dict[str, dict] = {}
            for name in MODELS:
                residuals, shifts = [], []
                for train, test in folds:
                    predicted = predicted_error(name, frame, train, test, deployed)
                    corrected = frame["error"][test] - predicted
                    residuals.append(corrected)
                    shifts.append(float(np.linalg.norm(corrected.mean(axis=0))))
                stacked = np.vstack(residuals)
                per_model[name] = {
                    "parameters": PARAMETERS[name],
                    "rms_m": float(np.sqrt(np.mean(np.sum(stacked**2, axis=1)))),
                    "heldout_bias_m": float(np.linalg.norm(stacked.mean(axis=0))),
                    "worst_fold_bias_m": float(max(shifts)),
                    "mean_fold_bias_m": float(np.mean(shifts)),
                    "folds": len(folds),
                }
            entry[scheme] = per_model
        results[camera] = entry
    return results


def paired_bootstrap_gain(per_camera: dict, models_geo: dict, calib: dict,
                          scheme: str = "leave_region_out",
                          challenger: str = "M4_bearing_2dof",
                          incumbent: str = "MD_deployed") -> dict:
    """Bootstrap CI on the held-out RMS difference, resampled by FOLD.

    Resampling folds rather than samples respects the block structure: adjacent
    frames are not independent, whole held-out regions approximately are.
    """

    out: dict[str, dict] = {}
    for camera in CAMERAS:
        rows = per_camera[camera]
        if len(rows) < MIN_CAMERA_SAMPLES:
            continue
        frame = bearing_frame(rows, models_geo[camera])
        deployed = calib.get(camera, {"intercept_m": 0.0, "slope_per_m": 0.0})
        folds = dict(fold_schemes(rows, frame))[scheme]
        per_fold = []
        for train, test in folds:
            a = frame["error"][test] - predicted_error(incumbent, frame, train, test, deployed)
            b = frame["error"][test] - predicted_error(challenger, frame, train, test, deployed)
            per_fold.append((np.sum(a**2, axis=1), np.sum(b**2, axis=1)))
        draws = []
        for _ in range(BOOT):
            picked = RNG.integers(0, len(per_fold), len(per_fold))
            incumbent_sq = np.concatenate([per_fold[i][0] for i in picked])
            challenger_sq = np.concatenate([per_fold[i][1] for i in picked])
            draws.append(math.sqrt(incumbent_sq.mean()) - math.sqrt(challenger_sq.mean()))
        draws = np.sort(draws)
        out[camera] = {
            "median_rms_gain_m": float(np.median(draws)),
            "ci95_m": [float(draws[int(0.025 * BOOT)]), float(draws[int(0.975 * BOOT)])],
            "fraction_challenger_better": float(np.mean(np.asarray(draws) > 0.0)),
            "folds_resampled": len(per_fold),
        }
    return out


def variance_loop_closure(per_camera: dict, models_geo: dict, calib: dict,
                          bias_model: str) -> dict:
    """Does correcting the bias make the conditional covariance estimable?

    Reruns the projection-amplification variance comparison with the bias model
    doing the centring instead of an oracle mean. This is the deployable
    configuration: everything is fitted on the training fold.
    """

    out: dict[str, dict] = {}
    for camera in CAMERAS:
        rows = per_camera[camera]
        if len(rows) < MIN_CAMERA_SAMPLES:
            continue
        camera_model = models_geo[camera]
        frame = bearing_frame(rows, camera_model)
        deployed = calib.get(camera, {"intercept_m": 0.0, "slope_per_m": 0.0})
        projection_kwargs = GEO.projection_kwargs(calib, camera)
        jacobians = []
        keep = []
        for index, row in enumerate(rows):
            jacobian = GEO.jacobian_at(camera_model, row["u"], row["v"], projection_kwargs)
            if jacobian is None or abs(np.linalg.det(jacobian)) < 1e-12:
                continue
            jacobians.append(jacobian)
            keep.append(index)
        if len(keep) < MIN_CAMERA_SAMPLES:
            continue
        keep = np.asarray(keep)
        shapes = GEO.geometric_shapes(np.asarray(jacobians))
        folds = dict(fold_schemes(rows, frame))["leave_region_out"]
        collected = {name: {"res": [], "cov": []} for name in ("R1-iso", "R2-geom")}
        for train, test in folds:
            train_local = np.flatnonzero(np.isin(keep, train))
            test_local = np.flatnonzero(np.isin(keep, test))
            if train_local.size < 8 or test_local.size == 0:
                continue
            train_rows, test_rows = keep[train_local], keep[test_local]
            train_res = (frame["error"][train_rows]
                         - predicted_error(bias_model, frame, train, train_rows, deployed))
            test_res = (frame["error"][test_rows]
                        - predicted_error(bias_model, frame, train, test_rows, deployed))
            iso = GEO.fit_iso(train_res)
            pixel_var = GEO.fit_geometric_pixel_variance(train_res, shapes[train_local])
            for name, covariances in (
                ("R1-iso", [iso] * test_local.size),
                ("R2-geom", [pixel_var * shape for shape in shapes[test_local]]),
            ):
                collected[name]["res"].extend(test_res.tolist())
                collected[name]["cov"].extend(
                    [GEO._psd_floor(np.asarray(c)).tolist() for c in covariances]
                )
        entry = {}
        for name, payload in collected.items():
            if not payload["res"]:
                continue
            entry[name] = {
                "nll": matrix_nll(payload["res"], payload["cov"]),
                "coverage_90": chi2_coverage(payload["res"], payload["cov"], 0.90),
                "coverage_95": chi2_coverage(payload["res"], payload["cov"], 0.95),
            }
        out[camera] = entry
    return out


def fit_full_calibration(per_camera: dict, models_geo: dict) -> dict:
    """The 2-DOF constants fitted on ALL data, in a deployable JSON schema.

    Emitted for inspection and for a future deliberate promotion. NOT loaded by
    anything: the runtime projection accepts one along-bearing degree of freedom,
    so consuming ``cross_bearing_offset_m`` requires a code change first.
    """

    cameras = {}
    for camera in CAMERAS:
        rows = per_camera[camera]
        if len(rows) < MIN_CAMERA_SAMPLES:
            continue
        frame = bearing_frame(rows, models_geo[camera])
        cameras[camera] = {
            # Correction constants: what to ADD to the raw projected point, so the
            # sign is opposite to the measured error.
            "along_bearing_offset_m": -float(frame["e_along"].mean()),
            "cross_bearing_offset_m": -float(frame["e_cross"].mean()),
            "slope_per_m": 0.0,
            "cross_slope_per_m": 0.0,
            "samples": len(rows),
            "range_span_m": [float(frame["distance"].min()), float(frame["distance"].max())],
        }
    return {
        "kind": "projection_bearing_frame_2dof_offsets",
        "convention": "p_corrected = p_raw + along*e_along + cross*e_cross; "
                      "e_along points away from the camera, e_cross to its left",
        "status": "CANDIDATE — not deployed; runtime supports along-bearing only",
        "fitted_from": str(RESIDUALS_CSV.relative_to(REPO)),
        "cameras": cameras,
    }


# --------------------------------------------------------------------- figures


def fig_x1(results: dict) -> None:
    """Held-out remaining bias and RMS per model, per camera, per fold scheme."""

    usable = [c for c in CAMERAS if results.get(c, {}).get("status") == "ok"]
    shown = ("M0_raw", "MD_deployed", "M3_bearing_affine", "M1_world_const",
             "M4_bearing_2dof", "M5_bearing_2dof_aff", "M6_world_affine")
    colors = {"M0_raw": C_RAW, "MD_deployed": C_DEPLOYED, "M3_bearing_affine": "#B07AA1",
              "M1_world_const": C_OTHER, "M4_bearing_2dof": C_2DOF,
              "M5_bearing_2dof_aff": "#56B4E9", "M6_world_affine": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    for row, scheme in enumerate(("time_blocked", "leave_region_out")):
        for column, (metric, label) in enumerate(
            (("heldout_bias_m", "remaining held-out bias  $|E[r]|$  [mm]"),
             ("rms_m", "held-out RMS error  [mm]")),
        ):
            ax = axes[row][column]
            width = 0.11
            positions = np.arange(len(usable))
            for index, name in enumerate(shown):
                offset = (index - (len(shown) - 1) / 2) * width
                values = [1000 * results[c][scheme][name][metric] for c in usable]
                ax.bar(positions + offset, values, width, color=colors[name],
                       label=f"{name} ({PARAMETERS[name]}p)" if row == 0 and column == 0 else None)
            ax.set_xticks(positions, [c.replace("camera_", "") for c in usable])
            ax.set_ylabel(label)
            ax.set_title(f"{scheme.replace('_', ' ')}", fontweight="bold", fontsize=10)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle("Adding the cross-bearing degree of freedom at equal parameter count\n"
                 "(M4 vs the deployed form M3 and the world-constant M1)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_x1_heldout_bias_and_rms.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_x2(per_camera: dict, models_geo: dict, calib: dict) -> None:
    """Residual scatter in the bearing frame: what each correction actually removes."""

    usable = [c for c in CAMERAS if len(per_camera[c]) >= MIN_CAMERA_SAMPLES]
    fig, axes = plt.subplots(1, len(usable), figsize=(3.7 * len(usable), 3.9))
    for ax, camera in zip(np.atleast_1d(axes), usable):
        rows = per_camera[camera]
        frame = bearing_frame(rows, models_geo[camera])
        deployed = calib.get(camera, {"intercept_m": 0.0, "slope_per_m": 0.0})
        every = np.arange(len(rows))
        for name, color, marker in (("M0_raw", C_RAW, "."),
                                    ("MD_deployed", C_DEPLOYED, "."),
                                    ("M4_bearing_2dof", C_2DOF, ".")):
            # In-sample here on purpose: this figure shows WHAT each correction
            # removes, not how well it generalises (that is fig_x1).
            corrected = frame["error"] - predicted_error(name, frame, every, every, deployed)
            along = np.sum(corrected * frame["unit_along"], axis=1) * 1000.0
            cross = np.sum(corrected * frame["unit_cross"], axis=1) * 1000.0
            ax.scatter(along, cross, s=5, alpha=0.35, color=color, marker=marker)
            ax.plot([along.mean()], [cross.mean()], "o", color=color, ms=9,
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=f"{name}: |bias| {math.hypot(along.mean(), cross.mean()):.0f} mm")
        ax.axhline(0.0, color="#333333", lw=0.9)
        ax.axvline(0.0, color="#333333", lw=0.9)
        ax.set_aspect("equal")
        ax.set_xlabel("along-bearing residual [mm]")
        ax.set_title(camera.replace("camera_", "camera "), fontweight="bold", fontsize=10)
        ax.legend(fontsize=6.8, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("cross-bearing residual [mm]")
    fig.suptitle("The deployed correction collapses the along axis and leaves the cross axis "
                 "where it was\n(large markers = mean; in-sample, to show what is removed)",
                 fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_x2_bearing_frame_residuals.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_x3(results: dict, bootstrap: dict) -> dict:
    """The decision rule: when is fitting a cross-bearing term worth it?

    Four cameras is four points, so this is a hypothesis with a mechanism, not a
    fitted law. The mechanism is not subtle: a systematic you cannot resolve
    against your own scatter is a systematic you will estimate mostly as noise,
    and subtracting noise fitted in one region from another region adds error.
    """

    usable = [c for c in bootstrap if results.get(c, {}).get("status") == "ok"]
    ratios = [results[c]["cross_bias_to_sigma"] for c in usable]
    gains = [1000 * bootstrap[c]["median_rms_gain_m"] for c in usable]
    lows = [1000 * (bootstrap[c]["median_rms_gain_m"] - bootstrap[c]["ci95_m"][0])
            for c in usable]
    highs = [1000 * (bootstrap[c]["ci95_m"][1] - bootstrap[c]["median_rms_gain_m"])
             for c in usable]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axhline(0.0, color="#333333", lw=1.2)
    ax.axvspan(0.0, 1.0, color=C_DEPLOYED, alpha=0.08)
    ax.errorbar(ratios, gains, yerr=[lows, highs], fmt="o", color=C_2DOF, ms=9,
                capsize=4, lw=1.6, markeredgecolor="white", markeredgewidth=1.2)
    for camera, ratio, gain in zip(usable, ratios, gains):
        ax.annotate(camera.replace("camera_", ""), xy=(ratio, gain),
                    xytext=(6, 6), textcoords="offset points", fontsize=10,
                    fontweight="bold")
    ax.set_xlabel(r"$|b_{\rm cross}|\ /\ \sigma_{\rm cross}$   "
                  "(is the lateral systematic resolvable at all?)")
    ax.set_ylabel("held-out RMS gain from adding\nthe cross term  [mm]  "
                  "(positive = better)")
    ax.set_title("Adding the cross-bearing DOF pays exactly where the lateral bias is\n"
                 "resolvable, and costs where it is not (shaded: below own scatter)",
                 fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_x3_when_the_cross_term_pays.{ext}", bbox_inches="tight")
    plt.close(fig)
    return {
        camera: {"cross_bias_to_sigma": ratio, "rms_gain_mm": gain,
                 "p_better": bootstrap[camera]["fraction_challenger_better"]}
        for camera, ratio, gain in zip(usable, ratios, gains)
    }


def main() -> int:
    if not RESIDUALS_CSV.is_file():
        print(f"missing input: {RESIDUALS_CSV}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    calib = load_projection_calibration(RA.DEPLOYED_CALIB)
    models_geo = {
        camera: camera_model_from_world(RA.WORLD_SDF, include_name=include)
        for camera, include in RA.MODEL_INCLUDES.items()
    }
    per_camera = load_rows()
    results = evaluate(per_camera, models_geo, calib)
    payload = {
        "inputs": {"residuals_csv": str(RESIDUALS_CSV.relative_to(REPO)),
                   "deployed_calibration": str(RA.DEPLOYED_CALIB.relative_to(REPO))},
        "models": {name: PARAMETERS[name] for name in MODELS},
        "heldout": results,
        # The decisive comparison is the last one: same frozen along-bearing
        # correction, one extra cross parameter. The first two conflate the added
        # cross DOF with DROPPING the deployed range slope.
        "paired_bootstrap_M4_vs_deployed": paired_bootstrap_gain(per_camera, models_geo, calib),
        "paired_bootstrap_M5_vs_M3_same_form": paired_bootstrap_gain(
            per_camera, models_geo, calib, challenger="M5_bearing_2dof_aff",
            incumbent="M3_bearing_affine"),
        "paired_bootstrap_MDcross_vs_MD": paired_bootstrap_gain(
            per_camera, models_geo, calib, challenger="MD_plus_cross_const",
            incumbent="MD_deployed"),
        "variance_loop_closure": {
            model: variance_loop_closure(per_camera, models_geo, calib, model)
            for model in ("MD_deployed", "MD_plus_world_const", "MD_plus_cross_const",
                          "M4_bearing_2dof")
        },
        "candidate_calibration": fit_full_calibration(per_camera, models_geo),
    }
    fig_x1(results)
    fig_x2(per_camera, models_geo, calib)
    payload["decision_rule"] = fig_x3(results, payload["paired_bootstrap_MDcross_vs_MD"])
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "projection_calibration_2dof_candidate.json").write_text(
        json.dumps(payload["candidate_calibration"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    for camera in CAMERAS:
        entry = results[camera]
        if entry.get("status") != "ok":
            print(f"{camera}: {entry}")
            continue
        print(f"{camera}: n={entry['n']} raw along={entry['raw_bias_along_m']:+.4f} "
              f"cross={entry['raw_bias_cross_m']:+.4f} m  "
              f"runtime_equiv={entry['runtime_equivalence_max_m']:.2e} m")
        for scheme in ("time_blocked", "leave_region_out"):
            best = min(MODELS, key=lambda n: entry[scheme][n]["rms_m"])
            print(f"    {scheme:>16}: " + "  ".join(
                f"{n.replace('bearing_', '').replace('world_', 'w_')}"
                f"={1000 * entry[scheme][n]['rms_m']:.0f}/"
                f"{1000 * entry[scheme][n]['heldout_bias_m']:.0f}"
                for n in ("M0_raw", "MD_deployed", "M3_bearing_affine", "M1_world_const",
                          "M4_bearing_2dof", "M5_bearing_2dof_aff", "M6_world_affine"))
                  + f"   [rms/bias mm; best rms = {best}]")
    for key in ("paired_bootstrap_M4_vs_deployed", "paired_bootstrap_M5_vs_M3_same_form",
                "paired_bootstrap_MDcross_vs_MD"):
        print(f"\n{key} (leave-region-out, fold-resampled):")
        for camera, stats in payload[key].items():
            print(f"  {camera}: gain {1000 * stats['median_rms_gain_m']:+.1f} mm "
                  f"CI95 [{1000 * stats['ci95_m'][0]:+.1f}, {1000 * stats['ci95_m'][1]:+.1f}] "
                  f"P(better)={stats['fraction_challenger_better']:.2f}")
    print("\nvariance loop closure (deployable centring, leave-region-out):")
    for model, per_camera_stats in payload["variance_loop_closure"].items():
        print(f"  centred by {model}:")
        for camera, stats in per_camera_stats.items():
            parts = "  ".join(f"{k}: NLL {v['nll']:7.3f} cov90 {v['coverage_90']:.2f}"
                              for k, v in stats.items())
            print(f"    {camera}: {parts}")
    print("\nwrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
