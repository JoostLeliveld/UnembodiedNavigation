#!/usr/bin/env python3
"""E2 — availability and conditional accuracy are different fields.

The claim under test is C1: a single scalar trust score cannot stand in for what
an external camera delivers. The sharp version, on one grid and one capture:

* ``P_D(x)``      — does a usable detection arrive at this pose?
* ``R_cond(x)``   — how accurate is it, given that one arrived?

If these were the same field up to a monotone transform, one number would do, and
the planner could keep folding availability into covariance. This experiment
measures whether they are.

Data. The 2026-08-07 commissioning grid contains every ATTEMPTED pose (942
positions x 8 headings x 4 cameras) and, separately, the projection residual for
every pose that produced a detection. Availability and conditional accuracy are
therefore measured on the same positions, from the same capture, under the current
zero-parameter IPM.

Ground truth is used for the residual magnitude only, which is why this is an
offline characterisation and not a runtime field. No result here licenses a
navigation or safety reading.

Run:
    python3 experiments/availability_paper/e2_availability_vs_accuracy/run_experiment.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402
import metrics as M  # noqa: E402

RESULTS = C.OUT_ROOT / "e2_availability_vs_accuracy"
GRID_ROOT = C.REPO / "logs/visibility_comparison/commissioning_grid_20260807"
SAMPLES = GRID_ROOT / "samples.csv"
RESIDUALS = GRID_ROOT / "grid_residuals_raw_ipm.csv"

CAMERA_FROM_FRAME = {
    "external_camera": "camera_A",
    "external_camera_b": "camera_B",
    "external_camera_c": "camera_C",
    "external_camera_d": "camera_D",
}
#: Positions need enough attempted headings for a rate to mean anything, and
#: enough detections for a conditional mean to mean anything.
MIN_ATTEMPTS = 8
MIN_DETECTIONS = 2
QUANT = 3  # decimals used to key a position


def _key(x, y) -> tuple[float, float]:
    return (round(float(x), QUANT), round(float(y), QUANT))


def rho(a, b) -> float:
    """Spearman rho only. ``metrics.spearman`` returns (rho, n_finite)."""

    return float(M.spearman(a, b)[0])


def load_grid() -> dict[str, dict[tuple[float, float], dict[str, float]]]:
    """Per camera, per position: attempts, detections, mean conditional error."""

    if not SAMPLES.is_file() or not RESIDUALS.is_file():
        raise RuntimeError(f"Commissioning grid missing under {GRID_ROOT}")

    table: dict[str, dict[tuple[float, float], dict[str, float]]] = {c: {} for c in C.CAMERAS}
    with open(SAMPLES, newline="") as fh:
        for row in csv.DictReader(fh):
            camera = CAMERA_FROM_FRAME.get(row["camera_frame"])
            if camera is None:
                continue
            cell = table[camera].setdefault(
                _key(row["x"], row["y"]),
                {"attempts": 0.0, "detections": 0.0, "error_sum": 0.0, "oracle_visible": 0.0},
            )
            cell["attempts"] += 1.0
            cell["oracle_visible"] += float(row["oracle_visible"] or 0.0)

    unmatched = 0
    with open(RESIDUALS, newline="") as fh:
        for row in csv.DictReader(fh):
            camera = row["camera"]
            if camera not in table:
                continue
            key = _key(row["true_x"], row["true_y"])
            cell = table[camera].get(key)
            if cell is None:
                unmatched += 1
                continue
            norm = float(row["raw_norm"])
            if not np.isfinite(norm):
                continue
            cell["detections"] += 1.0
            cell["error_sum"] += norm
    if unmatched:
        raise RuntimeError(
            f"{unmatched} residual rows had no matching attempted pose; the two files "
            "do not describe the same grid and must not be joined."
        )
    return table


def per_camera_frames(table) -> dict[str, dict[str, np.ndarray]]:
    """Assemble aligned arrays of availability and conditional error per camera."""

    out: dict[str, dict[str, np.ndarray]] = {}
    for camera in C.CAMERAS:
        xs, ys, avail, err, n_det, oracle = [], [], [], [], [], []
        for (x, y), cell in sorted(table[camera].items()):
            if cell["attempts"] < MIN_ATTEMPTS:
                continue
            xs.append(x)
            ys.append(y)
            avail.append(cell["detections"] / cell["attempts"])
            n_det.append(cell["detections"])
            oracle.append(cell["oracle_visible"] / cell["attempts"])
            err.append(cell["error_sum"] / cell["detections"] if cell["detections"] > 0 else np.nan)
        out[camera] = {
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "availability": np.asarray(avail, dtype=float),
            "conditional_error_m": np.asarray(err, dtype=float),
            "n_detections": np.asarray(n_det, dtype=float),
            "oracle_visible_rate": np.asarray(oracle, dtype=float),
        }
    return out


CORR_COLUMNS = (
    "camera", "n_positions", "n_positions_with_conditional_error",
    "spearman_availability_vs_error", "spearman_cad_vs_availability",
    "spearman_cad_vs_error", "spearman_monodepth_vs_availability",
    "spearman_monodepth_vs_error", "median_conditional_error_m",
    "median_error_low_availability_m", "median_error_high_availability_m",
)
DISAGREE_COLUMNS = (
    "n_positions_compared", "fraction_argmax_availability_ne_argmin_error",
    "median_error_penalty_following_availability_m",
    "median_availability_penalty_following_accuracy",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()
    out = Path(args.out)

    apparatus = C.build_apparatus()
    frames = per_camera_frames(load_grid())

    corr_rows: list[dict] = []
    for camera in C.CAMERAS:
        f = frames[camera]
        pts = np.column_stack([f["x"], f["y"]])
        cad = C.sample_field_at(apparatus.field("cad_reference", camera), apparatus.xs, apparatus.ys, pts)
        mono = C.sample_field_at(apparatus.field("mono_depth", camera), apparatus.xs, apparatus.ys, pts)

        ok = f["n_detections"] >= MIN_DETECTIONS
        err = f["conditional_error_m"]
        avail = f["availability"]

        lo = ok & (avail <= np.nanquantile(avail[ok], 0.25))
        hi = ok & (avail >= np.nanquantile(avail[ok], 0.75))

        corr_rows.append(
            {
                "camera": camera,
                "n_positions": int(avail.size),
                "n_positions_with_conditional_error": int(ok.sum()),
                "spearman_availability_vs_error": f"{rho(avail[ok], err[ok]):.6f}",
                "spearman_cad_vs_availability": f"{rho(cad, avail):.6f}",
                "spearman_cad_vs_error": f"{rho(cad[ok], err[ok]):.6f}",
                "spearman_monodepth_vs_availability": f"{rho(mono, avail):.6f}",
                "spearman_monodepth_vs_error": f"{rho(mono[ok], err[ok]):.6f}",
                "median_conditional_error_m": f"{np.nanmedian(err[ok]):.6f}",
                "median_error_low_availability_m": f"{np.nanmedian(err[lo]):.6f}" if lo.any() else "",
                "median_error_high_availability_m": f"{np.nanmedian(err[hi]):.6f}" if hi.any() else "",
            }
        )

    # Camera disagreement: where the most-available camera is not the most accurate.
    common_keys = sorted(
        set.intersection(*[{(x, y) for x, y in zip(frames[c]["x"], frames[c]["y"])} for c in C.CAMERAS])
    )
    index = {c: {(x, y): i for i, (x, y) in enumerate(zip(frames[c]["x"], frames[c]["y"]))} for c in C.CAMERAS}

    compared = 0
    disagree = 0
    error_penalty: list[float] = []
    availability_penalty: list[float] = []
    for key in common_keys:
        avail = []
        err = []
        for camera in C.CAMERAS:
            i = index[camera][key]
            f = frames[camera]
            if f["n_detections"][i] < MIN_DETECTIONS:
                avail.append(np.nan)
                err.append(np.nan)
            else:
                avail.append(f["availability"][i])
                err.append(f["conditional_error_m"][i])
        avail_a = np.asarray(avail)
        err_a = np.asarray(err)
        usable = np.isfinite(avail_a) & np.isfinite(err_a)
        if int(usable.sum()) < 2:
            continue
        compared += 1
        by_availability = int(np.nanargmax(np.where(usable, avail_a, -np.inf)))
        by_accuracy = int(np.nanargmin(np.where(usable, err_a, np.inf)))
        if by_availability != by_accuracy:
            disagree += 1
            error_penalty.append(float(err_a[by_availability] - err_a[by_accuracy]))
            availability_penalty.append(float(avail_a[by_availability] - avail_a[by_accuracy]))

    disagree_rows = [
        {
            "n_positions_compared": compared,
            "fraction_argmax_availability_ne_argmin_error": (
                f"{disagree / compared:.6f}" if compared else ""
            ),
            "median_error_penalty_following_availability_m": (
                f"{np.median(error_penalty):.6f}" if error_penalty else ""
            ),
            "median_availability_penalty_following_accuracy": (
                f"{np.median(availability_penalty):.6f}" if availability_penalty else ""
            ),
        }
    ]

    C.write_csv(out / "e2_correlations.csv", CORR_COLUMNS, corr_rows)
    C.write_csv(out / "e2_camera_disagreement.csv", DISAGREE_COLUMNS, disagree_rows)
    C.write_json(
        out / "manifest.json",
        {
            "experiment_id": "EXP-AVAIL-VS-ACC",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "capture": str(GRID_ROOT.relative_to(C.REPO)),
            "grid": "942 positions x 8 headings x 4 cameras; current zero-parameter IPM",
            "availability_definition": "detections / attempted headings at a position",
            "conditional_accuracy_definition": "mean raw_norm over the detections at a position",
            "thresholds": {"min_attempts": MIN_ATTEMPTS, "min_detections": MIN_DETECTIONS},
            "evaluation_only_inputs": ["commanded ground-truth x/y"],
            "prohibited_interpretation": [
                "any navigation, safety or closed-loop reading",
                "per-camera R_cond beats pooled R_cond (that is EXP-RCOND's null)",
            ],
            "inputs_sha256": C.input_manifest(extra=[SAMPLES, RESIDUALS]),
        },
    )

    print(f"wrote {out}/e2_correlations.csv")
    print(f"wrote {out}/e2_camera_disagreement.csv\n")
    header = (
        f"{'camera':<10}{'n pos':>7}{'n w/err':>9}{'rho(avail,err)':>16}"
        f"{'rho(CAD,avail)':>16}{'rho(CAD,err)':>14}{'med err m':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in corr_rows:
        print(
            f"{row['camera']:<10}{row['n_positions']:>7}{row['n_positions_with_conditional_error']:>9}"
            f"{float(row['spearman_availability_vs_error']):>16.3f}"
            f"{float(row['spearman_cad_vs_availability']):>16.3f}"
            f"{float(row['spearman_cad_vs_error']):>14.3f}"
            f"{float(row['median_conditional_error_m']):>11.4f}"
        )
    d = disagree_rows[0]
    print(
        f"\nPositions with two or more usable cameras: {d['n_positions_compared']}\n"
        f"Most-available is not most-accurate at: "
        f"{float(d['fraction_argmax_availability_ne_argmin_error']) * 100:.1f}% of them\n"
        f"Median accuracy cost of following availability: "
        f"{float(d['median_error_penalty_following_availability_m']) * 100:.2f} cm"
    )


if __name__ == "__main__":
    main()
