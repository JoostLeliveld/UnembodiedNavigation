#!/usr/bin/env python3
"""Render thesis figures from the exact locked Stage-05--08 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis_stage09_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "logs/thesis_final_pipeline_v1"
SOURCES = {
    "detector_640": ROOT / "stage05_detector_training/imgsz640/validation_report.json",
    "detector_960": ROOT / "stage05_detector_training/imgsz960/validation_report.json",
    "gate": ROOT / "stage06_detector_gate/gate_selection_v1/gate_selection.json",
    "sliver_report": ROOT / "stage06_detector_gate/camera_D_sliver_challenge_result/report.json",
    "sliver_image": ROOT / "stage06_detector_gate/camera_D_sliver_challenge_result/camera_D_sliver_annotated.png",
    "correction": ROOT / "stage07_correction_covariance/run_v3/correction_report.json",
    "covariance": ROOT / "stage07_correction_covariance/run_v3/covariance_report.json",
    "correction_oof": ROOT / "stage07_correction_covariance/run_v3/oof_predictions.csv",
    "availability": ROOT / "stage08_availability/run_v2/availability_report.json",
    "availability_oof": ROOT / "stage08_availability/run_v2/oof_predictions.csv",
    "availability_curve": ROOT / "stage08_availability/run_v2/commissioning_size_curve.csv",
}
COLORS = {
    "C0_raw": "#657789",
    "C1_visual_hull": "#5b9a85",
    "C2_ridge_residual": "#c57a2a",
    "C3_mlp_residual": "#8d67a7",
    "Q0_camera_constant": "#657789",
    "Q1_geometry_logistic": "#c57a2a",
    "Q2_rbf_heading_logistic": "#8d67a7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(key: str) -> dict:
    return json.loads(SOURCES[key].read_text())


def read_csv(key: str) -> list[dict]:
    with SOURCES[key].open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, output: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def detector_figure(output: Path) -> dict:
    reports = {size: read_json(f"detector_{size}") for size in (640, 960)}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)
    for size, marker in ((640, "o"), (960, "s")):
        report = reports[size]
        iou = np.asarray([float(key) for key in report["ap_by_iou"]])
        ap = np.asarray(list(report["ap_by_iou"].values()))
        axes[0].plot(iou, ap, marker=marker, ms=3.2, lw=1.2, label=f"{size}px")
        points = report["operating_points"]
        axes[1].plot(
            [row["recall"] for row in points],
            [row["precision"] for row in points],
            marker=marker, ms=3.0, lw=1.1, label=f"{size}px",
        )
        selected = next(row for row in points if row["confidence"] == 0.25)
        axes[1].scatter(selected["recall"], selected["precision"],
                        s=50, facecolors="none", edgecolors="black", linewidths=0.9)
    axes[0].set(xlabel="IoU threshold", ylabel="Average precision", ylim=(0, 1.03))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0.95, 1.001), ylim=(0.93, 1.003))
    axes[0].set_title("Spatially held-out validation")
    axes[1].set_title("Confidence operating points")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    save(fig, output, "stage05_detector_validation")
    selected = next(row for row in reports[960]["operating_points"]
                    if row["confidence"] == 0.25)
    return {
        "selected_resolution_px": 960,
        "validation_images": reports[960]["images"],
        "positive_images": reports[960]["positives"],
        "negative_images": reports[960]["negatives"],
        "ap50": reports[960]["ap50"],
        "ap50_95": reports[960]["ap50_95"],
        "precision_at_0_25": selected["precision"],
        "recall_at_0_25": selected["recall"],
        "false_detections_on_negative_images_at_0_25": selected["false_detections_on_negatives"],
        "smallest_evaluated_visible_width_bin_px": "32-63",
    }


def gate_figure(output: Path) -> dict:
    gate = read_json("gate")["selected"]
    image = mpimg.imread(SOURCES["sliver_image"])
    cameras = sorted(gate["by_camera"])
    recall = [gate["by_camera"][camera]["recall"] for camera in cameras]
    precision = [gate["by_camera"][camera]["precision"] for camera in cameras]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.8, 1]})
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Named Camera-D sliver challenge")
    x = np.arange(len(cameras))
    axes[1].bar(x - 0.18, precision, 0.36, label="Precision", color="#5b9a85")
    axes[1].bar(x + 0.18, recall, 0.36, label="Recall", color="#c57a2a")
    axes[1].set(xticks=x, xticklabels=[c.replace("camera_", "") for c in cameras],
                xlabel="Camera", ylabel="Held-out gate metric", ylim=(0.72, 1.015))
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, loc="lower left")
    axes[1].set_title("Selected gate by camera")
    save(fig, output, "stage06_validity_gate")
    return {
        "selected_gate": gate["candidate"],
        "pooled_precision": gate["pooled"]["precision"],
        "pooled_recall": gate["pooled"]["recall"],
        "admitted": gate["pooled"]["admitted"],
        "opportunities": gate["pooled"]["views"],
        "zero_mask_false_admissions": gate["false_admitted_zero_mask"],
        "sliver_outcome": read_json("sliver_report")["camera_D_required_outcome"],
    }


def correction_figure(output: Path) -> dict:
    report, covariance = read_json("correction"), read_json("covariance")
    rows = read_csv("correction_oof")
    candidates = ["C0_raw", "C1_visual_hull", "C2_ridge_residual", "C3_mlp_residual"]
    labels = ["Raw", "Hull", "Ridge", "MLP"]
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.15), constrained_layout=True)
    for candidate, label in zip(candidates, labels):
        values = np.sort(np.asarray([float(row[f"{candidate}_error_m"]) for row in rows]))
        axes[0].plot(100 * values, np.arange(1, values.size + 1) / values.size,
                     color=COLORS[candidate], lw=1.25, label=label)
    axes[0].set(xlabel="Out-of-fold position error [cm]", ylabel="Empirical CDF", xlim=(0, 25))
    axes[0].legend(frameon=False)
    axes[0].set_title("Correction candidates")
    cameras = ["camera_A", "camera_B", "camera_C", "camera_D", "camera_E"]
    x = np.arange(5)
    raw = [100 * report["candidate_reports"]["C0_raw"]["by_camera"][c]["p95_m"] for c in cameras]
    ridge = [100 * report["candidate_reports"]["C2_ridge_residual"]["by_camera"][c]["p95_m"] for c in cameras]
    axes[1].bar(x - .18, raw, .36, label="Raw", color=COLORS["C0_raw"])
    axes[1].bar(x + .18, ridge, .36, label="Ridge", color=COLORS["C2_ridge_residual"])
    axes[1].set(xticks=x, xticklabels=list("ABCDE"), xlabel="Camera", ylabel="OOF 95th percentile [cm]")
    axes[1].legend(frameon=False)
    axes[1].set_title("Error by camera")
    rcandidates = list(covariance["candidates"])
    containment = [100 * covariance["candidates"][key]["containment"]["95"] for key in rcandidates]
    axes[2].bar(rcandidates, containment,
                color=["#657789" if key != "R2C" else "#c57a2a" for key in rcandidates])
    axes[2].axhline(95, color="black", ls="--", lw=0.9, label="Nominal 95%")
    axes[2].set(xlabel="Covariance candidate", ylabel="OOF 95% containment [%]", ylim=(82, 100))
    axes[2].legend(frameon=False)
    axes[2].set_title("Covariance calibration")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    save(fig, output, "stage07_correction_covariance")
    selected = report["candidate_reports"][report["selected"]]["pooled"]
    r2c = covariance["candidates"][covariance["selected"]]
    return {
        "observations": report["observations"],
        "positions": report["positions"],
        "spatial_blocks": report["spatial_blocks"],
        "selected_correction": report["selected"],
        "selected_median_m": selected["median_m"],
        "selected_p95_m": selected["p95_m"],
        "selected_rmse_m": selected["rms_m"],
        "selected_covariance": covariance["selected"],
        "deployment_covariance_scale": covariance["deployment_R2C_scale"],
        "selected_95_containment": r2c["containment"]["95"],
    }


def availability_figure(output: Path) -> dict:
    report = read_json("availability")
    oof, curve = read_csv("availability_oof"), read_csv("availability_curve")
    truth = np.asarray([float(row["available"]) for row in oof])
    probability = np.asarray([float(row["Q1_geometry_logistic"]) for row in oof])
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.15), constrained_layout=True)
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (probability >= lo) & (probability < hi if hi < 1 else probability <= hi)
        if np.any(use):
            axes[0].scatter(np.mean(probability[use]), np.mean(truth[use]),
                            s=18 + 2 * np.sqrt(np.sum(use)), color="#c57a2a")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.9)
    axes[0].set(xlabel="Mean predicted availability", ylabel="Observed available fraction",
                xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[0].set_title("Spatial OOF calibration")
    candidates = list(report["candidates"])
    cameras = list("ABCDE")
    x = np.arange(5)
    for offset, candidate in zip((-.24, 0, .24), candidates):
        values = [report["candidates"][candidate]["by_camera"][f"camera_{c}"]["brier"] for c in cameras]
        axes[1].bar(x + offset, values, .24, color=COLORS[candidate],
                    label={"Q0_camera_constant":"Q0 constant", "Q1_geometry_logistic":"Q1 geometry",
                           "Q2_rbf_heading_logistic":"Q2 RBF"}[candidate])
    axes[1].set(xticks=x, xticklabels=cameras, xlabel="Camera", ylabel="OOF Brier score")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title("Candidate error by camera")
    sizes = np.asarray([float(row["requested_positions"]) for row in curve])
    brier = np.asarray([float(row["block_macro_brier"]) for row in curve])
    axes[2].plot(sizes, brier, "o-", color="#c57a2a", lw=1.25, ms=4)
    axes[2].set(xlabel="Requested commissioning positions", ylabel="Block-macro Brier score",
                xticks=sizes)
    axes[2].set_title("Commissioning-size curve")
    for ax in axes:
        ax.grid(alpha=0.2)
    save(fig, output, "stage08_availability")
    selected = report["candidates"][report["selected"]]
    return {
        "opportunities": len(oof),
        "selected": report["selected"],
        "block_macro_brier": selected["block_macro_brier"],
        "pooled_brier": selected["pooled_brier"],
        "pooled_auprc": selected["pooled_auprc"],
        "pooled_ece_10": selected["pooled_ece_10"],
        "calibration_intercept": selected["calibration"]["intercept"],
        "calibration_slope": selected["calibration"]["slope"],
        "commissioning_size_curve": report["commissioning_size_curve"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    owned = [
        output / f"stage{stage}_{name}.{suffix}"
        for stage, name in (("05", "detector_validation"), ("06", "validity_gate"),
                            ("07", "correction_covariance"), ("08", "availability"))
        for suffix in ("pdf", "svg", "png")
    ] + [output / "report.json"]
    if any(path.exists() for path in owned):
        raise FileExistsError("refusing to overwrite Stage-05--08 thesis figures")
    plt.rcParams.update({"font.size": 8.4, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    results = {
        "stage05": detector_figure(output),
        "stage06": gate_figure(output),
        "stage07": correction_figure(output),
        "stage08": availability_figure(output),
    }
    report = {
        "schema": "thesis_stage05_08_figure_report.v1",
        "status": "derived_from_locked_artifacts",
        "selection_or_fitting_performed": False,
        "sources": {str(path.relative_to(REPO)): sha256(path) for path in SOURCES.values()},
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "results": results,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
