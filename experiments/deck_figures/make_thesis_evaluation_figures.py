#!/usr/bin/env python3
"""Render and export the sealed reference-position audit results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "papers" / "Thesis"
REPORT = ROOT / "logs/thesis_final_pipeline_v1/recapture_v5/final_audit/report.json"
OUT = PAPER / "figures"
EXPORT = ROOT / "logs/thesis_current/paper_exports"
MODELS = ("R0_global_full", "R1_per_camera_full", "R2_spatial_full")
LABELS = (r"global $R_0$", r"per-camera $R_1$", r"spatial $R_2$")
COLOURS = ("#0072B2", "#E69F00", "#009E73")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or not report.get("final_audit_accessed")
            or report.get("selection_or_fitting_performed")):
        raise RuntimeError("sealed final-audit report is not valid")
    metrics = report["covariance"]
    nominal = np.asarray([0.50, 0.90, 0.95, 0.99])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), constrained_layout=True)
    axes[0].plot(nominal, nominal, color="#777777", lw=1.0, ls="--", label="ideal")
    for model, label, colour in zip(MODELS, LABELS, COLOURS, strict=True):
        observed = [metrics[model]["equal_position_coverage"][str(int(100 * p))]
                    for p in nominal]
        axes[0].plot(nominal, observed, marker="o", ms=4, lw=1.5,
                     color=colour, label=label)
    axes[0].set(xlabel="nominal ellipse probability", ylabel="empirical containment",
                xlim=(0.47, 1.0), ylim=(0.47, 1.0), title="(a) Calibration")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    area = [metrics[model]["equal_position_mean_ellipse_area_95_cm2"] for model in MODELS]
    bars = axes[1].bar(np.arange(3), area, color=COLOURS, width=0.66)
    axes[1].set_xticks(np.arange(3), LABELS)
    axes[1].set(ylabel=r"mean 95\% ellipse area (cm$^2$)", title="(b) Sharpness")
    for bar, value in zip(bars, area, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}",
                     ha="center", va="bottom", fontsize=7.2)
    for axis in axes:
        axis.grid(color="#dddddd", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Covariance calibration and sharpness on held-out positions",
                 fontsize=10.2, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"covariance_diagnostics.{suffix}", dpi=240,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    export = {
        "schema": "thesis_reference_final_audit_tables.v1",
        "status": "sealed_final_audit",
        "source": {"path": str(REPORT), "sha256": digest(REPORT)},
        "sample_unit": report["sample_unit"], "population": report["population"],
        "detector": report["detector"], "admission": report["admission"],
        "raw_error": report["raw_error"], "corrected_error": report["corrected_error"],
        "covariance": report["covariance"],
        "planning_information": report["planning_information"],
    }
    EXPORT.mkdir(parents=True, exist_ok=True)
    target = EXPORT / "camera_network_table.json"
    target.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
