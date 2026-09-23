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
    # The sharpness panel of earlier versions repeated the ellipse-area column
    # of the covariance table verbatim, so it is not drawn. Only the calibration
    # curve, which the table cannot carry, is plotted.
    nominal = np.asarray([0.50, 0.90, 0.95, 0.99])
    fig, axis = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    axes = [axis]
    axis.plot(nominal, nominal, color="#777777", lw=1.0, ls="--", label="ideal")
    for model, label, colour in zip(MODELS, LABELS, COLOURS, strict=True):
        observed = [metrics[model]["equal_position_coverage"][str(int(100 * p))]
                    for p in nominal]
        axis.plot(nominal, observed, marker="o", ms=4, lw=1.5,
                  color=colour, label=label)
    axis.set(xlabel="nominal ellipse probability", ylabel="empirical containment",
             xlim=(0.47, 1.0), ylim=(0.47, 1.0))
    axis.legend(frameon=False, fontsize=7, loc="lower right")

    for axis in axes:
        axis.grid(color="#dddddd", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
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
