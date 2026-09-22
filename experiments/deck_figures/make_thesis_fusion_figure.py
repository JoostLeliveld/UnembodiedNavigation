#!/usr/bin/env python3
"""Render the sealed final-audit multi-camera fusion result."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / (
    "logs/thesis_final_pipeline_v1/final_bayesian/"
    "final_audit_fusion_v2/report.json"
)
OUT = ROOT.parent / "papers/Thesis/figures"

LABELS = {
    "best_spatial_single": "best single",
    "equal": "equal",
    "global": "global",
    "per_camera": "per-camera",
    "spatial": "spatial",
}
COLOURS = {
    "best_spatial_single": "#777777",
    "equal": "#999999",
    "global": "#0072B2",
    "per_camera": "#E69F00",
    "spatial": "#009E73",
}


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if (report.get("status") != "complete"
            or report.get("selection_or_fitting_performed")
            or report.get("sample_unit") != "position-heading camera batch"):
        raise RuntimeError("final-audit fusion report is not valid")

    methods = tuple(LABELS)
    rmse = [100.0 * report["metrics"][name]["rmse_m"] for name in methods]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), constrained_layout=True)

    bars = axes[0].bar(np.arange(len(methods)), rmse,
                       color=[COLOURS[name] for name in methods], width=0.68)
    axes[0].set_xticks(np.arange(len(methods)), [LABELS[name] for name in methods],
                       rotation=25, ha="right")
    axes[0].set(ylabel="fused RMSE (cm)", title="(a) All matched batches")
    for bar, value in zip(bars, rmse, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.06,
                     f"{value:.2f}", ha="center", va="bottom", fontsize=6.8)

    counts = (2, 3, 4)
    compared = ("equal", "global", "per_camera", "spatial")
    x = np.arange(len(counts), dtype=float)
    width = 0.19
    for index, name in enumerate(compared):
        values = [100.0 * report["by_camera_count"][name][str(count)]["rmse_m"]
                  for count in counts]
        axes[1].bar(x + (index - 1.5) * width, values, width=width,
                    color=COLOURS[name], label=LABELS[name])
    axes[1].set_xticks(x, [str(value) for value in counts])
    axes[1].set(xlabel="admitted cameras", ylabel="fused RMSE (cm)",
                title="(b) Result by camera count")
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2)

    for axis in axes:
        axis.grid(axis="y", color="#dddddd", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.0)
        axis.title.set_fontsize(8.5)
        axis.xaxis.label.set_size(7.5)
        axis.yaxis.label.set_size(7.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        target = OUT / f"fusion_final_audit.{suffix}"
        fig.savefig(target, dpi=240, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
