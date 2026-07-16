#!/usr/bin/env python3
"""Assemble the Option-A suite showcase: one tile per experiment."""
from __future__ import annotations

import numpy as np
import optA_common as oc

OUT = oc.OUT_ROOT

TILES = [
    ("exp0_confidence_audit/fig1_confidence_vs_error.png",
     "Exp0 — confidence audit: tail-informative, not a metric-error proxy (PARTIAL gate)"),
    ("exp1_synthetic_gp/fig1_setup_and_gate.png",
     "Exp1 — uncertain-input GP: converges to point GP as P→0 (gate PASS); fixes calibration, not the mean"),
    ("exp2_operational_mapping/fig4_semisynthetic_sweep.png",
     "Exp2 — real data: methods tie at σ≈0.02 m; uncertainty-aware family separates from σ≈0.2–0.4 m"),
    ("exp34_init_budget/fig2_budget_curves.png",
     "Exp3/4 — priors dominate low-data regime; the single-route false-confidence trap"),
    ("exp5_trajectory_smoothing/fig2_calibration.png",
     "Exp5 — smoothing repairs covariance honesty (NEES 16.8 → 2.8), not the mean"),
    ("exp6_stress_test/fig1_inflation_curves.png",
     "Exp6 — inflation fixes local overconfidence (dynamics change), cannot fix a biased stale map"),
    ("exp7_planner_replay/fig1_interface_and_paths.png",
     "Exp7 — τ→R_plan→predicted belief through the existing planner seam"),
]


def main():
    fig = oc.plt.figure(figsize=(15.5, 17.5), dpi=115)
    rows = len(TILES) + 1
    ax = fig.add_subplot(rows, 1, 1)
    ax.axis("off")
    ax.text(0.5, 0.75, "Option A — Realistic commissioning of external-camera trust maps\nunder uncertain robot poses",
            ha="center", va="center", fontsize=15, color=oc.INK, weight="bold")
    ax.text(0.5, 0.22,
            "calibration prior → passive baseline data → detector evidence at uncertain beliefs → belief-aware GP\n"
            "→ held-out + false-confidence validation → inflation-guarded reuse → frozen deployment → R_plan seam",
            ha="center", va="center", fontsize=10, color=oc.INK2)
    for i, (rel, cap) in enumerate(TILES):
        ax = fig.add_subplot(rows, 1, i + 2)
        img = oc.plt.imread(OUT / rel)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(cap, fontsize=9.5, color=oc.INK, pad=4, loc="left")
    fig.tight_layout(h_pad=1.4)
    oc.save(fig, OUT, "SHOWCASE.png")


if __name__ == "__main__":
    main()
