#!/usr/bin/env python3
"""Render the WP5 calibration-drift detection figure from the probe summary.

Reads logs/studies/single_camera_uigp_reliability/wp5_self_monitoring/summary.json
(produced by wp5_drift_detection.py) and plots the real go/no-go result: continuous
health + hard-reject onset vs injected calibration-drift magnitude, with the
zero-false-alarm healthy baseline. Reuses the repo house figure style (optA_common)
so it matches the other Paper-2 figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "optionA_commissioning"))
import optA_common as oc  # noqa: E402  house style (SURFACE/INK/colors, save)
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / "logs/studies/single_camera_uigp_reliability/wp5_self_monitoring"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    sweep = summary["sweep"]
    beta = [r["beta_px"] for r in sweep]
    deg = [r["approx_yaw_deg"] for r in sweep]
    health = [r["health_post_end"] for r in sweep]
    fracnis = [r["frac_nis_over_gate_post"] for r in sweep]
    degraded = [r["detected_degraded"] for r in sweep]
    fa = summary["nominal_false_alarm"]
    thresh = summary.get("detection_threshold_px")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    oc.style_ax(ax, title="WP5 — innovation health monitor vs controlled calibration drift")
    ax.plot(beta, health, "-o", color=oc.BLUE, lw=2, ms=4, label="continuous health h (drift half)")
    ax.plot(beta, fracnis, "-s", color=oc.RED, lw=2, ms=4, label="fraction NIS > 9.21 gate (drift half)")
    ax.axhline(0.5, color=oc.MUTED, lw=0.8, ls=":", zorder=1)

    # mark the hard-reject (DEGRADED) onset
    deg_betas = [b for b, d in zip(beta, degraded) if d]
    if deg_betas:
        b0 = min(deg_betas)
        ax.axvline(b0, color=oc.GREEN, lw=1.2, ls="--", zorder=2)
        ax.annotate(f"hard DEGRADED reject\n≥ {b0:.0f} px (~{[d for b, d in zip(beta, deg) if b == b0][0]:.1f}°)",
                    xy=(b0, 0.55), xytext=(b0 + 0.4, 0.62), fontsize=7.2, color=oc.GREEN)

    ax.set_xlabel("injected calibration-drift bias (px)")
    ax.set_ylabel("health  /  fraction over NIS gate")
    ax.set_ylim(-0.03, 1.03)
    oc.badge(ax, f"healthy false-alarm = {fa:.3f} over {summary['n']} frames", loc="upper right")
    ax.legend(loc="center left", fontsize=7.4)

    # secondary approx-degree axis
    secax = ax.secondary_xaxis("top", functions=(lambda x: x / summary["validation"]["focal_px"] * 57.2958,
                                                 lambda d: d * summary["validation"]["focal_px"] / 57.2958))
    secax.set_xlabel("≈ camera yaw drift (deg)", fontsize=7.5, color=oc.INK2)

    path = oc.save(fig, OUT, "fig_wp5_drift_detection.png")
    print("wrote", path)


if __name__ == "__main__":
    main()
