#!/usr/bin/env python3
"""Consolidate the Tier-1 single-cam results into paper figures.

Reads the CSVs produced by the real-Gazebo runs and renders three figures:
  fig1_health_trace   — HEALTHY -> DEGRADED on real perception (detection pilot)
  fig2_detection_envelope — C1 detectability surface (severity x onset -> latency)
  fig3_regime_belief  — health-gated rejection recovers accuracy where the fixed gate fails

Usage: python3 make_tier1_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
STUD = REPO / "logs" / "studies" / "single_camera_uigp_reliability"
PILOT = STUD / "tier1_pilot_1784728217" / "health_trace.csv"
ENV = STUD / "tier1_envelope_1784729404" / "envelope.csv"
PRED = STUD / "tier1_predictive_1784750174" / "predictive.csv"

C = {"B1_nis_gate": "#8a8a8a", "B2_degraded": "#1f6fb2", "B2p_early": "#7bb6de"}
LBL = {"B1_nis_gate": "B1 fixed NIS gate", "B2_degraded": "B2 health-gated (DEGRADED)",
       "B2p_early": "B2p health-gated (early h<0.5)"}


def _rows(path):
    return list(csv.DictReader(open(path)))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def fig_health_trace(out):
    rows = _rows(PILOT)
    t0 = _f(rows[0]["t_wall"])
    t = np.array([_f(r["t_wall"]) - t0 for r in rows])
    h = np.array([_f(r["h"]) for r in rows])
    nis = np.array([_f(r["nis"]) for r in rows])
    deg = np.array([int(_f(r["degraded"])) for r in rows])
    onset = 18.0  # fault-after-s in the pilot
    first_deg = next((t[i] for i in range(len(deg)) if deg[i] == 1), None)

    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax1.axvspan(0, onset, color="#e8f4e8", label="_healthy")
    if first_deg is not None:
        ax1.axvspan(first_deg, t[-1], color="#fdeaea", label="_degraded")
    ax1.plot(t, h, color="#1f6fb2", lw=2, label="health $h$")
    ax1.axhline(0.5, color="#1f6fb2", ls=":", lw=1, alpha=0.6)
    ax1.set_ylabel("health $h$", color="#1f6fb2")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("time since first detection (s)")
    ax1.axvline(onset, color="k", ls="--", lw=1)
    ax1.text(onset + 0.3, 0.9, "drift onset", fontsize=8, rotation=0)
    if first_deg is not None:
        ax1.axvline(first_deg, color="#c0392b", ls="--", lw=1)
        ax1.text(first_deg + 0.3, 0.6, f"DEGRADED\n(+{first_deg-onset:.0f}s)", fontsize=8, color="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(t, np.clip(nis, 0, 30), color="#c0392b", lw=1, alpha=0.5, label="NIS")
    ax2.axhline(9.21, color="#c0392b", ls=":", lw=1, alpha=0.6)
    ax2.set_ylabel("NIS (clipped @30)", color="#c0392b")
    ax2.set_ylim(0, 32)
    ax1.set_title("Health monitor on real perception: HEALTHY $\\to$ DEGRADED after a calibration drift")
    fig.tight_layout()
    p = out / "fig1_health_trace.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_detection_envelope(out):
    rows = _rows(ENV)
    sevs = sorted({_f(r["severity"]) for r in rows})
    onsets = sorted({_f(r["onset"]) for r in rows})
    lat = np.full((len(sevs), len(onsets)), np.nan)
    detected = np.zeros_like(lat, dtype=bool)
    for r in rows:
        i, j = sevs.index(_f(r["severity"])), onsets.index(_f(r["onset"]))
        if r["reached_degraded"] == "True":
            detected[i, j] = True
            lat[i, j] = _f(r["degraded_latency_s"])

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    masked = np.ma.masked_invalid(lat)
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap="viridis_r",
                   extent=[-0.5, len(onsets) - 0.5, -0.5, len(sevs) - 0.5])
    for i in range(len(sevs)):
        for j in range(len(onsets)):
            if detected[i, j]:
                ax.text(j, i, f"{lat[i, j]:.0f}s", ha="center", va="center", color="w", fontsize=9)
            else:
                ax.text(j, i, "none", ha="center", va="center", color="#c0392b", fontsize=9)
    ax.set_xticks(range(len(onsets)))
    ax.set_xticklabels([f"{o:.0f} s" for o in onsets])
    ax.set_yticks(range(len(sevs)))
    ax.set_yticklabels([f"{s:.2f}" for s in sevs])
    ax.set_xlabel("fault onset (route progress / coverage)")
    ax.set_ylabel("drift severity — aim slide (m)")
    ax.set_title("C1 detectability envelope\n(latency onset$\\to$DEGRADED; 'none' = undetected)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("detection latency (s)")
    fig.tight_layout()
    p = out / "fig2_detection_envelope.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_regime_belief(out):
    rows = _rows(PRED)
    conds = ["B1_nis_gate", "B2_degraded", "B2p_early"]
    sevs = sorted({_f(r["severity"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.0))
    width = 0.25
    xbase = np.arange(len(sevs))
    for k, cond in enumerate(conds):
        means, pts, goals = [], [], []
        for s in sevs:
            ms = [r for r in rows if _f(r["severity"]) == s and r["condition"] == cond]
            be = [_f(r["mean_belief_error_gt_m"]) for r in ms if not math.isnan(_f(r["mean_belief_error_gt_m"]))]
            means.append(np.mean(be) if be else np.nan)
            pts.append(be)
            goals.append(sum(1 for r in ms if r["goal_reached"] == "True"))
        xs = xbase + (k - 1) * width
        ax.bar(xs, means, width, color=C[cond], label=LBL[cond], edgecolor="k", lw=0.5)
        for xi, be, g in zip(xs, pts, goals):
            ax.scatter([xi] * len(be), be, color="k", s=10, zorder=3, alpha=0.6)
            ax.text(xi, (np.nanmean(be) if be else 0) + 0.03, f"{g}/5\ngoals", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xbase)
    ax.set_xticklabels([f"{s:.1f} m\n(moderate)" if s == 0.5 else f"{s:.1f} m\n(gross)" for s in sevs])
    ax.set_xlabel("calibration-drift severity")
    ax.set_ylabel("mean localization error vs GT (m)")
    ax.set_title("Health-gated rejection recovers accuracy where the fixed gate fails\n"
                 "(dots = per-seed, n=5)")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    p = out / "fig3_regime_belief.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUD / "tier1_consolidated" / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = [fig_health_trace(out), fig_detection_envelope(out), fig_regime_belief(out)]
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
