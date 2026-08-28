#!/usr/bin/env python3
"""Two defects, two fixes, and the sensor that was there all along.

    python3 experiments/anomaly_investigation/before_and_after.py

Scores each correction ONCE, at the moment it is published. Scoring a held correction against
a moving robot measures the robot's travel, not the sensor -- that artefact is what made the
p95 look like 151 cm.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "deck_figures"))
import style as D                                        # noqa: E402

OUT = D.REPO / "logs/studies/anomaly_investigations"
RUNS = (("neither fix", "drives", D.MUTED),
        ("correction carried\nto the time it is used", "drives_timestamped", D.OLD),
        ("...and the admission\ncheck switched on", "drives_gated", D.ROBOT),
        ("...and the residual\ndeclared along travel", "drives_floor", D.GOOD))


def honesty(pattern):
    """Fraction of samples where the truth sits inside the belief's own 95% ellipse."""
    inside = []
    for f in sorted(glob.glob(str(D.REPO / f"logs/studies/fusion_on_fixed_routes/{pattern}"
                                  "/fusion_network_traverse/*/seed0/experiment_*/experiment.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                cov = np.array([[float(r["planner_cov_x"]), float(r["planner_cov_xy"])],
                                [float(r["planner_cov_xy"]), float(r["planner_cov_y"])]])
                res = np.array([float(r["gt_x"]) - float(r["planner_belief_x"]),
                                float(r["gt_y"]) - float(r["planner_belief_y"])])
            except (TypeError, ValueError, KeyError):
                continue
            if np.linalg.det(cov) <= 0 or not np.all(np.isfinite(res)):
                continue
            inside.append(float(res @ np.linalg.solve(cov, res)) <= 5.991)
    return float(np.mean(inside)) * 100 if inside else float("nan")


def corrections(pattern):
    out, gaps = [], []
    for f in sorted(glob.glob(str(D.REPO / f"logs/studies/fusion_on_fixed_routes/{pattern}"
                                  "/fusion_network_traverse/*/seed0/experiment_*/experiment.csv"))):
        rows = list(csv.DictReader(open(f)))

        def col(k):
            return np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan
                             for r in rows])

        st, err = col("state_stamp"), col("state_error_gt_m") * 100
        ok = np.isfinite(st) & np.isfinite(err)
        landed = np.concatenate([[True], np.diff(st) > 1e-9]) & ok
        out.append(err[landed])
        unique = np.unique(st[np.isfinite(st)])
        if unique.size > 2:
            gaps.append(np.diff(unique))
    return (np.concatenate(out) if out else np.array([])), \
           (np.concatenate(gaps) if gaps else np.array([]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {label: corrections(pat) for label, pat, _c in RUNS}
    data = {k: v for k, v in data.items() if v[0].size}
    if not data:
        raise SystemExit("no drives to compare yet")

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.0), constrained_layout=True)
    ax = axes[0]
    labels, colours = list(data), [c for _l, _p, c in RUNS if _l in data]
    med = [float(np.median(data[k][0])) for k in labels]
    p95 = [float(np.percentile(data[k][0], 95)) for k in labels]
    x = np.arange(len(labels))
    ax.bar(x, med, 0.55, color=colours, label="median")
    ax.bar(x, np.array(p95) - np.array(med), 0.55, bottom=med, color=colours, alpha=0.35,
           label="up to the 95th percentile")
    for i, (m, p) in enumerate(zip(med, p95)):
        ax.text(i, p * 0.92, f"{m:.2f} cm\np95 {p:.0f}", ha="center", va="top", fontsize=12,
                color=D.INK)
    ax.set_ylim(0, max(p95) * 1.12)
    ax.axhline(1.44, color=D.INK, lw=1.6, ls=(0, (5, 3)))
    ax.text(len(labels) - 0.45, 2.2, "commissioning: 1.44 cm", ha="right", fontsize=12,
            color=D.INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylabel("error of one published correction (cm)", fontsize=12.5)
    ax.grid(True, axis="y", color="#eeede8", lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11.5, loc="upper right")
    ax.set_title("The sensor was always this good", loc="left", fontsize=16, color=D.INK)

    ax = axes[1]
    for (label, _pat, colour) in RUNS:
        if label not in data:
            continue
        gaps = data[label][1]
        if gaps.size:
            ax.hist(gaps, bins=np.logspace(-1.2, 1.3, 40), color=colour, alpha=0.55,
                    label=f"{label.splitlines()[0]} — worst {gaps.max():.1f} s")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time between corrections (s, log)", fontsize=12.5)
    ax.set_ylabel("intervals (log — the rare long ones are the point)", fontsize=12.5)
    ax.grid(True, color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("What the check costs: absent measurements", loc="left", fontsize=16,
                 color=D.INK)

    fig.suptitle("Two defects, two fixes", x=0.004, ha="left", fontsize=20, color=D.INK)
    fig.text(0.004, -0.045,
             "Left: each correction scored once, at the moment it is published. Carrying the "
             "correction to the time it is used takes the median; the admission check takes the "
             "tail. Together they land on the accuracy commissioning measured on a stationary "
             "robot.\n"
             "Right: the check refuses roughly a third of detections, so corrections become "
             "less frequent and outages get longer. A refused reading is an ABSENT measurement, "
             "not a bad one — which is the availability side of the problem, not the accuracy "
             "side.\nPrimary route, hull arms.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(OUT / "03_two_defects_two_fixes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = {label: {"belief_honesty_pct": round(honesty(dict(
                           (l, p) for l, p, _c in RUNS)[label]), 1),
                       "n": int(v[0].size),
                       "median_cm": round(float(np.median(v[0])), 2),
                       "p95_cm": round(float(np.percentile(v[0], 95)), 2),
                       "worst_cm": round(float(v[0].max()), 1),
                       "worst_gap_s": round(float(v[1].max()), 1) if v[1].size else None}
               for label, v in data.items()}
    (OUT / "before_and_after.json").write_text(json.dumps(summary, indent=2) + "\n")
    for k, v in summary.items():
        print(f"{k.replace(chr(10), ' '):46s} median {v['median_cm']:5.2f} cm  p95 "
              f"{v['p95_cm']:6.2f}  worst {v['worst_cm']:6.1f}  worst gap {v['worst_gap_s']}s")
    print(f"wrote {OUT.relative_to(D.REPO)}/03_two_defects_two_fixes.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
