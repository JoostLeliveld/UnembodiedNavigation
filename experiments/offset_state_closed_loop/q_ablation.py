#!/usr/bin/env python3
"""Could the offset model's win just be a looser predict step?

Hypothesis under test: A5 (per-camera offset states) beats A0/A4 not because it
estimates a per-camera lean, but because it effectively runs a bigger process
noise Q, so its predict step is less confident and everything downstream looks
more honest.

Direct test: sweep Q on the BASELINE arms over two orders of magnitude and see
whether ANY value reaches A5's honesty AND A5's accuracy at the same time.

If honesty is buyable with Q alone, some row will land near NEES 1.9 with RMSE
4.8 cm. If Q only ever trades one for the other, the hypothesis is dead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
sys.path.insert(0, str(REPO / "scripts" / "shared"))
for rel in ("src/reliability", "src/unav_common", "src/state",
            "experiments/operational_residual_rcond",
            "experiments/bayesian_filter_showcase"):
    sys.path.insert(0, str(REPO / rel))

import rcond_common as rc
import exp1_graceful_vs_trusting as f1
import demo_state_space_model as d2

BASELINE_Q = f1.PROCESS_SIGMA_PER_SQRT_M          # 0.04
SWEEP = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
ARMS = ["A0_trust_everything", "A2_factorized"]


def load():
    models = rc.camera_models()
    calib = rc.deployed_calibration()
    out = []
    for name in rc.CAPTURES:
        capture = rc.load_operational_capture(name, models=models, calib=calib)
        table = rc.load_truth_table(name)
        out.append((name, capture, lambda s, _t=table: rc.truth_at(_t, s)))
    return out


def score_arm(captures, arm):
    records = []
    for _name, capture, truth in captures:
        records.extend(f1.run_arm(capture, arm, truth)["records"])
    return f1.summarize(records)


def score_offset(captures, *, sigma_prior, sigma_walk):
    records = []
    for _name, capture, truth in captures:
        records.extend(d2.run_as_ladder_arm(
            capture, truth,
            sigma_bias_prior=sigma_prior,
            sigma_bias_walk_per_sqrt_s=sigma_walk))
    return f1.summarize(records)


def row(label, s):
    return (f"{label:<34}{s['median_nees']:8.2f}{100 * s['unearned_confidence_fraction']:11.1f}"
            f"{100 * s['rmse_m']:9.1f}{100 * s['mean_stated_sigma_m']:11.1f}")


def main() -> int:
    captures = load()
    print("all three captures, exp1 protocol (filtered, per detection)\n")
    print(f"{'':<34}{'medNEES':>8}{'unearned%':>11}{'RMSE cm':>9}{'stated cm':>11}")
    print("-" * 73)

    print("\n[1] sweep the predict step on the BASELINE arms")
    for arm in ARMS:
        print(f"\n  {arm}")
        for q in SWEEP:
            f1.PROCESS_SIGMA_PER_SQRT_M = q
            s = score_arm(captures, arm)
            mark = "   <- deployed" if q == BASELINE_Q else ""
            print("  " + row(f"sigma_q = {q:.2f} /sqrt(m)", s) + mark)
    f1.PROCESS_SIGMA_PER_SQRT_M = BASELINE_Q

    print("\n[2] the arms being explained, all at the DEPLOYED Q")
    for arm in ("A0_trust_everything", "A4_correlation_floor"):
        print("  " + row(arm, score_arm(captures, arm)))
    print("  " + row("A5_offset_states", score_offset(
        captures, sigma_prior=0.05, sigma_walk=0.0016)))

    print("\n[3] does A5 even need its own Q? sweep the predict step under A5")
    for q in [0.01, 0.04, 0.16, 0.64]:
        f1.PROCESS_SIGMA_PER_SQRT_M = q
        s = score_offset(captures, sigma_prior=0.05, sigma_walk=0.0016)
        mark = "   <- deployed" if q == BASELINE_Q else ""
        print("  " + row(f"A5, sigma_q = {q:.2f}", s) + mark)
    f1.PROCESS_SIGMA_PER_SQRT_M = BASELINE_Q

    print("\n[4] and the offsets' OWN random walk, which is A5's only extra process noise")
    for walk in [0.0, 0.0004, 0.0016, 0.0064, 0.0256]:
        s = score_offset(captures, sigma_prior=0.05, sigma_walk=walk)
        mark = "   <- used" if walk == 0.0016 else ""
        print("  " + row(f"A5, offset walk = {walk:.4f}", s) + mark)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
