#!/usr/bin/env python3
"""Held-out validation of the DEPLOYED GP reliability field against detector hit/miss.

The shipped GP (paper_artifacts/gp/warehouse_visibility_gp_v1) was fit on a separate
uniform teleport capture. This script queries its reliability field rho(x,y) at the
robot's position on every logged detector frame of the navigation campaign and scores
those predictions against the observed detection outcome (hit/miss). The campaign
trajectories are held-out data the GP never saw, so this is a genuine generalisation
test of "does the deployed reliability field predict where the camera detects the robot".

Compares the deployed GP against a constant global-rate baseline using the canonical
scorers in scripts/shared/metrics.py (Brier / log-loss / AUROC / ECE).

Usage:
  python3 scripts/visibility_comparison/validate_gp_heldout.py \
      [logs/visibility_comparison/honest_campaign_v2] \
      [paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz]
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "scripts/shared")
import metrics as M  # noqa: E402

CAMP = sys.argv[1] if len(sys.argv) > 1 else "logs/visibility_comparison/honest_campaign_v2"
GP = sys.argv[2] if len(sys.argv) > 2 else "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
OUT = "paper_artifacts/gp/warehouse_visibility_gp_v1"


def load_gp(path):
    d = np.load(path, allow_pickle=True)
    return d["xs"], d["ys"], d["P_mean_map"], d["P_conservative_plan_map"]


def query_grid(xs, ys, grid, x, y):
    """Nearest-cell lookup on the (len(ys), len(xs)) map (grid ~0.05 m, adequate)."""
    ix = np.clip(np.searchsorted(xs, x), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, y), 0, len(ys) - 1)
    return grid[iy, ix]


def load_detections(camp):
    """Pool (x, y, hit) over all campaign perception.csv frames.

    Position = robot true pose at capture (perception.csv true_x/true_y; = odom at
    capture time, within ~1-6 cm of GT, negligible at the GP's 0.9 m length scale).
    hit = `detected` (a usable robot detection was produced for that frame).
    """
    X, Y, H = [], [], []
    for pc in glob.glob(f"{camp}/**/perception.csv", recursive=True):
        with open(pc) as f:
            for r in csv.DictReader(f):
                try:
                    x = float(r["true_x"]); y = float(r["true_y"])
                except (KeyError, ValueError):
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                hit = str(r.get("detected", "")).strip() in ("1", "1.0", "True", "true")
                X.append(x); Y.append(y); H.append(1.0 if hit else 0.0)
    return np.array(X), np.array(Y), np.array(H)


def main():
    xs, ys, P_mean, P_plan = load_gp(GP)
    X, Y, hit = load_detections(CAMP)
    if hit.size == 0:
        print("no detections found under", CAMP)
        return
    rho = np.array([query_grid(xs, ys, P_mean, x, y) for x, y in zip(X, Y)])
    rho_plan = np.array([query_grid(xs, ys, P_plan, x, y) for x, y in zip(X, Y)])
    base = np.full_like(hit, hit.mean())

    print(f"\n=== Held-out validation: deployed GP rho vs detector hit/miss ===")
    print(f"campaign: {CAMP}")
    print(f"frames: {hit.size}   observed detection rate: {hit.mean():.3f}   "
          f"misses: {int((hit == 0).sum())}\n")
    print(f"{'model':22s} {'AUROC':>7s} {'Brier':>7s} {'ECE':>7s} {'logloss':>8s}")
    for name, p in [("deployed rho (P_mean)", rho),
                    ("deployed rho_plan(LCB)", rho_plan),
                    ("constant global rate", base)]:
        pc = M.clip_prob(p)
        try:
            auc = M.auroc(hit, pc)
        except Exception:
            auc = float("nan")
        print(f"{name:22s} {auc:>7.3f} {M.brier(hit, pc):>7.3f} "
              f"{M.ece(hit, pc):>7.3f} {M.logloss(hit, pc):>8.3f}")

    # reliability diagram (observed hit-rate per predicted-rho bin)
    print("\nreliability diagram (deployed rho): predicted-bin -> observed rate (n)")
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        m = (rho >= edges[i]) & (rho < edges[i + 1] if i < 9 else rho <= edges[i + 1])
        if m.sum():
            print(f"  [{edges[i]:.1f},{edges[i+1]:.1f})  obs={hit[m].mean():.3f}  n={int(m.sum())}")

    # write a small CSV artifact (predicted vs observed, binned) for provenance
    with open(f"{OUT}/heldout_reliability_bins.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho_bin_lo", "rho_bin_hi", "observed_rate", "n"])
        for i in range(10):
            m = (rho >= edges[i]) & (rho < edges[i + 1] if i < 9 else rho <= edges[i + 1])
            if m.sum():
                w.writerow([round(edges[i], 2), round(edges[i + 1], 2),
                            round(float(hit[m].mean()), 4), int(m.sum())])
    print(f"\nwrote {OUT}/heldout_reliability_bins.csv")


if __name__ == "__main__":
    main()
