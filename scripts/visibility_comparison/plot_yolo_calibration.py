#!/usr/bin/env python3
"""Plot empirical YOLO confidence reliability against Oracle visible ground truth."""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from common import CURRENT_GP_DIR, LOGS_ROOT

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gp-dir', default=str(CURRENT_GP_DIR))
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    gp_dir = Path(args.gp_dir).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve() if str(args.out).strip() else (gp_dir / 'plots').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_file = gp_dir / 'yolo_confidence_gp.npz'
    oracle_file = gp_dir / 'oracle_visibility_gp.npz'

    if not yolo_file.is_file() or not oracle_file.is_file():
        print(f"Skipping calibration plot. Missing {yolo_file} or {oracle_file}")
        return 0

    with np.load(yolo_file, allow_pickle=False) as data:
        yolo_map = np.asarray(data['P_map'], dtype=float)
    
    with np.load(oracle_file, allow_pickle=False) as data:
        oracle_map = np.asarray(data['P_map'], dtype=float)

    if yolo_map.shape != oracle_map.shape:
        print("Shape mismatch. Cannot directly correlate without interpolation.")
        return 0

    yolo_flat = yolo_map.flatten()
    oracle_flat = oracle_map.flatten()

    # Bin the YOLO confidences
    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    empirical_prob = np.zeros_like(bin_centers)
    counts = np.zeros_like(bin_centers)

    for i in range(len(bins) - 1):
        mask = (yolo_flat >= bins[i]) & (yolo_flat < bins[i+1])
        if i == len(bins) - 2:  # Include exactly 1.0 in the last bin
            mask = (yolo_flat >= bins[i]) & (yolo_flat <= 1.0)
        
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            empirical_prob[i] = np.mean(oracle_flat[mask])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax1.scatter(yolo_flat, oracle_flat, alpha=0.01, color='blue', s=1)
    ax1.set_xlabel('YOLO Confidence (Planner P_map)')
    ax1.set_ylabel('Oracle Geometric Visibility')
    ax1.set_title('Raw Correlation')

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    valid = counts > 5
    ax2.plot(bin_centers[valid], empirical_prob[valid], 'mo-', linewidth=2, label='Empirical Oracle Visibility')
    
    ax2_twin = ax2.twinx()
    ax2_twin.bar(bin_centers, counts, width=0.08, color='gray', alpha=0.2, label='Map Cell Count')
    ax2_twin.set_ylabel('Number of map cells')

    ax2.set_xlabel('YOLO Confidence Proxy')
    ax2.set_ylabel('Mean Oracle Visibility Fraction')
    ax2.set_title('YOLO Calibration Curve')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    out_file = output_dir / 'yolo_confidence_calibration.png'
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    print(f"Wrote YOLO calibration curve to {out_file}")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
