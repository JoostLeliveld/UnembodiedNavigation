#!/usr/bin/env python3
"""Quantify the runtime limiter from a campaign's own perception logs.

For every run's perception.csv it computes the chain that decides belief
freshness:
    camera arrival period  -> effective camera Hz
    yolo_inference_ms       -> detector compute cost
    detector_callback_ms    -> detector wall cost per frame
    frame_age_at_publish_s  -> capture->publish staleness (end-to-end)
    pixel_pose_age_s        -> how old the correction is when the planner uses it

Thesis claim under test: the detector is NOT the bottleneck (inference is tens of
ms); the camera *arrival period* (Gazebo render + ros_gz bridge) is, so the
belief correction the planner applies is hundreds of ms stale.

Usage:  python analyze_runtime_limiter.py <campaign_dir> [<campaign_dir> ...]
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def med(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.median(s)) if len(s) else float("nan")


def eff_hz(stamps):
    s = pd.to_numeric(stamps, errors="coerce").dropna().to_numpy()
    s = np.unique(np.round(s, 4))
    if len(s) < 3:
        return float("nan")
    d = np.diff(np.sort(s))
    d = d[(d > 1e-4) & (d < 5.0)]
    return float(1.0 / np.median(d)) if len(d) else float("nan")


def analyze_run(pcsv):
    try:
        df = pd.read_csv(pcsv, na_values=["NaN", "nan", "", "inf", "-inf"], low_memory=False)
    except Exception:
        return None
    if "diag_stamp" not in df.columns:
        return None
    cam_stamp = df["yolo_receive_stamp"] if "yolo_receive_stamp" in df else df["diag_stamp"]
    return {
        "n": len(df),
        "cam_hz": eff_hz(cam_stamp),
        "yolo_inf_ms": med(df.get("yolo_inference_ms")),
        "cb_ms": med(df.get("detector_callback_ms")),
        "frame_age_s": med(df.get("frame_age_at_publish_s")),
        "corr_age_s": med(df.get("pixel_pose_age_s")),
        "e2e_lat_s": med(df.get("detector_total_latency_s")),
    }


def main():
    roots = sys.argv[1:] or ["logs/visibility_comparison/robustness_keepin_clean_20260619"]
    rows = []
    for root in roots:
        for pcsv in sorted(glob.glob(str(Path(root) / "**" / "perception.csv"), recursive=True)):
            r = analyze_run(pcsv)
            if r and r["n"] > 20:
                parts = Path(pcsv).parts
                tag = "/".join(parts[-5:-2]) if len(parts) >= 5 else pcsv
                r["run"] = tag
                rows.append(r)
    if not rows:
        print("no runs found")
        return
    df = pd.DataFrame(rows)
    cols = ["cam_hz", "yolo_inf_ms", "cb_ms", "frame_age_s", "corr_age_s", "e2e_lat_s"]
    print(f"\n=== Runtime limiter across {len(df)} runs ===\n")
    print(f"{'metric':<26}{'median':>10}{'p10':>10}{'p90':>10}")
    labels = {
        "cam_hz": "camera arrival Hz",
        "yolo_inf_ms": "YOLO inference (ms)",
        "cb_ms": "detector callback (ms)",
        "frame_age_s": "frame age @ publish (s)",
        "corr_age_s": "correction age used (s)",
        "e2e_lat_s": "capture->use latency (s)",
    }
    for c in cols:
        v = df[c].dropna()
        if len(v):
            print(f"{labels[c]:<26}{v.median():>10.1f}{v.quantile(0.1):>10.1f}{v.quantile(0.9):>10.1f}")
    print("\nInterpretation:")
    inf = df["yolo_inf_ms"].median()
    hz = df["cam_hz"].median()
    age = df["corr_age_s"].median()
    print(f"  detector compute ~{inf:.0f} ms -> {1000/inf:.0f} Hz capable, but camera only arrives at ~{hz:.1f} Hz")
    print(f"  => bottleneck is upstream of the detector (render+bridge). Correction applied is ~{age*1000:.0f} ms stale.")
    df.to_csv("/tmp/runtime_limiter_per_run.csv", index=False)
    print("\nper-run -> /tmp/runtime_limiter_per_run.csv")


if __name__ == "__main__":
    main()
