#!/usr/bin/env python3
"""Quantitative summary of ONE experiment run: timing, projection, update, drift.

Diagnostics-first: trusts no module. Reads the existing perception.csv +
experiment.csv + run_summary.json and reports hard numbers so we can see where
the camera->belief pipeline loses accuracy and throughput.

Usage:
    python analyze_run.py <run_dir> [<run_dir> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _pctl(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def _stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return "n=0"
    return (f"mean={x.mean():.3f} med={np.median(x):.3f} "
            f"p90={np.percentile(x,90):.3f} p99={np.percentile(x,99):.3f} "
            f"max={x.max():.3f} n={x.size}")


def load_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path, na_values=["NaN", "nan", "", "inf", "-inf"],
                       low_memory=False)


def first_cmd_time(exp):
    """Time of first nonzero commanded velocity."""
    if exp is None or "cmd_v" not in exp:
        return None
    moving = exp[(exp["cmd_v"].abs() > 1e-4) | (exp.get("cmd_w", 0).abs() > 1e-4)]
    if moving.empty:
        return None
    return float(moving["stamp"].iloc[0])


def analyze(run_dir: Path):
    run_dir = Path(run_dir)
    perc = load_csv(run_dir / "perception.csv")
    exp = load_csv(run_dir / "experiment.csv")
    summary = {}
    sp = run_dir / "run_summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())

    print("=" * 100)
    print(f"RUN: {run_dir}")
    print(f"  outcome: {summary.get('completion_reason','?')}  "
          f"valid={summary.get('valid_run','?')}  "
          f"crashed={summary.get('crashed','?')}  "
          f"reason={summary.get('collision_reason','')}")

    t0 = first_cmd_time(exp)
    if t0 is not None:
        print(f"  first nonzero cmd at t={t0:.1f}s")

    # ---------- TIMING (perception.csv = one row per processed frame) ----------
    if perc is not None and len(perc) > 2:
        ds = perc["diag_stamp"].to_numpy(float)
        ds = ds[np.isfinite(ds)]
        span = ds.max() - ds.min() if ds.size > 1 else float("nan")
        n_frames = len(perc)
        n_det = int((perc["detected"] == 1).sum()) if "detected" in perc else -1
        dts = np.diff(np.sort(ds))
        dts = dts[(dts > 0) & (dts < 5)]
        print("\n  [TIMING] (perception.csv: 1 row = 1 processed frame)")
        print(f"    frames={n_frames}  detections={n_det}  span={span:.1f}s")
        if span > 0:
            print(f"    EFFECTIVE detector throughput = {n_frames/span:.2f} Hz   "
                  f"detection rate = {n_det/span:.2f} Hz")
        if dts.size:
            print(f"    inter-frame dt: med={np.median(dts)*1e3:.0f}ms "
                  f"-> instantaneous {1.0/np.median(dts):.2f} Hz  "
                  f"(p90 dt={np.percentile(dts,90)*1e3:.0f}ms)")
        for col, label in [("yolo_inference_ms", "yolo_inference_ms"),
                           ("detector_callback_ms", "detector_callback_ms"),
                           ("frame_age_at_publish_s", "frame_age_at_publish_s (STALENESS)"),
                           ("detector_total_latency_s", "detector_total_latency_s")]:
            if col in perc:
                print(f"    {label:42s}: {_stat(perc[col])}")
        # phase split: global-solve (before first cmd) vs driving (after)
        if t0 is not None and "diag_stamp" in perc:
            for phase, mask in [("global-solve (t<t0)", perc["diag_stamp"] < t0),
                                ("DRIVING   (t>=t0)", perc["diag_stamp"] >= t0)]:
                sub = perc[mask]
                if len(sub) > 2:
                    s = np.sort(sub["diag_stamp"].dropna().to_numpy(float))
                    sp = s.max() - s.min()
                    rate = len(sub) / sp if sp > 0 else float("nan")
                    inf = sub["yolo_inference_ms"].dropna()
                    age = sub["frame_age_at_publish_s"].dropna()
                    print(f"    {phase}: rate={rate:.2f}Hz  inf_med={np.median(inf):.0f}ms  "
                          f"stale_med={np.median(age):.2f}s  n={len(sub)}")

    # ---------- PROJECTION accuracy (perception.csv, detected rows) ----------
    if perc is not None and "detected" in perc:
        det = perc[perc["detected"] == 1]
        print(f"\n  [PROJECTION] (detected frames n={len(det)}; error vs truth, metres)")
        for col, label in [("localization_error_m", "raw projection (pred_world)"),
                           ("localization_error_calibrated_m", "AFFINE-calibrated (=/state/bev path)"),
                           ("localization_error_captime_m", "capture-time-truth (latency removed)"),
                           ("state_error_captime_m", "state node captime")]:
            if col in det:
                print(f"    {label:40s}: {_stat(det[col])}")
        if "camera_relative_bearing_deg" in det and "localization_error_calibrated_m" in det:
            # periphery bias: correlate error with bearing magnitude
            b = det["camera_relative_bearing_deg"].abs().to_numpy(float)
            e = det["localization_error_calibrated_m"].to_numpy(float)
            m = np.isfinite(b) & np.isfinite(e)
            if m.sum() > 5:
                near = e[m][b[m] < np.median(b[m])]
                far = e[m][b[m] >= np.median(b[m])]
                print(f"    periphery check: err near-bearing={np.nanmean(near):.3f}m "
                      f"vs far-bearing={np.nanmean(far):.3f}m")

    # ---------- UPDATE sanity (experiment.csv pixel correction ledger) -------
    if exp is not None and "planner_pixel_correction_available" in exp:
        corr = exp[exp["planner_pixel_correction_available"] == 1]
        print(f"\n  [UPDATE] pixel corrections logged n={len(corr)}")
        if len(corr):
            acc = corr["pixel_corr_accepted"]
            n_acc = int((acc == 1).sum())
            print(f"    accepted={n_acc}/{len(corr)} ({100*n_acc/len(corr):.0f}%)")
            if "pixel_corr_reject_reason" in corr:
                rej = corr[corr["pixel_corr_accepted"] != 1]["pixel_corr_reject_reason"]
                if len(rej):
                    print("    reject reasons:", dict(rej.value_counts()))
            for col, label in [("pixel_corr_nis", "NIS (thresh~9.21)"),
                               ("pixel_corr_age_s", "applied correction age_s (STALENESS)"),
                               ("pixel_corr_xy_update_norm_m", "xy update norm m"),
                               ("pixel_corr_innov_u", "innov_u (px)"),
                               ("pixel_corr_innov_v", "innov_v (px)")]:
                if col in corr:
                    print(f"    {label:40s}: {_stat(corr[col])}")

    # ---------- DRIFT (experiment.csv) ----------
    if exp is not None and "belief_error_odom_m" in exp:
        print("\n  [DRIFT] truth vs belief/state (metres, radians)")
        for col, label in [("belief_error_odom_m", "truth-belief pos err"),
                           ("state_error_odom_m", "truth-state pos err")]:
            if col in exp:
                print(f"    {label:32s}: {_stat(exp[col])}")
                if t0 is not None:
                    after = exp[exp["stamp"] >= t0][col]
                    print(f"    {'  (after first cmd)':32s}: {_stat(after)}")
        for col, label in [("yaw_error_odom_map_vs_belief_rad", "truth-belief YAW err (rad)"),
                           ("yaw_error_odom_map_vs_state_rad", "truth-state YAW err (rad)"),
                           ("yaw_error_odom_map_vs_odom_rad", "truth-odom YAW err (rad)")]:
            if col in exp:
                v = exp[col].abs()
                print(f"    |{label:31s}|: {_stat(v)}  (deg med={np.degrees(_pctl(v,50)):.1f})")
                if t0 is not None:
                    after = exp[exp["stamp"] >= t0][col].abs()
                    print(f"    {'  (after first cmd, deg)':32s}: "
                          f"med={np.degrees(_pctl(after,50)):.1f} "
                          f"p90={np.degrees(_pctl(after,90)):.1f} "
                          f"max={np.degrees(_pctl(after,100)):.1f}")

        # occluded vs visible drift split using p_vis_plan_eff
        for vcol in ("p_vis_plan_eff", "p_vis_plan"):
            if vcol in exp and "belief_error_odom_m" in exp:
                d = exp.dropna(subset=[vcol, "belief_error_odom_m"])
                if len(d) > 10:
                    vis = d[d[vcol] >= 0.5]["belief_error_odom_m"]
                    occ = d[d[vcol] < 0.5]["belief_error_odom_m"]
                    print(f"\n  [VIS SPLIT by {vcol}>=0.5] "
                          f"VISIBLE err med={_pctl(vis,50):.3f}m (n={len(vis)})  "
                          f"OCCLUDED err med={_pctl(occ,50):.3f}m (n={len(occ)})")
                    break

        # heading drift attribution during turns
        if {"planner_diag_odom_delta_theta", "planner_diag_cmd_delta_theta", "cmd_w"}.issubset(exp.columns):
            turning = exp[exp["cmd_w"].abs() > 0.1]
            if len(turning) > 5:
                od = turning["planner_diag_odom_delta_theta"].abs()
                cd = turning["planner_diag_cmd_delta_theta"].abs()
                print(f"  [HEADING in turns |cmd_w|>0.1, n={len(turning)}] "
                      f"odom_dtheta med={_pctl(od,50):.4f} cmd_dtheta med={_pctl(cd,50):.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        analyze(Path(d))
