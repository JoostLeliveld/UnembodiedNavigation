#!/usr/bin/env python3
"""Gate R0 — detection<->odometry timing and per-camera coverage.

Answers two questions before any covariance is fitted:

1. **Is the join sound?** The recorded ICRA blocker says the detection<->odometry
   join "yields no in-window pairs" and names a spawn-grid re-capture as the
   unblocking step. This measures the actual offsets.
2. **How much data is there?** Per-camera usable-detection counts are published
   *before* any fit, so a spatial claim can never be made on 15 detections.

Operational streams only; ground truth is not opened.

Outputs -> logs/studies/operational_residual_rcond/exp1_timing_and_coverage/
"""

from __future__ import annotations

import numpy as np

import rcond_common as rc


OUT = rc.OUT_ROOT / "exp1_timing_and_coverage"


def audit_capture(name: str, models, calib) -> dict:
    cap = rc.load_operational_capture(name, models=models, calib=calib)
    odom_dt = np.diff(cap.stamps)
    per_camera = {}
    for cam in rc.CAMERAS:
        detections = cap.detections[cam]
        gaps, associated = [], 0
        for det in detections:
            idx = int(np.argmin(np.abs(cap.stamps - det.stamp)))
            gap = abs(float(cap.stamps[idx]) - det.stamp)
            gaps.append(gap)
            if gap <= rc.ASSOC_TOL_S:
                associated += 1
        entry: dict[str, float | int] = {
            "detections": len(detections),
            "associated": associated,
            "dropped_outside_tolerance": len(detections) - associated,
        }
        if gaps:
            entry.update(
                {
                    "gap_median_ms": round(float(np.median(gaps)) * 1e3, 2),
                    "gap_p99_ms": round(float(np.percentile(gaps, 99)) * 1e3, 2),
                    "gap_max_ms": round(float(np.max(gaps)) * 1e3, 2),
                    "range_m_median": round(float(np.median([d.range_m for d in detections])), 3),
                }
            )
        per_camera[cam] = entry

    return {
        "capture": name,
        "odometry_steps": cap.n_steps,
        "duration_s": round(cap.duration_s, 2),
        "odometry_rate_hz": round(float(1.0 / np.median(odom_dt)), 2) if odom_dt.size else None,
        "odometry_covariance_logged": bool(np.all(np.isfinite(cap.odom_cov))),
        "association_tolerance_s": rc.ASSOC_TOL_S,
        "per_camera": per_camera,
        "total_detections": sum(v["detections"] for v in per_camera.values()),
        "total_associated": sum(v["associated"] for v in per_camera.values()),
    }


def main() -> None:
    models, calib = rc.camera_models(), rc.deployed_calibration()
    reports = [audit_capture(name, models, calib) for name in rc.CAPTURES]

    total_det = sum(r["total_detections"] for r in reports)
    total_assoc = sum(r["total_associated"] for r in reports)
    worst_median = max(
        (
            cam["gap_median_ms"]
            for r in reports
            for cam in r["per_camera"].values()
            if "gap_median_ms" in cam
        ),
        default=float("nan"),
    )

    verdict = {
        "gate": "R0",
        "captures": reports,
        "total_detections": total_det,
        "total_associated": total_assoc,
        "association_rate": round(total_assoc / total_det, 4) if total_det else None,
        "worst_per_camera_median_gap_ms": worst_median,
        "join_is_sound": bool(total_det and total_assoc / total_det > 0.95),
        "uses_ground_truth": False,
    }
    rc.write_json(OUT / "timing_and_coverage.json", verdict)

    print(f"{'capture':<26} {'cam':<9} {'det':>5} {'assoc':>6} {'med ms':>7} {'max ms':>7} {'rng m':>7}")
    for report in reports:
        for cam, entry in report["per_camera"].items():
            if not entry["detections"]:
                print(f"{report['capture']:<26} {cam:<9} {0:>5} {'-':>6} {'-':>7} {'-':>7} {'-':>7}")
                continue
            print(
                f"{report['capture']:<26} {cam:<9} {entry['detections']:>5} "
                f"{entry['associated']:>6} {entry['gap_median_ms']:>7.1f} "
                f"{entry['gap_max_ms']:>7.1f} {entry['range_m_median']:>7.2f}"
            )
    print(
        f"\ntotal {total_assoc}/{total_det} detections associated within "
        f"{rc.ASSOC_TOL_S:g}s; worst per-camera median gap {worst_median:.1f} ms"
    )
    print(f"join_is_sound = {verdict['join_is_sound']}")
    print(f"-> {OUT / 'timing_and_coverage.json'}")


if __name__ == "__main__":
    main()
