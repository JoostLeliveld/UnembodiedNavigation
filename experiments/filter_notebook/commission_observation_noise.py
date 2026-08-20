#!/usr/bin/env python3
"""Commission the per-camera observation noise `R`, on the projection path in force.

The notebook needs a baseline `R` to compare a learned one against. The figures it
used to carry were fitted through the `projection_calibration_v2` along-bearing
corrections, which were removed from the runtime in August 2026 when plain inverse
perspective mapping beat every fitted variant. Comparing against them measured the
retired projection as much as it measured `R`.

This re-derives them from the same recorded pixels, back-projected with the deployed
parameter-free homography, `camera.pixel_to_world(u, v)`.

**Held out on purpose.** The commissioning drives are never the drive the notebook then
analyses. A covariance fitted on the run it is scored against would be guaranteed to look
good and would prove nothing. In the single-camera AWS world the three commissioning
drives are also three DIFFERENT routes, so what is held out is a viewing geometry and not
just a different stretch of the same line.

**Two covariances, and the choice matters.** For each camera the residual
`y - x_true` has a non-zero mean, because the detector reports the bottom of a
silhouette rather than the wheel contact point:

* `R_spread` is the covariance *about that mean* -- how much the observations scatter.
* `R_total` is the second moment *about zero*, `E[(y-x)(y-x)^T]` -- scatter plus offset.

A filter whose model says `y = x + noise` with zero-mean noise is entitled to the
second one. `R_spread` describes a quantity the filter never sees, because it has no
term for the mean and so cannot subtract it. Commissioning honestly for the model as
written therefore means handing over `R_total`, and this script reports both so the
difference is on the record rather than assumed.

Ground truth is used here. That is what commissioning is: an offline calibration step
against a reference, done once, before deployment. What is forbidden is a *filter*
consuming truth at run time, and nothing here does.

    python3 commission_observation_noise.py                      # the four-camera world
    python3 commission_observation_noise.py --world warehouse_aws \
        --captures aws_apron_west_to_east aws_aisle_west_north aws_mid_cross_east
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import notebook_data as nd  # noqa: E402

TRUTH_TOL_S = 0.05


def residuals_for(name: str, models) -> dict[str, np.ndarray]:
    """Per-camera (y - truth) residuals in metres, on the current projection path."""
    capture = nd.load_capture(name, models=models)
    truth = nd.load_truth(name)
    out: dict[str, list] = {cam: [] for cam in nd.CAMERAS}
    for cam in nd.CAMERAS:
        for detection in capture.detections[cam]:
            hit = nd.truth_at(truth, detection.stamp, tol_s=TRUTH_TOL_S)
            if hit is None:
                continue
            out[cam].append((detection.world[0] - hit[0], detection.world[1] - hit[1]))
    return {cam: np.asarray(rows, dtype=float) for cam, rows in out.items() if rows}


def summarise(res: np.ndarray) -> dict:
    mean = res.mean(axis=0)
    centred = res - mean
    r_spread = centred.T @ centred / max(len(res) - 1, 1)
    r_total = res.T @ res / len(res)               # second moment about zero
    return {
        "n": int(len(res)),
        "mean_offset_m": [float(mean[0]), float(mean[1])],
        "offset_magnitude_m": float(np.hypot(*mean)),
        "R_spread": r_spread.tolist(),
        "R_total": r_total.tolist(),
        "sigma_spread_m": float(math.sqrt(np.trace(r_spread) / 2.0)),
        "sigma_total_m": float(math.sqrt(np.trace(r_total) / 2.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=nd.FULL_4CAM.key, choices=sorted(nd.WORLDS))
    parser.add_argument("--captures", nargs="+", default=None,
                        help="run tags to fit on; default is the four-camera "
                             "commissioning set")
    parser.add_argument("--out", default=None, help="override the output file name")
    args = parser.parse_args()

    world = nd.use_world(args.world)
    captures = list(args.captures or nd.COMMISSIONING_CAPTURES)
    out_path = nd.STUDY_ROOT / (args.out or world.commissioned_file)
    print(f"commissioning {world.key} ({world.description}) on {len(captures)} drive(s)\n")

    models = nd.camera_models()

    pooled: dict[str, list] = {cam: [] for cam in nd.CAMERAS}
    per_capture = {}
    for name in captures:
        try:
            found = residuals_for(name, models)
        except (FileNotFoundError, KeyError) as exc:
            print(f"  {name}: unreadable ({exc})")
            continue
        per_capture[name] = {cam: int(len(res)) for cam, res in found.items()}
        for cam, res in found.items():
            pooled[cam].append(res)
        counts = ", ".join(f"{cam.replace('camera_', '')}={len(res)}"
                           for cam, res in sorted(found.items()))
        print(f"  {name}: {counts}")

    print()
    commissioned = {}
    for cam in nd.CAMERAS:
        if not pooled[cam]:
            print(f"  {cam}: NO observations across the commissioning captures -- "
                  f"cannot commission, will fall back to the pooled value")
            continue
        commissioned[cam] = summarise(np.vstack(pooled[cam]))

    if not commissioned:
        raise SystemExit("no camera had any commissioning observations")

    # A camera with no commissioning data still has to be given something, and the
    # pooled estimate over the others is the least-assuming choice available.
    everything = np.vstack([r for rows in pooled.values() for r in rows])
    pooled_summary = summarise(everything)
    for cam in nd.CAMERAS:
        commissioned.setdefault(cam, dict(pooled_summary, source="pooled_fallback"))

    print(f"  {'camera':8s}{'n':>7s}{'offset cm':>11s}{'spread cm':>11s}{'total cm':>10s}"
          f"{'  inflation':>12s}")
    for cam in nd.CAMERAS:
        s = commissioned[cam]
        print(f"  {cam.replace('camera_', ''):8s}{s['n']:>7d}"
              f"{100 * s['offset_magnitude_m']:>11.2f}{100 * s['sigma_spread_m']:>11.2f}"
              f"{100 * s['sigma_total_m']:>10.2f}"
              f"{s['sigma_total_m'] / s['sigma_spread_m']:>11.2f}x"
              + ("   (pooled fallback)" if s.get("source") == "pooled_fallback" else ""))
    s = pooled_summary
    print(f"  {'pooled':8s}{s['n']:>7d}{100 * s['offset_magnitude_m']:>11.2f}"
          f"{100 * s['sigma_spread_m']:>11.2f}{100 * s['sigma_total_m']:>10.2f}"
          f"{s['sigma_total_m'] / s['sigma_spread_m']:>11.2f}x")

    payload = {
        "projection": "camera.pixel_to_world -- plain inverse perspective mapping, zero fitted parameters",
        "world": world.key,
        "commissioning_captures": [str(nd.capture_root(name)) for name in captures],
        "held_out": "every drive not listed above, including the one the notebook analyses",
        "observations_per_capture": per_capture,
        "per_camera": commissioned,
        "pooled": pooled_summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
