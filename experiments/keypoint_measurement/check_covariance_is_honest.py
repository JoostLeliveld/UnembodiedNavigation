#!/usr/bin/env python3
"""Does the geometric covariance match the errors the detector actually makes?

A reading is only usable by the filter if its stated uncertainty is honest. The
covariance in ``state.core.marker_keypoint_reading`` is not fitted to the floor
errors — it is pixel noise pushed through the geometry — so it can be checked
against them, which is the whole point.

Given the per-sample readings that ``evaluate_keypoint_model.py`` wrote:

1. measure the detector's pixel spread (the one fitted number),
2. state a covariance for every reading from geometry alone,
3. ask how far the truth actually fell in units of that covariance.

If the answer is 1, the reading can go straight into the filter. Above 1 it is
overconfident and needs a floor; below 1 it is needlessly cautious.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/state', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_keypoint_model import cameras_from_manifest  # noqa: E402
from state.core.marker_keypoint_reading import read_marker_keypoints  # noqa: E402

# How far the truth is allowed to be, in units of what the filter claims. A
# 2-degree-of-freedom chi-square has median 1.386, so this is the honest level.
HONEST_MEDIAN_CHI2 = 1.386


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--readings', required=True,
                        help='per_sample.csv from evaluate_keypoint_model.py')
    parser.add_argument('--out', required=True)
    parser.add_argument('--visible-only', action='store_true', default=True,
                        help='Use only readings where both markers actually rendered.')
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    manifest = json.loads((dataset / 'capture_manifest.json').read_text(encoding='utf-8'))
    plane_z = float(manifest.get('keypoint_marker_world_z', 0.210))
    geom = manifest['marker_geometry']
    front_x, rear_x = float(geom['front_x']), float(geom['rear_x'])
    cameras = cameras_from_manifest(manifest, 1280, 720)
    default_camera = next(iter(cameras))

    with Path(args.readings).expanduser().resolve().open(encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r.get('detected') == '1']
    if args.visible_only:
        rows = [r for r in rows
                if r.get('front_labelled_visible') == '1' and r.get('rear_labelled_visible') == '1']
    if not rows:
        raise RuntimeError('No usable readings in that per_sample.csv')

    # Step 1: the detector's pixel spread, about its own mean, across all four
    # coordinates. Spread and not RMS, because a constant offset belongs in the
    # observation model, not in the noise.
    residuals = np.array([
        [float(r['res_front_u']), float(r['res_front_v']),
         float(r['res_rear_u']), float(r['res_rear_v'])]
        for r in rows
    ])
    pixel_bias = residuals.mean(axis=0)
    pixel_sigma = float(residuals.std(ddof=1))
    centred_sigma = float((residuals - pixel_bias).std(ddof=1))

    # Steps 2 and 3.
    records = []
    for row in rows:
        camera = cameras[row.get('camera') or default_camera]
        reading = read_marker_keypoints(
            camera,
            (float(row['pred_front_u']), float(row['pred_front_v'])),
            (float(row['pred_rear_u']), float(row['pred_rear_v'])),
            front_offset_x=front_x, rear_offset_x=rear_x,
            plane_z=plane_z, pixel_sigma=pixel_sigma,
        )
        if reading is None:
            continue
        truth = np.array([float(row['x']), float(row['y'])])
        error = np.array(reading.xy_m) - truth
        cov = np.array(reading.covariance_xy_m2)
        try:
            chi2 = float(error @ np.linalg.solve(cov, error))
        except np.linalg.LinAlgError:
            continue
        true_yaw = float(row['yaw_rad'])
        yaw_err = (reading.heading_rad - true_yaw + math.pi) % (2.0 * math.pi) - math.pi
        records.append({
            'x': truth[0], 'y': truth[1], 'yaw_rad': true_yaw,
            'range_m': float(row['range_m']),
            'separation_px': reading.marker_separation_px,
            'err_x_cm': 100.0 * error[0], 'err_y_cm': 100.0 * error[1],
            'stated_sigma_x_cm': 100.0 * math.sqrt(cov[0, 0]),
            'stated_sigma_y_cm': 100.0 * math.sqrt(cov[1, 1]),
            'chi2': chi2,
            'err_yaw_deg': math.degrees(yaw_err),
            'stated_sigma_yaw_deg': math.degrees(math.sqrt(reading.heading_variance_rad2)),
        })

    if not records:
        raise RuntimeError('Every reading failed to invert')

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'per_reading.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    chi2 = np.array([r['chi2'] for r in records])
    err = np.array([[r['err_x_cm'], r['err_y_cm']] for r in records])
    stated = np.array([[r['stated_sigma_x_cm'], r['stated_sigma_y_cm']] for r in records])
    yaw_err = np.array([r['err_yaw_deg'] for r in records])
    yaw_stated = np.array([r['stated_sigma_yaw_deg'] for r in records])
    rng = np.array([r['range_m'] for r in records])
    times_too_confident = math.sqrt(float(np.median(chi2)) / HONEST_MEDIAN_CHI2)

    with (out_dir / 'RESULTS.md').open('w', encoding='utf-8') as handle:
        handle.write('# Is the keypoint reading\'s stated uncertainty honest?\n\n')
        handle.write(f'- {len(records)} readings, both markers rendered\n')
        handle.write(f'- detector pixel spread used for the covariance: '
                     f'**{pixel_sigma:.2f} px** per coordinate '
                     f'({centred_sigma:.2f} px about its own mean)\n')
        handle.write(f'- detector pixel offset, which the covariance does NOT model: '
                     f'({pixel_bias[0]:+.2f}, {pixel_bias[1]:+.2f}) front, '
                     f'({pixel_bias[2]:+.2f}, {pixel_bias[3]:+.2f}) rear px\n\n')
        handle.write('## The headline\n\n')
        handle.write(f'**The truth lands {times_too_confident:.2f}x further away than the '
                     f'reading claims it should** (1.00 would be honest; above 1 is '
                     f'overconfident).\n\n')
        handle.write(f'- actual error: {np.hypot(*err.T).mean():.2f} cm mean, '
                     f'{np.median(np.hypot(*err.T)):.2f} cm median\n')
        handle.write(f'- stated 1 sigma: {stated[:,0].mean():.2f} cm across, '
                     f'{stated[:,1].mean():.2f} cm along\n')
        handle.write(f'- heading: actual {np.abs(yaw_err).mean():.1f} deg mean error vs '
                     f'stated {yaw_stated.mean():.1f} deg\n\n')
        handle.write('## Does it stay honest with range?\n\n')
        handle.write('A single number can hide a covariance that is right on average and wrong '
                     'everywhere. Position and heading are separated because heading precision '
                     'scales with apparent marker separation, so it should degrade faster.\n\n')
        handle.write('| range to camera | n | actual error | stated 1 sigma | times too confident | '
                     'heading error | stated heading 1 sigma |\n')
        handle.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for lo, hi in ((0, 3), (3, 5), (5, 7), (7, 9), (9, 20)):
            sel = (rng >= lo) & (rng < hi)
            if not sel.any():
                continue
            band_chi2 = math.sqrt(float(np.median(chi2[sel])) / HONEST_MEDIAN_CHI2)
            handle.write(
                f'| {lo}-{hi} m | {int(sel.sum())} | '
                f'{np.median(np.hypot(*err[sel].T)):.2f} cm | '
                f'{stated[sel].mean():.2f} cm | {band_chi2:.2f}x | '
                f'{np.median(np.abs(yaw_err[sel])):.1f} deg | '
                f'{yaw_stated[sel].mean():.1f} deg |\n'
            )
        handle.write('\n## What this means for the filter\n\n')
        if times_too_confident <= 1.3:
            handle.write('The stated covariance is close enough to the truth that the reading '
                         'can be handed to the filter as it stands.\n')
        else:
            handle.write(f'The reading is overconfident by {times_too_confident:.2f}x. The pixel '
                         f'spread alone does not account for the error, so something the geometry '
                         f'does not model is left over — check the pixel offset above before '
                         f'reaching for an inflation factor.\n')

    print((out_dir / 'RESULTS.md').read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
