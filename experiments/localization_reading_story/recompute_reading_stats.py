#!/usr/bin/env python3
"""Recompute every number this folder's Part 1 figures state, and write them down.

Folder rule 3: no new numbers without a recorded run. The figures compute their own
statistics from the two scored runs in logs/studies/keypoint_measurement/; this script
computes the same ones and records them, so a caption can be checked without rerunning
matplotlib.

Writes to logs/studies/localization_reading_story/:
  reading_stats.json          overall bias/spread split per reading, incl. by marker visibility,
                              the camera-frame anisotropy test, the geometry check and the
                              availability field (Part 2's P1-P4)
  by_world_heading.csv        north-south error per world-frame heading bin (figure L2, panel A)
  by_relative_heading.csv     error along the camera's line of sight, by relative heading (L2 B)
  by_range.csv                miss and bias per range band, split by marker visibility
                              (L4 panel C, P1, P3's floors)

Run: python3 experiments/localization_reading_story/recompute_reading_stats.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keypoint_geometry as kg  # noqa: E402
import reading_data as rd  # noqa: E402

OUT = rd.REPO_ROOT / 'logs/studies/localization_reading_story'
KEYS = ('box_bottom', 'keypoint_old_camera', 'keypoint_retrained')


def _split(err: np.ndarray) -> dict:
    mean = err.mean(axis=0)
    cov = np.cov(err - mean, rowvar=False, ddof=1)
    eig = np.sqrt(np.linalg.eigvalsh(cov))
    mag = np.hypot(err[:, 0], err[:, 1])
    return {
        'n': int(len(err)),
        'mean_east_cm': float(mean[0]), 'mean_north_cm': float(mean[1]),
        'systematic_cm': float(np.hypot(*mean)),
        'sigma_minor_cm': float(eig[0]), 'sigma_major_cm': float(eig[1]),
        'spread_cm': float(math.sqrt(np.trace(cov))),
        'correlation': float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])),
        'median_miss_cm': float(np.median(mag)),
        'p90_miss_cm': float(np.percentile(mag, 90)),
    }


def camera_frame_test(cam) -> dict:
    """Is the spread round, or an ellipse along the camera's line of sight? (P2)

    Reported in world axes the marked-point residuals look round. That is a frame
    artefact: the line of sight turns across the floor, so pooling in east/north averages
    ellipses of different orientation into a circle. The test below uses FIXED axes -- the
    sight line -- so no shape is chosen from the data, and an F test says whether the
    ratio is more than sampling noise.
    """
    out = {}
    for key in KEYS:
        r = rd.load_reading(key)
        mask = (kg.both_rendered_mask(r) if key != 'box_bottom'
                else np.ones(int(r.detected.sum()), bool))
        err = r.err_cm[mask]
        bearing = np.arctan2(r.col('y')[mask] - cam.cam_pos[1],
                             r.col('x')[mask] - cam.cam_pos[0])
        along = err[:, 0] * np.cos(bearing) + err[:, 1] * np.sin(bearing)
        across = -err[:, 0] * np.sin(bearing) + err[:, 1] * np.cos(bearing)
        n = len(err)
        va = float(np.var(along, ddof=1))
        vc = float(np.var(across, ddof=1))
        mad = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))
        out[key] = {
            'n': n,
            'sd_along_cm': math.sqrt(va), 'sd_across_cm': math.sqrt(vc),
            'ratio': math.sqrt(va / vc),
            'p_one_sided_f_test': float(1 - sps.f.cdf(va / vc, n - 1, n - 1)),
            'ratio_robust_mad': mad(along) / mad(across),
            'mean_along_cm': float(along.mean()), 'mean_across_cm': float(across.mean()),
            'world_axes_sd_east_cm': float(err[:, 0].std(ddof=1)),
            'world_axes_sd_north_cm': float(err[:, 1].std(ddof=1)),
        }
    return out


def geometry_check() -> dict:
    """Does the measured pixel scatter predict the measured floor scatter? (P4)

    Worked in the camera's frame, like the figure, so the along-sight bias is what gets
    removed as "the mean" rather than a world-axis mixture of it.
    """
    r = rd.load_reading('keypoint_retrained')
    cam = rd.camera()
    mask = kg.both_rendered_mask(r)
    pixel_cov = kg.pixel_covariance(r, mask)
    predicted = kg.predicted_floor_covariance(r, pixel_cov)[mask]
    err = r.err_cm[mask]
    bearing = np.arctan2(r.col('y')[mask] - cam.cam_pos[1],
                         r.col('x')[mask] - cam.cam_pos[0])
    rot = np.array([[[math.cos(b), math.sin(b)], [-math.sin(b), math.cos(b)]]
                    for b in bearing])
    err = np.einsum('nij,nj->ni', rot, err)
    predicted = np.einsum('nij,njk,nlk->nil', rot, predicted, rot)
    err = err - err.mean(axis=0)
    measured = np.cov(err, rowvar=False, ddof=1)
    whitened = kg.whiten(err, predicted)
    ratio = (whitened ** 2).sum(axis=1) / 2.0
    rng = r.col('range_m')[mask]
    per_band = {}
    for lo, hi in ((4.5, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 12.0)):
        sel = (rng >= lo) & (rng < hi)
        if sel.sum() >= 5:
            per_band[f'{lo}-{hi} m'] = {'n': int(sel.sum()),
                                        'measured_over_predicted': float(ratio[sel].mean())}
    return {
        'n': int(mask.sum()),
        'pixel_covariance_px2': pixel_cov.tolist(),
        'pixel_sd_px': np.sqrt(np.diag(pixel_cov)).tolist(),
        'front_rear_correlation_u': float(pixel_cov[0, 2]
                                          / math.sqrt(pixel_cov[0, 0] * pixel_cov[2, 2])),
        'measured_sd_along_cm': float(math.sqrt(measured[0, 0])),
        'measured_sd_across_cm': float(math.sqrt(measured[1, 1])),
        'predicted_median_sd_along_cm': float(math.sqrt(np.median(predicted, axis=0)[0, 0])),
        'predicted_median_sd_across_cm': float(math.sqrt(np.median(predicted, axis=0)[1, 1])),
        'measured_trace_cm2': float(np.trace(measured)),
        'predicted_mean_trace_cm2': float(np.trace(predicted, axis1=1, axis2=2).mean()),
        'mean_measured_over_predicted': float(ratio.mean()),
        'per_range_band': per_band,
    }


def availability() -> dict:
    """p_use over the floor, from the capture itself (P3)."""
    with (rd.DATASET / 'capture_diagnostics.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    cells: dict[tuple[float, float], list[int]] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        key = (round(float(row['x']), 3), round(float(row['y']), 3))
        entry = cells.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += int(row['accepted'] == '1')
        if row['accepted'] != '1':
            reasons[row['rejection_reason']] = reasons.get(row['rejection_reason'], 0) + 1
    p = np.array([ok / planned for planned, ok in cells.values()])
    return {
        'planned_poses': len(rows), 'floor_cells': len(cells),
        'headings_per_cell': max(c[0] for c in cells.values()),
        'mean_p_use': float(p.mean()),
        'cells_never_readable_pct': float(100.0 * (p == 0).mean()),
        'cells_readable_at_every_heading_pct': float(100.0 * (p == 1).mean()),
        'rejection_reasons': reasons,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cam = rd.camera()
    stats: dict = {'source': rd.SOURCE, 'dataset': rd.DATASET.name, 'readings': {}}

    world_rows, rel_rows, range_rows = [], [], []
    for key in KEYS:
        r = rd.load_reading(key)
        err = r.err_cm
        rendered = np.array([int(x['front_rendered']) + int(x['rear_rendered'])
                             for x in r.rows])[r.detected]
        entry = {
            'label': r.label,
            'poses': len(r.rows), 'read': int(r.detected.sum()),
            'found_pct': float(100.0 * r.detected.mean()),
            'all_read': _split(err),
            'both_markers_rendered': _split(err[rendered == 2]),
            'a_marker_hidden': _split(err[rendered < 2]) if (rendered < 2).sum() > 2 else None,
        }
        stats['readings'][key] = entry

        yaw = r.col('yaw_rad') % (2 * math.pi)
        bearing = np.arctan2(r.col('y') - cam.cam_pos[1], r.col('x') - cam.cam_pos[0])
        toward = -(err[:, 0] * np.cos(bearing) + err[:, 1] * np.sin(bearing))
        rel = (r.col('yaw_rad') - bearing + math.pi) % (2 * math.pi) - math.pi
        rng = r.col('range_m')

        edges = np.linspace(0.0, 2 * math.pi, 9)
        for i in range(8):
            sel = (yaw >= edges[i]) & (yaw < edges[i + 1])
            if sel.sum():
                world_rows.append({
                    'reading': key,
                    'heading_from_deg': round(math.degrees(edges[i])),
                    'heading_to_deg': round(math.degrees(edges[i + 1])),
                    'n': int(sel.sum()),
                    'mean_north_cm': round(float(err[sel, 1].mean()), 3),
                    'sem_north_cm': round(float(err[sel, 1].std(ddof=1) / math.sqrt(sel.sum())), 3),
                    'median_miss_cm': round(float(np.median(np.hypot(*err[sel].T))), 3),
                })

        redges = np.linspace(-math.pi, math.pi, 9)
        for i in range(8):
            sel = (rel >= redges[i]) & (rel < redges[i + 1])
            if sel.sum():
                rel_rows.append({
                    'reading': key,
                    'relative_heading_from_deg': round(math.degrees(redges[i])),
                    'relative_heading_to_deg': round(math.degrees(redges[i + 1])),
                    'n': int(sel.sum()),
                    'mean_towards_camera_cm': round(float(toward[sel].mean()), 3),
                    'sem_cm': round(float(toward[sel].std(ddof=1) / math.sqrt(sel.sum())), 3),
                })

        both = (np.array([int(x['front_rendered']) + int(x['rear_rendered'])
                          for x in r.rows])[r.detected] == 2)
        for lo, hi in ((4.5, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 12.0)):
            in_band = (rng >= lo) & (rng < hi)
            for subset, pick in (('all read', in_band),
                                 ('both markers rendered', in_band & both),
                                 ('a marker hidden', in_band & ~both)):
                if pick.sum() < 3:
                    continue
                e = err[pick]
                mean = e.mean(axis=0)
                cov = np.cov(e - mean, rowvar=False, ddof=1)
                range_rows.append({
                    'reading': key, 'subset': subset,
                    'range_from_m': lo, 'range_to_m': hi,
                    'n': int(pick.sum()),
                    'mean_east_cm': round(float(mean[0]), 3),
                    'mean_north_cm': round(float(mean[1]), 3),
                    'systematic_cm': round(float(np.hypot(*mean)), 3),
                    'spread_cm': round(float(math.sqrt(np.trace(cov))), 3),
                    'sigma_one_axis_cm': round(float(math.sqrt(np.trace(cov) / 2)), 3),
                    'median_miss_cm': round(float(np.median(np.hypot(*e.T))), 3),
                })

        stats['readings'][key]['miss_vs_range_correlation'] = round(
            float(np.corrcoef(np.hypot(*err.T), rng)[0, 1]), 3)

    stats['camera_frame_anisotropy'] = camera_frame_test(cam)
    stats['geometry_check'] = geometry_check()
    stats['availability'] = availability()

    (OUT / 'reading_stats.json').write_text(json.dumps(stats, indent=2) + '\n',
                                           encoding='utf-8')
    for name, rows in (('by_world_heading.csv', world_rows),
                       ('by_relative_heading.csv', rel_rows),
                       ('by_range.csv', range_rows)):
        with (OUT / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {OUT / name}  ({len(rows)} rows)')
    print(f'wrote {OUT / "reading_stats.json"}')

    aniso = stats['camera_frame_anisotropy']
    geo = stats['geometry_check']
    av = stats['availability']
    print('\ncamera-frame shape (fixed axes, so no shape is chosen from the data):')
    for key, v in aniso.items():
        print(f"  {key:22s} along {v['sd_along_cm']:.2f} across {v['sd_across_cm']:.2f} cm, "
              f"ratio {v['ratio']:.2f} (p={v['p_one_sided_f_test']:.2g}), "
              f"world axes {v['world_axes_sd_east_cm']:.2f} x "
              f"{v['world_axes_sd_north_cm']:.2f} cm")
    print(f"\ngeometry check: measured {geo['measured_trace_cm2']:.2f} cm2 vs predicted "
          f"{geo['predicted_mean_trace_cm2']:.2f} cm2, per-reading ratio "
          f"{geo['mean_measured_over_predicted']:.2f}")
    print(f"availability: {av['floor_cells']} cells x {av['headings_per_cell']} headings, "
          f"mean p_use {av['mean_p_use']:.2f}, "
          f"{av['cells_never_readable_pct']:.0f}% never readable")

    for key in KEYS:
        e = stats['readings'][key]
        a, b = e['all_read'], e['both_markers_rendered']
        print(f"\n{key}: found {e['found_pct']:.1f}% of {e['poses']} poses")
        print(f"  all read              median {a['median_miss_cm']:.2f} cm, "
              f"systematic {a['systematic_cm']:.2f} cm, spread {a['spread_cm']:.2f} cm, "
              f"sigma {a['sigma_major_cm']:.2f} x {a['sigma_minor_cm']:.2f} cm")
        print(f"  both markers rendered median {b['median_miss_cm']:.2f} cm, "
              f"systematic {b['systematic_cm']:.2f} cm, "
              f"sigma {b['sigma_major_cm']:.2f} x {b['sigma_minor_cm']:.2f} cm, "
              f"correlation {b['correlation']:+.2f}")


if __name__ == '__main__':
    main()
