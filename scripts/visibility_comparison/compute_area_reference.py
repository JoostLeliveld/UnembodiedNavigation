#!/usr/bin/env python3
"""Compute a simple spatial reference map for red-blob area normalization."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import CURRENT_TARGETS_DIR, LOGS_ROOT, parse_bool01, parse_float, read_csv_rows, write_manifest


def _nearest_finite_fill(values: np.ndarray, fallback: float) -> np.ndarray:
    filled = np.asarray(values, dtype=float).copy()
    finite_mask = np.isfinite(filled)
    if np.any(finite_mask):
        finite_idx = np.argwhere(finite_mask)
        finite_vals = filled[finite_mask]
        missing_idx = np.argwhere(~finite_mask)
        for iy, ix in missing_idx:
            dist2 = np.square(finite_idx[:, 0] - iy) + np.square(finite_idx[:, 1] - ix)
            nearest = int(np.argmin(dist2))
            filled[iy, ix] = float(finite_vals[nearest])
    else:
        filled[...] = float(fallback)
    filled[~np.isfinite(filled)] = float(fallback)
    return filled


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute the red-area reference map used for blob-area normalization.')
    parser.add_argument('--perception-targets', default=str(CURRENT_TARGETS_DIR / 'perception_targets.csv'))
    parser.add_argument('--out', default=str(CURRENT_TARGETS_DIR / 'area_reference.npz'))
    parser.add_argument('--percentile', type=float, default=95.0)
    parser.add_argument('--grid-nx', type=int, default=24)
    parser.add_argument('--grid-ny', type=int, default=24)
    args = parser.parse_args()

    targets_path = Path(args.perception_targets).expanduser().resolve()
    if not targets_path.is_file():
        raise RuntimeError(f'Perception targets CSV not found: {targets_path}')
    out_path = Path(args.out).expanduser().resolve()
    allowed_root = LOGS_ROOT.resolve()
    if allowed_root not in out_path.parents:
        raise RuntimeError(f'Area-reference output must stay under {allowed_root}: {out_path}')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(targets_path)
    all_xy = []
    detected_rows = []
    for row in rows:
        x = parse_float(row.get('x', ''), math.nan)
        y = parse_float(row.get('y', ''), math.nan)
        if math.isfinite(x) and math.isfinite(y):
            all_xy.append((x, y))
        if parse_bool01(row.get('red_detected', '0')) != 1:
            continue
        area = parse_float(row.get('red_area', ''), math.nan)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(area)):
            continue
        detected_rows.append((x, y, area))

    available = len(detected_rows) > 0
    if available:
        pts = np.asarray(detected_rows, dtype=float)
        all_xy_arr = np.asarray(all_xy if all_xy else pts[:, :2], dtype=float)
        xs = np.linspace(float(np.min(all_xy_arr[:, 0])), float(np.max(all_xy_arr[:, 0])), max(int(args.grid_nx), 2))
        ys = np.linspace(float(np.min(all_xy_arr[:, 1])), float(np.max(all_xy_arr[:, 1])), max(int(args.grid_ny), 2))
        area_ref = np.full((ys.size, xs.size), np.nan, dtype=float)
        x_step = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
        y_step = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
        bins: dict[tuple[int, int], list[float]] = {}
        for x, y, area in pts:
            ix = int(np.clip(round((x - float(xs[0])) / max(x_step, 1e-9)), 0, xs.size - 1))
            iy = int(np.clip(round((y - float(ys[0])) / max(y_step, 1e-9)), 0, ys.size - 1))
            bins.setdefault((iy, ix), []).append(float(area))
        for (iy, ix), values in bins.items():
            area_ref[iy, ix] = float(np.percentile(np.asarray(values, dtype=float), float(args.percentile)))
        global_ref = float(np.percentile(pts[:, 2], float(args.percentile)))
        area_ref = _nearest_finite_fill(area_ref, fallback=global_ref)
    else:
        xs = np.asarray([], dtype=float)
        ys = np.asarray([], dtype=float)
        area_ref = np.asarray([], dtype=float)
        global_ref = math.nan

    np.savez_compressed(
        out_path,
        xs=xs,
        ys=ys,
        A_ref_map=area_ref,
        available=np.asarray([1 if available else 0], dtype=np.int32),
        percentile=np.asarray([float(args.percentile)], dtype=float),
        global_ref=np.asarray([float(global_ref) if math.isfinite(global_ref) else math.nan], dtype=float),
    )
    write_manifest(out_path.with_name('area_reference_manifest.json'), {
        'perception_targets': str(targets_path),
        'output_path': str(out_path),
        'available': bool(available),
        'row_count': int(len(rows)),
        'detected_rows_used': int(len(detected_rows)),
        'percentile': float(args.percentile),
        'grid_nx': int(max(int(args.grid_nx), 2)),
        'grid_ny': int(max(int(args.grid_ny), 2)),
        'global_reference_area': float(global_ref) if math.isfinite(global_ref) else math.nan,
        'notes': [
            'A_ref(x,y) is estimated from a high percentile of detected red areas per spatial bin.',
            'Missing bins are filled from the nearest finite bin to reduce perspective-size domination without introducing a complex model.',
        ],
    })
    print(f'Wrote area reference to {out_path}')
    print(f'Available: {available}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
