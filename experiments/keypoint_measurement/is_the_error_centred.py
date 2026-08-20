#!/usr/bin/env python3
"""Is the reading's error actually centred on zero, or only small on average?

Those are different claims and only the second one is easy. A reading can have a
mean of zero overall while still being wrong in a way that depends on range or
heading — and a filter inherits the conditional error, not the average one. So
this asks three separate questions of the same readings:

1. Is the overall mean distinguishable from zero at all, given the scatter?
2. How much of the total error is offset, and how much is scatter? Only the
   offset is a candidate for correction; the scatter is what `R` is for.
3. Does the offset depend on range or heading? A term of the form
   a + b*cos(yaw) + c*sin(yaw) is fitted because that is the shape a
   heading-dependent reference-point error takes.

Run it on both readings to compare like with like.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

AXES = ('east-west', 'north-south')


def load(path: Path, visible_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r.get('detected') == '1']
    if visible_only:
        rows = [r for r in rows
                if r.get('front_labelled_visible', '1') == '1'
                and r.get('rear_labelled_visible', '1') == '1']
    if not rows:
        raise RuntimeError(f'No usable readings in {path}')
    err = np.array([[float(r['err_x_m']), float(r['err_y_m'])] for r in rows]) * 100.0
    rng = np.array([float(r['range_m']) for r in rows])
    yaw = np.array([float(r['yaw_rad']) for r in rows])
    return err, rng, yaw


def offset_and_scatter(err: np.ndarray) -> dict:
    """Split the error into a constant part and a varying part."""
    mean = err.mean(axis=0)
    scatter = err.std(axis=0, ddof=1)
    n = len(err)
    stderr = scatter / math.sqrt(n)
    offset_size = float(np.hypot(*mean))
    scatter_size = float(np.hypot(*scatter))
    total_var = float((err ** 2).mean())
    return {
        'n': n, 'mean': mean, 'scatter': scatter, 'stderr': stderr,
        'sigmas_from_zero': mean / stderr,
        'offset_size': offset_size, 'scatter_size': scatter_size,
        # What fraction of the mean squared error a perfect offset removal buys.
        'share_of_error_that_is_offset': float(offset_size ** 2 / total_var) if total_var else 0.0,
    }


def heading_and_range_structure(err: np.ndarray, rng: np.ndarray, yaw: np.ndarray) -> list[dict]:
    """Least squares: error = a + b*cos(yaw) + c*sin(yaw) + d*range."""
    design = np.column_stack([np.ones_like(rng), np.cos(yaw), np.sin(yaw), rng])
    out = []
    for axis in range(2):
        coef, *_ = np.linalg.lstsq(design, err[:, axis], rcond=None)
        fitted = design @ coef
        resid = err[:, axis] - fitted
        dof = max(len(err) - design.shape[1], 1)
        sigma2 = float(resid @ resid) / dof
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        se = np.sqrt(np.diag(cov))
        out.append({
            'axis': AXES[axis],
            'constant_cm': coef[0], 'constant_sigmas': coef[0] / se[0],
            'heading_amplitude_cm': float(math.hypot(coef[1], coef[2])),
            'heading_sigmas': float(math.hypot(coef[1] / se[1], coef[2] / se[2])),
            'range_slope_cm_per_m': coef[3], 'range_sigmas': coef[3] / se[3],
            'spread_left_cm': float(math.sqrt(sigma2)),
            'spread_before_cm': float(err[:, axis].std(ddof=1)),
        })
    return out


def report(name: str, path: Path, visible_only: bool, handle) -> None:
    err, rng, yaw = load(path, visible_only)
    stats = offset_and_scatter(err)
    handle.write(f'\n## {name}\n\n')
    handle.write(f'{stats["n"]} readings.\n\n')
    handle.write('| | offset (mean) | scatter (1 sigma) | offset in units of its own uncertainty |\n')
    handle.write('|---|---:|---:|---:|\n')
    for axis in range(2):
        handle.write(
            f'| {AXES[axis]} | {stats["mean"][axis]:+.2f} cm | {stats["scatter"][axis]:.2f} cm | '
            f'{stats["sigmas_from_zero"][axis]:+.1f} sigma |\n'
        )
    handle.write(
        f'\nOffset size **{stats["offset_size"]:.2f} cm** against scatter '
        f'**{stats["scatter_size"]:.2f} cm**. Removing the offset perfectly would remove '
        f'**{100.0*stats["share_of_error_that_is_offset"]:.1f}%** of the mean squared error.\n\n'
    )
    handle.write('Structure in the offset (is it the same everywhere?):\n\n')
    handle.write('| axis | constant | swing with heading | drift with range | scatter left after removing all three |\n')
    handle.write('|---|---:|---:|---:|---:|\n')
    for row in heading_and_range_structure(err, rng, yaw):
        handle.write(
            f'| {row["axis"]} | {row["constant_cm"]:+.2f} cm ({row["constant_sigmas"]:+.1f}s) | '
            f'{row["heading_amplitude_cm"]:.2f} cm ({row["heading_sigmas"]:.1f}s) | '
            f'{row["range_slope_cm_per_m"]:+.3f} cm/m ({row["range_sigmas"]:+.1f}s) | '
            f'{row["spread_left_cm"]:.2f} cm (was {row["spread_before_cm"]:.2f}) |\n'
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keypoint-readings', required=True)
    parser.add_argument('--bbox-readings', default='')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'RESULTS.md').open('w', encoding='utf-8') as handle:
        handle.write('# Is the error centred on zero?\n\n')
        handle.write('"s" is how many standard errors a term sits from zero. Above about 2 the '
                     'term is real; its *size* is a separate question from whether it is real, '
                     'and with several hundred readings a 2 mm effect is easily real and easily '
                     'irrelevant.\n')
        report('Marked point on the robot', Path(args.keypoint_readings), True, handle)
        if str(args.bbox_readings).strip():
            report('Bottom of the detected box (for comparison)',
                   Path(args.bbox_readings), False, handle)
    print((out_dir / 'RESULTS.md').read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
