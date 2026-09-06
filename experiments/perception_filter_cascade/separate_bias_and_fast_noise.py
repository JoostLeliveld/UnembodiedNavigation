#!/usr/bin/env python3
"""Split the camera observation error into a smooth part and a fast part.

The model under test is

    e_t = b(s_t) + q_t

`b(s)` is the smooth perception/projection bias. It varies on a 3-5 m scale, measured on
the frozen 0.67 m characterization grid, so it is effectively constant across one dense
line sampled at a few centimetres.

`q_t` is the fast component. A moving robot renders to a different pixel raster at every
step, so the detector response changes slightly from sample to sample even though the
simulator is deterministic and free of motion blur and timing jitter.

Method. Along each dense line, fit a low-order polynomial in arc length to each observed
quantity and treat the fit as `b` and the residual as `q`. The polynomial absorbs the
smooth trend and its curvature; anything varying faster than the polynomial can follow is
left in the residual. Two guards keep this honest:

  * The polynomial order is swept. If the reported `q` keeps shrinking as the order rises,
    the split is soaking up real signal and the number is not trustworthy.
  * A lag-1 autocorrelation is reported for the residual. A genuinely fast, near-white `q`
    has autocorrelation near zero; a residual that is still strongly correlated means the
    trend model has not captured `b` and the split has failed.

Both the pixel-space quantities (what a bbox filter would act on) and the ground-plane
error (what the localization EKF ultimately sees) are reported, because the filter's value
depends on the fast fraction in the space where it operates.

Outputs a JSON summary and a CSV of per-sample residuals into the study directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


TRUE_VALUES = {'True', 'true', '1'}


def polyfit(xs: list[float], ys: list[float], order: int) -> list[float]:
    """Least-squares polynomial coefficients via normal equations, lowest order first."""
    size = order + 1
    matrix = [[sum(x ** (row + col) for x in xs) for col in range(size)] for row in range(size)]
    target = [sum(y * x ** row for x, y in zip(xs, ys)) for row in range(size)]

    # Gauss-Jordan with partial pivoting; the systems here are tiny.
    augmented = [matrix[row][:] + [target[row]] for row in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError('singular normal equations')
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        for other in range(size):
            if other == col:
                continue
            factor = augmented[other][col] / divisor
            for term in range(col, size + 1):
                augmented[other][term] -= factor * augmented[col][term]
    return [augmented[row][size] / augmented[row][row] for row in range(size)]


def polyval(coeffs: list[float], x: float) -> float:
    return sum(coeff * x ** power for power, coeff in enumerate(coeffs))


def lag1_autocorr(values: list[float]) -> float:
    """Lag-1 autocorrelation. Near zero indicates a white residual."""
    if len(values) < 3:
        return float('nan')
    mean = st.mean(values)
    centred = [value - mean for value in values]
    denom = sum(value * value for value in centred)
    if denom <= 0.0:
        return float('nan')
    numer = sum(centred[index] * centred[index + 1] for index in range(len(centred) - 1))
    return numer / denom


def robust_sd(values: list[float]) -> float:
    """MAD-based scale, so a few outliers cannot inflate the reported fast component."""
    if len(values) < 2:
        return float('nan')
    median = st.median(values)
    mad = st.median([abs(value - median) for value in values])
    return mad / 0.6745


KEY_FIELDS = ('pose_id', 'position_id', 'heading_id', 'repetition_id', 'camera_id')


def load_rows(capture: Path) -> list[dict[str, str]]:
    """Load the derived interpretations, ensuring each row carries its line label.

    `dataset_split` holds the line identity. Captures written before that column was
    added to the derived schema lack it, so it is joined back from `capture_index.csv`
    on the shared observation key rather than being assumed present.
    """
    path = capture / 'observation_interpretations.csv'
    if not path.exists():
        raise SystemExit(
            f'{path} not found. Run run_bbox_detector.py then derive_interpretations.py first.')
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if rows and not rows[0].get('dataset_split'):
        index_path = capture / 'capture_index.csv'
        if not index_path.exists():
            raise SystemExit(
                f'{path} has no dataset_split column and {index_path} is missing, so line '
                'membership cannot be recovered.')
        with index_path.open(newline='', encoding='utf-8') as handle:
            labels = {tuple(row[field] for field in KEY_FIELDS): row.get('dataset_split', '')
                      for row in csv.DictReader(handle)}
        for row in rows:
            row['dataset_split'] = labels.get(tuple(row[field] for field in KEY_FIELDS), '')
    return rows


def group_lines(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Group detected rows by (camera, line label), ordered along the line."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get('detected') not in TRUE_VALUES:
            continue
        label = row.get('dataset_split') or ''
        if not label.startswith('line'):
            continue
        groups[(row['camera_id'], label)].append(row)

    ordered: dict[tuple[str, str], list[dict[str, str]]] = {}
    for key, items in groups.items():
        axis = 'x' if '_x_' in key[1] else 'y'
        column = 'robot_x' if axis == 'x' else 'robot_y'
        ordered[key] = sorted(items, key=lambda row: float(row[column]))
    return ordered


# Quantities to decompose: pixel-space is where a bbox filter would act, ground-plane
# error is what the localization EKF receives.
PIXEL_SIGNALS = ('u_bbox_bottom', 'v_bbox_bottom')
GROUND_SIGNALS = ('raw_dx', 'raw_dy')


def decompose(groups: dict[tuple[str, str], list[dict[str, str]]], *, order: int,
              min_samples: int) -> tuple[dict[str, dict[str, object]], list[dict[str, object]],
                                        dict[str, dict[str, dict[str, float]]]]:
    residuals: dict[str, list[float]] = defaultdict(list)
    autocorrs: dict[str, list[float]] = defaultdict(list)
    trend_spans: dict[str, list[float]] = defaultdict(list)
    per_sample: list[dict[str, object]] = []

    for (camera, label), items in sorted(groups.items()):
        if len(items) < min_samples:
            continue
        axis = 'x' if '_x_' in label else 'y'
        column = 'robot_x' if axis == 'x' else 'robot_y'
        coords = [float(row[column]) for row in items]
        origin = coords[0]
        arc = [coord - origin for coord in coords]

        for signal in PIXEL_SIGNALS + GROUND_SIGNALS:
            if signal not in items[0]:
                continue
            raw: list[float] = []
            usable: list[int] = []
            for index, row in enumerate(items):
                text = row.get(signal, '')
                if text in ('', 'nan', 'None'):
                    continue
                raw.append(float(text))
                usable.append(index)
            if len(raw) < min_samples:
                continue
            scale = 100.0 if signal in GROUND_SIGNALS else 1.0  # metres -> cm
            values = [value * scale for value in raw]
            positions = [arc[index] for index in usable]
            try:
                coeffs = polyfit(positions, values, order)
            except ValueError:
                continue
            fitted = [polyval(coeffs, position) for position in positions]
            resid = [value - fit for value, fit in zip(values, fitted)]

            residuals[signal].extend(resid)
            autocorrs[signal].append(lag1_autocorr(resid))
            trend_spans[signal].append(max(fitted) - min(fitted))

            for index, position, value, fit, res in zip(usable, positions, values, fitted, resid):
                row = items[index]
                per_sample.append({
                    'camera_id': camera,
                    'line': label,
                    'axis': axis,
                    'arc_m': round(position, 4),
                    'signal': signal,
                    'observed': round(value, 5),
                    'trend': round(fit, 5),
                    'residual': round(res, 5),
                    'camera_range_m': row.get('camera_range_m', ''),
                    'confidence': row.get('confidence', ''),
                    'robot_x': row.get('robot_x', ''),
                    'robot_y': row.get('robot_y', ''),
                    'robot_yaw': row.get('robot_yaw', ''),
                })

    summary: dict[str, dict[str, object]] = {}
    for signal, values in residuals.items():
        finite_ac = [value for value in autocorrs[signal] if not math.isnan(value)]
        summary[signal] = {
            'unit': 'cm' if signal in GROUND_SIGNALS else 'px',
            'n': len(values),
            'n_lines': len(autocorrs[signal]),
            'sd': st.pstdev(values) if len(values) > 1 else float('nan'),
            'robust_sd': robust_sd(values),
            'median_abs': st.median([abs(value) for value in values]),
            'mean': st.mean(values),
            'lag1_autocorr_median': st.median(finite_ac) if finite_ac else float('nan'),
            'trend_span_median': st.median(trend_spans[signal]) if trend_spans[signal] else float('nan'),
        }

    # Pooling hides the thing that matters. The fast component is pixel noise pushed
    # through the projection Jacobian, so in centimetres on the floor it must grow with
    # range even when the pixel-space component is constant. Reporting one pooled number
    # would average a near-camera and a far-camera reading into a value describing
    # neither, so both breakdowns are carried alongside.
    grouped: dict[str, dict[str, list[float]]] = {
        'by_camera': defaultdict(list), 'by_range_bucket': defaultdict(list)}
    for row in per_sample:
        signal = str(row['signal'])
        key_camera = f'{signal}|{row["camera_id"]}'
        grouped['by_camera'][key_camera].append(float(row['residual']))
        text = str(row.get('camera_range_m', ''))
        if text not in ('', 'nan'):
            low = int(float(text) // 4) * 4
            grouped['by_range_bucket'][f'{signal}|{low}-{low + 4}m'].append(
                float(row['residual']))

    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for name, buckets in grouped.items():
        breakdown[name] = {}
        for key, values in sorted(buckets.items()):
            if len(values) < min_samples:
                continue
            breakdown[name][key] = {
                'n': len(values),
                'robust_sd': robust_sd(values),
                'median_abs': st.median([abs(value) for value in values]),
            }

    return summary, per_sample, breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--order', type=int, default=2,
                        help='Polynomial order treated as the smooth b(s) trend.')
    parser.add_argument('--order-sweep', default='1,2,3,4',
                        help='Orders to sweep as a sensitivity guard.')
    parser.add_argument('--min-samples', type=int, default=8)
    args = parser.parse_args()

    rows = load_rows(args.capture)
    groups = group_lines(rows)
    if not groups:
        raise SystemExit('no dense-line detections found; was the capture run with dense_lines.json?')

    summary, per_sample, breakdown = decompose(
        groups, order=args.order, min_samples=args.min_samples)

    sweep: dict[str, dict[str, dict[str, object]]] = {}
    for order in (int(part) for part in args.order_sweep.split(',') if part.strip()):
        sweep_summary, _, _ = decompose(groups, order=order, min_samples=args.min_samples)
        sweep[str(order)] = {
            signal: {'robust_sd': stats['robust_sd'],
                     'lag1_autocorr_median': stats['lag1_autocorr_median']}
            for signal, stats in sweep_summary.items()
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    residual_csv = args.out_dir / 'line_residuals.csv'
    if per_sample:
        with residual_csv.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample)

    detected = sum(1 for row in rows if row.get('detected') in TRUE_VALUES)
    payload = {
        'schema': 'perception_filter_cascade_bq_split.v1',
        'capture': str(args.capture),
        'rows_total': len(rows),
        'rows_detected': detected,
        'lines_used': len(groups),
        'trend_order': args.order,
        'min_samples': args.min_samples,
        'summary': summary,
        'breakdown': breakdown,
        'order_sweep': sweep,
    }
    (args.out_dir / 'bq_split.json').write_text(json.dumps(payload, indent=2) + '\n',
                                                encoding='utf-8')

    print(f'capture   : {args.capture}')
    print(f'rows      : {len(rows)} total, {detected} detected')
    print(f'lines used : {len(groups)} (camera x line), trend order {args.order}')
    print()
    header = (f'{"signal":16s} {"unit":5s} {"n":>6s} {"robust_sd":>10s} {"sd":>9s} '
              f'{"med|q|":>8s} {"lag1_ac":>8s} {"trend_span":>11s}')
    print(header)
    print('-' * len(header))
    for signal, stats in summary.items():
        print(f'{signal:16s} {stats["unit"]:5s} {stats["n"]:6d} {stats["robust_sd"]:10.3f} '
              f'{stats["sd"]:9.3f} {stats["median_abs"]:8.3f} '
              f'{stats["lag1_autocorr_median"]:8.3f} {stats["trend_span_median"]:11.3f}')

    print()
    print('order sweep (robust_sd | lag1 autocorrelation)')
    signals = sorted(summary)
    print(f'{"order":6s} ' + ' '.join(f'{signal:>22s}' for signal in signals))
    for order in sorted(sweep, key=int):
        cells = []
        for signal in signals:
            stats = sweep[order].get(signal)
            cells.append('n/a'.rjust(22) if stats is None
                         else f'{stats["robust_sd"]:10.3f} | {stats["lag1_autocorr_median"]:+7.3f}')
        print(f'{order:6s} ' + ' '.join(cells))

    for name, title in (('by_range_bucket',
                         'fast component by camera range (the axis that matters most)'),
                        ('by_camera', 'fast component by camera')):
        rows_out = breakdown.get(name, {})
        if not rows_out:
            continue
        print()
        print(title)
        print(f'  {"group":34s} {"n":>6s} {"robust_sd":>10s} {"median|q|":>10s}')
        for key, stats in rows_out.items():
            print(f'  {key:34s} {stats["n"]:6d} {stats["robust_sd"]:10.3f} '
                  f'{stats["median_abs"]:10.3f}')

    print()
    print(f'wrote {args.out_dir / "bq_split.json"}')
    if per_sample:
        print(f'wrote {residual_csv} ({len(per_sample)} residual rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
