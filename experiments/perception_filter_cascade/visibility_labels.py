#!/usr/bin/env python3
"""Build supervision labels for the perception quality head, from simulation truth.

Everything computed here is a LABEL. None of it may reach a runtime feature vector. The
point of computing it carefully is that the earlier analysis used a crude proxy -- the ratio
of the detected box area to the projected hull box area -- which conflates three different
things: the robot hidden behind stock, the robot running off the edge of the image, and the
detector simply drawing a slightly different box than the hull projection.

With semantic masks captured, the causes separate:

  occlusion_fraction   how much of the robot that *should* be visible in frame is missing
                       from the mask, i.e. hidden behind something
  truncation_fraction  how much of the robot's projected hull falls outside the image
  visible_fraction     mask pixels over the pixels the robot would occupy if it were
                       neither occluded nor truncated -- the combined effect

The primary training target is deliberately none of these. It is whether the resulting
measurement is usable as a localization likelihood, because that is what the robot filter
needs to know. The visibility quantities are auxiliary supervision: informative if they
generalize, and falsifiable if they turn out to have stood in for one stock arrangement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

IMG_W, IMG_H = 1280, 720
TRUE_VALUES = {'True', 'true', '1'}
USABLE_CM = 20.0

# A solid robot does not fill its own bounding box. The ratio that converts a hull bbox area
# into an expected mask pixel count is estimated from clean sightings rather than assumed.
FALLBACK_FILL_RATIO = 0.53


def clipped_area(x0: float, y0: float, x1: float, y1: float) -> float:
    cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
    cx1, cy1 = min(x1, float(IMG_W)), min(y1, float(IMG_H))
    return max(cx1 - cx0, 0.0) * max(cy1 - cy0, 0.0)


def expected_box(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(row[f'expected_{key}'])  # type: ignore[return-value]
                     for key in ('x0', 'y0', 'x1', 'y1'))
    except (KeyError, ValueError, TypeError):
        return None


def estimate_fill_ratio(rows: list[dict[str, str]]) -> float:
    """Mask pixels per unit of hull bbox area, measured on unoccluded, untruncated views.

    Truncated views are excluded so the reference is not reduced by clipping, and the upper
    decile is used so mildly occluded views cannot drag the reference down.
    """
    ratios: list[float] = []
    for row in rows:
        pixels = row.get('semantic_robot_pixels', '')
        if pixels in ('', '0', None):
            continue
        box = expected_box(row)
        if box is None:
            continue
        full = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
        if full <= 0.0 or clipped_area(*box) < 0.999 * full:
            continue
        ratios.append(int(pixels) / full)
    if len(ratios) < 20:
        return FALLBACK_FILL_RATIO
    return sorted(ratios)[int(0.90 * (len(ratios) - 1))]


def labels_for(row: dict[str, str], fill_ratio: float) -> dict[str, float] | None:
    """Occlusion, truncation and visible fraction for one camera opportunity."""
    box = expected_box(row)
    if box is None:
        return None
    full = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    if full <= 0.0:
        return None

    in_frame = clipped_area(*box)
    truncation = 1.0 - in_frame / full

    pixels_text = row.get('semantic_robot_pixels', '')
    pixels = int(pixels_text) if pixels_text not in ('', None) else 0

    expected_in_frame = fill_ratio * in_frame
    expected_total = fill_ratio * full
    # Occlusion is judged against what should have been visible IN frame, so it is not
    # contaminated by the robot hanging off the image edge.
    occlusion = (1.0 - pixels / expected_in_frame) if expected_in_frame > 0 else 1.0

    return {
        'occlusion_fraction': min(max(occlusion, 0.0), 1.0),
        'truncation_fraction': min(max(truncation, 0.0), 1.0),
        'visible_fraction': min(max(pixels / expected_total if expected_total > 0 else 0.0,
                                    0.0), 1.5),
        'mask_pixels': float(pixels),
        'line_of_sight': float(row.get('line_of_sight') or 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    index_path = args.capture / 'capture_index.csv'
    if not index_path.exists():
        raise SystemExit(f'{index_path} not found')
    with index_path.open(newline='', encoding='utf-8') as handle:
        index_rows = list(csv.DictReader(handle))

    fill_ratio = estimate_fill_ratio(index_rows)

    interp: dict[tuple[str, str], dict[str, str]] = {}
    interp_path = args.capture / 'observation_interpretations.csv'
    if interp_path.exists():
        with interp_path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                interp[(row['pose_id'], row['camera_id'])] = row

    out_rows: list[dict[str, float | int | str]] = []
    for row in index_rows:
        labels = labels_for(row, fill_ratio)
        if labels is None:
            continue
        split_label = row.get('dataset_split') or ''
        record: dict[str, float | int | str] = {
            'pose_id': row['pose_id'],
            'camera_id': row['camera_id'],
            'split': split_label.split('-')[0],
            'centre': split_label.split('_')[0],
            'line': split_label,
            **{name: round(value, 5) for name, value in labels.items()},
        }
        detection = interp.get((row['pose_id'], row['camera_id']))
        if detection is not None:
            record['detected'] = 1 if detection.get('detected') in TRUE_VALUES else 0
            if detection.get('hull_valid') in TRUE_VALUES:
                error = math.hypot(float(detection['hull_dx']),
                                   float(detection['hull_dy'])) * 100.0
                record['hull_error_cm'] = round(error, 3)
                record['usable'] = 1 if error <= USABLE_CM else 0
        out_rows.append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if out_rows:
        fields = sorted({key for row in out_rows for key in row})
        with (args.out_dir / 'visibility_labels.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out_rows)

    summary: dict[str, object] = {
        'schema': 'perception_filter_cascade_visibility_labels.v1',
        'capture': str(args.capture),
        'these_are_labels_not_runtime_features': True,
        'fill_ratio': fill_ratio,
        'rows': len(out_rows),
        'usable_threshold_cm': USABLE_CM,
    }

    scored = [row for row in out_rows if 'usable' in row]
    if scored:
        buckets: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
        for row in scored:
            key = f'{min(int(float(row["visible_fraction"]) * 5) / 5, 1.0):.1f}'
            buckets[key].append(row)
        table = {}
        for key, values in sorted(buckets.items()):
            errors = [float(row['hull_error_cm']) for row in values]
            table[key] = {
                'n': len(values),
                'usable_rate': st.mean([float(row['usable']) for row in values]),
                'median_err_cm': st.median(errors),
                'centres': len({str(row['centre']) for row in values}),
            }
        summary['visible_fraction_vs_usable'] = table

        def cause(row: dict[str, float | int | str]) -> str:
            occluded = float(row['occlusion_fraction']) > 0.2
            truncated = float(row['truncation_fraction']) > 0.02
            if occluded and truncated:
                return 'both'
            if occluded:
                return 'occluded_only'
            if truncated:
                return 'truncated_only'
            return 'neither'

        causes: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
        for row in scored:
            causes[cause(row)].append(row)
        summary['cause_split'] = {
            name: {
                'n': len(values),
                'usable_rate': st.mean([float(row['usable']) for row in values]),
                'median_err_cm': st.median([float(row['hull_error_cm']) for row in values]),
                'centres': len({str(row['centre']) for row in values}),
            }
            for name, values in sorted(causes.items())
        }

    (args.out_dir / 'visibility_labels.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    print(f'fill ratio (mask pixels per hull bbox area, clean sightings): {fill_ratio:.3f}')
    print(f'rows labelled {len(out_rows)}, with a scored detection {len(scored)}')
    if scored:
        print()
        print(f'{"visible frac":13s} {"n":>6s} {"centres":>8s} {"usable rate":>12s} '
              f'{"median err cm":>14s}')
        for key, stats in summary['visible_fraction_vs_usable'].items():  # type: ignore[index]
            print(f'{key:13s} {stats["n"]:6d} {stats["centres"]:8d} '
                  f'{stats["usable_rate"]:12.3f} {stats["median_err_cm"]:14.2f}')
        print()
        print('cause of degradation, separated by the semantic mask:')
        print(f'{"cause":16s} {"n":>6s} {"centres":>8s} {"usable rate":>12s} '
              f'{"median err cm":>14s}')
        for name, stats in summary['cause_split'].items():  # type: ignore[index]
            print(f'{name:16s} {stats["n"]:6d} {stats["centres"]:8d} '
                  f'{stats["usable_rate"]:12.3f} {stats["median_err_cm"]:14.2f}')
    print()
    print(f'wrote {args.out_dir / "visibility_labels.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
