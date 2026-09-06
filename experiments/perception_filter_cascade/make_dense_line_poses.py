#!/usr/bin/env python3
"""Write a predeclared pose file of densely sampled straight lines.

Purpose: separate the two error components in

    e_t = b(s_t) + q_t

`b(s)` is the smooth, spatially correlated perception/projection bias. On the frozen
0.67 m characterization grid it dominates completely, so that grid cannot measure `q`.
`q_t` is the fast component produced because a moving robot renders to a different
raster of pixels at every step, so the detector response changes slightly even though
the scene is deterministic.

The separation works by sampling far below the correlation length of `b`. Consecutive
samples on one line sit `--step-m` apart (centimetres), while `b` was measured to change
on a 3-5 m scale. Across a short window `b` is therefore almost constant plus a gentle
slope, and anything left after removing a low-order polynomial along the line is `q`.

Lines are axis-aligned so that the sampled direction is unambiguous, and each line is
repeated at several headings because the box shape (and hence the bias) depends on the
robot's orientation relative to the camera.

Poses are emitted in travel order along each line. `capture_bbox_grid.py --pose-file`
validates every entry against the world collision scene and refuses the file if any
pose is invalid, so the geometry here stays deliberately conservative.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def build_lines(
    *,
    step_m: float,
    span_m: float,
    headings: int,
    centres: list[tuple[float, float]],
    axes: tuple[str, ...],
) -> list[dict[str, object]]:
    """Return pose dicts for every (centre, axis, heading) line."""
    n_steps = int(round(span_m / step_m)) + 1
    yaws = [2.0 * math.pi * index / headings for index in range(headings)]

    poses: list[dict[str, object]] = []
    for centre_index, (cx, cy) in enumerate(centres):
        for axis in axes:
            for yaw_index, yaw in enumerate(yaws):
                for step_index in range(n_steps):
                    offset = -0.5 * span_m + step_index * step_m
                    if axis == 'x':
                        x, y = cx + offset, cy
                    else:
                        x, y = cx, cy + offset
                    poses.append({
                        'x': round(x, 6),
                        'y': round(y, 6),
                        'yaw': round(yaw, 6),
                        # `stratum` is carried through the capture as `dataset_split`,
                        # so encode the line identity to allow grouping downstream.
                        'stratum': f'line{centre_index}_{axis}_yaw{yaw_index}',
                    })
    return poses


def build_from_probe(probe: dict, *, thin: int = 1) -> list[dict[str, object]]:
    """Return pose dicts for every usable line reported by `probe_line_validity.py`.

    Each line keeps only the symmetric extent the probe proved collision-valid, so the
    capture's all-or-nothing pose-file validation cannot reject the result.
    """
    step_m = float(probe['step_m'])
    poses: list[dict[str, object]] = []
    for line in probe['lines']:
        if not line.get('usable'):
            continue
        cx, cy = (float(value) for value in line['centre'])
        axis = str(line['axis'])
        yaw = float(line['yaw'])
        half = int(line['half_steps'])
        # The split is carried in the label so the partition is fixed at capture time and
        # cannot be reassigned later to suit a result.
        split = str(line.get('split', '')) or 'unassigned'
        label = (f'{split}-c{line["centre_index"]:02d}_{axis}_yaw{line["yaw_index"]}')
        for step_index in range(-half, half + 1, thin):
            offset = step_index * step_m
            if axis == 'x':
                x, y = cx + offset, cy
            else:
                x, y = cx, cy + offset
            poses.append({
                'x': round(x, 6),
                'y': round(y, 6),
                'yaw': round(yaw, 6),
                'stratum': label,
            })
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--thin', type=int, default=1,
                        help='Keep every Nth sample along a probed line. Frames are not '
                             'independent samples, so thinning trades redundant frames '
                             'for capture time without losing place diversity.')
    parser.add_argument(
        '--auto-span', type=Path, default=None,
        help='Line probe JSON from probe_line_validity.py. Each line is emitted at the '
             'largest collision-valid span the probe found, which is the recommended path.')
    parser.add_argument(
        '--step-m', type=float, default=0.04,
        help='Spacing between consecutive samples along a line. Must be far below the '
             'measured 3-5 m correlation length of b(s).')
    parser.add_argument(
        '--span-m', type=float, default=1.2,
        help='Length of each line. Kept short so b(s) stays nearly constant across it.')
    parser.add_argument(
        '--headings', type=int, default=4,
        help='Evenly spaced yaws per line.')
    parser.add_argument(
        '--axes', default='x,y',
        help='Comma-separated line directions to sample.')
    args = parser.parse_args()

    if args.step_m <= 0.0 or args.span_m <= 0.0:
        raise SystemExit('--step-m and --span-m must be positive')
    if args.headings < 1:
        raise SystemExit('--headings must be at least 1')

    # Centres are taken from positions the frozen grid already captured, so they are known
    # collision-valid and known to be seen by 3-4 cameras with a box at all 8 headings.
    # They are spread along the camera-range axis because range is the variable that
    # separates the near-range consistency tail, so q must be reported against it.
    centres = [
        (0.0, -8.056),    # camera_B range ~2.2 m, inside the near-range tail region
        (1.338, -6.767),  # ~4.1 m
        (-7.356, -4.189), # ~8.1 m
        (10.031, -6.122), # ~12.1 m
        (10.7, 0.967),    # ~16.2 m, beyond the 16 m bias gate
    ]
    axes = tuple(part.strip() for part in args.axes.split(',') if part.strip())
    for axis in axes:
        if axis not in {'x', 'y'}:
            raise SystemExit(f'unsupported axis {axis!r}; use x and/or y')

    if args.auto_span is not None:
        probe = json.loads(args.auto_span.read_text(encoding='utf-8'))
        poses = build_from_probe(probe, thin=max(int(args.thin), 1))
        if not poses:
            raise SystemExit(f'{args.auto_span} reported no usable lines')
        step_m = float(probe['step_m']) * max(int(args.thin), 1)
        provenance: dict[str, object] = {
            'span_source': 'probe_line_validity',
            'probe_file': str(args.auto_span),
            'step_m': step_m,
            'probe_step_m': float(probe['step_m']),
            'thin': max(int(args.thin), 1),
            'lines': sorted({str(pose['stratum']) for pose in poses}),
        }
    else:
        poses = build_lines(
            step_m=args.step_m,
            span_m=args.span_m,
            headings=args.headings,
            centres=centres,
            axes=axes,
        )
        step_m = args.step_m
        provenance = {
            'span_source': 'fixed_span_argument',
            'step_m': args.step_m,
            'span_m': args.span_m,
            'headings': args.headings,
            'axes': list(axes),
            'centres': [list(centre) for centre in centres],
        }

    payload = {
        'schema': 'perception_filter_cascade_dense_lines.v1',
        'purpose': (
            'Separate smooth bias b(s) from the fast raster/detector-response component q '
            'by sampling far below the 3-5 m correlation length of b.'),
        **provenance,
        'pose_count': len(poses),
        'poses': poses,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    lines = sorted({str(pose['stratum']) for pose in poses})
    print(f'wrote {len(poses)} poses to {args.out}')
    print(f'  lines        : {len(lines)}')
    print(f'  spacing      : {step_m * 100:.0f} cm')
    print(f'  camera rows  : {len(poses) * 5}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
