#!/usr/bin/env python3
"""Report the longest collision-valid dense line at each candidate centre.

`capture_bbox_grid.py --pose-file` refuses the whole file if any single pose is
collision-invalid, so the span of each line has to be known before the file is written.
This probe calls exactly the same validity filter the capture uses
(`_filter_pose_records` with the world's collision prisms), so a line reported valid here
is accepted there.

The output is consumed by `make_dense_line_poses.py --auto-span`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for rel in ('scripts/perception', 'src/experiments', 'src/perception', 'src/unav_common'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from capture_yolo_dataset import _filter_pose_records  # noqa: E402
from experiments.core.world_profiles import load_profile  # noqa: E402
from unav_common.occlusion_geometry import parse_collision_scene_from_world  # noqa: E402


def build_filter(world: str, profiles: Path, *, wall_margin: float, region_shrink_m: float,
                 collision_clearance_m: float):
    """Return a callable mirroring the capture script's pose validity test."""
    profile, _intrinsics, resolved_world, _pose = load_profile(str(profiles), str(world))
    vis = dict(profile.get('visibility_defaults') or {})
    bounds = (
        float(vis['visibility_map_min_x']) + wall_margin,
        float(vis['visibility_map_max_x']) - wall_margin,
        float(vis['visibility_map_min_y']) + wall_margin,
        float(vis['visibility_map_max_y']) - wall_margin,
    )
    known = list(profile.get('known_2d_regions') or [])
    traversable = [item for item in known
                   if str(item.get('type', '')).strip().lower() == 'traversable']
    excluded = [item for item in known
                if str(item.get('type', '')).strip().lower() not in {'traversable', 'site_boundary'}]
    prisms = parse_collision_scene_from_world(resolved_world).prisms

    def is_valid(x: float, y: float, yaw: float) -> bool:
        xmin, xmax, ymin, ymax = bounds
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
        record = {'x': x, 'y': y, 'yaw': yaw, 'x_idx': 0, 'y_idx': 0, 'yaw_idx': 0,
                  'position_id': 0, 'dataset_split': 'probe', 'random_draw_index': ''}
        kept, _ = _filter_pose_records(
            [record],
            traversable_regions=traversable,
            excluded_regions=excluded,
            region_shrink_m=region_shrink_m,
            collision_prisms=tuple(prisms),
            collision_clearance_m=collision_clearance_m,
            camera_xy=(0.0, 0.0),
            min_camera_range_m=0.0,
            max_camera_range_m=float('inf'),
        )
        return len(kept) == 1

    return is_valid, bounds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', default='warehouse_v2.world.sdf')
    parser.add_argument('--world-profiles', type=Path,
                        default=REPO / 'src/experiments/config/world_profiles.yaml')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--step-m', type=float, default=0.04)
    parser.add_argument('--max-span-m', type=float, default=1.2)
    parser.add_argument('--min-samples', type=int, default=12,
                        help='A line shorter than this is dropped as too short to fit a trend.')
    parser.add_argument('--headings', type=int, default=4)
    parser.add_argument('--centre-plan', type=Path, default=None,
                        help='Centre plan from select_capture_centres.py. Each centre '
                             'carries its pre-assigned train/val/test partition.')
    parser.add_argument('--wall-margin', type=float, default=0.65)
    parser.add_argument('--region-shrink-m', type=float, default=0.05)
    parser.add_argument('--collision-clearance-m', type=float, default=0.25)
    args = parser.parse_args()

    is_valid, bounds = build_filter(
        args.world, args.world_profiles,
        wall_margin=args.wall_margin,
        region_shrink_m=args.region_shrink_m,
        collision_clearance_m=args.collision_clearance_m,
    )

    if args.centre_plan is not None:
        plan = json.loads(args.centre_plan.read_text(encoding='utf-8'))
        centres = [(float(item['x']), float(item['y'])) for item in plan['centres']]
        splits = [str(item.get('split', '')) for item in plan['centres']]
    else:
        # Fallback: the five centres of the first dense-line capture.
        centres = [
            (0.0, -8.056),
            (1.338, -6.767),
            (-7.356, -4.189),
            (10.031, -6.122),
            (10.7, 0.967),
        ]
        splits = [''] * len(centres)
    yaws = [2.0 * math.pi * index / args.headings for index in range(args.headings)]
    max_steps = int(round(args.max_span_m / args.step_m / 2))

    results = []
    for centre_index, (cx, cy) in enumerate(centres):
        for axis in ('x', 'y'):
            for yaw_index, yaw in enumerate(yaws):
                # Grow symmetrically from the centre while every sampled pose stays valid.
                reach = 0
                for step in range(1, max_steps + 1):
                    offset = step * args.step_m
                    dx, dy = (offset, 0.0) if axis == 'x' else (0.0, offset)
                    if (is_valid(cx + dx, cy + dy, yaw)
                            and is_valid(cx - dx, cy - dy, yaw)):
                        reach = step
                    else:
                        break
                samples = 2 * reach + 1
                results.append({
                    'centre_index': centre_index,
                    'split': splits[centre_index],
                    'centre': [cx, cy],
                    'axis': axis,
                    'yaw_index': yaw_index,
                    'yaw': round(yaw, 6),
                    'half_steps': reach,
                    'samples': samples,
                    'span_m': round(2 * reach * args.step_m, 4),
                    'usable': bool(is_valid(cx, cy, yaw) and samples >= args.min_samples),
                })

    payload = {
        'schema': 'perception_filter_cascade_line_probe.v1',
        'world': args.world,
        'step_m': args.step_m,
        'max_span_m': args.max_span_m,
        'min_samples': args.min_samples,
        'bounds': list(bounds),
        'lines': results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    usable = [row for row in results if row['usable']]
    print(f'probed {len(results)} lines, {len(usable)} usable (>= {args.min_samples} samples)')
    print(f'{"centre":22s} {"axis":5s} {"yaw_i":6s} {"samples":8s} {"span_m":7s} usable')
    for row in results:
        centre = f'({row["centre"][0]:.3f}, {row["centre"][1]:.3f})'
        print(f'{centre:22s} {row["axis"]:5s} {row["yaw_index"]:<6d} '
              f'{row["samples"]:<8d} {row["span_m"]:<7.2f} {row["usable"]}')
    print(f'\ntotal poses if all usable lines are captured: '
          f'{sum(row["samples"] for row in usable)}')
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
