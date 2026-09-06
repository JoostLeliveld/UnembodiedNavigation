#!/usr/bin/env python3
"""Predeclare P1 transects from warehouse and camera geometry only.

The selector deliberately has no detector, bounding-box, residual, or learned-model input.
It searches the declared traversable rectangles for 1.2 m lines, labels their frozen
ray-cast visibility/range properties, and greedily covers the requested camera/range and
visibility-boundary strata.  Every selected line is then expanded to 0.15 m positions and
eight headings for the all-camera capture program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for rel in (
    'src/experiments', 'src/unav_common', 'scripts/perception',
    'experiments/perception_filter_cascade', 'experiments/warehouse_v2_sketches',
):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from experiments.core.world_profiles import load_profile  # noqa: E402
from probe_line_validity import build_filter  # noqa: E402
from coverage import coverage, grid  # noqa: E402
from warehouse_v2 import build  # noqa: E402

#: The robot's chassis skirt. The hull spans 0-0.40 m, but only the wheels and casters touch
#: the floor; the continuous body starts here, and this is the height whose occlusion moves
#: the detected box bottom off the true contact row.
ROBOT_BASE_Z_M = 0.09


CAMERA_NAMES = tuple('ABCDE')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    lane: str
    axis: str
    centre: tuple[float, float]
    points: tuple[tuple[float, float], ...]
    covers: frozenset[str]
    visibility: tuple[tuple[int, ...], ...]
    range_bins: tuple[str, ...]
    overlap_positions: int

    @property
    def key(self) -> tuple:
        return (self.lane, self.axis, self.centre[0], self.centre[1])


def _footprint_partial_fields(layout, xs, ys, *, half_extent_m: float) -> dict[str, np.ndarray]:
    """Per-camera boolean grid: the robot's contact edge is clipped, but it is still seen.

    A partial view is the robot PARTLY hidden -- enough of the body visible to be detected,
    with the contact edge occluded so the box bottom no longer marks the ground. Testing one
    point at the centre cannot express that. This tests the base height at the centre and at
    four footprint offsets; a cell is partial when the centre's base is visible and at least
    one offset's is not.

    Uses only occluder geometry, camera calibration and the robot's declared footprint.
    """
    import coverage as _cov

    height_map = _cov.height_map(layout, 'A', xs, ys)
    step_x = float(xs[1] - xs[0])
    step_y = float(ys[1] - ys[0])
    shift_x = int(round(half_extent_m / step_x))
    shift_y = int(round(half_extent_m / step_y))

    def visible_at(cam, z):
        previous = _cov.MARKER_Z
        _cov.MARKER_Z = z
        try:
            return _cov.visible_from(cam, height_map, xs, ys)
        finally:
            _cov.MARKER_Z = previous

    fields: dict[str, np.ndarray] = {}
    for name, camera in zip(CAMERA_NAMES, layout.cameras, strict=True):
        model = _cov.make_cam(camera)
        base = visible_at(model, ROBOT_BASE_Z_M)
        clipped = np.zeros_like(base)
        for dx, dy in ((shift_x, 0), (-shift_x, 0), (0, shift_y), (0, -shift_y)):
            offset = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
            clipped |= base & ~offset
        fields[name] = clipped
    return fields


def _nearest(values: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(values - float(value))))


def _declared_driveable_mask(xs: np.ndarray, ys: np.ndarray, lanes: list[dict]) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys)
    mask = np.zeros(gx.shape, dtype=bool)
    for lane in lanes:
        mask |= (
            (gx >= float(lane['xmin'])) & (gx <= float(lane['xmax']))
            & (gy >= float(lane['ymin'])) & (gy <= float(lane['ymax']))
        )
    return mask


def _range_edges(layout, xs, ys, per_cam, drive) -> dict[str, tuple[float, float]]:
    gx, gy = np.meshgrid(xs, ys)
    result = {}
    for name, cam, visible in zip(CAMERA_NAMES, layout.cameras, per_cam, strict=True):
        distance = np.hypot(gx - float(cam.x), gy - float(cam.y))
        values = distance[drive & visible]
        if len(values) < 3:
            raise RuntimeError(f'camera {name} has too few visible driveable cells')
        result[name] = tuple(float(v) for v in np.quantile(values, [1 / 3, 2 / 3]))
    return result


def _bin(value: float, edges: tuple[float, float]) -> str:
    if value <= edges[0]:
        return 'near'
    if value <= edges[1]:
        return 'mid'
    return 'far'


def _candidate_centres(lane: dict, *, span_m: float, centre_step_m: float):
    xmin, xmax = float(lane['xmin']), float(lane['xmax'])
    ymin, ymax = float(lane['ymin']), float(lane['ymax'])
    half = 0.5 * span_m
    # Search on several cross-lane offsets. This makes shelf-edge sightline
    # boundaries eligible without using observed localization errors.
    for axis in ('x', 'y'):
        lo, hi = (xmin + half, xmax - half) if axis == 'x' else (ymin + half, ymax - half)
        if hi < lo - 1e-9:
            continue
        along = np.arange(lo, hi + 0.5 * centre_step_m, centre_step_m)
        cross_lo, cross_hi = (ymin, ymax) if axis == 'x' else (xmin, xmax)
        width = cross_hi - cross_lo
        fractions = (0.25, 0.5, 0.75) if width >= 0.9 else (0.5,)
        for a in along:
            for fraction in fractions:
                cross = cross_lo + fraction * width
                yield (float(a), float(cross)) if axis == 'x' else (float(cross), float(a)), axis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', default='warehouse_v2.world.sdf')
    parser.add_argument('--world-profiles', type=Path,
                        default=REPO / 'src/experiments/config/world_profiles.yaml')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--step-m', type=float, default=0.15)
    parser.add_argument('--span-m', type=float, default=1.20)
    parser.add_argument('--headings', type=int, default=8)
    parser.add_argument('--centre-step-m', type=float, default=0.30)
    parser.add_argument('--max-lines', type=int, default=18)
    parser.add_argument('--range-cover-count', type=int, default=1)
    parser.add_argument('--transition-cover-count', type=int, default=1)
    parser.add_argument('--overlap-cover-count', type=int, default=1)
    # Experiment 37's spacing_freeze rule needs >=3 transects per camera partial-view
    # stratum. The pilot ran with the default 1 everywhere and four of five cameras came
    # back starved (4-20 position pairs at 0.15 m against the 20 required), so this
    # stratum defaults to the count the analysis actually asks for.
    parser.add_argument('--partial-view-cover-count', type=int, default=3)
    parser.add_argument('--footprint-half-m', type=float, default=0.35,
                        help='Half-extent used for the footprint occlusion test; the robot '
                             'is 0.80 x 0.55 m, so a rack edge clips a corner well before '
                             'it hides the centre.')
    args = parser.parse_args()

    if args.step_m <= 0 or args.span_m <= 0 or args.headings < 1:
        raise SystemExit('step/span must be positive and headings must be >= 1')
    intervals = int(round(args.span_m / args.step_m))
    if not math.isclose(intervals * args.step_m, args.span_m, abs_tol=1e-9):
        raise SystemExit('--span-m must be an integer multiple of --step-m')

    profile, _intrinsics, resolved_world, _pose = load_profile(
        str(args.world_profiles), str(args.world)
    )
    lanes = [
        item for item in profile.get('known_2d_regions', [])
        if str(item.get('type', '')).lower() == 'traversable'
    ]
    if not lanes:
        raise RuntimeError('world profile has no declared traversable regions')
    is_valid, _bounds = build_filter(
        args.world, args.world_profiles,
        wall_margin=0.65, region_shrink_m=0.05, collision_clearance_m=0.25,
    )

    layout = build()
    xs, ys = grid()
    geometry = coverage(layout, 'A', xs, ys)
    per_cam = geometry['per_cam']
    # Per-camera footprint-occlusion field: where is the robot's BODY clipped while its
    # centre is still seen? `coverage.visible_from` ray-casts one point at MARKER_Z, which
    # answers "is the centre visible" -- not "is part of the robot hidden". Measured on the
    # frozen world, a centre-only two-height test finds 0.3-1.8% of floor partial against
    # 14-33% of observed detections, while the footprint test below finds 26-47% of visible
    # floor and brackets the observed rate on every camera. See
    # logs/studies/gate1b_partial_view_targeting/. Geometry only: no detector, no residual,
    # no evaluation quantity, so the selection contract is unchanged.
    partial_fields = _footprint_partial_fields(
        layout, xs, ys, half_extent_m=float(args.footprint_half_m)
    )
    drive = _declared_driveable_mask(xs, ys, lanes)
    edges = _range_edges(layout, xs, ys, per_cam, drive)

    candidates: list[Candidate] = []
    offsets = np.linspace(-0.5 * args.span_m, 0.5 * args.span_m, intervals + 1)
    for lane in lanes:
        for centre, axis in _candidate_centres(
            lane, span_m=args.span_m, centre_step_m=args.centre_step_m
        ):
            points = tuple(
                (centre[0] + float(d), centre[1]) if axis == 'x'
                else (centre[0], centre[1] + float(d))
                for d in offsets
            )
            # Yaw does not affect collision validity in the current circular-footprint
            # filter, but check all eight because that is the actual capture contract.
            yaws = [2 * math.pi * i / args.headings for i in range(args.headings)]
            if not all(is_valid(x, y, yaw) for x, y in points for yaw in yaws):
                continue
            vis = []
            range_bins = []
            cover = set()
            for name, cam, field in zip(CAMERA_NAMES, layout.cameras, per_cam, strict=True):
                flags = tuple(int(field[_nearest(ys, y), _nearest(xs, x)]) for x, y in points)
                vis.append(flags)
                visible_ranges = [
                    math.hypot(x - cam.x, y - cam.y)
                    for (x, y), flag in zip(points, flags, strict=True) if flag
                ]
                if visible_ranges:
                    range_bin = _bin(float(np.median(visible_ranges)), edges[name])
                    range_bins.append(range_bin)
                    if sum(flags) >= 3:
                        cover.add(f'camera_{name}:range:{range_bin}')
                else:
                    range_bins.append('not_visible')
                if 0 < sum(flags) < len(flags):
                    cover.add(f'camera_{name}:sightline_transition')
                partial_flags = [
                    bool(partial_fields[name][_nearest(ys, y), _nearest(xs, x)])
                    for x, y in points
                ]
                # Same threshold the range strata use: at least 3 of the line's positions,
                # so a single grazing cell cannot claim the stratum.
                if sum(partial_flags) >= 3:
                    cover.add(f'camera_{name}:partial_view')
            overlap = sum(sum(flags[i] for flags in vis) >= 2 for i in range(len(points)))
            if overlap >= 3:
                cover.add('multi_camera_overlap')
            candidates.append(Candidate(
                lane=str(lane['name']), axis=axis, centre=centre, points=points,
                covers=frozenset(cover), visibility=tuple(vis),
                range_bins=tuple(range_bins), overlap_positions=overlap,
            ))

    requirements = {
        **{
            f'camera_{name}:range:{band}': max(int(args.range_cover_count), 1)
            for name in CAMERA_NAMES for band in ('near', 'mid', 'far')
        },
        **{
            f'camera_{name}:sightline_transition': max(int(args.transition_cover_count), 1)
            for name in CAMERA_NAMES
        },
        **{
            f'camera_{name}:partial_view': max(int(args.partial_view_cover_count), 1)
            for name in CAMERA_NAMES
        },
        'multi_camera_overlap': max(int(args.overlap_cover_count), 1),
    }
    requested = set(requirements)
    achievable = set().union(*(candidate.covers for candidate in candidates)) if candidates else set()
    missing = sorted(requested - achievable)
    if missing:
        raise RuntimeError(f'geometry search cannot cover requested strata: {missing}')
    short = {
        key: sum(key in candidate.covers for candidate in candidates)
        for key, count in requirements.items()
        if sum(key in candidate.covers for candidate in candidates) < count
    }
    if short:
        raise RuntimeError(f'geometry search has insufficient distinct lines: {short}')

    selected: list[Candidate] = []
    deficits = dict(requirements)
    remaining = list(candidates)
    while any(value > 0 for value in deficits.values()) and remaining and len(selected) < args.max_lines:
        # Prefer the largest new stratum gain, then actual sightline transitions,
        # overlap, and finally a stable lexical/coordinate order.
        remaining.sort(key=lambda candidate: (
            -sum(deficits[item] > 0 for item in candidate.covers),
            -sum('transition' in item and deficits[item] > 0 for item in candidate.covers),
            -candidate.overlap_positions,
            candidate.key,
        ))
        best = remaining.pop(0)
        if not any(deficits[item] > 0 for item in best.covers):
            break
        selected.append(best)
        for item in best.covers:
            deficits[item] = max(deficits[item] - 1, 0)
    uncovered = {key: value for key, value in deficits.items() if value > 0}
    if uncovered:
        raise RuntimeError(
            f'--max-lines={args.max_lines} leaves uncovered multiplicities: {uncovered}'
        )

    poses = []
    line_rows = []
    for line_index, line in enumerate(selected):
        line_id = f'p1g{line_index:02d}_{line.lane}_{line.axis}'
        for point_index, (x, y) in enumerate(line.points):
            for heading_index in range(args.headings):
                poses.append({
                    'x': round(x, 6), 'y': round(y, 6),
                    'yaw': round(2 * math.pi * heading_index / args.headings, 6),
                    'stratum': f'{line_id}_p{point_index:02d}',
                })
        line_rows.append({
            'line_id': line_id, 'declared_lane': line.lane, 'axis': line.axis,
            'centre': list(line.centre), 'points': [list(point) for point in line.points],
            'covers': sorted(line.covers),
            'geometry_visibility_by_camera': {
                f'camera_{name}': list(flags)
                for name, flags in zip(CAMERA_NAMES, line.visibility, strict=True)
            },
            'geometry_range_bin_by_camera': {
                f'camera_{name}': value
                for name, value in zip(CAMERA_NAMES, line.range_bins, strict=True)
            },
            'overlap_positions': line.overlap_positions,
        })

    source = Path(__file__).resolve()
    world_path = Path(resolved_world).resolve()
    payload = {
        'schema': 'camera_observation_characterization_icra_p1_geometry.v1',
        'purpose': 'Fresh dense spatial-scale and correction/refusal transfer pilot.',
        'selection_contract': {
            'uses_detector_outputs': False,
            'uses_localization_residuals': False,
            'uses_learned_model_outputs': False,
            'inputs': [
                'declared traversable rectangles', 'frozen warehouse occluder geometry',
                'frozen external-camera poses and intrinsics',
            ],
            'method': 'deterministic greedy set cover over camera range and sightline strata',
            'stock_state': 'peak',
        },
        'source_sha256': _sha256(source),
        'world_path': str(world_path),
        'world_sha256': _sha256(world_path),
        'world_profiles_sha256': _sha256(args.world_profiles.resolve()),
        'step_m': args.step_m, 'span_m': args.span_m, 'headings': args.headings,
        'candidate_line_count': len(candidates),
        'selected_line_count': len(selected),
        'requested_strata_multiplicity': dict(sorted(requirements.items())),
        'range_tertile_edges_m': {
            f'camera_{name}': list(value) for name, value in edges.items()
        },
        'lines': line_rows,
        'pose_count': len(poses),
        'planned_camera_rows': len(poses) * 5,
        'poses': poses,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'candidate_lines': len(candidates), 'selected_lines': len(selected),
        'pose_count': len(poses), 'planned_camera_rows': len(poses) * 5,
        'output': str(args.out),
    }, indent=2))
    for line in line_rows:
        print(f"{line['line_id']}: centre={line['centre']} covers={line['covers']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
