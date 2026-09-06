#!/usr/bin/env python3
"""Choose stratified line centres for the generalization capture, and assign the split.

The previous capture had five centres and about eighteen independent camera/place units,
which is too few to decide whether an image-quality model generalizes. This selects many
more places and, critically, spreads the difficult ones around the building instead of
concentrating them in one aisle: in the old capture every occluded situation sat at a
single centre, so "visibility predicts usability" could not be separated from "one pallet
stack behaves this way".

Selection uses the frozen 0.67 m characterization grid only as a *scouting* map. For each
grid position and camera it already knows the median hull error, so a centre can be
labelled clear or degraded before any new frame is captured. That label is scouting
metadata, never a runtime input and never a training feature.

Centres are chosen to satisfy, per camera, a target number of independent clear situations
and independent degraded situations, while spreading them across the warehouse. Spacing is
enforced only as a mild de-duplication so two centres are not effectively the same place;
it is deliberately not tied to a correlation-length figure, because the within-track
structure measured so far (roughly 0.25-0.45 m) is bounded by the short trajectories and
does not justify a large fixed separation.

The train/validation/test split is assigned HERE, before capture, and by centre. Every
frame from one centre stays in one partition, so no frame-level leakage is possible.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CAMERAS = ('camera_A', 'camera_B', 'camera_C', 'camera_D', 'camera_E')
USABLE_CM = 20.0


def scout(grid_csv: Path) -> dict[tuple[float, float], dict[str, dict[str, float]]]:
    """Median hull error and range per (position, camera) from the frozen grid."""
    import csv

    with grid_csv.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['detected'] in ('True', 'true', '1')
                and row['hull_valid'] in ('True', 'true', '1')]
    gathered: dict[tuple[float, float], dict[str, list[tuple[float, float]]]] = \
        collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        key = (round(float(row['robot_x']), 3), round(float(row['robot_y']), 3))
        error = math.hypot(float(row['hull_dx']), float(row['hull_dy'])) * 100.0
        gathered[key][row['camera_id']].append((error, float(row['camera_range_m'])))

    out: dict[tuple[float, float], dict[str, dict[str, float]]] = {}
    for key, cameras in gathered.items():
        entry: dict[str, dict[str, float]] = {}
        for camera, values in cameras.items():
            if len(values) < 3:
                continue
            errors = [value for value, _ in values]
            entry[camera] = {
                'median_err_cm': st.median(errors),
                'range_m': st.median([rng for _, rng in values]),
                'n_headings': len(values),
                'degraded': 1.0 if st.median(errors) > USABLE_CM else 0.0,
            }
        if entry:
            out[key] = entry
    return out


def pick_centres(scouted: dict, *, clear_per_camera: int, degraded_per_camera: int,
                 min_separation_m: float, seed: int) -> list[dict]:
    """Greedily choose centres meeting per-camera clear/degraded targets, spread out."""
    rng = random.Random(seed)
    positions = sorted(scouted)
    rng.shuffle(positions)

    need_clear = {camera: clear_per_camera for camera in CAMERAS}
    need_degraded = {camera: degraded_per_camera for camera in CAMERAS}
    chosen: list[dict] = []

    def far_enough(candidate: tuple[float, float]) -> bool:
        return all(math.hypot(candidate[0] - item['x'], candidate[1] - item['y'])
                   >= min_separation_m for item in chosen)

    # Two passes: satisfy the scarce degraded situations first, then fill clear ones.
    for want_degraded in (True, False):
        for position in positions:
            if not far_enough(position):
                continue
            entry = scouted[position]
            helps = False
            for camera, stats in entry.items():
                degraded = stats['degraded'] > 0.5
                if degraded == want_degraded:
                    pool = need_degraded if degraded else need_clear
                    if pool.get(camera, 0) > 0:
                        helps = True
            if not helps:
                continue
            for camera, stats in entry.items():
                pool = need_degraded if stats['degraded'] > 0.5 else need_clear
                if pool.get(camera, 0) > 0:
                    pool[camera] -= 1
            chosen.append({
                'x': position[0], 'y': position[1],
                'cameras': {camera: {'median_err_cm': round(stats['median_err_cm'], 2),
                                     'range_m': round(stats['range_m'], 2),
                                     'character': ('degraded' if stats['degraded'] > 0.5
                                                   else 'clear')}
                            for camera, stats in sorted(entry.items())},
            })
            if not any(need_clear.values()) and not any(need_degraded.values()):
                return chosen
    return chosen


def assign_split(centres: list[dict], *, seed: int,
                 fractions: tuple[float, float, float]) -> None:
    """Assign train/val/test BY CENTRE so no frame-level leakage is possible.

    Difficulty is dealt round-robin: centres are ordered by how many degraded situations
    they contribute, then handed out one at a time to whichever partition is furthest
    below its quota. That keeps hard views in every partition, which matters because a
    test set of only easy places cannot falsify a quality model.
    """
    rng = random.Random(seed + 1)
    total = len(centres)
    quotas = {'train': fractions[0] * total,
              'val': fractions[1] * total,
              'test': fractions[2] * total}
    filled: dict[str, int] = {'train': 0, 'val': 0, 'test': 0}
    hard_filled: dict[str, int] = {'train': 0, 'val': 0, 'test': 0}

    def difficulty(centre: dict) -> int:
        return sum(1 for stats in centre['cameras'].values()
                   if stats['character'] == 'degraded')

    total_hard = sum(difficulty(centre) for centre in centres)
    hard_quotas = {name: fraction * total_hard
                   for name, fraction in zip(('train', 'val', 'test'), fractions)}

    ordered = sorted(centres, key=lambda item: (-difficulty(item), rng.random()))
    for centre in ordered:
        hard = difficulty(centre)
        # Prefer the partition furthest below its share of the difficult situations, and
        # fall back to overall count when a centre contributes none.
        def deficit(name: str) -> tuple[float, float]:
            hard_short = (hard_quotas[name] - hard_filled[name]) / max(hard_quotas[name], 1e-9)
            count_short = (quotas[name] - filled[name]) / max(quotas[name], 1e-9)
            return (-hard_short, -count_short) if hard else (-count_short, -hard_short)

        target = min(('train', 'val', 'test'), key=deficit)
        centre['split'] = target
        filled[target] += 1
        hard_filled[target] += hard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-csv', type=Path,
                        default=REPO / 'logs/perception_datasets'
                                       'warehouse_v2_bbox_characterization_20260831'
                                       '/observation_interpretations.csv')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--clear-per-camera', type=int, default=10)
    parser.add_argument('--degraded-per-camera', type=int, default=10)
    parser.add_argument('--min-separation-m', type=float, default=1.5,
                        help='Mild de-duplication so two centres are not the same place. '
                             'Deliberately NOT tied to a correlation-length figure.')
    parser.add_argument('--seed', type=int, default=20260902)
    parser.add_argument('--train-frac', type=float, default=0.6)
    parser.add_argument('--val-frac', type=float, default=0.2)
    args = parser.parse_args()

    grid = args.grid_csv
    if not grid.exists():
        grid = (REPO / 'logs/perception_datasets'
                / 'warehouse_v2_bbox_characterization_20260831'
                / 'observation_interpretations.csv')
    if not grid.exists():
        raise SystemExit(f'scouting grid not found at {grid}')

    scouted = scout(grid)
    centres = pick_centres(scouted, clear_per_camera=args.clear_per_camera,
                           degraded_per_camera=args.degraded_per_camera,
                           min_separation_m=args.min_separation_m, seed=args.seed)
    assign_split(centres, seed=args.seed,
                 fractions=(args.train_frac, args.val_frac,
                            1.0 - args.train_frac - args.val_frac))

    counts: dict[str, collections.Counter] = {
        'clear': collections.Counter(), 'degraded': collections.Counter()}
    per_split: collections.Counter = collections.Counter()
    per_split_degraded: collections.Counter = collections.Counter()
    for centre in centres:
        per_split[centre['split']] += 1
        for camera, stats in centre['cameras'].items():
            counts[stats['character']][camera] += 1
            if stats['character'] == 'degraded':
                per_split_degraded[centre['split']] += 1

    payload = {
        'schema': 'perception_filter_cascade_centre_plan.v1',
        'purpose': ('Stratified centres for a generalization capture, with the '
                    'train/val/test split assigned by centre before capture.'),
        'scouting_source': str(grid),
        'scouting_is_metadata_only': True,
        'usable_threshold_cm': USABLE_CM,
        'min_separation_m': args.min_separation_m,
        'seed': args.seed,
        'centre_count': len(centres),
        'centres': centres,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    print(f'scouted {len(scouted)} grid positions')
    print(f'selected {len(centres)} centres')
    print()
    print(f'{"camera":10s} {"clear situations":>17s} {"degraded situations":>20s}')
    for camera in CAMERAS:
        print(f'{camera:10s} {counts["clear"][camera]:17d} {counts["degraded"][camera]:20d}')
    print()
    print(f'{"split":8s} {"centres":>8s} {"degraded situations":>20s}')
    for name in ('train', 'val', 'test'):
        print(f'{name:8s} {per_split[name]:8d} {per_split_degraded[name]:20d}')
    print()
    spread = [(centre['x'], centre['y']) for centre in centres]
    print(f'x span {min(p[0] for p in spread):.1f} to {max(p[0] for p in spread):.1f} m, '
          f'y span {min(p[1] for p in spread):.1f} to {max(p[1] for p in spread):.1f} m')
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
