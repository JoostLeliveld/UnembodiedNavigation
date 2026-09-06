#!/usr/bin/env python3
"""One sheet per correction per camera: where on the floor that correction still misses.

This replaces the four-panel-per-camera sheets that used to live in
`02_the_error/02_error_fields_full_grid/`. Those panels each carried their own arrow gain and
their own colour cap -- they had to, because the ladder spans roughly 30 cm down to 2 cm -- so
the sheet invited exactly the arrow-against-arrow comparison it could not support. Splitting
the ladder into one folder per rung, with one full-page map per camera inside it, removes the
invitation and gives each map the whole page.

Which floor each rung is scored on is a property of the rung, not a choice:

* the raw box, the fixed offset and the shape model fit nothing to this capture, so they are
  honest at every captured position and their maps have no holes;
* the two learned rungs can only be scored on the floor squares they were never fitted on, so
  their maps carry the checkerboard gaps of the held-out split.

Every sheet states which of the two it is, in words, above the map. The strict five-rung
like-for-like comparison on one common floor is a different figure and lives in
`05_the_fixes/14_where_each_fix_still_misses/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / 'logs/studies/camera_observation_characterization_20260831'
for rel in ('experiments/deck_figures', 'experiments/camera_observation_characterization'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from _arrowmap import (  # noqa: E402
    CAMERAS, CORRECTIONS, camera_title, direction_phrase, position_medians, sheet, stats,
)

FOLDER = '02_the_error/02_error_fields_by_correction'
RETIRED = '02_the_error/02_error_fields_full_grid'


def grid_words(correction) -> tuple[str, str]:
    """How to describe, in words, the floor this rung may be scored on."""
    if correction.fits_anything:
        return ('held-out floor squares only',
                'Only the floor squares this correction was never fitted on. The\n'
                'checkerboard gaps are the squares it learned from, and scoring it\n'
                'there would flatter it. The gaps are not missed detections.')
    return ('every captured floor position',
            'Every floor position in the capture. This rung fits nothing to this\n'
            'data, so there is no floor it has to be kept away from, and the map\n'
            'has no holes.')


def write_sheets(bias_rows: list[dict], out: Path) -> list[dict]:
    written = []
    for correction in CORRECTIONS:
        grid_short, grid_long = grid_words(correction)
        source = [row for row in bias_rows
                  if not correction.fits_anything or row['split'] == 'test']
        for camera in CAMERAS:
            rows = [row for row in source if row['camera_id'] == camera]
            cells, blanks = position_medians(rows, correction.key)
            summary = stats(cells)
            opportunities = len(rows)
            readings = sum(row[f'{correction.key}_valid'] == '1' for row in rows)
            path = (out / FOLDER / correction.folder
                    / f'02_{camera}_{correction.slug}.png')
            sheet(
                path=path,
                title=f'{camera_title(camera)} — {correction.label}',
                finding=(
                    f'Half of this camera’s floor positions land within '
                    f'{100 * summary["median"]:.1f} cm of the truth, and '
                    f'{direction_phrase(summary["along"], summary["across"])}'),
                provenance=(
                    f'{summary["n"]} of {len(cells) + len(blanks)} floor positions returned '
                    f'a reading, from {readings} of {opportunities} camera views  ·  '
                    f'simulated warehouse, one frozen detector, no admission gate'),
                grid_short=grid_short,
                cells=cells, blanks=blanks,
                note_title='How to read this map',
                note_lines=[
                    'Each arrow starts at where the robot really was and points to',
                    'where this camera said it was. Longer arrow, bigger mistake.',
                    'At a position the robot was photographed from eight headings;',
                    'the arrow is the middle one of those readings.',
                    '',
                    'Colour repeats the same distance, so a pale arrow is a bad',
                    'reading whichever way it points.',
                    '',
                    'Arrows are drawn many times life size or they would be',
                    'invisible on a 24 m floor. The black bar at the bottom left of',
                    'the map is what a round number of centimetres looks like at',
                    'this magnification. Never compare arrow lengths between two',
                    'sheets; compare the numbers in the titles.',
                    '',
                    'A grey × is a floor position where no heading produced a',
                    'usable reading. That is a missing reading, not a good one.',
                    '',
                    grid_long,
                ],
                hist_xlabel='distance from truth at one floor position (cm)',
                hist_title='How far the readings miss',
                colour=correction.colour,
                footer=f'{correction.subtitle}  ·  camera readings only: no filter, no fusion',
            )
            written.append({
                'correction': correction.key, 'camera_id': camera,
                'figure': str(path.relative_to(out)), 'grid': grid_short,
                'positions_with_a_reading': summary['n'],
                'positions_total': len(cells) + len(blanks),
                'camera_views': opportunities, 'readings': readings,
                'median_error_m': summary['median'], 'p90_error_m': summary['p90'],
                'worst_position_error_m': summary['worst'],
                'median_along_m': summary['along'], 'median_across_m': summary['across'],
            })
            print(f'wrote {path.relative_to(out)}')
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    table = capture / 'bias_update_interpretations.csv'
    manifest_path = capture / 'bias_update_interpretations_manifest.json'
    if not table.is_file() or not manifest_path.is_file():
        raise RuntimeError('Run fit_bias_updates.py first')
    if json.loads(manifest_path.read_text(encoding='utf-8')).get('status') != 'complete':
        raise RuntimeError('bias_update_interpretations manifest is not complete')
    bias_rows = list(csv.DictReader(table.open(encoding='utf-8')))

    out = (args.out or DEFAULT_OUT).expanduser().resolve()
    target = out / FOLDER
    if target.exists() and not args.overwrite:
        raise RuntimeError(f'Output already exists: {target}; pass --overwrite to refresh')
    if target.exists():
        shutil.rmtree(target)
    retired = out / RETIRED
    if retired.exists():
        shutil.rmtree(retired)
        print(f'removed superseded {RETIRED}')
    written = write_sheets(bias_rows, out)
    (target / 'error_fields_by_correction_manifest.json').write_text(json.dumps({
        'status': 'complete',
        'schema': 'error_fields_by_correction.v1',
        'quantity': 'camera reading scored against the commanded ground-truth position',
        'aggregation': 'median over the headings that returned a box at that position',
        'grid_rule': (
            'Rungs that fit nothing to this capture (raw box, fixed offset, shape model) are '
            'drawn on every captured position. The two learned rungs are drawn on held-out '
            'floor squares only.'),
        'interpretation_limit': (
            'One capture per camera-position-heading state, so an arrow is an observed '
            'error, never an estimate of a conditional mean bias or covariance.'),
        'sheets': written,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(target), 'sheets': len(written)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
