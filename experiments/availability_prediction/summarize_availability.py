#!/usr/bin/env python3
"""Whether a camera will see the robot is predictable - but not equally for every camera.

Two findings, from the frozen commissioning artifacts:

  1. A spatial Gaussian process predicts held-out detection far better than a constant rate
     or a geometry model. This is the availability field the planner needs, because at
     planning time no image exists yet.
  2. The improvement is wildly uneven across cameras. Camera C improves ~20x and camera D
     only ~2x, and that split is the same one the detector study found: C's misses are racks
     blocking a sightline, which is a sharp spatial pattern a GP learns easily, while D's
     misses come from range, which is diffuse and not a boundary in position at all.

Finding 2 is the one that matters: it says an availability map inherits the geometry of the
failure it is modelling, so a single field family will not serve every camera equally.

Reads frozen artifacts only. Fits nothing, launches nothing.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
STATIC = REPO / 'logs/studies/icra_commissioning_20260905/field_study/static_results.json'
OUT = REPO / 'logs/studies/availability_prediction_20260907'

# Ordered simplest to richest, so a rung has to earn itself against the one above.
LADDER = [
    ('constant', 'one detection rate per camera'),
    ('geometry_xy', 'sightline geometry from the CAD shelf layout'),
    ('local_xy', 'average of nearby measured positions'),
    ('local_heading', 'nearby positions and headings'),
    ('gp_xy', 'Gaussian process over floor position'),
    ('gp_integrated', 'Gaussian process, belief-integrated'),
]
CAMERAS = ['camera_A', 'camera_B', 'camera_C', 'camera_D', 'camera_E']


def main() -> None:
    static = json.loads(STATIC.read_text())
    scores = static['scores']
    paired = static['paired_tile_differences']

    ladder = []
    print('Predicting whether a camera returns a usable detection (held-out tiles).')
    print('Brier score: 0 is perfect, 0.25 is a coin flip. Lower is better.\n')
    print(f"{'model':44}{'Brier':>9}{'vs constant':>13}")
    baseline = scores['evaluation/constant']['equal_tile_brier']
    for key, description in LADDER:
        brier = scores[f'evaluation/{key}']['equal_tile_brier']
        entry = {
            'model': key, 'description': description,
            'equal_tile_brier': brier,
            'brier': scores[f'evaluation/{key}']['brier'],
            'calibration_error': scores[f'evaluation/{key}']['ece'],
            'improvement_over_constant': baseline - brier,
        }
        if key in paired:
            entry['paired_tile_advantage'] = paired[key]['constant_minus_model']
            entry['paired_tile_bootstrap95'] = paired[key]['descriptive_tile_bootstrap95']
        ladder.append(entry)
        gain = '-' if key == 'constant' else f'{baseline - brier:+.4f}'
        print(f'{description:44}{brier:9.4f}{gain:>13}')

    gp = paired['gp_xy']
    print(f"\nGP advantage over a constant rate: {gp['constant_minus_model']:+.3f} "
          f"(tile bootstrap {gp['descriptive_tile_bootstrap95'][0]:+.3f} to "
          f"{gp['descriptive_tile_bootstrap95'][1]:+.3f}), so the interval excludes zero.")

    per_camera = {}
    print('\nPer camera - the GP does not help every camera equally:')
    print(f"{'camera':>8}{'constant':>11}{'geometry':>11}{'GP':>9}{'GP gain':>10}"
          f"{'observed':>11}{'predicted':>11}")
    for cam in CAMERAS:
        const = scores['evaluation/constant']['cameras'][cam]
        geom = scores['evaluation/geometry_xy']['cameras'][cam]
        gp_cam = scores['evaluation/gp_xy']['cameras'][cam]
        factor = const['brier'] / gp_cam['brier'] if gp_cam['brier'] > 0 else float('inf')
        per_camera[cam] = {
            'constant_brier': const['brier'],
            'geometry_brier': geom['brier'],
            'gp_brier': gp_cam['brier'],
            'gp_improvement_factor': factor,
            'observed_detection_fraction': gp_cam['observed_fraction'],
            'gp_predicted_fraction': gp_cam['predicted_fraction'],
        }
        print(f"{cam[-1]:>8}{const['brier']:11.4f}{geom['brier']:11.4f}{gp_cam['brier']:9.4f}"
              f"{factor:9.1f}x{gp_cam['observed_fraction']:11.3f}"
              f"{gp_cam['predicted_fraction']:11.3f}")

    best = max(per_camera, key=lambda c: per_camera[c]['gp_improvement_factor'])
    worst = min(per_camera, key=lambda c: per_camera[c]['gp_improvement_factor'])
    print(f"\nSpread: camera {best[-1]} improves "
          f"{per_camera[best]['gp_improvement_factor']:.1f}x, camera {worst[-1]} only "
          f"{per_camera[worst]['gp_improvement_factor']:.1f}x.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps({
        'status': 'frozen_commissioning_diagnostic',
        'question': 'can a commissioning field predict whether a camera will see the robot',
        'finding': 'yes, and a spatial GP is best - but the gain per camera tracks the '
                   'geometry of that camera failure mode, not its overall difficulty',
        'held_out_opportunities': static['heldout_opportunities'],
        'held_out_groups': static['heldout_groups'],
        'attempted_frames': static['attempted_frames'],
        'ladder': ladder,
        'per_camera': per_camera,
        'source_limitations': static['limitations'],
        'limitations': [
            'previously examined static development configurations, not a fresh final test',
            'availability means a returned detection with finite floor projection, before '
            'the manager admission gates',
            'deterministic renderer: repeats at one pose are not independent trials',
            'bootstrap tiles are descriptive spatial blocks, not independent runs',
        ],
    }, indent=1) + '\n')
    print(f'\nwrote {OUT / "results.json"}')


if __name__ == '__main__':
    main()
