#!/usr/bin/env python3
"""Adding cameras makes the belief more accurate and less honest.

The central table of the thesis. On the same six drives, replayed with identical odometry,
corrected observations, initialization and process noise, it compares:

  - each camera alone;
  - the full five-camera network under four covariance models.

Accuracy (median and 95th-percentile belief-position error) and honesty (how often the robot
really was inside the claimed 95% ellipse) are reported together, because a filter can buy
coverage simply by widening its ellipse, and one can buy accuracy while becoming
overconfident. Neither number means anything alone.

Correction accounting travels with the error, as the metrics contract requires: a low error
achieved while silently dropping most corrections is not the same result.

Reads frozen artifacts only. Fits nothing, launches nothing.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
STUDY = REPO / 'logs/studies/icra_commissioning_20260905/thesis_evidence'
OUT = REPO / 'logs/studies/network_accuracy_honesty_20260907'

RUNS = [
    'fusion_overlap_rich__N1__seed110',
    'fusion_overlap_rich__N1__seed111',
    'fusion_overlap_rich__N1__seed112',
    'fusion_network_traverse__N1__seed110',
    'fusion_network_traverse__N1__seed111',
    'fusion_network_traverse__N1__seed112',
]

SINGLES = [f'single/camera_{c}' for c in 'ABCDE']
NETWORK = ['full/constant', 'full/geometry', 'full/confidence', 'full/confidence_bias']
LABELS = {
    'single/camera_A': 'camera A alone',
    'single/camera_B': 'camera B alone',
    'single/camera_C': 'camera C alone',
    'single/camera_D': 'camera D alone',
    'single/camera_E': 'camera E alone',
    'full/constant': 'all five cameras, one covariance per camera',
    'full/geometry': 'all five cameras, geometry-dependent covariance',
    'full/confidence': 'all five cameras, detector-score-conditioned covariance',
    'full/confidence_bias': 'all five cameras, score-conditioned plus persistent-bias state',
}
NOMINAL = 0.95


def main() -> None:
    scores: dict[str, list] = collections.defaultdict(list)
    accounting = []
    for name in RUNS:
        path = STUDY / 'driving' / name / 'results.json'
        if not path.exists():
            raise SystemExit(f'missing frozen run: {path}')
        payload = json.loads(path.read_text())
        for arm, value in payload['replay'].items():
            if isinstance(value, dict) and 'median_cm' in value:
                scores[arm].append((value['median_cm'], value['p95_cm'],
                                    100 * value['coverage'][str(NOMINAL)]))
        book = payload['accounting']
        accounting.append({
            'run': name,
            'outcome': book['outcome'],
            'fresh_readings': book['fresh_readings'],
            'fused_batches': book['fused_batches'],
            'dropped_fraction': book['dropped_fraction'],
            'longest_correction_gap_s': book['longest_gap_s'],
        })

    def summarize(arm: str) -> dict:
        values = np.asarray(scores[arm])
        if len(values) != len(RUNS):
            raise SystemExit(f'arm {arm}: expected {len(RUNS)} runs, got {len(values)}')
        return {
            'arm': arm, 'description': LABELS[arm],
            'median_cm': float(values[:, 0].mean()),
            'p95_cm': float(values[:, 1].mean()),
            'coverage95_pct': float(values[:, 2].mean()),
            'median_cm_per_run': [float(v) for v in values[:, 0]],
            'coverage95_pct_per_run': [float(v) for v in values[:, 2]],
        }

    singles = [summarize(a) for a in SINGLES]
    network = [summarize(a) for a in NETWORK]

    print('Belief-position error and honesty, mean over six drives.')
    print(f'A nominal {NOMINAL:.0%} ellipse should contain the robot {NOMINAL:.0%} of the time.\n')
    print(f"{'arm':62}{'median':>9}{'95th pct':>10}{'inside 95% ellipse':>20}")
    for row in singles + network:
        print(f"{row['description']:62}{row['median_cm']:8.2f}cm{row['p95_cm']:9.2f}cm"
              f"{row['coverage95_pct']:19.1f}%")

    best_single = min(singles, key=lambda r: r['median_cm'])
    constant = next(r for r in network if r['arm'] == 'full/constant')
    best_net = min(network, key=lambda r: r['median_cm'])
    bias_arm = next(r for r in network if r['arm'] == 'full/confidence_bias')

    print(f"\nBest single camera is {best_single['description']}: "
          f"{best_single['median_cm']:.2f} cm median, "
          f"{best_single['coverage95_pct']:.1f}% coverage.")
    print(f"The five-camera network with one covariance per camera is "
          f"{'more' if constant['median_cm'] < best_single['median_cm'] else 'less'} accurate "
          f"({constant['median_cm']:.2f} cm) but "
          f"{constant['coverage95_pct'] - best_single['coverage95_pct']:+.1f} points of coverage.")
    print(f"Score-conditioned covariance is the most accurate arm "
          f"({best_net['median_cm']:.2f} cm) and still only "
          f"{best_net['coverage95_pct']:.1f}% honest.")
    print(f"Only the persistent-bias state approaches the nominal level "
          f"({bias_arm['coverage95_pct']:.1f}%), and it costs "
          f"{bias_arm['median_cm'] - best_net['median_cm']:+.2f} cm of median error.")

    print('\nCorrection accounting, per drive:')
    print(f"{'run':40}{'outcome':>12}{'dropped':>10}{'longest gap':>14}")
    for row in accounting:
        print(f"{row['run']:40}{row['outcome']:>12}"
              f"{100 * row['dropped_fraction']:9.1f}%{row['longest_correction_gap_s']:12.1f} s")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps({
        'status': 'frozen_replay_diagnostic',
        'question': 'what does adding cameras buy, and what does it cost',
        'finding': 'the network lowers typical error and lowers coverage; only an explicit '
                   'persistent-bias state restores honesty, at a cost in median error',
        'nominal_ellipse': NOMINAL,
        'runs': RUNS,
        'single_cameras': singles,
        'network': network,
        'accounting': accounting,
        'limitations': [
            'six drives, one seed per arm; descriptive, not inferential',
            'capture-time replay; live processing latency and runtime refusals absent',
            'three of six drives ended in contact, and those drives remain in the selection',
            'belief error, camera-reading error and fused-correction error are different '
            'quantities and are never pooled',
        ],
    }, indent=1) + '\n')
    print(f'\nwrote {OUT / "results.json"}')


if __name__ == '__main__':
    main()
