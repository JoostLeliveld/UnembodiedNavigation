#!/usr/bin/env python3
"""Why the network's stated uncertainty is dishonest: readings are not independent.

A Kalman filter shrinks its stated uncertainty as if every reading were a fresh draw.
Camera readings arriving 0.2 s apart are ~90% correlated, so twenty of them carry about
one reading's worth of independent information. The filter takes the sqrt(N) credit anyway,
which is the overconfidence that shows up as 77% coverage of a nominal 95% ellipse.

Two independent lines of evidence, both from the frozen six-run selection:

  1. measured residual autocorrelation against time lag, per camera;
  2. the thinning test - the same drives replayed at 1 Hz instead of 5 Hz. Decimation
     removes correlated readings without changing the sensor or the covariance model, and
     coverage IMPROVES. No covariance model can produce that; only dependence can.

Reads frozen artifacts only. Runs nothing, launches nothing, fits nothing.
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
OUT = REPO / 'logs/studies/reading_independence_20260907'

# The frozen selection. Never globbed: a missing or extra run is an error, not a smaller mean.
RUNS = [
    'fusion_overlap_rich__N1__seed110',
    'fusion_overlap_rich__N1__seed111',
    'fusion_overlap_rich__N1__seed112',
    'fusion_network_traverse__N1__seed110',
    'fusion_network_traverse__N1__seed111',
    'fusion_network_traverse__N1__seed112',
]

# Replay arms: full rate against the same arm decimated to 1 Hz.
ARMS = ['constant', 'geometry', 'confidence', 'confidence_bias']
ARM_LABELS = {
    'constant': 'one fitted covariance per camera',
    'geometry': 'geometry-dependent covariance',
    'confidence': 'detector-score-conditioned covariance',
    'confidence_bias': 'score-conditioned plus persistent-bias state',
}


def load_runs() -> dict[str, dict]:
    runs = {}
    for name in RUNS:
        path = STUDY / 'driving' / name / 'results.json'
        if not path.exists():
            raise SystemExit(f'missing frozen run: {path}')
        runs[name] = json.loads(path.read_text())
    return runs


def autocorrelation(runs: dict[str, dict]) -> list[dict]:
    """Mean residual autocorrelation per time lag, pooled over cameras and runs."""
    by_lag: dict[float, list] = collections.defaultdict(list)
    travel: dict[float, list] = collections.defaultdict(list)
    for payload in runs.values():
        temporal = payload['temporal']
        rows = temporal['correlations'] if isinstance(temporal, dict) and 'correlations' in temporal else temporal
        for row in rows:
            lag = round(float(row['lag_s']), 1)
            by_lag[lag].append(row['correlation'])
            travel[lag].append(float(row['distance_m']))

    out = []
    for lag in sorted(by_lag):
        values = np.asarray(by_lag[lag], dtype=float)
        if len(values) < 5:  # a lag seen by only one or two camera-run pairs is not a mean
            continue
        rho_x, rho_y = float(values[:, 0].mean()), float(values[:, 1].mean())
        rho = 0.5 * (rho_x + rho_y)
        out.append({
            'lag_s': lag,
            'robot_travel_m': float(np.mean(travel[lag])),
            'corr_x': rho_x,
            'corr_y': rho_y,
            'camera_run_pairs': int(len(values)),
            # AR(1) effective sample size for N correlated readings.
            'independent_fraction': float((1 - rho) / (1 + rho)),
            'overconfidence_factor': float(np.sqrt((1 + rho) / (1 - rho))),
        })
    return out


def thinning_test(runs: dict[str, dict]) -> list[dict]:
    """Same drives, same covariance model, readings decimated 5x."""
    scores: dict[str, list] = collections.defaultdict(list)
    for payload in runs.values():
        for arm, value in payload['replay'].items():
            if isinstance(value, dict) and 'median_cm' in value:
                scores[arm].append((
                    value['median_cm'], value['p95_cm'], 100 * value['coverage']['0.95'],
                ))

    out = []
    for arm in ARMS:
        full, thin = np.asarray(scores[f'full/{arm}']), np.asarray(scores[f'1Hz_policy/{arm}'])
        if len(full) != len(RUNS) or len(thin) != len(RUNS):
            raise SystemExit(f'arm {arm}: expected {len(RUNS)} runs, got {len(full)}/{len(thin)}')
        out.append({
            'covariance_model': arm,
            'description': ARM_LABELS[arm],
            'full_rate': {'median_cm': full[:, 0].mean(), 'p95_cm': full[:, 1].mean(),
                          'coverage95_pct': full[:, 2].mean()},
            'thinned_1hz': {'median_cm': thin[:, 0].mean(), 'p95_cm': thin[:, 1].mean(),
                            'coverage95_pct': thin[:, 2].mean()},
            'coverage_change_pp': float(thin[:, 2].mean() - full[:, 2].mean()),
            'median_change_cm': float(thin[:, 0].mean() - full[:, 0].mean()),
        })
    return out


def main() -> None:
    runs = load_runs()
    lags = autocorrelation(runs)
    thinning = thinning_test(runs)

    print('Residual autocorrelation (5 cameras x 6 runs):')
    print(f"{'lag (s)':>8}{'travel (m)':>12}{'corr x':>9}{'corr y':>9}"
          f"{'independent share':>19}{'overconfident':>15}")
    for row in lags:
        print(f"{row['lag_s']:8.1f}{row['robot_travel_m']:12.3f}{row['corr_x']:9.3f}"
              f"{row['corr_y']:9.3f}{row['independent_fraction']:18.1%}"
              f"{row['overconfidence_factor']:14.2f}x")

    print('\nThinning test - identical drives replayed at 1 Hz instead of 5 Hz:')
    print(f"{'covariance model':34}{'coverage 5Hz':>14}{'coverage 1Hz':>14}{'change':>9}{'median cost':>13}")
    for row in thinning:
        print(f"{row['covariance_model']:34}{row['full_rate']['coverage95_pct']:13.1f}%"
              f"{row['thinned_1hz']['coverage95_pct']:13.1f}%"
              f"{row['coverage_change_pp']:+8.1f}{row['median_change_cm']:+12.2f} cm")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps({
        'status': 'frozen_replay_diagnostic',
        'question': 'why is the stated uncertainty dishonest while position error is small',
        'finding': 'camera readings are strongly autocorrelated; the filter credits them as independent',
        'runs': RUNS,
        'nominal_ellipse': 0.95,
        'autocorrelation': lags,
        'thinning_test': thinning,
        'limitations': [
            'six drives, one seed per arm; descriptive, not inferential',
            'capture-time replay; live processing latency and refusals are not reproduced',
            'thinning changes both correlation and reading count, so it corroborates the '
            'mechanism rather than isolating rho',
            'AR(1) effective sample size is an approximation of the true dependence structure',
        ],
    }, indent=1) + '\n')
    print(f'\nwrote {OUT / "results.json"}')


if __name__ == '__main__':
    main()
