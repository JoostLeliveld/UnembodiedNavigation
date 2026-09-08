#!/usr/bin/env python3
"""The independence figure: why more readings do not buy proportionally more certainty."""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

OUT = repo_root() / 'logs/studies/reading_independence_20260907'
INK, GRID = '#1d2530', '#c8ced6'
FULL, THIN, WARN = '#2367a2', '#25856a', '#c23d36'


def main() -> None:
    data = json.loads((OUT / 'results.json').read_text())
    lags = data['autocorrelation']
    thinning = data['thinning_test']

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.6, 4.8))

    # -- left: correlation decays slowly, so consecutive readings repeat the same error
    lag = np.array([r['lag_s'] for r in lags])
    rho = np.array([0.5 * (r['corr_x'] + r['corr_y']) for r in lags])
    share = np.array([100 * r['independent_fraction'] for r in lags])

    left.plot(lag, 100 * rho, 'o-', color=WARN, lw=2.2, ms=7, label='how alike two readings are')
    left.plot(lag, share, 's--', color=FULL, lw=2.2, ms=7, label='how much genuinely new information')
    for x, y, r in zip(lag, share, rho):
        left.annotate(f'{y:.0f}%', (x, y), textcoords='offset points', xytext=(0, -16),
                      ha='center', fontsize=8.5, color=FULL)
    left.set_xlabel('Time between two readings from the same camera (s)\n'
                    'the robot travels about 21 cm per second', fontsize=9.5)
    left.set_ylabel('Percent', fontsize=9.5)
    left.set_title('Consecutive readings repeat the same error\n'
                   'Two readings 0.2 s apart are 89% alike: only 6% is new information',
                   fontsize=10.5, fontweight='bold')
    left.set_ylim(0, 100)
    left.legend(fontsize=8.5, loc='center right')
    left.grid(alpha=.25, color=GRID)

    # -- right: the thinning test. Throwing readings away makes the filter more honest.
    names = [r['covariance_model'] for r in thinning]
    full = [r['full_rate']['coverage95_pct'] for r in thinning]
    thin = [r['thinned_1hz']['coverage95_pct'] for r in thinning]
    x = np.arange(len(names))

    right.bar(x - .2, full, .38, color=FULL, label='every reading used (5 per second)')
    right.bar(x + .2, thin, .38, color=THIN, label='4 in 5 readings discarded (1 per second)')
    right.axhline(95, color=WARN, ls='--', lw=1.6)
    right.text(len(names) - .45, 95.9, 'what the filter claims: 95%', color=WARN,
               fontsize=8.5, ha='right', fontweight='bold')
    for xi, (f, t) in enumerate(zip(full, thin)):
        right.text(xi - .2, f + .7, f'{f:.0f}', ha='center', fontsize=8.5, color=FULL)
        right.text(xi + .2, t + .7, f'{t:.0f}', ha='center', fontsize=8.5, color=THIN)

    right.set_xticks(x)
    right.set_xticklabels(['one covariance\nper camera', 'geometry\ndependent',
                           'detector-score\nconditioned', 'score plus\nbias state'], fontsize=8.5)
    right.set_ylabel('How often the robot really was inside\nthe claimed 95% ellipse (%)', fontsize=9.5)
    right.set_ylim(60, 100)
    right.set_title('Using fewer readings makes the filter more honest\n'
                    'Discarding 80% of the data raises coverage by up to 9 points',
                    fontsize=10.5, fontweight='bold')
    right.legend(fontsize=8.5, loc='lower right')
    right.grid(axis='y', alpha=.25, color=GRID)

    fig.subplots_adjust(wspace=.34)
    fig.suptitle('The stated uncertainty is dishonest because camera readings are not independent',
                 fontsize=12.5, fontweight='bold', y=1.02)
    fig.text(.5, -.07, 'Five cameras, six drives in the simulated warehouse. Nominal 95% ellipse. '
                       'Higher coverage is more honest; the dashed line is the target.',
             ha='center', fontsize=8.5, color=INK)

    for path in (OUT / 'reading_independence.pdf', OUT / 'reading_independence.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'reading_independence.pdf')


if __name__ == '__main__':
    main()
