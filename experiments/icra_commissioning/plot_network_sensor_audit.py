#!/usr/bin/env python3
"""Plot every camera's within-run residual correlations from the selected pilot."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/icra_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from network_navigation_analysis import ARMS, COLORS, NAMES, style, savefig


def plot(results_path):
    data = json.loads(results_path.read_text())['results']
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.25, top=.73, wspace=.15)
    for camera, ax in zip('ABCDE', axes):
        counts = []
        for arm in ARMS:
            record = next(r for r in data if r['arm'] == arm)
            matches = [c for c in record.get('sensor_diagnostics', {}).get('cameras', []) if c['camera'] == camera]
            if not matches:
                counts.append(0); continue
            c = matches[0]; counts.append(c['observations'])
            lag = [r['lag_s'] for r in c['temporal']]
            rho = np.array([[np.nan if v is None else v for v in r['whitened_coordinate_correlation']]
                            for r in c['temporal']])
            for j, ls in enumerate(('-', '--')):
                ax.plot(lag, rho[:, j], ls, color=COLORS[arm], marker='o', ms=3, lw=1.)
        ax.set(title=f"Camera {camera}\nn = {' / '.join(map(str, counts))}", xlabel='Lag [s, simulation]',
               xlim=(.1, 2.1), ylim=(-1.05, 1.05), xticks=[.2, 1., 2.])
        ax.axhline(0, color='#888', lw=.7); ax.grid(alpha=.16)
        if not any(counts): ax.text(.5, .55, 'No readings on\nthese route prefixes',
                                    transform=ax.transAxes, ha='center', fontsize=8)
    axes[0].set_ylabel('Residual correlation')
    handles = [Line2D([0], [0], color=COLORS[a], label=NAMES[a]) for a in ARMS]
    handles += [Line2D([0], [0], color='black', ls=ls, label=f'Whitened coordinate {j}')
                for j, ls in ((1, '-'), (2, '--'))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .025), ncols=5, frameon=False, fontsize=8)
    fig.suptitle('Residuals remain temporally structured after static camera calibration\n'
                 'One run per field; n is camera readings in P0 / P1 / P2. No cross-run lag pairs.', fontsize=11, y=.98)
    savefig(fig, results_path.parent/'sensor_temporal_audit')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    style(); plot(p.parse_args().results.resolve())
