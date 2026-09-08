#!/usr/bin/env python3
"""Problem-statement figure: what camera corrections are worth, from one real drive.

Two stretches of the same 242 s drive, drawn from the logged belief rather than sketched.
In the first the cameras deliver continuously and the belief tracks the robot. In the
second no correction is accepted for 45 s while the robot drives 11.8 m, and the belief
drifts away from the truth while its own stated uncertainty grows with it.

Everything here is logged: belief mean and covariance from the planner, ground truth
sampled at the belief timestamp as the metrics contract requires, and accepted corrections
from the runtime's own flag. Ground truth is used for scoring only; the estimator never saw
it.
"""
from __future__ import annotations

import glob
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
RUN = ('logs/studies/icra_commissioning_20260905/network_navigation_runtime_pilot/'
       '*/P2/seed210/experiment_*/experiment.csv')
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'

# Two windows of the same drive, in seconds from its first sample.
CORRECTED = (68.0, 113.0)
STARVED = (186.5, 231.6)

TRUTH, BELIEF, MEAS, ELLIPSE = '#4a5058', '#1f6fb8', '#c23d36', '#1f6fb8'


def load() -> tuple[pd.DataFrame, np.ndarray]:
    matches = glob.glob(str(REPO / RUN))
    if not matches:
        raise SystemExit(f'no drive matching {RUN}')
    frame = pd.read_csv(matches[0], low_memory=False)
    stamp = frame['stamp'].astype(float)
    start = stamp.min()

    scored = frame[(frame['planner_belief_available'] == 1)
                   & (frame['gt_available'] == 1)].copy()
    scored = scored.dropna(subset=['gt_x_at_belief_stamp', 'gt_y_at_belief_stamp',
                                   'planner_belief_x', 'planner_cov_x'])
    scored['t'] = scored['stamp'].astype(float) - start
    accepted = (stamp[frame['pixel_corr_accepted'].fillna(0).astype(float) == 1] - start).values
    return scored, accepted


def draw(ax, rows: pd.DataFrame, accepted: np.ndarray, window: tuple[float, float],
         title: str, subtitle: str) -> None:
    low, high = window
    seg = rows[(rows['t'] >= low) & (rows['t'] <= high)]
    hits = accepted[(accepted >= low) & (accepted <= high)]

    ax.plot(seg['gt_x_at_belief_stamp'], seg['gt_y_at_belief_stamp'],
            color=TRUTH, lw=2.4, label='where the robot really was', zorder=3)
    ax.plot(seg['planner_belief_x'], seg['planner_belief_y'],
            color=BELIEF, lw=2.0, ls='--', label='where the robot thought it was', zorder=4)

    # One ellipse every few seconds, at the belief's own stated covariance.
    for _, row in seg.iloc[::max(1, len(seg) // 9)].iterrows():
        cov = np.array([[row['planner_cov_x'], row['planner_cov_xy']],
                        [row['planner_cov_xy'], row['planner_cov_y']]])
        values, vectors = np.linalg.eigh(cov)
        values = np.clip(values, 1e-9, None)
        angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
        # 2 sigma, the usual drawing convention for a planar belief
        ax.add_patch(Ellipse((row['planner_belief_x'], row['planner_belief_y']),
                             4 * np.sqrt(values[-1]), 4 * np.sqrt(values[0]), angle=angle,
                             fc=ELLIPSE, alpha=.16, ec=ELLIPSE, lw=.9, zorder=2))

    if len(hits):
        marks = rows.iloc[np.searchsorted(rows['t'].values, hits[::max(1, len(hits) // 6)])]
        ax.plot(marks['planner_belief_x'], marks['planner_belief_y'], '*',
                color=MEAS, ms=8, mec='white', mew=.7, ls='none', alpha=.85,
                label='camera correction accepted', zorder=5)

    travelled = np.hypot(np.diff(seg['gt_x_at_belief_stamp']),
                          np.diff(seg['gt_y_at_belief_stamp'])).sum()
    error = 100 * np.hypot(seg['planner_belief_x'] - seg['gt_x_at_belief_stamp'],
                           seg['planner_belief_y'] - seg['gt_y_at_belief_stamp'])
    ax.set_title(f'{title}\n{subtitle}', fontsize=10.5, fontweight='bold')
    ax.text(.03, .96, f'{len(hits)} corrections in {high - low:.0f} s\n'
                      f'{travelled:.1f} m driven\n'
                      f'final error {error.iloc[-1]:.0f} cm',
            transform=ax.transAxes, fontsize=9.5, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#c8ced6', alpha=.92))
    ax.set(xlabel='x (m)', aspect='equal')
    ax.grid(alpha=.2)


def main() -> None:
    rows, accepted = load()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    draw(axes[0], rows, accepted, CORRECTED,
         'Cameras keep correcting the belief',
         'the estimate stays on the robot')
    draw(axes[1], rows, accepted, STARVED,
         'No camera correction arrives for 45 seconds',
         'the estimate drifts and its uncertainty grows')

    # One stretch runs north-south and the other east-west, so give each panel its own
    # centre but the same span in metres; equal aspect then keeps distances comparable.
    spans = []
    for ax in axes:
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        spans.append(max(x1 - x0, y1 - y0))
    span = max(spans) * 1.08
    for ax in axes:
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set(xlim=(cx - span / 2, cx + span / 2), ylim=(cy - span / 2, cy + span / 2))
    axes[0].set_ylabel('y (m)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'localization_example.pdf', OUT / 'localization_example.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'localization_example.pdf')


if __name__ == '__main__':
    main()
