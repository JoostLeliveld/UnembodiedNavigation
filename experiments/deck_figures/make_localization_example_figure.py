#!/usr/bin/env python3
"""Problem-statement figure: what camera corrections are worth, from one real drive.

Two stretches of the same 242 s drive, drawn from the logged belief rather than sketched,
each on the warehouse floor plan with the racks that block the cameras' views. In the first
the cameras deliver continuously and the belief tracks the robot. In the second no
correction is accepted for 45 s while the robot drives 9.3 m, and the belief drifts away
from the truth while its own stated uncertainty grows with it. A third panel follows the
error and the stated uncertainty over time for both stretches.

Everything here is logged: belief mean and covariance from the planner, ground truth
sampled at the belief timestamp as the metrics contract requires, and accepted corrections
from the runtime's own flag. Rack footprints come from the world file. Ground truth is used
for scoring only; the estimator never saw it.
"""
from __future__ import annotations

import glob
import pathlib
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
RUN = ('logs/studies/icra_commissioning_20260905/network_navigation_runtime_pilot/'
       '*/P2/seed210/experiment_*/experiment.csv')
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'

# Two windows of the same drive, in seconds from its first sample.
CORRECTED = (68.0, 113.0)
STARVED = (186.5, 231.6)

TRUTH, BELIEF, MEAS = '#2b3038', '#1f6fb8', '#c23d36'
RACK, START, GOAL = '#b9c0c8', '#2e8b57', '#c23d36'


def racks() -> list[tuple[float, float, float, float]]:
    block = WORLD.read_text()
    block = block[block.index('warehouse_v2_occluders'):]
    block = block[:block.index('</model>')]
    out = []
    for pose, size in re.findall(
            r'<pose>([^<]+)</pose>.*?<box><size>([^<]+)</size></box>', block, flags=re.S):
        px, py = (float(v) for v in pose.split()[:2])
        sx, sy = (float(v) for v in size.split()[:2])
        out.append((px, py, sx, sy))
    return out


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
    scored['err'] = np.hypot(scored['planner_belief_x'] - scored['gt_x_at_belief_stamp'],
                             scored['planner_belief_y'] - scored['gt_y_at_belief_stamp'])
    scored['sigma'] = np.sqrt(scored['planner_cov_x'] + scored['planner_cov_y'])
    accepted = (stamp[frame['pixel_corr_accepted'].fillna(0).astype(float) == 1] - start).values
    return scored, accepted


def floor(ax, boxes) -> None:
    for px, py, sx, sy in boxes:
        ax.add_patch(Rectangle((px - sx / 2, py - sy / 2), sx, sy,
                               facecolor=RACK, edgecolor='none', zorder=1))
    ax.add_patch(Rectangle((-12, -10), 24, 20, fill=False, ec='#6d747c', lw=1.3, zorder=1))
    ax.set(xlim=(-12.8, 12.8), ylim=(-10.8, 10.8), aspect='equal', xlabel='x (m)')
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.tick_params(labelsize=8.5)


def draw(ax, rows, accepted, window, boxes, title, subtitle) -> None:
    low, high = window
    seg = rows[(rows['t'] >= low) & (rows['t'] <= high)]
    hits = accepted[(accepted >= low) & (accepted <= high)]
    floor(ax, boxes)

    for _, row in seg.iloc[::max(1, len(seg) // 10)].iterrows():
        cov = np.array([[row['planner_cov_x'], row['planner_cov_xy']],
                        [row['planner_cov_xy'], row['planner_cov_y']]])
        values, vectors = np.linalg.eigh(cov)
        values = np.clip(values, 1e-9, None)
        ax.add_patch(Ellipse((row['planner_belief_x'], row['planner_belief_y']),
                             4 * np.sqrt(values[-1]), 4 * np.sqrt(values[0]),
                             angle=np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])),
                             fc=BELIEF, alpha=.20, ec=BELIEF, lw=.8, zorder=3))

    ax.plot(seg['gt_x_at_belief_stamp'], seg['gt_y_at_belief_stamp'],
            color=TRUTH, lw=2.6, label='where the robot really was', zorder=4)
    ax.plot(seg['planner_belief_x'], seg['planner_belief_y'],
            color=BELIEF, lw=2.0, ls='--', label='where the robot thought it was', zorder=5)

    if len(hits):
        marks = rows.iloc[np.searchsorted(rows['t'].to_numpy(), hits[::max(1, len(hits) // 8)])]
        ax.plot(marks['planner_belief_x'], marks['planner_belief_y'], '*', color=MEAS,
                ms=9, mec='white', mew=.6, ls='none', zorder=6,
                label='camera correction accepted')

    ax.plot(*seg[['gt_x_at_belief_stamp', 'gt_y_at_belief_stamp']].iloc[0], 'o',
            color=START, ms=10, mec='white', mew=1.4, zorder=7, label='stretch starts here')
    ax.plot(*seg[['gt_x_at_belief_stamp', 'gt_y_at_belief_stamp']].iloc[-1], 'X',
            color=GOAL, ms=11, mec='white', mew=1.4, zorder=7, label='stretch ends here')

    ax.set_title(f'{title}\n{subtitle}', fontsize=10, fontweight='bold')
    ax.text(.03, .97, f'{len(hits)} corrections\n'
                      f'ends {100 * seg["err"].iloc[-1]:.0f} cm from the robot',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#c8ced6', alpha=.93))


def timeline(ax, rows) -> None:
    for window, colour, name in ((CORRECTED, TRUTH, 'with corrections'),
                                 (STARVED, MEAS, 'during the 45 s gap')):
        seg = rows[(rows['t'] >= window[0]) & (rows['t'] <= window[1])]
        elapsed = seg['t'] - window[0]
        ax.plot(elapsed, 100 * seg['err'], color=colour, lw=2.0, label=f'error, {name}')
        ax.plot(elapsed, 100 * seg['sigma'], color=colour, lw=1.6, ls='--',
                label=f'uncertainty the robot reported, {name}')

    ax.set(xlabel='seconds into the stretch', ylabel='centimetres', xlim=(0, 45))
    ax.set_title('The robot only knows it is lost in one of them',
                 fontsize=10, fontweight='bold')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='upper left', framealpha=.92)


def main() -> None:
    rows, accepted = load()
    boxes = racks()

    fig = plt.figure(figsize=(12.2, 6.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=.34, wspace=.16)
    left, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

    draw(left, rows, accepted, CORRECTED, boxes,
         'Cameras keep correcting the belief', 'the estimate stays on the robot')
    draw(right, rows, accepted, STARVED, boxes,
         'No camera correction arrives for 45 seconds',
         'the estimate drifts and its uncertainty grows')
    left.set_ylabel('y (m)')

    timeline(fig.add_subplot(grid[1, :]), rows)

    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.035))

    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'localization_example.pdf', OUT / 'localization_example.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'localization_example.pdf')


if __name__ == '__main__':
    main()
