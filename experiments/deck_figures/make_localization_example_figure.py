#!/usr/bin/env python3
"""Problem-statement figure: one route, two moments, from a recorded drive.

Both panels show the same drive on the warehouse floor, at two times. In the first the
cameras are still delivering and the belief sits on the robot. In the second the robot has
driven on into a stretch where nothing is accepted for 45 s, and the belief has separated
from the truth while its own stated uncertainty has grown.

Everything is logged: belief mean and covariance from the planner, ground truth sampled at
the belief timestamp as the metrics contract requires, and accepted corrections from the
runtime's own flag. Rack footprints come from the world file. Ground truth scores the
result; the estimator never saw it.

Note on the growth rate: across the starved stretch the stated standard deviation grows
126 cm while the actual error grows 39 cm, so the belief ends about 2.6x wider than its own
error warrants. The filter is conservative here, not overconfident. The figure shows what
happens when corrections stop; it is not evidence that the process noise is well tuned.
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

ROUTE_START = 120.0          # where the drawn route begins
SNAPSHOTS = (186.0, 231.0)   # corrections still arriving; 45 s later with none
GAP_OPENS = 186.5            # last accepted correction before the starved stretch

TRUTH, BELIEF, MEAS = '#2b3038', '#1f6fb8', '#c23d36'
RACK, GONE, CAM = '#b9c0c8', '#f2d9a8', '#1f6fb8'

# Camera world positions and yaws, read from the world file's include poses.
CAMERAS = {'A': (-11.45, -9.45, 0.785), 'B': (-1.50, -9.72, 2.007),
           'C': (-6.95, 9.45, -1.047), 'D': (11.45, 7.20, -2.443),
           'E': (11.45, -9.45, 2.304)}


def racks() -> list[tuple[float, float, float, float]]:
    block = WORLD.read_text()
    block = block[block.index('warehouse_v2_occluders'):]
    block = block[:block.index('</model>')]
    return [((lambda p, s: (p[0], p[1], s[0], s[1]))(
        [float(v) for v in pose.split()[:2]], [float(v) for v in size.split()[:2]]))
        for pose, size in re.findall(
            r'<pose>([^<]+)</pose>.*?<box><size>([^<]+)</size></box>', block, flags=re.S)]


def load() -> tuple[pd.DataFrame, np.ndarray]:
    matches = glob.glob(str(REPO / RUN))
    if not matches:
        raise SystemExit(f'no drive matching {RUN}')
    frame = pd.read_csv(matches[0], low_memory=False)
    stamp = frame['stamp'].astype(float)
    start = stamp.min()

    rows = frame[(frame['planner_belief_available'] == 1)
                 & (frame['gt_available'] == 1)].copy()
    rows = rows.dropna(subset=['gt_x_at_belief_stamp', 'gt_y_at_belief_stamp',
                               'planner_belief_x', 'planner_cov_x'])
    rows['t'] = rows['stamp'].astype(float) - start
    rows['err'] = np.hypot(rows['planner_belief_x'] - rows['gt_x_at_belief_stamp'],
                           rows['planner_belief_y'] - rows['gt_y_at_belief_stamp'])
    accepted = (stamp[frame['pixel_corr_accepted'].fillna(0).astype(float) == 1] - start).values
    return rows, accepted


def snapshot(ax, rows, accepted, boxes, now: float, title: str) -> None:
    seen = rows[(rows['t'] >= ROUTE_START) & (rows['t'] <= now)]
    hits = accepted[(accepted >= ROUTE_START) & (accepted <= now)]
    head = seen.iloc[-1]

    # the stretch the robot crossed without any accepted correction
    starved = seen[seen['t'] >= GAP_OPENS]
    if len(starved) > 2:
        pad = 1.1
        ax.add_patch(Rectangle(
            (starved['gt_x_at_belief_stamp'].min() - pad, starved['gt_y_at_belief_stamp'].min() - pad),
            np.ptp(starved['gt_x_at_belief_stamp']) + 2 * pad,
            np.ptp(starved['gt_y_at_belief_stamp']) + 2 * pad,
            fc=GONE, ec='none', alpha=.75, zorder=1,
            label='stretch crossed with no accepted correction'))

    for px, py, sx, sy in boxes:
        ax.add_patch(Rectangle((px - sx / 2, py - sy / 2), sx, sy,
                               facecolor=RACK, edgecolor='none', zorder=2))
    ax.add_patch(Rectangle((-12, -10), 24, 20, fill=False, ec='#6d747c', lw=1.3, zorder=2))

    for name, (cx, cy, yaw) in CAMERAS.items():
        ax.plot(cx, cy, 's', color=CAM, ms=7, mec='white', mew=1.2, zorder=7,
                label='fixed camera' if name == 'C' else None)
        ax.annotate('', xy=(cx + 1.9 * np.cos(yaw), cy + 1.9 * np.sin(yaw)), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='-|>', lw=1.5, color=CAM), zorder=7)
        ax.text(cx, cy + (0.95 if cy < 0 else -1.15), name, color=CAM, fontsize=8.5,
                fontweight='bold', ha='center', va='center', zorder=8)

    for _, row in seen.iloc[::max(1, len(seen) // 11)].iterrows():
        cov = np.array([[row['planner_cov_x'], row['planner_cov_xy']],
                        [row['planner_cov_xy'], row['planner_cov_y']]])
        values, vectors = np.linalg.eigh(cov)
        values = np.clip(values, 1e-9, None)
        ax.add_patch(Ellipse((row['planner_belief_x'], row['planner_belief_y']),
                             4 * np.sqrt(values[-1]), 4 * np.sqrt(values[0]),
                             angle=np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])),
                             fc=BELIEF, alpha=.20, ec=BELIEF, lw=.8, zorder=3))

    ax.plot(seen['gt_x_at_belief_stamp'], seen['gt_y_at_belief_stamp'],
            color=TRUTH, lw=2.6, label='where the robot really was', zorder=4)
    ax.plot(seen['planner_belief_x'], seen['planner_belief_y'],
            color=BELIEF, lw=2.0, ls='--', label='where the robot thought it was', zorder=5)

    if len(hits):
        marks = rows.iloc[np.searchsorted(rows['t'].to_numpy(), hits[::max(1, len(hits) // 9)])]
        ax.plot(marks['planner_belief_x'], marks['planner_belief_y'], '*', color=MEAS,
                ms=9, mec='white', mew=.6, ls='none', zorder=6,
                label='camera correction accepted')

    ax.plot(head['gt_x_at_belief_stamp'], head['gt_y_at_belief_stamp'], 'o', color=TRUTH,
            ms=11, mec='white', mew=1.6, zorder=8, label='robot now')

    ax.set(xlim=(-12.9, 12.9), ylim=(-10.9, 10.9), aspect='equal', xlabel='x (m)')
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.tick_params(labelsize=8.5)
    ax.set_title(f'{title}\n$t={now - ROUTE_START:.0f}$ s into the route',
                 fontsize=10.5, fontweight='bold')
    ax.text(.03, .97, f'belief is {100 * head["err"]:.0f} cm from the robot',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#c8ced6', alpha=.93))


def main() -> None:
    rows, accepted = load()
    boxes = racks()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    snapshot(axes[0], rows, accepted, boxes, SNAPSHOTS[0],
             '(a) cameras still correcting the belief')
    snapshot(axes[1], rows, accepted, boxes, SNAPSHOTS[1],
             '(b) 45 s later, after driving on with none')
    axes[0].set_ylabel('y (m)')

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8.8,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'localization_example.pdf', OUT / 'localization_example.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'localization_example.pdf')


if __name__ == '__main__':
    main()
