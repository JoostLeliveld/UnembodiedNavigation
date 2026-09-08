#!/usr/bin/env python3
"""Problem-statement figure: what a camera correction is worth, on a recorded drive.

Three panels on one drive, telling the story in time:

  (a) the route while cameras are still correcting, and the belief on the robot,
  (b) 45 s later, after crossing a stretch where nothing was accepted,
  (c) the single correction that ends that stretch, drawn as the filter update it is.

On this drive the accepted corrections stop at t = 186.5 s and resume at t = 231.6 s,
a gap of 45.1 s. Panel (c) is that resuming correction. It is worth drawing because it
is the one update on the drive large enough to see: the median accepted correction here
moves the belief 0.97 mm, since corrections normally arrive several times a second and
each only nudges. After 45 s of prediction the prior has drifted, so this correction
moves the belief 65.9 cm and takes the error from 67 cm to 8 cm.

Everything is logged: belief mean and covariance from the planner, ground truth sampled at
the belief timestamp as the metrics contract requires, and accepted corrections from the
runtime's own flag. Panel (c) reads the prior and posterior straight out of the runtime's
own correction record (pixel_corr_pred_* and pixel_corr_next_*). The update is applied in
pixel space, so the world-frame reading is not logged directly; it is reconstructed from
the logged prior, posterior and prior covariance by inverting the Kalman update, under a
stated reading noise. Rack footprints come from the world file. Ground truth scores the
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
from matplotlib.patches import (Ellipse, FancyArrowPatch, Polygon,  # noqa: E402
                                Rectangle)
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
GAP_CLOSES = 231.605         # the correction that ends it, drawn in panel (c)
READING_SIGMA_M = 0.25       # assumed reading noise, to reconstruct the world reading

TRUTH, BELIEF, MEAS = '#2b3038', '#1f6fb8', '#c23d36'
RACK, GONE, CAM = '#b9c0c8', '#f2d9a8', '#1f6fb8'
GAP_INK = '#8a6d1f'

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


def correction_record(frame: pd.DataFrame, start: float) -> dict:
    """The runtime's own record of the correction that ends the starved stretch."""
    t = frame['stamp'].astype(float) - start
    row = frame[np.isclose(t, GAP_CLOSES, atol=1e-3)]
    if row.empty:
        raise SystemExit(f'no correction logged at t={GAP_CLOSES}')
    row = row.iloc[0]
    if float(row['pixel_corr_accepted']) != 1:
        raise SystemExit(f'the correction at t={GAP_CLOSES} was not accepted')
    prior = np.array([row['pixel_corr_pred_x'], row['pixel_corr_pred_y']], dtype=float)
    post = np.array([row['pixel_corr_next_x'], row['pixel_corr_next_y']], dtype=float)
    prior_cov = np.array([[row['planner_cov_x'], row['planner_cov_xy']],
                          [row['planner_cov_xy'], row['planner_cov_y']]], dtype=float)
    truth = np.array([row['gt_x_at_belief_stamp'], row['gt_y_at_belief_stamp']],
                     dtype=float)

    # The update is applied in pixel space, so no world-frame reading is logged. Invert
    # the Kalman update for the reading that carries this prior to this posterior:
    # m+ = m- + P(P+R)^-1 (z - m-)  =>  z = m- + (P+R) P^-1 (m+ - m-).
    meas_cov = np.eye(2) * READING_SIGMA_M ** 2
    reading = prior + (prior_cov + meas_cov) @ np.linalg.inv(prior_cov) @ (post - prior)
    gain = prior_cov @ np.linalg.inv(prior_cov + meas_cov)
    post_cov = prior_cov - gain @ prior_cov
    # the reconstruction must return the posterior the runtime actually logged
    check = prior + gain @ (reading - prior)
    if not np.allclose(check, post, atol=1e-9):
        raise SystemExit(f'reading reconstruction is inconsistent: {check} vs {post}')
    return dict(prior=prior, prior_cov=prior_cov, post=post, post_cov=post_cov,
                reading=reading, meas_cov=meas_cov, truth=truth,
                yaw=float(row['pixel_corr_pred_yaw']))


def update_panel(ax, record: dict, boxes) -> None:
    """(c) the correction itself: prior, reading and posterior, on the warehouse floor."""
    prior, post = record['prior'], record['post']
    reading, truth = record['reading'], record['truth']

    for px, py, sx, sy in boxes:
        ax.add_patch(Rectangle((px - sx / 2, py - sy / 2), sx, sy,
                               facecolor=RACK, edgecolor='none', zorder=1))
    ax.add_patch(Rectangle((-12, -10), 24, 20, fill=False, ec='#6d747c', lw=1.3, zorder=1))

    def ellipse(centre, cov, colour, style, zorder):
        values, vectors = np.linalg.eigh(cov)
        values = np.clip(values, 1e-9, None)
        ax.add_patch(Ellipse(centre, 2 * np.sqrt(values[-1]), 2 * np.sqrt(values[0]),
                             angle=np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])),
                             fc=colour, alpha=.20, ec=colour, lw=1.8, ls=style,
                             zorder=zorder))

    ellipse(prior, record['prior_cov'], BELIEF, '--', 2)
    ellipse(reading, record['meas_cov'], MEAS, '-', 3)
    ellipse(post, record['post_cov'], BELIEF, '-', 5)

    # the innovation: from where the filter expected the robot to where the camera put it
    ax.annotate('', xy=reading, xytext=prior,
                arrowprops=dict(arrowstyle='-|>,head_width=0.28,head_length=0.62',
                                lw=2.1, color=MEAS, shrinkA=4, shrinkB=4), zorder=6)

    # the robot, at its true pose, drawn at the AMR's real 0.80 x 0.55 m footprint
    heading = np.array([np.cos(record['yaw']), np.sin(record['yaw'])])
    across = np.array([-heading[1], heading[0]])
    body = np.array([truth + sl * 0.40 * heading + sw * 0.275 * across
                     for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    ax.add_patch(Polygon(body, closed=True, fc='white', ec=TRUTH, lw=1.8, alpha=.45,
                         zorder=4))
    ax.annotate('', xy=truth + 0.85 * heading, xytext=truth,
                arrowprops=dict(arrowstyle='-|>,head_width=0.30,head_length=0.62',
                                lw=2.0, color=TRUTH, alpha=.85, shrinkA=1, shrinkB=0),
                zorder=6)
    ax.plot(*truth, 'o', color=TRUTH, ms=5, mec='white', mew=1.0, zorder=6)

    ax.plot(*prior, 'o', color=BELIEF, ms=7, mec='white', mew=1.2, zorder=9)
    ax.plot(*reading, '*', color=MEAS, ms=17, mec='white', mew=1.0, zorder=9)
    ax.plot(*post, 'o', color=BELIEF, ms=8, mec='white', mew=1.3, zorder=9)

    # labels, offset in metres with leaders, so they follow the window
    halo = dict(boxstyle='square,pad=0.16', fc='white', ec='none', alpha=.84)
    for anchor, colour, label, offset, ha, va in (
            (prior, BELIEF, 'predicted\n$m_k^-,S_k^-$', (-1.30, 0.42), 'right', 'center'),
            (reading, MEAS, 'reading\n$z_k,R_c$', (1.15, -0.60), 'left', 'center'),
            (post, BELIEF, 'updated\n$m_k^+,S_k^+$', (-1.30, -0.62), 'right', 'center'),
            (truth, TRUTH, 'true pose $s_k$', (1.15, 0.62), 'left', 'center')):
        ax.annotate(label, xy=anchor, xytext=anchor + np.array(offset), color=colour,
                    fontsize=9.2, fontweight='bold', ha=ha, va=va, zorder=10, bbox=halo,
                    arrowprops=dict(arrowstyle='-', lw=0.8, color=colour, alpha=.65,
                                    shrinkA=3, shrinkB=6))
    mid = (prior + reading) / 2
    ax.text(mid[0] + 0.15, mid[1] + 0.55, '$z_k-h(m_k^-)$', color=MEAS, fontsize=9.2,
            fontweight='bold', ha='center', va='bottom', zorder=10, bbox=halo)

    # The window must hold the prior ellipse, which is the widest thing drawn: after
    # 45 s of prediction the belief's own stated sigma is 1.7 m across track, against
    # an update of 0.66 m. That ratio is the point of the panel, so it is not clipped.
    spread = 1.05 * float(np.sqrt(np.max(np.linalg.eigvalsh(record['prior_cov']))))
    drawn = np.vstack([prior, reading, post, truth])
    centre = (drawn.min(axis=0) + drawn.max(axis=0)) / 2
    half_h = max(2.15, spread + 0.55)
    ax.set(xlim=(centre[0] - half_h * 1.05, centre[0] + half_h * 1.05),
           ylim=(centre[1] - half_h, centre[1] + half_h), aspect='equal', xlabel='x (m)')
    ax.tick_params(labelsize=8.5)


def main() -> None:
    rows, accepted = load()
    boxes = racks()
    matches = sorted(glob.glob(str(REPO / RUN)))
    frame = pd.read_csv(matches[0], low_memory=False)
    record = correction_record(frame, frame['stamp'].astype(float).min())

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0),
                             gridspec_kw=dict(width_ratios=(1, 1, 0.92)))
    snapshot(axes[0], rows, accepted, boxes, SNAPSHOTS[0],
             '(a) cameras still correcting')
    snapshot(axes[1], rows, accepted, boxes, SNAPSHOTS[1],
             '(b) 45 s later, none accepted')
    update_panel(axes[2], record, boxes)
    axes[2].set_title('(c) the correction that ends the gap\n'
                      f'$t={GAP_CLOSES - ROUTE_START:.0f}$ s into the route',
                      fontsize=10.5, fontweight='bold')
    axes[0].set_ylabel('y (m)')

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8.8,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    # The temporal connector: the 45 s of prediction that separates (a) from (b) is the
    # point of the figure, so it is drawn between those panels rather than left to the
    # caption. It is placed after tight_layout, in figure coordinates.
    left, right = axes[0].get_position(), axes[1].get_position()
    x0, x1 = left.x1 + 0.004, right.x0 - 0.004
    y = (left.y0 + left.y1) / 2
    fig.patches.append(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure, color=GAP_INK, lw=2.6,
        arrowstyle='-|>', mutation_scale=20, zorder=5))
    fig.text((x0 + x1) / 2, y + 0.030, '45 s', fontsize=11, fontweight='bold',
             color=GAP_INK, ha='center', va='bottom')
    fig.text((x0 + x1) / 2, y - 0.034, 'no accepted\ncorrection\n\nprediction only',
             fontsize=8.6, color=GAP_INK, ha='center', va='top')

    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'localization_example.pdf', OUT / 'localization_example.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'localization_example.pdf')
    for name in ('prior', 'reading', 'post'):
        print(f'  {name} {np.round(record[name], 3)}, '
              f'{100 * np.linalg.norm(record[name] - record["truth"]):.0f} cm from truth')
    print(f'  correction moves the belief '
          f'{100 * np.linalg.norm(record["post"] - record["prior"]):.1f} cm')


if __name__ == '__main__':
    main()
