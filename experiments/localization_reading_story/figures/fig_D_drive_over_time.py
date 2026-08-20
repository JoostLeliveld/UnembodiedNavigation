#!/usr/bin/env python3
"""D1 -- the drive: what each pass of the R loop does to the estimate, over time.

The static capture cannot show this, because a standing robot has nothing for R to trade
against: the belief is the average of the readings whatever R says. On a drive there IS a
trade, between the odometry and the camera, and R sets it. So each pass of the loop bends
the estimate further onto the readings -- which is what the box bottom's residuals were
shrinking towards all along, while its error stayed where it was.

One column per reading, both from the SAME frames of the SAME drive. Rows are the two
axes; the route runs north up the aisle, so the north row is where the box bottom's lean
lives.

Run: python3 experiments/localization_reading_story/figures/fig_D_drive_over_time.py <drive>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drive_filter as df  # noqa: E402
import reading_data as rd  # noqa: E402

sys.path.insert(0, str(rd.REPO_ROOT / 'experiments/filter_notebook'))
import notebook_model as nm  # noqa: E402

TITLE = {'box_bottom': 'bottom of the detected box  (the deployed reading)',
         'keypoint': 'the two marker disks  (retrained)'}
COLOUR = {'box_bottom': '#9b2226', 'keypoint': '#2a9d8f'}
PASS_COLOUR = {1: '#a7c957', 'last': '#3d348b'}


def main() -> None:
    drive = Path(sys.argv[1]).expanduser().resolve()
    rd.style()
    results = {key: df.run(drive, key) for key in df.READINGS}

    fig, axes = plt.subplots(3, 2, figsize=(15.2, 10.4), sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [1.0, 1.0, 0.5]})
    for column, key in enumerate(df.READINGS):
        seq, history = results[key]['seq'], results[key]['history']
        time = seq.stamps - seq.stamps[0]
        obs = seq.observed
        truth_ok = np.isfinite(seq.truth[:, 0])
        last = len(history) - 1

        forwards = {p: nm.kalman_filter(seq, history[p]['R_in']) for p in (1, last)}
        for row, axis_name in enumerate(('north (along the aisle)', 'east (across it)')):
            axis = 1 - row          # north is component 1, east is component 0
            ax = axes[row, column]
            ax.axhline(0, color=rd.INK, lw=1.8, zorder=4,
                       label='the truth' if row == 0 and column == 0 else None)
            ax.scatter(time[obs], 100 * (seq.y[obs, axis] - seq.truth[obs, axis]),
                       s=12, lw=0, alpha=0.5, color=COLOUR[key],
                       label='the camera\'s readings' if row == 0 else None)
            # The claim is drawn at the steps where a reading was just absorbed: between
            # them the covariance grows again, and plotting that makes a picket fence of
            # the band without adding anything.
            used = forwards[last]['used'] & truth_ok
            band = 100 * np.sqrt(forwards[last]['P'][:, axis, axis]) * 2
            error = 100 * (forwards[last]['m'][:, axis] - seq.truth[:, axis])
            ax.fill_between(time[used], (error - band)[used], (error + band)[used],
                            color=PASS_COLOUR['last'], alpha=0.16, lw=0,
                            label='what the last pass CLAIMS, 2 sd' if row == 0 else None)
            for p, colour, name in ((1, PASS_COLOUR[1], 'pass 1'),
                                    (last, PASS_COLOUR['last'], f'pass {last}')):
                err = 100 * (forwards[p]['m'][:, axis] - seq.truth[:, axis])
                R = 1e4 * history[p]['R_in'][df.CAMERA]
                ax.plot(time[truth_ok], err[truth_ok], lw=1.9, color=colour, zorder=5,
                        label=(f'{name}  (R = {math.sqrt(np.trace(R) / 2):.2f} cm)'
                               if row == 0 else None))
            ax.set_ylabel(f'estimated position minus truth\n{axis_name}, cm')
            if row == 0:
                ok = np.isfinite(seq.truth[:, 0])
                final = forwards[last]
                belief = 100 * (final['m'][ok] - seq.truth[ok])
                nees = np.array([float(e @ np.linalg.inv(1e4 * P) @ e)
                                 for e, P in zip(belief, final['P'][ok])])
                inside = 100 * float(np.mean(nees <= nm.GATE_CHI2_2DOF))
                ax.set_title(f'{TITLE[key]}\n'
                             f'{seq.n_readings} readings in '
                             f'{seq.stamps[-1] - seq.stamps[0]:.0f} s   |   after the last '
                             f'pass the belief is {np.median(np.hypot(*belief.T)):.2f} cm '
                             f'from truth\nand the truth is inside its own 95% region '
                             f'{inside:.0f}% of the time',
                             loc='left', fontsize=10.4, color=COLOUR[key])
    # ---- a strip showing where the robot turned, so the steps in the error can be read
    for column, key in enumerate(df.READINGS):
        seq = results[key]['seq']
        time = seq.stamps - seq.stamps[0]
        ax = axes[2, column]
        heading = np.degrees((seq.truth_yaw + np.pi) % (2 * np.pi) - np.pi)
        ax.plot(time, heading, lw=1.8, color=rd.MUTED)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_yticklabels(['south', 'east', 'north', 'west', 'south'], fontsize=8.5)
        ax.set_ylim(-200, 200)
        ax.set_ylabel('which way the\nrobot faces')
        ax.set_xlabel('time since the start of the drive, seconds')
    axes[0, 0].legend(fontsize=8.4, loc='lower left', ncols=2)

    box, kp = results['box_bottom'], results['keypoint']
    def belief_error(result):
        seq, history = result['seq'], result['history']
        ok = np.isfinite(seq.truth[:, 0])
        m = nm.kalman_filter(seq, history[-1]['R_in'])['m']
        return float(np.median(np.hypot(*(100 * (m[ok] - seq.truth[ok])).T)))
    fig.suptitle('Each pass bends the estimate further onto the readings -- which is a '
                 'repair on one reading and a mistake on the other.\n'
                 f'On the same drive and the same frames the belief ends '
                 f'{belief_error(box):.1f} cm from truth on the box bottom and '
                 f'{belief_error(kp):.1f} cm on the marked point',
                 fontsize=12.8, fontweight='bold', y=0.995)
    rd.note(fig, f'One drive through warehouse_aws with the marker disks '
                 f'rendered ({drive.name}), 1280x720 frames read twice: the frozen detector '
                 f'`warehouse_yolo_detector_v1` for the box bottom and `yolo_pose_aws_v4` for '
                 f'the marked point, both at 960 px. Odometry is the recorded noisy stream; '
                 f'truth is used only to draw these differences. The filter, the gate, the '
                 f'process noise and the R-learning loop are '
                 f'experiments/filter_notebook/notebook_model.py, unmodified -- the only '
                 f'difference between the two columns is which pixel the reading came from. '
                 f'Evidence: {drive.relative_to(rd.REPO_ROOT)}.', width=200)
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.075, right=0.99, hspace=0.16,
                        wspace=0.13)
    tag = drive.name.replace('markers_', '').rsplit('_', 2)[0]
    rd.save(fig, f'D1_drive_{tag}')


if __name__ == '__main__':
    main()
