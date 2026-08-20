#!/usr/bin/env python3
"""L3 -- one frame, both readings, so the ladder's top and bottom rows become concrete.

Panel A  the whole camera frame: the robot is 24 x 25 pixels at 5.9 m.
Panel B  the same robot, magnified: the box the detector drew, the pixel the deployed
         reading uses (the bottom edge, centred), and the two marker keypoints the new
         reading uses -- predicted against where the markers actually project to.
Panel C  what those pixels become on the floor: truth, the box-bottom position, the
         marked-point position, in centimetres.

The pose is one of the 423 held-out poses, picked for being typical rather than
flattering: its box-bottom miss is 7.9 cm against that reading's 7.8 cm median.
Nothing is re-detected here -- the pixels are the ones the two scored runs recorded.

Run: python3 experiments/localization_reading_story/figures/fig_L3_what_the_camera_sees.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reading_data as rd  # noqa: E402

SAMPLE = 119        # held-out pose index; see the module docstring for why this one
CROP = 46           # half-width of the magnified crop, pixels
FRONT_C, REAR_C = '#3fbfd6', '#c2569b'


def main() -> None:
    rd.style()
    import cv2

    bb = rd.load_reading('box_bottom')
    kp = rd.load_reading('keypoint_retrained')
    b, k = bb.rows[SAMPLE], kp.rows[SAMPLE]
    image = cv2.cvtColor(cv2.imread(str(bb.images[SAMPLE])), cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    truth = np.array([float(b['x']), float(b['y'])])
    box_u, box_v = float(b['obs_u']), float(b['obs_v'])
    box_w, box_h = float(b['box_width_px']), float(b['box_height_px'])
    box_est = np.array([float(b['est_x']), float(b['est_y'])])
    kp_est = np.array([float(k['est_x']), float(k['est_y'])])
    rng = float(b['range_m'])

    fig = plt.figure(figsize=(15.2, 5.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.16)

    # ------------------------------------------------------------------ A: whole frame
    axa = fig.add_subplot(grid[0, 0])
    axa.imshow(image)
    axa.add_patch(Rectangle((box_u - CROP, box_v - CROP - box_h / 2), 2 * CROP, 2 * CROP,
                            fill=False, edgecolor='#f4a261', lw=1.8))
    axa.annotate(f'the robot: {box_w:.0f} x {box_h:.0f} pixels at {rng:.1f} m',
                 xy=(box_u - CROP, box_v - box_h / 2), xytext=(box_u - 150, box_v - 165),
                 fontsize=9.5, color='#e07b39', ha='right', fontweight='bold',
                 arrowprops={'arrowstyle': '-', 'color': '#e07b39', 'lw': 1.2})
    axa.set_xlim(0, img_w); axa.set_ylim(img_h, 0)
    axa.set_xticks([]); axa.set_yticks([])
    axa.grid(False)
    axa.set_title('A.  What the one camera sees: 1280 x 720, looking north\n'
                  'from the south wall, 4.8 m up', loc='left', fontsize=10.5)

    # ------------------------------------------------------------------ B: the crop
    axb = fig.add_subplot(grid[0, 1])
    cu, cv_ = box_u, box_v - box_h / 2
    axb.imshow(image)
    axb.add_patch(Rectangle((box_u - box_w / 2, box_v - box_h), box_w, box_h,
                            fill=False, edgecolor='#f4a261', lw=1.6,
                            label='the box the detector drew'))
    axb.plot([box_u], [box_v], 'v', color=rd.BOX_BOTTOM, ms=13, mec='white', mew=1.0,
             label='pixel the deployed reading uses\n(bottom edge, centred)')
    for tag, colour, pu, pv, gu, gv in (
            ('front marker', FRONT_C, float(k['pred_front_u']), float(k['pred_front_v']),
             float(k['gt_front_u']), float(k['gt_front_v'])),
            ('rear marker', REAR_C, float(k['pred_rear_u']), float(k['pred_rear_v']),
             float(k['gt_rear_u']), float(k['gt_rear_v']))):
        axb.plot([gu], [gv], 'x', color=colour, ms=9, mew=2.0,
                 label=f'where the {tag} really is')
        axb.add_patch(Circle((pu, pv), 2.4, facecolor='none', edgecolor=colour, lw=2.0))
        axb.plot([], [], 'o', mfc='none', mec=colour, mew=2.0, ms=8,
                 label=f'{tag}, as predicted')
    axb.set_xlim(cu - CROP, cu + CROP)
    axb.set_ylim(cv_ + CROP, cv_ - CROP)
    axb.set_xticks([]); axb.set_yticks([])
    axb.grid(False)
    axb.legend(loc='upper left', fontsize=7.6, labelspacing=0.55,
               facecolor='white', framealpha=0.82, frameon=True)
    res = np.hypot(float(k['res_front_u']), float(k['res_front_v']))
    axb.set_title('B.  Magnified: the two readings use different pixels\n'
                  f'predicted marker within {res:.1f} px of where it projects to',
                  loc='left', fontsize=10.5)

    # ------------------------------------------------------------------ C: on the floor
    axc = fig.add_subplot(grid[0, 2])
    cam = rd.camera()
    axc.plot([0], [0], 'o', mfc='none', mec=rd.INK, mew=2.2, ms=17, zorder=6,
             label='where the robot actually is')
    for point, colour, marker, label in (
            (box_est, rd.BOX_BOTTOM, 'v', 'bottom of the box says'),
            (kp_est, rd.KEYPOINT, 's', 'marked point says')):
        d = 100.0 * (point - truth)
        axc.plot([d[0]], [d[1]], marker, color=colour, ms=11, mec='white', mew=1.2,
                 label=label, zorder=4)
    for point, colour in ((box_est, rd.BOX_BOTTOM), (kp_est, rd.KEYPOINT)):
        d = 100.0 * (point - truth)
        axc.annotate('', xy=(d[0], d[1]), xytext=(0, 0),
                     arrowprops={'arrowstyle': '->', 'color': colour, 'lw': 1.4,
                                 'alpha': 0.8})
        axc.text(d[0] + 0.8, d[1] - 1.0, f'{np.hypot(*d):.2f} cm off',
                 fontsize=9, color=colour, fontweight='bold')
    bearing = math.atan2(truth[1] - cam.cam_pos[1], truth[0] - cam.cam_pos[0])
    axc.annotate('', xy=(-6.0 * math.cos(bearing), -6.0 * math.sin(bearing)),
                 xytext=(-2.0 * math.cos(bearing), -2.0 * math.sin(bearing)),
                 arrowprops={'arrowstyle': '->', 'color': rd.MUTED, 'lw': 1.6})
    axc.text(-6.6 * math.cos(bearing), -6.6 * math.sin(bearing), 'towards the camera',
             fontsize=8.5, color=rd.MUTED, ha='center', va='top')
    axc.set_aspect('equal')
    axc.set_xlim(-9.5, 9.5); axc.set_ylim(-9.5, 9.5)
    axc.axhline(0, color=rd.MUTED, lw=0.6, ls=':')
    axc.axvline(0, color=rd.MUTED, lw=0.6, ls=':')
    axc.set_xlabel('centimetres east of the true position')
    axc.set_ylabel('centimetres north of the true position')
    axc.legend(loc='upper left', fontsize=8.2)
    axc.set_title('C.  On the floor: the box bottom lands 8 cm short,\n'
                  'the marked point within a couple of millimetres',
                  loc='left', fontsize=10.5)

    fig.suptitle('The same frame read two ways: the deployed reading uses the box\'s bottom '
                 'edge, the new one uses two disks that are fixed to the robot',
                 fontsize=13.5, fontweight='bold', y=1.0)
    rd.note(fig, f'One held-out pose (#{SAMPLE} of 423, {rd.DATASET.name}, image '
                 f'{bb.images[SAMPLE].name}, robot at ({truth[0]:.2f}, {truth[1]:.2f}) m, '
                 f'facing {math.degrees(float(b["yaw_rad"])):.0f} deg, {rng:.2f} m from the '
                 f'camera). Pixels and positions are the ones recorded by the two scored runs '
                 f'in logs/studies/keypoint_measurement/ -- nothing is re-detected here. '
                 f'Its box-bottom miss ({100 * np.hypot(*(box_est - truth)):.1f} cm) is typical '
                 f'of that reading (median 7.78 cm).')
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.035, right=0.99)
    rd.save(fig, 'L3_what_the_camera_sees')


if __name__ == '__main__':
    main()
