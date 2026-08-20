#!/usr/bin/env python3
"""L1 -- the method ladder: every way the camera has been asked to read position.

Two panels, because the seven rows are not all evidence of the same kind:

  left   the three readings measured on the SAME 423 held-out images, with each
         one's error split into the part a filter can average away (random) and
         the part it cannot (systematic).
  right  the ladder as a whole. Rows 2-5 were corrections applied to row 1 and
         were measured on other data, so they carry their own result and are
         drawn as such -- never as bars on the same axis.

Run: python3 experiments/localization_reading_story/figures/fig_L1_ladder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reading_data as rd  # noqa: E402

# Rows 2-5: corrections to row 1, each measured on its own data. Quoted, with the
# file that measured them, never recomputed here.
INTERMEDIATE = [
    (2, 'fitted projection corrections (v2/v3/v4)',
     'RETIRED: every fitted correction was worse than none', True),
    (3, 'plain IPM, zero fitted parameters',
     'the honest baseline -- no fitted term survives holdout', False),
    (4, 'mesh / silhouette bias correction',
     'lean 9.50 -> 1.93 cm, on drives in the four-camera world', False),
    (5, 'offset-state filter (offset estimated as state)',
     'recovers the offset without truth, to 0.7-2.4 cm', False),
]

# Named in the provenance line rather than crowded onto the ladder rows.
EVIDENCE = ('rows 1, 6, 7: logs/studies/keypoint_measurement/RESULTS.md  |  '
            'row 2-3: logs/studies/pixel_ground_path/  |  '
            'row 4: logs/studies/filter_notebook/RESULT_learning_r_after_the_bias_fix_2026-08-14.md  |  '
            'row 5: logs/studies/offset_state_model/')

LIKE_FOR_LIKE = [
    ('box_bottom', 1, 'bottom of the detected box\n(the deployed reading)'),
    ('keypoint_old_camera', 6, 'marked point,\nmodel for the OLD camera'),
    ('keypoint_retrained', 7, 'marked point,\nretrained for THIS camera'),
]


def main() -> None:
    rd.style()
    stats = {k: rd.summarise(rd.load_reading(k)) for k, _, _ in LIKE_FOR_LIKE}

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(14.4, 5.8),
                                  gridspec_kw={'width_ratios': [1.0, 1.25]})

    # ---------------------------------------------------------------- left: the split
    ys = range(len(LIKE_FOR_LIKE))
    for y, (key, row, label) in zip(ys, LIKE_FOR_LIKE):
        s = stats[key]
        ax.barh(y, s['systematic_cm'], height=0.52, color=rd.COLOUR[key],
                label='the same error every time (systematic)' if y == 0 else None)
        ax.barh(y, s['spread_cm'], height=0.52, left=s['systematic_cm'],
                color=rd.COLOUR[key], alpha=0.33, hatch='///', edgecolor='white',
                label='scatter around it (random)' if y == 0 else None)
        ax.text(s['systematic_cm'] + s['spread_cm'] + 0.25, y,
                f"typical miss {s['median_miss_cm']:.2f} cm\n"
                f"found the robot {s['found_pct']:.1f}% of the time",
                va='center', fontsize=8.5, color=rd.INK)
        if s['systematic_cm'] > 1.0:
            ax.text(s['systematic_cm'] / 2, y, f"{s['systematic_cm']:.2f}", ha='center',
                    va='center', fontsize=9, color='white', fontweight='bold')
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f'{row}. {label}' for _, row, label in LIKE_FOR_LIKE], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 15.5)
    ax.set_xlabel('centimetres of error  (shorter is better)')
    ax.set_title('The deployed reading is mostly bias, so a filter is stuck with it;\n'
                 'the marked point is mostly scatter, which averaging removes',
                 loc='left', fontsize=10.5)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(axis='y', visible=False)

    # ---------------------------------------------------------------- right: the ladder
    axr.set_axis_off()
    axr.set_xlim(0, 1); axr.set_ylim(0, 1)
    rows = [(1, 'bottom of the detected box -> floor  (DEPLOYED)',
             f"{stats['box_bottom']['median_miss_cm']:.2f} cm typical miss, "
             f"{stats['box_bottom']['mean_north']:+.2f} cm the same way every time",
             False, 'box_bottom')]
    rows += [(n, name, result, retired, None) for n, name, result, retired in INTERMEDIATE]
    rows += [(6, 'marked point, model for the old camera',
              f"{stats['keypoint_old_camera']['median_miss_cm']:.2f} cm typical miss, "
              f"and a heading for the first time",
              False, 'keypoint_old_camera'),
             (7, 'marked point, RETRAINED for this camera',
              f"{stats['keypoint_retrained']['median_miss_cm']:.2f} cm typical miss, "
              f"{stats['keypoint_retrained']['mean_north']:+.2f} cm systematic, heading 3.8 deg",
              False, 'keypoint_retrained')]

    top, step = 0.90, 0.125
    for i, (n, name, result, retired, key) in enumerate(rows):
        y = top - i * step
        colour = rd.COLOUR[key] if key else rd.MUTED
        axr.add_patch(plt.Rectangle((0.0, y - 0.048), 0.028, 0.082,
                                    color=colour, alpha=1.0 if key else 0.45,
                                    transform=axr.transAxes, clip_on=False))
        axr.text(0.012, y - 0.007, str(n), ha='center', va='center', fontsize=9,
                 color='white', fontweight='bold')
        axr.text(0.045, y + 0.016, name, fontsize=9.5, color=rd.INK,
                 fontweight='bold' if key else 'normal')
        axr.text(0.045, y - 0.026, result, fontsize=8.5,
                 color='#9b2226' if retired else rd.MUTED, style='italic')
        axr.text(0.999, y - 0.005,
                 'measured on the\nsame 423 images' if key else 'measured on\nother data',
                 fontsize=7.4, color=rd.INK if key else rd.MUTED, ha='right',
                 va='center', linespacing=1.35)
    axr.plot([0.014, 0.014], [top - 0.048, top - (len(rows) - 1) * step - 0.048],
             color=rd.MUTED, lw=1.0, alpha=0.35, zorder=0)
    axr.set_title('Rows 2-5 corrected the box bottom; rows 6-7 stopped using it.\n'
                  'The bottom edge of a silhouette is not a fixed point on the robot,\n'
                  'so every correction to it was fighting a symptom.',
                  loc='left', fontsize=10.5)

    fig.suptitle('Seven ways to read the robot\'s position: the marked point misses by '
                 'ten times less than the box bottom, without a calibration step',
                 fontsize=13, fontweight='bold', y=1.06)
    rd.note(fig, rd.SOURCE + '\nRows 2-5 quote their own study on their own data: they are '
                             'corrections applied to row 1, not replacements for it.\n' + EVIDENCE)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    rd.save(fig, 'L1_method_ladder')


if __name__ == '__main__':
    main()
