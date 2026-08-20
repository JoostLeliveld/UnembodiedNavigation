#!/usr/bin/env python3
"""B1-B3 -- the same R-learning filter on both readings, kept strictly apart.

(B for belief. Part 3 of the README owes figures it calls F1/F2 about the pixel-space
update; these are not those, and the names are kept apart on purpose -- though B1's
middle-and-bottom rows are the health-check-looks-fine picture that F1 was asking for.)

One figure per reading, identical layout and identical axis scales so they can be laid
side by side, plus a third figure that puts only the verdict next to each other.

Each per-reading figure has the notebook's three questions as rows, at three passes of the
loop (the prior, one pass, ten passes):

  1  what the loop believes R is, against where the errors really fell
  2  what that belief predicts for the next reading, against where readings landed
  3  what the belief then SAYS about the robot's position, against where the robot was

Row 3 is the one that decides whether any of it was honest, and it is the row the
deployed reading fails: the loop gets R right, the forecast looks healthy, and the stated
belief is still 6.9 cm from the truth with a 2.0 cm error bar on it.

Run: python3 experiments/localization_reading_story/figures/fig_F_filter_both_readings.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import filter_on_both_readings as fb  # noqa: E402
import reading_data as rd  # noqa: E402

PICKS = (0, 1, 10)
BANNER = {
    'box_bottom': ('THE DEPLOYED READING  --  bottom of the detected box',
                   'The loop learns the scatter it can see and ends up 7.0 cm from the '
                   'truth while claiming 1.6 cm: every reading of a place carries the same '
                   'lean, so the lean lands in the position estimate and never in a '
                   'residual. The truth falls inside its own 95% region 8% of the time'),
    'keypoint_retrained': ('THE NEW READING  --  the two marker disks',
                           'Same loop, same code, same places: the belief sits 0.6 cm from '
                           'the truth while claiming 0.6 cm, and the truth falls inside its '
                           'stated 95% region 90% of the time. What costs it the last few '
                           'points is its own half-centimetre of leftover bias'),
}
TRUTH_C, BELIEF_C, OBS_C = '#22333b', '#457b9d', '#8a8f8a'


def ellipse(ax, centre, cov, n_sigma, colour, alpha, label=None, lw=2.0, ls='-'):
    vals, vecs = np.linalg.eigh(cov)
    e = Ellipse(centre, 2 * n_sigma * math.sqrt(max(vals[-1], 1e-12)),
                2 * n_sigma * math.sqrt(max(vals[0], 1e-12)),
                angle=math.degrees(math.atan2(vecs[1, -1], vecs[0, -1])),
                facecolor=colour, alpha=alpha, edgecolor=colour, lw=lw, ls=ls,
                label=label, zorder=4)
    ax.add_patch(e)
    return e


def square(ax, span, xlabel, ylabel=None):
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color=rd.MUTED, lw=0.6, ls=':')
    ax.axvline(0, color=rd.MUTED, lw=0.6, ls=':')
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def one_reading(key: str, result: dict, spans: dict) -> None:
    rd.style()
    history = result['history']
    cell_list = result['cells']
    errors = result['truth_errors_cm']
    picks = [p for p in PICKS if p < len(history)]

    fig = plt.figure(figsize=(4.5 * len(picks), 13.6))
    grid = fig.add_gridspec(4, len(picks), height_ratios=[1.2, 1.2, 1.2, 0.85],
                            hspace=0.42, wspace=0.24)

    for column, p in enumerate(picks):
        R = history[p]['R_in']
        R_cm = 1e4 * R
        run = fb.sweep(cell_list, R)
        told = fb.honesty(cell_list, R)

        # ---- row 1: R against where the errors really fell
        ax = fig.add_subplot(grid[0, column])
        ax.scatter(errors[:, 0], errors[:, 1], s=11, lw=0, alpha=0.35, color=OBS_C,
                   label='where the errors fell' if column == 0 else None)
        ellipse(ax, (0, 0), R_cm, 2.0, '#f4a261', 0.28,
                'what the loop believes R is, at 2 sd' if column == 0 else None, lw=2.2)
        ax.plot(0, 0, marker='+', ms=13, mew=2.2, color=TRUTH_C, zorder=6)
        when = 'before any learning' if p == 0 else f"after {p} pass{'' if p == 1 else 'es'}"
        ax.set_title(f'{when}\n{math.sqrt(R_cm[0, 0]):.2f} cm across the aisle, '
                     f'{math.sqrt(R_cm[1, 1]):.2f} cm along it', fontsize=10.4)
        square(ax, spans['error'], 'error across the aisle, cm',
               'WHAT IT BELIEVES R IS\n\nerror along the aisle, cm' if column == 0 else None)
        if column == 0:
            ax.legend(fontsize=7.6, loc='upper left')

        # ---- row 2: the forecast against where readings landed
        ax = fig.add_subplot(grid[1, column])
        inno = 100 * run['innovations']
        S_typ = 1e4 * np.median(run['S'], axis=0)
        ax.scatter(inno[:, 0], inno[:, 1], s=13, lw=0, alpha=0.40, color=BELIEF_C,
                   label='where each reading landed,\nrelative to its prediction'
                         if column == 0 else None)
        ellipse(ax, (0, 0), S_typ, math.sqrt(fb.GATE_CHI2_2DOF), '#f4a261', 0.17,
                'the 95% region it forecasts' if column == 0 else None, lw=1.9)
        ax.plot(0, 0, marker='+', ms=13, mew=2.2, color=BELIEF_C, zorder=6)
        miss = float(np.median(np.hypot(*inno.T)))
        width = float(np.median([math.sqrt(np.trace(1e4 * S) / 2) for S in run['S']]))
        ax.set_title(f'readings land {miss:.2f} cm from the prediction;\n'
                     f'it forecasts $\\pm${width:.2f} cm', fontsize=10.4)
        square(ax, spans['inno'], 'miss across the aisle, cm',
               'WHAT IT PREDICTS\n\nmiss along the aisle, cm' if column == 0 else None)
        if column == 0:
            ax.legend(fontsize=7.4, loc='lower left')

        # ---- row 3: the belief it states, against where the robot actually was
        ax = fig.add_subplot(grid[2, column])
        err = told['errors_cm']
        ax.scatter(err[:, 0], err[:, 1], s=18, lw=0, alpha=0.55, color=BELIEF_C,
                   label='where the belief ended up,\none per place' if column == 0 else None)
        ellipse(ax, (0, 0), np.median(told['covs_cm2'], axis=0),
                math.sqrt(fb.GATE_CHI2_2DOF), '#f4a261', 0.20,
                'the 95% region it states\nfor its own position' if column == 0 else None,
                lw=1.9)
        ax.plot(0, 0, marker='+', ms=14, mew=2.4, color=TRUTH_C, zorder=6,
                label='where the robot actually was' if column == 0 else None)
        ax.set_title(f"belief {told['median_error_cm']:.2f} cm off, claiming "
                     f"$\\pm${told['median_stated_sigma_cm']:.2f} cm\n"
                     f"truth inside its 95% region: "
                     f"{told['coverage_95_pct']:.0f}%   (honest = 95%)",
                     fontsize=9.9,
                     color='#9b2226' if told['coverage_95_pct'] < 80 else '#2a9d8f')
        square(ax, spans['belief'], 'belief error across the aisle, cm',
               'WHAT IT SAYS ABOUT ITSELF\n\nbelief error along the aisle, cm'
               if column == 0 else None)
        if column == 0:
            ax.legend(fontsize=7.4, loc='upper left')

    # ---- row 4: the objective
    ax = fig.add_subplot(grid[3, :])
    bound = np.array([h['elbo'] for h in history])
    steps = np.arange(len(history))
    settled = int(np.argmax(bound >= bound.max() - 1e-3))
    ax.plot(steps, bound, marker='o', ms=4, lw=2.2, color=BELIEF_C,
            label='the ELBO -- what the loop provably climbs')
    ax.plot(steps, [h['plug_in'] for h in history], marker='o', ms=3, lw=1.8, ls='--',
            color=OBS_C, label='the plug-in fit -- a different quantity, not the objective')
    from matplotlib.transforms import blended_transform_factory
    at_bottom = blended_transform_factory(ax.transData, ax.transAxes)
    for p in picks:
        ax.axvline(p, color='#e07b39', lw=1.0, ls=':')
        ax.annotate('the prior' if p == 0 else f"{p} pass{'' if p == 1 else 'es'}",
                    (p, 0.04), xycoords=at_bottom, textcoords='offset points',
                    xytext=(5, 0), fontsize=8.6, color='#e07b39')
    ax.set_xlim(-0.4, len(history) - 0.6)
    ax.set_xlabel('passes of the loop completed')
    ax.set_ylabel('the score being maximised')
    ax.set_title(f'The ELBO rises {bound[0]:.0f} -> {bound.max():.0f}, never falls, and '
                 f'stops moving after {settled} passes -- so ten passes and a hundred are '
                 f'the same answer', fontsize=10.6)
    ax.legend(fontsize=8.8, loc='lower right')

    headline, sub = BANNER[key]
    import textwrap
    fig.suptitle(f'{headline}\n' + '\n'.join(textwrap.wrap(sub, 96)),
                 fontsize=12.2, fontweight='bold', y=0.997, linespacing=1.4)
    rd.note(fig, f'{len(cell_list)} floor cells holding '
                 f'{int(sum(len(c["y"]) for c in cell_list))} readings of the same places at '
                 f'different headings, from the 423 held-out poses of {rd.DATASET.name} '
                 f'(2026-08-17). Same loop for both readings: the inverse-Wishart prior '
                 f'(nu=6, 5 cm) is the notebook\'s, and the ELBO\'s inverse-Wishart terms '
                 f'are imported from experiments/filter_notebook/notebook_model.py. These are teleported '
                 f'poses, so there is no odometry and none is invented -- each place is a '
                 f'standing robot read several times. Ground truth enters only in row 3, to '
                 f'score what the belief claimed. Recomputed by '
                 f'experiments/localization_reading_story/filter_on_both_readings.py.',
            width=205)
    fig.subplots_adjust(top=0.885, bottom=0.085, left=0.085, right=0.985)
    rd.save(fig, 'B1_filtered_box_bottom' if key == 'box_bottom'
            else 'B2_filtered_marked_point')


ORDER = ('box_bottom', 'box_bottom_heading_aware', 'keypoint_retrained')
PANEL_NAME = {
    'box_bottom': 'bottom of the detected box\n(the deployed reading)',
    'box_bottom_heading_aware': 'the same box bottom, lean corrected\nper heading '
                                '(and given the true heading)',
    'keypoint_retrained': 'the two marker disks\n(retrained, no correction)',
}
PANEL_COLOUR = {'box_bottom': '#9b2226', 'box_bottom_heading_aware': '#e07b39',
                'keypoint_retrained': '#2a9d8f'}


def verdict(results: dict, spans: dict) -> None:
    """B3 -- R passes its own test; the position fails. Those are the same event.

    The framing matters, and it is the supervisor's point: R is the spread around the
    prediction, and 1 cm is a perfectly reasonable value for it. So "coverage fell when R got
    tighter" is not an argument against learning R -- it is what happens when a CORRECT
    variance is put next to a wrong mean. This figure therefore shows both tests: the one R
    can be held to (its innovations, no ground truth) and the one the belief can be held to
    (its position, ground truth), pass by pass.
    """
    rd.style()
    fig = plt.figure(figsize=(15.6, 11.0))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.9], hspace=0.46, wspace=0.20)

    for column, key in enumerate(ORDER):
        ax = fig.add_subplot(grid[0, column])
        result = results[key]
        told, consistent = result['honesty'], result['consistency']
        err = told['errors_cm']
        colour = PANEL_COLOUR[key]
        ax.scatter(err[:, 0], err[:, 1], s=20, lw=0, alpha=0.6, color=BELIEF_C,
                   label='where the belief ended up,\none per place' if column == 0 else None)
        ellipse(ax, (0, 0), np.median(told['covs_cm2'], axis=0),
                math.sqrt(fb.GATE_CHI2_2DOF), '#f4a261', 0.22,
                'the 95% region it states' if column == 0 else None, lw=2.0)
        ax.plot(0, 0, marker='+', ms=15, mew=2.4, color=TRUTH_C, zorder=6,
                label='where the robot actually was' if column == 0 else None)
        ax.set_title(f'{PANEL_NAME[key]}\n'
                     f"R says $\\pm${told['median_stated_sigma_cm']:.2f} cm and its own test "
                     f"agrees (NIS {consistent['mean_nis']:.2f}, right = 2)\n"
                     f"but the mean is {told['median_error_cm']:.2f} cm out, so coverage is "
                     f"{told['coverage_95_pct']:.0f}%",
                     fontsize=10.2, color=colour, loc='left')
        square(ax, spans['belief'], 'belief error across the aisle, cm',
               'belief error along the aisle, cm' if column == 0 else None)
        if column == 0:
            ax.legend(fontsize=8.0, loc='upper left')

    # ---- bottom left: the test R can be held to
    ax = fig.add_subplot(grid[1, 0:2])
    for key in ORDER:
        result = results[key]
        nis = [fb.innovation_consistency(result['cells'], h['R_in'])['mean_nis']
               for h in result['history']]
        ax.plot(range(len(nis)), nis, marker='o', ms=5, lw=2.4, color=PANEL_COLOUR[key],
                label=f"{PANEL_NAME[key].splitlines()[0]} -- ends at {nis[-1]:.2f}")
    ax.axhline(2.0, color=TRUTH_C, lw=1.6)
    ax.text(0.2, 2.06, 'R is exactly right here', fontsize=8.8, color=TRUTH_C, va='bottom')
    ax.set_ylim(0, 2.6)
    ax.set_xlabel('passes of the loop completed')
    ax.set_ylabel('squared innovation, in units of\nthe covariance R forecast for it')
    ax.set_title('THE TEST R CAN BE HELD TO -- and it needs no ground truth.\n'
                 'Learning R walks all three from "far too cautious" up to about right; '
                 'nothing here is broken', fontsize=10.4, loc='left')
    ax.legend(fontsize=8.6, loc='lower right')

    # ---- bottom right: the test the belief can be held to
    ax = fig.add_subplot(grid[1, 2])
    for key in ORDER:
        result = results[key]
        cover = [fb.honesty(result['cells'], h['R_in'])['coverage_95_pct']
                 for h in result['history']]
        ax.plot(range(len(cover)), cover, marker='o', ms=4, lw=2.2,
                color=PANEL_COLOUR[key])
        ax.annotate(f'{cover[-1]:.0f}%', (len(cover) - 1, cover[-1]),
                    textcoords='offset points', xytext=(6, -3), fontsize=8.6,
                    color=PANEL_COLOUR[key], fontweight='bold')
    ax.axhline(95, color=TRUTH_C, lw=1.6)
    ax.text(0.3, 90, 'honest = 95%', fontsize=8.8, color=TRUTH_C, va='top')
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.5, len(results[ORDER[0]]['history']) + 1.6)
    ax.set_xlabel('passes of the loop completed')
    ax.set_ylabel('truth inside the belief\'s own 95% region')
    ax.set_title('THE TEST THE POSITION CAN BE HELD TO.\nIt fails -- and fails harder the '
                 'righter R gets,\nbecause the error it is missing is a mean',
                 fontsize=10.4, loc='left')

    fig.suptitle('Learning R makes R right -- and that is exactly what exposes the mean: '
                 'a correct variance beside a wrong average',
                 fontsize=13.4, fontweight='bold', y=0.985)
    weak = {key: results[key]['prior_sensitivity']['prior 1 cm'] for key in ORDER}
    rd.note(fig, 'Top row: the final pass, same places, same scale. Bottom: both tests after '
                 'every pass. R is not the broken part -- pushed to its best value it passes '
                 'its own test almost exactly, and the position gets WORSE, which is the '
                 'proof that coverage is a verdict on the mean and not on R. Under a 1 cm '
                 'prior instead of the notebook\'s 5 cm, R lands on the within-place scatter '
                 'and its test reads '
                 + ', '.join(f'{weak[k]["mean_nis"]:.2f}' for k in ORDER)
                 + ' (right = 2) while coverage falls to '
                 + ', '.join(f'{weak[k]["coverage_95_pct"]:.0f}%' for k in ORDER)
                 + ' -- so the 5 cm prior was buying the marked point coverage it had not '
                   'earned. The planning quantity is therefore not R but |bias(x)| + '
                   'k*sigma(x)/sqrt(n): the second term is what R honestly gives you and '
                   'shrinks with looking longer, the first is what R cannot give you and only '
                   'a better reading removes. Evidence: '
                   'logs/studies/localization_reading_story/filter_on_both_readings/.',
            width=200)
    fig.subplots_adjust(top=0.855, bottom=0.165, left=0.065, right=0.985)
    rd.save(fig, 'B3_r_is_right_the_mean_is_not')


def learned_r_versus_error(results: dict) -> None:
    """B4 -- what learning R tells you, against what the error actually is.

    The loop reports one number and one number only: how much the readings of a place
    scatter. This figure puts that number beside the error the robot actually has, for each
    reading, so the gap is visible rather than argued. It is the same decomposition L1 does
    on the raw readings, except that here the first bar is LEARNED -- produced without
    ground truth, the way a deployed filter would produce it.
    """
    rd.style()
    fig = plt.figure(figsize=(15.2, 10.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.85], hspace=0.44, wspace=0.20)

    stats = {}
    for key in ORDER:
        result = results[key]
        cells = result['cells']
        err = 100 * np.concatenate([c['y'] - c['truth'] for c in cells])
        place = 100 * np.array([c['y'].mean(axis=0) - c['truth'] for c in cells])
        R = 1e4 * result['R_final']
        within = result['within_place_scatter_cm2']
        stats[key] = {
            'err': err,
            'R_sigma': math.sqrt(np.trace(R) / 2),
            'R_cov': R,
            'within': math.sqrt(np.trace(within) / 2),
            'place_bias': float(np.median(np.hypot(*place.T))),
            'miss': float(np.median(np.hypot(*err.T))),
            'mean': err.mean(axis=0),
        }

    span = 1.15 * float(np.abs(stats['box_bottom']['err']).max())
    for column, key in enumerate(ORDER):
        ax = fig.add_subplot(grid[0, column])
        st = stats[key]
        colour = PANEL_COLOUR[key]
        ax.scatter(st['err'][:, 0], st['err'][:, 1], s=11, lw=0, alpha=0.30, color=OBS_C,
                   label='every reading\'s error' if column == 0 else None)
        ellipse(ax, (0, 0), st['R_cov'], 2.0, '#f4a261', 0.26,
                'what the loop LEARNED R is,\nat 2 sd (no truth used)' if column == 0 else None,
                lw=2.2)
        ax.plot(0, 0, marker='+', ms=14, mew=2.2, color=TRUTH_C, zorder=6,
                label='where the reading should land' if column == 0 else None)
        ax.annotate('', xy=tuple(st['mean']), xytext=(0, 0),
                    arrowprops={'arrowstyle': '-|>', 'color': colour, 'lw': 2.4,
                                'mutation_scale': 14})
        ax.plot([st['mean'][0]], [st['mean'][1]], 'o', ms=9, color=colour, mec='white',
                mew=1.2, zorder=7,
                label='where they actually land' if column == 0 else None)
        ratio = st['miss'] / st['R_sigma']
        verdict = (f'{ratio:.1f}x too small' if ratio > 1.15
                   else (f'{1 / ratio:.1f}x too big -- conservative' if ratio < 0.87
                         else 'about right'))
        ax.set_title(f'{PANEL_NAME[key]}\n'
                     f"the loop learns R = $\\pm${st['R_sigma']:.2f} cm; the typical miss is "
                     f"{st['miss']:.2f} cm\nso the number it states is {verdict}",
                     fontsize=10.2, color=colour, loc='left')
        square(ax, span, 'error across the aisle, cm',
               'error along the aisle, cm' if column == 0 else None)
        if column == 0:
            ax.legend(fontsize=8.0, loc='upper left')

    # ---- the decomposition, all three side by side
    ax = fig.add_subplot(grid[1, :])
    labels = ['what the loop LEARNS R is\n(no ground truth used)',
              'the scatter it is fitting\n(readings of one place)',
              'the part it never sees\n(how far a place\'s average is out)',
              'what actually happens\n(typical miss from truth)']
    keys = ['R_sigma', 'within', 'place_bias', 'miss']
    width = 0.24
    xs = np.arange(len(labels))
    for offset, key in zip((-width, 0.0, width), ORDER):
        st = stats[key]
        values = [st[k] for k in keys]
        bars = ax.bar(xs + offset, values, width, color=PANEL_COLOUR[key],
                      label=PANEL_NAME[key].replace('\n', ' '),
                      alpha=0.95 if key != 'box_bottom_heading_aware' else 0.9)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.2f}', (bar.get_x() + bar.get_width() / 2, value),
                        textcoords='offset points', xytext=(0, 3), ha='center',
                        fontsize=8.4, color=PANEL_COLOUR[key], fontweight='bold')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylabel('centimetres')
    ax.set_ylim(0, 1.22 * max(stats[k]['miss'] for k in ORDER))
    ax.legend(fontsize=8.8, loc='upper left')
    ax.set_title('The first two bars track each other on all three readings: the loop learns '
                 'the scatter, and the 5 cm prior holds it a\nlittle high on the accurate '
                 'ones. The third bar is the one it can never reach -- and it is the one that '
                 'decides whether the belief is any good',
                 fontsize=10.4, loc='left')
    ax.grid(axis='x', visible=False)

    fig.suptitle('What learning R can tell you, and what it cannot: the same loop on the box '
                 'bottom and on the marked point',
                 fontsize=13.4, fontweight='bold', y=0.985)
    bb, kp = stats['box_bottom'], stats['keypoint_retrained']
    rd.note(fig, f'Same 423 held-out poses, same loop, only the reading swapped '
                 f'({rd.DATASET.name}, 2026-08-17). The first bar is produced without ground '
                 f'truth, exactly as a deployed filter would produce it; the other three are '
                 f'scored against truth afterwards. On the box bottom the loop states '
                 f'{bb["R_sigma"]:.2f} cm while the robot is typically {bb["miss"]:.2f} cm '
                 f'away -- {bb["miss"] / bb["R_sigma"]:.1f}x too small, and nothing in the data '
                 f'can tell it so. On the marked point it states {kp["R_sigma"]:.2f} cm against '
                 f'a {kp["miss"]:.2f} cm miss, i.e. slightly conservative, and that is what '
                 f'makes the belief usable. The first bar sits a little above the second on '
                 f'the accurate readings only because of the notebook\'s 5 cm prior; re-learnt '
                 f'under a 1 cm prior the two land on each other (2.75 / 1.07 / 0.73 cm against '
                 f'2.69 / 1.11 / 0.77 cm), which changes nothing about the third bar. Evidence: '
                 f'logs/studies/localization_reading_story/filter_on_both_readings/.',
            width=200)
    fig.subplots_adjust(top=0.86, bottom=0.155, left=0.065, right=0.985)
    rd.save(fig, 'B4_what_R_learns_versus_the_error')


RATE_HZ = 3.0            # the deployed usable-detection rate, as in achievable_precision_map
PASS_SHADES = {0: '#c9d6df', 1: '#93b0c4', 2: '#5f87a6', 10: '#1d3557'}


def error_over_time(results: dict) -> None:
    """B5 -- what learning R changes over time, and what it does not.

    The robot stands in one place and the camera keeps seeing it. Time is therefore
    sightings: at the deployed 3 Hz, one every third of a second. The measured line is how
    far the belief really is from truth; the shaded steps are how far it SAYS it might be,
    at four passes of the loop.

    The point of the figure is that the two do not move together. Learning R does not shift
    the belief at all -- with the readings of a place averaged, where the belief sits does not
    depend on R -- it only lowers what the belief claims. So every pass drags the claim down
    past an error that is standing still.
    """
    rd.style()
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.4), sharey=True)
    horizon_s = 8.0
    dense = np.linspace(1, RATE_HZ * horizon_s, 300)

    for ax, key in zip(axes, ORDER):
        result = results[key]
        cells, history = result['cells'], result['history']
        colour = PANEL_COLOUR[key]
        measured = fb.error_over_time(cells, history[-1]['R_in'])
        times = (np.array(measured['sightings']) - 1) / RATE_HZ
        bias = result['honesty']['median_error_cm']

        for p in (0, 1, 2, 10):
            R = history[p]['R_in']
            sigma = 100 * math.sqrt(np.trace(R) / 2)
            claim = 2 * sigma / np.sqrt(dense)          # 2 sd after n sightings, R/n
            label = ('what it CLAIMS, at 2 sd:  ' if p == 0 else '') + (
                'the prior' if p == 0 else f'pass {p}')
            ax.plot((dense - 1) / RATE_HZ, claim, lw=2.0, color=PASS_SHADES[p], label=label)

        # only draw the measured line where enough places have that many sightings
        solid = [i for i, n in enumerate(measured['places']) if n >= 10]
        last = solid[-1]
        ax.plot(times[solid], [measured['median_error_cm'][i] for i in solid], lw=3.0,
                color=colour, marker='o', ms=6, zorder=6,
                label='where the belief REALLY is (measured)')
        ax.plot([times[last], horizon_s], [bias, bias], lw=3.0, color=colour, ls=':', zorder=6)
        y_text = bias * 0.42 if bias > 3 else bias * 2.4
        ax.annotate(f'it stops falling here: {bias:.2f} cm,\nthe place\'s own average error',
                    xy=(horizon_s * 0.72, bias), xytext=(horizon_s * 0.34, y_text),
                    fontsize=8.4, color=colour, linespacing=1.35,
                    arrowprops={'arrowstyle': '->', 'color': colour, 'lw': 1.0})

        # where the claim at the last pass drops below the error that is actually there
        sigma_final = 100 * math.sqrt(np.trace(history[-1]['R_in']) / 2)
        n_cross = (2 * sigma_final / bias) ** 2
        t_cross = (n_cross - 1) / RATE_HZ
        if t_cross <= 0:
            note = 'it is claiming to be better than it is\nfrom the very first sighting'
        else:
            note = (f'after {t_cross:.1f} s of standing still it claims\n'
                    f'to be better than it is')
            ax.axvline(t_cross, color=rd.INK, lw=1.2, ls='--')
        ax.text(0.98, 0.965, note, transform=ax.transAxes, ha='right', va='top',
                fontsize=8.6, color=rd.INK, linespacing=1.35)

        ax.set_yscale('log')
        ax.set_xlim(-0.15, horizon_s)
        ax.set_ylim(0.15, 22)
        ax.set_xlabel('seconds of standing still\n(sightings arrive at the deployed 3 Hz)')
        ax.set_title(PANEL_NAME[key], fontsize=10.6, color=colour, loc='left')
        secondary = ax.secondary_xaxis('top', functions=(lambda t: t * RATE_HZ + 1,
                                                         lambda n: (n - 1) / RATE_HZ))
        secondary.set_xlabel('sightings absorbed', fontsize=8.6)
        secondary.tick_params(labelsize=8)
    axes[0].set_ylabel('centimetres  (log scale)')
    axes[0].legend(fontsize=8.0, loc='lower left')

    fig.suptitle('Learning R does not move the belief -- it only lowers what the belief claims. '
                 'So each pass drags the claim down past an error that is standing still',
                 fontsize=13.0, fontweight='bold', y=0.985)
    rd.note(fig, f'Each place is a standing robot seen 2-6 times; the measured line is the '
                 f'median over the places that have that many sightings (121/121/73/35/11/3 '
                 f'for the box bottom), and the line is drawn only while at least ten places '
                 f'remain; past that the dotted line continues at the measured limit. The claim curves are the learned R '
                 f'divided by the number of sightings, which is arithmetic, not an '
                 f'extrapolation of data. The belief itself is identical at every pass -- '
                 f'averaging the readings of a place does not depend on R -- which is why one '
                 f'measured line serves all four. Evidence: '
                 f'logs/studies/localization_reading_story/filter_on_both_readings/. The same '
                 f'picture on a real drive, for the box bottom only, is '
                 f'`smoothing_bands` in experiments/filter_notebook/notebook_views.py.',
            width=200)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.06, right=0.99, wspace=0.12)
    rd.save(fig, 'B5_error_and_claim_over_time')


def main() -> None:
    results = {key: fb.run(key) for key in ORDER}
    # One set of scales for both readings, taken from the bigger one.
    spans = {}
    for name, getter in (
            ('error', lambda r: np.abs(r['truth_errors_cm']).max()),
            ('inno', lambda r: 1.15 * max(
                math.sqrt(fb.GATE_CHI2_2DOF * np.max(np.linalg.eigvalsh(
                    1e4 * np.median(fb.sweep(r['cells'], h['R_in'])['S'], axis=0))))
                for h in [r['history'][p] for p in PICKS if p < len(r['history'])])),
            ('belief', lambda r: 1.25 * max(
                np.abs(r['honesty']['errors_cm']).max(),
                np.abs(r['honesty_prior']['errors_cm']).max()))):
        spans[name] = float(max(getter(results[k]) for k in results))
    for key in ('box_bottom', 'keypoint_retrained'):
        one_reading(key, results[key], spans)
    verdict(results, spans)
    learned_r_versus_error(results)
    error_over_time(results)

    for key, r in results.items():
        h, R = r['honesty'], r['R_final']
        print(f"{key:20s} learned R {math.sqrt(1e4 * R[0, 0]):.2f} x "
              f"{math.sqrt(1e4 * R[1, 1]):.2f} cm | belief {h['median_error_cm']:.2f} cm off, "
              f"states {h['median_stated_sigma_cm']:.2f} cm | NEES {h['mean_nees']:.1f} | "
              f"coverage {h['coverage_95_pct']:.1f}%")


if __name__ == '__main__':
    main()
