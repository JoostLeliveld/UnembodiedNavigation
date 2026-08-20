#!/usr/bin/env python3
"""P1 -- the part that averages away against the part that does not.

For a planner the two halves of the error are not interchangeable: repeated looks shrink
the random half like 1/sqrt(n) and do nothing at all to the systematic half. So the
question is not "how big is the error" but "after a few looks, which half is left".

  A  systematic part and random part per range band, with the random part also drawn
     after 4 and after 16 looks. Where the falling curves cross the flat one is the
     point past which looking longer buys nothing.
  B  why the far bands look worse than they are: at 8 m and beyond, a fifth to a third
     of the readings have one marker hidden with the model guessing at it, and those
     carry a -2 to -3.5 cm pull. Among readings where both disks actually rendered the
     residual bias stays under 1 cm at every range.

Run: python3 experiments/localization_reading_story/figures/fig_P1_bias_versus_noise.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import keypoint_geometry as kg  # noqa: E402
import reading_data as rd  # noqa: E402

BANDS = ((4.5, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 12.0))
BIAS_C, NOISE_C, HIDDEN_C = '#9b2226', '#2a9d8f', '#e76f51'


def band_stats(err, rng, sel):
    out = []
    for lo, hi in BANDS:
        pick = sel & (rng >= lo) & (rng < hi)
        if pick.sum() < 5:
            out.append(None)
            continue
        e = err[pick]
        mean = e.mean(axis=0)
        cov = np.cov(e - mean, rowvar=False, ddof=1)
        out.append({
            'mid': 0.5 * (lo + hi), 'lo': lo, 'hi': hi, 'n': int(pick.sum()),
            'bias': float(np.hypot(*mean)), 'north': float(mean[1]),
            'sigma': float(math.sqrt(np.trace(cov) / 2)),
        })
    return out


def main() -> None:
    rd.style()
    reading = rd.load_reading('keypoint_retrained')
    err, rng = reading.err_cm, reading.col('range_m')
    both = kg.both_rendered_mask(reading)
    clean = [s for s in band_stats(err, rng, both) if s]
    pooled = [s for s in band_stats(err, rng, np.ones(len(err), bool)) if s]
    hidden = [s for s in band_stats(err, rng, ~both) if s]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13.8, 5.5))

    # ------------------------------------------------------------------ A
    mids = [s['mid'] for s in clean]
    bias = [s['bias'] for s in clean]
    sigma = [s['sigma'] for s in clean]
    ax.plot(mids, bias, marker='o', color=BIAS_C, lw=2.6, ms=7,
            label='systematic part: the same error every time')
    for n_looks, style, alpha in ((1, '-', 1.0), (4, '--', 0.8), (16, ':', 0.65)):
        ax.plot(mids, [s / math.sqrt(n_looks) for s in sigma], marker='s', ms=5,
                color=NOISE_C, lw=2.2, ls=style, alpha=alpha,
                label=f'random part after {n_looks} look' + ('s' if n_looks > 1 else ''))
    for s in clean:
        n_needed = max(1, math.ceil((s['sigma'] / s['bias']) ** 2)) if s['bias'] > 0 else 0
        ax.annotate(f'{n_needed}', (s['mid'], s['bias']), textcoords='offset points',
                    xytext=(0, -20), ha='center', fontsize=8.5, color=BIAS_C,
                    fontweight='bold')
    ax.text(0.5, 0.015,
            'red numbers: looks needed before the systematic part is the bigger one',
            transform=ax.transAxes, fontsize=8.4, color=BIAS_C, ha='center', va='bottom')
    ax.set_xlabel('distance from the camera (m)')
    ax.set_ylabel('error contribution (cm)')
    ax.set_ylim(0, max(max(bias), max(sigma)) * 1.45)
    ax.legend(loc='upper left', fontsize=8.6)
    ax.set_title('A.  Averaging removes the random half in a handful of looks;\n'
                 'past 7 m only one to three of them are worth taking',
                 loc='left', fontsize=10.5)

    # ------------------------------------------------------------------ B
    width = 0.36
    xs = np.arange(len(clean))
    axb.bar(xs - width / 2, [s['north'] for s in clean], width, color=NOISE_C,
            label='both marker disks actually rendered')
    hid_by_mid = {s['mid']: s for s in hidden}
    hid_vals = [hid_by_mid[s['mid']]['north'] if s['mid'] in hid_by_mid else np.nan
                for s in clean]
    axb.bar(xs + width / 2, hid_vals, width, color=HIDDEN_C,
            label='one disk hidden, the model guessed at it')
    axb.plot(xs, [s['north'] for s in pooled], marker='D', ms=6, color=rd.INK, lw=1.6,
             ls='--', label='what the pooled number says (the mixture)')
    for x, s in zip(xs, clean):
        axb.annotate(f"n={s['n']}", (x - width / 2, s['north']), textcoords='offset points',
                     xytext=(0, 6), ha='center', fontsize=7.5, color=rd.MUTED)
        if s['mid'] in hid_by_mid:
            h = hid_by_mid[s['mid']]
            axb.annotate(f"n={h['n']}", (x + width / 2, h['north']),
                         textcoords='offset points', xytext=(0, -12), ha='center',
                         fontsize=7.5, color=rd.MUTED)
    axb.axhline(0, color=rd.INK, lw=0.9)
    axb.set_xticks(xs)
    axb.set_xticklabels([f"{s['lo']:g}-{s['hi']:g} m" for s in clean], fontsize=9)
    axb.set_xlabel('distance from the camera')
    axb.set_ylabel('north-south error (cm)\nnegative = read too close to the camera')
    axb.set_ylim(min(hid_vals) * 1.18, max(0.6, max(hid_vals) * 1.4))
    axb.legend(loc='lower left', fontsize=8.4)
    axb.set_title('B.  The far-range pull is mostly a hidden marker, not geometry:\n'
                  'clean two-disk readings stay inside 1 cm everywhere',
                  loc='left', fontsize=10.5)

    fig.suptitle('What the planner has to live with is the systematic centimetre, not the '
                 'random one -- and most of the range trend is a visibility artefact',
                 fontsize=13, fontweight='bold', y=1.0)
    rd.note(fig, f'{int(both.sum())} held-out readings with both disks rendered and '
                 f'{int((~both).sum())} with one hidden, of 423 poses '
                 f'({rd.DATASET.name}, 2026-08-17). The random part is sqrt(trace(cov)/2), '
                 f'i.e. the one number that describes a round covariance; the systematic part '
                 f'is the length of the band mean. Both recomputed from '
                 f'logs/studies/keypoint_measurement/v4_retrained/per_sample.csv by '
                 f'experiments/localization_reading_story/. The 1/sqrt(n) curves assume '
                 f'independent looks, which the same-place-same-heading case does not satisfy '
                 f'-- they are an upper bound on what averaging can do, so the small red '
                 f'numbers are optimistic.', width=195)
    fig.subplots_adjust(top=0.84, bottom=0.20, left=0.065, right=0.985, wspace=0.26)
    rd.save(fig, 'P1_bias_versus_noise')

    for s in clean:
        print(f"{s['lo']:.1f}-{s['hi']:.1f} m  n={s['n']:3d}  bias {s['bias']:.2f} cm  "
              f"sigma {s['sigma']:.2f} cm  north {s['north']:+.2f} cm")
    for s in hidden:
        print(f"  hidden marker {s['lo']:.1f}-{s['hi']:.1f} m  n={s['n']:3d}  "
              f"north {s['north']:+.2f} cm")


if __name__ == '__main__':
    main()
