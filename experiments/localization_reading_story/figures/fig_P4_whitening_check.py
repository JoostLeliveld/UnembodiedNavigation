#!/usr/bin/env python3
"""P4 -- does pixel scatter alone now explain the marked point's floor scatter?

Under the box-bottom reading this check failed badly: the covariance that geometry
predicted was about 6x too small, because the target it was compared against was
dominated by a bias no noise model can represent. Redone on the marked-point reading:

  A  the floor residual cloud in the CAMERA's frame -- along the line of sight against
     across it -- with the covariance measured from it and the covariance the pixel
     scatter predicts drawn on top of each other. The camera frame is the frame this
     reading is actually anisotropic in; see P2
  B  the same residuals divided by the prediction. Round and unit-sized = the prediction
     is right; mean |z|^2 / 2 is the number to read
  C  that same ratio per range band -- flat at 1 is the pass condition, because a model
     can be right on average and wrong everywhere in particular

Run: python3 experiments/localization_reading_story/figures/fig_P4_whitening_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import keypoint_geometry as kg  # noqa: E402
import reading_data as rd  # noqa: E402

BANDS = ((4.5, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 12.0))
PREDICTED = '#457b9d'


def ellipse(ax, cov, colour, label, ls='-'):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    e = Ellipse((0, 0), 2 * np.sqrt(vals[-1]), 2 * np.sqrt(vals[0]), angle=angle,
                fill=False, edgecolor=colour, lw=2.2, ls=ls, label=label, zorder=5)
    ax.add_patch(e)
    return np.sqrt(vals[-1]), np.sqrt(vals[0])


def main() -> None:
    rd.style()
    reading = rd.load_reading('keypoint_retrained')
    camera = rd.camera()
    mask = kg.both_rendered_mask(reading)
    err = reading.err_cm[mask]
    pixel_cov = kg.pixel_covariance(reading, mask)
    predicted = kg.predicted_floor_covariance(reading, pixel_cov)[mask]
    rng = reading.col('range_m')[mask]

    # Into the camera's frame: first axis along its line of sight, second across it.
    bearing = np.arctan2(reading.col('y')[mask] - camera.cam_pos[1],
                         reading.col('x')[mask] - camera.cam_pos[0])
    rot = np.array([[[math.cos(b), math.sin(b)], [-math.sin(b), math.cos(b)]]
                    for b in bearing])
    err = np.einsum('nij,nj->ni', rot, err)
    predicted = np.einsum('nij,njk,nlk->nil', rot, predicted, rot)
    err_centred = err - err.mean(axis=0)

    measured_cov = np.cov(err_centred, rowvar=False, ddof=1)
    z = kg.whiten(err_centred, predicted)
    ratio = (z ** 2).sum(axis=1) / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.3))

    # ------------------------------------------------------------------ A: the two clouds
    ax = axes[0]
    ax.scatter(err_centred[:, 0], err_centred[:, 1], s=13, color=rd.KEYPOINT, alpha=0.45,
               edgecolors='none', label=f'{mask.sum()} readings (mean removed)')
    ma, mi = ellipse(ax, measured_cov, rd.INK, None)
    med = np.median(predicted, axis=0)
    pa, pi = ellipse(ax, med, PREDICTED, None, ls='--')
    ax.plot([], [], color=rd.INK, lw=2.2,
            label=f'measured spread: {ma:.2f} x {mi:.2f} cm')
    ax.plot([], [], color=PREDICTED, lw=2.2, ls='--',
            label=f'predicted from pixel scatter: {pa:.2f} x {pi:.2f} cm')
    ax.annotate('the two ellipses very nearly coincide -\nthat is the whole finding',
                xy=(0.62 * ma, -0.62 * mi), xytext=(0.35, -3.45), fontsize=8.2,
                color=rd.INK, linespacing=1.35, ha='center',
                arrowprops={'arrowstyle': '->', 'color': rd.INK, 'lw': 0.9})
    ax.set_aspect('equal')
    lim = 4.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('residual along the camera\'s line of sight (cm)')
    ax.set_ylabel('residual across the line of sight (cm)')
    ax.legend(loc='upper left', fontsize=8.2)
    ax.set_title('A.  Pixel scatter predicts the right size AND the right shape:\n'
                 'both are stretched along the line of sight, by about 1.4 to 1',
                 loc='left', fontsize=10.5)

    # ------------------------------------------------------------------ B: whitened
    ax = axes[1]
    ax.scatter(z[:, 0], z[:, 1], s=13, color=PREDICTED, alpha=0.5, edgecolors='none')
    for k, style in ((1.0, '-'), (2.0, ':')):
        ax.add_patch(Ellipse((0, 0), 2 * k, 2 * k, fill=False, edgecolor=rd.INK, lw=1.6,
                             ls=style, zorder=5,
                             label=f'{k:.0f}x the predicted uncertainty'))
    ax.set_aspect('equal')
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel('along-sight residual, in units of predicted uncertainty')
    ax.set_ylabel('across-sight residual, in units of predicted uncertainty')
    ax.legend(loc='upper left', fontsize=8.2)
    inside = float(np.mean(np.hypot(z[:, 0], z[:, 1]) <= 2.0))
    ax.set_title(f'B.  Divided by the prediction: mean squared size '
                 f'{ratio.mean():.2f}\n(1.00 would be exact), {100 * inside:.0f}% inside twice it',
                 loc='left', fontsize=10.5)

    # ------------------------------------------------------------------ C: per range band
    ax = axes[2]
    mids, vals, sems, ns = [], [], [], []
    for lo, hi in BANDS:
        sel = (rng >= lo) & (rng < hi)
        if sel.sum() < 5:
            continue
        mids.append(0.5 * (lo + hi))
        vals.append(ratio[sel].mean())
        sems.append(ratio[sel].std(ddof=1) / np.sqrt(sel.sum()))
        ns.append(int(sel.sum()))
    ax.errorbar(mids, vals, yerr=sems, marker='o', color=PREDICTED, lw=2.2, capsize=3, ms=7)
    for m, v, n in zip(mids, vals, ns):
        ax.annotate(f'n={n}', (m, v), textcoords='offset points', xytext=(0, -16),
                    ha='center', fontsize=7.5, color=rd.MUTED)
    ax.axhline(1.0, color=rd.INK, lw=1.4)
    ax.text(mids[-1], 1.03, 'exactly right', fontsize=8.5, color=rd.INK, ha='right',
            va='bottom')
    ax.axhspan(0.0, 1.0, color=rd.KEYPOINT, alpha=0.07, zorder=0)
    ax.text(mids[0] - 0.2, 0.12, 'below the line = the reading is better\n'
                                 'than the geometry predicts (safe direction)',
            fontsize=8, color=rd.MUTED, va='bottom')
    ax.set_ylim(0, max(2.0, max(vals) * 1.35))
    ax.set_xlabel('distance from the camera (m)')
    ax.set_ylabel('measured scatter / predicted scatter')
    ax.set_title(f'C.  And it holds at every range: the worst band underpredicts\n'
                 f'by {100 * (max(vals) - 1):.0f}%, the rest are on it or safely below',
                 loc='left', fontsize=10.5)

    fig.suptitle('The old "geometry predicts six times too little uncertainty" gap is gone:\n'
                 'on the marked-point reading, pixel scatter explains the floor scatter, in '
                 'size and in shape, to within a few percent',
                 fontsize=13, fontweight='bold', y=1.03)
    total_with_bias = np.trace(measured_cov) + float(err.mean(axis=0) @ err.mean(axis=0))
    rd.note(fig, f'{mask.sum()} held-out readings with both marker disks rendered '
                 f'({rd.DATASET.name}, 2026-08-17). The pixel covariance is measured from these '
                 f'same readings and pushed through the reading\'s own Jacobian; nothing is '
                 f'fitted to the floor errors. Evidence: '
                 f'logs/studies/keypoint_measurement/v4_retrained/per_sample.csv, recomputed by '
                 f'experiments/localization_reading_story/keypoint_geometry.py.\n'
                 f'The answer no longer depends on whether the leftover bias is counted in: '
                 f'with it the measured total goes from {np.trace(measured_cov):.2f} to '
                 f'{total_with_bias:.2f} cm2 against '
                 f'{np.trace(predicted, axis1=1, axis2=2).mean():.2f} cm2 predicted. That '
                 f'ambiguity is what made the old six-times claim unreadable.')
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.055, right=0.99, wspace=0.30)
    rd.save(fig, 'P4_does_geometry_explain_the_scatter')

    print(f'camera-frame sd along {math.sqrt(measured_cov[0, 0]):.2f} across '
          f'{math.sqrt(measured_cov[1, 1]):.2f} cm, ratio '
          f'{math.sqrt(measured_cov[0, 0] / measured_cov[1, 1]):.2f}; predicted ratio '
          f'{math.sqrt(med[0, 0] / med[1, 1]):.2f}')
    print(f'measured  {np.sqrt(np.linalg.eigvalsh(measured_cov))} cm '
          f'(trace {np.trace(measured_cov):.3f} cm^2)')
    print(f'predicted median sigma {np.sqrt(np.linalg.eigvalsh(med))} cm, '
          f'mean trace {np.trace(predicted, axis1=1, axis2=2).mean():.3f} cm^2')
    print(f'mean squared whitened size {ratio.mean():.3f}; per band '
          f'{[round(v, 2) for v in vals]}')


if __name__ == '__main__':
    main()
