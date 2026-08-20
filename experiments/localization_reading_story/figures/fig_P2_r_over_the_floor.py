#!/usr/bin/env python3
"""P2 -- what R looks like over the floor, and which frame you have to look at it in.

Reported in world axes the marked point's spread is 0.73 x 0.80 cm with essentially no
correlation, which reads as "round, one number describes it". That is a frame artefact:
the camera's line of sight turns as the robot moves across the floor, so pooling in
east/north averages ellipses of different orientation into a circle. Resolved in the
camera's own frame -- along the line of sight against across it -- the spread is an
ellipse of 1.45 to 1, and the geometry predicts 1.32 to 1.

  A  the marked point block by block: an ellipse for the spread, drawn at the block's
     mean error, with the camera's line of sight through each block for comparison
  B  the box bottom on the same blocks and the same scale
  C  the test that does not depend on choosing axes from the data: along-sight variance
     against across-sight variance per block, against the band a genuinely round R
     would produce at that sample size (an F interval), plus what the pixel geometry
     predicts

Run: python3 experiments/localization_reading_story/figures/fig_P2_r_over_the_floor.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import keypoint_geometry as kg  # noqa: E402
import reading_data as rd  # noqa: E402

X_EDGES = np.array([-5.0, -1.7, 1.7, 5.0])
Y_EDGES = np.array([-4.7, -2.5, 0.5, 4.7])
MIN_N = 25                 # below this, a block's ellipse shape is not measurable
CM_PER_METRE = 0.25        # drawing scale: 1 cm of error drawn as 25 cm of floor
PREDICTED = '#457b9d'


def camera_frame(reading, cam):
    """Errors rotated so the first axis points away from the camera."""
    bearing = np.arctan2(reading.col('y') - cam[1], reading.col('x') - cam[0])
    err = reading.err_cm
    along = err[:, 0] * np.cos(bearing) + err[:, 1] * np.sin(bearing)
    across = -err[:, 0] * np.sin(bearing) + err[:, 1] * np.cos(bearing)
    return np.column_stack([along, across])


def blocks(reading, mask, cam):
    err, x, y = reading.err_cm, reading.col('x'), reading.col('y')
    cframe = camera_frame(reading, cam)
    out = []
    for i in range(len(Y_EDGES) - 1):
        for j in range(len(X_EDGES) - 1):
            sel = (mask & (x >= X_EDGES[j]) & (x < X_EDGES[j + 1])
                   & (y >= Y_EDGES[i]) & (y < Y_EDGES[i + 1]))
            if sel.sum() < 5:
                continue
            e = err[sel]
            mean = e.mean(axis=0)
            cov = np.cov(e - mean, rowvar=False, ddof=1)
            vals, vecs = np.linalg.eigh(cov)
            c = cframe[sel] - cframe[sel].mean(axis=0)
            out.append({
                'cx': 0.5 * (X_EDGES[j] + X_EDGES[j + 1]),
                'cy': 0.5 * (Y_EDGES[i] + Y_EDGES[i + 1]),
                'x0': X_EDGES[j], 'x1': X_EDGES[j + 1],
                'y0': Y_EDGES[i], 'y1': Y_EDGES[i + 1],
                'n': int(sel.sum()), 'mean': mean,
                'sigma_major': float(math.sqrt(vals[-1])),
                'sigma_minor': float(math.sqrt(vals[0])),
                'angle_deg': float(math.degrees(math.atan2(vecs[1, -1], vecs[0, -1]))),
                'sd_along': float(c[:, 0].std(ddof=1)),
                'sd_across': float(c[:, 1].std(ddof=1)),
            })
    return out


def draw_blocks(ax, stats_list, colour, title, cam):
    for j in X_EDGES[1:-1]:
        ax.axvline(j, color=rd.MUTED, lw=0.6, ls=':')
    for i in Y_EDGES[1:-1]:
        ax.axhline(i, color=rd.MUTED, lw=0.6, ls=':')
    for b in stats_list:
        weak = b['n'] < MIN_N
        centre = (b['cx'] + CM_PER_METRE * b['mean'][0],
                  b['cy'] + CM_PER_METRE * b['mean'][1])
        # the camera's line of sight through this block, for comparison with the ellipse
        bearing = math.atan2(b['cy'] - cam[1], b['cx'] - cam[0])
        ax.plot([b['cx'] - 1.15 * math.cos(bearing), b['cx'] + 1.15 * math.cos(bearing)],
                [b['cy'] - 1.15 * math.sin(bearing), b['cy'] + 1.15 * math.sin(bearing)],
                color='#1d3557', lw=0.9, ls='--', alpha=0.6, zorder=2)
        ax.add_patch(Ellipse(centre,
                             2 * CM_PER_METRE * b['sigma_major'],
                             2 * CM_PER_METRE * b['sigma_minor'],
                             angle=b['angle_deg'], facecolor=colour,
                             alpha=0.20 if weak else 0.36, edgecolor=colour,
                             lw=1.0 if weak else 1.8, ls=':' if weak else '-', zorder=4))
        ax.plot([b['cx']], [b['cy']], '+', color=rd.INK, ms=9, mew=1.4, zorder=5)
        ax.annotate('', xy=centre, xytext=(b['cx'], b['cy']),
                    arrowprops={'arrowstyle': '->', 'color': colour, 'lw': 1.3})
        label = (f"{b['sigma_major']:.2f} x {b['sigma_minor']:.2f} cm, n={b['n']}" if not weak
                 else f"n={b['n']}: too few for a shape")
        top_row = b['y1'] >= Y_EDGES[-1] - 1e-9      # keep clear of the legend
        ax.text(0.5 * (b['x0'] + b['x1']),
                b['y0'] + 0.15 if top_row else b['y1'] - 0.12, label, fontsize=7.4,
                ha='center', va='bottom' if top_row else 'top',
                color=rd.INK if not weak else rd.MUTED)
    ax.plot([], [], '+', color=rd.INK, ms=9, mew=1.4, label='where the robot really is')
    ax.plot([], [], color=colour, lw=1.8, label='spread (1 sigma), drawn at the mean error')
    ax.plot([], [], color='#1d3557', lw=0.9, ls='--', label='the camera\'s line of sight')
    ax.set_aspect('equal')
    ax.set_xlim(X_EDGES[0] - 0.4, X_EDGES[-1] + 0.4)
    ax.set_ylim(-6.1, Y_EDGES[-1] + 0.5)
    ax.plot([cam[0]], [cam[1]], marker='^', color='#1d3557', ms=12, mec='white', mew=1.1,
            clip_on=False, zorder=6)
    ax.text(cam[0] + 0.35, cam[1], 'the camera', fontsize=8.5, color='#1d3557', va='center')
    ax.plot([X_EDGES[0] + 0.3, X_EDGES[0] + 0.3 + 5 * CM_PER_METRE], [-5.75, -5.75],
            color=rd.INK, lw=2.4)
    ax.text(X_EDGES[0] + 0.3 + 2.5 * CM_PER_METRE, -5.62, '5 cm of error', fontsize=8,
            ha='center', color=rd.INK)
    ax.set_xlabel('metres east')
    ax.set_title(title, loc='left', fontsize=10.5)
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=7.8, framealpha=0.88, frameon=True)


def main() -> None:
    rd.style()
    cam = rd.camera().cam_pos
    kp = rd.load_reading('keypoint_retrained')
    bb = rd.load_reading('box_bottom')
    kp_mask = kg.both_rendered_mask(kp)
    kp_blocks = blocks(kp, kp_mask, cam)
    bb_blocks = blocks(bb, np.ones(int(bb.detected.sum()), bool), cam)

    fig, axes = plt.subplots(1, 3, figsize=(16.6, 6.6),
                             gridspec_kw={'width_ratios': [1.0, 1.0, 1.05]})
    elongated = sum(1 for b in kp_blocks if b['sd_along'] > b['sd_across'])
    draw_blocks(axes[0], kp_blocks, rd.KEYPOINT,
                f'A.  Marked point: 0.5-0.9 cm wide, and in {elongated} of '
                f'{len(kp_blocks)} blocks\nthe long axis follows the camera\'s line of sight',
                cam)
    draw_blocks(axes[1], bb_blocks, rd.BOX_BOTTOM,
                'B.  Bottom of the box, same scale: four times wider, and\n'
                'every block pushed towards the camera', cam)
    axes[0].set_ylabel('metres north')

    # ------------------------------------------------------------------ C: the axis-free test
    ax = axes[2]
    pooled = camera_frame(kp, cam)[kp_mask]
    pooled = pooled - pooled.mean(axis=0)
    n_pool = len(pooled)
    ratio_pool = pooled[:, 0].std(ddof=1) / pooled[:, 1].std(ddof=1)
    p_pool = 1 - stats.f.cdf(ratio_pool ** 2, n_pool - 1, n_pool - 1)

    pixel_cov = kg.pixel_covariance(kp, kp_mask)
    pred = kg.predicted_floor_covariance(kp, pixel_cov)[kp_mask]
    bearing = np.arctan2(kp.col('y')[kp_mask] - cam[1], kp.col('x')[kp_mask] - cam[0])
    pred_ratio = []
    for cov, b in zip(pred, bearing):
        rot = np.array([[math.cos(b), math.sin(b)], [-math.sin(b), math.cos(b)]])
        c = rot @ cov @ rot.T
        pred_ratio.append(math.sqrt(c[0, 0] / c[1, 1]))
    pred_ratio = np.array(pred_ratio)

    order = sorted(kp_blocks, key=lambda b: b['n'])
    ys = np.arange(len(order) + 1)
    ax.axvspan(np.percentile(pred_ratio, 10), np.percentile(pred_ratio, 90),
               color=PREDICTED, alpha=0.13, zorder=0,
               label=f'what the pixel geometry predicts\n(median {np.median(pred_ratio):.2f}, '
                     f'10-90% band)')
    ax.axvline(np.median(pred_ratio), color=PREDICTED, lw=1.6, zorder=1)
    for y, b in zip(ys, order):
        lo = math.sqrt(stats.f.ppf(0.05, b['n'] - 1, b['n'] - 1))
        hi = math.sqrt(stats.f.ppf(0.95, b['n'] - 1, b['n'] - 1))
        ax.plot([lo, hi], [y, y], color=rd.MUTED, lw=7, alpha=0.30, solid_capstyle='butt')
        ax.plot([b['sd_along'] / b['sd_across']], [y], 'o', color=rd.KEYPOINT, ms=9,
                mec='white', mew=1.0, zorder=5)
    y = ys[-1]
    lo = math.sqrt(stats.f.ppf(0.05, n_pool - 1, n_pool - 1))
    hi = math.sqrt(stats.f.ppf(0.95, n_pool - 1, n_pool - 1))
    ax.plot([lo, hi], [y, y], color=rd.MUTED, lw=7, alpha=0.30, solid_capstyle='butt')
    ax.plot([ratio_pool], [y], 'D', color=rd.INK, ms=10, mec='white', mew=1.0, zorder=6)
    ax.axvline(1.0, color=rd.INK, lw=1.2)
    ax.text(1.02, -0.62, 'round', fontsize=8.5, color=rd.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"block of {b['n']} readings" for b in order]
                       + [f'ALL {n_pool} readings'], fontsize=8.5)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_xlabel('spread along the line of sight / spread across it')
    ax.plot([], [], color=rd.MUTED, lw=7, alpha=0.30,
            label='what a genuinely round R would give\nat that many readings (5-95%)')
    ax.plot([], [], 'o', color=rd.KEYPOINT, ms=9, label='measured, one block')
    ax.plot([], [], 'D', color=rd.INK, ms=9, label='measured, all readings pooled')
    ax.legend(loc='center right', fontsize=7.9)
    ax.set_xlim(0.55, 2.62)
    ax.set_title(f'C.  Pooled, the spread is {ratio_pool:.2f} times longer along the sight\n'
                 f'line than across it (p = {p_pool:.0e}), as geometry predicts',
                 loc='left', fontsize=10.5)
    ax.grid(axis='y', visible=False)

    fig.suptitle('R on this reading is small and much the same size everywhere -- but it is '
                 'not a circle: in the camera\'s frame it is 1.45 to 1 along the line of sight',
                 fontsize=13, fontweight='bold', y=1.0)
    rd.note(fig, f'{int(kp_mask.sum())} held-out marked-point readings with both disks '
                 f'rendered, and {int(bb.detected.sum())} box-bottom readings on the same images '
                 f'({rd.DATASET.name}, 2026-08-17). In WORLD axes the same marked-point '
                 f'residuals read 0.73 x 0.80 cm with correlation -0.03, i.e. round -- that is '
                 f'the frame artefact this figure exists to correct: the line of sight turns '
                 f'across the floor, so pooling in east/north averages ellipses of different '
                 f'orientation into a circle. The box bottom goes the other way, 0.62 along '
                 f'over across, and its spread is dominated by the heading-dependent lean '
                 f'rather than by pixel noise. Panel C uses fixed axes (the sight line), so no '
                 f'shape is chosen from the data. Evidence: '
                 f'logs/studies/keypoint_measurement/, recomputed by '
                 f'experiments/localization_reading_story/.', width=205)
    fig.subplots_adjust(top=0.85, bottom=0.19, left=0.05, right=0.99, wspace=0.24)
    rd.save(fig, 'P2_r_over_the_floor')

    print(f'pooled along/across {ratio_pool:.3f} (p={p_pool:.3g}), predicted median '
          f'{np.median(pred_ratio):.3f}')
    for b in kp_blocks:
        print(f"block ({b['cx']:+.1f},{b['cy']:+.1f}) n={b['n']:3d} "
              f"along {b['sd_along']:.2f} across {b['sd_across']:.2f} "
              f"ratio {b['sd_along'] / b['sd_across']:.2f}")


if __name__ == '__main__':
    main()
