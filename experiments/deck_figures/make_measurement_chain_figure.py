#!/usr/bin/env python3
"""Problem-statement figure: how a camera image becomes a position measurement.

Three panels following one real detection from the frozen capture:

  (a) the camera image with the detected box and its bottom-centre pixel,
  (b) that pixel carried to the floor by the ground-plane homography,
  (c) the filter update the resulting world position drives.

Panels (a) and (b) use a real captured frame, the real detected box and the real
back-projected ground point, all from the characterization dataset. Panel (c) is drawn
schematically: it shows what a Kalman update does to a belief, and the covariances are
chosen for legibility rather than taken from a run, because the point is the mechanism.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'shared'))
from paths import repo_root  # noqa: E402

REPO = repo_root()
CAPTURE = REPO / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'
FRAME = 'camera_B/images/pose_001211_r00.png'

DETECT, MEAS, BELIEF, INK = '#c23d36', '#c23d36', '#1f6fb8', '#1d2530'


def panel_image(ax, row) -> None:
    """(a) the frame the camera delivered, with the detector's box on it."""
    image = mpimg.imread(CAPTURE / row.image)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (float(row[k]) for k in ('x0', 'y0', 'x1', 'y1'))

    ax.imshow(image)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           ec=DETECT, lw=2.4, zorder=3))
    ax.plot((x0 + x1) / 2, y1, 'o', color=DETECT, ms=8, mec='white', mew=1.4, zorder=4)
    ax.annotate('bottom-centre pixel $u_k$', xy=((x0 + x1) / 2, y1),
                xytext=((x0 + x1) / 2 - 330, y1 + 105), color=DETECT, fontsize=9.5,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.8, color=DETECT))
    ax.text(x0, y0 - 14, f'detector: {row.confidence:.2f}', color='white', fontsize=9,
            fontweight='bold', va='bottom',
            bbox=dict(boxstyle='square,pad=0.25', fc=DETECT, ec='none'))

    # crop to the robot's neighbourhood so the box is readable at column width
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = 430
    half_h = half_w * 0.72
    ax.set(xlim=(max(0, cx - half_w), min(width, cx + half_w)),
           ylim=(min(height, cy + half_h), max(0, cy - half_h)))
    ax.axis('off')
    ax.set_title('(a) what the camera delivers\na frame and one detected box',
                 fontsize=10, fontweight='bold')


def panel_homography(ax, row) -> None:
    """(b) the same pixel carried to the floor."""
    ax.axis('off')
    ax.set(xlim=(0, 1), ylim=(0, 1))

    # image plane, drawn as a tilted quad with the detection on it
    plane = np.array([[.08, .62], [.52, .74], [.52, .40], [.08, .28]])
    ax.add_patch(plt.Polygon(plane, closed=True, fc='#dce6f2', ec=BELIEF, lw=1.4, alpha=.85))
    ax.plot(.30, .50, 'o', color=DETECT, ms=8, mec='white', mew=1.3, zorder=4)
    ax.text(.30, .565, '$u_k$', color=DETECT, fontsize=11, fontweight='bold', ha='center')
    ax.text(.13, .70, 'image plane', fontsize=9, color=INK)

    # the floor
    floor = np.array([[.30, .22], [.94, .30], [.94, .06], [.30, .02]])
    ax.add_patch(plt.Polygon(floor, closed=True, fc='#eef0f2', ec='#9aa3ac', lw=1.2))
    ax.text(.62, -.02, 'floor, $z=0$', fontsize=9, color=INK, ha='center')

    ax.add_patch(FancyArrowPatch((.33, .47), (.72, .19), color=INK, lw=2.0,
                                 arrowstyle='-|>', mutation_scale=17,
                                 connectionstyle='arc3,rad=-0.28', zorder=5))
    ax.text(.60, .40, 'ground-plane\nhomography $H_c^{-1}$', fontsize=9.5, ha='center',
            color=INK, fontweight='bold')

    ax.plot(.74, .17, '*', color=MEAS, ms=17, mec='white', mew=1.0, zorder=6)
    ax.text(.79, .12, '$z_k=(x_k,y_k)$', color=MEAS, fontsize=10.5, fontweight='bold')
    ax.set_title('(b) pixel to a position on the floor\nthe measurement the filter receives',
                 fontsize=10, fontweight='bold')


def panel_update(ax) -> None:
    """(c) what the filter does with that position. Schematic: the mechanism, not a run."""
    prior, meas, post = np.array([0.0, 0.0]), np.array([1.55, 0.42]), np.array([0.82, 0.22])
    prior_cov = np.array([[.62, .12], [.12, .20]])
    meas_cov = np.array([[.20, -.05], [-.05, .13]])
    post_cov = np.array([[.15, .03], [.03, .08]])

    for centre, cov, colour, style, label in (
            (prior, prior_cov, BELIEF, '--', 'predicted belief $m_k^-,S_k^-$'),
            (meas, meas_cov, MEAS, '-', 'measurement $z_k$ with $R_c$'),
            (post, post_cov, BELIEF, '-', 'updated belief $m_k^+,S_k^+$')):
        values, vectors = np.linalg.eigh(cov)
        ax.add_patch(Ellipse(centre, 4 * np.sqrt(values[-1]), 4 * np.sqrt(values[0]),
                             angle=np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])),
                             fc=colour, alpha=.18, ec=colour, lw=1.6, ls=style, label=label))

    ax.add_patch(FancyArrowPatch(prior, meas, color=MEAS, lw=2.0, arrowstyle='-|>',
                                 mutation_scale=16, zorder=5))
    ax.text(.72, .46, 'innovation\n$z_k-h(m_k^-)$', color=MEAS, fontsize=9.5,
            fontweight='bold', ha='center')
    ax.plot(*prior, 'o', color=BELIEF, ms=7, zorder=6)
    ax.plot(*meas, '*', color=MEAS, ms=16, mec='white', mew=.9, zorder=6)
    ax.plot(*post, 'o', color=BELIEF, ms=8, mec='white', mew=1.2, zorder=6)

    ax.set(xlim=(-1.9, 3.3), ylim=(-1.9, 1.7), aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(c) the filter update\nthe belief moves toward the reading',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.2, loc='lower left', framealpha=.94, ncol=1)


def main() -> None:
    rows = pd.read_csv(CAPTURE / 'observation_interpretations.csv')
    row = rows[(rows.image == FRAME) & (rows.detected == 1)].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    panel_image(axes[0], row)
    panel_homography(axes[1], row)
    panel_update(axes[2])

    for a, b in ((axes[0], axes[1]), (axes[1], axes[2])):
        fig.patches.append(FancyArrowPatch(
            (a.get_position().x1 + .006, .52), (b.get_position().x0 - .006, .52),
            transform=fig.transFigure, color='#8b939c', lw=2.4,
            arrowstyle='-|>', mutation_scale=18))

    for ax in axes:
        ax.set_title(ax.get_title(), fontsize=10, fontweight='bold', y=1.02)
    fig.tight_layout(pad=0.5, w_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'measurement_chain.pdf', OUT / 'measurement_chain.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    print('wrote', OUT / 'measurement_chain.pdf')
    print(f'frame {FRAME}, range {row.camera_range_m:.2f} m, confidence {row.confidence:.3f}')


if __name__ == '__main__':
    main()
