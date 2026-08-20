#!/usr/bin/env python3
"""L4 -- where on the floor each reading is wrong.

Three maps of the same 423 held-out poses, aggregated to the capture's 20 x 14 grid
(each cell holds 1-6 poses, one per sampled heading; blank cells are places where no
held-out pose was accepted, mostly because the racks hide the robot there):

  A  bottom of the box, on a 0-10 cm scale
  B  marked point, retrained, on the SAME 0-10 cm scale -- the honest comparison
  C  the marked point's remaining SYSTEMATIC part -- the mean north-south error per
     cell, on a +-1.5 cm scale -- because on the shared scale it is flat everywhere.
     Panel C uses only readings where both marker disks actually rendered: the pooled
     "-1.10 cm at 9-12 m" range trend is mostly readings where one disk was hidden and
     the model guessed at it (-3.46 cm there, n=16). Among clean readings the pull is
     at most half a centimetre anywhere. See P1 panel B for that split.

Run: python3 experiments/localization_reading_story/figures/fig_L4_error_maps.py
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

SHARED_MAX = 10.0     # cm
SIGNED_MAX = 1.5      # cm, for the north-south map in panel C


def grid_of(reading, xs, ys, what: str = 'miss', mask=None) -> np.ndarray:
    """Cell means, NaN where the split has no reading.

    what='miss'  distance from truth (cm)
    what='ns'    signed north-south error (cm), i.e. the systematic part
    """
    out = np.full((len(ys), len(xs)), np.nan)
    counts = np.zeros_like(out)
    miss = reading.miss_cm if what == 'miss' else reading.err_cm[:, 1]
    x, y = reading.col('x'), reading.col('y')
    if mask is not None:
        miss, x, y = miss[mask], x[mask], y[mask]
    for xi, yi, m in zip(x, y, miss):
        j = int(np.argmin(np.abs(xs - xi)))
        i = int(np.argmin(np.abs(ys - yi)))
        out[i, j] = m if np.isnan(out[i, j]) else out[i, j] + m
        counts[i, j] += 1
    return np.where(counts > 0, out / np.maximum(counts, 1), np.nan)


def draw(ax, xs, ys, values, vmax, title, cam, cmap='magma_r', vmin=0.0):
    dx, dy = np.diff(xs).mean(), np.diff(ys).mean()
    edges_x = np.append(xs - dx / 2, xs[-1] + dx / 2)
    edges_y = np.append(ys - dy / 2, ys[-1] + dy / 2)
    mesh = ax.pcolormesh(edges_x, edges_y, np.ma.masked_invalid(values),
                         cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    ax.plot([cam[0]], [cam[1]], marker='^', color='#1d3557', ms=13, mec='white', mew=1.2,
            zorder=5, clip_on=False)
    ax.annotate('the camera', xy=(cam[0], cam[1]), xytext=(cam[0] + 0.4, cam[1] - 0.05),
                fontsize=8.5, color='#1d3557', va='center')
    ax.set_aspect('equal')
    ax.set_xlim(edges_x[0] - 0.2, edges_x[-1] + 0.2)
    ax.set_ylim(cam[1] - 0.35, edges_y[-1] + 0.2)
    ax.set_xlabel('metres east')
    ax.set_title(title, loc='left', fontsize=10.5)
    ax.grid(False)
    return mesh


def _range_rings(ax, reading, cam, mask=None) -> None:
    """Arcs at fixed distance from the camera, labelled with the measured trend.

    Single cells hold 1-6 poses and are noisy; the ring labels are the aggregate,
    which is where the range trend is actually visible.
    """
    from matplotlib.patches import Circle
    rng = reading.col('range_m')
    ns = reading.err_cm[:, 1]
    if mask is not None:
        rng, ns = rng[mask], ns[mask]
    height = cam[2] - 0.21          # camera height above the marker plane
    for r3d, lo, hi in ((6.0, 5.0, 7.0), (8.0, 7.0, 9.0), (10.0, 9.0, 12.0)):
        ground = math.sqrt(max(r3d ** 2 - height ** 2, 0.01))
        ax.add_patch(Circle((cam[0], cam[1]), ground, fill=False, edgecolor=rd.INK,
                            lw=0.9, ls='--', alpha=0.55, zorder=4))
        sel = (rng >= lo) & (rng < hi)
        if sel.sum() < 5:
            continue
        ax.text(cam[0] - 0.15, cam[1] + ground - 0.22,
                f'{lo:.0f}-{hi:.0f} m from the camera: {ns[sel].mean():+.2f} cm',
                fontsize=7.8, color=rd.INK, ha='center', va='top', zorder=6,
                bbox={'facecolor': 'white', 'alpha': 0.80, 'edgecolor': 'none',
                      'pad': 1.4})


def main() -> None:
    rd.style()
    bb = rd.load_reading('box_bottom')
    kp = rd.load_reading('keypoint_retrained')
    xs = np.array(sorted({round(v, 3) for v in kp.col('x')}))
    ys = np.array(sorted({round(v, 3) for v in kp.col('y')}))
    cam = rd.camera().cam_pos

    clean = kg.both_rendered_mask(kp)      # both marker disks actually rendered
    g_bb, g_kp = grid_of(bb, xs, ys), grid_of(kp, xs, ys)
    g_kp_ns = grid_of(kp, xs, ys, 'ns', mask=clean)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 7.4))
    m1 = draw(axes[0], xs, ys, g_bb, SHARED_MAX,
              'A.  Bottom of the box: wrong everywhere, and worst\n'
              'close to the camera (8.4 cm at 5 m, 5.5 cm at 10 m)', cam)
    draw(axes[1], xs, ys, g_kp, SHARED_MAX,
         'B.  Marked point, same colour scale: the error has\n'
         f'gone (mean {np.nanmean(g_kp):.1f} cm, every cell at the floor)', cam)
    m3 = draw(axes[2], xs, ys, g_kp_ns, SIGNED_MAX,
              'C.  What is left of the marked point, counting only readings\n'
              'where both disks rendered: half a centimetre at most', cam,
              cmap='RdBu_r', vmin=-SIGNED_MAX)
    _range_rings(axes[2], kp, cam, mask=clean)
    axes[0].set_ylabel('metres north')

    cb1 = fig.colorbar(m1, ax=axes[:2], orientation='horizontal', location='bottom',
                       fraction=0.05, pad=0.30, aspect=42)
    cb1.set_label('typical miss in this cell (cm) -- shared scale, lower is better')
    cb3 = fig.colorbar(m3, ax=axes[2], orientation='horizontal', location='bottom',
                       fraction=0.05, pad=0.30, aspect=21)
    cb3.set_label('mean north-south error (cm); blue = read too far south,\n'
                  'i.e. too close to the camera')

    fig.suptitle('The deployed reading is wrong across the whole floor; the marked point '
                 'is not, and what remains of it is a range effect',
                 fontsize=13.5, fontweight='bold', y=0.99)
    rd.note(fig, rd.SOURCE + '  |  each cell is the mean miss of the 1-6 held-out poses '
                             'sampled there (one per heading), so a single cell is noisy; in '
                             'panel C the dashed arcs carry the aggregate trend over the '
                             '4.7-11.5 m of range this capture covers. Blank cells hold no '
                             'accepted held-out pose, mostly places where the racks hide the '
                             'robot from this one camera -- which is why the northern half is '
                             'sparse. Panels A and B count every detection, as a deployment '
                             'would; panel C counts only the 369 readings where both marker '
                             'disks actually rendered, because the pooled far-range pull '
                             '(-1.10 cm at 9-12 m) is mostly the 16 readings there where one '
                             'disk was hidden and the model guessed at it (-3.46 cm). P1 panel '
                             'B shows that split.', width=205)
    fig.subplots_adjust(top=0.87, bottom=0.33, left=0.05, right=0.98)
    rd.save(fig, 'L4_error_over_the_floor')


if __name__ == '__main__':
    main()
