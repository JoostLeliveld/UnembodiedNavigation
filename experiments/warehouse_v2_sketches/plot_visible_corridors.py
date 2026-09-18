#!/usr/bin/env python3
"""Is every corridor drivable IN A VISIBLE WAY? One large, readable plan view.

The criterion is not "no blind cells". It is that the robot can drive every
corridor while being seen, so the VISIBLE part of the driveable floor must stay
one connected network that spans the building.

    driveable   site boundary minus the declared obstacle boxes
    body-safe   eroded by the robot half-width, so a cell means the BODY fits
    visible     max_i q_i >= --threshold on the commissioned availability field

The panel colours the body-safe floor by whether it is visible, marks the dark
patches, and reports the connectivity that decides the question.

    python3 experiments/warehouse_v2_sketches/plot_visible_corridors.py
    python3 experiments/warehouse_v2_sketches/plot_visible_corridors.py --threshold 0.2
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in ('src/unav_common', 'src/experiments')]

WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
PROFILES = REPO / 'src/experiments/config/world_profiles.yaml'
FIELD = REPO / ('logs/studies/reference_controlled_commissioning_v1/'
                'selected_correction_r_planner_fields_20260915_v4/u0_planner_field.npz')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threshold', type=float, default=0.10)
    ap.add_argument('--out', type=Path,
                    default=Path('experiments/warehouse_v2_sketches/visible_corridors.png'))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from scipy import ndimage
    import yaml

    field = np.load(FIELD, allow_pickle=True)
    xs, ys = field['xs'], field['ys']
    q = field['availability'].max(axis=0)

    profile = yaml.safe_load(PROFILES.read_text())['worlds']['warehouse_v2.world.sdf']
    regions = profile['known_2d_regions']
    site = next(r for r in regions if r.get('type') == 'site_boundary')
    blocks = [r for r in regions if r.get('type') == 'non_driveable_obstacle']

    X, Y = np.meshgrid(xs, ys)
    drive = ((X >= site['xmin']) & (X <= site['xmax'])
             & (Y >= site['ymin']) & (Y <= site['ymax']))
    for r in blocks:
        drive &= ~((X >= r['xmin']) & (X <= r['xmax'])
                   & (Y >= r['ymin']) & (Y <= r['ymax']))
    # One cell of erosion on the 0.20 m grid approximates the 0.275 m half-width:
    # a surviving cell means the body fits, not just the centre point.
    safe = ndimage.binary_erosion(drive, np.ones((3, 3)))
    visible = safe & (q >= args.threshold)
    dark = safe & ~visible

    lab, n = ndimage.label(visible)
    sizes = ndimage.sum(visible, lab, range(1, n + 1)) if n else []
    main = lab == (1 + int(np.argmax(sizes))) if n else visible
    stranded = visible & ~main

    labels = {inc: str(cid).replace('camera_', '')
              for inc, cid in zip(profile.get('camera_model_includes') or (),
                                  profile.get('camera_ids') or ())}
    cams = []
    for m in re.finditer(r'<include><name>(external_camera[a-z_]*)</name>'
                         r'<uri>model://[^<]*</uri>(.*?)</include>', WORLD.read_text()):
        pose = re.search(r'<pose>([^<]+)</pose>', m.group(2))
        if pose:
            v = [float(t) for t in pose.group(1).split()]
            cams.append((labels.get(m.group(1), m.group(1)), v[0], v[1], v[5]))

    extent = [xs[0] - .1, xs[-1] + .1, ys[0] - .1, ys[-1] + .1]
    fig, ax = plt.subplots(figsize=(16.5, 13.4))
    fig.subplots_adjust(left=.06, right=.985, top=.905, bottom=.075)

    ax.imshow(np.where(visible, 1., np.nan), origin='lower', extent=extent,
              cmap=ListedColormap(['#4c9a6a']), vmin=0, vmax=1, alpha=.80,
              zorder=2, interpolation='nearest')
    ax.imshow(np.where(dark, 1., np.nan), origin='lower', extent=extent,
              cmap=ListedColormap(['#d81e3f']), vmin=0, vmax=1,
              zorder=4, interpolation='nearest')
    if stranded.any():
        ax.imshow(np.where(stranded, 1., np.nan), origin='lower', extent=extent,
                  cmap=ListedColormap(['#8e44ad']), vmin=0, vmax=1,
                  zorder=5, interpolation='nearest')

    for r in blocks:
        ax.add_patch(Rectangle((r['xmin'], r['ymin']), r['xmax'] - r['xmin'],
                               r['ymax'] - r['ymin'], fc='#9aa0a6', ec='#6b7075',
                               lw=.7, alpha=.55, zorder=3))
    ax.add_patch(Rectangle((site['xmin'], site['ymin']),
                           site['xmax'] - site['xmin'], site['ymax'] - site['ymin'],
                           fill=False, ec='0.35', lw=1.4, ls=(0, (7, 5)), zorder=7))

    for name, cx, cy, yaw in cams:
        ax.plot([cx], [cy], marker='o', ms=11, mfc='#1f4e9c', mec='white', mew=1.6, zorder=9)
        ax.annotate('', xy=(cx + 2.6 * math.cos(yaw), cy + 2.6 * math.sin(yaw)),
                    xytext=(cx, cy), zorder=9,
                    arrowprops=dict(arrowstyle='-|>', color='#1f4e9c', lw=2.0))
        dx = -1.25 if cx > 0 else 1.25
        dy = 1.15 if cy < 0 else -1.15
        ax.text(cx + dx, cy + dy, name, ha='center', va='center', fontsize=13,
                weight='bold', color='#123a78', zorder=10,
                bbox=dict(boxstyle='round,pad=.28', fc='white', ec='#1f4e9c', lw=1.1, alpha=.94))

    ax.set_xlim(xs[0] - 1.0, xs[-1] + 1.0)
    ax.set_ylim(ys[0] - 1.0, ys[-1] + 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.18, zorder=1)

    pct = 100 * main.sum() / safe.sum()
    verdict = ('EVERY corridor is drivable in a visible way'
               if n == 1 else f'visible floor splits into {n} components')
    ax.set_title(
        f'warehouse_v2 — can the robot drive every corridor while being seen?\n'
        f'visible = $\\max_i q_i \\geq {args.threshold:g}$ on body-safe floor    '
        f'{verdict}: one network of {int(main.sum())} cells = {pct:.1f}% of drivable',
        fontsize=14.5, weight='bold', pad=16)

    handles = [
        Line2D([], [], marker='s', ls='', ms=15, mfc='#4c9a6a', mec='none',
               label=f'drivable AND visible  ({int(visible.sum())} cells, '
                     f'{visible.sum() * .04:.0f} m$^2$)'),
        Line2D([], [], marker='s', ls='', ms=15, mfc='#d81e3f', mec='none',
               label=f'drivable but blind  ({int(dark.sum())} cells, '
                     f'{dark.sum() * .04:.1f} m$^2$)'),
        Line2D([], [], marker='s', ls='', ms=15, mfc='#9aa0a6', mec='none',
               label=f'declared obstacle  ({len(blocks)} boxes)'),
        Line2D([], [], marker='o', ls='', ms=11, mfc='#1f4e9c', mec='white',
               label='camera mount and facing'),
    ]
    if stranded.any():
        handles.insert(2, Line2D([], [], marker='s', ls='', ms=15, mfc='#8e44ad',
                                 mec='none', label=f'visible but CUT OFF ({int(stranded.sum())})'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.065),
              ncol=2, fontsize=12, framealpha=.96)

    out = REPO / args.out if not args.out.is_absolute() else args.out
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'written: {out}')
    print(f'  body-safe drivable : {int(safe.sum())} cells ({safe.sum() * .04:.1f} m^2)')
    print(f'  visible            : {int(visible.sum())} ({100 * visible.sum() / safe.sum():.1f}%)')
    print(f'  blind              : {int(dark.sum())} ({dark.sum() * .04:.2f} m^2)')
    print(f'  visible components : {n}  largest {int(main.sum())} ({pct:.1f}%)')
    print(f'  stranded visible   : {int(stranded.sum())}')


if __name__ == '__main__':
    raise SystemExit(main())
