#!/usr/bin/env python3
"""The four maps that define the warehouse_v2 commissioning setup, in one figure.

  1 TRAVERSABLE   the driveable region. The profile now declares OBSTACLES and a
                  site boundary, so the traversable region is DERIVED as
                  boundary minus obstacles -- it is no longer a lane list.
  2 OBSTACLES     the declared non-driveable boxes, which are what
                  ``nogo_mode: keep_out`` hands the planner.
  3 MASTER CAPTURE  where the 22130 commissioning views were actually taken:
                  695 positions from the v3 lattice plus the v4 edge extension.
  4 VISIBILITY    the commissioned availability field, max_i q_i, with the
                  blind-and-driveable cells marked.

Every panel is read from the CURRENT repo state; nothing is hard-coded here.

**Read the staleness note the figure prints.** The world was edited on
2026-09-18 (camera C 38 -> 44 deg, Cm barrels de-stacked) AFTER the capture and
after the availability field was fitted, so panels 3 and 4 describe the previous
world. The figure says so on its face rather than quietly mixing them.

    python3 experiments/warehouse_v2_sketches/plot_system_maps.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in
                ('src/planning', 'src/unav_common', 'src/experiments', 'src/reliability')]

WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
PROFILES = REPO / 'src/experiments/config/world_profiles.yaml'
FIELD = REPO / ('logs/studies/reference_controlled_commissioning_v1/'
                'selected_correction_r_planner_fields_20260915_v4/u0_planner_field.npz')
CAPTURE_MANIFEST = REPO / 'experiments/thesis_pipeline_lock/camera_capture_map_manifest.json'

TARGET_Z = 0.35
BLIND_Q = 0.10


def camera_poses(world_text, labels):
    out = []
    pattern = (r'<include><name>(external_camera[a-z_]*)</name>'
               r'<uri>model://[^<]*</uri>(.*?)</include>')
    for m in re.finditer(pattern, world_text):
        pose = re.search(r'<pose>([^<]+)</pose>', m.group(2))
        if not pose:
            continue
        v = [float(x) for x in pose.group(1).split()]
        out.append((labels.get(m.group(1), m.group(1)), v[0], v[1], v[2], v[4], v[5]))
    return out


def derived_traversable(profile, xs, ys):
    """Boundary minus obstacles, on the availability grid."""
    regions = profile['known_2d_regions']
    site = next(r for r in regions if r.get('type') == 'site_boundary')
    blocks = [r for r in regions if r.get('type') == 'non_driveable_obstacle']
    lanes = [r for r in regions if r.get('type') == 'traversable']
    X, Y = np.meshgrid(xs, ys)
    if lanes:  # older profile style: an explicit lane list
        mask = np.zeros(X.shape, dtype=bool)
        for r in lanes:
            mask |= ((X >= r['xmin']) & (X <= r['xmax'])
                     & (Y >= r['ymin']) & (Y <= r['ymax']))
        return mask, 'declared lanes', lanes, blocks, site
    mask = ((X >= site['xmin']) & (X <= site['xmax'])
            & (Y >= site['ymin']) & (Y <= site['ymax']))
    for r in blocks:
        mask &= ~((X >= r['xmin']) & (X <= r['xmax'])
                  & (Y >= r['ymin']) & (Y <= r['ymax']))
    return mask, 'boundary minus obstacles', lanes, blocks, site


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path,
                    default=Path('experiments/warehouse_v2_sketches/system_maps.png'))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import yaml

    profile = yaml.safe_load(PROFILES.read_text())['worlds']['warehouse_v2.world.sdf']
    labels = {inc: str(cid).replace('camera_', '')
              for inc, cid in zip(profile.get('camera_model_includes') or (),
                                  profile.get('camera_ids') or ())}
    cams = camera_poses(WORLD.read_text(), labels)

    field = np.load(FIELD, allow_pickle=True)
    xs, ys = field['xs'], field['ys']
    q_max = field['availability'].max(axis=0)
    drive, drive_how, lanes, blocks, site = derived_traversable(profile, xs, ys)

    sys.path.insert(0, str(REPO / 'experiments/reference_controlled_commissioning_v1'))
    import combined_master_capture as cmc
    rows = cmc.load_rows()
    cap = {}
    for r in rows:
        cap.setdefault((float(r['robot_x']), float(r['robot_y'])), set()).add(r['capture_source'])

    world_sha = hashlib.sha256(WORLD.read_bytes()).hexdigest()
    cap_sha = json.loads(CAPTURE_MANIFEST.read_text())['spatial_design']['world_sha256']

    extent = [xs[0] - .1, xs[-1] + .1, ys[0] - .1, ys[-1] + .1]
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 15.4))
    fig.subplots_adjust(left=.05, right=.965, top=.905, bottom=.075, wspace=.14, hspace=.20)

    def frame(ax, title):
        ax.add_patch(Rectangle((site['xmin'], site['ymin']),
                               site['xmax'] - site['xmin'], site['ymax'] - site['ymin'],
                               fill=False, ec='0.4', lw=1.3, ls=(0, (6, 4)), zorder=6))
        for name, cx, cy, _cz, _p, yaw in cams:
            ax.plot([cx], [cy], marker='o', ms=8, mfc='#1f4e9c', mec='white', mew=1.3, zorder=9)
            ax.annotate('', xy=(cx + 2.1 * math.cos(yaw), cy + 2.1 * math.sin(yaw)),
                        xytext=(cx, cy), zorder=9,
                        arrowprops=dict(arrowstyle='-|>', color='#1f4e9c', lw=1.5))
            dx = -1.1 if cx > 0 else 1.1
            dy = 1.0 if cy < 0 else -1.0
            ax.text(cx + dx, cy + dy, name, ha='center', va='center', fontsize=9,
                    weight='bold', color='#123a78', zorder=10,
                    bbox=dict(boxstyle='round,pad=.2', fc='white', ec='#1f4e9c', lw=.8, alpha=.92))
        ax.set_xlim(xs[0] - .8, xs[-1] + .8)
        ax.set_ylim(ys[0] - .8, ys[-1] + .8)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title, fontsize=11.5)

    # ---- 1 traversable -------------------------------------------------
    ax = axes[0][0]
    ax.imshow(np.where(drive, 1., np.nan), origin='lower', extent=extent,
              cmap=matplotlib.colors.ListedColormap(['#4c9a6a']), vmin=0, vmax=1,
              alpha=.75, zorder=2, interpolation='nearest')
    frame(ax, f'1  TRAVERSABLE  ({drive_how})\n'
              f'{int(drive.sum())} of {drive.size} grid cells '
              f'= {drive.sum() * 0.04:.1f} m$^2$ driveable')

    # ---- 2 obstacles ---------------------------------------------------
    ax = axes[1][0]
    for r in blocks:
        ax.add_patch(Rectangle((r['xmin'], r['ymin']), r['xmax'] - r['xmin'],
                               r['ymax'] - r['ymin'], fc='#c2453a', ec='#8d2f26',
                               lw=.7, alpha=.62, zorder=3))
    frame(ax, f'2  OBSTACLES  (declared non-driveable)\n'
              f'{len(blocks)} boxes — what nogo_mode: keep_out gives the planner')

    # ---- 3 master capture ----------------------------------------------
    ax = axes[0][1]
    ax.imshow(np.where(drive, .18, np.nan), origin='lower', extent=extent,
              cmap='Greys', vmin=0, vmax=1, zorder=1, interpolation='nearest')
    for r in blocks:
        ax.add_patch(Rectangle((r['xmin'], r['ymin']), r['xmax'] - r['xmin'],
                               r['ymax'] - r['ymin'], fc='#cccccc', ec='#aaaaaa',
                               lw=.6, alpha=.55, zorder=2))
    v3 = [p for p, s in cap.items() if 'v3' in s]
    ex = [p for p, s in cap.items() if 'v3' not in s]
    ax.scatter([p[0] for p in v3], [p[1] for p in v3], s=13, c='#1b6ca8',
               marker='o', zorder=7, label=f'v3 master ({len(v3)})')
    ax.scatter([p[0] for p in ex], [p[1] for p in ex], s=22, c='#e8a33d',
               marker='^', zorder=8, label=f'v4 edge extension ({len(ex)})')
    frame(ax, f'3  MASTER CAPTURE  ({len(rows)} views, {len(cap)} positions)\n'
              f'8 headings x 5 cameras per position')
    ax.legend(loc='lower left', fontsize=8.5, framealpha=.94)

    # ---- 4 visibility --------------------------------------------------
    ax = axes[1][1]
    im = ax.imshow(np.where(drive, q_max, np.nan), origin='lower', extent=extent,
                   cmap='viridis', vmin=0, vmax=1, zorder=2, interpolation='nearest')
    blind = drive & (q_max < BLIND_Q)
    ax.imshow(np.where(blind, 1., np.nan), origin='lower', extent=extent,
              cmap=matplotlib.colors.ListedColormap(['#d81e3f']), vmin=0, vmax=1,
              zorder=4, interpolation='nearest')
    frame(ax, f'4  VISIBILITY  best single-camera availability $\\max_i q_i$\n'
              f'red = blind on driveable ground ($q<{BLIND_Q:g}$): {int(blind.sum())} cells, '
              f'{blind.sum() * 0.04:.2f} m$^2$')
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
    cb.set_label(r'$\max_i q_i$')
    ax.legend(handles=[Line2D([], [], marker='s', ls='', ms=10, mfc='#d81e3f',
                              mec='none', label=f'blind ($q<{BLIND_Q:g}$)')],
              loc='lower left', fontsize=8.5, framealpha=.94)

    stale = world_sha[:12] != cap_sha[:12]
    note = (f'world on disk {world_sha[:12]}…   capture manifest world {cap_sha[:12]}…'
            + ('   ⚠ panels 3 and 4 were produced against a DIFFERENT world '
               '(camera C 38→44°, Cm de-stacked) and are stale'
               if stale else '   (consistent)'))
    fig.suptitle('warehouse_v2 — the four maps that define the commissioning setup',
                 fontsize=15, weight='bold', y=.972)
    fig.text(.5, .938, note, ha='center', fontsize=9.5,
             color='#8d2f26' if stale else '#2f6f4f')

    out = REPO / args.out if not args.out.is_absolute() else args.out
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'written: {out}')
    print(f'  traversable : {int(drive.sum())} cells ({drive.sum() * 0.04:.1f} m^2) via {drive_how}')
    print(f'  obstacles   : {len(blocks)} declared boxes, {len(lanes)} declared lanes')
    print(f'  capture     : {len(rows)} views at {len(cap)} positions')
    print(f'  blind       : {int(blind.sum())} driveable cells at q<{BLIND_Q}')
    print(f'  world sha   : {world_sha[:12]}   capture manifest: {cap_sha[:12]}'
          + ('   STALE' if stale else ''))


if __name__ == '__main__':
    raise SystemExit(main())
