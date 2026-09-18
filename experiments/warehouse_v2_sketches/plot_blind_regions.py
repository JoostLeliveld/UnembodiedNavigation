#!/usr/bin/env python3
"""Plan view of warehouse_v2: cameras, lanes, occluders and the blind regions.

Two panels:

  left   the commissioned availability field, max_i q_i over the five cameras,
         with the traversable lanes and the camera mounts drawn over it.
  right  the same floor with the blind-and-driveable cells classified by CAUSE:
         blocked by a C-block stack (removable by deleting the barrels) versus
         low availability from range/angle (removing the barrels does nothing).

Blind is ``max_i q_i(cell) < --threshold`` on the 0.20 m commissioned grid,
intersected with the declared traversable lanes -- ground the robot can actually
reach. Availability is identical across arms u0-u4 (only R differs), so one map
covers every commissioned arm; w0/w3 are q=1 arms with no blind cells at all.

Cause is decided by ray-casting camera -> cell at the 0.35 m visibility target
height, once against all 37 occluder prisms and once with the ten C-block stacks
removed. A cell counts as barrel-caused only if it is blocked now and clear
without them.

    python3 experiments/warehouse_v2_sketches/plot_blind_regions.py
    python3 experiments/warehouse_v2_sketches/plot_blind_regions.py --threshold 0.05
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in ('src/planning', 'src/unav_common', 'src/experiments')]

WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
PROFILES = REPO / 'src/experiments/config/world_profiles.yaml'
FIELD = REPO / ('logs/studies/reference_controlled_commissioning_v1/'
                'selected_correction_r_planner_fields_20260915_v4/u0_planner_field.npz')

#: the ten section-C floor stacks -- the "high stacked red barrels"
C_STACK_PREFIXES = ('obs_Cs_p', 'obs_Cm_p', 'obs_Cn_p')
#: height the visibility model targets on the robot
TARGET_Z = 0.35


def camera_labels(profile):
    """include name -> short camera label, taken from the profile's own pairing.

    ``camera_model_includes`` and ``camera_ids`` are positionally paired in
    world_profiles.yaml, so the mapping is read rather than derived: the bare
    include is ``external_camera`` with no suffix, and string-munging the names
    produced labels like "CAM" and "CAMO".
    """
    includes = list(profile.get('camera_model_includes') or ())
    ids = list(profile.get('camera_ids') or ())
    if len(includes) != len(ids):
        raise SystemExit('camera_model_includes and camera_ids disagree in length')
    return {inc: str(cid).replace('camera_', '') for inc, cid in zip(includes, ids)}


def camera_poses(world_text):
    """(name, x, y, yaw_rad) for each external camera include in the world."""
    out = []
    pattern = (r'<include><name>(external_camera[a-z_]*)</name>'
               r'<uri>model://[^<]*</uri>(.*?)</include>')
    for m in re.finditer(pattern, world_text):
        pose = re.search(r'<pose>([^<]+)</pose>', m.group(2))
        if not pose:
            continue
        v = [float(x) for x in pose.group(1).split()]
        out.append((m.group(1), v[0], v[1], v[2], v[5]))
    return out


def load_occluders():
    from unav_common.occlusion_geometry import parse_occlusion_scene_from_world
    scene = parse_occlusion_scene_from_world(
        str(WORLD), model_name=('warehouse_v2_occluders',),
        geometry_tags=('collision',), robot_z_range=None)
    prisms = list(scene.prisms)
    is_stack = lambda p: any(k in (getattr(p, 'name', '') or '') for k in C_STACK_PREFIXES)
    return prisms, [p for p in prisms if not is_stack(p)], [p for p in prisms if is_stack(p)]


def visible(prisms, cam, target, samples=260):
    """True when no prism intersects the segment camera -> target."""
    c = np.asarray(cam, dtype=float)
    t = np.asarray(target, dtype=float)
    for s in np.linspace(0.02, 0.98, samples):
        p = c + (t - c) * s
        for q in prisms:
            if (q.xmin <= p[0] <= q.xmax and q.ymin <= p[1] <= q.ymax
                    and q.zmin <= p[2] <= q.zmax):
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threshold', type=float, default=0.10,
                    help='blind when max_i q_i is below this (default 0.10)')
    ap.add_argument('--out', type=Path,
                    default=Path('experiments/warehouse_v2_sketches/blind_regions_map.png'))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import yaml

    field = np.load(FIELD, allow_pickle=True)
    xs, ys = field['xs'], field['ys']
    q_max = field['availability'].max(axis=0)

    profile = yaml.safe_load(PROFILES.read_text())['worlds']['warehouse_v2.world.sdf']
    regions = profile['known_2d_regions']
    labels = camera_labels(profile)
    lanes = [r for r in regions if r.get('type') == 'traversable']
    blocks = [r for r in regions if r.get('type') == 'non_driveable_obstacle']
    site = next(r for r in regions if r.get('type') == 'site_boundary')

    X, Y = np.meshgrid(xs, ys)
    driveable = np.zeros_like(q_max, dtype=bool)
    for lane in lanes:
        driveable |= ((X >= lane['xmin']) & (X <= lane['xmax'])
                      & (Y >= lane['ymin']) & (Y <= lane['ymax']))

    blind = (q_max < args.threshold) & driveable
    all_prisms, without_stacks, _stacks = load_occluders()
    cams = camera_poses(WORLD.read_text())
    eyes = [(c[1], c[2], c[3]) for c in cams]

    barrel = np.zeros_like(blind)
    for j, i in zip(*np.where(blind)):
        target = (xs[i], ys[j], TARGET_Z)
        now = any(visible(all_prisms, e, target) for e in eyes)
        without = any(visible(without_stacks, e, target) for e in eyes)
        barrel[j, i] = without and not now

    extent = [xs[0] - .1, xs[-1] + .1, ys[0] - .1, ys[-1] + .1]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))
    fig.subplots_adjust(left=.055, right=.975, top=.885, bottom=.175, wspace=.16)

    def draw_floor(ax):
        ax.add_patch(Rectangle((site['xmin'], site['ymin']),
                               site['xmax'] - site['xmin'], site['ymax'] - site['ymin'],
                               fill=False, ec='0.45', lw=1.2, ls=(0, (6, 4)), zorder=3))
        for lane in lanes:
            ax.add_patch(Rectangle((lane['xmin'], lane['ymin']),
                                   lane['xmax'] - lane['xmin'], lane['ymax'] - lane['ymin'],
                                   fill=False, ec='#2f6f4f', lw=1.0, alpha=.85, zorder=4))
        for b in blocks:
            stack = b['name'].startswith('C')
            ax.add_patch(Rectangle((b['xmin'], b['ymin']),
                                   b['xmax'] - b['xmin'], b['ymax'] - b['ymin'],
                                   fc='#d94a3d' if stack else '#9a9a9a',
                                   alpha=.30 if stack else .22,
                                   ec='#d94a3d' if stack else '#7a7a7a',
                                   lw=1.0, zorder=2))
            ax.text((b['xmin'] + b['xmax']) / 2, (b['ymin'] + b['ymax']) / 2, b['name'],
                    ha='center', va='center', fontsize=7.5,
                    color='#7a2018' if stack else '#555555', zorder=6)
        for name, cx, cy, _cz, yaw in cams:
            ax.plot([cx], [cy], marker='o', ms=9, mfc='#1f4e9c', mec='white',
                    mew=1.4, zorder=9)
            ax.annotate('', xy=(cx + 2.3 * math.cos(yaw), cy + 2.3 * math.sin(yaw)),
                        xytext=(cx, cy), zorder=9,
                        arrowprops=dict(arrowstyle='-|>', color='#1f4e9c', lw=1.7))
            # Offset the label INWARD, so a corner mount does not push it off-axis.
            dx = -1.15 if cx > 0 else 1.15
            dy = 1.0 if cy < 0 else -1.0
            ax.text(cx + dx, cy + dy, labels[name],
                    ha='center', va='center', fontsize=9.5, color='#123a78',
                    weight='bold', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#1f4e9c',
                              lw=.9, alpha=.92))
        ax.set_xlim(xs[0] - .6, xs[-1] + .6)
        ax.set_ylim(ys[0] - .6, ys[-1] + .6)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    ax = axes[0]
    im = ax.imshow(q_max, origin='lower', extent=extent, cmap='viridis',
                   vmin=0, vmax=1, zorder=1, interpolation='nearest')
    draw_floor(ax)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
    cb.set_label(r'best single-camera availability  $\max_i q_i$')
    ax.set_title('Commissioned coverage, five cameras\n'
                 'red boxes are the section-C floor stacks (the barrels)', fontsize=11)

    ax = axes[1]
    ax.imshow(np.where(driveable, .22, np.nan), origin='lower', extent=extent,
              cmap='Greys', vmin=0, vmax=1, zorder=1, interpolation='nearest')
    shown = np.full(q_max.shape, np.nan)
    shown[blind & ~barrel] = 0.
    shown[blind & barrel] = 1.
    ax.imshow(shown, origin='lower', extent=extent,
              cmap=ListedColormap(['#f0a500', '#d94a3d']),
              vmin=0, vmax=1, zorder=5, interpolation='nearest')
    draw_floor(ax)
    n_barrel, n_other = int(barrel.sum()), int((blind & ~barrel).sum())
    ax.set_title(f'Blind and driveable at $\\max_i q_i<{args.threshold:g}$: '
                 f'{int(blind.sum())} cells of {int(driveable.sum())} '
                 f'({100 * blind.sum() / driveable.sum():.1f}% of driveable)', fontsize=11)
    # Below the axes: the floor fills the frame, so an inset legend covers data.
    ax.legend(handles=[
        Line2D([], [], marker='s', ls='', ms=11, mfc='#d94a3d', mec='none',
               label=f'blocked by a C-block stack  ({n_barrel} cells, '
                     f'{n_barrel * 0.04:.2f} m$^2$) — removable'),
        Line2D([], [], marker='s', ls='', ms=11, mfc='#f0a500', mec='none',
               label=f'range / viewing angle  ({n_other} cells, '
                     f'{n_other * 0.04:.2f} m$^2$) — NOT fixed by removing barrels'),
        Line2D([], [], marker='o', ls='', ms=9, mfc='#1f4e9c', mec='white', label='camera mount'),
        Line2D([], [], color='#2f6f4f', lw=1.4, label='traversable lane'),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.115), ncol=2,
        fontsize=9.5, framealpha=.96)

    fig.suptitle('warehouse_v2 \u2014 camera layout and blind regions on driveable ground',
                 fontsize=14, weight='bold', y=.975)
    out = REPO / args.out if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'written: {out}')
    print(f'  blind & driveable      {int(blind.sum()):5d} cells  '
          f'{blind.sum() * 0.04:6.2f} m^2')
    print(f'  caused by C-block      {n_barrel:5d} cells  {n_barrel * 0.04:6.2f} m^2')
    print(f'  range / angle          {n_other:5d} cells  {n_other * 0.04:6.2f} m^2')


if __name__ == '__main__':
    raise SystemExit(main())
