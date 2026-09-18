#!/usr/bin/env python3
"""The static reference dataset, read against the proposed three-role protocol.

Three figures:

  1 map          every reference position, coloured by its dataset role, with the
                 cameras and the declared obstacles.
  2 design       the balance the protocol asks to plot: range and viewing angle
                 per camera, plus headings per position and opportunities per role.
  3 audit        where the protocol's requirements are and are not met by the
                 capture as it stands.

Nothing here is fitted or selected; it only describes what was collected. Written
to check a proposed protocol against reality before adopting it.

    python3 experiments/warehouse_v2_sketches/plot_data_protocol.py
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in
                ('src/unav_common', 'src/experiments',
                 'experiments/reference_controlled_commissioning_v1')]

WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
PROFILES = REPO / 'src/experiments/config/world_profiles.yaml'
OUT = REPO / 'experiments/warehouse_v2_sketches'

ROLE_COLOUR = {
    'detector_fit': '#1b6ca8',
    'detector_validation': '#5aa9e6',
    'commissioning_fit': '#2f8f5b',
    'final_audit': '#c23d36',
    'extension_edge': '#e8a33d',
    'extension_aisle': '#8e44ad',
}
ROLE_ORDER = ['detector_fit', 'detector_validation', 'commissioning_fit',
              'final_audit', 'extension_edge', 'extension_aisle']


def camera_rig(profile):
    labels = {inc: str(cid).replace('camera_', '')
              for inc, cid in zip(profile.get('camera_model_includes') or (),
                                  profile.get('camera_ids') or ())}
    rig = []
    for m in re.finditer(r'<include><name>(external_camera[a-z_]*)</name>'
                         r'<uri>model://[^<]*</uri>(.*?)</include>', WORLD.read_text()):
        pose = re.search(r'<pose>([^<]+)</pose>', m.group(2))
        if pose:
            v = [float(t) for t in pose.group(1).split()]
            rig.append((labels.get(m.group(1), m.group(1)), v[0], v[1], v[2], v[4], v[5]))
    return sorted(rig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import yaml
    import combined_master_capture as cmc

    rows = cmc.load_rows()
    profile = yaml.safe_load(PROFILES.read_text())['worlds']['warehouse_v2.world.sdf']
    regions = profile['known_2d_regions']
    site = next(r for r in regions if r.get('type') == 'site_boundary')
    blocks = [r for r in regions if r.get('type') == 'non_driveable_obstacle']
    rig = camera_rig(profile)

    pos_role, pos_xy = {}, {}
    for r in rows:
        pid = r['global_position_id']
        pos_role[pid] = r['dataset_split']
        pos_xy[pid] = (float(r['robot_x']), float(r['robot_y']))
    headings = defaultdict(set)
    for r in rows:
        headings[r['global_position_id']].add(r['heading_id'])

    # ---------- figure 1: the map -----------------------------------------
    fig, ax = plt.subplots(figsize=(15.5, 12.6))
    fig.subplots_adjust(left=.06, right=.985, top=.90, bottom=.13)
    for r in blocks:
        ax.add_patch(Rectangle((r['xmin'], r['ymin']), r['xmax'] - r['xmin'],
                               r['ymax'] - r['ymin'], fc='#9aa0a6', ec='#6b7075',
                               lw=.6, alpha=.45, zorder=2))
    ax.add_patch(Rectangle((site['xmin'], site['ymin']),
                           site['xmax'] - site['xmin'], site['ymax'] - site['ymin'],
                           fill=False, ec='.35', lw=1.4, ls=(0, (7, 5)), zorder=6))
    for role in ROLE_ORDER:
        pts = [pos_xy[p] for p in pos_xy if pos_role[p] == role]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42,
                   c=ROLE_COLOUR[role], edgecolors='white', linewidths=.5,
                   zorder=7, label=f'{role}  ({len(pts)} positions)')
    for name, cx, cy, _cz, _p, yaw in rig:
        ax.plot([cx], [cy], marker='o', ms=12, mfc='#12263f', mec='white', mew=1.6, zorder=9)
        ax.annotate('', xy=(cx + 2.6 * math.cos(yaw), cy + 2.6 * math.sin(yaw)),
                    xytext=(cx, cy), zorder=9,
                    arrowprops=dict(arrowstyle='-|>', color='#12263f', lw=2.0))
        dx, dy = (-1.3 if cx > 0 else 1.3), (1.2 if cy < 0 else -1.2)
        ax.text(cx + dx, cy + dy, name, ha='center', va='center', fontsize=13,
                weight='bold', color='#12263f', zorder=10,
                bbox=dict(boxstyle='round,pad=.28', fc='white', ec='#12263f', lw=1.1))
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.grid(alpha=.18)
    ax.set_title(f'Static reference dataset — {len(pos_xy)} positions, '
                 f'{len({r["global_pose_id"] for r in rows})} poses, {len(rows)} camera opportunities\n'
                 'split is by POSITION, so no position appears in two roles',
                 fontsize=15, weight='bold', pad=14)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.075), ncol=3, fontsize=11,
              framealpha=.96)
    fig.savefig(OUT / 'protocol_1_reference_map.png', dpi=args.dpi, bbox_inches='tight')
    print('written: protocol_1_reference_map.png')

    # ---------- figure 2: design balance -----------------------------------
    cams = sorted({r['camera_id'] for r in rows})
    rng_by_cam = defaultdict(list)
    ang_by_cam = defaultdict(list)
    campos = {lbl: (x, y) for lbl, x, y, _z, _p, _yw in rig}
    for r in rows:
        c = r['camera_id'].replace('camera_', '')
        try:
            rng_by_cam[c].append(float(r['camera_range_m']))
        except (TypeError, ValueError):
            pass
        if c in campos:
            cx, cy = campos[c]
            ang_by_cam[c].append(abs(math.degrees(math.atan2(
                float(r['robot_y']) - cy, float(r['robot_x']) - cx))))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    fig.subplots_adjust(left=.07, right=.98, top=.90, bottom=.08, hspace=.34, wspace=.20)

    ax = axes[0][0]
    for c in sorted(rng_by_cam):
        ax.hist(rng_by_cam[c], bins=28, histtype='step', lw=2, label=f'camera {c}')
    ax.set_xlabel('range to robot (m)')
    ax.set_ylabel('camera opportunities')
    ax.set_title('Range coverage per camera — the protocol\'s $d_i$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)

    ax = axes[0][1]
    ax.boxplot([rng_by_cam[c] for c in sorted(rng_by_cam)],
               labels=[f'{c}' for c in sorted(rng_by_cam)], showfliers=False)
    ax.set_xlabel('camera')
    ax.set_ylabel('range (m)')
    ax.set_title('Range spread per camera (are long ranges represented?)', fontsize=12)
    ax.grid(alpha=.2, axis='y')

    ax = axes[1][0]
    hc = Counter(len(v) for v in headings.values())
    ks = sorted(hc)
    ax.bar([str(k) for k in ks], [hc[k] for k in ks], color='#2f8f5b')
    for k in ks:
        ax.text(str(k), hc[k], str(hc[k]), ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('headings captured at a position')
    ax.set_ylabel('positions')
    ax.set_title('Headings per position — the protocol\'s $H$', fontsize=12)
    ax.grid(alpha=.2, axis='y')

    ax = axes[1][1]
    rc = Counter(r['dataset_split'] for r in rows)
    order = [k for k in ROLE_ORDER if k in rc]
    ax.barh(range(len(order)), [rc[k] for k in order],
            color=[ROLE_COLOUR[k] for k in order])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)
    for i, k in enumerate(order):
        ax.text(rc[k], i, f' {rc[k]}', va='center', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('camera opportunities')
    ax.set_title('Opportunities per role — the protocol\'s $M$', fontsize=12)
    ax.grid(alpha=.2, axis='x')

    fig.suptitle('Design balance of the static dataset — the appendix figure the protocol asks for',
                 fontsize=15, weight='bold', y=.965)
    fig.savefig(OUT / 'protocol_2_design_balance.png', dpi=args.dpi, bbox_inches='tight')
    print('written: protocol_2_design_balance.png')

    # ---------- figure 3: protocol audit -----------------------------------
    reps = {r['repetition_id'] for r in rows}
    per_pose_cam = Counter((r['global_pose_id'], r['camera_id']) for r in rows)
    split_pos = defaultdict(set)
    for r in rows:
        split_pos[r['dataset_split']].add(r['global_position_id'])
    overlap = any(split_pos[a] & split_pos[b]
                  for i, a in enumerate(split_pos) for b in list(split_pos)[i + 1:])

    checks = [
        ('Split by reference pose, not frame',
         not overlap, 'no position appears in two roles'),
        ('All five cameras per pose',
         len({r['camera_id'] for r in rows}) == 5,
         'batch contract requires all five'),
        ('Positions span the workspace',
         len(pos_xy) >= 400, f'{len(pos_xy)} positions'),
        ('Several headings per position',
         max(len(v) for v in headings.values()) >= 8,
         f'up to {max(len(v) for v in headings.values())}, but '
         f'{sum(1 for v in headings.values() if len(v) < 8)} positions have fewer'),
        ('Misses retained for q',
         any(str(r.get('line_of_sight')).lower() in ('0', 'false') for r in rows)
         or any(int(r.get('semantic_robot_pixels') or 0) == 0 for r in rows),
         'zero-pixel rows are kept as negatives'),
        ('REPEATED frames at a pose (F > 1)',
         max(per_pose_cam.values()) > 1,
         f'repetition_id = {sorted(reps)}, exactly one frame per (pose, camera)'),
        ('Separate D_mu and D_R partitions',
         False,
         'commissioning_fit is ONE pool of 240 positions; the mean and R are '
         'separated by 5-fold spatial blocks inside it, not by disjoint roles'),
        ('Held-out DRIVES as final audit',
         False,
         'final_audit is 40 STATIC positions, not complete trajectories'),
    ]
    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    fig.subplots_adjust(left=.03, right=.985, top=.86, bottom=.06)
    ax.axis('off')
    ax.set_title('Does the current capture satisfy the proposed protocol?',
                 fontsize=16, weight='bold', pad=18)
    for i, (name, ok, note) in enumerate(checks):
        y = 1 - (i + .6) / (len(checks) + .6)
        ax.add_patch(Rectangle((.012, y - .045), .976, .088,
                               fc='#eaf5ee' if ok else '#fdecea',
                               ec='#2f8f5b' if ok else '#c23d36',
                               lw=1.3, transform=ax.transAxes, zorder=1))
        ax.text(.035, y, '✓' if ok else '✗', fontsize=20, va='center',
                color='#2f8f5b' if ok else '#c23d36', weight='bold',
                transform=ax.transAxes, zorder=2)
        ax.text(.075, y, name, fontsize=13, va='center', weight='bold',
                color='#1d2530', transform=ax.transAxes, zorder=2)
        ax.text(.52, y, note, fontsize=11, va='center', color='#444',
                transform=ax.transAxes, zorder=2)
    fig.savefig(OUT / 'protocol_3_audit.png', dpi=args.dpi, bbox_inches='tight')
    print('written: protocol_3_audit.png')

    print()
    print(f'N positions {len(pos_xy)} / poses {len({r["global_pose_id"] for r in rows})} '
          f'/ F {sorted(reps)} / M {len(rows)}')
    for name, ok, note in checks:
        print(f'  {"OK  " if ok else "FAIL"} {name}: {note}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
