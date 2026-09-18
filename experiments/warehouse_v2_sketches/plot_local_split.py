#!/usr/bin/env python3
"""Local-stratum split: every operating region contributes to every role.

This replaces the large-exclusive-block design. The warehouse is a FIXED
INSTALLATION, so there is no unseen floor to generalise to and no reason to
withhold whole regions from the mean model. The one independence that still
matters is that `R` must be fitted to residuals of a FROZEN correction, i.e. on
reference poses that correction did not train on.

So roles are assigned WITHIN each local patch:

    one 2 m patch          instead of        whole patch = D_mu
      mu   mu   R                            next patch  = D_R
      mu   dev  mu                           next patch  = D_dev
      R    mu   audit

2 m is chosen from the data, not by taste: at 1.0 m only 22% of patches hold
three or more reference positions, at 1.6 m 63%, at 2.0 m 95%. Below that most
patches cannot host all three roles and the design silently degrades to the
block split it is meant to replace.

The atomic unit is the reference POSITION: every camera and every heading at a
position goes to the same role, so D_mu and D_R never share a scene.

Hard regions (few cameras ever see the robot) are additionally balanced, so each
role gets its proportional share of the difficult sensing conditions rather than
whatever the greedy pass leaves over.

Nothing is fitted here; this proposes and audits the partition.

    python3 experiments/warehouse_v2_sketches/plot_local_split.py
    python3 experiments/warehouse_v2_sketches/plot_local_split.py --patch 1.6
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

TARGET = [('D_mu', 0.55), ('D_R', 0.28), ('D_dev', 0.17)]
COLOUR = {'D_mu': '#2f8f5b', 'D_R': '#1b6ca8', 'D_dev': '#e8a33d',
          'final_audit': '#c23d36'}
ROLES = ['D_mu', 'D_R', 'D_dev']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patch', type=float, default=2.0, help='local stratum size (m)')
    ap.add_argument('--seed', type=int, default=20260918)
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import yaml
    import combined_master_capture as cmc

    rows = cmc.load_rows()
    profile = yaml.safe_load(PROFILES.read_text())['worlds']['warehouse_v2.world.sdf']
    regions = profile['known_2d_regions']
    site = next(r for r in regions if r.get('type') == 'site_boundary')
    obstacles = [r for r in regions if r.get('type') == 'non_driveable_obstacle']

    labels = {inc: str(cid).replace('camera_', '')
              for inc, cid in zip(profile.get('camera_model_includes') or (),
                                  profile.get('camera_ids') or ())}
    rig = []
    for m in re.finditer(r'<include><name>(external_camera[a-z_]*)</name>'
                         r'<uri>model://[^<]*</uri>(.*?)</include>', WORLD.read_text()):
        pose = re.search(r'<pose>([^<]+)</pose>', m.group(2))
        if pose:
            v = [float(t) for t in pose.group(1).split()]
            rig.append((labels.get(m.group(1), m.group(1)), v[0], v[1], v[5]))
    rig.sort()

    audit = {r['global_position_id'] for r in rows if r['dataset_split'] == 'final_audit'}
    pos_xy = {}
    for r in rows:
        pos_xy[r['global_position_id']] = (float(r['robot_x']), float(r['robot_y']))
    use = [p for p in pos_xy if p not in audit]

    seen = defaultdict(set)
    for r in rows:
        try:
            px = int(r['semantic_robot_pixels'])
        except (TypeError, ValueError):
            px = 0
        if px > 0:
            seen[r['global_position_id']].add(r['camera_id'])
    ncam = {p: len(seen.get(p, ())) for p in use}
    hard = {p for p in use if ncam[p] <= 1}

    # ---- assign roles WITHIN each local patch --------------------------
    cell = args.patch
    patch = defaultdict(list)
    for p in use:
        x, y = pos_xy[p]
        patch[(int(np.floor(x / cell)), int(np.floor(y / cell)))].append(p)

    rng = np.random.default_rng(args.seed)
    role_of = {}
    quota = np.array([t[1] for t in TARGET], dtype=float)
    for key, members in patch.items():
        # order members so hard positions are spread across roles, not clumped
        members = sorted(members, key=lambda p: (p not in hard, pos_xy[p]))
        # deal them round-robin against the cumulative quota: this keeps the
        # role mix right INSIDE the patch even when it holds only 2-3 positions
        want = quota * len(members)
        got = np.zeros(3)
        for p in members:
            deficit = (want - got) / np.maximum(want, 1e-9)
            best = int(np.argmax(deficit + rng.uniform(0, 1e-6, 3)))
            role_of[p] = TARGET[best][0]
            got[best] += 1
    for p in audit:
        role_of[p] = 'final_audit'

    npos = Counter(role_of[p] for p in pos_xy)
    opp = Counter(role_of[r['global_position_id']] for r in rows)
    order = ROLES + ['final_audit']

    # nearest same-role / cross-role distance, the independence diagnostic
    from scipy.spatial import cKDTree
    pts = {k: np.array([pos_xy[p] for p in use if role_of[p] == k]) for k in ROLES}
    tree_mu = cKDTree(pts['D_mu'])
    d_R, _ = tree_mu.query(pts['D_R'], k=1)
    d_dev, _ = tree_mu.query(pts['D_dev'], k=1)

    # ---------------- figure --------------------------------------------
    fig = plt.figure(figsize=(18.5, 12.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], height_ratios=[1.35, 1],
                          left=.042, right=.985, top=.885, bottom=.075,
                          wspace=.23, hspace=.28)

    ax = fig.add_subplot(gs[:, 0])
    for r in obstacles:
        ax.add_patch(Rectangle((r['xmin'], r['ymin']), r['xmax'] - r['xmin'],
                               r['ymax'] - r['ymin'], fc='#9aa0a6', ec='#6b7075',
                               lw=.5, alpha=.40, zorder=2))
    for gx in np.arange(math.floor(site['xmin'] / cell) * cell,
                        site['xmax'] + cell, cell):
        ax.axvline(gx, color='#bbb', lw=.5, zorder=1)
    for gy in np.arange(math.floor(site['ymin'] / cell) * cell,
                        site['ymax'] + cell, cell):
        ax.axhline(gy, color='#bbb', lw=.5, zorder=1)
    ax.add_patch(Rectangle((site['xmin'], site['ymin']),
                           site['xmax'] - site['xmin'], site['ymax'] - site['ymin'],
                           fill=False, ec='.35', lw=1.3, ls=(0, (7, 5)), zorder=6))
    for role in order:
        pp = [pos_xy[p] for p in pos_xy if role_of[p] == role]
        ax.scatter([q[0] for q in pp], [q[1] for q in pp], s=26, c=COLOUR[role],
                   edgecolors='white', linewidths=.35, zorder=7,
                   label=f'{role}  {len(pp)} pos / {opp[role]} opp')
    for name, cx, cy, yaw in rig:
        ax.plot([cx], [cy], marker='o', ms=10, mfc='#12263f', mec='white', mew=1.4, zorder=9)
        ax.annotate('', xy=(cx + 2.3 * math.cos(yaw), cy + 2.3 * math.sin(yaw)),
                    xytext=(cx, cy), zorder=9,
                    arrowprops=dict(arrowstyle='-|>', color='#12263f', lw=1.7))
        dx, dy = (-1.2 if cx > 0 else 1.2), (1.1 if cy < 0 else -1.1)
        ax.text(cx + dx, cy + dy, name, ha='center', va='center', fontsize=11,
                weight='bold', color='#12263f', zorder=10,
                bbox=dict(boxstyle='round,pad=.24', fc='white', ec='#12263f', lw=1))
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Roles assigned INSIDE each {cell:g} m patch\n'
                 'every operating region feeds every role', fontsize=12.5, weight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.06), ncol=2, fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(len(order)), [npos[k] for k in order], color=[COLOUR[k] for k in order])
    for i, k in enumerate(order):
        ax.text(i, npos[k], f'{npos[k]}\n{100*npos[k]/len(pos_xy):.0f}%',
                ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel('reference positions')
    ax.set_ylim(0, max(npos.values()) * 1.3)
    ax.set_title('Positions per role', fontsize=11.5)
    ax.grid(alpha=.2, axis='y')

    ax = fig.add_subplot(gs[0, 2])
    tot_hard = len(hard)
    share = [100 * sum(1 for p in use if role_of[p] == k and p in hard) / max(tot_hard, 1)
             for k in ROLES]
    ax.bar(range(3), share, color=[COLOUR[k] for k in ROLES])
    for i, k in enumerate(ROLES):
        ax.axhline(TARGET[i][1] * 100, color=COLOUR[k], ls=':', lw=1.2)
        ax.text(i, share[i], f'{share[i]:.0f}%', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels(ROLES, fontsize=9)
    ax.set_ylabel(f'% of the {tot_hard} hard positions')
    ax.set_title('Hard ground (0–1 cameras)\ndotted = quota', fontsize=11.5)
    ax.grid(alpha=.2, axis='y')

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(d_R, bins=np.arange(0, 3.1, .1), color='#1b6ca8', alpha=.85, label='D_R')
    ax.hist(d_dev, bins=np.arange(0, 3.1, .1), color='#e8a33d', alpha=.7, label='D_dev')
    ax.axvline(np.median(d_R), color='#0d3b5c', lw=2,
               label=f'median {np.median(d_R):.2f} m')
    ax.set_xlabel('distance to the nearest D_mu position (m)')
    ax.set_ylabel('positions')
    ax.set_title('Independence is LOCAL, by design\n'
                 'near enough to describe the same region', fontsize=11.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)

    ax = fig.add_subplot(gs[1, 2])
    cams = sorted({r['camera_id'] for r in rows})
    w = .26
    for j, role in enumerate(ROLES):
        sh = []
        for c in cams:
            n = sum(1 for r in rows if r['camera_id'] == c
                    and role_of[r['global_position_id']] == role)
            t = sum(1 for r in rows if r['camera_id'] == c)
            sh.append(100 * n / t)
        ax.bar([i + (j - 1) * w for i in range(len(cams))], sh, w,
               color=COLOUR[role], label=role)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([c.replace('camera_', '') for c in cams])
    ax.set_xlabel('camera')
    ax.set_ylabel("% of that camera's opportunities")
    ax.set_title('Camera balance', fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.2, axis='y')

    patched = sum(1 for v in patch.values() if len(v) >= 3)
    fig.suptitle('Local-stratum commissioning split — fixed installation, '
                 f'{len(pos_xy)} positions, {len(rows)} camera opportunities',
                 fontsize=15.5, weight='bold', y=.955)
    fig.text(.5, .026,
             f'{len(patch)} patches of {cell:g} m; {patched} ({100*patched/len(patch):.0f}%) hold three or more positions.  '
             'Atomic unit is the reference POSITION: every camera and heading at a position stays in one role.',
             ha='center', fontsize=10.5, color='#333')
    fig.savefig(OUT / 'local_split.png', dpi=args.dpi, bbox_inches='tight')
    print('written: local_split.png')

    print()
    print(f'{len(patch)} patches of {cell:g} m, {patched} with >=3 positions')
    for k in order:
        h = sum(1 for p in use if role_of[p] == k and p in hard)
        print(f'  {k:12s} {npos[k]:4d} positions ({100*npos[k]/len(pos_xy):4.1f}%)  '
              f'{opp[k]:6d} opportunities  hard {h:3d} ({100*h/max(tot_hard,1):4.1f}%)')
    print(f'\n  D_R   nearest D_mu: median {np.median(d_R):.2f} m, p5 {np.percentile(d_R,5):.2f}, p95 {np.percentile(d_R,95):.2f}')
    print(f'  D_dev nearest D_mu: median {np.median(d_dev):.2f} m')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
