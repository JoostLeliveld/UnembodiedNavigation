#!/usr/bin/env python3
"""Admitted-residual support per role — the diagnostic that proves R can be fitted.

Camera OPPORTUNITIES are not what the covariance models see. `R` is conditional
on an admitted observation, so a role's real sample size is its count of
detector returns, not its row count. A camera can hold 1200 opportunities in a
region and still offer 150 usable residuals there.

This plots, per split and per camera, the ADMITTED counts, and breaks D_R down
over range and viewing angle so it is visible where the covariance models
actually have support.

Also shown: availability difficulty (how many cameras ever see a position) is
NOT the same as conditional localization difficulty. A 0-camera position
contributes nothing to `R` at all, while a well-seen position at long range or
near an image edge can still be hard. The two panels on the right separate them.

The split is recomputed here exactly as `plot_local_split.py` does, so the two
figures describe the same partition.

Stratifying on the residual error itself would leak the target into the split,
so every descriptor used here is an ACQUISITION descriptor — range, angle, box
size, image position — never the residual.

    python3 experiments/warehouse_v2_sketches/plot_admitted_support.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in
                ('src/unav_common', 'src/experiments',
                 'experiments/reference_controlled_commissioning_v1')]

PROFILES = REPO / 'src/experiments/config/world_profiles.yaml'
OUT = REPO / 'experiments/warehouse_v2_sketches'
RECORDS = {
    'v3': REPO / 'logs/thesis_final_pipeline_v1/stage06_detector_gate/commissioning_inference/records.jsonl',
    'extension': REPO / 'logs/studies/reference_controlled_commissioning_v1/extension_inference_20260918_v1/records.jsonl',
}
V3_POSITIONS = 400

TARGET = [('D_mu', 0.55), ('D_R', 0.28), ('D_dev', 0.17)]
COLOUR = {'D_mu': '#2f8f5b', 'D_R': '#1b6ca8', 'D_dev': '#e8a33d'}
ROLES = ['D_mu', 'D_R', 'D_dev']
CAM_XY = {'camera_A': (-11.450, -9.450), 'camera_B': (-1.500, -9.720),
          'camera_C': (-6.950, 9.450), 'camera_D': (11.450, 7.200),
          'camera_E': (11.450, -9.450)}


def local_split(pos_xy, use, hard, cell, seed):
    """Identical to plot_local_split.py: roles assigned inside each patch."""
    patch = defaultdict(list)
    for p in use:
        x, y = pos_xy[p]
        patch[(int(np.floor(x / cell)), int(np.floor(y / cell)))].append(p)
    rng = np.random.default_rng(seed)
    quota = np.array([t[1] for t in TARGET], dtype=float)
    role_of = {}
    for members in patch.values():
        members = sorted(members, key=lambda p: (p not in hard, pos_xy[p]))
        want = quota * len(members)
        got = np.zeros(3)
        for p in members:
            deficit = (want - got) / np.maximum(want, 1e-9)
            best = int(np.argmax(deficit + rng.uniform(0, 1e-6, 3)))
            role_of[p] = TARGET[best][0]
            got[best] += 1
    return role_of


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patch', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=20260918)
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import combined_master_capture as cmc

    rows = cmc.load_rows()
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
    role_of = local_split(pos_xy, use, hard, args.patch, args.seed)

    # ---- load inference records and attach the role --------------------
    recs = []
    for src, path in RECORDS.items():
        if not path.is_file():
            print(f'  note: {path} missing, skipping {src}')
            continue
        off = 0 if src == 'v3' else V3_POSITIONS
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                gid = int(r['position_id']) + off
                role = role_of.get(gid)
                if role is None:
                    continue          # final_audit, deliberately untouched
                cam = r['camera_id']
                cx, cy = CAM_XY[cam]
                bearing = math.degrees(math.atan2(float(r['robot_y']) - cy,
                                                  float(r['robot_x']) - cx))
                recs.append({
                    'role': role, 'camera': cam.replace('camera_', ''),
                    'admitted': bool(r.get('detector_return_at_0.25')),
                    'range': float(r.get('camera_range_m') or np.nan),
                    'bearing': abs(((bearing + 180) % 360) - 180),
                    'box': r.get('best_box_xyxy'),
                })
    if not recs:
        raise SystemExit('no inference records found')

    cams = sorted({r['camera'] for r in recs})
    adm = [r for r in recs if r['admitted']]
    print(f'records {len(recs)}, admitted {len(adm)} ({100*len(adm)/len(recs):.1f}%)')

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.8))
    fig.subplots_adjust(left=.055, right=.985, top=.875, bottom=.075,
                        wspace=.26, hspace=.36)

    # 1 opportunities vs admitted, per role
    ax = axes[0][0]
    opp = Counter(r['role'] for r in recs)
    ok = Counter(r['role'] for r in adm)
    w = .38
    ax.bar([i - w / 2 for i in range(3)], [opp[k] for k in ROLES], w,
           color='#c9d6df', label='opportunities')
    ax.bar([i + w / 2 for i in range(3)], [ok[k] for k in ROLES], w,
           color=[COLOUR[k] for k in ROLES], label='ADMITTED residuals')
    for i, k in enumerate(ROLES):
        ax.text(i - w / 2, opp[k], f'{opp[k]}', ha='center', va='bottom', fontsize=9)
        ax.text(i + w / 2, ok[k], f'{ok[k]}\n{100*ok[k]/max(opp[k],1):.0f}%',
                ha='center', va='bottom', fontsize=9, weight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(ROLES)
    ax.set_ylabel('count')
    ax.set_ylim(0, max(opp.values()) * 1.3)
    ax.set_title('Only ~2 in 5 opportunities yield a residual', fontsize=11.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=.2, axis='y')

    # 2 admitted per camera x role -- the table that matters for R
    ax = axes[0][1]
    w = .26
    for j, role in enumerate(ROLES):
        vals = [sum(1 for r in adm if r['camera'] == c and r['role'] == role) for c in cams]
        ax.bar([i + (j - 1) * w for i in range(len(cams))], vals, w,
               color=COLOUR[role], label=role)
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v, str(v), ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels(cams)
    ax.set_xlabel('camera')
    ax.set_ylabel('admitted residuals')
    ax.set_title(r'$N^{\rm admitted}_{i,\rm role}$ — support per camera', fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.2, axis='y')

    # 3 D_R support over range
    ax = axes[0][2]
    bins = np.arange(0, 32, 2.5)
    for role in ROLES:
        v = [r['range'] for r in adm if r['role'] == role and np.isfinite(r['range'])]
        ax.hist(v, bins=list(bins), histtype='step', lw=2, color=COLOUR[role], label=role)
    ax.set_xlabel('range to robot (m)')
    ax.set_ylabel('admitted residuals')
    ax.set_title('Support over range — where R has evidence', fontsize=11.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)

    # 4 D_R heat map: camera x range, admitted counts
    ax = axes[1][0]
    rb = [0, 5, 10, 15, 20, 30]
    grid = np.zeros((len(cams), len(rb) - 1))
    for r in adm:
        if r['role'] != 'D_R' or not np.isfinite(r['range']):
            continue
        j = np.searchsorted(rb, r['range'], 'right') - 1
        if 0 <= j < len(rb) - 1:
            grid[cams.index(r['camera']), j] += 1
    im = ax.imshow(grid, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(rb) - 1))
    ax.set_xticklabels([f'{rb[i]}–{rb[i+1]}' for i in range(len(rb) - 1)])
    ax.set_yticks(range(len(cams)))
    ax.set_yticklabels(cams)
    ax.set_xlabel('range (m)')
    ax.set_title(r'$D_R$ admitted residuals: camera $\times$ range', fontsize=11.5)
    for i in range(len(cams)):
        for j in range(len(rb) - 1):
            ax.text(j, i, int(grid[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if grid[i, j] > grid.max() * .55 else '#123')
    fig.colorbar(im, ax=ax, fraction=.04, pad=.02)

    # 5 availability difficulty vs conditional difficulty
    ax = axes[1][1]
    pos_adm = defaultdict(lambda: [0, 0])
    for src, path in RECORDS.items():
        if not path.is_file():
            continue
        off = 0 if src == 'v3' else V3_POSITIONS
        with path.open() as fh:
            for line in fh:
                rr = json.loads(line)
                gid = int(rr['position_id']) + off
                if gid not in role_of:
                    continue
                pos_adm[gid][0] += 1
                if rr.get('detector_return_at_0.25'):
                    pos_adm[gid][1] += 1
    groups = {0: [], 1: [], 2: [], 3: [], 4: []}
    for gid, (n, a) in pos_adm.items():
        k = min(ncam.get(gid, 0), 4)
        groups[k].append(100 * a / max(n, 1))
    ks = [k for k in sorted(groups) if groups[k]]
    ax.boxplot([groups[k] for k in ks], labels=[str(k) for k in ks], showfliers=False)
    ax.set_xlabel('cameras that ever see the position (availability difficulty)')
    ax.set_ylabel('% of that position\'s opportunities admitted')
    ax.set_title('Availability difficulty ≠ conditional difficulty', fontsize=11.5)
    ax.grid(alpha=.2, axis='y')

    # 6 D_R support over viewing angle
    ax = axes[1][2]
    for role in ROLES:
        v = [r['bearing'] for r in adm if r['role'] == role]
        ax.hist(v, bins=18, histtype='step', lw=2, color=COLOUR[role], label=role)
    ax.set_xlabel('|bearing| from camera (deg)')
    ax.set_ylabel('admitted residuals')
    ax.set_title(r'Support over viewing angle $\alpha_i$', fontsize=11.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)

    fig.suptitle('Admitted-residual support — what the covariance models actually see',
                 fontsize=15.5, weight='bold', y=.945)
    fig.text(.5, .026,
             'Split descriptors are ACQUISITION descriptors only (range, angle, camera). '
             'Stratifying on the residual would leak the target the covariance model predicts.',
             ha='center', fontsize=10.5, color='#333')
    fig.savefig(OUT / 'admitted_support.png', dpi=args.dpi, bbox_inches='tight')
    print('written: admitted_support.png')

    print()
    print(f"{'role':8s} {'opportunities':>14} {'admitted':>10} {'rate':>7}")
    for k in ROLES:
        print(f'  {k:6s} {opp[k]:14d} {ok[k]:10d} {100*ok[k]/max(opp[k],1):6.1f}%')
    print(f"\n{'camera':8s} " + ' '.join(f'{k:>8s}' for k in ROLES))
    for c in cams:
        print(f'  {c:6s} ' + ' '.join(
            f'{sum(1 for r in adm if r["camera"]==c and r["role"]==k):8d}' for k in ROLES))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
