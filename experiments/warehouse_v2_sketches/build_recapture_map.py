#!/usr/bin/env python3
"""The spawn map for the new master capture: where the robot stands, and in which role.

Produces the position list and the figure that together define the recapture.
Everything is derived from the CURRENT world and the CURRENT keep-out region map;
nothing is inherited from the v3 lattice, which was built against a different
world and against a traversable-lane map that no longer exists.

Design, in the order the decisions bind:

  legality      a pose is legal when the robot's oriented footprint clears every
                declared obstacle by --clearance at EVERY sampled heading. The
                body is 0.80 x 0.55 m, so a point test would admit poses the
                robot cannot occupy.

  base lattice  deterministic maximin thinning over the legal candidates. This
                is the coverage floor and reproduces the v3 selection rule.

  densified     extra positions near visibility transitions, where the admitted/
                refused boundary lives and where both f_mu and R need resolution.
                A transition cell is one whose count of admitting cameras differs
                from a neighbour's.

  micro-cluster every selected anchor also gets companions at +/- --micro metres.
                These are what make D_R independent of D_mu without leaving the
                local operating region, and they give a local residual population
                rather than one residual per point.

  roles         assigned INSIDE each --patch metre stratum, so every region feeds
                D_mu, D_R and D_dev. The atomic unit is the position: all five
                cameras and all headings at a position share one role.

Writes:
    logs/thesis_final_pipeline_v1/recapture_v5/capture_positions_v5.csv
    logs/thesis_final_pipeline_v1/recapture_v5/capture_plan_v5.json
    experiments/warehouse_v2_sketches/recapture_map.png

    python3 experiments/warehouse_v2_sketches/build_recapture_map.py
    python3 experiments/warehouse_v2_sketches/build_recapture_map.py --base 420 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in ('src/unav_common', 'src/experiments')]

WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
PROFILES = REPO / 'src/experiments/config/world_profiles.yaml'
OUTDIR = REPO / 'logs/thesis_final_pipeline_v1/recapture_v5'
FIG = REPO / 'experiments/warehouse_v2_sketches/recapture_map.png'

ROBOT_L, ROBOT_W = 0.80, 0.55
TARGET_Z = 0.35
ROLES = [('D_mu', 0.55), ('D_R', 0.28), ('D_dev', 0.17)]
COLOUR = {'D_mu': '#2f8f5b', 'D_R': '#1b6ca8', 'D_dev': '#e8a33d',
          'final_audit': '#c23d36'}


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
            rig.append({'id': labels.get(m.group(1), m.group(1)),
                        'x': v[0], 'y': v[1], 'z': v[2],
                        'pitch': v[4], 'yaw': v[5]})
    return sorted(rig, key=lambda c: c['id'])


def footprint_clear(x, y, yaw, obstacles, site, clearance):
    """Oriented 0.80 x 0.55 m body clears every obstacle and stays in the site."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = ROBOT_L / 2, ROBOT_W / 2
    corners = [(x + c * dx - s * dy, y + s * dx + c * dy)
               for dx in (-hx, hx) for dy in (-hy, hy)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    if (min(xs) < site['xmin'] + clearance or max(xs) > site['xmax'] - clearance
            or min(ys) < site['ymin'] + clearance or max(ys) > site['ymax'] - clearance):
        return False
    # axis-aligned obstacles: the body's AABB inflated by clearance must not overlap
    lo_x, hi_x = min(xs) - clearance, max(xs) + clearance
    lo_y, hi_y = min(ys) - clearance, max(ys) + clearance
    for r in obstacles:
        if lo_x < r['xmax'] and hi_x > r['xmin'] and lo_y < r['ymax'] and hi_y > r['ymin']:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spacing', type=float, default=0.4, help='candidate lattice (m)')
    ap.add_argument('--headings', type=int, default=8,
                    help='headings per position; 4 halves the capture time and, measured on '
                         'the v3/v4 data, costs q almost nothing (visibility varies 2.9 '
                         'percentage points across the 8-way comb) while retaining ~90%% of '
                         'the median angular variation R sees')
    ap.add_argument('--clearance', type=float, default=0.05,
                    help='extra body clearance beyond the declared obstacle envelope (m)')
    ap.add_argument('--base', type=int, default=420, help='base lattice positions')
    ap.add_argument('--dense', type=int, default=180,
                    help='extra positions at visibility transitions')
    ap.add_argument('--micro', type=float, default=0.15,
                    help='micro-cluster offset (m); 0 disables')
    ap.add_argument('--patch', type=float, default=2.0, help='role stratum (m)')
    ap.add_argument('--audit-frac', type=float, default=0.06)
    ap.add_argument('--seed', type=int, default=20260919)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import yaml
    from unav_common.occlusion_geometry import parse_occlusion_scene_from_world

    prof_all = yaml.safe_load(PROFILES.read_text())
    profile = prof_all['worlds']['warehouse_v2.world.sdf']
    intr = prof_all['camera_intrinsics']
    fov_h = float(intr['fov_h_rad'])
    fov_v = 2 * math.atan(math.tan(fov_h / 2) * intr['img_height'] / intr['img_width'])
    regions = profile['known_2d_regions']
    site = next(r for r in regions if r.get('type') == 'site_boundary')
    obstacles = [r for r in regions if r.get('type') == 'non_driveable_obstacle']
    rig = camera_rig(profile)

    scene = parse_occlusion_scene_from_world(
        str(WORLD), model_name=('warehouse_v2_occluders',),
        geometry_tags=('collision',), robot_z_range=None)
    prisms = list(scene.prisms)

    def sees(cam, x, y):
        rng = math.hypot(x - cam['x'], y - cam['y'])
        dep = math.degrees(math.atan2(cam['z'] - TARGET_Z, rng))
        if abs(dep - math.degrees(cam['pitch'])) > math.degrees(fov_v) / 2:
            return False
        bearing = math.degrees(math.atan2(y - cam['y'], x - cam['x']))
        if abs(((bearing - math.degrees(cam['yaw']) + 180) % 360) - 180) > math.degrees(fov_h) / 2:
            return False
        a = np.array([cam['x'], cam['y'], cam['z']])
        b = np.array([x, y, TARGET_Z])
        for t in np.linspace(0.02, 0.98, 90):
            p = a + (b - a) * t
            for pr in prisms:
                if (pr.xmin <= p[0] <= pr.xmax and pr.ymin <= p[1] <= pr.ymax
                        and pr.zmin <= p[2] <= pr.zmax):
                    return False
        return True

    # ---- legal candidates ------------------------------------------------
    HEADINGS = int(args.headings)
    yaws = [2 * math.pi * k / HEADINGS for k in range(HEADINGS)]
    xs = np.arange(site['xmin'] + 0.4, site['xmax'] - 0.4 + 1e-9, args.spacing)
    ys = np.arange(site['ymin'] + 0.4, site['ymax'] - 0.4 + 1e-9, args.spacing)
    cand = []
    for y in ys:
        for x in xs:
            if all(footprint_clear(x, y, w, obstacles, site, args.clearance) for w in yaws):
                cand.append((round(float(x), 3), round(float(y), 3)))
    print(f'legal candidates (body clears at all {HEADINGS} headings): {len(cand)}')
    if len(cand) < args.base:
        raise SystemExit(f'only {len(cand)} legal candidates for base {args.base}')

    ncam = {p: sum(1 for c in rig if sees(c, *p)) for p in cand}
    print(f'  admitting-camera count: {dict(sorted(Counter(ncam.values()).items()))}')

    # ---- base lattice: deterministic maximin ------------------------------
    pts = np.asarray(cand, dtype=float)
    first = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    chosen = [first]
    near = np.sum((pts - pts[first]) ** 2, axis=1)
    for _ in range(1, args.base):
        i = int(np.argmax(near))
        chosen.append(i)
        near = np.minimum(near, np.sum((pts - pts[i]) ** 2, axis=1))
    base = [cand[i] for i in chosen]

    # ---- densify at visibility transitions --------------------------------
    cand_set = {p: i for i, p in enumerate(cand)}
    transition = []
    for p in cand:
        n = ncam[p]
        for dx, dy in ((args.spacing, 0), (-args.spacing, 0),
                       (0, args.spacing), (0, -args.spacing)):
            q = (round(p[0] + dx, 3), round(p[1] + dy, 3))
            if q in cand_set and ncam[q] != n:
                transition.append(p)
                break
    taken = set(base)
    pool = [p for p in transition if p not in taken]
    dense = []
    if pool and args.dense:
        pp = np.asarray(pool, dtype=float)
        ref = np.asarray(base, dtype=float)
        near = np.min(((pp[:, None, :] - ref[None, :, :]) ** 2).sum(-1), axis=1)
        for _ in range(min(args.dense, len(pool))):
            i = int(np.argmax(near))
            dense.append(pool[i])
            near = np.minimum(near, ((pp - pp[i]) ** 2).sum(-1))
    print(f'  visibility-transition candidates: {len(transition)}, densified with {len(dense)}')

    anchors = base + dense

    # ---- micro-clusters ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    positions = []
    for j, (x, y) in enumerate(anchors):
        positions.append({'x': x, 'y': y, 'anchor': j, 'kind': 'anchor'})
        if args.micro <= 0:
            continue
        for dx, dy in ((args.micro, 0.0), (0.0, args.micro)):
            nx, ny = round(x + dx, 3), round(y + dy, 3)
            if all(footprint_clear(nx, ny, w, obstacles, site, args.clearance) for w in yaws):
                positions.append({'x': nx, 'y': ny, 'anchor': j, 'kind': 'micro'})
    print(f'  anchors {len(anchors)} -> positions with micro-clusters {len(positions)}')

    # ---- roles, assigned inside each patch; a whole ANCHOR GROUP shares... --
    # ...no: D_R must differ from D_mu WITHIN a micro-cluster, which is the point
    # of the cluster. So roles are per POSITION, but the patch keeps the mix local.
    patch = defaultdict(list)
    for i, p in enumerate(positions):
        patch[(int(np.floor(p['x'] / args.patch)), int(np.floor(p['y'] / args.patch)))].append(i)
    quota = np.array([r[1] for r in ROLES], dtype=float)
    for members in patch.values():
        members = sorted(members, key=lambda i: (positions[i]['anchor'], positions[i]['kind']))
        want = quota * len(members)
        got = np.zeros(3)
        for i in members:
            deficit = (want - got) / np.maximum(want, 1e-9)
            k = int(np.argmax(deficit + rng.uniform(0, 1e-6, 3)))
            positions[i]['role'] = ROLES[k][0]
            got[k] += 1

    # ---- final audit: balanced over admitting-camera regimes ---------------
    n_audit = int(round(args.audit_frac * len(positions)))
    by_reg = defaultdict(list)
    for i, p in enumerate(positions):
        by_reg[min(ncam.get((p['x'], p['y']), 0), 4)].append(i)
    audit = []
    per = max(1, n_audit // max(len(by_reg), 1))
    for reg, idxs in sorted(by_reg.items()):
        pick = rng.choice(idxs, size=min(per, len(idxs)), replace=False)
        audit.extend(int(i) for i in pick)
    for i in audit[:n_audit]:
        positions[i]['role'] = 'final_audit'

    for i, p in enumerate(positions):
        p['position_id'] = f'P{i:04d}'
        p['n_cameras'] = int(ncam.get((p['x'], p['y']), 0))
        p['headings'] = HEADINGS

    role_n = Counter(p['role'] for p in positions)
    opp = {k: v * HEADINGS * len(rig) for k, v in role_n.items()}
    total_opp = len(positions) * HEADINGS * len(rig)

    # ---- write -------------------------------------------------------------
    world_sha = hashlib.sha256(WORLD.read_bytes()).hexdigest()
    if not args.dry_run:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        with (OUTDIR / 'capture_positions_v5.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['position_id', 'x', 'y', 'role', 'kind',
                                               'anchor', 'n_cameras', 'headings'])
            w.writeheader()
            for p in positions:
                w.writerow({k: p[k] for k in w.fieldnames})
        plan = {
            'schema': 'recapture_plan.v5',
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'world': str(WORLD.relative_to(REPO)),
            'world_sha256': world_sha,
            'rule': {
                'legality': f'oriented {ROBOT_L}x{ROBOT_W} m body clears every declared '
                            f'obstacle by {args.clearance} m at all {HEADINGS} headings',
                'base_selection': 'deterministic maximin thinning',
                'densification': 'visibility-transition cells, maximin within them',
                'micro_cluster_m': args.micro,
                'role_stratum_m': args.patch,
                'atomic_unit': 'position (all cameras and headings share one role)',
            },
            'counts': {
                'legal_candidates': len(cand), 'anchors': len(anchors),
                'base': len(base), 'densified': len(dense),
                'positions': len(positions), 'headings': HEADINGS,
                'cameras': len(rig), 'camera_opportunities': total_opp,
                'by_role': dict(role_n),
            },
            'cameras': rig,
        }
        (OUTDIR / 'capture_plan_v5.json').write_text(json.dumps(plan, indent=2))
        print(f'\nwritten: {OUTDIR}/capture_positions_v5.csv')
        print(f'written: {OUTDIR}/capture_plan_v5.json')

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(19, 9.6),
                             gridspec_kw=dict(width_ratios=[1.5, 1]))
    fig.subplots_adjust(left=.045, right=.985, top=.875, bottom=.10, wspace=.16)

    ax = axes[0]
    for r in obstacles:
        ax.add_patch(Rectangle((r['xmin'], r['ymin']), r['xmax'] - r['xmin'],
                               r['ymax'] - r['ymin'], fc='#9aa0a6', ec='#6b7075',
                               lw=.5, alpha=.45, zorder=2))
    ax.add_patch(Rectangle((site['xmin'], site['ymin']),
                           site['xmax'] - site['xmin'], site['ymax'] - site['ymin'],
                           fill=False, ec='.35', lw=1.3, ls=(0, (7, 5)), zorder=6))
    for role in ['D_mu', 'D_R', 'D_dev', 'final_audit']:
        pp = [p for p in positions if p['role'] == role]
        ax.scatter([p['x'] for p in pp], [p['y'] for p in pp], s=15, c=COLOUR[role],
                   edgecolors='none', zorder=7,
                   label=f'{role}  {len(pp)} pos / {opp.get(role,0)} opp')
    dn = [p for p in positions if p['kind'] == 'anchor' and (p['x'], p['y']) in set(dense)]
    ax.scatter([p['x'] for p in dn], [p['y'] for p in dn], s=52, facecolors='none',
               edgecolors='#6a51a3', linewidths=1.0, zorder=8,
               label=f'visibility-transition densified ({len(dense)})')
    for c in rig:
        ax.plot([c['x']], [c['y']], marker='o', ms=11, mfc='#12263f', mec='white',
                mew=1.5, zorder=9)
        ax.annotate('', xy=(c['x'] + 2.4 * math.cos(c['yaw']),
                            c['y'] + 2.4 * math.sin(c['yaw'])),
                    xytext=(c['x'], c['y']), zorder=9,
                    arrowprops=dict(arrowstyle='-|>', color='#12263f', lw=1.9))
        dx, dy = (-1.25 if c['x'] > 0 else 1.25), (1.15 if c['y'] < 0 else -1.15)
        ax.text(c['x'] + dx, c['y'] + dy, c['id'], ha='center', va='center',
                fontsize=12, weight='bold', color='#12263f', zorder=10,
                bbox=dict(boxstyle='round,pad=.26', fc='white', ec='#12263f', lw=1.1))
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(alpha=.18)
    ax.set_title(f'Spawn map — {len(positions)} positions x {HEADINGS} headings x '
                 f'{len(rig)} cameras = {total_opp} opportunities',
                 fontsize=13, weight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.07), ncol=3, fontsize=10)

    ax = axes[1]
    reg = Counter(p['n_cameras'] for p in positions)
    ks = sorted(reg)
    ax.bar([str(k) for k in ks], [reg[k] for k in ks], color='#4c6ef5')
    for k in ks:
        ax.text(str(k), reg[k], str(reg[k]), ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('cameras that can admit the robot at that position')
    ax.set_ylabel('positions')
    ax.set_title('Coverage regime of the new map', fontsize=12)
    ax.grid(alpha=.2, axis='y')

    fig.suptitle(f'Master recapture map v5 — world {world_sha[:12]}…  '
                 f'(C at 48°, Cm de-stacked)', fontsize=15, weight='bold', y=.95)
    fig.text(.5, .028,
             f'legal = oriented {ROBOT_L}×{ROBOT_W} m body clears every obstacle by '
             f'{args.clearance:g} m at all {HEADINGS} headings.  micro-clusters at '
             f'{args.micro:g} m give D_R residuals independent of D_mu inside the same region.',
             ha='center', fontsize=10.5, color='#333')
    fig.savefig(FIG, dpi=150, bbox_inches='tight')
    print(f'written: {FIG}')

    print()
    print(f'{"role":14s} {"positions":>10} {"%":>6} {"opportunities":>14}')
    for k in ['D_mu', 'D_R', 'D_dev', 'final_audit']:
        print(f'  {k:12s} {role_n[k]:10d} {100*role_n[k]/len(positions):5.1f}% {opp.get(k,0):14d}')
    print(f'  {"TOTAL":12s} {len(positions):10d} {100:5.1f}% {total_opp:14d}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
