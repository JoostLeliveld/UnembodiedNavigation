#!/usr/bin/env python3
"""Local D_R residual density per camera — can a spatial R(p, i) be identified?

Camera x range support says where a GEOMETRY model has evidence. It does not say
whether a SPATIAL model does, because a region can be well covered in aggregate
and still hold one or two residuals per neighbourhood. A learned R(p) that
predicts a small covariance somewhere should be checkable against how much local
evidence existed there.

So this maps, for every camera,

    N_i^R(p) = #{ admitted D_R residuals within rho of p }

on the driveable floor, and reports the fraction of covered floor that clears a
usable-support threshold.

Distinguishes the two ways a cell can be empty, which need different fixes:

  * the camera never sees that ground        -> an availability problem, q_i(p)
  * the camera sometimes sees it but the
    residual count is tiny                   -> a covariance-support problem, R

Only acquisition descriptors are used. Nothing is stratified on the residual the
covariance model is meant to predict.

    python3 experiments/warehouse_v2_sketches/plot_local_residual_support.py
    python3 experiments/warehouse_v2_sketches/plot_local_residual_support.py --rho 1.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
CAMS = ['A', 'B', 'C', 'D', 'E']
#: below this many local residuals a spatial covariance is not identifiable
USABLE = 5


def local_split(pos_xy, use, hard, cell, seed):
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
    ap.add_argument('--rho', type=float, default=2.0, help='neighbourhood radius (m)')
    ap.add_argument('--patch', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=20260918)
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import yaml
    import combined_master_capture as cmc

    rows = cmc.load_rows()
    profile = yaml.safe_load(PROFILES.read_text())['worlds']['warehouse_v2.world.sdf']
    regions = profile['known_2d_regions']
    site = next(r for r in regions if r.get('type') == 'site_boundary')
    obstacles = [r for r in regions if r.get('type') == 'non_driveable_obstacle']

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
    hard = {p for p in use if len(seen.get(p, ())) <= 1}
    role_of = local_split(pos_xy, use, hard, args.patch, args.seed)

    # admitted D_R residual locations, and "ever seen" locations, per camera
    resid = defaultdict(list)
    ever = defaultdict(list)
    for src, path in RECORDS.items():
        if not path.is_file():
            continue
        off = 0 if src == 'v3' else V3_POSITIONS
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                gid = int(r['position_id']) + off
                cam = r['camera_id'].replace('camera_', '')
                xy = (float(r['robot_x']), float(r['robot_y']))
                if r.get('detector_return_at_0.25'):
                    ever[cam].append(xy)
                    if role_of.get(gid) == 'D_R':
                        resid[cam].append(xy)

    # driveable grid
    step = 0.4
    xs = np.arange(site['xmin'], site['xmax'] + step, step)
    ys = np.arange(site['ymin'], site['ymax'] + step, step)
    X, Y = np.meshgrid(xs, ys)
    drive = np.ones(X.shape, dtype=bool)
    for r in obstacles:
        drive &= ~((X >= r['xmin']) & (X <= r['xmax'])
                   & (Y >= r['ymin']) & (Y <= r['ymax']))

    from scipy.spatial import cKDTree
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.2))
    fig.subplots_adjust(left=.045, right=.975, top=.875, bottom=.075,
                        wspace=.16, hspace=.24)
    cmap = ListedColormap(['#f2f4f5', '#fdd6cf', '#f8ad9d', '#8fc7a8', '#3f9b6d', '#1f6f4a'])
    bounds = [0, 1, 3, USABLE, 12, 30, 10_000]
    norm = BoundaryNorm(bounds, cmap.N)

    summary = {}
    for ax, cam in zip(axes.ravel(), CAMS):
        pts = np.array(resid[cam]) if resid[cam] else np.empty((0, 2))
        dens = np.zeros(X.shape)
        if len(pts):
            tree = cKDTree(pts)
            flat = np.column_stack([X.ravel(), Y.ravel()])
            dens = np.array([len(i) for i in tree.query_ball_point(flat, args.rho)]
                            ).reshape(X.shape)
        evp = np.array(ever[cam]) if ever[cam] else np.empty((0, 2))
        cover = np.zeros(X.shape, dtype=bool)
        if len(evp):
            t2 = cKDTree(evp)
            flat = np.column_stack([X.ravel(), Y.ravel()])
            cover = (np.array([len(i) for i in t2.query_ball_point(flat, args.rho)]
                              ).reshape(X.shape) > 0)
        shown = np.where(drive, dens, np.nan)
        im = ax.imshow(shown, origin='lower', cmap=cmap, norm=norm,
                       extent=(xs[0], xs[-1], ys[0], ys[-1]), interpolation='nearest')
        for r in obstacles:
            ax.add_patch(Rectangle((r['xmin'], r['ymin']), r['xmax'] - r['xmin'],
                                   r['ymax'] - r['ymin'], fc='#8d949a', ec='none',
                                   alpha=.55, zorder=3))
        covered = drive & cover
        thin = covered & (dens < USABLE)
        summary[cam] = (int(covered.sum()), int(thin.sum()), len(pts))
        ax.contour(X, Y, covered.astype(float), levels=[.5], colors='#1b3a5c',
                   linewidths=1.1, zorder=4)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'camera {cam} — {len(pts)} $D_R$ residuals\n'
                     f'{100*thin.sum()/max(covered.sum(),1):.0f}% of its covered floor '
                     f'has < {USABLE} within {args.rho:g} m',
                     fontsize=11.5)

    ax = axes.ravel()[5]
    ax.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=.5, aspect=14, pad=.02)
    cb.set_label(f'$N_i^R(p)$: $D_R$ residuals within {args.rho:g} m', fontsize=11)
    cb.set_ticks([0, 1, 3, USABLE, 12, 30])
    ax.text(.5, .17, 'dark outline = ground this camera\never admits the robot on\n\n'
                     'pale INSIDE the outline is the\ncovariance-support problem;\n'
                     'pale outside it is availability',
            ha='center', va='center', fontsize=11, color='#333', transform=ax.transAxes)

    fig.suptitle('Local $D_R$ residual density — is a spatial $R(p, i)$ identifiable?',
                 fontsize=15.5, weight='bold', y=.945)
    fig.text(.5, .028,
             f'Support thins toward each camera\'s far field. A shared R(d, alpha, i) or '
             f'R(p, i) with camera identity pools this evidence; five independent per-camera '
             f'models would not be identified.',
             ha='center', fontsize=10.5, color='#333')
    fig.savefig(OUT / 'local_residual_support.png', dpi=args.dpi, bbox_inches='tight')
    print('written: local_residual_support.png')
    print()
    print(f'{"cam":5s} {"D_R residuals":>14} {"covered cells":>14} {"thin (<%d)" % USABLE:>12} {"% thin":>8}')
    for cam in CAMS:
        cov, thin, n = summary[cam]
        print(f'  {cam:3s} {n:14d} {cov:14d} {thin:12d} {100*thin/max(cov,1):7.1f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
