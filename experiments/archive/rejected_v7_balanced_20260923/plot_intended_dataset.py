#!/usr/bin/env python3
"""Intended reference-position dataset: thin the oversampled area, add a small camera C supplement.

Two rules, both derived rather than tuned:

* Density cap. R2 uses the K nearest positions with a Gaussian length scale l_R. For the
  K neighbours to lie within 2 l_R, the density must be K / (pi (2 l_R)^2). With K=16 and
  l_R=0.4 m this is ~8 positions per m^2. Cells holding more than the cap are thinned to
  it by greedy max-min spacing, so the kept positions stay spread over the cell. Cells
  at or below the cap, and every cell inside the camera C region, are not touched.
* Camera C region. The range bins where camera C admits at least half of its peak
  admission rate, measured on the fitting opportunities. Only cells inside that radius
  that already hold a captured position are topped up to the cap times the fraction of
  the cell where the robot footprint fits, at points where the
  0.80 x 0.55 m footprint clears the collision scene by the capture body clearance at
  every heading. Nothing is added elsewhere.

Recapture candidates are proposals. They must pass the capture pose-validity filter
(`--plan-only`, read filter_counts) before any run.

    python3 experiments/warehouse_v2_sketches/plot_intended_dataset.py
"""
from __future__ import annotations

import collections
import csv
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'src' / 'unav_common'))
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from unav_common.occlusion_geometry import parse_collision_scene_from_world  # noqa: E402
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402

V5 = REPO / 'logs/thesis_final_pipeline_v1/recapture_v5'
OUT_PNG = HERE.parent / 'intended_dataset.png'
OUT_JSON = HERE.parent / 'intended_dataset.json'

K, ELL = 16, 0.4
CAP = K / (math.pi * (2 * ELL) ** 2)          # positions per m^2
CELL = 1.0
CAMS = {'A': (-11.45, -9.45), 'B': (-1.5, -9.72), 'C': (-6.95, 9.45),
        'D': (11.45, 7.2), 'E': (11.45, -9.45)}
RANGE_BINS = [0, 3, 6, 9, 12, 15, 20, 30]
WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
BODY_CLEARANCE = 0.0499
HEADINGS = np.linspace(0.0, math.pi, 8, endpoint=False)


def load_positions():
    rows = list(csv.DictReader(open(V5 / 'capture_positions_v5.csv')))
    return {r['position_id']: (float(r['x']), float(r['y']), r['role'], r['kind']) for r in rows}


def camera_c_radius(pos):
    opp, adm = collections.Counter(), collections.Counter()
    for line in open(V5 / 'gate_dataset/opportunities.jsonl'):
        r = json.loads(line)
        if r['camera_id'] != 'camera_C' or r['position_key'] not in pos:
            continue
        opp[r['position_key']] += 1
        adm[r['position_key']] += r['opportunity_outcome'] == 'admitted'
    cx, cy = CAMS['C']
    rng = np.array([math.hypot(pos[k][0] - cx, pos[k][1] - cy) for k in opp])
    rate = np.array([adm[k] / opp[k] for k in opp])
    per_bin = []
    for lo, hi in zip(RANGE_BINS, RANGE_BINS[1:]):
        m = (rng >= lo) & (rng < hi)
        per_bin.append(float(rate[m].mean()) if m.any() else 0.0)
    peak = max(per_bin)
    radius = 0.0
    for (lo, hi), r in zip(zip(RANGE_BINS, RANGE_BINS[1:]), per_bin):
        if r < 0.5 * peak:
            break
        radius = hi
    return radius, per_bin, sum(1 for k in adm if adm[k] > 0)


def maximin(points, n):
    pts = np.asarray(points)
    centre = pts.mean(axis=0)
    chosen = [int(np.argmin(((pts - centre) ** 2).sum(1)))]
    dist = ((pts - pts[chosen[0]]) ** 2).sum(1)
    while len(chosen) < n:
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, ((pts - pts[i]) ** 2).sum(1))
    return chosen


def main() -> int:
    pos = load_positions()
    radius, c_rate, c_admitted_positions = camera_c_radius(pos)
    cap_n = int(round(CAP * CELL ** 2))

    scene = parse_collision_scene_from_world(
        WORLD, model_names=('warehouse_shell', 'warehouse_v2_occluders'), robot_z_range=(0.0, 0.55))
    footprint = RectangularFootprint(tuple(scene.prisms), length=0.80, width=0.55)

    def fits(x, y):
        return all(footprint.clearance((x, y, float(h))) >= BODY_CLEARANCE for h in HEADINGS)


    cells = collections.defaultdict(list)
    for pid, (x, y, _, _) in pos.items():
        cells[(math.floor(x / CELL), math.floor(y / CELL))].append(pid)

    sub = np.arange(0.05, 1.0, 0.1)

    def target(cell):
        i, j = cell
        free = np.mean([fits(i + u, j + v) for u in sub for v in sub])
        return int(round(CAP * CELL ** 2 * free))

    targets = {c: target(c) for c in cells}
    keep, remove = [], []
    ccx, ccy = CAMS['C']
    for cell, members in cells.items():
        n = cap_n
        in_c = math.hypot(cell[0] + 0.5 * CELL - ccx, cell[1] + 0.5 * CELL - ccy) <= radius
        if len(members) <= n or in_c:
            keep += members
            continue
        pts = [pos[p][:2] for p in members]
        idx = set(maximin(pts, n))
        for i, p in enumerate(members):
            (keep if i in idx else remove).append(p)

    cx, cy = CAMS['C']
    keep_set = set(keep)
    kept_xy = np.array([pos[p][:2] for p in keep])
    add = []
    pitch = 1.0 / math.sqrt(cap_n)
    for i in range(math.floor(cx - radius), math.ceil(cx + radius)):
        for j in range(math.floor(cy - radius), math.ceil(cy + radius)):
            if (i, j) not in cells:
                continue
            have = len([p for p in cells[(i, j)] if p in keep_set])
            need = targets[(i, j)] - have
            if need <= 0:
                continue
            cand = []
            for gx in np.arange(i + pitch / 2, i + 1, pitch):
                for gy in np.arange(j + pitch / 2, j + 1, pitch):
                    if math.hypot(gx - cx, gy - cy) > radius or not fits(gx, gy):
                        continue
                    d = np.sqrt(((kept_xy - (gx, gy)) ** 2).sum(1)).min()
                    cand.append((d, gx, gy))
            cand.sort(reverse=True)
            for d, gx, gy in cand[:need]:
                if d >= 0.5 * pitch:
                    add.append((round(gx, 3), round(gy, 3)))

    def density(points):
        pts = np.asarray(points) if len(points) else np.zeros((0, 2))
        h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1],
                                 bins=[np.arange(-11, 12, CELL), np.arange(-9, 10, CELL)])
        return np.ma.masked_equal(h, 0)

    before = [pos[p][:2] for p in pos]
    after = [pos[p][:2] for p in keep] + add
    south = lambda pts: sum(1 for x, y in pts if y < -5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    vmax = max(density(before).max(), cap_n)
    for ax, title in zip(axes, ['Current v5 dataset', 'Intended changes', 'Intended dataset']):
        for pr in scene.prisms:
            if 'wall' in pr.name:
                continue
            ax.add_patch(Rectangle((pr.xmin, pr.ymin), pr.xmax - pr.xmin, pr.ymax - pr.ymin,
                                   fc='#d9c9a8', ec='#8a6d3b', lw=0.5, zorder=0))
        for c, (x, y) in CAMS.items():
            ax.plot(x, y, 's', color='#1f6fb8', ms=8)
            ax.annotate(c, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=10,
                        color='#1f6fb8', weight='bold')
        ax.add_patch(Circle(CAMS['C'], radius, fill=False, ls='--', color='#c0392b'))
        ax.set_aspect('equal'); ax.set_xlim(-12, 12); ax.set_ylim(-10, 10)
        ax.set_title(title, fontsize=12)

    im = axes[0].imshow(density(before).T, origin='lower', extent=[-11, 11, -9, 9],
                        cmap='viridis', vmin=0, vmax=vmax, alpha=0.9)
    axes[2].imshow(density(after).T, origin='lower', extent=[-11, 11, -9, 9],
                   cmap='viridis', vmin=0, vmax=vmax, alpha=0.9)
    fig.colorbar(im, ax=[axes[0], axes[2]], shrink=0.8, label='positions per m$^2$')

    k = np.array([pos[p][:2] for p in keep]); r = np.array([pos[p][:2] for p in remove])
    axes[1].scatter(k[:, 0], k[:, 1], s=3, c='#7f8c8d', label=f'keep ({len(keep)})')
    if len(r):
        axes[1].scatter(r[:, 0], r[:, 1], s=4, c='#e74c3c', marker='x', lw=0.6,
                        label=f'remove ({len(remove)})')
    if add:
        a = np.array(add)
        axes[1].scatter(a[:, 0], a[:, 1], s=14, c='#27ae60', label=f'recapture ({len(add)})')
    axes[1].legend(loc='lower right', fontsize=9, framealpha=0.9)

    axes[0].set_xlabel(f'{len(before)} positions, {south(before)} at y < -5')
    axes[2].set_xlabel(f'{len(after)} positions, {south(after)} at y < -5')
    axes[1].set_xlabel(f'cap {CAP:.1f}/m$^2$ from K={K}, 2$\\ell_R$={2*ELL} m;  '
                       f'camera C radius {radius:.0f} m (dashed)')
    fig.savefig(OUT_PNG, dpi=150)

    roles_removed = collections.Counter(pos[p][2] for p in remove)
    summary = {
        'cap_per_m2': CAP, 'cap_per_cell': cap_n, 'cell_m': CELL,
        'camera_c_radius_m': radius, 'camera_c_admission_by_range': dict(
            zip([f'{a}-{b}' for a, b in zip(RANGE_BINS, RANGE_BINS[1:])], c_rate)),
        'camera_c_admitted_positions': c_admitted_positions,
        'positions_before': len(before), 'positions_after': len(after),
        'south_before': south(before), 'south_after': south(after),
        'remove': sorted(remove), 'remove_by_role': dict(roles_removed),
        'recapture_xy': add,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('remove', 'recapture_xy')}, indent=1))
    print('recapture', len(add), '->', OUT_PNG)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
