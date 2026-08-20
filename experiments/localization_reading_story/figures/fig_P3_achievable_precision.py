#!/usr/bin/env python3
"""P3 -- being seen is not the same as being known, rebuilt on the marked-point reading.

The achievable-precision model (experiments/achievable_precision_map) says that at a
point seen with probability p per opportunity, at rate f, with odometry accumulating
variance q per second, the sharpest belief that can be sustained there is

    sigma*(x)^2  =  floor(x)^2  +  q / (f * p_use(x))
                    ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^
                    what the       drift accumulated while waiting
                    reading's      for the next usable sighting
                    bias forbids

Its constants are imported from that study, so this is the same model, not a lookalike.
What is new is both inputs: floor(x) is now the marked point's residual bias (a few
millimetres) instead of the box bottom's 6-7 cm, and p_use(x) is measured directly from
this capture -- 280 floor cells x 12 headings, counting a heading as usable only if both
marker disks actually rendered.

  A  p_use(x): at how many of its 12 headings the robot can be read at all
  B  sigma*(x) with the marked point's floor
  C  the term that binds. With a 6-7 cm floor almost the whole floor was bias-bound and
     loitering bought nothing. With a 0.3-0.8 cm floor, nothing is bias-bound: every
     cell is limited by how often it is seen.

Run: python3 experiments/localization_reading_story/figures/fig_P3_achievable_precision.py
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import keypoint_geometry as kg  # noqa: E402
import reading_data as rd  # noqa: E402

BANDS = ((4.5, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 20.0))
MARKER_Z = kg.MARKER_Z


def precision_model():
    """The constants from experiments/achievable_precision_map, imported not copied."""
    path = (rd.REPO_ROOT / 'experiments/achievable_precision_map'
            / 'exp1_precision_vs_coverage.py')
    spec = importlib.util.spec_from_file_location('achievable_precision_map_exp1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        'q_rate': float(module.Q_RATE_M2_PER_S),
        'rate_hz': float(module.DETECTION_RATE_HZ),
        'min_p': float(module.MIN_USABLE_P),
        'odom_sigma': float(module.ODOM_SIGMA_PER_SQRT_M),
        'speed': float(module.NOMINAL_SPEED_MPS),
        'source': str(path.relative_to(rd.REPO_ROOT)),
    }


def availability_grid():
    """p_use per floor cell: fraction of the capture's 12 headings that gave a reading."""
    with (rd.DATASET / 'capture_diagnostics.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    cells: dict[tuple[float, float], list[int]] = {}
    for row in rows:
        key = (round(float(row['x']), 3), round(float(row['y']), 3))
        entry = cells.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += int(row['accepted'] == '1')
    xs = np.array(sorted({k[0] for k in cells}))
    ys = np.array(sorted({k[1] for k in cells}))
    grid = np.full((len(ys), len(xs)), np.nan)
    for (x, y), (planned, ok) in cells.items():
        grid[int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))] = ok / planned
    return xs, ys, grid, len(rows), max(c[0] for c in cells.values())


def measured_floor_by_range(reading, mask):
    """|mean error| per range band, in metres: the part repeated looks cannot remove."""
    err, rng = reading.err_cm[mask], reading.col('range_m')[mask]
    out = []
    for lo, hi in BANDS:
        sel = (rng >= lo) & (rng < hi)
        if sel.sum() < 5:
            out.append(np.nan)
            continue
        out.append(float(np.hypot(*err[sel].mean(axis=0)) / 100.0))
    return np.array(out)


def floor_field(xs, ys, cam, floors):
    """Assign each cell the floor measured at its range from the camera."""
    field = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            r = math.dist((x, y, MARKER_Z), tuple(cam))
            band = np.searchsorted([hi for _, hi in BANDS], r, side='right')
            band = min(band, len(BANDS) - 1)
            value = floors[band]
            if not np.isfinite(value):          # a band with too few readings
                value = np.nanmax(floors)
            field[i, j] = value
    return field


def sigma_star(floor_m, p_use, model):
    usable = np.maximum(p_use, model['min_p'])
    sigma = np.sqrt(floor_m ** 2 + model['q_rate'] / (model['rate_hz'] * usable))
    return np.where(p_use < model['min_p'], np.nan, sigma)


def draw_map(ax, xs, ys, values, cam, cmap, vmin, vmax, title, label):
    dx, dy = np.diff(xs).mean(), np.diff(ys).mean()
    ex = np.append(xs - dx / 2, xs[-1] + dx / 2)
    ey = np.append(ys - dy / 2, ys[-1] + dy / 2)
    mesh = ax.pcolormesh(ex, ey, np.ma.masked_invalid(values), cmap=cmap, vmin=vmin,
                         vmax=vmax, shading='flat')
    ax.plot([cam[0]], [cam[1]], marker='^', color='#1d3557', ms=12, mec='white', mew=1.1,
            clip_on=False, zorder=6)
    ax.text(cam[0] + 0.4, cam[1], 'the camera', fontsize=8.5, color='#1d3557', va='center')
    ax.set_aspect('equal')
    ax.set_xlim(ex[0] - 0.2, ex[-1] + 0.2)
    ax.set_ylim(cam[1] - 0.35, ey[-1] + 0.2)
    ax.set_xlabel('metres east')
    ax.set_title(title, loc='left', fontsize=10.5)
    ax.grid(False)
    bar = ax.figure.colorbar(mesh, ax=ax, orientation='horizontal', location='bottom',
                             fraction=0.05, pad=0.22, aspect=22)
    bar.set_label(label)
    return mesh


def main() -> None:
    rd.style()
    model = precision_model()
    cam = rd.camera().cam_pos
    kp = rd.load_reading('keypoint_retrained')
    bb = rd.load_reading('box_bottom')
    kp_mask = kg.both_rendered_mask(kp)

    xs, ys, p_use, n_rows, headings = availability_grid()
    kp_floors = measured_floor_by_range(kp, kp_mask)
    bb_floors = measured_floor_by_range(bb, np.ones(int(bb.detected.sum()), bool))
    kp_floor_field = floor_field(xs, ys, cam, kp_floors)
    sigma_kp = sigma_star(kp_floor_field, p_use, model)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.4),
                             gridspec_kw={'width_ratios': [1.0, 1.0, 1.15]})

    never = float(np.mean(p_use < model['min_p']))
    always = float(np.mean(p_use >= 0.999))
    draw_map(axes[0], xs, ys, p_use, cam, 'YlGnBu', 0.0, 1.0,
             f'A.  Where the robot can be read at all: {100 * always:.0f}% of cells at\n'
             f'every heading, {100 * never:.0f}% at none of them',
             'fraction of the 12 headings that gave a reading')
    axes[0].set_ylabel('metres north')

    draw_map(axes[1], xs, ys, 100 * sigma_kp, cam, 'magma_r', 0.0,
             float(np.nanmax(100 * sigma_kp)),
             f'B.  The sharpest belief this reading can sustain:\n'
             f'{np.nanmin(100 * sigma_kp):.1f} to {np.nanmax(100 * sigma_kp):.1f} cm, '
             f'set by availability not by the camera',
             'achievable precision, 1 sigma (cm)')

    # ------------------------------------------------------------------ C
    ax = axes[2]
    ps = np.linspace(0.03, 1.0, 400)
    drift = np.sqrt(model['q_rate'] / (model['rate_hz'] * ps))
    for floor_m, colour, name in ((float(np.nanmax(bb_floors)), rd.BOX_BOTTOM,
                                   'bottom of the box'),
                                  (float(np.nanmax(kp_floors)), rd.KEYPOINT,
                                   'marked point, retrained')):
        ax.plot(ps, 100 * np.sqrt(floor_m ** 2 + drift ** 2), color=colour, lw=2.6,
                label=f'{name}: floor {100 * floor_m:.2f} cm')
        ax.axhline(100 * floor_m, color=colour, lw=1.3, ls='--', alpha=0.8)
        ax.text(0.99, 100 * floor_m * 1.06, f'its floor, {100 * floor_m:.2f} cm',
                fontsize=8.2, color=colour, ha='right')
        cross = model['q_rate'] / (model['rate_hz'] * floor_m ** 2)
        if cross <= 1.0:
            ax.axvline(cross, color=colour, lw=1.0, ls=':')
            ax.annotate(f'past {100 * cross:.0f}% availability it is bias-bound:\n'
                        f'looking longer stops helping',
                        xy=(0.11, 100 * math.sqrt(floor_m ** 2 + model['q_rate']
                                                  / (model['rate_hz'] * 0.11))),
                        xytext=(0.26, 9.6), fontsize=8.4, color=colour, linespacing=1.35,
                        arrowprops={'arrowstyle': '->', 'color': colour, 'lw': 1.0})
    ax.plot(ps, 100 * drift, color=rd.MUTED, lw=1.8, ls='-.',
            label='drift between sightings alone')
    best = 100 * math.sqrt(float(np.nanmax(kp_floors)) ** 2
                           + model['q_rate'] / model['rate_hz'])
    ax.annotate(f'even seen at every heading this reading\nsustains only {best:.1f} cm -- '
                f'{best / (100 * float(np.nanmax(kp_floors))):.1f} times its own floor,\n'
                f'so the drift term is what binds',
                xy=(0.97, best), xytext=(0.42, 3.4), fontsize=8.4, color=rd.KEYPOINT,
                linespacing=1.35,
                arrowprops={'arrowstyle': '->', 'color': rd.KEYPOINT, 'lw': 1.0})
    ax.set_yscale('log')
    ax.set_ylim(0.6, 26)
    ax.set_xlabel('how often the robot is usably seen here (p_use)')
    ax.set_ylabel('sharpest sustainable belief, 1 sigma (cm)')
    ax.set_title('C.  What binds has changed: with the old reading almost every cell\n'
                 'was bias-bound; with this one, none is', loc='left', fontsize=10.5)
    ax.legend(loc='upper right', fontsize=8.4)
    ax.grid(True, which='both', alpha=0.22)

    fig.suptitle('Being seen is not the same as being known -- but with the marked point the '
                 'limit is availability, not the camera\'s bias',
                 fontsize=13, fontweight='bold', y=1.0)
    rd.note(fig, f'Availability from all {n_rows} planned poses of {rd.DATASET.name} '
                 f'({len(xs)} x {len(ys)} floor cells x {headings} headings, 2026-08-17); a '
                 f'heading counts as usable only if both marker disks actually rendered. Floors '
                 f'are the measured residual bias per range band of each reading '
                 f'(marked point {100 * np.nanmin(kp_floors):.2f}-{100 * np.nanmax(kp_floors):.2f} cm, '
                 f'box bottom {100 * np.nanmin(bb_floors):.2f}-{100 * np.nanmax(bb_floors):.2f} cm). '
                 f'Model constants imported from {model["source"]}: '
                 f'{model["rate_hz"]:.0f} Hz usable-detection rate, odometry '
                 f'{model["odom_sigma"]:.2f} m per sqrt(m) at {model["speed"]:.1f} m/s, cells '
                 f'below {model["min_p"]:.0%} availability left blank. The model is quoted, its '
                 f'historical floor numbers are not.', width=200)
    fig.subplots_adjust(top=0.85, bottom=0.24, left=0.05, right=0.985, wspace=0.24)
    rd.save(fig, 'P3_achievable_precision')

    print(f'p_use: mean {np.nanmean(p_use):.2f}, never-readable cells {100 * never:.0f}%, '
          f'always {100 * always:.0f}%')
    print(f'sigma* {np.nanmin(100 * sigma_kp):.2f}-{np.nanmax(100 * sigma_kp):.2f} cm')
    print(f'floors: marked point {100 * kp_floors} cm, box bottom {100 * bb_floors} cm')


if __name__ == '__main__':
    main()
