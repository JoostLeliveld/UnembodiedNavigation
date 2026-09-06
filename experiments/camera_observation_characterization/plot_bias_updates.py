#!/usr/bin/env python3
"""Reader-first figures for the five-rung box-interpretation ladder.

Every figure here is scored on held-out warehouse tiles only, and every method is scored on
exactly the same rows, so the comparison is like-for-like.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / 'logs/studies/camera_observation_characterization_20260831'
for rel in ('experiments/deck_figures',):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
from _fieldscale import drawn_shrink, panel_scale, scale_bar  # noqa: E402

LADDER = (
    ('raw', 'Raw box → floor', 'no correction', '#eb6834'),
    ('fixed', 'Fixed 30.9 cm shift', 'one constant, everywhere', '#2a78d6'),
    ('learned', 'Learned linear update', 'per camera, from the box alone', '#4a3aa7'),
    ('nn', 'Neural bias update', 'one network, from the box alone', '#d4267b'),
    ('hull', 'Hull residual around reference', 'needs the offline reference pose', '#1baf7a'),
)
CAMERAS = tuple(f'camera_{letter}' for letter in 'ABCDE')
SHORT = ('Raw box', 'Fixed 30.9 cm', 'Learned linear', 'Neural net', 'Hull reference')
SHORT_TWO_LINE = tuple(name.replace(' ', '\n', 1) for name in SHORT)
ERROR_FOLDER = '02_the_error'      # what the reading error is
FIXES_FOLDER = '05_the_fixes'      # what each correction does about it


def camera_title(camera_id: str) -> str:
    return f'Camera {camera_id[-1]}'


def floats(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows if row[field] not in ('', 'nan')])


def valid(rows: list[dict[str, str]], method: str, field: str) -> np.ndarray:
    return floats([row for row in rows if row[f'{method}_valid'] == '1'], f'{method}_{field}')


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def median_proxy() -> Line2D:
    return Line2D([0], [0], color=D.MUTED, lw=1.4,
                  label='thin vertical line = that method’s median')


# ---------------------------------------------------------------------------------------
# 06  signed components, the figure the ladder is built to be read in
# ---------------------------------------------------------------------------------------
def draw_components(rows: list[dict[str, str]], out: Path, holdout: dict) -> None:
    spread = np.asarray([
        abs(float(row[f'{method}_{field}']))
        for row in rows for method, _l, _s, _c in LADDER for field in ('along_m', 'across_m')
        if row[f'{method}_valid'] == '1'
    ])
    limit = max(0.5, float(np.quantile(spread, 0.99))) if spread.size else 1.0
    bins = np.linspace(-limit, limit, 49)

    fig, axes = plt.subplots(5, 2, figsize=(16.5, 18.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for row_axes, camera in zip(axes, CAMERAS):
        camera_rows = [row for row in rows if row['camera_id'] == camera]
        for ax, field in zip(row_axes, ('along_m', 'across_m')):
            legend_lines = []
            for index, (method, label, _sub, colour) in enumerate(LADDER):
                values = valid(camera_rows, method, field)
                if not values.size:
                    continue
                clipped = np.clip(values, np.nextafter(-limit, 0.0), np.nextafter(limit, 0.0))
                baseline = method in ('raw', 'fixed')
                ax.hist(clipped, bins=bins, histtype='step',
                        lw=1.9 if baseline else 2.4, alpha=0.85 if baseline else 1.0,
                        color=colour, zorder=3 if baseline else 4)
                ax.axvline(float(np.median(values)), color=colour, lw=1.4, alpha=0.75, zorder=2)
                legend_lines.append((colour, f'{SHORT[index]}  {100 * np.median(values):+.1f} cm'))
            ax.axvline(0.0, color=D.INK, lw=1.2, linestyle='--', zorder=5)
            step = 0.052
            height = step * (len(legend_lines) + 1) + 0.020
            ax.add_patch(Rectangle((0.762, 0.980 - height), 0.232, height,
                                   transform=ax.transAxes, fc='white', ec='#d0cec7', lw=0.9,
                                   alpha=0.95, zorder=5.5))
            ax.text(0.878, 0.972, 'median', transform=ax.transAxes, ha='center', va='top',
                    fontsize=8.6, color=D.MUTED, zorder=6)
            for index, (colour, text) in enumerate(legend_lines):
                ax.text(0.982, 0.972 - step * (index + 1), text, transform=ax.transAxes,
                        ha='right', va='top', fontsize=9.0, color=colour, fontweight='bold',
                        zorder=6)
        row_axes[0].set_ylabel(f'{camera_title(camera)}\ncount')

    axes[0, 0].set_title('Along the camera ray\nnegative = reading falls toward the camera',
                         fontsize=14.5, fontweight='bold')
    axes[0, 1].set_title('Across the camera ray\npositive = left of the ray',
                         fontsize=14.5, fontweight='bold')
    handles = [Line2D([0], [0], color=colour, lw=2.4, label=f'{label} — {sub}')
               for _m, label, sub, colour in LADDER] + [median_proxy()]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11.5, frameon=True,
               bbox_to_anchor=(0.5, -0.035))
    for ax in axes[-1]:
        ax.set_xlabel('Signed position error (m)')
    fig.suptitle(
        'Which direction is wrong, and what removes it?\n'
        'A constant radial shift leaves lateral error intact; a learned update reaches both directions\n'
        f'Held-out warehouse tiles only ({holdout["test"]} readings); all YOLO returns retained, no admission gate',
        fontsize=19, fontweight='bold')
    save(fig, out / ERROR_FOLDER / '03_signed_error_components.png')


# ---------------------------------------------------------------------------------------
# 07  the same medians as one uncluttered summary
# ---------------------------------------------------------------------------------------
def draw_ladder_summary(rows: list[dict[str, str]], out: Path, holdout: dict) -> None:
    spread = np.asarray([
        abs(float(row[f'{method}_{field}']))
        for row in rows for method, _l, _s, _c in LADDER for field in ('along_m', 'across_m')
        if row[f'{method}_valid'] == '1'
    ])
    limit = max(0.3, float(np.quantile(spread, 0.93))) if spread.size else 0.5
    fig, axes = plt.subplots(5, 2, figsize=(14.0, 15.0), sharex=True, constrained_layout=True)
    order = list(range(len(LADDER)))[::-1]
    for row_axes, camera in zip(axes, CAMERAS):
        camera_rows = [row for row in rows if row['camera_id'] == camera]
        for ax, field, name in zip(row_axes, ('along_m', 'across_m'),
                                   ('along the ray', 'across the ray')):
            for slot, (method, label, _sub, colour) in zip(order, LADDER):
                values = valid(camera_rows, method, field)
                if not values.size:
                    continue
                low10, low25, mid, high75, high90 = np.quantile(
                    values, [0.10, 0.25, 0.50, 0.75, 0.90])
                ax.plot([low10, high90], [slot, slot], color=colour, lw=1.6, alpha=0.5,
                        solid_capstyle='butt')
                ax.plot([low25, high75], [slot, slot], color=colour, lw=6.5, alpha=0.85,
                        solid_capstyle='butt')
                ax.plot([mid], [slot], marker='o', ms=9, color='white', mec=colour, mew=2.4,
                        zorder=5)
            ax.axvline(0.0, color=D.INK, lw=1.2, linestyle='--')
            ax.set_yticks(order)
            ax.set_yticklabels([label for _m, label, _s, _c in LADDER] if ax is row_axes[0]
                               else ['' for _ in LADDER], fontsize=10)
            ax.set_ylim(-0.7, len(LADDER) - 0.3)
            ax.grid(axis='x', color='#e4e2dc', lw=0.8)
            ax.set_axisbelow(True)
            ax.set_xlim(-limit, limit)
            if camera == CAMERAS[0]:
                ax.set_title(f'Signed error {name}', fontsize=14.5, fontweight='bold')
        row_axes[0].text(-0.34, 0.5, camera_title(camera), transform=row_axes[0].transAxes,
                         rotation=90, va='center', ha='center', fontsize=13, fontweight='bold')
    for ax in axes[-1]:
        ax.set_xlabel('Signed position error (m)')
    handles = [
        Line2D([0], [0], color=D.MUTED, marker='o', ms=9, mfc='white', mec=D.MUTED, mew=2.4,
               lw=0, label='circle = median'),
        Line2D([0], [0], color=D.MUTED, lw=6.5, alpha=0.85, label='thick bar = middle half (25–75%)'),
        Line2D([0], [0], color=D.MUTED, lw=1.6, alpha=0.5, label='thin bar = 10–90%'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11.5, frameon=True,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('The same five readings, read as a ladder\n'
                 'A method is unbiased when its circle sits on the dashed zero line, '
                 'and precise when its bars are short\n'
                 f'Held-out warehouse tiles only ({holdout["test"]} readings)',
                 fontsize=19, fontweight='bold')
    save(fig, out / FIXES_FOLDER / '12_fix_ladder_summary.png')


# ---------------------------------------------------------------------------------------
# 08  distance error, five rungs
# ---------------------------------------------------------------------------------------
def draw_magnitudes(rows: list[dict[str, str]], out: Path, holdout: dict) -> None:
    pooled = np.concatenate([valid(rows, method, 'error_m') for method, _l, _s, _c in LADDER])
    upper = max(0.5, float(np.quantile(pooled, 0.99))) if pooled.size else 1.0
    bins = np.linspace(0, upper, 41)
    fig, axes = plt.subplots(5, 5, figsize=(19.0, 15.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for row_axes, camera in zip(axes, CAMERAS):
        camera_rows = [row for row in rows if row['camera_id'] == camera]
        for ax, (method, label, sub, colour) in zip(row_axes, LADDER):
            values = valid(camera_rows, method, 'error_m')
            if values.size:
                shown = np.minimum(values, np.nextafter(upper, 0.0))
                ax.hist(shown, bins=bins, color=colour, alpha=0.9, edgecolor='white', lw=0.35)
                median = float(np.median(values))
                ax.axvline(min(median, upper), color=D.INK, lw=1.6, linestyle='--')
                ax.text(0.96, 0.93,
                        f'median {100 * median:.1f} cm\n90% below {100 * np.quantile(values, 0.9):.0f} cm\n'
                        f'≥{upper:.2f} m: {int(np.sum(values >= upper))}',
                        transform=ax.transAxes, ha='right', va='top', fontsize=9.3)
            if camera == CAMERAS[0]:
                ax.set_title(f'{label}\n{sub}', fontsize=12.5, fontweight='bold')
        row_axes[0].set_ylabel(f'{camera_title(camera)}\ncount')
    for ax in axes[-1]:
        ax.set_xlabel('Distance from truth (m)')
    fig.suptitle('How far from truth does each interpretation of the same box land?\n'
                 'Shared bins and axes; the dashed line is the median; the last bin holds everything beyond it\n'
                 f'Held-out warehouse tiles only ({holdout["test"]} readings)',
                 fontsize=19, fontweight='bold')
    save(fig, out / ERROR_FOLDER / '04_error_magnitude_histograms.png')


# ---------------------------------------------------------------------------------------
# 09  what the learned rungs cost when they meet unseen floor
# ---------------------------------------------------------------------------------------
def draw_generalisation(rows: list[dict[str, str]], out: Path, holdout: dict) -> None:
    stats = ('median_m', 'p90_m', 'rmse_m')
    titles = ('Median error', '90th-percentile error', 'RMS error')
    table = {}
    for portion in ('train', 'test'):
        subset = [row for row in rows if row['split'] == portion]
        for method, _l, _s, _c in LADDER:
            values = valid(subset, method, 'error_m')
            table[(portion, method)] = {
                'median_m': float(np.median(values)),
                'p90_m': float(np.quantile(values, 0.90)),
                'rmse_m': float(np.sqrt(np.mean(values ** 2))),
            }
    x = np.arange(len(LADDER))
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.6), constrained_layout=True)
    for ax, stat, title in zip(axes, stats, titles):
        train = [100 * table[('train', m)][stat] for m, _l, _s, _c in LADDER]
        test = [100 * table[('test', m)][stat] for m, _l, _s, _c in LADDER]
        for index in x:
            ax.plot([index, index], [train[index], test[index]], color='#c9c7c0', lw=2.2, zorder=1)
        ax.plot(x, train, marker='o', ms=11, lw=0, color='white', mec=D.MUTED, mew=2.4,
                label='tiles it was fitted on', zorder=3)
        ax.plot(x, test, marker='o', ms=11, lw=0, color=D.INK, label='held-out tiles', zorder=3)
        for index in x:
            ax.annotate(f'{test[index]:.1f}', (index, test[index]), textcoords='offset points',
                        xytext=(0, 13), ha='center', fontsize=10.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_TWO_LINE, fontsize=10.5)
        for tick, (_m, _l, _s, colour) in zip(ax.get_xticklabels(), LADDER):
            tick.set_color(colour)
            tick.set_fontweight('bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylabel('cm')
        ax.grid(axis='y', color='#e4e2dc', lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(train + test) * 1.18)
    axes[0].legend(loc='upper right', fontsize=11, frameon=True)
    fig.suptitle('Does the correction survive floor it never saw?\n'
                 'Only the two learned rungs can move between the circles; raw, fixed and hull fit nothing\n'
                 f'{holdout["train"]} readings on fitted tiles, {holdout["test"]} on held-out tiles, '
                 f'{holdout["tile_m"]:.0f} m checkerboard',
                 fontsize=19, fontweight='bold')
    save(fig, out / FIXES_FOLDER / '13_generalisation_to_unseen_tiles.png')


# ---------------------------------------------------------------------------------------
# 10  where on the floor each rung still fails
# ---------------------------------------------------------------------------------------
def aggregate_positions(rows: list[dict[str, str]], method: str):
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row['position_id']].append(row)
    result = []
    for group in groups.values():
        first = group[0]
        hits = [row for row in group if row[f'{method}_valid'] == '1']
        if not hits:
            result.append((float(first['robot_x']), float(first['robot_y']),
                           math.nan, math.nan, math.nan))
            continue
        result.append((
            float(first['robot_x']), float(first['robot_y']),
            float(np.median(floats(hits, f'{method}_dx'))),
            float(np.median(floats(hits, f'{method}_dy'))),
            float(np.median(floats(hits, f'{method}_error_m'))),
        ))
    return result


def draw_error_fields(rows: list[dict[str, str]], out: Path, holdout: dict) -> None:
    """One warehouse sheet per camera, each rung on its own arrow and colour scale.

    A single shared scale is unusable here: the raw rung is ~30 cm and the neural rung ~3 cm,
    so a gain that suits raw renders the corrections as dots. Each panel therefore states its
    own gain and cap and carries a scale bar, and magnitudes are compared from the printed
    numbers, never from arrow length across panels.
    """
    for camera in CAMERAS:
        camera_rows = [row for row in rows if row['camera_id'] == camera]
        fig, axes = plt.subplots(2, 3, figsize=(17.5, 12.6), constrained_layout=True)
        for ax, (method, label, sub, _colour) in zip(axes.flat, LADDER):
            agg = aggregate_positions(camera_rows, method)
            good = [item for item in agg if math.isfinite(item[2])]
            blank = [item for item in agg if not math.isfinite(item[2])]
            clipped = 0
            scale = panel_scale(np.asarray([item[4] for item in good]))
            D.draw_warehouse(ax, D.layout(), show_cameras=True, camera_labels=True,
                             rack_alpha=0.72)
            if blank:
                ax.scatter([i[0] for i in blank], [i[1] for i in blank], marker='x', s=11,
                           c='#aaa9a4', linewidths=0.7, zorder=3)
            if good:
                gx = np.asarray([i[0] for i in good]); gy = np.asarray([i[1] for i in good])
                dx = np.asarray([i[2] for i in good]); dy = np.asarray([i[3] for i in good])
                shrink, clipped = drawn_shrink(dx, dy, scale['gain'])
                scalar = ax.quiver(gx, gy, scale['gain'] * dx * shrink,
                                   scale['gain'] * dy * shrink,
                                   np.asarray([i[4] for i in good]), cmap='magma',
                                   norm=Normalize(0, scale['cap']), angles='xy',
                                   scale_units='xy', scale=1, width=0.0045, headwidth=3.5,
                                   headlength=4.2, zorder=5)
                bar = fig.colorbar(scalar, ax=ax, fraction=0.040, pad=0.015, extend='max')
                bar.set_label(f'median error at this position (m), capped {scale["cap"]:.2f}',
                              fontsize=8.6)
                bar.ax.tick_params(labelsize=8)
                scale_bar(ax, scale['gain'], layout=D.layout(), colour=D.INK)
            tail = f'  ·  {clipped} clipped' if good and clipped else ''
            ax.set_title(
                f'{label}\n{sub}\n'
                f'median {100 * scale["median_m"]:.1f} cm  ·  arrows \u00d7{scale["gain"]:.0f}{tail}',
                fontsize=12.6, fontweight='bold')
        note = axes.flat[-1]
        note.axis('off')
        note.text(0.02, 0.97, 'How to read these maps', fontsize=16, fontweight='bold',
                  va='top')
        note.text(
            0.02, 0.86,
            'Each arrow sits at a true floor position and points\n'
            'to where the reading landed, taking the median over\n'
            'that position\u2019s detected headings.\n\n'
            '\u26a0 Every panel has its OWN arrow gain and its OWN\n'
            'colour cap, both printed above it, because the rungs\n'
            'differ about tenfold. Never compare arrow length or\n'
            'colour between panels \u2014 compare the printed medians,\n'
            'or use the scale bar drawn in each panel.\n\n'
            'Grey \u00d7 means no usable reading at any heading.\n\n'
            'Only held-out tiles appear: the checkerboard gaps\n'
            'are the floor the learned rungs were fitted on.',
            fontsize=12.0, va='top', linespacing=1.38)
        fig.suptitle(f'{camera_title(camera)} \u2014 where each correction still misses\n'
                     f'Held-out tiles only; each panel on its own scale',
                     fontsize=19, fontweight='bold')
        save(fig, out / FIXES_FOLDER / '14_where_each_fix_still_misses'
             / f'14_{camera}_still_misses.png')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-fields', action='store_true',
                        help='Skip the five per-camera warehouse map sheets.')
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    table = capture / 'bias_update_interpretations.csv'
    manifest_path = capture / 'bias_update_interpretations_manifest.json'
    if not table.is_file() or not manifest_path.is_file():
        raise RuntimeError('Run fit_bias_updates.py first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise RuntimeError('bias_update_interpretations manifest is not complete')
    holdout = manifest['holdout']

    every = list(csv.DictReader(table.open(encoding='utf-8')))
    fitted_and_held = [row for row in every if row['raw_valid'] == '1']
    held_out = [row for row in fitted_and_held if row['split'] == 'test']
    # The maps keep the views where nothing was detected, so blank floor stays visible.
    held_out_views = [row for row in every if row['split'] == 'test']

    out = (args.out or DEFAULT_OUT).expanduser().resolve()
    targets = [out / ERROR_FOLDER, out / FIXES_FOLDER]
    for target in targets:
        if target.exists() and not args.overwrite:
            raise RuntimeError(
                f'Output already exists: {target}; pass --overwrite to refresh figures')
        target.mkdir(parents=True, exist_ok=True)

    draw_components(held_out, out, holdout)
    draw_ladder_summary(held_out, out, holdout)
    draw_magnitudes(held_out, out, holdout)
    draw_generalisation(fitted_and_held, out, holdout)
    if not args.skip_fields:
        draw_error_fields(held_out_views, out, holdout)
    print(json.dumps({'output': [str(item) for item in targets],
                      'figures': sum(len(list(item.rglob('*.png'))) for item in targets)},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
