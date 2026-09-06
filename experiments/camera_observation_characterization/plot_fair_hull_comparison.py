#!/usr/bin/env python3
"""The fair comparison figure: every method starting from the same information.

Two panels.  The left one shows what each method is GIVEN before it starts, because that is
the thing the published ladder got wrong.  The right one shows how far each lands from the
truth on held-out warehouse tiles.  Methods that were handed the answer are drawn greyed and
hatched so a reader cannot mistake them for a result.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for rel in ('experiments/deck_figures',):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402

DEFAULT_OUT = REPO / 'logs/studies/camera_observation_characterization_20260831/07_fair_hull'

# method, label, what it is given, fair?, colour
LADDER = (
    ('raw', 'Raw box → floor',
     'the box and its camera', True, '#eb6834'),
    ('fixed', 'One fixed 30.9 cm shift',
     'the box and its camera', True, '#2a78d6'),
    ('learned', 'Learned linear correction',
     'the box and its camera', True, '#4a3aa7'),
    ('nn', 'Neural correction',
     'the box and its camera', True, '#d4267b'),
    ('hull_operational', 'Robot-shape model, heading solved',
     'the box and its camera', True, '#1baf7a'),
    ('hull_heading', 'Robot-shape model, TRUE heading given',
     'the box, its camera, and the true heading', False, '#9a9a94'),
    ('hull', 'Published “hull” rung: started AT the answer',
     'the true position and the true heading', False, '#9a9a94'),
)


def load(capture: Path):
    table = capture / 'fair_hull_interpretations.csv'
    manifest_path = capture / 'fair_hull_interpretations_manifest.json'
    if not table.is_file() or not manifest_path.is_file():
        raise RuntimeError('Run fair_hull_comparison.py first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = list(csv.DictReader(table.open(encoding='utf-8')))
    return rows, manifest


def errors(rows: list[dict], method: str) -> np.ndarray:
    return np.asarray([float(row[f'{method}_error_m']) * 100.0 for row in rows
                       if str(row[f'{method}_valid']) == '1'])


def draw(rows: list[dict], manifest: dict, out: Path) -> None:
    test = [row for row in rows
            if row['split'] == 'test' and row['raw_valid'] == '1'
            and str(row['hull_operational_valid']) == '1']
    data = {key: errors(test, key) for key, *_ in LADDER}
    n = len(test)

    fig = plt.figure(figsize=(15.4, 8.9))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.45), wspace=0.06)
    ax_info = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])

    positions = np.arange(len(LADDER))[::-1]

    # ---- left: what each method was allowed to know ------------------------------------
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(-1.15, len(LADDER) - 0.3)
    ax_info.axis('off')
    ax_info.text(0.0, len(LADDER) - 0.45, 'What the method is given before it starts',
                 fontsize=12.5, fontweight='bold', color=D.INK)
    for (key, label, given, fair, colour), y in zip(LADDER, positions):
        mark = '●' if fair else '✕'
        mark_colour = colour if fair else '#c0392b'
        ax_info.text(0.0, y + 0.16, mark, fontsize=13, color=mark_colour,
                     va='center', ha='left')
        ax_info.text(0.055, y + 0.16, label, fontsize=11.5, va='center', ha='left',
                     color=D.INK if fair else '#7a7a74',
                     fontweight='bold' if fair else 'normal')
        ax_info.text(0.055, y - 0.19, given, fontsize=10.2, va='center', ha='left',
                     color=D.INK2 if fair else '#c0392b',
                     style='normal' if fair else 'italic')
    ax_info.axhline(1.5, xmin=0.0, xmax=0.98, color='#d5d4cf', lw=1.0)
    ax_info.text(0.0, -0.80,
                 'The two greyed rows are given something the robot never has at run time,\n'
                 'so their scores are not achievable performance.',
                 fontsize=10.2, color='#c0392b', va='center')

    # ---- right: how far from the truth --------------------------------------------------
    for (key, label, _given, fair, colour), y in zip(LADDER, positions):
        values = data[key]
        box = ax.boxplot([values], positions=[y], vert=False, widths=0.56,
                         whis=(10, 90), showfliers=False, patch_artist=True)
        face = colour if fair else '#e2e1dc'
        for patch in box['boxes']:
            patch.set_facecolor(face)
            patch.set_alpha(0.85 if fair else 1.0)
            patch.set_edgecolor(colour if fair else '#9a9a94')
            patch.set_linewidth(1.4)
            if not fair:
                patch.set_hatch('////')
        for part in ('whiskers', 'caps'):
            for line in box[part]:
                line.set_color(colour if fair else '#9a9a94')
                line.set_linewidth(1.3)
        for line in box['medians']:
            line.set_color(D.INK)
            line.set_linewidth(2.2)
        median = float(np.median(values))
        ax.text(median, y + 0.40, f'{median:.1f} cm', fontsize=10.5, ha='center',
                va='bottom', color=D.INK if fair else '#7a7a74', fontweight='bold')

    ax.set_yticks(positions)
    ax.set_yticklabels([label for _k, label, *_ in LADDER], fontsize=11)
    for tick, (_k, _l, _g, fair, _c) in zip(ax.get_yticklabels(), LADDER):
        tick.set_color(D.INK if fair else '#9a9a94')
    ax.tick_params(axis='y', length=0)
    ax.set_yticklabels([])          # the left panel already names them
    ax.set_xlabel('Distance from the true robot centre  (cm) — lower is better',
                  fontsize=12)
    ax.text(0.995, -0.088, 'box = middle half of readings · whiskers = 10th to 90th percentile',
            transform=ax.transAxes, fontsize=10, ha='right', va='top', color=D.INK2)
    ax.set_xlim(0, 55)
    ax.grid(axis='x', color='#e6e5e0', lw=0.9)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    best_fair = min((key for key, *_r in LADDER if _r[2]),
                    key=lambda key: float(np.median(data[key])))
    ax.axvline(float(np.median(data[best_fair])), color=D.INK, lw=0.9, ls=(0, (4, 3)),
               alpha=0.45, zorder=0)

    heading = manifest['solved_heading_error_deg']
    fig.suptitle(
        'Given the same starting information, the robot-shape model and the neural\n'
        'correction are level — the shape model was winning on a head start',
        fontsize=15.5, fontweight='bold', x=0.5, y=0.995, va='top')
    fig.text(0.5, 0.905,
             f'{n:,} detections on held-out warehouse tiles the corrections never saw. '
             'Every method scored on exactly the same readings.',
             fontsize=11.2, ha='center', color=D.INK2)

    fig.text(0.5, 0.022,
             'Solving the heading from the box instead of being told it costs the shape model '
             f'only {(np.median(data["hull_operational"]) - np.median(data["hull_heading"])):.1f} cm '
             f'(heading recovered to {heading["median"]:.0f}° typical, front/back not identifiable '
             'from one box).\n'
             'The shape model still has the worst tail: its 90th percentile is '
             f'{np.quantile(data["hull_operational"], 0.9):.0f} cm against '
             f'{np.quantile(data["nn"], 0.9):.0f} cm for the neural correction.',
             fontsize=10.6, ha='center', color=D.INK2)

    fig.subplots_adjust(top=0.865, bottom=0.185, left=0.012, right=0.985)
    out.mkdir(parents=True, exist_ok=True)
    path = out / '01_fair_comparison_same_starting_information.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f'wrote {path}')

    summary = {
        'n_test_rows': n,
        'median_cm': {key: float(np.median(data[key])) for key, *_ in LADDER},
        'p90_cm': {key: float(np.quantile(data[key], 0.9)) for key, *_ in LADDER},
        'rmse_cm': {key: float(np.sqrt(np.mean(data[key] ** 2))) for key, *_ in LADDER},
    }
    (out / 'fair_comparison_summary.json').write_text(json.dumps(summary, indent=2),
                                                      encoding='utf-8')
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows, manifest = load(args.capture.expanduser().resolve())
    draw(rows, manifest, args.out.expanduser().resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
