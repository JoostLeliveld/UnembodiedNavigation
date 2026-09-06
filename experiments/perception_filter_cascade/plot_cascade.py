#!/usr/bin/env python3
"""Figures for the perception-filter cascade study.

Three panels, each answering one question a reader will ask:

  1. How much of the error can a temporal filter even reach? The smooth part of the error
     travels with the robot and cannot be averaged away; only the fast part can. This
     panel shows both, along one dense line, so the ceiling on any filter is visible
     before any filter is judged.
  2. Is the fast part actually fast? A residual that is still correlated from sample to
     sample means the smooth part was not removed and the split cannot be trusted.
  3. Do the arms earn their place? Accuracy and honesty are plotted together, because a
     filter that merely widens its ellipse passes any consistency test and is useless.

Every panel states its finding in the title and labels which direction is good.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Consistency reference for a two-dimensional measurement.
NSE_REFERENCE = 2.0
INK = '#22252a'
MUTED = '#8a9099'
ACCENT = '#1f5da8'
WARN = '#b5423a'
GOOD = '#16834b'


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def load_residuals(study_dir: Path) -> list[dict[str, str]]:
    path = study_dir / 'line_residuals.csv'
    if not path.exists():
        raise SystemExit(f'{path} not found; run separate_bias_and_fast_noise.py first')
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def plot_trend_vs_residual(rows: list[dict[str, str]], study_dir: Path) -> None:
    """Panel 1: the smooth part and the fast part along one representative line."""
    signal = 'v_bbox_bottom'
    candidates = [row for row in rows if row['signal'] == signal]
    if not candidates:
        print(f'no {signal} residuals; skipping trend figure')
        return

    # Pick the line with the most samples so the panel is the best-supported one.
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        groups[(row['camera_id'], row['line'])].append(row)
    key = max(groups, key=lambda item: len(groups[item]))
    series = sorted(groups[key], key=lambda row: float(row['arc_m']))

    arc = [float(row['arc_m']) * 100.0 for row in series]
    observed = [float(row['observed']) for row in series]
    trend = [float(row['trend']) for row in series]
    residual = [float(row['residual']) for row in series]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True,
                             gridspec_kw={'height_ratios': [2.0, 1.0]},
                             constrained_layout=True)

    span = max(trend) - min(trend)
    axes[0].plot(arc, observed, 'o-', color=ACCENT, ms=4, lw=1.2,
                 label='detector output, one sample per 4 cm of travel')
    axes[0].plot(arc, trend, '-', color=WARN, lw=2.0,
                 label=f'smooth part: moves {span:.1f} px along this line')
    axes[0].set_ylabel('box bottom edge, in pixels down the image')
    axes[0].legend(loc='best', fontsize=9, frameon=False)
    axes[0].grid(alpha=0.25)

    fast = st.median([abs(value) for value in residual])
    axes[1].axhline(0.0, color=MUTED, lw=1.0)
    axes[1].plot(arc, residual, 'o-', color=INK, ms=4, lw=1.0)
    axes[1].set_ylabel('what is left over\n(pixels)')
    axes[1].set_xlabel('distance travelled along the line, in centimetres')
    axes[1].grid(alpha=0.25)
    axes[1].set_title(
        f'Only this leftover part can be averaged away: typically {fast:.2f} px '
        f'against a {span:.1f} px smooth drift',
        fontsize=10, color=INK)

    camera = key[0].replace('camera_', 'Camera ')
    # The headline must follow the data, not a prior expectation: whichever of the two
    # components is larger decides whether a temporal filter has much to work on.
    if span > 3.0 * fast:
        finding = ('Most of the error drifts smoothly as the robot drives, '
                   'so repeated looks cannot average it away')
    elif fast > span:
        finding = ('The error jumps from frame to frame more than it drifts, '
                   'so repeated looks can average much of it away')
    else:
        finding = ('The error has a smooth drift and a comparable frame-to-frame part, '
                   'so a filter can reach only part of it')
    fig.suptitle(
        f'{finding}\n'
        f'{camera}, one straight line, {len(series)} placements 4 cm apart',
        fontsize=12)
    save(fig, study_dir / 'figures' / '01_smooth_versus_fast.png')


def plot_autocorrelation(study_dir: Path) -> None:
    """Panel 2: is the leftover part actually fast, or still slowly varying?"""
    payload = json.loads((study_dir / 'bq_split.json').read_text(encoding='utf-8'))
    summary = payload.get('summary', {})
    if not summary:
        print('no summary in bq_split.json; skipping autocorrelation figure')
        return

    labels, values, colours = [], [], []
    nice = {'u_bbox_bottom': 'box centre,\nsideways (px)',
            'v_bbox_bottom': 'box bottom edge,\nvertical (px)',
            'raw_dx': 'position error,\neast (cm)',
            'raw_dy': 'position error,\nnorth (cm)'}
    for signal, stats in summary.items():
        ac = stats['lag1_autocorr_median']
        if ac != ac:
            continue
        labels.append(nice.get(signal, signal))
        values.append(ac)
        colours.append(GOOD if abs(ac) <= 0.35 else WARN)

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    positions = range(len(labels))
    ax.bar(positions, values, color=colours, width=0.6)
    ax.axhline(0.35, color=MUTED, ls='--', lw=1.2)
    ax.text(len(labels) - 0.45, 0.37,
            'above this line the leftover is still\nslowly varying, so the split failed',
            fontsize=8.5, color=MUTED, va='bottom', ha='right')
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('how much one sample resembles the next\n(0 = independent, 1 = identical)')
    ax.set_ylim(min(-0.1, min(values) - 0.1), max(1.0, max(values) + 0.15))
    ax.grid(axis='y', alpha=0.25)
    ax.set_title(
        'Green: the leftover really is fast, so a filter could average it down.\n'
        'Red: the leftover still drifts, so it is bias in disguise.',
        fontsize=11)
    save(fig, study_dir / 'figures' / '02_is_the_leftover_fast.png')


def plot_arms(study_dir: Path) -> None:
    """Panel 3: accuracy and honesty side by side, never one without the other."""
    path = study_dir / 'arm_comparison.json'
    if not path.exists():
        print(f'{path} not found; skipping arm figure')
        return
    payload = json.loads(path.read_text(encoding='utf-8'))
    overall = payload.get('overall', {})

    order = ['A_raw', 'E_static_R', 'B_kf', 'C_robust_kf', 'D_smoother']
    nice = {'A_raw': 'per-frame\n(today)',
            'E_static_R': 'per-frame with\nmeasured spread\n(to beat)',
            'B_kf': 'box filter',
            'C_robust_kf': 'box filter with\nsoft rejection',
            'D_smoother': 'fixed-lag\nsmoother'}
    arms = [arm for arm in order if overall.get(arm, {}).get('n')]
    if not arms:
        print('no scored arms; skipping arm figure')
        return

    median = [overall[arm]['median_err_cm'] for arm in arms]
    nse = [overall[arm]['mean_nse'] for arm in arms]
    tail = [overall[arm]['beyond_4sigma_pct'] for arm in arms]
    labels = [nice.get(arm, arm) for arm in arms]
    positions = list(range(len(arms)))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=True)

    axes[0].bar(positions, median, color=ACCENT, width=0.6)
    axes[0].set_ylabel('typical distance from the true position (cm)')
    axes[0].set_title('Accuracy — lower is better', fontsize=11)

    baseline = overall['E_static_R']['mean_nse'] if 'E_static_R' in overall else None
    colours = [GOOD if abs(value - NSE_REFERENCE) <= abs((baseline or value) - NSE_REFERENCE)
               else WARN for value in nse]
    axes[1].bar(positions, nse, color=colours, width=0.6)
    axes[1].axhline(NSE_REFERENCE, color=INK, ls='--', lw=1.4)
    axes[1].text(len(arms) - 0.45, NSE_REFERENCE * 1.08,
                 'honest level: stated uncertainty\nmatches the real error',
                 fontsize=8.5, color=INK, va='bottom', ha='right')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('stated uncertainty versus real error\n(log scale)')
    axes[1].set_title('Honesty — above the line it claims\nmore precision than it has',
                      fontsize=11)

    axes[2].bar(positions, tail, color=WARN, width=0.6)
    axes[2].set_ylabel('share of readings far outside\ntheir stated uncertainty (%)')
    axes[2].set_title('The tail — lower is better', fontsize=11)

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=8.5)
        axis.grid(axis='y', alpha=0.25)

    settings = payload.get('settings', {})
    fig.suptitle(
        'Does filtering the box over time produce a better and more honest measurement?\n'
        f'{overall[arms[0]]["n"]} readings per arm, warehouse_v2, '
        f'{payload.get("lines_used", "?")} camera-line sequences, '
        f'sample spacing {settings.get("step_s", "?")} s',
        fontsize=12)
    save(fig, study_dir / 'figures' / '03_do_the_arms_earn_their_place.png')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()

    rows = load_residuals(args.study_dir)
    plot_trend_vs_residual(rows, args.study_dir)
    plot_autocorrelation(args.study_dir)
    plot_arms(args.study_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
