#!/usr/bin/env python3
"""Folder 03: does the camera error depend on which way the robot is facing?

Everything in this folder is scored on held-out floor squares -- the ones no learned
correction was fitted on -- so all five rungs of the ladder are measured on exactly the same
readings and every number here pairs with every other number here.

What changed, and why. This folder used to hold one eight-panel sheet per camera, for the
neural rung only. Eight warehouse maps on one page left each map about a thumbnail wide, the
arrows a few pixels long, and the other four rungs undrawn. It is now split all the way down:
one folder per correction, one folder per camera inside it, and inside that one full-page map
per robot heading plus an eight-up overview. The eight heading sheets in one camera folder
share a single arrow scale and a single colour scale, so arrow length there IS comparable
between headings -- which is the comparison the folder exists to support.

The interpretation limit that governs the whole folder: there is one capture per
camera-position-heading state. Every arrow is an observed error. Nothing here estimates a
conditional mean bias or a conditional covariance, and it must never be described as doing
so; separating those needs the repeat panel described in `09_what_R_needs/`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / 'logs/studies/camera_observation_characterization_20260831'
for rel in ('experiments/deck_figures', 'experiments/camera_observation_characterization'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
from _fieldscale import panel_scale  # noqa: E402
from _arrowmap import (  # noqa: E402
    CAMERAS, CORRECTIONS, camera_title, direction_phrase, draw_compass, draw_field,
    heading_dial, heading_slug, note_block, open_sheet, per_observation, save, sheet,
    stats,
)

FOLDER = '03_why_the_error'
CURVES = f'{FOLDER}/05_error_by_robot_heading'
SPREAD = f'{FOLDER}/06_heading_induced_spread'
SCORECARDS = f'{FOLDER}/07_camera_heading_scorecards'
FIELDS = f'{FOLDER}/08_residual_by_heading'
DELTAS = f'{FOLDER}/09_neural_minus_linear_by_heading'

SUMMARY_FIELDS = (
    'correction', 'correction_label', 'camera_id', 'heading_id', 'heading_deg',
    'camera_views', 'readings', 'detection_coverage', 'median_m', 'p90_m', 'rmse_m',
    'worse_than_10cm', 'mean_along_m', 'mean_across_m', 'median_along_m', 'median_across_m',
    'worse_than_raw', 'worse_than_learned_linear',
)
MIN_HEADINGS_FOR_SPREAD = 3
GRID_WORDS = 'held-out floor squares only'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def heading_degrees(rows: list[dict]) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in rows:
        heading = int(row['heading_id'])
        result.setdefault(heading, math.degrees(float(row['robot_yaw'])) % 360.0)
    return result


def finite(rows: list[dict], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows
                       if row.get(field, '') not in ('', 'nan')], dtype=float)


# --------------------------------------------------------------------------------------
# the numbers behind every figure in this folder


def summarize(rows: list[dict], correction, camera: str, heading: int,
              degrees: float) -> dict:
    views = [row for row in rows
             if row['camera_id'] == camera and int(row['heading_id']) == heading]
    hits = [row for row in views if row[f'{correction.key}_valid'] == '1']
    if not hits:
        raise RuntimeError(f'No held-out reading for {camera}, heading {heading}')
    error = finite(hits, f'{correction.key}_error_m')
    raw = finite(hits, 'raw_error_m')
    linear = finite(hits, 'learned_error_m')
    if not (error.size == raw.size == linear.size):
        raise RuntimeError(f'Unpaired rungs for {camera}, heading {heading}')
    along = finite(hits, f'{correction.key}_along_m')
    across = finite(hits, f'{correction.key}_across_m')
    return {
        'correction': correction.key, 'correction_label': correction.label,
        'camera_id': camera, 'heading_id': heading, 'heading_deg': degrees,
        'camera_views': len(views), 'readings': len(hits),
        'detection_coverage': len(hits) / len(views),
        'median_m': float(np.median(error)),
        'p90_m': float(np.quantile(error, 0.90)),
        'rmse_m': float(np.sqrt(np.mean(error ** 2))),
        'worse_than_10cm': float(np.mean(error > 0.10)),
        'mean_along_m': float(np.mean(along)), 'mean_across_m': float(np.mean(across)),
        'median_along_m': float(np.median(along)),
        'median_across_m': float(np.median(across)),
        'worse_than_raw': float(np.mean(error > raw)),
        'worse_than_learned_linear': float(np.mean(error > linear)),
    }


def matrix(summary: list[dict], correction_key: str, field: str,
           headings: list[int]) -> np.ndarray:
    by_key = {(row['camera_id'], row['heading_id']): row[field] for row in summary
              if row['correction'] == correction_key}
    return np.asarray([[float(by_key[(camera, heading)]) for heading in headings]
                       for camera in CAMERAS], dtype=float)


# --------------------------------------------------------------------------------------
# 05 -- the along-ray error against heading, all five rungs on one axis per camera


def draw_error_by_heading(rows: list[dict], headings: list[int], degrees: dict[int, float],
                          out: Path) -> None:
    """Per camera, how the along-ray error moves as the robot turns, for every rung."""
    for signed, name, ylabel, title, finding in (
        (True, 'along_the_camera_ray',
         'error along the camera ray (cm)\nnegative = falls toward the camera',
         'How far along the camera ray does the reading land, as the robot turns?',
         'The raw reading sits tens of centimetres short of the robot at every heading; '
         'the constant offset overshoots at some headings and undershoots at others'),
        (False, 'distance_from_truth', 'distance from truth (cm)',
         'How far from the truth does the reading land, as the robot turns?',
         'Turning the robot on the spot changes the typical error, and no rung is flat'),
    ):
        fig, axes = plt.subplots(5, 1, figsize=(13.4, 16.2), sharex=True,
                                 constrained_layout=True)
        for ax, camera in zip(axes, CAMERAS):
            for correction in CORRECTIONS:
                values = []
                for heading in headings:
                    hits = [row for row in rows
                            if row['camera_id'] == camera
                            and int(row['heading_id']) == heading
                            and row[f'{correction.key}_valid'] == '1']
                    field = f'{correction.key}_along_m' if signed \
                        else f'{correction.key}_error_m'
                    sample = finite(hits, field)
                    values.append(100 * float(np.median(sample)) if sample.size else math.nan)
                ax.plot([degrees[heading] for heading in headings], values, marker='o', ms=5.5,
                        lw=2.3, color=correction.colour, label=correction.label.split(':')[0])
            if signed:
                ax.axhline(0.0, color=D.INK, lw=1.1)
                ax.text(0.995, 0.06, 'below this line the reading falls toward the camera',
                        transform=ax.transAxes, ha='right', fontsize=9.6, color=D.INK2)
            ax.text(0.008, 0.93, camera_title(camera), transform=ax.transAxes, fontsize=13.5,
                    fontweight='bold', va='top')
            ax.set_ylabel(ylabel, fontsize=10.6)
            ax.grid(color='#e4e2dc', lw=0.8)
            ax.set_axisbelow(True)
        axes[0].legend(loc='upper right', ncol=3, fontsize=9.8, frameon=True)
        axes[-1].set_xlabel('which way the robot was facing (degrees)', fontsize=12)
        axes[-1].set_xticks([degrees[heading] for heading in headings])
        fig.suptitle(f'{title}\n{finding}\n'
                     f'{GRID_WORDS}; each point is the middle reading over the floor '
                     f'positions that returned a box at that heading',
                     fontsize=17, fontweight='bold')
        save(fig, out / CURVES / f'05_{name}.png')
        print(f'wrote {CURVES}/05_{name}.png')


# --------------------------------------------------------------------------------------
# 06 -- how much turning on the spot alone moves the reading


def heading_spread(rows: list[dict], key: str) -> dict[tuple[str, str], dict]:
    """The part of the error that rotating in place creates, per camera and position.

    At one floor position the robot was photographed at eight headings. Average those error
    vectors; the spread is the mean distance of each heading's error from that average. What
    is left over is the part of the error that does not care which way the robot faces.

    Positions with fewer than three returned headings are dropped: with one or two samples
    this is not a spread, it is noise.
    """
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        if row[f'{key}_valid'] != '1':
            continue
        grouped[(row['camera_id'], row['position_id'])].append(
            (float(row[f'{key}_dx']), float(row[f'{key}_dy']),
             float(row['robot_x']), float(row['robot_y'])))
    result = {}
    for group, items in grouped.items():
        if len(items) < MIN_HEADINGS_FOR_SPREAD:
            continue
        residuals = np.asarray([[item[0], item[1]] for item in items], dtype=float)
        centre = residuals.mean(axis=0)
        result[group] = {
            'x': items[0][2], 'y': items[0][3], 'headings': len(items),
            'spread_m': float(np.mean(np.hypot(*(residuals - centre).T))),
            'invariant_m': float(np.hypot(*centre)),
        }
    return result


def draw_spread_sheets(rows: list[dict], out: Path) -> list[dict]:
    """One full page per correction per camera, plus one page comparing all five rungs."""
    fields = {correction.key: heading_spread(rows, correction.key)
              for correction in CORRECTIONS}
    pooled = np.concatenate([
        np.asarray([cell['spread_m'] for cell in fields[correction.key].values()])
        for correction in CORRECTIONS])
    # one colour scale across all 25 sheets, so a bright square means the same thing
    # wherever it appears in this folder
    cap = 100 * float(np.quantile(pooled, 0.95))
    written = []

    for correction in CORRECTIONS:
        cells = fields[correction.key]
        for camera in CAMERAS:
            here = [cell for (cam, _position), cell in cells.items() if cam == camera]
            spread = 100 * np.asarray([cell['spread_m'] for cell in here], dtype=float)
            stays = 100 * np.asarray([cell['invariant_m'] for cell in here], dtype=float)
            fig, ax_map, ax_note, ax_hist = open_sheet()
            D.draw_warehouse(ax_map, D.layout(), show_cameras=True, camera_labels=True,
                             rack_alpha=0.80)
            if here:
                scalar = ax_map.scatter(
                    [cell['x'] for cell in here], [cell['y'] for cell in here], c=spread,
                    cmap='inferno', norm=Normalize(0, cap), marker='s', s=118,
                    linewidths=0, zorder=4)
                bar = fig.colorbar(scalar, ax=ax_map, fraction=0.040, pad=0.012,
                                   extend='max')
                bar.set_label('how far turning the robot on the spot moves the reading (cm)',
                              fontsize=10.8)
                bar.ax.tick_params(labelsize=9.5)
            ax_map.set_title(
                f'{GRID_WORDS}  ·  {len(here)} floor positions with '
                f'{MIN_HEADINGS_FOR_SPREAD}+ returned headings  ·  '
                f'colour stops at {cap:.0f} cm, shared by all 25 sheets in this folder',
                fontsize=12.0, fontweight='bold', pad=7)
            note_block(ax_note, 'What this map measures', [
                'Stand the robot on one floor square and turn it through eight',
                'headings without moving it. The reading moves anyway. This map',
                'shows by how much.',
                '',
                'For each square the eight error vectors are averaged. The',
                'colour is the average distance of a single heading from that',
                'average — the part of the error that turning creates.',
                '',
                'Bright square: turning the robot in place throws the reading',
                'around, so this camera cannot be trusted here without knowing',
                'which way the robot faces.',
                '',
                'Dark square: the error is still there, but it does not care',
                'which way the robot is facing — so a correction that never sees',
                'the heading can remove it.',
                '',
                f'Squares with fewer than {MIN_HEADINGS_FOR_SPREAD} returned headings are '
                'left blank:',
                'with one or two samples this is not a spread, it is noise.',
            ], fontsize=11.2)
            ax_hist.grid(color='#e4e2dc', lw=0.8)
            ax_hist.set_axisbelow(True)
            if spread.size:
                limit = max(1.0, float(np.quantile(np.concatenate([spread, stays]), 0.98)))
                bins = np.linspace(0, limit, 30)
                ax_hist.hist(np.minimum(spread, limit), bins=bins, color=correction.colour,
                             alpha=0.78, edgecolor='white', linewidth=0.5,
                             label='created by turning the robot')
                ax_hist.hist(np.minimum(stays, limit), bins=bins, histtype='step', lw=2.2,
                             color=D.INK, label='there whatever way it faces')
                ax_hist.axvline(float(np.median(spread)), color=correction.colour, lw=1.8,
                                linestyle='--')
                ax_hist.set_xlim(0, limit)
                ax_hist.legend(fontsize=9.5, frameon=True, loc='upper right')
            ax_hist.set_xlabel('centimetres at one floor position', fontsize=10.4)
            ax_hist.set_ylabel('floor positions', fontsize=10.4)
            ax_hist.set_title('Turned on the spot, against what stays put',
                              fontsize=11.6, fontweight='bold')
            fig.suptitle(
                f'{camera_title(camera)} — {correction.label}\n'
                f'Turning the robot on the spot alone moves this reading '
                f'{np.median(spread):.1f} cm typically, and up to {spread.max():.1f} cm '
                f'somewhere on this floor',
                fontsize=16.5, fontweight='bold', y=0.988, linespacing=1.5)
            fig.text(0.5, 0.014,
                     f'{GRID_WORDS}  ·  the part that stays put is '
                     f'{np.median(stays):.1f} cm typically  ·  {correction.subtitle}  ·  '
                     f'camera readings only: no filter, no fusion',
                     ha='center', fontsize=9.9, color=D.INK2)
            path = out / SPREAD / correction.folder / f'06_{camera}_{correction.slug}.png'
            save(fig, path)
            print(f'wrote {path.relative_to(out)}')
            written.append({
                'correction': correction.key, 'camera_id': camera,
                'figure': str(path.relative_to(out)), 'positions': int(spread.size),
                'median_spread_m': float(np.median(spread)) / 100,
                'p90_spread_m': float(np.quantile(spread, 0.90)) / 100,
                'worst_spread_m': float(spread.max()) / 100,
                'median_heading_invariant_m': float(np.median(stays)) / 100,
            })

    # the one page that compares the rungs against each other
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 7.4), constrained_layout=True)
    limit = float(np.quantile(100 * pooled, 0.99))
    bins = np.linspace(0, limit, 40)
    table = []
    for correction in CORRECTIONS:
        values = 100 * np.asarray([cell['spread_m']
                                   for cell in fields[correction.key].values()])
        axes[0].hist(np.minimum(values, limit), bins=bins, histtype='step', lw=2.5,
                     color=correction.colour, label=correction.label.split(':')[0])
        axes[0].axvline(float(np.median(values)), color=correction.colour, lw=1.3,
                        linestyle='--')
        table.append((correction, float(np.median(values)),
                      float(np.quantile(values, 0.90)), float(values.max())))
    axes[0].set_xlabel('how far turning the robot on the spot moves the reading (cm)',
                       fontsize=11.5)
    axes[0].set_ylabel('floor positions', fontsize=11.5)
    axes[0].set_title('Dashed line = the typical value for that correction',
                      fontsize=12.6, fontweight='bold')
    axes[0].legend(fontsize=10.2, frameon=True)
    axes[0].grid(color='#e4e2dc', lw=0.8)
    axes[0].set_axisbelow(True)
    axes[1].axis('off')
    axes[1].text(0.0, 1.0, 'Turning the robot in place, and what it costs', fontsize=15.2,
                 fontweight='bold', va='top')
    lines = [f'{"correction":<32}{"typical":>9}{"1 in 10":>10}{"worst":>9}', '']
    for correction, median, p90, worst in table:
        lines.append(f'{correction.label.split(":")[0][:31]:<32}'
                     f'{median:>7.1f} cm{p90:>8.1f} cm{worst:>7.1f} cm')
    axes[1].text(0.0, 0.86, '\n'.join(lines), fontsize=10.6, va='top', family='monospace',
                 linespacing=1.5)
    axes[1].text(
        0.0, 0.44,
        'Neither learned correction is told which way the robot is facing, so\n'
        'neither can represent this part of the error. They average over it,\n'
        'which helps where it is small and hurts where it is large — which is why\n'
        'the learned rungs cut the typical value but not the worst one.\n\n'
        'Recovering that missing variable from the box itself is what\n'
        '11_a_better_gate/ tests.',
        fontsize=11.6, va='top', linespacing=1.42)
    fig.suptitle('How much of the camera error is created purely by turning the robot?\n'
                 'Stand the robot on one floor square, turn it through eight headings '
                 'without moving it, and the reading moves anyway\n'
                 f'{GRID_WORDS}; all five corrections on identical readings',
                 fontsize=17, fontweight='bold', linespacing=1.45)
    path = out / SPREAD / '06_00_all_corrections_compared.png'
    save(fig, path)
    print(f'wrote {path.relative_to(out)}')
    return written


# --------------------------------------------------------------------------------------
# 07 -- one camera-by-heading scorecard per correction


def annotate(ax, values: np.ndarray, *, scale: float, suffix: str, signed: bool) -> None:
    finite_values = values[np.isfinite(values)]
    threshold = 0.58 * float(np.max(np.abs(finite_values))) if finite_values.size else math.inf
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not math.isfinite(value):
                continue
            colour = 'white' if abs(value) >= threshold else D.INK
            template = '{:+.1f}' if signed else '{:.1f}'
            ax.text(column, row, template.format(scale * value) + suffix, ha='center',
                    va='center', fontsize=8.0, color=colour, fontweight='bold')


def draw_scorecards(summary: list[dict], headings: list[int], degrees: dict[int, float],
                    out: Path) -> None:
    panels = (
        ('detection_coverage', 'Views that returned a box (%)', 'viridis', 100.0, False),
        ('median_m', 'Typical distance from truth (cm)', 'magma', 100.0, False),
        ('p90_m', 'Worst tenth: 1 in 10 is worse than (cm)', 'magma', 100.0, False),
        ('rmse_m', 'Root-mean-square error (cm)\nthe tail counts double', 'magma', 100.0,
         False),
        ('mean_along_m', 'Average lean along the ray (cm)\nnegative = toward the camera',
         'RdBu_r', 100.0, True),
        ('mean_across_m', 'Average lean across the ray (cm)\npositive = left of the ray',
         'RdBu_r', 100.0, True),
        ('worse_than_10cm', 'Readings worse than 10 cm (%)', 'YlOrRd', 100.0, False),
    )
    for correction in CORRECTIONS:
        fig, axes = plt.subplots(2, 4, figsize=(21.5, 11.0), constrained_layout=True)
        for ax, (field, title, cmap, scale, signed) in zip(axes.flat, panels):
            values = matrix(summary, correction.key, field, headings)
            if signed:
                cap = max(0.01, float(np.nanmax(np.abs(values))))
                norm = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)
            else:
                norm = Normalize(0.0, max(float(np.nanmax(values)), 1e-12))
            image = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')
            annotate(ax, values, scale=scale, suffix='', signed=signed)
            ax.set_title(title, fontsize=12.4, fontweight='bold')
            ax.set_xticks(range(len(headings)))
            ax.set_xticklabels([f'{degrees[heading]:.0f}°' for heading in headings],
                               fontsize=10)
            ax.set_yticks(range(len(CAMERAS)))
            ax.set_yticklabels([camera_title(camera) for camera in CAMERAS], fontsize=10.5)
            ax.set_xlabel('which way the robot was facing', fontsize=10.2)
            bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
            # the cells are printed in the panel's own unit, so the bar must use it too
            bar.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(
                lambda value, _pos, factor=scale: f'{factor * value:.0f}'))
            bar.ax.tick_params(labelsize=9.5)
        median = matrix(summary, correction.key, 'median_m', headings)
        best = np.unravel_index(int(np.argmin(median)), median.shape)
        worst = np.unravel_index(int(np.argmax(median)), median.shape)
        note_block(
            axes.flat[-1], 'Why split the error by heading at all?',
            ['Neither learned correction is told which way the robot is',
             'facing. A single pooled number can therefore hide one',
             'camera-and-heading combination that fails outright, or',
             'cancel two opposite leans against each other.',
             '',
             'On this sheet the typical error runs from',
             f'{100 * median[best]:.1f} cm ({camera_title(CAMERAS[best[0]])} facing '
             f'{degrees[headings[best[1]]]:.0f}°)',
             f'to {100 * median[worst]:.1f} cm '
             f'({camera_title(CAMERAS[worst[0]])} facing '
             f'{degrees[headings[worst[1]]]:.0f}°)',
             f'— a spread of {100 * (median[worst] - median[best]):.1f} cm '
             f'that a pooled median hides.',
             '',
             'Coverage is on this sheet for the same reason it is in',
             'folder 01: a reading that never arrived is not a small',
             'error, and every other panel here is conditional on one',
             'arriving.',
             '',
             'Held-out floor squares only. All five corrections are',
             'scored on exactly the same readings, so a cell here can be',
             'compared with the same cell on the other four sheets.',
             '',
             'These are raw camera readings. No filter, no fusion, no',
             'belief — and one capture per state, so a cell is an',
             'observed error, not an estimated bias.'],
            fontsize=11.0)
        fig.suptitle(f'{correction.label}\n'
                     f'The pooled number does not describe every camera and every heading\n'
                     f'{GRID_WORDS}; one frozen detector; every returned box retained',
                     fontsize=18.5, fontweight='bold', linespacing=1.4)
        path = out / SCORECARDS / f'07_{correction.folder[:2]}_{correction.slug}.png'
        save(fig, path)
        print(f'wrote {path.relative_to(out)}')


# --------------------------------------------------------------------------------------
# 08 -- one full-page map per correction, per camera, per heading


def overview(rows: list[dict], summary_by_key: dict, correction, camera: str,
             headings: list[int], degrees: dict[int, float], scale: dict,
             out: Path) -> Path:
    """The eight headings on one page -- an index to the eight full-page sheets."""
    fig, axes = plt.subplots(2, 4, figsize=(23.0, 12.4), constrained_layout=True)
    scalar = None
    for ax, heading in zip(axes.flat, headings):
        subset = [row for row in rows if row['camera_id'] == camera
                  and int(row['heading_id']) == heading]
        cells, blanks = per_observation(subset, correction.key)
        drawn = draw_field(fig, ax, cells, blanks, colour_cap=scale['cap'],
                           arrow_gain=scale['gain'], marker_size=10.0, cbar=False)
        scalar = drawn['scalar'] or scalar
        cell = summary_by_key[(correction.key, camera, heading)]
        ax.set_title(f'robot facing {degrees[heading]:.0f}°  ·  '
                     f'{cell["readings"]} of {cell["camera_views"]} views returned a box\n'
                     f'typical {100 * cell["median_m"]:.1f} cm  ·  '
                     f'1 in 10 worse than {100 * cell["p90_m"]:.1f} cm',
                     fontsize=12.2, fontweight='bold')
    if scalar is not None:
        bar = fig.colorbar(scalar, ax=axes.flatten().tolist(), fraction=0.016, pad=0.008,
                           extend='max')
        bar.set_label(f'how far the reading missed (cm) — colour stops at '
                      f'{100 * scale["cap"]:.0f} cm', fontsize=11.5)
        bar.ax.yaxis.set_major_formatter(
            plt.matplotlib.ticker.FuncFormatter(lambda value, _pos: f'{100 * value:.0f}'))
    medians = [100 * summary_by_key[(correction.key, camera, heading)]['median_m']
               for heading in headings]
    fig.suptitle(
        f'{camera_title(camera)} — {correction.label}, one panel per robot heading\n'
        f'Turning the robot changes the typical error from {min(medians):.1f} cm to '
        f'{max(medians):.1f} cm without the robot moving at all\n'
        f'{GRID_WORDS}; every arrow is one reading, drawn {scale["gain"]:.0f}× life size; '
        f'all eight panels and the eight full-page sheets beside them share this scale; '
        f'grey × = no box',
        fontsize=18.0, fontweight='bold', linespacing=1.42)
    path = out / FIELDS / correction.folder / camera / \
        f'08_{camera}_{correction.slug}_00_all_headings.png'
    save(fig, path)
    return path


def draw_heading_fields(rows: list[dict], summary: list[dict], headings: list[int],
                        degrees: dict[int, float], out: Path) -> list[dict]:
    summary_by_key = {(row['correction'], row['camera_id'], row['heading_id']): row
                      for row in summary}
    written = []
    for correction in CORRECTIONS:
        for camera in CAMERAS:
            camera_rows = [row for row in rows if row['camera_id'] == camera]
            # one scale for all eight headings of this camera, so the sheets compare
            scale = panel_scale(np.asarray(
                [float(row[f'{correction.key}_error_m']) for row in camera_rows
                 if row[f'{correction.key}_valid'] == '1'], dtype=float))
            index = overview(camera_rows, summary_by_key, correction, camera, headings,
                             degrees, scale, out)
            print(f'wrote {index.relative_to(out)}')
            for position, heading in enumerate(headings, start=1):
                subset = [row for row in camera_rows if int(row['heading_id']) == heading]
                cells, blanks = per_observation(subset, correction.key)
                cell = summary_by_key[(correction.key, camera, heading)]
                local = stats(cells)
                path = (out / FIELDS / correction.folder / camera /
                        f'08_{camera}_{correction.slug}_{position:02d}_'
                        f'{heading_slug(heading, degrees[heading])}.png')
                sheet(
                    path=path,
                    title=f'{camera_title(camera)}, robot facing '
                          f'{degrees[heading]:.0f}° — {correction.label}',
                    finding=(
                        f'At this one heading half the readings land within '
                        f'{100 * cell["median_m"]:.1f} cm of the truth, and '
                        f'{direction_phrase(local["along"], local["across"])}'),
                    provenance=(
                        f'{cell["readings"]} of {cell["camera_views"]} camera views at this '
                        f'heading returned a box  ·  1 reading in 10 is worse than '
                        f'{100 * cell["p90_m"]:.1f} cm  ·  '
                        f'{100 * cell["worse_than_10cm"]:.0f}% are worse than 10 cm'),
                    grid_short=GRID_WORDS,
                    cells=cells, blanks=blanks,
                    note_title='How to read this map',
                    note_lines=[
                        'The robot was placed on each floor square and photographed',
                        'facing one fixed direction, shown on the dial. Each arrow',
                        'starts where the robot really was and points to where this',
                        'camera said it was.',
                        '',
                        'One arrow is ONE photograph, not an average. There is a',
                        'single capture per square at this heading, so an arrow is an',
                        'error that was observed, not a bias that was estimated.',
                        '',
                        'All eight heading sheets in this folder share one arrow',
                        'scale and one colour scale, so arrow length HERE can be',
                        'compared with arrow length on the other seven. That is the',
                        'comparison this folder exists for: if the arrows swing round',
                        'as you page through the headings, the error depends on which',
                        'way the robot faces.',
                        '',
                        'A grey × is a floor square where the camera returned no box',
                        'at this heading. A missing reading is not an accurate one.',
                        '',
                        'Only floor squares no correction was fitted on appear, so',
                        'the checkerboard gaps are by design.',
                    ],
                    hist_xlabel='distance from truth of one reading (cm)',
                    hist_title='How far the readings miss at this heading',
                    colour=correction.colour,
                    heading_deg=degrees[heading],
                    colour_cap=scale['cap'],
                    arrow_gain=scale['gain'],
                    footer=f'{correction.subtitle}  ·  camera readings only: '
                           f'no filter, no fusion, no admission gate',
                )
                written.append({
                    'correction': correction.key, 'camera_id': camera,
                    'heading_id': heading, 'heading_deg': degrees[heading],
                    'figure': str(path.relative_to(out)),
                    'arrow_gain': scale['gain'], 'colour_cap_m': scale['cap'],
                })
                print(f'wrote {path.relative_to(out)}')
    return written


# --------------------------------------------------------------------------------------
# 09 -- where the network beats the straight-line correction, heading by heading


def draw_delta_fields(rows: list[dict], summary: list[dict], headings: list[int],
                      degrees: dict[int, float], out: Path) -> None:
    paired = [row for row in rows if row['nn_valid'] == '1' and row['learned_valid'] == '1']
    everything = np.asarray([float(row['nn_error_m']) - float(row['learned_error_m'])
                             for row in paired], dtype=float)
    cap = max(0.05, float(np.quantile(np.abs(everything), 0.95)))
    norm = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)
    by_key = {(row['correction'], row['camera_id'], row['heading_id']): row
              for row in summary}

    def panel(ax, camera: str, heading: int, *, marker_size: float):
        subset = [row for row in rows if row['camera_id'] == camera
                  and int(row['heading_id']) == heading]
        good = [row for row in subset
                if row['nn_valid'] == '1' and row['learned_valid'] == '1']
        blank = [row for row in subset if row['nn_valid'] != '1']
        D.draw_warehouse(ax, D.layout(), show_cameras=True, camera_labels=True,
                         rack_alpha=0.72)
        if blank:
            ax.scatter([float(row['robot_x']) for row in blank],
                       [float(row['robot_y']) for row in blank], marker='x', s=10,
                       c='#aaa9a4', linewidths=0.7, zorder=3)
        delta = np.asarray([float(row['nn_error_m']) - float(row['learned_error_m'])
                            for row in good], dtype=float)
        scalar = None
        if good:
            scalar = ax.scatter([float(row['robot_x']) for row in good],
                                [float(row['robot_y']) for row in good], c=delta,
                                cmap='RdBu_r', norm=norm, marker='s', s=marker_size,
                                linewidths=0, zorder=5)
        return scalar, delta

    for camera in CAMERAS:
        fig, axes = plt.subplots(2, 4, figsize=(23.0, 12.4), constrained_layout=True)
        scalar = None
        shares = []
        for ax, heading in zip(axes.flat, headings):
            drawn, delta = panel(ax, camera, heading, marker_size=26)
            scalar = drawn or scalar
            cell = by_key[('nn', camera, heading)]
            shares.append(100 * cell['worse_than_learned_linear'])
            ax.set_title(f'robot facing {degrees[heading]:.0f}°  ·  network worse on '
                         f'{100 * cell["worse_than_learned_linear"]:.0f}% of readings\n'
                         f'typical change {100 * np.median(delta):+.1f} cm',
                         fontsize=12.2, fontweight='bold')
        if scalar is not None:
            bar = fig.colorbar(scalar, ax=axes.flatten().tolist(), fraction=0.016, pad=0.008,
                               extend='both')
            bar.set_label('network error minus straight-line error (cm)  ·  '
                          'red = the network is worse here', fontsize=11.5)
            bar.ax.yaxis.set_major_formatter(
                plt.matplotlib.ticker.FuncFormatter(lambda value, _pos: f'{100 * value:.0f}'))
        fig.suptitle(
            f'{camera_title(camera)} — does the neural correction beat the straight-line one, '
            f'heading by heading?\n'
            f'The network is the worse of the two on {min(shares):.0f}% to {max(shares):.0f}% '
            f'of readings depending only on which way the robot faces\n'
            f'{GRID_WORDS}; the two are compared on the same square, the same heading and '
            f'the same photograph; grey × = no box',
            fontsize=18.0, fontweight='bold', linespacing=1.42)
        path = out / DELTAS / camera / f'09_{camera}_00_all_headings.png'
        save(fig, path)
        print(f'wrote {path.relative_to(out)}')

        for position, heading in enumerate(headings, start=1):
            fig, ax_map, ax_note, ax_hist = open_sheet(left=0.060)
            drawn, delta = panel(ax_map, camera, heading, marker_size=110)
            cell = by_key[('nn', camera, heading)]
            if drawn is not None:
                bar = fig.colorbar(drawn, ax=ax_map, fraction=0.040, pad=0.012,
                                   extend='both')
                bar.set_label('network error minus straight-line error (cm)', fontsize=10.8)
                bar.ax.yaxis.set_major_formatter(
                    plt.matplotlib.ticker.FuncFormatter(
                        lambda value, _pos: f'{100 * value:+.0f}'))
            ax_map.set_title(f'{GRID_WORDS}  ·  {delta.size} readings compared  ·  '
                             f'colour stops at ±{100 * cap:.0f} cm', fontsize=12.0,
                             fontweight='bold', pad=7)
            note_block(ax_note, 'How to read this map', [
                'Both learned corrections were run on the SAME photograph of',
                'the robot on the same floor square at the same heading, and',
                'this map shows which of the two landed closer.',
                '',
                'Blue square: the network won there.',
                'Red square: the straight-line correction won there.',
                'Pale square: the two were within a centimetre or so.',
                '',
                'The size of the win is the colour, not the size of the',
                'square. Colour saturates at the value printed above the map.',
                '',
                'A grey × is a floor square with no box at this heading.',
                '',
                'Neither correction is told which way the robot is facing, so',
                'if the red and blue regions move as you page through the',
                'eight headings, that is the missing variable showing itself.',
                '',
                f'{GRID_WORDS}: only squares neither correction was fitted on.',
            ], fontsize=11.0)
            draw_compass(heading_dial(fig), degrees[heading])
            ax_hist.grid(color='#e4e2dc', lw=0.8)
            ax_hist.set_axisbelow(True)
            if delta.size:
                limit = max(1.0, 100 * float(np.quantile(np.abs(delta), 0.98)))
                bins = np.linspace(-limit, limit, 33)
                values = np.clip(100 * delta, -limit, limit)
                ax_hist.hist(values[values <= 0], bins=bins, color='#2a78d6', alpha=0.8,
                             edgecolor='white', linewidth=0.5, label='network closer')
                ax_hist.hist(values[values > 0], bins=bins, color='#c0392b', alpha=0.8,
                             edgecolor='white', linewidth=0.5, label='straight line closer')
                ax_hist.axvline(0.0, color=D.INK, lw=1.6)
                ax_hist.legend(fontsize=9.6, frameon=True)
                ax_hist.set_xlim(-limit, limit)
            ax_hist.set_xlabel('network error minus straight-line error (cm)', fontsize=10.4)
            ax_hist.set_ylabel('readings', fontsize=10.4)
            ax_hist.set_title('Which correction landed closer?', fontsize=11.6,
                              fontweight='bold')
            fig.suptitle(
                f'{camera_title(camera)}, robot facing {degrees[heading]:.0f}° — '
                f'neural correction against the straight-line one\n'
                f'At this heading the network is the worse of the two on '
                f'{100 * cell["worse_than_learned_linear"]:.0f}% of readings, and the typical '
                f'change is {100 * np.median(delta) if delta.size else float("nan"):+.1f} cm',
                fontsize=16.5, fontweight='bold', y=0.988, linespacing=1.5)
            fig.text(0.5, 0.014,
                     f'Paired on the same photograph  ·  {GRID_WORDS}  ·  camera readings '
                     f'only: no filter, no fusion, no admission gate',
                     ha='center', fontsize=9.9, color=D.INK2)
            path = (out / DELTAS / camera /
                    f'09_{camera}_{position:02d}_{heading_slug(heading, degrees[heading])}.png')
            save(fig, path)
            print(f'wrote {path.relative_to(out)}')


# --------------------------------------------------------------------------------------


def write_summary(summary: list[dict], spread: list[dict], fields: list[dict],
                  source: Path, source_manifest: Path, out: Path) -> None:
    folder = out / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / 'heading_conditioned_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    (folder / 'heading_conditioned_manifest.json').write_text(json.dumps({
        'status': 'complete',
        'schema': 'heading_conditioned_bias_diagnostics.v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'evidence': 'held-out 2 m checkerboard tiles from the frozen one-shot field capture',
        'quantity': 'camera reading scored against commanded ground-truth position',
        'conditioning': 'correction, camera_id and captured robot heading',
        'grid_rule': (
            'Every figure and every row in this folder is scored on held-out floor squares '
            'only, so all five corrections are measured on identical readings and every '
            'number pairs with every other number here.'),
        'scale_rule': (
            'The eight heading sheets of one camera-correction folder share one arrow gain '
            'and one colour cap, so arrow length is comparable between headings within a '
            'folder and never between folders.'),
        'interpretation_limit': (
            'One observation per camera-position-heading cell: mapped arrows are observed '
            'errors, not estimates of conditional mean bias or conditional covariance.'),
        'selection_limit': (
            'Accuracy is conditional on a box being returned; the share of camera views '
            'that returned one is reported in every cell and on every scorecard.'),
        'source_sha256': sha256(source),
        'source_manifest_sha256': sha256(source_manifest),
        'summary_csv_sha256': sha256(csv_path),
        'cells': summary,
        'heading_induced_spread': spread,
        'field_sheets': fields,
        'figures': {
            '05_error_by_robot_heading/': 'two sheets, all five corrections per camera',
            '06_heading_induced_spread/': 'one sheet per correction plus one comparison',
            '07_camera_heading_scorecards/': 'one camera-by-heading scorecard per correction',
            '08_residual_by_heading/<correction>/<camera>/':
                'an eight-up overview plus one full-page map per heading',
            '09_neural_minus_linear_by_heading/<camera>/':
                'an eight-up overview plus one full-page map per heading',
        },
        'next_required_evidence': (
            'A separately frozen repeat panel is required to separate conditional mean bias '
            'from repeated-sampling spread at fixed camera, position and heading.'),
    }, indent=2), encoding='utf-8')


OWNED = (CURVES, SPREAD, SCORECARDS, FIELDS, DELTAS)
RETIRED = ('03_why_the_error/05_bias_by_robot_heading.png',
           '03_why_the_error/06_heading_induced_spread.png',
           '03_why_the_error/07_camera_heading_scorecard.png',
           '03_why_the_error/08_nn_residual_by_heading',
           '03_why_the_error/09_nn_minus_linear_by_heading')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--only', choices=('curves', 'spread', 'scorecards', 'fields',
                                           'deltas'), action='append',
                        help='Redraw only these parts; the summary CSV is always rewritten.')
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    source = capture / 'bias_update_interpretations.csv'
    source_manifest = capture / 'bias_update_interpretations_manifest.json'
    if not source.is_file() or not source_manifest.is_file():
        raise RuntimeError('Run fit_bias_updates.py first')
    manifest = json.loads(source_manifest.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise RuntimeError('bias-update interpretations are not complete')
    if sha256(source) != manifest['bias_update_interpretations_sha256']:
        raise RuntimeError('bias_update_interpretations.csv changed after fitting')

    rows = [row for row in csv.DictReader(source.open(encoding='utf-8'))
            if row['split'] == 'test']
    degrees = heading_degrees(rows)
    headings = sorted(degrees)
    summary = [summarize(rows, correction, camera, heading, degrees[heading])
               for correction in CORRECTIONS for camera in CAMERAS for heading in headings]

    out = (args.out or DEFAULT_OUT).expanduser().resolve()
    wanted = set(args.only or ('curves', 'spread', 'scorecards', 'fields', 'deltas'))
    existing = [out / name for name in OWNED if (out / name).exists()]
    if existing and not args.overwrite:
        raise RuntimeError(f'Output already exists under {out / FOLDER}; '
                           f'pass --overwrite to refresh')
    for name in RETIRED:
        stale = out / name
        if stale.is_dir():
            shutil.rmtree(stale)
            print(f'removed superseded {name}')
        elif stale.is_file():
            stale.unlink()
            print(f'removed superseded {name}')

    if 'curves' in wanted:
        draw_error_by_heading(rows, headings, degrees, out)
    spread = draw_spread_sheets(rows, out) if 'spread' in wanted else []
    if 'scorecards' in wanted:
        draw_scorecards(summary, headings, degrees, out)
    fields = draw_heading_fields(rows, summary, headings, degrees, out) \
        if 'fields' in wanted else []
    if 'deltas' in wanted:
        draw_delta_fields(rows, summary, headings, degrees, out)
    previous_path = out / FOLDER / 'heading_conditioned_manifest.json'
    if args.only and previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        if 'spread' not in wanted:
            spread = previous.get('heading_induced_spread', [])
        if 'fields' not in wanted:
            fields = previous.get('field_sheets', [])
    write_summary(summary, spread, fields, source, source_manifest, out)
    print(json.dumps({'output': str(out / FOLDER), 'cells': len(summary),
                      'figures': len(list((out / FOLDER).rglob('*.png')))}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
