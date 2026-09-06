#!/usr/bin/env python3
"""Make reader-first warehouse maps from the frozen box characterization table."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / 'logs/studies/camera_observation_characterization_20260831'
for rel in ('experiments/deck_figures',):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
from _fieldscale import drawn_shrink, panel_scale, scale_bar  # noqa: E402

METHODS = (
    ('raw', 'Raw box → floor', '#eb6834'),
    ('fixed', 'Fixed 30.9 cm shift', '#2a78d6'),
    ('hull', 'Hull residual around reference', '#1baf7a'),
)
CAMERAS = tuple(f'camera_{letter}' for letter in 'ABCDE')


def floats(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows if row[field] not in ('', 'nan')])


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def camera_title(camera_id: str) -> str:
    return f'Camera {camera_id[-1]}'


def aggregate_positions(rows: list[dict[str, str]], method: str):
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row['position_id']].append(row)
    result = []
    for group in groups.values():
        hits = [row for row in group if row['detected'] == '1' and row[f'{method}_valid'] == '1']
        first = group[0]
        if not hits:
            result.append((float(first['robot_x']), float(first['robot_y']), math.nan, math.nan, math.nan, 0.0))
            continue
        dx = float(np.median(floats(hits, f'{method}_dx')))
        dy = float(np.median(floats(hits, f'{method}_dy')))
        error = float(np.median(floats(hits, f'{method}_error_m')))
        result.append((float(first['robot_x']), float(first['robot_y']), dx, dy, error, len(hits) / len(group)))
    return result


def aggregate_confidence(rows: list[dict[str, str]]):
    """Median YOLO confidence per floor position, over the headings that returned a box.

    Confidence is read only where `detected == 1`. Undetected rows still carry the
    best sub-threshold candidate score from the detector's 0.001 prediction floor, and
    averaging those in would invent a reading that the detector never made.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row['position_id']].append(row)
    result = []
    for group in groups.values():
        first = group[0]
        hits = [row for row in group if row['detected'] == '1']
        confidence = float(np.median(floats(hits, 'confidence'))) if hits else math.nan
        result.append((float(first['robot_x']), float(first['robot_y']), confidence))
    return result


def draw_coverage(rows: list[dict[str, str]], out: Path, threshold: float) -> None:
    """Where a box comes back at all (top), and how confident it is where it does (bottom)."""
    columns = len(CAMERAS) + 1
    fig, axes = plt.subplots(2, columns, figsize=(4.9 * columns, 10.6),
                             constrained_layout=True)
    coverage_scalar = confidence_scalar = None
    all_confidence = []

    for column, camera_id in enumerate(CAMERAS):
        camera_rows = [row for row in rows if row['camera_id'] == camera_id]
        detections = sum(row['detected'] == '1' for row in camera_rows)

        top = axes[0, column]
        agg = aggregate_positions(camera_rows, 'raw')
        D.draw_warehouse(top, D.layout(), show_cameras=True, camera_labels=True, rack_alpha=0.80)
        coverage_scalar = top.scatter(
            [item[0] for item in agg], [item[1] for item in agg],
            c=[item[5] for item in agg], cmap=D.SUPPORT, norm=Normalize(0, 1),
            marker='s', s=31, linewidths=0, zorder=4,
        )
        top.set_title(
            f'{camera_title(camera_id)}: box returned in {detections}/{len(camera_rows)} views\n'
            f'{100 * detections / len(camera_rows):.0f}% of this camera\u2019s opportunities',
            fontsize=13, fontweight='bold',
        )

        bottom = axes[1, column]
        confidence = aggregate_confidence(camera_rows)
        seen = [item for item in confidence if math.isfinite(item[2])]
        blank = [item for item in confidence if not math.isfinite(item[2])]
        all_confidence.extend(item[2] for item in seen)
        D.draw_warehouse(bottom, D.layout(), show_cameras=True, camera_labels=True,
                         rack_alpha=0.80)
        if blank:
            bottom.scatter([item[0] for item in blank], [item[1] for item in blank],
                           marker='x', s=12, c='#aaa9a4', linewidths=0.7, zorder=3)
        if seen:
            confidence_scalar = bottom.scatter(
                [item[0] for item in seen], [item[1] for item in seen],
                c=[item[2] for item in seen], cmap='viridis',
                norm=Normalize(threshold, 1.0), marker='s', s=31, linewidths=0, zorder=4,
            )
        median = float(np.median(floats([row for row in camera_rows if row['detected'] == '1'],
                                        'confidence'))) if detections else math.nan
        low = sum(item[2] < 0.70 for item in seen)
        bottom.set_title(
            f'{camera_title(camera_id)}: median confidence {median:.2f}\n'
            f'{low}/{len(seen)} positions below 0.70',
            fontsize=13, fontweight='bold',
        )

    pooled = np.asarray(all_confidence, dtype=float)
    axes[0, -1].axis('off')
    axes[0, -1].text(0.02, 0.94, 'Top row: does a box come back?', fontsize=15.5,
                     fontweight='bold', va='top')
    axes[0, -1].text(
        0.02, 0.80,
        'Each square is one floor position. The robot was\n'
        'turned through 8 headings there.\n\n'
        'Dark blue: YOLO returned a robot box for all\n'
        '8 headings. White: it returned none.\n\n'
        'Racks and camera aim are drawn so a miss can be\n'
        'related to the physical warehouse.\n\n'
        'Read this before any accuracy figure: every error\n'
        'number in folders 02\u201306 is conditional on a box\n'
        'being returned here.',
        fontsize=12.5, va='top', linespacing=1.35,
    )
    axes[1, -1].axis('off')
    axes[1, -1].text(0.02, 0.94, 'Bottom row: how sure is it?', fontsize=15.5,
                     fontweight='bold', va='top')
    axes[1, -1].text(
        0.02, 0.80,
        'Median YOLO confidence over the headings that\n'
        'returned a box at that position.\n\n'
        f'The scale starts at the detector\u2019s own admission\n'
        f'threshold, {threshold:.2f}: a square at the bottom of the\n'
        'colour bar only just cleared it.\n\n'
        'Grey \u00d7 means no box at any heading, so there is no\n'
        'confidence to show. It is not low confidence.\n\n'
        f'Across all cameras the per-position median runs\n'
        f'{np.quantile(pooled, 0.05):.2f} (5th pct) to {np.quantile(pooled, 0.95):.2f} '
        f'(95th pct), median {np.median(pooled):.2f}.\n\n'
        'Confidence is high almost everywhere a box comes\n'
        'back, so it separates good readings from bad only\n'
        'weakly \u2014 see 06_what_a_gate_costs/.',
        fontsize=12.5, va='top', linespacing=1.35,
    )

    coverage_bar = fig.colorbar(coverage_scalar, ax=axes[0, :len(CAMERAS)].tolist(),
                                fraction=0.021, pad=0.010)
    coverage_bar.set_label('Fraction of 8 headings with a YOLO box', fontsize=12)
    confidence_bar = fig.colorbar(confidence_scalar, ax=axes[1, :len(CAMERAS)].tolist(),
                                  fraction=0.021, pad=0.010, extend='min')
    confidence_bar.set_label(f'Median YOLO confidence (admission threshold {threshold:.2f})',
                             fontsize=12)
    fig.suptitle(
        'Where does each camera produce a YOLO bounding box, and how confident is it?\n'
        'All returns; no post-detection admission gate',
        fontsize=20, fontweight='bold',
    )
    save(fig, out / '01_where_cameras_see/01_yolo_box_coverage_by_camera.png')


def draw_bias_maps(rows: list[dict[str, str]], bias_rows: list[dict[str, str]],
                   out: Path) -> None:
    """Where each interpretation puts the robot, across the warehouse floor.

    The first three rungs fit nothing, so they are drawn on every captured position and the
    map has no holes. The neural rung is drawn beside them because that is the comparison
    that matters, but it can only be scored off its training tiles, so its panel necessarily
    has checkerboard gaps. The panel titles say which grid each one uses.
    """
    bias_by_camera: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bias_rows:
        if row['split'] == 'test':
            bias_by_camera[row['camera_id']].append(row)

    for camera_id in CAMERAS:
        camera_rows = [row for row in rows if row['camera_id'] == camera_id]
        panels = [(method, label, 'every captured position', camera_rows, method)
                  for method, label, _colour in METHODS]
        panels.append(('nn', 'Neural bias update', 'held-out tiles only',
                       bias_by_camera[camera_id], 'nn'))

        fig, axes = plt.subplots(1, 4, figsize=(24.5, 7.8), constrained_layout=True)
        for ax, (_key, label, grid_note, source, method) in zip(axes, panels):
            agg = aggregate_positions(source, method)
            valid = [item for item in agg if math.isfinite(item[2])]
            empty = [item for item in agg if not math.isfinite(item[2])]
            clipped = 0
            scale = panel_scale(np.asarray([item[4] for item in valid]))
            D.draw_warehouse(ax, D.layout(), show_cameras=True, camera_labels=True,
                             rack_alpha=0.72)
            if empty:
                ax.scatter([x[0] for x in empty], [x[1] for x in empty], marker='x', s=12,
                           c='#aaa9a4', linewidths=0.7, zorder=3)
            if valid:
                x = np.asarray([item[0] for item in valid]); y = np.asarray([item[1] for item in valid])
                dx = np.asarray([item[2] for item in valid]); dy = np.asarray([item[3] for item in valid])
                shrink, clipped = drawn_shrink(dx, dy, scale['gain'])
                scalar = ax.quiver(
                    x, y, scale['gain'] * dx * shrink, scale['gain'] * dy * shrink,
                    np.asarray([item[4] for item in valid]), cmap='magma',
                    norm=Normalize(0, scale['cap']), angles='xy', scale_units='xy', scale=1,
                    width=0.0042, headwidth=3.5, headlength=4.2, zorder=5,
                )
                bar = fig.colorbar(scalar, ax=ax, fraction=0.042, pad=0.014, extend='max')
                bar.set_label(f'median error here (m), capped {scale["cap"]:.2f}', fontsize=8.8)
                bar.ax.tick_params(labelsize=8)
                scale_bar(ax, scale['gain'], layout=D.layout(), colour=D.INK)
            tail = f'  ·  {clipped} arrow(s) clipped' if valid and clipped else ''
            ax.set_title(
                f'{label}\n{grid_note}\n'
                f'median {100 * scale["median_m"]:.1f} cm  ·  arrows \u00d7{scale["gain"]:.0f}{tail}',
                fontsize=13, fontweight='bold')
        fig.suptitle(
            f'{camera_title(camera_id)} — where the same YOLO box places the robot, '
            'and what the network leaves behind',
            fontsize=20, fontweight='bold',
        )
        fig.text(
            0.5, -0.012,
            'At each floor cell the arrow starts at truth and shows the median over detected '
            'headings; grey × means no box at any heading.\n'
            '⚠ Every panel has its own arrow gain and colour cap, both printed above it, '
            'because the rungs differ about tenfold — compare the printed medians or the '
            'per-panel scale bar, never arrow length between panels.',
            ha='center', fontsize=10.8, linespacing=1.5,
        )
        save(fig, out / '02_the_error/02_error_fields_full_grid'
             / f'02_{camera_id}_error_field.png')


def draw_heading_effect(rows: list[dict[str, str]], out: Path) -> None:
    headings = sorted({int(row['heading_id']) for row in rows})
    degrees = np.asarray(headings, dtype=float) * (360.0 / max(len(headings), 1))
    values = {
        (camera, method, heading): floats([
            row for row in rows
            if row['camera_id'] == camera and int(row['heading_id']) == heading
            and row[f'{method}_valid'] == '1'
        ], f'{method}_along_m')
        for camera in CAMERAS for method, _label, _colour in METHODS for heading in headings
    }
    finite = np.asarray([abs(value) for group in values.values() for value in group], dtype=float)
    limit = max(0.25, float(np.quantile(finite, 0.98))) if finite.size else 0.5
    fig, axes = plt.subplots(5, 1, figsize=(12.8, 15.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, camera in zip(axes, CAMERAS):
        for method, label, colour in METHODS:
            medians = [
                float(np.median(values[(camera, method, heading)]))
                if values[(camera, method, heading)].size else math.nan
                for heading in headings
            ]
            ax.plot(degrees, medians, marker='o', ms=5, lw=2.2, color=colour, label=label)
        ax.axhline(0.0, color=D.INK, lw=1.0)
        ax.fill_between([-5, 365], [-limit, -limit], [0, 0], color='#f3ede8', zorder=0)
        ax.text(0.01, 0.88, camera_title(camera), transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')
        ax.set_ylabel('along-ray\nerror (m)')
        ax.set_xlim(-5, 320); ax.set_ylim(-limit, limit)
    axes[0].legend(loc='upper right', ncol=3, fontsize=10, frameon=True)
    axes[-1].set_xlabel('Robot heading (degrees)')
    axes[-1].set_xticks(degrees)
    fig.suptitle('Does turning the robot change where the box places its centre?\n'
                 'Negative means the reading falls toward the camera; each point pools warehouse positions\n'
                 'All YOLO returns retained; no post-detection admission gate',
                 fontsize=18, fontweight='bold')
    save(fig, out / '03_why_the_error/05_bias_by_robot_heading.png')


def draw_teaching_example(rows: list[dict[str, str]], capture: Path, out: Path) -> None:
    candidates = [
        row for row in rows
        if row['detected'] == '1' and all(row[f'{m}_valid'] == '1' for m, _l, _c in METHODS)
        and float(row['raw_error_m']) < 1.0 and float(row['hull_error_m']) < 1.0
    ]
    if not candidates:
        return
    target = float(np.median([float(row['raw_error_m']) for row in candidates]))
    sample = min(candidates, key=lambda row: abs(float(row['raw_error_m']) - target))
    image = cv2.imread(str(capture / sample['image']))
    if image is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(16, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.28, 1.0))
    ax_image = fig.add_subplot(gs[0, 0]); ax_map = fig.add_subplot(gs[0, 1])
    ax_image.imshow(image)
    x0, y0, x1, y1 = (float(sample[key]) for key in ('x0', 'y0', 'x1', 'y1'))
    ax_image.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='#ffdd00', lw=3))
    ax_image.scatter([float(sample['u_bbox_bottom'])], [float(sample['v_bbox_bottom'])],
                     s=85, c='#ffdd00', ec=D.INK, lw=1.2, zorder=5)
    ax_image.annotate('one measured point:\nbottom-centre of the YOLO box',
                      xy=(float(sample['u_bbox_bottom']), float(sample['v_bbox_bottom'])),
                      xytext=(0.04, 0.10), textcoords='axes fraction', fontsize=13,
                      bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d0cec7'),
                      arrowprops=dict(arrowstyle='->', color=D.INK, lw=1.8))
    ax_image.set_axis_off(); ax_image.set_title(f'{camera_title(sample["camera_id"])} RGB image', fontsize=16, fontweight='bold')

    D.draw_warehouse(ax_map, D.layout(), show_cameras=False, rack_alpha=0.72)
    tx, ty = float(sample['robot_x']), float(sample['robot_y'])
    ax_map.scatter([tx], [ty], s=130, c=D.INK, ec='white', lw=1.5, zorder=8, label='true centre')
    for method, label, colour in METHODS:
        ex, ey = float(sample[f'{method}_x']), float(sample[f'{method}_y'])
        ax_map.annotate('', xy=(ex, ey), xytext=(tx, ty),
                        arrowprops=dict(arrowstyle='-|>', color=colour, lw=3), zorder=7)
        ax_map.scatter([ex], [ey], s=65, c=colour, ec='white', lw=1, zorder=8,
                       label=f'{label}: {float(sample[f"{method}_error_m"]):.2f} m')
    span = max(1.6, 2.3 * max(float(sample[f'{method}_error_m']) for method, _l, _c in METHODS))
    ax_map.set_xlim(tx - span, tx + span); ax_map.set_ylim(ty - span, ty + span)
    camera = next(item for item in D.layout().cameras if item.name == sample['camera_id'][-1])
    toward = np.asarray([camera.x - tx, camera.y - ty], dtype=float)
    toward /= max(float(np.linalg.norm(toward)), 1e-12)
    ax_map.annotate(f'toward Camera {camera.name}', xy=(tx + toward[0] * span * 0.82,
                                                       ty + toward[1] * span * 0.82),
                    xytext=(tx + toward[0] * span * 0.28, ty + toward[1] * span * 0.28),
                    arrowprops=dict(arrowstyle='-|>', color=D.MUTED, lw=1.8),
                    color=D.MUTED, fontsize=11, ha='center')
    ax_map.legend(loc='upper left', frameon=True, fontsize=11)
    ax_map.set_title('Three consequences of that same box', fontsize=16, fontweight='bold')
    ax_map.text(0.02, 0.02, 'Hull residual is evaluated around the offline reference pose.',
                transform=ax_map.transAxes, fontsize=9.5, color=D.INK2, va='bottom')
    fig.suptitle('What one error arrow means', fontsize=21, fontweight='bold')
    save(fig, out / '00_what_the_data_is/00_what_one_arrow_means.png')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite this script\'s named PNGs in an existing output directory.',
    )
    args = parser.parse_args()
    capture = args.capture.expanduser().resolve()
    table = capture / 'observation_interpretations.csv'
    manifest = capture / 'observation_interpretations_manifest.json'
    if not table.is_file() or not manifest.is_file():
        raise RuntimeError('Run derive_interpretations.py first')
    if json.loads(manifest.read_text(encoding='utf-8')).get('status') != 'complete':
        raise RuntimeError('Interpretation manifest is not complete')
    rows = list(csv.DictReader(table.open(encoding='utf-8')))

    # The confidence colour scale is anchored on the detector's own frozen admission
    # threshold, never on a value chosen by looking at the resulting picture.
    detector_manifest = capture / 'bbox_detector_manifest.json'
    if not detector_manifest.is_file():
        raise RuntimeError(f'Missing detector manifest: {detector_manifest}')
    detector = json.loads(detector_manifest.read_text(encoding='utf-8'))
    if detector.get('status') != 'complete':
        raise RuntimeError('Detector manifest is not complete')
    threshold = float(detector['runtime']['confidence_threshold'])

    out = (args.out or DEFAULT_OUT).expanduser().resolve()
    if out.exists() and not args.overwrite:
        raise RuntimeError(f'Output already exists: {out}; pass --overwrite to refresh figures')
    out.mkdir(parents=True, exist_ok=True)
    draw_teaching_example(rows, capture, out)
    draw_coverage(rows, out, threshold)
    print(json.dumps({'output': str(out), 'figures': len(list(out.rglob('*.png')))}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
