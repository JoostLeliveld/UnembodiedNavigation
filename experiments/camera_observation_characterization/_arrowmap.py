#!/usr/bin/env python3
"""One full-size warehouse error map per sheet, readable without a caption.

Folders 02 and 03 used to stack four or eight of these maps onto a single sheet. That made
every arrow a few pixels long, and it put panels side by side that do not share an arrow
scale, so the one comparison a reader naturally makes -- arrow against arrow -- was the one
comparison the sheet could not support.

Each sheet written through this module therefore carries exactly ONE correction, ONE camera
and, in the heading folders, ONE robot heading. The map gets the whole page, and the numbers
that describe it sit beside it instead of in a shared footnote.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np

import style as D
from _fieldscale import drawn_shrink, panel_scale, scale_bar


@dataclass(frozen=True)
class Correction:
    """One rung of the correction ladder, and the floor it may honestly be scored on."""
    key: str            # column prefix in the interpretation tables
    folder: str         # subfolder name, numbered so the ladder reads in order
    slug: str           # filename fragment
    label: str          # what a reader is told it is
    subtitle: str       # what it costs / needs
    fits_anything: bool  # False -> scoreable on every captured position
    colour: str


CORRECTIONS = (
    Correction('raw', '01_no_correction', 'no_correction',
               'No correction: the raw YOLO box on the floor',
               'zero parameters; the box bottom projected straight onto the floor plane',
               False, '#eb6834'),
    Correction('fixed', '02_fixed_offset', 'fixed_offset',
               'One fixed offset: every reading pushed 30.9 cm away from its camera',
               'one constant, the same number everywhere in the building',
               False, '#2a78d6'),
    Correction('hull', '03_hull_residual', 'hull_residual',
               'Robot-shape model: what the box bottom should have been',
               'zero fitted parameters, but it needs the robot pose measured offline first',
               False, '#1baf7a'),
    Correction('learned', '04_learned_linear', 'learned_linear',
               'Learned straight-line correction, one per camera',
               'a handful of numbers per camera, read from the box alone',
               True, '#4a3aa7'),
    Correction('nn', '05_neural_network', 'neural_network',
               'Learned neural correction, one network for all cameras',
               'a small network, read from the box alone',
               True, '#d4267b'),
)
BY_KEY = {item.key: item for item in CORRECTIONS}
CAMERAS = tuple(f'camera_{letter}' for letter in 'ABCDE')
HEADING_STEP_DEG = 45.0


def camera_title(camera_id: str) -> str:
    return f'Camera {camera_id[-1]}'


def heading_label(heading_id: int, degrees: float) -> str:
    return f'facing {degrees:.0f}°'


def heading_slug(heading_id: int, degrees: float) -> str:
    return f'h{heading_id}_{round(degrees):03d}deg'


# --------------------------------------------------------------------------------------
# turning table rows into map cells


def _float(row: dict, field: str) -> float:
    value = row.get(field, '')
    return math.nan if value in ('', 'nan') else float(value)


def position_medians(rows: list[dict], key: str) -> tuple[list[dict], list[tuple[float, float]]]:
    """One arrow per floor position: the median over the headings that returned a box.

    A position where no heading produced a usable reading becomes a blank marker, never an
    arrow of length zero -- an absent reading is not an accurate reading.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row['position_id']].append(row)
    cells: list[dict] = []
    blanks: list[tuple[float, float]] = []
    for group in groups.values():
        first = group[0]
        x, y = float(first['robot_x']), float(first['robot_y'])
        hits = [row for row in group if row[f'{key}_valid'] == '1']
        if not hits:
            blanks.append((x, y))
            continue
        cells.append({
            'x': x, 'y': y,
            'dx': float(np.median([_float(row, f'{key}_dx') for row in hits])),
            'dy': float(np.median([_float(row, f'{key}_dy') for row in hits])),
            'error': float(np.median([_float(row, f'{key}_error_m') for row in hits])),
            'along': float(np.median([_float(row, f'{key}_along_m') for row in hits])),
            'across': float(np.median([_float(row, f'{key}_across_m') for row in hits])),
            'headings': len(hits),
            'opportunities': len(group),
        })
    return cells, blanks


def per_observation(rows: list[dict], key: str) -> tuple[list[dict], list[tuple[float, float]]]:
    """One arrow per captured reading -- used where a sheet already fixes the heading."""
    cells: list[dict] = []
    blanks: list[tuple[float, float]] = []
    for row in rows:
        x, y = float(row['robot_x']), float(row['robot_y'])
        if row[f'{key}_valid'] != '1':
            blanks.append((x, y))
            continue
        cells.append({
            'x': x, 'y': y,
            'dx': _float(row, f'{key}_dx'), 'dy': _float(row, f'{key}_dy'),
            'error': _float(row, f'{key}_error_m'),
            'along': _float(row, f'{key}_along_m'),
            'across': _float(row, f'{key}_across_m'),
            'headings': 1, 'opportunities': 1,
        })
    return cells, blanks


def stats(cells: list[dict]) -> dict:
    if not cells:
        return {'n': 0, 'median': math.nan, 'p90': math.nan, 'worst': math.nan,
                'along': math.nan, 'across': math.nan}
    error = np.asarray([cell['error'] for cell in cells], dtype=float)
    along = np.asarray([cell['along'] for cell in cells], dtype=float)
    across = np.asarray([cell['across'] for cell in cells], dtype=float)
    return {
        'n': int(error.size),
        'median': float(np.median(error)),
        'p90': float(np.quantile(error, 0.90)),
        'worst': float(error.max()),
        'along': float(np.median(along)),
        'across': float(np.median(across)),
    }


def direction_phrase(along_m: float, across_m: float) -> str:
    """A finished clause saying where the reading lands, from the signed medians.

    Radial (along the camera ray) and sideways (across it) are reported separately because a
    single constant offset can only remove the radial part -- which is the whole reason the
    fixed-offset rung sits where it does on the ladder.
    """
    if not math.isfinite(along_m):
        return 'no camera view here returned a usable reading'
    radial, lateral = abs(along_m), abs(across_m)
    toward = ('short of the robot, toward the camera' if along_m < 0
              else 'past the robot, away from the camera')
    side = 'left' if across_m > 0 else 'right'
    if radial < 0.02 and lateral < 0.02:
        return 'no consistent direction is left \u2014 the typical lean is under 2 cm either way'
    if radial >= 2.0 * lateral:
        return f'what is left leans {100 * radial:.1f} cm {toward}'
    if lateral > 2.0 * radial:
        return (f'what is left leans {100 * lateral:.1f} cm sideways, to the {side} of the '
                f'camera ray')
    return (f'what is left leans {100 * radial:.1f} cm {toward} and {100 * lateral:.1f} cm '
            f'sideways to the {side}')


# --------------------------------------------------------------------------------------
# drawing


def draw_field(fig, ax, cells: list[dict], blanks, *, colour_cap: float | None = None,
               arrow_gain: float | None = None, cbar_fraction: float = 0.040,
               cbar_label: str | None = None, marker_size: float = 15.0,
               cbar: bool = True) -> dict:
    """The warehouse, the blanks, and one arrow per cell. Returns the scale it chose."""
    D.draw_warehouse(ax, D.layout(), show_cameras=True, camera_labels=True, rack_alpha=0.72)
    if blanks:
        ax.scatter([item[0] for item in blanks], [item[1] for item in blanks],
                   marker='x', s=marker_size, c='#aaa9a4', linewidths=0.8, zorder=3)
    scale = panel_scale(np.asarray([cell['error'] for cell in cells], dtype=float))
    if arrow_gain is not None:
        scale = dict(scale, gain=float(arrow_gain))
    if colour_cap is not None:
        scale = dict(scale, cap=float(colour_cap))
    scale['clipped'] = 0
    scale['scalar'] = None
    if not cells:
        return scale
    x = np.asarray([cell['x'] for cell in cells], dtype=float)
    y = np.asarray([cell['y'] for cell in cells], dtype=float)
    dx = np.asarray([cell['dx'] for cell in cells], dtype=float)
    dy = np.asarray([cell['dy'] for cell in cells], dtype=float)
    error = np.asarray([cell['error'] for cell in cells], dtype=float)
    shrink, clipped = drawn_shrink(dx, dy, scale['gain'])
    scalar = ax.quiver(
        x, y, scale['gain'] * dx * shrink, scale['gain'] * dy * shrink, error,
        cmap='magma', norm=Normalize(0.0, scale['cap']), angles='xy', scale_units='xy',
        scale=1, width=0.0040, headwidth=3.6, headlength=4.3, zorder=5,
    )
    scale_bar(ax, scale['gain'], layout=D.layout(), colour=D.INK, fontsize=10.0)
    scale['clipped'] = clipped
    scale['scalar'] = scalar
    if cbar:
        bar = fig.colorbar(scalar, ax=ax, fraction=cbar_fraction, pad=0.012, extend='max')
        bar.set_label(cbar_label or 'how far the reading missed (cm)', fontsize=10.8)
        bar.ax.tick_params(labelsize=9.5)
        bar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f'{100 * value:.0f}'))
    return scale


def heading_dial(fig):
    """A page-corner badge for the heading dial.

    It sits in the figure's own top-left corner rather than inside the note column: the note
    text is a different length on every sheet, and a dial anchored to it landed on top of the
    first three lines on the longer ones.
    """
    return fig.add_axes([0.006, 0.700, 0.047, 0.070])


def draw_compass(ax, degrees: float) -> None:
    """A small dial saying which way the robot was pointing, so 'h3' never has to be read."""
    ax.set_aspect('equal')
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(plt.Circle((0, 0), 1.0, facecolor='#f2f0ea', edgecolor='#d0cec7', lw=1.2))
    for tick in range(0, 360, 45):
        angle = math.radians(tick)
        ax.plot([0.88 * math.cos(angle), math.cos(angle)],
                [0.88 * math.sin(angle), math.sin(angle)], color='#b8b7ae', lw=1.0)
    ax.text(1.18, 0, '+x', fontsize=8.5, color=D.MUTED, va='center')
    ax.text(0, 1.18, '+y', fontsize=8.5, color=D.MUTED, ha='center')
    angle = math.radians(degrees)
    ax.annotate('', xy=(0.86 * math.cos(angle), 0.86 * math.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=D.ROBOT, lw=2.6, shrinkA=0, shrinkB=0))
    ax.set_title(f'robot faces {degrees:.0f}°', fontsize=9.8, fontweight='bold', pad=3)


def draw_histogram(ax, values_cm: np.ndarray, *, colour: str, xlabel: str,
                   title: str) -> None:
    ax.grid(color='#e4e2dc', lw=0.8)
    ax.set_axisbelow(True)
    if values_cm.size:
        limit = max(1.0, float(np.quantile(values_cm, 0.99)))
        bins = np.linspace(0, limit, 30)
        counts, _, _ = ax.hist(np.minimum(values_cm, limit), bins=bins, color=colour,
                               alpha=0.72, edgecolor='white', linewidth=0.5)
        median, p90 = float(np.median(values_cm)), float(np.quantile(values_cm, 0.90))
        ax.axvline(median, color=D.INK, lw=1.9)
        ax.axvline(p90, color=D.INK, lw=1.4, linestyle='--')
        headroom = max(float(counts.max()), 1.0) * 1.34
        ax.set_ylim(0, headroom)
        ax.set_xlim(0, limit)
        ax.annotate(f'half are closer\nthan {median:.1f} cm', xy=(median, headroom * 0.755),
                    xytext=(median + 0.03 * limit, headroom * 0.99), fontsize=9.6,
                    va='top', ha='left', fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=D.INK, lw=0.9))
        ax.annotate(f'1 in 10 is worse\nthan {p90:.1f} cm', xy=(p90, headroom * 0.40),
                    xytext=(min(p90 + 0.03 * limit, 0.62 * limit), headroom * 0.62),
                    fontsize=9.6, va='top', ha='left', color=D.INK2,
                    arrowprops=dict(arrowstyle='-', color=D.INK2, lw=0.9))
    ax.set_xlabel(xlabel, fontsize=10.2)
    ax.set_ylabel('floor positions', fontsize=10.2)
    ax.set_title(title, fontsize=11.4, fontweight='bold')
    ax.tick_params(labelsize=9.4)


def note_block(ax, title: str, lines: list[str], *, fontsize: float = 11.4) -> None:
    ax.axis('off')
    ax.text(0.0, 1.0, title, fontsize=14.2, fontweight='bold', va='top')
    ax.text(0.0, 0.895, '\n'.join(lines), fontsize=fontsize, va='top', linespacing=1.42)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def open_sheet(*, left: float = 0.012):
    """The standard page used by every single-map sheet in folders 02 and 03.

    Map on the left, words top right, the distribution of the same quantity bottom right.
    The proportions are chosen so the map axes box has roughly the warehouse's own aspect
    ratio -- without that the box is much wider than the building it draws, and the colour
    bar, which attaches to the box rather than to the drawing, floats in empty space halfway
    to the text column.
    """
    fig = plt.figure(figsize=(18.6, 11.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.66, 1.0), height_ratios=(1.0, 0.55),
                            wspace=0.10, hspace=0.20, left=left, right=0.988,
                            top=0.905, bottom=0.082)
    return (fig, fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1]))


def sheet(*, path: Path, title: str, finding: str, provenance: str, cells: list[dict],
          blanks, note_title: str, note_lines: list[str], hist_xlabel: str,
          hist_title: str, colour: str, grid_short: str = '', heading_deg: float | None = None,
          colour_cap: float | None = None, arrow_gain: float | None = None,
          footer: str | None = None) -> dict:
    """One correction, one camera, one page of arrows, with its own numbers beside it."""
    fig, ax_map, ax_note, ax_hist = open_sheet(left=0.060 if heading_deg is not None else 0.012)
    scale = draw_field(fig, ax_map, cells, blanks, colour_cap=colour_cap,
                       arrow_gain=arrow_gain)
    summary = stats(cells)
    pieces = [piece for piece in (
        grid_short,
        f'{summary["n"]} arrows drawn',
        f'arrows {scale["gain"]:.0f}× life size',
        f'colour stops at {100 * scale["cap"]:.0f} cm',
        f'{scale["clipped"]} arrow(s) shortened to fit' if scale['clipped'] else '',
    ) if piece]
    ax_map.set_title('  ·  '.join(pieces), fontsize=12.0, fontweight='bold', pad=7)
    note_block(ax_note, note_title, note_lines)
    if heading_deg is not None:
        draw_compass(heading_dial(fig), heading_deg)
    draw_histogram(ax_hist, 100 * np.asarray([cell['error'] for cell in cells], dtype=float),
                   colour=colour, xlabel=hist_xlabel, title=hist_title)
    fig.suptitle(f'{title}\n{finding}', fontsize=16.5, fontweight='bold', y=0.988,
                 linespacing=1.5)
    tail = '   —   '.join(piece for piece in (provenance, footer) if piece)
    if tail:
        fig.text(0.5, 0.014, tail, ha='center', fontsize=9.9, color=D.INK2)
    save(fig, path)
    return summary
