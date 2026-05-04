#!/usr/bin/env python3
"""Plot GP P_conservative_map as a heatmap with task corridors and shelf footprint.

Usage:
    python plot_gp_map_with_tasks.py \
        --artifact logs/visibility_comparison/current_gp/oracle_visibility_gp.npz \
        --tasks-yaml src/experiments/config/tasks.yaml \
        --out gp_map_tasks.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'visibility_comparison'))
import common  # noqa: F401 — imported for side-effects / path constants

# ── constants ────────────────────────────────────────────────────────────────

SHELF_X = (-0.85, 0.75)   # x bounds of shelf footprint
SHELF_Y = (-0.29, -0.01)  # y bounds of shelf footprint

TASK_COLORS = {
    'shadow_tradeoff_a': '#e41a1c',   # red
    'shadow_tradeoff_b': '#377eb8',   # blue
    'sanity_open':       '#4daf4a',   # green
}

TASK_LABELS = {
    'shadow_tradeoff_a': 'Task A (shadow tradeoff)',
    'shadow_tradeoff_b': 'Task B (shadow tradeoff)',
    'sanity_open':       'Task S (sanity open)',
}

# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True,
                        help='Path to GP .npz artifact (must exist).')
    parser.add_argument('--tasks-yaml', required=True,
                        help='Path to tasks.yaml (must exist).')
    parser.add_argument('--out', required=True,
                        help='Output PNG path.')
    return parser.parse_args()


def _load_gp(artifact_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xs, ys (1-D grid axes) and P_conservative_map (2-D, shape ys×xs)."""
    if not artifact_path.is_file():
        print(f'ERROR: GP artifact not found: {artifact_path}', file=sys.stderr)
        sys.exit(1)
    with np.load(artifact_path, allow_pickle=False) as data:
        if 'xs' not in data.files or 'ys' not in data.files:
            print(
                f'ERROR: GP artifact missing xs/ys keys. Found: {data.files}',
                file=sys.stderr,
            )
            sys.exit(1)
        xs = np.asarray(data['xs'], dtype=float)
        ys = np.asarray(data['ys'], dtype=float)
        # Prefer P_conservative_map; fall back gracefully.
        for key in ('P_conservative_map', 'P_conservative_plan_map', 'P_mean_map', 'P_map'):
            if key in data.files:
                p_map = np.asarray(data[key], dtype=float)
                print(f'Loaded GP map key: {key}  shape={p_map.shape}')
                break
        else:
            print(
                f'ERROR: GP artifact has none of the expected map keys. Found: {data.files}',
                file=sys.stderr,
            )
            sys.exit(1)
    return xs, ys, p_map


def _load_tasks(tasks_yaml: Path) -> dict[str, dict]:
    """Return {task_name: {start: {x,y}, goal: {x,y}}} for the 3 IWAI tasks."""
    if not tasks_yaml.is_file():
        print(f'ERROR: tasks.yaml not found: {tasks_yaml}', file=sys.stderr)
        sys.exit(1)
    payload = yaml.safe_load(tasks_yaml.read_text(encoding='utf-8'))
    # tasks are under payload['tasks'][world_key]
    tasks_section = payload.get('tasks', {})
    all_tasks: dict[str, dict] = {}
    for world_key, task_list in tasks_section.items():
        if not isinstance(task_list, list):
            continue
        for entry in task_list:
            name = entry.get('name', '')
            if name in TASK_COLORS:
                all_tasks[name] = {
                    'start': {'x': float(entry['start']['x']), 'y': float(entry['start']['y'])},
                    'goal':  {'x': float(entry['goal']['x']),  'y': float(entry['goal']['y'])},
                }
    missing = set(TASK_COLORS) - set(all_tasks)
    if missing:
        print(f'ERROR: tasks.yaml is missing task entries: {missing}', file=sys.stderr)
        sys.exit(1)
    return all_tasks


def _draw_arrow(ax, sx: float, sy: float, gx: float, gy: float, color: str, label: str) -> None:
    """Draw a start→goal corridor arrow on ax."""
    ax.annotate(
        '',
        xy=(gx, gy),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle='->', color=color, lw=1.8,
            connectionstyle='arc3,rad=0.0',
        ),
        zorder=5,
    )


def main() -> int:
    args = _parse_args()
    artifact_path = Path(args.artifact).expanduser().resolve()
    tasks_yaml = Path(args.tasks_yaml).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    # ── load data ──────────────────────────────────────────────────────────
    xs, ys, p_map = _load_gp(artifact_path)
    tasks = _load_tasks(tasks_yaml)

    # ── figure ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)

    # Heatmap: p_map rows=ys, cols=xs → imshow with origin='lower'
    im = ax.imshow(
        p_map,
        origin='lower',
        extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
        aspect='equal',
        zorder=0,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r'$\rho_\mathrm{plan}$ (P_conservative_map)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Shelf footprint rectangle
    shelf_rect = mpatches.Rectangle(
        (SHELF_X[0], SHELF_Y[0]),
        SHELF_X[1] - SHELF_X[0],
        SHELF_Y[1] - SHELF_Y[0],
        linewidth=1.5,
        edgecolor='white',
        facecolor='none',
        linestyle='--',
        zorder=4,
        label='Shelf footprint',
    )
    ax.add_patch(shelf_rect)
    ax.text(
        0.5 * (SHELF_X[0] + SHELF_X[1]),
        SHELF_Y[1] + 0.05,
        'shelf',
        color='white',
        fontsize=7,
        ha='center',
        va='bottom',
        zorder=6,
    )

    # Task corridors
    legend_handles = [shelf_rect]
    for task_name, color in TASK_COLORS.items():
        info = tasks[task_name]
        sx, sy = info['start']['x'], info['start']['y']
        gx, gy = info['goal']['x'],  info['goal']['y']
        _draw_arrow(ax, sx, sy, gx, gy, color, TASK_LABELS[task_name])
        # Start marker
        ax.scatter([sx], [sy], s=55, color=color, marker='o',
                   edgecolors='white', linewidths=0.8, zorder=7)
        # Goal marker
        ax.scatter([gx], [gy], s=65, color=color, marker='*',
                   edgecolors='white', linewidths=0.6, zorder=7)
        # Small label offsets
        ax.text(sx - 0.1, sy + 0.08, 'S', color=color, fontsize=7, fontweight='bold',
                ha='right', va='bottom', zorder=8)
        ax.text(gx + 0.1, gy + 0.08, 'G', color=color, fontsize=7, fontweight='bold',
                ha='left', va='bottom', zorder=8)
        # Legend proxy
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=2,
                       label=TASK_LABELS[task_name])
        )

    # Legend proxies for start/goal markers
    legend_handles.append(
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
                   markeredgecolor='white', markersize=7, label='Start (S)')
    )
    legend_handles.append(
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='gray',
                   markeredgecolor='white', markersize=9, label='Goal (G)')
    )

    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(float(ys[0]), float(ys[-1]))
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('y [m]', fontsize=9)
    ax.set_title('GP visibility map with task corridors', fontsize=10)
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        fontsize=7,
        frameon=True,
        framealpha=0.85,
        fancybox=False,
        borderpad=0.4,
        handlelength=1.6,
        labelspacing=0.3,
    )
    ax.tick_params(labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
