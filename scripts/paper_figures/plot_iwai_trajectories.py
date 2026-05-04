#!/usr/bin/env python3
"""Overlay robot trajectories from all 3 conditions on the GP visibility map for one task.

Usage:
    python plot_iwai_trajectories.py \
        --campaign-log logs/visibility_comparison/iwai_campaign/campaign_log.json \
        --artifact logs/visibility_comparison/current_gp/oracle_visibility_gp.npz \
        --task shadow_tradeoff_a \
        --out figures/trajectories_shadow_tradeoff_a.png

Hard-fail rules:
  - --campaign-log, --artifact must exist.
  - --task must be present in campaign_log.json.
  - experiment.csv must be present for at least one run of each condition.
  - Position columns (x/y or robot_x/robot_y) must exist in each CSV that is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'visibility_comparison'))
import common  # noqa: F401

# ── constants ─────────────────────────────────────────────────────────────────

CONDITIONS = ['C1', 'C2', 'C3']
CONDITION_COLORS = {
    'C1': '#4393c3',   # blue
    'C2': '#d6604d',   # red/orange
    'C3': '#74c476',   # green
}
CONDITION_LABELS = {
    'C1': 'C1 (constant_R_efe)',
    'C2': 'C2 (visibility_aware_efe)',
    'C3': 'C3 (risk_only_ablation)',
}

SHELF_X = (-0.85, 0.75)
SHELF_Y = (-0.29, -0.01)

# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-log', required=True,
                        help='Path to campaign_log.json (must exist).')
    parser.add_argument('--artifact', required=True,
                        help='Path to GP .npz artifact (must exist).')
    parser.add_argument('--task', required=True,
                        help='Task name to plot (e.g. shadow_tradeoff_a).')
    parser.add_argument('--out', required=True,
                        help='Output PNG path.')
    return parser.parse_args()


def _load_gp(artifact_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, ys, P_conservative_map)."""
    if not artifact_path.is_file():
        print(f'ERROR: GP artifact not found: {artifact_path}', file=sys.stderr)
        sys.exit(1)
    with np.load(artifact_path, allow_pickle=False) as data:
        missing_axes = [k for k in ('xs', 'ys') if k not in data.files]
        if missing_axes:
            print(
                f'ERROR: GP artifact missing keys {missing_axes}. Found: {data.files}',
                file=sys.stderr,
            )
            sys.exit(1)
        xs = np.asarray(data['xs'], dtype=float)
        ys = np.asarray(data['ys'], dtype=float)
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


def _load_campaign_log(log_path: Path) -> dict:
    if not log_path.is_file():
        print(f'ERROR: campaign_log.json not found: {log_path}', file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(log_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        print(f'ERROR: failed to read campaign_log.json: {exc}', file=sys.stderr)
        sys.exit(1)


def _find_position_columns(headers: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Detect which x/y column names are present in a CSV header."""
    for xc, yc in (('truth_x', 'truth_y'), ('x', 'y'), ('robot_x', 'robot_y')):
        if xc in headers and yc in headers:
            return xc, yc
    return None, None


def _load_trajectory(run_dir: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load (x_array, y_array) from experiment.csv in run_dir; return None on failure."""
    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        return None
    rows = []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        xcol, ycol = _find_position_columns(headers)
        if xcol is None:
            print(
                f'  WARNING: no position columns (x/y, robot_x/robot_y, truth_x/truth_y) '
                f'in {csv_path}. Available: {headers}',
                file=sys.stderr,
            )
            return None
        for row in reader:
            try:
                xv = float(row[xcol])
                yv = float(row[ycol])
            except (TypeError, ValueError):
                continue
            if math.isfinite(xv) and math.isfinite(yv):
                rows.append((xv, yv))
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def _is_goal_reached(run_dir: Path) -> bool:
    """Return True if run_summary.json says goal_reached."""
    summary_path = run_dir / 'run_summary.json'
    if not summary_path.is_file():
        # Search recursively
        candidates = sorted(run_dir.rglob('run_summary.json'))
        if not candidates:
            return False
        summary_path = candidates[-1]
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        bool(summary.get('goal_reached', False))
        or str(summary.get('completion_reason', '')).strip() == 'goal_reached'
    )


def _task_start_goal(campaign_log: dict, task: str) -> tuple[Optional[dict], Optional[dict]]:
    """Extract start/goal from any entry that has the task metadata."""
    for entry in campaign_log.values():
        if str(entry.get('task', '')) != task:
            continue
        start = entry.get('start') or entry.get('task_start')
        goal = entry.get('goal') or entry.get('task_goal')
        if start and goal:
            return start, goal
    return None, None


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    artifact_path = Path(args.artifact).expanduser().resolve()
    log_path = Path(args.campaign_log).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    task = str(args.task).strip()

    # ── load GP ───────────────────────────────────────────────────────────────
    xs, ys, p_map = _load_gp(artifact_path)

    # ── load campaign log ─────────────────────────────────────────────────────
    campaign_log = _load_campaign_log(log_path)

    # Filter entries by task
    task_entries = {k: v for k, v in campaign_log.items() if str(v.get('task', '')) == task}
    if not task_entries:
        print(f'ERROR: no entries found for task={task!r} in campaign_log.json.', file=sys.stderr)
        print(f'  Available tasks: {sorted({v.get("task") for v in campaign_log.values()})}',
              file=sys.stderr)
        sys.exit(1)

    # ── collect trajectories per condition ────────────────────────────────────
    # Structure: {condition: [(x_arr, y_arr, goal_reached_bool), ...]}
    trajs: dict[str, list] = {c: [] for c in CONDITIONS}

    for entry in task_entries.values():
        cond = str(entry.get('condition', ''))
        if cond not in CONDITIONS:
            continue
        outcome = str(entry.get('outcome', ''))
        if outcome == 'infra_invalid':
            continue
        run_dir_str = str(entry.get('run_dir', ''))
        if not run_dir_str:
            continue
        run_dir = Path(run_dir_str)
        result = _load_trajectory(run_dir)
        if result is None:
            print(f'  Skipping {cond} run_dir={run_dir}: no trajectory data.', file=sys.stderr)
            continue
        x_arr, y_arr = result
        goal_reached = _is_goal_reached(run_dir)
        trajs[cond].append((x_arr, y_arr, goal_reached))
        status = 'goal_reached' if goal_reached else 'other'
        print(f'  Loaded {cond} ({status}): {len(x_arr)} points from {run_dir.name}')

    # Fail if no condition has any trajectory
    total_trajs = sum(len(v) for v in trajs.values())
    if total_trajs == 0:
        print(
            f'ERROR: no trajectory data found for task={task!r}. '
            'Check that run_dir paths in campaign_log.json exist and '
            'that experiment.csv files contain position columns.',
            file=sys.stderr,
        )
        sys.exit(1)

    # ── try to find start/goal from campaign log ───────────────────────────────
    start_info, goal_info = _task_start_goal(campaign_log, task)

    # Fallback: use TASK_INFO from compute_paper_metrics
    FALLBACK_TASK_INFO = {
        'shadow_tradeoff_a': {'start': (-2.0, 0.5),  'goal': (2.0, 0.5)},
        'shadow_tradeoff_b': {'start': (-2.0, -1.0), 'goal': (2.0, 0.5)},
        'sanity_open':       {'start': (-2.0, -1.5), 'goal': (2.0, -1.5)},
    }
    if start_info is None or goal_info is None:
        fb = FALLBACK_TASK_INFO.get(task)
        if fb:
            start_info = {'x': fb['start'][0], 'y': fb['start'][1]}
            goal_info  = {'x': fb['goal'][0],  'y': fb['goal'][1]}
            print(f'  Using fallback start/goal for task={task!r}')

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(6.0, 5.5), constrained_layout=True)

    # GP heatmap background
    im = ax.imshow(
        p_map,
        origin='lower',
        extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
        aspect='equal',
        alpha=0.75,
        zorder=0,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
    cbar.set_label(r'$\rho_\mathrm{plan}$ (P_conservative_map)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Shelf footprint
    shelf_rect = mpatches.Rectangle(
        (SHELF_X[0], SHELF_Y[0]),
        SHELF_X[1] - SHELF_X[0],
        SHELF_Y[1] - SHELF_Y[0],
        linewidth=1.5,
        edgecolor='white',
        facecolor='white',
        alpha=0.35,
        linestyle='--',
        zorder=3,
        label='Shelf footprint',
    )
    ax.add_patch(shelf_rect)
    ax.text(
        0.5 * (SHELF_X[0] + SHELF_X[1]),
        SHELF_Y[1] + 0.06,
        'shelf',
        color='white',
        fontsize=7,
        ha='center',
        va='bottom',
        zorder=5,
    )

    # Trajectories
    legend_handles = [shelf_rect]
    for cond in CONDITIONS:
        color = CONDITION_COLORS[cond]
        runs = trajs[cond]
        if not runs:
            continue
        added_label = False
        for x_arr, y_arr, goal_reached in runs:
            linestyle = '-' if goal_reached else '--'
            lw = 1.6 if goal_reached else 1.2
            alpha = 0.90 if goal_reached else 0.65
            label = CONDITION_LABELS[cond] if not added_label else None
            ax.plot(
                x_arr, y_arr,
                color=color,
                linewidth=lw,
                linestyle=linestyle,
                alpha=alpha,
                label=label,
                zorder=4,
            )
            added_label = True

        # Legend proxy (solid line representative)
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=2, label=CONDITION_LABELS[cond])
        )

    # Line style legend proxies
    legend_handles.append(
        plt.Line2D([0], [0], color='#888888', linewidth=1.5, linestyle='-',
                   label='Goal reached')
    )
    legend_handles.append(
        plt.Line2D([0], [0], color='#888888', linewidth=1.2, linestyle='--',
                   label='Other outcome')
    )

    # Start / goal markers
    if start_info is not None:
        sx, sy = float(start_info['x']), float(start_info['y'])
        ax.scatter([sx], [sy], s=80, color='#0a8f2a', marker='o',
                   edgecolors='white', linewidths=0.8, zorder=8, label='Start')
        ax.text(sx - 0.1, sy + 0.10, 'start', color='white', fontsize=7,
                ha='right', va='bottom', zorder=9)
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='none',
                       markerfacecolor='#0a8f2a', markeredgecolor='white',
                       markersize=8, label='Start')
        )
    if goal_info is not None:
        gx, gy = float(goal_info['x']), float(goal_info['y'])
        ax.scatter([gx], [gy], s=90, color='#e41a1c', marker='*',
                   edgecolors='white', linewidths=0.6, zorder=8, label='Goal')
        ax.text(gx + 0.1, gy + 0.10, 'goal', color='white', fontsize=7,
                ha='left', va='bottom', zorder=9)
        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='none',
                       markerfacecolor='#e41a1c', markeredgecolor='white',
                       markersize=9, label='Goal')
        )

    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(float(ys[0]), float(ys[-1]))
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('y [m]', fontsize=9)
    ax.set_title(f'IWAI trajectories — task: {task}', fontsize=10)
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        fontsize=7,
        frameon=True,
        framealpha=0.85,
        fancybox=False,
        borderpad=0.4,
        handlelength=1.8,
        labelspacing=0.3,
    )
    ax.tick_params(labelsize=8)
    ax.grid(True, color='#444444', linewidth=0.3, alpha=0.4, zorder=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
