#!/usr/bin/env python3
"""Section 0a GP artifact verification for the IWAI campaign.

Loads the fitted GP artifact and checks the three required conditions before
any campaign run:

  1. Shadow contrast: rho_plan < 0.3 directly behind the shelf and
     rho_plan > 0.7 in the open region below the shelf.
     Contrast >= 0.4 units required.

  2. Start/goal visibility: rho_plan > 0.6 at every task start and goal.

  3. Shadow path intersection: the straight line from start to goal for
     shadow-tradeoff tasks passes through >= 1.5 m of rho_plan < 0.35 region.

Produces a plot: gp_verification.png with three panels
  (a) rho_plan posterior mean
  (b) rho_plan conservative planning map (what the planner sees)
  (c) rho_plan uncertainty

Usage:
    python verify_gp_artifact.py --artifact path/to/gp.npz [--out gp_verification.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Task definitions matching tasks.yaml
TASKS = {
    'shadow_tradeoff_a': {
        'start': (-2.0, 0.5),
        'goal':  (2.0,  0.5),
        'is_shadow_task': True,
    },
    'shadow_tradeoff_b': {
        'start': (-2.0, -1.0),
        'goal':  (2.0,  0.5),
        'is_shadow_task': True,
    },
    'sanity_open': {
        'start': (-2.0, -1.5),
        'goal':  (2.0,  -1.5),
        'is_shadow_task': False,
    },
}

# Protocol thresholds (Section 0a)
RHO_VISIBLE_THRESHOLD = 0.6      # start/goal must exceed this
RHO_SHADOW_MAX = 0.35            # "in shadow" if below this
SHADOW_CONTRAST_MIN = 0.4        # required contrast between shadow and open regions
SHADOW_PATH_MIN_M = 1.5          # minimum shadow-path length for shadow tasks
OPEN_REGION_RHO_MIN = 0.7        # open region reference (below shelf, camera side)
SHADOW_REGION_RHO_MAX = 0.3      # shadow region reference (behind shelf)


def _load_artifact(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        xs = np.asarray(data['xs'], dtype=float)
        ys = np.asarray(data['ys'], dtype=float)
        p_mean = np.asarray(data['P_mean_map'], dtype=float)
        p_plan = np.asarray(data.get('P_conservative_plan_map', data.get('P_conservative_map')), dtype=float)
        p_std = np.asarray(data.get('P_std_map', np.zeros_like(p_mean)), dtype=float)
    return {'xs': xs, 'ys': ys, 'p_mean': p_mean, 'p_plan': p_plan, 'p_std': p_std}


def _make_interpolator(xs: np.ndarray, ys: np.ndarray, values: np.ndarray):
    # values expected shape (ny, nx) — RegularGridInterpolator needs (y, x) order
    return RegularGridInterpolator(
        (ys, xs), values, method='linear', bounds_error=False, fill_value=np.nan
    )


def _query(interp, points: np.ndarray) -> np.ndarray:
    # points: (N, 2) as (x, y) — swap to (y, x) for interpolator
    return interp(points[:, [1, 0]])


def _path_shadow_length(interp, start: tuple, goal: tuple, n_samples: int = 500, dt: float = 0.0) -> float:
    """Approximate length of straight-line path passing through rho < RHO_SHADOW_MAX."""
    xs = np.linspace(start[0], goal[0], n_samples)
    ys = np.linspace(start[1], goal[1], n_samples)
    pts = np.column_stack([xs, ys])
    rho = _query(interp, pts)
    total_length = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
    segment = total_length / (n_samples - 1)
    in_shadow = np.isfinite(rho) & (rho < RHO_SHADOW_MAX)
    return float(np.sum(in_shadow) * segment)


def _check_contrast(interp) -> tuple[float, float, float]:
    """Sample shadow and open reference regions; return (shadow_val, open_val, contrast)."""
    # Shadow reference: region directly behind the shelf relative to camera (upper area)
    # Using x in [0, 1.5], y in [0.3, 0.8]
    xs_s = np.linspace(0.0, 1.5, 20)
    ys_s = np.linspace(0.3, 0.8, 20)
    pts_shadow = np.array([[x, y] for x in xs_s for y in ys_s])
    rho_shadow = np.nanmean(_query(interp, pts_shadow))

    # Open reference: below the shelf on camera side, x in [-2, 2], y in [-1.5, -0.5]
    xs_o = np.linspace(-2.0, 2.0, 30)
    ys_o = np.linspace(-1.5, -0.5, 20)
    pts_open = np.array([[x, y] for x in xs_o for y in ys_o])
    rho_open = np.nanmean(_query(interp, pts_open))

    contrast = float(rho_open - rho_shadow)
    return float(rho_shadow), float(rho_open), contrast


def run_checks(artifact: dict) -> bool:
    xs, ys = artifact['xs'], artifact['ys']
    p_plan = artifact['p_plan']

    interp_plan = _make_interpolator(xs, ys, p_plan)

    all_ok = True
    print('\n=== Section 0a GP Artifact Verification ===\n')

    # --- Check 1: Shadow contrast ---
    rho_shadow, rho_open, contrast = _check_contrast(interp_plan)
    ok1 = contrast >= SHADOW_CONTRAST_MIN and rho_shadow < SHADOW_REGION_RHO_MAX + 0.05 and rho_open > OPEN_REGION_RHO_MIN - 0.05
    status1 = 'PASS' if ok1 else 'FAIL'
    print(f'[{status1}] Shadow contrast')
    print(f'       Shadow region mean rho_plan : {rho_shadow:.3f}  (want < {SHADOW_REGION_RHO_MAX:.2f})')
    print(f'       Open region mean rho_plan   : {rho_open:.3f}  (want > {OPEN_REGION_RHO_MIN:.2f})')
    print(f'       Contrast (open - shadow)    : {contrast:.3f}  (want >= {SHADOW_CONTRAST_MIN:.2f})')
    if not ok1:
        print('  ACTION: Refit GP with denser samples or re-examine YOLO score distribution.')
        all_ok = False

    # --- Check 2: Start/goal visibility ---
    print()
    for task_name, task in TASKS.items():
        start, goal = task['start'], task['goal']
        rho_start = float(_query(interp_plan, np.array([[start[0], start[1]]]))[0])
        rho_goal  = float(_query(interp_plan, np.array([[goal[0],  goal[1]]]))[0])
        ok_s = rho_start > RHO_VISIBLE_THRESHOLD
        ok_g = rho_goal  > RHO_VISIBLE_THRESHOLD
        ok2 = ok_s and ok_g
        status2 = 'PASS' if ok2 else 'FAIL'
        print(f'[{status2}] {task_name}: start rho={rho_start:.3f} ({"ok" if ok_s else "LOW"}), '
              f'goal rho={rho_goal:.3f} ({"ok" if ok_g else "LOW"})  '
              f'(want both > {RHO_VISIBLE_THRESHOLD:.2f})')
        if not ok2:
            print(f'  ACTION: Adjust task coordinates or shelf position for {task_name}.')
            all_ok = False

    # --- Check 3: Shadow path intersection for shadow tasks ---
    print()
    for task_name, task in TASKS.items():
        if not task['is_shadow_task']:
            continue
        start, goal = task['start'], task['goal']
        shadow_len = _path_shadow_length(interp_plan, start, goal)
        ok3 = shadow_len >= SHADOW_PATH_MIN_M
        status3 = 'PASS' if ok3 else 'FAIL'
        total_len = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
        print(f'[{status3}] {task_name}: direct path shadow length = {shadow_len:.2f} m '
              f'/ {total_len:.2f} m total  (want >= {SHADOW_PATH_MIN_M:.1f} m)')
        if not ok3:
            print(f'  ACTION: The direct path does not pass through enough shadow. '
                  f'Adjust task coordinates or shelf position.')
            all_ok = False

    print()
    if all_ok:
        print('All checks PASSED. Proceed to campaign.\n')
    else:
        print('One or more checks FAILED. Do NOT proceed until resolved.\n')

    return all_ok


def plot_maps(artifact: dict, out_path: Path) -> None:
    xs, ys = artifact['xs'], artifact['ys']
    X, Y = np.meshgrid(xs, ys)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    cmap = 'RdYlGn'

    panels = [
        (artifact['p_mean'], 'GP posterior mean $\\sigma(\\mu_h)$', r'(a)'),
        (artifact['p_plan'], 'Conservative planning map $\\rho_\\mathrm{plan}$', r'(b)'),
        (artifact['p_std'],  'GP uncertainty $\\sigma_h$', r'(c)'),
    ]
    for ax, (data, title, label) in zip(axes, panels):
        im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=1 if label != r'(c)' else None)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'{label} {title}', fontsize=10)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

    # Overlay task markers on the planning map (middle panel)
    ax = axes[1]
    colors = {'shadow_tradeoff_a': 'blue', 'shadow_tradeoff_b': 'orange', 'sanity_open': 'green'}
    for task_name, task in TASKS.items():
        color = colors.get(task_name, 'black')
        s, g = task['start'], task['goal']
        ax.plot(*s, 'o', color=color, markersize=10, markeredgecolor='k', zorder=5)
        ax.plot(*g, '*', color=color, markersize=12, markeredgecolor='k', zorder=5)
        ax.plot([s[0], g[0]], [s[1], g[1]], '--', color=color, linewidth=1, alpha=0.6, zorder=4)
        ax.text(s[0], s[1] + 0.12, f'{task_name[:1].upper()}s', fontsize=7, ha='center', color=color)
        ax.text(g[0], g[1] + 0.12, f'{task_name[:1].upper()}g', fontsize=7, ha='center', color=color)

    # Add shelf region marker (approximate)
    shelf = mpatches.Rectangle((-0.5, -0.2), 2.0, 0.5, linewidth=2,
                                edgecolor='black', facecolor='none', linestyle='-', zorder=6)
    axes[1].add_patch(shelf)
    axes[1].text(0.5, 0.05, 'shelf', fontsize=8, ha='center', va='center', color='black')

    # Threshold overlay on planning map: show rho < 0.35 as shadow mask
    rho_plan = artifact['p_plan']
    shadow_mask = np.where(rho_plan < RHO_SHADOW_MAX, 0.3, np.nan)
    axes[1].pcolormesh(X, Y, shadow_mask, cmap='Greys', vmin=0, vmax=1, alpha=0.3, zorder=3)

    fig.suptitle(f'GP Artifact Verification — {out_path.stem}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to: {out_path}')
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description='Verify GP artifact against Section 0a requirements.')
    parser.add_argument('--artifact', required=True, help='Path to the GP .npz artifact.')
    parser.add_argument('--out', default='gp_verification.png', help='Output plot path.')
    parser.add_argument('--plot-only', action='store_true', help='Skip checks, only plot.')
    args = parser.parse_args()

    artifact_path = Path(args.artifact).expanduser().resolve()
    if not artifact_path.is_file():
        print(f'ERROR: artifact not found: {artifact_path}', file=sys.stderr)
        return 1

    print(f'Loading artifact: {artifact_path}')
    try:
        artifact = _load_artifact(artifact_path)
    except Exception as e:
        print(f'ERROR loading artifact: {e}', file=sys.stderr)
        return 1

    print(f'GP grid: xs={artifact["xs"].shape}, ys={artifact["ys"].shape}')
    print(f'rho_plan range: [{artifact["p_plan"].min():.3f}, {artifact["p_plan"].max():.3f}]')

    out_path = Path(args.out).expanduser().resolve()
    plot_maps(artifact, out_path)

    if args.plot_only:
        return 0

    all_ok = run_checks(artifact)
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
