#!/usr/bin/env python3
"""Generate thesis figures from the IWAI campaign data.

Mechanism-first results layout. Main-text figures (Task A is the headline):

    gp_pipeline.pdf            data -> rho_plan -> induced ambiguity (\xa75)
    mechanism_taskA.pdf        single representative C2 run with truth, one
                                belief mean, fresh camera-correction evidence,
                                YOLO-score diagnostics, 2sigma ellipses, and
                                time series
    compare_taskA.pdf          two-panel C1 vs C2 multi-seed truth paths
    (Tables 1 & 2 are written into 07_results.tex directly.)

Appendix figures:

    paths_overview_BS.pdf      Task B and Task S overlays
    paths_per_seed.pdf         per-seed paths for every (task, condition)
    compare_C3_taskA.pdf       C3 ablation pulled out of main text
    metrics_box.pdf            pooled boxplots (supporting visualization)

Problem-setup helper:

    problem_setup_panels.pdf   replaces (b)/(c) of the legacy figure with
                                current Q-noise dynamics; (a) Gazebo screenshot
                                must be captured separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
import numpy as np


REPO = Path('/home/joostleliveld/Thesis/UnembodiedNavigation')
CAMPAIGN_DIR = REPO / 'logs' / 'visibility_comparison' / 'iwai_campaign'
GP_PATH = REPO / 'logs' / 'visibility_comparison' / 'current_gp' / 'yolo_score_raw_gp.npz'

TASKS = ['shadow_tradeoff_a', 'shadow_tradeoff_b', 'sanity_open']
TASK_LABELS = {
    'shadow_tradeoff_a': 'Task A: shadow tradeoff (front)',
    'shadow_tradeoff_b': 'Task B: shadow tradeoff (diagonal)',
    'sanity_open': 'Task S: sanity, no shadow',
}
TASK_INFO = {
    'shadow_tradeoff_a': {'start': (-2.0, 0.5), 'goal': (2.0, -0.5)},
    'shadow_tradeoff_b': {'start': (-2.0, -1.0), 'goal': (2.0, -0.5)},
    'sanity_open':       {'start': (-2.0, -1.5), 'goal': (2.0, -1.5)},
}

CONDS = ['C1', 'C2', 'C3']
COND_LABEL = {
    'C1': r'C1: constant $R_0$',
    'C2': r'C2: visibility-aware (full)',
    'C3': r'C3: GP-risk only (ablation)',
}
COND_COLOR = {
    'C1': '#d62728',  # red
    'C2': '#1f77b4',  # blue
    'C3': '#2ca02c',  # green
}
OUTCOME_COLOR = {
    'goal_reached': '#2ca02c',
    'collision': '#d62728',
    'timeout': '#ff9f40',
    'infra_invalid': '#7f7f7f',
}

SHELF = dict(xmin=-0.9, xmax=0.8, ymin=-0.33, ymax=0.03)
CAM_XY = (-2.45, -2.45)
FRESH_CAMERA_UPDATE_MAX_AGE_S = 1.0


def load_campaign(log_path: Path) -> dict:
    return json.loads(log_path.read_text(encoding='utf-8'))


def load_metrics_csv(path: Path) -> list[dict]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_run_csv(run_dir: Path) -> list[dict]:
    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        for p in run_dir.rglob('experiment.csv'):
            csv_path = p
            break
    if not csv_path.is_file():
        return []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_plan_samples(run_dir: Path) -> list[dict]:
    p = run_dir / 'plan_samples.csv'
    if not p.is_file():
        for q in run_dir.rglob('plan_samples.csv'):
            p = q
            break
    if not p.is_file():
        return []
    with p.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_perception_csv(run_dir: Path) -> list[dict]:
    p = run_dir / 'perception.csv'
    if not p.is_file():
        for q in run_dir.rglob('perception.csv'):
            p = q
            break
    if not p.is_file():
        return []
    with p.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_run_manifest(run_dir: Path) -> dict:
    p = run_dir / 'run_manifest.json'
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def _resolve_for_compare(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve(strict=False)


def validate_campaign_gp_artifact(rows: list[dict], gp_path: Path) -> None:
    """Hard-fail if visibility-aware runs do not match the plotted GP artifact."""
    expected = _resolve_for_compare(str(gp_path))
    mismatches = []
    for r in rows:
        condition = str(r.get('condition', ''))
        if condition == 'C1':
            continue
        run_dir_str = str(r.get('run_dir', '') or '')
        if not run_dir_str:
            continue
        run_dir = Path(run_dir_str)
        manifest = load_run_manifest(run_dir)
        actual_str = str(manifest.get('visibility_artifact_path', '') or '')
        actual = _resolve_for_compare(actual_str) if actual_str else None
        if actual != expected:
            mismatches.append((condition, r.get('task', ''), r.get('seed', ''),
                               run_dir, actual_str or '<missing>'))
    if mismatches:
        lines = [
            'ERROR: refusing to make paper figures from mixed GP artifacts.',
            f'Expected plotted/evaluated artifact: {expected}',
            'The following visibility-aware runs used a different artifact:',
        ]
        for condition, task, seed, run_dir, actual in mismatches[:12]:
            lines.append(
                f'  {task}/{condition}/seed{seed}: {actual} ({run_dir})'
            )
        if len(mismatches) > 12:
            lines.append(f'  ... and {len(mismatches) - 12} more')
        lines.append('Rerun the campaign with the current raw-YOLO-score GP, or point the figure script at the exact artifact used by the runs.')
        raise RuntimeError('\n'.join(lines))


def _f(row: dict, key: str) -> float:
    v = row.get(key, '')
    if v in (None, '', 'nan', 'NaN'):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def extract_truth_path(rows: list[dict]):
    ts, xs, ys = [], [], []
    for r in rows:
        if _f(r, 'truth_available') >= 0.5:
            x, y = _f(r, 'truth_x'), _f(r, 'truth_y')
            t = _f(r, 'stamp')
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x); ys.append(y); ts.append(t)
    return np.asarray(ts), np.asarray(xs), np.asarray(ys)


def extract_belief_path(rows: list[dict]):
    """Planner-belief trajectory, deduplicated by belief stamp.

    Reads `planner_belief_*`, the planner-side EKF state. This propagates
    with motion model and growing covariance even when no perception
    update arrives. The `state_*` columns carry the perception-only
    output and only exist when YOLO actually fired, so they are *not*
    what we want for visualizing belief dynamics.
    """
    ts, mx, my, sxx, sxy, syy, upd = [], [], [], [], [], [], []
    seen_stamp = None
    for r in rows:
        if _f(r, 'planner_belief_available') < 0.5:
            continue
        b_stamp = _f(r, 'planner_belief_stamp')
        if not math.isfinite(b_stamp):
            continue
        if seen_stamp is not None and b_stamp == seen_stamp:
            continue
        seen_stamp = b_stamp
        x, y = _f(r, 'planner_belief_x'), _f(r, 'planner_belief_y')
        cxx = _f(r, 'planner_cov_x')
        cxy = _f(r, 'planner_cov_xy')
        cyy = _f(r, 'planner_cov_y')
        t = _f(r, 'stamp')
        u = _f(r, 'planner_pixel_correction_available')
        age = _f(r, 'planner_pixel_correction_age_s')
        if math.isfinite(x) and math.isfinite(y):
            ts.append(t); mx.append(x); my.append(y)
            sxx.append(cxx); sxy.append(cxy); syy.append(cyy)
            fresh = (
                math.isfinite(u) and u >= 0.5
                and math.isfinite(age)
                and age <= FRESH_CAMERA_UPDATE_MAX_AGE_S
            )
            upd.append(1.0 if fresh else 0.0)
    return (np.asarray(ts), np.asarray(mx), np.asarray(my),
            np.asarray(sxx), np.asarray(sxy), np.asarray(syy),
            np.asarray(upd))


def extract_yolo_scores(rows: list[dict]):
    """Raw YOLO score diagnostics from the runtime detector stream."""
    ts, scores, detected = [], [], []
    for r in rows:
        t = _f(r, 'log_stamp')
        s = _f(r, 'yolo_selected_score')
        if not math.isfinite(s):
            s = _f(r, 'yolo_raw_best_score')
        d = _f(r, 'yolo_detected_after_threshold')
        if math.isfinite(t) and math.isfinite(s):
            ts.append(t); scores.append(s)
            detected.append(1.0 if (math.isfinite(d) and d >= 0.5) else 0.0)
    return np.asarray(ts), np.asarray(scores), np.asarray(detected)


def load_gp(path: Path):
    d = np.load(path)
    return {
        'xs': d['xs'].astype(float),
        'ys': d['ys'].astype(float),
        'P_plan': d['P_conservative_plan_map'].astype(float),
        'P_mean': d['P_mean_map'].astype(float),
        'F_std': d['F_std_map'].astype(float),
        'X_train': d['X_train'].astype(float),
        'p_train': d['p_train'].astype(float),
        'cam': d['camera_pos'].astype(float),
        'beta': float(np.asarray(d['beta']).reshape(-1)[0]),
    }


def draw_workspace_overlay(ax, alpha=0.85, edgecolor='black'):
    ax.add_patch(Rectangle(
        (SHELF['xmin'], SHELF['ymin']),
        SHELF['xmax'] - SHELF['xmin'],
        SHELF['ymax'] - SHELF['ymin'],
        facecolor='dimgray', edgecolor=edgecolor, alpha=alpha, linewidth=0.6,
        zorder=3,
    ))


def draw_camera_marker(ax, color='black'):
    ax.scatter([CAM_XY[0]], [CAM_XY[1]], marker='v', s=70,
               color=color, zorder=11, edgecolor='white', linewidth=0.6,
               label='camera')


def resample_by_arclength(x, y, n=200, extra=None):
    if x.size < 2:
        return (np.zeros(n),
                np.full(n, x[0] if x.size else np.nan),
                np.full(n, y[0] if y.size else np.nan),
                {k: np.full(n, np.nan) for k in (extra or {})})
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if not math.isfinite(L) or L <= 1e-9:
        return (np.zeros(n), np.full(n, x[0]), np.full(n, y[0]),
                {k: np.full(n, np.nan) for k in (extra or {})})
    s_norm = s / L
    s_grid = np.linspace(0.0, 1.0, n)
    extras = {}
    for k, arr in (extra or {}).items():
        if arr is None or arr.size != x.size:
            extras[k] = np.full(n, np.nan)
        else:
            extras[k] = np.interp(s_grid, s_norm, arr)
    return s_grid, np.interp(s_grid, s_norm, x), np.interp(s_grid, s_norm, y), extras


def _cov_eig(sxx, sxy, syy):
    a = 0.5 * (sxx + syy)
    b = 0.5 * (sxx - syy)
    d = math.sqrt(b * b + sxy * sxy)
    lam1 = max(a + d, 1e-12)
    lam2 = max(a - d, 1e-12)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return lam1, lam2, theta


def _draw_cov_ellipses(ax, x_arr, y_arr, sxx, sxy, syy, n_ellipses=10,
                       scale=2.0, color='purple'):
    if x_arr.size == 0:
        return
    idx = np.linspace(0, x_arr.size - 1, n_ellipses).astype(int)
    for i in idx:
        sxx_i = float(sxx[i]) if i < sxx.size else math.nan
        syy_i = float(syy[i]) if i < syy.size else math.nan
        sxy_i = float(sxy[i]) if i < sxy.size else 0.0
        if not (math.isfinite(sxx_i) and math.isfinite(syy_i)):
            continue
        lam1, lam2, theta = _cov_eig(sxx_i, sxy_i, syy_i)
        e = Ellipse(xy=(x_arr[i], y_arr[i]),
                    width=2.0 * scale * math.sqrt(lam1),
                    height=2.0 * scale * math.sqrt(lam2),
                    angle=math.degrees(theta), facecolor=color, alpha=0.18,
                    edgecolor=color, linewidth=0.8, zorder=4)
        ax.add_patch(e)


def _draw_plan_horizon_snapshots(ax, plan_rows: list[dict],
                                 fractions=(0.18, 0.52, 0.84),
                                 color='#d95f02') -> None:
    """Draw a few planned horizons from the run without cluttering the path.

    The campaign logs contain every optimized horizon. For the paper figure we
    only need enough snapshots to show the receding-horizon mechanism, not a
    dense spaghetti plot.
    """
    by_stamp: dict[float, list[tuple[int, float, float]]] = {}
    for r in plan_rows:
        try:
            stamp = round(float(r['plan_stamp']), 3)
            idx = int(float(r['point_idx']))
            x = float(r['x'])
            y = float(r['y'])
        except (KeyError, TypeError, ValueError):
            continue
        by_stamp.setdefault(stamp, []).append((idx, x, y))
    if not by_stamp:
        return

    stamps = sorted(by_stamp)
    chosen = []
    for frac in fractions:
        i = min(len(stamps) - 1, max(0, int(round(frac * (len(stamps) - 1)))))
        if stamps[i] not in chosen:
            chosen.append(stamps[i])

    for j, stamp in enumerate(chosen):
        pts = sorted(by_stamp[stamp], key=lambda p: p[0])
        if len(pts) < 2:
            continue
        arr = np.asarray([(x, y) for _, x, y in pts], dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=1.05,
                alpha=0.75 if j == len(chosen) - 1 else 0.45,
                zorder=6,
                label='planned horizon snapshots' if j == 0 else None)


# -------------------------------------------------------------------------
# Figure: GP pipeline (data -> rho_plan -> induced ambiguity)
# -------------------------------------------------------------------------

def plot_gp_pipeline(gp, out_path, r_visible_uv=2.5, r_miss_uv=120.0,
                     min_prob=1e-4):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    extent = (gp['xs'][0], gp['xs'][-1], gp['ys'][0], gp['ys'][-1])

    ax = axes[0]
    sc = ax.scatter(gp['X_train'][:, 0], gp['X_train'][:, 1], c=gp['p_train'],
                    cmap='viridis', vmin=0.0, vmax=1.0, s=20, edgecolor='none')
    draw_workspace_overlay(ax, alpha=0.6)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='YOLO score')
    ax.set_title(r'(a) raw YOLO-score samples')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ (m)'); ax.set_ylabel(r'$y$ (m)')

    ax = axes[1]
    im = ax.imshow(gp['P_plan'], extent=extent, origin='lower', cmap='viridis',
                   vmin=0.0, vmax=max(0.62, float(np.max(gp['P_plan']))),
                   aspect='equal')
    draw_workspace_overlay(ax, alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$\rho_{\mathrm{plan}}$')
    ax.set_title(r'(b) planner reliability $\rho_{\mathrm{plan}}$')
    ax.set_xlabel(r'$x$ (m)'); ax.tick_params(labelleft=False)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    ax = axes[2]
    p_eff = np.clip(gp['P_plan'], min_prob, 1.0 - min_prob)
    visible_var = float(r_visible_uv) ** 2
    miss_var = float(r_miss_uv) ** 2
    plan_var = 1.0 / np.maximum(p_eff / visible_var + (1.0 - p_eff) / miss_var, 1e-9)
    log_det = 0.5 * np.log(np.clip(plan_var * plan_var, 1e-12, None))
    im = ax.imshow(log_det, extent=extent, origin='lower', cmap='magma', aspect='equal')
    draw_workspace_overlay(ax, alpha=0.6, edgecolor='white')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r'$\frac{1}{2}\log|R_{\mathrm{eff}}|$')
    ax.set_title('(c) induced ambiguity')
    ax.set_xlabel(r'$x$ (m)'); ax.tick_params(labelleft=False)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.14, wspace=0.30)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure: mechanism (representative C2 run, path + time series)
# -------------------------------------------------------------------------

def plot_mechanism_taskA(rows, gp, out_path, cond='C2', seed=0):
    target = next((r for r in rows
                   if r['task'] == 'shadow_tradeoff_a'
                   and r['condition'] == cond
                   and int(r.get('seed') or -1) == seed), None)
    if target is None or not target.get('run_dir'):
        return
    run_dir = Path(target['run_dir'])
    csv_rows = load_run_csv(run_dir)
    plan_rows = load_plan_samples(run_dir)
    perception_rows = load_perception_csv(run_dir)
    manifest = load_run_manifest(run_dir)
    t_t, t_x, t_y = extract_truth_path(csv_rows)
    b_t, b_x, b_y, sxx, sxy, syy, b_upd = extract_belief_path(csv_rows)
    yolo_t, yolo_score, yolo_detected = extract_yolo_scores(perception_rows)
    yolo_threshold = float(manifest.get('yolo_conf_threshold', 0.25))

    t_p = np.asarray([_f(r, 'stamp') for r in csv_rows])
    pvis = np.asarray([_f(r, 'p_vis_plan_eff') for r in csv_rows])
    err = np.asarray([_f(r, 'truth_belief_error_m') for r in csv_rows])
    cov_tr = np.asarray([_f(r, 'state_cov_trace') for r in csv_rows])
    efe_risk = np.asarray([_f(r, 'efe_risk') for r in csv_rows])
    efe_amb = np.asarray([_f(r, 'efe_ambiguity') for r in csv_rows])

    fig = plt.figure(figsize=(11.8, 6.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1.0],
                          height_ratios=[1.0, 1.0, 1.0],
                          hspace=0.40, wspace=0.28)
    ax_path = fig.add_subplot(gs[:, 0])
    ax_pvis = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1], sharex=ax_pvis)
    ax_efe = fig.add_subplot(gs[2, 1], sharex=ax_pvis)

    extent = (gp['xs'][0], gp['xs'][-1], gp['ys'][0], gp['ys'][-1])
    im = ax_path.imshow(gp['P_plan'], extent=extent, origin='lower',
                        cmap='viridis', vmin=0.0, vmax=0.62, aspect='equal',
                        alpha=0.85, zorder=1)
    draw_workspace_overlay(ax_path, alpha=0.85)
    draw_camera_marker(ax_path)
    ti = TASK_INFO['shadow_tradeoff_a']
    ax_path.scatter([ti['start'][0]], [ti['start'][1]], marker='o', s=80,
                    facecolor='lime', edgecolor='black', zorder=10, label='start')
    ax_path.scatter([ti['goal'][0]], [ti['goal'][1]], marker='*', s=170,
                    facecolor='red', edgecolor='black', zorder=10, label='goal')
    ax_path.add_patch(Circle(ti['goal'], 0.20, fill=False, edgecolor='red',
                             linewidth=1.0, linestyle='--', zorder=9))

    ax_path.plot(t_x, t_y, color='black', linewidth=2.0, zorder=8, label='truth')
    _draw_plan_horizon_snapshots(ax_path, plan_rows)

    if b_x.size:
        # Break the line whenever the planner state-stamp gap is too large,
        # otherwise matplotlib draws a phantom diagonal across an interval
        # in which the EKF was not publishing.
        max_gap_s = 1.5
        gap_mask = np.concatenate([[False], np.diff(b_t) > max_gap_s])
        bx_seg = b_x.copy().astype(float)
        by_seg = b_y.copy().astype(float)
        bx_seg[gap_mask] = np.nan
        by_seg[gap_mask] = np.nan
        ax_path.plot(bx_seg, by_seg, color='purple', linewidth=1.8,
                     linestyle='--', alpha=0.95, zorder=7,
                     label='belief mean')
        upd_mask = b_upd >= 0.5
        if np.any(upd_mask):
            ax_path.scatter(bx_seg[upd_mask], by_seg[upd_mask],
                            s=14, facecolor='white', edgecolor='#1f77b4',
                            linewidth=0.7, alpha=0.9, zorder=8,
                            label='fresh camera correction')
    _draw_cov_ellipses(ax_path, b_x, b_y, sxx, sxy, syy,
                       n_ellipses=10, scale=2.0, color='purple')

    ax_path.set_xlim(-2.6, 2.6); ax_path.set_ylim(-2.4, 2.4)
    ax_path.set_aspect('equal')
    ax_path.set_xlabel(r'$x$ (m)'); ax_path.set_ylabel(r'$y$ (m)')
    ax_path.set_title(f'truth, belief mean, $2\\sigma$ ellipses, and fresh corrections\n'
                      f'over $\\rho_{{\\mathrm{{plan}}}}$ — {COND_LABEL[cond]}, seed {seed}',
                      fontsize=10)
    cbar = fig.colorbar(im, ax=ax_path, fraction=0.04, pad=0.02,
                        label=r'$\rho_{\mathrm{plan}}$')
    cbar.ax.tick_params(labelsize=8)
    ax_path.legend(loc='upper left', fontsize=8, frameon=True)

    finite_pvis = np.isfinite(pvis)
    if np.any(finite_pvis):
        ax_pvis.plot(t_p[finite_pvis], pvis[finite_pvis], color='#1f77b4',
                     linewidth=1.4)
        ax_pvis.set_ylim(0.0, max(0.5, float(np.nanmax(pvis[finite_pvis])) * 1.05))
    else:
        ax_pvis.text(0.5, 0.5, r'$\rho_{\mathrm{plan}}$ not logged for C1',
                     ha='center', va='center', transform=ax_pvis.transAxes,
                     fontsize=9, color='gray')
        ax_pvis.set_ylim(0, 1)
    if yolo_t.size:
        det_mask = yolo_detected >= 0.5
        miss_mask = ~det_mask
        if np.any(miss_mask):
            ax_pvis.scatter(yolo_t[miss_mask], yolo_score[miss_mask],
                            s=10, color='orange', alpha=0.45,
                            label='YOLO score below threshold', zorder=3)
        if np.any(det_mask):
            ax_pvis.scatter(yolo_t[det_mask], yolo_score[det_mask],
                            s=12, color='green', alpha=0.60,
                            label='YOLO accepted detection', zorder=4)
        ax_pvis.axhline(yolo_threshold, color='orange', linestyle=':',
                        linewidth=0.9, alpha=0.8,
                        label=f'YOLO threshold {yolo_threshold:.2f}')
    ax_pvis.set_ylabel(r'$\rho_{\mathrm{plan}}$ / YOLO score', fontsize=9)
    ax_pvis.legend(fontsize=7.0, loc='upper right', frameon=False)
    ax_pvis.grid(alpha=0.3, linestyle=':')

    finite_err = np.isfinite(err)
    if np.any(finite_err):
        ax_err.plot(t_p[finite_err], err[finite_err], color='black',
                    linewidth=1.2, label=r'$\|p_{\mathrm{truth}}-\hat p\|$')
    if np.any(np.isfinite(cov_tr)):
        sigma_pos = np.sqrt(np.clip(cov_tr / 2.0, 1e-9, None))
        ax_err.plot(t_p, sigma_pos, color='purple', linewidth=1.0,
                    linestyle='--', label=r'$\sqrt{\mathrm{tr}\,\Sigma_{xy}/2}$')
    upd_available = np.asarray([_f(r, 'planner_pixel_correction_available')
                                for r in csv_rows])
    upd_age = np.asarray([_f(r, 'planner_pixel_correction_age_s')
                          for r in csv_rows])
    upd_mask_rt = (
        (upd_available >= 0.5)
        & np.isfinite(upd_age)
        & (upd_age <= FRESH_CAMERA_UPDATE_MAX_AGE_S)
    )
    if np.any(upd_mask_rt):
        ymin, ymax = ax_err.get_ylim() if ax_err.has_data() else (0.0, 0.2)
        if ymax <= ymin:
            ymax = ymin + 0.2
        # Reserve a strip below the data for the update rug, then re-clamp ylim.
        strip = (ymax - ymin) * 0.12
        rug_y = ymin - strip * 0.5
        ax_err.scatter(t_p[upd_mask_rt],
                       np.full(int(upd_mask_rt.sum()), rug_y),
                       marker='|', s=24, color='#1f77b4', alpha=0.85,
                       label='fresh camera correction', zorder=4)
        ax_err.set_ylim(ymin - strip, ymax)
    ax_err.set_ylabel('error / std (m)', fontsize=9)
    ax_err.legend(fontsize=7.5, loc='upper right', frameon=False)
    ax_err.grid(alpha=0.3, linestyle=':')

    finite_risk = np.isfinite(efe_risk)
    finite_amb = np.isfinite(efe_amb)
    if np.any(finite_risk):
        ax_efe.plot(t_p[finite_risk], efe_risk[finite_risk], color='#d62728',
                    linewidth=1.2, label='risk')
    if np.any(finite_amb):
        ax_efe.plot(t_p[finite_amb], efe_amb[finite_amb], color='#ff9f40',
                    linewidth=1.2, label='ambiguity')
    ax_efe.set_xlabel('time (s)', fontsize=9)
    ax_efe.set_ylabel('EFE component', fontsize=9)
    ax_efe.legend(fontsize=8, loc='upper right', frameon=False)
    ax_efe.grid(alpha=0.3, linestyle=':')

    fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.10)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure: multi-seed C1 vs C2 comparison on Task A
# -------------------------------------------------------------------------

def _plot_cond_panel(ax, gp, task, cond, by_cond, n_grid=200,
                     show_belief=False):
    extent = (gp['xs'][0], gp['xs'][-1], gp['ys'][0], gp['ys'][-1])
    ax.imshow(gp['P_plan'], extent=extent, origin='lower', cmap='viridis',
              vmin=0.0, vmax=0.62, aspect='equal', alpha=0.85, zorder=1)
    draw_workspace_overlay(ax, alpha=0.85)
    draw_camera_marker(ax)
    ti = TASK_INFO[task]
    ax.scatter([ti['start'][0]], [ti['start'][1]], marker='o', s=80,
               facecolor='lime', edgecolor='black', zorder=10, label='start')
    ax.scatter([ti['goal'][0]], [ti['goal'][1]], marker='*', s=170,
               facecolor='red', edgecolor='black', zorder=10, label='goal')
    ax.add_patch(Circle(ti['goal'], 0.20, fill=False, edgecolor='red',
                        linewidth=1.0, linestyle='--', zorder=9))

    truth_grid = []
    for run_dir, outc in by_cond.get(cond, []):
        csv_rows = load_run_csv(run_dir)
        _, t_x, t_y = extract_truth_path(csv_rows)
        if t_x.size < 5:
            continue
        ls = '-' if outc == 'goal_reached' else ':'
        lw = 1.3 if outc == 'goal_reached' else 1.6
        ax.plot(t_x, t_y, color=COND_COLOR[cond], alpha=0.55, linewidth=lw,
                linestyle=ls, zorder=4)
        if outc == 'goal_reached':
            _, x_rs, y_rs, _ = resample_by_arclength(t_x, t_y, n=n_grid)
            truth_grid.append((x_rs, y_rs))
        if show_belief:
            _, b_x, b_y, *_ = extract_belief_path(csv_rows)
            if b_x.size >= 5:
                ax.plot(b_x, b_y, color='purple', alpha=0.20, linewidth=0.9,
                        linestyle='--', zorder=4)

    if truth_grid:
        tx = np.stack([g[0] for g in truth_grid], axis=0)
        ty = np.stack([g[1] for g in truth_grid], axis=0)
        mx, my = np.mean(tx, axis=0), np.mean(ty, axis=0)
        sy = np.std(ty, axis=0)
        ax.plot(mx, my, color=COND_COLOR[cond], linewidth=2.4, zorder=7,
                label=f'{cond} truth mean (success)')
        ax.fill_between(mx, my - sy, my + sy, color=COND_COLOR[cond],
                        alpha=0.20, zorder=2, linewidth=0,
                        label=fr'{cond} $\pm 1\sigma$ across seeds')
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.4, 2.4)


def plot_compare_taskA(rows, gp, out_path, conds=('C1', 'C2')):
    task = 'shadow_tradeoff_a'
    by_cond = {c: [] for c in conds}
    for r in rows:
        if r['task'] != task or r['condition'] not in conds:
            continue
        rd = r.get('run_dir', '')
        if rd:
            by_cond[r['condition']].append((Path(rd), r.get('outcome', '')))

    fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 4.0),
                             sharey=True)
    if len(conds) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds):
        _plot_cond_panel(ax, gp, task, cond, by_cond)
        ax.set_title(COND_LABEL[cond], fontsize=10)
        ax.set_xlabel(r'$x$ (m)')
    axes[0].set_ylabel(r'$y$ (m)')

    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc='lower center',
               bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.18, wspace=0.05)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure: appendix — Task B and Task S overview
# -------------------------------------------------------------------------

def plot_paths_overview_BS(rows, gp, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    extent = (gp['xs'][0], gp['xs'][-1], gp['ys'][0], gp['ys'][-1])
    by_task_cond = {}
    for r in rows:
        rd = r.get('run_dir', '')
        if rd:
            by_task_cond.setdefault((r['task'], r['condition']), []).append(Path(rd))
    for ax, task in zip(axes, ['shadow_tradeoff_b', 'sanity_open']):
        im = ax.imshow(gp['P_plan'], extent=extent, origin='lower',
                       cmap='viridis', vmin=0.0, vmax=0.62, aspect='equal',
                       alpha=0.9, zorder=1)
        draw_workspace_overlay(ax)
        draw_camera_marker(ax)
        ti = TASK_INFO[task]
        ax.scatter([ti['start'][0]], [ti['start'][1]], marker='o', s=70,
                   facecolor='lime', edgecolor='black', zorder=10, label='start')
        ax.scatter([ti['goal'][0]], [ti['goal'][1]], marker='*', s=160,
                   facecolor='red', edgecolor='black', zorder=10, label='goal')
        ax.add_patch(Circle(ti['goal'], 0.20, fill=False, edgecolor='red',
                            linewidth=1.0, linestyle='--', zorder=9))
        for cond in CONDS:
            for i, run_dir in enumerate(by_task_cond.get((task, cond), [])):
                _, xs, ys = extract_truth_path(load_run_csv(run_dir))
                if xs.size < 3:
                    continue
                ax.plot(xs, ys, color=COND_COLOR[cond], alpha=0.55,
                        linewidth=1.1, zorder=5,
                        label=COND_LABEL[cond] if i == 0 else None)
        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.4, 2.4)
        ax.set_xlabel(r'$x$ (m)')
    axes[0].set_ylabel(r'$y$ (m)')
    cbar_ax = fig.add_axes([0.93, 0.20, 0.014, 0.62])
    fig.colorbar(im, cax=cbar_ax, label=r'$\rho_{\mathrm{plan}}(p)$')
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc='lower center',
               bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.91, top=0.92, bottom=0.20, wspace=0.05)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure: appendix — per-seed paths
# -------------------------------------------------------------------------

def plot_paths_per_seed(rows, gp, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(10.0, 9.0), sharex=True, sharey=True)
    extent = (gp['xs'][0], gp['xs'][-1], gp['ys'][0], gp['ys'][-1])
    by_tc = {}
    for r in rows:
        rd = r.get('run_dir', '')
        seed = int(r['seed']) if r.get('seed') not in (None, '') else -1
        outc = r.get('outcome', '')
        by_tc.setdefault((r['task'], r['condition']), []).append(
            (seed, Path(rd) if rd else None, outc))
    for ti_idx, task in enumerate(TASKS):
        for ci_idx, cond in enumerate(CONDS):
            ax = axes[ti_idx, ci_idx]
            ax.imshow(gp['P_plan'], extent=extent, origin='lower',
                      cmap='viridis', vmin=0.0, vmax=0.62, aspect='equal',
                      alpha=0.9, zorder=1)
            draw_workspace_overlay(ax)
            tinfo = TASK_INFO[task]
            ax.scatter([tinfo['start'][0]], [tinfo['start'][1]], marker='o',
                       s=40, facecolor='lime', edgecolor='black', zorder=10)
            ax.scatter([tinfo['goal'][0]], [tinfo['goal'][1]], marker='*',
                       s=80, facecolor='red', edgecolor='black', zorder=10)
            ax.add_patch(Circle(tinfo['goal'], 0.20, fill=False,
                                edgecolor='red', linewidth=0.8, linestyle='--',
                                zorder=9))
            entries = by_tc.get((task, cond), [])
            n_total = len(entries)
            n_goal = sum(1 for _, _, o in entries if o == 'goal_reached')
            for seed, run_dir, outc in entries:
                if run_dir is None:
                    continue
                _, xs, ys = extract_truth_path(load_run_csv(run_dir))
                if xs.size < 3:
                    continue
                color = '#1f77b4' if outc == 'goal_reached' else '#d62728'
                ax.plot(xs, ys, color=color, alpha=0.7, linewidth=1.1, zorder=5)
            if ti_idx == 0:
                ax.set_title(COND_LABEL[cond], fontsize=9)
            if ci_idx == 0:
                ax.set_ylabel(TASK_LABELS[task], fontsize=9)
            ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.4, 2.4)
            ax.text(0.03, 0.96, f'goal {n_goal}/{n_total}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='black', linewidth=0.6, alpha=0.85))
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$x$ (m)')
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.07,
                        wspace=0.05, hspace=0.07)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure: appendix — pooled metrics box
# -------------------------------------------------------------------------

def plot_metrics_box(rows, out_path):
    metrics = [
        ('path_length_m', r'path length $L$ (m)', (0, 8)),
        ('mean_loc_error_m', r'mean localization error $\bar e$ (m)', (0, 0.3)),
        ('mean_overconf', r'mean overconfidence $\bar c$', (0, 6)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, (mkey, mlabel, ylim) in zip(axes, metrics):
        data_per_cond = {c: [] for c in CONDS}
        for r in rows:
            if r.get('outcome') != 'goal_reached':
                continue
            try:
                v = float(r.get(mkey, '') or 'nan')
            except ValueError:
                v = float('nan')
            if math.isfinite(v):
                data_per_cond[r['condition']].append(v)
        positions = list(range(len(CONDS)))
        bp = ax.boxplot([data_per_cond[c] for c in CONDS], positions=positions,
                        widths=0.55, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='black',
                                       markeredgecolor='black', markersize=4))
        for patch, c in zip(bp['boxes'], CONDS):
            patch.set_facecolor(COND_COLOR[c]); patch.set_alpha(0.55)
        for line in bp['medians']:
            line.set_color('black')
        for i, c in enumerate(CONDS):
            xs = np.full(len(data_per_cond[c]), i) + np.random.uniform(
                -0.10, 0.10, len(data_per_cond[c]))
            ax.scatter(xs, data_per_cond[c], s=12, color='black',
                       alpha=0.55, zorder=5)
        ax.set_xticks(positions); ax.set_xticklabels(CONDS, fontsize=10)
        ax.set_ylabel(mlabel); ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.3, linestyle=':')
    fig.suptitle('(only successful runs are pooled across tasks)',
                 fontsize=8, y=0.02)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.18, wspace=0.30)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure: problem-setup belief-rollout panels
# -------------------------------------------------------------------------

def plot_problem_setup_panels(rows, gp, out_path):
    target = next((r for r in rows
                   if r['task'] == 'shadow_tradeoff_a'
                   and r['condition'] == 'C1'
                   and r.get('outcome') in ('collision', 'timeout', 'infra_invalid')
                   and r.get('run_dir')), None)
    if target is None:
        return
    run_dir = Path(target['run_dir'])
    csv_rows = load_run_csv(run_dir)
    plan_rows = load_plan_samples(run_dir)
    t_t, t_x, t_y = extract_truth_path(csv_rows)
    b_t, b_x, b_y, sxx, sxy, syy, _b_upd = extract_belief_path(csv_rows)
    if t_x.size < 3 or b_x.size < 3:
        return
    plan_by_stamp = {}
    for pr in plan_rows:
        try:
            ts = round(float(pr['plan_stamp']), 3)
            plan_by_stamp.setdefault(ts, []).append((float(pr['x']), float(pr['y'])))
        except (KeyError, TypeError, ValueError):
            continue
    plan_stamps = sorted(plan_by_stamp.keys())
    if not plan_stamps:
        return
    early_stamp = plan_stamps[max(0, len(plan_stamps) // 6)]
    late_stamp = plan_stamps[max(0, int(0.85 * len(plan_stamps)))]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    extent = (gp['xs'][0], gp['xs'][-1], gp['ys'][0], gp['ys'][-1])
    ti = TASK_INFO['shadow_tradeoff_a']
    for ax, stamp, title in zip(
            axes, [early_stamp, late_stamp],
            ['(b) initial constant-$R_0$ rollout',
             '(c) near reduced camera-update reliability']):
        ax.imshow(gp['P_plan'], extent=extent, origin='lower', cmap='Greys',
                  vmin=0.0, vmax=0.62, aspect='equal', alpha=0.55, zorder=1)
        cone_xy = np.asarray([
            [CAM_XY[0], CAM_XY[1]],
            [3.0, 2.6], [3.0, -2.6],
        ])
        ax.fill(cone_xy[:, 0], cone_xy[:, 1], color='goldenrod', alpha=0.18,
                zorder=2,
                label='reduced camera-update reliability' if stamp == late_stamp else None)
        draw_workspace_overlay(ax)
        draw_camera_marker(ax)
        ax.scatter([ti['start'][0]], [ti['start'][1]], marker='o', s=70,
                   facecolor='lime', edgecolor='black', zorder=10, label='start')
        ax.scatter([ti['goal'][0]], [ti['goal'][1]], marker='*', s=170,
                   facecolor='red', edgecolor='black', zorder=10, label='goal')
        idx = int(np.searchsorted(t_t, stamp))
        idx = max(1, min(idx, t_t.size))
        ax.plot(t_x[:idx], t_y[:idx], color='black', linewidth=1.6,
                zorder=8, label='truth path')
        b_idx = int(np.searchsorted(b_t, stamp))
        b_idx = max(1, min(b_idx, b_x.size))
        ax.plot(b_x[:b_idx], b_y[:b_idx], color='purple', linewidth=1.2,
                linestyle='--', zorder=7, label='belief mean')
        if b_x.size and b_idx > 0:
            i = min(b_idx - 1, sxx.size - 1)
            sxx_i = float(sxx[i]); syy_i = float(syy[i])
            sxy_i = float(sxy[i] if i < sxy.size else 0.0)
            if math.isfinite(sxx_i) and math.isfinite(syy_i):
                lam1, lam2, theta = _cov_eig(sxx_i, sxy_i, syy_i)
                e = Ellipse(xy=(b_x[b_idx - 1], b_y[b_idx - 1]),
                            width=2.0 * 2.0 * math.sqrt(lam1),
                            height=2.0 * 2.0 * math.sqrt(lam2),
                            angle=math.degrees(theta), facecolor='purple',
                            alpha=0.30, edgecolor='purple', linewidth=1.0,
                            zorder=6, label=r'$2\sigma$ covariance')
                ax.add_patch(e)
        plan_pts = plan_by_stamp.get(stamp, [])
        if plan_pts:
            pa = np.asarray(plan_pts)
            ax.plot(pa[:, 0], pa[:, 1], color='red', linewidth=1.0,
                    zorder=8, label='current horizon')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-3.0, 3.0); ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel('position x (m)')
    axes[0].set_ylabel('position y (m)')
    handles, labels = axes[1].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc='lower center',
               bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.20, wspace=0.05)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics-csv', default='/tmp/paper_metrics.csv')
    p.add_argument('--out-dir',
                   default='/home/joostleliveld/Thesis/thesis-report/figures/campaign')
    args = p.parse_args()

    rows = load_metrics_csv(Path(args.metrics_csv))
    log = load_campaign(CAMPAIGN_DIR / 'campaign_log.json')
    for r in rows:
        key = f"{r['task']}__{r['condition']}__seed{r['seed']}"
        entry = log.get(key, {})
        if 'run_dir' in entry and entry['run_dir']:
            r['run_dir'] = entry['run_dir']
        if not r.get('outcome'):
            r['outcome'] = entry.get('outcome', '')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_campaign_gp_artifact(rows, GP_PATH)
    gp = load_gp(GP_PATH)

    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'

    # main text — mechanism-first ordering
    plot_gp_pipeline(gp, out_dir / 'gp_pipeline.pdf')
    print(f'wrote {out_dir / "gp_pipeline.pdf"}')

    plot_mechanism_taskA(rows, gp, out_dir / 'mechanism_taskA.pdf',
                         cond='C2', seed=0)
    print(f'wrote {out_dir / "mechanism_taskA.pdf"}')

    plot_compare_taskA(rows, gp, out_dir / 'compare_taskA.pdf',
                       conds=('C1', 'C2'))
    print(f'wrote {out_dir / "compare_taskA.pdf"}')

    # appendix
    plot_compare_taskA(rows, gp, out_dir / 'compare_C3_taskA.pdf',
                       conds=('C2', 'C3'))
    print(f'wrote {out_dir / "compare_C3_taskA.pdf"}')

    plot_paths_overview_BS(rows, gp, out_dir / 'paths_overview_BS.pdf')
    print(f'wrote {out_dir / "paths_overview_BS.pdf"}')

    plot_paths_per_seed(rows, gp, out_dir / 'paths_per_seed.pdf')
    print(f'wrote {out_dir / "paths_per_seed.pdf"}')

    plot_metrics_box(rows, out_dir / 'metrics_box.pdf')
    print(f'wrote {out_dir / "metrics_box.pdf"}')

    plot_problem_setup_panels(rows, gp, out_dir.parent / 'problem_setup_panels.pdf')
    print(f'wrote {out_dir.parent / "problem_setup_panels.pdf"}')


if __name__ == '__main__':
    main()
