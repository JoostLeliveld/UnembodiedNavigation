#!/usr/bin/env python3
"""Compute protocol-defined paper metrics from completed run directories.

Supports two campaign-log formats:
  * `campaign_log.json` from `run_visibility_campaign.py`, with explicit
    `task` / `condition` / `outcome` per entry;
  * current `grid_log.json` from `run_model_selection.py`, where each entry
    carries `label` (e.g. `C1_constant_R`, `C2_h40_*`) and `merged_config.task`.

The figure-side condition vocabulary is fixed: C1 = constant covariance baseline,
C2 = learned-observability EFE (any C2_* label collapses to `C2`).

Outcome classifier (added 2026-05): each row also gets boolean flags
  is_clean_success / is_near_success / is_collision / is_penetration
  / is_timeout / is_interrupted / is_invalid
plus the underlying `valid_run` and penetration depths from run_summary.json.
This mirrors the paper categories defined in the Experiments section.

Usage (current data):
    python compute_paper_metrics.py \\
        --campaign-log logs/visibility_comparison/aws_f31b1_final_v1/campaign_log.json \\
        --gp-artifact paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz \\
        --out paper_metrics.csv

Outputs:
    paper_metrics.csv   : one row per run, all metrics
    paper_summary.txt   : mean ± std per (task, condition)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

RHO_SHADOW_THRESHOLD = 0.35

PERCEPTION_WRITER_FIELDS = [
    'diag_stamp', 'log_stamp', 'detected', 'true_available',
    'true_x', 'true_y', 'true_yaw',
    'state_available', 'state_x', 'state_y', 'state_yaw',
    'state_age_s', 'state_fresh', 'state_pos_error', 'state_yaw_error_deg',
    'obs_u', 'obs_v', 'obs_yaw', 'obs_yaw_error_deg',
    'pixel_pose_available', 'pixel_pose_stamp', 'pixel_pose_u', 'pixel_pose_v',
    'pixel_pose_yaw', 'pixel_pose_age_s', 'pixel_pose_fresh',
    'pred_world_x', 'pred_world_y', 'localization_error_m',
    'pred_world_x_calibrated', 'pred_world_y_calibrated',
    'localization_error_calibrated_m', 'bev_y_calibration_offset_m',
    'u_red', 'v_red', 'red_area_px', 'u_blue', 'v_blue', 'blue_area_px',
    'separation_px', 'border_margin_px',
    'yolo_score_raw', 'yolo_score_selected', 'yolo_detected_after_threshold',
    'yolo_best_class_id', 'yolo_target_candidate_count',
    'bbox_area_px', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
    'logit_margin', 'class_entropy',
    'mask_area_px', 'mask_bottom_u', 'mask_bottom_v', 'mask_used',
    'mask_polygon_points', 'confidence_logit', 'mask_compactness',
    'mask_border_frac', 'mask_score',
    'selected_pixel_source_code', 'yolo_raw_best_score',
    'yolo_selected_score', 'yolo_num_target_candidates',
    'yolo_selected_class_id', 'yolo_selected_pixel_source',
    'yolo_bbox_area', 'yolo_mask_area', 'yolo_inference_ms',
    'detector_callback_ms', 'yolo_receive_stamp', 'yolo_start_stamp',
    'yolo_finish_stamp', 'yolo_publish_stamp', 'yolo_latency_s',
    'frame_age_at_publish_s', 'detector_total_latency_s',
    'camera_relative_bearing_deg', 'seed',
]


def _load_gp(artifact_path: Path):
    with np.load(artifact_path, allow_pickle=False) as data:
        xs = np.asarray(data['xs'], dtype=float)
        ys = np.asarray(data['ys'], dtype=float)
        if 'P_conservative_plan_map' not in data.files:
            raise RuntimeError(f'Paper GP artifact is missing P_conservative_plan_map: {artifact_path}')
        p_plan = np.asarray(data['P_conservative_plan_map'], dtype=float)
    interp = RegularGridInterpolator(
        (ys, xs), p_plan, method='linear', bounds_error=False, fill_value=np.nan
    )
    return interp


def _resolve_for_compare(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve(strict=False)


def _visibility_artifact_suffix(path: Path) -> Path:
    parts = path.parts
    if 'visibility_comparison' in parts:
        idx = parts.index('visibility_comparison')
        return Path(*parts[idx + 1:])
    return Path(path.name)


def _same_visibility_artifact(actual: Path | None, expected: Path) -> bool:
    if actual is None:
        return False
    return actual == expected or _visibility_artifact_suffix(actual) == _visibility_artifact_suffix(expected)


def _load_run_manifest(run_dir: Path) -> dict:
    p = run_dir / 'run_manifest.json'
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _label_to_condition(label: str) -> str:
    """Collapse model-selection labels into paper conditions C1 / C2.

    Anything starting `C1` → `C1` (constant covariance).
    Anything starting `C2` or `C3` → `C2` (learned covariance) for paper rows;
    callers that need to distinguish the C3 ablation should branch on the
    raw label, not the condition column.
    """
    s = label.strip()
    if s.startswith('C1'):
        return 'C1'
    if s.startswith('C2'):
        return 'C2'
    if s.startswith('C3'):
        return 'C3'
    return s


def _normalize_entry(key: str, entry: dict, campaign_root: Path) -> dict:
    """Return a flat record with task/condition/seed/planner/run_dir/outcome.

    Handles both campaign `campaign_log.json` rows (which already have those
    columns) and current `grid_log.json` rows (label + merged_config).
    Resolves stale `run_dir` paths by re-locating the experiment_id under
    `<campaign_root>/<axis>/<label>/seed<N>/`.
    """
    task = str(entry.get('task') or entry.get('merged_config', {}).get('task', ''))
    raw_label = str(entry.get('condition') or entry.get('label', ''))
    condition = _label_to_condition(raw_label)
    seed = entry.get('seed', '')
    planner = str(entry.get('planner')
                  or entry.get('overrides', {}).get('planner')
                  or entry.get('merged_config', {}).get('planner', ''))
    outcome = str(entry.get('outcome', ''))
    run_dir_str = str(entry.get('run_dir', '') or '')

    run_dir = Path(run_dir_str) if run_dir_str else None
    if (run_dir is None or not run_dir.is_dir()) and campaign_root and seed != '':
        axis = str(entry.get('axis', 'monte_carlo_compare'))
        candidate_parents = [
            campaign_root / task / condition / f'seed{seed}',
            campaign_root / task / raw_label / f'seed{seed}',
            campaign_root / axis / raw_label / f'seed{seed}',
        ]
        exp_id = run_dir.name if run_dir is not None else ''
        for candidate_parent in candidate_parents:
            if not candidate_parent.is_dir():
                continue
            picked = (candidate_parent / exp_id) if exp_id else None
            if picked is None or not picked.is_dir():
                experiments = sorted(candidate_parent.glob('experiment_*'))
                if experiments:
                    picked = experiments[-1]
            if picked is not None and picked.is_dir():
                run_dir = picked
                break

    return {
        'task': task,
        'condition': condition,
        'label': raw_label,
        'seed': seed,
        'planner': planner,
        'outcome': outcome,
        'run_dir': str(run_dir) if run_dir is not None else '',
    }


def _validate_campaign_gp_artifact(records: list[dict], gp_artifact: Path) -> None:
    """Hard-fail if learned-condition run manifests do not match the metrics GP."""
    expected = _resolve_for_compare(str(gp_artifact))
    mismatches = []
    for r in records:
        if r['condition'] == 'C1':
            continue
        run_dir_str = r['run_dir']
        if not run_dir_str:
            continue
        run_dir = Path(run_dir_str)
        manifest = _load_run_manifest(run_dir)
        actual_str = str(manifest.get('visibility_artifact_path', '') or '')
        actual = _resolve_for_compare(actual_str) if actual_str else None
        if not _same_visibility_artifact(actual, expected):
            mismatches.append((r.get('task', ''), r['condition'],
                               r.get('seed', ''), run_dir,
                               actual_str or '<missing>'))
    if mismatches:
        print('ERROR: refusing to compute paper metrics from mixed GP artifacts.', file=sys.stderr)
        print(f'Expected metrics artifact: {expected}', file=sys.stderr)
        for task, condition, seed, run_dir, actual in mismatches[:12]:
            print(f'  {task}/{condition}/seed{seed}: {actual} ({run_dir})', file=sys.stderr)
        if len(mismatches) > 12:
            print(f'  ... and {len(mismatches) - 12} more', file=sys.stderr)
        print('Rerun the campaign with the current raw-YOLO-score GP before computing paper metrics.', file=sys.stderr)
        raise SystemExit(1)


def _query_rho(interp, x: float, y: float) -> float:
    if not (math.isfinite(x) and math.isfinite(y)):
        return math.nan
    return float(interp([[y, x]])[0])


def _load_experiment_csv(run_dir: Path) -> list[dict]:
    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        return []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        rows = []
        for values in reader:
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))
            elif len(values) == len(PERCEPTION_WRITER_FIELDS):
                rows.append(dict(zip(PERCEPTION_WRITER_FIELDS, values)))
        return rows


def _load_perception_csv(run_dir: Path) -> list[dict]:
    csv_path = run_dir / 'perception.csv'
    if not csv_path.is_file():
        for p in run_dir.rglob('perception.csv'):
            csv_path = p
            break
    if not csv_path.is_file():
        return []
    rows = []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _pf(row: dict, key: str) -> float:
    v = row.get(key, '')
    if v in (None, '', 'nan', 'NaN'):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


NEAR_SUCCESS_RADIUS_M = 0.40  # entered goal region but did not satisfy hold


def _loc_nll_nees(dx: float, dy: float, cov_x: float, cov_xy: float, cov_y: float) -> tuple[float, float]:
    """Return Gaussian position NLL and NEES for the 2D planner belief."""
    if not all(math.isfinite(v) for v in (dx, dy, cov_x, cov_y)):
        return math.nan, math.nan
    cov_xy = float(cov_xy) if math.isfinite(cov_xy) else 0.0
    S = np.array([[float(cov_x), cov_xy], [cov_xy, float(cov_y)]], dtype=float)
    S = 0.5 * (S + S.T)
    try:
        vals, vecs = np.linalg.eigh(S)
    except np.linalg.LinAlgError:
        return math.nan, math.nan
    vals = np.clip(vals, 1e-9, None)
    S_pd = (vecs * vals) @ vecs.T
    e = np.array([float(dx), float(dy)], dtype=float)
    try:
        solved = np.linalg.solve(S_pd, e)
    except np.linalg.LinAlgError:
        return math.nan, math.nan
    nees = float(e @ solved)
    logdet = float(np.log(vals[0]) + np.log(vals[1]))
    nll = 0.5 * (nees + logdet + 2.0 * math.log(2.0 * math.pi))
    return nll, nees


def _classify_outcome(summary: dict, completed_externally: bool = True) -> dict:
    """Map run_summary fields to the paper outcome categories.

    Categories (mutually exclusive for primary outcome flags):
      is_invalid       : run_summary.valid_run is False for non-outcome reasons
                          (logger / infra / frame-sanity failure)
      is_interrupted   : run did not reach a planned termination (no completion_reason)
      is_collision     : collision_any True
      is_penetration   : max_obstacle_penetration_m or max_wall_penetration_m > 0
                          (recorded separately even when no collision was logged)
      is_timeout       : completion_reason == 'timeout_after_first_cmd'
      is_clean_success : completion_reason == 'goal_reached' AND not collision/penetration
      is_near_success  : not clean_success and goal_region_entered or
                          minimum_goal_distance <= NEAR_SUCCESS_RADIUS_M
    """
    valid_run = summary.get('valid_run')
    completion_reason = str(summary.get('completion_reason', '') or '')
    coll_any = bool(summary.get('collision_any', False))
    pen_o = float(summary.get('max_obstacle_penetration_m', 0.0) or 0.0)
    pen_w = float(summary.get('max_wall_penetration_m', 0.0) or 0.0)
    has_penetration = (pen_o > 0.0) or (pen_w > 0.0)
    goal_region_entered = bool(summary.get('goal_region_entered', False))
    min_goal = float(summary.get('minimum_goal_distance', math.nan) or math.nan)

    is_collision = coll_any or completion_reason == 'collision'
    is_penetration = has_penetration  # reported separately, may overlap with collision
    is_timeout = completion_reason == 'timeout_after_first_cmd'
    is_invalid = (
        valid_run is False
        and not is_collision
        and not is_penetration
        and not is_timeout
    )
    is_interrupted = (not is_invalid) and (not completion_reason) and completed_externally is False
    is_clean_success = (
        completion_reason == 'goal_reached'
        and not coll_any
        and not has_penetration
    )
    is_near_success = (
        (not is_clean_success)
        and (goal_region_entered or
             (math.isfinite(min_goal) and min_goal <= NEAR_SUCCESS_RADIUS_M))
        and not is_collision and not is_invalid
    )
    return {
        'valid_run': bool(valid_run) if valid_run is not None else True,
        'is_clean_success': is_clean_success,
        'is_near_success': is_near_success,
        'is_collision': is_collision,
        'is_penetration': is_penetration,
        'is_timeout': is_timeout,
        'is_interrupted': is_interrupted,
        'is_invalid': is_invalid,
        'goal_region_entered': goal_region_entered,
        'max_obstacle_penetration_m': pen_o,
        'max_wall_penetration_m': pen_w,
    }


def _compute_run_metrics(run_dir: Path, summary: dict, gp_interp, task_info: dict | None) -> dict:
    rows = _load_experiment_csv(run_dir)
    perception_rows = _load_perception_csv(run_dir)
    manifest = _load_run_manifest(run_dir)

    # --- From summary ---
    goal_reached = bool(summary.get('goal_reached', False)) or summary.get('completion_reason') == 'goal_reached'
    crashed = bool(summary.get('crashed', False))
    completion_reason = str(summary.get('completion_reason', ''))
    path_length_m = float(summary.get('path_length_m', math.nan) or math.nan)
    min_goal_distance = float(summary.get('minimum_goal_distance', math.nan) or math.nan)
    final_goal_distance = float(summary.get('final_goal_distance', math.nan) or math.nan)
    elapsed_s = float(summary.get('elapsed_after_first_cmd_s', math.nan) or math.nan)
    mean_solve_time_ms = float(summary.get('mean_solve_time_ms', math.nan) or math.nan)
    # Prefer explicit driving-only summary fields when available. Older logs
    # only have mean_truth_belief_error_m, which can include launch/global-solve
    # rows; the CSV recomputation below remains the canonical path when rows exist.
    summary_mean_truth_belief = float(
        summary.get(
            'mean_truth_belief_error_after_first_cmd_m',
            summary.get('mean_truth_belief_error_m', math.nan),
        ) or math.nan
    )
    summary_mean_p_vis_eff = float(summary.get('mean_p_vis_plan_eff', math.nan) or math.nan)

    if not rows:
        return {
            'goal_reached': goal_reached, 'collision': crashed,
            'completion_reason': completion_reason,
            'path_length_m': path_length_m, 'min_goal_distance': min_goal_distance,
            'final_goal_distance': final_goal_distance,
            'elapsed_s': elapsed_s,
            'mean_loc_error_m': summary_mean_truth_belief, 'mean_overconf': math.nan,
            'mean_loc_nll': math.nan, 'mean_loc_nees': math.nan,
            'f_shadow': math.nan, 'mean_rho_plan': summary_mean_p_vis_eff,
            'path_efficiency': math.nan, 'mean_cov_trace': math.nan,
            'mean_solve_time_ms': mean_solve_time_ms,
            'p90_solve_time_ms': math.nan,
            'yolo_detection_rate': math.nan,
            'n_rows': 0,
        }

    # Runtime means are defined only after the first non-trivial command.
    # Pre-command rows include launch, global-solve, and estimator warm-up time;
    # mixing them into localization/control metrics makes route comparisons
    # look cleaner or worse for the wrong reason.
    first_cmd_stamp = math.nan
    for row in rows:
        cv = _pf(row, 'cmd_v')
        cw = _pf(row, 'cmd_w')
        if (math.isfinite(cv) and abs(cv) > 0.01) or (math.isfinite(cw) and abs(cw) > 0.05):
            first_cmd_stamp = _pf(row, 'stamp')
            break
    runtime_rows = rows
    if math.isfinite(first_cmd_stamp):
        runtime_rows = [
            row for row in rows
            if math.isfinite(_pf(row, 'stamp')) and _pf(row, 'stamp') >= first_cmd_stamp
        ]

    # --- From CSV after first command ---
    loc_errors = []
    overconf_values = []
    loc_nll_values = []
    loc_nees_values = []
    rho_values = []
    cov_traces = []
    solve_times_ms = []

    for row in runtime_rows:
        # Use planner belief as the estimator (that's what drives control)
        truth_x = _pf(row, 'truth_x')
        truth_y = _pf(row, 'truth_y')
        truth_ok = _pf(row, 'truth_available') >= 0.5 if math.isfinite(_pf(row, 'truth_available')) else False

        belief_x = _pf(row, 'planner_belief_x')
        belief_y = _pf(row, 'planner_belief_y')
        belief_ok = _pf(row, 'planner_belief_available') >= 0.5 if math.isfinite(_pf(row, 'planner_belief_available')) else False
        cov_x = _pf(row, 'planner_cov_x')
        cov_xy = _pf(row, 'planner_cov_xy')
        cov_y = _pf(row, 'planner_cov_y')

        if truth_ok and belief_ok and math.isfinite(belief_x) and math.isfinite(belief_y):
            dx = truth_x - belief_x
            dy = truth_y - belief_y
            err = math.hypot(dx, dy)
            loc_errors.append(err)

            if math.isfinite(cov_x) and math.isfinite(cov_y) and (cov_x + cov_y) > 0:
                cov_trace = cov_x + cov_y
                cov_traces.append(cov_trace)
                overconf = err / math.sqrt(cov_trace)
                overconf_values.append(overconf)
                nll, nees = _loc_nll_nees(dx, dy, cov_x, cov_xy, cov_y)
                if math.isfinite(nll):
                    loc_nll_values.append(nll)
                if math.isfinite(nees):
                    loc_nees_values.append(nees)

        if truth_ok and math.isfinite(truth_x) and math.isfinite(truth_y) and gp_interp is not None:
            rho = _query_rho(gp_interp, truth_x, truth_y)
            if math.isfinite(rho):
                rho_values.append(rho)

        st = _pf(row, 'solve_time_ms')
        if math.isfinite(st):
            solve_times_ms.append(st)

    mean_loc_error = float(np.mean(loc_errors)) if loc_errors else math.nan
    mean_overconf = float(np.mean(overconf_values)) if overconf_values else math.nan
    mean_loc_nll = float(np.mean(loc_nll_values)) if loc_nll_values else math.nan
    mean_loc_nees = float(np.mean(loc_nees_values)) if loc_nees_values else math.nan
    mean_cov_trace = float(np.mean(cov_traces)) if cov_traces else math.nan

    f_shadow = math.nan
    mean_rho_plan = math.nan
    if rho_values:
        f_shadow = float(np.mean(np.array(rho_values) < RHO_SHADOW_THRESHOLD))
        mean_rho_plan = float(np.mean(rho_values))

    # Path efficiency: straight-line distance / path length
    path_efficiency = math.nan
    if task_info and math.isfinite(path_length_m) and path_length_m > 0:
        start = task_info.get('start')
        goal = task_info.get('goal')
        if start and goal:
            straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
            path_efficiency = straight / path_length_m

    p90_solve_time_ms = (
        float(np.percentile(solve_times_ms, 90)) if solve_times_ms else math.nan
    )

    # Empirical YOLO detection rate from perception.csv after first command.
    yolo_detection_rate = math.nan
    yolo_pixel_fresh_rate = math.nan
    mean_state_error_fresh_m = math.nan
    if perception_rows:
        runtime_perception_rows = perception_rows
        if math.isfinite(first_cmd_stamp):
            runtime_perception_rows = [
                r for r in perception_rows
                if math.isfinite(_pf(r, 'log_stamp')) and _pf(r, 'log_stamp') >= first_cmd_stamp
            ]
        n_detected = sum(
            1 for r in runtime_perception_rows
            if _pf(r, 'yolo_detected_after_threshold') >= 0.5
        )
        yolo_detection_rate = n_detected / len(runtime_perception_rows) if runtime_perception_rows else math.nan
        fresh_flags = []
        state_fresh_errors = []
        for r in runtime_perception_rows:
            fresh = _pf(r, 'pixel_pose_fresh')
            if not math.isfinite(fresh):
                age = _pf(r, 'pixel_pose_age_s')
                # Fallback for old logs: use the campaign's common stale-pixel
                # threshold where available, otherwise the paper default.
                timeout = float(manifest.get('pixel_timeout_s', 0.5) or 0.5)
                fresh = 1.0 if math.isfinite(age) and age <= timeout else 0.0
            fresh_flags.append(fresh >= 0.5)
            err = _pf(r, 'state_pos_error')
            if fresh >= 0.5 and math.isfinite(err):
                state_fresh_errors.append(err)
        yolo_pixel_fresh_rate = (
            float(np.mean(fresh_flags)) if fresh_flags else math.nan
        )
        mean_state_error_fresh_m = (
            float(np.mean(state_fresh_errors)) if state_fresh_errors else math.nan
        )

    state_fresh_rate = math.nan
    if runtime_rows:
        flags = []
        for row in runtime_rows:
            fresh = _pf(row, 'state_fresh')
            if not math.isfinite(fresh):
                age = _pf(row, 'state_age_s')
                timeout = float(manifest.get('pixel_timeout_s', 0.5) or 0.5)
                fresh = 1.0 if math.isfinite(age) and age <= timeout else math.nan
            if math.isfinite(fresh):
                flags.append(fresh >= 0.5)
        state_fresh_rate = float(np.mean(flags)) if flags else math.nan

    # Use the CSV runtime mean for truth-belief error. Summary-level means may
    # include pre-command launch/global-solve rows and are only a no-CSV fallback.
    mean_loc_error_out = mean_loc_error
    if not math.isfinite(mean_loc_error_out):
        mean_loc_error_out = summary_mean_truth_belief

    return {
        'goal_reached': goal_reached,
        'collision': crashed,
        'completion_reason': completion_reason,
        'path_length_m': path_length_m,
        'min_goal_distance': min_goal_distance,
        'final_goal_distance': final_goal_distance,
        'elapsed_s': elapsed_s,
        'mean_loc_error_m': mean_loc_error_out,
        'mean_overconf': mean_overconf,
        'mean_loc_nll': mean_loc_nll,
        'mean_loc_nees': mean_loc_nees,
        'f_shadow': f_shadow,
        'mean_rho_plan': mean_rho_plan,
        'path_efficiency': path_efficiency,
        'mean_cov_trace': mean_cov_trace,
        'mean_solve_time_ms': mean_solve_time_ms,
        'p90_solve_time_ms': p90_solve_time_ms,
        'yolo_detection_rate': yolo_detection_rate,
        'yolo_pixel_fresh_rate': yolo_pixel_fresh_rate,
        'state_fresh_rate': state_fresh_rate,
        'mean_state_error_fresh_m': mean_state_error_fresh_m,
        'n_rows': len(rows),
    }


TASK_INFO = {
    # Compact benchmark legacy/core tasks.
    'shadow_tradeoff_a': {'start': (-2.0, 0.5), 'goal': (2.0, -0.5)},
    'shadow_tradeoff_b': {'start': (-2.0, -1.0), 'goal': (2.0, -0.5)},
    'sanity_open':       {'start': (-2.0, -1.5), 'goal': (2.0, -1.5)},
    # Current AWS candidate task. This entry only enables task-derived metrics
    # such as path efficiency; evidence status is still controlled by the
    # experiment registry and artifact chain.
    'F31_b1_apron_a3_mid': {'start': (3.30, -1.00), 'goal': (1.00, 1.75)},
}

FIELDNAMES = [
    'task', 'condition', 'label', 'seed', 'planner', 'run_dir',
    'goal_reached', 'collision', 'completion_reason',
    'path_length_m', 'min_goal_distance', 'final_goal_distance', 'elapsed_s',
    'mean_loc_error_m', 'mean_loc_nll', 'mean_loc_nees', 'mean_overconf',
    'f_shadow', 'mean_rho_plan',
    'path_efficiency', 'mean_cov_trace',
    'yolo_detection_rate',
    'mean_solve_time_ms', 'p90_solve_time_ms',
    'valid_run', 'is_clean_success', 'is_near_success',
    'is_collision', 'is_penetration', 'is_timeout', 'is_interrupted', 'is_invalid',
    'goal_region_entered',
    'max_obstacle_penetration_m', 'max_wall_penetration_m',
    'n_rows', 'outcome',
]


def _format(v) -> str:
    if v is None:
        return ''
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        return '' if not math.isfinite(v) else f'{v:.6f}'
    return str(v)


CONDITION_DISPLAY = {
    'C1': 'Constant cov.',
    'C2': 'Learned cov.',
    'C3': 'GP-risk only (abl.)',
}


def _print_summary(rows: list[dict]) -> str:
    lines = []
    header = (f'{"Task":<22} {"Cond":<22} {"N":>3} {"Clean%":>7} {"Near%":>6} '
              f'{"Coll%":>6} {"Pen%":>5} {"Inv%":>5} '
              f'{"L(m)":>10} {"ē(m)":>10} {"NLL":>9} {"NEES":>9} '
              f'{"c̄":>9} '
              f'{"f_shad":>8} {"det_rate":>9} {"η":>8} {"solve(ms)":>10}')
    lines.append(header)
    lines.append('-' * len(header))

    tasks = sorted({r['task'] for r in rows if r.get('task')})
    conditions = ['C1', 'C2', 'C3']

    for task in tasks:
        for cond in conditions:
            subset = [r for r in rows if r['task'] == task and r['condition'] == cond]
            if not subset:
                continue
            n = len(subset)
            def _pct(flag):
                return 100.0 * sum(1 for r in subset if r.get(flag)) / n
            clean_pct = _pct('is_clean_success')
            near_pct = _pct('is_near_success')
            coll_pct = _pct('is_collision')
            pen_pct = _pct('is_penetration')
            inv_pct = _pct('is_invalid')
            valid = [r for r in subset if r.get('is_clean_success')]

            def _mean_std(key):
                vals = []
                for r in valid:
                    v = r.get(key)
                    try:
                        vf = float(v if v not in (None, '') else 'nan')
                    except (TypeError, ValueError):
                        vf = math.nan
                    if math.isfinite(vf):
                        vals.append(vf)
                if not vals:
                    return 'n/a'
                return f'{np.mean(vals):.2f}±{np.std(vals):.2f}'

            lines.append(
                f'{task:<22} {CONDITION_DISPLAY.get(cond, cond):<22} {n:>3} '
                f'{clean_pct:>6.0f}% {near_pct:>5.0f}% '
                f'{coll_pct:>5.0f}% {pen_pct:>4.0f}% {inv_pct:>4.0f}% '
                f'{_mean_std("path_length_m"):>10} {_mean_std("mean_loc_error_m"):>10} '
                f'{_mean_std("mean_loc_nll"):>9} {_mean_std("mean_loc_nees"):>9} '
                f'{_mean_std("mean_overconf"):>9} {_mean_std("f_shadow"):>8} '
                f'{_mean_std("yolo_detection_rate"):>9} '
                f'{_mean_std("path_efficiency"):>8} {_mean_std("mean_solve_time_ms"):>10}'
            )
        lines.append('')

    lines.append('Outcome counts pool every attempted run; continuous metrics '
                 'are pooled over clean successes only.')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute paper metrics from a campaign run log.')
    parser.add_argument('--campaign-log', required=True,
                        help='Path to campaign_log.json or grid_log.json.')
    parser.add_argument('--gp-artifact', default='',
                        help='Path to GP .npz artifact for rho_plan queries (needed for f_shadow).')
    parser.add_argument('--out', default='paper_metrics.csv',
                        help='Output CSV path.')
    parser.add_argument('--summary-out', default='paper_summary.txt',
                        help='Output summary text path.')
    args = parser.parse_args()

    campaign_log_path = Path(args.campaign_log).expanduser().resolve()
    if not campaign_log_path.is_file():
        print(f'ERROR: campaign log not found: {campaign_log_path}', file=sys.stderr)
        return 1

    campaign_log = json.loads(campaign_log_path.read_text(encoding='utf-8'))
    campaign_root = campaign_log_path.parent
    print(f'Loaded {len(campaign_log)} run entries from {campaign_log_path}')

    records = [_normalize_entry(k, v, campaign_root) for k, v in campaign_log.items()]
    n_resolved = sum(1 for r in records if r['run_dir'])
    print(f'  resolved run_dir for {n_resolved}/{len(records)} entries')

    gp_interp = None
    if args.gp_artifact:
        gp_path = Path(args.gp_artifact).expanduser().resolve()
        if gp_path.is_file():
            print(f'Loading GP artifact: {gp_path}')
            _validate_campaign_gp_artifact(records, gp_path)
            gp_interp = _load_gp(gp_path)
        else:
            print(f'WARNING: GP artifact not found: {gp_path} — f_shadow and mean_rho will be NaN')

    out_path = Path(args.out).expanduser().resolve()
    summary_path = Path(args.summary_out).expanduser().resolve()

    all_rows = []
    for r in records:
        base_row = dict(r)
        run_dir_str = r['run_dir']
        task = r['task']
        condition = r['condition']
        seed = r['seed']

        if r['outcome'] == 'infra_invalid' or not run_dir_str:
            base_row.update({k: math.nan for k in FIELDNAMES if k not in base_row})
            base_row['goal_reached'] = False
            base_row['collision'] = False
            base_row['completion_reason'] = 'infra_invalid'
            base_row['valid_run'] = False
            base_row['is_invalid'] = True
            for flag in ('is_clean_success', 'is_near_success', 'is_collision',
                         'is_penetration', 'is_timeout', 'is_interrupted',
                         'goal_region_entered'):
                base_row[flag] = False
            all_rows.append(base_row)
            print(f'  infra_invalid: {task}/{condition}/seed{seed}')
            continue

        run_dir = Path(run_dir_str)
        summary_path_run = run_dir / 'run_summary.json'
        if not summary_path_run.is_file():
            candidates = sorted(run_dir.rglob('run_summary.json'))
            if candidates:
                summary_path_run = candidates[-1]
                run_dir = summary_path_run.parent

        if not summary_path_run.is_file():
            print(f'  WARNING: no run_summary.json for {task}/{condition}/seed{seed}')
            base_row.update({k: math.nan for k in FIELDNAMES if k not in base_row})
            base_row['outcome'] = 'infra_invalid'
            base_row['is_invalid'] = True
            all_rows.append(base_row)
            continue

        summary = json.loads(summary_path_run.read_text(encoding='utf-8'))
        task_info = TASK_INFO.get(task)
        metrics = _compute_run_metrics(run_dir, summary, gp_interp, task_info)
        classification = _classify_outcome(summary)
        base_row.update(metrics)
        base_row.update(classification)
        all_rows.append(base_row)
        tag = (
            'clean' if classification['is_clean_success']
            else 'near'  if classification['is_near_success']
            else 'coll'  if classification['is_collision']
            else 'pen'   if classification['is_penetration']
            else 'time'  if classification['is_timeout']
            else 'inv'   if classification['is_invalid']
            else '?'
        )
        f_shadow = metrics['f_shadow'] if math.isfinite(metrics['f_shadow']) else float('nan')
        print(f'  {task:<22} {condition} seed{seed}: {tag:<5} '
              f'L={metrics["path_length_m"]:.2f}m  ē={metrics["mean_loc_error_m"]:.3f}m  '
              f'f_shadow={f_shadow:.2f}')

    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: _format(row.get(k)) for k in FIELDNAMES})
    print(f'\nMetrics written to: {out_path}')

    summary_text = _print_summary(all_rows)
    print('\n' + summary_text)
    summary_path.write_text(summary_text, encoding='utf-8')
    print(f'Summary written to: {summary_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
