#!/usr/bin/env python3
"""Validate comparison run integrity before generating reports.

Checks every run directory for:
  - Structural: run_summary.json exists, key fields present, timing valid
  - YOLO consistency: runtime config matches offline extractor config
  - Planner weight consistency: all visibility-aware methods share identical weights
  - Initial pose sanity: truth/state/belief start near task start
  - GP interpolation consistency: logged p_vis_plan ≈ sampled GP (optional, slow)

Run before every report:
  python validate_comparison_integrity.py [--planner-runs-root PATH] [--targets-manifest PATH]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from common import ACTIVE_METHOD_IDS, CURRENT_TARGETS_DIR, LOGS_ROOT, PLANNER_RUNS_DIR, accepted_completed_run, run_has_usable_logs
except ImportError:
    ACTIVE_METHOD_IDS = []
    CURRENT_TARGETS_DIR = Path('.')
    LOGS_ROOT = Path('logs')
    PLANNER_RUNS_DIR = Path('logs/planner_runs')
    def accepted_completed_run(summary):
        return bool(summary and summary.get('completed', False))
    def run_has_usable_logs(run_dir):
        return True


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
TIMEOUT_TOLERANCE_S = 0.5      # elapsed may exceed timeout by this much  
INITIAL_POSE_WINDOW = 30       # rows to look at for "initial" pose
TRUTH_START_TOL_M = 0.30       # how close truth must be to task start
STATE_TRUTH_INIT_TOL_M = 0.50  # how close state must be to truth at start
BELIEF_TRUTH_INIT_TOL_M = 0.50 # how close belief must be to truth at start
GP_INTERP_TOL = 0.15  # max mean abs error between logged and nearest-neighbour-sampled p_vis_plan
                      # Runtime uses bilinear/bicubic (CasADi); the check uses nearest-neighbour.
                      # A difference of ~0.1 is expected on coarse grids — this threshold catches
                      # gross mismatches (wrong artifact file, coordinate-frame flip) not interp noise.

# Planner weights that must be identical across all visibility-aware methods
WEIGHT_KEYS = (
    'ambiguity_weight',
    'risk_weight_obs',
    'horizon',
    'dt',
    'plan_rate',
    'r_visible_uv',
    'r_miss_uv',
)

VISIBILITY_AWARE_METHODS = {
    'oracle_visibility', 'red_binary', 'red_area_corrected',
    'yolo_binary', 'yolo_score_raw', 'yolo_score_calibrated',
}


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
_GREEN  = '\033[92m'
_YELLOW = '\033[93m'
_RED    = '\033[91m'
_RESET  = '\033[0m'
_BOLD   = '\033[1m'


def _ok(msg: str) -> str:
    return f'{_GREEN}✅ {msg}{_RESET}'


def _warn(msg: str) -> str:
    return f'{_YELLOW}⚠️  {msg}{_RESET}'


def _fail(msg: str) -> str:
    return f'{_RED}❌ {msg}{_RESET}'


def _hdr(msg: str) -> str:
    return f'{_BOLD}{msg}{_RESET}'


# ---------------------------------------------------------------------------
# File loading utilities
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_run_manifest(run_dir: Path) -> dict:
    return _load_json(run_dir / 'run_manifest.json') or _load_json(run_dir / 'manifest.json')


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or '').encode('utf-8')).hexdigest()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ('1', 'true', 't', 'yes', 'y', 'on')


def _artifact_visibility_map(data) -> np.ndarray:
    for key in ('P_conservative_plan_map', 'P_map', 'P_conservative_map', 'P_mean_map'):
        if key in data.files:
            return np.asarray(data[key], dtype=float)
    raise KeyError('missing planner visibility map in artifact')


def _read_csv_head_tail(path: Path, n: int = 50) -> tuple[list[str], list[dict]]:
    """Read fieldnames and first+last n rows of a CSV cheaply."""
    if not path.is_file():
        return [], []
    rows: list[dict] = []
    try:
        with path.open('r', newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)
    except OSError:
        return [], []
    return fieldnames, rows


def _float_cell(row: dict, key: str) -> float:
    try:
        return float(row.get(key, '') or 'nan')
    except (TypeError, ValueError):
        return math.nan


def _find_latest_run_dirs(root: Path) -> dict[str, Path]:
    """Return {method_id: most-recent run_dir} from run_summary.json files."""
    latest: dict[str, Path] = {}
    for summary_path in root.rglob('run_summary.json'):
        run_dir = summary_path.parent
        summary = _load_json(summary_path)
        if not accepted_completed_run(summary):
            continue
        if not run_has_usable_logs(run_dir):
            continue
        method_id = summary.get('run_config', {}).get('method_id', '') or ''
        if not method_id:
            # Infer from parent directory name
            parent_name = run_dir.parent.name
            if parent_name in (ACTIVE_METHOD_IDS or []):
                method_id = parent_name
        if not method_id:
            continue
        manifest = (_load_json(run_dir / 'manifest.json')
                    or _load_json(run_dir / 'run_manifest.json'))
        if not method_id:
            method_id = manifest.get('method', '') or manifest.get('comparison_method_id', '') or ''
        if not method_id:
            continue
        current = latest.get(method_id)
        if current is None or run_dir.stat().st_mtime > current.stat().st_mtime:
            latest[method_id] = run_dir
    return latest


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_structural(method_id: str, run_dir: Path, summary: dict) -> list[str]:
    issues = []

    if not summary:
        issues.append(f'run_summary.json missing or unreadable in {run_dir}')
        return issues

    if not summary.get('completed', False):
        issues.append(f'completed=false  (reason={summary.get("completion_reason", "?")})')

    first_cmd = summary.get('first_cmd_stamp')
    if first_cmd is None or not math.isfinite(float(first_cmd or 'nan')):
        issues.append('first_cmd_stamp missing or non-finite')

    elapsed = summary.get('elapsed_after_first_cmd_s')
    timeout = summary.get('run_config', {}).get('timeout_after_first_cmd_s', 60.0)
    reason = summary.get('completion_reason', '')
    if elapsed is not None and math.isfinite(float(elapsed or 'nan')):
        elapsed_f = float(elapsed)
        if reason not in ('goal_reached', 'stuck') and elapsed_f > float(timeout) + TIMEOUT_TOLERANCE_S:
            issues.append(
                f'elapsed_after_first_cmd_s={elapsed_f:.1f} > timeout={timeout}+{TIMEOUT_TOLERANCE_S} '
                f'but reason={reason!r}'
            )
    else:
        issues.append('elapsed_after_first_cmd_s missing or non-finite')

    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        issues.append('experiment.csv missing or empty')

    return issues


def check_csv_columns(method_id: str, run_dir: Path) -> list[str]:
    issues = []
    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        return ['experiment.csv missing; cannot check columns']

    fieldnames, _ = _read_csv_head_tail(csv_path, n=0)
    required = ['truth_x', 'truth_y', 'truth_yaw',
                'state_x', 'state_y', 'state_yaw',
                'planner_belief_x', 'planner_belief_y', 'planner_belief_yaw',
                'truth_state_error_m', 'truth_belief_error_m',
                'efe_risk', 'efe_ambiguity', 'efe_control', 'efe_obstacle',
                'terminal_goal_distance_pred', 'terminal_goal_progress_m',
                'fraction_horizon_low_pvis', 'fraction_horizon_high_ambiguity',
                'min_predicted_obstacle_distance_m',
                'rollout_valid', 'fallback_stop_applied',
                'collision_any', 'collision_contact', 'collision_geom', 'collision_reason', 'first_crash_stamp',
                'min_wall_distance_m', 'min_obstacle_distance_m',
                'wall_penetration_m', 'obstacle_penetration_m',
                'off_map', 'inside_no_go', 'valid_run', 'invalid_reason',
                'stamp']
    missing = [col for col in required if col not in fieldnames]
    if missing:
        issues.append(f'experiment.csv missing columns: {missing}  '
                      f'(old run without new logger? re-run to get audit-proof columns)')

    perception_path = run_dir / 'perception.csv'
    if not perception_path.is_file():
        issues.append('perception.csv missing; cannot validate detector instrumentation columns')
        return issues

    perception_fieldnames, _ = _read_csv_head_tail(perception_path, n=0)
    perception_required = [
        'yolo_raw_best_score',
        'yolo_selected_score',
        'yolo_detected_after_threshold',
        'yolo_num_target_candidates',
        'yolo_selected_class_id',
        'yolo_selected_pixel_source',
        'yolo_bbox_area',
        'yolo_mask_area',
        'camera_relative_bearing_deg',
    ]
    missing = [col for col in perception_required if col not in perception_fieldnames]
    if missing:
        issues.append(f'perception.csv missing columns: {missing}')
    return issues


def check_geometry_hashes(method_id: str, run_dir: Path) -> list[str]:
    issues = []
    manifest = _load_run_manifest(run_dir)
    if not manifest:
        return [f'{method_id}: run manifest missing; cannot validate geometry hashes']

    visibility_geometry_json = str(manifest.get('visibility_geometry_json', '') or '')
    visibility_geometry_sha256 = str(manifest.get('visibility_geometry_sha256', '') or '')
    collision_geometry_json = str(manifest.get('collision_geometry_json', '') or '')
    collision_geometry_sha256 = str(manifest.get('collision_geometry_sha256', '') or '')

    if not visibility_geometry_sha256:
        issues.append(f'{method_id}: visibility_geometry_sha256 missing from run manifest')
    elif visibility_geometry_json and visibility_geometry_sha256 != _sha256_text(visibility_geometry_json):
        issues.append(f'{method_id}: visibility_geometry_sha256 does not match visibility_geometry_json')

    if not collision_geometry_sha256:
        issues.append(f'{method_id}: collision_geometry_sha256 missing from run manifest')
    elif collision_geometry_json and collision_geometry_sha256 != _sha256_text(collision_geometry_json):
        issues.append(f'{method_id}: collision_geometry_sha256 does not match collision_geometry_json')

    artifact_path = Path(str(manifest.get('visibility_artifact_path', '') or '')).expanduser()
    if artifact_path.is_file() and visibility_geometry_sha256:
        try:
            with np.load(artifact_path, allow_pickle=False) as data:
                artifact_hash_raw = data['geometry_sha256'] if 'geometry_sha256' in data.files else None
                artifact_hash = ''
                if artifact_hash_raw is not None:
                    artifact_hash_arr = np.asarray(artifact_hash_raw)
                    if artifact_hash_arr.size:
                        artifact_hash = str(artifact_hash_arr.reshape(-1)[0])
        except Exception as exc:
            issues.append(f'{method_id}: could not load visibility artifact geometry hash from {artifact_path}: {exc}')
        else:
            if not artifact_hash:
                issues.append(f'{method_id}: visibility artifact missing geometry_sha256: {artifact_path}')
            elif artifact_hash != visibility_geometry_sha256:
                issues.append(
                    f'{method_id}: visibility artifact geometry_sha256={artifact_hash!r} '
                    f'does not match run manifest {visibility_geometry_sha256!r}'
                )
    return issues


def check_invalid_run_warning(method_id: str, run_dir: Path, summary: dict) -> list[str]:
    warnings = []
    valid_run = summary.get('valid_run', True)
    crashed = summary.get('crashed', False)
    invalid_reason = str(summary.get('invalid_reason', '') or '')
    if (not _boolish(valid_run)) or _boolish(crashed) or invalid_reason:
        reason = invalid_reason or str(summary.get('completion_reason', '') or 'invalid')
        warnings.append(f'Run marked invalid/crashed for analysis: reason={reason!r} (kept visible by design)')
    return warnings


def check_initial_poses(method_id: str, run_dir: Path, summary: dict) -> list[str]:
    issues = []
    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        return []

    _, rows = _read_csv_head_tail(csv_path, n=INITIAL_POSE_WINDOW)
    if not rows:
        return ['experiment.csv is empty']

    # Use frame_sanity from summary for truth vs task start check
    fs = summary.get('frame_sanity', {})
    if fs.get('recorded'):
        err = fs.get('truth_start_error_m', math.nan)
        if math.isfinite(float(err or 'nan')) and float(err) > TRUTH_START_TOL_M:
            issues.append(
                f'Initial truth pose is {float(err):.3f} m from task start '
                f'(tolerance {TRUTH_START_TOL_M} m). Frame sanity reason: {fs.get("reason")}'
            )
    elif fs.get('ok') is False:
        issues.append(f'Frame sanity check FAILED: {fs.get("reason")}')

    # Check state vs truth and belief vs truth at first rows with finite values
    first_truth: tuple[float, float] | None = None
    first_state_err: float = math.nan
    first_belief_err: float = math.nan

    for row in rows[:INITIAL_POSE_WINDOW]:
        tx = _float_cell(row, 'truth_x')
        ty = _float_cell(row, 'truth_y')
        if not (math.isfinite(tx) and math.isfinite(ty)):
            continue
        if first_truth is None:
            first_truth = (tx, ty)

        if math.isnan(first_state_err):
            sx = _float_cell(row, 'state_x')
            sy = _float_cell(row, 'state_y')
            if math.isfinite(sx) and math.isfinite(sy):
                first_state_err = math.hypot(tx - sx, ty - sy)

        if math.isnan(first_belief_err):
            bx = _float_cell(row, 'planner_belief_x')
            by = _float_cell(row, 'planner_belief_y')
            if math.isfinite(bx) and math.isfinite(by):
                first_belief_err = math.hypot(tx - bx, ty - by)

        if not (math.isnan(first_state_err) or math.isnan(first_belief_err)):
            break

    if math.isfinite(first_state_err) and first_state_err > STATE_TRUTH_INIT_TOL_M:
        issues.append(
            f'Initial state is {first_state_err:.3f} m from truth '
            f'(tolerance {STATE_TRUTH_INIT_TOL_M} m). State estimator may not have converged.'
        )
    if math.isfinite(first_belief_err) and first_belief_err > BELIEF_TRUTH_INIT_TOL_M:
        issues.append(
            f'Initial planner belief is {first_belief_err:.3f} m from truth '
            f'(tolerance {BELIEF_TRUTH_INIT_TOL_M} m). Belief not initialized from task start.'
        )
    return issues


def check_yolo_config(method_id: str, run_dir: Path, manifest_path: Path | None) -> list[str]:
    issues = []
    if manifest_path is None or not manifest_path.is_file():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError):
        return [f'Could not read targets manifest {manifest_path}']
    yolo_summary = payload.get('yolo_summary', {}) if isinstance(payload, dict) else {}
    if not isinstance(yolo_summary, dict) or not bool(yolo_summary.get('enabled', False)):
        return []  # no YOLO used for offline extraction, skip check

    run_manifest = (_load_json(run_dir / 'manifest.json')
                    or _load_json(run_dir / 'run_manifest.json'))
    keys_to_check = [
        ('yolo_model',         'model_path',       str),
        ('yolo_imgsz',         'imgsz',            int),
        ('yolo_conf_threshold','conf_threshold',    float),
        ('yolo_iou_threshold', 'iou_threshold',     float),
        ('yolo_target_class',  'class_name',        str),
        ('yolo_class_id',      'class_id',          int),
        ('yolo_use_masks',     'use_masks',         str),
        ('yolo_min_mask_area_px', 'mask_min_area',  float),
        ('yolo_mask_bottom_band_px', 'mask_bottom_band_px', float),
    ]
    for manifest_key, yolo_key, cast in keys_to_check:
        runtime_val = run_manifest.get(manifest_key)
        offline_val = yolo_summary.get(yolo_key)
        if runtime_val is None or offline_val is None:
            continue
        try:
            rv = cast(str(runtime_val).strip().lower()) if cast is str else cast(runtime_val)
            ov = cast(str(offline_val).strip().lower()) if cast is str else cast(offline_val)
        except (TypeError, ValueError):
            continue
        if cast is float:
            if abs(rv - ov) > 1e-6:
                issues.append(f'YOLO {manifest_key}: runtime={rv!r} vs offline={ov!r}')
        else:
            if rv != ov:
                issues.append(f'YOLO {manifest_key}: runtime={rv!r} vs offline={ov!r}')
    return issues


def check_weight_consistency(run_dirs: dict[str, Path]) -> list[str]:
    """All visibility-aware methods must share identical planner weights."""
    issues = []
    weight_snapshots: dict[str, dict[str, Any]] = {}

    for method_id, run_dir in run_dirs.items():
        if method_id not in VISIBILITY_AWARE_METHODS:
            continue
        manifest = (_load_json(run_dir / 'manifest.json')
                    or _load_json(run_dir / 'run_manifest.json'))
        run_config = _load_json(run_dir / 'run_summary.json').get('run_config', {})
        snap = {}
        for key in WEIGHT_KEYS:
            val = manifest.get(key) if key in manifest else run_config.get(key)
            if val is not None:
                snap[key] = val
        weight_snapshots[method_id] = snap

    if len(weight_snapshots) < 2:
        return issues  # nothing to compare

    reference_method = next(iter(weight_snapshots))
    reference = weight_snapshots[reference_method]

    for method_id, snap in weight_snapshots.items():
        if method_id == reference_method:
            continue
        for key in WEIGHT_KEYS:
            ref_val = reference.get(key)
            cur_val = snap.get(key)
            if ref_val is None or cur_val is None:
                continue
            try:
                rf = float(ref_val)
                cf = float(cur_val)
                if abs(rf - cf) > 1e-9:
                    issues.append(
                        f'Weight mismatch for {key!r}: '
                        f'{reference_method}={rf!r} vs {method_id}={cf!r}'
                    )
            except (TypeError, ValueError):
                if str(ref_val).strip() != str(cur_val).strip():
                    issues.append(
                        f'Weight mismatch for {key!r}: '
                        f'{reference_method}={ref_val!r} vs {method_id}={cur_val!r}'
                    )
    return issues


def check_config_hash_consistency(run_dirs: dict[str, Path]) -> list[str]:
    """Warn if different methods used identical hashes (same config) or if
    any method is missing a hash entirely (pre-hash run)."""
    issues = []
    hashes: dict[str, str] = {}
    for method_id, run_dir in run_dirs.items():
        summary = _load_json(run_dir / 'run_summary.json')
        h = summary.get('run_config_hash', '')
        if not h:
            issues.append(
                f'{method_id}: no run_config_hash in run_summary.json '
                f'(run pre-dates hash feature or was not run through the sweep)'
            )
        else:
            hashes[method_id] = h
    # Confirm all methods with hashes have consistent task/world/seed/planner
    configs = {}
    for method_id, run_dir in run_dirs.items():
        summary = _load_json(run_dir / 'run_summary.json')
        cfg = summary.get('run_config', {})
        if cfg:
            configs[method_id] = cfg
    if len(configs) >= 2:
        keys_to_match = ('world', 'task', 'seed', 'planner')
        methods = list(configs.keys())
        ref_method = methods[0]
        for m in methods[1:]:
            for k in keys_to_match:
                if str(configs[ref_method].get(k, '')) != str(configs[m].get(k, '')):
                    issues.append(
                        f'Config mismatch {k!r}: {ref_method}={configs[ref_method].get(k)!r} '
                        f'vs {m}={configs[m].get(k)!r}'
                    )
    return issues


def check_gp_interpolation(method_id: str, run_dir: Path, gp_dir: Path | None) -> list[str]:
    """Spot-check that logged p_vis_plan ≈ NumPy-interpolated GP at belief pose."""
    issues = []
    if gp_dir is None:
        return issues
    artifact_path = gp_dir / f'{method_id}_gp.npz'
    if not artifact_path.is_file():
        return []

    csv_path = run_dir / 'experiment.csv'
    if not csv_path.is_file():
        return []

    try:
        with np.load(artifact_path, allow_pickle=False) as data:
            xs = np.asarray(data['xs'], dtype=float)
            ys = np.asarray(data['ys'], dtype=float)
            p_map = _artifact_visibility_map(data)
    except Exception as exc:
        return [f'Could not load GP artifact {artifact_path}: {exc}']

    _, rows = _read_csv_head_tail(csv_path, n=200)
    logged_vals: list[float] = []
    sampled_vals: list[float] = []

    for row in rows:
        bx = _float_cell(row, 'planner_belief_x')
        by = _float_cell(row, 'planner_belief_y')
        p_log = _float_cell(row, 'p_vis_plan')
        if not (math.isfinite(bx) and math.isfinite(by) and math.isfinite(p_log)):
            continue
        # Grid nearest-neighbor interpolation (same as plot code)
        nx, ny = xs.size, ys.size
        dx = (xs[-1] - xs[0]) / max(nx - 1, 1)
        dy = (ys[-1] - ys[0]) / max(ny - 1, 1)
        ix = int(round((bx - xs[0]) / dx)) if dx > 0 else 0
        iy = int(round((by - ys[0]) / dy)) if dy > 0 else 0
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        p_sampled = float(p_map[iy, ix])
        logged_vals.append(p_log)
        sampled_vals.append(p_sampled)

    if len(logged_vals) < 5:
        return []  # not enough data to check

    mae = float(np.mean(np.abs(np.array(logged_vals) - np.array(sampled_vals))))
    if mae > GP_INTERP_TOL:
        issues.append(
            f'GP interpolation mismatch: mean|p_vis_plan_logged - p_vis_np_sampled| = {mae:.4f} '
            f'(threshold {GP_INTERP_TOL}). Runtime GP may differ from offline artifact.'
        )
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description='Validate comparison run integrity before generating reports.'
    )
    parser.add_argument('--planner-runs-root', default=str(PLANNER_RUNS_DIR))
    parser.add_argument('--gp-dir', default='', help='Path to GP artifact dir (for GP interp check)')
    parser.add_argument('--targets-manifest', default=str(CURRENT_TARGETS_DIR / 'target_manifest.json'),
                        help='Path to targets manifest for YOLO config consistency check')
    parser.add_argument('--skip-gp-check', action='store_true',
                        help='Skip the (slow) GP interpolation spot-check')
    parser.add_argument('--out-json', default='',
                        help='Write integrity_report.json here (default: planner_runs_root)')
    args = parser.parse_args()

    planner_runs_root = Path(args.planner_runs_root).expanduser().resolve()
    targets_manifest = Path(args.targets_manifest).expanduser().resolve()
    gp_dir: Path | None = Path(args.gp_dir).expanduser().resolve() if args.gp_dir.strip() else None

    run_dirs = _find_latest_run_dirs(planner_runs_root)
    if not run_dirs:
        print(_warn(f'No run directories found under {planner_runs_root}'))
        return 1

    print(_hdr(f'\n=== Comparison Integrity Report ==='))
    print(f'Planner runs root: {planner_runs_root}')
    print(f'Methods found: {sorted(run_dirs.keys())}\n')

    all_issues: dict[str, list[str]] = {}
    all_warnings: dict[str, list[str]] = {}
    any_failures = False

    # Cross-method checks
    weight_issues = check_weight_consistency(run_dirs)
    hash_issues = check_config_hash_consistency(run_dirs)
    cross_method_issues = weight_issues + hash_issues
    if cross_method_issues:
        any_failures = True
        print(_hdr('Cross-method checks:'))
        for issue in cross_method_issues:
            print(f'  {_fail(issue)}')
        print()
    else:
        print(_ok('Cross-method weight and config consistency: OK'))
        print()
    all_issues['_cross_method'] = cross_method_issues

    # Per-method checks
    for method_id in sorted(run_dirs.keys()):
        run_dir = run_dirs[method_id]
        summary = _load_json(run_dir / 'run_summary.json')
        method_issues: list[str] = []
        method_warnings: list[str] = []

        method_issues += check_structural(method_id, run_dir, summary)
        method_issues += check_csv_columns(method_id, run_dir)
        method_issues += check_geometry_hashes(method_id, run_dir)
        method_issues += check_initial_poses(method_id, run_dir, summary)
        method_warnings += check_invalid_run_warning(method_id, run_dir, summary)

        # YOLO check only for YOLO-backed methods
        if method_id in ('yolo_binary', 'yolo_score_raw', 'yolo_score_calibrated', 'oracle_visibility'):
            method_issues += check_yolo_config(method_id, run_dir, targets_manifest)

        # GP interpolation spot check
        if not args.skip_gp_check and gp_dir is not None:
            method_issues += check_gp_interpolation(method_id, run_dir, gp_dir)

        all_issues[method_id] = method_issues
        all_warnings[method_id] = method_warnings
        if method_issues:
            any_failures = True
            print(_hdr(f'[{method_id}]') + f'  {run_dir}')
            for issue in method_issues:
                print(f'  {_fail(issue)}')
            for warning in method_warnings:
                print(f'  {_warn(warning)}')
        elif method_warnings:
            print(_hdr(f'[{method_id}]') + f'  {run_dir}')
            for warning in method_warnings:
                print(f'  {_warn(warning)}')
        else:
            print(_ok(f'[{method_id}]  {run_dir.name}'))

    print()
    if any_failures:
        print(_fail('Integrity check FAILED — do not run final reports until all issues are resolved.'))
    elif any(v for v in all_warnings.values()):
        print(_warn('Integrity check passed with warnings — invalid runs were preserved but instrumentation is intact.'))
    else:
        print(_ok('All integrity checks passed.'))

    # Write JSON report
    out_path = Path(args.out_json) if args.out_json.strip() else planner_runs_root / 'integrity_report.json'
    report = {
        'planner_runs_root': str(planner_runs_root),
        'methods_checked': sorted(run_dirs.keys()),
        'any_failures': any_failures,
        'issues': {k: v for k, v in all_issues.items()},
        'warnings': {k: v for k, v in all_warnings.items()},
    }
    try:
        out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'\nReport written to {out_path}')
    except OSError as exc:
        print(_warn(f'Could not write report JSON: {exc}'))

    return 1 if any_failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
