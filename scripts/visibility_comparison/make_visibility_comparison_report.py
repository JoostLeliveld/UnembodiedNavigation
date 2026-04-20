#!/usr/bin/env python3
"""Assemble a compact report for the shared perception-to-visibility comparison."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import tempfile
from pathlib import Path

from common import (
    ACTIVE_METHOD_IDS,
    CURRENT_CAPTURE_DIR,
    CURRENT_GP_DIR,
    CURRENT_TARGETS_DIR,
    LOGS_ROOT,
    PLANNER_RUNS_DIR,
    REPORT_DIR,
    accepted_completed_run,
    choose_preview_rows,
    read_csv_rows,
    run_has_usable_logs,
    write_csv,
    write_manifest,
)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.is_file() else []


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    import json
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _target_column_available(rows: list[dict[str, str]], column: str) -> bool:
    for row in rows:
        if str(row.get(column, '') or '').strip() != '':
            return True
    return False


def _copy_if_exists(src: Path, dst: Path) -> str:
    if not src.is_file():
        return ''
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _copy_tree_pngs(src_root: Path, dst_root: Path) -> list[str]:
    copied = []
    if not src_root.is_dir():
        return copied
    for src in src_root.rglob('*.png'):
        rel = src.relative_to(src_root)
        out = _copy_if_exists(src, dst_root / rel)
        if out:
            copied.append(out)
    return copied


def _latest_run_dir(root: Path, method_id: str) -> Path | None:
    candidates = []
    for p in root.rglob('run_summary.json'):
        if p.parent.parent.name == method_id or p.parent.name == method_id:
            summary = _load_json(p)
            if accepted_completed_run(summary) and run_has_usable_logs(p.parent):
                candidates.append(p.parent)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_metrics(run_dir: Path | None, method_id: str, field_summary_rows: list[dict]) -> dict[str, str]:
    base = {
        'run_available': '0', 'run_dir': '',
        'completed': '', 'completion_reason': '',
        'frame_sanity_recorded': '', 'frame_sanity_ok': '', 'frame_sanity_reason': '',
        'frame_truth_start_error_m': '', 'frame_raw_start_error_m': '',
        'elapsed_after_first_cmd_s': '', 'path_length_m': '',
        'final_goal_distance': '', 'minimum_goal_distance': '',
        'mean_solve_time_ms': '', 'mean_efe_risk': '',
        'mean_efe_ambiguity': '', 'mean_p_vis_plan': '',
        'mean_p_vis_plan_eff': '', 'mean_r_plan_u_std': '', 'mean_r_plan_v_std': '',
        'mean_truth_state_error_m': '', 'mean_truth_belief_error_m': '',
        'mean_state_pos_error_m': '', 'max_state_pos_error_m': '',
        'mean_state_cov_trace': '', 'max_state_cov_trace': '',
        'mean_state_sigma_major_m': '', 'max_state_sigma_major_m': '',
        'mean_path_ambiguity': '', 'max_path_ambiguity': '',
        'mean_path_r_plan_uv_std': '', 'max_path_r_plan_uv_std': '',
        'fraction_path_in_high_ambiguity_region': '',
        'fraction_time_in_high_ambiguity_region': '',
        'fraction_time_p_vis_below_0_2': '',
        'fraction_time_p_vis_eff_below_0_2': '',
        'field_p_vis_mean': '', 'field_ambiguity_mean': '', 'field_r_plan_uv_std_mean': ''
    }
    
    for row in field_summary_rows:
        if row.get('method_id') == method_id:
            base['field_p_vis_mean'] = str(row.get('p_vis_mean', ''))
            base['field_ambiguity_mean'] = str(row.get('ambiguity_mean', ''))
            base['field_r_plan_uv_std_mean'] = str(row.get('r_plan_uv_std_mean', ''))
            break
            
    if run_dir is None:
        return base
        
    summary_path = run_dir / 'run_summary.json'
    if not summary_path.is_file():
        return base
        
    summary = _load_json(summary_path)
    base['run_available'] = '1' if _run_has_usable_logs(run_dir) else '0'
    base['run_dir'] = str(run_dir)
    
    for k in [
        'completed', 'completion_reason',
        'frame_sanity_recorded', 'frame_sanity_ok', 'frame_sanity_reason',
        'frame_truth_start_error_m', 'frame_raw_start_error_m',
        'elapsed_after_first_cmd_s', 'path_length_m',
        'final_goal_distance', 'minimum_goal_distance', 'mean_solve_time_ms',
        'mean_efe_risk', 'mean_efe_ambiguity', 'mean_p_vis_plan', 'mean_p_vis_plan_eff',
        'mean_r_plan_u_std', 'mean_r_plan_v_std',
        'mean_truth_state_error_m', 'mean_truth_belief_error_m',
        'mean_state_pos_error_m', 'max_state_pos_error_m',
        'mean_state_cov_trace', 'max_state_cov_trace',
        'mean_state_sigma_major_m', 'max_state_sigma_major_m',
        'mean_path_ambiguity', 'max_path_ambiguity',
        'mean_path_r_plan_uv_std', 'max_path_r_plan_uv_std',
        'fraction_path_in_high_ambiguity_region', 'fraction_time_in_high_ambiguity_region',
        'fraction_time_p_vis_below_0_2', 'fraction_time_p_vis_eff_below_0_2',
        'max_r_plan_std',
    ]:
        val = summary.get(k, '')
        if isinstance(val, float):
            base[k] = f'{val:.6f}'
        else:
            base[k] = str(val)
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description='Assemble the shared visibility-comparison report folder.')
    parser.add_argument('--capture-dir', default=str(CURRENT_CAPTURE_DIR))
    parser.add_argument('--targets-dir', default=str(CURRENT_TARGETS_DIR))
    parser.add_argument('--gp-dir', default=str(CURRENT_GP_DIR))
    parser.add_argument('--planner-runs-root', default=str(PLANNER_RUNS_DIR))
    parser.add_argument('--out', default=str(REPORT_DIR))
    parser.add_argument('--preview-count', type=int, default=16)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    targets_dir = Path(args.targets_dir).expanduser().resolve()
    gp_dir = Path(args.gp_dir).expanduser().resolve()
    planner_runs_root = Path(args.planner_runs_root).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    allowed_root = LOGS_ROOT.resolve()
    if allowed_root not in output_dir.parents and output_dir != allowed_root:
        raise RuntimeError(f'Report output must stay under {allowed_root}: {output_dir}')
    staged_path_plots_root = None
    staged_path_plots = None
    if output_dir.exists():
        existing_path_plots = output_dir / 'path_plots'
        if existing_path_plots.is_dir():
            staged_path_plots_root = Path(tempfile.mkdtemp(prefix='path_plots_', dir=str(LOGS_ROOT)))
            staged_path_plots = staged_path_plots_root / 'path_plots'
            shutil.copytree(existing_path_plots, staged_path_plots)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    assets_capture = output_dir / 'assets' / 'capture_previews'
    assets_gp = output_dir / 'assets' / 'gp_maps'
    assets_calibration = output_dir / 'assets' / 'calibration'
    assets_paths = output_dir / 'assets' / 'path_plots'
    assets_capture.mkdir(parents=True, exist_ok=False)
    assets_gp.mkdir(parents=True, exist_ok=False)
    assets_calibration.mkdir(parents=True, exist_ok=False)
    assets_paths.mkdir(parents=True, exist_ok=False)

    sample_rows = _load_csv_rows(capture_dir / 'samples.csv')
    gp_target_rows = _load_csv_rows(targets_dir / 'gp_targets.csv')
    capture_preview_rows = choose_preview_rows(sample_rows, int(args.preview_count))
    copied_capture = []
    for row in capture_preview_rows:
        preview_rel = str(row.get('preview_path', '')).strip()
        if not preview_rel:
            continue
        src = capture_dir / preview_rel
        dst = assets_capture / Path(preview_rel).name
        out = _copy_if_exists(src, dst)
        if out:
            copied_capture.append(out)

    gp_summary_rows = _load_csv_rows(gp_dir / 'gp_fit_summary.csv')
    gp_summary_by_method = {row['method_id']: row for row in gp_summary_rows if row.get('method_id')}
    field_summary_rows = _load_csv_rows(gp_dir / 'field_method_summary.csv')
    if not field_summary_rows:
        field_summary_rows = _load_csv_rows(gp_dir / 'plots' / 'field_method_summary.csv')
    calibration_manifest = _load_json(targets_dir / 'calibration' / 'manifest.json')
    gp_plot_manifest = _load_json(gp_dir / 'plots' / 'plot_manifest.json')
    path_plot_manifest = _load_json((output_dir.parent / 'path_plots' / 'plot_manifest.json'))
    if not path_plot_manifest:
        path_plot_manifest = _load_json(REPORT_DIR / 'path_plots' / 'plot_manifest.json')
    if not path_plot_manifest and staged_path_plots is not None:
        path_plot_manifest = _load_json(staged_path_plots / 'plot_manifest.json')

    method_rows = []
    for method_id in ACTIVE_METHOD_IDS:
        gp_row = gp_summary_by_method.get(method_id, {})
        gp_available = str(gp_row.get('available', '0') or '0')
        run_dir = _latest_run_dir(planner_runs_root, method_id)
        run_metrics = _run_metrics(run_dir, method_id, field_summary_rows)
        target_available = '0' if method_id == 'visibility_unaware_baseline' else str(
            int(_target_column_available(gp_target_rows, method_id))
        )
        method_rows.append({
            'method_id': method_id,
            'target_available': target_available,
            'gp_available': gp_available,
            'train_points': str(gp_row.get('train_points', '') or ''),
            'target_mean': str(gp_row.get('target_mean', '') or ''),
            **run_metrics,
        })

    gp_assets = []
    for entry in gp_plot_manifest.get('method_plots', []):
        plot_path = Path(str(entry.get('plot_path', '') or '')).expanduser()
        if plot_path.is_file():
            dst = assets_gp / plot_path.name
            gp_assets.append(_copy_if_exists(plot_path, dst))
    combined_gp = Path(str(gp_plot_manifest.get('combined_plot', '') or '')).expanduser()
    if combined_gp.is_file():
        _copy_if_exists(combined_gp, assets_gp / combined_gp.name)

    calibration_assets = _copy_tree_pngs(targets_dir / 'calibration', assets_calibration)
    calibration_artifact_dst = _copy_if_exists(
        targets_dir / 'yolo_score_calibration.json',
        assets_calibration / 'yolo_score_calibration.json',
    )

    path_assets = []
    for entry in path_plot_manifest.get('method_entries', []):
        plot_path = Path(str(entry.get('plot_path', '') or '')).expanduser()
        if plot_path.is_file():
            dst = assets_paths / plot_path.name
            path_assets.append(_copy_if_exists(plot_path, dst))

    path_plots_dir = output_dir.parent / 'path_plots'
    if not path_plots_dir.is_dir():
        path_plots_dir = REPORT_DIR / 'path_plots'
    if not path_plots_dir.is_dir() and staged_path_plots is not None:
        path_plots_dir = staged_path_plots
    path_assets.extend(_copy_tree_pngs(path_plots_dir, assets_paths))
    if path_plots_dir.is_dir():
        _copy_tree_pngs(path_plots_dir, output_dir / 'path_plots')

    write_csv(
        output_dir / 'method_table.csv',
        (
            'method_id',
            'target_available',
            'gp_available',
            'train_points',
            'target_mean',
            'run_available',
            'run_dir',
            'completed',
            'completion_reason',
            'frame_sanity_recorded',
            'frame_sanity_ok',
            'frame_sanity_reason',
            'frame_truth_start_error_m',
            'frame_raw_start_error_m',
            'elapsed_after_first_cmd_s',
            'path_length_m',
            'final_goal_distance',
            'minimum_goal_distance',
            'mean_solve_time_ms',
            'mean_efe_risk',
            'mean_efe_ambiguity',
            'mean_p_vis_plan',
            'mean_p_vis_plan_eff',
            'mean_r_plan_u_std',
            'mean_r_plan_v_std',
            'mean_truth_state_error_m',
            'mean_truth_belief_error_m',
            'mean_state_pos_error_m',
            'max_state_pos_error_m',
            'mean_state_cov_trace',
            'max_state_cov_trace',
            'mean_state_sigma_major_m',
            'max_state_sigma_major_m',
            'mean_path_ambiguity',
            'max_path_ambiguity',
            'mean_path_r_plan_uv_std',
            'max_path_r_plan_uv_std',
            'fraction_path_in_high_ambiguity_region',
            'fraction_time_in_high_ambiguity_region',
            'fraction_time_p_vis_below_0_2',
            'fraction_time_p_vis_eff_below_0_2',
            'max_r_plan_std',
            'field_p_vis_mean',
            'field_ambiguity_mean',
            'field_r_plan_uv_std_mean',
        ),
        method_rows,
    )

    report_md = output_dir / 'report.md'
    report_md.write_text(
        '\n'.join([
            '# Visibility Comparison Report',
            '',
            '## Summary',
            '',
            'This report summarizes the current state of the shared perception-to-visibility comparison framework.',
            'Planner traces are explicitly aligned to the first command, and only accepted completed runs are included in the report tables and path summaries.',
            'Per-run frame sanity fields now expose whether transformed truth in `map_bev` matched the task start pose before motion began.',
            'Oracle visibility, red binary, and the three YOLO-derived targets can all appear here when they are present in the current targets/GP folders.',
            'Red corrected area may still be absent if it has not been built yet.',
            'The path-plots folder now includes actual-vs-inferred state overlays, state-certainty maps, ambiguity-region overlays, and uncertainty-propagation sheets when experiment logs contain the needed belief/state fields.',
            'YOLO calibration assets are copied alongside the GP and path figures so score reliability is visible next to the planner results.',
            '',
            '## Included assets',
            '',
            f'- Capture previews copied: {len(copied_capture)}',
            f'- GP map figures copied: {len([p for p in gp_assets if p])}',
            f'- Calibration figures copied: {len([p for p in calibration_assets if p])}',
            f'- Path figures copied: {len([p for p in path_assets if p])}',
            '',
            '## Method status',
            '',
            'See `method_table.csv` for the current per-method availability and run metrics.',
        ]),
        encoding='utf-8',
    )

    write_manifest(output_dir / 'report_manifest.json', {
        'capture_dir': str(capture_dir),
        'targets_dir': str(targets_dir),
        'gp_dir': str(gp_dir),
        'planner_runs_root': str(planner_runs_root),
        'capture_previews_copied': int(len(copied_capture)),
        'gp_figures_copied': int(len([p for p in gp_assets if p])),
        'calibration_figures_copied': int(len([p for p in calibration_assets if p])),
        'path_figures_copied': int(len([p for p in path_assets if p])),
        'lineage': {
            'capture_manifest': str(capture_dir / 'capture_manifest.json') if (capture_dir / 'capture_manifest.json').is_file() else '',
            'perception_targets_manifest': str(targets_dir / 'manifest.json') if (targets_dir / 'manifest.json').is_file() else '',
            'gp_targets_manifest': str(targets_dir / 'target_manifest.json') if (targets_dir / 'target_manifest.json').is_file() else '',
            'yolo_calibration_artifact': calibration_artifact_dst,
            'yolo_calibration_manifest': str(targets_dir / 'calibration' / 'manifest.json') if (targets_dir / 'calibration' / 'manifest.json').is_file() else '',
            'gp_manifest': str(gp_dir / 'gp_manifest.json') if (gp_dir / 'gp_manifest.json').is_file() else '',
            'gp_plot_manifest': str(gp_dir / 'plots' / 'plot_manifest.json') if (gp_dir / 'plots' / 'plot_manifest.json').is_file() else '',
            'path_plot_manifest': str(path_plots_dir / 'plot_manifest.json') if path_plots_dir.is_dir() and (path_plots_dir / 'plot_manifest.json').is_file() else '',
        },
        'calibration_summary': calibration_manifest,
        'notes': [
            'This report is intentionally compact and resilient to missing methods during the shared-backbone stage.',
            'Missing methods are left blank rather than causing report generation to fail.',
            'Report tables and copied path assets exclude interrupted/incomplete runs by construction.',
        ],
    })
    if staged_path_plots_root is not None:
        shutil.rmtree(staged_path_plots_root, ignore_errors=True)
    print(f'Wrote visibility comparison report to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
