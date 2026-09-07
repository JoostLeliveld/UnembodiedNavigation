"""Read only the three registry-selected pilots and retain terminal evidence.

Never reruns navigation analysis/freeze or edits existing experiments. Uses aligned.py
and verifies all files in each frozen selection before reporting measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/fusion_on_fixed_routes'))
import aligned

GROUPS = ('network_navigation_pilot', 'network_navigation_tracking_pilot', 'network_navigation_runtime_pilot')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''): h.update(data)
    return h.hexdigest()


def f(row, key):
    try: return float(row.get(key, 'nan'))
    except (TypeError, ValueError): return math.nan


def snapshot(row):
    keys = ['stamp', 'planner_belief_stamp', 'planner_belief_age_s',
            'planner_belief_x', 'planner_belief_y', 'planner_belief_yaw',
            'planner_cov_x', 'planner_cov_xy', 'planner_cov_y',
            'planner_belief_cov_theta_theta', 'planner_belief_cov_x_theta', 'planner_belief_cov_y_theta',
            'goal_x', 'goal_y', 'operational_goal_dist_m', 'cmd_v', 'cmd_w',
            'cmd_raw_v', 'cmd_raw_w', 'cmd_stamp', 'cmd_raw_stamp',
            'gt_x', 'gt_y', 'gt_yaw', 'gt_stamp', 'goal_dist',
            'exec_wp_idx', 'exec_wp_count', 'exec_wp_target_x', 'exec_wp_target_y',
            'exec_wp_dist_m', 'exec_tracking_yaw', 'exec_yaw_error',
            'exec_cmd_v', 'exec_cmd_w', 'exec_plan_remaining_s', 'exec_controls_len',
            'contact_topic_publishers', 'contact_messages_seen', 'collision_contact',
            'collision_geom', 'first_crash_stamp', 'odom_noisy_stamp', 'odom_noisy_v', 'odom_noisy_w']
    return {k: f(row, k) for k in keys} | {'collision_reason': row.get('collision_reason')}


def main(out):
    registry_path = ROOT / 'docs/localization_metrics_registry.json'
    registry = json.loads(registry_path.read_text())
    records, terminal_rows = [], []
    sources = {str(p.relative_to(ROOT)): digest(p) for p in
               (Path(__file__), Path(aligned.__file__), registry_path)}
    for group in GROUPS:
        selection_path = ROOT / registry[group]['selection']
        selection = json.loads(selection_path.read_text())
        sources[str(selection_path.relative_to(ROOT))] = digest(selection_path)
        assert len(selection['runs']) == 3
        for entry in selection['runs']:
            assert not entry['missing'] and entry['seed'] == 210
            run = ROOT / entry['run']
            for name, sha in entry['files'].items():
                assert digest(run / name) == sha, f'Changed selection: {run / name}'
            table = aligned.rows(run)
            summary = json.loads((run / 'run_summary.json').read_text())
            manifest = json.loads((run / 'run_manifest.json').read_text())
            start, stop = summary['first_cmd_stamp'], summary['stop_stamp']
            selected = [(i, r) for i, r in enumerate(table) if start <= f(r, 'stamp') <= stop]
            index, last = selected[-1]
            truth = aligned.truth_series(run, table)
            aligned_belief = aligned.aligned_error_cm(run, 'belief', table)
            use = (aligned.landed_mask(aligned_belief['stamp']) &
                   np.isfinite(aligned_belief['aligned_cm']) &
                   (aligned_belief['stamp'] >= start) & (aligned_belief['stamp'] <= stop))
            yaw = np.array([f(r, 'planner_belief_yaw') for r in table])[use]
            yaw_truth = truth.yaw_at(aligned_belief['stamp'][use])
            yaw_error = np.degrees(np.abs(np.arctan2(np.sin(yaw-yaw_truth), np.cos(yaw-yaw_truth))))
            finite_yaw = np.isfinite(yaw_error)
            event_rows = aligned.assimilations(run)
            events = [r for r in event_rows if start <= r['apply_stamp'] <= stop]
            accepted = sorted(r['apply_stamp'] for r in events if r['accepted'])
            # Terminal snapshot is the last record received/logged by decision
            # time. A belief first logged AFTER a contact is offline evidence,
            # not a substitute for the state available to online termination.
            terminal_belief = last
            terminal_belief_stamp = f(terminal_belief, 'planner_belief_stamp')
            gxy = np.array(truth.at(np.array([terminal_belief_stamp])))[:, 0]
            target = np.array([f(last, 'exec_wp_target_x'), f(last, 'exec_wp_target_y')])
            with (run / 'global_waypoints.csv').open() as stream:
                waypoints = list(csv.DictReader(stream))
            # Held raw fields are not an authoritative event ledger; retain bounds.
            zero_tail = []
            for _, row in reversed(selected):
                if f(row, 'cmd_v') == f(row, 'cmd_w') == 0.:
                    zero_tail.append(row)
                else: break
            operational = np.array([f(terminal_belief, 'planner_belief_x'), f(terminal_belief, 'planner_belief_y')])
            terminal_yaw_delta = f(terminal_belief, 'planner_belief_yaw')-truth.yaw_at(np.array([terminal_belief_stamp]))[0]
            terminal_yaw_error = abs(math.atan2(math.sin(terminal_yaw_delta), math.cos(terminal_yaw_delta)))
            terminal_P = np.array([
                [f(last, 'planner_cov_x'), f(last, 'planner_cov_xy'), f(last, 'planner_belief_cov_x_theta')],
                [f(last, 'planner_cov_xy'), f(last, 'planner_cov_y'), f(last, 'planner_belief_cov_y_theta')],
                [f(last, 'planner_belief_cov_x_theta'), f(last, 'planner_belief_cov_y_theta'),
                 f(last, 'planner_belief_cov_theta_theta')],
            ])
            assert np.isfinite(terminal_P).all() and np.linalg.eigvalsh(terminal_P).min() > 0
            terminal_e = np.r_[operational-gxy, math.atan2(math.sin(terminal_yaw_delta), math.cos(terminal_yaw_delta))]
            planar_nees = float(terminal_e[:2] @ np.linalg.solve(terminal_P[:2, :2], terminal_e[:2]))
            metric = dict(belief_stamps=int(use.sum()), heading_samples=int(finite_yaw.sum()),
                belief_position_median_cm=float(np.median(aligned_belief['aligned_cm'][use])),
                belief_position_p95_cm=float(np.quantile(aligned_belief['aligned_cm'][use], .95)),
                belief_heading_p95_deg=float(np.nanquantile(yaw_error, .95)),
                terminal_belief_stamp=terminal_belief_stamp,
                terminal_belief_position_error_cm=float(np.linalg.norm(operational-gxy)*100),
                terminal_belief_heading_error_deg=float(np.degrees(terminal_yaw_error)),
                terminal_gt_aligned_xy=gxy.tolist(),
                terminal_planar_nees=planar_nees,
                terminal_planar_inside_nominal_95=bool(planar_nees <= 5.991),
                terminal_pose_nees=float(terminal_e @ np.linalg.solve(terminal_P, terminal_e)),
                reference=truth.source, terminal_samples=1,
                correction_dropped_fraction=sum(r['status']=='dropped' for r in events)/max(len(events), 1),
                longest_correction_gap_s=float(max(np.diff([start, *accepted, stop]))),
                assimilation_status_counts=dict(Counter(r['status'] for r in events)))
            record = dict(group=group, arm=entry['arm'], key=entry['key'], run=entry['run'],
                selection=str(selection_path.relative_to(ROOT)), selected_files_sha256=entry['files'],
                completion_reason=summary['completion_reason'], campaign_outcome=entry['event']['outcome'],
                first_cmd_stamp=start, stop_stamp=stop, final_csv_stamp=f(table[-1], 'stamp'),
                rows_after_stop=sum(f(r, 'stamp') > stop for r in table),
                nearest_pre_stop_csv_line=index+2, last_before_stop=snapshot(last),
                stop_minus_last_log_s=stop-f(last, 'stamp'),
                observed_terminal_zero_since=f(zero_tail[-1], 'stamp') if zero_tail else None,
                offline_final_goal_distance_m=summary['final_goal_distance'],
                contact=summary['collision_contact'], geometry_collision=summary['collision_geom'],
                contact_messages=summary['contact_messages_seen'], contact_publishers=summary['contact_topic_publishers'],
                crash_stamp=summary['first_crash_stamp'], collision_reason=summary['collision_reason'],
                target_matches_saved_waypoint=bool(np.allclose(target, [float(waypoints[int(f(last,'exec_wp_idx'))][k]) for k in ('x','y')], atol=1e-5)),
                route_count=len(waypoints), route_terminal=waypoints[-1], metrics=metric,
                config={k: manifest.get(k) for k in ('logging_schema_version', 'use_hierarchical',
                    'global_planner_mode', 'local_controller_type', 'waypoint_spacing_m',
                    'waypoint_arrival_radius_m', 'goal_success_radius', 'goal_success_hold_s',
                    'goal_stable_radius', 'goal_stable_hold_s', 'stuck_window_s',
                    'stuck_max_displacement_m', 'stuck_max_goal_improvement_m',
                    'stuck_cmd_fraction_min', 'stuck_idle_cmd_fraction_max', 'local_plan_rate', 'cmd_publish_rate')})
            records.append(record)
            terminal_rows.append(dict(group=group, arm=entry['arm'], run=run.name,
                reason=record['completion_reason'], decision_s=stop, csv_s=f(last, 'stamp'),
                belief_s=f(last,'planner_belief_stamp'), op_gap_m=f(last, 'operational_goal_dist_m'),
                raw_v=f(last,'cmd_raw_v'), raw_w=f(last,'cmd_raw_w'),
                out_v=f(last,'cmd_v'), out_w=f(last,'cmd_w'),
                wp=f(last,'exec_wp_idx'), wp_count=len(waypoints),
                final_heading_error_deg=metric['terminal_belief_heading_error_deg'],
                final_position_error_cm=metric['terminal_belief_position_error_cm'],
                contact=summary['collision_contact'], contact_count=summary['contact_messages_seen'],
                offline_goal_gap_m=summary['final_goal_distance']))
            print(terminal_rows[-1], flush=True)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'run_traces.json').write_text(json.dumps(dict(
        kind='three_separate_registry_selected_diagnostic_pilots_not_pooled',
        sources=sources, results=records,
        limits='Exact logger decision/contact stamps retained. A last logged/published belief is not '
               'the controller computation snapshot; output Twist is not physical velocity; no '
               'applied command acknowledgement or stopped-pose dwell is available.'), indent=2) + '\n')
    with (out / 'terminal_rows.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(terminal_rows[0]))
        writer.writeheader(); writer.writerows(terminal_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
