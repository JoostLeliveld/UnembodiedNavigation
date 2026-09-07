#!/usr/bin/env python3
"""Explicitly selected network-planner pilot: aligned measurements and figures.

No selection by directory recency; every configured attempt, including failed
infrastructure, remains in selection.json. One seed per field is descriptive only.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/icra_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from collections import Counter
from functools import lru_cache
import csv
import json
from pathlib import Path
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in ('src/planning', 'src/unav_common', 'src/reliability',
                                      'experiments/fusion_on_fixed_routes')]
import aligned
from study import OUT, digest, writejson
from unav_common.occlusion_geometry import scene_from_json
from planning.planners.base_planner import UnicyclePlannerBase

ARMS = ('P0', 'P1', 'P2')
NAMES = dict(P0='Uniform score', P1='Geometry score', P2='GP score')
COLORS = dict(P0='#687687', P1='#c78435', P2='#207b70')
PROBE = OUT / 'network_planner/full_route_v1'
CONFIG = REPO / 'experiments/icra_commissioning/network_navigation_pilot.yaml'
CAMPAIGN = OUT / 'network_navigation_pilot'
REQUIRED = ('run_manifest.json', 'run_summary.json', 'experiment.csv', 'fusion_observations.csv',
            'correction_assimilations.csv', 'camera_opportunities.jsonl', 'global_plan.csv',
            'global_waypoints.csv', 'global_plan_meta.json')


def style():
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def savefig(fig, path):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(path.with_suffix('.' + extension), dpi=160)
    plt.close(fig)


@lru_cache(maxsize=1)
def camera_positions():
    import joblib
    # Geometry packaged with the frozen mean model; used only for figure annotation.
    model = joblib.load(REPO/'logs/perception_models/box_feature_bias_correction_20260831/models.joblib')
    return {name: np.asarray(value['xy']) for name, value in model['camera_geometry'].items()}


def map_background(ax, settings):
    for p in scene_from_json(settings['driveable_geometry_json']).prisms:
        ax.add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin,
                               fc='#eef1f2', ec='none', zorder=0))
    for p in scene_from_json(settings['collision_geometry_json']).prisms:
        ax.add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin,
                               fc='#c6ced3', ec='#9da7ae', lw=.3, zorder=1))
    for name, xy in camera_positions().items():
        ax.plot(*xy, '^', color='#d89a35', mec='#704710', mew=.4, ms=5, zorder=5)
        ax.annotate(name[-1], xy, xytext=(4, 0), textcoords='offset points', fontsize=7, va='center')
    ax.set(xlim=(-12.3, 12.8), ylim=(-10.4, 10.2), aspect='equal', xlabel='World x [m]', ylabel='World y [m]')
    ax.grid(alpha=.12)


def preflight(out):
    out.mkdir(parents=True, exist_ok=True)
    cached = {}
    if (out/'preflight_results.json').exists():
        previous = json.loads((out/'preflight_results.json').read_text())
        for arm in ARMS:
            p = PROBE/f'{arm}_result.json'
            if previous['sources'][str(p.relative_to(REPO))] != digest(p):
                raise ValueError('Frozen preflight inputs changed')
        cached = {r['arm']: r for r in previous['results']}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.), sharex=True, sharey=True)
    fig.subplots_adjust(left=.06, right=.985, bottom=.17, top=.79, wspace=.14)
    scores = []
    for arm, ax in zip(ARMS, axes):
        record = json.loads((PROBE / f'{arm}_result.json').read_text())
        planner = UnicyclePlannerBase(**record['settings'])
        map_background(ax, record['settings'])
        if arm != 'P0': ax.set_ylabel('')
        for seed in record['seed_results']:
            s = np.asarray(seed['states'])
            ax.plot(s[:, 0], s[:, 1], ':', color='#8d979d', lw=.8, alpha=.7)
        path = np.asarray(record['result']['states'])
        ax.plot(path[:, 0], path[:, 1], color=COLORS[arm], lw=2, label='Selected optimized route')
        ax.plot(*record['state'][:2], 'ko', ms=3)
        ax.plot(*record['goal'], 'k*', ms=9)
        checks = []
        # Evaluate all seed/optimized candidates under the exact runtime validity gate.
        for index, candidate in enumerate([] if arm in cached else record['attempts']):
            d = planner._trajectory_plan_diagnostics(record['state'], record['initial_P'],
                                                     candidate['controls'], record['goal'])
            checks.append(dict(index=index, **d, scaled_cost=candidate['scaled_total']))
        r = record['result']
        # Dense independent physical-clearance walk, including the starting point.
        points = np.vstack([a + t[:, None]*(b-a) for a, b in zip(path[:-1], path[1:])
                            for t in [np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(b[:2]-a[:2])/.02))+1))]])
        clearance = (cached[arm]['min_physical_clearance_m'] if arm in cached else
                     min(planner.collision_clearance_state_np(p) for p in points))
        if arm in cached: checks = cached[arm]['attempts']
        item = dict(arm=arm, goal_gap_m=r['terminal_goal_distance_pred'],
                    path_length_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
                    solve_seconds=r['solve_time_s'], optimizer_success=r['optimizer_success'],
                    optimizer_message=r['optimizer_message'], rollout_valid=r['rollout_valid'],
                    min_physical_clearance_m=float(clearance),
                    max_cost_chart_condition=record['maximum_chart_condition'],
                    selected_source=r['selected_source'], attempts=checks)
        scores.append(item)
        ax.set_title(f"{NAMES[arm]}\nGoal gap {100*item['goal_gap_m']:.1f} cm; solve {item['solve_seconds']:.0f} s")
    axes[0].plot([], [], ':', color='#8d979d', label='Geometry-only seeds')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .025), ncols=2, fontsize=8, frameon=False)
    fig.suptitle('Full-route preflight: same upper corridor for all three fields\n'
                 'All selected solutions reached the 80-iteration limit', fontsize=11, y=.975)
    savefig(fig, out / 'full_route_preflight')
    writejson(out / 'preflight_results.json', dict(kind='model_and_implementation_only',
        conclusion='All three select the upper corridor; all hit the iteration limit. No realized sensor samples.',
        results=scores, sources={str(p.relative_to(REPO)): digest(p) for p in [Path(__file__),
            *[PROBE / f'{a}_result.json' for a in ARMS]]}))


def configured_trials():
    cfg=yaml.safe_load(CONFIG.read_text())
    if len(cfg['tasks'])!=1:raise ValueError('This analyzer requires exactly one task per pilot')
    task,group=next(iter(cfg['tasks'].items()))
    if len(group['seeds'])!=1:raise ValueError('This analyzer plots exactly one declared seed per arm')
    if len(group['conditions'])!=len(set(group['conditions'])):raise ValueError('duplicate configured condition')
    if not set(group['conditions'])<=set(NAMES):raise ValueError('unknown configured condition')
    return cfg,task,group


def freeze(out):
    cfg,_,_=configured_trials()
    ledger_path = CAMPAIGN / 'campaign_log.json'
    ledger = json.loads(ledger_path.read_text())
    expected={f'{t}__{a}__seed{s}' for t,g in cfg['tasks'].items() for a in g['conditions'] for s in g['seeds']}
    if set(ledger)!=expected:raise ValueError('campaign ledger differs from exact configured trial set')
    entries = [];seen_runs=set()
    for task, group in cfg['tasks'].items():
        for arm in group['conditions']:
            for seed in group['seeds']:
                key = f'{task}__{arm}__seed{seed}'
                event = ledger.get(key)
                if not event or not event.get('finished_at'):
                    raise RuntimeError(f'Campaign incomplete: {key}; no final selection written')
                entry = dict(key=key, arm=arm, task=task, seed=seed, event=event)
                if event.get('run_dir'):
                    run = Path(event['run_dir'])
                    if run.resolve() in seen_runs:raise ValueError('same logged path reused for distinct trials')
                    seen_runs.add(run.resolve())
                    m=json.loads((run/'run_manifest.json').read_text())
                    if (m.get('task'),m.get('seed'))!=(task,seed):raise ValueError('ledger task/seed mismatch')
                    entry['run'] = str(run.relative_to(REPO))
                    entry['files'] = {name: digest(run/name) for name in aligned.required_artifacts(run,REQUIRED) if (run/name).exists()}
                    entry['missing'] = [name for name in aligned.required_artifacts(run,REQUIRED) if not (run/name).exists()]
                entries.append(entry)
    selection = dict(status=('three_arm_one_seed_integration_pilot_not_confirmatory' if len(entries)==3
                             else 'scoped_integration_pilot_not_confirmatory'),
                     protocol_sha256=digest(out / 'protocol.json'), config_sha256=digest(CONFIG),
                     ledger_sha256=digest(ledger_path), runs=entries)
    dest = out / 'selection.json'
    if dest.exists() and json.loads(dest.read_text()) != selection:
        raise ValueError('Frozen selection changed')
    writejson(dest, selection)
    return entries


def f(row, key):
    try: return float(row.get(key, 'nan'))
    except (TypeError, ValueError): return float('nan')


def audit_live_camera_model(run, delivered):
    """Check logged camera means/R before robust fusion against frozen deployment."""
    from reliability.reference_calibration import ReferenceCalibration
    from reliability.learned_box_correction import LearnedBoxCorrection
    mean_path = REPO/'logs/perception_models/box_feature_bias_correction_20260831/models.joblib'
    mean = LearnedBoxCorrection(mean_path)
    cal = ReferenceCalibration(OUT/'network_planner/reference_calibration.json', mean_path, mean.camera_ids)
    worst_z = worst_R = 0.; count = 0; seen = set()
    with (run/'fusion_observations.csv').open() as stream:
        for row in csv.DictReader(stream):
            # Schema-7 fusion CSV abbreviates camera_A to A; opportunity contracts
            # retain the full camera identity. This is a declared ID mapping.
            camera = {c[-1]: c for c in mean.camera_ids}.get(row['camera'], row['camera'])
            key = row['source_batch_id'], camera
            if key in seen or key not in delivered: continue
            seen.add(key); observation = delivered[key]
            raw = [f(row, 'raw_obs_x'), f(row, 'raw_obs_y')]
            if not np.isfinite(raw).all(): raise ValueError('Raw pre-NN observation not logged')
            nn = mean.correct(key[1], raw, observation['bbox_xyxy'], observation['detector_score'])
            expected_z, expected_R = cal.apply(key[1], nn)
            logged_z = [f(row, 'obs_x'), f(row, 'obs_y')]
            logged_R = [[f(row, 'obs_cov_xx'), f(row, 'obs_cov_xy')],
                        [f(row, 'obs_cov_xy'), f(row, 'obs_cov_yy')]]
            dz = float(np.max(np.abs(np.asarray(logged_z)-expected_z)))
            dR = float(np.max(np.abs(np.asarray(logged_R)-expected_R)))
            worst_z, worst_R = max(worst_z, dz), max(worst_R, dR); count += 1
            if dz > 1e-8 or dR > 1e-10: raise ValueError('Logged camera model differs from frozen residual model')
    if count == 0: raise ValueError('No logged camera model outputs to verify')
    return dict(observations_checked=count, maximum_mean_difference_m=worst_z,
                maximum_R_difference_m2=worst_R)


def sensor_diagnostics(run, start, stop, *, max_reference_gap_s=None):
    """Exploratory conditional-error audit; never fit noise from these drive rows."""
    selected = [r for r in aligned.readings(run, admitted_only=False, max_reference_gap_s=max_reference_gap_s,require_reference=False)
                if start <= r['obs_stamp'] <= stop]
    readings=[r for r in selected if r['reference_supported']]
    output = []
    for camera in sorted({r['camera'] for r in readings}):
        rows = [r for r in readings if r['camera'] == camera]
        t = np.array([r['obs_stamp'] for r in rows])
        errors = np.array([r['error'] for r in rows])
        aligned.validate_covariances(np.array([r['cov'] for r in rows]))
        normalized = np.array([np.linalg.solve(np.linalg.cholesky(r['cov']), r['error']) for r in rows])
        distance2 = np.sum(normalized**2, axis=1)
        correlations = []
        for lag in (.2, 1., 2.):
            pairs = []
            for i, stamp in enumerate(t):
                j = int(np.searchsorted(t, stamp+lag))
                candidates = [k for k in (j-1, j) if i < k < len(t)]
                if not candidates: continue
                k = min(candidates, key=lambda k: abs(t[k]-stamp-lag))
                if abs(t[k]-stamp-lag) <= .05: pairs.append((i, k))
            rho = [None, None]
            if len(pairs) >= 10:
                left, right = np.asarray(pairs).T
                for j in range(2):
                    if min(np.std(normalized[left, j]), np.std(normalized[right, j])) > 1e-9:
                        rho[j] = float(np.corrcoef(normalized[left, j], normalized[right, j])[0, 1])
            correlations.append(dict(lag_s=lag, pairs=len(pairs), whitened_coordinate_correlation=rho))
        output.append(dict(camera=camera, observations=len(rows), residual_mean_cm=(100*errors.mean(axis=0)).tolist(),
            position_median_cm=float(np.median(np.linalg.norm(errors, axis=1))*100),
            nominal_95_ellipse_coverage=float(np.mean(distance2 <= 5.991464547)),
            temporal=correlations))
    return dict(selected_readings=len(selected),reference_supported_readings=len(readings),
                unscoreable_reference_readings=len(selected)-len(readings),
                population='camera-manager outputs before fusion selection, after its admission gate; capture-time reference',
                interpretation='Within-run exploratory diagnostic. View-dependent mean and scale can cause correlation; no independence or Gaussian claim.',
                cameras=output)


def analyze_run(entry, *, max_reference_gap_s=None):
    if entry['event']['outcome'] == 'infra_invalid' or entry.get('missing'):
        return dict(arm=entry['arm'], status='infrastructure_invalid',
                    reason=entry['event']['completion_reason']), None
    run,manifest,summary=aligned.verify_frozen_entry(entry,REQUIRED,minimum_schema=7,repo=REPO)
    ledger=aligned.validate_run_ledger(run)
    if manifest['campaign_config_sha256'] != digest(CONFIG): raise ValueError('Wrong runtime config')
    if (manifest['process_noise_xy'], manifest['process_noise_theta']) != (.01, .02): raise ValueError('Q mismatch')
    if manifest['manager_covariance_profile'] != 'commissioned_reference_r': raise ValueError('Live R mismatch')
    if manifest['manager_commissioned_world_covariance_sha256'] != digest(OUT/'network_planner/reference_calibration.json'):
        raise ValueError('Live calibration bytes differ')
    if manifest['camera_network_artifact_sha256'] != digest(REPO/yaml.safe_load(CONFIG.read_text())['conditions'][entry['arm']]['camera_network_artifact_path']):
        raise ValueError('Wrong field artifact')
    table = aligned.rows(run)
    truth = aligned.truth_series(run, table, max_reference_gap_s=max_reference_gap_s)
    a = aligned.aligned_error_cm(run, 'belief', table, max_reference_gap_s=max_reference_gap_s)
    try:
        start, stop = aligned.mission_interval(run)
    except ValueError as exc:
        return dict(arm=entry['arm'],task=entry['task'],seed=entry['seed'],status=entry['event']['outcome'],
                    accuracy_status='unscoreable_mission_interval',reason=str(exc)),None
    population=aligned.landed_mask(a['stamp'])&a['have']&(a['stamp']>=start)&(a['stamp']<=stop)
    use = aligned.landed_mask(a['stamp']) & np.isfinite(a['aligned_cm']) & (a['stamp'] >= start) & (a['stamp'] <= stop)
    P = np.array([[[f(r, 'planner_cov_x'), f(r, 'planner_cov_xy')],
                   [f(r, 'planner_cov_xy'), f(r, 'planner_cov_y')]] for r in table])[use]
    errors = a['aligned_cm'][use]
    e = np.column_stack([a['x'][use]-a['gt_x'][use], a['y'][use]-a['gt_y'][use]])
    nees = aligned.nees(e,P)
    ass = aligned.assimilations(run)
    events = [r for r in ass if start <= r['apply_stamp'] <= stop]
    accounting=aligned.correction_accounting(run)
    if not len(errors):
        return dict(arm=entry['arm'],task=entry['task'],seed=entry['seed'],status=entry['event']['outcome'],
            accuracy_status='no_reference_supported_beliefs',belief_samples=0,
            belief_reference_unscoreable=int(population.sum()),reference_max_gap_s=max_reference_gap_s,
            correction_accounting=accounting),None
    ty = truth.yaw_at(a['stamp'][use])
    ey = np.array([f(r, 'planner_belief_yaw') for r in table])[use]-ty
    yaw_error_deg = np.rad2deg(np.abs(np.arctan2(np.sin(ey), np.cos(ey))))
    P_pose = np.zeros((len(errors), 3, 3))
    P_pose[:, :2, :2] = P
    P_pose[:, 0, 2] = P_pose[:, 2, 0] = np.array([f(r, 'planner_belief_cov_x_theta') for r in table])[use]
    P_pose[:, 1, 2] = P_pose[:, 2, 1] = np.array([f(r, 'planner_belief_cov_y_theta') for r in table])[use]
    P_pose[:, 2, 2] = np.array([f(r, 'planner_belief_cov_theta_theta') for r in table])[use]
    error_pose = np.column_stack([e, np.arctan2(np.sin(ey), np.cos(ey))])
    pose_valid = np.isfinite(P_pose).all(axis=(1, 2)) & np.isfinite(error_pose).all(axis=1)
    aligned.validate_covariances(P_pose[np.isfinite(error_pose).all(axis=1)],dimension=3)
    pose_nees = np.array([z @ np.linalg.solve(C, z) for C, z in zip(P_pose[pose_valid], error_pose[pose_valid])])
    delivery,duplicate_deliveries=aligned.camera_opportunities(run)
    opportunities=[o for o in delivery if start<=o['timestamp_s']<=stop]
    by_batch = {}
    for r in opportunities:
        k = (r['source_batch_id'], r['camera_id'])
        if k in by_batch: raise ValueError('Duplicate delivered observation')
        by_batch[k] = r
    camera_audit = audit_live_camera_model(run, by_batch)
    latencies = [r['frame_age_at_publish_s'] for r in opportunities
                 if r.get('frame_age_at_publish_s') is not None and np.isfinite(r['frame_age_at_publish_s'])]
    inference = [r['yolo_inference_wall_ms'] for r in opportunities
                 if r.get('yolo_inference_wall_ms') is not None and np.isfinite(r['yolo_inference_wall_ms'])]
    global_meta = json.loads((run/'global_plan_meta.json').read_text())
    record = dict(arm=entry['arm'], status=entry['event']['outcome'],
        selected_source=global_meta['selected_source'], camera_model_audit=camera_audit,
        task=entry['task'],seed=entry['seed'],
        sensor_diagnostics=sensor_diagnostics(run, start, stop,max_reference_gap_s=max_reference_gap_s),
        reference_max_gap_s=max_reference_gap_s,belief_reference_unscoreable=int(population.sum())-len(errors),
        stop_reason=summary['completion_reason'], belief_samples=int(len(errors)),
        duration_sim_s=stop-start, path_length_m=summary.get('path_length_m'),
        offline_final_goal_distance_m=summary.get('final_goal_distance'),
        offline_final_goal_distance_reference=summary.get('final_goal_distance_reference','unspecified'),
        belief_position_median_cm=float(np.median(errors)), belief_position_p95_cm=float(np.quantile(errors, .95)),
        belief_position_rmse_cm=float(np.sqrt(np.mean(errors**2))),
        planar_95_ellipse_coverage=float(np.mean(nees <= 5.991464547)), planar_nees_median=float(np.median(nees)),
        belief_heading_samples=int(np.isfinite(yaw_error_deg).sum()),
        belief_heading_median_deg=float(np.nanmedian(yaw_error_deg)) if np.isfinite(yaw_error_deg).any() else None,
        belief_heading_p95_deg=float(np.nanquantile(yaw_error_deg,.95)) if np.isfinite(yaw_error_deg).any() else None,
        belief_heading_final_deg=float(yaw_error_deg[-1]) if np.isfinite(yaw_error_deg[-1]) else None,
        pose_nees_samples=int(len(pose_nees)),
        pose_95_ellipsoid_coverage=float(np.mean(pose_nees <= 7.814727903)) if len(pose_nees) else None,
        pose_nees_median=float(np.median(pose_nees)) if len(pose_nees) else None,
        assimilations=len(events), assimilation_status_counts=dict(Counter(r['status'] for r in events)),
        refusal_reasons=dict(Counter(r['reason'] for r in events if not r['accepted'])),
        correction_dropped_fraction=accounting['correction_dropped_fraction'],
        longest_correction_gap_s=accounting['longest_correction_gap_s'], opportunities=len(opportunities),
        duplicate_deliveries=duplicate_deliveries,
        detections=sum(bool(r['detection_valid']) for r in opportunities),
        frame_age_publish_median_sim_s=float(np.median(latencies)) if latencies else None,
        inference_median_wall_ms=float(np.median(inference)) if inference else None,
        reference=truth.source, sample_weighting='first occurrence of each planner belief stamp within first-command to stop',
        scope='one closed-loop run, not a replicate-based effect estimate')
    gt_mask = (truth.t >= start) & (truth.t <= stop)
    plot = dict(t=a['stamp'][use]-start, e=errors, std_trace=100*np.sqrt(np.trace(P, axis1=1, axis2=2)),
                x=a['x'][use], y=a['y'][use], gx=truth.x[gt_mask], gy=truth.y[gt_mask])
    return record, plot


def navigation(out, *, max_reference_gap_s=None):
    protocol_data=json.loads((out/'protocol.json').read_text())
    if protocol_data.get('reference_policy')!={'max_reference_gap_s':max_reference_gap_s}:
        raise ValueError('protocol reference policy differs or is absent; preserve old output and use a new protocol directory')
    entries = freeze(out)
    analysis=dict(selection_sha256=digest(out/'selection.json'),max_reference_gap_s=max_reference_gap_s,
        sources={str(p):digest(p) for p in [Path(__file__),Path(aligned.__file__),
            REPO/'src/unav_common/unav_common/correction_ledger.py']})
    aligned.freeze_analysis_protocol(out,'analysis_protocol.json',analysis,
        owned_outputs=['results.json','results.csv','navigation_pilot.png'])
    results, curves = [], {}
    for entry in entries:
        result, curve = analyze_run(entry,max_reference_gap_s=max_reference_gap_s)
        results.append(result); curves[entry['arm']] = curve
    fig, axes = plt.subplots(2, len(ARMS), figsize=(max(5.4,3.83*len(ARMS)), 7.),
                             squeeze=False, layout='constrained')
    background = json.loads((PROBE / 'P0_result.json').read_text())['settings']
    all_scales = np.concatenate([np.empty(0),*[np.r_[c['e'], c['std_trace']] for c in curves.values() if c is not None]])
    positive = all_scales[np.isfinite(all_scales) & (all_scales > 0)]
    scale_limits = (max(.01, positive.min()*.8), positive.max()*1.25) if positive.size else (.1, 10.)
    for arm, ax, err_ax in zip(ARMS, axes[0], axes[1]):
        map_background(ax, background)
        record = next(r for r in results if r['arm'] == arm)
        entry = next(r for r in entries if r['arm'] == arm)
        curve = curves[arm]
        ax.set_title(f"{NAMES[arm]}: {record['status'].replace('_', ' ')}")
        if curve is None:
            ax.text(.5, .5, record.get('accuracy_status','infrastructure_invalid').replace('_',' ')+'\nNo accuracy comparison', transform=ax.transAxes, ha='center')
            err_ax.set_axis_off(); continue
        with (REPO/entry['run']/'global_plan.csv').open() as stream:
            planned = np.array([[f(r, 'x'), f(r, 'y')] for r in csv.DictReader(stream)])
        ax.plot(planned[:, 0], planned[:, 1], ':', color='#88949e', lw=1., label='Planned route')
        # Goal comes from the selected run's route metadata when explicitly logged.
        goal=json.loads((REPO/entry['run']/'global_plan_meta.json').read_text()).get('goal')
        if isinstance(goal,list) and len(goal)>=2:ax.plot(*goal[:2],'k*',ms=7)
        ax.plot(curve['gx'], curve['gy'], color='black', lw=1.4, label='Simulator reference')
        ax.plot(curve['x'], curve['y'], color=COLORS[arm], lw=1., label='Online belief')
        err_ax.plot(curve['t'], curve['e'], color=COLORS[arm], lw=1., label='Position error')
        err_ax.plot(curve['t'], curve['std_trace'], color='#37424d', ls='--', lw=1., label=r'$\sqrt{\mathrm{tr}(P_{xy})}$')
        err_ax.set(xlabel='Time since first command [s, simulation]', ylabel='Position scale [cm, log]',
                   yscale='log', ylim=scale_limits)
        err_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
        err_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        err_ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].legend(fontsize=7, frameon=False)
    title = yaml.safe_load(CONFIG.read_text()).get('study_title',
        'Matched navigation pilot: one run per field, identical camera model and estimator')
    if len(ARMS) == 1:
        import textwrap
        title = '\n'.join(textwrap.fill(line, width=57) for line in title.splitlines())
    fig.suptitle(title, fontsize=10 if len(ARMS) == 1 else 12)
    savefig(fig, out/'navigation_pilot')
    writejson(out / 'results.json', dict(selection_sha256=digest(out/'selection.json'), results=results,
        analysis_source_sha256=digest(Path(__file__)),
        interpretation='Descriptive integration pilot; no statistical significance or calibrated future-quality claim'))
    columns = sorted(set().union(*(r.keys() for r in results)))
    with (out/'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(results)


def protocol(out, *, max_reference_gap_s=None):
    out.mkdir(parents=True, exist_ok=True)
    path = out/'protocol.json'
    if path.exists(): raise ValueError('Protocol already frozen')
    cfg,task,group=configured_trials()
    files = [CONFIG, REPO/'src/reliability/reliability/reference_calibration.py',
             REPO/'src/reliability/reliability/nodes/camera_manager_node.py',
             REPO/'src/reliability/reliability/fusion.py',
             REPO/'src/planning/planning/nodes/unicycle_planner_node.py',
             REPO/'src/planning/planning/nodes/efe_agent_node.py',
             REPO/'src/planning/planning/core/belief_correction.py',
             REPO/'src/planning/planning/core/dynamics.py',
             REPO/'src/planning/planning/core/motion_history.py',
             REPO/'src/planning/planning/core/tracker_guard.py',
             REPO/'src/sim/sim/actuation_noise_node.py',
             REPO/'scripts/visibility_comparison/run_visibility_campaign.py',
             OUT/'network_planner/reference_calibration.json', Path(__file__)]
    writejson(path, dict(kind='matched_integration_pilot', task=task, seeds=group['seeds'], arms=group['conditions'],
        reference_policy=dict(max_reference_gap_s=max_reference_gap_s),
        comparison=cfg.get('study_comparison',
            'Change only the future detector-score field; keep mean, camera R, fusion, Q and controller fixed.'),
        primary_outputs=['termination including failures', 'route choice', 'time and travel cost',
                         'aligned belief median and p95 position error', 'planar coverage', 'correction gaps'],
        scope='One seed per arm is descriptive; static calibration and robust runtime fusion do not establish forecast calibration.',
        sources={str(p.relative_to(REPO)): digest(p) for p in files}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['protocol', 'preflight', 'navigation'])
    parser.add_argument('--out', type=Path, default=OUT/'network_navigation_evidence')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--max-reference-gap-s',type=float)
    args = parser.parse_args(); style()
    CONFIG, CAMPAIGN = args.config.resolve(), args.campaign.resolve()
    if args.mode != 'preflight':
        cfg=yaml.safe_load(CONFIG.read_text())
        if len(cfg['tasks']) != 1:raise ValueError('This analyzer requires exactly one task per pilot')
        ARMS=tuple(next(iter(cfg['tasks'].values()))['conditions'])
    if args.mode=='preflight':preflight(args.out.resolve())
    else:globals()[args.mode](args.out.resolve(),max_reference_gap_s=args.max_reference_gap_s)
