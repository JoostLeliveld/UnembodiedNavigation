#!/usr/bin/env python3
"""A frozen single-event diagnosis, not a counterfactual closed-loop result."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

import network_navigation_analysis as nav
from study import digest, writejson


def diagnose(selection_path, capture_stamp, out):
    entry = json.loads(selection_path.read_text())
    run = nav.REPO/entry['run']
    for name, sha in entry['files'].items():
        if digest(run/name) != sha:
            raise ValueError(f'Frozen input changed: {name}')
    table = nav.aligned.rows(run)
    truth = nav.aligned.truth_series(run, table)
    belief = nav.aligned.aligned_error_cm(run, 'belief', table)
    assimilations = nav.aligned.assimilations(run)
    event = next(r for r in assimilations if abs(r['correction_stamp']-capture_stamp) < 1e-6)
    if event['reason'] != 'replay_gap_too_large':
        raise ValueError('Requested event is not a recorded replay-gap refusal')
    before = [r for r in assimilations if r['apply_stamp'] < event['apply_stamp']]
    anchor = before[-1]
    # A refusal before innovation formation has no NIS; retain it as missing.
    event = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in event.items()}
    anchor = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in anchor.items()}
    prior_row = next(r for r in table if abs(nav.f(r, 'pixel_corr_apply_stamp')-anchor['apply_stamp']) < 1e-6)
    prior = np.array([nav.f(prior_row, f'pixel_corr_next_{k}') for k in ('x', 'y', 'yaw')])
    if not np.isfinite(prior).all():
        raise ValueError('No logged committed anchor for the requested event')
    from_s = anchor['correction_stamp']
    # Logger samples the measured odometry stream at 10 Hz. The live node has
    # denser input history, so this is a mechanism check, not exact arrival replay.
    inputs = []
    for row in table:
        item = tuple(nav.f(row, k) for k in ('odom_noisy_stamp', 'odom_noisy_v', 'odom_noisy_w'))
        if np.isfinite(item).all() and (not inputs or item[0] > inputs[-1][0]):
            inputs.append(item)
    from planning.core.dynamics import unicycle_step
    from planning.core.motion_history import covers_interval

    def replay(start):
        state = prior.copy()
        preceding = [r for r in inputs if r[0] <= start]
        if not preceding:
            raise ValueError('Missing input at interval start')
        u = np.asarray(preceding[-1][1:])
        last = start
        for t, v, w in inputs:
            if start < t <= capture_stamp:
                state = unicycle_step(state, u, t-last)
                u = np.array([v, w]); last = t
        return unicycle_step(state, u, capture_stamp-last)

    complete = replay(from_s)
    capped = replay(capture_stamp-1.5)
    gx, gy = truth.at(capture_stamp)
    gyaw = truth.yaw_at(capture_stamp)
    reference = np.array([gx, gy, gyaw])

    def score(state):
        return dict(mean=state.tolist(), position_error_cm=float(100*np.linalg.norm(state[:2]-reference[:2])),
                    heading_error_deg=float(abs(np.rad2deg(np.arctan2(np.sin(state[2]-gyaw), np.cos(state[2]-gyaw))))))

    readings = [r for r in nav.aligned.readings(run, admitted_only=False)
                if abs(r['obs_stamp']-capture_stamp) < 1e-6]
    output = dict(kind='single_event_10Hz_logged_motion_diagnostic_not_closed_loop_replay',
        run=entry['run'], camera_return_event=event, previous_committed_event=anchor,
        previous_committed_mean=prior.tolist(), correction_interval_s=capture_stamp-from_s,
        motion_history_covers_interval=covers_interval(inputs, from_s, capture_stamp, 1.5),
        reference_at_capture=reference.tolist(), complete_motion=score(complete), legacy_capped_motion=score(capped),
        returning_camera_readings=[dict(camera=r['camera'], error_cm=float(100*np.linalg.norm(r['error']))) for r in readings],
        scope='Compares predictions at one observed capture time from the same committed state. '
              'It does not predict what the robot would have done under different feedback.',
        sources={str(selection_path.relative_to(nav.REPO)): digest(selection_path),
                 str(Path(__file__).relative_to(nav.REPO)): digest(Path(__file__))})
    out.mkdir(parents=True, exist_ok=True)
    writejson(out/'outage_diagnostic.json', output)
    nav.style()
    fig, axes = nav.plt.subplots(2, 1, figsize=(9., 5.2), sharex=True, layout='constrained')
    start, stop = from_s-3., capture_stamp+7.
    selected = nav.aligned.landed_mask(belief['stamp']) & (belief['stamp'] >= start) & (belief['stamp'] <= stop)
    t = belief['stamp'][selected]
    yaw = np.array([nav.f(r, 'planner_belief_yaw') for r in table])[selected]
    error_yaw = np.rad2deg(np.arctan2(np.sin(yaw-truth.yaw_at(t)), np.cos(yaw-truth.yaw_at(t))))
    axes[0].plot(t-from_s, belief['aligned_cm'][selected], color='#207b70', label='Online belief position error')
    for camera in ('A', 'B', 'C', 'D', 'E'):
        rr = [r for r in nav.aligned.readings(run, admitted_only=False)
              if r['camera'] == camera and start <= r['obs_stamp'] <= stop]
        if rr:
            axes[0].scatter([r['obs_stamp']-from_s for r in rr],
                            [100*np.linalg.norm(r['error']) for r in rr], s=12,
                            label=f'Camera {camera} reading error')
    axes[0].set(ylabel='Position error [cm, log]', yscale='log')
    axes[0].yaxis.set_major_formatter(nav.FuncFormatter(lambda value, _: f'{value:g}'))
    axes[1].plot(t-from_s, error_yaw, color='#207b70', label='Online belief heading error')
    odom_t = np.array([nav.f(r, 'odom_noisy_stamp') for r in table])
    odom_mask = nav.aligned.landed_mask(odom_t) & (odom_t >= start) & (odom_t <= stop)
    odom_yaw = np.array([nav.f(r, 'odom_noisy_yaw') for r in table])[odom_mask]
    odom_e = odom_yaw-truth.yaw_at(odom_t[odom_mask])
    axes[1].plot(odom_t[odom_mask]-from_s, np.rad2deg(np.arctan2(np.sin(odom_e), np.cos(odom_e))),
                 color='#667788', ls='--', label='Measured odometry heading error')
    axes[1].set(xlabel='Time since previous committed correction [s, simulation]', ylabel='Wrapped heading error [deg]')
    for ax in axes:
        ax.axvspan(0, capture_stamp-from_s, color='#eef1f2', zorder=0)
        ax.axvline(capture_stamp-from_s, color='#a85b35', ls=':', label='Camera returns / gap refusal')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('A camera gap is not an odometry gap: recorded P0 failure', fontsize=12)
    nav.savefig(fig, out/'outage_diagnostic')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--capture-stamp', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    diagnose(args.selection.resolve(), args.capture_stamp, args.out.resolve())
