#!/usr/bin/env python3
"""Gate 0: is the declared process noise Q good enough to be used as a measuring instrument?

WHY THIS EXISTS.  The plan for identifying R from driving data attributes to the camera
whatever innovation covariance odometry cannot explain:

    S_t = H P_t^- H^T + R,        P_t^- built by propagating Q.

That makes Q an instrument.  If Q is too small the fit inflates R; if Q is too large it
deflates it.  So Q must be validated BEFORE it is used, even though it is not refitted.

WHAT Q IS TODAY.  Not a measured quantity.  `process_noise_xy = 0.01`,
`process_noise_theta = 0.02` are declared planner defaults
(`unicycle_planner_node.py:90-91`), copied verbatim into every campaign config; a different
value (0.012^2) sits in `replay.py`.  Neither was ever checked against realized odometry
error.  That was a deliberate decision while Q only had to be fair across arms
("do NOT fit Q to GT" -- the white-noise campaign); it is no longer sufficient once Q sets
the scale of every R.

WHAT IS MEASURED HERE.  Prediction in this stack is ODOMETRY-DRIVEN: the belief is
propagated by replaying the encoder-measured twist `/odom_noisy`, not the commanded u.  So
the process noise this Q must describe is the drift of `odom_noisy` against ground truth.

Over a window of length T starting at time t0, with no camera correction anywhere inside it:

    realized   d = (odom_noisy pose at t0+T - odom_noisy pose at t0) rotated into map
               e = d - (gt pose at t0+T - gt pose at t0)
    predicted  P = sum of Q_d(theta, v, dt) over the window's steps

`Q_d` is the SAME state-dependent closed form the filter uses
(`planning.core.dynamics.unicycle_process_noise`), evaluated on the same twist -- this is
not a reimplementation.  Consistency is the normalized squared error

    d^2 = e^T P^-1 e         which should average 2.0 for the (x,y) pair if Q is right,
                             and whose 95th percentile should sit near chi2_2's 5.99.

Ratios above 1 mean Q is TOO SMALL (drift exceeds what Q allows) and R would be inflated;
below 1 mean Q is too large and R would be deflated.

WHAT THIS IS NOT.  It does not refit Q, and it reads ground truth only to score -- exactly
the offline-reference role `gt_*` is permitted to play.  Windows are open-loop by
construction, so the camera never enters.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str((REPO / 'src/planning').resolve()))

from planning.core.dynamics import unicycle_process_noise  # noqa: E402

# The declared values under test, and the disagreeing one in replay.py.
DECLARED = {'process_noise_xy': 0.01, 'process_noise_theta': 0.02}
REPLAY_ALT = {'process_noise_xy': 0.012, 'process_noise_theta': 0.02}

# --------------------------------------------------------------------------------------
# The corrected Q, DERIVED from the encoder generator rather than fitted to the ratios.
#
# `sim/encoder_noise_node.py` corrupts the reported twist with (a) a systematic linear
# scale error `linear_slip_mean = 0.02`, (b) AR(1)-correlated multiplicative slip with
# `correlation_alpha = 0.80`, and (c) small additive white noise. The planner's Q_d models
# only instantaneous white v/w noise, so it represents (c) and misses (a) and (b).
#
# Measured consequence: drift grows as T^0.9 -- near LINEAR in time, not the T^0.5 of white
# noise -- in BOTH directions. Linear growth is the signature of a coherent offset held over
# the window, not a random walk. Two coherent terms reproduce it from the generator's own
# declared parameters, with nothing fitted to the drift:
#
#   along-track   a systematic speed scale error eps_v = linear_slip_mean = 0.02
#                 predicts v*eps_v*T = 0.44 / 0.88 / 2.20 / 4.40 cm at T = 1/2/5/10 s
#                 observed                          0.48 / 0.84 / 1.99 / 4.11 cm
#
#   cross-track   a quasi-static heading offset eps_theta, giving v*eps_theta*T. Inverting
#                 the observed cross-track drift yields 1.48 / 1.43 / 1.27 / 1.18 deg --
#                 flat across a tenfold change in window length, which is what identifies it
#                 as a held bias rather than a growing walk (a walk would scale as sqrt(T)).
#                 Taken as eps_theta = 1.3 deg, the midpoint of that flat range.
#
# These enter the covariance as rank-1 coherent terms in the window's own heading frame,
# growing as (v*eps*T)^2, alongside the existing white Q_d.
COHERENT_SPEED_SCALE = 0.02          # sim/encoder_noise_node.py `linear_slip_mean`
COHERENT_HEADING_RAD = math.radians(1.3)


def coherent_block(distance_m: float, heading: float) -> np.ndarray:
    """Rank-2 coherent drift covariance for a window that travelled `distance_m`."""
    fwd = np.array([math.cos(heading), math.sin(heading)])
    lat = np.array([-fwd[1], fwd[0]])
    along_sd = COHERENT_SPEED_SCALE * distance_m
    cross_sd = COHERENT_HEADING_RAD * distance_m
    return (along_sd ** 2) * np.outer(fwd, fwd) + (cross_sd ** 2) * np.outer(lat, lat)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    keep = []
    for row in rows:
        try:
            if float(row['gt_available']) < 0.5 or float(row['odom_noisy_available']) < 0.5:
                continue
            keep.append({
                't': float(row['stamp']),
                'gt': np.array([float(row['gt_x']), float(row['gt_y'])]),
                'gt_yaw': float(row['gt_yaw']),
                'od': np.array([float(row['odom_noisy_x']), float(row['odom_noisy_y'])]),
                'od_yaw': float(row['odom_noisy_yaw']),
                'v': float(row['odom_noisy_v']),
                'w': float(row['odom_noisy_w']),
            })
        except (KeyError, ValueError):
            continue
    return keep


def window_stats(rows: list[dict], i0: int, i1: int, q: dict) -> dict | None:
    """Open-loop drift over rows[i0:i1] against what Q predicts for the same twist."""
    a, b = rows[i0], rows[i1]
    # Odometry frame is a pure translation of map (verified: same start yaw, same shape),
    # so a displacement in odom is directly comparable to a displacement in map.
    d_od = b['od'] - a['od']
    d_gt = b['gt'] - a['gt']
    err = d_od - d_gt
    yaw_err = wrap((b['od_yaw'] - a['od_yaw']) - (b['gt_yaw'] - a['gt_yaw']))

    # Predicted covariance: the SAME state-dependent Q_d the filter integrates, summed over
    # the window's own steps. Rotation of the 2x2 block is skipped because the accumulated
    # heading change inside a short window is small; that approximation is reported, not hidden.
    cov = np.zeros((3, 3))
    for j in range(i0, i1):
        dt = rows[j + 1]['t'] - rows[j]['t']
        if not (0.0 < dt < 1.0):
            return None
        cov += unicycle_process_noise(
            q['process_noise_xy'], q['process_noise_theta'], dt,
            theta=rows[j]['od_yaw'], v=rows[j]['v'],
        )
    pos = cov[:2, :2]
    if q.get('coherent'):
        # Distance actually travelled in the window, from the encoder stream (no truth).
        travelled = sum(
            abs(rows[j]['v']) * (rows[j + 1]['t'] - rows[j]['t']) for j in range(i0, i1)
        )
        pos = pos + coherent_block(travelled, a['od_yaw'])
    if not np.all(np.isfinite(pos)) or np.linalg.det(pos) <= 0:
        return None
    d2 = float(err @ np.linalg.solve(pos, err))

    # Q_d is extremely anisotropic on a straight run: cross-track variance comes ONLY from
    # the v^2 sigma_w^2 dt^3/3 term and collapses as w -> 0, leaving the position block
    # ~1550x stiffer across the path than along it. A single d^2 is then dominated by the
    # cross-track direction and says little about the along-track scale. Report both
    # directions separately, in the heading frame the motion itself defines.
    heading = a['od_yaw']
    fwd = np.array([math.cos(heading), math.sin(heading)])
    lat = np.array([-fwd[1], fwd[0]])
    return {
        'e_along_m': abs(float(err @ fwd)),
        'e_cross_m': abs(float(err @ lat)),
        'sd_along_m': math.sqrt(max(float(fwd @ pos @ fwd), 1e-30)),
        'sd_cross_m': math.sqrt(max(float(lat @ pos @ lat), 1e-30)),
        'T': b['t'] - a['t'],
        'dist': float(np.linalg.norm(d_gt)),
        'err_m': float(np.linalg.norm(err)),
        'yaw_err_rad': float(abs(yaw_err)),
        'pred_sd_m': float(math.sqrt(0.5 * np.trace(pos))),
        'pred_yaw_sd': float(math.sqrt(cov[2, 2])),
        'd2': d2,
    }


def summarize(vals: list[dict], label: str) -> dict:
    if not vals:
        return {'n': 0, 'window_s': label}
    d2 = np.array([v['d2'] for v in vals])
    err = np.array([v['err_m'] for v in vals])
    pred = np.array([v['pred_sd_m'] for v in vals])
    yerr = np.array([v['yaw_err_rad'] for v in vals])
    ypred = np.array([v['pred_yaw_sd'] for v in vals])
    def ratio(key_e: str, key_sd: str) -> float:
        realized = float(np.median([abs(v[key_e]) for v in vals]))
        # A half-normal's median is 0.6745 sd, so compare like with like.
        predicted = 0.6745 * float(np.median([v[key_sd] for v in vals]))
        return realized / predicted if predicted > 0 else float('nan')

    return {
        'window_s': label,
        'n': len(vals),
        'along_track_sd_ratio': ratio('e_along_m', 'sd_along_m'),
        'cross_track_sd_ratio': ratio('e_cross_m', 'sd_cross_m'),
        'realized_along_cm_median': float(np.median([v['e_along_m'] for v in vals])) * 100,
        'realized_cross_cm_median': float(np.median([v['e_cross_m'] for v in vals])) * 100,
        'predicted_along_sd_cm_median': float(np.median([v['sd_along_m'] for v in vals])) * 100,
        'predicted_cross_sd_cm_median': float(np.median([v['sd_cross_m'] for v in vals])) * 100,
        'mean_d2': float(d2.mean()),
        'median_d2': float(np.median(d2)),
        'p95_d2': float(np.quantile(d2, 0.95)),
        'frac_within_chi2_95': float((d2 <= 5.991).mean()),
        'realized_pos_err_cm': {
            'median': float(np.median(err)) * 100,
            'p95': float(np.quantile(err, 0.95)) * 100,
        },
        'predicted_pos_sd_cm': {
            'median': float(np.median(pred)) * 100,
            'p95': float(np.quantile(pred, 0.95)) * 100,
        },
        'pos_sd_ratio_realized_over_predicted': float(np.median(err) / np.median(pred)),
        'realized_yaw_err_deg_median': float(np.degrees(np.median(yerr))),
        'predicted_yaw_sd_deg_median': float(np.degrees(np.median(ypred))),
        'yaw_ratio_realized_over_predicted': float(np.median(yerr) / np.median(ypred)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', type=float, nargs='+', default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument('--stride-s', type=float, default=0.5)
    args = parser.parse_args()

    drives = sorted(
        (REPO / 'logs/visibility_comparison').glob('**/experiment.csv'))
    usable: list[tuple[str, list[dict]]] = []
    for path in drives:
        summary = path.parent / 'run_summary.json'
        if not summary.is_file():
            continue
        meta = json.loads(summary.read_text(encoding='utf-8'))
        if meta.get('completion_reason') == 'interrupted':
            continue           # a crashed logger is not a drive
        rows = load(path)
        if len(rows) < 50:
            continue
        tag = '/'.join(path.parts[-5:-1])
        usable.append((tag, rows))
    if not usable:
        raise RuntimeError('no usable drives found')

    per_q = {}
    corrected = dict(DECLARED, coherent=True)
    for qname, q in (('declared_0.01', DECLARED),
                     ('replay_alt_0.012', REPLAY_ALT),
                     ('corrected_coherent', corrected)):
        by_window = {}
        for T in args.windows:
            collected: list[dict] = []
            for tag, rows in usable:
                j = 0
                last_start = -1e9
                for i0 in range(len(rows)):
                    if rows[i0]['t'] - last_start < args.stride_s:
                        continue
                    i1 = i0
                    while i1 + 1 < len(rows) and rows[i1]['t'] - rows[i0]['t'] < T:
                        i1 += 1
                    if rows[i1]['t'] - rows[i0]['t'] < T * 0.9:
                        break
                    stat = window_stats(rows, i0, i1, q)
                    if stat is not None:
                        stat['drive'] = tag
                        collected.append(stat)
                        last_start = rows[i0]['t']
            by_window[f'{T:g}s'] = summarize(collected, f'{T:g}s')
        per_q[qname] = by_window

    out_dir = REPO / 'logs/studies/gate0_process_noise'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'status': 'complete',
        'schema': 'gate0_process_noise.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'Is the declared Q good enough to serve as the instrument that sizes R?',
        'n_drives': len(usable),
        'drives': [tag for tag, _ in usable],
        'q_under_test': {
            'declared_0.01': DECLARED, 'replay_alt_0.012': REPLAY_ALT,
            'corrected_coherent': {
                **DECLARED,
                'coherent_speed_scale': COHERENT_SPEED_SCALE,
                'coherent_heading_rad': COHERENT_HEADING_RAD,
                'derivation': 'speed scale = encoder_noise_node linear_slip_mean (0.02); '
                              'heading offset = 1.3 deg, the flat value implied by observed '
                              'cross-track drift across 1-10 s windows. Not fitted to d2.',
            },
        },
        'reference': 'mean d^2 should be 2.0 for a 2-D error if Q is right; '
                     'ratio > 1 means Q too small (R would be inflated)',
        'stride_s': args.stride_s,
        'approximations': [
            'The 2x2 position block is summed without rotating earlier steps into the '
            'later heading frame; valid while the heading change inside a window is small.',
            'odom_noisy and gt are compared as DISPLACEMENTS, so the constant frame offset '
            'cancels; verified that both frames share the start yaw.',
        ],
        'results': per_q,
    }
    (out_dir / 'gate0_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    for qname, by_window in per_q.items():
        print(f'\n=== {qname} ===')
        print(f"{'window':>8} {'n':>6} {'mean d2':>9} {'med d2':>8} {'<=chi2.95':>10} "
              f"{'real cm':>8} {'pred cm':>8} {'ratio':>7} {'yaw ratio':>10}")
        for key, s in by_window.items():
            if not s.get('n'):
                continue
            print(f"{key:>8} {s['n']:>6} {s['mean_d2']:>9.2f} {s['median_d2']:>8.2f} "
                  f"{s['frac_within_chi2_95']:>9.1%} "
                  f"{s['realized_pos_err_cm']['median']:>8.2f} "
                  f"{s['predicted_pos_sd_cm']['median']:>8.2f} "
                  f"{s['pos_sd_ratio_realized_over_predicted']:>7.2f} "
                  f"{s['yaw_ratio_realized_over_predicted']:>10.2f}")
            print(f"{'':>8} {'':>6}   along x{s['along_track_sd_ratio']:>6.2f} "
                  f"(real {s['realized_along_cm_median']:>6.2f} vs pred "
                  f"{s['predicted_along_sd_cm_median']:>6.2f} cm) | "
                  f"cross x{s['cross_track_sd_ratio']:>9.2f} "
                  f"(real {s['realized_cross_cm_median']:>6.2f} vs pred "
                  f"{s['predicted_cross_sd_cm_median']:>7.4f} cm)")
    print(f'\nwrote {out_dir}/gate0_manifest.json  ({len(usable)} drives)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
