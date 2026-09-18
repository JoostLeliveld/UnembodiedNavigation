#!/usr/bin/env python3
"""Gate 4/5 reporting from a score_anchored_route_contrast.py artifact.

Reports, per task x arm:
  - the route each accounting selects (LEGACY absolute vs ANCHORED)
  - whether the choice flips
  - for every flip, the BREAK-EVEN w_amb: the ambiguity weight at which the
    two routes cost the same, i.e. where the extra travel stops paying
  - per-step ambiguity summaries (gate 5), not only totals

Break-even. Each candidate's total is

    J(c) = other(c) + w_amb * amb(c)

where ``other`` is risk + obstacle + control, all evaluated at the SAME rollout.
The recorded totals used the deployed weight w0, so other(c) = J(c) - w0*amb(c).
Two candidates a, b cost the same when

    w* = -(other(a) - other(b)) / (amb(a) - amb(b))

A w* that is negative, infinite, or below the deployed weight means the flip is
not weight-driven at all. A w* close to the deployed value is a THIN margin:
report it as a finding, not a success.
"""
import argparse
import json
import math
from pathlib import Path


def break_even(a, b, w0, amb_key, total_key):
    """w_amb at which candidates a and b tie. None when they never tie."""
    d_amb = a[amb_key] - b[amb_key]
    other_a = a[total_key] - w0 * a[amb_key]
    other_b = b[total_key] - w0 * b[amb_key]
    d_other = other_a - other_b
    if abs(d_amb) < 1e-12:
        return None  # parallel in w_amb: no weight makes them tie
    return -d_other / d_amb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--w-amb', type=float, required=True,
                        help='deployed ambiguity weight the totals were scored at')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    records = json.loads(Path(args.artifact).read_text())
    w0 = args.w_amb
    report = []

    for record in records:
        feasible = [c for c in record['candidates'] if c['feasible']]
        if not feasible:
            report.append({'task': record['task'], 'arm': record['arm'],
                           'status': 'no feasible candidate'})
            continue

        # ANCHORED totals are what the planner now returns.
        anchored = min(feasible, key=lambda c: (c['total_cost_anchored'], c['name']))
        # LEGACY totals: swap the anchored ambiguity for the absolute one on the
        # same rollout. other(c) is identical, so only the ambiguity term moves.
        def legacy_total(c):
            other = c['total_cost_anchored'] - w0 * c['ambiguity_cost_anchored']
            return other + w0 * c['per_step_legacy_sum']
        legacy = min(feasible, key=lambda c: (legacy_total(c), c['name']))

        entry = {
            'task': record['task'], 'arm': record['arm'],
            'feasible_count': len(feasible),
            'selected_legacy': legacy['name'],
            'selected_anchored': anchored['name'],
            'flipped': legacy['name'] != anchored['name'],
            'per_step': [{
                'name': c['name'],
                'length_m': c['length_m'],
                'steps_scored': len(c['per_step']),
                'anchored_sum': c['per_step_anchored_sum'],
                'anchored_min': c['per_step_anchored_min'],
                'anchored_mean': (c['per_step_anchored_sum'] / len(c['per_step'])
                                  if c['per_step'] else math.nan),
                'legacy_sum': c['per_step_legacy_sum'],
                'legacy_mean': (c['per_step_legacy_sum'] / len(c['per_step'])
                                if c['per_step'] else math.nan),
                'total_cost_anchored': c['total_cost_anchored'],
            } for c in sorted(feasible, key=lambda c: c['name'])],
        }
        if entry['flipped']:
            w_star = break_even(
                anchored, legacy, w0,
                'ambiguity_cost_anchored', 'total_cost_anchored')
            entry['break_even_w_amb'] = w_star
            entry['deployed_w_amb'] = w0
            if w_star is None:
                entry['margin'] = 'no tie: the two routes never cross in w_amb'
            else:
                ratio = w_star / w0 if w0 else math.inf
                entry['break_even_over_deployed'] = ratio
                entry['margin'] = (
                    'THIN: break-even is within 2x of the deployed weight'
                    if 0.5 <= ratio <= 2.0 else 'wide')
            length_delta = None
            if anchored['length_m'] is not None and legacy['length_m'] is not None:
                length_delta = anchored['length_m'] - legacy['length_m']
            entry['extra_travel_m'] = length_delta
        report.append(entry)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
    flips = [e for e in report if e.get('flipped')]
    print(f'task x arm cells: {len(report)}')
    print(f'route choice flips: {len(flips)}')
    for entry in report:
        if entry.get('status'):
            print(f"  {entry['task']:44s} {entry['arm']:4s} {entry['status']}")
            continue
        flag = 'FLIP' if entry['flipped'] else 'same'
        line = (f"  {entry['task']:44s} {entry['arm']:4s} {flag} "
                f"legacy={entry['selected_legacy']} anchored={entry['selected_anchored']}")
        if entry.get('flipped'):
            w_star = entry.get('break_even_w_amb')
            line += (f"  break-even w_amb={w_star!r}"
                     f"  ({entry.get('margin')})")
        print(line)
    if args.out:
        print(f'written: {args.out}')


if __name__ == '__main__':
    main()
