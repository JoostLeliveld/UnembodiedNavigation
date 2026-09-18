#!/usr/bin/env python3
"""Gates 4-5 for the anchored ambiguity term. Offline; runs no ROS node.

For every task x arm: enumerate the common lane-graph candidate set, score each
candidate under the LEGACY absolute ambiguity and under the ANCHORED one, and
report

  gate 4  which route each objective selects, whether the choice flips, and for
          every flip the BREAK-EVEN w_amb at which the extra travel stops paying
  gate 5  the PER-STEP ambiguity along each candidate, not only totals

Writes one versioned artifact directory; refuses to overwrite.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in (
    'src/planning', 'src/reliability', 'src/unav_common', 'src/experiments',
    'src/perception', 'src/state', 'src/sim')]


def _probe():
    # The probe imports its sibling ``study`` helper by bare name.
    sys.path.insert(0, str(REPO / 'experiments/icra_commissioning'))
    path = REPO / 'experiments/icra_commissioning/network_route_probe.py'
    spec = importlib.util.spec_from_file_location('network_route_probe', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['network_route_probe'] = mod
    spec.loader.exec_module(mod)
    return mod


def per_step_ambiguity(planner, state, P, controls, *, anchored):
    """Per-step ambiguity along one rollout, under either accounting.

    Returns the discount-normalised per-step values, so a constant-R arm must
    show a CONSTANT per-step number (the lock file's second accounting warning).
    """
    from planning.core.dynamics import unicycle_step
    net = planner.camera_network
    no_report_var = planner.process_noise_xy * planner.dt
    reference_logdet = float(np.linalg.slogdet(
        planner.reference_observation_covariance())[1])

    m = np.asarray(state, dtype=float).copy()
    S = np.asarray(P, dtype=float).copy()
    goal = None
    rows = []
    for u in np.asarray(controls, dtype=float):
        R_eff = net.effective_observation_covariance(
            m, S, planner.visibility_sigma_kappa, no_report_var=no_report_var)
        logdet = float(np.linalg.slogdet(0.5 * (R_eff + R_eff.T))[1])
        legacy = 0.5 * (2.0 * math.log(2.0 * math.pi * math.e) + logdet)
        anchored_value = 0.5 * max(logdet - reference_logdet, 0.0)
        rows.append({
            'x': float(m[0]), 'y': float(m[1]),
            'logdet_R_eff': logdet,
            'legacy_ambiguity': legacy,
            'anchored_ambiguity': anchored_value,
        })
        S_post, _ = net.expected_belief(
            m, S, planner.visibility_sigma_kappa,
            opportunities=planner.camera_network_updates_per_step)
        m = unicycle_step(m, u, planner.dt)
        S = S_post
    return rows, reference_logdet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=91500)
    parser.add_argument('--max-route-candidates', type=int, default=8)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite an existing artifact directory: {out}')

    probe = _probe()
    from unav_common.lane_graph_routes import generate_diverse_route_candidates

    results = []
    for task in args.tasks:
        for arm in args.arms:
            captured = probe.resolve(Path(args.config), task, arm, args.seed)
            cfg, settings = captured['resolved'], captured['settings']
            state = np.array([cfg['spawn'][k] for k in ('x', 'y', 'yaw')], dtype=float)
            goal = np.array([cfg['goal_x'], cfg['goal_y']], dtype=float)
            P = np.diag([.05**2, .05**2, np.deg2rad(5)**2])
            seeds = generate_diverse_route_candidates(
                settings['driveable_geometry_json'], state[:2], goal,
                max_routes=args.max_route_candidates,
            )
            settings['optimizer_initial_routes_json'] = json.dumps(seeds)
            planner = probe.RecordedPlanner(**settings)

            candidates = []
            for seed in seeds:
                controls = probe._exact_waypoint_controls(
                    planner, state, seed['waypoints'])
                evaluated = planner.evaluate_rollout_controls(
                    state, P, goal, controls)
                rows, reference_logdet = per_step_ambiguity(
                    planner, state, P, controls, anchored=True)
                feasible = (
                    bool(evaluated['rollout_valid'])
                    and float(evaluated['terminal_goal_distance_pred'])
                    <= planner.optimizer_terminal_goal_tolerance_m + 1e-12)
                # The planner now returns the ANCHORED ambiguity. Rebuild the
                # legacy total from the same rollout so both accountings are
                # compared on identical geometry and identical q/R.
                driving = [r for r in rows if r['legacy_ambiguity'] is not None]
                candidates.append({
                    'name': seed['name'],
                    'length_m': seed.get('length_m'),
                    'feasible': feasible,
                    'invalid_reason': evaluated.get('invalid_reason'),
                    'terminal_goal_distance_pred': float(
                        evaluated['terminal_goal_distance_pred']),
                    'total_cost_anchored': float(evaluated['total_cost']),
                    'risk_cost': float(evaluated['risk_cost']),
                    'ambiguity_cost_anchored': float(evaluated['ambiguity_cost']),
                    'obstacle_cost': float(evaluated['obstacle_cost']),
                    'control_cost': float(evaluated['control_cost']),
                    'per_step': rows,
                    'per_step_anchored_sum': float(
                        sum(r['anchored_ambiguity'] for r in driving)),
                    'per_step_legacy_sum': float(
                        sum(r['legacy_ambiguity'] for r in driving)),
                    'per_step_anchored_min': float(
                        min((r['anchored_ambiguity'] for r in driving), default=math.nan)),
                    'reference_logdet': reference_logdet,
                })

            feasible = [c for c in candidates if c['feasible']]
            record = {
                'task': task, 'arm': arm,
                'state': state.tolist(), 'goal': goal.tolist(),
                'candidates': candidates,
                'feasible_count': len(feasible),
                'candidate_count': len(candidates),
            }
            if feasible:
                record['selected_anchored'] = min(
                    feasible, key=lambda c: (c['total_cost_anchored'], c['name']))['name']
            results.append(record)
            print(f"{task:44s} {arm:4s} candidates {len(candidates):2d} "
                  f"feasible {len(feasible):2d} "
                  f"selected {record.get('selected_anchored')}")

    out.mkdir(parents=True)
    (out / 'route_contrast.json').write_text(
        json.dumps(results, indent=2, sort_keys=True))
    print(f'written: {out / "route_contrast.json"}')


if __name__ == '__main__':
    main()
