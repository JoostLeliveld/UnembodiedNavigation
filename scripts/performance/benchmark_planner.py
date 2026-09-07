#!/usr/bin/env python3
"""Time construction, disk reuse and evaluation on a declared synthetic problem."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src/planning'), str(ROOT/'src/unav_common')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizon', type=int, default=40)
    parser.add_argument('--calls', type=int, default=10)
    parser.add_argument('--network', type=Path,
                        default=ROOT/'logs/studies/icra_commissioning_20260905/network_planner/gp.npz')
    parser.add_argument('--jit', action='store_true', help='Also benchmark C compilation and its persistent reload.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.horizon < 1 or args.calls < 1:
        parser.error('horizon and calls must be positive')
    import numpy as np
    from planning.planners.base_planner import UnicyclePlannerBase
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['UNAV_CASADI_CACHE_DIR'] = str((args.out/'cache').resolve())
    settings = dict(horizon=args.horizon, dt=.25, v_min=0., v_max=.22, w_min=-.8, w_max=.8,
        control_weight=.02, process_noise_xy=.01, process_noise_theta=.02, obs_noise_uv=2.5,
        goal_sigma_uv=30., risk_weight_obs=1., ambiguity_weight=1., optimizer_maxiter=10,
        optimizer_gtol=1e-5, optimizer_warm_start=False, seed=5, use_visibility_model=True,
        camera_network_artifact_path=str(args.network.resolve()),
        camera_params=dict(cam_pos=[-5.,-5.5,4.8], look_at=[1.5,1.5,0.],
                           img_width=1280, img_height=720, fov_h_rad=1.2))
    cases = [(np.tile([.15, .03 + shift*.01], args.horizon),
              [-1.+shift, .5, .2], np.diag([.04, .05, .02]), [640.,380.], [2.,1.], 0.)
             for shift in (0., .1, -.1)]
    protocol = dict(kind='synthetic_objective_and_gradient_timing_not_navigation', settings=settings,
        calls=args.calls, network_sha256=hashlib.sha256(args.network.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    results, reference = {}, None
    for mode in (['native', 'compiled'] if args.jit else ['native']):
        os.environ['UNAV_CASADI_JIT'] = '1' if mode == 'compiled' else '0'
        for attempt in ('cold', 'warm'):
            started = time.perf_counter()
            planner = UnicyclePlannerBase(**settings)
            fn = planner._get_casadi_valgrad([2.,1.], None,
                                           use_observation_risk=True, use_ambiguity_term=True)
            prepare_s = time.perf_counter()-started
            outputs = [fn(*case) for case in cases]
            if reference is None:
                reference = outputs
            for actual, expected in zip(outputs, reference, strict=True):
                np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-9)
                np.testing.assert_allclose(actual[1], expected[1], rtol=1e-8, atol=1e-8)
            samples = []
            for i in range(args.calls):
                start = time.perf_counter()
                fn(*cases[i % len(cases)])
                samples.append((time.perf_counter()-start)*1000.)
            report = dict(prepare_s=prepare_s, median_eval_ms=float(np.median(samples)),
                          cache=fn.cache_info, checked_cases=len(cases), equivalent=True)
            results[f'{mode}_{attempt}'] = report
            print(mode, attempt, json.dumps(report), flush=True)
            (args.out/'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
