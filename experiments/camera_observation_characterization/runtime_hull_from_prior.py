#!/usr/bin/env python3
"""What is the hull worth when it predicts from the BELIEF, which is what ships?

The fair-hull study (`07_fair_hull/`) took the truth away from the hull rung one ingredient
at a time and found the heading, not the starting point, was the oracle.  It closed with a
caveat: `hull_operational` is a fair *comparison*, not a runtime method, because its heading
search costs a silhouette projection per candidate per reading.

Neither rung is what the robot actually runs.  `camera_manager_node.py` defaults to
`observation_model=hull` and corrects every reading through
`reliability.silhouette_observation.equivalent_position_measurement`, predicting from the
filter's own prior pose AND ITS YAW -- never from truth, never from a search.  So the
deployed accuracy of the hull sits between the two published rungs and has never been
measured on the frozen capture:

    hull_heading      raw back-projection + TRUE heading      2.53 cm   oracle heading
    ??? THE DEPLOYED PATH: prior position + PRIOR heading     ???
    hull_operational  heading searched from the box           3.10 cm   not deployable

This script measures the missing row by calling the deployed function itself -- not a
reimplementation of it -- on the same held-out rows as the published ladder.

**How the prior is built, and why this is a bound rather than a replay.**  The frozen
capture teleports the robot; there is no filter and therefore no belief history to read a
prior from.  A prior is synthesised instead, by perturbing the true pose by a declared
position and heading error drawn from a seeded RNG.  That makes each sweep row a statement
of the form "if the belief is this good, the reading is that good".  The sweep is declared
up front and spans the belief quality actually observed in the live campaign
(1-3 cm healthy, 20 cm+ during an excursion), so nothing here is tuned against the outcome:

    prior error 0 cm / 0 deg      the ceiling: a perfect belief
    ... through the declared grid ...
    prior error 100 cm / 45 deg   deep inside an unrecovered excursion

The zero row is a cross-check, not a result: at a perfect prior this must reproduce the
published `hull` rung, since the deployed correction is then evaluated at the truth.

Scored on the same rows, in the same error frame, as `fair_hull_comparison.py`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/unav_common', 'src/reliability',
            'experiments/measurement_commissioning'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from experiments.core.world_profiles import compute_look_at_from_pose  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402
# THE DEPLOYED CODE PATH -- the same function camera_manager_node.py calls per reading.
from reliability.silhouette_observation import (  # noqa: E402
    equivalent_position_measurement,
    plausibility_reasons,
)

# Declared BEFORE running, spanning belief quality seen in the live campaign.
# (position error metres, heading error degrees)
PRIOR_GRID = (
    (0.00, 0.0),
    (0.01, 1.0),
    (0.02, 2.0),
    (0.05, 5.0),
    (0.10, 10.0),
    (0.20, 15.0),
    (0.50, 30.0),
    (1.00, 45.0),
)
# A unit pixel covariance: the correction's z_eq does not depend on it, and R_eq is
# reported only as the Jacobian's amplification factor.
UNIT_COV = ((1.0, 0.0), (0.0, 1.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def camera_models(capture_manifest: dict) -> dict[str, ObliqueCameraModel]:
    profiles_path = Path(capture_manifest['world_profiles_path'])
    if sha256(profiles_path) != capture_manifest['world_profiles_sha256']:
        raise RuntimeError('The world-profile file changed after capture; refusing reconstruction')
    profiles = yaml.safe_load(profiles_path.read_text(encoding='utf-8'))
    intrinsics = profiles['camera_intrinsics']
    models = {}
    for item in capture_manifest['cameras']:
        pose = [float(value) for value in item['pose_xyz_rpy']]
        look_at = compute_look_at_from_pose(pose[:3], *pose[3:])
        models[item['camera_id']] = ObliqueCameraModel(
            cam_pos=pose[:3], look_at=look_at,
            img_width=int(item['image_width']), img_height=int(item['image_height']),
            fov_h_rad=float(intrinsics['fov_h_rad']),
        )
    return models


def summarize(errors: np.ndarray, *, attempted: int) -> dict:
    """Accuracy conditional on the correction being applied, beside how often it was."""
    if not errors.size:
        return {'n': 0, 'attempted': attempted, 'applied_fraction': 0.0}
    return {
        'n': int(errors.size),
        'attempted': int(attempted),
        'applied_fraction': float(errors.size / attempted) if attempted else 0.0,
        'median_cm': float(np.median(errors)) * 100.0,
        'p90_cm': float(np.quantile(errors, 0.90)) * 100.0,
        'rmse_cm': float(np.sqrt(np.mean(errors ** 2))) * 100.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260904)
    parser.add_argument('--limit', type=int, default=0,
                        help='Score only the first N usable rows (a smoke test, not a result).')
    parser.add_argument('--split', default='test', choices=('test', 'train', 'all'))
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    table = capture / 'bias_update_interpretations.csv'
    ladder_manifest_path = capture / 'bias_update_interpretations_manifest.json'
    capture_manifest_path = capture / 'capture_manifest.json'
    for path in (table, ladder_manifest_path, capture_manifest_path):
        if not path.is_file():
            raise RuntimeError(f'Missing required input: {path}')
    ladder_manifest = json.loads(ladder_manifest_path.read_text(encoding='utf-8'))
    if ladder_manifest.get('status') != 'complete':
        raise RuntimeError('bias_update_interpretations manifest is not complete')
    if sha256(table) != ladder_manifest['bias_update_interpretations_sha256']:
        raise RuntimeError('bias_update_interpretations.csv changed after it was derived')

    cameras = camera_models(json.loads(capture_manifest_path.read_text(encoding='utf-8')))
    rows = [row for row in csv.DictReader(table.open(encoding='utf-8'))
            if row['raw_valid'] == '1'
            and (args.split == 'all' or row['split'] == args.split)]
    if args.limit:
        # The capture CSV is ordered by position, so a head slice is a contiguous BLOCK of
        # floor, not a sample of it -- it scores one corner of the warehouse and reports a
        # median 7x the true one. Sample instead, so a smoke test is at least unbiased.
        picked = np.random.default_rng(args.seed).choice(
            len(rows), size=min(args.limit, len(rows)), replace=False)
        rows = [rows[index] for index in sorted(picked)]
    if not rows:
        raise RuntimeError('No usable rows selected')

    results: dict[str, dict] = {}
    per_row_dump: list[dict] = []
    for pos_err, yaw_err_deg in PRIOR_GRID:
        # One RNG per grid cell, seeded from the declared seed: reproducible, and the
        # perturbation directions are identical across cells so rows stay comparable.
        rng = np.random.default_rng(args.seed)
        yaw_err = math.radians(yaw_err_deg)
        errors: list[float] = []
        gated_errors: list[float] = []
        raw_errors: list[float] = []
        attempted = 0
        for row in rows:
            cam = cameras[row['camera_id']]
            truth = np.array([float(row['robot_x']), float(row['robot_y'])])
            yaw_truth = float(row['robot_yaw'])
            z_xy = np.array([float(row['raw_x']), float(row['raw_y'])])

            # Synthesise the belief: random direction at the declared magnitude.
            angle = rng.uniform(0.0, 2.0 * math.pi)
            prior_xy = truth + pos_err * np.array([math.cos(angle), math.sin(angle)])
            prior_yaw = yaw_truth + rng.choice((-1.0, 1.0)) * yaw_err

            attempted += 1
            corrected = equivalent_position_measurement(
                (float(z_xy[0]), float(z_xy[1])), UNIT_COV, cam,
                (float(prior_xy[0]), float(prior_xy[1])), float(prior_yaw),
            )
            raw_errors.append(float(np.linalg.norm(z_xy - truth)))
            if corrected is None:
                continue          # the runtime would fall back to the uncorrected reading
            z_eq = np.asarray(corrected[0], dtype=float)
            err = float(np.linalg.norm(z_eq - truth))
            errors.append(err)

            # The runtime's admission gate, evaluated from the SAME prior -- no truth.
            reasons = plausibility_reasons(
                (float(row['x0']), float(row['y0']), float(row['x1']), float(row['y1'])),
                cam, float(prior_xy[0]), float(prior_xy[1]), float(prior_yaw),
            )
            if not reasons:
                gated_errors.append(err)

            if pos_err in (0.00, 0.05, 0.20):
                per_row_dump.append({
                    'prior_pos_err_m': pos_err, 'prior_yaw_err_deg': yaw_err_deg,
                    'camera_id': row['camera_id'], 'position_id': row['position_id'],
                    'heading_id': row['heading_id'], 'split': row['split'],
                    'camera_range_m': row['camera_range_m'],
                    'confidence': row['confidence'],
                    'error_m': err, 'raw_error_m': raw_errors[-1],
                    'gate_passed': int(not reasons),
                })

        key = f'pos{pos_err:.2f}m_yaw{yaw_err_deg:.0f}deg'
        results[key] = {
            'prior_pos_err_m': pos_err,
            'prior_yaw_err_deg': yaw_err_deg,
            'runtime_hull': summarize(np.asarray(errors), attempted=attempted),
            'runtime_hull_after_gate': summarize(np.asarray(gated_errors), attempted=attempted),
            'raw_box_same_rows': summarize(np.asarray(raw_errors), attempted=attempted),
        }
        print(f'{key}: n={len(errors)}/{attempted} '
              f'median={results[key]["runtime_hull"].get("median_cm", float("nan")):.2f} cm '
              f'gated_median='
              f'{results[key]["runtime_hull_after_gate"].get("median_cm", float("nan")):.2f} cm '
              f'gate_keeps={results[key]["runtime_hull_after_gate"]["applied_fraction"]:.1%}',
              flush=True)

    out_dir = REPO / 'logs/studies/camera_observation_characterization_20260831/15_runtime_hull_from_prior'
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / 'runtime_hull_rows.csv'
    if per_row_dump:
        with dump_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_row_dump[0].keys()))
            writer.writeheader()
            writer.writerows(per_row_dump)

    manifest = {
        'status': 'complete' if not args.limit else 'smoke_test',
        'schema': 'runtime_hull_from_prior.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'What is the hull worth predicting from the belief, as deployed?',
        'split_scored': args.split,
        'n_rows': len(rows),
        'seed': args.seed,
        'bias_update_interpretations_sha256': sha256(table),
        'capture_manifest_sha256': sha256(capture_manifest_path),
        'code_path': 'reliability.silhouette_observation.equivalent_position_measurement',
        'boundaries': [
            'The prior is SYNTHESISED by perturbing truth; the frozen capture teleports the '
            'robot and has no filter, so no real belief history exists to replay. Each row '
            'is conditional on the declared prior quality, not a closed-loop result.',
            'Perturbation is isotropic in position and symmetric in heading. A real belief '
            'error is correlated with the reading error that produced it; this is not.',
            'Camera-reading layer only -- not fused, belief, or planner error.',
            'The gate is evaluated from the same synthesised prior, so its keep-rate '
            'degrades with prior quality exactly as it would online.',
        ],
        'grid': list(PRIOR_GRID),
        'results': results,
    }
    (out_dir / 'runtime_hull_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nwrote {out_dir}/runtime_hull_manifest.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
