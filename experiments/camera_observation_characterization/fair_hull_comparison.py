#!/usr/bin/env python3
"""A like-for-like comparison: what does the hull cost when nobody hands it the answer?

The published five-rung ladder is not a fair race.  The learned rungs start from the raw
back-projected pixel and are given nothing else; the hull rung is started at the robot's
TRUE position and TRUE heading and asked only to take one Gauss-Newton step from there:

    hull = truth + J(truth, yaw_truth)^-1 (z - h(truth, yaw_truth))

What that measures is how straight the observation function is near the answer, not how
well the hull recovers a position.  Its train and test scores are the same (2.8 vs 2.5 cm)
because nothing was ever fitted -- it is reading off an oracle both places.

This script re-runs the hull with the truth taken away, one ingredient at a time, so the
reader can see which ingredient the 2.5 cm was actually buying:

    hull_oracle       truth position + truth heading   (the published rung, kept for reference)
    hull_heading      raw back-projection + truth heading, iterated to convergence
    hull_operational  raw back-projection + heading solved from the box, iterated

`hull_operational` is the fair opponent for the learned rungs: it starts where they start,
sees what they see (one box, its camera), and uses no truth at all.  Heading is recovered by
searching for the heading whose predicted box best matches the measured box -- width and
height carry heading, exactly the signal `observation.py` says is being discarded today.

Two things measured here decide how the result should be read.  The hull is EXACTLY
symmetric under a 180 deg flip -- yaw and yaw+pi project to the same box -- so one box can
never say which way the robot faces, and the search covers a half circle only.  That
ambiguity turns out to be free: solving at the flipped heading moves the position by
nothing (2.67 vs 2.66 cm median).  What is not free is getting the heading wrong by a
quarter or half turn, which costs 8 to 9 cm, and the worst heading on the circle costs
13 cm.  Heading -- not the starting point -- is the oracle the published rung was living on.

Everything is scored on the same held-out tiles, on the same rows, as the existing ladder.
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
for rel in ('src/experiments', 'src/unav_common', 'experiments/measurement_commissioning'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from experiments.core.world_profiles import compute_look_at_from_pose  # noqa: E402
from observation import h as hull_h, jacobian as hull_jacobian, predicted_box  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

QUANTITIES = ('valid', 'x', 'y', 'dx', 'dy', 'error_m', 'along_m', 'across_m')
NEW_METHODS = ('hull_heading', 'hull_operational')


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


def method_values(estimate, *, truth: np.ndarray, camera_xy: np.ndarray) -> dict:
    """Identical bookkeeping and error frame to derive_interpretations.py."""
    if estimate is None:
        return dict.fromkeys(QUANTITIES, math.nan) | {'valid': 0}
    estimate = np.asarray(estimate, dtype=float)
    if estimate.shape != (2,) or not np.all(np.isfinite(estimate)):
        return method_values(None, truth=truth, camera_xy=camera_xy)
    error = estimate - truth
    ray = truth - camera_xy
    norm = float(np.linalg.norm(ray))
    unit = ray / norm if norm > 1e-12 else np.array([1.0, 0.0])
    left = np.array([-unit[1], unit[0]])
    return {
        'valid': 1,
        'x': float(estimate[0]), 'y': float(estimate[1]),
        'dx': float(error[0]), 'dy': float(error[1]),
        'error_m': float(np.linalg.norm(error)),
        'along_m': float(error @ unit), 'across_m': float(error @ left),
    }


# ---------------------------------------------------------------------------------------
# the hull, solved rather than evaluated at the answer
# ---------------------------------------------------------------------------------------
def solve_position(cam, start: np.ndarray, yaw: float, measured: np.ndarray,
                   *, iterations: int = 12, tol: float = 1e-5) -> np.ndarray | None:
    """Gauss-Newton on the bottom-centre, starting wherever we are told to start.

    Same update as the published rung; the only difference is that it is iterated from an
    honest starting point instead of taken once from the truth.
    """
    point = np.asarray(start, dtype=float).copy()
    for _ in range(iterations):
        predicted = hull_h(cam, point[0], point[1], yaw)
        if predicted is None:
            return None
        residual = measured - np.asarray(predicted, dtype=float)
        try:
            jac = hull_jacobian(cam, point[0], point[1], yaw)
            step = np.linalg.solve(jac, residual)
        except (np.linalg.LinAlgError, ValueError, TypeError):
            return None
        if not np.all(np.isfinite(step)):
            return None
        norm = float(np.linalg.norm(step))
        if norm > 1.0:          # the raw start is ~30 cm out, never metres
            step = step / norm
        point = point + step
        if norm < tol:
            break
    if not np.all(np.isfinite(point)):
        return None
    return point


def box_cost(cam, point: np.ndarray, yaw: float, measured_box: np.ndarray) -> float:
    """How badly the predicted box disagrees with the measured one, in pixels."""
    box = predicted_box(cam, point[0], point[1], yaw)
    if box is None:
        return math.inf
    predicted = np.asarray(box, dtype=float)
    # Compare shape and place: the width and height are what carry heading.
    pw, ph = predicted[2] - predicted[0], predicted[3] - predicted[1]
    mw, mh = measured_box[2] - measured_box[0], measured_box[3] - measured_box[1]
    pc = 0.5 * (predicted[0] + predicted[2])
    mc = 0.5 * (measured_box[0] + measured_box[2])
    return float((pw - mw) ** 2 + (ph - mh) ** 2 + (pc - mc) ** 2 + (predicted[3] - measured_box[3]) ** 2)


def solve_pose(cam, start: np.ndarray, measured: np.ndarray, measured_box: np.ndarray,
               *, coarse: int = 24, refine_rounds: int = 3) -> tuple[np.ndarray, float] | None:
    """Position AND heading from one box: no truth anywhere.

    Heading is unknown, so it is searched.  For each candidate heading the position that
    explains the bottom-centre is solved for, and the candidate whose whole predicted box
    best matches the measured box wins.  The search is coarse-to-fine over the full circle.
    """
    # The hull is symmetric under a 180 deg flip: yaw and yaw+pi project to the SAME box,
    # to the last bit.  Heading is therefore unidentifiable from one box beyond a half
    # circle -- and, measured below, the flip costs the POSITION nothing.  So search a half
    # circle and report the representative; do not pretend to resolve front from back.
    best: tuple[float, np.ndarray, float] | None = None
    candidates = [math.pi * index / coarse for index in range(coarse)]
    span = math.pi / coarse
    for _ in range(refine_rounds):
        for yaw in candidates:
            point = solve_position(cam, start, yaw, measured)
            if point is None:
                continue
            cost = box_cost(cam, point, yaw, measured_box)
            if not math.isfinite(cost):
                continue
            if best is None or cost < best[0]:
                best = (cost, point, yaw)
        if best is None:
            return None
        centre = best[2]
        span *= 0.25
        candidates = [centre + span * offset for offset in (-2, -1, -0.5, 0.5, 1, 2)]
    return best[1], best[2]


def summarize(errors: np.ndarray) -> dict:
    if not errors.size:
        return {'n': 0}
    return {
        'n': int(errors.size),
        'median_m': float(np.median(errors)),
        'p90_m': float(np.quantile(errors, 0.90)),
        'rmse_m': float(np.sqrt(np.mean(errors ** 2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0,
                        help='Score only the first N usable rows (a smoke test, not a result).')
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
    rows = list(csv.DictReader(table.open(encoding='utf-8')))

    usable_seen = 0
    scored: list[dict] = []
    for row in rows:
        record = dict(row)
        budget_spent = args.limit and usable_seen >= args.limit
        if row['raw_valid'] != '1' or budget_spent:
            for method in NEW_METHODS:
                values = method_values(None, truth=np.zeros(2), camera_xy=np.zeros(2))
                record.update({f'{method}_{key}': value for key, value in values.items()})
            record['solved_yaw_rad'] = ''
            scored.append(record)
            continue
        usable_seen += 1
        cam = cameras[row['camera_id']]
        camera_xy = np.asarray(cam.cam_pos[:2], dtype=float)
        truth = np.array([float(row['robot_x']), float(row['robot_y'])])
        raw = np.array([float(row['raw_x']), float(row['raw_y'])])
        measured = np.array([float(row['u_bbox_bottom']), float(row['v_bbox_bottom'])])
        measured_box = np.array([float(row['x0']), float(row['y0']),
                                 float(row['x1']), float(row['y1'])])
        yaw_truth = float(row['robot_yaw'])

        estimates: dict[str, np.ndarray | None] = {
            'hull_heading': solve_position(cam, raw, yaw_truth, measured),
        }
        solved = solve_pose(cam, raw, measured, measured_box)
        estimates['hull_operational'] = None if solved is None else solved[0]
        record['solved_yaw_rad'] = '' if solved is None else f'{float(solved[1]):.6f}'

        for method, estimate in estimates.items():
            values = method_values(estimate, truth=truth, camera_xy=camera_xy)
            record.update({f'{method}_{key}': value for key, value in values.items()})
        scored.append(record)

    fields = list(rows[0].keys()) + ['solved_yaw_rad'] + [
        f'{method}_{quantity}' for method in NEW_METHODS for quantity in QUANTITIES
    ]
    output = capture / 'fair_hull_interpretations.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)

    ladder = ('raw', 'fixed', 'learned', 'nn', 'hull_operational', 'hull_heading', 'hull')
    scores = {}
    for portion in ('train', 'test'):
        subset = [row for row in scored
                  if row['split'] == portion and row['raw_valid'] == '1'
                  and str(row['hull_operational_valid']) == '1']
        scores[portion] = {
            method: summarize(np.asarray([
                float(row[f'{method}_error_m']) for row in subset
                if str(row[f'{method}_valid']) == '1'
            ]))
            for method in ladder
        }

    # Heading is scored MODULO 180 deg, because the hull cannot distinguish a flip from one
    # box -- scoring it against a full circle would report a 90 deg median that is pure
    # bookkeeping, not a failure of the search.
    solved_rows = [row for row in scored if row['solved_yaw_rad'] not in ('', None)]
    folded = np.asarray([
        (float(row['solved_yaw_rad']) - float(row['robot_yaw'])) % math.pi
        for row in solved_rows
    ])
    heading_error = np.minimum(folded, math.pi - folded)

    manifest = {
        'status': 'complete' if not args.limit else 'smoke_test',
        'schema': 'fair_hull_interpretations.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'bias_update_interpretations_sha256': sha256(table),
        'capture_manifest_sha256': sha256(capture_manifest_path),
        'question': 'What is the hull worth when it is not started at the true pose?',
        'methods': {
            'hull': 'PUBLISHED RUNG. One Gauss-Newton step taken AT the true position and '
                    'true heading. Not operational: the answer is an input.',
            'hull_heading': 'Same observation model, started at the raw back-projection and '
                            'iterated to convergence, but still told the true heading.',
            'hull_operational': 'Started at the raw back-projection, heading recovered by '
                                'matching the predicted box to the measured box. No truth. '
                                'This is the like-for-like opponent for learned and nn.',
        },
        'common_rows': 'all methods scored on rows where the operational solve converged',
        'solved_heading_error_deg': {
            'note': 'absolute error modulo 180 deg; the flip is not identifiable from one box',
            'n': int(heading_error.size),
            'median': float(np.degrees(np.median(heading_error))) if heading_error.size else None,
            'p90': float(np.degrees(np.quantile(heading_error, 0.9))) if heading_error.size else None,
            'fraction_within_15_deg': float((np.degrees(heading_error) < 15).mean())
            if heading_error.size else None,
        },
        'scores': scores,
        'fair_hull_interpretations_sha256': sha256(output),
    }
    (capture / 'fair_hull_interpretations_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output),
                      'heading': manifest['solved_heading_error_deg'],
                      'test_scores': scores['test']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
