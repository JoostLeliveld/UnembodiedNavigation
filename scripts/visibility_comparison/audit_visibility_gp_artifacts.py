#!/usr/bin/env python3
"""Audit the locked visibility GP artifact and its generated mirror."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import REPO_ROOT, read_csv_rows, sha256_file


EXPECTED_ARTIFACT_SHA256 = 'ccbc058311f0e6feeac9aacf034f474af202fba712bd2141752cdfc62de192c8'
EXPECTED_TARGETS_SHA256 = 'f7698d64316a13dd36c0c10ce3571a842e1922e2cfa45f84ed4e95906700e775'
EXPECTED_ARCHIVED_MISMATCH_SHA256 = '1305f94f4041d2ad7f80c53be80286d31fc335edd34d66388844766d9f084373'

DEFAULT_LOCKED_GP = REPO_ROOT / 'paper_artifacts' / 'gp' / 'warehouse_visibility_gp_v1' / 'yolo_score_raw_gp.npz'
DEFAULT_GENERATED_GP = REPO_ROOT / 'logs' / 'visibility_comparison' / 'warehouse_visibility_gp_v1' / 'yolo_score_raw_gp.npz'
DEFAULT_TARGETS = REPO_ROOT / 'logs' / 'visibility_comparison' / 'warehouse_visibility_targets_v1' / 'gp_targets_xy_aggregated.csv'
DEFAULT_ARCHIVED_MISMATCH = (
    REPO_ROOT
    / 'logs'
    / 'visibility_comparison'
    / 'archive'
    / 'mismatched_warehouse_visibility_gp_v1_20260709'
    / 'yolo_score_raw_gp.npz'
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload if isinstance(payload, dict) else {}


def _parse_float(raw: str, default: float = math.nan) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _target_arrays(path: Path, *, min_prob: float) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[float, float], list[float]] = {}
    for row in read_csv_rows(path):
        x = _parse_float(row.get('x', ''))
        y = _parse_float(row.get('y', ''))
        p = _parse_float(row.get('yolo_score_raw', ''))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(p)):
            continue
        grouped.setdefault((round(x, 6), round(y, 6)), []).append(float(p))
    if not grouped:
        raise RuntimeError(f'No finite yolo_score_raw targets in {path}')
    keys = sorted(grouped.keys(), key=lambda item: (item[1], item[0]))
    X = np.asarray([[key[0], key[1]] for key in keys], dtype=float)
    p = np.asarray([float(np.mean(grouped[key])) for key in keys], dtype=float)
    return X, np.clip(p, min_prob, 1.0 - min_prob)


def _artifact_summary(path: Path) -> dict[str, float | int]:
    with np.load(path, allow_pickle=False) as data:
        p_train = np.asarray(data['p_train'], dtype=float)
        p_map = np.asarray(data['P_mean_map'], dtype=float)
        p_plan = np.asarray(data['P_conservative_plan_map'], dtype=float)
        x_train = np.asarray(data['X_train'], dtype=float)
    return {
        'train_points': int(x_train.shape[0]),
        'p_train_mean': float(np.mean(p_train)),
        'p_train_min': float(np.min(p_train)),
        'p_train_max': float(np.max(p_train)),
        'p_mean_map_mean': float(np.mean(p_map)),
        'p_conservative_plan_map_mean': float(np.mean(p_plan)),
    }


def _check(condition: bool, message: str, failures: list[str], passes: list[str]) -> None:
    if condition:
        passes.append(message)
    else:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description='Verify the current/correct visibility GP artifact and archive markers.')
    parser.add_argument('--locked-gp', default=str(DEFAULT_LOCKED_GP))
    parser.add_argument('--generated-gp', default=str(DEFAULT_GENERATED_GP))
    parser.add_argument('--targets', default=str(DEFAULT_TARGETS))
    parser.add_argument('--archived-mismatch-gp', default=str(DEFAULT_ARCHIVED_MISMATCH))
    parser.add_argument('--atol', type=float, default=1e-8)
    args = parser.parse_args()

    locked_gp = Path(args.locked_gp).expanduser().resolve()
    generated_gp = Path(args.generated_gp).expanduser().resolve()
    targets = Path(args.targets).expanduser().resolve()
    archived_mismatch_gp = Path(args.archived_mismatch_gp).expanduser().resolve()
    manifest_path = locked_gp.with_name('gp_manifest.json')
    generated_manifest_path = generated_gp.with_name('gp_manifest.json')

    failures: list[str] = []
    passes: list[str] = []

    for label, path in (
        ('locked GP', locked_gp),
        ('generated GP mirror', generated_gp),
        ('target table', targets),
        ('archived mismatched GP', archived_mismatch_gp),
        ('locked manifest', manifest_path),
        ('generated manifest mirror', generated_manifest_path),
    ):
        _check(path.is_file(), f'{label} exists: {path}', failures, passes)
    if failures:
        for message in failures:
            print(f'FAIL {message}')
        return 1

    locked_sha = sha256_file(locked_gp)
    generated_sha = sha256_file(generated_gp)
    targets_sha = sha256_file(targets)
    archived_sha = sha256_file(archived_mismatch_gp)
    manifest = _load_json(manifest_path)
    generated_manifest = _load_json(generated_manifest_path)

    _check(locked_sha == EXPECTED_ARTIFACT_SHA256, f'locked artifact sha256 = {locked_sha}', failures, passes)
    _check(generated_sha == locked_sha, f'generated mirror matches locked artifact sha256 = {generated_sha}', failures, passes)
    _check(targets_sha == EXPECTED_TARGETS_SHA256, f'target table sha256 = {targets_sha}', failures, passes)
    _check(archived_sha == EXPECTED_ARCHIVED_MISMATCH_SHA256, f'archived mismatched artifact sha256 = {archived_sha}', failures, passes)
    _check(archived_sha != locked_sha, 'archived mismatched artifact differs from locked artifact', failures, passes)

    _check(manifest.get('status') == 'LOCKED_CURRENT_CORRECT', 'locked manifest status is LOCKED_CURRENT_CORRECT', failures, passes)
    _check(generated_manifest == manifest, 'generated manifest mirrors locked manifest', failures, passes)
    _check(manifest.get('artifact_sha256') == locked_sha, 'manifest artifact_sha256 matches locked artifact', failures, passes)
    _check(manifest.get('gp_targets_sha256') == targets_sha, 'manifest gp_targets_sha256 matches target table', failures, passes)
    archived_manifest = dict(manifest.get('archived_mismatched_artifact') or {})
    _check(
        archived_manifest.get('artifact_sha256') == archived_sha,
        'manifest archived_mismatched_artifact sha256 matches archived artifact',
        failures,
        passes,
    )

    min_prob = float(manifest.get('min_prob', 1e-4))
    target_X, target_p = _target_arrays(targets, min_prob=min_prob)
    with np.load(locked_gp, allow_pickle=False) as data:
        X_train = np.asarray(data['X_train'], dtype=float)
        p_train = np.asarray(data['p_train'], dtype=float)
        required_keys = {'xs', 'ys', 'X_train', 'p_train', 'P_mean_map', 'P_conservative_plan_map'}
        missing_keys = sorted(required_keys.difference(data.files))
    _check(not missing_keys, f'locked artifact contains required keys: {sorted(required_keys)}', failures, passes)
    _check(X_train.shape == target_X.shape, f'X_train shape matches targets: {X_train.shape}', failures, passes)
    _check(p_train.shape == target_p.shape, f'p_train shape matches targets: {p_train.shape}', failures, passes)
    if X_train.shape == target_X.shape:
        _check(np.allclose(X_train, target_X, atol=float(args.atol), rtol=0.0), 'X_train coordinates match target table order', failures, passes)
    if p_train.shape == target_p.shape:
        _check(np.allclose(p_train, target_p, atol=float(args.atol), rtol=0.0), 'p_train matches clipped yolo_score_raw targets', failures, passes)

    summary = _artifact_summary(locked_gp)
    expected_summary = {
        'train_points': 139,
        'p_train_mean': 0.5971043165467627,
        'p_train_min': 0.0001,
        'p_train_max': 0.9999,
        'p_mean_map_mean': 0.6454923347436693,
        'p_conservative_plan_map_mean': 0.5727276236451633,
    }
    for key, expected in expected_summary.items():
        actual = summary[key]
        if isinstance(expected, int):
            _check(actual == expected, f'{key} = {actual}', failures, passes)
        else:
            _check(abs(float(actual) - float(expected)) <= 1e-10, f'{key} = {actual:.16g}', failures, passes)

    for message in passes:
        print(f'PASS {message}')
    for message in failures:
        print(f'FAIL {message}')
    if failures:
        print(f'Visibility GP audit failed with {len(failures)} issue(s).')
        return 1
    print('Visibility GP audit passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
