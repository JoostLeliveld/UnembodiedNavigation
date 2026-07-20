#!/usr/bin/env python3
"""Build the paper GP target table from YOLO detector-score samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import (
    CURRENT_TARGETS_DIR,
    GP_TARGET_COLUMNS,
    LOGS_ROOT,
    parse_float,
    read_csv_rows,
    write_csv,
    write_manifest,
)


AGGREGATED_GP_TARGET_COLUMNS = (
    'sample_id',
    'x',
    'y',
    'heading_sample_count',
    'yolo_score_raw',
    'yolo_raw_best_score',
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _finite_score(row: dict[str, str]) -> float:
    score = parse_float(row.get('yolo_raw_best_score', ''), math.nan)
    if not math.isfinite(score):
        score = parse_float(row.get('yolo_score_raw', ''), math.nan)
    if not math.isfinite(score):
        return math.nan
    return float(np.clip(score, 0.0, 1.0))


def _aggregate_rows_by_xy(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[float, float], dict[str, object]] = {}
    for row in rows:
        x = parse_float(row.get('x', ''), math.nan)
        y = parse_float(row.get('y', ''), math.nan)
        score = parse_float(row.get('yolo_score_raw', ''), math.nan)
        best_score = parse_float(row.get('yolo_raw_best_score', ''), math.nan)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(score)):
            continue
        key = (round(float(x), 6), round(float(y), 6))
        bucket = grouped.setdefault(
            key,
            {
                'x': float(x),
                'y': float(y),
                'count': 0,
                'scores': [],
                'best_scores': [],
            },
        )
        bucket['count'] = int(bucket['count']) + 1
        bucket['scores'].append(float(score))
        if math.isfinite(best_score):
            bucket['best_scores'].append(float(best_score))

    out_rows: list[dict[str, str]] = []
    for idx, key in enumerate(sorted(grouped.keys(), key=lambda item: (item[1], item[0]))):
        bucket = grouped[key]
        scores = list(bucket['scores'])
        best_scores = list(bucket['best_scores'])
        out_rows.append(
            {
                'sample_id': f'xy_{idx:06d}',
                'x': f"{float(bucket['x']):.8f}",
                'y': f"{float(bucket['y']):.8f}",
                'heading_sample_count': str(int(bucket['count'])),
                'yolo_score_raw': f'{float(np.mean(scores)):.8f}',
                'yolo_raw_best_score': '' if not best_scores else f'{float(np.mean(best_scores)):.8f}',
            }
        )
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Create gp_targets.csv and heading-marginal gp_targets_xy_aggregated.csv from YOLO perception targets.'
    )
    parser.add_argument('--perception-targets', default=str(CURRENT_TARGETS_DIR / 'perception_targets.csv'))
    parser.add_argument('--out', default=str(CURRENT_TARGETS_DIR))
    args = parser.parse_args()

    targets_path = Path(args.perception_targets).expanduser().resolve()
    if not targets_path.is_file():
        raise RuntimeError(f'Perception targets CSV not found: {targets_path}')

    output_dir = Path(args.out).expanduser().resolve()
    allowed_root = LOGS_ROOT.resolve()
    if allowed_root not in output_dir.parents and output_dir != allowed_root:
        raise RuntimeError(f'GP target output must stay under {allowed_root}: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(targets_path)
    extractor_manifest = _load_json(targets_path.parent / 'manifest.json')

    out_rows: list[dict[str, str]] = []
    yolo_raw_values: list[float] = []
    for row in rows:
        score = _finite_score(row)
        if math.isfinite(score):
            yolo_raw_values.append(float(score))
        out_rows.append(
            {
                'sample_id': str(row.get('sample_id', '')).strip(),
                'x': str(row.get('x', '')).strip(),
                'y': str(row.get('y', '')).strip(),
                'yolo_score_raw': '' if not math.isfinite(score) else f'{float(score):.8f}',
                'yolo_raw_best_score': '' if not math.isfinite(score) else f'{float(score):.8f}',
            }
        )

    if not yolo_raw_values:
        raise RuntimeError(
            'No finite YOLO raw scores were found in perception_targets.csv. '
            'Run extract_perception_targets.py with the paper YOLO model first.'
        )

    gp_targets_path = output_dir / 'gp_targets.csv'
    write_csv(gp_targets_path, GP_TARGET_COLUMNS, out_rows)

    aggregated_rows = _aggregate_rows_by_xy(out_rows)
    aggregated_gp_targets_path = output_dir / 'gp_targets_xy_aggregated.csv'
    write_csv(aggregated_gp_targets_path, AGGREGATED_GP_TARGET_COLUMNS, aggregated_rows)

    write_csv(
        output_dir / 'target_summary.csv',
        ('method_id', 'available', 'row_count', 'mean_target', 'min_target', 'max_target'),
        [
            {
                'method_id': 'yolo_score_raw',
                'available': '1',
                'row_count': str(int(len(yolo_raw_values))),
                'mean_target': f'{float(np.mean(yolo_raw_values)):.8f}',
                'min_target': f'{float(np.min(yolo_raw_values)):.8f}',
                'max_target': f'{float(np.max(yolo_raw_values)):.8f}',
            }
        ],
    )

    write_manifest(
        output_dir / 'target_manifest.json',
        {
            'perception_targets': str(targets_path),
            'gp_targets_path': str(gp_targets_path),
            'gp_targets_xy_aggregated_path': str(aggregated_gp_targets_path),
            'row_count': int(len(out_rows)),
            'xy_aggregated_row_count': int(len(aggregated_rows)),
            'capture_metadata': dict(extractor_manifest.get('capture_metadata', {})),
            'yolo_summary': dict(extractor_manifest.get('yolo_summary', {})),
            'implemented_targets': ['yolo_score_raw', 'yolo_raw_best_score'],
            'yolo_score_raw_summary': {
                'mean_target': float(np.mean(yolo_raw_values)),
                'min_target': float(np.min(yolo_raw_values)),
                'max_target': float(np.max(yolo_raw_values)),
                'uses_temperature_scaling': False,
            },
            'notes': [
                'Paper GP fitting uses the raw YOLO detector score only.',
                'Missed detections remain encoded as zero-valued raw scores when emitted by the extractor.',
                'gp_targets_xy_aggregated.csv averages raw scores over sampled headings at each (x, y) position.',
            ],
        },
    )
    print(f'Wrote GP targets to {gp_targets_path}')
    print(f'Rows: {len(out_rows)}')
    print(f'Wrote heading-aggregated GP targets to {aggregated_gp_targets_path}')
    print(f'Aggregated rows: {len(aggregated_rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
