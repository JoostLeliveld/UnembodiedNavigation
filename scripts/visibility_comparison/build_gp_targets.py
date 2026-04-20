#!/usr/bin/env python3
"""Build the planner-facing scalar GP target table from perception targets."""

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
    parse_bool01,
    parse_float,
    read_csv_rows,
    write_csv,
    write_manifest,
)
from yolo_calibration_utils import apply_temperature_scaling


def _load_area_reference(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _lookup_area_ref(area_ref_payload: dict[str, np.ndarray] | None, *, x: float, y: float) -> float:
    if not area_ref_payload:
        return math.nan
    available = np.asarray(area_ref_payload.get('available', np.asarray([0], dtype=np.int32))).reshape(-1)
    if available.size == 0 or int(available[0]) != 1:
        return math.nan
    xs = np.asarray(area_ref_payload.get('xs', np.asarray([], dtype=float)), dtype=float).reshape(-1)
    ys = np.asarray(area_ref_payload.get('ys', np.asarray([], dtype=float)), dtype=float).reshape(-1)
    area_map = np.asarray(area_ref_payload.get('A_ref_map', np.asarray([], dtype=float)), dtype=float)
    if xs.size == 0 or ys.size == 0 or area_map.size == 0:
        return math.nan
    ix = int(np.argmin(np.abs(xs - float(x))))
    iy = int(np.argmin(np.abs(ys - float(y))))
    if iy < 0 or iy >= area_map.shape[0] or ix < 0 or ix >= area_map.shape[1]:
        return math.nan
    ref = float(area_map[iy, ix])
    if math.isfinite(ref) and ref > 0.0:
        return ref
    global_ref = np.asarray(area_ref_payload.get('global_ref', np.asarray([math.nan], dtype=float)), dtype=float).reshape(-1)
    if global_ref.size > 0:
        ref = float(global_ref[0])
        if math.isfinite(ref) and ref > 0.0:
            return ref
    return math.nan


def _load_calibration_payload(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(
            f'YOLO calibrated targets require a calibration artifact. Missing: {path}. '
            'Run plot_yolo_calibration.py first.'
        )
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Failed to read calibration artifact {path}: {exc}') from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f'Calibration artifact is not a JSON object: {path}')
    if str(payload.get('calibration_type', '')).strip() != 'temperature_scaling':
        raise RuntimeError(f'Unsupported calibration artifact type in {path}: {payload.get("calibration_type")!r}')
    return payload


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description='Create gp_targets.csv from the shared perception target contract.')
    parser.add_argument('--perception-targets', default=str(CURRENT_TARGETS_DIR / 'perception_targets.csv'))
    parser.add_argument('--area-reference', default=str(CURRENT_TARGETS_DIR / 'area_reference.npz'))
    parser.add_argument('--yolo-calibration', default=str(CURRENT_TARGETS_DIR / 'yolo_score_calibration.json'))
    parser.add_argument('--out', default=str(CURRENT_TARGETS_DIR))
    parser.add_argument('--r-min', type=float, default=0.05)
    parser.add_argument('--r-ok', type=float, default=0.35)
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
    area_reference_path = Path(args.area_reference).expanduser().resolve()
    area_ref_payload = _load_area_reference(area_reference_path)
    calibration_path = Path(args.yolo_calibration).expanduser().resolve()

    yolo_raw_present = any(str(row.get('yolo_score_raw', '')).strip() != '' for row in rows)
    calibration_payload = _load_calibration_payload(calibration_path) if yolo_raw_present else {}
    calibration_temperature = float(calibration_payload.get('temperature', 1.0)) if calibration_payload else 1.0

    out_rows = []
    oracle_values: list[float] = []
    red_values: list[float] = []
    red_area_corrected_values: list[float] = []
    yolo_binary_values: list[float] = []
    yolo_raw_values: list[float] = []
    yolo_calibrated_values: list[float] = []

    for row in rows:
        x = parse_float(row.get('x', ''), math.nan)
        y = parse_float(row.get('y', ''), math.nan)
        oracle_visible = int(max(0, min(1, parse_bool01(row.get('oracle_visible', '0')))))
        oracle_values.append(float(oracle_visible))

        red_detected_raw = str(row.get('red_detected', '')).strip()
        red_detected = int(max(0, min(1, parse_bool01(red_detected_raw))))
        if red_detected_raw != '':
            red_values.append(float(red_detected))

        red_area = parse_float(row.get('red_area', ''), math.nan)
        red_area_corrected = math.nan
        if red_detected_raw != '':
            area_ref = _lookup_area_ref(area_ref_payload, x=x, y=y) if math.isfinite(x) and math.isfinite(y) else math.nan
            measured_area = 0.0 if not math.isfinite(red_area) else max(float(red_area), 0.0)
            ratio = measured_area / area_ref if math.isfinite(area_ref) and area_ref > 0.0 else math.nan
            if math.isfinite(ratio):
                red_area_corrected = float(
                    np.clip(
                        (ratio - float(args.r_min)) / max(float(args.r_ok) - float(args.r_min), 1e-9),
                        0.0,
                        1.0,
                    )
                )
                red_area_corrected_values.append(float(red_area_corrected))

        yolo_raw = parse_float(row.get('yolo_score_raw', ''), math.nan)
        yolo_selected = parse_float(row.get('yolo_score_selected', ''), math.nan)
        yolo_detected_raw = str(row.get('yolo_detected_after_threshold', '')).strip()
        yolo_detected = int(max(0, min(1, parse_bool01(yolo_detected_raw))))
        if yolo_detected_raw != '':
            yolo_binary_values.append(float(yolo_detected))

        if math.isfinite(yolo_raw):
            yolo_raw = float(np.clip(yolo_raw, 0.0, 1.0))
            yolo_raw_values.append(float(yolo_raw))
            yolo_calibrated = float(apply_temperature_scaling([yolo_raw], calibration_temperature)[0])
            yolo_calibrated_values.append(float(yolo_calibrated))
        else:
            yolo_calibrated = math.nan

        out_rows.append({
            'sample_id': str(row.get('sample_id', '')).strip(),
            'x': str(row.get('x', '')).strip(),
            'y': str(row.get('y', '')).strip(),
            'red_binary': '' if red_detected_raw == '' else str(int(red_detected)),
            'red_area_corrected': '' if not math.isfinite(red_area_corrected) else f'{float(red_area_corrected):.8f}',
            'yolo_binary': '' if yolo_detected_raw == '' else str(int(yolo_detected)),
            'yolo_score_raw': '' if not math.isfinite(yolo_raw) else f'{float(yolo_raw):.8f}',
            'yolo_score_calibrated': '' if not math.isfinite(yolo_calibrated) else f'{float(yolo_calibrated):.8f}',
            'oracle_visibility': str(int(oracle_visible)),
        })

    gp_targets_path = output_dir / 'gp_targets.csv'
    write_csv(gp_targets_path, GP_TARGET_COLUMNS, out_rows)
    write_csv(
        output_dir / 'target_summary.csv',
        ('method_id', 'available', 'row_count', 'mean_target', 'min_target', 'max_target'),
        [
            {
                'method_id': 'red_binary',
                'available': '1' if red_values else '0',
                'row_count': str(int(len(red_values))),
                'mean_target': '' if not red_values else f'{float(sum(red_values) / len(red_values)):.8f}',
                'min_target': '' if not red_values else f'{float(min(red_values)):.8f}',
                'max_target': '' if not red_values else f'{float(max(red_values)):.8f}',
            },
            {
                'method_id': 'oracle_visibility',
                'available': '1',
                'row_count': str(int(len(oracle_values))),
                'mean_target': '' if not oracle_values else f'{float(sum(oracle_values) / len(oracle_values)):.8f}',
                'min_target': '' if not oracle_values else f'{float(min(oracle_values)):.8f}',
                'max_target': '' if not oracle_values else f'{float(max(oracle_values)):.8f}',
            },
            {
                'method_id': 'red_area_corrected',
                'available': '1' if red_area_corrected_values else '0',
                'row_count': str(int(len(red_area_corrected_values))),
                'mean_target': '' if not red_area_corrected_values else f'{float(sum(red_area_corrected_values) / len(red_area_corrected_values)):.8f}',
                'min_target': '' if not red_area_corrected_values else f'{float(min(red_area_corrected_values)):.8f}',
                'max_target': '' if not red_area_corrected_values else f'{float(max(red_area_corrected_values)):.8f}',
            },
            {
                'method_id': 'yolo_binary',
                'available': '1' if yolo_binary_values else '0',
                'row_count': str(int(len(yolo_binary_values))),
                'mean_target': '' if not yolo_binary_values else f'{float(sum(yolo_binary_values) / len(yolo_binary_values)):.8f}',
                'min_target': '' if not yolo_binary_values else f'{float(min(yolo_binary_values)):.8f}',
                'max_target': '' if not yolo_binary_values else f'{float(max(yolo_binary_values)):.8f}',
            },
            {
                'method_id': 'yolo_score_raw',
                'available': '1' if yolo_raw_values else '0',
                'row_count': str(int(len(yolo_raw_values))),
                'mean_target': '' if not yolo_raw_values else f'{float(sum(yolo_raw_values) / len(yolo_raw_values)):.8f}',
                'min_target': '' if not yolo_raw_values else f'{float(min(yolo_raw_values)):.8f}',
                'max_target': '' if not yolo_raw_values else f'{float(max(yolo_raw_values)):.8f}',
            },
            {
                'method_id': 'yolo_score_calibrated',
                'available': '1' if yolo_calibrated_values else '0',
                'row_count': str(int(len(yolo_calibrated_values))),
                'mean_target': '' if not yolo_calibrated_values else f'{float(sum(yolo_calibrated_values) / len(yolo_calibrated_values)):.8f}',
                'min_target': '' if not yolo_calibrated_values else f'{float(min(yolo_calibrated_values)):.8f}',
                'max_target': '' if not yolo_calibrated_values else f'{float(max(yolo_calibrated_values)):.8f}',
            },
        ],
    )
    write_manifest(output_dir / 'target_manifest.json', {
        'perception_targets': str(targets_path),
        'area_reference_path': str(area_reference_path),
        'yolo_calibration_path': str(calibration_path) if calibration_payload else '',
        'gp_targets_path': str(gp_targets_path),
        'row_count': int(len(out_rows)),
        'capture_metadata': dict(extractor_manifest.get('capture_metadata', {})),
        'yolo_summary': dict(extractor_manifest.get('yolo_summary', {})),
        'implemented_targets': ['red_binary', 'oracle_visibility']
        + (['red_area_corrected'] if red_area_corrected_values else [])
        + (['yolo_binary', 'yolo_score_raw', 'yolo_score_calibrated'] if (yolo_binary_values or yolo_raw_values or yolo_calibrated_values) else []),
        'deferred_targets': ([] if red_area_corrected_values else ['red_area_corrected']),
        'red_binary_summary': {
            'mean_target': float(sum(red_values) / len(red_values)) if red_values else math.nan,
            'detected_count': int(sum(1 for value in red_values if value >= 0.5)),
            'undetected_count': int(sum(1 for value in red_values if value < 0.5)),
        },
        'red_area_corrected_summary': {
            'mean_target': float(sum(red_area_corrected_values) / len(red_area_corrected_values)) if red_area_corrected_values else math.nan,
            'min_target': float(min(red_area_corrected_values)) if red_area_corrected_values else math.nan,
            'max_target': float(max(red_area_corrected_values)) if red_area_corrected_values else math.nan,
            'r_min': float(args.r_min),
            'r_ok': float(args.r_ok),
        },
        'oracle_summary': {
            'mean_target': float(sum(oracle_values) / len(oracle_values)) if oracle_values else math.nan,
            'visible_count': int(sum(1 for value in oracle_values if value >= 0.5)),
            'hidden_count': int(sum(1 for value in oracle_values if value < 0.5)),
        },
        'yolo_binary_summary': {
            'mean_target': float(sum(yolo_binary_values) / len(yolo_binary_values)) if yolo_binary_values else math.nan,
            'detected_count': int(sum(1 for value in yolo_binary_values if value >= 0.5)),
            'undetected_count': int(sum(1 for value in yolo_binary_values if value < 0.5)),
        },
        'yolo_score_raw_summary': {
            'mean_target': float(sum(yolo_raw_values) / len(yolo_raw_values)) if yolo_raw_values else math.nan,
            'min_target': float(min(yolo_raw_values)) if yolo_raw_values else math.nan,
            'max_target': float(max(yolo_raw_values)) if yolo_raw_values else math.nan,
            'uses_temperature_scaling': False,
        },
        'yolo_score_calibrated_summary': {
            'mean_target': float(sum(yolo_calibrated_values) / len(yolo_calibrated_values)) if yolo_calibrated_values else math.nan,
            'min_target': float(min(yolo_calibrated_values)) if yolo_calibrated_values else math.nan,
            'max_target': float(max(yolo_calibrated_values)) if yolo_calibrated_values else math.nan,
            'uses_temperature_scaling': bool(calibration_payload),
            'temperature': calibration_temperature if calibration_payload else math.nan,
        },
        'notes': [
            'Red binary, red-area-corrected, oracle/reference visibility, YOLO binary, YOLO raw score, and YOLO calibrated score are populated when their upstream artifacts exist.',
            'YOLO calibrated targets are built from the temperature-scaling artifact generated by plot_yolo_calibration.py.',
            'Missed YOLO detections remain in the target table as zero-valued raw/calibrated scores rather than being dropped.',
        ],
    })
    print(f'Wrote GP targets to {gp_targets_path}')
    print(f'Rows: {len(out_rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
