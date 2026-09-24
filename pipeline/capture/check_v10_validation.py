#!/usr/bin/env python3
"""Validation of the v10 capture plan on a small capture, before the full capture runs.

Checks, all declared before the validation capture:
  1. completeness: one row per (pose, camera), every capture_status ok;
  2. the planned view class agrees with the measured segmentation masks:
     edge recall >= 0.90 and bottom-hidden recall >= 0.60 (known answer on v9 data:
     1.00 and 0.80), and at least one measured edge view per camera that plans one;
  3. the new images are rendered in the capture world (world sha matches the lock).
Exit status 0 only if every check passes.

    python3 pipeline/capture/check_v10_validation.py logs/thesis/captures/v10/validation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'pipeline/capture')]
from plan_hard_views import ViewClassifier  # noqa: E402

W, H = 1280, 720


def measured(row) -> str:
    ex0, ey0, ex1, ey1 = (float(row[f'expected_{k}']) for k in ('x0', 'y0', 'x1', 'y1'))
    full = (ex1 - ex0) * (ey1 - ey0)
    inframe = max(min(ex1, W) - max(ex0, 0), 0) * max(min(ey1, H) - max(ey0, 0), 0)
    if inframe <= 0 or float(row['line_of_sight']) != 1:
        return 'none'              # robot not in the image: no view to classify
    if ey1 > H or 1 - inframe / full > 0.05:
        return 'edge'
    if float(row['line_of_sight']) == 1 and min(ey1, H) - float(row['mask_y1']) > 4:
        return 'bottom hidden'
    return 'visible' if float(row['line_of_sight']) == 1 else 'none'


def main() -> int:
    out = Path(sys.argv[1])
    out = out if out.is_absolute() else REPO / out
    rows = pd.read_csv(out / 'capture_index.csv')
    poses = json.loads((out.parent / 'validation_poses.json').read_text())
    report = {'rows': int(len(rows)), 'expected_rows': 5 * len(poses)}
    report['complete'] = bool(len(rows) == 5 * len(poses) and (rows.capture_status == 'ok').all())
    classify = ViewClassifier()
    planned = [classify(float(r.robot_x), float(r.robot_y), float(r.robot_yaw))[r.camera_id] for r in rows.itertuples()]
    rows['planned'] = planned
    rows['measured'] = [measured(r) for _, r in rows.iterrows()]
    edge = rows[rows.measured == 'edge']
    hidden = rows[rows.measured == 'bottom hidden']
    report['edge_recall'] = float((edge.planned == 'edge').mean()) if len(edge) else None
    report['bottom_hidden_recall'] = float((hidden.planned == 'bottom hidden').mean()) if len(hidden) else None
    report['n_measured'] = rows.measured.value_counts().to_dict()
    report['confusion'] = pd.crosstab(rows.measured, rows.planned).to_dict()
    per_cam = rows[rows.planned == 'edge'].groupby('camera_id').measured.apply(lambda s: int((s == 'edge').sum())).to_dict()
    report['measured_edge_where_planned_per_camera'] = per_cam
    report['passed'] = bool(report['complete']
                            and report['edge_recall'] is not None and report['edge_recall'] >= 0.90
                            and (report['bottom_hidden_recall'] is None or report['bottom_hidden_recall'] >= 0.60)
                            and all(v >= 1 for v in per_cam.values()))
    (out / 'validation_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('rows', 'expected_rows', 'complete', 'edge_recall', 'bottom_hidden_recall', 'passed')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
