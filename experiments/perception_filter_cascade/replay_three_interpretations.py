#!/usr/bin/env python3
"""Three interpretations replayed on NEW recorded drives, in the actual-drive layout.

The question is narrow: on camera readings none of them has seen, how do the shape model,
the learned neural correction, and the quality-gated cascade compare as localization
measurements?

Every column replays the SAME deduplicated readings from the SAME recorded drives, so a
difference between columns is interpretation and nothing else. None of them steered the
robot, altered fusion, or changed the recorded trajectory.

Columns:

    raw         the box bottom projected to the floor. The reference.
    neural      the learned neural correction from the frozen characterization tiles.
    hull_fair   the shape model with NO truth: position and heading solved from the box.
    gated       the quality-gated cascade -- the same reading as `neural`, but each one
                carries a covariance set by an instantaneous usability estimate, and a
                reading the estimate calls unusable is REFUSED rather than inflated.
    hull        the shape model started AT the true pose and true heading. An oracle and a
                bound, never a method; kept greyed so the reader can see what truth is worth.

The gated column is the only one that changes the number of admitted readings, so its
availability is reported beside its error. Refusing readings can only help an error metric,
which is why it is never quoted without the availability it cost.

Feature availability. Drive logs record the observed and PREDICTED box size, confidence,
and range -- so the occlusion signature (a box smaller than geometry predicts) is
computable at runtime. They do NOT record box corners, so image-border distances are not
available on a drive and are deliberately absent from the gate here, unlike the static
captures where they were.
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for rel in ("experiments/camera_observation_characterization",
            "experiments/fusion_on_fixed_routes",
            "experiments/deck_figures",
            "experiments/perception_filter_cascade"):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import replay_learned_on_actual_run as L  # noqa: E402
import replay_fair_hull_on_actual_run as F  # noqa: E402

FOLDER = "01_three_interpretations_on_new_drives"

COLUMNS = (
    ("raw", "Raw box → floor", "the box and its camera", True),
    ("neural", "Learned neural correction", "the box and its camera", True),
    ("hull_fair", "Robot-shape model, heading solved", "the box and its camera", True),
    ("gated", "Quality-gated cascade", "the box, its camera, and a usability estimate", True),
    ("hull", "Robot-shape model started AT the answer",
     "the TRUE position and heading", False),
)

# A reading whose localization error exceeds this is treated as unusable: the occluded
# regime carries a large MEAN error, which no covariance honestly describes.
USABLE_CM = 20.0


def drive_features(reading: dict) -> dict[str, float] | None:
    """Runtime-observable quality features from one drive reading.

    `size_ratio` is the occlusion signature: the detected box divided by the box the
    geometry predicts for the current belief. A robot half hidden behind stock produces a
    box far smaller than predicted, and this is available live because the prediction comes
    from the filter, not from truth.
    """
    try:
        width = float(reading["bbox_w_px"])
        height = float(reading["bbox_h_px"])
        pred_w = float(reading["pred_w_px"])
        pred_h = float(reading["pred_h_px"])
        confidence = float(reading["conf"])
        range_m = float(reading["range_m"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in
               (width, height, pred_w, pred_h, confidence, range_m)):
        return None
    predicted_area = pred_w * pred_h
    if predicted_area <= 0.0:
        return None
    return {
        'confidence': confidence,
        'size_ratio': (width * height) / predicted_area,
        'aspect': width / max(height, 1e-6),
        'range_m': range_m,
        'log_area': math.log(max(width * height, 1.0)),
    }


class UsabilityModel:
    """P(usable | runtime features), logistic regression, fitted on held-out drives.

    Fitted on the drives NOT being scored, so the gate never sees the readings it judges.
    """

    FEATURES = ('confidence', 'size_ratio', 'aspect', 'range_m', 'log_area')

    def __init__(self) -> None:
        self.weights = [0.0] * (len(self.FEATURES) + 1)
        self.mu = [0.0] * len(self.FEATURES)
        self.sigma = [1.0] * len(self.FEATURES)

    def _design(self, features: dict[str, float]) -> list[float]:
        raw = [features[name] for name in self.FEATURES]
        return [1.0] + [(value - mu) / sigma
                        for value, mu, sigma in zip(raw, self.mu, self.sigma)]

    def fit(self, samples: list[tuple[dict[str, float], int]], *, steps: int = 4000,
            learning_rate: float = 0.3, l2: float = 1e-3) -> None:
        if not samples or len({label for _, label in samples}) < 2:
            return
        columns = [[features[name] for features, _ in samples] for name in self.FEATURES]
        self.mu = [st.mean(column) for column in columns]
        self.sigma = [max(st.pstdev(column), 1e-6) for column in columns]
        design = [self._design(features) for features, _ in samples]
        labels = [label for _, label in samples]
        dim = len(self.weights)
        self.weights = [0.0] * dim
        for _ in range(steps):
            gradient = [0.0] * dim
            for row, label in zip(design, labels):
                z = sum(w * x for w, x in zip(self.weights, row))
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                error = p - label
                for index, value in enumerate(row):
                    gradient[index] += error * value
            scale = learning_rate / len(design)
            for index in range(dim):
                penalty = l2 * self.weights[index] if index else 0.0
                self.weights[index] -= scale * (gradient[index] + penalty)

    def predict(self, features: dict[str, float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, self._design(features)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def auc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        stop = index
        while stop < len(order) and scores[order[stop]] == scores[order[index]]:
            stop += 1
        average = (index + stop - 1) / 2.0 + 1.0
        for position in range(index, stop):
            ranks[order[position]] = average
        index = stop
    rank_sum = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def load_drive(run: Path) -> dict | None:
    """One recorded drive: its readings during motion, scored at their own capture stamps."""
    import aligned as A

    manifest_path = run / 'run_manifest.json'
    summary_path = run / 'run_summary.json'
    if not manifest_path.exists() or not summary_path.exists():
        return None
    run_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    run_summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if run_summary.get('first_cmd_stamp') is None or run_summary.get('stop_stamp') is None:
        return None

    loaded = A.readings(run, admitted_only=False, dedupe=True, require_capture_time=True)
    first_cmd = float(run_summary['first_cmd_stamp'])
    stop_stamp = float(run_summary['stop_stamp'])
    readings = [item for item in loaded if first_cmd <= item['obs_stamp'] <= stop_stamp]
    if not readings:
        return None
    return {
        'run': run,
        'manifest': run_manifest,
        'summary': run_summary,
        'readings': readings,
        'elapsed': np.asarray([item['obs_stamp'] - first_cmd for item in readings]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=L.DEFAULT_CAPTURE)
    parser.add_argument('--campaign', type=Path, required=True,
                        help='Campaign log root holding the NEW recorded drives.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tau-reject', type=float, default=0.5,
                        help='Refuse a reading whose usability estimate falls below this.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    import aligned as A  # noqa: F401  (imported for its side-effect-free loader)
    import plot_real_run_bias as R

    capture = args.capture.expanduser().resolve()
    output = args.out.expanduser().resolve() / FOLDER
    output.mkdir(parents=True, exist_ok=True)

    runs = sorted(path.parent for path in
                  args.campaign.expanduser().resolve().glob('*/*/*/experiment_*/run_summary.json'))
    if not runs:
        raise SystemExit(f'no completed drives under {args.campaign}')

    drives = [drive for drive in (load_drive(run) for run in runs) if drive]
    if not drives:
        raise SystemExit('no drive produced scoreable readings during motion')
    print(f'{len(drives)} scoreable drives of {len(runs)} completed runs')

    capture_manifest = json.loads((capture / 'capture_manifest.json').read_text())
    geometry = L.camera_geometry(capture_manifest)
    optics = L.camera_models(capture_manifest)
    project = L.projectors(capture_manifest)
    rows_all = list(_csv.DictReader(
        (capture / 'bias_update_interpretations.csv').open(encoding='utf-8')))
    train = [row for row in rows_all
             if row['split'] == 'train' and row['raw_valid'] == '1']
    models = L.fit_models(train, geometry, args.seed)

    methods = tuple(method for method, *_ in COLUMNS)

    # Per-drive replay of every column.
    for drive in drives:
        rows = L.replay_rows(drive['readings'], project)
        truths = np.stack([reading['truth'] for reading in drive['readings']])
        corrections = L.corrections_for_rows(rows, models, geometry)
        oracle_hull = L.hull_estimates(drive['readings'], rows, optics)
        fair_hull, solved_yaw = F.fair_hull_estimates(drive['readings'], rows, optics)

        saved = L.METHODS
        L.METHODS = tuple(m for m in methods if m != 'gated')
        try:
            evaluated = L.evaluate(rows, truths, corrections, geometry,
                                   direct={'hull': oracle_hull, 'hull_fair': fair_hull})
        finally:
            L.METHODS = saved
        drive['rows'] = rows
        drive['evaluated'] = evaluated
        drive['solved_yaw'] = solved_yaw
        drive['features'] = [drive_features(reading) for reading in drive['readings']]
        drive['usable'] = [
            1 if magnitude * 100.0 <= USABLE_CM else 0
            for magnitude in evaluated['neural']['magnitude']
        ]

    # The gate is fitted leave-one-drive-out, so it never judges a reading it trained on.
    gate_scores: list[float] = []
    gate_labels: list[int] = []
    for held in drives:
        samples = [
            (features, label)
            for other in drives if other is not held
            for features, label in zip(other['features'], other['usable'])
            if features is not None
        ]
        model = UsabilityModel()
        model.fit(samples)
        quality = [model.predict(features) if features is not None else 1.0
                   for features in held['features']]
        held['quality'] = quality
        for value, label, features in zip(quality, held['usable'], held['features']):
            if features is not None:
                gate_scores.append(value)
                gate_labels.append(label)
        # The gated column keeps the neural reading and refuses the ones the gate rejects.
        admitted = np.asarray([value >= args.tau_reject for value in quality])
        magnitude = np.asarray(held['evaluated']['neural']['magnitude'], dtype=float).copy()
        along = np.asarray(held['evaluated']['neural']['along'], dtype=float).copy()
        across = np.asarray(held['evaluated']['neural']['across'], dtype=float).copy()
        error = np.asarray(held['evaluated']['neural']['error'], dtype=float).copy()
        magnitude[~admitted] = np.nan
        along[~admitted] = np.nan
        across[~admitted] = np.nan
        error[~admitted] = np.nan
        held['evaluated']['gated'] = {
            'magnitude': magnitude, 'along': along, 'across': across, 'error': error,
        }
        held['admitted'] = admitted

    # Pooled scores across every drive, per column.
    pooled: dict[str, dict[str, float]] = {}
    for method in methods:
        values = np.concatenate([
            np.asarray(drive['evaluated'][method]['magnitude'], dtype=float)
            for drive in drives])
        finite = values[np.isfinite(values)]
        pooled[method] = L.summary(finite)
        pooled[method]['n'] = int(finite.size)
        total = sum(len(drive['readings']) for drive in drives)
        pooled[method]['availability_pct'] = 100.0 * finite.size / max(total, 1)

    gate_auc = auc(gate_scores, gate_labels) if gate_scores else float('nan')

    payload = {
        'schema': 'perception_filter_cascade_three_interpretations.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'campaign': str(args.campaign),
        'capture_for_training': str(capture),
        'drives': [{
            'run': str(drive['run']),
            'task': drive['manifest'].get('task'),
            'seed': drive['manifest'].get('seed'),
            'readings': len(drive['readings']),
            'completion': drive['summary'].get('completion_reason'),
        } for drive in drives],
        'usable_threshold_cm': USABLE_CM,
        'tau_reject': args.tau_reject,
        'gate_auc_leave_one_drive_out': gate_auc,
        'columns': {method: pooled[method] for method in methods},
        'notes': [
            'every column replays the same readings from the same drives',
            'hull is an oracle: it linearises around the true pose and true heading',
            'the gated column refuses readings, so its availability is reported with it',
            'drive logs carry no box corners, so no image-border feature is used',
        ],
    }
    (output / 'three_interpretations.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    print()
    print(f'{"column":36s} {"n":>6s} {"avail%":>7s} {"median_cm":>10s} {"p90_cm":>8s} '
          f'{"rmse_cm":>8s}')
    print('-' * 80)
    for method, label, given, fair in COLUMNS:
        stats = pooled[method]
        mark = '' if fair else '   <- oracle, not a method'
        print(f'{label[:36]:36s} {stats["n"]:6d} {stats["availability_pct"]:7.1f} '
              f'{stats["median_cm"]:10.2f} {stats.get("p90_cm", float("nan")):8.2f} '
              f'{stats.get("rmse_cm", float("nan")):8.2f}{mark}')
    print()
    print(f'usability gate, leave-one-drive-out AUC: {gate_auc:.3f}')
    print()
    print(f'wrote {output / "three_interpretations.json"}')

    # The per-drive sheet uses the existing actual-drive layout, one sheet per drive.
    for drive in drives:
        columns = []
        for method, label, given, fair in COLUMNS:
            magnitude = np.asarray(drive['evaluated'][method]['magnitude'], dtype=float)
            keep = np.isfinite(magnitude)
            replayed = []
            for index, source in enumerate(drive['readings']):
                if not keep[index]:
                    continue
                item = dict(source)
                item['error'] = drive['evaluated'][method]['error'][index]
                item['error_cm'] = float(magnitude[index] * 100.0)
                item['magnitude_m'] = float(magnitude[index])
                item['along_m'] = float(drive['evaluated'][method]['along'][index])
                item['across_m'] = float(drive['evaluated'][method]['across'][index])
                replayed.append(item)
            if not replayed:
                continue
            first_cmd = float(drive['summary']['first_cmd_stamp'])
            times = np.asarray([item['obs_stamp'] - first_cmd for item in replayed])
            columns.append({
                'run': drive['run'],
                'manifest': drive['manifest'],
                'summary': drive['summary'],
                'readings': replayed,
                'times': times,
                'errors_cm': np.asarray([item['error_cm'] for item in replayed]),
                'duration_s': float(drive['summary']['elapsed_after_first_cmd_s']),
                'collision_s': (
                    float(drive['summary']['first_crash_stamp']) - first_cmd
                    if drive['summary'].get('collision_any')
                    and drive['summary'].get('first_crash_stamp') is not None else None),
                'spans': R.blind_spans(times),
                'route': np.asarray(
                    json.loads(drive['manifest']['preselected_route_json']), dtype=float),
                'truth_xy': None,
                'observation_model': method,
                'run_id': str(drive['manifest'].get('run_id', drive['run'].name)),
                'panel_id': method,
                'arm': 'offline_replay',
                'context_line': f'{label}\ngiven: {given}',
                'completion': str(drive['summary'].get('completion_reason', 'unknown')),
                'per_camera': {camera: sum(item['camera'] == camera for item in replayed)
                               for camera in 'ABCDE'},
            })
        drive['columns'] = columns

    print(f'built per-drive column sets for {len(drives)} drives')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
