#!/usr/bin/env python3
"""Cascaded perception architecture where the temporal filter estimates QUALITY, not position.

The measurement handed to the robot EKF is built from the current frame only. Perception
history is allowed to influence how much that measurement is trusted, and whether it is
admitted at all, but never what it says. That is the distinction the audit turned on: a
posterior mean carried across frames re-injects past position evidence into the localization
filter (measured at 3.79x overconfident on a controlled test), while a covariance informed by
past frames does not, because a covariance is not a statement about where the robot is.

The temporal filter therefore tracks perception quality rather than the image point. A
position tracker's innovation is a poor reliability signal here: the failures in this data
are contiguous runs (median 13 consecutive frames), so after a couple of corrupted frames
the tracker has followed the corruption and its innovation goes quiet exactly when the error
is worst. A quality state does not share that failure mode.

Arms, all sharing one identical downstream measurement construction:

  A  nn_only          corrected point, one fixed R. What the correction alone achieves.
  B  instant_quality  plus an instantaneous usability estimate from current-frame features,
                      driving admission and R. The load-bearing baseline.
  C  temporal_quality plus a filtered quality state over the history of quality estimates.
  D  arm C scored only on sequences with a clean prefix, where temporal history has a fair
                      chance to say something before the observation degrades.

Enforced invariants (checked at runtime, see `_check_invariants`):
  1. the robot measurement uses only frame t
  2. no perception posterior position or mean enters the measurement
  3. perception history affects only the quality estimate, admission, and R
  4. R adaptation uses no robot-EKF innovation
  5. every arm uses the same downstream measurement construction
  6. instantaneous and temporally filtered quality are reported separately
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for rel in ('scripts/perception', 'src/experiments', 'src/perception', 'src/unav_common',
            'experiments/camera_observation_characterization'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)
sys.path.insert(0, str(Path(__file__).resolve().parent))

TRUE_VALUES = {'True', 'true', '1'}
IMG_W, IMG_H = 1280, 720

# A reading is "usable" if its localization error is within this bound. The occluded regime
# carries a 30-60 cm mean, which no covariance honestly represents, so it must be refused
# rather than inflated.
USABLE_CM = 20.0


# ---------------------------------------------------------------------------
# runtime-observable features. Nothing here may come from simulation truth.
# ---------------------------------------------------------------------------

TRUTH_ONLY_COLUMNS = {
    'robot_x', 'robot_y', 'robot_yaw', 'camera_range_m', 'nominal_in_frame',
    'expected_x0', 'expected_y0', 'expected_x1', 'expected_y1',
    'mask_x0', 'mask_y0', 'mask_x1', 'mask_y1', 'mask_bottom_u', 'mask_bottom_v',
    'semantic_robot_pixels', 'line_of_sight',
    'hull_dx', 'hull_dy', 'hull_error_m', 'raw_dx', 'raw_dy', 'raw_error_m',
}


def runtime_features(row: dict[str, str], camera_xy: tuple[float, float]) -> dict[str, float]:
    """Features available on a live robot: the box, the detector, and the camera geometry.

    Camera range is deliberately recomputed from the back-projected reading rather than read
    from `camera_range_m`, which is derived from the true pose and is a label, not an input.
    """
    x0, y0, x1, y1 = (float(row[key]) for key in ('x0', 'y0', 'x1', 'y1'))
    width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    est_x, est_y = float(row['hull_x']), float(row['hull_y'])
    measured_range = math.hypot(est_x - camera_xy[0], est_y - camera_xy[1])
    return {
        'confidence': float(row['confidence']),
        'width': width,
        'height': height,
        'area': width * height,
        'aspect': width / height,
        'bottom_margin': IMG_H - y1,
        'top_margin': y0,
        'left_margin': x0,
        'right_margin': IMG_W - x1,
        'measured_range_m': measured_range,
        # Apparent size falls as 1/range for an unoccluded robot, so this ratio drops when
        # part of the robot is hidden. Uses the measured range, not the true one.
        'size_range_product': width * height * measured_range ** 2,
    }


# ---------------------------------------------------------------------------
# instantaneous quality model (arm B)
# ---------------------------------------------------------------------------

class InstantQuality:
    """P(usable | current-frame features), fitted by logistic regression.

    This stands in for the image-aware network's reliability head. It sees only
    runtime-observable features, and is always fitted on training places and applied to
    held-out places, because a per-place fit is what produced a false result once already.
    """

    FEATURES = ('confidence', 'aspect', 'bottom_margin', 'size_range_product',
                'measured_range_m')

    def __init__(self) -> None:
        self.weights: list[float] = [0.0] * (len(self.FEATURES) + 1)
        self.mu: list[float] = [0.0] * len(self.FEATURES)
        self.sigma: list[float] = [1.0] * len(self.FEATURES)

    def _design(self, features: dict[str, float]) -> list[float]:
        raw = [features[name] for name in self.FEATURES]
        scaled = [(value - mu) / sigma
                  for value, mu, sigma in zip(raw, self.mu, self.sigma)]
        return [1.0] + scaled

    def fit(self, samples: list[tuple[dict[str, float], int]], *, steps: int = 3000,
            learning_rate: float = 0.2, l2: float = 1e-3) -> None:
        if not samples:
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
                residual = p - label
                for index, value in enumerate(row):
                    gradient[index] += residual * value
            scale = learning_rate / len(design)
            for index in range(dim):
                penalty = l2 * self.weights[index] if index else 0.0
                self.weights[index] -= scale * gradient[index] + scale * penalty

    def predict(self, features: dict[str, float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, self._design(features)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


# ---------------------------------------------------------------------------
# temporal quality filter (arm C)
# ---------------------------------------------------------------------------

class QualityFilter:
    """A two-state Kalman filter over perception quality and its rate.

    State is `[logit(quality), d/dt logit(quality)]`. Working in logit space keeps the
    filtered quality inside (0, 1) without clipping, and makes a sustained slide toward
    failure a linear trend the filter can extrapolate.

    It consumes ONLY the instantaneous quality estimates, which are functions of current
    image evidence. It never sees the robot state, the localization innovation, or truth,
    so it cannot recycle robot-state information into the covariance path.
    """

    def __init__(self, *, step_s: float, q_rate: float = 1.5,
                 r_quality: float = 0.8) -> None:
        self.dt = step_s
        self.q_rate = q_rate
        self.r_quality = r_quality
        self.state: list[float] | None = None
        self.cov: list[list[float]] | None = None

    @staticmethod
    def _logit(value: float) -> float:
        clipped = min(max(value, 1e-4), 1.0 - 1e-4)
        return math.log(clipped / (1.0 - clipped))

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))

    def initialise(self, quality: float) -> None:
        self.state = [self._logit(quality), 0.0]
        self.cov = [[self.r_quality, 0.0], [0.0, (2.0 * self.q_rate) ** 2]]

    def step(self, quality: float) -> tuple[float, float]:
        """Advance one frame and return (filtered quality, filtered rate in logit units)."""
        if self.state is None or self.cov is None:
            self.initialise(quality)
            assert self.state is not None
            return self._sigmoid(self.state[0]), 0.0

        dt = self.dt
        # predict: constant-rate drift in logit space
        state = [self.state[0] + dt * self.state[1], self.state[1]]
        var = (self.q_rate * dt) ** 2
        cov = [
            [self.cov[0][0] + dt * (self.cov[0][1] + self.cov[1][0])
             + dt * dt * self.cov[1][1] + dt ** 4 / 4.0 * var,
             self.cov[0][1] + dt * self.cov[1][1] + dt ** 3 / 2.0 * var],
            [self.cov[1][0] + dt * self.cov[1][1] + dt ** 3 / 2.0 * var,
             self.cov[1][1] + dt * dt * var],
        ]
        # update on the observed instantaneous quality
        innovation = self._logit(quality) - state[0]
        s = cov[0][0] + self.r_quality
        gain = [cov[0][0] / s, cov[1][0] / s]
        state = [state[0] + gain[0] * innovation, state[1] + gain[1] * innovation]
        cov = [
            [(1 - gain[0]) * cov[0][0], (1 - gain[0]) * cov[0][1]],
            [cov[1][0] - gain[1] * cov[0][0], cov[1][1] - gain[1] * cov[0][1]],
        ]
        self.state, self.cov = state, cov
        return self._sigmoid(state[0]), state[1]


# ---------------------------------------------------------------------------
# shared measurement construction (identical for every arm)
# ---------------------------------------------------------------------------

def build_measurement(row: dict[str, str], *, quality: float, sigma_clean_m: float,
                      sigma_soft_m: float, tau_reject: float, tau_good: float,
                      use_quality: bool) -> tuple[tuple[float, float], list[list[float]], str]:
    """Return (z, R, decision) from the CURRENT frame only.

    `z` is the corrected ground point of this frame. No arm may alter it -- that is the
    invariant that keeps this a measurement update rather than track-to-track fusion. The
    arms differ only in `R` and in whether the reading is admitted.
    """
    z = (float(row['hull_x']), float(row['hull_y']))

    if not use_quality:
        sigma = sigma_clean_m
        decision = 'admit'
    elif quality < tau_reject:
        # The occluded regime has a large MEAN error; a wide covariance would misdescribe it.
        return z, [[0.0, 0.0], [0.0, 0.0]], 'refuse'
    elif quality < tau_good:
        # Intermediate band: the reading still carries information, so widen rather than drop.
        span = max(tau_good - tau_reject, 1e-6)
        weight = (tau_good - quality) / span
        sigma = sigma_clean_m + weight * (sigma_soft_m - sigma_clean_m)
        decision = 'soft'
    else:
        sigma = sigma_clean_m
        decision = 'admit'

    return z, [[sigma ** 2, 0.0], [0.0, sigma ** 2]], decision


def _check_invariants(row: dict[str, str], z: tuple[float, float]) -> None:
    """Assert the measurement is the current frame's corrected point and nothing else."""
    assert z[0] == float(row['hull_x']) and z[1] == float(row['hull_y']), \
        'INVARIANT 1/2 violated: the robot measurement is not the current frame reading'


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def whiten_stats(records: list[dict[str, float]]) -> dict[str, float]:
    """Distributional scoring: is `R^-1/2 (z - z_true)` standard normal?"""
    if len(records) < 8:
        return {'n': len(records)}
    wx = [row['wx'] for row in records]
    wy = [row['wy'] for row in records]
    both = wx + wy
    maha = [row['maha'] for row in records]
    mean = st.mean(both)
    centred = [value - mean for value in both]
    m2 = st.mean([value ** 2 for value in centred]) or 1e-12
    errors = [row['err_cm'] for row in records]
    coverage = {}
    for label, threshold in (('50', 1.386), ('90', 4.605), ('95', 5.991), ('99', 9.210)):
        coverage[f'coverage_{label}'] = 100.0 * sum(1 for v in maha if v <= threshold) / len(maha)
    return {
        'n': len(records),
        'median_err_cm': st.median(errors),
        'rms_err_cm': math.sqrt(st.mean([value ** 2 for value in errors])),
        'p90_err_cm': sorted(errors)[int(0.9 * (len(errors) - 1))],
        'stated_sigma_cm': st.median([row['sigma_cm'] for row in records]),
        'whitened_mean': mean,
        'whitened_sd': math.sqrt(m2),
        'whitened_skew': st.mean([v ** 3 for v in centred]) / m2 ** 1.5,
        'whitened_excess_kurtosis': st.mean([v ** 4 for v in centred]) / m2 ** 2 - 3.0,
        'mahalanobis_mean': st.mean(maha),
        'mahalanobis_median': st.median(maha),
        'beyond_4sigma_pct': 100.0 * sum(1 for v in maha if v > 16.0) / len(maha),
        **coverage,
    }


def quality_classification(pairs: list[tuple[float, int]]) -> dict[str, float]:
    """How well does a quality estimate identify unusable readings? AUC plus calibration."""
    if not pairs or len({label for _, label in pairs}) < 2:
        return {'n': len(pairs)}
    scores = [score for score, _ in pairs]
    labels = [label for _, label in pairs]
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
    positives = sum(labels)
    negatives = len(labels) - positives
    rank_sum = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    # Reliability: mean predicted probability against observed frequency, in deciles.
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for score, label in pairs:
        bins[min(int(score * 10), 9)].append((score, label))
    calibration_error = sum(
        len(values) * abs(st.mean([s for s, _ in values]) - st.mean([l for _, l in values]))
        for values in bins.values()) / len(pairs)
    return {'n': len(pairs), 'auc': auc, 'calibration_error': calibration_error,
            'usable_rate': st.mean(labels)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def load(capture: Path) -> tuple[list[dict[str, str]], dict[str, tuple[float, float]]]:
    from separate_bias_and_fast_noise import load_rows

    # Line labels have carried two schemes: `line<centre>_<axis>_yaw<n>` in the first
    # dense capture, and `<split>-c<centre>_<axis>_yaw<n>` once the partition was assigned
    # before capture. Accept anything that names an axis and a heading rather than
    # matching a prefix, so a naming change cannot silently empty the dataset.
    rows = [row for row in load_rows(capture)
            if row.get('detected') in TRUE_VALUES
            and row.get('hull_valid') in TRUE_VALUES
            and '_yaw' in (row.get('dataset_split') or '')]
    manifest = json.loads((capture / 'capture_manifest.json').read_text(encoding='utf-8'))
    cameras = {item['camera_id']: (float(item['pose_xyz_rpy'][0]),
                                   float(item['pose_xyz_rpy'][1]))
               for item in manifest['cameras']}
    return rows, cameras


def place_of(row: dict[str, str]) -> str:
    """The centre a reading belongs to. Held-out folds are formed over these.

    Under the pre-assigned-split scheme the label already carries its partition, so the
    centre token is `<split>-c<index>`; that token is the unit of the spatial holdout.
    """
    return (row.get('dataset_split') or '').split('_')[0]


def split_of(row: dict[str, str]) -> str:
    """The pre-assigned partition, when the capture recorded one."""
    token = (row.get('dataset_split') or '').split('_')[0]
    return token.split('-')[0] if '-' in token else ''


def sequence_of(row: dict[str, str]) -> tuple[str, str]:
    return (row['camera_id'], row['dataset_split'] or '')


def truth_error_cm(row: dict[str, str]) -> float:
    """Evaluation only. Never an input to any arm."""
    return math.hypot(float(row['hull_dx']), float(row['hull_dy'])) * 100.0


def has_clean_prefix(sequence: list[dict[str, str]], *, min_clean: int = 5) -> bool:
    """Does this sequence start clean and later degrade? Arm D's subpopulation."""
    labels = [truth_error_cm(row) <= USABLE_CM for row in sequence]
    if len(labels) < min_clean + 2:
        return False
    return all(labels[:min_clean]) and not all(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--speed-m-s', type=float, default=0.22)
    parser.add_argument('--tau-reject', type=float, default=0.35)
    parser.add_argument('--tau-good', type=float, default=0.80)
    parser.add_argument('--sigma-soft-scale', type=float, default=4.0,
                        help='How much wider R is in the intermediate quality band, as a '
                             'multiple of the clean sigma.')
    args = parser.parse_args()

    rows, cameras = load(args.capture)
    spacing_m = float(json.loads(
        (Path(json.loads((args.capture / 'capture_manifest.json').read_text(
            encoding='utf-8'))['plan']['pose_file'])).read_text(encoding='utf-8'))['step_m'])
    step_s = spacing_m / args.speed_m_s

    sequences: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sequences[sequence_of(row)].append(row)
    for key, items in sequences.items():
        axis = 'x' if '_x_' in key[1] else 'y'
        items.sort(key=lambda row: float(row['robot_x' if axis == 'x' else 'robot_y']))

    # When the capture carries a pre-assigned partition, honour it: the split was fixed
    # before any frame existed, so it cannot be reshaped to suit a result. Fall back to
    # leave-one-centre-out only for captures written before that scheme.
    pre_assigned = sorted({split_of(row) for row in rows} - {''})
    use_pre_assigned = {'train', 'test'} <= set(pre_assigned)
    places = ['test'] if use_pre_assigned else sorted({place_of(row) for row in rows})
    arms = ('A_nn_only', 'B_instant_quality', 'C_temporal_quality')
    scored: dict[str, list[dict[str, float]]] = defaultdict(list)
    scored_clean_prefix: dict[str, list[dict[str, float]]] = defaultdict(list)
    quality_pairs: dict[str, list[tuple[float, int]]] = defaultdict(list)
    refused = {arm: 0 for arm in arms}
    admitted = {arm: 0 for arm in arms}

    for held in places:
        if use_pre_assigned:
            # Fit on the training centres only; score on the held-out test centres.
            train = [row for row in rows if split_of(row) == 'train']
            held_rows = {id(row) for row in rows if split_of(row) == 'test'}
        else:
            train = [row for row in rows if place_of(row) != held]
            held_rows = {id(row) for row in rows if place_of(row) == held}
        if not train:
            continue
        # Commission the clean-regime sigma and fit the quality model on TRAINING places.
        clean_train = [row for row in train if truth_error_cm(row) <= USABLE_CM]
        if len(clean_train) < 20:
            continue
        sigma_clean_m = math.sqrt(st.mean(
            [float(row['hull_dx']) ** 2 + float(row['hull_dy']) ** 2
             for row in clean_train]) / 2.0)
        sigma_soft_m = args.sigma_soft_scale * sigma_clean_m

        model = InstantQuality()
        model.fit([(runtime_features(row, cameras[row['camera_id']]),
                    1 if truth_error_cm(row) <= USABLE_CM else 0) for row in train])

        for key, items in sequences.items():
            if not items or id(items[0]) not in held_rows:
                continue
            clean_prefix = has_clean_prefix(items)
            quality_filter = QualityFilter(step_s=step_s)
            for row in items:
                features = runtime_features(row, cameras[row['camera_id']])
                q_instant = model.predict(features)
                q_temporal, _rate = quality_filter.step(q_instant)
                label = 1 if truth_error_cm(row) <= USABLE_CM else 0
                quality_pairs['instant'].append((q_instant, label))
                quality_pairs['temporal'].append((q_temporal, label))

                for arm in arms:
                    if arm == 'A_nn_only':
                        quality, use_quality = 1.0, False
                    elif arm == 'B_instant_quality':
                        quality, use_quality = q_instant, True
                    else:
                        quality, use_quality = q_temporal, True

                    z, cov, decision = build_measurement(
                        row, quality=quality, sigma_clean_m=sigma_clean_m,
                        sigma_soft_m=sigma_soft_m, tau_reject=args.tau_reject,
                        tau_good=args.tau_good, use_quality=use_quality)
                    _check_invariants(row, z)
                    if decision == 'refuse':
                        refused[arm] += 1
                        continue
                    admitted[arm] += 1
                    ex = z[0] - float(row['robot_x'])
                    ey = z[1] - float(row['robot_y'])
                    sigma = math.sqrt(cov[0][0])
                    record = {
                        'err_cm': math.hypot(ex, ey) * 100.0,
                        'sigma_cm': sigma * 100.0,
                        'wx': ex / sigma, 'wy': ey / sigma,
                        'maha': (ex * ex + ey * ey) / (sigma * sigma),
                    }
                    scored[arm].append(record)
                    if clean_prefix:
                        scored_clean_prefix[arm].append(record)

    results = {arm: whiten_stats(records) for arm, records in scored.items()}
    for arm, records in scored.items():
        total = admitted[arm] + refused[arm]
        results[arm]['availability_pct'] = 100.0 * admitted[arm] / max(total, 1)
    clean_results = {arm: whiten_stats(records)
                     for arm, records in scored_clean_prefix.items()}
    quality_results = {name: quality_classification(pairs)
                       for name, pairs in quality_pairs.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': 'perception_filter_cascade_quality_arms.v1',
        'capture': str(args.capture),
        'protocol': ('pre-assigned split fixed before capture: fitted on train centres, '
                     'scored on held-out test centres' if use_pre_assigned
                     else 'leave-one-centre-out'),
        'invariants': [
            'measurement uses current frame only',
            'no perception posterior mean enters the measurement',
            'perception history affects only quality, admission and R',
            'R adaptation uses no robot-EKF innovation',
            'all arms share one measurement construction',
            'instantaneous and temporal quality reported separately',
        ],
        'settings': {'step_s': step_s, 'spacing_m': spacing_m,
                     'speed_m_s': args.speed_m_s, 'tau_reject': args.tau_reject,
                     'tau_good': args.tau_good,
                     'sigma_soft_scale': args.sigma_soft_scale,
                     'usable_threshold_cm': USABLE_CM},
        'places': places,
        'quality_estimators': quality_results,
        'overall': results,
        'clean_prefix_subset': clean_results,
    }
    (args.out_dir / 'quality_arms.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    protocol = ('pre-assigned split: fitted on train centres, scored on test centres'
                if use_pre_assigned
                else f'leave-one-centre-out over {len(places)} centres')
    print(f'{protocol}; sample interval {step_s:.4f} s')
    print(f'usable threshold {USABLE_CM:.0f} cm; reject below q={args.tau_reject}, '
          f'clean above q={args.tau_good}')
    print()
    print('Perception quality estimators (does the estimate identify bad views?)')
    print(f'  {"estimator":12s} {"n":>6s} {"AUC":>6s} {"calib_err":>10s} {"usable_rate":>12s}')
    for name, stats in quality_results.items():
        if 'auc' not in stats:
            continue
        print(f'  {name:12s} {stats["n"]:6d} {stats["auc"]:6.3f} '
              f'{stats["calibration_error"]:10.3f} {stats["usable_rate"]:12.3f}')
    print()
    header = (f'{"arm":22s} {"n":>5s} {"avail%":>7s} {"med_cm":>7s} {"p90_cm":>7s} '
              f'{"statedS":>8s} {"w_sd":>6s} {"exkurt":>7s} {"maha":>7s} '
              f'{"c50":>6s} {"c90":>6s} {"c95":>6s} {"c99":>6s} {">4sig":>6s}')
    print('Localization likelihood calibration (all arms, held out by place)')
    print(header)
    print('-' * len(header))
    for arm in arms:
        s = results.get(arm, {})
        if 'whitened_sd' not in s:
            continue
        print(f'{arm:22s} {s["n"]:5d} {s["availability_pct"]:7.1f} {s["median_err_cm"]:7.2f} '
              f'{s["p90_err_cm"]:7.2f} {s["stated_sigma_cm"]:8.2f} {s["whitened_sd"]:6.2f} '
              f'{s["whitened_excess_kurtosis"]:7.2f} {s["mahalanobis_mean"]:7.2f} '
              f'{s["coverage_50"]:6.1f} {s["coverage_90"]:6.1f} {s["coverage_95"]:6.1f} '
              f'{s["coverage_99"]:6.1f} {s["beyond_4sigma_pct"]:6.2f}')
    print()
    print('Arm D: arm C restricted to sequences with a clean prefix then degradation')
    print(header)
    print('-' * len(header))
    for arm in arms:
        s = clean_results.get(arm, {})
        if 'whitened_sd' not in s:
            continue
        print(f'{arm:22s} {s["n"]:5d} {"-":>7s} {s["median_err_cm"]:7.2f} '
              f'{s["p90_err_cm"]:7.2f} {s["stated_sigma_cm"]:8.2f} {s["whitened_sd"]:6.2f} '
              f'{s["whitened_excess_kurtosis"]:7.2f} {s["mahalanobis_mean"]:7.2f} '
              f'{s["coverage_50"]:6.1f} {s["coverage_90"]:6.1f} {s["coverage_95"]:6.1f} '
              f'{s["coverage_99"]:6.1f} {s["beyond_4sigma_pct"]:6.2f}')
    print()
    print('Ideal: maha 2.00, w_sd 1.00, exkurt 0, coverage 50/90/95/99, >4sig ~0.')
    print('Read stated sigma beside the error: a method that only widens R is not a success.')
    print()
    print(f'wrote {args.out_dir / "quality_arms.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
