#!/usr/bin/env python3
"""Compare perception-side filtering arms on the dense-line captures.

The question is narrow and pre-registered: does filtering the bounding box over time turn
imperfect per-frame detections into a better localization measurement, and does the
covariance it reports describe the error that is actually left?

Arms, applied to the box-bottom pixel of one camera along one dense line:

  A  raw          per-frame observation, no temporal processing (the current pipeline)
  B  kf           constant-velocity Kalman filter on the box pixel
  C  robust_kf    same filter, but an observation whose normalised innovation exceeds a
                  gate inflates that observation's covariance instead of being discarded
  D  smoother     fixed-lag smoother: the same filter run forward, then corrected using
                  the following `--lag` observations
  E  static_R     no temporal filter; the per-frame observation is kept but paired with a
                  covariance measured offline for this camera and range bucket

Arm E is the arm to beat. It represents the alternative in which the covariance comes from
a commissioned field rather than from a filter, and it is what makes the comparison a real
test instead of a demonstration.

Every arm is scored two ways:

  accuracy     median and RMS distance from the true position
  calibration  normalised squared error e' R^-1 e, whose mean should be 2.0 for a
               well-calibrated 2-D measurement, plus the share of readings beyond
               4 sigma, because a median-based statistic is blind to the tail

The filter runs in pixel space and the reported covariance is pushed through the ground
projection with the local Jacobian, which is the interface the localization EKF expects.
The Jacobian is obtained numerically from the same projection the capture used, so no
separate calibration is introduced here.
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
for rel in ('scripts/perception', 'src/experiments', 'src/perception', 'src/unav_common'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

TRUE_VALUES = {'True', 'true', '1'}


# ---------------------------------------------------------------------------
# small 2x2 / 4x4 linear algebra, kept explicit so the filter is auditable
# ---------------------------------------------------------------------------

def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows, inner, cols = len(a), len(b), len(b[0])
    return [[sum(a[r][k] * b[k][c] for k in range(inner)) for c in range(cols)]
            for r in range(rows)]


def mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[r][c] + b[r][c] for c in range(len(a[0]))] for r in range(len(a))]


def mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[r][c] - b[r][c] for c in range(len(a[0]))] for r in range(len(a))]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def inv2(a: list[list[float]]) -> list[list[float]]:
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) < 1e-15:
        raise ValueError('singular 2x2 matrix')
    return [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]]


def inv4(a: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse for the 4x4 predicted covariance in the smoother."""
    size = 4
    aug = [a[row][:] + [1.0 if col == row else 0.0 for col in range(size)]
           for row in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-15:
            raise ValueError('singular 4x4 matrix')
        aug[col], aug[pivot] = aug[pivot], aug[col]
        divisor = aug[col][col]
        aug[col] = [value / divisor for value in aug[col]]
        for other in range(size):
            if other == col:
                continue
            factor = aug[other][col]
            if factor == 0.0:
                continue
            aug[other] = [value - factor * pivot_value
                          for value, pivot_value in zip(aug[other], aug[col])]
    return [row[size:] for row in aug]


def identity(size: int) -> list[list[float]]:
    return [[1.0 if r == c else 0.0 for c in range(size)] for r in range(size)]


# ---------------------------------------------------------------------------
# constant-velocity box filter
# ---------------------------------------------------------------------------

class BoxFilter:
    """Constant-velocity Kalman filter on the box-bottom pixel [u, v, du, dv].

    Velocity is included because the robot drives: without it the filter would treat real
    motion as measurement error and lag behind. The process noise is an acceleration model
    whose scale is not tuned against the score: `q_accel_px` is measured, as the projection
    scale in pixels per metre times the robot acceleration the filter must tolerate.
    """

    def __init__(self, *, q_accel_px: float, r_px: float, step_s: float) -> None:
        self.dt = step_s
        self.q_accel = q_accel_px
        self.r_px = r_px
        self.state: list[float] | None = None
        self.cov: list[list[float]] | None = None

    def _transition(self) -> list[list[float]]:
        dt = self.dt
        return [[1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]

    def _process_noise(self) -> list[list[float]]:
        dt, var = self.dt, self.q_accel ** 2
        # Discrete-time integrated acceleration covariance.
        p = dt ** 4 / 4.0 * var
        c = dt ** 3 / 2.0 * var
        v = dt ** 2 * var
        return [[p, 0.0, c, 0.0],
                [0.0, p, 0.0, c],
                [c, 0.0, v, 0.0],
                [0.0, c, 0.0, v]]

    def initialise(self, u: float, v: float) -> None:
        self.state = [u, v, 0.0, 0.0]
        big = (20.0 * self.r_px) ** 2  # velocity is unknown at the first sample
        self.cov = [[self.r_px ** 2, 0.0, 0.0, 0.0],
                    [0.0, self.r_px ** 2, 0.0, 0.0],
                    [0.0, 0.0, big, 0.0],
                    [0.0, 0.0, 0.0, big]]

    def predict(self) -> None:
        if self.state is None or self.cov is None:
            return
        f = self._transition()
        self.state = [sum(f[r][c] * self.state[c] for c in range(4)) for r in range(4)]
        self.cov = mat_add(mat_mul(mat_mul(f, self.cov), transpose(f)), self._process_noise())

    def innovation(self, u: float, v: float, r_px: float) -> tuple[list[float], list[list[float]]]:
        assert self.state is not None and self.cov is not None
        nu = [u - self.state[0], v - self.state[1]]
        s = [[self.cov[0][0] + r_px ** 2, self.cov[0][1]],
             [self.cov[1][0], self.cov[1][1] + r_px ** 2]]
        return nu, s

    def update(self, u: float, v: float, r_px: float) -> None:
        assert self.state is not None and self.cov is not None
        nu, s = self.innovation(u, v, r_px)
        s_inv = inv2(s)
        h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        gain = mat_mul(mat_mul(self.cov, transpose(h)), s_inv)
        self.state = [self.state[r] + sum(gain[r][c] * nu[c] for c in range(2)) for r in range(4)]
        keep = mat_sub(identity(4), mat_mul(gain, h))
        cov = mat_mul(mat_mul(keep, self.cov), transpose(keep))
        noise = [[r_px ** 2, 0.0], [0.0, r_px ** 2]]
        self.cov = mat_add(cov, mat_mul(mat_mul(gain, noise), transpose(gain)))

    def pixel_estimate(self) -> tuple[float, float, list[list[float]]]:
        assert self.state is not None and self.cov is not None
        return self.state[0], self.state[1], [row[:2] for row in self.cov[:2]]


# ---------------------------------------------------------------------------
# projection and its Jacobian
# ---------------------------------------------------------------------------

def make_projector(camera_id: str, capture: Path):
    """Return `project(u, v) -> (x, y)` for one camera, plus a numeric Jacobian.

    The camera model is rebuilt with the characterization study's own helper, which
    verifies the world-profile bytes against the capture manifest and uses the same
    `pixel_to_world` homography the frozen interpretations were derived with. Nothing is
    re-calibrated here, so arm A reproduces the pipeline's existing `raw` reading.
    """
    import json as _json

    sys.path.insert(0, str(REPO / 'experiments/camera_observation_characterization'))
    from derive_interpretations import camera_models  # noqa: E402

    manifest = _json.loads((capture / 'capture_manifest.json').read_text(encoding='utf-8'))
    camera = camera_models(manifest)[camera_id]

    def project(u: float, v: float) -> tuple[float, float]:
        point = camera.pixel_to_world(float(u), float(v))
        if point is None:
            raise ValueError('pixel does not back-project to the ground plane')
        return float(point[0]), float(point[1])

    def jacobian(u: float, v: float, delta: float = 0.5) -> list[list[float]]:
        x_up, y_up = project(u + delta, v)
        x_dn, y_dn = project(u - delta, v)
        x_vp, y_vp = project(u, v + delta)
        x_vn, y_vn = project(u, v - delta)
        return [[(x_up - x_dn) / (2 * delta), (x_vp - x_vn) / (2 * delta)],
                [(y_up - y_dn) / (2 * delta), (y_vp - y_vn) / (2 * delta)]]

    return project, jacobian


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(records: list[dict[str, float]]) -> dict[str, float]:
    """Accuracy and calibration for one arm."""
    if not records:
        return {'n': 0}
    errors = [math.hypot(row['ex'], row['ey']) for row in records]
    nse = [row['nse'] for row in records if math.isfinite(row['nse'])]
    beyond4 = sum(1 for value in nse if value > 16.0)  # 4 sigma in 2-D -> squared radius 16
    return {
        'n': len(records),
        'median_err_cm': st.median(errors) * 100.0,
        'rms_err_cm': math.sqrt(st.mean([value ** 2 for value in errors])) * 100.0,
        'p90_err_cm': sorted(errors)[int(0.9 * (len(errors) - 1))] * 100.0,
        'mean_nse': st.mean(nse) if nse else float('nan'),
        'median_nse': st.median(nse) if nse else float('nan'),
        'beyond_4sigma_pct': 100.0 * beyond4 / len(nse) if nse else float('nan'),
    }


def normalised_squared_error(ex: float, ey: float, cov: list[list[float]],
                             floor_m: float) -> float:
    total = [[cov[0][0] + floor_m ** 2, cov[0][1]],
             [cov[1][0], cov[1][1] + floor_m ** 2]]
    try:
        inv = inv2(total)
    except ValueError:
        return float('nan')
    return (ex * (inv[0][0] * ex + inv[0][1] * ey)
            + ey * (inv[1][0] * ex + inv[1][1] * ey))


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def run_line(items: list[dict[str, str]], *, project, jacobian,
             sigma_px: float, q_accel_px: float, step_s: float, gate: float,
             lag: int, floor_m: float, static_sigma_px: float | None,
             commissioned_sigma_m: float | None = None
             ) -> dict[str, list[dict[str, float]]]:
    """Run every arm along one dense line and return per-arm scored records."""
    out: dict[str, list[dict[str, float]]] = defaultdict(list)

    observations: list[tuple[float, float, float, float]] = []
    for row in items:
        text_u, text_v = row.get('u_bbox_bottom', ''), row.get('v_bbox_bottom', '')
        if text_u in ('', 'nan') or text_v in ('', 'nan'):
            continue
        observations.append((float(text_u), float(text_v),
                             float(row['robot_x']), float(row['robot_y'])))
    if len(observations) < 4:
        return out

    def record(arm: str, u: float, v: float, cov_px: list[list[float]] | None,
               truth_x: float, truth_y: float,
               cov_xy_override: list[list[float]] | None = None) -> None:
        try:
            x, y = project(u, v)
            jac = jacobian(u, v)
        except ValueError:
            return
        if cov_xy_override is not None:
            cov_xy = cov_xy_override
        else:
            assert cov_px is not None
            cov_xy = mat_mul(mat_mul(jac, cov_px), transpose(jac))
        ex, ey = x - truth_x, y - truth_y
        out[arm].append({
            'ex': ex, 'ey': ey,
            'nse': normalised_squared_error(ex, ey, cov_xy, floor_m),
        })

    # --- arm A: raw per-frame observation, covariance from pixel noise alone
    raw_cov = [[sigma_px ** 2, 0.0], [0.0, sigma_px ** 2]]
    for u, v, truth_x, truth_y in observations:
        record('A_raw', u, v, raw_cov, truth_x, truth_y)

    # --- arm E: per-frame observation with a commissioned ground-plane covariance.
    #     The commissioned value is already a spread on the floor, so it is used directly
    #     rather than being pushed through the Jacobian a second time.
    if commissioned_sigma_m is not None:
        commissioned_cov = [[commissioned_sigma_m ** 2, 0.0],
                            [0.0, commissioned_sigma_m ** 2]]
        for u, v, truth_x, truth_y in observations:
            record('E_static_R', u, v, None, truth_x, truth_y,
                   cov_xy_override=commissioned_cov)
    elif static_sigma_px is not None:
        static_cov = [[static_sigma_px ** 2, 0.0], [0.0, static_sigma_px ** 2]]
        for u, v, truth_x, truth_y in observations:
            record('E_static_R', u, v, static_cov, truth_x, truth_y)

    # --- arm B: plain constant-velocity filter
    kf = BoxFilter(q_accel_px=q_accel_px, r_px=sigma_px, step_s=step_s)
    kf.initialise(observations[0][0], observations[0][1])
    for index, (u, v, truth_x, truth_y) in enumerate(observations):
        if index > 0:
            kf.predict()
            kf.update(u, v, sigma_px)
        mu_u, mu_v, cov_px = kf.pixel_estimate()
        record('B_kf', mu_u, mu_v, cov_px, truth_x, truth_y)

    # --- arm C: same filter, inflating the covariance of surprising observations.
    #     This is the soft-rejection alternative to a hard gate: instead of discarding a
    #     detection that disagrees with the prediction, its own noise is scaled up so it
    #     still contributes, but weakly. The scale is not an arbitrary curve -- it is the
    #     factor that brings the normalised innovation back to the gate, which is the
    #     Huber-style rescaling used for heavy-tailed observations. Below the gate nothing
    #     changes, so a well-behaved reading is untouched.
    rkf = BoxFilter(q_accel_px=q_accel_px, r_px=sigma_px, step_s=step_s)
    rkf.initialise(observations[0][0], observations[0][1])
    for index, (u, v, truth_x, truth_y) in enumerate(observations):
        if index > 0:
            rkf.predict()
            nu, s = rkf.innovation(u, v, sigma_px)
            try:
                s_inv = inv2(s)
                d2 = (nu[0] * (s_inv[0][0] * nu[0] + s_inv[0][1] * nu[1])
                      + nu[1] * (s_inv[1][0] * nu[0] + s_inv[1][1] * nu[1]))
            except ValueError:
                d2 = 0.0
            # Solve for the noise scale c that would put d2 at the gate. The predicted
            # part of S does not scale with c, so c^2 = (d2 / gate - 1) * S_pred / R + 1
            # is the exact factor when S is isotropic; the isotropic form is used because
            # the observation noise supplied here is isotropic by construction.
            inflate = 1.0
            if gate > 0.0 and d2 > gate:
                predicted_var = max(0.5 * (rkf.cov[0][0] + rkf.cov[1][1]), 0.0) \
                    if rkf.cov is not None else 0.0
                ratio = d2 / gate
                scaled = (ratio - 1.0) * (predicted_var / sigma_px ** 2) + ratio
                inflate = math.sqrt(max(scaled, 1.0))
            rkf.update(u, v, sigma_px * inflate)
        mu_u, mu_v, cov_px = rkf.pixel_estimate()
        record('C_robust_kf', mu_u, mu_v, cov_px, truth_x, truth_y)

    # --- arm D: fixed-lag smoother, as a proper Rauch-Tung-Striebel backward pass.
    #     Back-propagating the filtered mean with the constant-velocity model is NOT the
    #     smoother: it leaves the covariance at its filtered value, so the arm would claim
    #     the wrong uncertainty and the calibration comparison would be meaningless.
    for index in range(len(observations)):
        end = min(len(observations) - 1, index + lag)
        window = observations[:end + 1]
        sm = BoxFilter(q_accel_px=q_accel_px, r_px=sigma_px, step_s=step_s)
        sm.initialise(window[0][0], window[0][1])

        # Forward pass, storing what the backward recursion needs at every step.
        filtered: list[tuple[list[float], list[list[float]]]] = []
        predicted: list[tuple[list[float], list[list[float]]]] = []
        assert sm.state is not None and sm.cov is not None
        filtered.append(([value for value in sm.state], [row[:] for row in sm.cov]))
        predicted.append(([value for value in sm.state], [row[:] for row in sm.cov]))
        for step, (u, v, _truth_x, _truth_y) in enumerate(window):
            if step == 0:
                continue
            sm.predict()
            assert sm.state is not None and sm.cov is not None
            predicted.append(([value for value in sm.state], [row[:] for row in sm.cov]))
            sm.update(u, v, sigma_px)
            assert sm.state is not None and sm.cov is not None
            filtered.append(([value for value in sm.state], [row[:] for row in sm.cov]))

        # Backward pass from the window end down to the target index.
        transition = sm._transition()
        state, cov = filtered[-1]
        state = [value for value in state]
        cov = [row[:] for row in cov]
        for step in range(len(window) - 2, index - 1, -1):
            state_f, cov_f = filtered[step]
            state_p, cov_p = predicted[step + 1]
            try:
                gain = mat_mul(mat_mul(cov_f, transpose(transition)), inv4(cov_p))
            except ValueError:
                break
            diff = [state[row] - state_p[row] for row in range(4)]
            state = [state_f[row] + sum(gain[row][col] * diff[col] for col in range(4))
                     for row in range(4)]
            delta = mat_sub(cov, cov_p)
            cov = mat_add(cov_f, mat_mul(mat_mul(gain, delta), transpose(gain)))

        truth_x, truth_y = observations[index][2], observations[index][3]
        record('D_smoother', state[0], state[1], [row[:2] for row in cov[:2]],
               truth_x, truth_y)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--sigma-px', type=float, default=0.764,
                        help="Per-frame pixel noise the filters assume. Default is the "
                             "repo's commissioned value.")
    parser.add_argument('--static-sigma-px', type=float, default=None,
                        help='Pixel sigma for arm E. Defaults to the measured fast '
                             'component from bq_split.json when available.')
    parser.add_argument('--q-accel-px', type=float, default=None,
                        help='Process noise as a pixel acceleration, in px/s^2. Left unset '
                             'it is derived from the measured projection scale rather than '
                             'tuned: see --px-per-metre and --turn-rate-rad-s.')
    parser.add_argument('--px-per-metre', type=float, default=41.4,
                        help='Pixel motion per metre of robot travel. Measured on the frozen '
                             'grid as the median step-to-step box-bottom motion (22.4 px/m '
                             'vertically, 41.4 px/m horizontally); the larger axis is used so '
                             'the filter is not over-confident about motion it cannot follow.')
    parser.add_argument('--accel-m-s2', type=float, default=0.5,
                        help='Robot acceleration the filter must tolerate without lagging. '
                             'The process noise is px_per_metre * accel_m_s2.')
    parser.add_argument('--step-s', type=float, default=None,
                        help='Time between consecutive samples. Left unset it is derived '
                             'as spacing / speed from the capture pose file, so it cannot '
                             'silently disagree with the geometry.')
    parser.add_argument('--speed-m-s', type=float, default=0.22,
                        help='Driving speed the sampled line stands for. 0.22 m/s is the '
                             'campaign speed.')
    parser.add_argument('--spacing-m', type=float, default=None,
                        help='Sample spacing. Left unset it is read from the pose file '
                             'named in the capture manifest.')
    parser.add_argument('--gate', type=float, default=9.0,
                        help='Normalised innovation above which arm C inflates covariance.')
    parser.add_argument('--lag', type=int, default=3)
    parser.add_argument('--floor-m', type=float, default=0.0,
                        help='Additive per-camera residual floor in R, in metres.')
    parser.add_argument('--bq-split', type=Path, default=None,
                        help='bq_split.json, used to default --static-sigma-px.')
    args = parser.parse_args()

    # Reuse the split script's loader so line labels are recovered identically, including
    # the fallback for captures whose derived schema predates the dataset_split column.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from separate_bias_and_fast_noise import load_rows  # noqa: E402

    rows = load_rows(args.capture)

    # The sample interval is geometry, not a knob: the line was sampled in space, so the
    # time between samples is the spacing divided by the speed it stands for. Reading the
    # spacing from the capture's own pose file stops it drifting out of step with the data.
    spacing_m = args.spacing_m
    if spacing_m is None:
        try:
            manifest = json.loads(
                (args.capture / 'capture_manifest.json').read_text(encoding='utf-8'))
            pose_file = Path(manifest.get('plan', {}).get('pose_file', ''))
            spacing_m = float(json.loads(
                pose_file.read_text(encoding='utf-8')).get('step_m'))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            spacing_m = None
    if args.step_s is not None:
        step_s = args.step_s
    elif spacing_m:
        step_s = spacing_m / args.speed_m_s
        print(f'sample interval derived: {spacing_m * 100:.0f} cm / {args.speed_m_s} m/s '
              f'= {step_s:.4f} s')
    else:
        raise SystemExit(
            'could not read the sample spacing from the capture manifest; pass '
            '--spacing-m or --step-s explicitly so the filter timing is stated, not guessed')

    # The process noise is a physical quantity: how many pixels the box can accelerate
    # per second squared, given how far the box moves per metre the robot travels.
    q_accel_px = (args.q_accel_px if args.q_accel_px is not None
                  else args.px_per_metre * args.accel_m_s2)
    if args.q_accel_px is None:
        print(f'process noise derived: {args.px_per_metre:.1f} px/m x '
              f'{args.accel_m_s2:.2f} m/s^2 = {q_accel_px:.1f} px/s^2')

    # Arm E stands for a commissioned covariance: the spread of the error this camera
    # actually makes at this place. It must NOT be the frame-to-frame component from
    # bq_split.json -- that describes only how much a reading jitters between samples,
    # not how far it sits from the truth, so using it would hand arm E a covariance
    # hundreds of times too small and guarantee it looks overconfident by construction.
    # It is commissioned per camera and range bucket from the residuals themselves,
    # which is exactly how R_c(s) is defined.
    static_sigma = args.static_sigma_px
    commissioned: dict[tuple[str, int], float] = {}
    if static_sigma is None:
        buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in rows:
            if row.get('detected') not in TRUE_VALUES:
                continue
            if row.get('raw_valid') not in TRUE_VALUES:
                continue
            text = row.get('camera_range_m', '')
            if text in ('', 'nan'):
                continue
            key = (row['camera_id'], int(float(text) // 4) * 4)
            for field in ('raw_dx', 'raw_dy'):
                value = row.get(field, '')
                if value not in ('', 'nan'):
                    buckets[key].append(float(value))
        for key, values in buckets.items():
            if len(values) < 8:
                continue
            # Root-mean-square about zero: the commissioned covariance has to cover the
            # bias this camera carries, not just its scatter about its own mean.
            commissioned[key] = math.sqrt(st.mean([value ** 2 for value in values]))
        if commissioned:
            print(f'arm E covariance commissioned per camera and range bucket from the '
                  f'residuals: {len(commissioned)} buckets, '
                  f'{min(commissioned.values()) * 100:.1f}-'
                  f'{max(commissioned.values()) * 100:.1f} cm')
    if static_sigma is None and not commissioned:
        static_sigma = args.sigma_px

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get('detected') not in TRUE_VALUES:
            continue
        label = row.get('dataset_split') or ''
        if not label.startswith('line'):
            continue
        groups[(row['camera_id'], label)].append(row)

    if not groups:
        detected = sum(1 for row in rows if row.get('detected') in TRUE_VALUES)
        labels = {row.get('dataset_split', '') for row in rows}
        raise SystemExit(
            'no dense-line detections found in the capture: '
            f'{detected} detections but no row carries a line label. '
            f'Labels present: {sorted(label for label in labels if label)[:5] or "none"}. '
            'Was this capture run with dense_lines.json?')

    projectors: dict[str, tuple[object, object]] = {}
    combined: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_camera: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list))

    for (camera_id, label), items in sorted(groups.items()):
        if camera_id not in projectors:
            projectors[camera_id] = make_projector(camera_id, args.capture)
        project, jacobian = projectors[camera_id]
        axis = 'x' if '_x_' in label else 'y'
        column = 'robot_x' if axis == 'x' else 'robot_y'
        items = sorted(items, key=lambda row: float(row[column]))

        # Arm E's commissioned covariance is looked up for this camera and the range
        # bucket this line sits in, so it varies with place as R_c(s) is meant to.
        bucket_sigma = None
        if commissioned:
            ranges = [float(row['camera_range_m']) for row in items
                      if row.get('camera_range_m', '') not in ('', 'nan')]
            if ranges:
                bucket = int(st.median(ranges) // 4) * 4
                bucket_sigma = commissioned.get((camera_id, bucket))

        per_arm = run_line(
            items, project=project, jacobian=jacobian,
            sigma_px=args.sigma_px, q_accel_px=q_accel_px, step_s=step_s,
            gate=args.gate, lag=args.lag, floor_m=args.floor_m,
            static_sigma_px=static_sigma, commissioned_sigma_m=bucket_sigma,
        )
        for arm, records in per_arm.items():
            combined[arm].extend(records)
            by_camera[camera_id][arm].extend(records)

    results = {arm: score(records) for arm, records in sorted(combined.items())}
    per_camera_results = {
        camera: {arm: score(records) for arm, records in sorted(arms.items())}
        for camera, arms in sorted(by_camera.items())
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': 'perception_filter_cascade_arm_comparison.v1',
        'capture': str(args.capture),
        'settings': {
            'sigma_px': args.sigma_px,
            'static_sigma_px': static_sigma,
            'commissioned_buckets': {f'{cam}|{low}-{low + 4}m': value
                                     for (cam, low), value in sorted(commissioned.items())},
            'q_accel_px': q_accel_px,
            'px_per_metre': args.px_per_metre,
            'accel_m_s2': args.accel_m_s2,
            'step_s': step_s,
            'spacing_m': spacing_m,
            'speed_m_s': args.speed_m_s,
            'gate': args.gate,
            'lag': args.lag,
            'floor_m': args.floor_m,
        },
        'lines_used': len(groups),
        'overall': results,
        'per_camera': per_camera_results,
    }
    (args.out_dir / 'arm_comparison.json').write_text(json.dumps(payload, indent=2) + '\n',
                                                      encoding='utf-8')

    print()
    if commissioned:
        arm_e = (f'arm E: commissioned per camera and range, '
                 f'{min(commissioned.values()) * 100:.1f}-'
                 f'{max(commissioned.values()) * 100:.1f} cm')
    else:
        arm_e = f'arm E sigma={static_sigma:.3f} px'
    print(f'lines used: {len(groups)}   sigma_px={args.sigma_px}   {arm_e}   '
          f'floor={args.floor_m * 100:.1f} cm')
    print()
    header = (f'{"arm":14s} {"n":>6s} {"median_cm":>10s} {"rms_cm":>8s} {"p90_cm":>8s} '
              f'{"mean_NSE":>9s} {"med_NSE":>8s} {">4sig_%":>8s}')
    print(header)
    print('-' * len(header))
    for arm, stats in results.items():
        if not stats.get('n'):
            continue
        print(f'{arm:14s} {stats["n"]:6d} {stats["median_err_cm"]:10.2f} '
              f'{stats["rms_err_cm"]:8.2f} {stats["p90_err_cm"]:8.2f} '
              f'{stats["mean_nse"]:9.2f} {stats["median_nse"]:8.2f} '
              f'{stats["beyond_4sigma_pct"]:8.2f}')
    print()
    print('Calibration reading: mean_NSE 2.0 is consistent for a 2-D measurement;')
    print('above 2.0 is overconfident, below is conservative. >4sig_% exposes the tail.')
    print()
    print(f'wrote {args.out_dir / "arm_comparison.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
