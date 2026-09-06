#!/usr/bin/env python3
"""Compare three perception-side observation models under identical data.

The perception stage is treated as a Bayesian observation estimator, not a denoiser. For
camera `i` the corrected observation is modelled as

    y_t = H xi_t + b_t + v_t,      v_t ~ N(0, R_t)

where `xi_t` is the latent visual state (the ground-contact pixel and its rate), `b_t` is
the systematic error that survives the per-frame correction, and `R_t` is the remaining
random scatter. Writing `b_t` down explicitly is the point: a filter that folds a non-zero
mean into zero-mean noise estimates scatter around a biased trajectory and becomes
confidently wrong, which is the failure this repo already recorded once.

The three models, in the order they must be tried:

  M1 fixed_R      xi only, R fixed. The reference. Any remaining b_t is absorbed as noise.
  M2 bias_aware   xi augmented with a slowly varying b_t (a random walk). R still fixed.
                  This is the state-augmentation form of Friedland's two-stage separation
                  of state and bias in recursive filtering.
  M3 niw_vb       xi and b_t as in M2, plus a Normal-inverse-Wishart belief over the
                  residual mean and covariance, updated by variational Bayes. This is the
                  joint unknown-mean, unknown-covariance treatment.

Identifiability is the live risk in M2 and M3: the visual point and the bias both enter the
observation additively, so they separate only because they move on different timescales.
That is imposed through the ratio of their process noises and reported, never tuned against
the score. The ratio is set from measurement -- the visual point moves at the projection
rate times the robot speed, while the bias is measured to drift far more slowly -- and the
run records what was used.

Scoring is distributional, because the goal is a measurement whose Gaussian description is
true, not merely a small error. Every model reports the whitened residual
`w = R^-1/2 (z - z_true)` and is judged on whether `w ~ N(0, I)`: mean, covariance, the
chi-squared behaviour of the Mahalanobis distance, plus skew, kurtosis and a tail count.
"""
from __future__ import annotations

import argparse
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

from compare_perception_filters import (  # noqa: E402
    identity,
    inv2,
    mat_add,
    mat_mul,
    mat_sub,
    transpose,
)
from separate_bias_and_fast_noise import load_rows  # noqa: E402

TRUE_VALUES = {'True', 'true', '1'}


def inv_n(a: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse for the small square systems used here."""
    size = len(a)
    aug = [a[row][:] + [1.0 if col == row else 0.0 for col in range(size)]
           for row in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-15:
            raise ValueError('singular matrix')
        aug[col], aug[pivot] = aug[pivot], aug[col]
        divisor = aug[col][col]
        aug[col] = [value / divisor for value in aug[col]]
        for other in range(size):
            if other == col:
                continue
            factor = aug[other][col]
            if factor:
                aug[other] = [value - factor * top
                              for value, top in zip(aug[other], aug[col])]
    return [row[size:] for row in aug]


def sqrt_inv2(cov: list[list[float]]) -> list[list[float]]:
    """Inverse square root of a 2x2 covariance, for whitening a residual."""
    a, b, d = cov[0][0], 0.5 * (cov[0][1] + cov[1][0]), cov[1][1]
    trace, det = a + d, a * d - b * b
    disc = max(trace * trace / 4.0 - det, 0.0)
    root = math.sqrt(disc)
    lam1, lam2 = trace / 2.0 + root, trace / 2.0 - root
    if lam1 <= 0.0 or lam2 <= 0.0:
        raise ValueError('covariance is not positive definite')
    if abs(b) < 1e-18:
        return [[1.0 / math.sqrt(a), 0.0], [0.0, 1.0 / math.sqrt(d)]]
    # Eigenvectors of a symmetric 2x2.
    v1 = [b, lam1 - a]
    n1 = math.hypot(*v1) or 1.0
    v1 = [value / n1 for value in v1]
    v2 = [-v1[1], v1[0]]
    s1, s2 = 1.0 / math.sqrt(lam1), 1.0 / math.sqrt(lam2)
    return [
        [s1 * v1[0] * v1[0] + s2 * v2[0] * v2[0], s1 * v1[0] * v1[1] + s2 * v2[0] * v2[1]],
        [s1 * v1[1] * v1[0] + s2 * v2[1] * v2[0], s1 * v1[1] * v1[1] + s2 * v2[1] * v2[1]],
    ]


class ObservationModel:
    """`y_t = H xi_t + b_t + v_t` with an optional bias state and optional NIW belief.

    State layout is `[u, v, du, dv]` followed by `[bu, bv]` when the bias is modelled.
    The bias enters the observation with an identity block, which is what makes it a
    nuisance parameter rather than part of the tracked point.
    """

    def __init__(self, *, mode: str, sigma_px: float, q_accel_px: float,
                 q_bias_px_s: float, step_s: float, niw_kappa: float = 5.0,
                 niw_nu: float = 8.0, niw_forget: float = 0.98) -> None:
        if mode not in {'fixed_R', 'bias_aware', 'niw_vb'}:
            raise ValueError(f'unknown mode {mode!r}')
        self.mode = mode
        self.dt = step_s
        self.q_accel = q_accel_px
        self.q_bias = q_bias_px_s
        self.sigma_px = sigma_px
        self.dim = 6 if mode in {'bias_aware', 'niw_vb'} else 4
        self.state: list[float] = [0.0] * self.dim
        self.cov: list[list[float]] = identity(self.dim)
        # Normal-inverse-Wishart belief over the residual mean and covariance.
        self.niw_m = [0.0, 0.0]
        self.niw_kappa = niw_kappa
        self.niw_nu = niw_nu
        self.niw_psi = [[niw_nu * sigma_px ** 2, 0.0], [0.0, niw_nu * sigma_px ** 2]]
        self.niw_forget = niw_forget

    # -- model matrices ----------------------------------------------------
    def transition(self) -> list[list[float]]:
        dt, dim = self.dt, self.dim
        f = identity(dim)
        f[0][2] = dt
        f[1][3] = dt
        return f

    def process_noise(self) -> list[list[float]]:
        dt, dim = self.dt, self.dim
        var = self.q_accel ** 2
        q = [[0.0] * dim for _ in range(dim)]
        q[0][0] = q[1][1] = dt ** 4 / 4.0 * var
        q[2][2] = q[3][3] = dt ** 2 * var
        q[0][2] = q[2][0] = q[1][3] = q[3][1] = dt ** 3 / 2.0 * var
        if dim == 6:
            # The bias is a random walk, and it must be far slower than the visual point
            # or the two are not separable from one additive observation.
            q[4][4] = q[5][5] = (self.q_bias * dt) ** 2
        return q

    def observation_matrix(self) -> list[list[float]]:
        h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        if self.dim == 6:
            h = [row + [1.0, 0.0] for row in h[:1]] + [h[1] + [0.0, 1.0]]
        return h

    def measurement_cov(self) -> list[list[float]]:
        if self.mode != 'niw_vb':
            return [[self.sigma_px ** 2, 0.0], [0.0, self.sigma_px ** 2]]
        # Posterior mean of the inverse-Wishart belief.
        scale = max(self.niw_nu - 2.0 - 1.0, 1.0)
        return [[self.niw_psi[0][0] / scale, self.niw_psi[0][1] / scale],
                [self.niw_psi[1][0] / scale, self.niw_psi[1][1] / scale]]

    # -- recursion ---------------------------------------------------------
    def initialise(self, u: float, v: float) -> None:
        self.state = [u, v, 0.0, 0.0] + ([0.0, 0.0] if self.dim == 6 else [])
        big = (20.0 * self.sigma_px) ** 2
        cov = [[0.0] * self.dim for _ in range(self.dim)]
        cov[0][0] = cov[1][1] = self.sigma_px ** 2
        cov[2][2] = cov[3][3] = big
        if self.dim == 6:
            # The bias prior has to be wide enough to be learnable but not so wide that
            # it swallows the visual point on the first update.
            cov[4][4] = cov[5][5] = (4.0 * self.sigma_px) ** 2
        self.cov = cov

    def predict(self) -> None:
        f = self.transition()
        self.state = [sum(f[r][c] * self.state[c] for c in range(self.dim))
                      for r in range(self.dim)]
        self.cov = mat_add(mat_mul(mat_mul(f, self.cov), transpose(f)), self.process_noise())

    def update(self, u: float, v: float) -> None:
        h = self.observation_matrix()
        r = self.measurement_cov()
        pred = [sum(h[row][col] * self.state[col] for col in range(self.dim))
                for row in range(2)]
        nu = [u - pred[0], v - pred[1]]
        hph = mat_mul(mat_mul(h, self.cov), transpose(h))
        s = mat_add(hph, r)
        try:
            gain = mat_mul(mat_mul(self.cov, transpose(h)), inv2(s))
        except ValueError:
            return
        self.state = [self.state[row] + sum(gain[row][col] * nu[col] for col in range(2))
                      for row in range(self.dim)]
        keep = mat_sub(identity(self.dim), mat_mul(gain, h))
        cov = mat_mul(mat_mul(keep, self.cov), transpose(keep))
        self.cov = mat_add(cov, mat_mul(mat_mul(gain, r), transpose(gain)))

        if self.mode == 'niw_vb':
            self._update_niw(nu, hph)

    def _update_niw(self, nu: list[float], hph: list[list[float]]) -> None:
        """Update the Normal-inverse-Wishart belief from this innovation.

        The innovation carries both the state-prediction spread and the measurement
        scatter, so the predicted part `hph` is removed before the remainder is credited
        to the measurement covariance. Without that subtraction the estimate absorbs
        prediction uncertainty and inflates without bound.
        """
        forget = self.niw_forget
        self.niw_kappa = forget * self.niw_kappa + 1.0
        self.niw_nu = forget * self.niw_nu + 1.0
        weight = 1.0 / self.niw_kappa
        residual = [nu[0] - self.niw_m[0], nu[1] - self.niw_m[1]]
        self.niw_m = [self.niw_m[0] + weight * residual[0],
                      self.niw_m[1] + weight * residual[1]]
        outer = [[residual[row] * residual[col] for col in range(2)] for row in range(2)]
        psi = [[forget * self.niw_psi[row][col]
                + (1.0 - weight) * outer[row][col]
                - hph[row][col]
                for col in range(2)] for row in range(2)]
        # Keep the scale matrix positive definite: an innovation smaller than the
        # predicted spread is evidence of small measurement noise, not negative noise.
        floor = (0.05 * self.sigma_px) ** 2 * self.niw_nu
        psi[0][0] = max(psi[0][0], floor)
        psi[1][1] = max(psi[1][1], floor)
        off = 0.5 * (psi[0][1] + psi[1][0])
        limit = 0.99 * math.sqrt(psi[0][0] * psi[1][1])
        off = max(-limit, min(limit, off))
        self.niw_psi = [[psi[0][0], off], [off, psi[1][1]]]

    # -- output ------------------------------------------------------------
    def corrected_pixel(self) -> tuple[float, float]:
        """The visual point with the estimated residual bias removed.

        This is the quantity that should be projected: the bias is a property of the
        observation, not of the robot, so it must not travel downstream.
        """
        return self.state[0], self.state[1]

    def pixel_cov(self) -> list[list[float]]:
        cov = [row[:2] for row in self.cov[:2]]
        return [[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]]

    def estimated_bias(self) -> tuple[float, float]:
        if self.dim == 6:
            return self.state[4], self.state[5]
        return 0.0, 0.0


def make_projector(camera_id: str, capture: Path):
    """Ground projection and its numeric Jacobian, from the capture's own camera model."""
    from derive_interpretations import camera_models

    manifest = json.loads((capture / 'capture_manifest.json').read_text(encoding='utf-8'))
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


def gaussianity(records: list[dict[str, float]]) -> dict[str, float]:
    """Is the whitened residual standard normal? Shape and scale, reported separately."""
    if len(records) < 8:
        return {'n': len(records)}
    wx = [row['wx'] for row in records]
    wy = [row['wy'] for row in records]
    both = wx + wy
    maha = [row['maha'] for row in records]

    mean = st.mean(both)
    sd = st.pstdev(both)
    centred = [value - mean for value in both]
    m2 = st.mean([value ** 2 for value in centred]) or 1e-12
    skew = st.mean([value ** 3 for value in centred]) / m2 ** 1.5
    kurtosis = st.mean([value ** 4 for value in centred]) / m2 ** 2 - 3.0

    # Kolmogorov-Smirnov distance of the Mahalanobis distance from chi-squared with 2
    # degrees of freedom, whose CDF is 1 - exp(-x/2). This tests the joint scale.
    ordered = sorted(maha)
    count = len(ordered)
    ks = max(abs((index + 1) / count - (1.0 - math.exp(-value / 2.0)))
             for index, value in enumerate(ordered))

    errors = [math.hypot(row['ex'], row['ey']) for row in records]
    return {
        'n': len(records),
        'whitened_mean': mean,
        'whitened_sd': sd,
        'whitened_skew': skew,
        'whitened_excess_kurtosis': kurtosis,
        'mahalanobis_mean': st.mean(maha),
        'mahalanobis_median': st.median(maha),
        'chi2_ks_distance': ks,
        'beyond_2sigma_pct': 100.0 * sum(1 for value in maha if value > 6.18) / count,
        'beyond_4sigma_pct': 100.0 * sum(1 for value in maha if value > 16.0) / count,
        'median_err_cm': st.median(errors) * 100.0,
        'rms_err_cm': math.sqrt(st.mean([value ** 2 for value in errors])) * 100.0,
    }


def run_sequence(items: list[dict[str, str]], *, mode: str, project, jacobian,
                 reading: str, sigma_px: float, q_accel_px: float, q_bias_px_s: float,
                 step_s: float, floor_m: float) -> list[dict[str, float]]:
    """Run one observation model along one dense line.

    Two readings stand in for the per-frame correction stage, and they are filtered in
    different frames because that is where each one lives:

      `raw`  the bounding-box bottom pixel. Filtered in pixels, then projected. The
             observation noise is a pixel sigma and the Jacobian carries it to the floor.
      `hull` the analytic-hull ground point, which is the corrected reading this repo
             already has. It is produced in the ground frame, not as a pixel offset, so
             it is filtered directly in metres. Its process noise and observation noise
             are the pixel quantities divided by the local projection scale, so the two
             arms make the same physical assumptions in different units.
    """
    observations: list[tuple[float, float, float, float]] = []
    for row in items:
        if reading == 'hull':
            if row.get('hull_valid') not in TRUE_VALUES:
                continue
            text_a, text_b = row.get('hull_x', ''), row.get('hull_y', '')
        else:
            if row.get('raw_valid') not in TRUE_VALUES:
                continue
            text_a, text_b = row.get('u_bbox_bottom', ''), row.get('v_bbox_bottom', '')
        if text_a in ('', 'nan') or text_b in ('', 'nan'):
            continue
        observations.append((float(text_a), float(text_b),
                             float(row['robot_x']), float(row['robot_y'])))
    if len(observations) < 8:
        return []

    if reading == 'hull':
        # Convert the pixel-space noise assumptions into metres using the projection
        # scale at the middle of this line, so nothing is re-tuned per arm.
        mid = observations[len(observations) // 2]
        try:
            jac_mid = jacobian_for_ground(project, jacobian, mid[0], mid[1])
        except ValueError:
            return []
        scale = math.sqrt(abs(jac_mid[0][0] * jac_mid[1][1] - jac_mid[0][1] * jac_mid[1][0]))
        scale = scale if scale > 1e-9 else 1e-9
        model = ObservationModel(mode=mode, sigma_px=sigma_px * scale,
                                 q_accel_px=q_accel_px * scale,
                                 q_bias_px_s=q_bias_px_s * scale, step_s=step_s)
    else:
        model = ObservationModel(mode=mode, sigma_px=sigma_px, q_accel_px=q_accel_px,
                                 q_bias_px_s=q_bias_px_s, step_s=step_s)
    model.initialise(observations[0][0], observations[0][1])

    out: list[dict[str, float]] = []
    for index, (a, b, truth_x, truth_y) in enumerate(observations):
        if index > 0:
            model.predict()
            model.update(a, b)
        mu_a, mu_b = model.corrected_pixel()
        bias_a, bias_b = model.estimated_bias()
        # The bias belongs to the observation, not the robot, so it is removed before the
        # estimate travels downstream.
        est_a, est_b = mu_a - bias_a, mu_b - bias_b
        cov_state = model.pixel_cov()
        cov_meas = model.measurement_cov()
        total = mat_add(cov_state, cov_meas)

        if reading == 'hull':
            x, y = est_a, est_b          # already a ground point
            cov_xy = total               # already in metres
        else:
            try:
                x, y = project(est_a, est_b)
                jac = jacobian(est_a, est_b)
            except ValueError:
                continue
            cov_xy = mat_mul(mat_mul(jac, total), transpose(jac))

        cov_xy = [[cov_xy[0][0] + floor_m ** 2, cov_xy[0][1]],
                  [cov_xy[1][0], cov_xy[1][1] + floor_m ** 2]]
        ex, ey = x - truth_x, y - truth_y
        try:
            whitener = sqrt_inv2(cov_xy)
            inv = inv2(cov_xy)
        except ValueError:
            continue
        out.append({
            'ex': ex, 'ey': ey,
            'wx': whitener[0][0] * ex + whitener[0][1] * ey,
            'wy': whitener[1][0] * ex + whitener[1][1] * ey,
            'maha': ex * (inv[0][0] * ex + inv[0][1] * ey)
                    + ey * (inv[1][0] * ex + inv[1][1] * ey),
            'bias_a': bias_a, 'bias_b': bias_b,
        })
    return out


def jacobian_for_ground(project, jacobian, x: float, y: float) -> list[list[float]]:
    """Projection Jacobian near a ground point, found by inverting the projection locally.

    The hull arm knows a floor position, not a pixel, so the scale that converts pixel
    noise into metres has to be evaluated at the pixel that maps there. A short search is
    enough: the projection is smooth and only the local scale is needed.
    """
    best = None
    for u in range(40, 1280, 80):
        for v in range(40, 720, 40):
            try:
                px, py = project(float(u), float(v))
            except ValueError:
                continue
            distance = math.hypot(px - x, py - y)
            if best is None or distance < best[0]:
                best = (distance, float(u), float(v))
    if best is None:
        raise ValueError('no pixel maps near this ground point')
    return jacobian(best[1], best[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--reading', default='raw', choices=('raw', 'hull'),
                        help='Which per-frame reading stands in for the NN correction.')
    parser.add_argument('--sigma-px', type=float, default=0.764)
    parser.add_argument('--px-per-metre', type=float, default=41.4)
    parser.add_argument('--accel-m-s2', type=float, default=0.5)
    parser.add_argument('--speed-m-s', type=float, default=0.22)
    parser.add_argument('--bias-drift-px-s', type=float, default=None,
                        help='Random-walk rate of the residual bias. Left unset it is set '
                             'from the identifiability ratio below rather than tuned.')
    parser.add_argument('--bias-timescale-ratio', type=float, default=100.0,
                        help='How many times slower the bias must drift than the visual '
                             'point moves. This is what makes the two separable.')
    parser.add_argument('--floor-m', type=float, default=0.0)
    args = parser.parse_args()

    rows = load_rows(args.capture)
    spacing_m = None
    try:
        manifest = json.loads(
            (args.capture / 'capture_manifest.json').read_text(encoding='utf-8'))
        pose_file = Path(manifest.get('plan', {}).get('pose_file', ''))
        spacing_m = float(json.loads(pose_file.read_text(encoding='utf-8'))['step_m'])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise SystemExit('could not read the sample spacing from the capture manifest')
    step_s = spacing_m / args.speed_m_s

    q_accel_px = args.px_per_metre * args.accel_m_s2
    pixel_rate = args.px_per_metre * args.speed_m_s
    q_bias = (args.bias_drift_px_s if args.bias_drift_px_s is not None
              else pixel_rate / args.bias_timescale_ratio)

    print(f'sample interval   : {spacing_m * 100:.0f} cm / {args.speed_m_s} m/s '
          f'= {step_s:.4f} s')
    print(f'visual motion     : {args.px_per_metre} px/m x {args.speed_m_s} m/s '
          f'= {pixel_rate:.2f} px/s')
    print(f'process noise     : {q_accel_px:.1f} px/s^2')
    print(f'bias drift        : {q_bias:.4f} px/s '
          f'(1/{args.bias_timescale_ratio:.0f} of the visual rate, for identifiability)')
    print(f'reading           : {args.reading}')

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get('detected') not in TRUE_VALUES:
            continue
        label = row.get('dataset_split') or ''
        if not label.startswith('line'):
            continue
        groups[(row['camera_id'], label)].append(row)
    if not groups:
        raise SystemExit('no dense-line detections found in the capture')

    modes = ('fixed_R', 'bias_aware', 'niw_vb')
    projectors: dict[str, tuple[object, object]] = {}
    pooled: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_camera: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list))

    for (camera_id, label), items in sorted(groups.items()):
        if camera_id not in projectors:
            projectors[camera_id] = make_projector(camera_id, args.capture)
        project, jacobian = projectors[camera_id]
        axis = 'x' if '_x_' in label else 'y'
        column = 'robot_x' if axis == 'x' else 'robot_y'
        ordered = sorted(items, key=lambda row: float(row[column]))
        for mode in modes:
            records = run_sequence(
                ordered, mode=mode, project=project, jacobian=jacobian,
                reading=args.reading, sigma_px=args.sigma_px, q_accel_px=q_accel_px,
                q_bias_px_s=q_bias, step_s=step_s, floor_m=args.floor_m)
            pooled[mode].extend(records)
            per_camera[camera_id][mode].extend(records)

    results = {mode: gaussianity(records) for mode, records in pooled.items()}
    per_camera_results = {
        camera: {mode: gaussianity(records) for mode, records in sorted(modes_.items())}
        for camera, modes_ in sorted(per_camera.items())
    }

    names = {'fixed_R': 'M1 fixed R (reference)',
             'bias_aware': 'M2 bias-aware state augmentation',
             'niw_vb': 'M3 joint mean and covariance (NIW/VB)'}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': 'perception_filter_cascade_bayesian_models.v1',
        'capture': str(args.capture),
        'reading': args.reading,
        'settings': {
            'sigma_px': args.sigma_px, 'q_accel_px': q_accel_px,
            'q_bias_px_s': q_bias, 'bias_timescale_ratio': args.bias_timescale_ratio,
            'step_s': step_s, 'spacing_m': spacing_m, 'speed_m_s': args.speed_m_s,
            'px_per_metre': args.px_per_metre, 'floor_m': args.floor_m,
        },
        'sequences': len(groups),
        'model_names': names,
        'overall': results,
        'per_camera': per_camera_results,
    }
    out_path = args.out_dir / f'bayesian_models_{args.reading}.json'
    out_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    print()
    print(f'{len(groups)} camera-line sequences, reading = {args.reading}')
    print()
    header = (f'{"model":38s} {"n":>5s} {"med_cm":>7s} {"w_mean":>7s} {"w_sd":>6s} '
              f'{"skew":>6s} {"exkurt":>7s} {"maha_mean":>9s} {"chi2_KS":>8s} '
              f'{">2sig%":>7s} {">4sig%":>7s}')
    print(header)
    print('-' * len(header))
    for mode in modes:
        stats = results.get(mode, {})
        if not stats.get('n') or 'whitened_mean' not in stats:
            continue
        print(f'{names[mode]:38s} {stats["n"]:5d} {stats["median_err_cm"]:7.2f} '
              f'{stats["whitened_mean"]:7.3f} {stats["whitened_sd"]:6.2f} '
              f'{stats["whitened_skew"]:6.2f} {stats["whitened_excess_kurtosis"]:7.2f} '
              f'{stats["mahalanobis_mean"]:9.2f} {stats["chi2_ks_distance"]:8.3f} '
              f'{stats["beyond_2sigma_pct"]:7.2f} {stats["beyond_4sigma_pct"]:7.2f}')
    print()
    print('A truthful Gaussian interface has whitened mean 0, whitened sd 1, skew 0,')
    print('excess kurtosis 0, Mahalanobis mean 2.0, small chi-squared KS distance,')
    print('about 5% beyond 2 sigma and almost nothing beyond 4 sigma.')
    print()
    print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
