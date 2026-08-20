"""What a Bayesian filter does with each reading, and how confidently wrong it gets.

The question is not which reading is more accurate — L1 settled that. It is what a filter
that *learns its own R from the data* concludes about how much to trust each reading, and
whether the belief it then states is honest. That is the question the deployed stack
actually answers, and the box bottom answers it very badly: it learns a small R, states a
tight belief, and sits 6-7 cm away from the truth with every internal check passing.

WHAT IS BEING FILTERED. The same capture as everything else in this folder: 423 held-out
teleported poses, both readings scored on the same images. Those are not a drive, so this
is not a drive replay — there is no odometry here and none is invented. The poses group
into floor cells, and a cell holds up to 6 readings of the *same position* taken at
different headings. So each cell is one honest little estimation problem: the robot stands
still, the camera reads it several times, and the filter has to say where it is and how
sure it is. Cells with at least two readings are used (121 cells / 364 readings for the box
bottom, 122 / 368 for the marked point).

THE MODEL, and it is the notebook's model with the trajectory taken out:

    x_k             the robot's position in cell k, unknown
    y_{k,i} = x_k + e,   e ~ N(0, R)      the readings, all sharing one R
    R ~ inverse-Wishart(Psi_0, nu_0)      the same prior learn_R uses: nu_0 = 6, 5 cm

    q(x_k) exact given R_bar; q(R) conjugate. Coordinate ascent, ELBO after each x step,
    with the two corrections `notebook_model.elbo` applies: E[log|R|] rather than
    log|R_bar|, and -KL(q(R) || p(R)). Those two helpers are IMPORTED from
    experiments/filter_notebook/notebook_model.py, so the objective here is the same
    objective as the notebook's, not a lookalike.

The position prior is flat — the filter is told nothing about where the robot is, only what
the camera says — so the belief after a cell's readings is their mean with covariance R/n,
and the bound is written out in closed form rather than read off a filter's log evidence.
That makes it exact, and every run asserts that it rose on every pass.

For the pictures the same cells are also walked reading by reading, starting at the first
reading with its own covariance R, which gives the innovations and the forecast the middle
row needs. Starting instead from the notebook's 5 cm `INITIAL_SIGMA_M` treats the anchor as
better than it is and inflates the learned R by about 1.7x per axis; measured and rejected,
not assumed.

Nothing in the loop sees ground truth: truth enters only afterwards, to ask whether the
stated belief was honest.

WHY THE LOOP CANNOT SEE THE BOX BOTTOM'S LEAN. Every reading in a cell carries the same
lean, so the lean lands in the cell's *position estimate* and never in a residual. R is
learned from within-cell scatter, which is exactly what a filter's innovations can see.
That is the concealment mechanism of README Part 3, in a form small enough to run.

Outputs -> logs/studies/localization_reading_story/filter_on_both_readings/
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reading_data as rd  # noqa: E402

sys.path.insert(0, str(rd.REPO_ROOT / 'experiments/filter_notebook'))
import notebook_model as nm  # noqa: E402   the ELBO helpers and the priors come from here

CELL_MIN = 2
HEADING_BINS = 8                     # 45-degree bins, as everywhere else in this folder
PRIOR_NU = 6.0                       # learn_R default
PRIOR_SIGMA_M = 0.05                 # learn_R default
INITIAL_SIGMA_M = nm.INITIAL_SIGMA_M  # 0.05 m, the notebook's first-sighting covariance
GATE_CHI2_2DOF = nm.GATE_CHI2_2DOF   # 5.991
PASSES = 15
OUT = rd.REPO_ROOT / 'logs/studies/localization_reading_story/filter_on_both_readings'


# ---------------------------------------------------------------- the data, grouped

def cells(reading, min_readings: int = CELL_MIN) -> list[dict]:
    """One entry per floor cell: its readings (metres), its headings, and the truth.

    `truth` is carried for scoring only; nothing in the filter or the learning touches it.
    """
    x, y = reading.col('x'), reading.col('y')
    est = np.column_stack([reading.col('est_x'), reading.col('est_y')])
    yaw = reading.col('yaw_rad')
    order: dict[tuple[float, float], list[int]] = {}
    for i, (xi, yi) in enumerate(zip(x, y)):
        order.setdefault((round(float(xi), 3), round(float(yi), 3)), []).append(i)
    out = []
    for (cx, cy), idx in order.items():
        if len(idx) < min_readings:
            continue
        out.append({'truth': np.array([cx, cy]), 'y': est[idx], 'yaw': yaw[idx]})
    return out


def heading_aware_cells(reading, min_readings: int = CELL_MIN,
                        bins: int = HEADING_BINS) -> list[dict]:
    """The box bottom with its heading-dependent lean corrected, held out cell by cell.

    Row 4-5 of the ladder tried to correct the box bottom rather than replace it, and the
    strongest form of that is a per-heading correction: the lean swings -2.8 to -8.9 cm with
    heading, so subtract the mean error of the heading bin the robot is in. Two honesty
    conditions:

      * the correction for a cell is fitted WITHOUT that cell -- leave-one-place-out, so
        nothing is corrected using its own errors;
      * the correction needs to know the heading at runtime, which the box bottom itself
        cannot supply. Here it is given the true heading, so this is the BEST such a
        correction can do. The marked-point reading is what would actually supply it
        (3.8 deg median), and wheel odometry could too.

    Everything else -- the cells, the loop, the scoring -- is unchanged.
    """
    x, y = reading.col('x'), reading.col('y')
    est = np.column_stack([reading.col('est_x'), reading.col('est_y')])
    truth = np.column_stack([x, y])
    error = est - truth
    yaw = reading.col('yaw_rad') % (2 * math.pi)
    which_bin = np.minimum((yaw / (2 * math.pi / bins)).astype(int), bins - 1)
    cell_of = np.array([hash((round(float(a), 3), round(float(b), 3))) for a, b in truth])

    corrected = est.copy()
    for i in range(len(est)):
        others = (which_bin == which_bin[i]) & (cell_of != cell_of[i])
        if others.sum() >= 3:
            corrected[i] = est[i] - error[others].mean(axis=0)
        else:                       # too thin to correct: leave it alone, and say so
            corrected[i] = est[i]

    order: dict[tuple[float, float], list[int]] = {}
    for i, (xi, yi) in enumerate(zip(x, y)):
        order.setdefault((round(float(xi), 3), round(float(yi), 3)), []).append(i)
    out = []
    for (cx, cy), idx in order.items():
        if len(idx) < min_readings:
            continue
        out.append({'truth': np.array([cx, cy]), 'y': corrected[idx], 'yaw': yaw[idx]})
    return out


# ---------------------------------------------------------------- one cell, one R

def filter_cell(cell: dict, R: np.ndarray) -> dict:
    """Start at the first reading, absorb the rest, and report what was said and seen."""
    mean = cell['y'][0].copy()
    cov = np.array(R, dtype=float)      # the anchor is a reading, so it carries R
    innovations, shapes, loglik = [], [], 0.0
    for reading in cell['y'][1:]:
        innovation = reading - mean
        S = cov + R
        S_inv = np.linalg.inv(S)
        sign, logdet = np.linalg.slogdet(S)
        loglik += -0.5 * (2 * math.log(2 * math.pi) + logdet
                          + float(innovation @ S_inv @ innovation))
        gain = cov @ S_inv
        mean = mean + gain @ innovation
        cov = cov - gain @ cov
        innovations.append(innovation)
        shapes.append(S)
    return {'mean': mean, 'cov': cov, 'loglik': loglik,
            'innovations': np.asarray(innovations), 'S': np.asarray(shapes),
            'n': len(cell['y']), 'n_updates': len(cell['y']) - 1}


def sweep(cell_list: list[dict], R: np.ndarray) -> dict:
    """Every cell filtered at this R: the log evidence, the innovations, the beliefs."""
    runs = [filter_cell(c, R) for c in cell_list]
    return {
        'log_evidence': float(sum(r['loglik'] for r in runs)),
        'innovations': np.concatenate([r['innovations'] for r in runs]),
        'S': np.concatenate([r['S'] for r in runs]),
        'means': np.asarray([r['mean'] for r in runs]),
        'covs': np.asarray([r['cov'] for r in runs]),
        'n_readings': int(sum(r['n'] for r in runs)),
        'n_likelihood_terms': int(sum(r['n_updates'] for r in runs)),
    }


# ---------------------------------------------------------------- the loop

def sufficient_statistics(cell_list: list[dict]) -> tuple[list[np.ndarray], np.ndarray]:
    """Per cell: how many readings, and their scatter about their own mean.

    Those two are everything the loop can learn from. The scatter about a cell's own mean
    is the only part of the error a filter can see, because whatever is common to all the
    readings of a place is indistinguishable from the place being somewhere else.
    """
    counts = np.array([len(c['y']) for c in cell_list], dtype=float)
    scatters = []
    for cell in cell_list:
        centred = cell['y'] - cell['y'].mean(axis=0)
        scatters.append(centred.T @ centred)
    return scatters, counts


def within_place_scatter(cell_list: list[dict]) -> np.ndarray:
    """The only part of the error a filter can see, in cm^2: scatter about each own mean."""
    scatters, counts = sufficient_statistics(cell_list)
    dof = float(sum(counts) - len(counts))
    return 1e4 * sum(scatters) / dof


def collapsed_log_evidence(cell_list: list[dict], R: np.ndarray) -> float:
    """log p(readings | R) with each cell's position marginalised out (flat prior).

    Closed form, so it needs no filter: for n readings of one place, the position absorbs
    the mean and n-1 residuals are left.
    """
    scatters, counts = sufficient_statistics(cell_list)
    R_inv = np.linalg.inv(R)
    _, logdet = np.linalg.slogdet(R)
    d = 2
    total = 0.0
    for W, n in zip(scatters, counts):
        total += (-0.5 * (n - 1) * d * math.log(2 * math.pi) - 0.5 * d * math.log(n)
                  - 0.5 * (n - 1) * logdet - 0.5 * float(np.trace(W @ R_inv)))
    return float(total)


def learn_R(cell_list: list[dict], *, iterations: int = PASSES,
            prior_nu: float = PRIOR_NU, prior_sigma_m: float = PRIOR_SIGMA_M) -> list[dict]:
    """Coordinate ascent on (q(x), q(R)) -- the notebook's loop, with the driving removed.

    x step: with a flat prior on where each place is, the posterior is the mean of that
    cell's readings with covariance R_bar / n, and R_bar = (E_q[R^-1])^-1 = Psi/nu is
    exactly what the filter should use.

    R step: the conjugate update from the expected residuals, which is what makes R come
    out as the within-place scatter and nothing else.

    The ELBO is written out in full rather than taken from the filter's log evidence: the
    x-dependent part is exact here, so the bound is exact, and it must rise on every pass.
    `plug_in` is the collapsed evidence at the R coming OUT of the pass -- a different
    quantity, and not what the loop climbs.
    """
    from scipy.special import digamma  # noqa: F401  (used inside nm's helpers)
    d = 2
    scatters, counts = sufficient_statistics(cell_list)
    n_total = float(counts.sum())
    Psi_prior = np.eye(d) * (prior_sigma_m ** 2) * prior_nu
    R_bar = Psi_prior / prior_nu
    q = {'Psi': Psi_prior.copy(), 'nu': float(prior_nu)}
    history = []
    for _ in range(iterations):
        # ---- x step at R_bar: posterior covariance per cell
        S = [R_bar / n for n in counts]

        # ---- the bound, at (this q(x), the q(R) whose mean is R_bar)
        e_logdet = nm.expected_log_det_iw(q['Psi'], q['nu'], d)
        e_inv = q['nu'] * np.linalg.inv(q['Psi'])
        bound = -0.5 * n_total * d * math.log(2 * math.pi) - 0.5 * n_total * e_logdet
        for W, n, S_k in zip(scatters, counts, S):
            bound -= 0.5 * float(np.trace((W + n * S_k) @ e_inv))
            sign, logdet_S = np.linalg.slogdet(S_k)
            bound += 0.5 * (d * math.log(2 * math.pi * math.e) + logdet_S)
        bound -= nm.iw_kl_from_prior(q['Psi'], q['nu'], Psi_prior, prior_nu, d)

        # ---- R step
        Psi_q = Psi_prior + sum(W + n * S_k for W, n, S_k in zip(scatters, counts, S))
        nu_q = prior_nu + n_total
        R_next = Psi_q / nu_q

        history.append({
            'R_in': R_bar.copy(),
            'elbo': float(bound),
            'plug_in': collapsed_log_evidence(cell_list, R_next),
            'R_out': R_next.copy(),
            'nu_q': float(nu_q),
            'n_readings': int(n_total),
        })
        R_bar, q = R_next, {'Psi': Psi_q.copy(), 'nu': float(nu_q)}
    return history


# ---------------------------------------------------------------- was R itself honest?

def innovation_consistency(cell_list: list[dict], R: np.ndarray) -> dict:
    """R's OWN test, and the only one it can be held to: does it predict its innovations?

    An honest R makes the squared innovation, divided by the covariance the filter forecast
    for it, average the number of dimensions -- 2 here. This uses no ground truth at all,
    which is the point: it is the test a running filter can actually perform, and it is a
    statement about spread around the prediction, not about whether the prediction is in the
    right place.
    """
    run = sweep(cell_list, R)
    nis = np.array([float(v @ np.linalg.inv(S) @ v)
                    for v, S in zip(run['innovations'], run['S'])])
    return {
        'n': int(len(nis)),
        'mean_nis': float(nis.mean()),          # 2 = R is exactly right
        'gate_pass_pct': float(100 * np.mean(nis <= GATE_CHI2_2DOF)),
    }


# ---------------------------------------------------------------- was the BELIEF honest?

def honesty(cell_list: list[dict], R: np.ndarray) -> dict:
    """What the belief said, against where the robot actually was.

    Ground truth enters HERE and nowhere earlier. `nees` is the squared error in units of
    the belief's own covariance; a filter that states its uncertainty honestly averages 2
    on two axes, and covers the truth 95% of the time.
    """
    runs = [filter_cell(c, R) for c in cell_list]
    errors = np.asarray([r['mean'] - c['truth'] for r, c in zip(runs, cell_list)])
    covs = np.asarray([r['cov'] for r in runs])
    nees = np.asarray([float(e @ np.linalg.inv(P) @ e) for e, P in zip(errors, covs)])
    sigma = np.asarray([math.sqrt(np.trace(P) / 2) for P in covs])
    return {
        'errors_cm': 100 * errors, 'covs_cm2': 1e4 * covs, 'nees': nees,
        'stated_sigma_cm': 100 * sigma,
        'median_error_cm': float(np.median(np.hypot(*(100 * errors).T))),
        'median_stated_sigma_cm': float(np.median(100 * sigma)),
        'mean_nees': float(nees.mean()),
        'coverage_95_pct': float(100 * np.mean(nees <= GATE_CHI2_2DOF)),
        'n_cells': len(cell_list),
    }


WEAK_PRIOR = {'prior_nu': 6.0, 'prior_sigma_m': 0.01}


def prior_sensitivity(cell_list: list[dict]) -> dict:
    """Is the learned R the data's answer or the prior's?

    The 5 cm prior is worth six readings, which is a lot next to a 0.7 cm scatter, so for the
    accurate readings it holds R up. Re-learning under a 1 cm prior settles who is speaking --
    and it matters, because a too-large R buys coverage it has not earned.
    """
    out = {}
    for name, kwargs in (('prior 5 cm (the notebook\'s)', {}),
                         ('prior 1 cm', WEAK_PRIOR)):
        R = learn_R(cell_list, **kwargs)[-1]['R_out']
        told, consistent = honesty(cell_list, R), innovation_consistency(cell_list, R)
        out[name] = {
            'R_cm': [math.sqrt(1e4 * R[0, 0]), math.sqrt(1e4 * R[1, 1])],
            'mean_nis': consistent['mean_nis'],
            'stated_sigma_cm': told['median_stated_sigma_cm'],
            'error_cm': told['median_error_cm'],
            'coverage_95_pct': told['coverage_95_pct'],
        }
    return out


def error_over_time(cell_list: list[dict], R: np.ndarray, max_sightings: int = 6) -> dict:
    """How the belief and its stated band develop as sightings arrive at one place.

    A place is a standing robot, so "one more sighting" is "a third of a second later" at the
    deployed 3 Hz. After each update this records how far the belief actually is from truth
    and how far it SAYS it might be, and takes the median over every place that has that many
    sightings. Ground truth is used for the first quantity only, after the fact.
    """
    error, stated, counts = [], [], []
    for n in range(1, max_sightings + 1):
        errs, bands = [], []
        for cell in cell_list:
            if len(cell['y']) < n:
                continue
            run = filter_cell({'truth': cell['truth'], 'y': cell['y'][:n]}, R)
            errs.append(100 * float(np.hypot(*(run['mean'] - cell['truth']))))
            bands.append(100 * math.sqrt(np.trace(run['cov']) / 2))
        error.append(float(np.median(errs)))
        stated.append(float(np.median(bands)))
        counts.append(len(errs))
    return {'sightings': list(range(1, max_sightings + 1)),
            'median_error_cm': error, 'median_stated_sigma_cm': stated, 'places': counts}


def run(key: str) -> dict:
    """`key` may be a reading name, or 'box_bottom_heading_aware' for the corrected box."""
    heading_aware = key.endswith('_heading_aware')
    reading = rd.load_reading(key.replace('_heading_aware', ''))
    cell_list = heading_aware_cells(reading) if heading_aware else cells(reading)
    history = learn_R(cell_list)
    bound = np.array([h['elbo'] for h in history])
    drop = float(np.min(np.diff(bound)))
    if drop < -1e-6:                     # coordinate ascent cannot lower its own bound
        raise AssertionError(f'{key}: the ELBO fell by {-drop:.3g} -- the pairing of q(x) '
                             f'and q(R) is wrong somewhere')
    final = history[-1]['R_out']
    return {
        'key': key,
        'label': (reading.label + ', lean corrected per heading (heading given)'
                  if heading_aware else reading.label),
        'cells': cell_list, 'history': history,
        'R_final': final, 'honesty': honesty(cell_list, final),
        'honesty_prior': honesty(cell_list, history[0]['R_in']),
        'consistency': innovation_consistency(cell_list, final),
        'prior_sensitivity': prior_sensitivity(cell_list),
        'within_place_scatter_cm2': within_place_scatter(cell_list),
        'truth_errors_cm': 100 * np.concatenate(
            [c['y'] - c['truth'] for c in cell_list]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = {}
    for key in ('box_bottom', 'box_bottom_heading_aware', 'keypoint_retrained'):
        result = run(key)
        h, R = result['honesty'], result['R_final']
        record[key] = {
            'label': result['label'],
            'cells': len(result['cells']),
            'readings': int(sum(len(c['y']) for c in result['cells'])),
            'learned_R_cm': [math.sqrt(1e4 * R[0, 0]), math.sqrt(1e4 * R[1, 1])],
            'learned_R_cm2': (1e4 * R).tolist(),
            'true_error_sd_cm': np.std(result['truth_errors_cm'], axis=0, ddof=1).tolist(),
            'true_error_mean_cm': result['truth_errors_cm'].mean(axis=0).tolist(),
            'elbo_first_last': [result['history'][0]['elbo'], result['history'][-1]['elbo']],
            'within_place_scatter_cm': [math.sqrt(result['within_place_scatter_cm2'][0, 0]),
                                        math.sqrt(result['within_place_scatter_cm2'][1, 1])],
            'honesty': {k: v for k, v in h.items() if not isinstance(v, np.ndarray)},
            'R_own_test': result['consistency'],
            'prior_sensitivity': prior_sensitivity(result['cells']),
        }
        print(f"\n=== {key}: {result['label']}")
        print(f"  {record[key]['cells']} cells, {record[key]['readings']} readings")
        print(f"  learned R      {record[key]['learned_R_cm'][0]:.2f} x "
              f"{record[key]['learned_R_cm'][1]:.2f} cm")
        print(f"  actual error   mean ({result['truth_errors_cm'].mean(axis=0)[0]:+.2f}, "
              f"{result['truth_errors_cm'].mean(axis=0)[1]:+.2f}) cm, sd "
              f"{np.std(result['truth_errors_cm'], axis=0, ddof=1)[0]:.2f} x "
              f"{np.std(result['truth_errors_cm'], axis=0, ddof=1)[1]:.2f} cm")
        print(f"  belief         off by {h['median_error_cm']:.2f} cm while stating "
              f"{h['median_stated_sigma_cm']:.2f} cm")
        print(f"  NEES {h['mean_nees']:.1f} (honest = 2), truth inside its own 95% "
              f"region {h['coverage_95_pct']:.1f}% of the time")
        print(f"  R's own test: mean normalised innovation "
              f"{result['consistency']['mean_nis']:.2f} (2 = R exactly right), "
              f"{result['consistency']['gate_pass_pct']:.1f}% inside the gate")
        for name, row in record[key]['prior_sensitivity'].items():
            print(f"    under {name}: R {row['R_cm'][0]:.2f} x {row['R_cm'][1]:.2f} cm, "
                  f"NIS {row['mean_nis']:.2f}, states {row['stated_sigma_cm']:.2f} cm, "
                  f"coverage {row['coverage_95_pct']:.1f}%")
    (OUT / 'filter_on_both_readings.json').write_text(
        json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f"\nwrote {OUT / 'filter_on_both_readings.json'}")


if __name__ == '__main__':
    main()
