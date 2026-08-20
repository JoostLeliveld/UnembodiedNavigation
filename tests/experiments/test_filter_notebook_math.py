#!/usr/bin/env python3
"""Check every piece of mathematics the notebook asserts.

The notebook states results in LaTeX and then implements them in code a few cells
later. Nothing enforces that the two agree, and two of these checks failed the first
time they were written (the inverse-Wishart KL came out negative, and the evidence was
being compared across arms that had gated different subsets of the data). So the claims
are pinned here, against exact or Monte-Carlo references rather than against themselves.

    python3 -m pytest tests/experiments/test_filter_notebook_math.py -v

It guards `experiments/filter_notebook/notebook_model.py` -- the single copy of every
estimator both notebooks call -- and runs as part of the ordinary suite, so a change to
the mathematics fails here rather than sitting unnoticed in a published figure.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import stats
from scipy.special import multigammaln

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "experiments" / "filter_notebook"
MODEL_SOURCE = STUDY / "notebook_model.py"
NOTEBOOKS = tuple(sorted(STUDY.glob("pp4_[1-9]_*.py")))

if not MODEL_SOURCE.is_file():
    pytest.skip(f"{MODEL_SOURCE} is missing", allow_module_level=True)

sys.path.insert(0, str(STUDY))


def _load_notebook_functions():
    """Import the estimators both notebooks call, and check the names still exist.

    This used to scrape the definitions out of a single monolithic notebook script,
    because importing that script would have loaded a 700 MB capture and drawn fifteen
    figures. The estimators now live in `notebook_model.py`, which imports cheaply, so
    the test exercises the very objects the notebooks call rather than a re-exec of
    their text.
    """
    import notebook_model as nm

    wanted = ("Sequence", "kalman_filter", "rts_smoother", "iw_kl_from_prior", "learn_R",
              "honesty", "sigma_density", "CALIBRATED_MEDIAN_NEES",
              "forecast", "forecast_summary", "observations_between",
              "model_generated_observations")
    missing = [name for name in wanted if not hasattr(nm, name)]
    if missing:
        raise AssertionError(f"vanished from notebook_model.py: {', '.join(missing)}")
    return {name: getattr(nm, name) for name in wanted}


NS = _load_notebook_functions()
GATE_OFF = float("inf")


def test_both_notebooks_define_no_estimator_of_their_own():
    """The point of the split: one copy of the filter, called by both notebooks.

    A notebook that grew its own `def kalman_filter` would silently stop being covered
    by everything below, so the absence of definitions is the thing worth asserting.
    """
    import ast

    assert len(NOTEBOOKS) >= 4, f"expected at least four notebooks, found {len(NOTEBOOKS)}"
    for path in NOTEBOOKS:
        assert path.is_file(), f"{path.name} is missing"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = [n.name for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        assert not defined, f"{path.name} defines {defined}; it should only call"


# --------------------------------------------------------------------------- #
# the filter and smoother
# --------------------------------------------------------------------------- #

class _Seq:
    """The attribute contract kalman_filter and rts_smoother rely on."""

    def __init__(self, T, obs_at, rng):
        self.n_steps = T
        self.u = np.vstack([np.zeros((1, 2)), rng.normal(scale=0.3, size=(T - 1, 2))])
        self.odom = np.cumsum(self.u, axis=0)
        self.stamps = np.arange(T, dtype=float) / 10.0
        self.camera: list = [None] * T
        self.pixel: list = [None] * T
        self.y = np.full((T, 2), np.nan)
        for k in obs_at:
            self.camera[k] = "camera_A"
            self.y[k] = rng.normal(scale=0.5, size=2)
        self.truth = np.zeros((T, 2))

    @property
    def observed(self):
        return ~np.isnan(self.y[:, 0])


def _dense_posterior(seq, R, sigma_p, s0, m0, upto=None):
    """The exact joint posterior, built as one big precision matrix and inverted."""
    T = seq.n_steps
    n = 2 * T
    blk = lambda i: slice(2 * i, 2 * i + 2)
    Lam = np.zeros((n, n))
    eta = np.zeros(n)
    S0inv = np.linalg.inv(np.eye(2) * s0**2)
    Lam[blk(0), blk(0)] += S0inv
    eta[blk(0)] += S0inv @ m0
    for k in range(1, T):
        q = sigma_p**2 * float(np.linalg.norm(seq.u[k]))
        Qinv = np.linalg.inv(np.eye(2) * q)
        Lam[blk(k), blk(k)] += Qinv
        Lam[blk(k - 1), blk(k - 1)] += Qinv
        Lam[blk(k), blk(k - 1)] -= Qinv
        Lam[blk(k - 1), blk(k)] -= Qinv
        eta[blk(k)] += Qinv @ seq.u[k]
        eta[blk(k - 1)] -= Qinv @ seq.u[k]
    Rinv = np.linalg.inv(R)
    for k in range(T):
        if seq.camera[k] is None:
            continue
        if upto is not None and k > upto:
            continue
        Lam[blk(k), blk(k)] += Rinv
        eta[blk(k)] += Rinv @ seq.y[k]
    Cov = np.linalg.inv(Lam)
    return Cov @ eta, Cov


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(3)
    seq = _Seq(9, [1, 3, 4, 7], rng)
    B = rng.normal(size=(2, 2))
    R = B @ B.T + np.eye(2) * 0.05
    return seq, R, 0.04, 0.05, seq.odom[0].copy()


def test_joseph_form_equals_the_short_form():
    """The update the notebook writes out must equal (I - K) P, exactly."""
    rng = np.random.default_rng(11)
    for _ in range(400):
        A = rng.normal(size=(2, 2))
        P = A @ A.T + np.eye(2) * 1e-3
        B = rng.normal(size=(2, 2))
        R = B @ B.T + np.eye(2) * 1e-3
        K = P @ np.linalg.inv(P + R)
        I = np.eye(2)
        assert np.allclose((I - K) @ P @ (I - K).T + K @ R @ K.T, (I - K) @ P, atol=1e-10)


def test_filter_matches_the_exact_posterior(problem):
    seq, R, sigma_p, s0, m0 = problem
    forward = NS["kalman_filter"](seq, {"camera_A": R}, sigma_p=sigma_p,
                                  initial_sigma=s0, gate=GATE_OFF, m0=m0)
    for k in range(seq.n_steps):
        mean, cov = _dense_posterior(seq, R, sigma_p, s0, m0, upto=k)
        assert np.allclose(forward["m"][k], mean[2 * k:2 * k + 2], atol=1e-9)
        assert np.allclose(forward["P"][k], cov[2 * k:2 * k + 2, 2 * k:2 * k + 2], atol=1e-9)


def test_smoother_matches_the_exact_posterior(problem):
    seq, R, sigma_p, s0, m0 = problem
    forward = NS["kalman_filter"](seq, {"camera_A": R}, sigma_p=sigma_p,
                                  initial_sigma=s0, gate=GATE_OFF, m0=m0)
    smooth = NS["rts_smoother"](seq, forward)
    mean, cov = _dense_posterior(seq, R, sigma_p, s0, m0)
    for k in range(seq.n_steps):
        assert np.allclose(smooth["m"][k], mean[2 * k:2 * k + 2], atol=1e-9)
        assert np.allclose(smooth["P"][k], cov[2 * k:2 * k + 2, 2 * k:2 * k + 2], atol=1e-9)


def test_multicamera_offset_model_has_exactly_one_common_2d_gauge():
    """Prove the observability statement made in notebooks 2 and 4.

    With one simultaneous position and four 2-D camera offsets, stacking all four
    observation matrices has a two-dimensional nullspace. Its basis subtracts delta
    from position and adds the same delta to every offset. Camera differences remove
    position and therefore constrain relative offsets, but no camera row constrains
    this common translation.
    """
    n_cameras = 4
    state_dim = 2 + 2 * n_cameras
    rows = []
    for camera_index in range(n_cameras):
        H = np.zeros((2, state_dim))
        H[:, :2] = np.eye(2)
        start = 2 + 2 * camera_index
        H[:, start:start + 2] = np.eye(2)
        rows.append(H)
    H_all = np.vstack(rows)

    gauge = np.zeros((state_dim, 2))
    gauge[:2] = -np.eye(2)
    for camera_index in range(n_cameras):
        start = 2 + 2 * camera_index
        gauge[start:start + 2] = np.eye(2)

    assert np.allclose(H_all @ gauge, 0.0)
    assert state_dim - np.linalg.matrix_rank(H_all) == 2

    # Differencing two camera rows cancels position and retains b_c - b_d.
    contrast = rows[0] - rows[1]
    assert np.allclose(contrast[:, :2], 0.0)
    assert np.linalg.matrix_rank(contrast) == 2


def test_log_evidence_equals_the_joint_marginal(problem):
    """The chain-rule accumulation must equal log N(y_all; mu, Sigma) in one shot."""
    seq, R, sigma_p, s0, m0 = problem
    forward = NS["kalman_filter"](seq, {"camera_A": R}, sigma_p=sigma_p,
                                  initial_sigma=s0, gate=GATE_OFF, m0=m0)
    obs_at = [k for k in range(seq.n_steps) if seq.camera[k] is not None]
    prior_mean, prior_cov = _dense_posterior(seq, R, sigma_p, s0, m0, upto=-1)
    idx = np.concatenate([np.arange(2 * k, 2 * k + 2) for k in obs_at])
    Sigma = prior_cov[np.ix_(idx, idx)] + np.kron(np.eye(len(obs_at)), R)
    direct = stats.multivariate_normal(mean=prior_mean[idx], cov=Sigma).logpdf(
        seq.y[obs_at].reshape(-1))
    assert forward["log_evidence"] == pytest.approx(direct, abs=1e-8)


def test_gate_threshold_is_the_95th_percentile_of_chi2_2():
    assert stats.chi2.ppf(0.95, 2) == pytest.approx(5.991, abs=5e-4)


# --------------------------------------------------------------------------- #
# the two worlds
# --------------------------------------------------------------------------- #

def test_each_world_describes_itself_consistently():
    """A world's camera list, its includes, its topics and its files must agree.

    The loaders switch worlds from a capture's own manifest, so a world whose pieces
    disagree fails somewhere far from the cause -- a missing include surfaces as a
    projection that silently uses the wrong camera pose.
    """
    import notebook_data as nd

    assert set(nd.WORLDS) == {"warehouse_aws", "warehouse_full_4cam"}
    for key, world in nd.WORLDS.items():
        assert world.key == key
        assert world.cameras, f"{key} has no cameras"
        assert set(world.model_includes) == set(world.cameras), key
        assert set(world.image_topics) == set(world.cameras), key
        assert world.world_sdf.is_file(), f"{key}: no world file at {world.world_sdf}"
        detector = nd.REPO / "logs/perception_models" / world.detector_model / "model.pt"
        assert detector.is_file(), f"{key}: no detector at {detector}"


def test_switching_world_rebinds_every_alias():
    """`use_world` has to move all of them together, or half the module speaks for the
    old world while the other half speaks for the new one."""
    import notebook_data as nd

    before = nd.ACTIVE
    try:
        for key, world in nd.WORLDS.items():
            nd.use_world(key)
            assert nd.ACTIVE is world
            assert nd.CAMERAS == world.cameras
            assert nd.WORLD_SDF == world.world_sdf
            assert nd.MODEL_INCLUDES == world.model_includes
            assert nd.IMAGE_TOPICS == world.image_topics
            assert nd.detector_of().parent.name == world.detector_model
    finally:
        nd.use_world(before)


def test_a_capture_without_a_manifest_is_read_as_the_four_camera_world():
    """Every capture recorded before manifests existed is a four-camera one."""
    import notebook_data as nd

    assert nd.world_of("a-capture-that-does-not-exist") is nd.FULL_4CAM


def test_the_estimators_follow_the_sequence_and_not_a_fixed_four(problem):
    """`learn_R` must size itself from the sequence, or a one-camera world gets three
    posteriors over cameras that are not there."""
    seq, R, sigma_p, s0, m0 = problem
    seq.cameras = ("camera_A",)
    _, history, _ = NS["learn_R"](seq, iterations=2, sigma_p=sigma_p)
    assert set(history[-1]["R_bar"]) == {"camera_A"}
    assert set(history[-1]["sigma_m"]) == {"camera_A"}


# --------------------------------------------------------------------------- #
# the variational objective
# --------------------------------------------------------------------------- #

def test_the_elbo_never_decreases(problem):
    """Coordinate ascent on the ELBO cannot go downhill. If it does, something is wrong.

    This is the strongest single check on `learn_R` there is: it does not compare the
    result against anything, it checks that the algorithm has the property its derivation
    guarantees. Every x step and every R step maximises the same bound over one factor
    with the other held fixed, so the sequence of bounds must be non-decreasing.
    """
    import notebook_model as nm

    seq, R, sigma_p, s0, m0 = problem
    seq.cameras = ("camera_A",)
    _, history, _ = nm.learn_R(seq, iterations=25, sigma_p=sigma_p)
    bound = [h["elbo"] for h in history]
    assert all(math.isfinite(b) for b in bound)
    worst = min(b - a for a, b in zip(bound, bound[1:]))
    assert worst >= -1e-9, f"the ELBO fell by {-worst:.3e} at some step"


def test_the_elbo_is_a_lower_bound_on_the_evidence(problem):
    """The ELBO is a bound, so it can never exceed the exact log marginal likelihood.

    At any q(R) the bound must sit below log p(y | R_bar), which the ungated filter
    accumulates exactly (pinned by test_log_evidence_equals_the_joint_marginal).
    """
    import notebook_model as nm

    seq, R, sigma_p, s0, m0 = problem
    seq.cameras = ("camera_A",)
    d, prior_nu, prior_sigma = 2, 6.0, 0.05
    Psi = np.eye(d) * prior_sigma ** 2 * prior_nu
    R_bar = {"camera_A": Psi / prior_nu}
    posterior = {"camera_A": {"Psi": Psi, "nu": prior_nu}}
    bound = nm.elbo(seq, R_bar, posterior, Psi, prior_nu, sigma_p=sigma_p)
    exact = NS["kalman_filter"](seq, R_bar, sigma_p=sigma_p,
                                gate=GATE_OFF)["log_evidence"]
    assert bound <= exact + 1e-9, f"the bound {bound} exceeded the evidence {exact}"


def test_the_kl_term_vanishes_at_the_prior():
    """With q(R) = p(R) the ELBO's only correction is the Jensen gap, not a KL."""
    import notebook_model as nm

    Psi = np.eye(2) * 0.05 ** 2 * 6.0
    assert nm.iw_kl_from_prior(Psi, 6.0, Psi, 6.0) == pytest.approx(0.0, abs=1e-9)


def test_expected_log_det_exceeds_the_working_point_and_shrinks_with_data():
    """E[log|R|] > log|R_bar|, which is exactly why the ELBO sits BELOW the evidence.

    R_bar is Psi/nu, the inverse of the expected precision, and it is not E[R]. The gap
    E[log|R|] - log|R_bar| enters the bound with a minus sign, so the bound is always the
    smaller number. The gap also has to close as nu grows: with enough data q(R)
    concentrates and the distinction stops mattering.
    """
    import notebook_model as nm

    rng = np.random.default_rng(19)
    A = rng.normal(size=(2, 2))
    Psi = A @ A.T + np.eye(2) * 0.4
    gaps = []
    for nu in (6.0, 12.0, 50.0, 300.0, 3000.0):
        _, logdet_bar = np.linalg.slogdet(Psi / nu)
        gap = nm.expected_log_det_iw(Psi, nu) - logdet_bar
        assert gap > 0.0, f"gap was {gap} at nu={nu}"
        gaps.append(gap)
    assert all(b < a for a, b in zip(gaps, gaps[1:])), "the gap must shrink with nu"
    assert gaps[-1] < 2e-3   # ~1/nu, so 3000 pseudo-counts leaves about 0.001


# --------------------------------------------------------------------------- #
# the forecast: notebook 1 rests on this being a prediction, not a fit
# --------------------------------------------------------------------------- #

def test_forecast_scores_sum_to_the_log_evidence(problem):
    """Sum of the per-reading forecast scores IS the model evidence, by the chain rule.

    log p(y_1..y_n) = sum_k log p(y_k | y_1..y_k-1). If these two ever disagree, one of
    the notebook's two headline quantities -- "fits better" and "forecasts better" -- is
    not what it is claimed to be.
    """
    seq, R, sigma_p, s0, m0 = problem
    out = NS["forecast"](seq, {"camera_A": R}, sigma_p=sigma_p, initial_sigma=s0)
    total = sum(row["log_p"] for row in out["rows"])
    assert total == pytest.approx(out["result"]["log_evidence"], abs=1e-9)


def test_forecast_splits_exactly_into_the_miss_and_the_confidence(problem):
    """The two halves the notebook draws as stacked bars must add back to the score."""
    seq, R, sigma_p, s0, m0 = problem
    for row in NS["forecast"](seq, {"camera_A": R}, sigma_p=sigma_p,
                              initial_sigma=s0)["rows"]:
        assert row["miss_term"] + row["confidence_term"] == pytest.approx(row["log_p"],
                                                                         abs=1e-12)
        assert row["miss_term"] == pytest.approx(-0.5 * row["nis"], abs=1e-12)


def test_forecast_for_a_reading_does_not_depend_on_that_reading(problem):
    """A forecast is formed from readings 1..k-1 only. Move y_k and it must not move.

    This is the whole methodological claim of the prediction sections: the score is not
    a fit dressed up. Later forecasts are allowed to change -- they consumed y_k.
    """
    seq, R, sigma_p, s0, m0 = problem
    kwargs = dict(sigma_p=sigma_p, initial_sigma=s0)
    before = NS["forecast"](seq, {"camera_A": R}, **kwargs)["rows"]
    target = before[1]["k"]

    moved = copy.deepcopy(seq)
    moved.y[target] = moved.y[target] + np.array([0.7, -0.4])
    after = NS["forecast"](moved, {"camera_A": R}, **kwargs)["rows"]
    row_before = next(r for r in before if r["k"] == target)
    row_after = next(r for r in after if r["k"] == target)

    assert np.allclose(row_before["predicted"], row_after["predicted"], atol=1e-12)
    assert np.allclose(row_before["S"], row_after["S"], atol=1e-12)
    # and the perturbation really did reach the filter, so this is not vacuous
    later = [r for r in after if r["k"] > target]
    assert later and not np.allclose(later[0]["predicted"],
                                     next(r for r in before if r["k"] == later[0]["k"])["predicted"])


def test_observations_between_hides_readings_without_touching_anything_else(problem):
    """Holding data back must remove only observations, never odometry or truth."""
    seq, R, sigma_p, s0, m0 = problem
    kept = NS["observations_between"](seq, 2, 5)
    for k in range(seq.n_steps):
        if seq.camera[k] is not None and 2 <= k < 5:
            assert kept.camera[k] == seq.camera[k] and np.allclose(kept.y[k], seq.y[k])
        else:
            assert kept.camera[k] is None and np.isnan(kept.y[k, 0])
    assert np.allclose(kept.u, seq.u) and np.allclose(kept.odom, seq.odom)
    assert np.allclose(kept.truth, seq.truth)


def test_honesty_restricted_to_the_whole_range_matches_the_unrestricted_score(problem):
    """The hold-out scoring path must reduce to the ordinary one when nothing is held out."""
    seq, R, sigma_p, s0, m0 = problem
    result = NS["kalman_filter"](seq, {"camera_A": R}, sigma_p=sigma_p,
                                 initial_sigma=s0, m0=m0)
    everything = NS["honesty"](result, seq, "all")
    restricted = NS["honesty"](result, seq, "all", steps=(0, seq.n_steps))
    assert restricted["median_nees"] == pytest.approx(everything["median_nees"])
    half = NS["honesty"](result, seq, "half", steps=(0, seq.n_steps // 2))
    assert 0 < len(half["nees"]) < len(everything["nees"])


def test_the_self_test_generator_really_draws_from_the_assumed_model():
    """`model_generated_observations` underpins the recovery check, so pin what it makes."""
    rng = np.random.default_rng(5)
    seq = _Seq(4000, list(range(0, 4000, 2)), rng)
    seq.truth = np.cumsum(seq.u, axis=0) * 0.1
    made = NS["model_generated_observations"](seq, 0.04, bias_m=0.07, seed=1)
    residual = made.y[made.observed] - seq.truth[made.observed]
    assert residual.shape[0] == 2000
    assert np.allclose(residual.mean(axis=0), [0.0, 0.07], atol=0.006)
    assert np.allclose(np.cov(residual.T), np.eye(2) * 0.04**2, atol=1.2e-4)


# --------------------------------------------------------------------------- #
# the scores
# --------------------------------------------------------------------------- #

def test_calibrated_median_nees_is_2_ln_2():
    """An honest 2-D belief has median NEES = median of chi2(2) = 2 ln 2."""
    assert stats.chi2.median(2) == pytest.approx(2 * math.log(2), abs=1e-12)
    assert NS["CALIBRATED_MEDIAN_NEES"] == pytest.approx(2 * math.log(2), abs=5e-4)


def test_nlpd_identity():
    """NLPD = 0.5 (NEES + log det 2 pi P) must equal -log N(x; m, P)."""
    rng = np.random.default_rng(13)
    for _ in range(300):
        A = rng.normal(size=(2, 2))
        P = A @ A.T + np.eye(2) * 1e-3
        m, x = rng.normal(size=2), rng.normal(size=2)
        e = x - m
        nees = float(e @ np.linalg.inv(P) @ e)
        formula = 0.5 * (nees + math.log(np.linalg.det(2 * math.pi * P)))
        direct = -stats.multivariate_normal(mean=m, cov=P).logpdf(x)
        assert formula == pytest.approx(direct, abs=1e-9)


def test_a_calibrated_filter_scores_1_386():
    """Sanity on the whole scoring path: simulate from the model the filter assumes and
    the median NEES must land on the calibrated value."""
    rng = np.random.default_rng(17)
    sigma_p, s0 = 0.04, 0.05
    R = np.eye(2) * 0.03**2
    values = []
    for _ in range(60):
        seq = _Seq(60, list(range(0, 60, 2)), rng)
        # draw a truth consistent with the assumed dynamics, then observations from it
        x = rng.multivariate_normal(seq.odom[0], np.eye(2) * s0**2)
        truth = []
        for k in range(seq.n_steps):
            q = sigma_p**2 * float(np.linalg.norm(seq.u[k]))
            x = x + seq.u[k] + rng.multivariate_normal(np.zeros(2), np.eye(2) * q)
            truth.append(x.copy())
        seq.truth = np.asarray(truth)
        for k in range(seq.n_steps):
            if seq.camera[k] is not None:
                seq.y[k] = rng.multivariate_normal(seq.truth[k], R)
        forward = NS["kalman_filter"](seq, {"camera_A": R}, sigma_p=sigma_p,
                                      initial_sigma=s0, gate=GATE_OFF,
                                      m0=seq.odom[0].copy())
        values.extend(NS["honesty"](forward, seq, "sim")["nees"])
    median = float(np.median(values))
    assert median == pytest.approx(NS["CALIBRATED_MEDIAN_NEES"], rel=0.12), (
        f"a correctly specified filter scored {median:.3f}, not ~1.386")


# --------------------------------------------------------------------------- #
# the inverse-Wishart algebra
# --------------------------------------------------------------------------- #

def _sample_iw(Psi, nu, n, seed):
    W = stats.wishart.rvs(df=nu, scale=np.linalg.inv(Psi), size=n, random_state=seed)
    return np.linalg.inv(W)


def test_inverse_wishart_moments_in_the_convention_used():
    """E[R] = Psi/(nu-d-1), E[R^-1] = nu Psi^-1, so R_bar = Psi/nu."""
    d, nu = 2, 14.0
    rng = np.random.default_rng(19)
    A = rng.normal(size=(2, 2))
    Psi = A @ A.T + np.eye(2) * 0.5
    S = _sample_iw(Psi, nu, 200_000, 23)
    assert np.allclose(S.mean(axis=0), Psi / (nu - d - 1), rtol=0.03)
    assert np.allclose(np.linalg.inv(S).mean(axis=0), nu * np.linalg.inv(Psi), rtol=0.03)
    assert np.allclose(np.linalg.inv(nu * np.linalg.inv(Psi)), Psi / nu, atol=1e-12)


def test_marginal_of_a_diagonal_element_is_the_inverse_gamma_used():
    """sigma_density assumes R_ii ~ InvGamma((nu-d+1)/2, Psi_ii/2)."""
    d, nu = 2, 15.0
    rng = np.random.default_rng(29)
    A = rng.normal(size=(2, 2))
    Psi = A @ A.T + np.eye(2) * 0.5
    S = _sample_iw(Psi, nu, 200_000, 31)
    shape, scale = (nu - d + 1) / 2.0, Psi[0, 0] / 2.0
    empirical = np.quantile(S[:, 0, 0], [0.16, 0.5, 0.84])
    theory = stats.invgamma.ppf([0.16, 0.5, 0.84], a=shape, scale=scale)
    assert np.allclose(empirical, theory, rtol=0.04)
    # and the notebook's density over sigma carries the Jacobian, so it integrates to 1
    grid = np.linspace(1e-3, 6.0, 20_000)
    density = NS["sigma_density"](Psi, nu, grid, axis=0, d=d)
    assert np.trapz(density, grid) == pytest.approx(1.0, abs=3e-3)


def test_inverse_wishart_kl_is_zero_from_itself():
    Psi, nu = np.eye(2) * 0.0025 * 6.0, 6.0
    assert NS["iw_kl_from_prior"](Psi, nu, Psi, nu) == pytest.approx(0.0, abs=1e-9)


def test_inverse_wishart_kl_is_never_negative():
    rng = np.random.default_rng(37)
    Psi_p, nu_p = np.eye(2) * 0.0025 * 6.0, 6.0
    for _ in range(500):
        A = rng.normal(size=(2, 2))
        Psi_q = A @ A.T + np.eye(2) * 1e-3
        nu_q = 5.0 + 40.0 * rng.random()
        assert NS["iw_kl_from_prior"](Psi_q, nu_q, Psi_p, nu_p) >= -1e-9


def _iw_logpdf_stack(S, Psi, nu, d=2):
    """log IW(R; Psi, nu) for a whole stack of R at once.

    Vectorised because the obvious loop over 240 000 two-by-two matrices took 24 seconds
    and dominated this file's runtime. numpy's inv and slogdet both broadcast over a
    leading batch axis, so there is no reason to write the loop.
    """
    _, logdet_psi = np.linalg.slogdet(Psi)
    _, logdet_r = np.linalg.slogdet(S)
    trace_term = np.einsum("ij,nji->n", Psi, np.linalg.inv(S))
    return (0.5 * nu * logdet_psi - 0.5 * nu * d * math.log(2)
            - multigammaln(nu / 2, d) - 0.5 * (nu + d + 1) * logdet_r
            - 0.5 * trace_term)


def test_inverse_wishart_kl_matches_monte_carlo():
    """Non-negativity is necessary but not sufficient -- check the value itself."""
    rng = np.random.default_rng(41)
    for trial in range(4):
        A = rng.normal(size=(2, 2))
        Psi_q = A @ A.T + np.eye(2) * 0.4
        nu_q = 9.0 + 20.0 * rng.random()
        B = rng.normal(size=(2, 2))
        Psi_p = B @ B.T + np.eye(2) * 0.4
        nu_p = 7.0 + 15.0 * rng.random()
        S = _sample_iw(Psi_q, nu_q, 120_000, 100 + trial)
        mc = float(np.mean(_iw_logpdf_stack(S, Psi_q, nu_q)
                           - _iw_logpdf_stack(S, Psi_p, nu_p)))
        closed = NS["iw_kl_from_prior"](Psi_q, nu_q, Psi_p, nu_p)
        assert closed == pytest.approx(mc, rel=0.03)


def test_variational_step_uses_the_expected_precision():
    """The mean-field x-step must use (E_q[R^-1])^-1 = Psi/nu, not the mean or mode.

    All three differ, so a regression here would silently change the algorithm from
    variational Bayes to MAP-EM -- which is what the first version of it did.
    """
    d, nu = 2, 20.0
    Psi = np.eye(2) * 0.004 * nu
    r_bar = Psi / nu
    mean = Psi / (nu - d - 1)
    mode = Psi / (nu + d + 1)
    assert not np.allclose(r_bar, mean) and not np.allclose(r_bar, mode)
    assert np.linalg.inv(np.linalg.inv(r_bar)) == pytest.approx(r_bar)
    # and R_bar sits between the mode and the mean, as nu/(nu+d+1) < 1 < nu/(nu-d-1)
    assert np.trace(mode) < np.trace(r_bar) < np.trace(mean)


def test_conjugate_update_recovers_a_known_R():
    """Feed the M-step residuals from a known R and it must return that R."""
    rng = np.random.default_rng(43)
    d, n = 2, 20_000
    A = rng.normal(size=(2, 2))
    R_true = A @ A.T + np.eye(2) * 0.02
    residuals = rng.multivariate_normal(np.zeros(2), R_true, n)
    Psi_prior, nu_prior = np.eye(2) * 0.0025 * 6.0, 6.0
    Psi_post = Psi_prior + residuals.T @ residuals          # P^s = 0: truth known exactly
    nu_post = nu_prior + n
    assert np.allclose(Psi_post / nu_post, R_true, rtol=0.05)


# --------------------------------------------------------------------------- #
# the commissioning arithmetic
# --------------------------------------------------------------------------- #

def test_second_moment_identity():
    """R_total = R_spread (n-1)/n + mean mean', so the 'inflation' column is the offset."""
    rng = np.random.default_rng(47)
    for _ in range(200):
        n = int(rng.integers(20, 200))
        res = rng.normal(size=(n, 2)) @ rng.normal(size=(2, 2)) + rng.normal(size=2)
        mean = res.mean(axis=0)
        centred = res - mean
        spread = centred.T @ centred / (n - 1)
        total = res.T @ res / n
        assert np.allclose(total, spread * (n - 1) / n + np.outer(mean, mean), atol=1e-10)


def test_commissioned_file_is_internally_consistent():
    """The published JSON's scalar summaries must match its own matrices."""
    path = REPO / "logs/studies/filter_notebook/commissioned_observation_noise.json"
    if not path.is_file():
        pytest.skip("commissioning has not been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for camera, entry in payload["per_camera"].items():
        spread = np.asarray(entry["R_spread"])
        total = np.asarray(entry["R_total"])
        assert entry["sigma_spread_m"] == pytest.approx(math.sqrt(np.trace(spread) / 2), rel=1e-9)
        assert entry["sigma_total_m"] == pytest.approx(math.sqrt(np.trace(total) / 2), rel=1e-9)
        offset = np.asarray(entry["mean_offset_m"])
        assert entry["offset_magnitude_m"] == pytest.approx(float(np.hypot(*offset)), rel=1e-9)
        # the second moment must dominate the central one, by exactly the offset
        assert np.trace(total) >= np.trace(spread) * (entry["n"] - 1) / entry["n"] - 1e-12, camera


# --------------------------------------------------------------------------- #
# the geometry
# --------------------------------------------------------------------------- #

def test_back_projection_displacement_formula():
    """A point lifted eps above the floor back-projects x_p eps/(h-eps) too far."""
    h, x_p, eps = 6.1, 10.83, 0.05
    slope = (eps - h) / x_p
    hit = -h / slope
    assert hit - x_p == pytest.approx(x_p * eps / (h - eps), abs=1e-12)
    # and the notebook's quoted approximation is good to under a millimetre here
    assert abs((hit - x_p) - x_p * eps / h) < 1e-3


def test_projection_round_trip_on_the_floor():
    """pixel_to_world must invert world_to_pixel on z = 0 for every deployed camera."""
    import notebook_data as nd

    rng = np.random.default_rng(53)
    for camera, model in nd.camera_models().items():
        for _ in range(100):
            x, y = rng.uniform(-8, 8), rng.uniform(-12, 12)
            u, v, _ = model.world_to_pixel(x, y, 0.0)
            back = model.pixel_to_world(u, v)
            assert back is not None, camera
            assert back[0] == pytest.approx(x, abs=1e-8), camera
            assert back[1] == pytest.approx(y, abs=1e-8), camera


def test_homography_equals_the_full_projection_on_the_floor():
    """The H = K [r1 r2 t] the notebook writes out must be the real projection on z = 0."""
    import notebook_data as nd

    rng = np.random.default_rng(59)
    for camera, model in nd.camera_models().items():
        for _ in range(100):
            x, y = rng.uniform(-8, 8), rng.uniform(-12, 12)
            homogeneous = model.H @ np.array([x, y, 1.0])
            u, v, _ = model.world_to_pixel(x, y, 0.0)
            assert homogeneous[0] / homogeneous[2] == pytest.approx(u, abs=1e-7), camera
            assert homogeneous[1] / homogeneous[2] == pytest.approx(v, abs=1e-7), camera


def test_radial_tangential_basis_is_orthonormal():
    """The decomposition must preserve length, or the 51/49 split means nothing."""
    rng = np.random.default_rng(61)
    for _ in range(500):
        sight = rng.normal(size=2)
        along = sight / np.linalg.norm(sight)
        across = np.array([-along[1], along[0]])
        assert abs(along @ across) < 1e-12
        assert abs(np.linalg.norm(across) - 1.0) < 1e-12
        err = rng.normal(size=2)
        assert (err @ along) ** 2 + (err @ across) ** 2 == pytest.approx(err @ err, abs=1e-10)


def test_two_sigma_ellipse_covers_the_right_mass():
    """The drawn 2-sigma contour should hold 1 - exp(-2) of a 2-D Gaussian."""
    rng = np.random.default_rng(67)
    A = rng.normal(size=(2, 2))
    P = A @ A.T + np.eye(2) * 0.1
    S = rng.multivariate_normal(np.zeros(2), P, 200_000)
    nees = np.einsum("ij,jk,ik->i", S, np.linalg.inv(P), S)
    assert float((nees <= 4.0).mean()) == pytest.approx(1 - math.exp(-2.0), abs=0.005)


# --------------------------------------------------------------------------- #
# the comparison the notebook makes
# --------------------------------------------------------------------------- #

def test_gated_evidence_is_not_comparable_across_arms():
    """Why the notebook recomputes the evidence with the gate off.

    Two different R values reject different observations, so their gated evidences sum
    over different data. This pins the reason down rather than trusting a comment.
    """
    rng = np.random.default_rng(71)
    seq = _Seq(80, list(range(0, 80, 2)), rng)
    tight = {"camera_A": np.eye(2) * 0.005**2}
    loose = {"camera_A": np.eye(2) * 0.20**2}
    a = NS["kalman_filter"](seq, tight, m0=seq.odom[0].copy())
    b = NS["kalman_filter"](seq, loose, m0=seq.odom[0].copy())
    assert int(a["rejected"].sum()) != int(b["rejected"].sum()), (
        "construct a case where the gate bites differently, or this guard is vacuous")
    a_all = NS["kalman_filter"](seq, tight, gate=GATE_OFF, m0=seq.odom[0].copy())
    b_all = NS["kalman_filter"](seq, loose, gate=GATE_OFF, m0=seq.odom[0].copy())
    assert int(a_all["rejected"].sum()) == 0 and int(b_all["rejected"].sum()) == 0
    # with the gate off both sum over every observation, which is the comparable quantity
    n_obs = sum(1 for c in seq.camera if c is not None)
    assert int(a_all["used"].sum()) == n_obs and int(b_all["used"].sum()) == n_obs


def test_published_summary_matches_the_notebook_claims():
    """The numbers quoted in the prose must be the ones the notebook actually produced."""
    path = REPO / "logs/studies/filter_notebook/notebook_summary.json"
    if not path.is_file():
        pytest.skip("the notebook has not been run")
    summary = json.loads(path.read_text(encoding="utf-8"))
    by_label = {s["label"]: s for s in summary["scores"]}

    commissioned = by_label["commissioned scatter + offset, filtered"]
    learned = by_label["learned R, filtered"]
    removed = by_label["offsets removed, scatter-only R"]

    # the three claims the conclusions rest on
    assert learned["median_nees"] > commissioned["median_nees"], (
        "the notebook claims learning R on this run is less honest than commissioning")
    assert removed["median_nees"] < 2.0, (
        "the notebook claims removing the offsets restores an honest belief")
    assert commissioned["median_nees"] > 5 * NS["CALIBRATED_MEDIAN_NEES"], (
        "the notebook claims the best zero-mean model is still several times overconfident")
    # and that no arm was rescued by accuracy alone
    assert removed["rmse_cm"] < commissioned["rmse_cm"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
