"""CasADi utilities for symbolic visibility-aware EFE planning."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    import casadi as ca
except Exception:  # pragma: no cover - optional dependency
    ca = None


def casadi_available() -> bool:
    return ca is not None


@dataclass
class CasadiEfeParams:
    # NOTE: there is no static `Q` field. Process noise is rebuilt per step inside the
    # EFE loop via unicycle_process_noise_ca(process_noise_xy, process_noise_theta, dt,
    # theta, v) — the exact integrated Q_d(theta, v, dt) — so a frozen Q would be unused.
    R_visible: object
    R_miss: object
    control_weight: float
    risk_scale: float
    ambiguity_scale: float
    discount_gamma: float
    process_noise_xy: float
    process_noise_theta: float
    visibility_sigma_kappa: float
    goal_prior_u_std_start: float
    goal_prior_v_std_start: float
    goal_prior_u_std_final: float
    goal_prior_v_std_final: float
    goal_tightening_power: float
    goal_progress_n_steps: int
    use_belief_nogo_cost: bool
    time_horizon: int
    dt: float
    Du: int


def _require_casadi():
    if not casadi_available():
        raise RuntimeError("CasADi is not available")


def _clip_expr(x, lo, hi):
    return ca.fmin(ca.fmax(x, lo), hi)


def make_g_from_homography(H):
    """
    Return a CasADi-friendly planar homography observation function.
    H: (3,3) projective mapping -> returns Function(state(3,) -> uv(2,))
    """
    _require_casadi()
    H_ca = ca.DM(np.asarray(H, dtype=float))
    state = ca.MX.sym('state', 3)
    pt = ca.vertcat(state[0], state[1], 1.0)
    pix = ca.mtimes(H_ca, pt)
    uv = ca.vertcat(pix[0] / pix[2], pix[1] / pix[2])
    return ca.Function('homography_uv', [state], [uv], ['state'], ['uv'])


def wrap_angle_ca(theta):
    return ca.atan2(ca.sin(theta), ca.cos(theta))


def unicycle_step_ca(state, control, dt):
    x = state[0] + control[0] * dt * ca.cos(state[2])
    y = state[1] + control[0] * dt * ca.sin(state[2])
    theta = wrap_angle_ca(state[2] + control[1] * dt)
    return ca.vertcat(x, y, theta)


def unicycle_jacobian_ca(state, control, dt):
    theta = state[2]
    v = control[0]
    return ca.vertcat(
        ca.horzcat(1.0, 0.0, -v * dt * ca.sin(theta)),
        ca.horzcat(0.0, 1.0, v * dt * ca.cos(theta)),
        ca.horzcat(0.0, 0.0, 1.0),
    )


def unicycle_process_noise_ca(process_noise_xy, process_noise_theta, dt, theta, v):
    c = ca.cos(theta)
    s = ca.sin(theta)
    sig_v2 = process_noise_xy ** 2
    sig_w2 = process_noise_theta ** 2

    # Q_d matrix elements
    # Row 0
    q00 = sig_v2 * (c ** 2) * dt + (1.0 / 3.0) * (v ** 2) * (s ** 2) * sig_w2 * (dt ** 3)
    q01 = sig_v2 * c * s * dt - (1.0 / 3.0) * (v ** 2) * c * s * sig_w2 * (dt ** 3)
    q02 = -0.5 * v * s * sig_w2 * (dt ** 2)

    # Row 1
    q10 = q01
    q11 = sig_v2 * (s ** 2) * dt + (1.0 / 3.0) * (v ** 2) * (c ** 2) * sig_w2 * (dt ** 3)
    q12 = 0.5 * v * c * sig_w2 * (dt ** 2)

    # Row 2
    q20 = q02
    q21 = q12
    q22 = sig_w2 * dt

    return ca.vertcat(
        ca.horzcat(q00, q01, q02),
        ca.horzcat(q10, q11, q12),
        ca.horzcat(q20, q21, q22)
    )


def _ensure_symmetric_pd(M, eps=1e-9):
    _require_casadi()
    dim = int(M.size1())
    return 0.5 * (M + M.T) + float(eps) * ca.DM.eye(dim)


def _det_2x2(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def _det_3x3(M):
    return (
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    )


def _logdet_small_pd(M):
    dim = int(M.size1())
    if dim == 2 and int(M.size2()) == 2:
        det_val = _det_2x2(M)
    elif dim == 3 and int(M.size2()) == 3:
        det_val = _det_3x3(M)
    else:
        raise RuntimeError(f"Unsupported matrix size for CasADi logdet surrogate: {dim}x{int(M.size2())}")
    return ca.log(ca.fmax(det_val, 1e-12))


def _chol_2x2(M, eps=1e-9):
    M = 0.5 * (M + M.T)
    a = ca.fmax(M[0, 0], eps)
    l11 = ca.sqrt(a)
    l21 = M[1, 0] / l11
    diag22 = ca.fmax(M[1, 1] - l21 * l21, eps)
    l22 = ca.sqrt(diag22)
    return ca.vertcat(
        ca.horzcat(l11, 0.0),
        ca.horzcat(l21, l22),
    )


def _xy_visibility_sigma_points_ca(mean_xy, cov_xy, kappa=1.0):
    mean_xy = ca.reshape(mean_xy, 2, 1)
    cov_xy = 0.5 * (cov_xy + cov_xy.T)
    kappa = max(float(kappa), 1e-6)
    scale = math.sqrt(2.0 + kappa)
    chol = _chol_2x2(cov_xy + 1e-9 * ca.DM.eye(2))
    spread = scale * chol
    sigma_points = (
        mean_xy,
        mean_xy + spread[:, 0],
        mean_xy - spread[:, 0],
        mean_xy + spread[:, 1],
        mean_xy - spread[:, 1],
    )
    weights = (
        kappa / (2.0 + kappa),
        1.0 / (2.0 * (2.0 + kappa)),
        1.0 / (2.0 * (2.0 + kappa)),
        1.0 / (2.0 * (2.0 + kappa)),
        1.0 / (2.0 * (2.0 + kappa)),
    )
    return sigma_points, weights


def expected_visibility_ca(mean, cov, prob_state, *, kappa=1.0, lo=1e-4, hi=1.0 - 1e-4):
    """
    Approximate expected visibility via Unscented Transform (Sigma Points).
    mean: (3,) S: (3,3) -> return p_vis: float (scalar MX)
    """
    sigma_points_xy, weights = _xy_visibility_sigma_points_ca(mean[:2], cov[:2, :2], kappa=kappa)
    total = 0
    for sigma_xy, weight in zip(sigma_points_xy, weights):
        state = ca.vertcat(sigma_xy[0], sigma_xy[1], mean[2])
        total += float(weight) * _clip_expr(prob_state(state), lo, hi)
    return _clip_expr(total, lo, hi)


def _smoothstep_ca(x):
    x = _clip_expr(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _softplus_expr(x):
    x = _clip_expr(x, -60.0, 60.0)
    return ca.log(1.0 + ca.exp(x))


def _visibility_effective_score_ca(p_vis, params: CasadiEfeParams):
    del params
    return _clip_expr(p_vis, 1e-4, 1.0 - 1e-4)


def _blend_observation_covariance_ca(p_vis_eff, params: CasadiEfeParams):
    visible_prec = ca.vertcat(
        1.0 / ca.fmax(params.R_visible[0, 0], 1e-6),
        1.0 / ca.fmax(params.R_visible[1, 1], 1e-6),
    )
    miss_prec = ca.vertcat(
        1.0 / ca.fmax(params.R_miss[0, 0], 1e-6),
        1.0 / ca.fmax(params.R_miss[1, 1], 1e-6),
    )
    blended_prec = p_vis_eff * visible_prec + (1.0 - p_vis_eff) * miss_prec
    return ca.diag(1.0 / ca.fmax(blended_prec, 1e-9))


def goal_obs_cov_ca_for_progress(params: CasadiEfeParams, progress):
    progress_fast = ca.power(_clip_expr(progress, 0.0, 1.0), params.goal_tightening_power)
    a = _smoothstep_ca(progress_fast)
    sigma_u = (1.0 - a) * params.goal_prior_u_std_start + a * params.goal_prior_u_std_final
    sigma_v = (1.0 - a) * params.goal_prior_v_std_start + a * params.goal_prior_v_std_final
    return ca.diag(ca.vertcat(ca.power(sigma_u, 2), ca.power(sigma_v, 2)))


def et1_ca(m, S, R_eff, g, dg):
    """
    Extended Transform (1st order) linearization.
    m: (3,) S: (3,3) R_eff: (2,2) g/dg: mapping functions
    returns: mu_y: (2,) Sigma_y: (2,2) Gamma_xy: (3,2)
    """
    Jm = dg(m)
    mu = g(m)
    Sigma = ca.mtimes([Jm, S, Jm.T]) + R_eff
    Gamma = ca.mtimes(S, Jm.T)
    return mu, Sigma, Gamma


def et2_ca(m, S, R_eff, g, dg, d2g):
    """
    Second-order extended transform using CasADi Jacobians and Hessians.
    m: (3,) S: (3,3) R_eff: (2,2) g/dg/d2g: mapping functions
    returns: mu_y: (2,) Sigma_y: (2,2) Gamma_xy: (3,2)
    """
    Jm = dg(m)
    mu0 = g(m)
    dim_y = int(mu0.size1())
    hessians = [h_fn(m) for h_fn in d2g]

    aux1 = []
    for H_i in hessians:
        aux1.append(ca.trace(ca.mtimes(H_i, S)))

    aux2 = ca.MX.zeros(dim_y, dim_y)
    for i, H_i in enumerate(hessians):
        for j, H_j in enumerate(hessians):
            aux2[i, j] = ca.trace(ca.mtimes([H_i, S, H_j, S]))

    mu = mu0 + 0.5 * ca.vertcat(*aux1)
    Sigma = ca.mtimes([Jm, S, Jm.T]) + 0.5 * aux2 + R_eff
    Gamma = ca.mtimes(S, Jm.T)
    return mu, Sigma, Gamma


def risk_ca(mu, Sigma, goal_mu, goal_S):
    """
    KL-Divergence based instrumental risk cost.
    mu: (2,) Sigma: (2,2) goal_mu: (2,) goal_S: (2,2)
    """
    Sigma_pd = _ensure_symmetric_pd(Sigma)
    goal_S_pd = _ensure_symmetric_pd(goal_S)
    diff = goal_mu - mu
    trace_term = ca.trace(ca.solve(goal_S_pd, Sigma_pd))
    quad_term = ca.mtimes([diff.T, ca.solve(goal_S_pd, diff)])
    dim = int(goal_mu.size1())
    logdet_goal = _logdet_small_pd(goal_S_pd)
    logdet_sigma = _logdet_small_pd(Sigma_pd)
    return 0.5 * (trace_term - dim + quad_term[0, 0] + logdet_goal - logdet_sigma)


def ambiguity_ca(Sigma, Gamma, S):
    """
    Expected information gain (epistemic uncertainty reduction).
    Sigma: (2,2) Gamma: (3,2) S: (3,3)
    """
    S_pd = _ensure_symmetric_pd(S)
    Sigma_cond = Sigma - ca.mtimes([Gamma.T, ca.solve(S_pd, Gamma)])
    Sigma_cond = _ensure_symmetric_pd(Sigma_cond)
    dim = int(Sigma_cond.size1())
    logdet = _logdet_small_pd(Sigma_cond)
    return 0.5 * (dim * math.log(2.0 * math.pi * math.e) + logdet)


def state_posterior_cov_ca(S, Sigma, Gamma):
    """Expected state covariance after the planner-facing camera update."""
    S_pd = _ensure_symmetric_pd(S)
    Sigma_pd = _ensure_symmetric_pd(Sigma)
    S_post = S_pd - ca.mtimes([Gamma, ca.solve(Sigma_pd, Gamma.T)])
    return _ensure_symmetric_pd(S_post)


def visibility_aware_unicycle_efe_ca(
    u_flat,
    m0,
    S0,
    goal_obs,
    goal_xy,
    progress_index0,
    params: CasadiEfeParams,
    g,
    dg,
    *,
    approx='ET1',
    d2g=None,
    p_vis_state=None,
    nogo_cost=None,
    nogo_belief_cost=None,
):
    """
    Core Expected Free Energy functional for a unicycle agent.
    Iteratively propagates Gaussian state (m, S) over params.time_horizon.
    Goal-seeking emerges from the EFE goal-prior in the risk term (no external
    goal-distance reward).
    """
    m = m0
    S = S0
    total_risk = 0
    total_amb = 0
    total_control = 0
    total_nogo = 0
    denom = float(max(params.goal_progress_n_steps, 1))  # goal-prior anneal schedule length

    for t in range(params.time_horizon):
        u_t = ca.vertcat(u_flat[2 * t], u_flat[2 * t + 1])
        m_prev = m
        m = unicycle_step_ca(m_prev, u_t, params.dt)
        F = unicycle_jacobian_ca(m_prev, u_t, params.dt)
        Q_t = unicycle_process_noise_ca(
            params.process_noise_xy,
            params.process_noise_theta,
            params.dt,
            m_prev[2],
            u_t[0]
        )
        S = ca.mtimes([F, S, F.T]) + Q_t

        p_vis = 1.0
        if p_vis_state is not None:
            p_vis = expected_visibility_ca(
                m,
                S,
                p_vis_state,
                kappa=params.visibility_sigma_kappa,
            )
        p_vis_eff = _visibility_effective_score_ca(p_vis, params)
        R_plan = _blend_observation_covariance_ca(p_vis_eff, params)
        if approx == 'ET1':
            mu, Sigma, Gamma = et1_ca(m, S, R_plan, g, dg)
        elif approx == 'ET2':
            mu, Sigma, Gamma = et2_ca(m, S, R_plan, g, dg, d2g or [])
        else:
            raise RuntimeError(f"Unsupported CasADi approximation: {approx}")

        progress = (progress_index0 + float(t)) / denom
        goal_cov_t = goal_obs_cov_ca_for_progress(params, progress)
        weight_t = params.discount_gamma ** t
        total_risk += weight_t * params.risk_scale * risk_ca(mu, Sigma, goal_obs, goal_cov_t)
        total_amb += weight_t * params.ambiguity_scale * ambiguity_ca(Sigma, Gamma, S)
        total_control += weight_t * params.control_weight * ca.sumsqr(u_t)
        if nogo_belief_cost is not None and params.use_belief_nogo_cost:
            S_drive = state_posterior_cov_ca(S, Sigma, Gamma)
            total_nogo += weight_t * nogo_belief_cost(m, S_drive)
        elif nogo_cost is not None:
            total_nogo += weight_t * nogo_cost(m)

    # Normalise per-step sums by the effective discounted horizon so that changing
    # time_horizon does not rescale the weight balance between terms.
    H_eff = sum(params.discount_gamma ** t for t in range(params.time_horizon))
    inv_H = 1.0 / max(H_eff, 1e-8)
    return (total_risk * inv_H + total_amb * inv_H
            + total_control * inv_H + total_nogo * inv_H)


def _make_valgrad_wrapper(valgrad):
    """Wrap a built valgrad ca.Function as a numpy-in/out callable."""

    def _wrapper(
        u_val,
        m_val,
        S_val,
        goal_obs_val,
        goal_xy_val,
        progress_index0_val,
    ):
        val, grad = valgrad(
            np.asarray(u_val, dtype=float).reshape((-1, 1)),
            np.asarray(m_val, dtype=float).reshape((3, 1)),
            np.asarray(S_val, dtype=float).reshape((3, 3)),
            np.asarray(goal_obs_val, dtype=float).reshape((2, 1)),
            np.asarray(goal_xy_val, dtype=float).reshape((2, 1)),
            np.asarray(float(progress_index0_val), dtype=float),
        )
        return float(np.asarray(val, dtype=float).reshape(-1)[0]), np.asarray(grad, dtype=float).reshape(-1)

    return _wrapper


def make_efe_valgrad_fn(
    params: CasadiEfeParams,
    H,
    *,
    approx='ET1',
    p_vis_state=None,
    nogo_cost=None,
    nogo_belief_cost=None,
):
    _require_casadi()
    approx = str(approx or 'ET1').upper()
    if approx not in ('ET1', 'ET2'):
        raise RuntimeError("CasADi EFE path supports only ET1 or ET2")

    g = make_g_from_homography(H)

    state_sym = ca.MX.sym('state_for_jac', 3)
    g_expr = g(state_sym)
    dg = ca.Function(
        'homography_uv_jac',
        [state_sym],
        [ca.jacobian(g_expr, state_sym)],
        ['state'],
        ['J'],
    )
    d2g = None
    if approx == 'ET2':
        d2g = []
        for i in range(int(g_expr.size1())):
            H_i, _ = ca.hessian(g_expr[i], state_sym)
            d2g.append(
                ca.Function(
                    f'homography_uv_hess_{i}',
                    [state_sym],
                    [H_i],
                    ['state'],
                    ['H'],
                )
            )

    u_flat = ca.MX.sym('u_flat', params.time_horizon * params.Du)
    m0 = ca.MX.sym('m0', 3)
    S0 = ca.MX.sym('S0', 3, 3)
    goal_obs = ca.MX.sym('goal_obs', 2)
    goal_xy = ca.MX.sym('goal_xy', 2)
    progress_index0 = ca.MX.sym('progress_index0')

    objective = visibility_aware_unicycle_efe_ca(
        u_flat,
        m0,
        S0,
        goal_obs,
        goal_xy,
        progress_index0,
        params,
        g,
        dg,
        approx=approx,
        d2g=d2g,
        p_vis_state=p_vis_state,
        nogo_cost=nogo_cost,
        nogo_belief_cost=nogo_belief_cost,
    )
    gradient = ca.gradient(objective, u_flat)
    valgrad = ca.Function(
        'visibility_aware_efe_valgrad',
        [u_flat, m0, S0, goal_obs, goal_xy, progress_index0],
        [objective, gradient],
        ['u_flat', 'm0', 'S0', 'goal_obs', 'goal_xy', 'progress_index0'],
        ['objective', 'gradient'],
    )

    return _make_valgrad_wrapper(valgrad)
