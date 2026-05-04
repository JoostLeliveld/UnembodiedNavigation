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
    Q: object
    R_visible: object
    R_miss: object
    control_weight: float
    risk_scale: float
    ambiguity_scale: float
    discount_gamma: float
    visibility_sigma_kappa: float
    goal_prior_u_std_start: float
    goal_prior_v_std_start: float
    goal_prior_u_std_final: float
    goal_prior_v_std_final: float
    goal_tightening_power: float
    goal_progress_n_steps: int
    robot_collision_radius_m: float
    min_terminal_goal_progress_m: float
    invalid_rollout_barrier_cost: float
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


def collision_barrier_penalty_ca(clearance, params: CasadiEfeParams):
    near_margin = 0.05
    near_term = 1e-3 * params.invalid_rollout_barrier_cost * _softplus_expr(
        (near_margin - clearance) / 0.02
    )
    penetration = ca.fmax(-clearance, 0.0)
    scale = max(float(params.robot_collision_radius_m), 1e-3)
    penetration_term = params.invalid_rollout_barrier_cost * (
        1.0 + ca.power(penetration / scale, 2)
    )
    return near_term + ca.if_else(penetration > 0.0, penetration_term, 0.0)


def terminal_progress_penalty_ca(m0, m_terminal, goal_xy, params: CasadiEfeParams):
    if float(params.min_terminal_goal_progress_m) <= 0.0:
        return 0.0
    current_goal_distance = ca.sqrt(
        ca.power(m0[0] - goal_xy[0], 2) + ca.power(m0[1] - goal_xy[1], 2)
    )
    terminal_goal_distance = ca.sqrt(
        ca.power(m_terminal[0] - goal_xy[0], 2) + ca.power(m_terminal[1] - goal_xy[1], 2)
    )
    progress = current_goal_distance - terminal_goal_distance
    shortfall = ca.fmax(params.min_terminal_goal_progress_m - progress, 0.0)
    denom = max(float(params.min_terminal_goal_progress_m), 1e-6)
    return 0.01 * params.invalid_rollout_barrier_cost * ca.power(shortfall / denom, 2)


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
    collision_signed_distance=None,
):
    """
    Core Expected Free Energy functional for a unicycle agent.
    Iteratively propagates Gaussian state (m, S) over params.time_horizon.
    """
    m = m0
    S = S0
    total_risk = 0
    total_amb = 0
    total_control = 0
    total_nogo = 0
    denom = float(max(params.goal_progress_n_steps, 1))

    for t in range(params.time_horizon):
        u_t = ca.vertcat(u_flat[2 * t], u_flat[2 * t + 1])
        m_prev = m
        m = unicycle_step_ca(m_prev, u_t, params.dt)
        F = unicycle_jacobian_ca(m_prev, u_t, params.dt)
        S = ca.mtimes([F, S, F.T]) + params.Q

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
        if nogo_cost is not None:
            total_nogo += weight_t * nogo_cost(m)
        if collision_signed_distance is not None:
            clearance = collision_signed_distance(m) - params.robot_collision_radius_m
            total_nogo += weight_t * collision_barrier_penalty_ca(clearance, params)

    total_risk += terminal_progress_penalty_ca(m0, m, goal_xy, params)
    return total_risk + total_amb + total_control + total_nogo


def make_efe_valgrad_fn(
    params: CasadiEfeParams,
    H,
    *,
    approx='ET1',
    p_vis_state=None,
    nogo_cost=None,
    collision_signed_distance=None,
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
        collision_signed_distance=collision_signed_distance,
    )
    gradient = ca.gradient(objective, u_flat)
    valgrad = ca.Function(
        'visibility_aware_efe_valgrad',
        [u_flat, m0, S0, goal_obs, goal_xy, progress_index0],
        [objective, gradient],
        ['u_flat', 'm0', 'S0', 'goal_obs', 'goal_xy', 'progress_index0'],
        ['objective', 'gradient'],
    )

    def _wrapper(u_val, m_val, S_val, goal_obs_val, goal_xy_val, progress_index0_val):
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
