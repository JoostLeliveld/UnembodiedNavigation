"""CasADi utilities for symbolic visibility-aware EFE planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import warnings

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
    # Metric (x,y) goal-progress incentive. Pure function of the mean state and
    # the current target; no GP / ambiguity / visibility dependence, so it is
    # identical for C1/C2/C3. Default 0.0 keeps every locked config unchanged.
    goal_progress_weight: float = 0.0
    # LOCAL reference-segment tracking. Pure metric (x,y) distance from the
    # predicted mean to a fixed per-step reference point computed OUTSIDE the
    # solve, plus a control-smoothness (du) term. No GP / ambiguity / visibility
    # dependence, so it is identical for C1/C2/C3. All three default to 0.0, so
    # every existing caller (global planner, locked configs) is numerically
    # unchanged: the terms vanish from the objective when the weights are 0.0.
    ref_weight: float = 0.0
    terminal_ref_weight: float = 0.0
    du_weight: float = 0.0


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
    ref_seq=None,
    prev_u=None,
):
    """
    Core Expected Free Energy functional for a unicycle agent.
    Iteratively propagates Gaussian state (m, S) over params.time_horizon.

    LOCAL tracking extension (condition-neutral, default-off):
      ref_seq: fixed per-step (x, y) reference points, MX of shape (2*H,) laid
        out [x0, y0, x1, y1, ...]. Computed OUTSIDE the solve so the optimizer
        sees only smooth squared-distance terms. Used only when ref_weight or
        terminal_ref_weight is > 0.0.
      prev_u: last applied control, MX of shape (2,). Used only when du_weight
        is > 0.0 for a control-smoothness penalty sum_t ||u_t - u_{t-1}||^2 with
        u_{-1} = prev_u.
    When ref_weight, terminal_ref_weight, and du_weight are all 0.0 these terms
    contribute nothing and the returned objective is numerically identical to the
    pre-extension behaviour (no symbolic terms are added at all).
    """
    m = m0
    S = S0
    total_risk = 0
    total_amb = 0
    total_control = 0
    total_nogo = 0
    total_progress = 0
    total_ref = 0
    total_du = 0
    denom = float(max(params.goal_progress_n_steps, 1))  # goal-prior anneal schedule length (legit)
    ref_weight = float(getattr(params, 'ref_weight', 0.0))
    terminal_ref_weight = float(getattr(params, 'terminal_ref_weight', 0.0))
    du_weight = float(getattr(params, 'du_weight', 0.0))
    use_ref = (ref_weight > 0.0 or terminal_ref_weight > 0.0) and ref_seq is not None
    use_du = du_weight > 0.0 and prev_u is not None

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
        # NOTE: the metric goal-distance reward (goal_progress_weight * ||mean-goal||^2)
        # was REMOVED (2026-06-10). It is a non-EFE goal attractor that imposes
        # goal-seeking externally instead of letting it emerge from the goal-prior
        # in the risk term; it had been added to mask the zero-velocity/stop-short
        # local minimum. Goal-seeking is now the EFE goal-prior only. (total_progress
        # stays 0; goal_progress_n_steps still drives the legit goal-prior anneal.)
        if use_ref:
            # Fixed per-step reference point r_t = (ref_seq[2t], ref_seq[2t+1]),
            # computed OUTSIDE the solve. Smooth squared tracking error of the
            # predicted mean toward the reference segment. Condition-neutral
            # (no GP / ambiguity / visibility). The final step also carries the
            # terminal reference weight so the tracker holds the segment endpoint.
            dref = ca.vertcat(m[0] - ref_seq[2 * t], m[1] - ref_seq[2 * t + 1])
            total_ref += weight_t * ref_weight * ca.sumsqr(dref)
            if t == params.time_horizon - 1:
                total_ref += weight_t * terminal_ref_weight * ca.sumsqr(dref)
        if use_du:
            # Control-smoothness: penalise change from the previous control. For
            # t == 0 the previous control is the last applied command prev_u;
            # afterwards it is u_{t-1}. Condition-neutral.
            u_prev = prev_u if t == 0 else ca.vertcat(u_flat[2 * (t - 1)], u_flat[2 * (t - 1) + 1])
            total_du += weight_t * du_weight * ca.sumsqr(u_t - u_prev)

    # Normalise per-step sums by the effective discounted horizon so that changing
    # time_horizon does not rescale the weight balance between terms.
    H_eff = sum(params.discount_gamma ** t for t in range(params.time_horizon))
    inv_H = 1.0 / max(H_eff, 1e-8)
    return (total_risk * inv_H + total_amb * inv_H
            + total_control * inv_H + total_nogo * inv_H
            + total_progress * inv_H
            + total_ref * inv_H + total_du * inv_H)


def _load_cached_valgrad(cache_path):
    """Best-effort load of a previously serialized valgrad ca.Function.

    Returns the loaded Function only if it round-trips with the exact I/O arity
    of the live build (8 inputs, 2 outputs); on ANY failure (missing file,
    corrupt blob, CasADi-version mismatch) it returns None so the caller falls
    back to a fresh symbolic build. CasADi serializes the full computational
    graph, so a successfully loaded function evaluates bit-identically to a
    freshly built one -- no numerical change, only the build cost is skipped.
    """
    if not cache_path:
        return None
    try:
        if not os.path.exists(cache_path):
            return None
        fn = ca.Function.load(cache_path)
        if int(fn.n_in()) == 8 and int(fn.n_out()) == 2:
            return fn
    except Exception:
        return None
    return None


def _save_cached_valgrad(valgrad, cache_path):
    """Atomically persist the built valgrad Function; never raises."""
    if not cache_path:
        return
    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        valgrad.save(tmp_path)
        os.replace(tmp_path, cache_path)  # atomic on POSIX
    except Exception:
        try:
            if 'tmp_path' in dir() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _build_valgrad_function(fn_args, jit_opts):
    """Construct the valgrad ca.Function, optionally JIT-compiling it to native
    code.

    ``jit_opts`` (when given) is the CasADi options dict passed straight to
    ``ca.Function`` -- e.g. ``{'jit': True, 'compiler': 'shell',
    'jit_options': {'flags': ['-O1'], 'compiler': 'gcc'}}``. JIT trades a
    one-time C-codegen+compile cost (paid once per run, since the built function
    is cached in-process) for much faster repeated evaluation: the global EFE
    solve calls this function tens of thousands of times, so the per-eval VM
    overhead dominates. On ANY JIT failure (no compiler, codegen too large,
    plugin missing) we fall back to the interpreted build so behaviour never
    regresses -- only speed does.
    """
    if jit_opts:
        try:
            return ca.Function(*fn_args, dict(jit_opts))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"CasADi JIT build failed ({exc}); falling back to interpreted "
                "evaluation. Ensure a C compiler (gcc/clang) is available to "
                "enable JIT."
            )
    return ca.Function(*fn_args)


def _make_valgrad_wrapper(valgrad, H_local):
    """Wrap a (built or loaded) valgrad ca.Function as a numpy-in/out callable."""

    def _wrapper(
        u_val,
        m_val,
        S_val,
        goal_obs_val,
        goal_xy_val,
        progress_index0_val,
        ref_seq_val=None,
        prev_u_val=None,
    ):
        if ref_seq_val is None:
            ref_seq_arr = np.zeros(H_local * 2, dtype=float)
        else:
            ref_seq_arr = np.asarray(ref_seq_val, dtype=float).reshape(-1)
            if ref_seq_arr.size != H_local * 2:
                raise ValueError(
                    f"ref_seq must have {H_local * 2} entries, got {ref_seq_arr.size}"
                )
        if prev_u_val is None:
            prev_u_arr = np.zeros(2, dtype=float)
        else:
            prev_u_arr = np.asarray(prev_u_val, dtype=float).reshape(-1)
            if prev_u_arr.size != 2:
                raise ValueError(f"prev_u must have 2 entries, got {prev_u_arr.size}")
        val, grad = valgrad(
            np.asarray(u_val, dtype=float).reshape((-1, 1)),
            np.asarray(m_val, dtype=float).reshape((3, 1)),
            np.asarray(S_val, dtype=float).reshape((3, 3)),
            np.asarray(goal_obs_val, dtype=float).reshape((2, 1)),
            np.asarray(goal_xy_val, dtype=float).reshape((2, 1)),
            np.asarray(float(progress_index0_val), dtype=float),
            ref_seq_arr.reshape((-1, 1)),
            prev_u_arr.reshape((2, 1)),
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
    cache_path=None,
    jit_opts=None,
):
    _require_casadi()
    approx = str(approx or 'ET1').upper()
    if approx not in ('ET1', 'ET2'):
        raise RuntimeError("CasADi EFE path supports only ET1 or ET2")

    # Disk cache: the symbolic graph below is identical for a fixed config, so a
    # previously serialized build can be reloaded (bit-identical evaluation) to
    # skip the multi-second assembly on each fresh-process run. Any cache miss or
    # load failure falls through to a normal build. Skipped when JIT is
    # requested: a serialized Function reloads as an interpreted graph, so honour
    # the JIT request by building (and compiling) fresh instead.
    if not jit_opts:
        cached = _load_cached_valgrad(cache_path)
        if cached is not None:
            return _make_valgrad_wrapper(cached, int(params.time_horizon))

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
    # New LOCAL-tracking inputs, appended after the existing args so callers that
    # always pass zeros keep the original behaviour. ref_seq is the flat per-step
    # reference segment [x0,y0,x1,y1,...] of length 2*H; prev_u is the last applied
    # control. When ref/du weights are 0.0 the objective ignores both inputs.
    ref_seq = ca.MX.sym('ref_seq', params.time_horizon * 2)
    prev_u = ca.MX.sym('prev_u', 2)

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
        ref_seq=ref_seq,
        prev_u=prev_u,
    )
    gradient = ca.gradient(objective, u_flat)
    valgrad = _build_valgrad_function(
        (
            'visibility_aware_efe_valgrad',
            [u_flat, m0, S0, goal_obs, goal_xy, progress_index0, ref_seq, prev_u],
            [objective, gradient],
            ['u_flat', 'm0', 'S0', 'goal_obs', 'goal_xy', 'progress_index0', 'ref_seq', 'prev_u'],
            ['objective', 'gradient'],
        ),
        jit_opts,
    )

    # Only persist the interpreted build; a JIT function's value is its compiled
    # code, which Function.save/load does not carry, so caching it to disk would
    # silently drop back to interpreted evaluation on reload.
    if not jit_opts:
        _save_cached_valgrad(valgrad, cache_path)

    return _make_valgrad_wrapper(valgrad, int(params.time_horizon))
