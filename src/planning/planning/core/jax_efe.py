"""JAX utilities for EFE planning (optional runtime dependency)."""

from dataclasses import dataclass
import os

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


def jax_available() -> bool:
    return jax is not None and jnp is not None


_JAX_CACHE_CONFIGURED = False


def configure_persistent_cache(cache_dir: str | None = None) -> str | None:
    """Enable JAX persistent compilation caching once per process."""
    global _JAX_CACHE_CONFIGURED
    if not jax_available():
        return None
    if _JAX_CACHE_CONFIGURED:
        return cache_dir

    resolved_dir = str(cache_dir or '').strip()
    if not resolved_dir:
        resolved_dir = os.path.expanduser('~/.cache/unembodied_navigation/jax_compilation_cache')

    try:
        from jax.experimental.compilation_cache import compilation_cache as cc
        os.makedirs(resolved_dir, exist_ok=True)
        cc.set_cache_dir(resolved_dir)
        _JAX_CACHE_CONFIGURED = True
        return resolved_dir
    except Exception:
        return None


@dataclass
class JaxUnicycleParams:
    Q: "jnp.ndarray"
    R: "jnp.ndarray"
    control_weight: float
    risk_weight_state: float
    risk_weight_obs: float
    ambiguity_weight: float
    time_horizon: int
    dt: float
    Du: int
    R_bad: object = None
    visibility_weight: float = 0.0
    vis_cov_pos_scale: float = 2.0
    vis_cov_theta_scale: float = 0.8


@dataclass
class JaxNotebookSimpleParams:
    Q: "jnp.ndarray"
    R_visible: "jnp.ndarray"
    R_miss: "jnp.ndarray"
    control_weight: float
    risk_scale: float
    ambiguity_scale: float
    discount_gamma: float
    visibility_power: float
    visibility_sigma_kappa: float
    goal_prior_u_std_start: float
    goal_prior_v_std_start: float
    goal_prior_u_std_final: float
    goal_prior_v_std_final: float
    goal_tightening_power: float
    goal_progress_n_steps: int
    time_horizon: int
    dt: float
    Du: int


def make_g_from_homography(H):
    """Return a JAX-friendly planar homography observation function."""
    if not jax_available():
        raise RuntimeError("JAX is not available")
    H_j = jnp.array(H)

    def g(x):
        x = jnp.asarray(x)
        pt = jnp.array([x[0], x[1], 1.0], dtype=x.dtype)
        pix = H_j @ pt
        u = pix[0] / pix[2]
        v = pix[1] / pix[2]
        return jnp.stack([u, v])

    return g


def wrap_angle_jax(theta):
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def unicycle_step_jax(state, control, dt):
    x, y, theta = state
    v, w = control
    x = x + v * dt * jnp.cos(theta)
    y = y + v * dt * jnp.sin(theta)
    theta = wrap_angle_jax(theta + w * dt)
    return jnp.array([x, y, theta], dtype=state.dtype)


def unicycle_jacobian_jax(state, control, dt):
    _, _, theta = state
    v, _ = control
    F = jnp.eye(3, dtype=state.dtype)
    F = F.at[0, 2].set(-v * dt * jnp.sin(theta))
    F = F.at[1, 2].set(v * dt * jnp.cos(theta))
    return F


def _safe_cholesky(M, eps=1e-9):
    d = M.shape[0]
    return jnp.linalg.cholesky(M + eps * jnp.eye(d, dtype=M.dtype))


def _xy_visibility_sigma_points_jax(mean_xy, cov_xy, kappa=1.0):
    mean_xy = jnp.asarray(mean_xy, dtype=jnp.float64).reshape((2,))
    cov_xy = jnp.asarray(cov_xy, dtype=jnp.float64).reshape((2, 2))
    cov_xy = 0.5 * (cov_xy + cov_xy.T)
    kappa = max(float(kappa), 1e-6)
    scale = jnp.sqrt(jnp.asarray(2.0 + kappa, dtype=jnp.float64))
    chol = jnp.linalg.cholesky(cov_xy + 1e-9 * jnp.eye(2, dtype=jnp.float64))
    spread = scale * chol
    sigma_points = jnp.vstack(
        [
            mean_xy,
            mean_xy + spread[:, 0],
            mean_xy - spread[:, 0],
            mean_xy + spread[:, 1],
            mean_xy - spread[:, 1],
        ]
    )
    weights = jnp.asarray(
        [kappa / (2.0 + kappa)] + [1.0 / (2.0 * (2.0 + kappa))] * 4,
        dtype=jnp.float64,
    )
    return sigma_points, weights


def expected_visibility_jax(mean, cov, prob_state, *, kappa=1.0, lo=1e-4, hi=1.0 - 1e-4):
    sigma_points_xy, weights = _xy_visibility_sigma_points_jax(mean[:2], cov[:2, :2], kappa=kappa)
    vals = jax.vmap(
        lambda xy: jnp.clip(prob_state(jnp.array([xy[0], xy[1], mean[2]], dtype=mean.dtype)), lo, hi)
    )(sigma_points_xy)
    return jnp.clip(jnp.sum(weights * vals), lo, hi)


def _smoothstep_jax(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def goal_obs_cov_jax_for_progress(params: JaxNotebookSimpleParams, progress):
    progress_fast = jnp.clip(progress, 0.0, 1.0) ** params.goal_tightening_power
    a = _smoothstep_jax(progress_fast)
    sigma_u = (1.0 - a) * params.goal_prior_u_std_start + a * params.goal_prior_u_std_final
    sigma_v = (1.0 - a) * params.goal_prior_v_std_start + a * params.goal_prior_v_std_final
    return jnp.diag(jnp.array([sigma_u ** 2, sigma_v ** 2], dtype=jnp.float64))


def et1_jax(m, S, R_eff, g, dg=None):
    # Reuse prebuilt Jacobian transform when available.
    Jm = dg(m) if dg is not None else jax.jacfwd(g)(m)
    mu = g(m)
    Sigma = Jm @ S @ Jm.T + R_eff
    Gamma = S @ Jm.T
    return mu, Sigma, Gamma


def et2_jax(m, S, R_eff, g, dg=None, d2g=None):
    # Reuse prebuilt Jacobian/Hessian transforms when available.
    Jm = dg(m) if dg is not None else jax.jacfwd(g)(m)
    H = d2g(m) if d2g is not None else jax.jacfwd(jax.jacrev(g))(m)
    aux1 = jnp.array([jnp.trace(H[i] @ S) for i in range(H.shape[0])])
    aux2 = jnp.array([
        [jnp.trace(H[i] @ S @ H[j] @ S) for j in range(H.shape[0])]
        for i in range(H.shape[0])
    ])
    mu = g(m) + 0.5 * aux1
    Sigma = Jm @ S @ Jm.T + 0.5 * aux2 + R_eff
    Gamma = S @ Jm.T
    return mu, Sigma, Gamma


def risk_jax(mu, Sigma, goal_mu, goal_S):
    eps = 1e-9
    L0 = _safe_cholesky(Sigma, eps=eps)
    L1 = _safe_cholesky(goal_S, eps=eps)
    M = jnp.linalg.solve(L1, L0)
    y = jnp.linalg.solve(L1, goal_mu - mu)
    d = goal_mu.shape[0]
    return 0.5 * (jnp.sum(M ** 2) - d + jnp.sum(y ** 2) +
                  2.0 * jnp.sum(jnp.log(jnp.diag(L1) / jnp.diag(L0))))


def ambiguity_jax(Sigma, Gamma, S):
    Sigma_cond = Sigma - Gamma.T @ jnp.linalg.solve(S, Gamma)
    sign, logdet = jnp.linalg.slogdet(Sigma_cond)
    d = Sigma_cond.shape[0]
    return 0.5 * (d * jnp.log(2 * jnp.pi * jnp.e) + logdet)


def efe_unicycle_jax(
    u,
    m0,
    S0,
    goal_state,
    goal_state_cov,
    goal_obs,
    goal_obs_cov,
    params: JaxUnicycleParams,
    g,
    approx: str = "ET2",
    add_ambiguity: bool = True,
    use_obs_risk: bool = True,
    use_state_risk: bool = True,
    p_vis=None,
    nogo_cost=None,
    dg=None,
    d2g=None,
    risk_scale=1.0,
    ambiguity_scale=1.0,
):
    u = jnp.asarray(u)
    if u.ndim == 1:
        u = u.reshape((params.time_horizon, params.Du))
    m = m0
    S = S0
    total_risk = 0.0
    total_amb = 0.0
    total_control = 0.0
    total_vis = 0.0
    total_nogo = 0.0

    for t in range(params.time_horizon):
        m = unicycle_step_jax(m, u[t], params.dt)
        F = unicycle_jacobian_jax(m, u[t], params.dt)
        S = F @ S @ F.T + params.Q

        p = jnp.array(1.0, dtype=m.dtype)
        q = jnp.array(0.0, dtype=m.dtype)
        R_eff = params.R
        S_eff = S
        if p_vis is not None:
            p = jnp.clip(p_vis(m, S), 1e-4, 1.0 - 1e-4)
            q = 1.0 - p
            R_bad = params.R if params.R_bad is None else params.R_bad
            R_eff = p * params.R + q * R_bad
            sxy = 1.0 + params.vis_cov_pos_scale * q
            sth = 1.0 + params.vis_cov_theta_scale * q
            S_eff = S
            S_eff = S_eff.at[0, 0].set(S[0, 0] * sxy)
            S_eff = S_eff.at[1, 1].set(S[1, 1] * sxy)
            S_eff = S_eff.at[2, 2].set(S[2, 2] * sth)

        mu = Sigma = Gamma = None
        if use_obs_risk or add_ambiguity:
            if approx == "ET1":
                mu, Sigma, Gamma = et1_jax(m, S_eff, R_eff, g, dg=dg)
            elif approx == "ET2":
                mu, Sigma, Gamma = et2_jax(m, S_eff, R_eff, g, dg=dg, d2g=d2g)
            else:
                raise ValueError("Approximation method unknown.")

        r_state = 0.0
        if use_state_risk:
            r_state = risk_jax(m, S, goal_state, goal_state_cov)

        r_obs = 0.0
        if use_obs_risk and mu is not None:
            r_obs = risk_jax(mu, Sigma, goal_obs, goal_obs_cov)

        total_risk += params.risk_weight_state * r_state + params.risk_weight_obs * r_obs

        if add_ambiguity and Sigma is not None:
            amb = ambiguity_jax(Sigma, Gamma, S_eff)
            if p_vis is not None:
                amb = p * amb
            total_amb += params.ambiguity_weight * amb

        total_control += params.control_weight * jnp.sum(u[t] ** 2)
        if p_vis is not None and params.visibility_weight > 0.0:
            total_vis += params.visibility_weight * q
        if nogo_cost is not None:
            total_nogo += nogo_cost(m)

    risk_scale = jnp.maximum(jnp.asarray(risk_scale, dtype=m.dtype), 1e-9)
    ambiguity_scale = jnp.maximum(jnp.asarray(ambiguity_scale, dtype=m.dtype), 1e-9)
    return (
        total_risk / risk_scale
        + total_amb / ambiguity_scale
        + total_control
        + total_vis
        + total_nogo
    )


def make_unicycle_valgrad_fn(
    params: JaxUnicycleParams,
    g,
    approx: str = "ET2",
    add_ambiguity: bool = True,
    use_obs_risk: bool = True,
    use_state_risk: bool = True,
    p_vis=None,
    nogo_cost=None,
    mode: str = "rev",
    jit: bool = True,
):
    """Return (value, grad) function w.r.t. u for unicycle dynamics."""
    dg = jax.jacfwd(g)
    d2g = jax.jacfwd(jax.jacrev(g)) if str(approx).upper() == "ET2" else None

    def _valgrad(u, m, S, goal_state, goal_state_cov, goal_obs, goal_obs_cov, risk_scale, ambiguity_scale):
        if mode == "fwd":
            val = efe_unicycle_jax(
                u, m, S, goal_state, goal_state_cov, goal_obs, goal_obs_cov, params, g,
                approx=approx,
                add_ambiguity=add_ambiguity,
                use_obs_risk=use_obs_risk,
                use_state_risk=use_state_risk,
                p_vis=p_vis,
                nogo_cost=nogo_cost,
                dg=dg,
                d2g=d2g,
                risk_scale=risk_scale,
                ambiguity_scale=ambiguity_scale,
            )
            grad = jax.jacfwd(
                lambda uu: efe_unicycle_jax(
                    uu, m, S, goal_state, goal_state_cov, goal_obs, goal_obs_cov, params, g,
                    approx=approx,
                    add_ambiguity=add_ambiguity,
                    use_obs_risk=use_obs_risk,
                    use_state_risk=use_state_risk,
                    p_vis=p_vis,
                    nogo_cost=nogo_cost,
                    dg=dg,
                    d2g=d2g,
                    risk_scale=risk_scale,
                    ambiguity_scale=ambiguity_scale,
                )
            )(u)
            return val, grad

        return jax.value_and_grad(
            lambda uu: efe_unicycle_jax(
                uu, m, S, goal_state, goal_state_cov, goal_obs, goal_obs_cov, params, g,
                approx=approx,
                add_ambiguity=add_ambiguity,
                use_obs_risk=use_obs_risk,
                use_state_risk=use_state_risk,
                p_vis=p_vis,
                nogo_cost=nogo_cost,
                dg=dg,
                d2g=d2g,
                risk_scale=risk_scale,
                ambiguity_scale=ambiguity_scale,
            )
        )(u)

    return jax.jit(_valgrad) if jit else _valgrad


def notebook_simple_unicycle_jax(
    u,
    m0,
    S0,
    goal_obs,
    progress_index0,
    params: JaxNotebookSimpleParams,
    g,
    p_vis_state=None,
    nogo_cost=None,
    dg=None,
):
    u = jnp.asarray(u)
    if u.ndim == 1:
        u = u.reshape((params.time_horizon, params.Du))
    m = m0
    S = S0
    total_risk = 0.0
    total_amb = 0.0
    total_control = 0.0
    total_nogo = 0.0
    denom = jnp.asarray(max(params.goal_progress_n_steps, 1), dtype=m.dtype)
    progress_index0 = jnp.asarray(progress_index0, dtype=m.dtype)

    for t in range(params.time_horizon):
        m = unicycle_step_jax(m, u[t], params.dt)
        F = unicycle_jacobian_jax(m, u[t], params.dt)
        S = F @ S @ F.T + params.Q
        p_vis = jnp.array(1.0, dtype=m.dtype)
        if p_vis_state is not None:
            p_vis = expected_visibility_jax(
                m,
                S,
                p_vis_state,
                kappa=params.visibility_sigma_kappa,
            )
        p_vis_eff = jnp.clip(p_vis ** params.visibility_power, 1e-4, 1.0 - 1e-4)
        R_plan = p_vis_eff * params.R_visible + (1.0 - p_vis_eff) * params.R_miss
        mu, Sigma, Gamma = et1_jax(m, S, R_plan, g, dg=dg)
        progress = (progress_index0 + jnp.asarray(t, dtype=m.dtype)) / denom
        goal_cov_t = goal_obs_cov_jax_for_progress(params, progress)
        weight_t = params.discount_gamma ** t
        total_risk += weight_t * params.risk_scale * risk_jax(mu, Sigma, goal_obs, goal_cov_t)
        total_amb += weight_t * params.ambiguity_scale * ambiguity_jax(Sigma, Gamma, S)
        total_control += weight_t * params.control_weight * jnp.sum(u[t] ** 2)
        if nogo_cost is not None:
            total_nogo += weight_t * nogo_cost(m)

    return total_risk + total_amb + total_control + total_nogo


def make_notebook_simple_valgrad_fn(
    params: JaxNotebookSimpleParams,
    g,
    p_vis_state=None,
    nogo_cost=None,
    jit: bool = True,
):
    dg = jax.jacfwd(g)

    def _valgrad(u, m, S, goal_obs, progress_index0):
        return jax.value_and_grad(
            lambda uu: notebook_simple_unicycle_jax(
                uu,
                m,
                S,
                goal_obs,
                progress_index0,
                params,
                g,
                p_vis_state=p_vis_state,
                nogo_cost=nogo_cost,
                dg=dg,
            )
        )(u)

    return jax.jit(_valgrad) if jit else _valgrad
