"""JAX utilities for EFE planning (optional runtime dependency)."""

from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


def jax_available() -> bool:
    return jax is not None and jnp is not None


@dataclass
class JaxUnicycleParams:
    Q: "jnp.ndarray"
    R: "jnp.ndarray"
    goal_state: "jnp.ndarray"
    goal_state_cov: "jnp.ndarray"
    goal_obs: "jnp.ndarray"
    goal_obs_cov: "jnp.ndarray"
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
    params: JaxUnicycleParams,
    g,
    approx: str = "ET2",
    add_ambiguity: bool = True,
    use_obs_risk: bool = True,
    use_state_risk: bool = True,
    p_vis=None,
    dg=None,
    d2g=None,
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
            r_state = risk_jax(m, S, params.goal_state, params.goal_state_cov)

        r_obs = 0.0
        if use_obs_risk and mu is not None:
            r_obs = risk_jax(mu, Sigma, params.goal_obs, params.goal_obs_cov)

        total_risk += params.risk_weight_state * r_state + params.risk_weight_obs * r_obs

        if add_ambiguity and Sigma is not None:
            amb = ambiguity_jax(Sigma, Gamma, S_eff)
            if p_vis is not None:
                amb = p * amb
            total_amb += params.ambiguity_weight * amb

        total_control += params.control_weight * jnp.sum(u[t] ** 2)
        if p_vis is not None and params.visibility_weight > 0.0:
            total_vis += params.visibility_weight * q

    return total_risk + total_amb + total_control + total_vis


def make_unicycle_valgrad_fn(
    params: JaxUnicycleParams,
    g,
    approx: str = "ET2",
    add_ambiguity: bool = True,
    use_obs_risk: bool = True,
    use_state_risk: bool = True,
    p_vis=None,
    mode: str = "rev",
    jit: bool = True,
):
    """Return (value, grad) function w.r.t. u for unicycle dynamics."""
    dg = jax.jacfwd(g)
    d2g = jax.jacfwd(jax.jacrev(g)) if str(approx).upper() == "ET2" else None

    def _valgrad(u, m, S):
        if mode == "fwd":
            val = efe_unicycle_jax(
                u, m, S, params, g,
                approx=approx,
                add_ambiguity=add_ambiguity,
                use_obs_risk=use_obs_risk,
                use_state_risk=use_state_risk,
                p_vis=p_vis,
                dg=dg,
                d2g=d2g,
            )
            grad = jax.jacfwd(
                lambda uu: efe_unicycle_jax(
                    uu, m, S, params, g,
                    approx=approx,
                    add_ambiguity=add_ambiguity,
                    use_obs_risk=use_obs_risk,
                    use_state_risk=use_state_risk,
                    p_vis=p_vis,
                    dg=dg,
                    d2g=d2g,
                )
            )(u)
            return val, grad

        return jax.value_and_grad(
            lambda uu: efe_unicycle_jax(
                uu, m, S, params, g,
                approx=approx,
                add_ambiguity=add_ambiguity,
                use_obs_risk=use_obs_risk,
                use_state_risk=use_state_risk,
                p_vis=p_vis,
                dg=dg,
                d2g=d2g,
            )
        )(u)

    return jax.jit(_valgrad) if jit else _valgrad
