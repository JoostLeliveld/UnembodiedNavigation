"""Legacy notebook JAX utilities for EFE planning.

Not used by the production ROS experiment runtime path.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class JaxAgentParams:
    A: jnp.ndarray
    B: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray
    goal_mu: jnp.ndarray
    goal_S: jnp.ndarray
    eta: float
    time_horizon: int
    Du: int


@dataclass
class JaxUnicycleParams:
    Q: jnp.ndarray
    R: jnp.ndarray
    goal_mu: jnp.ndarray
    goal_S: jnp.ndarray
    eta: float
    time_horizon: int
    Du: int
    dt: float


def bind_agent_jax(agent) -> JaxAgentParams:
    """Extract JAX-ready parameters from a numpy EFEAgent."""
    eta_val = float(getattr(agent, "eta", getattr(agent, "η", 0.0)))
    return JaxAgentParams(
        A=jnp.array(agent.A),
        B=jnp.array(agent.B),
        Q=jnp.array(agent.Q),
        R=jnp.array(agent.R),
        goal_mu=jnp.array(agent.goal[0]),
        goal_S=jnp.array(agent.goal[1]),
        eta=eta_val,
        time_horizon=int(agent.time_horizon),
        Du=int(agent.Du),
    )


def bind_unicycle_agent_jax(agent) -> JaxUnicycleParams:
    """Extract JAX-ready parameters from a numpy UnicycleEFEAgent."""
    eta_val = float(getattr(agent, "eta", getattr(agent, "η", 0.0)))
    return JaxUnicycleParams(
        Q=jnp.array(agent.Q),
        R=jnp.array(agent.R),
        goal_mu=jnp.array(agent.goal[0]),
        goal_S=jnp.array(agent.goal[1]),
        eta=eta_val,
        time_horizon=int(agent.time_horizon),
        Du=int(agent.Du),
        dt=float(agent.dt),
    )


def make_homography_g_jax(H):
    """Return a JAX-friendly planar homography observation function."""
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


def et1_jax(m, S, params: JaxAgentParams, g):
    Jm = jax.jacfwd(g)(m)
    mu = g(m)
    Sigma = Jm @ S @ Jm.T + params.R
    Gamma = S @ Jm.T
    return mu, Sigma, Gamma


def et2_jax(m, S, params: JaxAgentParams, g):
    Jm = jax.jacfwd(g)(m)
    H = jax.jacfwd(jax.jacrev(g))(m)
    aux1 = jnp.array([jnp.trace(H[i] @ S) for i in range(H.shape[0])])
    aux2 = jnp.array([
        [jnp.trace(H[i] @ S @ H[j] @ S) for j in range(H.shape[0])]
        for i in range(H.shape[0])
    ])
    mu = g(m) + 0.5 * aux1
    Sigma = Jm @ S @ Jm.T + 0.5 * aux2 + params.R
    Gamma = S @ Jm.T
    return mu, Sigma, Gamma


def risk_jax(mu, Sigma, params: JaxAgentParams):
    eps = 1e-9
    L0 = _safe_cholesky(Sigma, eps=eps)
    L1 = _safe_cholesky(params.goal_S, eps=eps)
    M = jnp.linalg.solve(L1, L0)
    y = jnp.linalg.solve(L1, params.goal_mu - mu)
    d = params.goal_mu.shape[0]
    # Standard Gaussian KL (L2 norm squared)
    return 0.5 * (jnp.sum(M ** 2) - d + jnp.sum(y ** 2) +
                  2.0 * jnp.sum(jnp.log(jnp.diag(L1) / jnp.diag(L0))))


def ambiguity_jax(Sigma, Gamma, S):
    Sigma_cond = Sigma - Gamma.T @ jnp.linalg.solve(S, Gamma)
    sign, logdet = jnp.linalg.slogdet(Sigma_cond)
    d = Sigma_cond.shape[0]
    return 0.5 * (d * jnp.log(2 * jnp.pi * jnp.e) + logdet)


def efe_jax(u, m0, S0, params: JaxAgentParams, g, approx="ET2", add_ambiguity=True):
    u = jnp.asarray(u)
    if u.ndim == 1:
        u = u.reshape((params.time_horizon, params.Du))
    m = m0
    S = S0
    cost = 0.0
    for t in range(params.time_horizon):
        m = params.A @ m + params.B @ u[t]
        S = params.A @ S @ params.A.T + params.Q
        if approx == "ET1":
            mu, Sigma, Gamma = et1_jax(m, S, params, g)
        elif approx == "ET2":
            mu, Sigma, Gamma = et2_jax(m, S, params, g)
        else:
            raise ValueError("Approximation method unknown.")
        # Standard quadratic control penalty on the 2-D control at step t
        cost += risk_jax(mu, Sigma, params) + params.eta * jnp.sum(u[t] ** 2)
        if add_ambiguity:
            cost += ambiguity_jax(Sigma, Gamma, S)
    return cost


def efe_unicycle_jax(u, m0, S0, params: JaxUnicycleParams, g, approx="ET2", add_ambiguity=True):
    u = jnp.asarray(u)
    if u.ndim == 1:
        u = u.reshape((params.time_horizon, params.Du))
    m = m0
    S = S0
    cost = 0.0
    for t in range(params.time_horizon):
        m = unicycle_step_jax(m, u[t], params.dt)
        F = unicycle_jacobian_jax(m, u[t], params.dt)
        S = F @ S @ F.T + params.Q
        if approx == "ET1":
            mu, Sigma, Gamma = et1_jax(m, S, params, g)
        elif approx == "ET2":
            mu, Sigma, Gamma = et2_jax(m, S, params, g)
        else:
            raise ValueError("Approximation method unknown.")
        cost += risk_jax(mu, Sigma, params) + params.eta * jnp.sum(u[t] ** 2)
        if add_ambiguity:
            cost += ambiguity_jax(Sigma, Gamma, S)
    return cost


def make_valgrad_fn(params: JaxAgentParams, g, approx="ET2", add_ambiguity=True, mode="rev", jit=True):
    """Return (value, grad) function w.r.t. u for fixed params and g."""

    def _valgrad(u, m, S):
        if mode == "fwd":
            val = efe_jax(u, m, S, params, g, approx=approx, add_ambiguity=add_ambiguity)
            grad = jax.jacfwd(lambda uu: efe_jax(uu, m, S, params, g,
                                                 approx=approx, add_ambiguity=add_ambiguity))(u)
            return val, grad
        # default reverse-mode
        return jax.value_and_grad(
            lambda uu: efe_jax(uu, m, S, params, g,
                               approx=approx, add_ambiguity=add_ambiguity)
        )(u)

    return jax.jit(_valgrad) if jit else _valgrad


def make_unicycle_valgrad_fn(params: JaxUnicycleParams, g, approx="ET2", add_ambiguity=True, mode="rev", jit=True):
    """Return (value, grad) function w.r.t. u for unicycle dynamics."""

    def _valgrad(u, m, S):
        if mode == "fwd":
            val = efe_unicycle_jax(u, m, S, params, g, approx=approx, add_ambiguity=add_ambiguity)
            grad = jax.jacfwd(lambda uu: efe_unicycle_jax(uu, m, S, params, g,
                                                         approx=approx, add_ambiguity=add_ambiguity))(u)
            return val, grad
        return jax.value_and_grad(
            lambda uu: efe_unicycle_jax(uu, m, S, params, g,
                                        approx=approx, add_ambiguity=add_ambiguity)
        )(u)

    return jax.jit(_valgrad) if jit else _valgrad
