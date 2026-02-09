"""Helper utilities for EFE planning with planar camera observations."""

import math
import numpy as np
from scipy.linalg import cholesky, inv
from scipy.stats import multivariate_normal


def compute_intrinsics(width, height, fov_h_rad):
    f = (width / 2.0) / math.tan(fov_h_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def compute_lookat_rotation(cam_pos, look_at, up_hint=(0.0, 0.0, 1.0)):
    cam_pos = np.array(cam_pos, dtype=float)
    look_at = np.array(look_at, dtype=float)
    up_hint = np.array(up_hint, dtype=float)

    z_cam = look_at - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)
    x_cam = np.cross(z_cam, up_hint)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    return np.array([x_cam, y_cam, z_cam])


class PlanarCamera:
    """Planar homography camera model (ground plane Z=0)."""

    def __init__(
        self,
        cam_pos,
        look_at,
        img_width,
        img_height,
        fov_h_rad,
        up_hint=(0.0, 0.0, 1.0),
    ):
        self.cam_pos = np.array(cam_pos, dtype=float)
        self.look_at = np.array(look_at, dtype=float)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.fov_h_rad = float(fov_h_rad)
        self.up_hint = np.array(up_hint, dtype=float)

        self.K = compute_intrinsics(self.img_width, self.img_height, self.fov_h_rad)
        self.R = compute_lookat_rotation(self.cam_pos, self.look_at, self.up_hint)
        self.t = -self.R @ self.cam_pos
        self.H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        self.H_inv = np.linalg.inv(self.H)

    def world_to_pixel(self, x, y):
        world_pt = np.array([x, y, 1.0], dtype=float)
        pix = self.H @ world_pt
        if abs(pix[2]) < 1e-9:
            return np.nan, np.nan
        return float(pix[0] / pix[2]), float(pix[1] / pix[2])

    def pixel_to_world(self, u, v):
        pixel_pt = np.array([u, v, 1.0], dtype=float)
        world_h = self.H_inv @ pixel_pt
        if abs(world_h[2]) < 1e-9:
            return np.nan, np.nan
        return float(world_h[0] / world_h[2]), float(world_h[1] / world_h[2])

    def g(self, state):
        x, y = float(state[0]), float(state[1])
        u, v = self.world_to_pixel(x, y)
        return np.array([u, v], dtype=float)


def _ensure_psd(S, eps=1e-9):
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return (vecs @ np.diag(vals) @ vecs.T + (vecs @ np.diag(vals) @ vecs.T).T) / 2.0


def finite_diff_jacobian(g, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(g(x), dtype=float)
    m = y0.size
    n = x.size
    J = np.zeros((m, n), dtype=float)
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        y1 = np.asarray(g(x + dx), dtype=float)
        y2 = np.asarray(g(x - dx), dtype=float)
        J[:, i] = (y1 - y2) / (2.0 * eps)
    return J


def finite_diff_hessian_scalar(g, x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dx_i = np.zeros_like(x)
            dx_i[i] = eps
            dx_j = np.zeros_like(x)
            dx_j[j] = eps
            f_pp = g(x + dx_i + dx_j)
            f_pm = g(x + dx_i - dx_j)
            f_mp = g(x - dx_i + dx_j)
            f_mm = g(x - dx_i - dx_j)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
    return H


def sigma_points(m, P, alpha=1e-3, kappa=0.0):
    m = np.asarray(m, dtype=float)
    P = np.asarray(P, dtype=float)
    n = m.size
    lam = alpha ** 2 * (n + kappa) - n
    scale = n + lam

    jitter = 1e-9
    for _ in range(5):
        try:
            S = np.linalg.cholesky(scale * P + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        S = np.linalg.cholesky(scale * _ensure_psd(P) + jitter * np.eye(n))

    sigmas = np.zeros((2 * n + 1, n), dtype=float)
    sigmas[0] = m
    for i in range(n):
        sigmas[1 + i] = m + S[:, i]
        sigmas[1 + n + i] = m - S[:, i]
    return sigmas


def ut_weights(n, alpha=1e-3, beta=2.0, kappa=0.0):
    lam = alpha ** 2 * (n + kappa) - n
    w0 = lam / (n + lam)
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wm[0] = w0
    Wc[0] = w0 + (1.0 - alpha ** 2 + beta)
    return Wm, Wc


def UT(m, P, g, addmatrix=None, forceHermitian=False, alpha=1e-3, beta=2.0, kappa=0.0):
    m = np.asarray(m, dtype=float)
    P = np.asarray(P, dtype=float)
    n = m.size
    sigmas = sigma_points(m, P, alpha=alpha, kappa=kappa)
    Wm, Wc = ut_weights(n, alpha=alpha, beta=beta, kappa=kappa)

    ys = np.array([np.asarray(g(s), dtype=float) for s in sigmas])
    mu = np.sum(Wm[:, None] * ys, axis=0)

    d_y = ys.shape[1]
    Sigma = np.zeros((d_y, d_y), dtype=float)
    Gamma = np.zeros((n, d_y), dtype=float)
    for i in range(2 * n + 1):
        dy = (ys[i] - mu).reshape(-1, 1)
        dx = (sigmas[i] - m).reshape(-1, 1)
        Sigma += Wc[i] * (dy @ dy.T)
        Gamma += Wc[i] * (dx @ dy.T)

    if addmatrix is not None:
        Sigma += addmatrix
    if forceHermitian:
        Sigma = (Sigma + Sigma.T) / 2.0
    return mu, Sigma, Gamma


def ET1(m, S, g, addmatrix=None, forceHermitian=False, eps=1e-6):
    m = np.asarray(m, dtype=float)
    S = np.asarray(S, dtype=float)
    Jm = finite_diff_jacobian(g, m, eps=eps)
    mE = np.asarray(g(m), dtype=float)
    SE = Jm @ S @ Jm.T
    CE = S @ Jm.T
    if addmatrix is not None:
        SE += addmatrix
    if forceHermitian:
        SE = (SE + SE.T) / 2.0
    return mE, SE, CE


def ET2(m, S, g, addmatrix=None, forceHermitian=False, eps=1e-4):
    m = np.asarray(m, dtype=float)
    S = np.asarray(S, dtype=float)
    y = np.asarray(g(m), dtype=float)
    n = m.size
    d = y.size
    Jm = finite_diff_jacobian(g, m, eps=eps)

    H = np.zeros((d, n, n), dtype=float)
    for i in range(d):
        def g_i(x):
            return np.asarray(g(x), dtype=float)[i]
        H[i] = finite_diff_hessian_scalar(g_i, m, eps=eps)

    aux1 = np.zeros(d, dtype=float)
    for i in range(d):
        aux1[i] = np.trace(H[i] @ S)

    aux2 = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            aux2[i, j] = np.trace(H[i] @ S @ H[j] @ S)

    mE = y + 0.5 * aux1
    SE = Jm @ S @ Jm.T + 0.5 * aux2
    CE = S @ Jm.T

    if addmatrix is not None:
        SE += addmatrix
    if forceHermitian:
        SE = (SE + SE.T) / 2.0
    return mE, SE, CE


def ambiguity(Sigma, Gamma, S):
    S_inv = np.linalg.inv(S)
    Sigma_cond = Sigma - Gamma.T @ S_inv @ Gamma
    Sigma_cond = _ensure_psd(Sigma_cond)
    sign, logdet = np.linalg.slogdet(Sigma_cond)
    if sign <= 0:
        logdet = np.log(np.maximum(np.linalg.det(Sigma_cond), 1e-12))
    d = Sigma_cond.shape[0]
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)


def risk(mu, Sigma, goal):
    m_star, S_star = goal
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    m_star = np.asarray(m_star, dtype=float)
    S_star = np.asarray(S_star, dtype=float)

    d = mu.size
    S_inv = np.linalg.inv(S_star)
    diff = (m_star - mu).reshape(-1, 1)

    sign_s, logdet_s = np.linalg.slogdet(Sigma)
    sign_t, logdet_t = np.linalg.slogdet(S_star)
    if sign_s <= 0:
        logdet_s = np.log(np.maximum(np.linalg.det(Sigma), 1e-12))
    if sign_t <= 0:
        logdet_t = np.log(np.maximum(np.linalg.det(S_star), 1e-12))

    term_trace = np.trace(S_inv @ Sigma)
    term_quad = float(diff.T @ S_inv @ diff)
    return 0.5 * (term_trace + term_quad - d + (logdet_t - logdet_s))


class EFEAgent:
    """Expected Free Energy Agent for active inference planning."""

    def __init__(self, goal, g, rho, sigma=1.0, eta=1.0, dt=1.0, time_horizon=1):
        """
        Construct agent

        Parameters:
        -----------
        goal : tuple
            Tuple of (goal_mean, goal_covariance) where goal_mean is a vector
            and goal_covariance is a matrix
        g : callable
            Measurement function
        rho : array-like
            Process noise parameters [rho_x, rho_y]
        sigma : float, optional
            Measurement noise standard deviation (default: 1.0)
        eta : float, optional
            Control cost weight (default: 1.0)
        dt : float, optional
            Time step (default: 1.0)
        time_horizon : int, optional
            Planning horizon (default: 1)
        """

        self.Dx = 4
        self.Du = 2
        self.Dy = len(g(np.zeros(self.Dx)))
        self.dt = dt

        self.g = g
        self.eta = eta
        self.goal = goal
        self.time_horizon = time_horizon

        # State transition matrix
        self.A = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Control matrix
        self.B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [dt, 0.0],
            [0.0, dt],
        ])

        # Process noise covariance matrix
        self.Q = np.array([
            [dt ** 3 / 3 * rho[0], 0.0, dt ** 2 / 2 * rho[0], 0.0],
            [0.0, dt ** 3 / 3 * rho[1], 0.0, dt ** 2 / 2 * rho[1]],
            [dt ** 2 / 2 * rho[0], 0.0, dt * rho[0], 0.0],
            [0.0, dt ** 2 / 2 * rho[1], 0.0, dt * rho[1]],
        ])

        # Measurement noise covariance matrix
        self.R = np.diag(sigma ** 2 * np.ones(self.Dy))


def predict(agent, m_kmin1, S_kmin1, u_kmin1):
    """
    Chapman-Kolmogorov for linear Gaussian state transition using known control u

    Parameters:
    -----------
    agent : EFEAgent
        The agent instance
    m_kmin1 : array-like
        Previous state mean
    S_kmin1 : array-like
        Previous state covariance
    u_kmin1 : array-like
        Previous control input

    Returns:
    --------
    m_k_pred : array
        Predicted state mean
    S_k_pred : array
        Predicted state covariance
    """

    m_k_pred = agent.A @ m_kmin1 + agent.B @ u_kmin1
    S_k_pred = agent.A @ S_kmin1 @ agent.A.T + agent.Q

    return m_k_pred, S_k_pred


def correct(agent, y_k, m_k_pred, S_k_pred, approx="ET2"):
    """
    Correction step based on Gaussian approximation to nonlinear measurement

    Parameters:
    -----------
    agent : EFEAgent
        The agent instance
    y_k : array-like
        Current measurement
    m_k_pred : array-like
        Predicted state mean
    S_k_pred : array-like
        Predicted state covariance
    approx : str, optional
        Approximation method: "ET1", "ET2", or "UT" (default: "ET2")

    Returns:
    --------
    m_k : array
        Corrected state mean
    S_k : array
        Corrected state covariance
    """

    if approx == "ET1":
        mu, Sigma, Gamma = ET1(m_k_pred, S_k_pred, agent.g, addmatrix=agent.R, forceHermitian=True)
    elif approx == "ET2":
        mu, Sigma, Gamma = ET2(m_k_pred, S_k_pred, agent.g, addmatrix=agent.R, forceHermitian=True)
    elif approx == "UT":
        mu, Sigma, Gamma = UT(m_k_pred, S_k_pred, agent.g, addmatrix=agent.R, forceHermitian=True)
    else:
        raise ValueError("Approximation method unknown.")

    Sigma_inv = inv(Sigma)
    m_k = m_k_pred + Gamma @ Sigma_inv @ (y_k - mu)
    S_k = S_k_pred - Gamma @ Sigma_inv @ Gamma.T

    return m_k, S_k


def condition_yx(m, S, dims=1):
    """
    Conditioning a Gaussian distribution.

    Appendix A(5), Särkkä (2013), Bayesian filtering & Smoothing.

    Parameters:
    -----------
    m : array-like
        Mean vector
    S : array-like
        Covariance matrix
    dims : int, optional
        Number of dimensions for first part (default: 1)

    Returns:
    --------
    m_y : callable
        Function that returns conditional mean given x
    S_y : callable
        Function that returns conditional covariance given x
    """

    m_a = m[:dims]
    m_b = m[dims:]

    S_A = S[:dims, :dims]
    S_B = S[dims:, dims:]
    S_C = S[:dims, dims:]

    S_A_inv = inv(S_A)

    def m_y(x):
        return m_b + S_C.T @ S_A_inv @ (x - m_a)

    def S_y(x):
        return S_B - S_C.T @ S_A_inv @ S_C

    return m_y, S_y


def ambiguity(Sigma, Gamma, S):
    """
    Conditional entropy term within expected free energy

    Parameters:
    -----------
    Sigma : array-like
        Measurement covariance
    Gamma : array-like
        Cross-covariance
    S : array-like
        State covariance

    Returns:
    --------
    float
        Ambiguity term
    """

    S_inv = inv(S)
    Sigma_cond = Sigma - Gamma.T @ S_inv @ Gamma
    return 0.5 * (Sigma.shape[0] * np.log(2 * np.pi * np.e) + np.log(np.linalg.det(Sigma_cond)))


def risk(mu, Sigma, goal):
    """
    Kullback-Leibler divergence term within expected free energy

    Parameters:
    -----------
    mu : array-like
        Predicted mean
    Sigma : array-like
        Predicted covariance
    goal : tuple
        Tuple of (goal_mean, goal_covariance)

    Returns:
    --------
    float
        Risk term (KL divergence)
    """

    m_star, S_star = goal
    D = len(m_star)

    L0 = cholesky(Sigma, lower=True)
    L1 = cholesky(S_star, lower=True)

    L1_inv = inv(L1)
    M = L1_inv @ L0
    y = L1_inv @ (m_star - mu)

    return 0.5 * (np.sum(M ** 2) - D + np.linalg.norm(y) ** 2 +
                  2 * np.sum([np.log(L1[i, i] / L0[i, i]) for i in range(D)]))


def evidence(agent, y_k, m_k, S_k, approx="ET2"):
    """
    Marginal likelihood

    Parameters:
    -----------
    agent : EFEAgent
        The agent instance
    y_k : array-like
        Current measurement
    m_k : array-like
        Current state mean
    S_k : array-like
        Current state covariance
    approx : str, optional
        Approximation method: "ET1", "ET2", or "UT" (default: "ET2")

    Returns:
    --------
    float
        Negative log likelihood
    """

    # Gaussian approximation
    if approx == "ET1":
        mu, Sigma, _ = ET1(m_k, S_k, agent.g, addmatrix=agent.R, forceHermitian=True)
    elif approx == "ET2":
        mu, Sigma, _ = ET2(m_k, S_k, agent.g, addmatrix=agent.R, forceHermitian=True)
    elif approx == "UT":
        mu, Sigma, _ = UT(m_k, S_k, agent.g, addmatrix=agent.R, forceHermitian=True)
    else:
        raise ValueError("Approximation method unknown.")

    return -multivariate_normal.logpdf(y_k, mu, Sigma)


def EFE(agent, u, state, approx="ET2", add_ambiguity=True):
    """
    Expected Free Energy

    Parameters:
    -----------
    agent : EFEAgent
        The agent instance
    u : array-like
        Control sequence (flattened)
    state : tuple
        Tuple of (current_mean, current_covariance)
    approx : str, optional
        Approximation method: "ET1", "ET2", or "UT" (default: "ET2")
    add_ambiguity : bool, optional
        Whether to include ambiguity term (default: True)

    Returns:
    --------
    float
        Expected Free Energy value
    """

    # Unpack parameters of current state
    m_tmin1, S_tmin1 = state

    # Start cumulative sum
    cEFE = 0.0
    for t in range(1, agent.time_horizon + 1):

        # State transition p(z_t | u_t)
        u_t = u[(t - 1) * 2:2 * t]
        m_t, S_t = predict(agent, m_tmin1, S_tmin1, u_t)

        # Gaussian approximation
        if approx == "ET1":
            mu, Sigma, Gamma = ET1(m_t, S_t, agent.g, addmatrix=agent.R, forceHermitian=True)
        elif approx == "ET2":
            mu, Sigma, Gamma = ET2(m_t, S_t, agent.g, addmatrix=agent.R, forceHermitian=True)
        elif approx == "UT":
            mu, Sigma, Gamma = UT(m_t, S_t, agent.g, addmatrix=agent.R, forceHermitian=True)
        else:
            raise ValueError("Approximation method unknown.")

        # Accumulate objective
        # Control cost: squared norm of control vector
        cEFE += risk(mu, Sigma, agent.goal) + agent.eta * np.sum(u_t ** 2)
        if add_ambiguity:
            cEFE += ambiguity(Sigma, Gamma, S_t)

        # Update state recursion
        m_tmin1 = m_t
        S_tmin1 = S_t

    return cEFE


def planned_trajectory(agent, policy, current_state, approx="ET2"):
    """
    Generate future states and observations

    Parameters:
    -----------
    agent : EFEAgent
        The agent instance
    policy : array-like
        Policy matrix of shape (Du, time_horizon)
    current_state : tuple
        Tuple of (current_mean, current_covariance)
    approx : str, optional
        Approximation method: "ET1", "ET2", or "UT" (default: "ET2")

    Returns:
    --------
    tuple
        Tuple of ((z_m, z_S), (y_m, y_S)) where:
        - z_m: state means (Dx, time_horizon)
        - z_S: state covariances (Dx, Dx, time_horizon)
        - y_m: observation means (Dy, time_horizon)
        - y_S: observation covariances (Dy, Dy, time_horizon)
    """

    # Unpack parameters of current state
    m_tmin1, S_tmin1 = current_state

    # Track predicted observations
    z_m = np.zeros((4, agent.time_horizon))
    z_S = np.zeros((4, 4, agent.time_horizon))
    y_m = np.zeros((agent.Dy, agent.time_horizon))
    y_S = np.zeros((agent.Dy, agent.Dy, agent.time_horizon))

    for t in range(agent.time_horizon):

        # State transition
        z_m[:, t] = agent.A @ m_tmin1 + agent.B @ policy[:, t]
        z_S[:, :, t] = agent.A @ S_tmin1 @ agent.A.T + agent.Q

        # Gaussian approximation
        if approx == "ET1":
            mu, Sigma, _ = ET1(z_m[:, t], z_S[:, :, t], agent.g, addmatrix=agent.R, forceHermitian=True)
            y_m[:, t] = mu
            y_S[:, :, t] = Sigma
        elif approx == "ET2":
            mu, Sigma, _ = ET2(z_m[:, t], z_S[:, :, t], agent.g, addmatrix=agent.R, forceHermitian=True)
            y_m[:, t] = mu
            y_S[:, :, t] = Sigma
        elif approx == "UT":
            mu, Sigma, _ = UT(z_m[:, t], z_S[:, :, t], agent.g, addmatrix=agent.R, forceHermitian=True)
            y_m[:, t] = mu
            y_S[:, :, t] = Sigma
        else:
            raise ValueError("Approximation method unknown.")

        # Update previous state
        m_tmin1 = z_m[:, t]
        S_tmin1 = z_S[:, :, t]

    return (z_m, z_S), (y_m, y_S)
