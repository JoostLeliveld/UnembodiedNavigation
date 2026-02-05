"""Dynamics models used by planners."""

import math
import numpy as np

from planning.efe_utils import wrap_angle


def unicycle_step(state, control, dt):
    """Unicycle step in SE(2). state=[x,y,theta], control=[v,w]."""
    x, y, theta = state
    v, w = control
    x = x + v * dt * math.cos(theta)
    y = y + v * dt * math.sin(theta)
    theta = wrap_angle(theta + w * dt)
    return np.array([x, y, theta], dtype=float)


def unicycle_jacobian(state, control, dt):
    """Jacobian of unicycle dynamics wrt state."""
    _, _, theta = state
    v, _ = control
    F = np.eye(3)
    F[0, 2] = -v * dt * math.sin(theta)
    F[1, 2] = v * dt * math.cos(theta)
    return F


def unicycle_process_noise(process_noise_xy, process_noise_theta, dt, base_dt=None):
    """Process noise matrix for unicycle; scaled by dt/base_dt if provided."""
    Q = np.diag([
        process_noise_xy ** 2,
        process_noise_xy ** 2,
        process_noise_theta ** 2,
    ])
    if base_dt is not None and base_dt > 1e-9:
        Q = Q * max(dt / base_dt, 0.0)
    return Q


def linear_dynamics_matrices(dt):
    """Linear 4D dynamics matrices from the reference free_energy_agents.py."""
    A = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [dt, 0.0],
        [0.0, dt],
    ])
    return A, B


def linear_process_noise(dt, rho):
    """Process noise covariance for the linear 4D dynamics (reference form)."""
    rho = np.asarray(rho, dtype=float).reshape(-1)
    if rho.size == 1:
        rho = np.array([rho[0], rho[0]], dtype=float)
    return np.array([
        [dt**3 / 3.0 * rho[0], 0.0, dt**2 / 2.0 * rho[0], 0.0],
        [0.0, dt**3 / 3.0 * rho[1], 0.0, dt**2 / 2.0 * rho[1]],
        [dt**2 / 2.0 * rho[0], 0.0, dt * rho[0], 0.0],
        [0.0, dt**2 / 2.0 * rho[1], 0.0, dt * rho[1]],
    ])
