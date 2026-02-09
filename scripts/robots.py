"""
Robots module for simulating robot dynamics and observations.
"""

import numpy as np
from scipy.stats import multivariate_normal


class FieldBot:
    """
    FieldBot robot with stochastic state transitions and observations.
    """
    
    def __init__(self, g, ρ, σ=1.0, Δt=1.0, control_lims=(-1.0, 1.0)):
        """
        Construct FieldBot
        
        Parameters:
        -----------
        g : callable
            Measurement function
        ρ : array-like
            Process noise parameters [ρ_x, ρ_y]
        σ : float, optional
            Measurement noise standard deviation (default: 1.0)
        Δt : float, optional
            Time step (default: 1.0)
        control_lims : tuple, optional
            Control limits (min, max) (default: (-1.0, 1.0))
        """
        
        self.Dx = 4
        self.Du = 2
        self.Dy = len(g(np.zeros(self.Dx)))
        self.Δt = Δt
        
        self.g = g
        self.control_lims = control_lims
        
        # State transition matrix
        self.A = np.array([
            [1., 0., Δt, 0.],
            [0., 1., 0., Δt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        
        # Control matrix
        self.B = np.array([
            [0., 0.],
            [0., 0.],
            [Δt, 0.],
            [0., Δt]
        ])
        
        # Process noise covariance matrix
        self.Q = np.array([
            [Δt**3/3*ρ[0], 0.0, Δt**2/2*ρ[0], 0.0],
            [0.0, Δt**3/3*ρ[1], 0.0, Δt**2/2*ρ[1]],
            [Δt**2/2*ρ[0], 0.0, Δt*ρ[0], 0.0],
            [0.0, Δt**2/2*ρ[1], 0.0, Δt*ρ[1]]
        ])
        
        # Measurement noise covariance matrix
        # Note: Julia version uses diagm(ρ), so ρ must have length Dy
        # The σ parameter is passed but not used in the original Julia code
        if len(ρ) == self.Dy:
            self.R = np.diag(ρ)
        else:
            # Fallback: use σ^2 if ρ length doesn't match
            self.R = np.diag(σ**2 * np.ones(self.Dy))


def step(bot, z_kmin1, u_k):
    """
    Stochastic state transition
    
    Parameters:
    -----------
    bot : FieldBot
        The robot instance
    z_kmin1 : array-like
        Previous state
    u_k : array-like
        Control input (will be clamped to control_lims)
    
    Returns:
    --------
    array
        New state sampled from transition distribution
    """
    
    # Clamp control to limits
    u_k = np.clip(u_k, bot.control_lims[0], bot.control_lims[1])
    
    # Sample from transition distribution
    mean = bot.A @ z_kmin1 + bot.B @ u_k
    return multivariate_normal.rvs(mean, bot.Q)


def emit(bot, z_k):
    """
    Stochastic observation
    
    Parameters:
    -----------
    bot : FieldBot
        The robot instance
    z_k : array-like
        Current state
    
    Returns:
    --------
    array
        Noisy observation sampled from observation distribution
    """
    
    # Sample from observation distribution
    mean = bot.g(z_k)
    return multivariate_normal.rvs(mean, bot.R)


def update(bot, z_kmin1, u_k):
    """
    Update environment: perform state transition and emit observation
    
    Parameters:
    -----------
    bot : FieldBot
        The robot instance
    z_kmin1 : array-like
        Previous state
    u_k : array-like
        Control input
    
    Returns:
    --------
    tuple
        (observation, new_state)
    """
    
    # State transition
    z_k = step(bot, z_kmin1, u_k)
    
    # Emit noisy observation
    y_k = emit(bot, z_k)
    
    return y_k, z_k
