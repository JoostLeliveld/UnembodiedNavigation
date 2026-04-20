from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/planning',):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from planning.core.casadi_efe import unicycle_jacobian_ca, unicycle_step_ca
from planning.core.dynamics import unicycle_jacobian, unicycle_step


def _finite_difference_jacobian(state: np.ndarray, control: np.ndarray, dt: float, eps: float = 1e-6) -> np.ndarray:
    base = unicycle_step(state, control, dt)
    J = np.zeros((3, 3), dtype=float)
    for idx in range(3):
        perturbed = np.array(state, dtype=float)
        perturbed[idx] += eps
        J[:, idx] = (unicycle_step(perturbed, control, dt) - base) / eps
    return J


def test_unicycle_jacobian_matches_pre_step_finite_difference() -> None:
    state = np.array([1.2, -0.4, 0.7], dtype=float)
    control = np.array([0.35, -0.2], dtype=float)
    dt = 0.15

    analytic = unicycle_jacobian(state, control, dt)
    numeric = _finite_difference_jacobian(state, control, dt)
    np.testing.assert_allclose(analytic, numeric, atol=5e-6, rtol=5e-6)


def test_casadi_unicycle_jacobian_uses_pre_step_heading() -> None:
    ca = pytest.importorskip('casadi')
    state = ca.DM([0.8, 0.1, 0.45])
    control = ca.DM([0.4, 0.9])
    dt = 0.25

    jacobian = np.asarray(unicycle_jacobian_ca(state, control, dt), dtype=float)
    expected = np.eye(3, dtype=float)
    expected[0, 2] = -float(control[0]) * dt * math.sin(float(state[2]))
    expected[1, 2] = float(control[0]) * dt * math.cos(float(state[2]))
    np.testing.assert_allclose(jacobian, expected, atol=1e-9, rtol=1e-9)

    stepped = np.asarray(unicycle_step_ca(state, control, dt), dtype=float).reshape(-1)
    assert jacobian[0, 2] != pytest.approx(-float(control[0]) * dt * math.sin(float(stepped[2])))
