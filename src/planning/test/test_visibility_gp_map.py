from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from planning.core.visibility_gp_map import GPVisibilityMapConfig, GPVisibilityMapModel


def test_prob_state_np_matches_casadi_interpolant(tmp_path: Path) -> None:
    pytest.importorskip('casadi')

    xs = np.array([-2.0, -0.5, 1.0, 2.5], dtype=float)
    ys = np.array([-1.5, 0.0, 1.5], dtype=float)
    p_map = np.array(
        [
            [0.12, 0.48, 0.77, 0.66],
            [0.31, 0.58, 0.84, 0.44],
            [0.27, 0.41, 0.69, 0.91],
        ],
        dtype=float,
    )
    artifact_path = tmp_path / 'test_gp.npz'
    np.savez(
        artifact_path,
        xs=xs,
        ys=ys,
        P_map=p_map,
        P_mean_map=p_map,
        P_conservative_map=p_map,
        camera_pos=np.array([-2.45, -2.45, 2.8], dtype=float),
        target_height=np.array([0.0], dtype=float),
    )

    model = GPVisibilityMapModel(GPVisibilityMapConfig(artifact_path=str(artifact_path)))
    prob_state_ca = model.make_prob_state_casadi()

    rng = np.random.default_rng(7)
    samples = np.column_stack(
        [
            rng.uniform(xs[0], xs[-1], size=32),
            rng.uniform(ys[0], ys[-1], size=32),
            np.zeros(32, dtype=float),
        ]
    )
    for state in samples:
        p_np = model.prob_state_np(state)
        p_ca = float(prob_state_ca(state))
        assert abs(p_np - p_ca) < 1e-6
