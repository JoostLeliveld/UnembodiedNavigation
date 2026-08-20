from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "experiments/usable_observation/supervisor_comparison"
    / "11_static_probability_planning/closed_loop_gazebo/solve_offline_routes.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("static_puse_offline_routes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validator_fails_closed_on_goal_mismatch():
    solver = _module()
    ctx = solver.lean_context()
    path = np.asarray([[-7.78, -7.5], [-7.78, -6.5]], dtype=float)
    with pytest.raises(solver.NoFeasibleRouteError, match="goal mismatch"):
        solver.validate_complete_route(
            path,
            np.asarray([-7.78, -7.5]),
            np.asarray([-5.68, 5.5]),
            ctx["driveable_prisms"],
        )


def test_validator_fails_closed_on_insufficient_clearance():
    solver = _module()
    ctx = solver.lean_context()
    # x=-8.22 lies only 1 cm inside the W3 aisle's west edge.
    path = np.asarray([[-8.22, -5.0], [-8.22, -4.0]], dtype=float)
    with pytest.raises(solver.NoFeasibleRouteError, match="route clearance"):
        solver.validate_complete_route(
            path, path[0], path[-1], ctx["driveable_prisms"], endpoint_tolerance_m=1e-6
        )


def test_offline_solver_returns_complete_exact_routes_and_discriminates():
    solver = _module()
    payload, _candidates, selected = solver.solve()
    assert set(payload["selections"]) == {"C1", "C2", "C3"}
    assert len(selected) == 3
    for row in selected:
        assert row["complete"] is True
        assert row["start_error_m"] == pytest.approx(0.0, abs=1e-12)
        assert row["goal_error_m"] == pytest.approx(0.0, abs=1e-12)
        assert row["minimum_driveable_clearance_m"] >= solver.MIN_DRIVEABLE_CLEARANCE_M
    # The scientific mechanism is route choice: blind and probability-aware differ.
    assert payload["selections"]["C1"]["candidate_id"] != payload["selections"]["C2"]["candidate_id"]
    # On this field the two probability-update formulations select the same route.
    assert payload["selections"]["C2"]["candidate_id"] == payload["selections"]["C3"]["candidate_id"]
